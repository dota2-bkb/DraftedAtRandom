"""Stats-DQN episode simulator.

Walks turns 0..49 of a synthetic draft. On the focal player's 5 turns the
StatsQNet picks (ε-greedy on the balanced composite argmax of the per-stat
vector). On the other 45 turns the BehaviorPolicy (V+1 output)
samples from its full distribution; if the "random" class is drawn, the
non-focal pick is drawn from the true timeout mechanism P_mech (side-coin
then uniform over heroes∪side; dota2ad.core.mechanism) AND the resulting
history event is flagged is_random=True so the encoder conditions on it
the next time it's queried.

At terminal, the StatsModel scores the completed draft to produce a
per-stat vector reward [K], z-normalized against precomputed (μ, σ) over
realized training stats.
"""

from __future__ import annotations

import random
from typing import cast

import torch

from dota2ad.core.collate import policy_collate, stats_collate
from dota2ad.core.encoding import encode_loadout, encode_mmr, encode_policy_sample
from dota2ad.core.mechanism import sample_mechanism_pick
from dota2ad.core.types import (
    MatchRow,
    NUM_PLAYERS,
    Per10,
    PickSlot,
    PolicySample,
    StatsRecord,
    UnifiedIdx,
    VocabKey,
    Vocabs,
)
from dota2ad.models import BehaviorPolicy, EnsembleStatsModel, QNetStats, StatsModel
from dota2ad.suggest.state import DraftState, make_forced_state
from dota2ad.eval.stats_specs import STAT_SPECS
from dota2ad.training.stats_replay import Transition


K_STATS = len(STAT_SPECS)
HERO_DAMAGE_DIM = 7  # STAT_SPECS index of hero_damage/min — the densified dim


def initial_state_from_match(match: MatchRow) -> DraftState:
    """Turn-0 state from a MatchRow's initial pools and MMR."""
    from dota2ad.core.types import PlayerPickState, Turn
    player_picks: dict[PickSlot, PlayerPickState] = {
        PickSlot(ps): PlayerPickState(hero=None, basics=[], ult=None)
        for ps in range(NUM_PLAYERS)
    }
    return DraftState(
        turn=Turn(0),
        pick_slot=PickSlot(0),
        player_picks=player_picks,
        hero_pool_remaining=list(match.hero_pool),
        basic_pool_remaining=list(match.basic_pool),
        ult_pool_remaining=list(match.ult_pool),
        action_key="",
        mmr=match.mmr,
        history=[],
        hero_id=None,
        draft_ability_id=None,
        radiant_win=match.radiant_win,
        match_id=match.match_id,
    )


def _build_stats_record(
    state: DraftState, vocabs: Vocabs, mmr_mean: float, mmr_std: float,
) -> StatsRecord:
    """Read final loadouts from a terminal state into a StatsRecord. Stat
    label tensors are zero placeholders (not used at forward time)."""
    pp = state.player_picks
    loadouts = cast(
        Per10[list[UnifiedIdx]],
        tuple(encode_loadout(pp[PickSlot(ps)], vocabs) for ps in range(NUM_PLAYERS)),
    )
    mmr_vals, mmr_mask = encode_mmr(state.mmr, mmr_mean, mmr_std)

    # ability_indices [10, 4]: per-player ability draft IDs (the 3 basics + 1
    # ult). The simulator doesn't know upgrade priority (it's a play-time
    # signal), so we just list them in pick order, pad with 0 for unfilled
    # slots. The priority head's output is irrelevant to scalar/matchup/
    # damage extraction.
    ability_idx_list: list[list[int]] = []
    for ps in range(NUM_PLAYERS):
        player = pp[PickSlot(ps)]
        ids: list[int] = []
        for a in player.basics:
            ids.append(int(vocabs.draft_id_to_index[VocabKey(f"a:{a}")]))
        if player.ult is not None:
            ids.append(int(vocabs.draft_id_to_index[VocabKey(f"a:{player.ult}")]))
        while len(ids) < 4:
            ids.append(0)
        ability_idx_list.append(ids[:4])
    ability_indices = torch.tensor(ability_idx_list, dtype=torch.long)

    z22 = torch.zeros(10, 22)
    z3 = torch.zeros(10, 3)
    z5 = torch.zeros(10, 5)
    z14 = torch.zeros(10, 14)
    z6 = torch.zeros(10, 6)
    z4 = torch.zeros(10, 4)
    time_mask = torch.zeros(10, 3, dtype=torch.bool)
    return StatsRecord(
        loadouts=loadouts,
        mmr_vals=mmr_vals,
        mmr_mask=mmr_mask,
        ability_indices=ability_indices,
        scalar_stats=z22,
        gold_t=z3, xp_t=z3, lh_t=z3,
        time_mask=time_mask,
        kill_counts=z5, death_counts=z5, damage_dealt=z5,
        gold_reasons=z14, xp_reasons=z6,
        ability_priorities=z4,
        spell_damage_dealt=z4,
        match_id=state.match_id,
    )


def compute_terminal_reward_vec(
    state: DraftState,
    focal_slot: PickSlot,
    stats_model: StatsModel | EnsembleStatsModel,
    vocabs: Vocabs,
    mmr_mean: float,
    mmr_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Terminal reward + per-ability spell damage for the dense dim-7 reward.

    Returns `(terminal_vec[K], spell_z7[4], focal_ability_idxs[4])`:
    - `terminal_vec` — StatsModel(final loadouts) → focal stats (z-space). Outputs
      are already z-normalized, so we return them directly (the realized anchor
      targets normalize raw→z into the same space).
    - `spell_z7` — each of the focal's 4 abilities' predicted DIRECT spell damage,
      expressed in `hero_damage` (dim-7) z-units: `(z_spell·σ_spell+μ_spell)/σ_hd`.
      Aligned with `focal_ability_idxs`. Used to densify the dim-7 reward per pick.
    - `focal_ability_idxs` — the focal's 4 ability vocab indices (= spell-head slot
      order), for mapping each focal pick to its spell_z7 entry.
    """
    record = _build_stats_record(state, vocabs, mmr_mean, mmr_std)
    batch = stats_collate([record], device=device)
    with torch.no_grad():
        outs = stats_model(batch)
    focal_t = torch.tensor([focal_slot], dtype=torch.long, device=device)
    raw = torch.empty(K_STATS)
    for k, spec in enumerate(STAT_SPECS):
        raw[k] = float(spec.pred_fn(outs, focal_t).cpu().item())

    sd_mean = float(stats_model.spell_damage_mean)
    sd_std = float(stats_model.spell_damage_std)
    sigma_hd = float(stats_model.scalar_std[HERO_DAMAGE_DIM])
    z_spell = outs[9][0, focal_slot].cpu()                 # [4], spell-head z-output
    spell_z7 = (z_spell * sd_std + sd_mean) / sigma_hd     # [4], in dim-7 z-units
    focal_ability_idxs = [int(x) for x in record.ability_indices[focal_slot].tolist()]
    return raw, spell_z7, focal_ability_idxs


def scalarize_q(q_vec: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """q_vec: [..., V, K]  →  scalar Q over actions: [..., V].

    Infeasible vocab rows are -inf across all K (the QNetStats forward sets
    them that way). Naive `(q_vec * weights).sum(-1)` produces NaN at
    infeasible rows when any weight is exactly 0 (−inf * 0 = NaN), and
    argmax of NaN is undefined. We detect infeasible rows via the first K
    coordinate (all K are -inf together) and re-mask back to -inf after
    the dot product.
    """
    infeas = q_vec[..., 0].isneginf()
    scalar = (q_vec * weights).sum(dim=-1)
    return scalar.masked_fill(infeas, float("-inf"))


def epsilon_greedy_stats(
    qnet: QNetStats,
    sample: PolicySample,
    weights: torch.Tensor,   # [K] on device
    device: torch.device,
    epsilon: float,
    rng: random.Random,
    policy: BehaviorPolicy | None = None,
    bc_mask_frac: float = 0.0,
) -> UnifiedIdx:
    """ε-greedy over feasible actions: argmax of composite-scalar Q.

    With `bc_mask_frac > 0` and a BC policy, the choice set is HARD-restricted to
    BC-plausible actions — those with π(a|s) ≥ bc_mask_frac × uniform — so the
    focal never explores or learns to value rare stat-padders. This is the hard
    analog of soft CQL (orthogonal to the CQL term; either, both, or neither).
    The set can't go empty for frac ≤ 1: max π ≥ uniform ≥ frac × uniform.

    Note this only silences the focal's SIMULATED stream — the real random-pick
    anchors (anchor_frac of every batch) still carry rare actions, so rare-Q
    keeps getting direct gradient from real outcomes regardless of the mask.
    """
    feasible = list(sample.cand_idx)
    masked = policy is not None and bc_mask_frac > 0.0 and len(feasible) > 1
    if masked:
        assert policy is not None
        with torch.no_grad():
            p = policy(policy_collate([sample], device=device)).squeeze(0).exp().cpu()
        # Keep exactly the BC-plausible actions, thresholding the FEASIBLE-
        # renormalized π (the UI rare-tag convention; p/z ≥ frac/n ⟺ p ≥
        # frac·z/n). Raw π can leave the set empty where the V+1 BC parks its
        # mass on the random class (timeout-like states); renormalized,
        # max π ≥ uniform ≥ frac×uniform for frac ≤ 1 — never empty.
        z = sum(float(p[int(a)]) for a in feasible)
        thresh = bc_mask_frac * z / len(feasible)
        feasible = [a for a in feasible if float(p[int(a)]) >= thresh]
    if rng.random() < epsilon:
        return rng.choice(feasible)
    batch = policy_collate([sample], device=device)
    with torch.no_grad():
        q_vec = qnet(batch).squeeze(0)        # [V, K]; infeas at -inf
        q_scalar = scalarize_q(q_vec, weights)  # [V]; infeas at -inf
    if masked:
        kept = torch.full_like(q_scalar, float("-inf"))
        idx = torch.tensor([int(a) for a in feasible], device=q_scalar.device)
        kept[idx] = q_scalar[idx]
        q_scalar = kept
    return UnifiedIdx(int(torch.argmax(q_scalar).item()))


def sample_non_focal_action(
    policy: BehaviorPolicy,
    sample: PolicySample,
    device: torch.device,
    rng: random.Random,
    random_class_idx: int,
) -> tuple[UnifiedIdx, bool]:
    """Sample a non-focal action from the BC (V+1 output).

    If the random class is sampled, draw the forced pick from the true timeout
    mechanism P_mech (side-coin then uniform over heroes∪side; see
    dota2ad.core.mechanism / experiments/random-mechanism) and report
    is_random=True. Otherwise return the sampled action with is_random=False.
    """
    batch = policy_collate([sample], device=device)
    with torch.no_grad():
        log_probs = policy(batch).squeeze(0)     # [V + 1]; infeas at -inf
    probs = log_probs.exp()
    # CUDA multinomial can pick zero-prob entries due to small numerical
    # noise; restrict to feasible-or-random indices explicitly.
    feasible_set = set(int(c) for c in sample.cand_idx)
    feasible_set.add(random_class_idx)
    mask = torch.zeros_like(probs)
    for i in feasible_set:
        mask[i] = 1.0
    probs = probs * mask
    s = probs.sum()
    assert s > 0, f"BC produced zero mass on feasibles+random for sample turn {sample.turn}"
    probs = probs / s
    seed = rng.getrandbits(63)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    sampled = int(torch.multinomial(probs, 1, generator=g).item())
    if sampled == random_class_idx:
        # Server timeout: draw from the true forced-pick mechanism P_mech (not uniform)
        action = UnifiedIdx(sample_mechanism_pick(rng, sample.cand_idx, sample.cand_type))
        return action, True
    return UnifiedIdx(sampled), False


def sample_focal_bc(
    policy: BehaviorPolicy,
    sample: PolicySample,
    device: torch.device,
    epsilon: float,
    rng: random.Random,
) -> UnifiedIdx:
    """Focal action under BC — the continuation for the Q^BC variant
    (`step_episode_vec(..., focal_continuation="bc")`). The focal's own picks
    follow BC over the feasible set (the timeout/random class is renormalized
    out — a focal's *other* picks in a real game are deliberate ≈ BC), matching
    the per-pick eval's BC-continuation. ε of the time take a uniform-random
    feasible action so the Q still gets coverage to rank with.
    """
    feasible = list(sample.cand_idx)
    if rng.random() < epsilon:
        return UnifiedIdx(rng.choice(feasible))
    with torch.no_grad():
        probs = policy(policy_collate([sample], device=device)).squeeze(0).exp().cpu()
    weights = [max(float(probs[int(a)]), 0.0) for a in feasible]
    r = rng.random() * (sum(weights) or 1.0)
    acc = 0.0
    for a, w in zip(feasible, weights, strict=True):
        acc += w
        if r <= acc:
            return UnifiedIdx(int(a))
    return UnifiedIdx(int(feasible[-1]))


def step_episode_vec(
    match: MatchRow,
    focal_slot: PickSlot,
    qnet: QNetStats,
    policy: BehaviorPolicy,                  # V+1 output BC
    stats_model: StatsModel | EnsembleStatsModel,
    vocabs: Vocabs,
    mmr_mean: float,
    mmr_std: float,
    weights: torch.Tensor,
    epsilon: float,
    device: torch.device,
    rng: random.Random,
    bc_mask_frac: float = 0.0,
    focal_continuation: str = "policy",   # "policy" (Q^π, on-policy) | "bc" (Q^BC, eval-matched)
) -> list[Transition]:
    """Run one focal-POV episode, return its semi-MDP transitions.

    The dim-7 (hero_damage) reward is DENSE: each focal ability pick is credited
    its predicted DIRECT spell damage (in dim-7 z-units); the unattributable
    remainder (attacks/items) is paid at the terminal. Every other dim is
    terminal-only. The per-pick dim-7 rewards sum to the terminal dim-7 value, so
    the redistribution is return-equivalent under γ=1. Non-focal seats sample
    from BC including the random class — when sampled, a server-uniform draw
    fills the position with is_random=True.
    """
    random_class_idx = policy.vocab_size

    state = initial_state_from_match(match)
    focal_steps: list[tuple[PolicySample, UnifiedIdx]] = []

    while state.turn < 50:
        sample = encode_policy_sample(
            state.to_row(), vocabs, state.history, mmr_mean, mmr_std,
        )
        if state.pick_slot == focal_slot:
            if focal_continuation == "bc":
                # Q^BC variant: the focal continues with BC (matched to the
                # per-pick eval's BC-continuation), not its own policy.
                action = sample_focal_bc(policy, sample, device, epsilon, rng)
            else:
                action = epsilon_greedy_stats(
                    qnet, sample, weights, device, epsilon, rng,
                    policy=policy, bc_mask_frac=bc_mask_frac,
                )
            focal_steps.append((sample, action))
            is_random = False
        else:
            action, is_random = sample_non_focal_action(
                policy, sample, device, rng, random_class_idx,
            )
        state = make_forced_state(state, action, vocabs, is_random=is_random)

    assert len(focal_steps) == 5, f"focal picked {len(focal_steps)} times, expected 5"
    terminal_vec, spell_z7, focal_abils = compute_terminal_reward_vec(
        state, focal_slot, stats_model, vocabs, mmr_mean, mmr_std, device,
    )
    return _build_episode_transitions(focal_steps, terminal_vec, spell_z7, focal_abils)


# ---------------------------------------------------------------------------
# Batched rollout: run B episodes in lockstep with batch-B forwards (vs
# step_episode_vec's batch-1 per turn). Statistically equivalent to running
# step_episode_vec B times — only the RNG consumption order differs — but it
# lifts the GPU from launch-overhead-bound (~34%) toward compute-bound.
# ---------------------------------------------------------------------------

def _build_episode_transitions(
    focal_steps: list[tuple[PolicySample, UnifiedIdx]],
    terminal_vec: torch.Tensor, spell_z7: torch.Tensor, focal_abils: list[int],
) -> list[Transition]:
    """Assemble one episode's semi-MDP transitions from its focal picks + terminal
    (dense dim-7 spell reward per pick; everything else terminal-only)."""
    spell_sum = float(spell_z7.sum())
    n = len(focal_steps)
    transitions: list[Transition] = []
    for i, (sample, action) in enumerate(focal_steps):
        is_term = i == n - 1
        slot = focal_abils.index(int(action)) if int(action) in focal_abils else -1
        spell = float(spell_z7[slot]) if slot >= 0 else 0.0
        if is_term:
            reward = terminal_vec.clone()
            reward[HERO_DAMAGE_DIM] = terminal_vec[HERO_DAMAGE_DIM] - spell_sum + spell
            next_sample = None
        else:
            reward = torch.zeros(K_STATS)
            reward[HERO_DAMAGE_DIM] = spell
            next_sample = focal_steps[i + 1][0]
        transitions.append(Transition(
            sample=sample, action_idx=action, reward=reward, next_sample=next_sample))
    return transitions


def epsilon_greedy_stats_batch(
    qnet: QNetStats, samples: list[PolicySample], weights: torch.Tensor,
    device: torch.device, epsilons: list[float], rng: random.Random,
    policy: BehaviorPolicy | None = None, bc_mask_frac: float = 0.0,
) -> list[UnifiedIdx]:
    """Batched `epsilon_greedy_stats` over a list of focal samples."""
    if not samples:
        return []
    batch = policy_collate(samples, device=device)
    with torch.no_grad():
        q_scalar = scalarize_q(qnet(batch), weights).cpu()          # [B, V]
    plausible: list[list[int] | None] = [None] * len(samples)
    if policy is not None and bc_mask_frac > 0.0:
        with torch.no_grad():
            p = policy(batch).exp().cpu()                           # [B, V+1]
        for i, s in enumerate(samples):
            feas = [int(a) for a in s.cand_idx]
            if len(feas) > 1:
                z = sum(float(p[i, a]) for a in feas)
                thr = bc_mask_frac * z / len(feas)
                plausible[i] = [a for a in feas if float(p[i, a]) >= thr]
    actions: list[UnifiedIdx] = []
    for i, s in enumerate(samples):
        pl = plausible[i]
        cand = pl if pl is not None else [int(a) for a in s.cand_idx]
        if rng.random() < epsilons[i]:
            actions.append(UnifiedIdx(rng.choice(cand)))
            continue
        row = q_scalar[i]
        if pl is not None:
            kept = torch.full_like(row, float("-inf"))
            kept[torch.tensor(pl)] = row[torch.tensor(pl)]
            row = kept
        actions.append(UnifiedIdx(int(torch.argmax(row).item())))
    return actions


def sample_non_focal_batch(
    policy: BehaviorPolicy, samples: list[PolicySample], device: torch.device,
    rng: random.Random, random_class_idx: int,
) -> tuple[list[UnifiedIdx], list[bool]]:
    """Batched `sample_non_focal_action`: one policy forward + one multinomial for
    all B non-focal seats; each random-class draw resolves via `sample_mechanism_pick`."""
    if not samples:
        return [], []
    batch = policy_collate(samples, device=device)
    with torch.no_grad():
        probs = policy(batch).exp()                                # [B, V+1]
    mask = torch.zeros_like(probs)
    for i, s in enumerate(samples):
        for a in s.cand_idx:
            mask[i, int(a)] = 1.0
        mask[i, random_class_idx] = 1.0
    probs = probs * mask
    probs = probs / probs.sum(dim=1, keepdim=True)
    g = torch.Generator(device=device)
    g.manual_seed(rng.getrandbits(63))
    sampled = torch.multinomial(probs, 1, generator=g).squeeze(1).cpu()   # [B]
    actions: list[UnifiedIdx] = []
    is_rand: list[bool] = []
    for i, s in enumerate(samples):
        a = int(sampled[i].item())
        if a == random_class_idx:
            actions.append(UnifiedIdx(sample_mechanism_pick(rng, s.cand_idx, s.cand_type)))
            is_rand.append(True)
        else:
            actions.append(UnifiedIdx(a))
            is_rand.append(False)
    return actions, is_rand


def compute_terminal_reward_batch(
    states: list[DraftState], focal_slots: list[PickSlot],
    stats_model: StatsModel | EnsembleStatsModel, vocabs: Vocabs,
    mmr_mean: float, mmr_std: float, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Batched `compute_terminal_reward_vec` (one StatsModel forward for all B
    final drafts). Returns (raw[B,K], spell_z7[B,4], focal_abils per episode)."""
    B = len(states)
    records = [_build_stats_record(s, vocabs, mmr_mean, mmr_std) for s in states]
    batch = stats_collate(records, device=device)
    with torch.no_grad():
        outs = stats_model(batch)
    fs_t = torch.tensor([int(f) for f in focal_slots], dtype=torch.long, device=device)
    raw = torch.stack([spec.pred_fn(outs, fs_t).cpu() for spec in STAT_SPECS], dim=1)  # [B, K]
    sd_mean = float(stats_model.spell_damage_mean)
    sd_std = float(stats_model.spell_damage_std)
    sigma_hd = float(stats_model.scalar_std[HERO_DAMAGE_DIM])
    z_spell = outs[9][torch.arange(B, device=device), fs_t].cpu()          # [B, 4]
    spell_z7 = (z_spell * sd_std + sd_mean) / sigma_hd
    focal_abils = [[int(x) for x in records[i].ability_indices[int(focal_slots[i])].tolist()]
                   for i in range(B)]
    return raw, spell_z7, focal_abils


def step_episode_batch(
    matches: list[MatchRow], focal_slots: list[PickSlot],
    qnet: QNetStats, policy: BehaviorPolicy,
    stats_model: StatsModel | EnsembleStatsModel, vocabs: Vocabs,
    mmr_mean: float, mmr_std: float, weights: torch.Tensor,
    epsilons: list[float], device: torch.device, rng: random.Random,
    bc_mask_frac: float = 0.0, focal_continuation: str = "policy",
) -> list[list[Transition]]:
    """Run B focal-POV episodes in lockstep, batching every per-turn forward.
    The snake-draft pick order is fixed, so all B drafts share the turn's pick_slot;
    each turn splits into the focal seats (argmax-Q, batched) and non-focal seats
    (BC sample, batched). Returns B transition-lists, each identical in form to
    `step_episode_vec`'s output."""
    B = len(matches)
    random_class_idx = policy.vocab_size
    states = [initial_state_from_match(m) for m in matches]
    focal_steps: list[list[tuple[PolicySample, UnifiedIdx]]] = [[] for _ in range(B)]

    while states[0].turn < 50:
        samples = [encode_policy_sample(s.to_row(), vocabs, s.history, mmr_mean, mmr_std)
                   for s in states]
        focal_ids = [i for i in range(B) if int(states[i].pick_slot) == int(focal_slots[i])]
        focal_set = set(focal_ids)
        nonfocal_ids = [i for i in range(B) if i not in focal_set]
        actions: list[UnifiedIdx | None] = [None] * B
        is_rand = [False] * B

        if nonfocal_ids:
            acts, rands = sample_non_focal_batch(
                policy, [samples[i] for i in nonfocal_ids], device, rng, random_class_idx)
            for k, i in enumerate(nonfocal_ids):
                actions[i] = acts[k]; is_rand[i] = rands[k]
        if focal_ids:
            if focal_continuation == "bc":
                acts = [sample_focal_bc(policy, samples[i], device, epsilons[i], rng)
                        for i in focal_ids]
            else:
                acts = epsilon_greedy_stats_batch(
                    qnet, [samples[i] for i in focal_ids], weights, device,
                    [epsilons[i] for i in focal_ids], rng,
                    policy=policy, bc_mask_frac=bc_mask_frac)
            for k, i in enumerate(focal_ids):
                actions[i] = acts[k]; focal_steps[i].append((samples[i], acts[k]))
        for i in range(B):
            a_i = actions[i]
            assert a_i is not None
            states[i] = make_forced_state(states[i], a_i, vocabs, is_random=is_rand[i])

    for i in range(B):
        assert len(focal_steps[i]) == 5, f"ep {i}: {len(focal_steps[i])} focal picks"
    raw, spell_z7, focal_abils = compute_terminal_reward_batch(
        states, focal_slots, stats_model, vocabs, mmr_mean, mmr_std, device)
    return [_build_episode_transitions(focal_steps[i], raw[i], spell_z7[i], focal_abils[i])
            for i in range(B)]


def compute_stat_norm(
    stats_rows_by_id, focal_slots_iter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-stat (μ, σ) over realized training stats. `focal_slots_iter`
    yields (match_id, focal_slot) pairs to iterate; each contributes one
    K-vector sample. Returns ([K], [K]) tensors on CPU."""
    samples: list[torch.Tensor] = []
    for match_id, focal_slot in focal_slots_iter:
        if match_id not in stats_rows_by_id:
            continue
        row = stats_rows_by_id[match_id]
        v = torch.empty(K_STATS)
        for k, spec in enumerate(STAT_SPECS):
            v[k] = float(spec.real_fn(row, int(focal_slot)))
        samples.append(v)
    assert samples, "no stats rows matched the focal_slots_iter for normalization"
    stacked = torch.stack(samples, dim=0)        # [N, K]
    mean = stacked.mean(dim=0)                   # [K]
    std = stacked.std(dim=0).clamp(min=1e-6)     # [K]
    return mean, std
