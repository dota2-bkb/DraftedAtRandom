"""State-rarity confidence band from the BC draft density.

How *typical the whole draft* is — an unusual draft is where the StatsModel
extrapolates (validated: state support predicts held-out error; action rarity
does NOT). We measure typicality by the BC policy's per-pick average surprisal
along the realized draft path, `support = -log p(s)/T = mean_t -log P(pick_t |
history_t)`, and bucket it against held-out support quantiles into a
high/medium/low confidence band.

The quantiles are calibrated at train time and ride on the policy checkpoint
(`policy.density_support_q`, written by experiments/train-policy via
`compute_support_quantiles`) — so they can never go stale against the policy
that produced them. The per-pick action rarity is surfaced separately in the UI
as the "rare here" tag, not folded in here. The composite-error evidence that
this signal predicts error lives in experiments/stats-density-validate.
"""

from __future__ import annotations

from typing import Literal

import torch

Band = Literal["high", "medium", "low"]

from dota2ad.core import (
    PickSlot, Turn, UnifiedIdx, VocabKey, encode_policy_sample, policy_collate,
)
from dota2ad.core.draft_logic import turn_to_pick_slot
from dota2ad.core.types import PlayerPickState
from dota2ad.suggest.state import DraftState as RolloutState, make_forced_state


def state_support(state, policy, vocabs, mmr_mean: float, mmr_std: float,
                  device: torch.device) -> float | None:
    """Per-pick average surprisal `-log p(s)/T` for the live draft (higher =
    rarer draft). Replays the history through BC from the original pools.
    Returns None before any pick (no trajectory yet)."""
    T = int(state.turn)
    if T == 0 or not state.history:
        return None
    cur = RolloutState(
        turn=Turn(0), pick_slot=turn_to_pick_slot(Turn(0)),
        player_picks={PickSlot(ps): PlayerPickState(hero=None, basics=[], ult=None) for ps in range(10)},
        hero_pool_remaining=[h for h in state.hero_pool_all if h is not None],
        basic_pool_remaining=[a for a in state.basic_pool_all if a is not None],
        ult_pool_remaining=[a for a in state.ult_pool_all if a is not None],
        action_key="", mmr=state.mmr, history=[],
    )
    cum = 0.0
    for ev in state.history:
        sample = encode_policy_sample(cur.to_row(), vocabs, cur.history, mmr_mean, mmr_std)
        key = f"h:{ev.hero_id}" if ev.hero_id is not None else f"a:{ev.draft_ability_id}"
        a = int(vocabs.draft_id_to_index[VocabKey(key)])
        with torch.no_grad():
            cum += float(policy(policy_collate([sample], device=device)).squeeze(0).cpu()[a])
        cur = make_forced_state(cur, UnifiedIdx(a), vocabs, is_random=ev.is_random)
    return -cum / T


def confidence_band(support: float | None, support_q: list[float] | None) -> Band | None:
    """high/medium/low CONFIDENCE band from how typical the draft is. Higher
    support = rarer draft = lower confidence. Thresholds are the held-out support
    quantiles (support_q = [10,30,50,70,90]%): ≤30th → high, ≥70th → low."""
    if support is None or not support_q or len(support_q) < 5:
        return None
    if support >= support_q[3]:
        return "low"
    if support <= support_q[1]:
        return "high"
    return "medium"


def state_confidence(state, policy, vocabs, mmr_mean: float, mmr_std: float,
                     device: torch.device) -> tuple[float | None, Band | None]:
    """Live entry point: (state_support, confidence band) for the draft. The band
    quantiles ride on the policy checkpoint (`policy.density_support_q`); both are
    None before the first pick or if the policy carries no calibration."""
    support = state_support(state, policy, vocabs, mmr_mean, mmr_std, device)
    return support, confidence_band(support, getattr(policy, "density_support_q", None))


# --- offline calibration (used by train-policy + stats-density-validate) ------

def path_supports(match, policy, vocabs, mmr_mean: float, mmr_std: float,
                  device: torch.device) -> list[float]:
    """Per-turn support `-log p(s_t)/t` for every prefix of a completed match;
    `out[t]` is the support at turn `t+1` (after `t+1` picks)."""
    from dota2ad.training.stats_simulator import initial_state_from_match
    state = initial_state_from_match(match)
    cum = 0.0
    out: list[float] = []
    for t in range(min(50, len(match.history))):
        sample = encode_policy_sample(state.to_row(), vocabs, state.history, mmr_mean, mmr_std)
        ev = match.history[t]
        key = f"h:{ev.hero_id}" if ev.hero_id is not None else f"a:{ev.draft_ability_id}"
        a = int(vocabs.draft_id_to_index[VocabKey(key)])
        with torch.no_grad():
            cum += float(policy(policy_collate([sample], device=device)).squeeze(0).cpu()[a])
        out.append(-cum / (t + 1))
        state = make_forced_state(state, UnifiedIdx(a), vocabs, is_random=ev.is_random)
    return out


def compute_support_quantiles(matches, policy, vocabs, mmr_mean: float, mmr_std: float,
                              device: torch.device,
                              quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)) -> list[float]:
    """Empirical quantiles of per-turn state support over completed matches — the
    band calibration baked into the policy checkpoint. BC policy only (no
    StatsModel, no outcomes). Use held-out matches so the policy isn't overconfident
    on its own training drafts."""
    vals: list[float] = []
    for m in matches:
        vals.extend(path_supports(m, policy, vocabs, mmr_mean, mmr_std, device))
    vals.sort()
    n = len(vals)
    return [vals[int(q * (n - 1))] for q in quantiles]
