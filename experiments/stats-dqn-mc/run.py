"""Monte-Carlo stats-DQN trainer — retains the DQN/semi-MDP, swaps TD→MC.

The bootstrap (TD(0)) trainer collapses Q's *across-action* signal — Q learns
the per-state mean (`--n-step 1` reproduces it; see README.md).
The across-action term `Var_a[E(target|s,a)]` is the signal a ranker needs,
and a single-sample TD bootstrap buries it under next-state noise. Monte-Carlo
returns carry the across-action signal by construction (forcing action a
changes the rollout's terminal stats); the regression averages the
within-action noise across episodes.

Same structure as the bootstrap trainer: focal is the agent, the other 9 are
the environment (BC-filled), transitions are focal-to-focal. ONLY the target
changes — from `r + γ·max Q(s',a')` to the Monte-Carlo return. With γ=1 and
terminal-only reward, the return for *every* focal pick in an episode is just
the terminal StatsModel stats vector. No target network, no bootstrap.

Two MC sample sources:
  - Simulated episodes (step_episode_vec): focal ε-greedy on Q (or BC with
    --focal-continuation bc), others BC, terminal stats reward → return assigned
    to all focal transitions.
  - Real random-pick anchors (free): Q(s, a_realised) → realised stats. These
    are real MC samples with exogenous, uniformly-distributed actions and a
    real continuation — broad coverage + causal grounding.

Rare-action handling (the shipped default): random-pick anchors give every
BC-rare action a handful of real-outcome gradient hits, and the max over many
such noisy estimates floats a few to the top of the ranking (winner's curse) — a
pathway removing anchors can't fix. Two orthogonal defenses run by default:
  - --bc-mask-frac (data-side, default 0.33): hard-restrict the focal's choice
    set to BC-plausible actions (π ≥ frac×uniform), so it never explores or
    values rare stat-padders.
  - --usup-alpha (loss-side, default 0.3): uniform-support CQL, logsumexp_a Q −
    mean_{a∈plausible} Q. Keeps standard CQL's stability certificates (zero-sum
    gradient across actions, shift-invariance, pressure decaying to zero) but
    replaces the π_BC target with uniform-over-plausible, so popularity never
    ranks the plausible set (standard CQL's BC-cloning failure).
Standard CQL (--cql-alpha) is off by default — it clones BC. usup is the
popularity-free replacement; its known cost is order-preserving compression of
the within-plausible spread at rate α.

Checkpoint selection: at each post-anneal eval we track the argmax of the
val-split mean per-stat gap t; in parallel we accumulate an SWA average of the
online weights (--swa-every). After training the two compete on the val metric
and the winner ships — SWA smooths the ±1z per-snapshot noise the argmax can
lock onto. Selection burns the val split only; the test split is never loaded
here, so it stays clean for the held-out test eval (stats-causal-rank).

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python -u -m \\
    experiments/stats-dqn-mc/run.py --num-episodes 12000 --eval-every 1000
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from dota2ad.core import (
    NUM_PLAYERS,
    compute_mmr_norm,
    default_paths,
    load_matches,
    load_split,
    load_stats_rows,
    load_vocabs,
)
from dota2ad.core.collate import policy_collate
from dota2ad.core.types import PickSlot, PolicySample, UnifiedIdx
from dota2ad.models import QNetStats, load_policy, load_stats_model
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.stats_specs import STAT_SPECS
from dota2ad.training.stats_diagnostics import (
    print_summary,
    report_per_stat_gaps,
)
from dota2ad.eval.stats_eval import (
    compute_realized_y_vec,
    evaluate_q1q4_stat,
    evaluate_q1q4_win,
)
from dota2ad.training.stats_simulator import (
    compute_stat_norm, scalarize_q, step_episode_batch,
)
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS, K_STATS


@dataclass
class StepSample:
    """One focal transition with its n-step target spec.

    If `bootstrap_sample is None`, the n-step window reached the terminal
    (or this is a real-random-pick anchor) → target is `terminal_return`.
    Otherwise the target bootstraps: γ^n · Q(bootstrap_sample, argmax).
    Pure MC (n ≥ focal picks per episode) → bootstrap_sample always None.
    """
    sample: PolicySample
    action_idx: UnifiedIdx
    terminal_return: torch.Tensor          # [K]; used when bootstrap_sample is None
    bootstrap_sample: PolicySample | None  # focal state n picks ahead, else None
    gamma_pow: float                        # γ^n applied to the bootstrap value


def _focal_prefix_spell_z7(t, row, idx_to_key, sigma_hd) -> float:
    """Realized direct spell damage of the focal's ALREADY-picked abilities at
    the random pick (raw, summed over enemies), in hero_damage(dim-7) z-units.
    The dim-7 reward is dense (each pick credited its spell damage), so the
    anchor target must be the damage from this pick onward = full-game total −
    this prefix, matching the simulator's suffix-sum semantics."""
    drafted = list(row.ability_draft_ids[t.focal_slot])
    sd = row.spell_damage_dealt[t.focal_slot]
    prefix_raw = 0.0
    for vidx in t.sample.loadouts[t.focal_slot]:
        key = idx_to_key[vidx]
        if not key.startswith("a:"):
            continue                       # skip the hero
        aid = int(key[2:])
        if aid in drafted:
            slot = drafted.index(aid)
            if sd and slot < len(sd):
                prefix_raw += float(sum(sd[slot]))
    return prefix_raw / sigma_hd


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    # The never-empty plausible-set guarantee (renormalized max π ≥ frac/n)
    # holds only for frac ≤ 1.
    assert 0.0 <= args.bc_mask_frac <= 1.0
    assert 0.0 < args.usup_frac <= 1.0

    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  num_episodes={args.num_episodes}  anchors={not args.no_anchors}")

    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    val_matches = [m for m in matches if m.match_id in split.val_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    print(f"Train: {len(train_matches)} matches, Val: {len(val_matches)} matches, vocab={vocab_size}")

    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    stats_model = load_stats_model(paths.stats_ckpt, vocabs, device)
    stats_model.requires_grad_(False)

    stats_rows = load_stats_rows()
    stats_rows_by_id = {r.match_id: r for r in stats_rows}
    print(f"Loaded {len(stats_rows_by_id)} StatsRows")

    print("Computing per-stat normalization...")
    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    stat_norm_mean, stat_norm_std = compute_stat_norm(stats_rows_by_id, _iter())

    # Selection tuples: the val split's forced picks. Selection burns val only;
    # the test split is never loaded here — test-split numbers come from
    # stats-causal-rank on test.
    val_tuples = [t for t in extract_tuples(val_matches, vocabs, mmr_mean, mmr_std)
                  if t.match_id in stats_rows_by_id]
    print(f"  val selection tuples: {len(val_tuples)}")

    # Real random-pick MC anchors (train split): (sample, a_realised, realised stats).
    # Always "terminal" style (no bootstrap) — real outcome, exogenous action.
    anchor_samples: list[StepSample] = []
    if not args.no_anchors:
        train_tuples = [t for t in extract_tuples(train_matches, vocabs, mmr_mean, mmr_std)
                        if t.match_id in stats_rows_by_id]
        ys = compute_realized_y_vec(train_tuples, stats_rows_by_id, stat_norm_mean, stat_norm_std)
        idx_to_key = {v: k for k, v in vocabs.draft_id_to_index.items()}
        sigma_hd = float(stat_norm_std[7])
        for t, y in zip(train_tuples, ys, strict=True):
            if y is not None:
                # Dim-7 dense reward ⇒ anchor target is suffix (this pick onward),
                # so subtract the focal's already-picked abilities' spell damage.
                prefix = _focal_prefix_spell_z7(t, stats_rows_by_id[t.match_id], idx_to_key, sigma_hd)
                y = y.clone()
                y[7] = y[7] - prefix
                anchor_samples.append(StepSample(t.sample, t.action_idx, y, None, 1.0))
        print(f"  real random-pick MC anchors: {len(anchor_samples)}")

    online = QNetStats.warm_start_from_policy(paths.policy_ckpt, vocab_size, K_STATS, device)
    online.mmr_mean.fill_(mmr_mean)
    online.mmr_std.fill_(mmr_std)
    optimizer = torch.optim.AdamW(online.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Target network: engaged when n-step actually bootstraps (n < focal picks
    # per episode = 5). Pure MC (n ≥ 5) never uses it.
    import copy
    target_net = copy.deepcopy(online)
    target_net.requires_grad_(False)
    target_net.eval()
    bootstraps = args.n_step < 5   # 5 focal picks/episode; n≥5 → pure MC

    weights_cpu = DEFAULT_BALANCED_WEIGHTS
    weights_device = weights_cpu.to(device)

    buffer: deque[StepSample] = deque(maxlen=args.replay_capacity)
    eps_end = max(1, int(args.epsilon_end_frac * args.num_episodes))
    best_eval = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    # Selection metric: mean per-stat Q1−Q4 gap t over the balanced dims
    # (averages 8 dims → far less noisy than the single composite lift).
    balanced_dims = (0, 1, 3, 4, 5, 7, 8, 11)
    # SWA: running average of the online weights over the post-anneal window, an
    # alternative to the argmax snapshot that smooths the ±1z per-snapshot val
    # noise. Compared against the argmax on the val metric after training; the
    # winner ships. (No BatchNorm in the Q-net, so averaged weights need no recal.)
    swa_state: dict[str, torch.Tensor] | None = None
    swa_n = 0
    loss_ema: float | None = None
    cql_ema: float | None = None
    usup_ema: float | None = None
    t0 = time.monotonic()

    def _polyak():
        with torch.no_grad():
            for tp, op in zip(target_net.parameters(), online.parameters(), strict=True):
                tp.mul_(1.0 - args.tau).add_(op.data, alpha=args.tau)

    def _train_step():
        nonlocal loss_ema
        if len(buffer) < max(args.warmup, args.batch_size):
            return
        online.train()
        batch = rng.sample(buffer, args.batch_size)
        if anchor_samples:
            n_anchor = max(1, int(args.batch_size * args.anchor_frac))
            batch = batch[: args.batch_size - n_anchor] + rng.sample(anchor_samples, n_anchor)
        b = policy_collate([s.sample for s in batch], device=device)
        actions = torch.tensor([int(s.action_idx) for s in batch], dtype=torch.long, device=device)
        q_all = online(b)                                                     # [B, V, K]
        B = q_all.shape[0]
        q_sa = q_all[torch.arange(B, device=device), actions]                 # [B, K]

        # Target: terminal_return by default; n-step bootstrap where applicable.
        targets = torch.stack([s.terminal_return for s in batch]).to(device)  # [B, K]
        boot_pairs = [(i, s.bootstrap_sample) for i, s in enumerate(batch)
                      if s.bootstrap_sample is not None]
        if boot_pairs:
            boot_idx = [i for i, _ in boot_pairs]
            boot = [bs for _, bs in boot_pairs]
            bb = policy_collate(boot, device=device)
            with torch.no_grad():
                q_next_online = online(bb)                                    # [M, V, K]
                scalar_next = scalarize_q(q_next_online, weights_device)      # [M, V]
                a_star = scalar_next.argmax(dim=1)                            # [M]
                q_next_target = target_net(bb)                               # [M, V, K]
                M = q_next_target.shape[0]
                boot_vals = q_next_target[torch.arange(M, device=device), a_star]  # [M, K]
            gpow = torch.tensor([batch[i].gamma_pow for i in boot_idx], device=device).unsqueeze(1)
            # terminal_return holds the partial reward sum (i..j-1); add the
            # bootstrapped tail. (Pure MC has no bootstrap, so this is a no-op there.)
            targets[boot_idx] = targets[boot_idx] + gpow * boot_vals

        mc_loss = F.smooth_l1_loss(q_sa, targets)

        infeas = q_all[..., 0].isneginf()                                    # [B, V]
        bc_probs: torch.Tensor | None = None
        if args.cql_alpha > 0 or args.usup_alpha > 0:
            with torch.no_grad():
                # Raw softmax over V (random class dropped, NOT renormalized) —
                # the same probability the mask and UI rare-tag threshold against.
                bc_probs = policy(b).exp()[:, :vocab_size]                   # [B, V]

        # CQL-style conservatism: push Q down for OOD (low-BC) actions, up for
        # BC-supported ones, per stat. The push-up is ∝ π_BC, so it ranks the
        # plausible set by popularity (the BC-cloning failure mode) — off by
        # default; usup is the popularity-free replacement.
        cql = torch.zeros((), device=device)
        if args.cql_alpha > 0:
            assert bc_probs is not None
            probs = bc_probs * (~infeas).float()
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
            # logsumexp over feasible actions per stat (−inf entries drop out)
            logsumexp_term = torch.logsumexp(q_all, dim=1)                   # [B, K]
            q_safe = torch.where(infeas.unsqueeze(-1), torch.zeros_like(q_all), q_all)
            data_term = (probs.unsqueeze(-1) * q_safe).sum(dim=1)            # [B, K]
            cql = (logsumexp_term - data_term).mean()

        # Uniform-support CQL: logsumexp_a Q − mean_{a∈plausible} Q, per stat.
        # Same convex, zero-sum, shift-invariant shape as standard CQL — so it
        # cannot drift the global level (the runaway a one-sided penalty causes) —
        # but the data term is UNIFORM over the BC-plausible set instead of
        # π_BC-weighted: 10% and 30% picks get the same push-up, so popularity
        # never ranks the plausible set. Rare actions (u=0) only ever feel the
        # logsumexp push-down, ∝ softmax(Q) — strongest exactly on the
        # winner's-curse escapees at the top of the ranking, decaying to zero
        # as they sink. Cost: order-preserving compression of within-plausible
        # spread at rate α (gradient softmax−u shrinks gaps, never flips them).
        usup = torch.zeros((), device=device)
        if args.usup_alpha > 0:
            assert bc_probs is not None
            feas = ~infeas
            n_feas = feas.sum(dim=1, keepdim=True).clamp(min=1)              # [B, 1]
            # Plausibility thresholds the FEASIBLE-renormalized π (the UI
            # rare-tag convention): "what a human takes here, conditional on
            # picking deliberately". Raw π fails at anchor states — they ARE
            # timeout states, where the V+1 BC parks most mass on the random
            # class, so no action clears frac/n raw. Renormalized, max π ≥
            # 1/n ≥ frac/n for frac ≤ 1 — the plausible set is never empty.
            p_feas = bc_probs * feas.float()
            p_feas = p_feas / p_feas.sum(dim=1, keepdim=True).clamp(min=1e-8)
            plaus = feas & (p_feas >= args.usup_frac / n_feas)               # [B, V]
            u = plaus.float() / plaus.sum(dim=1, keepdim=True)               # [B, V]
            logsumexp_term = torch.logsumexp(q_all, dim=1)                   # [B, K]
            q_safe = torch.where(infeas.unsqueeze(-1), torch.zeros_like(q_all), q_all)
            data_term = (u.unsqueeze(-1) * q_safe).sum(dim=1)                # [B, K]
            usup = (logsumexp_term - data_term).mean()

        loss = mc_loss + args.cql_alpha * cql + args.usup_alpha * usup
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if bootstraps:
            _polyak()
        lv = float(mc_loss.item())
        # Fail fast on divergence (z-scale targets ⇒ healthy mc_loss is O(0.1)).
        assert lv < 1e4, f"mc_loss diverged: {lv:.3e}"
        loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv
        nonlocal cql_ema, usup_ema
        cv = float(cql.item()) if args.cql_alpha > 0 else 0.0
        cql_ema = cv if cql_ema is None else 0.99 * cql_ema + 0.01 * cv
        uv = float(usup.item()) if args.usup_alpha > 0 else 0.0
        usup_ema = uv if usup_ema is None else 0.99 * usup_ema + 0.01 * uv

    episode_cache: list[list] = []
    for ep in range(1, args.num_episodes + 1):
        if not episode_cache:                       # refill: simulate a batch of drafts at once
            nb = min(args.sim_batch, args.num_episodes - ep + 1)
            b_matches = [train_matches[rng.randrange(len(train_matches))] for _ in range(nb)]
            b_focal = [PickSlot(rng.randrange(NUM_PLAYERS)) for _ in range(nb)]
            b_eps = [max(args.epsilon_min, 1.0 - (1.0 - args.epsilon_min) * (ep - 1 + k) / eps_end)
                     for k in range(nb)]
            online.eval()
            episode_cache = step_episode_batch(
                b_matches, b_focal, online, policy, stats_model, vocabs,
                mmr_mean, mmr_std, weights_device, b_eps, device, rng,
                bc_mask_frac=args.bc_mask_frac, focal_continuation=args.focal_continuation,
            )
        transitions = episode_cache.pop(0)
        # n-step targets. The dim-7 reward is dense (intermediate transitions
        # carry per-pick spell damage), so the return is a SUFFIX SUM, not just
        # the terminal. G[i] = Σ_{j>=i} r_j (γ=1). For dims that are zero until
        # the terminal, G[i] equals the terminal value (unchanged).
        L = len(transitions)
        G: list[torch.Tensor] = [torch.zeros(K_STATS) for _ in range(L)]
        acc = torch.zeros(K_STATS)
        for i in range(L - 1, -1, -1):
            acc = transitions[i].reward + acc
            G[i] = acc.clone()
        for i, tr in enumerate(transitions):
            j = i + args.n_step
            if j >= L:
                ret = G[i]                                # full suffix sum to terminal
                boot_sample = None
                gpow = 1.0
            else:
                ret = G[i] - G[j]                         # partial reward sum i..j-1
                boot_sample = transitions[j].sample
                gpow = args.gamma ** args.n_step
            buffer.append(StepSample(tr.sample, tr.action_idx, ret, boot_sample, gpow))
            _train_step()

        # SWA: fold the current weights into the running average once in the
        # fixed-ε (post-anneal) regime. Decided against the argmax snapshot after
        # training; here we just accumulate the candidate. (Inlined rather than a
        # nested fn so the post-loop `swa_state is not None` check stays reachable.)
        if ep >= eps_end and ep % args.swa_every == 0:
            cur_w = {k: v.detach().cpu().clone() for k, v in online.state_dict().items()}
            if swa_state is None:
                swa_state, swa_n = cur_w, 1
            else:
                swa_n += 1
                for k, v in cur_w.items():
                    if v.is_floating_point():
                        swa_state[k].add_(v - swa_state[k], alpha=1.0 / swa_n)
                    else:
                        swa_state[k] = v  # constant buffers — keep latest

        if ep % args.eval_every == 0 or ep == args.num_episodes:
            elapsed = time.monotonic() - t0
            online.eval()
            gaps = report_per_stat_gaps(
                online, val_tuples, stats_rows_by_id, weights_cpu, device, args.batch_size * 2,
            )
            # Selection metric: continuous composite-STAT Q1−Q4 (the objective,
            # ~3-5× lower variance than binary win). composite_win kept as a
            # reported sanity number only — its CI is far too wide to
            # discriminate at this n.
            q1q4s = evaluate_q1q4_stat(
                online, val_tuples, stats_rows_by_id, stat_norm_mean, stat_norm_std,
                weights_cpu, device, args.batch_size * 2,
            )
            q1q4w = evaluate_q1q4_win(online, val_tuples, weights_cpu, device, args.batch_size * 2)
            stat_lift = q1q4s.get("q1q4_stat_lift", 0.0)
            st = q1q4s.get("q1q4_stat_t", 0.0)
            win = q1q4w.get("q1q4_lift_pp", 0.0) * 100
            wt = q1q4w.get("q1q4_t", 0.0)
            ls = f"{loss_ema:.4f}" if loss_ema is not None else " n/a "
            cs = f"{cql_ema:.3f}" if cql_ema is not None else " n/a "
            us = f"{usup_ema:.3f}" if usup_ema is not None else " n/a "
            cur_eps = max(args.epsilon_min, 1.0 - (1.0 - args.epsilon_min) * (ep - 1) / eps_end)
            print(f"Ep {ep:5d}  mc_loss={ls}  cql={cs}  usup={us}  eps={cur_eps:.3f}  "
                  f"composite_stat={stat_lift:+.3f}z (t={st:+.2f})  "
                  f"composite_win={win:+.2f}pp (t={wt:+.2f})  ({elapsed:.1f}s)")
            print_summary("val gaps", gaps)
            # Select on the MEAN per-stat ranking t over the balanced dims
            # (stable — averages 8 dims) rather than the single composite_stat
            # lift (noisy; a single high-ε early snapshot can peak it).
            # Restrict to post-ε-anneal epochs so exploration-noise evals can't
            # be selected. Final held-out numbers come from stats-causal-rank on
            # the test split, which this script never loads.
            gap_ts = [gaps.get(f"q1q4_stat_t_{STAT_SPECS[k].label}", 0.0) for k in balanced_dims]
            metric = sum(gap_ts) / len(gap_ts)
            if metric > best_eval and ep >= eps_end:
                best_eval = metric
                best_state = {k: v.cpu().clone() for k, v in online.state_dict().items()}
                print(f"    new best: val mean per-stat gap t = {best_eval:+.2f}")

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in online.state_dict().items()}

    # Ship whichever of {argmax snapshot, SWA average} wins the val selection
    # metric. SWA averages out the ±1z per-snapshot noise that the argmax can
    # otherwise lock onto.
    if swa_state is not None:
        online.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
        online.eval()
        swa_gaps = report_per_stat_gaps(
            online, val_tuples, stats_rows_by_id, weights_cpu, device, args.batch_size * 2,
        )
        swa_metric = sum(swa_gaps.get(f"q1q4_stat_t_{STAT_SPECS[k].label}", 0.0)
                         for k in balanced_dims) / len(balanced_dims)
        print(f"\nCheckpoint selection — argmax snapshot val metric={best_eval:+.2f}  "
              f"vs  SWA(n={swa_n}) val metric={swa_metric:+.2f}")
        if swa_metric >= best_eval:
            best_state, best_eval = swa_state, swa_metric
            print("  → shipping SWA average")
        else:
            print("  → shipping argmax snapshot")

    online.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    online.eval()

    # Per-preset Q1−Q4 win + per-stat gaps on the val split (diagnostic only —
    # the shipped checkpoint was selected on these same picks).
    print("\nFinal val-split evaluation (diagnostic):")
    val_gaps = report_per_stat_gaps(
        online, val_tuples, stats_rows_by_id, weights_cpu, device, args.batch_size * 2,
    )
    print_summary("val gaps", val_gaps)

    paths.models.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output or (paths.models / "stats_dqn.pt.mc")
    torch.save({
        "state_dict": best_state, "d": online.d, "n_heads": online.n_heads,
        "k_stats": K_STATS, "vocab_size": vocab_size,
        "mmr_mean": mmr_mean, "mmr_std": mmr_std,
        "stat_names": [s.label for s in STAT_SPECS],
        "stat_norm_mean": stat_norm_mean, "stat_norm_std": stat_norm_std,
        "bc_mask_frac": args.bc_mask_frac,
        "usup": {"alpha": args.usup_alpha, "frac": args.usup_frac},
        "training": (f"n_step={args.n_step}"
                     + ("+anchors" if not args.no_anchors else "")
                     + f"+cql{args.cql_alpha}"
                     f"+bcmask{args.bc_mask_frac}+usup{args.usup_alpha}"),
    }, ckpt_path)
    print(f"\nSaved MC checkpoint to {ckpt_path}  (best val composite_stat = {best_eval:+.3f}z)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=12000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--swa-every", type=int, default=100,
                        help="Fold online weights into the SWA average every N "
                             "post-anneal episodes (ep >= epsilon_end_frac*num_episodes). "
                             "The SWA average competes with the argmax snapshot for shipping.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sim-batch", type=int, default=64,
                        help="episodes simulated per batched rollout (batch-B forwards vs "
                             "batch-1). 1 = fully sequential. Statistically equivalent; the "
                             "collection net is up to sim-batch updates stale (replay-DQN tolerant).")
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--replay-capacity", type=int, default=100000)
    parser.add_argument("--n-step", type=int, default=99,
                        help="n-step return horizon (in focal picks). n>=5 = pure "
                             "Monte Carlo (default; no bootstrap, no target net). "
                             "n=1 = TD(0) (the collapsed setting). 2-4 = intermediate "
                             "bias-variance points on the TD(λ)/n-step spectrum.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Per-focal-step discount (only matters when n-step bootstraps).")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Polyak rate for the target net (only used when n<5).")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--no-anchors", action="store_true",
                        help="Drop the real random-pick MC anchors from training batches.")
    parser.add_argument("--anchor-frac", type=float, default=0.25,
                        help="Fraction of each batch drawn from real random-pick anchors.")
    parser.add_argument("--bc-mask-frac", type=float, default=0.33,
                        help="Hard BC-plausibility mask (orthogonal to CQL): the "
                             "focal only explores/recommends actions with π(a|s) ≥ "
                             "frac×uniform. 0 = off. Saved in the ckpt so inference "
                             "applies the same mask.")
    parser.add_argument("--focal-continuation", choices=["policy", "bc"], default="policy",
                        help="What the focal's FUTURE picks follow in a simulated "
                             "episode. 'policy' (default) = its own ε-greedy Q → the "
                             "Q^π full-trajectory value the recommender ships. 'bc' = "
                             "BC → the Q^BC one-step-deviation value matched to the "
                             "per-pick BC-continuation eval (and the one-suggestion "
                             "use-case). See REPORT.md (§4).")
    parser.add_argument("--cql-alpha", type=float, default=0.0,
                        help="CQL-style conservatism weight. Per-stat penalty pushing Q "
                             "down for OOD (low-BC) actions, up for BC-supported ones. "
                             "0 = pure MC (no conservatism); higher = stay closer to "
                             "human-plausible picks.")
    parser.add_argument("--usup-alpha", type=float, default=0.3,
                        help="Uniform-support CQL weight: logsumexp_a Q − "
                             "mean over BC-plausible actions of Q, per stat. "
                             "Popularity-free conservatism — pushes down "
                             "whatever ranks high outside the plausible set, "
                             "uniform push-up inside it. 0 = off.")
    parser.add_argument("--usup-frac", type=float, default=0.33,
                        help="Plausibility threshold for uniform-support CQL: "
                             "plausible iff π(a|s) ≥ frac/n_feasible (same "
                             "convention as --bc-mask-frac and the UI tag).")
    parser.add_argument("--epsilon-min", type=float, default=0.2,
                        help="Sustained exploration for action coverage (MC needs it).")
    parser.add_argument("--epsilon-end-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
