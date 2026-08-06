"""CQL-vs-BC diagnostics: does the model beat BC, not just resemble it?

Three tests on held-out val random-pick tuples (causal identification via the
random-pick natural experiment — the realized action is exogenous, so realized
outcomes are unbiased samples of E[Y | do(a)]; the Δrank regression is weighted by
the propensity IW w=(1/m)/P_mech(A) so it targets the uniform-over-feasible estimand,
the forced pick being drawn from P_mech, not uniform — see experiments/random-mechanism):

  1. Δrank regression — primary "better than BC" test
     For each random-pick state, compute the realized action's BC_rank and
     Q-composite-rank over feasible actions, Δrank = BC_rank − Q_rank
     (positive = Q likes a more than BC does). Regress realized composite-
     stat on Δrank. Positive slope ⇒ Q's deviations from BC produce better
     realized outcomes (Q adds causal value over BC).

  2. BC-at-MMR composite-stat lift — premise check + skill baselines
     Composite-stat Q1-Q4 of BC as the ranker, with focal MMR overridden to
     {low (-1z), med (0z), high (+1z)}. Validates the premise that high-MMR
     play is genuinely better AND gives skill-conditioned BC baselines that
     Q is compared against.

  3. MMR-agreement gradient — corroborating, label-free
     Per-state Spearman(Q-composite-rank, BC@mmr-rank) over feasible actions,
     averaged across states, as a function of MMR. Rising with MMR ⇒ Q's
     ranking leans skillward (the stats-signal in Q pulls it toward what
     high-MMR players do, beyond the CQL-anchored average BC).

All three report match-clustered bootstrap 95% CIs (random picks share matches,
so an i.i.d. SE understates the uncertainty); significant = the CI excludes 0.

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python -u -m \\
    experiments/stats-cql-vs-bc/run.py
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches, load_split,
    load_stats_rows, load_vocabs, iw_to_uniform,
)
from dota2ad.core.io import DEFAULT_EXCLUDES, load_excluded_matches
from dota2ad.core.collate import policy_collate
from dota2ad.core.types import PolicySample, UnifiedIdx
from dota2ad.models import QNetStats, load_policy
from dota2ad.eval.tuples import RandomPickTuple, extract_tuples
from dota2ad.eval.bootstrap import (
    cluster_bootstrap_ci, quartile_gap, realized_percentiles, spearman,
)
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.results import write_results
from dota2ad.training.stats_simulator import compute_stat_norm, scalarize_q
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


# Match-clustered bootstrap resamples for all CIs (random picks share matches,
# so an i.i.d. SE understates the uncertainty).
BOOTSTRAP = 2000


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def _override_mmr(sample: PolicySample, focal_slot: int, mmr_z: float) -> PolicySample:
    """Return a copy of sample with focal's MMR overridden to mmr_z (z-score
    in the same space as PolicySample.mmr_vals — already z-normalized)."""
    new_vals = list(sample.mmr_vals)
    new_vals[focal_slot] = mmr_z
    new_mask = list(sample.mmr_mask)
    new_mask[focal_slot] = True
    return dataclasses.replace(
        sample,
        mmr_vals=tuple(new_vals),  # type: ignore[arg-type]
        mmr_mask=tuple(new_mask),  # type: ignore[arg-type]
    )


def _rank_of(action: UnifiedIdx, scores: dict[UnifiedIdx, float]) -> int:
    """0..n-1, higher rank = higher score (i.e. more preferred)."""
    sorted_actions = sorted(scores.items(), key=lambda kv: kv[1])
    return next(i for i, (a, _) in enumerate(sorted_actions) if a == action)


def _ols(xs: list[float], ys: list[float],
         ws: list[float] | None = None) -> tuple[float, float, float, float]:
    """(Weighted) OLS y = α + β·x. Returns (β, SE(β), t, R²). `ws` are the
    propensity weights w = (1/m)/P_mech(A) that correct the true forced-pick
    propensity onto the uniform estimand (None ⇒ unweighted, w≡1)."""
    n = len(xs)
    if ws is None:
        ws = [1.0] * n
    W = sum(ws)
    mx = sum(ws[i] * xs[i] for i in range(n)) / W
    my = sum(ws[i] * ys[i] for i in range(n)) / W
    sxx = sum(ws[i] * (xs[i] - mx) ** 2 for i in range(n))
    sxy = sum(ws[i] * (xs[i] - mx) * (ys[i] - my) for i in range(n))
    syy = sum(ws[i] * (ys[i] - my) ** 2 for i in range(n))
    beta = sxy / sxx if sxx else 0.0
    alpha = my - beta * mx
    resid = [ys[i] - (alpha + beta * xs[i]) for i in range(n)]
    rss = sum(ws[i] * resid[i] ** 2 for i in range(n))
    sigma2 = rss / max(n - 2, 1)
    se_beta = math.sqrt(sigma2 / sxx) if sxx else 0.0
    t = beta / se_beta if se_beta else 0.0
    r2 = 1 - rss / syy if syy else 0.0
    return beta, se_beta, t, r2


# ---------------------------------------------------------------------------- #
# Test 1: Δrank regression — primary "better than BC" test
# ---------------------------------------------------------------------------- #

def delta_rank_regression(
    qnet: QNetStats,
    policy,
    tuples: list[RandomPickTuple],
    stats_rows_by_id,
    stat_norm_mean: torch.Tensor,
    stat_norm_std: torch.Tensor,
    weights: torch.Tensor,                  # K-vec for composite
    device: torch.device,
    vocab_size: int,
    batch_size: int = 256,
    bc_topk_filter: bool = False,
    mask_frac: float = 0.33,
) -> dict[str, float]:
    """For each random-pick tuple, compute the realized action's rank under
    BC and under Q-composite, and regress realized composite-stat on the
    rank difference. Positive β ⇒ Q's deviations from BC are improvements.
    Also reports the realized random action's mean normalized rank under each
    ranker (Q vs BC) — a sanity contrast on where the exogenous action sits."""
    # 1) Realized composite-stat per tuple (z-normalized)
    ys = compute_realized_y_vec(tuples, stats_rows_by_id, stat_norm_mean, stat_norm_std)
    kept = [(t, y) for t, y in zip(tuples, ys, strict=True) if y is not None]
    if not kept:
        return {}
    k = qnet.score_mlp[-1].out_features
    w = weights[:k]
    realized_comp = [float((y[:k] * w).sum().item()) for _, y in kept]
    kept_tuples = [t for t, _ in kept]

    # 2) Per-tuple ranks (under BC and under Q-composite)
    delta_ranks: list[float] = []
    q_ranks: list[float] = []
    bc_ranks: list[float] = []
    w_props: list[float] = []    # IW (1/m)/P_mech(A): propensity → uniform
    top1_rare: list[float] = []  # is the ranker's argmax a BC-implausible action?
    with torch.no_grad():
        for i in range(0, len(kept_tuples), batch_size):
            chunk = kept_tuples[i:i + batch_size]
            b = policy_collate([t.sample for t in chunk], device=device)
            q_vec = qnet(b).cpu()                          # [B, V, K]
            q_scalar = scalarize_q(q_vec, w)               # [B, V]
            bc_lp = policy(b).cpu()
            bc = bc_lp.exp()[:, :vocab_size]               # [B, V]; drop random class
            for j, t in enumerate(chunk):
                feas = list(t.sample.cand_idx)
                if len(feas) < 3:
                    delta_ranks.append(float('nan'))
                    q_ranks.append(float('nan'))
                    bc_ranks.append(float('nan'))
                    w_props.append(float('nan'))
                    continue
                q_scores = {a: float(q_scalar[j, a].item()) for a in feas}
                bc_scores = {a: float(bc[j, a].item()) for a in feas}
                # BC-plausible set, canonical def (stats_simulator): p[a] >= frac·z/n
                z = sum(bc_scores.values()) or 1.0
                thresh = mask_frac * z / len(feas)
                plausible = {a for a in feas if bc_scores[a] >= thresh}
                if bc_topk_filter:
                    # inference BC-top-K gate: implausible actions are never ranked up
                    q_scores = {a: (q_scores[a] if a in plausible else -1e30) for a in feas}
                top1 = max(feas, key=lambda a: q_scores[a])
                top1_rare.append(0.0 if top1 in plausible else 1.0)
                a_real = t.action_idx
                qr = _rank_of(a_real, q_scores)
                br = _rank_of(a_real, bc_scores)
                # Normalize ranks to [0, 1] so cross-state aggregation is fair
                denom = len(feas) - 1
                qr_n = qr / denom
                br_n = br / denom
                q_ranks.append(qr_n)
                bc_ranks.append(br_n)
                delta_ranks.append(br_n - qr_n)
                w_props.append(iw_to_uniform(t.sample.cand_type, feas.index(a_real)))

    # 3) Drop NaN tuples, regress realized composite-stat on Δrank.
    pairs = [(delta_ranks[i], realized_comp[i], q_ranks[i], bc_ranks[i],
              kept_tuples[i].match_id, w_props[i])
             for i in range(len(kept_tuples))
             if not math.isnan(delta_ranks[i])]
    if not pairs:
        return {}
    dr = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    qr = [p[2] for p in pairs]
    brr = [p[3] for p in pairs]
    mids = [p[4] for p in pairs]
    wp = [p[5] for p in pairs]

    # _ols regresses realized stat on Δrank = BC_rank − Q_rank, WEIGHTED by the
    # propensity IW w=(1/m)/P_mech(A) (the forced pick is P_mech, not uniform);
    # negate β so positive = Q's deviations from BC are *improvements*. The CI is a
    # match-clustered bootstrap (random picks share matches → an i.i.d. SE
    # understates it); significance is "CI excludes 0".
    dr_a, yv_a, wp_a = np.asarray(dr, float), np.asarray(yv, float), np.asarray(wp, float)
    ols = _ols(dr, yv, wp)
    beta, r2 = -ols[0], ols[3]
    blo, _, bhi = cluster_bootstrap_ci(
        lambda idx: -_ols(list(dr_a[idx]), list(yv_a[idx]), list(wp_a[idx]))[0],
        mids, n_boot=BOOTSTRAP,
    )

    mean_dr = sum(dr) / len(dr)
    return {
        "n": float(len(pairs)),
        "delta_rank_std": math.sqrt(sum((d - mean_dr) ** 2 for d in dr) / len(dr)),
        "beta": beta,
        "beta_lo": blo,
        "beta_hi": bhi,
        "r2": r2,
        "mean_q_rank_of_realized": sum(qr) / len(qr),
        "mean_bc_rank_of_realized": sum(brr) / len(brr),
        "top1_rare": (sum(top1_rare) / len(top1_rare)) if top1_rare else float("nan"),
    }


# ---------------------------------------------------------------------------- #
# Test 2: BC-at-MMR composite-stat lift
# ---------------------------------------------------------------------------- #

def bc_at_mmr_q1q4_stat(
    policy,
    tuples: list[RandomPickTuple],
    stats_rows_by_id,
    stat_norm_mean: torch.Tensor,
    stat_norm_std: torch.Tensor,
    weights: torch.Tensor,
    vocab_size: int,
    mmr_z: float,
    device: torch.device,
    batch_size: int = 256,
    quantiles: int = 4,
) -> dict[str, float]:
    """Composite-stat Q1−Q4 lift using BC@mmr_z as the ranker. Realized
    composite-stat is the same continuous outcome used elsewhere."""
    ys = compute_realized_y_vec(tuples, stats_rows_by_id, stat_norm_mean, stat_norm_std)
    kept = [(t, y) for t, y in zip(tuples, ys, strict=True) if y is not None]
    if not kept:
        return {}
    k = len(weights)
    realized_comp = [float((y[:k] * weights).sum().item()) for _, y in kept]
    kept_tuples = [t for t, _ in kept]

    scores_per_tuple: list[dict[UnifiedIdx, float]] = [{} for _ in range(len(kept_tuples))]
    realized_actions = [t.action_idx for t in kept_tuples]
    with torch.no_grad():
        for i in range(0, len(kept_tuples), batch_size):
            chunk = kept_tuples[i:i + batch_size]
            # Override focal MMR for each sample
            overridden = [_override_mmr(t.sample, int(t.focal_slot), mmr_z) for t in chunk]
            b = policy_collate(overridden, device=device)
            bc_lp = policy(b).cpu()
            bc = bc_lp.exp()[:, :vocab_size]
            for j, t in enumerate(chunk):
                for a in t.sample.cand_idx:
                    scores_per_tuple[i + j][a] = float(bc[j, a].item())

    pct = realized_percentiles(scores_per_tuple, realized_actions)
    gap = quartile_gap(pct, realized_comp, quantiles)
    mids = [t.match_id for t in kept_tuples]
    pct_a, y_a = np.asarray(pct, float), np.asarray(realized_comp, float)
    glo, _, ghi = cluster_bootstrap_ci(
        lambda idx: quartile_gap(pct_a[idx], y_a[idx], quantiles),
        mids, n_boot=BOOTSTRAP,
    )
    return {
        "n": float(len(kept_tuples)),
        "mmr_z": mmr_z,
        "q1q4_stat_lift": gap,
        "ci_lo": glo,
        "ci_hi": ghi,
    }


# ---------------------------------------------------------------------------- #
# Test 3: MMR-agreement gradient
# ---------------------------------------------------------------------------- #

def mmr_agreement(
    qnet: QNetStats,
    policy,
    tuples: list[RandomPickTuple],
    weights: torch.Tensor,
    vocab_size: int,
    mmr_z: float,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float, float]:
    """Mean per-state Spearman(Q-composite-rank, BC@mmr-prob-rank) over feasible
    actions, with a match-clustered bootstrap 95% CI. Returns (mean_rho, lo, hi)."""
    k = qnet.score_mlp[-1].out_features
    w = weights[:k]

    rhos: list[float] = []
    mids: list[int] = []
    with torch.no_grad():
        for i in range(0, len(tuples), batch_size):
            chunk = tuples[i:i + batch_size]
            # Q uses original MMR (we evaluate the fixed Q); BC uses overridden MMR
            b_q = policy_collate([t.sample for t in chunk], device=device)
            q_scalar = scalarize_q(qnet(b_q).cpu(), w)
            overridden = [_override_mmr(t.sample, int(t.focal_slot), mmr_z) for t in chunk]
            b_bc = policy_collate(overridden, device=device)
            bc = policy(b_bc).cpu().exp()[:, :vocab_size]
            for j, t in enumerate(chunk):
                feas = list(t.sample.cand_idx)
                if len(feas) < 3:
                    continue
                qv = [float(q_scalar[j, a]) for a in feas]
                bv = [float(bc[j, a]) for a in feas]
                rho = spearman(qv, bv)
                if not np.isnan(rho):  # no score variance among feasible
                    rhos.append(rho)
                    mids.append(t.match_id)
    if not rhos:
        return float("nan"), float("nan"), float("nan")
    rho_a = np.asarray(rhos, float)
    lo, _, hi = cluster_bootstrap_ci(lambda idx: float(rho_a[idx].mean()), mids, n_boot=BOOTSTRAP)
    return float(rho_a.mean()), lo, hi


# ---------------------------------------------------------------------------- #
# Ablation: usup vs free vs free+inference-filter vs mask-only
# ---------------------------------------------------------------------------- #

def ablate():
    """Which Q beats BC (Δrank β CI > 0) while controlling rare-pick inflation
    (top-1-rare)? Tests moving plausibility from training (usup/mask) to
    inference (BC-top-K gate) so Q can train free of BC-cloning."""
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    val_matches = [m for m in matches if m.match_id in split.val_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    stats_rows_by_id = {r.match_id: r for r in load_stats_rows()}
    val_tuples = [t for t in extract_tuples(val_matches, vocabs, mmr_mean, mmr_std)
                  if t.match_id in stats_rows_by_id]

    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    stat_norm_mean, stat_norm_std = compute_stat_norm(stats_rows_by_id, _iter())
    weights = DEFAULT_BALANCED_WEIGHTS
    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)

    variants = [  # (label, ckpt filename, apply inference BC-top-K filter)
        ("A usup+mask (shipped)", "stats_dqn.pt",          False),
        ("B free (usup0,mask0)",  "stats_dqn_free.pt",     False),
        ("C free + infer-filter", "stats_dqn_free.pt",     True),
        ("D mask-only (usup0)",   "stats_dqn_maskonly.pt", False),
    ]
    print("\n" + "=" * 88)
    print(f"ABLATION — Δrank β vs BC + top-1-rare  (n={len(val_tuples)}, match-clustered bootstrap)")
    print("  β CI excludes 0 = beats BC;  top-1-rare = % states whose argmax is BC-implausible")
    print("=" * 88)
    print(f"  {'variant':24} {'Δrank β [95% CI]':30} {'top1-rare':10} realized-rank Q/BC")
    print("  " + "-" * 82)
    for label, fname, filt in variants:
        qp = paths.models / fname
        if not qp.exists():
            print(f"  {label:24} (missing {fname} — not trained)")
            continue
        q = QNetStats.load_from_ckpt(qp, vocab_size, device)
        q.eval()
        r = delta_rank_regression(
            q, policy, val_tuples, stats_rows_by_id, stat_norm_mean, stat_norm_std,
            weights, device, vocab_size, bc_topk_filter=filt,
        )
        star = "*" if (r["beta_lo"] > 0 or r["beta_hi"] < 0) else " "
        ci = f"{r['beta']:+.3f} [{r['beta_lo']:+.3f},{r['beta_hi']:+.3f}]{star}"
        print(f"  {label:24} {ci:30} {r['top1_rare']:8.1%}   "
              f"Q{r['mean_q_rank_of_realized']:.3f}/BC{r['mean_bc_rank_of_realized']:.3f}")
    print("\nReading guide: the de-cloning hypothesis predicts C (free training + inference")
    print("  plausibility filter) clears β>0 where A (clones BC) cannot; B isolates free")
    print("  training alone; D isolates the hard mask alone.")


def power_curve():
    """How much held-out val resolves the shipped Q's edge over BC? The A-vs-BC
    Δrank β point estimate is fixed by the model; only the CI tightens with
    N (a match-clustered SE ∝ 1/√N). So subsample the val random-picks (clustered by
    match) at increasing N, recompute β + clustered CI at each, fit the half-width
    h ≈ c/√N, and solve for the N* where the lower bound β−h clears 0. Reported in
    val-picks and, via the current picks↔matches ratio, in total matches.

    Caveat: this holds the model FIXED and assumes the measured point estimate is
    the true effect. More data also retrains the model — a second-order effect on the
    estimate the curve does not capture. It answers "if the edge is real, how much
    val to make it significant," not "will more data change the estimate.\""""
    import random
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    val_matches = [m for m in matches if m.match_id in split.val_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    stats_rows_by_id = {r.match_id: r for r in load_stats_rows()}
    val_tuples = [t for t in extract_tuples(val_matches, vocabs, mmr_mean, mmr_std)
                  if t.match_id in stats_rows_by_id]

    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    stat_norm_mean, stat_norm_std = compute_stat_norm(stats_rows_by_id, _iter())
    weights = DEFAULT_BALANCED_WEIGHTS
    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.models / "stats_dqn.pt", vocab_size, device)
    q.eval()

    # Group tuples by match for clustered subsampling — random picks share matches,
    # so subsampling must be at the match level to preserve the SE structure.
    by_match: dict[int, list[RandomPickTuple]] = {}
    for t in val_tuples:
        by_match.setdefault(t.match_id, []).append(t)
    match_ids = sorted(by_match)
    n_matches = len(match_ids)
    n_tuples = len(val_tuples)
    # Convert val-picks → *collected* matches: load_matches already dropped
    # bot/leaver/swap, so add them back to report what must actually be collected
    # (the actionable number), not just the usable post-exclusion count.
    n_usable = len(matches)
    n_raw = n_usable + len(load_excluded_matches(DEFAULT_EXCLUDES))

    rng = random.Random(42)
    REPS = 5
    fractions = [0.1, 0.2, 0.4, 0.7, 1.0]

    print("\n" + "=" * 86)
    print("POWER CURVE — A (shipped) Δrank β vs BC vs held-out val size (match-clustered)")
    print(f"  val: {n_tuples} random picks from {n_matches} matches  "
          f"(dataset: {n_raw} collected, {n_usable} usable after exclusions)")
    print("=" * 86)
    print(f"  {'N picks':>9} {'~collected':>10} {'mean β':>8} {'CI width':>9} "
          f"{'half h':>7} {'reps sig':>9}")
    print("  " + "-" * 70)

    cs: list[float] = []          # c = h·√N per grid row (constant if SE ∝ 1/√N)
    beta_full = float("nan")
    for f in fractions:
        k = max(1, round(f * n_matches))
        reps = 1 if f >= 1.0 else REPS
        betas, his, los, ns = [], [], [], []
        for _ in range(reps):
            sub_ids = match_ids if f >= 1.0 else rng.sample(match_ids, k)
            sub = [t for mid in sub_ids for t in by_match[mid]]
            r = delta_rank_regression(
                q, policy, sub, stats_rows_by_id, stat_norm_mean, stat_norm_std,
                weights, device, vocab_size,
            )
            betas.append(r["beta"])
            his.append(r["beta_hi"])
            los.append(r["beta_lo"])
            ns.append(int(r["n"]))
        N = sum(ns) / len(ns)
        mean_beta = sum(betas) / len(betas)
        width = sum(h - lo for h, lo in zip(his, los, strict=True)) / len(his)
        half = width / 2.0
        n_sig = sum(1 for lo in los if lo > 0)
        cs.append(half * math.sqrt(N))
        if f >= 1.0:
            beta_full = mean_beta
        print(f"  {N:9.0f} {N / n_tuples * n_raw:10.0f} {mean_beta:+8.3f} "
              f"{width:9.3f} {half:7.3f} {n_sig:>5}/{len(los)}")

    c = sum(cs) / len(cs)
    n_star = (c / beta_full) ** 2 if beta_full > 0 else float("inf")
    m_star = n_star / n_tuples * n_raw
    print("  " + "-" * 70)
    print(f"  Fit: CI half-width ≈ {c:.1f}/√N  (h·√N spread across grid: "
          f"{min(cs):.1f}–{max(cs):.1f})")
    print(f"  A's β = {beta_full:+.3f} resolves (lower bound > 0) at")
    print(f"    N* ≈ {n_star:,.0f} val random picks ({n_star / n_tuples:.2f}× current)")
    print(f"    ≈ {m_star:,.0f} collected matches (current {n_raw:,})")
    print("  Caveat: model held fixed, the measured β assumed real; more data also retrains it.")


def control_variate_adjustment():
    """Variance-reduce the Δrank β by regression-adjusting for pre-pick covariates.

    The forced pick is exogenous so β is unbiased, but its CI is wide because the
    realized stat yᵢ swings with draft quality and player skill/engagement —
    nuisance variation orthogonal to the random action. Adding action-INDEPENDENT
    predictors of yᵢ's level strips that swing from the residual → tighter CI on
    the same data, no bias (Lin 2013 covariate adjustment in a randomized design):
      ĝ(s)  = mean Q-composite OVER FEASIBLE (state/draft baseline; the Q is the
              decision-time stat predictor — must be the feasible mean, not the
              realized action, or it absorbs β itself)
      MMR   = focal MMR z          (skill)
      disc  = picker_disconnected  (engagement: focal was offline at the timeout)
      nrand = #random picks in the match (engagement)
    Nested models show each block's contribution + the implied N* (compare M0 to
    the unadjusted power-curve N*). Caveat: random picks are a SELECTED subpop (timeouts),
    so β is a LOCAL effect — adjustment tightens it, it does not de-localize it.
    """
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    val_matches = [m for m in matches if m.match_id in split.val_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    stats_rows_by_id = {r.match_id: r for r in load_stats_rows()}
    val_tuples = [t for t in extract_tuples(val_matches, vocabs, mmr_mean, mmr_std)
                  if t.match_id in stats_rows_by_id]

    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    stat_norm_mean, stat_norm_std = compute_stat_norm(stats_rows_by_id, _iter())
    weights = DEFAULT_BALANCED_WEIGHTS
    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.models / "stats_dqn.pt", vocab_size, device)
    q.eval()

    n_raw = len(matches) + len(load_excluded_matches(DEFAULT_EXCLUDES))
    n_random_by_match: dict[int, int] = {}
    for t in val_tuples:
        n_random_by_match[t.match_id] = n_random_by_match.get(t.match_id, 0) + 1

    ys = compute_realized_y_vec(val_tuples, stats_rows_by_id, stat_norm_mean, stat_norm_std)
    k = q.score_mlp[-1].out_features
    w = weights[:k]

    drank, yv, g_q, mmr, disc, nrand, mids, wp = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(val_tuples), 256):
            chunk = val_tuples[i:i + 256]
            ychunk = ys[i:i + 256]
            b = policy_collate([t.sample for t in chunk], device=device)
            q_scalar = scalarize_q(q(b).cpu(), w)            # [B, V]
            bc = policy(b).cpu().exp()[:, :vocab_size]        # [B, V]; drop random class
            for j, t in enumerate(chunk):
                y_j = ychunk[j]
                if y_j is None:
                    continue
                feas = list(t.sample.cand_idx)
                if len(feas) < 3:
                    continue
                q_scores = {a: float(q_scalar[j, a].item()) for a in feas}
                bc_scores = {a: float(bc[j, a].item()) for a in feas}
                denom = len(feas) - 1
                qr = _rank_of(t.action_idx, q_scores) / denom
                br = _rank_of(t.action_idx, bc_scores) / denom
                drank.append(br - qr)                         # same sign as T1 (negate coef below)
                yv.append(float((y_j[:k] * w).sum().item()))
                g_q.append(sum(q_scores.values()) / len(feas))  # ĝ(s): mean over feasible
                mmr.append(t.mmr_vals[t.focal_slot] if t.mmr_mask[t.focal_slot] else 0.0)
                disc.append(1.0 if t.picker_disconnected else 0.0)
                nrand.append(float(n_random_by_match[t.match_id]))
                mids.append(t.match_id)
                wp.append(iw_to_uniform(t.sample.cand_type, feas.index(t.action_idx)))

    n = len(drank)
    drank, yv = np.asarray(drank), np.asarray(yv)
    g_q, mmr, disc, nrand, wp = map(np.asarray, (g_q, mmr, disc, nrand, wp))
    sw = np.sqrt(wp)                                          # WLS: row-scale by √w (propensity → uniform)

    def beta_ci(cov_arrays):
        X = np.column_stack([np.ones(n), drank, *cov_arrays])
        def fit(ix):
            coef, *_ = np.linalg.lstsq(X[ix] * sw[ix, None], yv[ix] * sw[ix], rcond=None)
            return -coef[1]                                   # −(Δrank coef): + = Q improves
        beta = fit(np.arange(n))
        lo, _, hi = cluster_bootstrap_ci(fit, mids, n_boot=BOOTSTRAP)
        return beta, lo, hi

    models = [
        ("M0  Δrank only (= T1)",     []),
        ("M1  + MMR,disc,#rand",      [mmr, disc, nrand]),
        ("M2  + ĝ(s) Q-baseline",     [mmr, disc, nrand, g_q]),
    ]
    print("\n" + "=" * 88)
    print("CONTROL-VARIATE ADJUSTMENT — variance-reduce Δrank β with action-independent covariates")
    print(f"  held-out n={n} random picks ({int(disc.sum())} disconnected); "
          f"no bias (Lin 2013), pure precision")
    print("=" * 88)
    print(f"  {'model':24} {'β [95% CI]':28} {'CI width':>9} {'N* @β₀':>10}")
    print("  " + "-" * 78)
    results = [(label, *beta_ci(arrs)) for label, arrs in models]
    beta0 = results[0][1]
    w0 = results[0][3] - results[0][2]
    for label, beta, lo, hi in results:
        width = hi - lo
        # N* at the FIXED M0 effect size isolates PRECISION; the cross-model β drift
        # is finite-sample randomization imbalance (covariates are action-independent,
        # so there is no confound to remove), not a real gain — don't credit it.
        m_star = (width / 2 / beta0) ** 2 * n_raw
        star = "*" if lo > 0 else " "
        tag = f"  ({(1 - width / w0) * 100:+.1f}% narrower)" if width != w0 else ""
        print(f"  {label:24} {beta:+.3f} [{lo:+.3f},{hi:+.3f}]{star}   "
              f"{width:9.3f} {m_star:>10,.0f}{tag}")

    Xc = np.column_stack([np.ones(n), mmr, disc, nrand, g_q])
    coef, *_ = np.linalg.lstsq(Xc, yv, rcond=None)
    r2 = 1 - (yv - Xc @ coef).var() / yv.var()
    gain = (1 - (1 - r2) ** 0.5) * 100
    write_results("stats-cql-adjust", {"r2_covariates": float(r2),
                                       "precision_gain_pct": float(gain)})
    print("  " + "-" * 78)
    print(f"  Covariate R² on realized stat: {r2:.3f}  →  precision gain ≈ √(1−R²) ≈ {gain:.0f}%.")
    print("  Adjustment barely moves the CI: realized AD stats are ~all game noise, with almost")
    print("    no draft/player variance to strip. Any β drift across the nested models is")
    print("    finite-sample randomization imbalance, NOT de-confounding — a CI that newly")
    print("    clears 0 there is a point-shift, not robustness.")
    print("  ⇒ covariate adjustment can't shortcut the power-curve N*; the bind stays raw N /")
    print("    deployment A/B.")
    print("  (N* @β₀ holds the effect at M0's β to show precision only; timeouts are a SELECTED")
    print("   subpop, so β is LOCAL regardless.)")


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true",
                    help="run the usup/mask/inference-filter ablation instead of the 3 tests")
    ap.add_argument("--power", action="store_true",
                    help="data-size power curve: N* (val + total matches) to resolve A's edge over BC")
    ap.add_argument("--adjust", action="store_true",
                    help="control-variate adjustment: variance-reduce the Δrank β with pre-pick covariates")
    ap.add_argument("--subset", choices=["all", "online", "disconnect"], default="all",
                    help="timeout subpopulation for the 3 tests: online (present but slow) / disconnect (AFK)")
    args = ap.parse_args()
    if args.ablate:
        ablate()
        return
    if args.power:
        power_curve()
        return
    if args.adjust:
        control_variate_adjustment()
        return
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    val_matches = [m for m in matches if m.match_id in split.val_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    stats_rows_by_id = {r.match_id: r for r in load_stats_rows()}
    val_tuples = [t for t in extract_tuples(val_matches, vocabs, mmr_mean, mmr_std,
                                            disconnect_only=(args.subset == "disconnect"),
                                            online_only=(args.subset == "online"))
                  if t.match_id in stats_rows_by_id]
    print(f"Held-out val random picks: {len(val_tuples)} (subset={args.subset})")

    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    stat_norm_mean, stat_norm_std = compute_stat_norm(stats_rows_by_id, _iter())

    weights = DEFAULT_BALANCED_WEIGHTS
    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)

    # Evaluate the shipped Q checkpoint (models/stats_dqn.pt). Easy to re-run
    # on a variant checkpoint by editing this path.
    q_path = paths.models / "stats_dqn.pt"
    q = QNetStats.load_from_ckpt(q_path, vocab_size, device)
    q.eval()
    print(f"Q model: {q_path.name}  k_stats={q.score_mlp[-1].out_features}")

    print("\n" + "=" * 76)
    print("TEST 1 — Δrank regression: do Q's deviations from BC improve outcomes?")
    print("=" * 76)
    r = delta_rank_regression(
        q, policy, val_tuples, stats_rows_by_id,
        stat_norm_mean, stat_norm_std, weights, device, vocab_size,
    )
    print(f"  n = {int(r['n'])}  |  std(Q_rank − BC_rank) = {r['delta_rank_std']:.3f} (in [0,1] units)")
    print(f"  Realized action's mean normalized rank (0=worst…1=best feasible): "
          f"Q {r['mean_q_rank_of_realized']:.3f}  vs  BC {r['mean_bc_rank_of_realized']:.3f}")
    print("  Realized composite-stat regressed on (Q_rank − BC_rank):")
    print(f"    β = {r['beta']:+.4f}  95% CI [{r['beta_lo']:+.4f}, {r['beta_hi']:+.4f}]  "
          f"R² = {r['r2']:.4f}  (match-clustered)")
    sign = ("↑ improvements" if r['beta_lo'] > 0
            else "↓ regressions" if r['beta_hi'] < 0
            else "≈ noise (CI spans 0)")
    print(f"    Interpretation: Q's deviations from BC are {sign} on realized composite-stat.")
    figs = {"t1_slope": float(r["beta"]), "t1_slope_lo": float(r["beta_lo"]),
            "t1_slope_hi": float(r["beta_hi"]), "t1_r2": float(r["r2"]),
            "n_picks": int(r["n"])}

    print("\n" + "=" * 76)
    print("TEST 2 — BC@MMR composite-stat lift: does high-MMR play actually do better?")
    print("=" * 76)
    print("  (BC as the ranker, focal MMR overridden; if monotone in MMR, premise holds)")
    for label, z, suf in [("low MMR (-1z)", -1.0, "m1z"), ("med MMR ( 0z)", 0.0, "z0"),
                          ("high MMR (+1z)", +1.0, "p1z")]:
        r = bc_at_mmr_q1q4_stat(
            policy, val_tuples, stats_rows_by_id,
            stat_norm_mean, stat_norm_std, weights, vocab_size, z, device,
        )
        figs[f"t2_lift_{suf}"] = float(r["q1q4_stat_lift"])
        figs[f"t2_lift_{suf}_lo"] = float(r["ci_lo"])
        figs[f"t2_lift_{suf}_hi"] = float(r["ci_hi"])
        print(f"    BC@{label}:  stat lift = {r['q1q4_stat_lift']:+.3f}z  "
              f"95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")

    print("\n" + "=" * 76)
    print("TEST 3 — MMR-agreement gradient: does Q lean toward high-MMR BC?")
    print("=" * 76)
    print("  (Mean per-state Spearman(Q-rank, BC@mmr-rank) over feasible; rising = skillward)")
    for label, z, suf in [("low MMR (-1z)", -1.0, "m1z"), ("med MMR ( 0z)", 0.0, "z0"),
                          ("high MMR (+1z)", +1.0, "p1z")]:
        rho, rlo, rhi = mmr_agreement(q, policy, val_tuples, weights, vocab_size, z, device)
        figs[f"t3_rho_{suf}"] = float(rho)
        figs[f"t3_rho_{suf}_lo"] = float(rlo)
        figs[f"t3_rho_{suf}_hi"] = float(rhi)
        print(f"    ρ(Q, BC@{label}) = {rho:+.3f}  95% CI [{rlo:+.3f}, {rhi:+.3f}]")

    print("\nReading (all CIs are match-clustered bootstrap; significant = excludes 0):")
    print("  T1 β CI above 0: Q's deviations from BC point in the right direction → adds value.")
    print("  T2 monotone increasing in MMR: 'high MMR is better' premise confirmed, enabling T3.")
    print("  T3 monotone increasing in MMR: Q's stats-signal pulls it toward high-MMR play.")
    if args.subset == "all":
        write_results("stats-cql-vs-bc", figs)
    else:
        print("(non-headline subset — results manifest not written)")


if __name__ == "__main__":
    main()
