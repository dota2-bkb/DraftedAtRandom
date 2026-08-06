"""Design-based causal-ranking estimator (REPORT.md, Appendix A).

Scores feasible actions with a ranker at the decision-time state (BC
pick-probability, or the stats-DQN's per-action value), forms the within-state
deviation `δ` of the realized (and a permuted) action, and returns the per-tuple
arrays that both `stats-causal-rank` (the core test) and `stats-generalization`
(transportability) consume. `β̂ = mean_i δ(s_i,A_i)·ỹ_i` is unbiased for `E_s[Cov_a(σ, v)]` — provided each pick's contribution is weighted
by `w = (1/m)/P_mech(A)` (the `w_prop` field), which corrects the true forced-pick
propensity (coin-then-uniform, `dota2ad.core.mechanism`) back onto the uniform
estimand. Without it the unweighted mean targets the P_mech-weighted covariance, not
the uniform one (see `experiments/random-mechanism`). `w ≈ 1` (mean 1.00), so it
barely moves the point estimate, but it removes the per-kind tilt (basics under-, ults
over-sampled by the coin). Callers multiply their `δ·y` contributions by `w_prop`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.stats import rankdata

from dota2ad.core.collate import policy_collate
from dota2ad.core.mechanism import iw_to_uniform
from dota2ad.eval.bootstrap import cluster_bootstrap, cluster_bootstrap_ci
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.training.stats_simulator import scalarize_q


def rank_pct(vals: np.ndarray) -> np.ndarray:
    """Ascending percentile rank in [0,1] per element (1.0 = highest score).
    Ties receive the average (mid-)rank, so the ranks are invariant to input
    order — there is no position-dependent tie-breaking to bias δ."""
    m = len(vals)
    if m <= 1:
        return np.full(m, 0.5)
    return (rankdata(vals, method="average") - 1.0) / (m - 1)


def beta_ci(contrib: np.ndarray, mids, n_boot: int) -> tuple[float, float, float]:
    """β̂ = mean_i contrib_i (contrib_i = δ_i·ỹ_i) with a match-clustered bootstrap CI."""
    beta = float(contrib.mean())
    lo, _, hi = cluster_bootstrap_ci(lambda idx: float(contrib[idx].mean()), mids, n_boot=n_boot)
    return beta, lo, hi


def beta_ci_p(contrib: np.ndarray, mids, n_boot: int) -> tuple[float, float, float, float]:
    """β̂, its percentile 95% CI, and a two-sided match-clustered bootstrap p-value
    (H0: β=0), all from one clustered resample. The p-value is the resample mass on
    the far side of 0, doubled and (+1)-smoothed so it is never exactly 0 at finite
    `n_boot`; it is consistent with the CI (p < 0.05 ⟺ the 95% CI excludes 0). Used
    for the per-stat secondary battery, where the p-values feed a BH-FDR correction."""
    beta = float(contrib.mean())
    boots = cluster_bootstrap(lambda idx: float(contrib[idx].mean()), mids, n_boot=n_boot)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    nb = len(boots)
    tail = int(min((boots <= 0.0).sum(), (boots >= 0.0).sum()))
    p = min(1.0, 2.0 * (tail + 1) / (nb + 1))
    return beta, lo, hi, p


def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean mask of the hypotheses rejected while
    holding the expected false-discovery proportion of the family at ≤ `q`. The
    right control for a battery of correlated secondary outcomes (BH is valid under
    positive dependence), and more powerful than Bonferroni FWER control."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    passed = np.where(ranked <= q * np.arange(1, m + 1) / m)[0]
    if len(passed) == 0:
        return np.zeros(m, dtype=bool)
    return p <= ranked[passed.max()]


def _override_mmr(sample, focal_slot: int, mmr_z: float):
    """Copy of `sample` with the focal seat's (z-normalized) MMR set to `mmr_z` —
    the counterfactual "what would a player at this skill pick here" (used to test
    whether skill-conditioning changes the ranker's causal quality)."""
    nv = list(sample.mmr_vals); nv[focal_slot] = mmr_z
    nm = list(sample.mmr_mask); nm[focal_slot] = True
    return replace(sample, mmr_vals=tuple(nv), mmr_mask=tuple(nm))


@dataclass
class Deviations:
    """Per-tuple within-state deviations of the realized (and a permuted) action,
    under BC and Q, in both transforms — plus outcomes and pre-pick covariates.
    `*_rank` δ = percentile_rank − 0.5; `*_score` δ = (σ − mean)/sd."""
    n: int
    K: int
    mids: list
    Y: np.ndarray          # [n,K] z-normed realized stats
    comp: np.ndarray       # [n] balanced composite realized outcome
    win: np.ndarray        # [n] focal-team win (0/1)
    q_rank: np.ndarray; qc_rank: np.ndarray; bc_rank: np.ndarray       # [n,K] / [n] / [n]
    p_rank: np.ndarray; pc_rank: np.ndarray                            # permuted
    q_score: np.ndarray; qc_score: np.ndarray; bc_score: np.ndarray
    p_score: np.ndarray; pc_score: np.ndarray
    mmr: np.ndarray; turn: np.ndarray; n_feas: np.ndarray              # pre-pick covariates
    w_prop: np.ndarray     # [n] IW (1/m)/P_mech(A); multiply δ·y contributions by this


def compute_deviations(tuples, policy, qnet, weights, vocab_size,
                       stats_rows_by_id, stat_norm_mean, stat_norm_std,
                       device, seed: int = 0, batch_size: int = 256,
                       mmr_override: float | None = None) -> Deviations:
    """`mmr_override` (z-score) sets the focal's MMR before scoring — for the
    conditioned-skill analysis; None (default) uses the real MMR."""
    rng = np.random.default_rng(seed)
    with_rows = [t for t in tuples if t.match_id in stats_rows_by_id]
    ys = compute_realized_y_vec(with_rows, stats_rows_by_id, stat_norm_mean, stat_norm_std)
    K = qnet.score_mlp[-1].out_features
    w = weights[:K]
    w_np = np.asarray([float(x) for x in w])
    kept = [(t, y) for t, y in zip(with_rows, ys, strict=True)
            if y is not None and len(t.sample.cand_idx) >= 3]
    tup = [t for t, _ in kept]
    Y = np.stack([y.numpy()[:K] for _, y in kept])
    n = len(tup)

    q_rank = np.zeros((n, K)); p_rank = np.zeros((n, K))
    q_score = np.zeros((n, K)); p_score = np.zeros((n, K))
    qc_rank = np.zeros(n); pc_rank = np.zeros(n); bc_rank = np.zeros(n)
    qc_score = np.zeros(n); pc_score = np.zeros(n); bc_score = np.zeros(n)
    win = np.zeros(n); mmr = np.zeros(n); turn = np.zeros(n); n_feas = np.zeros(n)
    w_prop = np.zeros(n)
    mids: list = []

    ptr = 0
    with torch.no_grad():
        for i0 in range(0, n, batch_size):
            chunk = tup[i0:i0 + batch_size]
            samples = ([t.sample for t in chunk] if mmr_override is None else
                       [_override_mmr(t.sample, int(t.focal_slot), mmr_override) for t in chunk])
            b = policy_collate(samples, device=device)
            qv = qnet(b).cpu()                                # [B, V, K]
            qc = scalarize_q(qv, w)                            # [B, V]
            bc = policy(b).cpu().exp()[:, :vocab_size]         # [B, V]; drop random class
            for t in chunk:
                feas = list(t.sample.cand_idx)
                ridx = feas.index(t.action_idx)
                pidx = int(rng.integers(len(feas)))
                fi = torch.tensor(feas, dtype=torch.long)
                jj = ptr - i0

                qf = qv[jj].index_select(0, fi)[:, :K].numpy()   # [F, K]
                sd = qf.std(0); sd = np.where(sd > 1e-9, sd, 1.0)
                zf = (qf - qf.mean(0)) / sd
                rf = np.stack([rank_pct(qf[:, k]) for k in range(K)], axis=1) - 0.5
                q_score[ptr] = zf[ridx]; p_score[ptr] = zf[pidx]
                q_rank[ptr] = rf[ridx]; p_rank[ptr] = rf[pidx]

                qcf = qc[jj].index_select(0, fi).numpy()
                s = qcf.std() or 1.0
                zc = (qcf - qcf.mean()) / s
                qc_score[ptr] = zc[ridx]; pc_score[ptr] = zc[pidx]
                rc = rank_pct(qcf) - 0.5
                qc_rank[ptr] = rc[ridx]; pc_rank[ptr] = rc[pidx]

                bcf = bc[jj].index_select(0, fi).numpy()
                s2 = bcf.std() or 1.0
                bc_score[ptr] = ((bcf - bcf.mean()) / s2)[ridx]
                bc_rank[ptr] = (rank_pct(bcf) - 0.5)[ridx]

                win[ptr] = 1.0 if t.focal_team_won else 0.0
                mmr[ptr] = t.mmr_vals[t.focal_slot] if t.mmr_mask[t.focal_slot] else 0.0
                turn[ptr] = t.turn
                n_feas[ptr] = len(feas)
                w_prop[ptr] = iw_to_uniform(t.sample.cand_type, ridx)
                mids.append(t.match_id)
                ptr += 1

    return Deviations(
        n=n, K=K, mids=mids, Y=Y, comp=Y @ w_np, win=win,
        q_rank=q_rank, qc_rank=qc_rank, bc_rank=bc_rank, p_rank=p_rank, pc_rank=pc_rank,
        q_score=q_score, qc_score=qc_score, bc_score=bc_score, p_score=p_score, pc_score=pc_score,
        mmr=mmr, turn=turn, n_feas=n_feas, w_prop=w_prop)
