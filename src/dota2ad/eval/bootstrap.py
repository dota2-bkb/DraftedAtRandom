"""Match-clustered bootstrap CIs and descriptive gap primitives.

`cluster_bootstrap` / `cluster_bootstrap_ci` (resampling whole matches, since
multiple random picks share one) back every CI in the causal-rank estimator
(REPORT.md Appendix A) and the value tests. `spearman_cluster_ci` is a
rank-correlation variant. The quartile-gap helpers (`bucket_realized`,
`quartile_gap`, `continuous_gap_se`, `binary_gap_se`, `realized_percentiles`)
compute the descriptive top-vs-bottom-quartile outcome gap used by the
stats-DQN's checkpoint-selection diagnostic (`training.stats_diagnostics`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import rankdata

from dota2ad.core.types import UnifiedIdx


def bucket_realized(
    scores_per_tuple: list[dict[UnifiedIdx, float]],
    realized_actions: list[UnifiedIdx],
    quantiles: int,
) -> list[list[int]]:
    """Bucket tuple indices by the normalized rank of the realized action under
    `scores`. Returns `quantiles` lists of tuple indices, best (Q1) to worst."""
    n = len(scores_per_tuple)
    norm_ranks: list[float] = []
    for i, scores in enumerate(scores_per_tuple):
        items = sorted(scores.items(), key=lambda kv: -kv[1])  # descending = better
        rank = next(r for r, (a, _) in enumerate(items) if a == realized_actions[i])
        k = len(items)
        norm_ranks.append(rank / (k - 1) if k > 1 else 0.0)
    order = sorted(range(n), key=lambda i: norm_ranks[i])
    bucket_size = n // quantiles
    buckets: list[list[int]] = []
    for q in range(quantiles):
        lo = q * bucket_size
        hi = (q + 1) * bucket_size if q < quantiles - 1 else n
        buckets.append(order[lo:hi])
    return buckets


def continuous_gap_se(values_top: list[float], values_bot: list[float]) -> tuple[float, float, float]:
    """Returns (mean(top) − mean(bot), SE, t-stat) for two independent samples
    of a continuous outcome (Welch-style, unequal variance)."""
    n_t, n_b = len(values_top), len(values_bot)
    m_t = sum(values_top) / n_t
    m_b = sum(values_bot) / n_b
    var_t = sum((x - m_t) ** 2 for x in values_top) / max(n_t - 1, 1)
    var_b = sum((x - m_b) ** 2 for x in values_bot) / max(n_b - 1, 1)
    gap = m_t - m_b
    se = math.sqrt(var_t / n_t + var_b / n_b)
    tstat = gap / max(se, 1e-12)
    return gap, se, tstat


def binary_gap_se(values_top: list[float], values_bot: list[float]) -> tuple[float, float, float]:
    """Same shape as continuous_gap_se but using binomial SE for {0,1} outcomes."""
    n_t, n_b = len(values_top), len(values_bot)
    p_t = sum(values_top) / n_t
    p_b = sum(values_bot) / n_b
    gap = p_t - p_b
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_b * (1 - p_b) / n_b)
    tstat = gap / max(se, 1e-12)
    return gap, se, tstat


# ---------------------------------------------------------------------------
# Rank correlation + its match-clustered bootstrap CI
# ---------------------------------------------------------------------------


def realized_percentiles(
    scores_per_tuple: list[dict[UnifiedIdx, float]],
    realized_actions: list[UnifiedIdx],
) -> list[float]:
    """Per tuple: the model's within-tuple percentile of the realized action
    (1.0 = ranked highest among feasible actions, 0.0 = lowest). The continuous,
    full-ranking generalization of `bucket_realized`'s quartiles."""
    pct: list[float] = []
    for i, scores in enumerate(scores_per_tuple):
        items = sorted(scores.items(), key=lambda kv: -kv[1])  # descending = better
        rank = next(r for r, (a, _) in enumerate(items) if a == realized_actions[i])
        k = len(items)
        pct.append(1.0 - rank / (k - 1) if k > 1 else 1.0)
    return pct


def quartile_gap(
    pct: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    q: int = 4,
) -> float:
    """Q1−Q4 gap of outcome `y`: mean over the top 1/q of items (by `pct`, where
    high = the model ranked the realized action well) minus mean over the bottom
    1/q (which absorbs the remainder). Bucketing matches `bucket_realized`
    (Q1 = highest pct = best). The continuous-outcome analogue scored for its
    raw-units effect size; pair with `cluster_bootstrap_ci` for a clustered CI.
    Returns NaN when there are fewer than `q` items."""
    pa, ya = np.asarray(pct, float), np.asarray(y, float)
    n = len(pa)
    bs = n // q
    if bs == 0:
        return float("nan")
    order = np.argsort(-pa, kind="stable")  # best (highest pct) first
    return float(ya[order[:bs]].mean() - ya[order[(q - 1) * bs:]].mean())


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation (Pearson on average-ranks). NaN if <3 points or
    either side has no variance."""
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_bootstrap(
    stat_fn: Callable[[np.ndarray], float],
    groups: Sequence[int] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Resample whole clusters (e.g. matches) with replacement and return the
    array of `stat_fn` values (degenerate NaN resamples dropped). Resampling
    clusters — not items — respects within-cluster correlation, which an i.i.d.
    resample would understate. Shared core of the clustered-bootstrap CI and the
    clustered-bootstrap p-value."""
    rng = np.random.default_rng(seed)
    by_group: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    clusters = [np.array(v) for v in by_group.values()]
    nc = len(clusters)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = np.concatenate([clusters[c] for c in rng.integers(0, nc, size=nc)])
        v = stat_fn(idx)
        if not np.isnan(v):
            boots.append(v)
    return np.asarray(boots, dtype=float)


def cluster_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    groups: Sequence[int] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile CI for `stat_fn(indices)`, resampling whole clusters (e.g.
    matches) with replacement so within-cluster correlation is respected — an
    i.i.d. SE would understate it. `stat_fn` receives a numpy array of item
    indices (with repeats). Returns (lo, median, hi); degenerate (NaN) resamples
    are dropped."""
    boots = cluster_bootstrap(stat_fn, groups, n_boot=n_boot, seed=seed)
    if len(boots) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 50)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


def spearman_cluster_ci(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    groups: Sequence[int] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Spearman ρ(x, y) with a match-clustered bootstrap 95% CI. Returns
    (rho, ci_lo, ci_hi)."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    rho = spearman(xa, ya)
    lo, _, hi = cluster_bootstrap_ci(
        lambda idx: spearman(xa[idx], ya[idx]), groups, n_boot=n_boot, seed=seed,
    )
    return rho, lo, hi
