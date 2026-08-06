"""Q1−Q4 evaluation of the stats-DQN on random picks.

Ranks feasible actions by the stats-DQN composite score, buckets the realized
(exogenous) random action by rank quartile, and compares realized outcome
top-vs-bottom. The outcome is either the focal's realized continuous composite
stat (`evaluate_q1q4_stat`, the selection metric — ~3-5× lower variance) or the
binary focal-team win (`evaluate_q1q4_win`, a sanity number). `compute_realized_y_vec`
builds the z-normalized realized stat vectors. Model-free given the ranker — no
DR/IPW. See REPORT.md.
"""

from __future__ import annotations

import torch

from dota2ad.core.collate import policy_collate
from dota2ad.core.types import StatsRow, UnifiedIdx
from dota2ad.models import QNetStats
from dota2ad.eval.tuples import RandomPickTuple
from dota2ad.eval.stats_specs import STAT_SPECS
from dota2ad.training.stats_simulator import scalarize_q


K_STATS = len(STAT_SPECS)


def compute_realized_y_vec(
    tuples: list[RandomPickTuple],
    stats_rows_by_id: dict[int, StatsRow],
    stat_norm_mean: torch.Tensor,
    stat_norm_std: torch.Tensor,
) -> list[torch.Tensor | None]:
    """For each tuple, return the focal's realized z-normalized stats vector,
    or None if no StatsRow is available for the match."""
    out: list[torch.Tensor | None] = []
    for t in tuples:
        if t.match_id not in stats_rows_by_id:
            out.append(None)
            continue
        row = stats_rows_by_id[t.match_id]
        v = torch.empty(K_STATS)
        for k_idx, spec in enumerate(STAT_SPECS):
            v[k_idx] = float(spec.real_fn(row, int(t.focal_slot)))
        out.append((v - stat_norm_mean) / stat_norm_std.clamp(min=1e-6))
    return out


def evaluate_q1q4_win(
    qnet: QNetStats,
    tuples: list[RandomPickTuple],
    weights: torch.Tensor,                        # [K], CPU — preset weights
    device: torch.device,
    batch_size: int,
    quantiles: int = 4,
) -> dict[str, float]:
    """Per-preset Q1−Q4 lift on the BINARY focal_team_won label.

    Ranks feasible actions by composite score (Σ_k w_k·Q_vec[k]), buckets the
    realized random action by rank quartile, compares realized win top vs bottom.
    A reported sanity number — its ±~6pp CI can't discriminate at this n.
    """
    from dota2ad.eval.stats_specs import binary_gap_se, bucket_realized
    n = len(tuples)
    if n == 0:
        return {}

    scores_per_tuple: list[dict[UnifiedIdx, float]] = [{} for _ in range(n)]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = tuples[i:i + batch_size]
            batch = policy_collate([t.sample for t in chunk], device=device)
            q_vec = qnet(batch).cpu()                          # [B, V, K]
            q_scalar = scalarize_q(q_vec, weights)             # [B, V]
            for j, t in enumerate(chunk):
                for a in t.sample.cand_idx:
                    scores_per_tuple[i + j][a] = float(q_scalar[j, a].item())

    realized_actions = [t.action_idx for t in tuples]
    ys_win = [t.focal_team_won for t in tuples]
    buckets = bucket_realized(scores_per_tuple, realized_actions, quantiles)
    top = [ys_win[i] for i in buckets[0]]
    bot = [ys_win[i] for i in buckets[-1]]
    gap, se, t_stat = binary_gap_se(top, bot)
    return {
        "n_q1q4": float(n),
        "q1q4_lift_pp": gap,
        "q1q4_se_pp": se,
        "q1q4_t": t_stat,
        "q1_ybar": sum(top) / max(len(top), 1),
        "q4_ybar": sum(bot) / max(len(bot), 1),
    }


def evaluate_q1q4_stat(
    qnet: QNetStats,
    tuples: list[RandomPickTuple],
    stats_rows_by_id: dict[int, StatsRow],
    stat_norm_mean: torch.Tensor,
    stat_norm_std: torch.Tensor,
    weights: torch.Tensor,                        # [K], CPU — preset weights
    device: torch.device,
    batch_size: int,
    quantiles: int = 4,
) -> dict[str, float]:
    """Q1−Q4 lift on the realized CONTINUOUS composite-stat (Σ_k w_k·y_k).

    Same ranking/bucketing as `evaluate_q1q4_win`, but the bucketed outcome is
    the focal's realized z-normalized composite stat — ~3-5× lower variance than
    binary win, and exactly the objective the stats-DQN optimizes. This is the
    checkpoint-selection metric. Units are composite-z (Σ_k w_k·z_k).
    """
    from dota2ad.eval.stats_specs import bucket_realized, continuous_gap_se
    ys = compute_realized_y_vec(tuples, stats_rows_by_id, stat_norm_mean, stat_norm_std)
    kept_pairs = [(t, y) for t, y in zip(tuples, ys, strict=True) if y is not None]
    if not kept_pairs:
        return {}
    kept_tuples = [t for t, _ in kept_pairs]
    # slice y to the checkpoint's K (len(weights)).
    k = len(weights)
    realized_comp = [float((y[:k] * weights).sum().item()) for _, y in kept_pairs]

    n = len(kept_tuples)
    scores_per_tuple: list[dict[UnifiedIdx, float]] = [{} for _ in range(n)]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = kept_tuples[i:i + batch_size]
            batch = policy_collate([t.sample for t in chunk], device=device)
            q_vec = qnet(batch).cpu()                          # [B, V, K]
            q_scalar = scalarize_q(q_vec, weights)             # [B, V]
            for j, t in enumerate(chunk):
                for a in t.sample.cand_idx:
                    scores_per_tuple[i + j][a] = float(q_scalar[j, a].item())

    realized_actions = [t.action_idx for t in kept_tuples]
    buckets = bucket_realized(scores_per_tuple, realized_actions, quantiles)
    top = [realized_comp[i] for i in buckets[0]]
    bot = [realized_comp[i] for i in buckets[-1]]
    gap, se, t_stat = continuous_gap_se(top, bot)
    return {
        "n_q1q4_stat": float(n),
        "q1q4_stat_lift": gap,
        "q1q4_stat_se": se,
        "q1q4_stat_t": t_stat,
        "q1_sbar": sum(top) / max(len(top), 1),
        "q4_sbar": sum(bot) / max(len(bot), 1),
    }
