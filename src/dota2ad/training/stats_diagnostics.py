"""Checkpoint-selection diagnostics for the stats-DQN.

Measures Q's ranking quality on the random-pick subpopulation as top-vs-bottom
quartile (Q1−Q4) gaps of the realized outcome, applied to the Q-vec output
directly (per-stat, plus the composite): a larger gap means the ranker
separates high- from low-value actions more, so training selects the
checkpoint with the best gap. Bucketing/SE helpers come from `eval.bootstrap`.
"""

from __future__ import annotations

import torch

from dota2ad.core.collate import policy_collate
from dota2ad.core.types import StatsRow, UnifiedIdx
from dota2ad.models import QNetStats
from dota2ad.eval.tuples import RandomPickTuple
from dota2ad.eval.stats_specs import (
    STAT_SPECS,
    binary_gap_se,
    bucket_realized,
    continuous_gap_se,
)


def _qvec_per_action(
    qnet: QNetStats,
    tuples: list[RandomPickTuple],
    device: torch.device,
    batch_size: int,
) -> list[dict[UnifiedIdx, torch.Tensor]]:
    """For each tuple, return {action: Q_vec[K]} over feasible actions.

    Single Q-net forward per state scores every feasible action.
    """
    n = len(tuples)
    out: list[dict[UnifiedIdx, torch.Tensor]] = [{} for _ in range(n)]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = tuples[i:i + batch_size]
            batch = policy_collate([t.sample for t in chunk], device=device)
            q_vec_all = qnet(batch).cpu()              # [B, V, K]
            for j, t in enumerate(chunk):
                for a in t.sample.cand_idx:
                    out[i + j][a] = q_vec_all[j, a].clone()
    return out


def report_per_stat_gaps(
    qnet: QNetStats,
    tuples: list[RandomPickTuple],
    stats_rows_by_id: dict[int, StatsRow],
    weights: torch.Tensor,                          # [K], CPU; must match qnet's k_stats
    device: torch.device,
    batch_size: int = 256,
    quantiles: int = 4,
) -> dict[str, float]:
    """Per-stat Q1−Q4 quartile gap + composite quartile-gap (win-rate). Returns summary dict
    suitable for trainer logging. Uses STAT_SPECS truncated to the
    qnet's actual k_stats so checkpoints with fewer heads run in the same
    harness."""
    keep_mask = [t.match_id in stats_rows_by_id for t in tuples]
    kept = [t for t, k in zip(tuples, keep_mask, strict=True) if k]
    if not kept:
        return {}
    q_per = _qvec_per_action(qnet, kept, device, batch_size)
    k_active = q_per[0][next(iter(q_per[0]))].shape[0]
    active_specs = STAT_SPECS[:k_active]
    realized_actions = [t.action_idx for t in kept]
    ys_win = [t.focal_team_won for t in kept]
    focal_slots = [t.focal_slot for t in kept]

    # Realized stats per tuple
    realized_stats: list[list[float]] = []
    for k_idx in range(len(active_specs)):
        realized_stats.append([
            active_specs[k_idx].real_fn(stats_rows_by_id[t.match_id], int(focal_slots[i]))
            for i, t in enumerate(kept)
        ])

    summary: dict[str, float] = {"n": float(len(kept))}

    # Per-stat: bucket by Q_vec[k] rank, compare realized stat_k Q1−Q4
    for k_idx, spec in enumerate(active_specs):
        scores_per_tuple: list[dict[UnifiedIdx, float]] = [
            {a: float(q[a][k_idx].item()) for a in q} for q in q_per
        ]
        buckets = bucket_realized(scores_per_tuple, realized_actions, quantiles)
        top = [realized_stats[k_idx][i] for i in buckets[0]]
        bot = [realized_stats[k_idx][i] for i in buckets[-1]]
        gap, _, t_stat = continuous_gap_se(top, bot)
        summary[f"q1q4_stat_{spec.label}"] = gap
        summary[f"q1q4_stat_t_{spec.label}"] = t_stat

    # Composite: bucket by Σ_k w_k · Q_vec[k], compare realized win Q1−Q4
    composite_scores: list[dict[UnifiedIdx, float]] = [
        {a: float((q[a] * weights).sum().item()) for a in q} for q in q_per
    ]
    buckets = bucket_realized(composite_scores, realized_actions, quantiles)
    top_w = [ys_win[i] for i in buckets[0]]
    bot_w = [ys_win[i] for i in buckets[-1]]
    gap, se, t_stat = binary_gap_se(top_w, bot_w)
    summary["composite_q1q4_win"] = gap
    summary["composite_q1q4_win_se"] = se
    summary["composite_q1q4_win_t"] = t_stat
    return summary


def print_summary(label: str, summary: dict[str, float]) -> None:
    """Compact one-line print of the diagnostics summary."""
    if not summary:
        print(f"  [{label}] no data")
        return
    n = int(summary.get("n", 0))
    comp = summary.get("composite_q1q4_win", 0.0) * 100
    comp_se = summary.get("composite_q1q4_win_se", 0.0) * 100
    comp_t = summary.get("composite_q1q4_win_t", 0.0)
    parts = [f"n={n}", f"composite_win={comp:+.2f}pp±{comp_se:.2f} (t={comp_t:+.2f})"]
    # Top 3 stats by |t|
    stat_ts = [
        (key.removeprefix("q1q4_stat_t_"), val)
        for key, val in summary.items() if key.startswith("q1q4_stat_t_")
    ]
    stat_ts.sort(key=lambda kv: -abs(kv[1]))
    top3 = ", ".join(f"{name}:{t:+.1f}" for name, t in stat_ts[:3])
    parts.append(f"top stats: {top3}")
    print(f"  [{label}] " + "  ".join(parts))
