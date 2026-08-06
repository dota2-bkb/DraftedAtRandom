"""Validation: does state rarity track StatsModel error?

The suggestion server shows a state-rarity confidence band. The band's quantiles
are calibrated at train time and ride on the policy checkpoint
(experiments/train-policy → `policy.density_support_q`). This is NOT a build
step — it's the *evidence* that the rarity signal is meaningful: bin held-out
decisions by state support `-log p(s)/T` and report the balanced-composite
prediction RMSE per bin. Typical draft → tight, unusual → wide. It's the only
piece that touches the StatsModel and the realized outcomes.

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-density-validate/run.py
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from dota2ad.core import (
    NUM_PLAYERS,
    compute_mmr_norm,
    default_paths,
    load_matches,
    load_split,
    load_stats_rows,
    load_vocabs,
)
from dota2ad.models import load_policy, load_stats_model
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.stats_specs import STAT_SPECS, compute_stat_predictions
from dota2ad.eval.results import write_results
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.suggest.density import path_supports
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS

N_BINS = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-matches", type=int, default=300,
                    help="held-out matches to validate over (replay is the bottleneck)")
    args = ap.parse_args()

    p = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(p.matches)
    vocabs = load_vocabs(p.vocabs)
    split = load_split(p.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    mmr_mean, mmr_std = compute_mmr_norm(train)
    policy = load_policy(p.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    stats_model = load_stats_model(p.stats_ckpt, vocabs, device)

    srows = {r.match_id: r for r in load_stats_rows()}
    valm = [m for m in matches if m.match_id in split.val_ids and m.match_id in srows][: args.max_matches]
    by_id = {m.match_id: m for m in valm}
    tuples = [t for t in extract_tuples(valm, vocabs, mmr_mean, mmr_std) if t.match_id in srows]

    # per-tuple state support, via the same path-density used live + at calibration
    support_cache: dict[int, list[float]] = {}

    def support_at(tup) -> float:
        if tup.match_id not in support_cache:
            support_cache[tup.match_id] = path_supports(
                by_id[tup.match_id], policy, vocabs, mmr_mean, mmr_std, device)
        ps = support_cache[tup.match_id]
        return ps[min(max(1, int(tup.sample.turn)), len(ps)) - 1]

    nm, nd = compute_stat_norm(
        srows, ((m.match_id, ps) for m in train for ps in range(NUM_PLAYERS)))
    W = DEFAULT_BALANCED_WEIGHTS.tolist()
    sp = compute_stat_predictions(tuples, srows, stats_model, vocabs, device, 256)
    yv = compute_realized_y_vec(tuples, srows, nm, nd)
    kb = [k for k in range(len(STAT_SPECS)) if W[k] != 0]

    pairs = [
        (support_at(t),
         sum(W[k] * (sp[k][i] - float(y[k])) for k in kb))
        for i, (t, y) in enumerate(zip(tuples, yv, strict=True)) if y is not None
    ]
    pairs.sort(key=lambda pe: pe[0])
    m = len(pairs)

    sq = policy.density_support_q
    print(f"policy.density_support_q (10/30/50/70/90) = "
          f"{[round(s, 2) for s in sq] if sq else None}")
    print(f"\nVALIDATION — balanced-composite RMSE binned by state support (n={m}):")
    print(f"  {'bin':<5} {'support(med)':>12} {'RMSE(comp-z)':>13}")
    bars = []
    for b in range(N_BINS):
        chunk = pairs[b * m // N_BINS:(b + 1) * m // N_BINS]
        sup_med = statistics.median(pe[0] for pe in chunk)
        rmse = math.sqrt(statistics.mean(pe[1] ** 2 for pe in chunk))
        bars.append(rmse)
        print(f"  Q{b + 1:<4} {sup_med:>12.2f} {rmse:>13.2f}")
    print(f"typical (Q1) {bars[0]:.2f}  vs  unusual (Q{N_BINS}) {bars[-1]:.2f}  "
          f"→ {bars[-1] / bars[0]:.2f}× wider")
    write_results("density-validate", {
        "rmse_bins": bars, "rmse_typical": bars[0], "rmse_unusual": bars[-1],
        "rmse_ratio": bars[-1] / bars[0],
    })


if __name__ == "__main__":
    main()
