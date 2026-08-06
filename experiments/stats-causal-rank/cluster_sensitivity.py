"""cluster_sensitivity: does the match-only bootstrap understate β̂'s uncertainty?

The match-clustered bootstrap treats matches as independent, but players repeat
across matches (the 102k-match corpus shares 103,487 accounts; 71,607 appear in
≥2 matches, the most active in 180) and matches share calendar time. Sensitivity:
recompute the test-split composite-rank β̂ CI for BC and Q with the bootstrap
clustered (i) by match — the published baseline; (ii) by the FOCAL PICKER'S
ACCOUNT — absorbs repeated-player dependence in the treated seat, the only seat
whose contribution enters β̂; (iii) by calendar DAY of the match — absorbs
shared-time dependence (few clusters ⇒ a coarse but honest read; the count is
printed). Similar widths ⇒ match clustering suffices; wider ⇒ report the wider.

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-causal-rank/cluster_sensitivity.py
"""
from __future__ import annotations

import argparse
import json

import torch

from dota2ad.core import (
    NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches, load_split,
    load_stats_rows, load_vocabs,
)
from dota2ad.eval.bootstrap import cluster_bootstrap_ci
from dota2ad.eval.causal_rank import compute_deviations
from dota2ad.eval.tuples import extract_tuples
from dota2ad.models import QNetStats, load_policy
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS
from dota2ad.eval.results import write_results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    test_matches = [m for m in matches if m.match_id in split.test_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    stats_rows_by_id = {r.match_id: r for r in load_stats_rows()}

    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(stats_rows_by_id, _iter())

    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.models / "stats_dqn.pt", vocab_size, device)
    q.eval()

    tuples = extract_tuples(test_matches, vocabs, mmr_mean, mmr_std)
    d = compute_deviations(tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size,
                           stats_rows_by_id, snm, sns, device)
    kept = [t for t in tuples if t.match_id in stats_rows_by_id
            and len(t.sample.cand_idx) >= 3]
    assert len(kept) == d.n, (len(kept), d.n)

    # pick_slot → account_id + match start_time, from the raw parse (pick_slot
    # derived by first-appearance order exactly as build_dataset does).
    need = sorted({t.match_id for t in kept})
    acct_by, ts_by = {}, {}
    for mid in need:
        with open(paths.parsed / str(mid) / "match_details.json") as f:
            md = json.load(f)
        with open(paths.parsed / str(mid) / "draft_details.json") as f:
            dd = json.load(f)
        ev = [(hp["tick"], hp["player_slot"]) for hp in dd["hero_picks"]] \
            + [(p["tick"], p["player_slot"]) for p in dd["picks"]]
        ev.sort(key=lambda e: e[0])
        seen: list[int] = []
        for _, ps in ev:
            if ps not in seen:
                seen.append(ps)
        by_ps = {p["player_slot"]: p["account_id"] for p in md["players"]}
        acct_by[mid] = [by_ps[ps] for ps in seen]
        ts_by[mid] = md.get("start_time", 0)
    print(f"joined {len(need)} test matches (focal accounts + start times)")

    accts = [acct_by[t.match_id][int(t.focal_slot)] for t in kept]
    days = [ts_by[t.match_id] // 86400 for t in kept]
    labels = [("match (published)", list(d.mids)),
              ("focal account", accts),
              ("calendar day", days)]

    print(f"\nComposite-rank β̂ on TEST (n={d.n}) under alternative bootstrap clusterings")
    print(f"(same estimator, same picks — only the resampled unit changes; B={args.bootstrap}):")
    figs: dict[str, float | int] = {"n_picks": d.n}
    unit = {"match (published)": "match", "focal account": "account", "calendar day": "day"}
    for name, delta in (("BC", d.bc_rank), ("Q", d.qc_rank)):
        c = d.w_prop * delta * d.comp
        for lab, grp in labels:
            ncl = len(set(grp))
            lo, _, hi = cluster_bootstrap_ci(lambda ix, c=c: float(c[ix].mean()), grp,
                                             n_boot=args.bootstrap)
            figs[f"n_clusters_{unit[lab]}"] = ncl
            figs[f"width_{name.lower()}_{unit[lab]}"] = float(hi - lo)
            print(f"  {name:2s} {lab:20s} clusters={ncl:6d}  "
                  f"β̂={c.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  width={hi - lo:.4f}")
    for r in ("bc", "q"):
        figs[f"{r}_account_widening"] = figs[f"width_{r}_account"] / figs[f"width_{r}_match"] - 1.0
    write_results("cluster-sensitivity", figs)
    print("\nReading: similar widths across clusterings ⇒ the match-level bootstrap already")
    print("captures the dependence that matters. The day clustering has very few clusters, so")
    print("its read is coarse — compare widths, not exact bounds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
