"""static_rank: community-practice reference rankers, causally scored.

Three ranking *forms* the Ability Draft community actually drafts by, each fit
on TRAIN matches only and scored with the same estimator as BC/Q:

  popularity — a context-free "tier list": deliberate picks / pool appearances
      (items with <50 train pool appearances score 0 — rare = unpopular).
  win-rate — the observational per-ability win-rate table (the form served by
      community stat sites, e.g. windrun.io): P(team win | item in a
      player's final loadout), all picks counted. `raw` is the table as the
      community sees it; `shrunk` applies empirical-Bayes shrinkage toward 0.5
      with strength selected on VAL (fixed before the test read).
  pair-synergy — the "combos" table: a candidate is scored by the mean
      (shrunk) pair win-rate with the picker's CURRENT pre-pick loadout;
      unseen pairs and empty loadouts fall back to the marginal win-rate.

Besides each table's own β̂, the script scores BC on the same picks and reports
PAIRED same-pick contrasts (Δβ̂ with within-pick differencing, match-clustered)
— table-vs-table and BC-vs-table — so comparative claims never rest on
marginal-CI overlap. The win-rate and pair rankers were added after the
primary analysis plan was fixed — each is fully determined by train-split
tables plus a val-selected shrinkage strength; no parameter touches test.

Each run also writes the raw tables themselves (unshrunk counts, all items,
train split only) as CSVs to `<root>/results/tables/` — popularity.csv,
winrate.csv, pair_winrate.csv — which ship with the repo. `--tables-only`
rebuilds just those files without touching the evaluation.

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-causal-rank/static_rank.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches, load_split,
    load_stats_rows, load_vocabs,
)
from dota2ad.core.collate import policy_collate
from dota2ad.core.draft_logic import idx, replay_complete
from dota2ad.core.encoding import encode_loadout
from dota2ad.core.mechanism import iw_to_uniform
from dota2ad.core.types import PickSlot
from dota2ad.eval.causal_rank import beta_ci, rank_pct
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.tuples import extract_tuples
from dota2ad.models import load_policy
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS
from dota2ad.eval.results import write_results

SHRINK_GRID = [10, 30, 100, 300, 1000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--tables-only", action="store_true",
                    help="write results/tables/*.csv and exit (no eval)")
    args = ap.parse_args()
    paths = default_paths()

    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    val_matches = [m for m in matches if m.match_id in split.val_ids]
    test_matches = [m for m in matches if m.match_id in split.test_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    rows = {r.match_id: r for r in load_stats_rows()}

    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(rows, _iter())

    def U(kind, gid):
        return idx(vocabs, gid, kind)

    # --- popularity: deliberate picks / pool appearances ---
    pool_app, delib = {}, {}
    for m in train_matches:
        for a in ([U("h", h) for h in m.hero_pool] + [U("a", a) for a in m.basic_pool]
                  + [U("a", a) for a in m.ult_pool]):
            pool_app[a] = pool_app.get(a, 0) + 1
        for e in m.history:
            if e.is_random:
                continue
            a = U("h", e.hero_id) if e.hero_id is not None else U("a", e.draft_ability_id)
            delib[a] = delib.get(a, 0) + 1
    popularity = {a: delib.get(a, 0) / pool_app[a] for a in pool_app if pool_app[a] >= 50}
    print(f"popularity table from {len(train_matches)} train matches: {len(popularity)} items")

    # --- win-rate + pair tables from terminal loadouts (all picks, community-faithful) ---
    n_a: dict[int, int] = {}
    w_a: dict[int, float] = {}
    n_p: dict[tuple[int, int], int] = {}
    w_p: dict[tuple[int, int], float] = {}
    for m in train_matches:
        final_pp = replay_complete(m)
        for ps in range(NUM_PLAYERS):
            lo = encode_loadout(final_pp[PickSlot(ps)], vocabs)
            won = 1.0 if m.radiant_win == (ps % 2 == 0) else 0.0
            for a in lo:
                n_a[a] = n_a.get(a, 0) + 1
                w_a[a] = w_a.get(a, 0.0) + won
            for a, b in combinations(sorted(lo), 2):
                n_p[(a, b)] = n_p.get((a, b), 0) + 1
                w_p[(a, b)] = w_p.get((a, b), 0.0) + won
    print(f"win-rate table: {len(n_a)} items; pair table: {len(n_p)} pairs")

    # --- release the raw tables (unshrunk counts, train split only) ---
    inv: dict[int, str] = {i: k for k, i in vocabs.draft_id_to_index.items()}
    lookups = json.loads((paths.dataset / "lookups.json").read_text())

    def name(u: int) -> tuple[str, str]:
        kind, raw = inv[u].split(":")
        return (("hero", lookups["hero_id_to_name"][raw]) if kind == "h"
                else ("ability", lookups["ability_id_to_name"][raw]))

    tables_dir = paths.root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def write_table(fname, header, rows, key):
        rows = sorted(rows, key=key)
        with open(tables_dir / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"table → {tables_dir / fname} ({len(rows)} rows)")

    write_table(
        "popularity.csv",
        ["kind", "key", "pool_appearances", "deliberate_picks", "pick_rate"],
        ((*name(a), n, delib.get(a, 0), f"{delib.get(a, 0) / n:.6f}")
         for a, n in pool_app.items()),
        key=lambda r: (-r[3] / r[2], r[0], r[1]))       # pick_rate desc
    write_table(
        "winrate.csv",
        ["kind", "key", "matches", "wins", "win_rate"],
        ((*name(a), n, int(w_a[a]), f"{w_a[a] / n:.6f}") for a, n in n_a.items()),
        key=lambda r: (-r[3] / r[2], r[0], r[1]))       # win_rate desc
    write_table(
        "pair_winrate.csv",
        ["kind_a", "key_a", "kind_b", "key_b", "matches", "wins", "win_rate"],
        ((*name(a), *name(b), n, int(w_p[(a, b)]), f"{w_p[(a, b)] / n:.6f}")
         for (a, b), n in n_p.items()),
        key=lambda r: (-r[4], r[0], r[1], r[2], r[3]))  # most-drafted pairs first
    if args.tables_only:
        return 0

    def wr(a: int, m_: float) -> float:
        n = n_a.get(a, 0)
        return (w_a.get(a, 0.0) + 0.5 * m_) / (n + m_) if (n + m_) > 0 else 0.5

    def pair_score(c: int, loadout: Sequence[int], m_pair: float, m_marg: float) -> float:
        if not loadout:
            return wr(c, m_marg)
        vals = []
        for x in loadout:
            p = (c, x) if c < x else (x, c)
            n = n_p.get(p, 0)
            vals.append((w_p.get(p, 0.0) + 0.5 * m_pair) / (n + m_pair)
                        if (n + m_pair) > 0 else wr(c, m_marg))
        return float(np.mean(vals))

    # --- evaluation tuples (val for shrinkage selection; test for the one read) ---
    def prep(ms):
        tuples = [t for t in extract_tuples(ms, vocabs, mmr_mean, mmr_std)
                  if t.match_id in rows]
        ys = compute_realized_y_vec(tuples, rows, snm, sns)
        K = len(DEFAULT_BALANCED_WEIGHTS)
        wv = np.asarray([float(x) for x in DEFAULT_BALANCED_WEIGHTS])
        keep = [(t, y) for t, y in zip(tuples, ys, strict=True)
                if y is not None and len(t.sample.cand_idx) >= 3]
        comp = np.array([float(y.numpy()[:K] @ wv) for _, y in keep])
        win = np.array([t.focal_team_won for t, _ in keep])
        win_z = (win - win.mean()) / (win.std() or 1.0)
        mids = [t.match_id for t, _ in keep]
        return keep, comp, win_z, mids

    def focal_loadout(t):
        return list(t.sample.loadouts[t.focal_slot])

    # shrinkage selection on VAL (point-estimate β̂, composite)
    vkeep, vcomp, _, _ = prep(val_matches)
    print(f"\nshrinkage selection on val (n={len(vkeep)}):")

    def val_beta(score_fn):
        delta = np.zeros(len(vkeep))
        wp_ = np.zeros(len(vkeep))
        for i, (t, _) in enumerate(vkeep):
            feas = list(t.sample.cand_idx)
            ridx = feas.index(t.action_idx)
            s = np.array([score_fn(a, t) for a in feas])
            delta[i] = (rank_pct(s) - 0.5)[ridx]
            wp_[i] = iw_to_uniform(t.sample.cand_type, ridx)
        return float((wp_ * delta * vcomp).mean())

    m_wr = max(SHRINK_GRID, key=lambda m_: val_beta(lambda a, _t: wr(a, m_)))
    for m_ in SHRINK_GRID:
        print(f"  wr    m={m_:5d}  val β̂ = {val_beta(lambda a, _t, m_=m_: wr(a, m_)):+.4f}")
    m_pair = max(SHRINK_GRID,
                 key=lambda m_: val_beta(lambda a, t: pair_score(a, focal_loadout(t), m_, m_wr)))
    for m_ in SHRINK_GRID:
        b = val_beta(lambda a, t, m_=m_: pair_score(a, focal_loadout(t), m_, m_wr))
        print(f"  pair  m={m_:5d}  val β̂ = {b:+.4f}")
    print(f"  selected: m_wr={m_wr}, m_pair={m_pair} (fixed before the test read)")

    # --- the one test read: table rankers + BC on the same picks ---
    keep, comp, win_z, mids = prep(test_matches)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(paths.models / "policy.pt", vocabs, device)
    policy.eval()
    names = ["static", "wr_raw", "wr_shrunk", "pair_raw", "pair_shrunk", "bc"]
    d = {k: np.zeros(len(keep)) for k in names}
    wp = np.zeros(len(keep))
    B = 256
    with torch.no_grad():
        for i0 in range(0, len(keep), B):
            chunk = keep[i0:i0 + B]
            probs = policy(policy_collate([t.sample for t, _ in chunk], device)
                           ).exp().cpu().numpy()
            for j, (t, _) in enumerate(chunk):
                i = i0 + j
                feas = list(t.sample.cand_idx)
                ridx = feas.index(t.action_idx)
                wp[i] = iw_to_uniform(t.sample.cand_type, ridx)
                lo = focal_loadout(t)
                scores = {
                    "static": np.array([popularity.get(a, 0.0) for a in feas]),
                    "wr_raw": np.array([wr(a, 0) for a in feas]),
                    "wr_shrunk": np.array([wr(a, m_wr) for a in feas]),
                    "pair_raw": np.array([pair_score(a, lo, 0, 0) for a in feas]),
                    "pair_shrunk": np.array([pair_score(a, lo, m_pair, m_wr) for a in feas]),
                    "bc": probs[j][feas],
                }
                for k, s in scores.items():
                    d[k][i] = (rank_pct(s) - 0.5)[ridx]

    labels = {"static": "popularity", "wr_raw": "win-rate raw",
              "wr_shrunk": f"win-rate shrunk (m={m_wr})",
              "pair_raw": "pair-synergy raw",
              "pair_shrunk": f"pair-synergy shrunk (m={m_pair})",
              "bc": "BC (reference)"}
    print(f"\nStatic reference rankers on TEST (n={len(keep)}; same estimator as BC/Q):")
    figs: dict[str, float | int] = {"n_picks": len(keep), "n_matches": len(set(mids)),
                                    "m_wr": m_wr, "m_pair": m_pair}
    causal_p = paths.root / "results" / "stats-causal-rank.json"
    beta_bc = json.loads(causal_p.read_text())["beta_bc"] if causal_p.exists() else None
    for key in names:
        for out_lab, out in (("composite", comp), ("win", win_z)):
            b, lo_, hi = beta_ci(wp * d[key] * out, mids, args.bootstrap)
            if key == "bc":     # reference: number of record lives in stats-causal-rank
                if out_lab == "composite":
                    print(f"  {labels[key]:28s} β̂ = {b:+.4f} [{lo_:+.4f},{hi:+.4f}]")
                continue
            k = ("static_beta" if key == "static" else f"{key}_beta") \
                if out_lab == "composite" else \
                ("static_win" if key == "static" else f"{key}_win")
            figs[k], figs[f"{k}_lo"], figs[f"{k}_hi"] = float(b), float(lo_), float(hi)
            if out_lab == "composite":
                s = "*" if lo_ > 0 or hi < 0 else " "
                share = f"  ({b / beta_bc * 100.0:.0f}% of BC)" if beta_bc else ""
                print(f"  {labels[key]:28s} β̂ = {b:+.4f} [{lo_:+.4f},{hi:+.4f}]{s}{share}")
                if beta_bc:
                    figs[f"{key}_share_pct" if key != "static" else "share_of_bc_pct"] = (
                        b / beta_bc * 100.0)

    print("\nPaired same-pick contrasts (Δβ̂ with within-pick differencing, match-clustered):")
    contrasts = [("dd_pairraw_pairshrunk", "pair_raw", "pair_shrunk"),
                 ("dd_bc_pop", "bc", "static"),
                 ("dd_bc_wrraw", "bc", "wr_raw"),
                 ("dd_bc_pairraw", "bc", "pair_raw"),
                 ("dd_pairraw_pop", "pair_raw", "static"),
                 ("dd_pop_wrraw", "static", "wr_raw")]
    for key, a, b_ in contrasts:
        db, lo_, hi = beta_ci(wp * (d[a] - d[b_]) * comp, mids, args.bootstrap)
        figs[key], figs[f"{key}_lo"], figs[f"{key}_hi"] = float(db), float(lo_), float(hi)
        s = "*" if lo_ > 0 or hi < 0 else " "
        print(f"  {labels[a].split(' (')[0]:16s} − {labels[b_].split(' (')[0]:18s} "
              f"Δβ̂ = {db:+.4f} [{lo_:+.4f},{hi:+.4f}]{s}")
    write_results("static-rank", figs)
    print("\nComparative claims rest on the paired contrasts above, never on "
          "marginal-CI overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
