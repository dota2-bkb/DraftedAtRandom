"""Premise for the skill-headroom test: which player signal can even label a "good
drafter"? Reads raw account IDs + per-player stats from parsed/<mid>/match_details.json.

Three findings (numbers of record: the run log / results manifest):
  (0) GENERAL RANK is noise for AD. The team gap in rank-derived MMR estimates
      predicts the AD win only marginally above chance.
  (1) WIN is noise. A chronological AD-Elo over win/loss predicts *future* AD wins
      barely above chance (flat across K), while the identical machinery scores
      well above chance on synthetic games generated from an injected skill
      signal (positive control). So individual AD skill is not recoverable from
      win/loss — matchmaking + 10-player/50-min variance dominate.
  (2) STATS are a repeatable per-player trait. A balanced per-player stats
      composite has moderate split-half reliability, surviving out-of-sample.
      So skill CAN be labeled from stats, imperfectly.

Run:
  DOTA2AD_ROOT=work pixi run python experiments/stats-skill-headroom/premise.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, UTC

import numpy as np
from sklearn.metrics import roc_auc_score

from dota2ad.core import default_paths
from dota2ad.eval.results import write_results

FIELDS = ["gold_per_min", "xp_per_min", "last_hits", "hero_damage", "tower_damage",
          "kills", "assists", "deaths", "net_worth"]
SIGN = np.array([1, 1, 1, 1, 1, 1, 1, -1, 1.0])          # deaths negative
RATE = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1])             # 1 = total (÷ by duration); gpm/xpm already rates


def build_cache(paths, cache_path):
    """One pass over match_details → [match_id, start_time, [[account, isRadiant, won,
    stats9, duration], ...10]]. Cheap fields only; account_id sits next to the stats."""
    files = sorted(paths.parsed.glob("*/match_details.json"))
    print(f"extracting {len(files):,} match_details → {cache_path}")
    ok = bad = 0
    with open(cache_path, "w") as out:
        for f in files:
            try:
                d = json.loads(f.read_text())
                dur = max(d["duration"], 1)
                rw = bool(d["radiant_win"])
                rows = [[p["account_id"], bool(p["isRadiant"]),
                         int(bool(p["isRadiant"]) == rw),
                         [float(p.get(k) or 0.0) for k in FIELDS], dur,
                         p.get("computed_mmr")]
                        for p in d["players"]]
            except (json.JSONDecodeError, KeyError, OSError):
                bad += 1
                continue
            out.write(json.dumps([d["match_id"], d.get("start_time", 0), rows]) + "\n")
            ok += 1
    print(f"  wrote {ok:,}, skipped {bad}")


def per_min(stats, dur):
    """Rate-normalize the total-type stats to per-minute; leave gpm/xpm (already rates)."""
    v = np.array(stats, float)
    return np.where(RATE == 1, v * (60.0 / dur), v)


def elo_pass(games, K, outcome_fn=None, rng=None):
    """Chronological team-mean Elo. Records (min_prior_games, elo_gap, outcome)."""
    rating, seen, rec = {}, {}, []
    for _mid, _ts, players in games:
        rad = [a for a, isr, *_ in players if isr]
        dire = [a for a, isr, *_ in players if not isr]
        if len(rad) != 5 or len(dire) != 5:
            continue
        Rr = np.mean([rating.get(a, 1500.0) for a in rad])
        Rd = np.mean([rating.get(a, 1500.0) for a in dire])
        if outcome_fn:
            o = outcome_fn(rad, dire)
        else:
            o = next(won for _a, isr, won, *_ in players if isr)   # radiant_win
        rec.append((min(seen.get(a, 0) for a in rad + dire), Rr - Rd, o))
        Er = 1.0 / (1.0 + 10 ** ((Rd - Rr) / 400.0))
        for a in rad:
            rating[a] = rating.get(a, 1500.0) + K * (o - Er)
        for a in dire:
            rating[a] = rating.get(a, 1500.0) + K * ((1 - o) - (1 - Er))
        for a in rad + dire:
            seen[a] = seen.get(a, 0) + 1
    return np.array(rec), rating


def main() -> int:
    paths = default_paths()
    cdir = paths.cache / "skill-headroom"
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / "games.jsonl"
    if not cache.exists():
        build_cache(paths, cache)
    with open(cache) as f:
        games = [json.loads(l) for l in f]
    games.sort(key=lambda g: g[1])                        # chronological
    games = [g for g in games if len(g[2]) == 10]
    rng = np.random.default_rng(0)
    ts = [g[1] for g in games if g[1]]
    lo = datetime.fromtimestamp(min(ts), UTC)
    hi = datetime.fromtimestamp(max(ts), UTC)
    print(f"{len(games):,} games   corpus window (start_time, UTC): {lo.date()} → {hi.date()}\n")

    # --- (0) general rank is noise for AD ---
    gaps, outs = [], []
    for _m, _t, players in games:
        rmm = [p[5] for p in players if p[1] and p[5] is not None]
        dmm = [p[5] for p in players if not p[1] and p[5] is not None]
        if len(rmm) < 3 or len(dmm) < 3:
            continue
        gaps.append(float(np.mean(rmm) - np.mean(dmm)))
        outs.append(next(won for _a, isr, won, *_ in players if isr))
    auc_win = float(roc_auc_score(outs, gaps))
    print("=== GENERAL RANK (rank-derived MMR estimate; team gap → AD win) ===")
    print(f"  n={len(gaps):,} matches (≥3 rated players/side)  AUC={auc_win:.3f}")
    print("  → general rank barely predicts the AD outcome\n")

    # --- (1) win is noise ---
    print("=== WIN-ELO (prequential AUC: predict each game from ratings BEFORE its update) ===")
    rec, rating = elo_pass(games, 24)
    elo_auc = {m: float(roc_auc_score(rec[rec[:, 0] >= m, 2], rec[rec[:, 0] >= m, 1]))
               for m in (5, 10)}
    cells = [f">={m}:{elo_auc[m]:.3f}(n={int((rec[:, 0] >= m).sum())})" for m in (5, 10)]
    print(f"  K=24 rating_sd={np.std(list(rating.values())):.0f}  " + "  ".join(cells))
    accts = {a for g in games for a, *_ in g[2]}
    latent = {a: float(rng.normal(0, 300)) for a in accts}

    def synth(rad, dire):
        p = 1 / (1 + 10 ** ((np.mean([latent[a] for a in dire]) -
                             np.mean([latent[a] for a in rad])) / 400.0))
        return 1 if rng.random() < p else 0
    rec, _ = elo_pass(games, 24, outcome_fn=synth)
    elo_ctl_auc = {m: float(roc_auc_score(rec[rec[:, 0] >= m, 2], rec[rec[:, 0] >= m, 1]))
                   for m in (5, 10)}
    ctl = "  ".join(f">={m}:{elo_ctl_auc[m]:.3f}" for m in (5, 10))
    print(f"  POSITIVE CONTROL (synthetic skill sd=300): {ctl}")
    print("  → real ~0.51 vs control ~0.64 ⇒ AD wins are noise-dominated, NOT a broken rating\n")

    # --- (2) stats are a stable per-player trait ---
    accs, stats_mat, wins = [], [], []
    for _m, _t, players in games:
        for a, _isr, won, s, dur, _mmr in players:
            accs.append(a); stats_mat.append(per_min(s, dur)); wins.append(won)
    S = np.stack(stats_mat)
    comp = (((S - S.mean(0)) / S.std(0)) * SIGN).mean(1)
    W = np.array(wins, float)
    by_stat, by_win = defaultdict(list), defaultdict(list)
    for a, c, w in zip(accs, comp, W, strict=True):
        by_stat[a].append(c); by_win[a].append(w)
    print("=== STATS vs WIN reliability (split-half, Spearman-Brown) ===")
    sb: dict[str, float] = {}
    for m in (4, 8):
        row = []
        for name, d in (("STATS", by_stat), ("WIN", by_win)):
            a_, b_ = [], []
            for _acct, v in d.items():
                if len(v) < m:
                    continue
                v = np.array(v); idx = rng.permutation(len(v)); h = len(v) // 2
                a_.append(v[idx[:h]].mean()); b_.append(v[idx[h:2 * h]].mean())
            r = np.corrcoef(a_, b_)[0, 1]
            sb[f"sb_{name.lower()}_ge{m}"] = float(2 * r / (1 + r))
            row.append(f"{name} r={r:+.3f} SB={2 * r / (1 + r):.3f} (n={len(a_):,})")
        print(f"  >= {m} games:  " + "   ".join(row))
    print("  → the STATS split-half reliability dwarfs the WIN one ⇒ label skill from STATS, not win")
    write_results("premise", {
        "auc_win": auc_win, "n_games": len(games), "n_rank_matches": len(gaps),
        **{f"elo_auc_ge{m}": elo_auc[m] for m in (5, 10)},
        **{f"elo_ctl_auc_ge{m}": elo_ctl_auc[m] for m in (5, 10)},
        **sb,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
