"""retrieval.py — is the replay-retrieval censoring outcome-correlated?

A match enters the corpus only if Valve served its replay; `collect` marks the
refusals (CDN 403/404) with `parsed/errors/<id>.gone` alongside the OpenDota
details it already fetched, and `build-dataset` aggregates them into
`dataset/gone_matches.jsonl`. This compares
those matches to the retrieved corpus on the pre-replay observables both sides
share — duration, outcome, leaver rate, rank — the same balance-check pattern as
the stats-availability paragraph (REPORT.md §6). The retrieved-side baseline is a
seeded sample read from the identical source (`parsed/<id>/match_details.json`),
so the comparison is like-for-like.

Leaver = any player with a nonzero `leaver_status` — the same definition as the
corpus's leaver exclusion. `short` = duration under 15 minutes (an abandonment
proxy). If the gone matches are leaver-heavy or short, the retrieval censoring
is abandonment-linked (it then feeds the survivorship story — REPORT.md §3's third
caveat); if balanced, it is infrastructure noise.

Run (CPU; reads dataset/gone_matches.jsonl, written by `build-dataset`):
  DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/retrieval.py
"""
from __future__ import annotations

import argparse
import json
import random

from dota2ad.core.paths import default_paths
from dota2ad.eval.bootstrap import binary_gap_se, continuous_gap_se
from dota2ad.eval.results import write_results


def derive(rec: dict) -> dict:
    """Balance observables from one trimmed match record."""
    leavers = [s for s in (rec.get("leaver_statuses") or []) if s is not None]
    ranks = [r for r in (rec.get("rank_tiers") or []) if r is not None]
    dur = rec.get("duration")
    return {
        "duration": float(dur) if dur is not None else None,
        "short": (dur is not None) and dur < 900,
        "win": rec.get("radiant_win"),
        "leaver": any(s != 0 for s in leavers) if leavers else None,
        "rank": sum(ranks) / len(ranks) if ranks else None,
        "is_ad": rec.get("game_mode") == 18,
    }


def col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r[key] is not None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-n", type=int, default=10000,
                    help="seeded sample size of retrieved matches for the baseline")
    args = ap.parse_args()
    paths = default_paths()

    gone_path = paths.dataset / "gone_matches.jsonl"
    assert gone_path.exists(), f"{gone_path} missing — run `pixi run build-dataset` first"
    with open(gone_path) as f:
        gone_raw = [json.loads(line) for line in f]
    unknown = [g for g in gone_raw if g.get("missing")]
    gone = [derive(g) for g in gone_raw if not g.get("missing")]

    dirs = sorted(d for d in paths.parsed.iterdir() if d.is_dir() and d.name.isdigit())
    sample = random.Random(0).sample(dirs, min(args.baseline_n, len(dirs)))
    base, base_regions, base_start = [], [], []
    for d in sample:
        md = json.loads((d / "match_details.json").read_text())
        players = md.get("players") or []
        base.append(derive({
            "duration": md.get("duration"),
            "radiant_win": md.get("radiant_win"),
            "game_mode": md.get("game_mode"),
            "leaver_statuses": [p.get("leaver_status") for p in players],
            "rank_tiers": [p.get("rank_tier") for p in players],
        }))
        base_regions.append(md.get("region"))
        base_start.append(md.get("start_time"))

    print(f"gone (replay never served): n={len(gone)}  (+{len(unknown)} unknown to OpenDota too)")
    print(f"retrieved baseline: seeded sample n={len(base)} of {len(dirs)} parsed matches\n")
    print(f"{'observable':22s} {'gone':>10s} {'retrieved':>10s} {'gap':>8s} {'z':>6s}")
    obs_stats: dict[str, float] = {}
    for label, key, kind in [
        ("radiant win rate", "win", "b"),
        ("leaver rate", "leaver", "b"),
        ("short (<15 min)", "short", "b"),
        ("game_mode == AD", "is_ad", "b"),
        ("duration (s)", "duration", "c"),
        ("mean rank_tier", "rank", "c"),
    ]:
        g, b = col(gone, key), col(base, key)
        gap, _, z = (binary_gap_se if kind == "b" else continuous_gap_se)(g, b)
        mg, mb = sum(g) / len(g), sum(b) / len(b)
        obs_stats[f"{key}_gone"], obs_stats[f"{key}_retrieved"] = mg, mb
        obs_stats[f"{key}_gap"], obs_stats[f"{key}_z"] = gap, z
        print(f"{label:22s} {mg:10.3f} {mb:10.3f} {gap:+8.3f} {z:+6.1f}")

    gone_regions = [g.get("region") for g in gone_raw if not g.get("missing")]
    counts_g, counts_b = {}, {}
    for r in gone_regions:
        counts_g[r] = counts_g.get(r, 0) + 1
    for r in base_regions:
        counts_b[r] = counts_b.get(r, 0) + 1
    top = sorted(counts_g, key=lambda r: -counts_g[r])[:8]
    print("\nregion shares (replay publication is server-side; a skew here marks the censoring")
    print("as infrastructure-clustered rather than behavior-linked):")
    print(f"  {'region':>8s} {'gone':>8s} {'retrieved':>10s}")
    for r in top:
        print(f"  {r!s:>8s} {counts_g.get(r, 0) / len(gone_regions):8.1%} "
              f"{counts_b.get(r, 0) / len(base_regions):10.1%}")
    clusters = {}
    for g in gone_raw:
        if not g.get("missing"):
            c = g.get("cluster")
            clusters[c] = clusters.get(c, 0) + 1
    top_c = sorted(clusters, key=lambda c: -clusters[c])[:4]
    print("  gone by replay cluster (the serving host): " + ", ".join(
        f"{c}: {clusters[c]} ({clusters[c] / len(gone_regions):.0%})" for c in top_c))

    # Sub-channels: OpenDota's parse marker proves whether the replay was EVER served.
    # A parsed gone match = the replay existed (OpenDota fetched it while fresh) and
    # expired before our sweep; an unparsed one on a refusing cluster was plausibly
    # never served to anyone.
    live = [g for g in gone_raw if not g.get("missing")]
    parsed = [g for g in live if g.get("od_parsed")]
    day = 86400
    t0 = min(t for t in base_start if t is not None)

    def med(rows):
        offs = sorted((r.get("start_time") - t0) // day for r in rows
                      if r.get("start_time") is not None)
        return offs[len(offs) // 2]

    med_base = sorted((t - t0) // day for t in base_start if t is not None)[len(base_start) // 2]
    print(f"\n  OpenDota parse marker among gone: {len(parsed)}/{len(live)} — replays that existed "
          f"and were served\n  to OpenDota while fresh, then expired before our fetch "
          f"(median window-day {med(parsed) if parsed else '–'} vs corpus {med_base}); the "
          f"unparsed remainder\n  (median day {med([g for g in live if not g.get('od_parsed')])}) "
          f"sits on the refusing clusters — plausibly never served at all.")

    top12_share = sum(clusters[c] for c in top_c[:2]) / len(gone_regions)
    # Share of discovered matches the retrieval layer censored; the retrieved-corpus
    # size comes from the dataset manifest (build-dataset runs first — it also wrote
    # the gone_matches.jsonl this script reads).
    with open(default_paths().root / "results" / "dataset.json") as f:
        n_raw = json.load(f)["n_raw"]
    write_results("retrieval", {
        "n_gone": len(gone), "n_unknown": len(unknown),
        "gone_share": len(gone) / (n_raw + len(gone)),
        "od_parsed_share": len(parsed) / len(live),
        "win_gap_pp": obs_stats["win_gap"] * 100,
        "cluster_top1_id": top_c[0], "cluster_top1_share": clusters[top_c[0]] / len(gone_regions),
        "cluster_top2_id": top_c[1], "cluster_top2_share": clusters[top_c[1]] / len(gone_regions),
        "cluster_top12_share": top12_share, "cluster_rest_share": 1 - top12_share,
        **obs_stats,
    })
    print("\nReading: outcome balance (win, short) ⇒ the retrieval censoring is not selecting on")
    print("how games end; a leaver/short excess would mark it abandonment-linked (survivorship,")
    print("REPORT.md §3); region skew marks it infrastructure-clustered instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
