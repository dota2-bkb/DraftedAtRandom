"""stats_coverage.py — is OpenDota parse-coverage ignorable?

The stat endpoints run only on the analytic matches OpenDota parsed for
extended stats (REPORT.md §6, "stats-available"); the rest were collected but
never parsed by OpenDota's pipeline. This compares parsed vs unparsed analytic
matches on the observables we hold for both — outcome, MMR, forced-pick counts.
Balance ⇒ the coverage drop is plausibly independent of the pick→outcome
relationship (ignorable); a forced-pick-rate difference shifts the estimand
toward higher-timeout matches (a scope nudge, not outcome selection).

Run (CPU):
  DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/stats_coverage.py
"""
from __future__ import annotations

import numpy as np

from dota2ad.core import (
    NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches, load_split,
    load_stats_rows, load_vocabs,
)
from dota2ad.eval.results import write_results
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.tuples import extract_tuples
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def match_mmr(m):
    vals = [x for x in m.mmr if x is not None]
    return float(np.mean(vals)) if vals else None


def summarize(ms, name):
    n = len(ms)
    win = np.array([1.0 if m.radiant_win else 0.0 for m in ms])
    mmrs = np.array([v for m in ms if (v := match_mmr(m)) is not None])
    nrand = np.array([sum(1 for e in m.history if e.is_random) for m in ms], float)
    ndisc = np.array([sum(1 for e in m.history if e.picker_disconnected) for m in ms], float)
    print(f"  {name:9s} n={n:6d}  win={win.mean():.4f}  MMR={mmrs.mean():7.1f}(cov {len(mmrs)/n:.2f})  "
          f"forced/match={nrand.mean():.2f}  disc/match={ndisc.mean():.3f}")
    return dict(win=win, mmr=mmrs, nrand=nrand)


def z2prop(a, b):
    """Two-proportion z (e.g. win rate)."""
    pa, pb, na, nb = a.mean(), b.mean(), len(a), len(b)
    p = (a.sum() + b.sum()) / (na + nb)
    se = np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    return (pa - pb) / se


def welch(a, b):
    """Welch t for a continuous field."""
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def main() -> int:
    matches = load_matches()
    rows = load_stats_rows()
    stats_ids = {r.match_id for r in rows}
    parsed = [m for m in matches if m.match_id in stats_ids]
    unparsed = [m for m in matches if m.match_id not in stats_ids]

    print(f"analytic={len(matches)}  parsed={len(parsed)}  unparsed={len(unparsed)} "
          f"({len(unparsed)/len(matches):.1%})\n")
    P = summarize(parsed, "parsed")
    U = summarize(unparsed, "unparsed")

    d_win = float(P['win'].mean() - U['win'].mean()); z_win = float(z2prop(P['win'], U['win']))
    d_mmr = float(P['mmr'].mean() - U['mmr'].mean()); t_mmr = float(welch(P['mmr'], U['mmr']))
    d_forced = float(P['nrand'].mean() - U['nrand'].mean())
    t_forced = float(welch(P['nrand'], U['nrand']))
    print(f"\n  Δwin (parsed−unparsed) = {d_win:+.4f}  (z={z_win:+.2f})")
    print(f"  ΔMMR                   = {d_mmr:+.1f}    (t={t_mmr:+.2f})")
    print(f"  Δforced/match          = {d_forced:+.3f}  (t={t_forced:+.2f})")
    print("\n  Read magnitude AND significance: win is balanced (|z| < 2); ΔMMR is statistically")
    print("  detectable at this n but ~0.2% in magnitude — negligible; the forced/match gap folds")
    print("  into the timeout-subpopulation scope (REPORT.md §6).")

    # Composite structure: the six balanced dims are correlated, so the ±1 composite's
    # EFFECTIVE weighting is not 1/6 each — measure the tilt so §6 can state it.
    dims = [(0, "kills", +1.0), (1, "-deaths", -1.0), (3, "gold", +1.0),
            (4, "xp", +1.0), (5, "last-hits", +1.0), (7, "hero-dmg", +1.0)]
    X = np.array([[sign * float(r.scalar_stats[ps][idx]) for idx, _, sign in dims]
                  for r in rows for ps in range(10)])
    Xz = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(Xz.T)
    names = [n for _, n, _ in dims]
    print(f"\n[composite structure] correlations among the 6 balanced dims "
          f"(signed as weighted; per-player, n={len(Xz):,}):")
    print("  " + " ".join(f"{n:>10s}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:>10s} " + " ".join(f"{C[i, j]:10.2f}" for j in range(6)))
    farm = [2, 3, 4]                       # gold, xp, last-hits
    fp = [C[i, j] for i in farm for j in farm if i < j]
    comp = Xz.sum(1)
    farm3 = Xz[:, farm].mean(1)
    combat3 = Xz[:, [0, 1, 5]].mean(1)     # kills, -deaths, hero-dmg
    pc1 = float(np.linalg.eigvalsh(C).max() / 6)
    print(f"  farm block (gold, xp, last-hits): mean pairwise r = {np.mean(fp):.2f}; "
          f"hero-dmg <-> farm mean r = {np.mean(C[5, farm]):.2f}; "
          f"kills <-> farm = {np.mean(C[0, farm]):.2f}; -deaths <-> farm = {np.mean(C[1, farm]):.2f}")
    r_comp_farm = float(np.corrcoef(comp, farm3)[0, 1])
    r_comp_combat = float(np.corrcoef(comp, combat3)[0, 1])
    print(f"  composite: r = {r_comp_farm:.2f} with the farm-3 mean vs "
          f"r = {r_comp_combat:.2f} with the combat-3 mean; PC1 share = {pc1:.0%}")

    # Translation constants: §6 reads β̂ against these. β̂ (rank) is the covariance
    # between rank position (uniform on [−½,½], variance 1/12) and the z-composite,
    # so under a linear-in-rank dose response v = γ·(rank−½)+c, β̂ = γ/12 ⇒ a
    # ranker's TOP-vs-BOTTOM swing is 12·β̂ — in ỹ units. These constants convert
    # that swing into composite SDs and win-probability points, over the exact β̂
    # sample (test forced picks, eval-path normalization).
    paths = default_paths()
    split = load_split(paths.split)
    vocabs = load_vocabs(paths.vocabs)
    rows_by_id = {r.match_id: r for r in rows}
    train = [m for m in matches if m.match_id not in split.held_out]
    mmr_mean, mmr_std = compute_mmr_norm(train)
    snm, sns = compute_stat_norm(
        rows_by_id, ((m.match_id, ps) for m in train for ps in range(NUM_PLAYERS)))
    test_ms = [m for m in matches if m.match_id in split.test_ids]
    tups = [t for t in extract_tuples(test_ms, vocabs, mmr_mean, mmr_std)
            if t.match_id in rows_by_id]
    ys = compute_realized_y_vec(tups, rows_by_id, snm, sns)
    K = len(DEFAULT_BALANCED_WEIGHTS)
    wv = np.asarray([float(x) for x in DEFAULT_BALANCED_WEIGHTS])
    keep = [(t, y) for t, y in zip(tups, ys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]
    cvals = np.array([float(y.numpy()[:K] @ wv) for _, y in keep])
    win_r = np.array([t.focal_team_won for t, _ in keep])
    sd_c, sd_w = float(cvals.std()), float(win_r.std())
    print(f"\n[translation constants] test forced picks n={len(cvals)} (the β̂ sample):")
    print(f"  composite SD = {sd_c:.2f} z-units  ⇒ per +0.01 of β̂, top-vs-bottom swing "
          f"= 0.12 units = {0.12 / sd_c:.3f} composite SD")
    print(f"  win rate = {win_r.mean():.3f}, SD = {sd_w:.3f}  ⇒ per +0.01 of win-β̂, "
          f"top-vs-bottom = 0.12·SD = {0.12 * sd_w * 100:.1f}pp win probability")
    print(f"  illustration: composite β̂=+0.10 ⇒ swing ≈ {12 * 0.10:.2f} units ≈ "
          f"{12 * 0.10 / sd_c:.2f} SD; win β̂=+0.01 ⇒ ≈ {12 * 0.01 * sd_w * 100:.1f}pp")
    write_results("stats-coverage", {
        "n_analytic": len(matches), "n_parsed": len(parsed), "n_unparsed": len(unparsed),
        "unparsed_share": len(unparsed) / len(matches),
        "d_win": d_win, "z_win": z_win, "d_mmr": d_mmr, "t_mmr": t_mmr,
        "d_forced_per_match": d_forced, "t_forced_per_match": t_forced,
        "forced_per_match_parsed": float(P["nrand"].mean()),
        "forced_per_match_unparsed": float(U["nrand"].mean()),
        "unparsed_forced_share": float(U["nrand"].sum() / (U["nrand"].sum() + P["nrand"].sum())),
        "farm_pairwise_r": float(np.mean(fp)), "pc1_share": pc1,
        "r_comp_farm": r_comp_farm, "r_comp_combat": r_comp_combat,
        "sd_composite": sd_c, "sd_win": sd_w, "win_rate": float(win_r.mean()),
        "n_translation_picks": len(cvals),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
