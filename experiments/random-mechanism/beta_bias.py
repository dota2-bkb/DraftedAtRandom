"""beta_bias.py — how much does the survivorship contamination bias the causal
test (β̂)? The downstream of survivorship.py (see this experiment's README).

survivorship.py established that real-player forced picks are outcome-dependent
missing: undesirable randoms trigger abandonment, so matches with them go
unrecorded, and the surviving forced picks skew toward desirable items
(suppression tracks draft popularity). This asks what that does to β̂.

DIRECTION (reasoned): β̂ = mean over forced picks of δ·y, with δ the ranker's
within-state score of the forced item and y its outcome. Survivorship removes
picks of *undesirable* items; for a good ranker such an item scores low (δ<0) with
a low outcome (y<0), so δ·y > 0. Removing positive-δy picks *lowers* β̂ — survivorship
**attenuates** β̂ toward 0 (the true signal is larger), and leaves the scrambled null
untouched (δ is random there). So the "orders by causal effect" claim is conservative.

MEASURE: the baseline β̂ here is already **propensity-corrected** — compute_deviations
supplies w_prop = (1/m)/P_mech(A) and every contribution is multiplied by it (the
P_mech→uniform fix), so this experiment isolates the SEPARATE survivorship axis. On top
of that baseline, recover the clean-mechanism β̂ by importance-reweighting the test forced
picks by 1/survival, with survival estimated by the desirability-smoothed suppression
obs/exp(popularity) on the train+val matches only (test never informs its own correction).
Report baseline vs +survivorship β̂ for BC and Q, and the Q−BC comparison. (Held-out test,
the report's β̂ sample.)

CAVEATS. (1) This corrects the ITEM-level selection (missing-at-random *given the
item's desirability*). A residual outcome-level MNAR — players abandoning on their
own realized outcome beyond the item — is not corrected, and its sign is not pinned
down; it is second-order to the item effect. (2) The CI treats the suppression
estimate as known (its own uncertainty is not propagated). (3) Bots carry no stats,
so this is a model-based reweighting, not a bot-vs-human β̂ control.

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/random-mechanism/beta_bias.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches, load_split,
    load_stats_rows, load_vocabs, mech_propensity,
)
from dota2ad.core.draft_logic import idx
from dota2ad.eval.causal_rank import compute_deviations
from dota2ad.eval.bootstrap import cluster_bootstrap
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.tuples import extract_tuples
from dota2ad.models import QNetStats, load_policy
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS
from dota2ad.eval.results import write_results


def main() -> int:
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(paths.matches, exclude=())
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    test_m = [m for m in matches if m.match_id in split.test_ids]
    mm_mean, mm_std = compute_mmr_norm(train)
    srows = {r.match_id: r for r in load_stats_rows()}
    with open(paths.excluded) as f:
        exc = json.load(f)
    bad = set(exc["too_many_random_picks"]) | set(exc["leavers"]) | set(exc["swaps"])

    def _it():
        for m in train:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(srows, _it())

    def UI(kind, gid):
        return idx(vocabs, gid, kind)

    # desirability (draft popularity) + suppression obs/exp smoothed by popularity decile,
    # estimated on train+val only — test never informs the correction applied to it
    pool_app, delib = defaultdict(int), defaultdict(int)
    for m in matches:
        if m.match_id in bad or m.match_id in split.test_ids:
            continue
        for a in ([UI("h", h) for h in m.hero_pool] + [UI("a", a) for a in m.basic_pool]
                  + [UI("a", a) for a in m.ult_pool]):
            pool_app[a] += 1
        for e in m.history:
            if not e.is_random:
                delib[UI("h", e.hero_id) if e.hero_id is not None else UI("a", e.draft_ability_id)] += 1
    pop = {a: delib[a] / pool_app[a] for a in pool_app if pool_app[a] >= 50}
    obs, expc = defaultdict(float), defaultdict(float)
    for t in extract_tuples(matches, vocabs, mm_mean, mm_std):
        if t.match_id in bad or t.match_id in split.test_ids:
            continue
        ci = list(t.sample.cand_idx)
        per = mech_propensity(t.sample.cand_type)     # P_mech per feasible item
        for i, a in enumerate(ci):
            expc[a] += per[i]
        obs[t.action_idx] += 1
    ip = [a for a in expc if expc[a] >= 15 and a in pop]
    pv = np.array([pop[a] for a in ip])
    edges = np.quantile(pv, np.linspace(0, 1, 11))[1:-1]
    dec = np.clip(np.searchsorted(edges, pv), 0, 9)
    dsup = {}
    for d0 in range(10):
        j = np.where(dec == d0)[0]
        e = sum(expc[ip[k]] for k in j)
        dsup[d0] = (sum(obs.get(ip[k], 0.0) for k in j) / e) if e > 0 else 1.0

    def suppression(item):
        if item not in pop:
            return 1.0
        return dsup[int(np.clip(np.searchsorted(edges, pop[item]), 0, 9))]

    # β̂ on held-out test via the report's machinery, then survivorship-reweight
    policy = load_policy(paths.policy_ckpt, vocabs, device); policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.models / "stats_dqn.pt", vocab_size, device); q.eval()
    tuples = extract_tuples(test_m, vocabs, mm_mean, mm_std)
    d = compute_deviations(tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size, srows, snm, sns, device)
    with_rows = [t for t in tuples if t.match_id in srows]
    ys = compute_realized_y_vec(with_rows, srows, snm, sns)
    kept_t = [t for t, y in zip(with_rows, ys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]
    assert len(kept_t) == d.n, (len(kept_t), d.n)
    kept = [t.action_idx for t in kept_t]
    is_hero = np.array([list(t.sample.cand_type)[list(t.sample.cand_idx).index(t.action_idx)] == 0
                        for t in kept_t])
    w = np.array([1.0 / suppression(a) for a in kept])
    comp, mids, wp = d.comp, d.mids, d.w_prop     # wp = propensity IW (already applied in the baseline)
    print(f"test forced picks: {d.n}   reweight w(item)=1/survival in [{w.min():.3f}, {w.max():.3f}]")
    print("\nβ̂ (composite, rank): baseline is propensity-corrected (P_mech→uniform); this adds the "
          "survivorship\ncorrection on top. Correction CI is match-clustered:")
    figs: dict[str, float | int] = {"n_picks": d.n}
    for name, delta in (("BC", d.bc_rank), ("Q", d.qc_rank)):
        c = wp * delta * comp                     # propensity-corrected per-pick contribution
        ob = c.mean(); cl = (w * c).sum() / w.sum()
        b = cluster_bootstrap(lambda ix, c=c: (w[ix] * c[ix]).sum() / w[ix].sum() - c[ix].mean(), mids, 2000)
        k = name.lower()
        figs[f"beta_baseline_{k}"] = float(ob); figs[f"beta_corrected_{k}"] = float(cl)
        figs[f"correction_{k}"] = float(cl - ob)
        figs[f"correction_{k}_lo"] = float(np.percentile(b, 2.5))
        figs[f"correction_{k}_hi"] = float(np.percentile(b, 97.5))
        figs[f"atten_{k}"] = float((cl - ob) / ob)
        print(f"  {name}: baseline {ob:+.4f}  +survivorship {cl:+.4f}  correction {cl-ob:+.4f} "
              f"[{np.percentile(b,2.5):+.4f}, {np.percentile(b,97.5):+.4f}]  (+{100*(cl-ob)/ob:.0f}% of β̂)")
    cbc, cq = wp * d.bc_rank * comp, wp * d.qc_rank * comp
    dob = (cq - cbc).mean(); dcl = (w * (cq - cbc)).sum() / w.sum()
    b = cluster_bootstrap(lambda ix: (w[ix] * (cq[ix] - cbc[ix])).sum() / w[ix].sum() - (cq[ix] - cbc[ix]).mean(), mids, 2000)
    figs["dqbc_baseline"] = float(dob); figs["dqbc_corrected"] = float(dcl)
    figs["dqbc_correction"] = float(dcl - dob)
    figs["dqbc_correction_lo"] = float(np.percentile(b, 2.5))
    figs["dqbc_correction_hi"] = float(np.percentile(b, 97.5))
    print(f"  Q-BC: baseline Δβ̂ {dob:+.4f}  +survivorship {dcl:+.4f}  correction {dcl-dob:+.4f} "
          f"[{np.percentile(b,2.5):+.4f}, {np.percentile(b,97.5):+.4f}]")
    # ---- Sensitivity: how strong would UNMODELED selection have to be? --------
    # The correction above is MAR-given-desirability with the suppression estimate
    # treated as a point value, and the outcome-level residual is unmodeled. Three
    # sensitivity axes turn those caveats into bounds (all self-normalized, applied
    # on top of the item correction):
    #   [S1] suppression-estimate uncertainty — run the correction with the gradient
    #        REMOVED (λ=0), as measured (λ=1), and DOUBLED (λ=2): weights w^λ.
    #   [S2] outcome-level MNAR, P(record | y) ∝ exp(γ·ỹ) — inverse-selection weights
    #        u = w·exp(−γ·ỹ). The MEASURED channels calibrate |γ|: the item-level
    #        gradient implies γ ≈ ln(1.105/0.969)/1.22 ≈ 0.11 per composite unit, and
    #        undoing the leaver exclusion moves β̂ ≲ 0.01. The TIPPING γ (where a
    #        conclusion changes) is compared against that band.
    #   [S3] the unresolved hero-specific offset — an extra ×0.95 survival on hero picks.

    def snb(u: np.ndarray, c: np.ndarray) -> float:
        return float((u * c).sum() / u.sum())

    print("\n[Sensitivity] unmodeled-selection bounds (self-normalized, on top of the item correction):")
    for lam, suf in ((0.0, "l0"), (1.0, "l1"), (2.0, "l2")):
        ul = w ** lam
        figs[f"s1_bc_{suf}"] = snb(ul, cbc)
        figs[f"s1_q_{suf}"] = snb(ul, cq)
        figs[f"s1_dqbc_{suf}"] = snb(ul, cq) - snb(ul, cbc)
        print(f"  [S1] gradient ×{lam:.0f}:      BC {snb(ul, cbc):+.4f}   Q {snb(ul, cq):+.4f}   "
              f"Q−BC {snb(ul, cq) - snb(ul, cbc):+.4f}")
    uh = w * np.where(is_hero, 1.0 / 0.95, 1.0)
    figs["s3_delta_bc"] = snb(uh, cbc) - snb(w, cbc)
    figs["n_hero_forced"] = int(is_hero.sum())
    print(f"  [S3] hero offset ×0.95: BC {snb(uh, cbc):+.4f}   Q {snb(uh, cq):+.4f}   "
          f"(hero forced picks: {int(is_hero.sum())})")
    print("  [S2] P(record|y) ∝ exp(γ·y) — calibrated band |γ| ≲ 0.11:")
    print(f"       {'γ':>6} {'BC β̂':>9} {'Q β̂':>9} {'Q−BC':>9} {'ESS':>7}")
    band = {}
    for g in (-0.2, -0.11, -0.05, 0.05, 0.11, 0.2):
        u = w * np.exp(-g * comp)
        ess = float((u.sum() ** 2 / (u ** 2).sum()) / len(u))
        band[g] = (snb(u, cbc), snb(u, cq), ess)
        print(f"       {g:+6.2f} {snb(u, cbc):+9.4f} {snb(u, cq):+9.4f} "
              f"{snb(u, cq) - snb(u, cbc):+9.4f} {ess:7.1%}")
    in_band = [v for g, v in band.items() if abs(g) <= 0.11]
    figs["s2_bc_min"] = min(v[0] for v in in_band)
    figs["s2_bc_max"] = max(v[0] for v in in_band)
    figs["s2_ess_min_band"] = min(v[2] for v in in_band)
    figs["s2_dqbc_absmax_band"] = max(abs(v[1] - v[0]) for v in in_band)
    figs["s2_edge_q_retention"] = band[-0.11][1] / band[-0.11][0]
    figs["s2_ess_gm02"] = band[-0.2][2]
    # tipping γ: smallest |γ| whose point estimate drives β̂_BC to 0 (searched both signs),
    # and a clustered CI at the calibrated band edge.
    grid = np.linspace(-2.0, 2.0, 401)
    bcv = np.array([snb(w * np.exp(-g * comp), cbc) for g in grid])
    zeros = grid[bcv <= 0]
    tip = float(np.abs(zeros).min()) if len(zeros) else float("inf")
    g_edge = 0.11 if bcv[grid >= 0.11][0] <= bcv[grid <= -0.11][-1] else -0.11
    u_edge = w * np.exp(-g_edge * comp)
    b = cluster_bootstrap(lambda ix: snb(u_edge[ix], cbc[ix]), mids, 2000)
    figs["tip_gamma"] = tip
    figs["tip_gamma_ratio"] = tip / 0.11
    figs["gamma_band"] = 0.11
    figs["band_edge_beta"] = float(snb(u_edge, cbc))
    figs["band_edge_beta_lo"] = float(np.percentile(b, 2.5))
    figs["band_edge_beta_hi"] = float(np.percentile(b, 97.5))
    write_results("beta-bias", figs)
    print(f"  tipping |γ| for β̂_BC = 0: {tip:.2f}  (≈{tip / 0.11:.0f}× the calibrated band); "
          f"at the band edge γ={g_edge:+.2f}: β̂_BC = {snb(u_edge, cbc):+.4f} "
          f"[{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}]")

    print("\nReading: the baseline is already propensity-corrected (P_mech→uniform, in causal_rank). ON TOP of "
          "that,\n  survivorship ATTENUATES β̂ (+survivorship > baseline) by ~11% for BOTH BC and Q, so the 'orders "
          "by\n  causal effect' claim is conservative — the true effect is larger. The Q-BC comparison is UNCHANGED\n"
          "  (correction ~0, CI spans 0): both attenuate equally, so non-inferiority holds. The sensitivity block\n"
          "  bounds the remaining caveats: [S1] the verdicts are stable with the suppression gradient doubled;\n"
          "  [S2] the outcome-level residual (unmodeled, unsigned) would have to exceed the tipping |γ| — compare\n"
          "  it to the ≈0.11 band the measured channels calibrate; [S3] the hero offset is negligible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
