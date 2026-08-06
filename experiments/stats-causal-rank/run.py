"""stats-causal-rank: does a ranker order actions by their causal effect?

A design-based test on the random-pick natural experiment. At each held-out random
pick the server forced an *exogenous* action A given the state — drawn from the
reverse-engineered timeout mechanism P_mech (a side-coin then a uniform draw over
heroes∪side, NOT uniform-over-all-m; see experiments/random-mechanism). For a ranker
σ(s,a) scored at the decision-time state, with δ(s,a) the
within-state deviation of the realized action and w = (1/m)/P_mech(A) the propensity
weight that maps the P_mech sample onto the uniform estimand,

    β̂ = mean_i w_i · δ(s_i, A_i) · ỹ_i          (ỹ = z-normalized realized stat)

is unbiased for E_s[Cov_a(σ, v)] — the average within-state covariance between the
ranker's scores and the true action values (REPORT.md, Appendix A). The
match-level context cancels (constant within a state ⇒ killed by the centering), so
match variance is a power cost, not a bias. Two transforms: rank (δ = pct_rank−0.5,
primary) and score (δ = (σ−mean)/sd, secondary). Rankers: BC (policy prob — human-consensus
reference ranker, whose positivity is an empirical result), Q (stats-DQN — the shipped
recommender), permuted (negative control, null by construction). Match-clustered
bootstrap 95% CIs; significant = CI excludes 0. The PRIMARY endpoint is the composite β̂
(rank) — fixed, with the full analysis plan, before the test split's single read (held-out
protocol; selection of every kind confined to the val split). Win,
the score transform, and the per-stat battery are SECONDARY, and the per-stat p-values are
Benjamini-Hochberg FDR-controlled at 5% (raw per-stat counts are descriptive). Q-vs-BC is reported as **preservation of effect**: the
fraction ρ = β̂_Q/β̂_BC of BC's ranking skill Q retains, via a ratio bootstrap that re-estimates
β̂_BC on every resample — so the reference effect's own uncertainty is propagated and the Q,BC
correlation exploited (the synthesis approach, not a fixed margin that treats β̂_BC as known).
The worst-case retained fraction is the CI lower bound; κ (`--ni-margin`) is only a stated
tolerance. See REPORT.md §5 / Appendix A.

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-causal-rank/run.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches, load_split,
    load_stats_rows, load_vocabs,
)
from dota2ad.core.io import DEFAULT_EXCLUDES, load_excluded_matches
from dota2ad.core.mechanism import iw_to_uniform
from dota2ad.models import QNetStats, load_policy
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.stats_specs import STAT_SPECS
from dota2ad.eval.causal_rank import beta_ci, beta_ci_p, bh_fdr, compute_deviations
from dota2ad.eval.bootstrap import cluster_bootstrap
from dota2ad.eval.results import write_results
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--subset", choices=["all", "online", "disconnect"], default="all",
                    help="timeout subpopulation: online (present but slow) / disconnect (AFK)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ni-margin", type=float, default=0.25,
                    help="non-inferiority / equivalence margin as a fraction of BC's own β̂ "
                         "(reference-effect framing); 0.25 = a quarter of the human ranking skill")
    ap.add_argument("--q-ckpt", type=str, default=None,
                    help="stats-DQN checkpoint to score (default: models/stats_dqn.pt); "
                         "point at an alternative checkpoint, e.g. one retrained with --seed")
    ap.add_argument("--policy-ckpt", type=str, default=None,
                    help="BC policy checkpoint to score (default: models/policy.pt)")
    args = ap.parse_args()

    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boot = args.bootstrap

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
    stat_norm_mean, stat_norm_std = compute_stat_norm(stats_rows_by_id, _iter())

    policy_ckpt = Path(args.policy_ckpt) if args.policy_ckpt else paths.policy_ckpt
    q_ckpt = Path(args.q_ckpt) if args.q_ckpt else paths.models / "stats_dqn.pt"
    print(f"Scoring rankers  BC={policy_ckpt}  Q={q_ckpt}")
    policy = load_policy(policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(q_ckpt, vocab_size, device)
    q.eval()

    tuples = extract_tuples(test_matches, vocabs, mmr_mean, mmr_std,
                            disconnect_only=(args.subset == "disconnect"),
                            online_only=(args.subset == "online"))
    d = compute_deviations(tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size,
                           stats_rows_by_id, stat_norm_mean, stat_norm_std, device, seed=args.seed)
    K, mids, Y, comp = d.K, d.mids, d.Y, d.comp
    win_z = (d.win - d.win.mean()) / (d.win.std() or 1.0)
    w = d.w_prop     # IW (1/m)/P_mech(A): corrects the true forced-pick propensity to uniform
    n_matches = len(set(mids))

    def bci(c: np.ndarray):
        return beta_ci(c, mids, boot)

    def block(title, dq_c, dp_c, dbc, figures: dict | None = None, prefix: str = ""):
        print("\n" + "=" * 84); print(title); print("=" * 84)
        # Per-stat counts here use each ranker's SINGLE deployed ordering against every
        # stat (symmetric across rankers — the report's headline column); Q's
        # stat-specialized heads are the separate battery printed below.
        bc_ref: dict[str, tuple[float, float]] = {}
        for name, d_comp in [("BC", dbc), ("Q", dq_c), ("permuted", dp_c)]:
            d_ps = np.tile(d_comp[:, None], (1, K))
            bc_, lo_, hi_ = bci(w * d_comp * comp)
            bw, lw, hw = bci(w * d_comp * win_z)
            if name == "BC":
                bc_ref = {"composite": (bc_, lo_), "win": (bw, lw)}   # reference effect + its CI floor
            # per-stat battery is SECONDARY: report raw count and the BH-FDR-controlled count
            pv = np.array([beta_ci_p(w * d_ps[:, k] * Y[:, k], mids, boot)[3] for k in range(K)])
            nraw = int((pv < 0.05).sum()); nfdr = int(bh_fdr(pv).sum())
            if figures is not None:
                stem = {"BC": "bc", "Q": "q", "permuted": "perm"}[name]
                cstem = {"BC": "bc", "Q": "qc", "permuted": "perm"}[name]
                figures[f"{prefix}beta_{stem}"] = float(bc_)
                figures[f"{prefix}beta_{stem}_lo"] = float(lo_)
                figures[f"{prefix}beta_{stem}_hi"] = float(hi_)
                figures[f"{prefix}win_{stem}"] = float(bw)
                figures[f"{prefix}win_{stem}_lo"] = float(lw)
                figures[f"{prefix}win_{stem}_hi"] = float(hw)
                figures[f"{prefix}perstat_raw_{cstem}"] = nraw
                figures[f"{prefix}perstat_bhfdr_{cstem}"] = nfdr
                # Benjamini-Yekutieli count (manifest-only; the report quotes it
                # as the dependence-assumption sensitivity of the BH counts).
                c_k = float(np.sum(1.0 / np.arange(1, K + 1)))
                thr = 0.05 * (np.arange(1, K + 1) / K) / c_k
                hits = np.nonzero(np.sort(pv) <= thr)[0]
                figures[f"{prefix}perstat_byfdr_{cstem}"] = int(hits.max() + 1) if hits.size else 0
            sc = "*" if (lo_ > 0 or hi_ < 0) else " "
            sw = "*" if (lw > 0 or hw < 0) else " "
            print(f"  {name:9s} composite β̂={bc_:+.3f} [{lo_:+.3f},{hi_:+.3f}]{sc}   "
                  f"win β̂={bw:+.3f} [{lw:+.3f},{hw:+.3f}]{sw}   "
                  f"per-stat {nraw:2d}/{K} raw · {nfdr:2d}/{K} BH-FDR")
        m = args.ni_margin
        for nm, out in [("composite", comp), ("win", win_z)]:
            q_c = w * dq_c * out
            bc_c = w * dbc * out
            bd, ld, hd = bci(q_c - bc_c)
            if figures is not None:
                dkey = "dbeta" if nm == "composite" else "dbeta_win"
                figures[f"{prefix}{dkey}"] = float(bd)
                figures[f"{prefix}{dkey}_lo"] = float(ld)
                figures[f"{prefix}{dkey}_hi"] = float(hd)
            sd = "*" if (ld > 0 or hd < 0) else " "
            print(f"    Q−BC {nm:10s} Δβ̂ = {bd:+.4f} [{ld:+.4f},{hd:+.4f}]{sd}  (>0 ⇒ Q beats BC)")
            # Preservation-of-effect (synthesis, not fixed-margin): the fraction ρ = β̂_Q / β̂_BC of
            # BC's effect that Q retains, with a clustered-bootstrap CI that RE-ESTIMATES β̂_BC on
            # every resample — so it propagates the reference effect's OWN uncertainty and the Q,BC
            # correlation, instead of treating β̂_BC as a known constant. No pre-chosen margin: the
            # worst-case retained fraction is read straight off the CI lower bound. (κ, if stated, is
            # only a tolerance the reader picks; NI "holds" iff the data prove ρ ≥ 1−κ.)
            ref_pt, ref_lo = bc_ref[nm]
            if ref_lo <= 0:   # Fieller condition: a denominator whose own CI touches 0
                print(f"      vs BC effect {ref_pt:+.3f} (CI floor {ref_lo:+.3f} ≤ 0): the "
                      f"preservation ratio is Fieller-unstable — read the Δβ̂ CI above directly.")
                continue
            rho = float(q_c.sum() / bc_c.sum())
            rb = cluster_bootstrap(lambda ix, q_c=q_c, bc_c=bc_c: float(q_c[ix].sum() / bc_c[ix].sum()), mids, boot)
            rlo, rhi = float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))
            if figures is not None:
                rkey = "rho" if nm == "composite" else "rho_win"
                figures[f"{prefix}{rkey}"] = rho
                figures[f"{prefix}{rkey}_lo"] = rlo
                figures[f"{prefix}{rkey}_hi"] = rhi
            niv = "holds" if rlo >= (1 - m) else "not shown"
            print(f"      Q retains {rho:.0%} of BC's effect [{rlo:.0%}, {rhi:.0%}] "
                  f"(β̂_BC uncertainty propagated) → worst case loses ≤{max(0.0, 1 - rlo):.0%} of BC's "
                  f"ranking skill. NI at a stated κ={m:.0%} tolerance: {niv}.")

    print(f"\nHeld-out test random picks: {d.n} ({n_matches} matches, subset={args.subset}); "
          f"Q k_stats={K}; B={boot}")
    print("Primary endpoint (fixed before the test read; single test): composite β̂ (rank), the balanced-weight "
          "composite.\n  Secondary/descriptive: win, the score transform, and the per-stat battery "
          f"({K} stats), the last\n  multiplicity-controlled by Benjamini-Hochberg FDR at 5% (the "
          "'raw' counts are unadjusted).")
    figures: dict = {}
    block("RANK variant (primary)   δ = percentile_rank − 0.5",
          d.qc_rank, d.pc_rank, d.bc_rank, figures=figures)
    block("SCORE variant (secondary)   δ = (score − mean)/sd",
          d.qc_score, d.pc_score, d.bc_score, figures=figures, prefix="score_")

    print("\n  Per-stat rank-β̂ (secondary; per stat k: Q_k = Q's stat-k head, Qc = Q's composite "
          "ordering (symmetric to BC), BC = pick-probability ordering;\n  Q_k-vs-BC is NOT a "
          "head-to-head (specialized heads vs one ordering) — Qc-vs-BC is; "
          "'*' CI excludes 0, '†' survives BH-FDR 5% within its column):")
    q_res = [beta_ci_p(w * d.q_rank[:, k] * Y[:, k], mids, boot) for k in range(K)]
    qc_res = [beta_ci_p(w * d.qc_rank * Y[:, k], mids, boot) for k in range(K)]   # Q's ONE deployed (composite) ordering, scored vs each stat
    bc_res = [beta_ci_p(w * d.bc_rank * Y[:, k], mids, boot) for k in range(K)]   # BC's one ranking, scored vs each stat
    q_fdr = bh_fdr(np.array([r[3] for r in q_res]))
    qc_fdr = bh_fdr(np.array([r[3] for r in qc_res]))
    bc_fdr = bh_fdr(np.array([r[3] for r in bc_res]))
    for k in range(K):
        label = STAT_SPECS[k].label if k < len(STAT_SPECS) else f"stat{k}"
        bq, lq, hq, _ = q_res[k]
        bqc, lqc, hqc, _ = qc_res[k]
        bb, lb, hb, _ = bc_res[k]
        fq = ("*" if (lq > 0 or hq < 0) else " ") + ("†" if q_fdr[k] else " ")
        fqc = ("*" if (lqc > 0 or hqc < 0) else " ") + ("†" if qc_fdr[k] else " ")
        fb = ("*" if (lb > 0 or hb < 0) else " ") + ("†" if bc_fdr[k] else " ")
        print(f"    {label:34s} Q_k {bq:+.3f}[{lq:+.3f},{hq:+.3f}]{fq} "
              f"Qc {bqc:+.3f}[{lqc:+.3f},{hqc:+.3f}]{fqc} "
              f"BC {bb:+.3f}[{lb:+.3f},{hb:+.3f}]{fb}")
    print(f"    BH-FDR 5% counts — Q_k {int(q_fdr.sum())}/{K} · Qc {int(qc_fdr.sum())}/{K} · "
          f"BC {int(bc_fdr.sum())}/{K}")
    figures["perstat_bhfdr_q_heads"] = int(q_fdr.sum())
    # Benjamini–Yekutieli sensitivity: BH at q/c(K) is valid under ARBITRARY dependence,
    # so it needs no positive-dependence assumption (the price is conservatism).
    c_m = float(np.sum(1.0 / np.arange(1, K + 1)))
    by_q = bh_fdr(np.array([r[3] for r in q_res]), q=0.05 / c_m)
    by_qc = bh_fdr(np.array([r[3] for r in qc_res]), q=0.05 / c_m)
    by_bc = bh_fdr(np.array([r[3] for r in bc_res]), q=0.05 / c_m)
    print(f"    BY (arbitrary-dependence) counts — Q_k {int(by_q.sum())}/{K} · "
          f"Qc {int(by_qc.sum())}/{K} · BC {int(by_bc.sum())}/{K}")

    print("\nReading: BC & Q composite β̂ > 0 = both order actions by causal effect; permuted ≈ 0 "
          "= clean null.\n  Q vs BC is preservation-of-effect: the 'Q retains X% of BC's effect "
          "[lo, hi]' line is a ratio whose\n  bootstrap re-estimates β̂_BC each resample (reference "
          "uncertainty propagated); worst-case loss = 1−lo.\n  Q retains ~all of BC's ranking skill "
          "(no harm) with the upside open (unresolved) — but this conditions\n  on the single "
          "checkpoints; a recipe-level claim needs seed-inclusive retraining. The per-stat 'BH-FDR'\n"
          "  count is the multiplicity-controlled secondary readout; 'raw' is descriptive.")
    if args.subset == "all" and args.q_ckpt is None and args.policy_ckpt is None:
        figures["n_picks"] = int(d.n)
        figures["n_matches"] = int(n_matches)
        # Derived quantities the report quotes: the ρ-interval complements, the
        # mean importance weight, and the share of one-bag states (w ≡ 1 for
        # every feasible action — P_mech already uniform).
        figures["rho_loss_max"] = 1 - figures["rho_lo"]
        figures["rho_upside"] = figures["rho_hi"] - 1
        figures["mean_w"] = float(w.mean())
        onebag = [
            all(abs(iw_to_uniform(t.sample.cand_type, i) - 1.0) < 1e-9
                for i in range(len(t.sample.cand_idx)))
            for t in tuples
        ]
        figures["onebag_share"] = float(np.mean(onebag))
        # §6 magnitude translations (linear-in-rank reading): top-vs-bottom swing
        # = 12·β̂ — in composite z-units, composite SDs, and win percentage points
        # (win β̂ is on win's within-sample z scale; ×sd(win) restores probability units).
        figures["swing12_bc"] = 12 * figures["beta_bc"]
        figures["swing12_bc_sd"] = 12 * figures["beta_bc"] / float(comp.std())
        figures["swing12_win_pp_bc"] = 12 * figures["win_bc"] * float(d.win.std()) * 100
        # Data-size guidance for reproducers: Δβ̂ is a match-clustered mean, so its
        # CI half-width shrinks as 1/√N; the corpus multiple at which the CI would
        # exclude 0 — CONDITIONAL on the point estimate being the true effect and
        # the split shares staying fixed (more data also retrains the models).
        h = (figures["dbeta_hi"] - figures["dbeta_lo"]) / 2
        n_raw = len(matches) + len(load_excluded_matches(DEFAULT_EXCLUDES))
        x = (h / figures["dbeta"]) ** 2 if figures["dbeta"] > 0 else float("inf")
        figures["n_raw_matches"] = int(n_raw)
        figures["nstar_dbeta_x"] = float(x)
        figures["nstar_dbeta_matches"] = float(n_raw * x)
        figures["nstar_dbeta_kmatches"] = float(n_raw * x / 1000)
        print(f"\nData-size guidance (conditional — assumes Δβ̂ = {figures['dbeta']:+.3f} is the "
              f"true effect,\n  fixed split shares; SE ∝ 1/√N): the Q−BC CI excludes 0 at "
              f"≈{x:.1f}× the corpus ≈ {n_raw * x:,.0f}\n  collected matches (current {n_raw:,}).")
        write_results("stats-causal-rank", figures)
    else:
        print("\n(non-headline invocation — results manifest not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
