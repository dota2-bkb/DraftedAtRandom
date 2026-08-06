"""Transportability & robustness for the random-pick causal eval.

The natural experiment identifies a LOCAL effect on the timeout subpopulation (a
LATE-style estimand). This experiment characterizes and bounds the gap to the
deliberate-pick population using observables + placebo controls — definitively
closing it would need intervention (A/B), which the setting precludes. The anchor
is the **per-stat β̂(Q)** (`stats-causal-rank`): the same design-based
estimator that is the core causal test, stratified here.

  1. Covariate shift     how timeout picks differ from the population baseline
                         (focal MMR, draft turn; standardized mean difference).
  2. Effect homogeneity  does the per-stat causal β̂(Q) vary along the shifting axes
                         (MMR / draft phase / feasibility)? Flat ⇒ the observable
                         selection does not move the effect.
  3. Placebos            P1 exogeneity  (realized action's Q-rank ⊥ MMR/turn),
                         P2 permuted    (random ranker ⇒ no per-stat β̂),
                         P3 specificity (Q's stat-X ranking ⇏ unrelated stat-Y).
  4. Conditioned skill   β̂(BC/Q) with the focal MMR overridden low→high. Flat ⇒ no
                         headroom detectable through this knob — a weak AD-skill proxy
                         (see README caveat); the headroom question is settled on the
                         stats-based axis in stats-skill-headroom.

Companion: the online/disconnect engagement split (stats-causal-rank --subset).
Held-out test split. See REPORT.md (§7, Appendix A).

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-generalization/run.py
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches, load_split,
    load_stats_rows, load_vocabs,
)
from dota2ad.models import QNetStats, load_policy
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.bootstrap import cluster_bootstrap_ci, spearman_cluster_ci
from dota2ad.eval.causal_rank import beta_ci, compute_deviations
from dota2ad.eval.results import write_results
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def covariate_shift(mmr, turn):
    """Timeout-pick covariates vs the population baseline. MMR is z-scored
    (baseline mean 0, sd 1); a deliberate pick's draft turn is ~uniform[0,49].
    |SMD| > 0.1 = a notable shift."""
    mmr, turn = np.asarray(mmr), np.asarray(turn)
    u_mean, u_sd = 24.5, math.sqrt((50 ** 2 - 1) / 12)
    print("\n" + "=" * 82)
    print("1. COVARIATE SHIFT — timeout picks vs population baseline (|SMD|>0.1 = notable)")
    print("=" * 82)
    print(f"  {'covariate':16} {'timeout mean':>13} {'baseline':>10} {'SMD':>8}")
    smd_mmr = float(mmr.mean() / 1.0)
    smd_turn = float((turn.mean() - u_mean) / u_sd)
    print(f"  {'focal MMR (z)':16} {mmr.mean():13.2f} {0.0:10.2f} {smd_mmr:+8.2f}")
    print(f"  {'draft turn':16} {turn.mean():13.2f} {u_mean:10.2f} {smd_turn:+8.2f}")
    print("  → defines the OBSERVABLE gap to the deliberate population; the unobserved traits")
    print("    the timeout subgroup is selected on (engagement, attention, tilt, execution-skill,")
    print("    ability-complexity) are NOT captured here.")
    return smd_mmr, smd_turn


def homogeneity(d, n_boot, kind):
    """Per-stat causal β̂(Q) (rank) within strata of each shifting covariate.
    Overlapping mean β̂ across strata ⇒ the observable selection doesn't move the
    effect (so the timeout↔deliberate shift on these axes shouldn't either).
    The KIND axis is different in nature: the propensity model splits exactly
    along hero/basic/ult, so a divergence there would indicate P_mech miswiring —
    a pipeline-integrity diagnostic, not a transport check."""
    contrib = d.w_prop[:, None] * d.q_rank * d.Y          # [n, K] per-stat β̂ (propensity-weighted)
    mids = np.asarray(d.mids)
    mmr, turn, nf = d.mmr, d.turn, d.n_feas
    idx_all = np.arange(d.n)

    def stratum(idx):
        c = contrib[idx]
        point = float(c.mean())                          # mean over tuples & stats = mean per-stat β̂
        lo, _, hi = cluster_bootstrap_ci(lambda ix: float(c[ix].mean()), mids[idx], n_boot=n_boot)
        se_k = c.std(0) / math.sqrt(len(idx))            # i.i.d. per-stat bar (descriptive)
        nsig = int(np.sum(np.abs(c.mean(0)) > 1.96 * se_k))
        return point, lo, hi, nsig

    print("\n" + "=" * 82)
    print("2. EFFECT HOMOGENEITY — per-stat β̂(Q) by stratum (overlap ⇒ homogeneous)")
    print("=" * 82)
    qm, qt, med = np.quantile(mmr, [1/3, 2/3]), np.quantile(turn, [1/3, 2/3]), np.median(nf)
    groups = [
        ("MMR", [("low", mmr <= qm[0]), ("mid", (mmr > qm[0]) & (mmr <= qm[1])), ("high", mmr > qm[1])]),
        ("draft turn", [("early", turn <= qt[0]), ("mid", (turn > qt[0]) & (turn <= qt[1])), ("late", turn > qt[1])]),
        ("feasibility", [("low", nf <= med), ("high", nf > med)]),
        ("kind", [("hero", kind == 0), ("basic", kind == 1), ("ult", kind == 2)]),
    ]
    print(f"  {'axis':12} {'stratum':8} {'n':>6} {'mean β̂ [95% CI]':>26} {'stats sig':>10}")
    res: dict[str, float] = {}
    for axis, strata in groups:
        for label, mask in strata:
            idx = idx_all[mask]
            p, lo, hi, nsig = stratum(idx)
            k = f"beta_{axis.replace(' ', '_').lower()}_{label}"
            res[k], res[k + "_lo"], res[k + "_hi"] = p, lo, hi
            print(f"  {axis:12} {label:8} {len(idx):6d}   {p:+.3f} [{lo:+.3f},{hi:+.3f}]   {nsig:>4}/{d.K}")
    print("  → overlapping mean β̂ across strata = the causal ranking is homogeneous over the")
    print("    OBSERVED axes only. It does NOT rule out effect modification by the unobserved traits")
    print("    the timeout subgroup is selected on (engagement, attention, execution-skill,")
    print("    ability-complexity): the outcome is the REALIZED stat, which an ability delivers only")
    print("    as well as the player uses it, so a disengaged timed-out player plausibly converts a")
    print("    strong ability into a different realized effect than an engaged one. Transport is NOT")
    print("    established here — a deployment A/B (or measuring those traits) is needed to close it.")
    return res


def placebos(d, n_boot):
    mids = np.asarray(d.mids)
    K = d.K
    print("\n" + "=" * 82)
    print("3. PLACEBO / NEGATIVE CONTROLS")
    print("=" * 82)

    # P1 exogeneity: the realized action's Q-composite percentile ⊥ any pre-pick
    # covariate (the action is server-random given the state ⇒ its rank carries no
    # covariate info; P_mech depends only on the feasible-set composition).
    qc_pct = d.qc_rank + 0.5
    r_m, lo_m, hi_m = spearman_cluster_ci(qc_pct, d.mmr, mids, n_boot=n_boot)
    r_t, lo_t, hi_t = spearman_cluster_ci(qc_pct, d.turn, mids, n_boot=n_boot)
    print("  P1 exogeneity  ρ(Q-rank of realized action, pre-pick covariate) — expect ≈0:")
    print(f"       vs focal MMR : {r_m:+.3f} [{lo_m:+.3f},{hi_m:+.3f}]")
    print(f"       vs draft turn: {r_t:+.3f} [{lo_t:+.3f},{hi_t:+.3f}]")

    # P2 permuted ranker: a within-state shuffled ranker ⇒ no per-stat β̂.
    cp = d.w_prop[:, None] * d.p_rank * d.Y
    cq = d.w_prop[:, None] * d.q_rank * d.Y
    se_p = cp.std(0) / math.sqrt(d.n); se_q = cq.std(0) / math.sqrt(d.n)
    nsig_p = int(np.sum(np.abs(cp.mean(0)) > 1.96 * se_p))
    nsig_q = int(np.sum(np.abs(cq.mean(0)) > 1.96 * se_q))
    print(f"  P2 permuted    random ranker → {nsig_p}/{K} stats sig (vs Q's {nsig_q}/{K}), "
          f"mean|β̂| {np.mean(np.abs(cp.mean(0))):.4f} — expect ≈0")

    # P3 specificity: Q's stat-X ranking should predict realized stat X (diagonal)
    # but not an unrelated stat Y (off-diagonal). M[x,y] = mean_i q_rank[i,x]·Y[i,y].
    M = np.einsum("i,ix,iy->xy", d.w_prop, d.q_rank, d.Y) / d.n
    diag = float(np.mean(np.diag(M)))
    off = float(np.mean(np.abs(M[~np.eye(K, dtype=bool)])))
    own_is_best = sum(1 for x in range(K) if int(np.argmax(M[x])) == x)
    print(f"  P3 specificity stat×stat β̂: mean diagonal (X→X) {diag:+.4f}  vs  "
          f"mean |off-diagonal| (X→Y) {off:.4f}  (ratio {diag / max(off, 1e-9):.1f}×)")
    print(f"       own-stat ranking is the STRONGEST predictor for {own_is_best}/{K} stats — "
          f"stat-SPECIFIC causal effects (off-diagonal inflated by genuine inter-stat correlation).")
    return {
        "p3_diag": diag, "p3_offdiag": off,
        "p1_rho_mmr": r_m, "p1_rho_mmr_lo": lo_m, "p1_rho_mmr_hi": hi_m,
        "p1_rho_turn": r_t, "p1_rho_turn_lo": lo_t, "p1_rho_turn_hi": hi_t,
        "p2_permuted_sig": nsig_p, "p2_q_sig": nsig_q,
    }


def conditioned_skill(d_own, tuples, policy, q, w, vocab_size, stats_rows_by_id,
                      stat_norm_mean, stat_norm_std, device, n_boot):
    """β̂(BC@mmr) and β̂(Q@mmr) with the focal's MMR overridden to low/med/high.
    Flat across conditioned skill — and no positive headroom over the own-MMR Q —
    means no headroom is detectable through this knob. The knob is the imported
    general rank, a weak AD-skill proxy (README caveat), so this speaks to the
    estimate's stability, not to skill headroom itself."""
    print("\n" + "=" * 82)
    print("4. CONDITIONED-SKILL HEADROOM — β̂ with focal MMR overridden (flat ⇒ no headroom)")
    print("=" * 82)
    print(f"  {'cond':10} {'BC composite β̂':28} {'Q composite β̂':28}")

    figs = {}

    def show(name, key, d):
        bb, lb, hb = beta_ci(d.w_prop * d.bc_rank * d.comp, d.mids, n_boot)
        bq, lq, hq = beta_ci(d.w_prop * d.qc_rank * d.comp, d.mids, n_boot)
        figs[f"cond_bc_{key}"] = bb
        figs[f"cond_bc_{key}_lo"] = lb
        figs[f"cond_bc_{key}_hi"] = hb
        figs[f"cond_q_{key}"] = bq
        figs[f"cond_q_{key}_lo"] = lq
        figs[f"cond_q_{key}_hi"] = hq
        sb = "*" if (lb > 0 or hb < 0) else " "
        sq = "*" if (lq > 0 or hq < 0) else " "
        print(f"  {name:10} BC {bb:+.3f} [{lb:+.3f},{hb:+.3f}]{sb}    "
              f"Q {bq:+.3f} [{lq:+.3f},{hq:+.3f}]{sq}")
        return d

    R = {"own": show("own", "own", d_own)}
    for name, key, z in [("low(-1z)", "low", -1.0), ("med(0z)", "med", 0.0),
                         ("high(+1z)", "high", 1.0)]:
        R[name] = show(name, key, compute_deviations(
            tuples, policy, q, w, vocab_size, stats_rows_by_id,
            stat_norm_mean, stat_norm_std, device, mmr_override=z))
    own, lo, hi = R["own"], R["low(-1z)"], R["high(+1z)"]
    for label, key, a, b in [
        ("BC@high − BC@low (is high-skill consensus a better ranker?)", "dbeta_bc_high_low",
         hi.bc_rank, lo.bc_rank),
        ("BC@high − Q@own  (headroom above current Q?)", "dbeta_bchigh_qown",
         hi.bc_rank, own.qc_rank),
    ]:
        bd, ld, hd = beta_ci(own.w_prop * (a - b) * own.comp, own.mids, n_boot)
        figs[f"cond_{key}"] = bd
        figs[f"cond_{key}_lo"] = ld
        figs[f"cond_{key}_hi"] = hd
        s = "*" if (ld > 0 or hd < 0) else " "
        print(f"  {label:52} Δβ̂ = {bd:+.4f} [{ld:+.4f},{hd:+.4f}]{s}")
    print("  → flat across conditioned MMR and no positive headroom ⇒ nothing detectable through")
    print("    this knob — a weak AD-skill proxy; see stats-skill-headroom for the stats-based axis.")
    return figs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.models / "stats_dqn.pt", vocab_size, device)
    q.eval()

    tuples = extract_tuples(test_matches, vocabs, mmr_mean, mmr_std)
    d = compute_deviations(tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size,
                           stats_rows_by_id, stat_norm_mean, stat_norm_std, device, seed=args.seed)
    print(f"Held-out random-pick tuples: {d.n}  (anchor: β̂(Q), K={d.K})")

    ys = compute_realized_y_vec([t for t in tuples if t.match_id in stats_rows_by_id],
                                stats_rows_by_id, stat_norm_mean, stat_norm_std)
    kept = [t for t, y in zip([t for t in tuples if t.match_id in stats_rows_by_id], ys, strict=True)
            if y is not None and len(t.sample.cand_idx) >= 3]
    assert len(kept) == d.n, (len(kept), d.n)
    kind = np.array([list(t.sample.cand_type)[list(t.sample.cand_idx).index(t.action_idx)]
                     for t in kept])

    smd_mmr, smd_turn = covariate_shift(d.mmr, d.turn)
    homog = homogeneity(d, args.bootstrap, kind)
    plac = placebos(d, args.bootstrap)
    cond = conditioned_skill(d, tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size,
                             stats_rows_by_id, stat_norm_mean, stat_norm_std, device,
                             args.bootstrap)
    transport = [v for k, v in homog.items()
                 if k.startswith(("beta_mmr_", "beta_draft_turn_", "beta_feasibility_"))
                 and not k.endswith(("_lo", "_hi"))]
    homog["perstat_strata_min"] = min(transport)
    homog["perstat_strata_max"] = max(transport)
    write_results("stats-generalization", {
        "smd_mmr": smd_mmr, "smd_turn": smd_turn, "n_picks": d.n,
        **plac, **cond, **homog,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
