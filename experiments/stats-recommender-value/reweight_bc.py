"""reweight_bc.py — the best-justified recommender: BC prior + causal tilt.

A deployable policy: *tilts* the strong human-consensus prior toward Trial's clean
(random-only, unconfounded) causal value:

    π_β(a) ∝ π_BC(a) · exp(β · ẑ(a)),   ẑ = within-state z-scored Trial composite value
    score_β(a) = log π_BC(a) + β · ẑ(a)

β = 0 is exactly BC (a product-of-experts gate: an action BC deems implausible stays out no
matter its value — the downside protection); large β recovers Trial's ranking. The full β sweep —
V(argmax), V(soft = sample ~ π_β), β̂, top==BC — and its selection check run on the VAL split
(the selection set); the gaps at the fixed β=1 (the unit tilt: a 1-SD value bump trades
1:1 against log π_BC) are read on the held-out TEST split, so no β is chosen on the eval data
(decision-time states, exact IPW like run.py). EXPLORATORY: the tilt uses Trial's (unresolved) causal
value, so a − TYPICAL edge is a lead for an A/B, not a settled result. Inference only,
no training.

Run: DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/reweight_bc.py
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from dota2ad.core import (NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches,
                          load_split, load_stats_rows, load_vocabs, mech_propensity)
from dota2ad.core.collate import policy_collate
from dota2ad.models import QNetStats, load_policy
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.bootstrap import cluster_bootstrap_ci
from dota2ad.eval.results import write_results
from dota2ad.eval.causal_rank import rank_pct
from dota2ad.training.stats_simulator import compute_stat_norm, scalarize_q
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()
    p = default_paths()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(p.matches); vocabs = load_vocabs(p.vocabs)
    vs = len(vocabs.draft_id_to_index)
    split = load_split(p.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    valm = [m for m in matches if m.match_id in split.val_ids]
    testm = [m for m in matches if m.match_id in split.test_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train)
    rows = {r.match_id: r for r in load_stats_rows()}

    def _it():
        for m in train:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(rows, _it())
    K = len(DEFAULT_BALANCED_WEIGHTS)
    wnp = np.asarray([float(x) for x in DEFAULT_BALANCED_WEIGHTS[:K]])
    w_t = torch.tensor(wnp, dtype=torch.float32, device=dev)

    policy = load_policy(p.policy_ckpt, vocabs, dev); policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(p.stats_dqn_ckpt, vs, dev); q.eval(); q.requires_grad_(False)
    trial = QNetStats.load_from_ckpt(p.models / "trial.pt", vs, dev); trial.eval(); trial.requires_grad_(False)

    def pick_arrays(msel) -> dict[str, Any]:
        """Per-forced-pick arrays over `msel`: feasible-set scores + realized outcome."""
        vt = extract_tuples(msel, vocabs, mmr_mean, mmr_std)
        vwr = [t for t in vt if t.match_id in rows]
        vys = compute_realized_y_vec(vwr, rows, snm, sns)
        kept = [(t, y) for t, y in zip(vwr, vys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]
        LOGBC, ZEXO, FI = [], [], []                    # per-pick feasible arrays (ragged)
        M_, C, PIA, IWB, mids = [], [], [], [], []
        ARG_BC, ARG_Q, ARG_EXO = [], [], []
        with torch.no_grad():
            for i in range(0, len(kept), 256):
                ch = kept[i:i + 256]
                b = policy_collate([t.sample for t, _ in ch], device=dev)
                bc = policy(b).cpu().exp()[:, :vs]
                qc = scalarize_q(q(b), w_t).cpu(); ec = scalarize_q(trial(b), w_t).cpu()
                for j, (t, y) in enumerate(ch):
                    feas = list(t.sample.cand_idx); fi = feas.index(t.action_idx)
                    pr = bc[j, feas].numpy(); pr = pr / pr.sum()
                    qf = qc[j, feas].numpy(); ef = ec[j, feas].numpy()
                    z = (ef - ef.mean()) / (ef.std() or 1.0)              # within-state z-scored Trial value
                    LOGBC.append(np.log(pr)); ZEXO.append(z); FI.append(fi)
                    M_.append(len(feas)); C.append(float(y[:K].numpy() @ wnp)); PIA.append(pr[fi])
                    IWB.append(1.0 / mech_propensity(t.sample.cand_type)[fi])   # 1/P_mech(A): true IPW base
                    ARG_BC.append(1.0 if fi == int(pr.argmax()) else 0.0)
                    ARG_Q.append(1.0 if fi == int(qf.argmax()) else 0.0)
                    ARG_EXO.append(1.0 if fi == int(ef.argmax()) else 0.0)
                    mids.append(t.match_id)
        M_, C, PIA, IWB, ARG_BC, ARG_Q, ARG_EXO = map(np.array, (M_, C, PIA, IWB, ARG_BC, ARG_Q, ARG_EXO))
        return dict(LOGBC=LOGBC, ZEXO=ZEXO, FI=FI, M=M_, C=C, PIA=PIA, IWB=IWB, mids=mids,
                    ARG_BC=ARG_BC, ARG_Q=ARG_Q, ARG_EXO=ARG_EXO, WPROP=IWB / M_)

    def eval_beta(d, beta):
        n = len(d["C"])
        na = np.zeros(n); ns = np.zeros(n); rd = np.zeros(n); agree = np.zeros(n)
        for i in range(n):
            s = d["LOGBC"][i] + beta * d["ZEXO"][i]
            arg = int(s.argmax())
            na[i] = 1.0 if d["FI"][i] == arg else 0.0
            e = np.exp(s - s.max()); soft = e / e.sum(); ns[i] = soft[d["FI"][i]]
            rd[i] = (rank_pct(s) - 0.5)[d["FI"][i]]
            agree[i] = 1.0 if arg == int(d["LOGBC"][i].argmax()) else 0.0   # top pick == BC's top pick?
        return na, ns, rd, agree

    def ci(contrib, mids):
        lo, m, hi = cluster_bootstrap_ci(lambda ix: float(contrib[ix].mean()), mids, n_boot=a.bootstrap)
        return m, lo, hi, ("*" if (lo > 0 or hi < 0) else " ")

    va = pick_arrays(valm)
    ta = pick_arrays(testm)
    print(f"val (selection) random picks n={len(va['C'])} ({len(set(va['mids']))} matches); "
          f"test (single read) n={len(ta['C'])} ({len(set(ta['mids']))} matches); "
          f"outcome=z-composite (test sd={ta['C'].std():.2f})\n")

    print("β sweep on VAL (the selection set)  π ∝ π_BC · exp(β·ẑ_Trial)   (β=0 ⇒ BC; large β ⇒ Trial):")
    print(f"  {'β':>6} {'V(argmax)':>10} {'V(soft)':>9} {'β̂(rank)':>9} {'top==BC':>8}")
    betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0]
    results = []
    na_by_beta = []
    for beta in betas:
        na, ns, rd, agree = eval_beta(va, beta)
        na_by_beta.append(na)
        varg = (na * va["IWB"] * va["C"]).mean(); vsoft = (ns * va["IWB"] * va["C"]).mean()
        bhat = (va["WPROP"] * rd * va["C"]).mean()
        results.append((beta, varg))
        print(f"  {beta:6.2f} {varg:+10.3f} {vsoft:+9.3f} {bhat:+9.4f} {agree.mean():7.1%}")

    beta_star = 1.0   # fixed in advance, not tuned: the unit tilt (a 1-SD value bump trades 1:1 against log π_BC)
    grid_best = max(results, key=lambda r: r[1])[0]    # val-grid V(argmax) peak — reported alongside the fixed β=1

    # β=1 is fixed in advance and the sweep runs on val, so test never selects β — but check whether β=1
    # is a cherry-pick even within val: re-select β by V(argmax) inside a match-clustered bootstrap and
    # report how often it lands on 1.0 and how far select-vs-fix would move the − TYPICAL gap.
    nb = len(betas)
    Vc = np.stack([na_by_beta[k] * va["IWB"] * va["C"] for k in range(nb)])
    Gc = np.stack([va["IWB"] * (na_by_beta[k] - va["PIA"]) * va["C"] for k in range(nb)])
    naive = float(Gc[betas.index(beta_star)].mean())
    by_g: dict = {}
    for i, g in enumerate(va["mids"]):
        by_g.setdefault(g, []).append(i)
    clusters = [np.array(v) for v in by_g.values()]; nc = len(clusters)
    rng = np.random.default_rng(0)
    nboot_sel = min(a.bootstrap, 2000)
    sel_gaps, win1 = [], 0
    for _ in range(nboot_sel):
        ib = np.concatenate([clusters[c] for c in rng.integers(0, nc, nc)])
        kbest = int(np.argmax([Vc[k][ib].mean() for k in range(nb)]))
        win1 += int(betas[kbest] == beta_star)
        sel_gaps.append(float(Gc[kbest][ib].mean()))
    print(f"  selection check on val (B={nboot_sel}): grid-argmax lands on β=1 in {win1 / nboot_sel:.0%} of "
          f"match-resamples; select-vs-fix moves the gap by {float(np.mean(sel_gaps)) - naive:+.3f}.")

    v_random = float((ta['WPROP'] * ta['C']).mean())
    v_typical = float((ta['PIA'] * ta['IWB'] * ta['C']).mean())
    v_bcmode = float((ta['ARG_BC'] * ta['IWB'] * ta['C']).mean())
    v_q = float((ta['ARG_Q'] * ta['IWB'] * ta['C']).mean())
    v_trial = float((ta['ARG_EXO'] * ta['IWB'] * ta['C']).mean())
    print("\nheld-out TEST — reference recommenders (exact IPW V, weight π(A)/P_mech(A)):")
    print(f"  RANDOM legal        V={v_random:+.3f}")
    print(f"  TYPICAL (draw ~ BC) V={v_typical:+.3f}")
    print(f"  argmax BC (mode)    V={v_bcmode:+.3f}")
    print(f"  argmax Q (shipped)  V={v_q:+.3f}")
    print(f"  argmax Trial         V={v_trial:+.3f}")

    na, ns, rd, agree = eval_beta(ta, beta_star)
    print(f"\nβ = {beta_star:.1f} on held-out TEST (unit tilt, fixed in advance; val-grid V-peak is "
          f"β={grid_best:.2f}) — gaps with match-clustered CIs:")
    gaps: dict[str, float] = {}
    for key, lab, contrib in [
            ("gap_typical", "reweight-BC argmax − TYPICAL", ta["IWB"] * (na - ta["PIA"]) * ta["C"]),
            ("gap_bcmode", "reweight-BC argmax − argmax BC (mode)", ta["IWB"] * (na - ta["ARG_BC"]) * ta["C"]),
            ("gap_q", "reweight-BC argmax − argmax Q (shipped)", ta["IWB"] * (na - ta["ARG_Q"]) * ta["C"])]:
        m, lo, hi, s = ci(contrib, ta["mids"])
        gaps[key], gaps[f"{key}_lo"], gaps[f"{key}_hi"] = float(m), float(lo), float(hi)
        print(f"  Δ {lab:42} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{s}  ({m / ta['C'].std():+.1%} SD)")

    write_results("reweight-bc", {
        **gaps,
        "val_peak_beta": float(grid_best), "sel_beta1_rate": win1 / nboot_sel,
        "sel_vs_fix_shift": float(np.mean(sel_gaps)) - naive,
        "v_random": v_random, "v_typical": v_typical, "v_bcmode": v_bcmode,
        "v_q": v_q, "v_trial": v_trial,
        "n_test_picks": len(ta["C"]), "n_test_matches": len(set(ta["mids"])),
    })
    print("\nreweight-BC = human prior + causal tilt (β=0 is exactly BC; large β is Trial). EXPLORATORY:")
    print("the tilt leans on Trial's (unresolved) causal value, so a − TYPICAL edge is a lead for an A/B,")
    print("not a settled win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
