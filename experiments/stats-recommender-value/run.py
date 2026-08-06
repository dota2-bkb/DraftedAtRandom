"""How good is the recommender's top pick vs an actual human trajectory?

BC is the *average* human pick, but a recommender deploys its *mode* (always suggest
the top-ranked pick), which cancels the noise individuals add by experimenting or
misjudging. Using the forced-random natural experiment for causal ground truth, we value
five pick policies by realized composite effect. The forced action A is drawn from
the reverse-engineered timeout mechanism P_mech(A|s) (a side-coin then uniform over
heroes∪side — NOT uniform-1/m; see experiments/random-mechanism), so for any target
policy π the Horvitz-Thompson value is V(π) = E[ (π(A|s)/P_mech(A|s))·ỹ ]:

    RANDOM legal pick    π(A) = 1/m               →  UNIFORM baseline (theoretical, matches the β̂
                                                     estimand); weight (1/m)/P_mech(A)
    TIMEOUT (mechanism)  π(A) = P_mech(A)         →  the game's ACTUAL timeout (the real "let the timer
                                                     run" counterfactual); V = mean(y) — no reweighting
                                                     (P_mech cancels; no propensity model needed)
    TYPICAL human        π(A) = π_BC(A)           →  a real-player-like DRAW from BC
    RECOMMENDER (BC)     π(A) = 1[A=argmax π_BC]  →  the consensus / crowd-mode top pick
    RECOMMENDER (Q)      π(A) = 1[A=argmax Q]     →  the *shipped* recommender's top pick
    RECOMMENDER (Trial)   π(A) = 1[A=argmax Trial]  →  the state-aware value net's top pick

`TYPICAL − RANDOM` = humans draft better than chance; `RECOMMENDER − TYPICAL` = the
wisdom-of-crowds gap (the mode beats a typical draw when value tracks popularity). We
report it for the consensus mode (argmax BC) and for each *actually deployed* ranker
(argmax Q — the site default — and argmax Trial), so the deployed policies' value is
measured directly, not inferred from their β̂ / from Q ≈ BC. (Note argmax and β̂ can
diverge — ranker-average ≠ single best pick; the per-ranker gaps below are the direct
read, valid because test is held out from Trial's fit.) Each gap uses the
true propensity P_mech directly in the weight π(A)/P_mech(A) — no uniform assumption (a
uniform-1/m shortcut would misweight P_mech's across-kind coin). Decision-time
states; match-clustered bootstrap CIs.

Faithfulness of "draw from BC" ≈ average human trajectory rests on BC being calibrated
on held-out human picks (top-1/top-5/ECE — see the train-policy log).

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/run.py
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dota2ad.core import (NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches,
                          load_split, load_stats_rows, load_vocabs, mech_propensity)
from dota2ad.core.collate import policy_collate
from dota2ad.models import load_policy, QNetStats
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.causal_rank import rank_pct, beta_ci_p, bh_fdr
from dota2ad.eval.results import write_results
from dota2ad.training.stats_simulator import compute_stat_norm, scalarize_q
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vs = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    testm = [m for m in matches if m.match_id in split.test_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train)
    rows = {r.match_id: r for r in load_stats_rows()}

    def _it():
        for m in train:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(rows, _it())
    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    K = len(DEFAULT_BALANCED_WEIGHTS)
    w = np.asarray([float(x) for x in DEFAULT_BALANCED_WEIGHTS[:K]])
    q = QNetStats.load_from_ckpt(paths.stats_dqn_ckpt, vs, device)   # the shipped ranker
    q.eval(); q.requires_grad_(False)                               # eval() ⇒ deploy-mode (off: MMR + hist-transformer dropout)
    xq = QNetStats.load_from_ckpt(paths.models / "trial.pt", vs, device)  # Trial: random-subset value net
    xq.eval(); xq.requires_grad_(False)                            # test is held out from Trial's fit ⇒ IPW valid
    w_t = torch.tensor(w, dtype=torch.float32, device=device)

    tuples = extract_tuples(testm, vocabs, mmr_mean, mmr_std)
    wr = [t for t in tuples if t.match_id in rows]
    ys = compute_realized_y_vec(wr, rows, snm, sns)
    kept = [(t, y) for t, y in zip(wr, ys, strict=True)
            if y is not None and len(t.sample.cand_idx) >= 3]

    M, PIA, ARG, ARGQ, ARGX, AGREE, AGREEX, C, PCT, mids = ([] for _ in range(10))
    IWB = []   # per-pick 1/P_mech(A): the true IPW base (replaces the m of the uniform 1/m)
    with torch.no_grad():
        for i in range(0, len(kept), 256):
            ch = kept[i:i + 256]
            b = policy_collate([t.sample for t, _ in ch], device=device)
            bc = policy(b).cpu().exp()[:, :vs]
            qc = scalarize_q(q(b), w_t).cpu()                 # [B, V] composite Q over actions
            xc = scalarize_q(xq(b), w_t).cpu()                # [B, V] composite Trial over actions
            for j, (t, y) in enumerate(ch):
                feas = list(t.sample.cand_idx)
                pr = bc[j, feas].numpy(); pr = pr / pr.sum()
                qf = qc[j, feas].numpy(); xf = xc[j, feas].numpy()
                fi = feas.index(t.action_idx)
                ab, aq, ax = int(pr.argmax()), int(qf.argmax()), int(xf.argmax())
                M.append(len(feas)); PIA.append(pr[fi])
                IWB.append(1.0 / mech_propensity(t.sample.cand_type)[fi])
                ARG.append(1.0 if fi == ab else 0.0)
                ARGQ.append(1.0 if fi == aq else 0.0)
                ARGX.append(1.0 if fi == ax else 0.0)
                AGREE.append(1.0 if ab == aq else 0.0)
                AGREEX.append(1.0 if ab == ax else 0.0)
                PCT.append(rank_pct(pr)[fi]); C.append(float(y.numpy()[:K] @ w)); mids.append(t.match_id)
    M, PIA, ARG, ARGQ, ARGX, AGREE, AGREEX, C, PCT, IWB = map(
        np.array, (M, PIA, ARG, ARGQ, ARGX, AGREE, AGREEX, C, PCT, IWB))
    WPROP = IWB / M    # (1/m)/P_mech(A): the uniform-target (RANDOM legal pick) weight
    sd = C.std()
    print(f"held-out random picks n={len(C)} ({len(set(mids))} matches); "
          f"outcome=z-composite (sd={sd:.2f}); mean feasible m={M.mean():.1f}\n")

    print("realized value g(p) by BC-percentile quintile (shape; monotone ⇒ popularity tracks value):")
    for k in range(5):
        m = (k / 5 <= PCT) & ((k + 1) / 5 > PCT if k < 4 else PCT <= 1.0)
        print(f"  Q{k + 1} (pct {k * 20}-{(k + 1) * 20}%): {C[m].mean():+.3f}")

    Wt, Wa, Wq, Wx = PIA * IWB, ARG * IWB, ARGQ * IWB, ARGX * IWB   # IPW weights π(A)/P_mech(A)
    # TWO random baselines (see the note below): the UNIFORM policy π=1/m (theoretical, matches the
    # β̂ estimand; V=mean(WPROP·C), uses P_mech) and the game's ACTUAL TIMEOUT π=P_mech (the real
    # counterfactual "let the timer run"; V=mean(C), no reweighting — P_mech cancels).
    print(f"\n  RANDOM  (uniform, π=1/m)         V={(WPROP * C).mean():+.3f}   [theoretical; uses P_mech reweight]")
    print(f"  TIMEOUT (game's actual, P_mech)  V={C.mean():+.3f}   [no reweighting: raw mean of forced picks]")
    print(f"  TYPICAL human (draw ~ BC)        V={(Wt * C).mean():+.3f}")
    print(f"  RECOMMENDER (argmax BC, mode)    V={(Wa * C).mean():+.3f}")
    print(f"  RECOMMENDER (argmax Q, shipped)  V={(Wq * C).mean():+.3f}")
    print(f"  RECOMMENDER (argmax Trial)        V={(Wx * C).mean():+.3f}")
    print(f"  (argmax Q / Trial coincide with argmax BC on "
          f"{AGREE.mean():.1%} / {AGREEX.mean():.1%} of decisions)")
    # OPE weight diagnostics per policy: direct support (picks with nonzero weight),
    # effective sample size ESS = (ΣW)²/ΣW², the largest single weight, and the
    # self-normalized value ΣWC/ΣW as a weight-noise sensitivity (HT mean = primary).
    print(f"\n  weight diagnostics (n={len(C)}):")
    wdiag: dict[str, float | int] = {}
    for (lab, W), wkey in zip((("TYPICAL (π=BC)", Wt), ("argmax BC", Wa),
                               ("argmax Q", Wq), ("argmax Trial", Wx)),
                              ("typical", "bcmode", "q", "trial"), strict=True):
        sup = int((W > 0).sum())
        ess = float(W.sum() ** 2 / (W ** 2).sum())
        wdiag[f"support_{wkey}"] = sup
        wdiag[f"ess_{wkey}"] = ess
        wdiag[f"ess_{wkey}_share"] = ess / len(C)
        wdiag[f"max_w_{wkey}"] = float(W.max())
        print(f"    {lab:16s} support={sup:6d}  ESS={ess:7.1f} ({ess / len(C):5.1%} of n)  "
              f"max w={W.max():6.1f}  self-norm V={float((W * C).sum() / W.sum()):+.3f}")
    # V-contrast family. The "crowd" sub-family — does a recommender's top pick beat a TYPICAL human —
    # is the headline claim, multiplicity-controlled by BH-FDR 5% (as §5's per-stat battery). The
    # beats-random/timeout FLOORS and the recommender-vs-recommender comparisons are secondary/descriptive
    # (a near-tautological sanity floor / a different question), reported with CIs but no multiplicity mark.
    contrasts = [
        ("TYPICAL − RANDOM(uniform)   (humans beat uniform)", (Wt - WPROP) * C, "floor"),
        ("TYPICAL − TIMEOUT(mechanism)(humans beat letting timer run)", (Wt - 1.0) * C, "floor"),
        ("RECOMMENDER(BC mode) − TYPICAL (crowd-wisdom)", IWB * (ARG - PIA) * C, "crowd"),
        ("RECOMMENDER(Q shipped) − TYPICAL (deployed) ", IWB * (ARGQ - PIA) * C, "crowd"),
        ("RECOMMENDER(Trial) − TYPICAL (deployed)      ", IWB * (ARGX - PIA) * C, "crowd"),
        ("RECOMMENDER(Q) − RECOMMENDER(BC mode)", IWB * (ARGQ - ARG) * C, "aux"),
        ("RECOMMENDER(Trial) − RECOMMENDER(BC mode)", IWB * (ARGX - ARG) * C, "aux"),
        ("RECOMMENDER(BC mode) − RANDOM(uniform)  (total)", (Wa - WPROP) * C, "floor"),
        ("RECOMMENDER(BC mode) − TIMEOUT(mechanism) (total)", (Wa - 1.0) * C, "floor"),
    ]
    stats = [(lab, fam, *beta_ci_p(contrib, mids, n_boot=args.bootstrap)) for lab, contrib, fam in contrasts]
    crowd_p = np.array([p for (_, fam, _b, _lo, _hi, p) in stats if fam == "crowd"])
    crowd_pass = list(bh_fdr(crowd_p)) if len(crowd_p) else []
    ci = 0
    for lab, fam, b, lo, hi, p in stats:
        s = "*" if (lo > 0 or hi < 0) else " "
        tag = ""
        if fam == "crowd":
            tag = f"  p={p:.3f} {'†' if crowd_pass[ci] else '·'}"; ci += 1
        print(f"  Δ {lab:52} {b:+.3f} [{lo:+.3f},{hi:+.3f}]{s}{tag}  ({b / sd:+.1%} SD)")
    print("\n  '*' = raw 95% CI excludes 0. Crowd-wisdom sub-family (recommender − TYPICAL): '†' survives")
    print("  BH-FDR 5% over that family, '·' does not. Floors (beats random/timeout) + recommender-vs-")
    print("  recommender are secondary/descriptive (no multiplicity control). Point est = mean (as §5).")
    print("\nExact IPW with the true forced-pick propensity P_mech: TYPICAL = literal draw from BC,")
    print("RECOMMENDER = literal argmax of the named ranker, weight π(A)/P_mech(A) (RANDOM uses (1/m)/P_mech).")
    print("Both the consensus mode (argmax BC) and the shipped ranker (argmax Q) are measured vs TYPICAL.")
    figures: dict[str, float | int] = {
        "n_picks": len(C), "n_matches": len(set(mids)), "sd_composite": float(sd),
        "v_random_uniform": float((WPROP * C).mean()), "v_timeout": float(C.mean()),
        "v_typical": float((Wt * C).mean()), "v_bcmode": float((Wa * C).mean()),
        "v_q": float((Wq * C).mean()), "v_trial": float((Wx * C).mean()),
    }
    contrast_keys = ["typical_minus_random", "typical_minus_timeout", "bcmode_minus_typical",
                     "q_minus_typical", "trial_minus_typical", "q_minus_bcmode",
                     "trial_minus_bcmode", "bcmode_minus_random", "bcmode_minus_timeout"]
    for key, (_lab, _fam, b, lo, hi, pval) in zip(contrast_keys, stats, strict=True):
        figures[key], figures[f"{key}_lo"] = float(b), float(lo)
        figures[f"{key}_hi"], figures[f"{key}_p"] = float(hi), float(pval)
    qt_b, qt_lo, qt_hi, qt_p = beta_ci_p(IWB * (ARGQ - ARGX) * C, mids, n_boot=args.bootstrap)
    figures.update({"q_minus_trial": float(qt_b), "q_minus_trial_lo": float(qt_lo),
                    "q_minus_trial_hi": float(qt_hi), "q_minus_trial_p": float(qt_p),
                    "agree_q_bc": float(AGREE.mean()), "agree_trial_bc": float(AGREEX.mean()),
                    **wdiag})
    write_results("stats-recommender-value", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
