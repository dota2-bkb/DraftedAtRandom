"""AWR-Trial — the V-optimizing sibling of trial.py.

`trial.py` fits the value *surface* (MSE to the realized stat vector) and asks
"is there **β̂-room** above BC?" — a *ranking* question, for which the conditional
mean (MSE) is the right target. This asks the *policy-value* question instead:
trained on ONLY the random subset, is there **V-room** above BC — can a policy that
DIRECTLY maximizes deployed value beat BC's picks?

Method (no BC mimicry): advantage-weighted regression on the forced-random picks.
The forced action A is drawn from the timeout mechanism P_mech (experiments/random-
mechanism), so V(π) = E[(π(A|s)/P_mech(A|s))·ỹ] is an unbiased off-policy value for
ANY π. We maximize it via AWR: fit π_θ to imitate the random picks *weighted* by
exp(adv/λ), adv = ỹ − b(s) with a learned state baseline b(s); each pick is further
IW'd by (1/m)/P_mech(A) so the improved-over base is UNIFORM (not P_mech) — nothing
anchors to BC. Actor = a
QNetStats score head on BC's frozen encoder (Trial's backbone); critic = a small
state-value head. Two BC-free regularizers: the **temperature** λ (concentration)
and an **ensemble-LCB** at inference — rank by mean − κ·std across M members,
penalizing the net's OWN epistemic uncertainty (data-starved actions), not
BC-implausible ones.

Readouts on held-out TEST random picks (same design as trial.py / run.py):
  • V-room: V(argmax AWR-LCB) vs V(typical human) and vs V(argmax BC)
  • β̂-room: β̂(AWR-LCB, rank) vs β̂(BC) — for comparison to Trial's MSE β̂.
`--shuffle-y` is the negative control (break state-action→outcome; everything must
collapse to ≈ BC / random).

Run:
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/trial_awr.py
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dota2ad.core import (NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches,
                          load_split, load_stats_rows, load_vocabs, mech_propensity)
from dota2ad.core.collate import policy_collate
from dota2ad.models import QNetStats, load_policy
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.bootstrap import cluster_bootstrap_ci
from dota2ad.eval.causal_rank import rank_pct
from dota2ad.training.stats_simulator import compute_stat_norm, scalarize_q
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS
from dota2ad.eval.results import write_results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--members", type=int, default=4, help="ensemble size (for the LCB)")
    ap.add_argument("--lam", type=float, default=1.0, help="AWR temperature on the normalized advantage")
    ap.add_argument("--wmax", type=float, default=20.0, help="AWR weight clip")
    ap.add_argument("--kappa", type=float, default=1.0, help="ensemble-LCB pessimism (mean − κ·std)")
    ap.add_argument("--unfreeze", action="store_true",
                    help="robustness check: also train the BC encoder (low LR, dropout on)")
    ap.add_argument("--enc-lr", type=float, default=1e-4, help="encoder LR when --unfreeze")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-y", action="store_true",
                    help="negative control: permute outcomes (V/β̂ must collapse to ≈ BC)")
    ap.add_argument("--save", type=str, default=None,
                    help="save the ensemble (member actor state_dicts + config) to this path")
    a = ap.parse_args()
    p = default_paths()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(p.matches); vocabs = load_vocabs(p.vocabs)
    vsz = len(vocabs.draft_id_to_index)
    split = load_split(p.split)
    train = [m for m in matches if m.match_id not in split.held_out]
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
    bc_ck = torch.load(p.policy_ckpt, map_location="cpu", weights_only=False)
    D, NH = bc_ck["d"], bc_ck["n_heads"]

    print("extracting train random subset...")
    tt = extract_tuples(train, vocabs, mmr_mean, mmr_std)
    twr = [t for t in tt if t.match_id in rows]
    tys = compute_realized_y_vec(twr, rows, snm, sns)
    data = [(t, float(y[:K].numpy() @ wnp)) for t, y in zip(twr, tys, strict=True)
            if y is not None and len(t.sample.cand_idx) >= 3]                 # composite scalar target
    if a.shuffle_y:
        cs = [c for _, c in data]
        perm = np.random.default_rng(a.seed).permutation(len(cs))
        data = [(t, cs[perm[i]]) for i, (t, _) in enumerate(data)]
        print("NEGATIVE CONTROL: composite outcomes shuffled (state-action→outcome broken)")
    print(f"train random picks n={len(data):,}  (composite; ensemble={a.members}, λ={a.lam}, κ={a.kappa}, "
          f"encoder={'UNFROZEN@' + str(a.enc_lr) if a.unfreeze else 'frozen'})")

    def collate_c(chunk):
        b = policy_collate([t.sample for t, _ in chunk], device=dev)
        fa = torch.tensor([t.action_idx for t, _ in chunk], device=dev)
        c = torch.tensor([cc for _, cc in chunk], dtype=torch.float32, device=dev)
        mfeas = torch.tensor([len(t.sample.cand_idx) for t, _ in chunk],
                             dtype=torch.float32, device=dev)
        iwb = torch.tensor(   # 1/P_mech(A): true IPW base (replaces the uniform m)
            [1.0 / mech_propensity(t.sample.cand_type)[list(t.sample.cand_idx).index(t.action_idx)]
             for t, _ in chunk], dtype=torch.float32, device=dev)
        return b, fa, c, mfeas, iwb

    def train_member(seed_m):
        torch.manual_seed(seed_m)                                   # diversify the reinit'd actor head
        model = QNetStats.warm_start_from_policy(p.policy_ckpt, vsz, K, dev)
        critic = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1)).to(dev)  # state-value baseline
        if a.unfreeze:                                              # robustness: fine-tune the encoder too
            for _, prm in model.named_parameters():
                prm.requires_grad_(True)
            enc = [prm for n, prm in model.named_parameters() if not n.startswith("score_mlp")]
            head = [prm for n, prm in model.named_parameters() if n.startswith("score_mlp")]
            opt = torch.optim.AdamW([{"params": enc, "lr": a.enc_lr},
                                     {"params": head + list(critic.parameters()), "lr": a.lr}],
                                    weight_decay=1e-2)
        else:                                                       # freeze encoder; train actor head only
            for n, prm in model.named_parameters():
                prm.requires_grad_(n.startswith("score_mlp"))
            opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad]
                                    + list(critic.parameters()), lr=a.lr, weight_decay=1e-2)
        fit = [d for d in data if (d[0].match_id + seed_m) % 10 != 0]
        es = [d for d in data if (d[0].match_id + seed_m) % 10 == 0]

        def es_value():
            """Held-out centered IPW value  mean (π(A)/P_mech(A))·(ỹ − b(s))  — maximize."""
            model.eval(); critic.eval()
            tot = 0.0; n = 0
            with torch.no_grad():
                for i in range(0, len(es), 512):
                    b, fa, c, _mfeas, iwb = collate_c(es[i:i + 512])
                    comp = scalarize_q(model(b), w_t)
                    pa = F.softmax(comp, dim=-1)[torch.arange(len(fa), device=dev), fa]
                    bval = critic(model.encode_state(b)).squeeze(-1)
                    tot += float((iwb * pa * (c - bval)).sum()); n += len(fa)
            return tot / n

        best, best_state, bad = -float("inf"), None, 0
        for ep in range(a.epochs):
            model.train() if a.unfreeze else model.eval()   # unfrozen: dropout regularizes the trainable
            critic.train()                                   # encoder; frozen: eval ⇒ clean features for head
            order = np.random.default_rng(ep + seed_m * 100).permutation(len(fit))
            for i in range(0, len(fit), 256):
                chunk = [fit[k] for k in order[i:i + 256]]
                b, fa, c, mfeas, iwb = collate_c(chunk)
                comp = scalarize_q(model(b), w_t)                            # [B,V]; grad → actor head
                logpa = F.log_softmax(comp, dim=-1)[torch.arange(len(fa), device=dev), fa]
                with torch.no_grad():
                    z = model.encode_state(b)
                bval = critic(z).squeeze(-1)                                 # grad → critic
                adv = (c - bval).detach()
                adv_n = (adv - adv.mean()) / (adv.std() + 1e-6)
                wts = torch.clamp(torch.exp(adv_n / a.lam), max=a.wmax)      # AWR weight (no grad)
                wprop = (iwb / mfeas)                                        # (1/m)/P_mech(A): IW the P_mech base to uniform
                actor_loss = -(wprop * wts * logpa).mean()
                critic_loss = ((bval - c) ** 2).mean()
                (actor_loss + critic_loss).backward()
                opt.step(); opt.zero_grad()
            v = es_value()
            print(f"    epoch {ep + 1:2d}  es_value={v:+.4f}{'  *' if v > best else ''}")
            if v > best + 1e-5:
                best, best_state = v, {k: vv.detach().clone() for k, vv in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"    early stop @ {ep + 1}"); break
        return best_state, best

    members = []
    for mi in range(a.members):
        print(f"[member {mi + 1}/{a.members}]")
        st, best_v = train_member(a.seed * 100 + mi)
        members.append(st)
        print(f"  best es_value={best_v:+.4f}")

    if a.save:
        torch.save({"members": members, "d": D, "n_heads": NH, "k_stats": K,
                    "mmr_mean": mmr_mean, "mmr_std": mmr_std, "kappa": a.kappa,
                    "stat_names": torch.load(p.stats_dqn_ckpt, map_location="cpu",
                                             weights_only=False)["stat_names"],
                    "stat_norm_mean": snm[:K].cpu(), "stat_norm_std": sns[:K].cpu()}, a.save)
        print(f"saved AWR-Trial ensemble ({a.members} members) → {a.save} (serving not wired)")

    # --- eval on held-out TEST random picks: ensemble-LCB policy vs BC ---
    ens = []
    for st in members:
        m = QNetStats(vsz, d=D, n_heads=NH, k_stats=K).to(dev)
        m.load_state_dict(st); m.eval(); m.requires_grad_(False)
        ens.append(m)
    policy = load_policy(p.policy_ckpt, vocabs, dev); policy.requires_grad_(False)
    vt = extract_tuples(testm, vocabs, mmr_mean, mmr_std)
    vwr = [t for t in vt if t.match_id in rows]
    vys = compute_realized_y_vec(vwr, rows, snm, sns)
    kept = [(t, y) for t, y in zip(vwr, vys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]

    M_, PIA, PIA_AWR, C, IWB = [], [], [], [], []
    ARG_BC, ARG_MU, ARG_LCB, AGREE = [], [], [], []
    RD_BC, RD_LCB, mids = [], [], []
    with torch.no_grad():
        for i in range(0, len(kept), 256):
            ch = kept[i:i + 256]
            b = policy_collate([t.sample for t, _ in ch], device=dev)
            bc = policy(b).cpu().exp()[:, :vsz]
            comps = [scalarize_q(m(b), w_t).cpu() for m in ens]              # M × [B,V]
            for j, (t, y) in enumerate(ch):
                feas = list(t.sample.cand_idx); fi = feas.index(t.action_idx)
                pr = bc[j, feas].numpy(); pr = pr / pr.sum()
                cm = np.stack([comps[e][j, feas].numpy() for e in range(len(ens))])   # [M,F]
                cm = cm - cm.mean(axis=1, keepdims=True)                     # center each member within-state
                mu = cm.mean(0); lcb = mu - a.kappa * cm.std(0)
                ps = np.exp(mu - mu.max()); ps = ps / ps.sum()              # ensemble-mean soft policy π_AWR
                ab, amu, alcb = int(pr.argmax()), int(mu.argmax()), int(lcb.argmax())
                M_.append(len(feas)); PIA.append(pr[fi]); PIA_AWR.append(ps[fi]); C.append(float(y[:K].numpy() @ wnp))
                IWB.append(1.0 / mech_propensity(t.sample.cand_type)[fi])   # 1/P_mech(A)
                ARG_BC.append(1.0 if fi == ab else 0.0)
                ARG_MU.append(1.0 if fi == amu else 0.0)
                ARG_LCB.append(1.0 if fi == alcb else 0.0)
                AGREE.append(1.0 if alcb == ab else 0.0)
                RD_BC.append((rank_pct(pr) - 0.5)[fi])
                RD_LCB.append((rank_pct(lcb) - 0.5)[fi])
                mids.append(t.match_id)
    M_, PIA, PIA_AWR, C, IWB, ARG_BC, ARG_MU, ARG_LCB, AGREE, RD_BC, RD_LCB = map(
        np.array, (M_, PIA, PIA_AWR, C, IWB, ARG_BC, ARG_MU, ARG_LCB, AGREE, RD_BC, RD_LCB))
    WPROP = IWB / M_    # (1/m)/P_mech(A): uniform-target (RANDOM / β̂) weight
    sd = C.std()

    def ci(contrib):
        lo, m, hi = cluster_bootstrap_ci(lambda ix: float(contrib[ix].mean()), mids, n_boot=a.bootstrap)
        return m, lo, hi, ("*" if (lo > 0 or hi < 0) else " ")

    print(f"\nheld-out test random picks n={len(C)} ({len(set(mids))} matches); "
          f"outcome=z-composite (sd={sd:.2f}); B={a.bootstrap}")
    figs: dict[str, float | int] = {"n_picks": len(C), "n_matches": len(set(mids)),
                                    "sd_composite": float(sd)}
    print("\nβ̂-room (rank transform):")
    for lab, dv in [("BC", RD_BC), ("AWR-LCB", RD_LCB)]:
        m, lo, hi, s = ci(WPROP * dv * C)
        key = "beta_bc" if lab == "BC" else "beta_awr_lcb"
        figs[key], figs[f"{key}_lo"], figs[f"{key}_hi"] = m, lo, hi
        print(f"  β̂({lab:8}) = {m:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")
    m, lo, hi, s = ci(WPROP * (RD_LCB - RD_BC) * C)
    figs["dbeta_vs_bc"], figs["dbeta_vs_bc_lo"], figs["dbeta_vs_bc_hi"] = m, lo, hi
    print(f"  Δβ̂(AWR-LCB − BC) = {m:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")

    Wt, Wbc, Wmu, Wlcb = PIA * IWB, ARG_BC * IWB, ARG_MU * IWB, ARG_LCB * IWB
    print("\nV-room (exact IPW, weight π(A)/P_mech(A); the thing we built this for):")
    print(f"  RANDOM legal pick             V={(WPROP * C).mean():+.3f}")
    print(f"  TYPICAL human (draw ~ BC)     V={(Wt * C).mean():+.3f}")
    print(f"  argmax BC (mode)              V={(Wbc * C).mean():+.3f}")
    print(f"  argmax AWR (ensemble mean)    V={(Wmu * C).mean():+.3f}")
    print(f"  argmax AWR-LCB (κ={a.kappa})        V={(Wlcb * C).mean():+.3f}")
    print(f"  soft AWR policy (sample ~ π)   V={(PIA_AWR * IWB * C).mean():+.3f}")
    print(f"  (argmax AWR-LCB coincides with argmax BC on {AGREE.mean():.1%} of decisions)")
    ckeys = ["soft_minus_typical", "lcb_minus_typical", "lcb_minus_bcmode", "mu_minus_typical"]
    for ckey, (lab, contrib) in zip(ckeys, [
            ("soft AWR policy − TYPICAL (AWR's OWN objective)", IWB * (PIA_AWR - PIA) * C),
            ("AWR-LCB argmax − TYPICAL (V-room vs individual)", IWB * (ARG_LCB - PIA) * C),
            ("AWR-LCB argmax − argmax BC (V-room vs BC mode)", IWB * (ARG_LCB - ARG_BC) * C),
            ("AWR-mean argmax − TYPICAL (no pessimism)", IWB * (ARG_MU - PIA) * C)], strict=True):
        m, lo, hi, s = ci(contrib)
        figs[ckey], figs[f"{ckey}_lo"], figs[f"{ckey}_hi"] = m, lo, hi
        print(f"  Δ {lab:48} {m:+.3f} [{lo:+.3f},{hi:+.3f}]{s}  ({m / sd:+.1%} SD)")
    figs["agree_lcb_bc"] = float(AGREE.mean())
    write_results("trial-awr", figs)
    print("\nV-room = does the V-optimized random-subset policy beat BC? "
          "Δ(vs argmax BC)>0 & CI∌0 ⇒ yes; --shuffle-y must collapse it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
