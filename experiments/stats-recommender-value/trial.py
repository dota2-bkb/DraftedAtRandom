"""Is there a ranker fittable from the causal data that beats BC? The strongest offline
probe short of an A/B — a *state-aware* value ranker trained on ONLY the random subset.

The forced action is randomized, so the ~37k train forced picks are a clean supervised
set: regressing the realized outcome on (state, action) recovers the UNCONFOUNDED causal
value. We fit it state-aware but sample-efficiently: BC's encoder (trained on picks, not
outcomes — no leakage), frozen, provides the state representation; only a value head is
learned, on the train-split random picks, early-stopped on the val split's forced picks
(selection burns val only). Then β̂ of this ranker vs BC on held-out TEST forced picks
(the `stats-causal-rank` test).

`--shuffle-y` is the negative control: permute the fit outcomes so the value head
learns nothing — β̂ must collapse to ≈ 0 for the signal to be real, not artifact.
`--seed` varies the head init + fit shuffling for robustness. Numbers of record:
the run log / results manifest; the report carries the curated reading.

Run (repeat over --seed 0..3; add --shuffle-y for the control):
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/trial.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dota2ad.core import (NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches,
                          load_split, load_stats_rows, load_vocabs, iw_to_uniform)
from dota2ad.core.collate import policy_collate
from dota2ad.models import QNetStats, load_policy
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.bootstrap import cluster_bootstrap_ci
from dota2ad.eval.results import write_results
from dota2ad.eval.causal_rank import rank_pct
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-y", action="store_true",
                    help="negative control: permute outcomes (β̂ must collapse to ~0)")
    ap.add_argument("--save", type=str, default=None,
                    help="save the trained net as a serving-compatible QNetStats checkpoint (Trial)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)          # controls the reinit'd value head
    p = default_paths()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(p.matches); vocabs = load_vocabs(p.vocabs)
    vsz = len(vocabs.draft_id_to_index)
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

    print("extracting train random subset...")
    tt = extract_tuples(train, vocabs, mmr_mean, mmr_std)
    twr = [t for t in tt if t.match_id in rows]
    tys = compute_realized_y_vec(twr, rows, snm, sns)
    fit = [(t, y[:K]) for t, y in zip(twr, tys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]
    if a.shuffle_y:
        yss = [y for _, y in fit]
        perm = np.random.default_rng(a.seed).permutation(len(yss))
        fit = [(t, yss[perm[i]]) for i, (t, _) in enumerate(fit)]
        print("NEGATIVE CONTROL: fit outcomes shuffled (state-action→outcome link broken)")
    et = extract_tuples(valm, vocabs, mmr_mean, mmr_std)
    ewr = [t for t in et if t.match_id in rows]
    eys = compute_realized_y_vec(ewr, rows, snm, sns)
    es = [(t, y[:K]) for t, y in zip(ewr, eys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]
    print(f"fit={len(fit):,} (train forced picks)  earlystop={len(es):,} (val forced picks)")

    model = QNetStats.warm_start_from_policy(p.policy_ckpt, vsz, K, dev)
    for n, prm in model.named_parameters():
        prm.requires_grad_(n.startswith("score_mlp"))       # freeze BC encoder; learn value head only
    opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad],
                            lr=a.lr, weight_decay=1e-2)

    def fwd(chunk):
        b = policy_collate([t.sample for t, _ in chunk], device=dev)
        q = model(b)                                          # [B, V, K]
        fa = torch.tensor([t.action_idx for t, _ in chunk], device=dev)
        qf = q[torch.arange(len(chunk), device=dev), fa, :]   # [B, K] — forced action's Q vector
        yv = torch.stack([y for _, y in chunk]).to(dev)
        return qf, yv

    def es_loss():
        model.eval(); tot = n = 0.0
        with torch.no_grad():
            for i in range(0, len(es), 512):
                qf, yv = fwd(es[i:i + 512]); tot += float(((qf - yv) ** 2).sum()); n += qf.numel()
        return tot / n

    best, best_state, bad = float("inf"), None, 0
    for ep in range(a.epochs):
        model.eval()          # frozen encoder ⇒ eval mode: no dropout on the features the head fits
        order = np.random.default_rng(ep + a.seed * 100).permutation(len(fit))
        for i in range(0, len(fit), 256):
            chunk = [fit[k] for k in order[i:i + 256]]
            qf, yv = fwd(chunk); loss = ((qf - yv) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        e = es_loss()
        print(f"  epoch {ep + 1:2d}  es_mse={e:.4f}{'  *' if e < best else ''}")
        if e < best - 1e-5:
            best, best_state, bad = e, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop @ {ep + 1}"); break
    assert best_state is not None
    model.load_state_dict(best_state)

    if a.save:
        bc_ck = torch.load(p.policy_ckpt, map_location="cpu", weights_only=False)
        q_ck = torch.load(p.stats_dqn_ckpt, map_location="cpu", weights_only=False)
        torch.save({
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "d": bc_ck["d"], "n_heads": bc_ck["n_heads"], "k_stats": K,
            "mmr_mean": mmr_mean, "mmr_std": mmr_std,
            "stat_names": q_ck["stat_names"],
            "stat_norm_mean": snm[:K].cpu(), "stat_norm_std": sns[:K].cpu(),
            "bc_mask_frac": 0.0,
        }, a.save)
        print(f"saved Trial checkpoint → {a.save}")

    # --- β̂ on held-out TEST forced picks: this ranker vs BC ---
    policy = load_policy(p.policy_ckpt, vocabs, dev); policy.requires_grad_(False)
    vt = extract_tuples(testm, vocabs, mmr_mean, mmr_std)
    vwr = [t for t in vt if t.match_id in rows]
    vys = compute_realized_y_vec(vwr, rows, snm, sns)
    kept = [(t, y) for t, y in zip(vwr, vys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]
    DBC, DQ, C, WPROP, mids = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(kept), 256):
            ch = kept[i:i + 256]
            b = policy_collate([t.sample for t, _ in ch], device=dev)
            bc = policy(b).cpu().exp()[:, :vsz]; q = model(b).cpu()
            for j, (t, y) in enumerate(ch):
                feas = list(t.sample.cand_idx); fi = feas.index(t.action_idx)
                DBC.append((rank_pct(bc[j, feas].numpy()) - 0.5)[fi])
                DQ.append((rank_pct(q[j, feas, :].numpy() @ wnp) - 0.5)[fi])
                WPROP.append(iw_to_uniform(t.sample.cand_type, fi))   # (1/m)/P_mech(A) → uniform estimand
                C.append(float(y[:K].numpy() @ wnp)); mids.append(t.match_id)
    DBC, DQ, C, WPROP = map(np.array, (DBC, DQ, C, WPROP))

    def bci(c):
        lo, m, hi = cluster_bootstrap_ci(lambda i: float(c[i].mean()), mids, n_boot=a.bootstrap)
        return m, lo, hi
    print(f"\nheld-out test forced picks n={len(C)} ({len(set(mids))} matches); B={a.bootstrap}")
    bb, lb, hb = bci(WPROP * DBC * C); print(f"  β̂(BC)                      = {bb:+.4f} [{lb:+.4f},{hb:+.4f}]")
    bq, lq, hq = bci(WPROP * DQ * C); sq = "*" if (lq > 0 or hq < 0) else " "
    print(f"  β̂(random-subset value net) = {bq:+.4f} [{lq:+.4f},{hq:+.4f}]{sq}")
    bd, ld, hd = bci(WPROP * (DQ - DBC) * C); sd = "*" if (ld > 0 or hd < 0) else " "
    print(f"  Δ(value net − BC)          = {bd:+.4f} [{ld:+.4f},{hd:+.4f}]{sd}")
    print("\nΔ>0 & CI excludes 0 ⇒ fittable room; --shuffle-y control must give β̂≈0.")
    if not a.shuffle_y:
        # Cross-seed aggregate: every seed run upserts its Δβ̂; mean/min/max
        # recompute over the seeds recorded so far.
        agg_path = p.root / "results" / "trial-seeds.json"
        agg = json.loads(agg_path.read_text()) if agg_path.exists() else {}
        agg[f"dbeta_s{a.seed}"] = float(bd)
        seeds_done = [v for k, v in agg.items() if k.startswith("dbeta_s") and k[7:].isdigit()]
        agg["dbeta_seed_mean"] = float(np.mean(seeds_done))
        agg["dbeta_seed_min"] = float(min(seeds_done))
        agg["dbeta_seed_max"] = float(max(seeds_done))
        agg["n_seeds_done"] = len(seeds_done)
        write_results("trial-seeds", agg)
    if a.seed == 0 and a.shuffle_y:
        # The negative control's manifest (canonical seed-0 shuffle run).
        write_results("trial-shuffle", {
            "beta_shuffled": float(bq), "beta_shuffled_lo": float(lq),
            "beta_shuffled_hi": float(hq), "dbeta_vs_bc": float(bd),
            "n_picks": len(C),
        })
    if a.seed == 0 and not a.shuffle_y:
        # One manifest from the canonical seed-0 run; other seeds are robustness reruns.
        write_results("trial", {
            "dbeta_vs_bc": float(bd), "dbeta_vs_bc_lo": float(ld), "dbeta_vs_bc_hi": float(hd),
            "beta_trial": float(bq), "beta_trial_lo": float(lq), "beta_trial_hi": float(hq),
            "beta_bc": float(bb), "beta_bc_lo": float(lb), "beta_bc_hi": float(hb),
            "seed": int(a.seed), "n_picks": len(C), "n_matches": len(set(mids)),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
