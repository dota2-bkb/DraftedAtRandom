"""Headroom test: does conditioning the retrained BC on high PLAY-SKILL (the leak-free
residual rating, now in the mmr slot of the variant dataset) make it a more
causally-aligned ranker? β̂(BC | focal skill = z) on held-out random picks; the paired
high−low is the readout. >0 with CI excluding 0 ⇒ skilled players draft in a more
causally-effective way ⇒ offline headroom. Spans 0 ⇒ none on this axis.

A sanity line first reports how much BC's pick distribution actually MOVES under the
skill override — a flat β̂ with a live feature is a real null, not a dead feature.

Run (on the variant root, after retraining BC there with --init-from):
  DOTA2AD_ROOT=work_skill pixi run -e cuda python experiments/stats-skill-headroom/headroom.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dota2ad.core import (NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches,
                          load_split, load_stats_rows, load_vocabs)
from dota2ad.core.collate import policy_collate
from dota2ad.models import QNetStats, load_policy
from dota2ad.eval.tuples import extract_tuples
from dota2ad.eval.causal_rank import beta_ci, compute_deviations, _override_mmr
from dota2ad.eval.results import write_results
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matches = load_matches(paths.matches); vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index); split = load_split(paths.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    testm = [m for m in matches if m.match_id in split.test_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train)
    rows = {r.match_id: r for r in load_stats_rows()}

    def _it():
        for m in train:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(rows, _it())
    policy = load_policy(paths.policy_ckpt, vocabs, device); policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.stats_dqn_ckpt, vocab_size, device); q.eval()  # K only; Q unused
    tuples = extract_tuples(testm, vocabs, mmr_mean, mmr_std)
    print(f"skill-rating scale (z-norm): mean={mmr_mean:.3f} std={mmr_std:.3f}")

    # SANITY: does BC's pick distribution respond to the skill override at all?
    samp = tuples[:3000]

    def bc_probs(z):
        outs = []
        for i in range(0, len(samp), 256):
            ss = [_override_mmr(t.sample, int(t.focal_slot), z) for t in samp[i:i + 256]]
            with torch.no_grad():
                outs.append(policy(policy_collate(ss, device=device)).cpu().exp()[:, :vocab_size])
        return torch.cat(outs)
    ph, pl = bc_probs(1.0), bc_probs(-1.0)
    tv = []
    for i, t in enumerate(samp):
        f = list(t.sample.cand_idx)
        a = ph[i, f]; a = a / a.sum(); b = pl[i, f]; b = b / b.sum()
        tv.append(0.5 * float((a - b).abs().sum()))
    print(f"SANITY mean TV(BC@+1z, BC@-1z) over feasible = {np.mean(tv):.4f}  max={np.max(tv):.3f}")
    print("  (≈0 ⇒ BC ignores the skill feature; a flat β̂ below would then be 'feature dead')")

    d0 = compute_deviations(tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size,
                            rows, snm, sns, device, mmr_override=None)
    comp, mids, n, w = d0.comp, d0.mids, d0.n, d0.w_prop
    store: dict[float | None, np.ndarray] = {None: d0.bc_rank}
    for z in (-1.5, -1.0, 0.0, 1.0, 1.5):
        d = compute_deviations(tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size,
                               rows, snm, sns, device, mmr_override=z)
        store[z] = d.bc_rank
    figures: dict[str, float | int] = {
        "mean_tv": float(np.mean(tv)), "max_tv": float(np.max(tv)),
        "n_picks": int(n), "n_matches": len(set(mids)),
    }
    level_key = {None: "own", -1.5: "m1p5z", -1.0: "m1z", 0.0: "z0", 1.0: "p1z", 1.5: "p1p5z"}
    print(f"\nHeld-out random picks n={n} ({len(set(mids))} matches); B={args.bootstrap}")
    for z in (None, -1.5, -1.0, 0.0, 1.0, 1.5):
        b, lo, hi = beta_ci(w * store[z] * comp, mids, args.bootstrap)
        k = f"beta_{level_key[z]}"
        figures[k], figures[f"{k}_lo"], figures[f"{k}_hi"] = float(b), float(lo), float(hi)
        s = "*" if (lo > 0 or hi < 0) else " "
        lab = "own" if z is None else f"{z:+.1f}z"
        print(f"  skill={lab:8} β̂(BC)={b:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")
    print("\nPaired (same outcomes → Δβ̂; * = CI excludes 0):")
    for key, lab, a, c in [("dbeta_high_low", "BC@+1z − BC@-1z  (HEADROOM?)", store[1.0], store[-1.0]),
                           ("dbeta_high_low_wide", "BC@+1.5z − BC@-1.5z (wider)", store[1.5], store[-1.5]),
                           ("dbeta_p1z_vs_own", "BC@+1z − BC@own", store[1.0], store[None])]:
        b, lo, hi = beta_ci(w * (a - c) * comp, mids, args.bootstrap)
        figures[key], figures[f"{key}_lo"], figures[f"{key}_hi"] = float(b), float(lo), float(hi)
        s = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {lab:32} Δβ̂={b:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")
    causal_p = paths.root / "results" / "stats-causal-rank.json"
    if causal_p.exists():
        causal = json.loads(causal_p.read_text())
        figures["bound_pct_of_bc"] = (
            max(abs(figures["dbeta_high_low_lo"]), abs(figures["dbeta_high_low_hi"]))
            / causal["beta_bc"] * 100.0)
    write_results("stats-skill-headroom", figures)
    print("\n>0 & CI excludes 0 ⇒ high-play-skill picks rank forced actions more causally-aligned")
    print("⇒ imitating skilled players beats consensus = OFFLINE HEADROOM. spans 0 ⇒ none on this axis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
