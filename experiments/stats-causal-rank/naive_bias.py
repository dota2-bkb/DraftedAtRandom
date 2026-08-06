"""naive_bias: the confounded number the design exists to avoid.

Score the rankers' rank-δ on DELIBERATE picks (one per test match — chosen picks,
no randomization, no propensity) against the picker's realized composite. Skilled
players both pick in-model and play well, so this association mixes any causal
effect with skill/context confounding — §2's argument, here measured. The gap
against the forced-pick causal β̂ is the bias the natural experiment removes.

Run (cuda — scores BC and Q):
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-causal-rank/naive_bias.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, PickSlot, Turn, compute_mmr_norm, default_paths, encode_policy_sample,
    load_matches, load_split, load_stats_rows, load_vocabs,
)
from dota2ad.core.collate import policy_collate
from dota2ad.core.draft_logic import idx, replay_complete, replay_to_turn
from dota2ad.core.encoding import encode_loadout, encode_mmr
from dota2ad.eval.causal_rank import beta_ci, rank_pct
from dota2ad.eval.stats_eval import compute_realized_y_vec
from dota2ad.eval.tuples import RandomPickTuple
from dota2ad.models import QNetStats, load_policy
from dota2ad.training.stats_simulator import compute_stat_norm, scalarize_q
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS
from dota2ad.eval.results import write_results


def deliberate_tuples(matches, vocabs, mmr_mean, mmr_std, seed=0):
    """One uniformly chosen DELIBERATE pick per match, packaged like a forced-pick
    tuple (same encoding path as extract_tuples, minus the is_random filter)."""
    rng = np.random.default_rng(seed)
    out = []
    for match in matches:
        cand = [t for t, e in enumerate(match.history) if not e.is_random]
        if not cand:
            continue
        t = int(rng.choice(cand))
        event = match.history[t]
        final_pp = replay_complete(match)
        final_loadouts = [encode_loadout(final_pp[PickSlot(ps)], vocabs)
                         for ps in range(NUM_PLAYERS)]
        mmr_vals, mmr_mask = encode_mmr(match.mmr, mmr_mean, mmr_std)
        row = replay_to_turn(match, Turn(t))
        sample = encode_policy_sample(row, vocabs, history=match.history[:t],
                                      mmr_mean=mmr_mean, mmr_std=mmr_std)
        if event.hero_id is not None:
            a_idx = idx(vocabs, event.hero_id, "h")
        else:
            a_idx = idx(vocabs, event.draft_ability_id, "a")
        focal_slot = row.pick_slot
        fir = (focal_slot % 2 == 0)
        won = (match.radiant_win and fir) or (not match.radiant_win and not fir)
        out.append(RandomPickTuple(
            sample=sample, action_idx=a_idx, n_feasible=len(sample.cand_idx),
            turn=t, focal_slot=focal_slot,
            picker_disconnected=event.picker_disconnected,
            focal_team_won=1.0 if won else 0.0,
            final_loadouts_other=final_loadouts,
            mmr_vals=mmr_vals, mmr_mask=mmr_mask, match_id=match.match_id))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    test_matches = [m for m in matches if m.match_id in split.test_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    rows = {r.match_id: r for r in load_stats_rows()}

    def _iter():
        for m in train_matches:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(rows, _iter())

    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.models / "stats_dqn.pt", vocab_size, device)
    q.eval()

    tuples = [t for t in deliberate_tuples(test_matches, vocabs, mmr_mean, mmr_std)
              if t.match_id in rows]
    ys = compute_realized_y_vec(tuples, rows, snm, sns)
    K = len(DEFAULT_BALANCED_WEIGHTS)
    wv = DEFAULT_BALANCED_WEIGHTS[:K]
    w_np = np.asarray([float(x) for x in wv])
    keep = [(t, y) for t, y in zip(tuples, ys, strict=True) if y is not None and len(t.sample.cand_idx) >= 3]
    tup = [t for t, _ in keep]
    comp = np.array([float(y.numpy()[:K] @ w_np) for _, y in keep])
    mids = [t.match_id for t in tup]
    n = len(tup)

    bc_r = np.zeros(n); qc_r = np.zeros(n)
    with torch.no_grad():
        for i0 in range(0, n, 256):
            chunk = tup[i0:i0 + 256]
            b = policy_collate([t.sample for t in chunk], device=device)
            qc = scalarize_q(q(b).cpu(), wv)
            bc = policy(b).cpu().exp()[:, :vocab_size]
            for j, t in enumerate(chunk):
                feas = list(t.sample.cand_idx)
                ridx = feas.index(t.action_idx)
                fi = torch.tensor(feas, dtype=torch.long)
                qc_r[i0 + j] = (rank_pct(qc[j].index_select(0, fi).numpy()) - 0.5)[ridx]
                bc_r[i0 + j] = (rank_pct(bc[j].index_select(0, fi).numpy()) - 0.5)[ridx]

    print(f"Deliberate picks: one per test match, n={n} ({len(set(mids))} matches).")
    print("NAIVE association (no randomization, no propensity — mean δ·ỹ on chosen picks):")
    figs: dict[str, float | int] = {"n_picks": n, "n_matches": len(set(mids))}
    for name, delta in (("BC", bc_r), ("Q", qc_r)):
        b, lo, hi = beta_ci(delta * comp, mids, args.bootstrap)
        figs[f"naive_{name.lower()}"] = float(b)
        figs[f"naive_{name.lower()}_lo"] = float(lo)
        figs[f"naive_{name.lower()}_hi"] = float(hi)
        s = "*" if lo > 0 or hi < 0 else " "
        print(f"  {name:2s} naive = {b:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")
    causal_p = paths.root / "results" / "stats-causal-rank.json"
    if causal_p.exists():
        causal = json.loads(causal_p.read_text())
        figs["inflation_bc_pct"] = (figs["naive_bc"] / causal["beta_bc"] - 1.0) * 100.0
        figs["inflation_q_pct"] = (figs["naive_q"] / causal["beta_q"] - 1.0) * 100.0
    write_results("naive-bias", figs)
    print("\nReading: this number mixes causal effect with skill/context confounding — the")
    print("association §2 warns against. Compare the causal forced-pick β̂ in")
    print("stats-causal-rank.log: the difference is the bias the design removes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
