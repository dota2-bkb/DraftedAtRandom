"""Build the leak-free play-skill rating and the dataset variant for the headroom test.

Skill is measured as the RESIDUAL = actual composite − StatsModel-predicted composite
(both z-space, balanced weights): it removes the draft+role, leaving how much a player
over/under-performs what their draft predicts (play-skill), so conditioning on it is
non-circular. The rating fed as a feature is leak-free — LOO train-mean for training
picks, train-only mean for val picks, masked for accounts with no prior train game.

Stages (intermediate caches under <root>/.cache/skill-headroom/):
  1. residuals[match] = actual − predicted composite per pick_slot   (StatsModel, GPU)
  2. account per pick_slot                                            (draft_details + match_details)
  3. leak-free rating per (match, pick_slot) + reliability checkpoint
  4. variant dataset at --out-root: matches.jsonl with mmr := rating (rest symlinked)

Run (GPU):
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-skill-headroom/rating.py
Then retrain BC on the variant and run headroom.py (see README).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dota2ad.core import (NUM_PLAYERS, compute_mmr_norm, default_paths, load_matches,
                          load_split, load_stats_rows, load_vocabs)
from dota2ad.core.collate import stats_collate
from dota2ad.core.draft_logic import turn_to_pick_slot
from dota2ad.core.encoding import encode_mmr
from typing import cast

from dota2ad.core.types import Per10, StatsRecord, Turn, UnifiedIdx
from dota2ad.models import load_stats_model
from dota2ad.eval.results import write_results
from dota2ad.eval.stats_specs import STAT_SPECS
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def record_from_row(row, vocabs, mmr_mean, mmr_std):
    """StatsModel input from a StatsRow's completed draft (all 10 players)."""
    loadouts = tuple([vocabs.draft_id_to_index[k] for k in row.loadouts[ps]]
                     for ps in range(10))
    mmr_vals, mmr_mask = encode_mmr(row.mmr, mmr_mean, mmr_std)
    ab = [([vocabs.draft_id_to_index[f"a:{a}"] for a in row.ability_draft_ids[ps]]
           + [0, 0, 0, 0])[:4] for ps in range(10)]
    z = torch.zeros
    return StatsRecord(
        loadouts=cast(Per10[list[UnifiedIdx]], loadouts), mmr_vals=mmr_vals, mmr_mask=mmr_mask,
        ability_indices=torch.tensor(ab, dtype=torch.long),
        scalar_stats=z(10, 22), gold_t=z(10, 3), xp_t=z(10, 3), lh_t=z(10, 3),
        time_mask=z(10, 3, dtype=torch.bool), kill_counts=z(10, 5), death_counts=z(10, 5),
        damage_dealt=z(10, 5), gold_reasons=z(10, 14), xp_reasons=z(10, 6),
        ability_priorities=z(10, 4), spell_damage_dealt=z(10, 4), match_id=row.match_id)


def stage_residuals(paths, cache, device, bs=256):
    """Per match: actual − predicted composite for each pick_slot (z-space, weights)."""
    if cache.exists():
        return {int(k): v for k, v in (json.loads(l) for l in open(cache))}
    matches = load_matches(paths.matches); vocabs = load_vocabs(paths.vocabs)
    split = load_split(paths.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    mmr_mean, mmr_std = compute_mmr_norm(train)
    rows = {r.match_id: r for r in load_stats_rows()}
    model = load_stats_model(paths.stats_ckpt, vocabs, device)

    def _it():
        for m in train:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(rows, _it())
    K = len(STAT_SPECS); w = DEFAULT_BALANCED_WEIGHTS[:K]
    snmK, snsK = snm[:K], sns[:K].clamp(min=1e-6)
    ids = [m.match_id for m in matches if m.match_id in rows]
    print(f"residual pass over {len(ids):,} matches (StatsModel)")
    with open(cache, "w") as out, torch.no_grad():
        for i in range(0, len(ids), bs):
            chunk = ids[i:i + bs]
            recs = [record_from_row(rows[m], vocabs, mmr_mean, mmr_std) for m in chunk]
            outs = model(stats_collate(recs, device=device)); M = len(chunk)
            pred = torch.zeros(M, 10)
            for s in range(10):
                slots = torch.full((M,), s, dtype=torch.long, device=device)
                pz = torch.stack([STAT_SPECS[k].pred_fn(outs, slots).cpu() for k in range(K)], 1)
                pred[:, s] = (pz * w).sum(1)
            for j, m in enumerate(chunk):
                r = rows[m]; resid = []
                for s in range(10):
                    v = torch.tensor([STAT_SPECS[k].real_fn(r, s) for k in range(K)])
                    resid.append(round(float(((v - snmK) / snsK * w).sum()) - float(pred[j, s]), 4))
                out.write(json.dumps([m, resid]) + "\n")
    return {int(k): v for k, v in (json.loads(l) for l in open(cache))}


def stage_join(paths, cache):
    """Per match: account_id per pick_slot (draft_details → pick order; match_details → account)."""
    if cache.exists():
        return {int(k): (t, a) for k, t, a in (json.loads(l) for l in open(cache))}
    files = sorted(paths.parsed.glob("*/match_details.json"))
    print(f"account↔pick_slot join over {len(files):,} matches")
    ok = bad = 0
    with open(cache, "w") as out:
        for md in files:
            dd = md.with_name("draft_details.json")
            try:
                D = json.loads(dd.read_text()); M = json.loads(md.read_text())
                ev = sorted([(e["tick"], e["player_slot"])
                             for e in D["hero_picks"] + D["picks"]])
                ps2pick = {}
                for turn, (_tick, pslot) in enumerate(ev):
                    ps2pick.setdefault(pslot, int(turn_to_pick_slot(Turn(turn))))
                pick2acc = [None] * 10
                for p in M["players"]:
                    pick2acc[ps2pick[p["player_slot"]]] = p["account_id"]
                if any(x is None for x in pick2acc):
                    bad += 1; continue
            except (json.JSONDecodeError, KeyError, OSError):
                bad += 1; continue
            out.write(json.dumps([M["match_id"], M.get("start_time", 0), pick2acc]) + "\n"); ok += 1
    print(f"  joined {ok:,}, skipped {bad}")
    return {int(k): (t, a) for k, t, a in (json.loads(l) for l in open(cache))}


def build_rating(paths, resid, join):
    """Leak-free rating per (match, pick_slot): LOO train-mean for train matches,
    train-only mean for held-out (val/test) matches — held-out outcomes never enter
    the rating covariate."""
    sp = load_split(paths.split)
    held = sp.held_out
    mids = [m for m in resid if m in join]
    # Reliability by residual SOURCE: train-match residuals are in-sample for the
    # StatsModel (it trained on those outcomes), val-match residuals are honest
    # out-of-sample predictions. Comparable SB across sources ⇒ the stable-trait
    # premise is not an artifact of in-sample fit.
    pools = {"all matches": mids,
             "train only (in-sample for the StatsModel)": [m for m in mids if m not in held],
             "val only (out-of-sample)": [m for m in mids if m in sp.val_ids]}
    rng = np.random.default_rng(0)
    sb: dict[str, float] = {}
    pool_key = {"all matches": "all",
                "train only (in-sample for the StatsModel)": "train",
                "val only (out-of-sample)": "val"}
    print("  residual split-half reliability (Spearman-Brown), by residual source:")
    for pname, pm in pools.items():
        by_acct = defaultdict(list)
        for m in pm:
            for ps in range(10):
                by_acct[join[m][1][ps]].append(resid[m][ps])
        cells = []
        for thr in (4, 8):
            a_, b_ = [], []
            for _acct, v in by_acct.items():
                if len(v) < thr:
                    continue
                v = np.array(v); idx = rng.permutation(len(v)); h = len(v) // 2
                a_.append(v[idx[:h]].mean()); b_.append(v[idx[h:2 * h]].mean())
            if len(a_) < 10:
                cells.append(f">={thr}: n={len(a_)} (too few)")
                continue
            r = np.corrcoef(a_, b_)[0, 1]
            sb[f"sb_{pool_key[pname]}_ge{thr}"] = float(2 * r / (1 + r))
            cells.append(f">={thr} games: SB={2 * r / (1 + r):.3f} (n={len(a_):,})")
        print(f"    {pname:44s} " + "   ".join(cells))
    tsum, tcnt = defaultdict(float), defaultdict(int)
    for m in mids:
        if m in held:
            continue
        for ps in range(10):
            a = join[m][1][ps]; tsum[a] += resid[m][ps]; tcnt[a] += 1
    rating = {}
    for m in mids:
        is_tr = m not in held; accts = join[m][1]; rs = resid[m]
        row: list[float | None] = [None] * 10
        for ps in range(10):
            a = accts[ps]
            if is_tr and tcnt[a] > 1:
                row[ps] = round((tsum[a] - rs[ps]) / (tcnt[a] - 1), 4)
            elif not is_tr and tcnt[a] > 0:
                row[ps] = round(tsum[a] / tcnt[a], 4)
        rating[m] = row
    return rating, sb


def build_variant(paths, out_root, rating):
    """matches.jsonl with mmr := rating; everything else symlinked from the source root."""
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ("dataset", "models", "logs"):
        (out_root / sub).mkdir(exist_ok=True)
    n = nr = 0
    with open(paths.matches) as fin, open(out_root / "dataset/matches.jsonl", "w") as fout:
        for line in fin:
            d = json.loads(line); d["mmr"] = rating.get(d["match_id"], [None] * 10)
            nr += d["match_id"] in rating; n += 1; fout.write(json.dumps(d) + "\n")
    for f in ("vocabs.json", "split.json", "excluded_matches.json", "match_stats.jsonl"):
        t = out_root / "dataset" / f
        if not t.exists():
            t.symlink_to((paths.dataset / f).resolve())
    for f in ("match_stats.pt", "stats_dqn.pt"):
        s = paths.models / f; t = out_root / "models" / f
        if s.exists() and not t.exists():
            t.symlink_to(s.resolve())
    print(f"variant: {n} matches ({nr} rated) → {out_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=str, default=None,
                    help="variant root (default: <root>_skill)")
    args = ap.parse_args()
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdir = paths.cache / "skill-headroom"; cdir.mkdir(parents=True, exist_ok=True)
    resid = stage_residuals(paths, cdir / "residuals.jsonl", device)
    join = stage_join(paths, cdir / "pickslot_account.jsonl")
    rating, sb = build_rating(paths, resid, join)
    acc_seen: dict[int, int] = {}
    for _m, (_t, accts) in join.items():
        for acc in accts:
            acc_seen[acc] = acc_seen.get(acc, 0) + 1
    write_results("rating", {**sb, "n_accounts": len(acc_seen),
                             "n_accounts_ge2": sum(1 for v in acc_seen.values() if v >= 2),
                             "n_joined_matches": len(join)})
    out_root = Path(args.out_root) if args.out_root else Path(str(paths.root) + "_skill")
    build_variant(paths, out_root, rating)
    print(f"\nNext: DOTA2AD_ROOT={out_root} pixi run -e cuda python experiments/train-policy/run.py "
          f"--init-from {paths.policy_ckpt} --epochs 25")
    print(f"Then: DOTA2AD_ROOT={out_root} pixi run -e cuda python experiments/stats-skill-headroom/headroom.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
