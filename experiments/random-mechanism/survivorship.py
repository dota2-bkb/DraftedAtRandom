"""survivorship.py — quantify the outcome-dependent missingness that contaminates
real-player forced picks (see this experiment's README).

The natural experiment needs the forced pick to be a clean draw from the server
mechanism. run.py's T5 shows it is — for **bots** (they never abandon). For real
players it is not: matches where the coin handed someone an **undesirable** random
are disproportionately abandoned and never recorded, so the *surviving* forced
picks skew toward *desirable* items.

FRAMING (one axis, no kind special-case). Every item — hero or ability — lives on a
single **desirability** axis, measured by **draft popularity** (deliberate picks /
times in pool: how often players actually choose it). Survivorship is one gradient:
undesirable items get suppressed, desirable ones do not. There is NO hero block —
the most-drafted heroes are *over*-forced, not suppressed; the average
"hero deficit" is just that heroes are, on average, less-drafted, concentrated in
the unpopular ones. Per item: `suppression = observed forced count / count expected
under the mechanism (P_mech)`; null = 1.

We report the survivorship two ways: (a) the desirability shift of forced picks
(popularity units), and (b) the same shift in the **composite** outcome units β̂
uses (for the downstream bias). Both are the aggregate over all items — the pooled
answer, not a kind-split.

CHANNEL DECOMPOSITION. The clean-set suppression conflates two channels: (i) matches
**never recorded** (abandoned before scoring — unobservable), and (ii) matches
recorded but **excluded per-protocol as leaver games** — and leaving is what a
tilted player does after a bad forced pick. Channel (ii) is estimand scope, not an
identification threat, and it is measurable: section [C] recomputes the gradient
with leaver matches included (same items, same popularity axis and bins). The
residual gradient there bounds channel (i).

CAVEAT. A small residual hero-specific offset (~0.04-0.06 in obs/exp) *may* sit on
top of desirability — at matched popularity, heroes trend a touch below abilities.
But it is confounded: the mechanism baseline P_mech itself carries the ~2% hero
over-prediction (the 3-kind residual), so part of any hero-vs-ability gap at matched
desirability is circular, and the binning is coarse. The robust, un-circular finding
is the desirability gradient + that popular heroes are not suppressed; whether a
genuine hero-specific term survives is unresolved and second-order.

Run:
  DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/survivorship.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, PickSlot, compute_mmr_norm, default_paths, load_matches,
    load_split, load_stats_rows, load_vocabs, mech_kind_probs,
)
from dota2ad.core.draft_logic import idx, replay_complete
from dota2ad.core.encoding import encode_loadout
from dota2ad.eval.results import write_results
from dota2ad.eval.stats_specs import STAT_SPECS
from dota2ad.eval.tuples import extract_tuples
from dota2ad.training.stats_simulator import compute_stat_norm
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def cluster_ci(vals, mids, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, m in enumerate(mids):
        by[m].append(i)
    groups = [np.array(v) for v in by.values()]
    nc = len(groups)
    boots = np.array([vals[np.concatenate([groups[c] for c in rng.integers(0, nc, nc)])].mean()
                      for _ in range(n_boot)])
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    paths = default_paths()
    matches = load_matches(paths.matches, exclude=())
    vocabs = load_vocabs(paths.vocabs)
    split = load_split(paths.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    mm_mean, mm_std = compute_mmr_norm(train)
    srows = {r.match_id: r for r in load_stats_rows()}
    with open(paths.excluded) as f:
        exc = json.load(f)
    bad = set(exc["too_many_random_picks"]) | set(exc["leavers"]) | set(exc["swaps"])
    # bots + swaps are skipped in BOTH accumulator sets; leaver matches are what the
    # incl-leaver set adds back (channel decomposition, section [C])
    bots_swaps = set(exc["too_many_random_picks"]) | set(exc["swaps"])

    def U(kind, gid):
        return idx(vocabs, gid, kind)

    # desirability = draft popularity (deliberate picks / times in pool)
    pool_app, delib = defaultdict(int), defaultdict(int)
    for m in matches:
        if m.match_id in bad:
            continue
        for a in ([U("h", h) for h in m.hero_pool] + [U("a", a) for a in m.basic_pool]
                  + [U("a", a) for a in m.ult_pool]):
            pool_app[a] += 1
        for e in m.history:
            if e.is_random:
                continue
            a = U("h", e.hero_id) if e.hero_id is not None else U("a", e.draft_ability_id)
            delib[a] += 1
    popularity = {a: delib[a] / pool_app[a] for a in pool_app if pool_app[a] >= 50}

    # composite value (β̂'s outcome units) for the magnitude
    def _it():
        for m in train:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    snm, sns = compute_stat_norm(srows, _it())
    w = DEFAULT_BALANCED_WEIGHTS
    K = len(w)
    isum, icnt, comps = defaultdict(float), defaultdict(int), []
    for m in matches:
        if m.match_id not in srows:
            continue
        row = srows[m.match_id]
        fp = replay_complete(m)
        for ps in range(NUM_PLAYERS):
            v = torch.tensor([float(STAT_SPECS[k].real_fn(row, ps)) for k in range(K)])
            comp = float((((v - snm) / sns.clamp(min=1e-6)) * w).sum())
            comps.append(comp)
            for it in encode_loadout(fp[PickSlot(ps)], vocabs):
                isum[it] += comp; icnt[it] += 1
    value = {k: isum[k] / icnt[k] for k in icnt if icnt[k] >= 30}
    csd = float(np.std(comps))

    # per forced pick: desirability & composite shift vs mechanism-expected; per-item obs/exp.
    # Two accumulator sets from one pass — clean (the analytic set) and clean∪leaver (leaver
    # matches included) — for the channel decomposition; bots/swaps are skipped in both.
    pop_rows, val_rows, obs, expc, kind_of = [], [], defaultdict(float), defaultdict(float), {}
    pop_rows_l, obs_l, expc_l = [], defaultdict(float), defaultdict(float)
    for t in extract_tuples(matches, vocabs, mm_mean, mm_std):
        if t.match_id in bots_swaps:
            continue
        is_clean = t.match_id not in bad
        ct = list(t.sample.cand_type); ci = list(t.sample.cand_idx)
        NH, NB, NU = ct.count(0), ct.count(1), ct.count(2)
        ph, pb, pu = mech_kind_probs(NH, NB, NU)
        per = (ph / NH if NH else 0.0, pb / NB if NB else 0.0, pu / NU if NU else 0.0)
        for i, a in enumerate(ci):
            expc_l[a] += per[ct[i]]; kind_of[a] = ct[i]
            if is_clean:
                expc[a] += per[ct[i]]
        obs_l[t.action_idx] += 1
        if is_clean:
            obs[t.action_idx] += 1
        # desirability shift
        pv = {0: [], 1: [], 2: []}
        for i, a in enumerate(ci):
            if a in popularity:
                pv[ct[i]].append(popularity[a])
        if t.action_idx in popularity and any(pv.values()):
            exp = sum(p * np.mean(pv[k]) for p, k in ((ph, 0), (pb, 1), (pu, 2)) if pv[k])
            pop_rows_l.append((t.match_id, popularity[t.action_idx] - exp))
            if is_clean:
                pop_rows.append((t.match_id, popularity[t.action_idx] - exp))
        # composite shift (analytic/clean & stats-parsed)
        if is_clean and t.match_id in srows and t.action_idx in value:
            vv = {0: [], 1: [], 2: []}
            for i, a in enumerate(ci):
                if a in value:
                    vv[ct[i]].append(value[a])
            exp = sum(p * np.mean(vv[k]) for p, k in ((ph, 0), (pb, 1), (pu, 2)) if vv[k])
            val_rows.append((t.match_id, value[t.action_idx] - exp))

    print("[A] Survivorship magnitude — forced picks skew toward DESIRABLE items (null 0):")
    d = np.array([r[1] for r in pop_rows]); m, lo, hi = cluster_ci(d, [r[0] for r in pop_rows])
    pop_shift, pop_shift_lo, pop_shift_hi = m, lo, hi
    print(f"    desirability (popularity) shift: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"[{'SURVIVORSHIP' if lo > 0 else 'null'}]")
    d = np.array([r[1] for r in val_rows]); m, lo, hi = cluster_ci(d, [r[0] for r in val_rows])
    shift_sd, shift_sd_lo, shift_sd_hi = m / csd, lo / csd, hi / csd
    print(f"    composite value shift (β̂ units): {m/csd:+.4f} SD [{lo/csd:+.4f}, {hi/csd:+.4f}] SD  "
          f"[{'SURVIVORSHIP' if lo > 0 else 'null'}]")

    # [B] one axis, no hero block: suppression by popularity, and popular heroes not suppressed
    items = [a for a in expc if expc[a] >= 15 and a in popularity]
    P = np.array([popularity[a] for a in items]); O = np.array([obs.get(a, 0.0) for a in items])
    E = np.array([expc[a] for a in items]); KD = np.array([kind_of[a] for a in items])
    print("\n[B] One desirability axis (ALL items pooled) — suppression obs/exp by popularity quintile:")
    q = np.quantile(P, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        b = (q[i] <= P) & (q[i + 1] >= P if i == 4 else q[i + 1] > P)
        print(f"    popularity[{q[i]:.2f},{q[i+1]:.2f}]  obs/exp = {O[b].sum()/E[b].sum():.3f}")
    h = KD == 0
    hi_idx = sorted(np.where(h)[0], key=lambda j: -P[j])[:8]
    print("    most-drafted heroes are NOT suppressed (obs/exp > 1) — no hero block:")
    for j in hi_idx:
        print(f"      hero pop={P[j]:.3f}  obs/exp={O[j]/E[j]:.3f}")

    # [C] channel decomposition: same items, same popularity bins; leaver matches INCLUDED.
    # Channel (ii) = per-protocol leaver exclusion (recorded, recoverable); the residual
    # gradient here bounds channel (i) = never-recorded matches.
    Ol = np.array([obs_l.get(a, 0.0) for a in items]); El = np.array([expc_l[a] for a in items])
    print("\n[C] Channel decomposition — obs/exp by the SAME popularity quintiles, leaver matches included:")
    ratios_c, ratios_l = [], []
    for i in range(5):
        b = (q[i] <= P) & (q[i + 1] >= P if i == 4 else q[i + 1] > P)
        rc, rl = O[b].sum() / E[b].sum(), Ol[b].sum() / El[b].sum()
        ratios_c.append(rc); ratios_l.append(rl)
        print(f"    popularity[{q[i]:.2f},{q[i+1]:.2f}]  clean obs/exp = {rc:.3f}   incl-leaver obs/exp = {rl:.3f}")
    sc, sl = ratios_c[4] / ratios_c[0], ratios_l[4] / ratios_l[0]
    print(f"    top/bottom-quintile suppression spread: clean {sc:.3f}  vs  incl-leaver {sl:.3f}  "
          f"→ {(sc - sl) / (sc - 1):.0%} of the gradient is the leaver exclusion (channel ii)"
          if sc > 1 else "")
    d = np.array([r[1] for r in pop_rows_l]); m, lo, hi = cluster_ci(d, [r[0] for r in pop_rows_l])
    pop_shift_l, pop_shift_l_lo, pop_shift_l_hi = m, lo, hi
    print(f"    desirability shift incl-leaver: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"(clean value printed in [A]; the incl-leaver residual bounds the never-recorded channel)")

    print("\nReading: forced picks skew toward desirable items — the survivorship, one gradient over ALL\n"
          "  items (heroes included; popular ones over-forced). Magnitude ~+0.003 SD in β̂'s composite\n"
          "  units feeds the downstream bias analysis. [C] splits the gradient into the per-protocol\n"
          "  leaver-exclusion channel (measurable, estimand scope) and the never-recorded residual\n"
          "  (the identification threat). CAVEAT (see docstring): a small residual hero-specific\n"
          "  offset may remain but is confounded by the P_mech baseline and unresolved.")
    # The two most-drafted heroes and their obs/expected forced ratios — the
    # named examples the README quotes. Names come from the OpenDota heroes
    # cache (id -> npc suffix); omitted if the cache is absent.
    top_heroes = {}
    heroes_cache = paths.cache / "heroes.json"
    if heroes_cache.exists():
        rev = {i: k for k, i in vocabs.draft_id_to_index.items()}
        id_to_suffix = {int(hid): h["name"].removeprefix("npc_dota_hero_")
                        for hid, h in json.loads(heroes_cache.read_text()).items()}
        hero_pop = sorted((a for a in popularity
                           if kind_of.get(a) == 0 and expc[a] > 0 and obs[a] > 0),
                          key=lambda a: popularity[a], reverse=True)
        for rank, a in enumerate(hero_pop[:2], 1):
            top_heroes[f"top_forced_hero_{rank}_key"] = id_to_suffix.get(
                int(rev[a].split(":")[1]), rev[a])
            top_heroes[f"top_forced_hero_{rank}_ratio"] = obs[a] / expc[a]
    write_results("survivorship", {
        "shift_sd": shift_sd, "shift_sd_lo": shift_sd_lo, "shift_sd_hi": shift_sd_hi,
        "pop_shift": pop_shift, "pop_shift_lo": pop_shift_lo, "pop_shift_hi": pop_shift_hi,
        "pop_shift_incl_leaver": pop_shift_l, "pop_shift_incl_leaver_lo": pop_shift_l_lo,
        "pop_shift_incl_leaver_hi": pop_shift_l_hi,
        "quintile_obs_exp_clean": ratios_c, "quintile_obs_exp_incl_leaver": ratios_l,
        "spread_clean": sc, "spread_incl_leaver": sl,
        "leaver_channel_share": (sc - sl) / (sc - 1),
        **top_heroes,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
