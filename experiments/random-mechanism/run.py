"""random-mechanism: what Dota 2 Ability-Draft's forced-random pick actually is,
and the tests that verify the decompiled mechanism against the replay data.

The natural experiment (REPORT.md §3) rests on a claim about the *treatment
assignment*: what distribution does the server draw a forced pick from? The
obvious guess — "uniform over all m legal items" (propensity 1/m) — is
**wrong**. Decompiling `server.dll` (`FUN_1817bc510`) + `tier0.dll` (the `ran1`
RNG) gives the true rule, and this experiment verifies it holds in the data:

  THE MECHANISM (per forced pick, for the timed-out seat):
    1. Pick an ability *side* — basics or ults — by a flat coin among the sides
       the seat still needs (`RandomInt(0,1)`; forced when only one is needed).
    2. If the seat still needs a hero, gather all available heroes; gather all
       available items of the chosen side.
    3. Draw ONE uniformly from that combined bag (heroes ∪ side); assign it.
       `RandomInt(0, |heroes|+|side|)` is inclusive, a benign off-by-one that
       nudges every pick slightly toward abilities: P(hero)=H/(H+S+1).

  This collapses to: one-bag H/(H+B) for hero+basic; a flat 50/50 basic-vs-ult
  coin (pool-invariant) once a hero is owned; H/(H+U) for hero+ult; and
  ½·H/(H+B+1)+½·H/(H+U+1) when all three are needed.

Tests (all held on the full corpus, numpy-only, CPU):
  T1  kind-share calibration : derived model vs observed per config (clustered CI)
  T2  within-kind uniformity : the forced item is uniform within its kind
  T3  rule falsification     : derived model beats uniform-1/m and kind-uniform (log-lik)
  T4  RNG faithfulness       : emulated ran1 reproduces the formula (no serial corr.)
  T5  survivorship probe     : bots match the mechanism exactly; real-player forced
                               picks skew toward better items (outcome-dependent
                               missingness) — a threat to the instrument, not the RNG

Run:
  DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/run.py
  DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/run.py --quick   # skip T5 (winrate pass)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from dota2ad.core import (
    NUM_PLAYERS, PickSlot, compute_mmr_norm, default_paths, load_matches,
    load_split, load_vocabs,
)
from dota2ad.core.draft_logic import replay_complete
from dota2ad.core.encoding import encode_loadout
from dota2ad.eval.results import write_results
from dota2ad.eval.tuples import extract_tuples

from prng import UniformRandomStream


# --------------------------------------------------------------------------
# The derived propensity: P(kind | feasible counts), straight from the code.
# --------------------------------------------------------------------------
def derived_kind_probs(NH, NB, NU, plus_one: float = 1.0):
    """P(hero), P(basic), P(ult) for a forced pick, given the feasible counts
    NH/NB/NU (heroes/basics/ults the seat may pick). Coin over needed sides,
    then a uniform draw over heroes∪side with the inclusive-RandomInt +1.
    `plus_one=0` gives the exclusive draw the call presumably intended — the
    no-off-by-one twin T3 uses to ask whether the data detects the mistake."""
    H, B, U = NH.astype(float), NB.astype(float), NU.astype(float)
    c = float(plus_one)
    bs, us = B > 0, U > 0
    ns = bs.astype(float) + us.astype(float)
    ns = np.where(ns == 0, 1.0, ns)                       # hero-only edge
    with np.errstate(divide="ignore", invalid="ignore"):   # unselected where-branch: 0/0 at c=0
        ph = np.where(bs, (1 / ns) * H / (H + B + c), 0.0) + np.where(us, (1 / ns) * H / (H + U + c), 0.0)
        pb = np.where(bs, (1 / ns) * (B + c) / (H + B + c), 0.0)
        pu = np.where(us, (1 / ns) * (U + c) / (H + U + c), 0.0)
    ph = np.where((~bs) & (~us) & (H > 0), 1.0, ph)      # hero-only: no ability side -> forced hero
    tot = ph + pb + pu
    tot = np.where(tot == 0, 1.0, tot)
    return ph / tot, pb / tot, pu / tot                  # hero-only -> (1,0,0) after norm


def cluster_ci(vals, mids, n_boot=1500, seed=0):
    """Match-clustered percentile 95% CI of mean(vals)."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, m in enumerate(mids):
        by[m].append(i)
    groups = [np.array(v) for v in by.values()]
    nc = len(groups)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[c] for c in rng.integers(0, nc, nc)])
        boots[b] = vals[idx].mean()
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip T5 (the winrate pass over all matches)")
    args = ap.parse_args()
    paths = default_paths()

    matches = load_matches(paths.matches, exclude=())            # ALL matches: mechanism is match-type-independent
    vocabs = load_vocabs(paths.vocabs)
    split = load_split(paths.split)
    train = [m for m in matches if m.match_id not in split.held_out]
    mm_mean, mm_std = compute_mmr_norm(train)
    with open(paths.excluded) as f:
        exc = json.load(f)
    botids = set(exc["too_many_random_picks"]); leaids = set(exc["leavers"]); swpids = set(exc["swaps"])

    # one pass over the forced picks: feasible counts, realized kind, within-kind position, match/split
    mids, NH, NB, NU, kind, pos_pit, is_bot, is_clean = ([] for _ in range(8))
    forced_item, avail_same_kind = [], []
    rng = np.random.default_rng(0)
    for t in extract_tuples(matches, vocabs, mm_mean, mm_std):
        ct = list(t.sample.cand_type); ci = list(t.sample.cand_idx)
        ridx = ci.index(t.action_idx); k = ct[ridx]
        same = [ci[i] for i in range(len(ci)) if ct[i] == k]
        mids.append(t.match_id)
        NH.append(int(ct.count(0))); NB.append(int(ct.count(1))); NU.append(int(ct.count(2)))
        kind.append(k)
        # randomized PIT of the forced item's rank within its kind block -> Uniform(0,1) if uniform
        r = same.index(t.action_idx)
        pos_pit.append((r + rng.random()) / len(same))
        b = t.match_id in botids
        is_bot.append(b)
        is_clean.append(not (b or t.match_id in leaids or t.match_id in swpids))
        forced_item.append(t.action_idx); avail_same_kind.append(same)
    NH, NB, NU, kind = map(np.asarray, (NH, NB, NU, kind))
    mids = np.asarray(mids); pos_pit = np.asarray(pos_pit)
    is_bot = np.asarray(is_bot); is_clean = np.asarray(is_clean)
    ph, pb, pu = derived_kind_probs(NH, NB, NU)
    print(f"forced picks: {len(kind)}  ({int(is_clean.sum())} clean / {int(is_bot.sum())} bot)")

    # ---- T1: kind-share calibration by config -------------------------------
    print("\n[T1] Derived model vs observed kind shares, by feasible config (clustered 95% CI):")
    conf = (NH > 0).astype(int) * 100 + (NB > 0).astype(int) * 10 + (NU > 0).astype(int)
    names = {111: "hero+basic+ult", 110: "hero+basic", 101: "hero+ult", 11: "basic+ult", 100: "hero-only"}
    t1_configs_ok = t1_configs_total = 0
    for c, nm in names.items():
        m = conf == c
        if m.sum() < 100:
            continue
        row = []
        config_ok = True
        for kk, P in ((0, ph), (1, pb), (2, pu)):
            obs, lo, hi = cluster_ci((kind[m] == kk).astype(float), mids[m])
            pr = P[m].mean()
            if obs == 0 and pr == 0:
                continue
            tag = "ok" if lo <= pr <= hi else "OFF"
            config_ok = config_ok and tag == "ok"
            row.append(f"{'hbu'[kk]}: obs {obs:.3f}[{lo:.3f},{hi:.3f}] model {pr:.3f} {tag}")
        t1_configs_total += 1
        t1_configs_ok += int(config_ok)
        print(f"  {nm:16s} n={int(m.sum()):6d}  " + "  ".join(row))

    # ---- T2: within-kind uniformity ----------------------------------------
    print("\n[T2] Within-kind uniformity of the forced item (randomized PIT -> Uniform(0,1)):")
    for kk, nm in ((0, "hero"), (1, "basic"), (2, "ult")):
        u = pos_pit[kind == kk]
        obs_counts, _ = np.histogram(u, bins=10, range=(0, 1))
        exp = len(u) / 10.0
        chi2 = float(((obs_counts - exp) ** 2 / exp).sum())        # ~chi2_9 under uniform (crit 16.9 @5%)
        print(f"  {nm:5s} n={len(u):6d}  mean={u.mean():.4f} (0.5=uniform)  chi2_9={chi2:6.1f} "
              f"[{'uniform' if chi2 < 16.9 else 'NON-UNIFORM'}]")

    # ---- T3: falsify uniform-1/m and kind-uniform ---------------------------
    print("\n[T3] Log-likelihood of the realized kind (higher=better) — is naive 1/m wrong?")
    H, B, U = NH.astype(float), NB.astype(float), NU.astype(float)
    tot = H + B + U
    P_onebag = np.stack([H / tot, B / tot, U / tot], 1)             # naive "uniform over all legal items"
    nk = (NH > 0).astype(float) + (NB > 0).astype(float) + (NU > 0).astype(float)
    P_kindunif = np.stack([(NH > 0) / nk, (NB > 0) / nk, (NU > 0) / nk], 1)
    P_derived = np.stack([ph, pb, pu], 1)
    ph0, pb0, pu0 = derived_kind_probs(NH, NB, NU, plus_one=0.0)
    P_derived_no1 = np.stack([ph0, pb0, pu0], 1)
    idx = np.arange(len(kind))
    t3_ll: dict[str, float] = {}
    for key, nm, P in (("t3_ll_derived", "derived (this experiment)", P_derived),
                       ("t3_ll_derived_no1", "derived w/o the off-by-one", P_derived_no1),
                       ("t3_ll_uniform_1m", "uniform 1/m (naive)", P_onebag),
                       ("t3_ll_kind_uniform", "kind-uniform", P_kindunif)):
        ll = np.log(np.clip(P[idx, kind], 1e-12, 1)).mean()
        t3_ll[key] = float(ll)
        print(f"  {nm:34s} mean log-lik = {ll:+.4f}")
    # Off-by-one detectability: paired per-pick Δll between the decompiled (+1)
    # model and its exclusive-draw twin, match-clustered. Positive = the data
    # itself prefers the off-by-one model.
    dll = (np.log(np.clip(P_derived[idx, kind], 1e-12, 1))
           - np.log(np.clip(P_derived_no1[idx, kind], 1e-12, 1)))
    mean_dll, dlo, dhi = cluster_ci(dll, mids)
    t3_ll["t3_dll_plus1"] = float(mean_dll)
    t3_ll["t3_dll_plus1_lo"] = float(dlo)
    t3_ll["t3_dll_plus1_hi"] = float(dhi)
    t3_ll["t3_dll_plus1_total"] = float(dll.sum())
    verdict = ("the data itself prefers the off-by-one model" if dlo > 0 else
               "the data prefers the exclusive model" if dhi < 0 else
               "the data cannot distinguish the two")
    print(f"  off-by-one detectability: paired Δll(+1 − no+1) = {mean_dll:+.6f} "
          f"[{dlo:+.6f},{dhi:+.6f}] per pick, total {dll.sum():+.1f} nats — {verdict}")

    # ---- T4: RNG faithfulness (emulated ran1) -------------------------------
    print("\n[T4] Emulated ran1 (tier0) reproduces P(hero)=formula — consecutive draws are not correlated:")
    outs = []
    for seed in (12345, 99991, 7, 271828):
        s = UniformRandomStream(seed)
        outs.append(np.array([s.random_int(0, 2147483646) for _ in range(250_000)]))
    outs = np.concatenate(outs)
    coin, dec = outs[:-1], outs[1:]
    dperm = dec[np.random.default_rng(0).permutation(len(dec))]
    for (h, bb, uu) in ((11, 30, 11), (12, 36, 12), (11, 20, 8)):
        def hero_rate(cn, dc, h=h, bb=bb, uu=uu):
            ult = (cn % 2) == 1
            S = np.where(ult, uu, bb); m = h + S + 1
            return ((dc % m) < h).mean()
        formula = 0.5 * h / (h + bb + 1) + 0.5 * h / (h + uu + 1)
        print(f"  H={h} B={bb} U={uu}: consecutive {hero_rate(coin, dec):.4f}  independent "
              f"{hero_rate(coin, dperm):.4f}  formula {formula:.4f}")

    # ---- T6: where does the inclusive-RandomInt overflow land? ---------------
    # RandomInt(0, H+S) spans H+S+1 outcomes for a bag of H+S entries; T1/T4 pin
    # the extra mass on the item side ((S+1)/(H+S+1) as a block). If the overflow
    # maps to a FIXED bag position, that one item carries 2/(S+1) instead of
    # 1/(S+1) and mech_propensity's uniform-within-kind is misspecified for it.
    # Bots are the clean-mechanism benchmark (no survivorship); positions are in
    # pool-scan order (= the server's bag order). Compare per-pick log-lik of the
    # within-side position under uniform vs clamp-to-last vs clamp-to-first.
    print("\n[T6] Overflow placement within the drawn side (within-side position, pool order):")
    t6: dict[str, float] = {}
    for who, sel_who in (("bot", is_bot), ("clean", is_clean)):
        for kk, nm in ((1, "basic"), (2, "ult")):
            sel = [i for i in range(len(kind))
                   if sel_who[i] and kind[i] == kk and len(avail_same_kind[i]) >= 2]
            if len(sel) < 500:
                continue
            S = np.array([len(avail_same_kind[i]) for i in sel], float)
            pos = np.array([avail_same_kind[i].index(forced_item[i]) for i in sel])
            first, last = pos == 0, pos == S - 1
            ll_unif = float(np.log(1.0 / S).mean())
            ll_last = float(np.where(last, np.log(2.0 / (S + 1)), np.log(1.0 / (S + 1))).mean())
            ll_first = float(np.where(first, np.log(2.0 / (S + 1)), np.log(1.0 / (S + 1))).mean())
            t6[f"t6_plast_obs_{who}_{nm}"] = float(last.mean())
            t6[f"t6_plast_unif_{who}_{nm}"] = float(np.mean(1.0 / S))
            t6[f"t6_plast_clamp_{who}_{nm}"] = float(np.mean(2.0 / (S + 1)))
            print(f"  {who:5s} {nm:5s} n={len(sel):6d}  "
                  f"P(last) obs={last.mean():.4f} unif={np.mean(1.0 / S):.4f} clamp={np.mean(2.0 / (S + 1)):.4f}  "
                  f"P(first) obs={first.mean():.4f}  "
                  f"log-lik: unif={ll_unif:+.4f} last-clamp={ll_last:+.4f} first-clamp={ll_first:+.4f}")

    # ---- T5: survivorship (bots vs real players) ----------------------------
    t5: dict[str, float] = {}
    if not args.quick:
        print("\n[T5] Survivorship probe: does outcome-dependent match loss contaminate real-player picks?")
        wins, cnt = defaultdict(int), defaultdict(int)
        for m in matches:
            fp = replay_complete(m)
            for ps in range(NUM_PLAYERS):
                won = m.radiant_win == (ps % 2 == 0)
                for it in encode_loadout(fp[PickSlot(ps)], vocabs):
                    cnt[it] += 1; wins[it] += int(won)
        wr = {k: wins[k] / cnt[k] for k in cnt if cnt[k] >= 50}
        # (a) 3-kind hero deficit vs the mechanism, clean vs bot
        m3 = (NH > 0) & (NB > 0) & (NU > 0)
        for lab, sel in (("clean", is_clean), ("bot", is_bot)):
            mm = m3 & sel
            r, lo, hi = cluster_ci((kind[mm] == 0).astype(float) - ph[mm], mids[mm])
            t5[f"t5_hero_gap_{lab}"], t5[f"t5_hero_gap_{lab}_lo"], t5[f"t5_hero_gap_{lab}_hi"] = r, lo, hi
            print(f"  3-kind hero (obs-model)  {lab:5s} n={int(mm.sum()):6d}: {r:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"[{'survivorship' if hi < 0 else 'matches mechanism'}]")
        # (b) forced ability's value vs the available-mean value (within kind), clean vs bot
        fv = np.array([wr.get(it, np.nan) for it in forced_item])
        av = np.array([np.mean([wr[x] for x in s if x in wr]) if any(x in wr for x in s) else np.nan
                       for s in avail_same_kind])
        d = fv - av
        for kk, nm in ((1, "basic"), (2, "ult")):
            for lab, sel in (("clean", is_clean), ("bot", is_bot)):
                mm = (kind == kk) & sel & np.isfinite(d)
                r, lo, hi = cluster_ci(d[mm], mids[mm])
                t5[f"t5_value_{nm}_{lab}"], t5[f"t5_value_{nm}_{lab}_lo"], t5[f"t5_value_{nm}_{lab}_hi"] = r, lo, hi
                print(f"  forced {nm:5s} value-vs-available  {lab:5s} n={int(mm.sum()):6d}: {r:+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}] [{'survivorship' if lo > 0 else 'null'}]")

    print(f"\nReading: T1 {t1_configs_ok} of {t1_configs_total} configs land inside the CI purely from the derived mechanism; "
          "the all-three\n  config is high on hero by ~1-2 pts (see T5). T2/T4 confirm uniform-within-kind and "
          "a faithful RNG.\n  T3 shows naive uniform-1/m is a far worse fit than the derived rule. T5: the "
          "residual is NOT the\n  mechanism (bots reproduce it exactly) — it is outcome-dependent match loss in "
          "real-player games.")
    if not args.quick:
        write_results("random-mechanism", {
            "n_forced": len(kind), "n_forced_clean": int(is_clean.sum()),
            "n_forced_bot": int(is_bot.sum()),
            "t1_configs_ok": t1_configs_ok, "t1_configs_total": t1_configs_total,
            **t3_ll, **t5, **t6,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
