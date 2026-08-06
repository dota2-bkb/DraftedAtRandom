"""exclusions.py — the post-treatment exclusions, one shared logic.

Every exclusion conditions on an event downstream of the forced pick (a swap, a
leaver, OpenDota's parse coverage), so each faces the same two bias routes
(REPORT.md §6): an outcome→event edge (selection on outcome) and a
treatment→event edge (collider fuel). Under randomization the treatment→event
association IS the total causal effect of the assignment on the event — so it is
directly measurable, always as a *bound* on the axis tested, never as proof of
absence. Four readouts:

  [A] composition — the forced pick's state-centered desirability,
      popularity(realized) − E_mech[popularity | s], compared across each
      event's strata (swap vs non-swap; OpenDota-parsed vs unparsed). A shift
      bounds the treatment→event selection on the desirability axis.
  [B] consequence — β̂ (composite, rank; BC and Q) on the VAL split with the
      leaver exclusion undone: analytic picks, leaver-match picks alone, their
      union. This measures the exclusion's total effect on the estimate — no
      edge semantics needed; its only caveat is its own CI. (Val split: a
      design check, not a headline readout.)
  [C] differential attrition — the leaver exclusion is identified only if
      leaving does not respond to the assigned pick (leaving is decided AFTER
      the pick; ~37% of forced picks in leaver matches are the eventual
      leaver's own). The direct test, per pick: does the pick's state-centered
      desirability differ between picks whose PICKER eventually leaves and
      picks whose picker stays? (Picker→leaver join: pick_slot from the
      bundle timeline, exactly as build_dataset derives it.)
  [E] swap timing — evidence for REPORT.md §6's "swaps are strategy-time trades executed
      before the game starts": seconds between the last draft pick and each
      swap, from the raw parse (tick clock).
  [D] ITT-win — win-outcome β̂ with NO post-treatment filter at all: every
      retrieved match (leavers, swaps, no-extended-stats included; bots out —
      a measurement-validity exclusion, engine picks are not human timeouts).
      Win is observed for every match, so this sample applies no attrition
      filter — the filter-free check of the whole exclusion structure: it
      avoids the attrition assumptions the exclusions introduce (the design's
      own assumptions remain).
      Runs on VAL and TEST (TEST read added after the primary analysis plan
      was fixed — a disclosed amendment: externally-review-motivated,
      direction-adversarial, no parameter selected on test).

Run (cuda — [B]/[D] score BC and Q):
  DOTA2AD_ROOT=work pixi run -e cuda python experiments/random-mechanism/exclusions.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dota2ad.core import (
    NUM_PLAYERS, ExcludeReason, compute_mmr_norm, default_paths, load_matches,
    load_split, load_stats_rows, load_vocabs, mech_kind_probs,
)
from dota2ad.core.collate import policy_collate
from dota2ad.core.draft_logic import idx
from dota2ad.core.mechanism import iw_to_uniform
from dota2ad.eval.bootstrap import cluster_bootstrap_ci
from dota2ad.eval.causal_rank import compute_deviations, rank_pct
from dota2ad.eval.results import write_results
from dota2ad.eval.tuples import extract_tuples
from dota2ad.models import QNetStats, load_policy
from dota2ad.pipeline.build_dataset import load_bundle
from dota2ad.training.stats_simulator import compute_stat_norm, scalarize_q
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS


def mean_ci(vals: np.ndarray, mids: list, n_boot: int) -> tuple[float, float, float]:
    m = float(vals.mean())
    lo, _, hi = cluster_bootstrap_ci(lambda ix: float(vals[ix].mean()), mids, n_boot=n_boot)
    return m, lo, hi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_matches = load_matches(paths.matches, exclude=())
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    split = load_split(paths.split)
    with open(paths.excluded) as f:
        exc = json.load(f)
    botids = set(exc["too_many_random_picks"])
    leaids = set(exc["leavers"])
    swpids = set(exc["swaps"])
    stats_rows_by_id = {r.match_id: r
                        for r in load_stats_rows(exclude=(ExcludeReason.TOO_MANY_RANDOM_PICKS,
                                                          ExcludeReason.SWAPS))}

    # Norms exactly as the models saw them: analytic train.
    analytic_train = [m for m in all_matches
                      if m.match_id not in split.held_out
                      and m.match_id not in botids and m.match_id not in leaids
                      and m.match_id not in swpids]
    mmr_mean, mmr_std = compute_mmr_norm(analytic_train)

    def U(kind, gid):
        return idx(vocabs, gid, kind)

    # Picker→leaver join for [C]: which pick_slots eventually left, per leaver
    # match. pick_slot = first-appearance order in the tick-sorted hero+ability
    # timeline — replicated exactly from build_match_row (hero_picks appended
    # first, stable sort by tick).
    left_slots: dict[int, set[int]] = {}
    for mid in leaids:
        if mid in botids:
            continue
        b = load_bundle(paths.parsed / str(mid))
        ev = [(hp.tick, hp.player_slot) for hp in b.hero_picks] \
            + [(p.tick, p.player_slot) for p in b.picks]
        ev.sort(key=lambda e: e[0])
        seen: list[int] = []
        for _, ps in ev:
            if ps not in seen:
                seen.append(ps)
        left_ps = {p.player_slot for p in b.players if p.leaver_status != 0}
        left_slots[mid] = {i for i, ps in enumerate(seen) if ps in left_ps}
    print(f"picker→leaver join built over {len(left_slots)} leaver matches")

    # ---- [A] composition: state-centered desirability by exclusion stratum ----
    # popularity = deliberate picks / pool appearances, from analytic matches.
    pool_app, delib = {}, {}
    for m in analytic_train + [m for m in all_matches
                               if m.match_id in split.held_out
                               and m.match_id not in botids and m.match_id not in leaids
                               and m.match_id not in swpids]:
        for a in ([U("h", h) for h in m.hero_pool] + [U("a", a) for a in m.basic_pool]
                  + [U("a", a) for a in m.ult_pool]):
            pool_app[a] = pool_app.get(a, 0) + 1
        for e in m.history:
            if e.is_random:
                continue
            a = U("h", e.hero_id) if e.hero_id is not None else U("a", e.draft_ability_id)
            delib[a] = delib.get(a, 0) + 1
    popularity = {a: delib.get(a, 0) / pool_app[a] for a in pool_app if pool_app[a] >= 50}

    d_shift, mids_a, flags = [], [], []
    n_lp = n_fl = n_fc = 0            # [C] headline: leaver-match forced picks
    n_all = n_disc = 0                # all non-bot forced picks / disconnected pickers
    for t in extract_tuples(all_matches, vocabs, mmr_mean, mmr_std):
        if t.match_id in botids:
            continue
        n_all += 1
        n_disc += t.picker_disconnected
        pleft = int(t.focal_slot) in left_slots.get(t.match_id, ())
        if t.match_id in leaids:
            n_lp += 1
            if pleft:
                n_fl += 1
                n_fc += not t.picker_disconnected
        ct = list(t.sample.cand_type); ci = list(t.sample.cand_idx)
        if t.action_idx not in popularity:
            continue
        NH, NB, NU = ct.count(0), ct.count(1), ct.count(2)
        ph, pb, pu = mech_kind_probs(NH, NB, NU)
        pv = {0: [], 1: [], 2: []}
        for i, a in enumerate(ci):
            if a in popularity:
                pv[ct[i]].append(popularity[a])
        if not any(pv.values()):
            continue
        expd = sum(p * np.mean(pv[k]) for p, k in ((ph, 0), (pb, 1), (pu, 2)) if pv[k])
        d_shift.append(popularity[t.action_idx] - expd)
        mids_a.append(t.match_id)
        flags.append((t.match_id in swpids, t.match_id in leaids,
                      t.match_id in stats_rows_by_id, pleft))
    d_shift = np.asarray(d_shift)
    swp = np.asarray([f[0] for f in flags])
    lea = np.asarray([f[1] for f in flags])
    prs = np.asarray([f[2] for f in flags])
    plf = np.asarray([f[3] for f in flags])
    mids_a = np.asarray(mids_a)

    print("[A] Forced-pick composition by exclusion stratum — state-centered desirability shift")
    print("    (popularity(realized) − E_mech[popularity | s]; a stratum gap bounds the")
    print("    treatment→event selection on this axis; clustered 95% CIs):")

    def stratum(name, mask):
        m, lo, hi = mean_ci(d_shift[mask], list(mids_a[mask]), args.bootstrap)
        print(f"    {name:34s} n={int(mask.sum()):6d}  shift={m:+.4f} [{lo:+.4f},{hi:+.4f}]")
        return m, lo, hi

    def gap(name, mask_a, mask_b):
        # clustered CI of the difference in means via joint resampling
        both = mask_a | mask_b
        point = float(d_shift[mask_a].mean() - d_shift[mask_b].mean())
        sub = d_shift[both]; grp = list(mids_a[both]); is_a = mask_a[both]
        lo, _, hi = cluster_bootstrap_ci(
            lambda ix: float(sub[ix][is_a[ix]].mean() - sub[ix][~is_a[ix]].mean()),
            grp, n_boot=args.bootstrap)
        print(f"    Δ {name:32s} {point:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"{' *' if lo > 0 or hi < 0 else ''}")
        return point, lo, hi

    nonswap = ~swp
    stratum("swap matches", swp)
    stratum("non-swap matches", nonswap)
    gap_swap = gap("swap − non-swap", swp, nonswap)
    analytic = ~swp & ~lea
    stratum("analytic, OpenDota-parsed", analytic & prs)
    stratum("analytic, unparsed", analytic & ~prs)
    gap_parsed = gap("parsed − unparsed (analytic)", analytic & prs, analytic & ~prs)

    # ---- [C] differential attrition: does the pick predict the PICKER leaving? --
    print("\n[C] Differential attrition — pick desirability vs the picker's own leaving")
    connected_rate = 1 - n_disc / n_all
    print(f"    all forced picks (non-bot): {n_all}; picker connected at pick time: "
          f"{connected_rate:.1%}")
    print(f"    forced picks in leaver matches: {n_lp}; by the eventual leaver: "
          f"{n_fl} ({n_fl / n_lp:.1%}); of those, connected at pick time: {n_fc / n_fl:.1%}")
    print("    (leaving is decided after the pick ⇒ the exclusion is identified only if")
    print("    leaving ⫫ pick | state; a stratum gap here bounds that dependence):")
    leaver_own_shift = stratum("picker eventually leaves", plf)
    stayer_shift = stratum("picker stays", ~plf)
    gap_leave = gap("leaves − stays (all matches)", plf, ~plf)
    stratum("leaver matches: leaver's picks", lea & plf)
    stratum("leaver matches: others' picks", lea & ~plf)
    gap_leave_within = gap("within leaver matches only", lea & plf, lea & ~plf)

    # ---- [E] swap timing: the §6 "no outcome→swap edge" evidence ---------------
    # A swap is a strategy-time trade — it must land within moments of the last
    # draft pick, never mid-game. Measured from the raw parse (tick clock, 30/s).
    dts = []
    swp_mids = swpids - botids
    for mid in swp_mids:
        with open(paths.parsed / str(mid) / "draft_details.json") as f:
            dd = json.load(f)
        last = max(p["tick"] for p in dd["picks"] + dd["hero_picks"])
        dts += [(s["tick"] - last) / 30.0 for s in dd["swaps"]]
    dts = np.asarray(dts)
    print(f"\n[E] Swap timing (n={len(dts)} swaps in {len(swp_mids)} matches, seconds "
          f"after the last draft pick):")
    print(f"    p50={np.percentile(dts, 50):.0f}s  p95={np.percentile(dts, 95):.0f}s  "
          f"p99={np.percentile(dts, 99):.0f}s  max={dts.max():.0f}s  "
          f"within 120s: {(dts <= 120).mean():.1%}  before the last pick: {(dts < 0).mean():.2%}")

    # ---- [B] consequence: β̂ on VAL with the leaver exclusion undone ------------
    print("\n[B] β̂ (composite, rank) on the VAL split, leaver exclusion undone:")
    policy = load_policy(paths.policy_ckpt, vocabs, device)
    policy.requires_grad_(False)
    q = QNetStats.load_from_ckpt(paths.models / "stats_dqn.pt", vocab_size, device)
    q.eval()

    def _iter():
        for m in analytic_train:
            for ps in range(NUM_PLAYERS):
                yield m.match_id, ps
    stat_norm_mean, stat_norm_std = compute_stat_norm(stats_rows_by_id, _iter())

    val_union = [m for m in all_matches
                 if m.match_id in split.val_ids
                 and m.match_id not in botids and m.match_id not in swpids]
    tuples = extract_tuples(val_union, vocabs, mmr_mean, mmr_std)
    d = compute_deviations(tuples, policy, q, DEFAULT_BALANCED_WEIGHTS, vocab_size,
                           stats_rows_by_id, stat_norm_mean, stat_norm_std, device)
    kept = [t for t in tuples if t.match_id in stats_rows_by_id
            and len(t.sample.cand_idx) >= 3]
    assert len(kept) == d.n, (len(kept), d.n)
    is_lea = np.asarray([t.match_id in leaids for t in kept])
    mids_b = np.asarray(d.mids)
    res_b: dict[str, float] = {}
    b_key = {"analytic (as published)": "analytic", "leaver-match picks only": "leaver",
             "union (exclusion undone)": "union"}
    for name, delta in (("BC", d.bc_rank), ("Q", d.qc_rank)):
        c = d.w_prop * delta * d.comp
        for lab, mask in (("analytic (as published)", ~is_lea),
                          ("leaver-match picks only", is_lea),
                          ("union (exclusion undone)", np.ones(d.n, bool))):
            m, lo, hi = mean_ci(c[mask], list(mids_b[mask]), args.bootstrap)
            k = f"val_{name.lower()}_{b_key[lab]}"
            res_b[k], res_b[k + "_lo"], res_b[k + "_hi"] = m, lo, hi
            print(f"    {name:2s} {lab:26s} n={int(mask.sum()):6d}  β̂={m:+.4f} [{lo:+.4f},{hi:+.4f}]")

    # ---- [D] ITT-win: no post-treatment filter at all --------------------------
    # Win is observed for every retrieved match, so this sample retains leavers,
    # swaps, and no-extended-stats matches (bots stay out: measurement validity).
    def win_devs(tuples):
        """BC/Qc rank-δ of the realized action + w_prop + win — no stats needed."""
        tup = [t for t in tuples if len(t.sample.cand_idx) >= 3]
        n = len(tup)
        K = q.score_mlp[-1].out_features
        bc_r = np.zeros(n); qc_r = np.zeros(n); wp = np.zeros(n); win = np.zeros(n)
        mids: list = []
        ptr = 0
        with torch.no_grad():
            for i0 in range(0, n, 256):
                chunk = tup[i0:i0 + 256]
                b = policy_collate([t.sample for t in chunk], device=device)
                qc = scalarize_q(q(b).cpu(), DEFAULT_BALANCED_WEIGHTS[:K])
                bc = policy(b).cpu().exp()[:, :vocab_size]
                for t in chunk:
                    feas = list(t.sample.cand_idx)
                    ridx = feas.index(t.action_idx)
                    fi = torch.tensor(feas, dtype=torch.long)
                    jj = ptr - i0
                    qc_r[ptr] = (rank_pct(qc[jj].index_select(0, fi).numpy()) - 0.5)[ridx]
                    bc_r[ptr] = (rank_pct(bc[jj].index_select(0, fi).numpy()) - 0.5)[ridx]
                    win[ptr] = t.focal_team_won
                    wp[ptr] = iw_to_uniform(t.sample.cand_type, ridx)
                    mids.append(t.match_id)
                    ptr += 1
        return tup, bc_r, qc_r, wp, win, np.asarray(mids)

    print("\n[D] ITT-win: win β̂ (rank) with NO post-treatment filter (leavers, swaps,")
    print("    no-stats matches all retained; bots out). TEST read added after the primary")
    print("    plan was fixed — a disclosed amendment (review-motivated, direction-adversarial,")
    print("    no parameter selected on test):")
    res_d: dict[str, float] = {}
    for sname, sid in (("VAL", split.val_ids), ("TEST", split.test_ids)):
        ms = [m for m in all_matches if m.match_id in sid and m.match_id not in botids]
        tup, bc_r, qc_r, wp, win, mids_d = win_devs(extract_tuples(ms, vocabs, mmr_mean, mmr_std))
        lea_m = np.asarray([t.match_id in leaids for t in tup])
        swp_m = np.asarray([t.match_id in swpids for t in tup])
        prs_m = np.asarray([t.match_id in stats_rows_by_id for t in tup])
        pub = ~lea_m & ~swp_m & prs_m
        for name, delta in (("BC", bc_r), ("Q", qc_r)):
            for lab, mask in (("published-definition sample", pub),
                              ("ITT (all retrieved)", np.ones(len(tup), bool))):
                wz = (win[mask] - win[mask].mean()) / (win[mask].std() or 1.0)
                c = wp[mask] * delta[mask] * wz
                m, lo, hi = mean_ci(c, list(mids_d[mask]), args.bootstrap)
                k = f"win_{sname.lower()}_{name.lower()}_{'pub' if lab.startswith('published') else 'itt'}"
                res_d[k], res_d[k + "_lo"], res_d[k + "_hi"] = m, lo, hi
                s = " *" if lo > 0 or hi < 0 else ""
                print(f"    {sname:4s} {name:2s} {lab:27s} n={int(mask.sum()):6d}  "
                      f"β̂={m:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")

    write_results("exclusions", {
        "connected_rate": connected_rate,
        "n_forced_nonbot": n_all,
        "swap_forced_gain": float((swp & ~lea).sum() / analytic.sum()),
        "gap_swap": gap_swap[0], "gap_swap_lo": gap_swap[1], "gap_swap_hi": gap_swap[2],
        "gap_parsed": gap_parsed[0], "gap_parsed_lo": gap_parsed[1], "gap_parsed_hi": gap_parsed[2],
        "gap_leave": gap_leave[0], "gap_leave_lo": gap_leave[1], "gap_leave_hi": gap_leave[2],
        "gap_leave_within": gap_leave_within[0], "gap_leave_within_lo": gap_leave_within[1],
        "gap_leave_within_hi": gap_leave_within[2],
        "swap_p50_s": float(np.percentile(dts, 50)),
        "swap_within_120s": float((dts <= 120).mean()),
        "swap_max_s": float(dts.max()),
        "leaver_own_shift": leaver_own_shift[0], "leaver_own_shift_lo": leaver_own_shift[1],
        "leaver_own_shift_hi": leaver_own_shift[2],
        "stayer_shift": stayer_shift[0], "stayer_shift_lo": stayer_shift[1],
        "stayer_shift_hi": stayer_shift[2],
        "n_swap_events": len(dts),
        "n_leaver_forced": n_lp,
        "leaver_own_share": n_fl / n_lp,
        "leaver_own_connected_share": n_fc / n_fl,
        **res_b, **res_d,
    })
    print("\nReading: [A] bounds each exclusion's selection-on-treatment on the desirability")
    print("axis (a bound, not proof of absence — power and axis caveats apply). [C] is the")
    print("differential-attrition test: a ≈0 gap supports the leaver exclusion's identifying")
    print("assumption (leaving ⫫ pick | state) on the desirability axis. [B] and [D] are the")
    print("bottom line: [B] undoes the leaver exclusion on the composite; [D] applies no")
    print("post-treatment filter at all on the always-observed win outcome. If both track the")
    print("published estimates, the exclusion structure demonstrably does not drive the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
