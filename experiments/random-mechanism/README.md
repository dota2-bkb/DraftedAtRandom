# random-mechanism

**What:** the ground truth of the natural experiment's *treatment assignment* —
what distribution the Dota 2 server actually draws an Ability-Draft forced-random
pick from. The causal test (REPORT.md §3, §5) rests on knowing this propensity. The
obvious guess — **uniform over all `m` legal items** (`P(a|s)=1/m`) — is
**wrong**. This experiment documents the real mechanism, recovered by
decompiling the game binaries, and verifies it against the replay data — so a
reader can confirm what Dota's random *is*, either by re-decompiling or by running
the behavioral tests.

## The mechanism

On a pick-timer expiry (7 s = 210 ticks) for a seat, the server (`server.dll`,
`FUN_1817bc510`):

1. **Picks an ability *side* — basics or ults — by a flat coin.** Among the sides
   the seat still needs (has `<3` basics / has no ult), choose one with
   `RandomInt(0,1)` (forced when only one is needed). The coin is a bare 50/50,
   **independent of how many basic slots remain** (`b`-invariant).
2. **Gathers one combined bag:** all *available* heroes (only if the seat still
   needs a hero, gated by `FUN_182094cc0`) **plus** all available items of the
   chosen side.
3. **Draws one uniformly** from that bag and assigns it — with a genuine off-by-one
   mistake in the caller: the bag holds `H+S` entries, but the draw is
   `RandomInt(0, H+S)`, and tier0's `RandomInt(min, max)` is inclusive by contract —
   `H+S+1` outcomes for `H+S` entries (the correct call for a 0-indexed bag is
   `RandomInt(0, H+S−1)`). The extra outcome lands on the ability side — spread
   evenly across it, not on any fixed position (T6) — so every timeout is tilted
   slightly toward abilities: `P(hero)=H/(H+S+1)`, `P(item)=(S+1)/(H+S+1)`, where an
   exclusive draw would give `H/(H+S)` and `S/(H+S)`. Benign in play and invisible
   without the decompilation, but real — and modeled exactly in `core/mechanism.py`.

Item choice within a kind is uniform. This one routine collapses into everything
observed:

| seat still needs | shape | why |
|---|---|---|
| hero + basics (has ult) | **one bag** `H/(H+B)` | basics forced; hero∪basics drawn together |
| basics + ult (has hero) | **flat 50/50** basic-vs-ult, pool-invariant | heroes excluded ⇒ the side-coin *is* the outcome |
| hero + ult (has 3 basics) | `H/(H+U)` | ults forced; hero∪ults |
| hero + basics + ult | `½·H/(H+B+1) + ½·H/(H+U+1)` ≈ .375/.375/.25 | the coin averages the two bags |
| hero only (abilities full) | forced hero | no ability side |

The RNG is Valve's `CUniformRandomStream` (`tier0.dll`) — Numerical Recipes `ran1`:
a Park-Miller LCG `idum = 16807·idum mod 2147483647` with a Bays-Durham shuffle
(NTAB=32, index `j = iy>>26`). Faithfully re-implemented in `prng.py`.

## Decompilation map (verify by decompiling)

Addresses are from the build analyzed here; they drift across patches — the stable
anchors are the RTTI/export names and the algorithm.

- **`server.dll`**
  - `FUN_1817bc510(gamerules, seat)` — the timeout picker (steps 1–3 above);
    writes `m_bAbilityDraftCurrentPlayerHasPicked` (gamerules `+0x1248`).
  - `FUN_1817bd760` — counts the seat's basics and sets `has_ult` (feeds the coin).
  - `FUN_182094cc0` — the "does this seat already have a hero?" gate.
  - Ability pool: `*(gamerules+0x11c8)`, 48 entries × `0x40`, indices `[0,36)`
    basic / `[36,48)` ult; heroes `*(gamerules+0x1250)`, 12 × `0x38`;
    "available" marker `== 10` on field `+0x34`.
  - `RandomInt` free-function thunk `0x182c3560d` → imports from `tier0.dll`.
- **`tier0.dll`** (exported, no full analysis needed)
  - `RandomInt` `@0x18015e0f0` — `range≤1 ⇒ return min` (so `RandomInt(0,0)` does
    **not** draw); else rejection-sample then `min + g % range`.
  - `GenerateRandomNumber_Locked` `@0x18015d9d0` — `ran1` (constants `0x41a7`=16807,
    `0x7fffffff`, shuffle `iy>>26`).
  - `SetSeed` `@0x18015dfa0`.

## Verification (`run.py`)

- **T1 kind-share calibration** — the derived model reproduces observed shares
  within a match-clustered 95% CI in **<!--n random-mechanism: {t1_configs_ok} of {t1_configs_total}-->4 of 5<!--/n-->** configs; the all-three
  config is high on hero by ~1–2 pts (see T5).
- **T2 within-kind uniformity** — the forced item is uniform within its kind
  (randomized PIT, χ²₉ well under the 16.9 threshold for hero/basic/ult).
- **T3 rule falsification** — mean log-likelihood of the realized kind:
  derived <!--n random-mechanism: **{t3_ll_derived:+.3f}**-->**-0.568**<!--/n--> > <!--n random-mechanism: kind-uniform {t3_ll_kind_uniform:+.3f}-->kind-uniform −0.597<!--/n--> > <!--n random-mechanism: **uniform-1/m {t3_ll_uniform_1m:+.3f}**-->**uniform-1/m −0.619**<!--/n--> (the naive
  assumption is the worst fit). The off-by-one itself is **behaviorally
  detectable**: against an exclusive-draw twin of the derived rule
  (`plus_one=0`), the paired per-pick Δll is
  <!--n random-mechanism: {t3_dll_plus1:+.5f} [{t3_dll_plus1_lo:+.5f}, {t3_dll_plus1_hi:+.5f}]-->+0.00071 [+0.00055, +0.00087]<!--/n--> —
  <!--n random-mechanism: {t3_dll_plus1_total:+.0f} nats over the corpus-->+79 nats over the corpus<!--/n--> in favor of the buggy call: the
  data independently confirms the mistake.
- **T4 RNG faithfulness** — the emulated `ran1` coin-then-decision sequence
  reproduces `P(hero)=formula` and consecutive draws behave as independent
  (the shuffle decorrelates them; no serial-correlation artifact).
- **T6 overflow placement** — the inclusive-`RandomInt` extra outcome does **not**
  land on any fixed bag position: within-side positions are uniform on the bot
  benchmark (basics P(last) <!--n random-mechanism: obs {t6_plast_obs_bot_basic:.4f} vs uniform's {t6_plast_unif_bot_basic:.4f}; a clamp-to-last would
  predict {t6_plast_clamp_bot_basic:.4f}-->obs 0.0576 vs uniform's 0.0576; a clamp-to-last would
  predict 0.1075<!--/n-->; log-likelihood favors uniform over both clamp models in every
  cell), so `mech_propensity`'s uniform-within-kind is exact at the item level.
- **T5 survivorship probe** — bots time out every turn, so their forced picks are
  *pure* server-random and reproduce the mechanism **exactly** (3-kind hero
  <!--n random-mechanism: obs−model `{t5_hero_gap_bot:+.3f}`-->obs−model `+0.002`<!--/n-->, forced-ability value bias null). **Real-player** forced picks
  do not: a 3-kind <!--n random-mechanism: hero deficit of **{t5_hero_gap_clean:+.3f}**-->hero deficit of **−0.020**<!--/n--> and a small positive forced-ability
  value bias <!--n random-mechanism: (`{t5_value_basic_clean:+.4f}`/`{t5_value_ult_clean:+.4f}`, basics/ults)-->(`+0.0006`/`+0.0005`, basics/ults)<!--/n-->. This is **outcome-dependent match
  loss** — matches where the coin handed someone an undesirable random
  (early hero; a weak ability) are disproportionately abandoned and never recorded
  — not a flaw in the mechanism (bots prove it is exact) or the RNG (T4).

**`survivorship.py`** quantifies that contamination on **one desirability axis** for
all items — no hero special-case. Desirability is **draft popularity** (deliberate
picks / times in pool: how often players actually choose it); per item,
`suppression = observed forced count / count expected under the mechanism` (null 1).
Pooling every item and binning by popularity gives a clean monotone gradient —
undesirable items suppressed, desirable ones over-forced (obs/exp by popularity
quintile, low → high:
<!--n survivorship: {quintile_obs_exp_clean[0]:.3f} / {quintile_obs_exp_clean[1]:.3f} / {quintile_obs_exp_clean[2]:.3f} / {quintile_obs_exp_clean[3]:.3f} / {quintile_obs_exp_clean[4]:.3f}-->0.969 / 0.983 / 1.008 / 1.029 / 1.105<!--/n-->).

Crucially the **most-drafted heroes are over-forced, not suppressed**
(<!--n survivorship: {top_forced_hero_1_key} obs/exp≈{top_forced_hero_1_ratio:.2f}-->lina obs/exp≈1.18<!--/n-->,
<!--n survivorship: {top_forced_hero_2_key}≈{top_forced_hero_2_ratio:.2f}-->nevermore≈1.19<!--/n-->) — so there is no "hero block"; the <!--n random-mechanism: average {t5_hero_gap_clean:+.3f} hero-->average −0.020 hero<!--/n-->
deficit is just that heroes are, on average, less-drafted, concentrated in the
*unpopular* ones. The survivorship magnitude, aggregated over all items: forced
picks are value-shifted <!--n survivorship: **{shift_sd:+.4f} SD** upward-->**+0.0033 SD** upward<!--/n--> in the **composite** units β̂ uses
<!--n survivorship: (clustered CI [{shift_sd_lo:+.4f}, {shift_sd_hi:+.4f}])-->(clustered CI [+0.0016, +0.0051])<!--/n--> — the handle for the downstream β̂-bias analysis.

**Channel decomposition** (section [C]): the clean-set gradient conflates matches
**never recorded** (channel i) with matches recorded but **excluded per-protocol as
leaver games** (channel ii — recoverable, and estimand scope rather than an
identification threat). Recomputing the gradient with leaver matches included (same
items, same popularity bins) leaves it essentially intact — top/bottom-quintile
<!--n survivorship: spread {spread_incl_leaver:.3f} vs {spread_clean:.3f} clean-->spread 1.132 vs 1.140 clean<!--/n-->, <!--n survivorship: ~**{leaver_channel_share:.0%}** of the gradient-->~**6%** of the gradient<!--/n-->; <!--n survivorship: desirability shift {pop_shift_incl_leaver:+.4f} vs
{pop_shift:+.4f}-->desirability shift +0.0059 vs
+0.0060<!--/n--> — so the suppression sits at the **recording boundary itself** (channel i),
not in the exclusion. This also checks the leaver exclusion's scope argument
directly: adding those matches back does not move the forced-pick desirability
distribution.

**`retrieval.py`** — the replay-retrieval censoring, characterized. `collect` records
every replay Valve refuses (`errors/<id>.gone` plus the OpenDota details it already
fetched); `build-dataset` aggregates them into `dataset/gone_matches.jsonl` <!--n retrieval: — {n_gone:,}
matches-->— 2,104
matches<!--/n-->, <!--n retrieval: ~{gone_share:.0%} of discovered-->~2% of discovered<!--/n-->, every one known to OpenDota. Against a seeded retrieved
baseline they are outcome-balanced (win <!--n retrieval: {win_gap_pp:+.1f}pp-->+1.6pp<!--/n-->, <!--n retrieval: z={win_z:+.1f}-->z=+1.3<!--/n-->; no short-game excess), carry
**fewer** leavers <!--n retrieval: ({leaver_gone:.1%} vs {leaver_retrieved:.1%}, z={leaver_z:+.0f}-->(9.6% vs 17.1%, z=−10<!--/n--> — the opposite of abandonment selection), and
concentrated in infrastructure: <!--n retrieval: ~{cluster_top12_share:.0%} sit-->~89% sit<!--/n--> on two replay clusters (<!--n retrieval: {cluster_top1_id}: {cluster_top1_share:.0%}-->413: 72%<!--/n-->,
<!--n retrieval: {cluster_top2_id}: {cluster_top2_share:.0%}-->227: 17%<!--/n-->) whose hosts refused our fetches; the remaining <!--n retrieval: ~{cluster_rest_share:.0%}-->~11%<!--/n--> sit elsewhere.
Cutting across the clusters, OpenDota's `od_parsed` marker shows <!--n retrieval: {od_parsed_share:.0%}-->22%<!--/n--> of the gone
matches demonstrably **existed** — served to OpenDota while fresh, then expired or refused
before our oldest-first sweep — including some on the refusing clusters themselves. The
unparsed remainder sits mostly on those clusters and was plausibly never served to anyone,
though never-served is not provable from here.
**Infrastructure and expiry timing, not behavior.** The never-indexed layer above it
stays unmeasurable, which is where the survivorship gradient points.

**`stats_coverage.py`** — the OpenDota parse-coverage balance behind REPORT.md §6's
stats-available restriction: parsed vs unparsed analytic matches — win
<!--n stats-coverage: {d_win:+.1%}-->-0.6%<!--/n--> <!--n stats-coverage: (z={z_win:+.1f})-->(z=−0.8)<!--/n-->, <!--n stats-coverage: MMR {d_mmr:+.0f} on a ~4000 scale-->MMR +7 on a ~4000 scale<!--/n--> (0.2%), <!--n stats-coverage: forced picks/match {forced_per_match_parsed:.2f} vs {forced_per_match_unparsed:.2f}-->forced picks/match 0.82 vs 0.50<!--/n--> (the
estimand's timeout-scope nudge). The unparsed residual was left unrecovered by
choice (self-parsing the archived replays remains possible); the balance
measurements above are what make that safe.

**`exclusions.py`** — the post-treatment exclusions under one shared logic
(REPORT.md §6's two bias routes), four readouts. [A] composition bounds — the forced pick's
state-centered desirability shift by stratum: swap − non-swap <!--n exclusions: **Δ = {gap_swap:+.3f}
[{gap_swap_lo:+.3f}, {gap_swap_hi:+.3f}]**-->**Δ = +0.004
[-0.002, +0.011]**<!--/n-->; parsed − unparsed (analytic) <!--n exclusions: **Δ = {gap_parsed:+.3f} [{gap_parsed_lo:+.3f}, {gap_parsed_hi:+.3f}]**-->**Δ = −0.003 [−0.011, +0.005]**<!--/n-->.
Under randomization these are *causal* bounds on the treatment→event selection, on
the desirability axis (bounds, never proof of absence — power and axis caveats
apply). [B] consequence — β̂ (composite, rank) on the **val** split with the leaver
exclusion undone: <!--n exclusions: analytic BC {val_bc_analytic:+.3f} / Q {val_q_analytic:+.3f} vs union {val_bc_union:+.3f} / {val_q_union:+.3f}-->analytic BC +0.078 / Q +0.099 vs union +0.081 / +0.093<!--/n-->, and the
leaver-only picks carry a positive signal of their own <!--n exclusions: (BC {val_bc_leaver:+.3f} [{val_bc_leaver_lo:+.3f}, {val_bc_leaver_hi:+.3f}],
Q {val_q_leaver:+.3f} [{val_q_leaver_lo:+.3f}, {val_q_leaver_hi:+.3f}])-->(BC +0.091 [+0.028, +0.153],
Q +0.071 [+0.010, +0.130])<!--/n-->. Undoing the exclusion does not move the estimate.
[C] **differential attrition** — the leaver exclusion conditions on a post-treatment
event and is not somebody else's departure: <!--n exclusions: of {n_leaver_forced:,} forced picks-->of 19,049 forced picks<!--/n--> in leaver
matches, <!--n exclusions: **{leaver_own_share:.1%} belong to the eventual leaver**-->**37.5% belong to the eventual leaver**<!--/n--> <!--n exclusions: ({leaver_own_connected_share:.1%} connected at pick time)-->(95.6% connected at pick time)<!--/n-->, so
the exclusion is clean scoping only if leaving ⫫ pick | state. The direct test
**rejects that, mildly**: the leaver's own forced picks sit at the mechanism
expectation (shift <!--n exclusions: {leaver_own_shift:+.3f} [{leaver_own_shift_lo:+.3f}, {leaver_own_shift_hi:+.3f}]-->−0.000 [−0.005, +0.004]<!--/n-->) while stayers' carry the survivorship
tilt (<!--n exclusions: {stayer_shift:+.3f} [{stayer_shift_lo:+.3f}, {stayer_shift_hi:+.3f}]-->+0.007 [+0.005, +0.008]<!--/n-->) — <!--n exclusions: Δ **{gap_leave:+.3f} [{gap_leave_lo:+.3f}, {gap_leave_hi:+.3f}]** overall-->Δ **-0.007 [-0.011, -0.002]** overall<!--/n-->, <!--n exclusions: **{gap_leave_within:+.3f}
[{gap_leave_within_lo:+.3f}, {gap_leave_within_hi:+.3f}]** across seats-->**−0.009
[−0.015, −0.004]** across seats<!--/n--> within the same leaver matches. Reading: the leave
is the mild outcome of the same abandonment pressure whose severe outcome unrecords
the match (leaver matches carry the *unselected* picks that survived recording
anyway) — consistent with the leaver exclusion being <!--n survivorship: only ~{leaver_channel_share:.0%} of the suppression-->only ~6% of the suppression<!--/n-->
gradient. [D] **ITT-win** — the filter-free check: win is observed for every
match, so the win-β̂ runs with *no post-treatment filter at all* (leavers, swaps,
unparsed retained; bots out — measurement validity). It reproduces the filtered
effects on both splits: <!--n exclusions: test **BC {win_test_bc_itt:+.3f} [{win_test_bc_itt_lo:+.3f}, {win_test_bc_itt_hi:+.3f}]** (filtered {win_test_bc_pub:+.3f})-->test **BC +0.009 [+0.004, +0.014]** (filtered +0.010)<!--/n-->, <!--n exclusions: **Q
{win_test_q_itt:+.3f} [{win_test_q_itt_lo:+.3f}, {win_test_q_itt_hi:+.3f}]** (filtered {win_test_q_pub:+.3f})-->**Q
+0.010 [+0.005, +0.015]** (filtered +0.010)<!--/n-->; val the same pattern. (This test read was added
after the primary analysis plan was fixed — a disclosed amendment:
review-motivated, direction-adversarial, no parameter selected on test. Scoring leaver/swap states is mildly out-of-distribution
for the rankers, which can only attenuate their δ — the ITT read is conservative.)
[E] **swap timing** — §6's "strategy-time trades" evidence, from the raw parse: <!--n exclusions: median
{swap_p50_s:.0f} s after the last draft pick-->median
17 s after the last draft pick<!--/n-->, <!--n exclusions: {swap_within_120s:.1%} within 2 min-->99.6% within 2 min<!--/n-->, <!--n exclusions: max {swap_max_s:.0f} s-->max 144 s<!--/n-->, none before the last
pick, none mid-game (<!--n exclusions: {n_swap_events:,} swaps-->6,612 swaps<!--/n-->). [C] also prints the overall connection rate:
<!--n exclusions: **{connected_rate:.1%}** of all {n_forced_nonbot:,} non-bot forced picks-->**96.6%** of all 85,929 non-bot forced picks<!--/n--> had a connected picker.

**Caveat.** A small residual hero-specific offset (~0.04–0.06 in obs/exp) *may* sit
on top of desirability — at matched popularity heroes trend a touch below abilities
— but it is confounded: the mechanism baseline `P_mech` itself carries the 3-kind hero
residual (<!--n random-mechanism: obs−model {t5_hero_gap_clean:+.3f}-->obs−model -0.020<!--/n-->), so part of any hero-vs-ability gap at matched
desirability is circular, and the binning is coarse. The robust, un-circular finding
is the desirability gradient plus popular-heroes-not-suppressed; whether a genuine
hero-specific term survives is unresolved and second-order.

**`beta_bias.py`** — the downstream: what the survivorship does to the causal test.
β̂ = mean(δ·y); survivorship drops undesirable-item picks, and for a good ranker those
have δ<0, y<0 (so δ·y>0), which means it **attenuates β̂ toward 0** — the observed
effect is *smaller* than the truth, not inflated. Recovering the clean-mechanism β̂ by
reweighting the held-out test picks by 1/survival (the desirability-smoothed suppression,
estimated on train+val only — test never informs its own correction):

| | observed β̂ | corrected β̂ | correction |
|---|---|---|---|
<!--n beta-bias: \| BC \| {beta_baseline_bc:+.3f} \| {beta_corrected_bc:+.3f} \| **{correction_bc:+.3f}** ({atten_bc:+.0%}) \|-->| BC | +0.091 | +0.102 | **+0.011** (+12%) |<!--/n-->
<!--n beta-bias: \| Q \| {beta_baseline_q:+.3f} \| {beta_corrected_q:+.3f} \| **{correction_q:+.3f}** ({atten_q:+.0%}) \|-->| Q | +0.100 | +0.111 | **+0.011** (+11%) |<!--/n-->
<!--n beta-bias: \| Q−BC \| {dqbc_baseline:+.4f} \| {dqbc_corrected:+.4f} \| {dqbc_correction:+.4f} [{dqbc_correction_lo:+.4f}, {dqbc_correction_hi:+.4f}] \|-->| Q−BC | +0.0089 | +0.0088 | -0.0001 [-0.0008, +0.0006] |<!--/n-->

So survivorship attenuates the *absolute* β̂ <!--n beta-bias: by ~{atten_bc:.0%} (the-->by ~12% (the<!--/n--> "orders by causal effect"
claim is conservative — the real effect is larger), and it leaves the **Q−BC
comparison unchanged** (both attenuate equally ⇒ non-inferiority holds). This corrects
the **item-level** selection (missing-at-random given desirability); the script's
sensitivity block then bounds what the correction takes on faith:

- **[S1] suppression-estimate uncertainty** — rerun with the measured gradient removed
  (w⁰), as measured (w¹), and doubled (w²): BC <!--n beta-bias: {s1_bc_l0:+.3f} / {s1_bc_l1:+.3f} / {s1_bc_l2:+.3f}-->+0.091 / +0.102 / +0.112<!--/n-->, Q <!--n beta-bias: {s1_q_l0:+.3f} /
  {s1_q_l1:+.3f} / {s1_q_l2:+.3f}-->+0.100 /
  +0.111 / +0.121<!--/n-->, Q−BC pinned at <!--n beta-bias: {s1_dqbc_l1:+.3f} throughout-->+0.009 throughout<!--/n-->. Mis-estimating the gradient
  rescales the <!--n beta-bias: ~{atten_bc:.0%} correction-->~12% correction<!--/n-->; it cannot flip a verdict.
- **[S2] outcome-level MNAR** — players abandoning on their *realized outcome* beyond
  the item is invisible to the item axis. Model it as P(record | y) ∝ exp(γ·ỹ) and
  correct with u = w·exp(−γ·ỹ). The measured channels calibrate the plausible band:
  the item-level gradient itself sets the scale (its top-vs-bottom log-suppression
  spread against the corresponding outcome spread calibrates
  <!--n beta-bias: γ to ≈ **{gamma_band:.2f}** per-->γ to ≈ **0.11** per<!--/n-->
  composite unit — a stated calibration, fixed in the script), and undoing the
  leaver exclusion moves β̂ ≲ 0.01. Within |γ| ≤ 0.11,
  β̂_BC stays in <!--n beta-bias: [{s2_bc_min:+.2f}, {s2_bc_max:+.2f}] (ESS ≥ {s2_ess_min_band:.0%})-->[+0.10, +0.19] (ESS ≥ 69%)<!--/n-->; the **tipping point** where β̂_BC would
  reach 0 is <!--n beta-bias: \|γ\| ≈ **{tip_gamma:.2f} — ~{tip_gamma_ratio:.0f}× the calibrated band**-->|γ| ≈ **inf — ~inf× the calibrated band**<!--/n--> — and at the band edge γ=+0.11
  the match-clustered <!--n beta-bias: CI is {band_edge_beta:+.3f} [{band_edge_beta_lo:+.3f}, {band_edge_beta_hi:+.3f}]-->CI is +0.104 [+0.061, +0.146]<!--/n-->. The Q−BC gap stays within <!--n beta-bias: ±{s2_dqbc_absmax_band:.2f}-->±0.02<!--/n-->
  across the band (point estimates; at the worst edge, γ=−0.11, Q retains <!--n beta-bias: {s2_edge_q_retention:.0%} of BC-->97% of BC<!--/n--> —
  inside the κ=25% margin, though no CI is attached to these edge reads). Beyond ~2×
  the band the reweighting itself degenerates (ESS <!--n beta-bias: {s2_ess_gm02:.0%} at-->8% at<!--/n--> γ=−0.2), so the tipping
  regime is not one the data can even express.
- **[S3] the unresolved hero-specific offset** — an extra ×0.95 survival on the <!--n beta-bias: {n_hero_forced:,}-->3,179<!--/n-->
  hero forced picks moves β̂ by <!--n beta-bias: {s3_delta_bc:+.4f}-->+0.0001<!--/n-->. Negligible.

Residual caveat: the S2 model is single-parameter exponential tilting on the composite;
selection patterned on something orthogonal to both item desirability and ỹ is outside
all three axes (nothing measured hints at one).

## Run

```
DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/run.py
DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/run.py --quick   # skip the T5 winrate pass
DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/survivorship.py  # survivorship magnitude + channel decomposition
DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/retrieval.py     # retrieval-censoring balance (reads dataset/gone_matches.jsonl)
DOTA2AD_ROOT=work pixi run python experiments/random-mechanism/stats_coverage.py # OpenDota parse-coverage balance
DOTA2AD_ROOT=work pixi run -e cuda python experiments/random-mechanism/exclusions.py # exclusions: composition bounds, leaver-undone β̂, differential attrition, ITT-win
DOTA2AD_ROOT=work pixi run -e cuda python experiments/random-mechanism/beta_bias.py  # β̂-bias (loads BC/Q)
```

`run.py`/`survivorship.py` are CPU-only; `run.py` runs on the full corpus
(`exclude=()`) since the mechanism is match-type-independent, `survivorship.py` uses
the stats-parsed (analytic/clean) subset. `beta_bias.py` needs the **cuda** env (it
loads BC + Q for the held-out-test β̂).
