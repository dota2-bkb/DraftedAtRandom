# stats-recommender-value

**What:** how good is the recommender — vs what individuals actually do, vs the best
ranker/policy fittable from the causal data, and as a deployable BC+Trial blend? Four
probes, all on the forced-random natural experiment, at the decision-time state;
single test-split readouts, selection (sweeps, early stopping) on **val**.
Reading: [the report](../../REPORT.md) §6.

- **`run.py` — vs a typical trajectory.** BC is the *average* human pick; deploying its
  *mode* would cancel the noise individuals add if value tracks popularity. Value pick policies
  by realized causal effect: RANDOM legal → TYPICAL human (draw ~ BC) → RECOMMENDER top pick,
  for the consensus mode (argmax BC) and each *deployed* ranker (argmax Q, argmax Trial) —
  the deployed policies measured directly, so their value doesn't ride on β̂ / on the §5 Q–BC
  comparison.
- **`trial.py` — vs the best fittable *ranker* (β̂).** A *state-aware* value ranker trained on
  *only* the randomized picks (a value head on BC's frozen encoder + unconfounded outcome
  regression, early-stopped on the val split's forced picks), with a `--shuffle-y` negative
  control and `--seed` robustness. Compare β̂ to BC: a robust win = fittable room; a
  lean-but-unresolved = data-limited. A true ceiling isn't identifiable offline.
- **`trial_awr.py` — vs the best fittable *policy* (V).** Directly maximizes policy value on
  the random subset via advantage-weighted regression (uniform base — *no* BC prior), with a
  temperature and an ensemble-LCB (both BC-free). The *V-room* question: can a policy built from
  the randoms *alone* beat BC's value? `--unfreeze` (train the encoder) / `--shuffle-y` robustness.
- **`reweight_bc.py` — the deployable recommender.** BC prior tilted by
  Trial's causal value, `π ∝ π_BC · exp(β·v̂)` — a downside-protected inference-time blend
  (β=0 = BC). Sweeps β on val; reads β=1 (the unit tilt, fixed in advance) on test.

**Run:**
```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/run.py
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/trial.py
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/trial_awr.py
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-recommender-value/reweight_bc.py
```

**Results** (held-out test, <!--n stats-recommender-value: n={n_picks:,} random picks-->n=12,310 random picks<!--/n-->; outcome = z-composite, <!--n stats-recommender-value: sd {sd_composite:.2f}-->sd 4.58<!--/n-->):
- `run.py` (exact importance-weighted; TYPICAL = literal draw from BC, RECOMMENDER = literal
  argmax of the named ranker): value-vs-BC-percentile curve monotone. Per single pick,
  <!--n stats-recommender-value: **TYPICAL − RANDOM = {typical_minus_random:+.3f} [{typical_minus_random_lo:+.3f}, {typical_minus_random_hi:+.3f}]**-->**TYPICAL − RANDOM = +0.507 [+0.334, +0.676]**<!--/n--> (humans beat chance, about a tenth of an SD). The
  top-pick contrasts form the crowd-wisdom family (BH-FDR 5%): argmax BC − TYPICAL <!--n stats-recommender-value: **{bcmode_minus_typical:+.3f}
  [{bcmode_minus_typical_lo:+.3f}, {bcmode_minus_typical_hi:+.3f}]** (p={bcmode_minus_typical_p:.2f})-->**+0.313
  [-0.098, +0.709]** (p=0.13)<!--/n-->, argmax Q − TYPICAL <!--n stats-recommender-value: **{q_minus_typical:+.3f} [{q_minus_typical_lo:+.3f}, {q_minus_typical_hi:+.3f}]** (p={q_minus_typical_p:.3f})-->**+0.277 [-0.238, +0.769]** (p=0.286)<!--/n-->,
  argmax Trial − TYPICAL <!--n stats-recommender-value: **{trial_minus_typical:+.3f} [{trial_minus_typical_lo:+.3f}, {trial_minus_typical_hi:+.3f}]** (p={trial_minus_typical_p:.3f})-->**+0.263 [-0.226, +0.739]** (p=0.290)<!--/n--> — **0/3 significant** (all p ≥ 0.13): no deployed
  recommender separates from a typical human at this n (a deterministic top pick is observed on
  only <!--n stats-recommender-value: ~{support_q}-->~910<!--/n--> of the ~12k forced draws, with an effective sample of
  <!--n stats-recommender-value: ≈{ess_q:.0f}-->≈364<!--/n--> after weighting). argmax Q coincides with argmax BC <!--n stats-recommender-value: on ~{agree_q_bc:.0%} of decisions
  (Trial ~{agree_trial_bc:.0%})-->on ~29% of decisions
  (Trial ~22%)<!--/n-->; their own difference <!--n stats-recommender-value: {q_minus_trial:+.3f} [{q_minus_trial_lo:+.3f}, {q_minus_trial_hi:+.3f}]-->+0.015 [-0.649, +0.687]<!--/n--> is unresolved, consistent with §5.
  (Faithfulness of "draw from BC" ≈ trajectory rests on BC's held-out calibration, <!--n train-policy: ECE ≈ {ece:.2f}.)-->ECE ≈ 0.01.)<!--/n-->
- `trial.py` (state-aware value net on the random subset; shuffled control):
  <!--n trial: **β̂(net) = {beta_trial:+.3f} [{beta_trial_lo:+.3f}, {beta_trial_hi:+.3f}]**-->**β̂(net) = +0.085 [+0.060, +0.113]**<!--/n--> — control-verified (shuffled outcomes → <!--n trial-shuffle: β̂ ≈ {beta_shuffled:+.2f}-->β̂ ≈ -0.00<!--/n--> ns)
  — <!--n trial: vs **β̂(BC) = {beta_bc:+.3f}**-->vs **β̂(BC) = +0.091**<!--/n-->. **Δ(net − BC) = <!--n trial: {dbeta_vs_bc:+.3f} [{dbeta_vs_bc_lo:+.3f}, {dbeta_vs_bc_hi:+.3f}]-->-0.006 [-0.037, +0.024]<!--/n-->**,
  straddling 0: a clean state-aware causal fit lands at BC's level — no
  resolvable edge in either direction at ~12k picks.
- `trial_awr.py` (direct-V policy, 4 members, frozen & `--unfreeze`): **fails.** As a ranker it
  is significantly worse than BC (<!--n trial-awr: Δβ̂ = {dbeta_vs_bc:+.3f} [{dbeta_vs_bc_lo:+.3f}, {dbeta_vs_bc_hi:+.3f}]-->Δβ̂ = -0.044 [-0.079, -0.009]<!--/n-->); its *sampled* policy sits
  significantly below TYPICAL <!--n trial-awr: ({soft_minus_typical:+.2f} [{soft_minus_typical_lo:+.2f}, {soft_minus_typical_hi:+.2f}])-->(-0.37 [-0.53, -0.19])<!--/n-->; its LCB argmax merely matches BC's mode.
  A policy from the randoms *alone* can't beat BC — the ~37k picks *correct* the prior, they
  don't *replace* it. (Not a V-ceiling but a loose lower bound: V(argmax) has no well-behaved
  estimator here, unlike β̂.)
- `reweight_bc.py` (BC · exp(β·v̂_Trial), inference-only): the **deployable** answer. The β
  sweep and its selection diagnostics run on val — the grid <!--n reweight-bc: V-peak sits at β ≈ {val_peak_beta:.1f}-->V-peak sits at β ≈ 0.5<!--/n-->, and
  re-selecting β inside a match-resampled bootstrap moves the − TYPICAL gap <!--n reweight-bc: by ≈ {sel_vs_fix_shift:+.2f}-->by ≈ +0.20<!--/n--> (the grid
  lands on β=1 in <!--n reweight-bc: only ~{sel_beta1_rate:.0%} of resamples-->only ~11% of resamples<!--/n-->), so a
  grid-chosen β would carry a winner's-curse premium — while the **unit
  tilt β=1** (fixed in advance) gets the one test read: <!--n reweight-bc: **− TYPICAL {gap_typical:+.3f} [{gap_typical_lo:+.3f}, {gap_typical_hi:+.3f}]**, − BC-mode {gap_bcmode:+.3f}
  [{gap_bcmode_lo:+.3f}, {gap_bcmode_hi:+.3f}], − argmax Q {gap_q:+.3f} [{gap_q_lo:+.3f}, {gap_q_hi:+.3f}]-->**− TYPICAL +0.747 [+0.281, +1.201]**, − BC-mode +0.433
  [-0.086, +0.966], − argmax Q +0.469 [-0.179, +1.116]<!--/n--> (indistinguishable from the shipped Q).
  **EXPLORATORY**: the tilt leans on Trial's (unresolved) causal value and rests on the argmax
  reading, the noisiest in the report. Soft-V rises with β before plateauing on val. Human prior +
  causal correction ≈ Q, transparently.

These are single-pick effects in a noisy 50-min game — real and significant where marked,
but small, not naively multipliable across a draft; cumulative/deployed value needs A/B.
