# stats-skill-headroom

**What:** is there offline headroom above the human-consensus recommender — a
*stronger drafter* we could imitate? Answered on the outcome that matters (stats),
using the same β̂ as `stats-causal-rank`. Result: **none detected.** Skilled players
draft a little *differently* but not *detectably better* through a
moderate-reliability skill label; the measurable edge is execution, not draft
choice. Reading: [the report](../../REPORT.md) §6.

**The chain:**
1. **Win can't label good drafters.** A chronological AD-Elo over win/loss (103k
   accounts, real IDs on every seat) predicts *future* AD wins <!--n premise: at AUC ≈ {elo_auc_ge5:.2f} — flat-->at AUC ≈ 0.51 — flat<!--/n--> —
   while the identical machinery <!--n premise: reaches ≈ {elo_ctl_auc_ge5:.2f} on synthetic-->reaches ≈ 0.64 on synthetic<!--/n--> skill-driven games
   (positive control). AD outcomes are noise-dominated; win is the wrong axis.
2. **Stats can.** A player's over/under-performance vs what their draft predicts (a
   leak-free residual against the StatsModel) is a repeatable trait of moderate
   reliability <!--n rating: (stratified split-half ≈ {sb_all_ge4:.2f}/{sb_all_ge8:.2f}, out-of-sample ≈ {sb_val_ge4:.1f}–{sb_val_ge8:.1f}; raw-->(stratified split-half ≈ 0.57/0.68, out-of-sample ≈ 0.4–0.5; raw<!--/n-->
   win ≈ 0).
3. **Conditioning on it doesn't help.** Feed that play-skill rating to a retrained BC
   and override focal skill low→high: β̂ is flat <!--n stats-skill-headroom: (Δβ̂ = {dbeta_high_low:+.3f}, 95% CI [{dbeta_high_low_lo:+.3f},
   {dbeta_high_low_hi:+.3f}])-->(Δβ̂ = +0.001, 95% CI [-0.002,
   +0.004])<!--/n--> even though BC demonstrably shifts its picks <!--n stats-skill-headroom: (mean TV ≈ {mean_tv:.3f})-->(mean TV ≈ 0.017)<!--/n-->.

**Run** (each stage caches to `<root>/.cache/skill-headroom/`):
```
# 1. premise: win is noise, stats-skill is stable (CPU; ~6min to build the account cache)
DOTA2AD_ROOT=work pixi run python experiments/stats-skill-headroom/premise.py

# 2. build the leak-free rating + the variant dataset (GPU: StatsModel residual pass)
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-skill-headroom/rating.py
#    → writes work_skill/ (matches.jsonl with mmr := rating; rest symlinked)

# 3. retrain BC on the variant (warm-started from the shipped BC — only the mmr
#    pathway needs to relearn general-rank → skill; ~1hr)
DOTA2AD_ROOT=work_skill pixi run -e cuda python experiments/train-policy/run.py \
    --init-from work/models/policy.pt --epochs 25

# 4. the headroom readout: β̂(BC | focal skill = z), paired high−low
DOTA2AD_ROOT=work_skill pixi run -e cuda python experiments/stats-skill-headroom/headroom.py
```

**Result** (held-out test, <!--n stats-skill-headroom: n≈{n_picks:,} random picks-->n≈12,310 random picks<!--/n-->): <!--n premise: win-Elo AUC ≈ {elo_auc_ge5:.2f} (control {elo_ctl_auc_ge5:.2f})-->win-Elo AUC ≈ 0.51 (control 0.64)<!--/n-->;
stats-skill SB <!--n rating: reliability ≈ {sb_all_ge4:.2f} (≥4 games) / {sb_all_ge8:.2f} (≥8)-->reliability ≈ 0.57 (≥4 games) / 0.68 (≥8)<!--/n-->; headroom <!--n stats-skill-headroom: **Δβ̂(high−low) =
{dbeta_high_low:+.3f} [{dbeta_high_low_lo:+.3f}, {dbeta_high_low_hi:+.3f}]**-->**Δβ̂(high−low) =
+0.001 [-0.002, +0.004]**<!--/n--> — within <!--n stats-skill-headroom: ±{bound_pct_of_bc:.1f}%-->±4.6%<!--/n--> of BC's own effect <!--n stats-causal-rank: (β̂_BC={beta_bc:+.3f})-->(β̂_BC=+0.091)<!--/n-->, clearing
**tight equivalence** (<!--n stats-skill-headroom: κ ≈ {bound_pct_of_bc:.1f}%-->κ ≈ 4.6%<!--/n-->); and since <!--n stats-skill-headroom: BC-response TV ≈ {mean_tv:.3f}-->BC-response TV ≈ 0.017<!--/n--> shows the model *does* shift
its picks on the skill input, the flat β̂ means skill-conditioning changes *which* abilities are
drafted without a detectable change in their causal quality. No offline headroom detected —
human consensus (BC) is the operative target; a real edge needs a live A/B.
