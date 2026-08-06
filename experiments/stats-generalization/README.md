# stats-generalization

**What:** transportability & robustness for the causal test. Random picks are a
selected (timeout) subgroup, so the effect is *local*; this bounds the gap to the
deliberate/deployment population using the **same β̂** as
`stats-causal-rank`, stratified. Four analyses: covariate shift (SMD on focal MMR,
draft turn), effect homogeneity (β̂ by MMR / draft-phase / feasibility strata),
placebos (P1 exogeneity, P2 permuted, P3 specificity), and a conditioned-skill
headroom test (β̂ re-scored with the focal seat's MMR overridden low→high — is a
higher-skill ranker causally better?). Reading: [the report](../../REPORT.md)
§7.

**Run:**
```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-generalization/run.py
```

**Result** (held-out test, <!--n stats-generalization: n={n_picks:,})-->n=12,310)<!--/n-->: shift small (<!--n stats-generalization: MMR SMD {smd_mmr:+.2f}, turn {smd_turn:+.2f}-->MMR SMD −0.03, turn +0.10<!--/n-->); β̂(Q)
homogeneous (mean β̂ ≈ <!--n stats-generalization: {perstat_strata_min:+.3f}–{perstat_strata_max:+.3f}-->+0.009–+0.012<!--/n--> across all strata, overlapping CIs); P1 ρ(Q-rank,
pre-pick covariate) ≈ 0 (<!--n stats-generalization: MMR {p1_rho_mmr:+.3f}, turn {p1_rho_turn:+.3f}-->MMR +0.005, turn -0.006<!--/n-->); P2 permuted <!--n stats-generalization: {p2_permuted_sig}/26 (vs Q's {p2_q_sig}/26)-->3/26 (vs Q's 21/26)<!--/n-->. P3 is
**uninformative** here: <!--n stats-generalization: diagonal {p3_diag:+.4f} vs \|off-diagonal\| {p3_offdiag:.4f}-->diagonal +0.0106 vs |off-diagonal| 0.0083<!--/n--> — the per-stat rankings
and the stats themselves co-move too strongly for the design to separate stat-specific
effects from a shared quality axis. Conditioned skill is **flat** — composite β̂(BC)
<!--n stats-generalization: {cond_bc_low:+.3f} (low) → {cond_bc_high:+.3f} (high)-->+0.091 (low) → +0.091 (high)<!--/n-->, paired BC@high−BC@low <!--n stats-generalization: {cond_dbeta_bc_high_low:+.4f} [{cond_dbeta_bc_high_low_lo:+.4f}, {cond_dbeta_bc_high_low_hi:+.4f}]-->+0.0006 [-0.0019, +0.0028]<!--/n--> ns;
steering to high opens no headroom (BC@high−Q@own <!--n stats-generalization: {cond_dbeta_bchigh_qown:+.4f} [{cond_dbeta_bchigh_qown_lo:+.4f}, {cond_dbeta_bchigh_qown_hi:+.4f}]-->-0.0087 [-0.0295, +0.0136]<!--/n--> ns).
**Caveat (do not read this as skill-invariance):** the conditioned variable is the
*general* ranked medal, which barely predicts AD outcomes (team-rank→win <!--n premise: AUC ≈ {auc_win:.2f}-->AUC ≈ 0.51<!--/n-->,
and it is only weakly related to the model's compressed `computed_mmr`
input). A flat response to a knob that does not measure AD skill is
uninformative about AD-skill headroom — the headroom question is settled separately, on a
stats-based skill axis (see REPORT.md §6): a leak-free play-skill rating
(residual vs the StatsModel — moderate reliability, stratified <!--n rating: split-half ≈ {sb_all_ge4:.2f}/{sb_all_ge8:.2f},
out-of-sample ≈ {sb_val_ge4:.1f}–{sb_val_ge8:.1f}-->split-half ≈ 0.57/0.68,
out-of-sample ≈ 0.4–0.5<!--/n-->) fed to a retrained BC leaves β̂ flat
low→high (<!--n stats-skill-headroom: Δβ̂ = {dbeta_high_low:+.3f}, CI [{dbeta_high_low_lo:+.3f}, {dbeta_high_low_hi:+.3f}]-->Δβ̂ = +0.001, CI [-0.002, +0.004]<!--/n-->, BC demonstrably responsive), so skilled
players draft somewhat differently but not detectably better through this
moderate-reliability label — no offline headroom detected. The
transport story otherwise holds: local effect, small placebo-checked observable gap; a
definitive close needs deployment A/B.
