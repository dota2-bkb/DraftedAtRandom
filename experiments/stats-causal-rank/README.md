# stats-causal-rank

**What:** the core causal test. At each server-randomized (timeout) pick, a ranker
scores the legal abilities at the decision-time draft state, and we measure whether its
preference for the forced ability predicts the realized outcome —
`β̂ = mean_i w_i · δ(s_i, A_i) · ỹ_i`. Run
on **BC** (human baseline), **Q** (recommender), and a **scrambled control**, in two
transforms (rank / score), per stat + composite + win, on the **test split** (the
single held-out read). Method and derivation:
[the report](../../REPORT.md) §5 and Appendix A.

**Run:**
```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-causal-rank/run.py
```
Flags: `--subset {online,disconnect}`, `--bootstrap N`, `--ni-margin κ`.

**Result** (held-out test, <!--n stats-causal-rank: n={n_picks:,} picks / {n_matches:,} matches-->n=12,310 picks / 6,834 matches<!--/n-->): the **primary** endpoint —
fixed, with the full analysis plan, before the test split's single read — is the composite
β̂ (rank) — <!--n stats-causal-rank: BC **{beta_bc:+.3f}** [{beta_bc_lo:+.3f},{beta_bc_hi:+.3f}]-->BC **+0.091** [+0.063,+0.120]<!--/n-->, <!--n stats-causal-rank: Q **{beta_q:+.3f}**
[{beta_q_lo:+.3f},{beta_q_hi:+.3f}]-->Q **+0.100**
[+0.070,+0.128]<!--/n-->, <!--n stats-causal-rank: scrambled {beta_perm:+.3f} [{beta_perm_lo:+.3f},{beta_perm_hi:+.3f}]-->scrambled -0.026 [-0.054,+0.002]<!--/n-->. Win (secondary): <!--n stats-causal-rank: BC {win_bc:+.3f}-->BC +0.010<!--/n-->*, <!--n stats-causal-rank: Q {win_q:+.3f}-->Q +0.010<!--/n-->*.
The per-stat battery is **secondary**, under BH-FDR 5%, and prints three columns per stat —
Q's stat-specific head (Q_k), Q's deployed composite ordering (Qc, the symmetric counterpart
of BC's single ordering), and BC: deployed-ordering head-to-head <!--n stats-causal-rank: **Qc {perstat_bhfdr_qc}/26 vs BC {perstat_bhfdr_bc}/26**-->**Qc 19/26 vs BC 18/26**<!--/n-->;
Q's specialized <!--n stats-causal-rank: heads {perstat_bhfdr_q_heads}/26-->heads 19/26<!--/n-->; <!--n stats-causal-rank: control {perstat_bhfdr_perm}/26-->control 0/26<!--/n--> (Benjamini–Yekutieli sensitivity, printed
alongside: <!--n stats-causal-rank: Qc {perstat_byfdr_qc}/26, BC {perstat_byfdr_bc}/26-->Qc 14/26, BC 14/26<!--/n-->). <!--n stats-causal-rank: **Q − BC = {dbeta:+.3f} [{dbeta_lo:+.3f},{dbeta_hi:+.3f}]**-->**Q − BC = +0.009 [-0.013,+0.029]**<!--/n--> (rank, shipped checkpoint), assessed as
**preservation of effect** against BC's *own* ranking skill: <!--n stats-causal-rank: ρ = {rho:.0%} [{rho_lo:.0%},{rho_hi:.0%}]-->ρ = 109% [87%,137%]<!--/n--> —
non-inferior at the default κ=25% tolerance (the run prints the NI verdict). The
`cluster_sensitivity.py` checks the bootstrap's unit: focal-account clustering leaves the CI
widths essentially unchanged (BC identical, <!--n cluster-sensitivity: Q ≈{q_account_widening:+.0%}-->Q ≈-1%<!--/n-->), day blocks no wider — match clustering
stands. Training-seed
robustness is left as an on-demand rerun (the training scripts' `--seed` flags +
`run.py --q-ckpt` to score an alternative checkpoint). The score
transform is noisier <!--n stats-causal-rank: (BC {score_beta_bc:+.3f}, Q {score_beta_q:+.3f}; Q − BC = {score_dbeta:+.3f} [{score_dbeta_lo:+.3f},{score_dbeta_hi:+.3f}])-->(BC +0.308, Q +0.266; Q − BC = -0.042 [-0.140,+0.053])<!--/n-->; rank is primary.
See REPORT.md §5–6 for the margin logic.

**Raw tables:** `static_rank.py` scores the community-practice reference rankers
(popularity, win-rate, pair-synergy; REPORT.md §6) and writes the raw tables behind
them — unshrunk counts over train-split matches — to `work/results/tables/`:
[`popularity.csv`](../../work/results/tables/popularity.csv) (sorted by pick rate) and
[`winrate.csv`](../../work/results/tables/winrate.csv) (sorted by win rate) render as
searchable tables on GitHub;
[`pair_winrate.csv`](../../work/results/tables/pair_winrate.csv) (~13 MB, 185k pairs,
most-drafted first) is too large for GitHub's viewer — load it with pandas/DuckDB.
`--tables-only` rebuilds just those files.
