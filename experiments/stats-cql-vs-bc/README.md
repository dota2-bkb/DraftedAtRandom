# stats-cql-vs-bc

**What:** recommender **diagnostics** — *not* the primary causal test (that is
`stats-causal-rank`, where the answer is: Q is non-inferior to BC, edge unresolved). Three
checks on the validation split's random picks: (1) a Δrank regression *slope* — a
variance-normalized cousin of the causal test, kept for the ablation; (2) BC's ranking
lift with the focal MMR conditioned low→high; (3) the Q-vs-BC@MMR agreement gradient.
Plus `--ablate` (de-cloning: does removing the conservatism recover an edge? — no) and
`--power` (data needed to resolve the T1 slope).

**Run:**
```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-cql-vs-bc/run.py [--ablate|--power|--adjust]
```

**Result:** T1 slope <!--n stats-cql-vs-bc: {t1_slope:+.2f} [{t1_slope_lo:+.2f}, {t1_slope_hi:+.2f}]-->+0.24 [-0.09, +0.58]<!--/n--> ns (Q's deviations from BC ≈ noise on the
realized composite, <!--n stats-cql-vs-bc: R² ≈ {t1_r2:.0f}-->R² ≈ 0<!--/n--> — the design-based β̂ reports the underlying covariance directly);
T2: BC's Q1−Q4 stat lift is <!--n stats-cql-vs-bc: ≈ {t2_lift_z0:+.1f}z-->≈ +1.0z<!--/n--> at *every* conditioned MMR (flat across the knob —
consistent with the imported ladder rank being a weak AD-skill proxy; see
`stats-skill-headroom` for the skill axis that does measure something); T3 flat
(<!--n stats-cql-vs-bc: ρ(Q, BC@MMR) ≈ {t3_rho_z0:.2f}-->ρ(Q, BC@MMR) ≈ 0.74<!--/n--> at all conditioned levels — Q carries no skillward gradient). The
de-cloning ablation finds no variant beating BC. The headline Q-vs-BC result (Q
non-inferior to BC, edge unresolved) lives in `stats-causal-rank`.
