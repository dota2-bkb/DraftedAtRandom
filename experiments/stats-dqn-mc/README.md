# stats-dqn-mc

**What:** trains **Q**, the shipped recommender — a per-action stat-vector Q-net via
**Monte-Carlo returns** (not TD bootstrap) + **uniform-support CQL** + a hard
**BC-plausibility mask**, warm-started from BC. At inference a user-preference weight
vector reduces the per-stat Q to a composite score (balanced / kill / farm / support
/ push). It escapes three known failure modes: TD across-action collapse (MC fixes
it — `--n-step 1` reproduces the collapse as a demo), standard CQL's BC-cloning
(`usup` is the popularity-free replacement), and Goodhart rare-pick inflation (the
mask + `usup`). Its causal evaluation is `stats-causal-rank`
([the report](../../REPORT.md) §5–6): **Q is non-inferior to BC (no harm; a modest edge stays unresolved)**.

**Run:**
```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-dqn-mc/run.py
```
Flags: `--n-step 1` (TD-collapse demo), `--focal-continuation {policy,bc}`.

**Output:** `models/stats_dqn.pt`.
