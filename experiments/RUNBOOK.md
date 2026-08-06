# Runbook — reproducing results

**`work/`** (<!--n dataset: {n_raw:,} raw matches-->102,096 raw matches<!--/n-->, patch 7.41; <!--n dataset: {n_analytic:,} analytic-->79,240 analytic<!--/n--> after excluding bots/leavers/swaps
— see REPORT.md §6) is the work-dir of record —
self-contained: `dataset/`, `parsed/`, `models/`, `logs/`. Training stages are
guarded (`[ -f models/<name>.pt ]`), so a re-run skips the finished models and
just regenerates the evals; delete a model (or use a fresh root) to force a
retrain.

The match split (`dataset/split.json`, written by `build_dataset`) is three-way:
models **fit on train**; every **selection** decision — training epochs, the
stats-DQN's checkpoint metric, Trial's early stopping, the reweight β sweep,
calibration, diagnostics — reads the **validation** split; the **test** split is
read only by the final evaluations (stage 2 of `run_all.sh`).

## Run everything

```
ROOT=work bash experiments/run_all.sh      # re-run against the data of record
ROOT=work_new bash experiments/run_all.sh     # fresh root (needs its own dataset/ + parsed/)
```

The script runs, in dependency order: the foundational models (policy → stats →
stats-dqn), the probe models (Trial, the
skill-rating variant and its warm-started BC), then the evaluation suite —
final reads on test (stats-causal-rank, cluster-sensitivity, static-rank,
naive-bias, stats-generalization,
recommender-value + reweight + trial/awr readouts, skill-headroom, beta-bias) and
diagnostics on val (stats-cql-vs-bc, density-validate), plus the mechanism annexes.
Logs are unbuffered and tee'd to `$ROOT/logs/<name>.log`
(tail one for live progress). It uses the **cuda** pixi env for GPU steps.

## Run one experiment

Every experiment has a pixi task (see `pyproject.toml`) and a direct form:

```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/stats-causal-rank/run.py
DOTA2AD_ROOT=work pixi run stats-causal-rank      # via pixi task
```

See each `experiments/<name>/README.md` for purpose, flags, and expected result,
and the top-level [`REPORT.md`](../REPORT.md) for the full story.

## Notes
- `pixi run collect` fetches replays into `parsed/`; a permanently unserved replay
  leaves `parsed/errors/<id>.gone` plus the OpenDota details already in hand, and
  `build-dataset` aggregates those into `dataset/gone_matches.jsonl` — the
  retrieval-censoring record that `random-mechanism/retrieval.py` characterizes.
- The final evaluations run on the **held-out test split** — the
  StatsModel/Q are trained on train completions and selected on val, so in-sample or
  selection-set tuples would inflate the signal via memorization and checkpoint
  selection.
- `stats-causal-rank` is the core causal test: it scores each forced pick's
  decision-time state with the rankers (BC/Q). A few
  minutes to run; `--bootstrap` / `--subset` flags available.
- Re-running is safe; training overwrites its own checkpoint. The stats-DQN
  trainer's `--output` defaults to `models/stats_dqn.pt.mc`, so `run_all.sh`
  passes `--output models/stats_dqn.pt` to write the live name.
