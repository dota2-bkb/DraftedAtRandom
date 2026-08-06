# train-stats

Trains the StatsModel: completed draft → per-player game-end stat vector. It
provides the stats-DQN's Monte-Carlo reward targets and its encoder warm-starts the
stats-DQN — the recommender Q is built on it. (It is a training component, not itself
the object of the causal test — that is `stats-causal-rank`.) The model includes
a per-spell damage head.

## Run
```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/train-stats/run.py
```

## Design note
Targets are z-normalized. The per-spell damage head exists because per-spell
damage has per-pick ground truth (OpenDota's damage attribution) where aggregate
damage does not — densify only where attribution exists. Outputs feed
`STAT_SPECS` / the composite rankers.

See the top-level [`REPORT.md`](../../REPORT.md) for the full story.
