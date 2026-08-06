#!/usr/bin/env bash
# Reproduce all model artifacts + experiment results into a work-dir.
#
# Default root is work/ (the data of record; training stages skip if the model
# already exists). Override:  ROOT=work_new bash experiments/run_all.sh
#
# Logs are unbuffered (python -u) and tee'd to $ROOT/logs/<name>.log so a tail
# shows live progress. GPU steps use the cuda pixi env.
#
# Split roles (dataset/split.json): fit on train; select (best epoch, checkpoint
# metric, early stop, β sweep, calibration, diagnostics) on val; final
# numbers on test only — stage 2 is the test read.
set -euo pipefail

ROOT="${ROOT:-work}"
export DOTA2AD_ROOT="$ROOT"
mkdir -p "$ROOT/logs"
echo "DOTA2AD_ROOT=$ROOT"

run() {  # run <log-name> <experiment-args...>
  local name="$1"; shift
  echo ">>> $name"
  pixi run -e cuda python -u "$@" 2>&1 | tee "$ROOT/logs/$name.log"
}

# ---- 1. Foundational models (order matters; skip if artifact exists → resumable) ----
# Models are written only on successful completion, so an existing artifact means
# that stage finished — a relaunch resumes at the first incomplete stage.
[ -f "$ROOT/models/policy.pt" ]      || run train-policy    experiments/train-policy/run.py                     # → policy.pt (V+1)
[ -f "$ROOT/models/match_stats.pt" ] || run train-stats     experiments/train-stats/run.py                      # → match_stats.pt
[ -f "$ROOT/models/stats_dqn.pt" ]   || run train-stats-dqn experiments/stats-dqn-mc/run.py --output "$ROOT/models/stats_dqn.pt"  # MC + usup+bc-mask, cql off

# ---- 1b. Probe models ----
# Trial — state-aware value ranker on the random subset (also prints its test β̂)
[ -f "$ROOT/models/trial.pt" ] || run trial experiments/stats-recommender-value/trial.py --save "$ROOT/models/trial.pt"
# Skill variant: play-skill rating → ${ROOT}_skill (matches.jsonl with mmr := rating),
# then a skill-conditioned BC warm-started from the shipped BC.
[ -f "${ROOT}_skill/dataset/matches.jsonl" ] || run skill-rating experiments/stats-skill-headroom/rating.py
[ -f "${ROOT}_skill/models/policy.pt" ] || (
  export DOTA2AD_ROOT="${ROOT}_skill"
  run skill-train-policy experiments/train-policy/run.py --init-from "$ROOT/models/policy.pt" --epochs 25
)

# ---- 2. Causal evaluation (final read on TEST; diagnostics on val) ----
run stats-causal-rank      experiments/stats-causal-rank/run.py            # β̂(BC/Q/permuted)
run cluster-sensitivity    experiments/stats-causal-rank/cluster_sensitivity.py  # bootstrap unit check (account/day)
run static-rank            experiments/stats-causal-rank/static_rank.py    # community-reference rankers: popularity, win-rate, pair-synergy (CPU)
run naive-bias             experiments/stats-causal-rank/naive_bias.py     # confounded estimate on deliberate picks
run stats-generalization   experiments/stats-generalization/run.py         # transportability + placebos
run recommender-value      experiments/stats-recommender-value/run.py      # crowd-wisdom V contrasts
run recommender-value-reweight experiments/stats-recommender-value/reweight_bc.py  # BC·exp(β·v̂): sweep on val, β=1 on test
run trial-shuffle          experiments/stats-recommender-value/trial.py --shuffle-y   # negative control (β̂ ≈ 0)
run trial-awr              experiments/stats-recommender-value/trial_awr.py           # direct-V probe
run stats-cql-vs-bc        experiments/stats-cql-vs-bc/run.py              # recommender diagnostics (val)
run cql-adjust             experiments/stats-cql-vs-bc/run.py --adjust     # control-variate CI tightening (val)
run density-validate       experiments/stats-density-validate/run.py       # confidence signal (val)
mkdir -p "${ROOT}_skill/results"
cp "$ROOT/results/stats-causal-rank.json" "${ROOT}_skill/results/"         # headroom derives its bound-vs-β̂_BC from it
( export DOTA2AD_ROOT="${ROOT}_skill"
  run skill-headroom       experiments/stats-skill-headroom/headroom.py )  # Δβ̂(high−low) on test
cp "${ROOT}_skill/results/stats-skill-headroom.json" "$ROOT/results/"      # into the primary root (check-docs reads $ROOT)
run skill-premise          experiments/stats-skill-headroom/premise.py     # skill-signal premise (corpus-level, model-free)
run mechanism              experiments/random-mechanism/run.py             # P_mech verification (full corpus)
run survivorship           experiments/random-mechanism/survivorship.py    # suppression gradient + channel decomposition
run stats-coverage         experiments/random-mechanism/stats_coverage.py  # OpenDota parse-coverage balance
run exclusions             experiments/random-mechanism/exclusions.py      # post-treatment exclusions: composition bounds + leaver-undone β̂ (val)
if [ -f "$ROOT/dataset/gone_matches.jsonl" ]; then
  run retrieval            experiments/random-mechanism/retrieval.py       # retrieval-censoring balance
else
  echo ">>> retrieval — SKIPPED: $ROOT/dataset/gone_matches.jsonl missing (rebuild: pixi run build-dataset)"
fi
run beta-bias              experiments/random-mechanism/beta_bias.py       # survivorship → β̂ (test)
run figures                experiments/figures.py                          # README figures from the manifests (CPU)

# On-demand (not in the default run):
#   stats-cql-vs-bc/run.py --ablate   de-cloning ablation (needs the B/D variant ckpts)
#   stats-cql-vs-bc/run.py --power    data-size power curve (N* to resolve A's edge)
#   stats-dqn-mc/run.py --focal-continuation bc   train Q^BC (eval-matched) vs default Q^π
#   stats-{causal-rank,cql-vs-bc}/run.py --subset {online,disconnect}  engagement split

echo "ALL DONE → results in $ROOT/logs/"

# Notes:
# - `stats-dqn-mc/run.py --n-step 1` demonstrates the TD(0) collapse.
