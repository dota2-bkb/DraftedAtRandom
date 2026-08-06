# train-policy

Trains the BehaviorPolicy — predicts the *human* pick from a draft state. Ships
as the plausibility filter and warm-starts the stats-DQN. The policy has a V+1
head (extra class = "this seat will let the timer expire"), used by the
stats-DQN simulator for realistic non-focal continuations.

## Run
```
DOTA2AD_ROOT=work pixi run -e cuda python experiments/train-policy/run.py
```

## Key result
<!--n train-policy: Top-1 ≈ {top1:.0%} / top-5 ≈ {top5:.0%}-->Top-1 ≈ 30% / top-5 ≈ 71%<!--/n--> on held-out deliberate picks <!--n train-policy: (ECE ≈ {ece:.2f})-->(ECE ≈ 0.01)<!--/n-->.

See the top-level [`REPORT.md`](../../REPORT.md) for the full story.
