# stats-density-validate

**Validation / diagnostic** (not a build step, not the calibration). Evidence that
the **state-rarity confidence band** is meaningful: rarer drafts really do incur
larger StatsModel error.

The band itself is calibrated at train time and baked into the policy checkpoint
(`policy.density_support_q`, written by `train-policy`); the server reads it from
there. This script only *checks* that signal: it bins validation-split decisions by
state support `-log p(s)/T` and reports the balanced-composite prediction RMSE per bin.

Result: <!--n density-validate: ~{rmse_ratio:.1f}× wider-->~1.3× wider<!--/n--> RMSE in the rarest support quintile than the most typical
<!--n density-validate: ({rmse_typical:.2f} → {rmse_unusual:.2f})-->(3.62 → 4.53)<!--/n--> — i.e. unusual drafts are where the model extrapolates. It's the only
part of the confidence machinery that touches the StatsModel and realized outcomes.

## Run

```bash
DOTA2AD_ROOT=work pixi run -e cuda density-validate
```
