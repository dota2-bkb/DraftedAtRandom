# experiments/

One folder per experiment: a thin `run.py` (imports the library from
`dota2ad.eval` / `dota2ad.training`) plus a short `README.md` with its command and
expected numbers.

The full story — problem, method, results, limitations — is in the top-level
[`REPORT.md`](../REPORT.md). Reproduce everything with [`RUNBOOK.md`](./RUNBOOK.md).

Run one experiment (pick a work-dir root; use the cuda env for GPU steps):

```
DOTA2AD_ROOT=<work-dir> pixi run -e cuda python experiments/<name>/run.py
```

Most also have a pixi task shortcut (see `pyproject.toml`).
