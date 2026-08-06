"""Headline-figure manifests.

Each experiment ends its run by dumping the figures the report quotes into
`<root>/results/<experiment>.json`. The docs stay hand-written prose;
`experiments/check_docs.py` (pixi task `check-docs`) verifies that every quoted
headline still matches the manifests, so a rerun turns doc staleness into a
loud failure instead of silent drift.
"""
from __future__ import annotations

import json
from typing import Any

from dota2ad.core.paths import default_paths


def write_results(experiment: str, figures: dict[str, Any]) -> None:
    """Write `<root>/results/<experiment>.json` (overwrites the previous run's)."""
    out = default_paths().root / "results" / f"{experiment}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(figures, f, indent=2, sort_keys=True)
    print(f"results manifest → {out}")
