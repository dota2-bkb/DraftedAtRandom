"""Project paths — single source of truth.

The whole tree lives under one root directory (cwd by default; override
via the DOTA2AD_ROOT environment variable). Filenames are property-derived
so renaming an artifact is a single edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    # Top-level directories
    @property
    def parsed(self) -> Path:
        return self.root / "parsed"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def cache(self) -> Path:
        return self.root / ".cache"

    # Built dataset artifacts
    @property
    def matches(self) -> Path:
        return self.dataset / "matches.jsonl"

    @property
    def vocabs(self) -> Path:
        return self.dataset / "vocabs.json"

    @property
    def split(self) -> Path:
        return self.dataset / "split.json"

    @property
    def excluded(self) -> Path:
        return self.dataset / "excluded_matches.json"

    @property
    def match_stats(self) -> Path:
        return self.dataset / "match_stats.jsonl"

    # Trained checkpoints
    @property
    def policy_ckpt(self) -> Path:
        return self.models / "policy.pt"

    @property
    def stats_ckpt(self) -> Path:
        return self.models / "match_stats.pt"

    @property
    def stats_dqn_ckpt(self) -> Path:
        return self.models / "stats_dqn.pt"

    # External-API caches
    @property
    def replays(self) -> Path:
        return self.cache / "replays"


def default_paths() -> Paths:
    """Resolve `DOTA2AD_ROOT` (or cwd) at call time."""
    return Paths(root=Path(os.environ.get("DOTA2AD_ROOT", ".")))
