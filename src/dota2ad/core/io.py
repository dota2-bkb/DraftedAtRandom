"""Data loading: matches, vocabs, splits, stats rows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import msgspec

from dota2ad.core.paths import default_paths
from dota2ad.core.types import ExcludeReason, MatchRow, StatsRow, Vocabs

DEFAULT_EXCLUDES: tuple[ExcludeReason, ...] = (
    ExcludeReason.TOO_MANY_RANDOM_PICKS,
    ExcludeReason.LEAVERS,
    ExcludeReason.SWAPS,
)

_match_row_decoder = msgspec.json.Decoder(MatchRow)
_stats_row_decoder = msgspec.json.Decoder(StatsRow)
_excluded_decoder = msgspec.json.Decoder(dict[ExcludeReason, list[int]])
_split_decoder = msgspec.json.Decoder(dict[str, list[int]])
_vocabs_decoder = msgspec.json.Decoder(Vocabs)


def load_excluded_matches(
    reasons: Sequence[ExcludeReason], path: str | Path | None = None,
) -> set[int]:
    """Load set of excluded match IDs for the given exclusion reasons."""
    p = Path(path) if path is not None else default_paths().excluded
    with open(p, "rb") as f:
        data = _excluded_decoder.decode(f.read())
    result: set[int] = set()
    for reason in reasons:
        result.update(data[reason])
    return result


def load_matches(
    path: str | Path | None = None,
    exclude: Sequence[ExcludeReason] = DEFAULT_EXCLUDES,
) -> list[MatchRow]:
    paths = default_paths()
    p = Path(path) if path is not None else paths.matches
    excluded = load_excluded_matches(exclude, paths.excluded)
    decode = _match_row_decoder.decode
    with open(p, "rb") as f:
        matches = [decode(line) for line in f]
    return [m for m in matches if m.match_id not in excluded]


def load_vocabs(path: str | Path | None = None) -> Vocabs:
    p = Path(path) if path is not None else default_paths().vocabs
    with open(p, "rb") as f:
        return _vocabs_decoder.decode(f.read())


class SplitIds(NamedTuple):
    """Held-out match IDs from dataset/split.json. Roles: val = selection
    (best epoch, hyperparameters, calibration, diagnostics); test = read once
    (final report numbers only); train = every match not in `held_out`."""

    val_ids: set[int]
    test_ids: set[int]

    @property
    def held_out(self) -> set[int]:
        return self.val_ids | self.test_ids


def load_split(path: str | Path | None = None) -> SplitIds:
    """Load the val/test match-ID split from dataset/split.json."""
    p = Path(path) if path is not None else default_paths().split
    with open(p, "rb") as f:
        d = _split_decoder.decode(f.read())
    return SplitIds(val_ids=set(d["val_match_ids"]), test_ids=set(d["test_match_ids"]))


def load_stats_rows(
    path: str | Path | None = None,
    exclude: Sequence[ExcludeReason] = DEFAULT_EXCLUDES,
) -> list[StatsRow]:
    """Load match_stats.jsonl, filtered by exclusions. Returns raw StatsRows."""
    paths = default_paths()
    p = Path(path) if path is not None else paths.match_stats
    excluded = load_excluded_matches(exclude, paths.excluded)
    decode = _stats_row_decoder.decode
    with open(p, "rb") as f:
        return [r for line in f if (r := decode(line)).match_id not in excluded]
