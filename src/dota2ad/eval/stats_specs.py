"""Stat catalog (`STAT_SPECS`), the balanced composite signs, and the
StatsModel-prediction machinery used by `stats-density-validate`.

Stat predictions are made on the realized match's terminal draft — all 10
loadouts exactly as the game finished; `stats-density-validate` compares them
with the realized outcomes to validate the state-rarity confidence band.

For convenience this module re-exports the bootstrap + gap primitives
(`bucket_realized`, `binary_gap_se`, `continuous_gap_se`) from
`dota2ad.eval.bootstrap` so callers can import everything from one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from collections.abc import Callable

import torch

from dota2ad.core import stats_collate
from dota2ad.core.types import (
    Per10,
    StatsRecord,
    StatsRow,
    UnifiedIdx,
    VocabKey,
)
from dota2ad.eval.tuples import RandomPickTuple

# Re-exported for single-import convenience (see module docstring). The
# redundant `as` aliases mark these as intentional public re-exports.
from dota2ad.eval.bootstrap import (
    binary_gap_se as binary_gap_se,
    bucket_realized as bucket_realized,
    continuous_gap_se as continuous_gap_se,
)


@dataclass
class StatSpec:
    label: str
    # pred(model_outputs, focal_slot) → 1-D tensor [B] (predicted scalar)
    pred_fn: Callable[[tuple[torch.Tensor, ...], torch.Tensor], torch.Tensor]
    # real(stats_row, focal_slot_int) → float (realized scalar)
    real_fn: Callable[[StatsRow, int], float]


def _scalar_spec(name: str, idx: int) -> StatSpec:
    return StatSpec(
        label=name,
        pred_fn=lambda outs, fs: outs[0][torch.arange(outs[0].shape[0]), fs, idx],
        real_fn=lambda row, fs: float(row.scalar_stats[fs][idx]),
    )


def _sum_kills_spec() -> StatSpec:
    # kill_counts are per-min (build_match_stats divides by duration_min).
    return StatSpec(
        label="sum kills/min vs enemy heroes",
        pred_fn=lambda outs, fs: outs[4][torch.arange(outs[4].shape[0]), fs, :, 0].sum(-1),
        real_fn=lambda row, fs: float(sum(row.kill_counts[fs])),
    )


def _sum_deaths_spec() -> StatSpec:
    # death_counts are per-min (build_match_stats divides by duration_min).
    return StatSpec(
        label="sum deaths/min to enemy heroes",
        pred_fn=lambda outs, fs: outs[4][torch.arange(outs[4].shape[0]), fs, :, 1].sum(-1),
        real_fn=lambda row, fs: float(sum(row.death_counts[fs])),
    )


def _sum_damage_spec() -> StatSpec:
    return StatSpec(
        label="sum damage to enemy heroes",
        pred_fn=lambda outs, fs: outs[6][torch.arange(outs[6].shape[0]), fs, :].sum(-1),
        real_fn=lambda row, fs: float(sum(row.damage_dealt[fs])),
    )


def _farm_gold_spec() -> StatSpec:
    """Farm gold/min = creep + neutral gold, EXCLUDING hero-kill gold. Reads the
    gold_reasons head (outs[7], [B,10,14], in GOLD_REASON_KEYS order). Vec-index
    semantics verified empirically (corr with kills/last_hits/neutral):
    idx5=hero-kill (corr_kills .80, excluded), idx6=creep (corr_lh .84),
    idx7=neutral (corr_neu .98)."""
    return StatSpec(
        label="farm gold/min (creep+neutral, excl hero-kill)",
        pred_fn=lambda outs, fs: (
            outs[7][torch.arange(outs[7].shape[0]), fs, 6]
            + outs[7][torch.arange(outs[7].shape[0]), fs, 7]
        ),
        real_fn=lambda row, fs: float(row.gold_reasons[fs][6] + row.gold_reasons[fs][7]),
    )


def _farm_xp_spec() -> StatSpec:
    """Farm xp/min = creep xp, EXCLUDING hero-kill xp. Reads the xp_reasons head
    (outs[8], [B,10,6], in XP_REASON_KEYS order). Verified empirically: idx1=
    hero-kill (corr_kills .62, excluded), idx2=creep (corr_lh .92, captures lane
    + jungle xp)."""
    return StatSpec(
        label="farm xp/min (creep, excl hero-kill)",
        pred_fn=lambda outs, fs: outs[8][torch.arange(outs[8].shape[0]), fs, 2],
        real_fn=lambda row, fs: float(row.xp_reasons[fs][2]),
    )


def _team_scalar_spec(label: str, idx: int, enemy: bool = False) -> StatSpec:
    def pred_fn(outs: tuple[torch.Tensor, ...], fs: torch.Tensor) -> torch.Tensor:
        s = outs[0][..., idx]                                 # [B, 10]
        rad = s[:, 0::2].sum(-1)                              # [B]
        dire = s[:, 1::2].sum(-1)                             # [B]
        focal_even = (fs % 2 == 0)
        if enemy:
            return torch.where(focal_even, dire, rad)
        return torch.where(focal_even, rad, dire)
    def real_fn(row: StatsRow, fs: int) -> float:
        start = ((fs + 1) % 2) if enemy else (fs % 2)
        return float(sum(row.scalar_stats[s][idx] for s in range(start, 10, 2)))
    return StatSpec(label=label, pred_fn=pred_fn, real_fn=real_fn)


STAT_SPECS: list[StatSpec] = [
    _scalar_spec("kills/min", 0),
    _scalar_spec("deaths/min (lower = better)", 1),
    _scalar_spec("assists/min", 2),
    _scalar_spec("gold_per_min", 3),
    _scalar_spec("xp_per_min", 4),
    _scalar_spec("last_hits/min", 5),
    _scalar_spec("denies/min", 6),
    _scalar_spec("hero_damage/min", 7),
    _scalar_spec("tower_damage/min", 8),
    _scalar_spec("hero_healing/min", 9),
    _scalar_spec("stuns/min", 10),
    _scalar_spec("tower_kills/min", 11),
    _scalar_spec("teamfight_participation", 18),
    _sum_kills_spec(),
    _sum_deaths_spec(),
    _sum_damage_spec(),
    # Team-aggregate (ally, focal's 5): cross-player synergy missed by per-focal heads.
    _team_scalar_spec("team kills/min", 0),
    _team_scalar_spec("team hero_damage/min", 7),
    _team_scalar_spec("team tower_damage/min", 8),
    _team_scalar_spec("team tower_kills/min", 11),
    # Team-aggregate (enemy, the other 5): suppression picks. Weighted NEGATIVELY
    # in aggressive presets — lower enemy output = good.
    _team_scalar_spec("enemy team hero_damage/min", 7, enemy=True),
    _team_scalar_spec("enemy team tower_damage/min", 8, enemy=True),
    _team_scalar_spec("enemy team tower_kills/min", 11, enemy=True),
    # Team-aggregate (ally deaths): supports raise the team's survival floor.
    # Weighted NEGATIVELY in the support preset — fewer ally deaths = good.
    _team_scalar_spec("team deaths/min", 1),
    # Farm economy excluding hero-kill income (creep+neutral gold, creep xp).
    # No per-spell damage dims: summed they duplicate hero_damage/min; per-pick
    # damage attribution is instead the dense dim-7 damage reward (see
    # stats_simulator).
    _farm_gold_spec(),
    _farm_xp_spec(),
]


# Hand-picked signs for the "balanced" equal-weighted composite. Indices are
# into STAT_SPECS. +1 = more is better, -1 = less is better. Excluded (zero
# weight, but still trained targets in STAT_SPECS): teamfight_participation
# (12 — a role descriptor, not monotone-good); the per-enemy sums (13/14/15 —
# duplicate 0/1/7); and the tower dims (8/11 — tower output is a team OUTCOME
# you happen to deal, a causally-real but goal-misaligned proxy that over-ranks
# pushers like Arc Warden). What remains is the set of individually-
# attributable, goal-aligned per-min stats.
BALANCED_COMPOSITE_SIGNS: dict[int, float] = {
    0: +1.0,   # kills/min
    1: -1.0,   # deaths/min
    3: +1.0,   # gold_per_min
    4: +1.0,   # xp_per_min
    5: +1.0,   # last_hits/min
    7: +1.0,   # hero_damage/min
}


def _build_realized_records(
    tuples: list[RandomPickTuple],
    stats_rows_by_id: dict[int, StatsRow],
    vocabs,
) -> list[StatsRecord]:
    """One StatsRecord per tuple: the realized match's terminal draft verbatim
    (all 10 loadouts as the game finished). Outcome tensors are zero dummies —
    the StatsModel reads only the draft-side inputs at inference.
    """
    dummy_22 = torch.zeros(10, 22)
    dummy_3 = torch.zeros(10, 3)
    dummy_5 = torch.zeros(10, 5)
    dummy_14 = torch.zeros(10, 14)
    dummy_6 = torch.zeros(10, 6)
    dummy_4 = torch.zeros(10, 4)
    time_mask = torch.zeros(10, 3, dtype=torch.bool)
    records: list[StatsRecord] = []
    for t in tuples:
        row = stats_rows_by_id[t.match_id]
        ability_indices = torch.tensor(
            [
                [vocabs.draft_id_to_index[VocabKey(f"a:{did}")] for did in player]
                for player in row.ability_draft_ids
            ],
            dtype=torch.long,
        )
        records.append(StatsRecord(
            loadouts=cast(Per10[list[UnifiedIdx]], tuple(t.final_loadouts_other)),
            mmr_vals=t.mmr_vals,
            mmr_mask=t.mmr_mask,
            ability_indices=ability_indices,
            scalar_stats=dummy_22,
            gold_t=dummy_3,
            xp_t=dummy_3,
            lh_t=dummy_3,
            time_mask=time_mask,
            kill_counts=dummy_5,
            death_counts=dummy_5,
            damage_dealt=dummy_5,
            gold_reasons=dummy_14,
            xp_reasons=dummy_6,
            ability_priorities=dummy_4,
            spell_damage_dealt=dummy_4,
            match_id=t.match_id,
        ))
    return records


def compute_stat_predictions(
    tuples: list[RandomPickTuple],
    stats_rows_by_id: dict[int, StatsRow],
    stats_model,
    vocabs,
    device: torch.device,
    batch_size: int,
) -> dict[int, list[float]]:
    """Per stat spec, the StatsModel's focal prediction on each tuple's realized
    terminal draft. Returns stat_idx → per-tuple predicted scalars, in the
    model's normalized scale.
    """
    records = _build_realized_records(tuples, stats_rows_by_id, vocabs)
    n_records = len(records)
    spec_outputs: list[torch.Tensor] = [torch.empty(n_records) for _ in STAT_SPECS]
    with torch.no_grad():
        for i in range(0, n_records, batch_size):
            chunk = records[i:i + batch_size]
            batch = stats_collate(chunk, device=device)
            outs = stats_model(batch)
            B = batch["mmr"].shape[0]
            focal_slots = torch.tensor(
                [t.focal_slot for t in tuples[i:i + B]],
                dtype=torch.long, device=device,
            )
            for s_idx, spec in enumerate(STAT_SPECS):
                spec_outputs[s_idx][i:i + B] = spec.pred_fn(outs, focal_slots).cpu()
    return {s_idx: spec_outputs[s_idx].tolist() for s_idx in range(len(STAT_SPECS))}
