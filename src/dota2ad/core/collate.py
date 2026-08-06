"""Collation functions: batch construction for DataLoader."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from dota2ad.core.encoding import pad_sets
from dota2ad.core.types import (
    LabeledPolicyBatch,
    LabeledPolicySample,
    PolicyBatch,
    PolicySample,
    StatsBatch,
    StatsRecord,
)


def policy_collate(batch: Sequence[PolicySample], device: torch.device | None = None) -> PolicyBatch:
    """Build padded tensors from a batch of PolicySamples (no action label)."""
    # Loadout sets: B*10 merged per-player sets
    all_loadouts: list[Sequence[int]] = []
    all_load_is_random: list[Sequence[int]] = []
    all_load_is_disconnected: list[Sequence[int]] = []
    for s in batch:
        all_loadouts.extend(s.loadouts)
        # Pad the flag lists in parallel with loadouts; 0/1 ints reuse pad_sets.
        for flags in s.loadouts_is_random:
            all_load_is_random.append([1 if f else 0 for f in flags])
        for flags in s.loadouts_is_disconnected:
            all_load_is_disconnected.append([1 if f else 0 for f in flags])
    load_idx, load_mask = pad_sets(all_loadouts)
    load_is_random_int, _ = pad_sets(all_load_is_random)
    load_is_random = load_is_random_int.bool()
    load_is_disconnected_int, _ = pad_sets(all_load_is_disconnected)
    load_is_disconnected = load_is_disconnected_int.bool()

    # Pool: B merged sets
    pool_idx, pool_mask = pad_sets([s.pool for s in batch])

    # Candidates: pad variable-length candidate lists
    cand_idx_t, cand_mask_t = pad_sets([s.cand_idx for s in batch])
    cand_type_t, _ = pad_sets([s.cand_type for s in batch])

    # History: pad variable-length history sequences
    hist_idx_t, hist_mask_t = pad_sets([s.hist_idx for s in batch])
    hist_slot_t, _ = pad_sets([s.hist_slot for s in batch])

    result: PolicyBatch = {
        "load_idx": load_idx, "load_mask": load_mask,
        "load_is_random": load_is_random,
        "load_is_disconnected": load_is_disconnected,
        "pool_idx": pool_idx, "pool_mask": pool_mask,
        "pick_slot": torch.tensor([s.pick_slot for s in batch], dtype=torch.long),
        "mmr": torch.tensor([s.mmr_vals for s in batch], dtype=torch.float32),
        "mmr_mask": torch.tensor([s.mmr_mask for s in batch], dtype=torch.bool),
        "feasible_mask": torch.stack([s.feasible_mask for s in batch]),
        "cand_idx": cand_idx_t, "cand_mask": cand_mask_t,
        "cand_type": cand_type_t,
        "turn": torch.tensor([s.turn for s in batch], dtype=torch.long),
        "round_idx": torch.tensor([s.round_idx for s in batch], dtype=torch.long),
        "side_idx": torch.tensor([s.side_idx for s in batch], dtype=torch.long),
        "hero_filled": torch.tensor([s.hero_filled for s in batch], dtype=torch.long),
        "basics_count": torch.tensor([s.basics_count for s in batch], dtype=torch.long),
        "ult_filled": torch.tensor([s.ult_filled for s in batch], dtype=torch.long),
        "pool_heroes": torch.tensor([s.pool_heroes for s in batch], dtype=torch.float32),
        "pool_basics": torch.tensor([s.pool_basics for s in batch], dtype=torch.float32),
        "pool_ults": torch.tensor([s.pool_ults for s in batch], dtype=torch.float32),
        "hist_idx": hist_idx_t, "hist_mask": hist_mask_t,
        "hist_slot": hist_slot_t,
    }
    if device is not None:
        result = {k: v.to(device) for k, v in result.items()}  # type: ignore[misc]
    return result


def labeled_policy_collate(
    batch: Sequence[LabeledPolicySample], device: torch.device | None = None,
) -> LabeledPolicyBatch:
    """Training-time collate: PolicyBatch + the action_idx target tensor."""
    base = policy_collate(batch, device=None)
    action_idx = torch.tensor([s.action_idx for s in batch], dtype=torch.long)
    if device is not None:
        base = {k: v.to(device) for k, v in base.items()}  # type: ignore[misc]
        action_idx = action_idx.to(device)
    return cast(LabeledPolicyBatch, {**base, "action_idx": action_idx})


def stats_collate(batch: list[StatsRecord], device: torch.device | None = None) -> StatsBatch:
    """Build padded tensors from a batch of StatsRecords."""
    all_loadouts: list[Sequence[int]] = []
    for rec in batch:
        all_loadouts.extend(rec.loadouts)
    load_idx, load_mask = pad_sets(all_loadouts)

    result: StatsBatch = {
        "load_idx": load_idx, "load_mask": load_mask,
        "mmr": torch.tensor([rec.mmr_vals for rec in batch], dtype=torch.float32),
        "mmr_mask": torch.tensor([rec.mmr_mask for rec in batch], dtype=torch.bool),
        "ability_indices": torch.stack([rec.ability_indices for rec in batch]),
        "scalar_stats": torch.stack([rec.scalar_stats for rec in batch]),
        "gold_t": torch.stack([rec.gold_t for rec in batch]),
        "xp_t": torch.stack([rec.xp_t for rec in batch]),
        "lh_t": torch.stack([rec.lh_t for rec in batch]),
        "time_mask": torch.stack([rec.time_mask for rec in batch]),
        "kill_counts": torch.stack([rec.kill_counts for rec in batch]),
        "death_counts": torch.stack([rec.death_counts for rec in batch]),
        "damage_dealt": torch.stack([rec.damage_dealt for rec in batch]),
        "gold_reasons": torch.stack([rec.gold_reasons for rec in batch]),
        "xp_reasons": torch.stack([rec.xp_reasons for rec in batch]),
        "ability_priorities": torch.stack([rec.ability_priorities for rec in batch]),
        "spell_damage_dealt": torch.stack([rec.spell_damage_dealt for rec in batch]),
    }
    if device is not None:
        result = {k: v.to(device) for k, v in result.items()}  # type: ignore[misc]
    return result
