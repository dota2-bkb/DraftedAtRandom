"""Normalization utilities: MMR norm, stats norm, stats record building."""

from __future__ import annotations

import torch

from dota2ad.core.encoding import encode_mmr
from typing import cast

from dota2ad.core.types import (
    MatchRow,
    Per10,
    StatsNormDict,
    StatsRecord,
    StatsRow,
    UnifiedIdx,
    VocabKey,
    Vocabs,
)


def compute_mmr_norm(matches: list[MatchRow]) -> tuple[float, float]:
    """Compute MMR mean/std from matches."""
    all_mmr = [v for m in matches for v in m.mmr if v is not None]
    t = torch.tensor(all_mmr, dtype=torch.float32)
    return t.mean().item(), t.std().clamp(min=1e-6).item()


def compute_stats_norm(rows: list[StatsRow]) -> StatsNormDict:
    """Compute per-feature mean/std across all rows for z-normalization."""
    scalars = torch.tensor([r.scalar_stats for r in rows], dtype=torch.float32)      # [N, 10, 22]
    gold_t = torch.tensor([r.gold_t for r in rows], dtype=torch.float32)              # [N, 10, 3]
    xp_t = torch.tensor([r.xp_t for r in rows], dtype=torch.float32)                 # [N, 10, 3]
    lh_t = torch.tensor([r.lh_t for r in rows], dtype=torch.float32)                 # [N, 10, 3]
    time_mask = torch.tensor([r.time_mask for r in rows], dtype=torch.bool)            # [N, 10, 3]
    kills = torch.tensor([r.kill_counts for r in rows], dtype=torch.float32)           # [N, 10, 5]
    deaths = torch.tensor([r.death_counts for r in rows], dtype=torch.float32)         # [N, 10, 5]
    damage = torch.tensor([r.damage_dealt for r in rows], dtype=torch.float32)         # [N, 10, 5]
    gold_r = torch.tensor([r.gold_reasons for r in rows], dtype=torch.float32)         # [N, 10, 14]
    xp_r = torch.tensor([r.xp_reasons for r in rows], dtype=torch.float32)            # [N, 10, 6]
    priorities = torch.tensor([r.ability_priorities for r in rows], dtype=torch.float32)  # [N, 10, 4]

    def stats(x: torch.Tensor) -> tuple[list[float], list[float]]:
        """Per-feature mean/std. x: [N, 10, F] -> flat: [N*10, F] -> returns ([F], [F])."""
        flat = x.reshape(-1, x.shape[-1])                                              # [N*10, F]
        return flat.mean(0).tolist(), flat.std(0).clamp(min=1e-6).tolist()

    def masked_stats(x: torch.Tensor, mask: torch.Tensor) -> tuple[list[float], list[float]]:
        """Masked per-feature mean/std. x, mask: [N, 10, F] -> returns ([F], [F])."""
        flat_x = x.reshape(-1, x.shape[-1])                                           # [N*10, F]
        flat_m = mask.reshape(-1, x.shape[-1]).float()                                 # [N*10, F]
        total = flat_m.sum(0).clamp(min=1)                                             # [F]
        mean = (flat_x * flat_m).sum(0) / total                                        # [F]
        var = ((flat_x - mean) ** 2 * flat_m).sum(0) / total                           # [F]
        return mean.tolist(), var.sqrt().clamp(min=1e-6).tolist()

    all_mmr = [v for r in rows for v in r.mmr if v is not None]
    mmr_t = torch.tensor(all_mmr, dtype=torch.float32)                                 # [M]
    mmr_mean = mmr_t.mean().item()
    mmr_std = mmr_t.std().clamp(min=1e-6).item()

    matchup = torch.cat([kills, deaths], dim=1).reshape(-1)                            # [N*20*5]
    damage_flat = damage.reshape(-1)                                                    # [N*10*5]
    # Per-spell damage: sum over the 5 enemies → [N, 10, 4], then flatten for
    # single-scalar mean/std (same convention as `damage`).
    spell_dmg = torch.tensor(
        [[[sum(slot) for slot in player]
          for player in r.spell_damage_dealt] for r in rows],
        dtype=torch.float32,
    )                                                                                   # [N, 10, 4]
    spell_dmg_flat = spell_dmg.reshape(-1)
    return {
        "scalar": stats(scalars),                                                       # ([22], [22])
        "gold_t": masked_stats(gold_t, time_mask),                                      # ([3], [3])
        "xp_t": masked_stats(xp_t, time_mask),                                         # ([3], [3])
        "lh_t": masked_stats(lh_t, time_mask),                                         # ([3], [3])
        "matchup": (matchup.mean().item(), matchup.std().clamp(min=1e-6).item()),       # (float, float)
        "damage": (damage_flat.mean().item(), damage_flat.std().clamp(min=1e-6).item()),  # (float, float)
        "gold_reasons": stats(gold_r),                                                  # ([14], [14])
        "xp_reasons": stats(xp_r),                                                     # ([6], [6])
        "priority": stats(priorities),                                                  # ([4], [4])
        "spell_damage": (spell_dmg_flat.mean().item(),
                         spell_dmg_flat.std().clamp(min=1e-6).item()),                  # (float, float)
        "mmr": (mmr_mean, mmr_std),                                                    # (float, float)
    }


def build_stats_records(rows: list[StatsRow], norm: StatsNormDict, vocabs: Vocabs) -> list[StatsRecord]:
    """Apply z-normalization to raw StatsRows, producing StatsRecords."""
    s_mean = torch.tensor(norm["scalar"][0], dtype=torch.float32)
    s_std = torch.tensor(norm["scalar"][1], dtype=torch.float32)
    g_mean = torch.tensor(norm["gold_t"][0], dtype=torch.float32)
    g_std = torch.tensor(norm["gold_t"][1], dtype=torch.float32)
    x_mean = torch.tensor(norm["xp_t"][0], dtype=torch.float32)
    x_std = torch.tensor(norm["xp_t"][1], dtype=torch.float32)
    l_mean = torch.tensor(norm["lh_t"][0], dtype=torch.float32)
    l_std = torch.tensor(norm["lh_t"][1], dtype=torch.float32)
    m_mean, m_std = norm["matchup"]
    d_mean, d_std = norm["damage"]
    gr_mean = torch.tensor(norm["gold_reasons"][0], dtype=torch.float32)
    gr_std = torch.tensor(norm["gold_reasons"][1], dtype=torch.float32)
    xr_mean = torch.tensor(norm["xp_reasons"][0], dtype=torch.float32)
    xr_std = torch.tensor(norm["xp_reasons"][1], dtype=torch.float32)
    p_mean = torch.tensor(norm["priority"][0], dtype=torch.float32)
    p_std = torch.tensor(norm["priority"][1], dtype=torch.float32)
    sd_mean, sd_std = norm["spell_damage"]
    mmr_mean, mmr_std = norm["mmr"]

    records: list[StatsRecord] = []
    for row in rows:
        mmr_vals, mmr_mask = encode_mmr(row.mmr, mmr_mean, mmr_std)
        records.append(StatsRecord(
            loadouts=cast(
                Per10[list[UnifiedIdx]],
                tuple(
                    [vocabs.draft_id_to_index[key] for key in player]
                    for player in row.loadouts
                ),
            ),
            mmr_vals=mmr_vals,
            mmr_mask=mmr_mask,
            ability_indices=torch.tensor(
                [[vocabs.draft_id_to_index[VocabKey(f"a:{did}")] for did in player]
                 for player in row.ability_draft_ids],
                dtype=torch.long,
            ),
            scalar_stats=(torch.tensor(row.scalar_stats, dtype=torch.float32) - s_mean) / s_std,
            gold_t=(torch.tensor(row.gold_t, dtype=torch.float32) - g_mean) / g_std,
            xp_t=(torch.tensor(row.xp_t, dtype=torch.float32) - x_mean) / x_std,
            lh_t=(torch.tensor(row.lh_t, dtype=torch.float32) - l_mean) / l_std,
            time_mask=torch.tensor(row.time_mask, dtype=torch.bool),
            kill_counts=(torch.tensor(row.kill_counts, dtype=torch.float32) - m_mean) / m_std,
            death_counts=(torch.tensor(row.death_counts, dtype=torch.float32) - m_mean) / m_std,
            damage_dealt=(torch.tensor(row.damage_dealt, dtype=torch.float32) - d_mean) / d_std,
            gold_reasons=(torch.tensor(row.gold_reasons, dtype=torch.float32) - gr_mean) / gr_std,
            xp_reasons=(torch.tensor(row.xp_reasons, dtype=torch.float32) - xr_mean) / xr_std,
            ability_priorities=(torch.tensor(row.ability_priorities, dtype=torch.float32) - p_mean) / p_std,
            # Per-(player, slot) total spell damage = sum over 5 enemies.
            spell_damage_dealt=(torch.tensor(
                [[sum(slot) for slot in player]
                 for player in row.spell_damage_dealt],
                dtype=torch.float32,
            ) - sd_mean) / sd_std,
            match_id=row.match_id,
        ))
    return records
