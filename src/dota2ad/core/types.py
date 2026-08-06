"""Core type definitions: msgspec Structs, TypedDicts, dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, NewType, TypedDict, TypeVar

import msgspec
import torch

NUM_PLAYERS = 10


class ExcludeReason(StrEnum):
    """Match-exclusion categories. The values are the keys under which match
    IDs are grouped in dataset/excluded_matches.json."""

    TOO_MANY_RANDOM_PICKS = "too_many_random_picks"  # bot matches (n_random > 25)
    LEAVERS = "leavers"                               # any player abandoned/disconnected
    SWAPS = "swaps"                                   # post-draft loadout traded between players

# ID disambiguation. Runtime is plain int; NewType lets the type checker
# catch swaps. msgspec/JSON treat these as int.
PickSlot = NewType("PickSlot", int)      # 0..9, player position in the draft
Turn = NewType("Turn", int)              # 0..49, draft turn number
UnifiedIdx = NewType("UnifiedIdx", int)  # vocab index in draft_id_to_index
HeroId = NewType("HeroId", int)          # raw Dota hero ID (e.g. 2 = axe)
AbilityId = NewType("AbilityId", int)    # raw Dota draft_ability_id
VocabKey = NewType("VocabKey", str)      # "h:{hero_id}" | "a:{ability_id}" | "<empty>"

# Fixed-length positional tuples — msgspec validates length, the type
# checker tracks each position, OpenAPI emits `prefixItems`. We use the
# older TypeVar-based alias style (not PEP 695) so Litestar's OpenAPI
# plugin can resolve the alias.
_T = TypeVar("_T")
Per10 = tuple[_T, _T, _T, _T, _T, _T, _T, _T, _T, _T]
Per12 = tuple[_T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T]
Per36 = tuple[
    _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T,
    _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T,
    _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T, _T,
]


def per10[T](v: T) -> Per10[T]:
    """Build a length-10 tuple of `v`."""
    return (v, v, v, v, v, v, v, v, v, v)


# ---------------------------------------------------------------------------
# Vocab type
# ---------------------------------------------------------------------------


class Vocabs(msgspec.Struct):
    draft_id_to_index: dict[VocabKey, UnifiedIdx]


# ---------------------------------------------------------------------------
# Row types (msgspec Structs — written/read as JSONL)
# ---------------------------------------------------------------------------


class PlayerPickState(msgspec.Struct):
    hero: HeroId | None
    basics: list[AbilityId]
    ult: AbilityId | None


class TurnRow(msgspec.Struct):
    match_id: int
    turn: Turn
    pick_slot: PickSlot
    hero_id: HeroId | None
    draft_ability_id: AbilityId | None
    action_key: str
    is_random: bool
    # Connection state of the picking player at the tick the pick registered:
    # True iff CDOTA_PlayerResource m_iConnectionState != CONNECTED — which
    # also covers never-connected, still-loading, and abandoned seats, so
    # "not fully present", not only mid-game drops. Instantaneous: a player
    # who reconnects before the timer expires reads connected. Only
    # meaningful when is_random=True: every "random" pick is a server-side
    # timeout, and this flag splits timeouts into online timeouts (False —
    # the picker was present but let the timer run; ~97% of forced picks)
    # and disconnect timeouts (True — the picker was absent, so the timeout
    # is state-independent: no decision was made).
    picker_disconnected: bool
    radiant_win: bool
    mmr: Per10[float | None]
    hero_pool_remaining: list[HeroId]
    basic_pool_remaining: list[AbilityId]
    ult_pool_remaining: list[AbilityId]
    player_picks: dict[PickSlot, PlayerPickState]


class HistoryEvent(msgspec.Struct):
    hero_id: HeroId | None
    draft_ability_id: AbilityId | None
    action_key: str
    is_random: bool
    picker_disconnected: bool      # semantics: see TurnRow.picker_disconnected


class MatchRow(msgspec.Struct):
    match_id: int
    radiant_win: bool
    mmr: Per10[float | None]
    hero_pool: Per12[HeroId]
    basic_pool: Per36[AbilityId]
    ult_pool: Per12[AbilityId]
    history: list[HistoryEvent]   # ~50 events; mutated during pipeline build


class StatsRow(msgspec.Struct):
    match_id: int
    loadouts: Per10[list[VocabKey]]
    mmr: Per10[float | None]
    ability_draft_ids: Per10[list[AbilityId]]
    scalar_stats: Per10[list[float]]
    gold_t: Per10[list[float]]
    xp_t: Per10[list[float]]
    lh_t: Per10[list[float]]
    time_mask: Per10[list[bool]]
    kill_counts: Per10[list[float]]
    death_counts: Per10[list[float]]
    damage_dealt: Per10[list[float]]
    gold_reasons: Per10[list[float]]
    xp_reasons: Per10[list[float]]
    ability_priorities: Per10[list[float]]
    # Per-spell hero damage attribution, per-min.
    # Shape: [10 players][4 spells (priority-sorted, aligned with
    # ability_draft_ids)][5 enemies]. From OpenDota damage_targets[ability_key]
    # [npc_key]. Auto-attacks ('null') and item damage NOT included — only
    # spell-attributable damage to enemy heroes.
    spell_damage_dealt: Per10[list[list[float]]]


# ---------------------------------------------------------------------------
# Policy data types
# ---------------------------------------------------------------------------


@dataclass
class PolicySample:
    """Inference-time draft state encoding. No action label."""
    loadouts: Per10[list[UnifiedIdx]]            # one merged set per player
    # Per loadout-position random flags, parallel to `loadouts` in length and
    # order. `loadouts_is_random`: the position was filled by a server timeout
    # of either kind (event.is_random). `loadouts_is_disconnected`: the subset
    # whose picker was offline at the timeout (event.picker_disconnected) —
    # three position classes for the encoder: deliberate, online random,
    # disconnected random. Both are learned-embed features in the
    # BehaviorPolicy / QNetStats encoders.
    loadouts_is_random: Per10[list[bool]]
    loadouts_is_disconnected: Per10[list[bool]]
    pool: list[UnifiedIdx]                       # all remaining pool items merged
    pick_slot: PickSlot                          # acting player
    mmr_vals: Per10[float]
    mmr_mask: Per10[bool]
    feasible_mask: torch.Tensor                  # [vocab_size], bool
    cand_idx: list[UnifiedIdx]                   # feasible action vocab indices
    cand_type: list[Literal[0, 1, 2]]            # 0=hero, 1=basic, 2=ult per candidate
    turn: Turn
    round_idx: int                                # turn // 10, 0..4
    side_idx: Literal[0, 1]                       # 0=radiant, 1=dire
    hero_filled: bool                             # actor has hero
    basics_count: int                             # 0..3
    ult_filled: bool                              # actor has ult
    pool_heroes: float                            # len(hero_pool_remaining) / 12
    pool_basics: float                            # len(basic_pool_remaining) / 36
    pool_ults: float                              # len(ult_pool_remaining) / 12
    hist_idx: list[UnifiedIdx] = field(default_factory=list)
    hist_slot: list[PickSlot] = field(default_factory=list)


@dataclass(kw_only=True)
class LabeledPolicySample(PolicySample):
    """Training-time sample: a PolicySample plus the action that was taken."""
    action_idx: UnifiedIdx


class PolicyBatch(TypedDict):
    load_idx: torch.Tensor    # [B*10, max_len]
    load_mask: torch.Tensor   # [B*10, max_len]
    load_is_random: torch.Tensor  # [B*10, max_len], bool — per-position
    load_is_disconnected: torch.Tensor  # [B*10, max_len], bool — per-position
    pool_idx: torch.Tensor    # [B, max_pool_len]
    pool_mask: torch.Tensor   # [B, max_pool_len]
    pick_slot: torch.Tensor   # [B]
    mmr: torch.Tensor         # [B, 10]
    mmr_mask: torch.Tensor    # [B, 10]
    feasible_mask: torch.Tensor  # [B, vocab_size]
    cand_idx: torch.Tensor    # [B, max_cands]
    cand_mask: torch.Tensor   # [B, max_cands]
    cand_type: torch.Tensor   # [B, max_cands]
    turn: torch.Tensor        # [B]
    round_idx: torch.Tensor   # [B]
    side_idx: torch.Tensor    # [B]
    hero_filled: torch.Tensor # [B]
    basics_count: torch.Tensor # [B]
    ult_filled: torch.Tensor  # [B]
    pool_heroes: torch.Tensor # [B]
    pool_basics: torch.Tensor # [B]
    pool_ults: torch.Tensor   # [B]
    hist_idx: torch.Tensor    # [B, max_hist_len]
    hist_mask: torch.Tensor   # [B, max_hist_len]
    hist_slot: torch.Tensor   # [B, max_hist_len]


class LabeledPolicyBatch(PolicyBatch):
    action_idx: torch.Tensor  # [B]


# ---------------------------------------------------------------------------
# Stats data types
# ---------------------------------------------------------------------------


@dataclass
class StatsRecord:
    loadouts: Per10[list[UnifiedIdx]]
    mmr_vals: Per10[float]
    mmr_mask: Per10[bool]
    ability_indices: torch.Tensor       # [10, 4]
    scalar_stats: torch.Tensor          # [10, 22]
    gold_t: torch.Tensor                # [10, 3]
    xp_t: torch.Tensor                 # [10, 3]
    lh_t: torch.Tensor                 # [10, 3]
    time_mask: torch.Tensor             # [10, 3]
    kill_counts: torch.Tensor           # [10, 5]
    death_counts: torch.Tensor          # [10, 5]
    damage_dealt: torch.Tensor          # [10, 5]
    gold_reasons: torch.Tensor          # [10, 14]
    xp_reasons: torch.Tensor            # [10, 6]
    ability_priorities: torch.Tensor    # [10, 4]
    # Per-(player, spell-slot) total spell damage to enemy heroes, per-min,
    # z-normalized. Summed over the 5 enemies from row.spell_damage_dealt.
    spell_damage_dealt: torch.Tensor    # [10, 4]
    match_id: int


class StatsBatch(TypedDict):
    load_idx: torch.Tensor              # [B*10, max_len]
    load_mask: torch.Tensor             # [B*10, max_len]
    mmr: torch.Tensor                   # [B, 10]
    mmr_mask: torch.Tensor              # [B, 10]
    ability_indices: torch.Tensor       # [B, 10, 4]
    scalar_stats: torch.Tensor          # [B, 10, 22]
    gold_t: torch.Tensor               # [B, 10, 3]
    xp_t: torch.Tensor                 # [B, 10, 3]
    lh_t: torch.Tensor                 # [B, 10, 3]
    time_mask: torch.Tensor             # [B, 10, 3]
    kill_counts: torch.Tensor           # [B, 10, 5]
    death_counts: torch.Tensor          # [B, 10, 5]
    damage_dealt: torch.Tensor          # [B, 10, 5]
    gold_reasons: torch.Tensor          # [B, 10, 14]
    xp_reasons: torch.Tensor            # [B, 10, 6]
    ability_priorities: torch.Tensor    # [B, 10, 4]
    spell_damage_dealt: torch.Tensor    # [B, 10, 4]


class StatsNormDict(TypedDict):
    scalar: tuple[list[float], list[float]]
    gold_t: tuple[list[float], list[float]]
    xp_t: tuple[list[float], list[float]]
    lh_t: tuple[list[float], list[float]]
    matchup: tuple[float, float]
    damage: tuple[float, float]
    gold_reasons: tuple[list[float], list[float]]
    xp_reasons: tuple[list[float], list[float]]
    priority: tuple[list[float], list[float]]
    spell_damage: tuple[float, float]
    mmr: tuple[float, float]
