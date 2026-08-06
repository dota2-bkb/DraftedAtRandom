"""Feature encoding: loadouts, MMR, policy samples, set padding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch

from dota2ad.core.draft_logic import idx, turn_to_pick_slot
from dota2ad.core.types import (
    HistoryEvent,
    LabeledPolicySample,
    NUM_PLAYERS,
    Per10,
    PickSlot,
    PlayerPickState,
    PolicySample,
    Turn,
    TurnRow,
    UnifiedIdx,
    VocabKey,
    Vocabs,
)


EMPTY_KEY = VocabKey("<empty>")


def encode_mmr(
    mmr: Per10[float | None], mean: float, std: float,
) -> tuple[Per10[float], Per10[bool]]:
    """Encode per-player MMR with z-score normalization: returns (vals, mask) length-10 tuples."""
    vals = cast(
        Per10[float],
        tuple((v - mean) / std if v is not None else 0.0 for v in mmr),
    )
    mask = cast(Per10[bool], tuple(v is not None for v in mmr))
    return vals, mask


def pad_sets(all_sets: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sets into (indices, mask) tensors.

    Returns:
        indices: [n_sets, max_len] padded with 0
        mask: [n_sets, max_len] True = valid element
    """
    n = len(all_sets)
    max_len = max((len(s) for s in all_sets), default=0)
    if max_len == 0:
        return torch.zeros(n, 0, dtype=torch.long), torch.zeros(n, 0, dtype=torch.bool)
    padded = [list(s) + [0] * (max_len - len(s)) for s in all_sets]
    indices = torch.tensor(padded, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in all_sets], dtype=torch.long)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return indices, mask


def encode_loadout(state: PlayerPickState, vocabs: Vocabs) -> list[UnifiedIdx]:
    """Merge a single player's hero + basics + ult into one flat list."""
    items: list[UnifiedIdx] = []
    if state.hero is not None:
        items.append(idx(vocabs, state.hero, "h"))
    items.extend(idx(vocabs, b, "a") for b in state.basics)
    if state.ult is not None:
        items.append(idx(vocabs, state.ult, "a"))
    if not items:
        items.append(vocabs.draft_id_to_index[EMPTY_KEY])
    return items


def loadout_random_flags(
    state: PlayerPickState, player_events: list[HistoryEvent],
) -> tuple[list[bool], list[bool]]:
    """Parallel to encode_loadout's output: per-position (is_random,
    is_disconnected) flag lists.

    `is_random` marks positions filled by a server timeout of either kind;
    `is_disconnected` marks the subset whose picker was offline at the
    timeout — so the encoder distinguishes three position classes:
    deliberate, online random, disconnected random.

    Walks the player's history events in pick order, classifies each as
    hero/basic/ult by matching against the final state, and emits flags
    in encode_loadout's canonical hero→basics→ult order.
    """
    hero_f: tuple[bool, bool] | None = None
    basics_f: list[tuple[bool, bool]] = []
    ult_f: tuple[bool, bool] | None = None
    ult_id = state.ult
    for event in player_events:
        ev_f = (event.is_random, event.picker_disconnected)
        if event.hero_id is not None:
            hero_f = ev_f
        elif event.draft_ability_id is not None:
            if ult_id is not None and event.draft_ability_id == ult_id:
                ult_f = ev_f
            else:
                basics_f.append(ev_f)

    flags: list[tuple[bool, bool]] = []
    if state.hero is not None:
        flags.append(hero_f if hero_f is not None else (False, False))
    flags.extend(basics_f)
    if state.ult is not None:
        flags.append(ult_f if ult_f is not None else (False, False))
    if not flags:
        flags.append((False, False))
    return [f[0] for f in flags], [f[1] for f in flags]


def encode_policy_sample(
    row: TurnRow,
    vocabs: Vocabs,
    history: list[HistoryEvent],
    mmr_mean: float,
    mmr_std: float,
) -> PolicySample:
    pp = row.player_picks
    pick_slot = row.pick_slot

    # 10 merged loadouts (one per player)
    loadouts = cast(
        Per10[list[UnifiedIdx]],
        tuple(encode_loadout(pp[PickSlot(ps)], vocabs) for ps in range(NUM_PLAYERS)),
    )

    # Per-position random flags, parallel to loadouts. Bucket history by
    # player slot (events come in turn order, so per-player order is preserved).
    events_by_slot: list[list[HistoryEvent]] = [[] for _ in range(NUM_PLAYERS)]
    for t, event in enumerate(history):
        events_by_slot[turn_to_pick_slot(Turn(t))].append(event)
    per_player_flags = [
        loadout_random_flags(pp[PickSlot(ps)], events_by_slot[ps])
        for ps in range(NUM_PLAYERS)
    ]
    loadouts_is_random = cast(Per10[list[bool]], tuple(f[0] for f in per_player_flags))
    loadouts_is_disconnected = cast(Per10[list[bool]], tuple(f[1] for f in per_player_flags))

    # Single merged pool
    pool: list[UnifiedIdx] = (
        [idx(vocabs, h, "h") for h in row.hero_pool_remaining]
        + [idx(vocabs, a, "a") for a in row.basic_pool_remaining]
        + [idx(vocabs, a, "a") for a in row.ult_pool_remaining]
    )

    mmr_vals, mmr_mask = encode_mmr(row.mmr, mmr_mean, mmr_std)

    # Build candidate list with type tags, then derive feasible_mask
    vocab_size = len(vocabs.draft_id_to_index)
    cand_idx: list[UnifiedIdx] = []
    cand_type: list[Literal[0, 1, 2]] = []
    actor_state = pp[pick_slot]
    if actor_state.hero is None:
        for h in row.hero_pool_remaining:
            cand_idx.append(idx(vocabs, h, "h"))
            cand_type.append(0)
    if len(actor_state.basics) < 3:
        for a in row.basic_pool_remaining:
            cand_idx.append(idx(vocabs, a, "a"))
            cand_type.append(1)
    if actor_state.ult is None:
        for a in row.ult_pool_remaining:
            cand_idx.append(idx(vocabs, a, "a"))
            cand_type.append(2)
    feasible_mask = torch.zeros(vocab_size, dtype=torch.bool)
    if cand_idx:
        feasible_mask[cand_idx] = True

    turn = row.turn
    round_idx = turn // 10
    side_idx: Literal[0, 1] = cast(Literal[0, 1], 0 if pick_slot % 2 == 0 else 1)
    hero_filled = actor_state.hero is not None
    basics_count = len(actor_state.basics)
    ult_filled = actor_state.ult is not None
    pool_heroes = len(row.hero_pool_remaining) / 12
    pool_basics = len(row.basic_pool_remaining) / 36
    pool_ults = len(row.ult_pool_remaining) / 12

    # Encode history
    hist_idx: list[UnifiedIdx] = []
    hist_slot: list[PickSlot] = []
    for t, event in enumerate(history):
        if event.hero_id is not None:
            hist_idx.append(idx(vocabs, event.hero_id, "h"))
        else:
            assert event.draft_ability_id is not None
            hist_idx.append(idx(vocabs, event.draft_ability_id, "a"))
        hist_slot.append(turn_to_pick_slot(Turn(t)))

    return PolicySample(
        loadouts=loadouts,
        loadouts_is_random=loadouts_is_random,
        loadouts_is_disconnected=loadouts_is_disconnected,
        pool=pool,
        pick_slot=pick_slot,
        mmr_vals=mmr_vals,
        mmr_mask=mmr_mask,
        feasible_mask=feasible_mask,
        cand_idx=cand_idx,
        cand_type=cand_type,
        turn=turn,
        round_idx=round_idx,
        side_idx=side_idx,
        hero_filled=hero_filled,
        basics_count=basics_count,
        ult_filled=ult_filled,
        pool_heroes=pool_heroes,
        pool_basics=pool_basics,
        pool_ults=pool_ults,
        hist_idx=hist_idx,
        hist_slot=hist_slot,
    )


def encode_labeled_policy_sample(
    row: TurnRow,
    vocabs: Vocabs,
    history: list[HistoryEvent],
    mmr_mean: float,
    mmr_std: float,
) -> LabeledPolicySample:
    """Encode a row whose action (hero_id or draft_ability_id) is set."""
    base = encode_policy_sample(row, vocabs, history, mmr_mean, mmr_std)
    if row.hero_id is not None:
        action_idx = idx(vocabs, row.hero_id, "h")
    elif row.draft_ability_id is not None:
        action_idx = idx(vocabs, row.draft_ability_id, "a")
    else:
        raise ValueError("Cannot label sample: row has no hero_id or draft_ability_id")
    return LabeledPolicySample(
        loadouts=base.loadouts,
        loadouts_is_random=base.loadouts_is_random,
        loadouts_is_disconnected=base.loadouts_is_disconnected,
        pool=base.pool,
        pick_slot=base.pick_slot,
        mmr_vals=base.mmr_vals,
        mmr_mask=base.mmr_mask,
        feasible_mask=base.feasible_mask,
        cand_idx=base.cand_idx,
        cand_type=base.cand_type,
        turn=base.turn,
        round_idx=base.round_idx,
        side_idx=base.side_idx,
        hero_filled=base.hero_filled,
        basics_count=base.basics_count,
        ult_filled=base.ult_filled,
        pool_heroes=base.pool_heroes,
        pool_basics=base.pool_basics,
        pool_ults=base.pool_ults,
        hist_idx=base.hist_idx,
        hist_slot=base.hist_slot,
        action_idx=action_idx,
    )
