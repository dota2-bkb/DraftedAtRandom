"""Draft logic: turn ordering, replay helpers, split utilities."""

from __future__ import annotations

import random

from typing import Literal

from dota2ad.core.types import (
    MatchRow,
    NUM_PLAYERS,
    PickSlot,
    PlayerPickState,
    Turn,
    TurnRow,
    UnifiedIdx,
    VocabKey,
    Vocabs,
)


def idx(vocabs: Vocabs, raw_id: int, kind: Literal["h", "a"]) -> UnifiedIdx:
    """Look up unified index. kind is 'h' (hero) or 'a' (ability)."""
    return vocabs.draft_id_to_index[VocabKey(f"{kind}:{raw_id}")]


def pick_slot_team(ps: PickSlot) -> Literal["radiant", "dire"]:
    return "radiant" if ps % 2 == 0 else "dire"


def turn_to_pick_slot(turn: Turn) -> PickSlot:
    """Snake draft: which pick_slot acts at a given turn."""
    rnd = turn // 10
    pos = turn % 10
    return PickSlot(pos if rnd % 2 == 0 else 9 - pos)


def replay_to_turn(match: MatchRow, turn: Turn) -> TurnRow:
    """Replay history[:turn] to reconstruct state, return TurnRow for that turn."""
    ult_ids = set(match.ult_pool)
    hero_pool = list(match.hero_pool)
    basic_pool = list(match.basic_pool)
    ult_pool = list(match.ult_pool)
    player_picks: dict[PickSlot, PlayerPickState] = {
        PickSlot(ps): PlayerPickState(hero=None, basics=[], ult=None)
        for ps in range(NUM_PLAYERS)
    }

    for t in range(turn):
        event = match.history[t]
        ps = turn_to_pick_slot(Turn(t))
        if event.hero_id is not None:
            if event.hero_id in hero_pool:
                hero_pool.remove(event.hero_id)
            player_picks[ps].hero = event.hero_id
        else:
            aid = event.draft_ability_id
            assert aid is not None
            if aid in basic_pool:
                basic_pool.remove(aid)
            elif aid in ult_pool:
                ult_pool.remove(aid)
            if aid in ult_ids:
                player_picks[ps].ult = aid
            else:
                player_picks[ps].basics.append(aid)

    event = match.history[turn]
    return TurnRow(
        match_id=match.match_id,
        turn=Turn(turn),
        pick_slot=turn_to_pick_slot(Turn(turn)),
        hero_id=event.hero_id,
        draft_ability_id=event.draft_ability_id,
        action_key=event.action_key,
        is_random=event.is_random,
        picker_disconnected=event.picker_disconnected,
        radiant_win=match.radiant_win,
        mmr=match.mmr,
        hero_pool_remaining=hero_pool,
        basic_pool_remaining=basic_pool,
        ult_pool_remaining=ult_pool,
        player_picks=player_picks,
    )


def replay_complete(match: MatchRow) -> dict[PickSlot, PlayerPickState]:
    """Replay full history, return final player loadouts."""
    ult_ids = set(match.ult_pool)
    player_picks: dict[PickSlot, PlayerPickState] = {
        PickSlot(ps): PlayerPickState(hero=None, basics=[], ult=None)
        for ps in range(NUM_PLAYERS)
    }

    for t, event in enumerate(match.history):
        ps = turn_to_pick_slot(Turn(t))
        if event.hero_id is not None:
            player_picks[ps].hero = event.hero_id
        else:
            aid = event.draft_ability_id
            assert aid is not None
            if aid in ult_ids:
                player_picks[ps].ult = aid
            else:
                player_picks[ps].basics.append(aid)

    return player_picks


def make_split(
    match_ids: list[int], val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Deterministically partition match IDs into (val, test); train = the rest.

    One seeded shuffle of the sorted IDs, contiguous slices: val = ids[:nv],
    test = ids[nv:nv+nt]. val is for selection (best epoch, hyperparameters,
    calibration, diagnostics); test is read once, for final numbers only.
    """
    ids = sorted(match_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    n_test = max(1, int(len(ids) * test_frac))
    return set(ids[:n_val]), set(ids[n_val:n_val + n_test])
