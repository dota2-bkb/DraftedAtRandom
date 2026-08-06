"""Mutable draft state for the inference server."""

from __future__ import annotations

import msgspec

from dota2ad.core import (
    AbilityId,
    HeroId,
    HistoryEvent,
    Per10,
    PickSlot,
    PlayerPickState,
    Turn,
    TurnRow,
    per10,
    turn_to_pick_slot,
)
from dota2ad.suggest import DraftState as RolloutDraftState, apply_action


NUM_PLAYERS = 10


class DraftState(msgspec.Struct):
    player_picks: dict[PickSlot, PlayerPickState]
    hero_pool_remaining: list[HeroId]
    basic_pool_remaining: list[AbilityId]
    ult_pool_remaining: list[AbilityId]
    hero_pool_all: list[HeroId | None]
    basic_pool_all: list[AbilityId | None]
    ult_pool_all: list[AbilityId | None]
    mmr: Per10[float | None]
    pick_slot: PickSlot
    turn: Turn
    hero_id: HeroId | None
    draft_ability_id: AbilityId | None
    action_key: str
    is_random: bool
    radiant_win: bool
    match_id: int
    history: list[HistoryEvent]


def make_empty_state() -> DraftState:
    return DraftState(
        player_picks={PickSlot(i): PlayerPickState(hero=None, basics=[], ult=None) for i in range(NUM_PLAYERS)},
        hero_pool_remaining=[],
        basic_pool_remaining=[],
        ult_pool_remaining=[],
        hero_pool_all=[],
        basic_pool_all=[],
        ult_pool_all=[],
        mmr=per10(None),
        pick_slot=PickSlot(0),
        turn=Turn(0),
        hero_id=None,
        draft_ability_id=None,
        action_key="",
        is_random=False,
        radiant_win=False,
        match_id=0,
        history=[],
    )


def set_at[T](lst: list[T | None], idx: int, val: T | None) -> None:
    """Set lst[idx] = val, extending with None if needed."""
    while len(lst) <= idx:
        lst.append(None)
    lst[idx] = val


def state_to_turn_row(state: DraftState) -> TurnRow:
    return TurnRow(
        match_id=state.match_id,
        turn=state.turn,
        pick_slot=state.pick_slot,
        hero_id=state.hero_id,
        draft_ability_id=state.draft_ability_id,
        action_key=state.action_key,
        is_random=state.is_random,
        picker_disconnected=False,
        radiant_win=state.radiant_win,
        mmr=state.mmr,
        hero_pool_remaining=state.hero_pool_remaining,
        basic_pool_remaining=state.basic_pool_remaining,
        ult_pool_remaining=state.ult_pool_remaining,
        player_picks=state.player_picks,
    )


def advance_state(state: DraftState, is_random: bool = False) -> None:
    """Apply hero_id / draft_ability_id and advance turn (mutates in place).

    `is_random` records whether the user labeled this pick as a timeout —
    propagates into the HistoryEvent so the state encoder can condition on
    the deliberate vs random conditional for downstream Q queries.
    """
    state.pick_slot = turn_to_pick_slot(state.turn)
    draft = RolloutDraftState.from_row(state_to_turn_row(state), list(state.history))
    result = apply_action(draft, is_random=is_random)
    new_row = result.to_row()

    state.match_id = new_row.match_id
    state.turn = new_row.turn
    state.pick_slot = new_row.pick_slot
    state.hero_id = new_row.hero_id
    state.draft_ability_id = new_row.draft_ability_id
    state.action_key = new_row.action_key
    state.is_random = new_row.is_random
    state.radiant_win = new_row.radiant_win
    state.mmr = new_row.mmr
    state.hero_pool_remaining = list(new_row.hero_pool_remaining)
    state.basic_pool_remaining = list(new_row.basic_pool_remaining)
    state.ult_pool_remaining = list(new_row.ult_pool_remaining)
    state.player_picks = {
        slot: PlayerPickState(hero=p.hero, basics=list(p.basics), ult=p.ult)
        for slot, p in new_row.player_picks.items()
    }
    state.history = list(result.history)
