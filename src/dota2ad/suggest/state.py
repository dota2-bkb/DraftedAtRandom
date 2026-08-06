"""Draft state: mutable representation for rollout/simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from dota2ad.core.types import (
    AbilityId,
    HeroId,
    HistoryEvent,
    Per10,
    PickSlot,
    PlayerPickState,
    Turn,
    TurnRow,
    UnifiedIdx,
    VocabKey,
    Vocabs,
)
from dota2ad.core.draft_logic import turn_to_pick_slot


@dataclass
class DraftState:
    turn: Turn
    pick_slot: PickSlot
    player_picks: dict[PickSlot, PlayerPickState]
    hero_pool_remaining: list[HeroId]
    basic_pool_remaining: list[AbilityId]
    ult_pool_remaining: list[AbilityId]
    action_key: str
    mmr: Per10[float | None]
    history: list[HistoryEvent] = field(default_factory=list)
    hero_id: HeroId | None = None
    draft_ability_id: AbilityId | None = None
    radiant_win: bool = False
    match_id: int = 0

    @classmethod
    def from_row(cls, row: TurnRow, history: list[HistoryEvent] | None = None) -> DraftState:
        picks: dict[PickSlot, PlayerPickState] = {
            slot: PlayerPickState(hero=v.hero, basics=list(v.basics), ult=v.ult)
            for slot, v in row.player_picks.items()
        }
        return cls(
            turn=row.turn,
            pick_slot=row.pick_slot,
            player_picks=picks,
            hero_pool_remaining=list(row.hero_pool_remaining),
            basic_pool_remaining=list(row.basic_pool_remaining),
            ult_pool_remaining=list(row.ult_pool_remaining),
            mmr=row.mmr,
            history=list(history) if history is not None else [],
            hero_id=row.hero_id,
            draft_ability_id=row.draft_ability_id,
            action_key=row.action_key,
            radiant_win=row.radiant_win,
            match_id=row.match_id,
        )

    def to_row(self) -> TurnRow:
        """Convert back to TurnRow for encoding functions."""
        pp: dict[PickSlot, PlayerPickState] = {
            slot: PlayerPickState(hero=ps.hero, basics=list(ps.basics), ult=ps.ult)
            for slot, ps in self.player_picks.items()
        }
        return TurnRow(
            match_id=self.match_id,
            turn=self.turn,
            pick_slot=self.pick_slot,
            hero_id=self.hero_id,
            draft_ability_id=self.draft_ability_id,
            action_key=self.action_key,
            is_random=False,
            picker_disconnected=False,
            radiant_win=self.radiant_win,
            mmr=self.mmr,
            hero_pool_remaining=self.hero_pool_remaining,
            basic_pool_remaining=self.basic_pool_remaining,
            ult_pool_remaining=self.ult_pool_remaining,
            player_picks=pp,
        )


_REV_VOCAB: dict[int, dict[UnifiedIdx, VocabKey]] = {}


def _reverse_draft_vocab(vocabs: Vocabs) -> dict[UnifiedIdx, VocabKey]:
    """Cached index→"kind:id" map. Rebuilding it per call (per draft, per rollout
    turn) dominated `make_forced_state`; the map is fixed for a run."""
    rev = _REV_VOCAB.get(id(vocabs))
    if rev is None:
        rev = {i: key for key, i in vocabs.draft_id_to_index.items()}
        _REV_VOCAB[id(vocabs)] = rev
    return rev


def _copy_state(s: DraftState) -> DraftState:
    """Fast independent copy of the mutated containers only — replaces
    `copy.deepcopy`, which recursed the whole struct on every rollout step.
    History events are append-only (never mutated), so the list is shallow-copied;
    each PlayerPickState and its basics list are rebuilt (they are mutated in place)."""
    return DraftState(
        turn=s.turn,
        pick_slot=s.pick_slot,
        player_picks={slot: PlayerPickState(hero=v.hero, basics=list(v.basics), ult=v.ult)
                      for slot, v in s.player_picks.items()},
        hero_pool_remaining=list(s.hero_pool_remaining),
        basic_pool_remaining=list(s.basic_pool_remaining),
        ult_pool_remaining=list(s.ult_pool_remaining),
        action_key=s.action_key,
        mmr=s.mmr,
        history=list(s.history),
        hero_id=s.hero_id,
        draft_ability_id=s.draft_ability_id,
        radiant_win=s.radiant_win,
        match_id=s.match_id,
    )


def _apply_action_inplace(new: DraftState, is_random: bool = False) -> None:
    """Advance `new` by its pending action, mutating in place. The caller owns a
    fresh copy (`_copy_state`), so no copy is made here."""
    actor = new.player_picks[new.pick_slot]

    # Record this action in history
    new.history.append(HistoryEvent(
        hero_id=new.hero_id,
        draft_ability_id=new.draft_ability_id,
        action_key=new.action_key,
        is_random=is_random,
        picker_disconnected=False,
    ))

    if new.hero_id is not None:
        new.hero_pool_remaining.remove(new.hero_id)
        actor.hero = new.hero_id
    else:
        assert new.draft_ability_id is not None
        draft_ability_id = new.draft_ability_id
        if draft_ability_id in new.ult_pool_remaining:
            new.ult_pool_remaining.remove(draft_ability_id)
            actor.ult = draft_ability_id
        else:
            new.basic_pool_remaining.remove(draft_ability_id)
            actor.basics.append(draft_ability_id)

    new.turn = Turn(new.turn + 1)
    new.pick_slot = turn_to_pick_slot(new.turn)
    new.hero_id = None
    new.draft_ability_id = None


def apply_action(state: DraftState, is_random: bool = False) -> DraftState:
    """Apply the action in state to produce a new state for the next turn.

    `is_random` is recorded in the history event for the pick: True iff the
    pick was either a server forced-random draw (simulator non-focal random
    branch, drawn from P_mech) or a user-labeled timeout (inference). Drives the
    per-loadout-position is_random feature in the state encoder.
    """
    new = _copy_state(state)
    _apply_action_inplace(new, is_random=is_random)
    return new


def make_forced_state(
    state: DraftState, action_idx: UnifiedIdx, vocabs: Vocabs,
    is_random: bool = False,
) -> DraftState:
    """Force a specific action (by unified idx) on the current state.

    `is_random` propagates into the recorded HistoryEvent — set True for
    server forced-random draws (simulator, P_mech) or user-labeled timeouts
    (inference)."""
    raw_key = _reverse_draft_vocab(vocabs)[action_idx]
    kind, raw_id_str = raw_key.split(":")
    raw_id = int(raw_id_str)

    new = _copy_state(state)
    if kind == "h":
        new.hero_id = HeroId(raw_id)
    else:
        new.draft_ability_id = AbilityId(raw_id)
    _apply_action_inplace(new, is_random=is_random)
    return new
