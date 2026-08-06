"""Random-pick tuples — the natural-experiment data behind the stats causal eval.

A `RandomPickTuple` captures one server-timeout pick (exogenous action with a
known propensity — the decompiled coin-then-uniform `P_mech`; consumers weight by
`iw_to_uniform` to target the uniform estimand). `extract_tuples` pulls them from
matches. These feed the causal-ranking test (`eval/causal_rank.py`). See
REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from dota2ad.core import (
    NUM_PLAYERS,
    PickSlot,
    Turn,
    encode_policy_sample,
)
from dota2ad.core.draft_logic import idx, replay_complete, replay_to_turn
from dota2ad.core.encoding import encode_loadout, encode_mmr
from dota2ad.core.types import (
    MatchRow,
    Per10,
    PolicySample,
    UnifiedIdx,
    Vocabs,
)


@dataclass
class RandomPickTuple:
    """A single random-pick evaluation tuple."""
    sample: PolicySample           # state s_i
    action_idx: UnifiedIdx         # realized random action a_i
    n_feasible: int                # |feasible(s_i)|
    turn: int                      # draft position 0..49 (pre-pick covariate)
    focal_slot: PickSlot
    picker_disconnected: bool      # focal was offline at the timeout (disengagement signal)
    focal_team_won: float          # 0.0 or 1.0
    final_loadouts_other: list[list[UnifiedIdx]]  # all 10 terminal loadouts (focal's included)
    mmr_vals: Per10[float]
    mmr_mask: Per10[bool]
    match_id: int


def extract_tuples(
    matches: list[MatchRow],
    vocabs: Vocabs,
    mmr_mean: float,
    mmr_std: float,
    disconnect_only: bool = False,
    online_only: bool = False,
) -> list[RandomPickTuple]:
    """One Tuple per random pick in the matches (any of the 10 seats counts).

    Every "random" pick in AD is a server-side timeout. `disconnect_only`
    restricts to timeouts where the picker was offline (`picker_disconnected`),
    `online_only` to the complement.
    """
    assert not (disconnect_only and online_only), "pass at most one of disconnect_only / online_only"
    tuples: list[RandomPickTuple] = []
    for match in matches:
        if not any(e.is_random for e in match.history):
            continue
        final_pp = replay_complete(match)
        final_loadouts = [
            encode_loadout(final_pp[PickSlot(ps)], vocabs) for ps in range(NUM_PLAYERS)
        ]
        mmr_vals, mmr_mask = encode_mmr(match.mmr, mmr_mean, mmr_std)
        for t, event in enumerate(match.history):
            if not event.is_random:
                continue
            if disconnect_only and not event.picker_disconnected:
                continue
            if online_only and event.picker_disconnected:
                continue
            row = replay_to_turn(match, Turn(t))
            sample = encode_policy_sample(
                row, vocabs, history=match.history[:t],
                mmr_mean=mmr_mean, mmr_std=mmr_std,
            )
            if event.hero_id is not None:
                a_idx = idx(vocabs, event.hero_id, "h")
            else:
                assert event.draft_ability_id is not None
                a_idx = idx(vocabs, event.draft_ability_id, "a")
            focal_slot = row.pick_slot
            focal_is_radiant = (focal_slot % 2 == 0)
            focal_won = (match.radiant_win and focal_is_radiant) or (
                not match.radiant_win and not focal_is_radiant
            )
            tuples.append(RandomPickTuple(
                sample=sample,
                action_idx=a_idx,
                n_feasible=len(sample.cand_idx),
                turn=t,
                focal_slot=focal_slot,
                picker_disconnected=event.picker_disconnected,
                focal_team_won=1.0 if focal_won else 0.0,
                final_loadouts_other=final_loadouts,
                mmr_vals=mmr_vals,
                mmr_mask=mmr_mask,
                match_id=match.match_id,
            ))
    return tuples
