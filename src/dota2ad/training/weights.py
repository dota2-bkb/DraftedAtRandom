"""User-preference weight presets for the stats-DQN.

`STAT_SPECS` (in `eval/stats_specs.py`) defines the K stat dimensions the
Q-net predicts (13 per-focal scalar + 3 focal sums + 4 ally-team + 3
enemy-team + ally-deaths + farm-gold + farm-xp = K). Each preset here is a
length-K tensor of weights w; the inference-time composite score for any
(state, action) is `Σ_k w_k · Q_vec(s, a)[k]`.

`balanced` (`BALANCED_COMPOSITE_SIGNS`) is the hand-picked ±1 composite over
**six** individually-attributable, goal-aligned per-min stats — kills (+),
deaths (−), gold (+), xp (+), last-hits (+), hero-damage (+) — with the other
20 of the K dims zero-weighted (tower/heal/team dims are causally real but
goal-misaligned or diluting; see `stats_specs.BALANCED_COMPOSITE_SIGNS`). NOTE
this eval composite (6 dims) is *not* the same set as the model-**selection**
metric in `stats-dqn-mc` (`balanced_dims = 0,1,3,4,5,7,8,11`, which keeps the
two tower dims) — 6 for the endpoint, 8 for checkpoint selection. The other
presets dial focus toward specific play styles; `farm_focused` weights
farm-gold/farm-xp (creep+neutral income, hero-kill income excluded) over raw GPM/XPM.
"""

from __future__ import annotations

import torch

from dota2ad.eval.stats_specs import (
    BALANCED_COMPOSITE_SIGNS,
    STAT_SPECS,
)

K_STATS = len(STAT_SPECS)


def _vec(d: dict[int, float]) -> torch.Tensor:
    """Materialize a sparse {stat_idx → weight} dict into a length-K tensor."""
    w = torch.zeros(K_STATS, dtype=torch.float32)
    for k, v in d.items():
        w[k] = v
    return w


DEFAULT_BALANCED_WEIGHTS: torch.Tensor = _vec(BALANCED_COMPOSITE_SIGNS)


# Preset weight bundles for the user-preference axis. Each emphasizes a
# different style by reweighting (or zeroing) stats. Weights are unit-sign
# scale — the relative magnitudes within a preset are what matter; the
# composite score is reported in z-score space regardless.
PRESETS: dict[str, torch.Tensor] = {
    "balanced": DEFAULT_BALANCED_WEIGHTS.clone(),
    "kill_focused": _vec({
        # Sums 13/14/15 (per-enemy matchup head, summed) duplicate the per-min
        # kills/deaths/hero_damage at 0/1/7 — dropped to stop double-counting,
        # same as in balanced. teamfight_participation (12) dropped as an
        # ambiguous role descriptor. Team aggregates 16/17/20 are kept with a
        # caveat: as 1/5-diluted team stats their per-pick signal may be weak
        # (the team tower aggregates measured ≈ none — see the push preset).
        0: +2.0,   # kills/min
        1: -1.5,   # deaths/min
        7: +1.5,   # hero_damage/min
        10: +1.0,  # stuns/min
        16: +1.0,  # team kills/min (ally aggregate)
        17: +1.0,  # team hero_damage/min (ally aggregate)
        20: -1.0,  # enemy team hero_damage/min (suppression)
    }),
    "farm_focused": _vec({
        24: +2.0,  # farm gold/min (creep+neutral, excl hero-kill)
        25: +2.0,  # farm xp/min (creep, excl hero-kill)
        5: +1.5,   # last_hits/min
        6: +0.5,   # denies/min (low-information lane stat: small weight)
        1: -1.0,   # deaths/min (less critical but still avoid)
    }),
    "support": _vec({
        2: +1.5,   # assists/min
        9: +1.5,   # hero_healing/min
        10: +1.5,  # stuns/min
        12: +1.5,  # teamfight_participation
        1: -1.0,   # deaths/min (focal)
        16: +1.0,  # team kills/min (ally aggregate: enables ally kills)
        17: +1.0,  # team hero_damage/min (ally aggregate: amplifies ally damage)
        23: -1.0,  # team deaths/min (ally aggregate: saves/heals lower ally deaths)
    }),
    "push": _vec({
        # Tower output is a team OUTCOME you happen to deal — causally real but
        # goal-misaligned (it ranks pushers like Arc Warden above abilities), so
        # the focal tower dims are kept at low weight and the profile is
        # anchored on the individually-attributable, goal-aligned push
        # contributors. The team / enemy tower aggregates are dropped: at 1/5
        # dilution they carry ~no per-pick causal signal (held-out team
        # tower_damage Q1−Q4 t ≈ 0).
        7: +1.5,   # hero_damage/min (win fights → towers fall) — the anchor
        3: +1.0,   # gold_per_min (pushing items)
        8: +1.0,   # tower_damage/min
        11: +1.0,  # tower_kills/min
    }),
}


def get_preset(name: str) -> torch.Tensor:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name].clone()
