"""Runtime app context bundling mutable state + immutable lookups + model handle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import torch

from dota2ad.models import BehaviorPolicy
from dota2ad.suggest import StatsDQNSuggester

from dota2ad.inference.server.lookups import Lookups
from dota2ad.inference.server.state import DraftState


@dataclass
class AppContext:
    state: DraftState
    lookups: Lookups
    policy: BehaviorPolicy
    device: torch.device
    lock: asyncio.Lock
    # Selectable vector-Q recommenders, e.g. {"q": ..., "trial": ...}. "bc" is
    # implicit (rank by BC softmax). `active_recommender` drives the ranking; when it
    # is "bc" (or the named variant is missing) the server falls back to BC softmax.
    recommenders: dict[str, StatsDQNSuggester] = field(default_factory=dict)
    active_recommender: str = "bc"
    preset_name: str = "balanced"
    # reweight-BC tilt strength β: rank by  log π_BC(a) + β·ẑ(Trial value).  Only used
    # when active_recommender == "reweight_bc"; β=0 ⇒ pure BC (downside-protected).
    reweight_beta: float = 1.0
    stat_weights: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0),  # populated at bootstrap
    )

    @property
    def stats_dqn(self) -> StatsDQNSuggester | None:
        """The active vector-Q suggester (Trial's value net when reweight-BC is active),
        or None when BC softmax is active."""
        key = "trial" if self.active_recommender == "reweight_bc" else self.active_recommender
        return self.recommenders.get(key)

    @property
    def available_recommenders(self) -> list[str]:
        # reweight-BC (BC prior + Trial causal tilt) is offered whenever Trial is loaded.
        extra = ["reweight_bc"] if "trial" in self.recommenders else []
        return ["bc", *self.recommenders, *extra]
