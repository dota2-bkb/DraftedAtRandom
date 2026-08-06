"""Transition record for the stats-DQN. Reward is a per-stat vector."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dota2ad.core.types import PolicySample, UnifiedIdx


@dataclass
class Transition:
    sample: PolicySample
    action_idx: UnifiedIdx
    reward: torch.Tensor       # [K], z-normalized stats; zeros at non-terminal
    next_sample: PolicySample | None
