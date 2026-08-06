"""Dataset classes for training."""

from __future__ import annotations

from torch.utils.data import Dataset

from dota2ad.core.draft_logic import replay_to_turn
from dota2ad.core.encoding import encode_labeled_policy_sample
from typing import cast

from dota2ad.core.types import (
    LabeledPolicySample,
    MatchRow,
    Turn,
    UnifiedIdx,
    Vocabs,
)


class PolicyDataset(Dataset[LabeledPolicySample]):
    def __init__(self, matches: list[MatchRow], vocabs: Vocabs,
                 mmr_mean: float, mmr_std: float, random_class_idx: int):
        """Online random picks are relabeled with random_class_idx (typically
        vocab_size), the "random" timeout class the BC emits at its last output
        index. The realized action was server-uniform; the decision the policy
        must predict is "this player will let the timer expire."

        Disconnected-random turns (is_random && picker_disconnected) are not
        training samples at all: the timeout fired because the player was
        offline, independent of the state, so there is no decision to
        predict. Those picks stay visible as context — other samples carry
        them in the per-position loadout flags.
        """
        self.samples: list[LabeledPolicySample] = []
        for match in matches:
            for turn in range(len(match.history)):
                event = match.history[turn]
                if event.is_random and event.picker_disconnected:
                    continue
                row = replay_to_turn(match, Turn(turn))
                sample = encode_labeled_policy_sample(
                    row, vocabs, history=match.history[:turn],
                    mmr_mean=mmr_mean, mmr_std=mmr_std,
                )
                if event.is_random:
                    sample.action_idx = cast(UnifiedIdx, random_class_idx)
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> LabeledPolicySample:
        return self.samples[i]
