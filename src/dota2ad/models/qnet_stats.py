"""StatsQNet: Q-network with a per-stat vector head.

Predicts E[stat_k | do(focal action = a), s] for K stats per (state, action).
Combined to a scalar score at inference via user-preference weights w ∈ R^K
(default = `BALANCED_COMPOSITE_SIGNS`, the hand-picked ±1 composite over six
goal-aligned per-min stats; see `training/weights.py`).

Inherits BehaviorPolicy's state encoder (loadout/pool SetTransformers, MMR
gating, history transformer, fusion) plus the candidate-token features
(embedding, type embed, state-x-candidate fusion). Only the final layer of
`score_mlp` differs — it outputs K stats instead of 1 scalar — and there's
no sigmoid (targets are z-normalized stats, not bounded probabilities).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from dota2ad.core.types import PolicyBatch
from dota2ad.models.policy import BehaviorPolicy


class QNetStats(BehaviorPolicy):
    def __init__(
        self,
        vocab_size: int,
        d: int = 64,
        n_heads: int = 4,
        k_stats: int = 16,
    ):
        super().__init__(vocab_size, d=d, n_heads=n_heads)
        self.k_stats = k_stats
        # Stats-DQN keeps BC's encoder but not the random "timeout" output head
        # — the focal player is always deliberate at decision time. Drop the
        # inherited head so the Q-net stays a pure vector-Q model.
        del self.random_logit_head
        # Replace the final 1-output layer with a K-output one. The first two
        # layers (`Linear(5d, 2d) → ReLU`) are kept from BehaviorPolicy and
        # remain warm-startable from a BC checkpoint.
        self.score_mlp = nn.Sequential(
            nn.Linear(5 * d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, k_stats),
        )

    def forward(self, batch: PolicyBatch) -> torch.Tensor:  # type: ignore[override]
        """Returns Q vectors per (state, action).

        Shape: [B, vocab_size, K]. Infeasible vocab indices have all K
        entries at -inf so that any scalar reduction (e.g. composite by
        user weights) keeps the infeasible action at -inf and argmax
        skips it.
        """
        z = self.encode_state(batch)                             # [B, d]
        B = z.shape[0]
        cand_idx = batch["cand_idx"]                             # [B, C]
        cand_mask = batch["cand_mask"]                           # [B, C]
        e_a = self.embed(cand_idx)                               # [B, C, d]
        type_vec = self.cand_type_embed(batch["cand_type"])      # [B, C, d]
        z_exp = z.unsqueeze(1).expand_as(e_a)                    # [B, C, d]
        feats = torch.cat([z_exp, e_a, z_exp * e_a, z_exp - e_a, type_vec], dim=-1)
        cand_scores = self.score_mlp(feats)                      # [B, C, K]
        scores = torch.full(
            (B, self.vocab_size, self.k_stats), float("-inf"), device=z.device,
        )
        b_ix, c_ix = cand_mask.nonzero(as_tuple=True)
        scores[b_ix, cand_idx[b_ix, c_ix]] = cand_scores[b_ix, c_ix]
        return scores

    @classmethod
    def warm_start_from_policy(
        cls,
        path: Path | str,
        vocab_size: int,
        k_stats: int,
        device: torch.device,
    ) -> QNetStats:
        """Warm-start from a BehaviorPolicy checkpoint. The final layer of
        score_mlp has different output dim (K instead of 1) and QNetStats has
        no random_logit_head — those BC keys are dropped from the load.
        Everything else (encoder, candidate features, score_mlp's first two
        layers) carries over."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            vocab_size,
            d=ckpt["d"],
            n_heads=ckpt["n_heads"],
            k_stats=k_stats,
        ).to(device)
        state = {
            k: v for k, v in ckpt["state_dict"].items()
            if not k.startswith("score_mlp.2.")
            and not k.startswith("random_logit_head.")
        }
        info = model.load_state_dict(state, strict=False)
        # The two we expect to miss: the reinit'd K-output head.
        expected_missing = {"score_mlp.2.weight", "score_mlp.2.bias"}
        if set(info.missing_keys) - expected_missing:
            raise RuntimeError(f"Unexpected missing keys: {info.missing_keys}")
        if info.unexpected_keys:
            raise RuntimeError(f"Unexpected keys: {info.unexpected_keys}")
        return model

    @classmethod
    def load_from_ckpt(
        cls, path: Path | str, vocab_size: int, device: torch.device,
    ) -> QNetStats:
        """Load a previously-trained StatsQNet checkpoint (full state_dict)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            vocab_size,
            d=ckpt["d"],
            n_heads=ckpt["n_heads"],
            k_stats=ckpt["k_stats"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model
