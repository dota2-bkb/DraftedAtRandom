"""BehaviorPolicy model."""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from dota2ad.core.types import PolicyBatch
from dota2ad.models.set_transformer import SetTransformer


class BehaviorPolicy(nn.Module):
    mmr_mean: torch.Tensor
    mmr_std: torch.Tensor

    def __init__(
        self,
        vocab_size: int,
        d: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.n_heads = n_heads
        # State-rarity band calibration (held-out support quantiles), attached at
        # load time from the checkpoint; see suggest/density.compute_support_quantiles.
        self.density_support_q: list[float] | None = None

        self.embed = nn.Embedding(vocab_size, d)
        nn.init.zeros_(self.embed.weight[-1])  # <empty> token
        self._empty_idx = vocab_size - 1
        self.embed.weight.register_hook(
            lambda grad: grad.index_fill_(0, torch.tensor([self._empty_idx], device=grad.device), 0)
        )

        self.register_buffer("mmr_mean", torch.zeros(()))
        self.register_buffer("mmr_std", torch.ones(()))

        self.st_loadout = SetTransformer(d, d, n_heads)
        self.st_pool = SetTransformer(d, d, n_heads)

        # Per-position random-pick embeddings added at loadout positions:
        # `is_random_embed` where `load_is_random` is True (any server
        # timeout), plus `is_disconnected_embed` where `load_is_disconnected`
        # is True (the picker was offline at the timeout) — compositional, a
        # disconnected random gets both. Serving and the rollout simulator
        # cannot observe disconnection and only ever set is_random.
        self.is_random_embed = nn.Parameter(torch.zeros(d))
        self.is_disconnected_embed = nn.Parameter(torch.zeros(d))

        # Gated MMR projection: h = W(x*m) + V*m + b
        self.mmr_W = nn.Linear(10, d)
        self.mmr_V = nn.Linear(10, d)

        # Two-level MMR dropout (training only)
        self.mmr_drop_group = 0.3
        self.mmr_drop_player = 0.1

        # Actor indicator — added to acting player's vector
        self.actor_embed = nn.Parameter(torch.zeros(d))

        # Context features
        self.turn_embed = nn.Embedding(50, d)
        self.round_embed = nn.Embedding(5, d)
        self.side_embed = nn.Embedding(2, d)
        self.basics_count_embed = nn.Embedding(4, d)       # 0..3
        self.occupancy_proj = nn.Linear(2, d)               # hero_filled, ult_filled
        self.pool_proj = nn.Linear(3, d)                    # pool_heroes, pool_basics, pool_ults

        state_dim = 13 * d  # 10 loadouts + 1 pool + 1 mmr + 1 context
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )

        # History encoder
        self.hist_slot_embed = nn.Embedding(10, d)
        self.hist_pos_embed = nn.Embedding(50, d)
        self.hist_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, n_heads, 2 * d, batch_first=True),
            num_layers=2,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Candidate scoring
        self.cand_type_embed = nn.Embedding(3, d)  # hero/basic/ult
        self.score_mlp = nn.Sequential(
            nn.Linear(5 * d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, 1),
        )

        # "Random" output class: forward emits vocab_size+1 log-probs; the last
        # index (vocab_size) is the probability that this player lets the timer
        # expire at this state. QNetStats drops this head — it reuses only the
        # encoder, never the random output.
        self.random_logit_head = nn.Linear(d, 1)

    def _run_st(
        self,
        st: SetTransformer,
        idx: torch.Tensor,
        mask: torch.Tensor,
        is_random: torch.Tensor | None,
        is_disconnected: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed padded indices and run through a SetTransformer instance.

        `is_random` / `is_disconnected` (same shape as idx, bool) add the
        corresponding flag embedding at positions where the flag is True;
        None means the set type carries no such flags (the pool).
        """
        x = self.embed(idx)  # [n_sets, max_len, d]
        if is_random is not None:
            x = x + is_random.unsqueeze(-1).float() * self.is_random_embed
        if is_disconnected is not None:
            x = x + is_disconnected.unsqueeze(-1).float() * self.is_disconnected_embed
        out = st(x, mask)
        # Zero out sets containing only the <empty> token
        is_empty = (mask.sum(dim=1) == 1) & (idx[:, 0] == self._empty_idx)
        if is_empty.any():
            out = out.clone()
            out[is_empty] = 0.0
        return out

    def encode_history(self, batch: PolicyBatch) -> torch.Tensor:
        """Encode pick history -> [B, d]."""
        hist_idx = batch["hist_idx"]    # [B, L]
        hist_mask = batch["hist_mask"]  # [B, L]
        L = hist_idx.shape[1]
        if L == 0:
            B = batch["mmr"].shape[0]
            return torch.zeros(B, self.d, device=batch["mmr"].device)
        tokens = (
            self.embed(hist_idx)
            + self.hist_slot_embed(batch["hist_slot"])
            + self.hist_pos_embed(torch.arange(L, device=hist_idx.device))
        )
        out = self.hist_transformer(tokens, src_key_padding_mask=~hist_mask)  # [B, L, d]
        # Mean pool over valid positions
        lengths = hist_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        pooled = (out * hist_mask.unsqueeze(-1)).sum(dim=1) / lengths  # [B, d]
        return pooled

    def encode_state(self, batch: PolicyBatch) -> torch.Tensor:
        """Encode draft state -> [B, d]."""
        B = batch["mmr"].shape[0]

        load = self._run_st(
            self.st_loadout, batch["load_idx"], batch["load_mask"],
            is_random=batch["load_is_random"],
            is_disconnected=batch["load_is_disconnected"],
        )
        load = load.view(B, 10, -1)                            # [B, 10, d]

        # Mark the acting player
        ps = batch["pick_slot"]                                 # [B]
        load[torch.arange(B, device=load.device), ps] += self.actor_embed

        pool = self._run_st(self.st_pool, batch["pool_idx"], batch["pool_mask"],
                            is_random=None, is_disconnected=None)

        # MMR: h = W(x*m) + V*m + b  (per-player, [B, 10])
        m = batch["mmr_mask"].float()
        if self.training:
            group_drop = torch.rand(B, 1, device=m.device) < self.mmr_drop_group
            player_drop = torch.rand(B, 10, device=m.device) < self.mmr_drop_player
            m = m * (~group_drop).float() * (~player_drop).float()
        x_tilde = batch["mmr"] * m
        mmr_vec = self.mmr_W(x_tilde) + self.mmr_V(m)         # [B, d]

        turn_vec = self.turn_embed(batch["turn"])
        round_vec = self.round_embed(batch["round_idx"])
        side_vec = self.side_embed(batch["side_idx"])
        basic_vec = self.basics_count_embed(batch["basics_count"])
        occ = torch.stack([batch["hero_filled"].float(), batch["ult_filled"].float()], dim=1)
        occ_vec = self.occupancy_proj(occ)
        pool_counts = torch.stack([batch["pool_heroes"].float(), batch["pool_basics"].float(), batch["pool_ults"].float()], dim=1)
        pool_vec = self.pool_proj(pool_counts)
        context_vec = turn_vec + round_vec + side_vec + basic_vec + occ_vec + pool_vec

        state = torch.cat([load.reshape(B, -1), pool, mmr_vec, context_vec], dim=1)  # [B, 13*d]
        z_state = self.state_proj(state)  # [B, d]
        z_hist = self.encode_history(batch)  # [B, d]
        return self.fusion(torch.cat([z_state, z_hist], dim=1))  # [B, d]

    @overload
    def forward(self, batch: PolicyBatch, return_z: Literal[False] = ...) -> torch.Tensor: ...
    @overload
    def forward(self, batch: PolicyBatch, return_z: Literal[True]) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(self, batch: PolicyBatch, return_z: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probabilities over actions.

        Returns [B, vocab_size + 1]: per-action log-probs plus the random
        "timeout" class at the last index (`vocab_size`).
        """
        z = self.encode_state(batch)                             # [B, d]
        B = z.shape[0]
        cand_idx = batch["cand_idx"]                             # [B, C]
        cand_mask = batch["cand_mask"]                           # [B, C]
        e_a = self.embed(cand_idx)                               # [B, C, d]
        type_vec = self.cand_type_embed(batch["cand_type"])      # [B, C, d]
        z_exp = z.unsqueeze(1).expand_as(e_a)                    # [B, C, d]
        feats = torch.cat([z_exp, e_a, z_exp * e_a, z_exp - e_a, type_vec], dim=-1)
        cand_scores = self.score_mlp(feats).squeeze(-1)          # [B, C]
        cand_scores[~cand_mask] = float("-inf")
        # Scatter valid candidate scores into vocab-sized tensor
        scores = torch.full((B, self.vocab_size), float("-inf"), device=z.device)
        b_ix, c_ix = cand_mask.nonzero(as_tuple=True)
        scores[b_ix, cand_idx[b_ix, c_ix]] = cand_scores[b_ix, c_ix]
        random_logit = self.random_logit_head(z)                 # [B, 1]
        scores = torch.cat([scores, random_logit], dim=-1)       # [B, vocab_size + 1]
        log_probs = F.log_softmax(scores, dim=-1)
        if return_z:
            return log_probs, z
        return log_probs
