"""StatsModel: predicts per-player post-game stats from draft loadouts."""

from __future__ import annotations

import torch
import torch.nn as nn

from dota2ad.core.types import StatsBatch, StatsNormDict
from dota2ad.models.set_transformer import SetTransformer


class StatsModel(nn.Module):
    """Predicts per-player post-game stats from draft loadouts."""

    mmr_mean: torch.Tensor
    mmr_std: torch.Tensor
    scalar_mean: torch.Tensor
    scalar_std: torch.Tensor
    gold_t_mean: torch.Tensor
    gold_t_std: torch.Tensor
    xp_t_mean: torch.Tensor
    xp_t_std: torch.Tensor
    lh_t_mean: torch.Tensor
    lh_t_std: torch.Tensor
    matchup_mean: torch.Tensor
    matchup_std: torch.Tensor
    damage_mean: torch.Tensor
    damage_std: torch.Tensor
    gold_reasons_mean: torch.Tensor
    gold_reasons_std: torch.Tensor
    xp_reasons_mean: torch.Tensor
    xp_reasons_std: torch.Tensor
    priority_mean: torch.Tensor
    priority_std: torch.Tensor
    spell_damage_mean: torch.Tensor
    spell_damage_std: torch.Tensor

    def __init__(self, vocab_size: int, d: int = 32, n_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.n_heads = n_heads

        self.register_buffer("mmr_mean", torch.zeros(()))
        self.register_buffer("mmr_std", torch.ones(()))
        self.register_buffer("scalar_mean", torch.zeros(22))
        self.register_buffer("scalar_std", torch.ones(22))
        self.register_buffer("gold_t_mean", torch.zeros(3))
        self.register_buffer("gold_t_std", torch.ones(3))
        self.register_buffer("xp_t_mean", torch.zeros(3))
        self.register_buffer("xp_t_std", torch.ones(3))
        self.register_buffer("lh_t_mean", torch.zeros(3))
        self.register_buffer("lh_t_std", torch.ones(3))
        self.register_buffer("matchup_mean", torch.zeros(()))
        self.register_buffer("matchup_std", torch.ones(()))
        self.register_buffer("damage_mean", torch.zeros(()))
        self.register_buffer("damage_std", torch.ones(()))
        self.register_buffer("gold_reasons_mean", torch.zeros(14))
        self.register_buffer("gold_reasons_std", torch.ones(14))
        self.register_buffer("xp_reasons_mean", torch.zeros(6))
        self.register_buffer("xp_reasons_std", torch.ones(6))
        self.register_buffer("priority_mean", torch.zeros(4))
        self.register_buffer("priority_std", torch.ones(4))
        self.register_buffer("spell_damage_mean", torch.zeros(()))
        self.register_buffer("spell_damage_std", torch.ones(()))

        self.embed = nn.Embedding(vocab_size, d)
        nn.init.zeros_(self.embed.weight[-1])
        self._empty_idx = vocab_size - 1
        self.embed.weight.register_hook(
            lambda grad: grad.index_fill_(0, torch.tensor([self._empty_idx], device=grad.device), 0)
        )
        self.st_loadout = SetTransformer(d, d, n_heads)

        self.mmr_W = nn.Linear(10, d)
        self.mmr_V = nn.Linear(10, d)

        self.scalar_head = nn.Sequential(
            nn.Linear(d, 2 * d), nn.ReLU(),
            nn.Linear(2 * d, 2 * d), nn.ReLU(),
            nn.Linear(2 * d, 22),
        )
        self.gold_t_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 3))
        self.xp_t_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 3))
        self.lh_t_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 3))
        self.matchup_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 2))
        self.damage_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.gold_reasons_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 14))
        self.xp_reasons_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 6))
        self.priority_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        # Per-(player, slot) total spell damage to enemy heroes (per-min,
        # z-normed). Mirrors priority_head: (player_repr ⊕ ability_emb) → 1.
        self.spell_damage_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def _encode(self, batch: StatsBatch) -> torch.Tensor:
        """Shared loadout+MMR encoder → per-player representation [B, 10, d]."""
        B = batch["mmr"].shape[0]
        x = self.embed(batch["load_idx"])
        load = self.st_loadout(x, batch["load_mask"])
        is_empty = (batch["load_mask"].sum(dim=1) == 1) & (batch["load_idx"][:, 0] == self._empty_idx)
        if is_empty.any():
            load = load.clone()
            load[is_empty] = 0.0
        load = load.view(B, 10, -1)
        m = batch["mmr_mask"].float()
        if self.training:
            group_drop = torch.rand(B, 1, device=m.device) < 0.3
            player_drop = torch.rand(B, 10, device=m.device) < 0.1
            m = m * (~group_drop).float() * (~player_drop).float()
        x_tilde = batch["mmr"] * m
        mmr_vec = self.mmr_W(x_tilde) + self.mmr_V(m)
        return load + mmr_vec.unsqueeze(1)

    def set_norm(self, norm: StatsNormDict) -> None:
        """Set normalization buffers from a StatsNormDict."""
        self.mmr_mean.fill_(norm["mmr"][0])
        self.mmr_std.fill_(norm["mmr"][1])
        self.scalar_mean.copy_(torch.tensor(norm["scalar"][0], dtype=torch.float32))
        self.scalar_std.copy_(torch.tensor(norm["scalar"][1], dtype=torch.float32))
        self.gold_t_mean.copy_(torch.tensor(norm["gold_t"][0], dtype=torch.float32))
        self.gold_t_std.copy_(torch.tensor(norm["gold_t"][1], dtype=torch.float32))
        self.xp_t_mean.copy_(torch.tensor(norm["xp_t"][0], dtype=torch.float32))
        self.xp_t_std.copy_(torch.tensor(norm["xp_t"][1], dtype=torch.float32))
        self.lh_t_mean.copy_(torch.tensor(norm["lh_t"][0], dtype=torch.float32))
        self.lh_t_std.copy_(torch.tensor(norm["lh_t"][1], dtype=torch.float32))
        self.matchup_mean.fill_(norm["matchup"][0])
        self.matchup_std.fill_(norm["matchup"][1])
        self.damage_mean.fill_(norm["damage"][0])
        self.damage_std.fill_(norm["damage"][1])
        self.gold_reasons_mean.copy_(torch.tensor(norm["gold_reasons"][0], dtype=torch.float32))
        self.gold_reasons_std.copy_(torch.tensor(norm["gold_reasons"][1], dtype=torch.float32))
        self.xp_reasons_mean.copy_(torch.tensor(norm["xp_reasons"][0], dtype=torch.float32))
        self.xp_reasons_std.copy_(torch.tensor(norm["xp_reasons"][1], dtype=torch.float32))
        self.priority_mean.copy_(torch.tensor(norm["priority"][0], dtype=torch.float32))
        self.priority_std.copy_(torch.tensor(norm["priority"][1], dtype=torch.float32))
        self.spell_damage_mean.fill_(norm["spell_damage"][0])
        self.spell_damage_std.fill_(norm["spell_damage"][1])

    def forward(self, batch: StatsBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (scalars, gold_t, xp_t, lh_t, matchups, priorities, damage,
        gold_reasons, xp_reasons, spell_damage). spell_damage is per-(player,
        slot) shape [B, 10, 4], total spell damage to enemy heroes per-min."""
        B = batch["mmr"].shape[0]
        load = self._encode(batch)                            # [B, 10, d]

        # Per-player predictions
        scalars = self.scalar_head(load)                      # [B, 10, 22]
        gold_t = self.gold_t_head(load)                        # [B, 10, 3]
        xp_t = self.xp_t_head(load)                            # [B, 10, 3]
        lh_t = self.lh_t_head(load)                            # [B, 10, 3]

        # Matchups: each player vs 5 enemies (pick_slot: even=radiant, odd=dire)
        rad = load[:, 0::2]                                    # [B, 5, d]
        dire = load[:, 1::2]                                   # [B, 5, d]
        rad_exp = rad.unsqueeze(2).expand(-1, -1, 5, -1)      # [B, 5, 5, d]
        dire_exp = dire.unsqueeze(1).expand(-1, 5, -1, -1)    # [B, 5, 5, d]
        # Radiant player i vs dire player j
        rad_matchups = self.matchup_head(
            torch.cat([rad_exp, dire_exp], -1)
        )                                                      # [B, 5, 5, 2]
        # Dire player j vs radiant player i
        dire_matchups = self.matchup_head(
            torch.cat([dire.unsqueeze(2).expand(-1, -1, 5, -1),
                       rad.unsqueeze(1).expand(-1, 5, -1, -1)], -1)
        )                                                      # [B, 5, 5, 2]
        # Scatter back to pick_slot order (even=rad, odd=dire)
        matchups = load.new_empty(B, 10, 5, 2)
        matchups[:, 0::2] = rad_matchups
        matchups[:, 1::2] = dire_matchups

        # Damage dealt: same rad-vs-dire pairing as matchups
        rad_damage = self.damage_head(
            torch.cat([rad_exp, dire_exp], -1)
        ).squeeze(-1)                                           # [B, 5, 5]
        dire_damage = self.damage_head(
            torch.cat([dire.unsqueeze(2).expand(-1, -1, 5, -1),
                       rad.unsqueeze(1).expand(-1, 5, -1, -1)], -1)
        ).squeeze(-1)                                           # [B, 5, 5]
        damage = load.new_empty(B, 10, 5)
        damage[:, 0::2] = rad_damage
        damage[:, 1::2] = dire_damage

        # Gold/XP reasons: per-player heads
        gold_reasons = self.gold_reasons_head(load)              # [B, 10, 14]
        xp_reasons = self.xp_reasons_head(load)                  # [B, 10, 6]

        # Priority + per-spell damage: both heads take (player_repr ⊕ ability_emb).
        ability_emb = self.embed(batch["ability_indices"])      # [B, 10, 4, d]
        load_exp = load.unsqueeze(2).expand(-1, -1, 4, -1)     # [B, 10, 4, d]
        cat = torch.cat([load_exp, ability_emb], -1)            # [B, 10, 4, 2d]
        priorities = self.priority_head(cat).squeeze(-1)        # [B, 10, 4]
        spell_damage = self.spell_damage_head(cat).squeeze(-1)  # [B, 10, 4]

        return scalars, gold_t, xp_t, lh_t, matchups, priorities, damage, gold_reasons, xp_reasons, spell_damage


class EnsembleStatsModel(nn.Module):
    """Averages the forward outputs of K StatsModels. The per-stat ranking the
    StatsModel is used for is unstable run-to-run (the MSE objective has a
    degenerate basin: equal val-loss, very different counterfactual ordering);
    averaging predictions reduces that variance. All members share the same
    z-normalization (same train data), so their z-space outputs are directly
    averageable. Same forward interface as StatsModel (the 10-tuple), so it is a
    drop-in reward source. Members all have grad disabled."""

    def __init__(self, models: list[StatsModel]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.d = models[0].d
        self.n_heads = models[0].n_heads
        # Norm buffers (identical across members — same train data). Exposed for
        # read-only de-normalization (e.g. the dense damage reward converting the
        # spell head's z-output into hero_damage z-units).
        self.scalar_mean = models[0].scalar_mean
        self.scalar_std = models[0].scalar_std
        self.spell_damage_mean = models[0].spell_damage_mean
        self.spell_damage_std = models[0].spell_damage_std

    def forward(self, batch: StatsBatch) -> tuple[torch.Tensor, ...]:
        outs = [m(batch) for m in self.models]
        n_out = len(outs[0])
        return tuple(torch.stack([o[i] for o in outs]).mean(dim=0) for i in range(n_out))
