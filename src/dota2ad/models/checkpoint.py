"""Standardized model checkpoint save/load."""

from __future__ import annotations

from pathlib import Path

import torch

from dota2ad.core.types import Vocabs
from dota2ad.models.policy import BehaviorPolicy
from dota2ad.models.stats import EnsembleStatsModel, StatsModel


def load_policy(path: Path | str, vocabs: Vocabs, device: torch.device) -> BehaviorPolicy:
    """Load a BehaviorPolicy from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab_size = len(vocabs.draft_id_to_index)
    model = BehaviorPolicy(
        vocab_size,
        d=ckpt["d"],
        n_heads=ckpt["n_heads"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    # State-rarity band calibration travels with the policy (written by
    # experiments/train-policy).
    model.density_support_q = ckpt["density_support_q"]
    return model


def load_stats_model(
    path: Path | str, vocabs: Vocabs, device: torch.device
) -> StatsModel | EnsembleStatsModel:
    """Load a StatsModel from checkpoint. An ensemble checkpoint (`"ensemble"`
    key holding a list of state_dicts) loads as an EnsembleStatsModel that
    averages member predictions — a drop-in reward source."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab_size = len(vocabs.draft_id_to_index)
    d, n_heads = ckpt["d"], ckpt["n_heads"]
    if "ensemble" in ckpt:
        members = []
        for sd in ckpt["ensemble"]:
            m = StatsModel(vocab_size, d=d, n_heads=n_heads).to(device)
            m.load_state_dict(sd)
            m.eval()
            m.requires_grad_(False)
            members.append(m)
        ensemble = EnsembleStatsModel(members).to(device)
        ensemble.eval()
        return ensemble
    model = StatsModel(vocab_size, d=d, n_heads=n_heads).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
