"""Compute ranked action suggestions from the policy network."""

from __future__ import annotations

from typing import Literal

import torch

from dota2ad.core import AbilityId, HeroId, UnifiedIdx, encode_policy_sample, policy_collate
from dota2ad.models import BehaviorPolicy
from dota2ad.suggest import StatsDQNSuggester

from dota2ad.inference.server.lookups import Lookups
from dota2ad.inference.server.schemas import SuggestionEntry
from dota2ad.inference.server.state import DraftState, state_to_turn_row


PickType = Literal["hero", "basic", "ult"]


def resolve_action(
    unified_idx: UnifiedIdx, lookups: Lookups,
) -> tuple[int, str, PickType]:
    """Convert a unified vocab index to (raw_id, name, type)."""
    raw_key = lookups.idx_to_raw[unified_idx]
    kind, raw_id_str = raw_key.split(":")
    raw_id = int(raw_id_str)
    pick_type: PickType
    if kind == "h":
        hero_id = HeroId(raw_id)
        name = lookups.hero_id_to_name.get(hero_id, f"hero_{hero_id}")
        pick_type = "hero"
    else:
        ability_id = AbilityId(raw_id)
        name = lookups.ability_id_to_name.get(ability_id, f"ability_{ability_id}")
        pick_type = "ult" if ability_id in lookups.ult_ids else "basic"
    return raw_id, name, pick_type


def get_suggestions(
    state: DraftState,
    lookups: Lookups,
    policy: BehaviorPolicy,
    device: torch.device,
    stats_dqn: StatsDQNSuggester | None = None,
    stat_weights: torch.Tensor | None = None,
    reweight_beta: float | None = None,
) -> list[SuggestionEntry]:
    """Return all feasible actions ranked.

    Ranking primary key:
    - If `reweight_beta` is set (with stats-DQN = Trial's value net), rank by the
      reweight-BC blend  log π_BC(a) + β·ẑ(composite)  over feasibles.
    - Else if stats-DQN is supplied, rank by composite score = Σ_k w_k · Q_vec[k]
      using `stat_weights` (defaults to ones if None).
    - Else fall back to BC softmax probability.

    The composite score replaces the scalar `q` field; per-action q_vec is
    populated for stats-DQN so the UI can render per-stat breakdowns.
    """
    if state.turn >= 50:
        return []

    turn_row = state_to_turn_row(state)
    history = list(state.history)
    sample = encode_policy_sample(turn_row, lookups.vocabs, history, lookups.mmr_mean, lookups.mmr_std)
    if not sample.feasible_mask.any():
        return []

    batch = policy_collate([sample], device)
    with torch.no_grad():
        log_probs = policy(batch).squeeze(0)

    probs = log_probs.exp()
    # If BC is V+1 (has random class), drop the random class and renormalize
    # over the feasible vocab for display.
    V = sample.feasible_mask.shape[0]
    if probs.shape[0] > V:
        probs = probs[:V]
    probs[~sample.feasible_mask] = 0.0
    norm = probs.sum()
    if norm > 0:
        probs = probs / norm
    n_feasible = int(sample.feasible_mask.sum().item())
    if n_feasible == 0:
        return []

    # Stats-DQN: composite scoring path (preferred when present)
    if stats_dqn is not None:
        weights = stat_weights if stat_weights is not None else torch.ones(stats_dqn.k_stats)
        q_vec_full, composite = stats_dqn.score_composite(sample, weights)
        composite_cpu = composite.cpu()
        q_vec_cpu = q_vec_full.cpu()
        feas = sample.feasible_mask.cpu()
        if reweight_beta is not None:
            # reweight-BC: rank by  log π_BC(a) + β·ẑ(composite)  over feasibles
            # (ẑ = within-state z-scored Trial value). β=0 ⇒ pure BC.
            comp_f = composite_cpu[feas]
            z = (comp_f - comp_f.mean()) / (comp_f.std() + 1e-9)
            score = torch.full_like(composite_cpu, float("-inf"))
            score[feas] = probs.cpu()[feas].clamp_min(1e-12).log() + float(reweight_beta) * z
        else:
            score = composite_cpu.masked_fill(~feas, float("-inf"))
        ranked_idx = torch.topk(score, n_feasible).indices

        out: list[SuggestionEntry] = []
        for i in range(n_feasible):
            unified_idx = UnifiedIdx(int(ranked_idx[i].item()))
            prob = round(float(probs[unified_idx].item()), 4)
            q = round(float(score[unified_idx].item()), 4)
            qvec = [round(float(v), 4) for v in q_vec_cpu[unified_idx].tolist()]
            raw_id, name, pick_type = resolve_action(unified_idx, lookups)
            out.append(SuggestionEntry(
                id=raw_id, name=name, type=pick_type,
                prob=prob, q=q, q_vec=qvec,
            ))
        return out

    # BC softmax fallback (no stats-DQN loaded): rank by human-pick probability.
    ranked_idx = torch.topk(probs, n_feasible).indices
    out = []
    for i in range(n_feasible):
        unified_idx = UnifiedIdx(int(ranked_idx[i].item()))
        prob = round(float(probs[unified_idx].item()), 4)
        raw_id, name, pick_type = resolve_action(unified_idx, lookups)
        out.append(SuggestionEntry(
            id=raw_id, name=name, type=pick_type,
            prob=prob, q=None,
        ))
    return out
