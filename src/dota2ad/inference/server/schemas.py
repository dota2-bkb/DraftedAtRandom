"""Wire types — request bodies and response shapes."""

from __future__ import annotations

from typing import Literal

import msgspec

from dota2ad.core import AbilityId, HeroId, HistoryEvent, Per10, PickSlot, Turn


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class AddHeroBody(msgspec.Struct):
    name: str
    slot: int | None = None


class AddAbilityBody(msgspec.Struct):
    name: str
    slot: int | None = None
    kind: Literal["ult", "basic"] | None = None  # hint from UI; otherwise ult_ids decides


class PickBody(msgspec.Struct):
    id: int
    type: Literal["hero", "basic", "ult"]
    # User-labeled timeout flag: True iff the picking player let the draft
    # timer expire on this pick. Feeds the is_random per-loadout-position
    # input to the state encoder, so the Q queries condition on the actual
    # deliberate/random history rather than assuming all-deliberate.
    is_random: bool = False


class SetPreferencesBody(msgspec.Struct):
    """Switch the active recommender (`recommender`: "bc" | "q" | "trial" | "reweight_bc"),
    the stats-DQN preset (`preset`, changes ctx.stat_weights), and/or the reweight-BC tilt
    strength (`reweight_beta`). Any field may be null."""
    preset: str | None = None
    recommender: str | None = None
    reweight_beta: float | None = None


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------


class SuggestionEntry(msgspec.Struct):
    id: int
    name: str
    type: Literal["hero", "basic", "ult"]
    prob: float  # BC softmax probability over feasibles
    q: float | None = None  # DQN Q-value if loaded; the primary ranker when present
    # Stats-DQN: per-stat Q vector [K] for this action (in z-score units),
    # populated when the stats-DQN suggester is loaded. None when no
    # vector-Q recommender is loaded (BC-only mode).
    q_vec: list[float] | None = None


class PresetEntry(msgspec.Struct):
    name: str


class PreferencesState(msgspec.Struct):
    """Active stats-DQN preference state — surfaces to UI for the picker."""
    preset: str
    presets: list[PresetEntry]
    stat_names: list[str]
    # The active preset's weight vector. UI uses these to decompose the
    # composite Q into per-stat contributions for the top suggestion.
    active_weights: list[float]
    # Per-stat z-normalization (μ, σ) so the UI can denormalize q_vec into
    # physical units if it wants to display predicted stat values.
    stat_norm_mean: list[float] | None = None
    stat_norm_std: list[float] | None = None


class ConfidenceInfo(msgspec.Struct):
    """State-rarity confidence for the whole draft: how typical it is under the
    BC density (-log p(s)/T). Rarer draft → StatsModel extrapolating → lower
    confidence band. The raw support is kept for debugging / a percentile
    readout; the UI shows the band.
    """
    state_support: float | None  # -log p(s)/T under BC; None before first pick
    state_band: Literal["high", "medium", "low"] | None  # confidence band


class AnnotatedPlayerPick(msgspec.Struct):
    hero: HeroId | None
    basics: list[AbilityId]
    ult: AbilityId | None
    hero_name: str | None
    basics_names: list[str]
    ult_name: str | None


class AnnotatedDraft(msgspec.Struct):
    player_picks: dict[PickSlot, AnnotatedPlayerPick]
    hero_pool_remaining: list[HeroId]
    basic_pool_remaining: list[AbilityId]
    ult_pool_remaining: list[AbilityId]
    hero_pool_all: list[HeroId | None]
    basic_pool_all: list[AbilityId | None]
    ult_pool_all: list[AbilityId | None]
    hero_pool_names: list[str]
    basic_pool_names: list[str]
    ult_pool_names: list[str]
    hero_pool_all_names: list[str | None]
    basic_pool_all_names: list[str | None]
    ult_pool_all_names: list[str | None]
    mmr: Per10[float | None]
    pick_slot: PickSlot
    turn: Turn
    hero_id: HeroId | None
    draft_ability_id: AbilityId | None
    action_key: str
    is_random: bool
    radiant_win: bool
    match_id: int
    history: list[HistoryEvent]


class DraftEnvelope(msgspec.Struct):
    state: AnnotatedDraft
    suggestions: list[SuggestionEntry]
    confidence: ConfidenceInfo
    preferences: PreferencesState | None = None  # null when the active recommender is BC
    # Recommender selection (always present, so the UI can switch even in BC mode).
    recommenders: list[str] = []              # available: "bc" + loaded vector-Q variants (+ "reweight_bc")
    recommender: str = "bc"                   # the active one
    reweight_beta: float = 1.0                # current reweight-BC tilt strength β


class HeroEntry(msgspec.Struct):
    id: HeroId
    name: str


class AbilityEntry(msgspec.Struct):
    id: AbilityId
    name: str
    is_ult: bool


class ErrorResponse(msgspec.Struct):
    error: str
