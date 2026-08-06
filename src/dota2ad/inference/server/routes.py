"""HTTP routes — Litestar controller."""

from __future__ import annotations

from litestar import Controller, delete, get, post
from litestar.datastructures import State
from litestar.di import Provide
from litestar.exceptions import HTTPException

from dota2ad.core import AbilityId, HeroId, PickSlot, VocabKey

from dota2ad.inference.server.context import AppContext
from dota2ad.inference.server.lookups import HeroAbilitySet, Lookups
from dota2ad.inference.server.schemas import (
    AbilityEntry,
    AddAbilityBody,
    AddHeroBody,
    AnnotatedDraft,
    AnnotatedPlayerPick,
    ConfidenceInfo,
    DraftEnvelope,
    HeroEntry,
    PickBody,
    PresetEntry,
    PreferencesState,
    SetPreferencesBody,
)
from dota2ad.training.weights import PRESETS, get_preset
from dota2ad.inference.server.state import (
    DraftState,
    advance_state,
    make_empty_state,
    set_at,
)


def _remove_hero_from_pool(
    state: DraftState, lookups: Lookups, hero_id: HeroId,
) -> None:
    """Remove a hero from the pool *and* its abilities, but keep abilities
    that are also owned by another hero still in the pool (overlap-safe).

    Mutates state.hero_pool_all / remaining and basic/ult pools in place.
    """
    # 1. Null the hero's slot in hero_pool_all (and drop from remaining)
    if hero_id in state.hero_pool_all:
        idx = state.hero_pool_all.index(hero_id)
        state.hero_pool_all[idx] = None
    if hero_id in state.hero_pool_remaining:
        state.hero_pool_remaining.remove(hero_id)

    # 2. Collect abilities that should be kept (owned by some OTHER hero still
    #    in the pool). For most hero pairs this set is empty, but the check
    #    is cheap and prevents accidental removal on overlap.
    keep_basics: set[AbilityId] = set()
    keep_ults: set[AbilityId] = set()
    for h in state.hero_pool_all:
        if h is None or h == hero_id:
            continue
        ha_other = lookups.hero_abilities.get(h, HeroAbilitySet())
        keep_basics.update(ha_other.basics)
        if ha_other.ult is not None:
            keep_ults.add(ha_other.ult)

    # 3. Remove this hero's abilities from the basic/ult pools, except those
    #    in keep_basics / keep_ults.
    ha = lookups.hero_abilities.get(hero_id, HeroAbilitySet())
    for ab_id in ha.basics:
        if ab_id in keep_basics:
            continue
        if ab_id in state.basic_pool_all:
            j = state.basic_pool_all.index(ab_id)
            state.basic_pool_all[j] = None
        if ab_id in state.basic_pool_remaining:
            state.basic_pool_remaining.remove(ab_id)
    if ha.ult is not None and ha.ult not in keep_ults:
        if ha.ult in state.ult_pool_all:
            j = state.ult_pool_all.index(ha.ult)
            state.ult_pool_all[j] = None
        if ha.ult in state.ult_pool_remaining:
            state.ult_pool_remaining.remove(ha.ult)
from dota2ad.inference.server.suggestions import get_suggestions


def provide_ctx(state: State) -> AppContext:
    return state.ctx


class DraftError(HTTPException):
    """Domain error rendered as {error: msg} with status 400."""

    status_code = 400

    def __init__(self, msg: str):
        super().__init__(detail=msg)


def _annotate(state: DraftState, lookups: Lookups) -> AnnotatedDraft:
    h2n = lookups.hero_id_to_name
    a2n = lookups.ability_id_to_name

    annotated_picks: dict[PickSlot, AnnotatedPlayerPick] = {}
    for slot, p in state.player_picks.items():
        annotated_picks[slot] = AnnotatedPlayerPick(
            hero=p.hero,
            basics=list(p.basics),
            ult=p.ult,
            hero_name=h2n.get(p.hero) if p.hero is not None else None,
            basics_names=[a2n.get(a, str(a)) for a in p.basics],
            ult_name=a2n.get(p.ult) if p.ult is not None else None,
        )

    return AnnotatedDraft(
        player_picks=annotated_picks,
        hero_pool_remaining=list(state.hero_pool_remaining),
        basic_pool_remaining=list(state.basic_pool_remaining),
        ult_pool_remaining=list(state.ult_pool_remaining),
        hero_pool_all=list(state.hero_pool_all),
        basic_pool_all=list(state.basic_pool_all),
        ult_pool_all=list(state.ult_pool_all),
        hero_pool_names=[h2n.get(h, str(h)) for h in state.hero_pool_remaining],
        basic_pool_names=[a2n.get(a, str(a)) for a in state.basic_pool_remaining],
        ult_pool_names=[a2n.get(a, str(a)) for a in state.ult_pool_remaining],
        hero_pool_all_names=[h2n.get(h, str(h)) if h is not None else None for h in state.hero_pool_all],
        basic_pool_all_names=[a2n.get(a, str(a)) if a is not None else None for a in state.basic_pool_all],
        ult_pool_all_names=[a2n.get(a, str(a)) if a is not None else None for a in state.ult_pool_all],
        mmr=state.mmr,
        pick_slot=state.pick_slot,
        turn=state.turn,
        hero_id=state.hero_id,
        draft_ability_id=state.draft_ability_id,
        action_key=state.action_key,
        is_random=state.is_random,
        radiant_win=state.radiant_win,
        match_id=state.match_id,
        history=list(state.history),
    )


def _build_preferences(ctx: AppContext) -> PreferencesState | None:
    if ctx.stats_dqn is None:
        return None
    presets = [PresetEntry(name=name) for name in PRESETS]
    return PreferencesState(
        preset=ctx.preset_name,
        presets=presets,
        stat_names=list(ctx.stats_dqn.stat_names),
        active_weights=ctx.stat_weights.tolist(),
        stat_norm_mean=ctx.stats_dqn.stat_norm_mean.tolist(),
        stat_norm_std=ctx.stats_dqn.stat_norm_std.tolist(),
    )


def envelope(ctx: AppContext) -> DraftEnvelope:
    suggestions = get_suggestions(
        ctx.state, ctx.lookups, ctx.policy, ctx.device,
        stats_dqn=ctx.stats_dqn, stat_weights=ctx.stat_weights,
        reweight_beta=ctx.reweight_beta if ctx.active_recommender == "reweight_bc" else None,
    )
    # State-rarity confidence band (how typical the whole draft is under BC).
    # Unusual draft → StatsModel extrapolating → recommendations less reliable.
    from dota2ad.suggest.density import state_confidence
    state_support, state_band = state_confidence(
        ctx.state, ctx.policy, ctx.lookups.vocabs,
        ctx.lookups.mmr_mean, ctx.lookups.mmr_std, ctx.device,
    )
    return DraftEnvelope(
        state=_annotate(ctx.state, ctx.lookups),
        suggestions=suggestions,
        confidence=ConfidenceInfo(state_support=state_support, state_band=state_band),
        preferences=_build_preferences(ctx),
        recommenders=ctx.available_recommenders,
        recommender=ctx.active_recommender,
        reweight_beta=ctx.reweight_beta,
    )


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class DraftController(Controller):
    path = "/api/draft"
    dependencies = {"ctx": Provide(provide_ctx, sync_to_thread=False)}

    @get("/", sync_to_thread=False)
    def get_draft(self, ctx: AppContext) -> DraftEnvelope:
        return envelope(ctx)

    @post("/reset")
    async def reset(self, ctx: AppContext) -> DraftEnvelope:
        async with ctx.lock:
            saved_pools = (ctx.state.hero_pool_all, ctx.state.basic_pool_all, ctx.state.ult_pool_all)
            ctx.state = make_empty_state()
            ctx.state.hero_pool_all = list(saved_pools[0])
            ctx.state.basic_pool_all = list(saved_pools[1])
            ctx.state.ult_pool_all = list(saved_pools[2])
            ctx.state.hero_pool_remaining = [h for h in saved_pools[0] if h is not None]
            ctx.state.basic_pool_remaining = [a for a in saved_pools[1] if a is not None]
            ctx.state.ult_pool_remaining = [a for a in saved_pools[2] if a is not None]
            return envelope(ctx)

    @post("/undo")
    async def undo(self, ctx: AppContext) -> DraftEnvelope:
        async with ctx.lock:
            history = list(ctx.state.history)
            if not history:
                raise DraftError("Nothing to undo")
            saved_pools = (
                list(ctx.state.hero_pool_all),
                list(ctx.state.basic_pool_all),
                list(ctx.state.ult_pool_all),
            )
            ctx.state = make_empty_state()
            ctx.state.hero_pool_all = list(saved_pools[0])
            ctx.state.basic_pool_all = list(saved_pools[1])
            ctx.state.ult_pool_all = list(saved_pools[2])
            ctx.state.hero_pool_remaining = [h for h in saved_pools[0] if h is not None]
            ctx.state.basic_pool_remaining = [a for a in saved_pools[1] if a is not None]
            ctx.state.ult_pool_remaining = [a for a in saved_pools[2] if a is not None]
            for event in history[:-1]:
                ctx.state.hero_id = event.hero_id
                ctx.state.draft_ability_id = event.draft_ability_id
                advance_state(ctx.state)
            ctx.state.hero_pool_all = list(saved_pools[0])
            ctx.state.basic_pool_all = list(saved_pools[1])
            ctx.state.ult_pool_all = list(saved_pools[2])
            return envelope(ctx)

    @post("/picks")
    async def pick(self, data: PickBody, ctx: AppContext) -> DraftEnvelope:
        async with ctx.lock:
            if data.type == "hero":
                ctx.state.hero_id = HeroId(data.id)
                ctx.state.draft_ability_id = None
            else:
                ctx.state.hero_id = None
                ctx.state.draft_ability_id = AbilityId(data.id)
            advance_state(ctx.state, is_random=data.is_random)
            return envelope(ctx)

    @post("/preferences")
    async def set_preferences(
        self, data: SetPreferencesBody, ctx: AppContext,
    ) -> DraftEnvelope:
        """Switch the active recommender (bc / q / trial / …) and/or stats-DQN preset."""
        async with ctx.lock:
            if data.recommender is not None:
                if data.recommender not in ctx.available_recommenders:
                    raise DraftError(
                        f"Unknown recommender '{data.recommender}'. "
                        f"Available: {ctx.available_recommenders}"
                    )
                ctx.active_recommender = data.recommender
            if data.preset is not None:
                if data.preset not in PRESETS:
                    raise DraftError(
                        f"Unknown preset '{data.preset}'. Available: {sorted(PRESETS)}"
                    )
                ctx.preset_name = data.preset
                ctx.stat_weights = get_preset(data.preset)
            if data.reweight_beta is not None:
                if not 0.0 <= data.reweight_beta <= 50.0:
                    raise DraftError("reweight_beta must be in [0, 50]")
                ctx.reweight_beta = float(data.reweight_beta)
            return envelope(ctx)

    @post("/pool/heroes")
    async def add_hero(self, data: AddHeroBody, ctx: AppContext) -> DraftEnvelope:
        async with ctx.lock:
            name = data.name.strip()
            hero_id = ctx.lookups.hero_name_to_id.get(name)
            if hero_id is None:
                raise DraftError(f"Unknown hero: {name}")
            if VocabKey(f"h:{hero_id}") not in ctx.lookups.vocabs.draft_id_to_index:
                raise DraftError(f"Hero {name} not in vocabulary")
            if hero_id in ctx.state.hero_pool_all:
                raise DraftError(f"Hero {name} already in pool")
            slot = data.slot
            # If this slot is already occupied (replacing a hero), evict the
            # old hero AND its abilities first so the new hero's abilities
            # don't pile on top of the previous ones.
            if slot is not None and slot < len(ctx.state.hero_pool_all):
                old = ctx.state.hero_pool_all[slot]
                if old is not None and old != hero_id:
                    _remove_hero_from_pool(ctx.state, ctx.lookups, old)
            if slot is not None:
                set_at(ctx.state.hero_pool_all, slot, hero_id)
            else:
                ctx.state.hero_pool_all.append(hero_id)
            ctx.state.hero_pool_remaining.append(hero_id)

            ha: HeroAbilitySet = ctx.lookups.hero_abilities.get(hero_id, HeroAbilitySet())
            for j, ab_id in enumerate(ha.basics):
                if ab_id not in ctx.state.basic_pool_all:
                    ctx.state.basic_pool_remaining.append(ab_id)
                    if slot is not None:
                        set_at(ctx.state.basic_pool_all, slot * 3 + j, ab_id)
                    else:
                        ctx.state.basic_pool_all.append(ab_id)
            if ha.ult is not None and ha.ult not in ctx.state.ult_pool_all:
                ctx.state.ult_pool_remaining.append(ha.ult)
                if slot is not None:
                    set_at(ctx.state.ult_pool_all, slot, ha.ult)
                else:
                    ctx.state.ult_pool_all.append(ha.ult)
            return envelope(ctx)

    @post("/pool/abilities")
    async def add_ability(self, data: AddAbilityBody, ctx: AppContext) -> DraftEnvelope:
        async with ctx.lock:
            name = data.name.strip()
            ab_id = ctx.lookups.ability_name_to_id.get(name)
            if ab_id is None:
                raise DraftError(f"Unknown ability: {name}")
            if VocabKey(f"a:{ab_id}") not in ctx.lookups.vocabs.draft_id_to_index:
                raise DraftError(f"Ability {name} not in vocabulary")
            is_ult = (data.kind == "ult") if data.kind else (ab_id in ctx.lookups.ult_ids)
            slot = data.slot
            if is_ult:
                if ab_id in ctx.state.ult_pool_all:
                    raise DraftError(f"Ability {name} already in pool")
                ctx.state.ult_pool_remaining.append(ab_id)
                if slot is not None:
                    set_at(ctx.state.ult_pool_all, slot, ab_id)
                else:
                    ctx.state.ult_pool_all.append(ab_id)
            else:
                if ab_id in ctx.state.basic_pool_all:
                    raise DraftError(f"Ability {name} already in pool")
                ctx.state.basic_pool_remaining.append(ab_id)
                if slot is not None:
                    set_at(ctx.state.basic_pool_all, slot, ab_id)
                else:
                    ctx.state.basic_pool_all.append(ab_id)
            return envelope(ctx)

    @delete("/pool/heroes/{name:str}", status_code=200)
    async def remove_hero(self, name: str, ctx: AppContext) -> DraftEnvelope:
        async with ctx.lock:
            hero_id = ctx.lookups.hero_name_to_id.get(name)
            if hero_id is None or hero_id not in ctx.state.hero_pool_all:
                raise DraftError(f"Hero {name} not in pool")
            _remove_hero_from_pool(ctx.state, ctx.lookups, hero_id)
            return envelope(ctx)

    @delete("/pool/abilities/{name:str}", status_code=200)
    async def remove_ability(self, name: str, ctx: AppContext) -> DraftEnvelope:
        async with ctx.lock:
            ab_id = ctx.lookups.ability_name_to_id.get(name)
            if ab_id is None:
                raise DraftError(f"Unknown ability: {name}")
            if ab_id in ctx.state.ult_pool_all:
                idx = ctx.state.ult_pool_all.index(ab_id)
                ctx.state.ult_pool_all[idx] = None
                if ab_id in ctx.state.ult_pool_remaining:
                    ctx.state.ult_pool_remaining.remove(ab_id)
            elif ab_id in ctx.state.basic_pool_all:
                idx = ctx.state.basic_pool_all.index(ab_id)
                ctx.state.basic_pool_all[idx] = None
                if ab_id in ctx.state.basic_pool_remaining:
                    ctx.state.basic_pool_remaining.remove(ab_id)
            else:
                raise DraftError(f"Ability {name} not in pool")
            return envelope(ctx)


class LookupsController(Controller):
    path = "/api/lookups"
    dependencies = {"ctx": Provide(provide_ctx, sync_to_thread=False)}

    @get("/heroes", sync_to_thread=False)
    def heroes(self, ctx: AppContext) -> list[HeroEntry]:
        return [
            HeroEntry(id=hid, name=name)
            for hid, name in sorted(ctx.lookups.hero_id_to_name.items(), key=lambda x: x[1])
        ]

    @get("/abilities", sync_to_thread=False)
    def abilities(self, ctx: AppContext) -> list[AbilityEntry]:
        return [
            AbilityEntry(id=aid, name=name, is_ult=aid in ctx.lookups.ult_ids)
            for aid, name in sorted(ctx.lookups.ability_id_to_name.items(), key=lambda x: x[1])
        ]
