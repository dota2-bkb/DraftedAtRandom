"""Read-only startup data: name maps, hero->ability mappings, MMR rank conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from dota2ad.core import AbilityId, HeroId, Paths, UnifiedIdx, VocabKey, Vocabs




@dataclass
class HeroAbilitySet:
    basics: list[AbilityId] = field(default_factory=list)
    ult: AbilityId | None = None


@dataclass
class Lookups:
    vocabs: Vocabs
    ability_id_to_name: dict[AbilityId, str]
    hero_id_to_name: dict[HeroId, str]
    hero_name_to_id: dict[str, HeroId]
    ability_name_to_id: dict[str, AbilityId]
    hero_abilities: dict[HeroId, HeroAbilitySet]
    ult_ids: set[AbilityId]
    idx_to_raw: dict[UnifiedIdx, VocabKey]
    mmr_mean: float
    mmr_std: float


def build_idx_to_raw(vocabs: Vocabs) -> dict[UnifiedIdx, VocabKey]:
    return {i: key for key, i in vocabs.draft_id_to_index.items()}


def build_lookups(
    paths: Paths, vocabs: Vocabs, mmr_mean: float, mmr_std: float,
) -> Lookups:
    lut_path = paths.dataset / "lookups.json"
    if not lut_path.exists():
        raise FileNotFoundError(
            f"{lut_path} missing — it is a build-dataset artifact "
            "(run `pixi run build-dataset`)"
        )
    with open(lut_path) as f:
        t = json.load(f)
    ability_id_to_name = {AbilityId(int(k)): v for k, v in t["ability_id_to_name"].items()}
    hero_id_to_name = {HeroId(int(k)): v for k, v in t["hero_id_to_name"].items()}
    hero_name_to_id = {v: k for k, v in hero_id_to_name.items()}
    ability_name_to_id = {v: k for k, v in ability_id_to_name.items()}
    ult_ids = {AbilityId(a) for a in t["ult_ids"]}
    hero_abilities = {
        HeroId(int(hid)): HeroAbilitySet(
            basics=[AbilityId(a) for a in entry["basics"]],
            ult=AbilityId(entry["ult"]) if entry["ult"] is not None else None,
        )
        for hid, entry in t["hero_abilities"].items()
    }
    idx_to_raw = build_idx_to_raw(vocabs)
    return Lookups(
        vocabs=vocabs,
        ability_id_to_name=ability_id_to_name,
        hero_id_to_name=hero_id_to_name,
        hero_name_to_id=hero_name_to_id,
        ability_name_to_id=ability_name_to_id,
        hero_abilities=hero_abilities,
        ult_ids=ult_ids,
        idx_to_raw=idx_to_raw,
        mmr_mean=mmr_mean,
        mmr_std=mmr_std,
    )
