#!/usr/bin/env python3
"""
Build turn-level causal dataset from parsed match bundles.

Reads parsed/<match_id>/ directories (containing match_details.json +
draft_details.json), constructs a chronological timeline of hero picks
and ability picks per match.

Accepts multiple --parsed-dir arguments to combine data from several
parsed folders into one dataset.

Key ID concepts:
  player_slot: Dota 2 server slot encoding. 0-4 = radiant, 128-132 = dire.
  pick_slot: 0-9 index derived from the chronological order players first
      appear in the draft timeline. Gives a consistent positional encoding
      across matches (0 = first picker, 9 = last).
  draft_ability_id: Opaque integer assigned to each ability *in the draft
      pool* for a specific match. NOT the same as the game's canonical
      ability_id (used by OpenDota, the wiki, etc.). To get the canonical
      id you need to go through ability_key (the string name like
      "slardar_bash") and look it up externally.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import msgspec

from dota2ad.eval.results import write_results

from typing import cast

from dota2ad.core import (
    AbilityId,
    ExcludeReason,
    HeroId,
    HistoryEvent,
    MatchRow,
    Per10,
    Per12,
    Per36,
    UnifiedIdx,
    VocabKey,
    Vocabs,
    default_paths,
    make_split,
)
from dota2ad.pipeline.build_match_stats import build_match_stats


# ---------------------------------------------------------------------------
# JSON-decoded types (msgspec Structs — decoded directly from bytes)
# ---------------------------------------------------------------------------

class PoolItem(msgspec.Struct):
    draft_ability_id: int


class HeroPoolItem(msgspec.Struct):
    hero_id: int


class AbilityMapping(msgspec.Struct):
    draft_ability_id: int
    player_slot: int
    ability_slot: int
    ability_key: str


class AbilityPick(msgspec.Struct):
    draft_ability_id: int
    player_slot: int
    tick: int
    is_random: bool
    # True when the picker was disconnected at the moment the server timed
    # out the slot. Only meaningful when is_random=True.
    picker_disconnected: bool
    pick_duration: int


class HeroPick(msgspec.Struct):
    hero_id: int
    player_slot: int
    tick: int
    is_random: bool
    picker_disconnected: bool
    pick_duration: int
    hero_key: str


class Player(msgspec.Struct):
    player_slot: int
    leaver_status: int = 0
    rank_tier: int | None = None
    computed_mmr: float | None = None


class MatchDetailsFile(msgspec.Struct):
    match_id: int
    radiant_win: bool
    players: list[Player]
    patch: int  # OpenDota patch index


class Swap(msgspec.Struct):
    tick: int


class DraftDetailsFile(msgspec.Struct):
    hero_pool: list[HeroPoolItem]
    pool_items: list[PoolItem]
    ability_mappings: list[AbilityMapping]
    picks: list[AbilityPick]
    hero_picks: list[HeroPick]
    swaps: list[Swap] = []


# ---------------------------------------------------------------------------
# Internal types (plain dataclasses — constructed in code, not from JSON)
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    tick: int
    player_slot: int
    hero_id: int | None
    draft_ability_id: int | None
    action_key: str
    is_random: bool
    picker_disconnected: bool
    pick_duration: int


@dataclass
class DraftBundle:
    match_id: int
    radiant_win: bool
    players: list[Player]
    pool_items: list[PoolItem]
    hero_pool: list[HeroPoolItem]
    picks: list[AbilityPick]
    hero_picks: list[HeroPick]
    ability_mappings: list[AbilityMapping]
    patch: int
    has_swap: bool = False


def build_ability_id_to_key(bundle: DraftBundle) -> dict[int, str]:
    """Build mapping from draft ability_id to ability_key using ability_mappings."""
    return {m.draft_ability_id: m.ability_key for m in bundle.ability_mappings}


def split_pool(
    pool_items: list[PoolItem],
) -> tuple[list[int], list[int], set[int]]:
    """Split pool_items into basic and ultimate ability ID lists.

    Uses pool position: indices 0-35 are basics, 36-47 are ultimates.
    Returns (basics, ults, ult_ids_set).
    """
    basics = [item.draft_ability_id for item in pool_items[:36]]
    ults = [item.draft_ability_id for item in pool_items[36:]]
    ult_ids = set(ults)
    return basics, ults, ult_ids


def build_match_row(bundle: DraftBundle) -> MatchRow:
    """Build a single MatchRow from a parsed match bundle."""
    match_id = bundle.match_id
    radiant_win = bundle.radiant_win
    ability_id_to_key = build_ability_id_to_key(bundle)

    # MMR from players (ordered by slot)
    players_by_slot = sorted(bundle.players, key=lambda p: p.player_slot)
    mmr_vals = [p.computed_mmr for p in players_by_slot]

    # Build initial pools (pool position: 0-35 basic, 36-47 ult)
    hero_pool = [h.hero_id for h in bundle.hero_pool]
    basic_pool, ult_pool, ult_ids = split_pool(bundle.pool_items)

    # Sanity check: verify position-based ult classification matches ability_slot from parser
    slot_ult_ids = {
        m.draft_ability_id
        for m in bundle.ability_mappings
        if m.ability_slot == 3
    }
    slot_basic_ids = {
        m.draft_ability_id
        for m in bundle.ability_mappings
        if m.ability_slot != 3
    }
    bad_ults = slot_basic_ids & ult_ids
    bad_basics = slot_ult_ids - ult_ids
    if bad_ults or bad_basics:
        raise ValueError(
            f"Match {match_id}: position-based ult classification disagrees with ability_slot. "
            f"Position says ult but slot says basic: {bad_ults}. "
            f"Slot says ult but position says basic: {bad_basics}."
        )

    # Build chronological timeline: combine hero_picks and ability picks
    timeline: list[TimelineEvent] = []

    for hp in bundle.hero_picks:
        timeline.append(TimelineEvent(
            tick=hp.tick,
            player_slot=hp.player_slot,
            hero_id=hp.hero_id,
            draft_ability_id=None,
            action_key=hp.hero_key,
            is_random=hp.is_random,
            picker_disconnected=hp.picker_disconnected,
            pick_duration=hp.pick_duration,
        ))

    for pick in bundle.picks:
        draft_ability_id = pick.draft_ability_id
        timeline.append(TimelineEvent(
            tick=pick.tick,
            player_slot=pick.player_slot,
            hero_id=None,
            draft_ability_id=draft_ability_id,
            action_key=ability_id_to_key[draft_ability_id],
            is_random=pick.is_random,
            picker_disconnected=pick.picker_disconnected,
            pick_duration=pick.pick_duration,
        ))

    timeline.sort(key=lambda e: e.tick)

    # Derive pick_slot from first 10 unique player_slots in timeline
    seen: list[int] = []
    for event in timeline:
        ps = event.player_slot
        if ps not in seen:
            seen.append(ps)
        if len(seen) == 10:
            break
    assert len(seen) == 10, f"Match {match_id}: only {len(seen)} unique players in timeline"

    # Reorder MMR by pick_slot
    sorted_player_slots = sorted(p.player_slot for p in players_by_slot)
    mmr_by_player_slot = dict(zip(sorted_player_slots, mmr_vals, strict=True))
    mmr_by_pick_slot = [mmr_by_player_slot[seen[i]] for i in range(10)]

    # Build history events
    history: list[HistoryEvent] = []
    for event in timeline:
        history.append(HistoryEvent(
            hero_id=HeroId(event.hero_id) if event.hero_id is not None else None,
            draft_ability_id=AbilityId(event.draft_ability_id) if event.draft_ability_id is not None else None,
            action_key=event.action_key,
            is_random=event.is_random,
            picker_disconnected=event.picker_disconnected,
        ))

    return MatchRow(
        match_id=match_id,
        radiant_win=radiant_win,
        mmr=cast(Per10[float | None], tuple(mmr_by_pick_slot)),
        hero_pool=cast(Per12[HeroId], tuple(HeroId(h) for h in hero_pool)),
        basic_pool=cast(Per36[AbilityId], tuple(AbilityId(a) for a in basic_pool)),
        ult_pool=cast(Per12[AbilityId], tuple(AbilityId(a) for a in ult_pool)),
        history=history,
    )


def build_vocabs(
    hero_ids: set[int], draft_ability_ids: set[int],
) -> Vocabs:
    """Build unified vocab mapping from collected ID sets.

    Keys in draft_id_to_index are prefixed: "h:<id>" for heroes, "a:<id>" for abilities.
    """
    draft_id_to_index: dict[VocabKey, UnifiedIdx] = {}
    for h in sorted(hero_ids):
        draft_id_to_index[VocabKey(f"h:{h}")] = UnifiedIdx(len(draft_id_to_index))
    for a in sorted(draft_ability_ids):
        draft_id_to_index[VocabKey(f"a:{a}")] = UnifiedIdx(len(draft_id_to_index))
    draft_id_to_index[VocabKey("<empty>")] = UnifiedIdx(len(draft_id_to_index))

    return Vocabs(draft_id_to_index=draft_id_to_index)


def find_match_dirs(parsed_dirs: list[Path]) -> list[Path]:
    """Collect all <match_id>/ directories across the given parsed dirs.

    Deduplicates by match_id (later dirs win).
    """
    by_id: dict[str, Path] = {}
    for parsed_dir in parsed_dirs:
        for child in sorted(parsed_dir.iterdir()):
            if child.is_dir() and child.name.isdigit():
                by_id[child.name] = child
    return [by_id[k] for k in sorted(by_id)]


_md_decoder = msgspec.json.Decoder(MatchDetailsFile)
_dd_decoder = msgspec.json.Decoder(DraftDetailsFile)


def load_bundle(match_dir: Path) -> DraftBundle:
    """Load match_details.json + draft_details.json into a DraftBundle."""
    md = _md_decoder.decode((match_dir / "match_details.json").read_bytes())
    dd = _dd_decoder.decode((match_dir / "draft_details.json").read_bytes())

    return DraftBundle(
        match_id=md.match_id,
        radiant_win=md.radiant_win,
        players=md.players,
        pool_items=dd.pool_items,
        hero_pool=dd.hero_pool,
        picks=dd.picks,
        hero_picks=dd.hero_picks,
        ability_mappings=dd.ability_mappings,
        has_swap=len(dd.swaps) > 0,
        patch=md.patch,
    )


def write_lookups(match_dirs: list[Path], vocabs, output_dir: Path) -> None:
    """Serving lookups — dataset/lookups.json: id→name maps, the ult-ability set,
    and each hero's canonical basics/ult (majority pool position). The inference
    server requires this artifact at startup instead of re-scanning parsed/."""
    valid: set[tuple[str, int]] = set()
    for key in vocabs.draft_id_to_index:
        if key == "<empty>":
            continue
        kind, raw = key.split(":")
        valid.add((kind, int(raw)))

    ability_names: dict[int, str] = {}
    hero_names: dict[int, str] = {}
    ult_ids: set[int] = set()
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for d in match_dirs:
        with open(d / "draft_details.json") as f:
            game = json.load(f)
        for am in game["ability_mappings"]:
            ability_names[int(am["draft_ability_id"])] = am["ability_key"]
        for hp in game["hero_picks"]:
            hero_names[int(hp["hero_id"])] = hp["hero_key"]
        pool_items = game["pool_items"]
        hero_pool = game["hero_pool"]
        if len(hero_pool) != 12 or len(pool_items) != 48:
            continue
        for i in range(12):
            hid = int(hero_pool[i]["hero_id"])
            basics = [int(pool_items[i * 3 + j]["draft_ability_id"]) for j in range(3)]
            ult = int(pool_items[36 + i]["draft_ability_id"])
            ult_ids.add(ult)
            for ab in (*basics, ult):
                counts[hid][ab] += 1

    hero_abilities: dict[str, dict] = {}
    for hid, ab_counts in counts.items():
        if ("h", hid) not in valid:
            continue
        basics_l: list[tuple[int, int]] = []
        ults_l: list[tuple[int, int]] = []
        for ab, c in ab_counts.items():
            if ("a", ab) not in valid:
                continue
            (ults_l if ab in ult_ids else basics_l).append((ab, c))
        basics_l.sort(key=lambda x: -x[1])
        ults_l.sort(key=lambda x: -x[1])
        hero_abilities[str(hid)] = {
            "basics": [ab for ab, _ in basics_l[:3]],
            "ult": ults_l[0][0] if ults_l else None,
        }

    path = output_dir / "lookups.json"
    with open(path, "w") as f:
        json.dump({
            "ability_id_to_name": {str(k): v for k, v in ability_names.items()},
            "hero_id_to_name": {str(k): v for k, v in hero_names.items()},
            "ult_ids": sorted(ult_ids),
            "hero_abilities": hero_abilities,
        }, f, indent=0)
    print(f"Wrote serving lookups ({len(hero_names)} heroes, "
          f"{len(ability_names)} abilities) -> {path}")


def write_gone_matches(parsed_dirs: list[Path], output_dir: Path) -> int:
    """Aggregate the retrieval-censoring record into dataset/gone_matches.jsonl.

    A `errors/<id>.gone` marker means Valve refused the replay (CDN 403/404), so
    the match never reached matches.jsonl; `collect` persists the already-fetched
    OpenDota record beside the marker (`<id>.details.json`). A marker without a
    details file means the match is unknown to OpenDota too, recorded as
    `missing`. Consumed by
    `experiments/random-mechanism/retrieval.py` (the censoring balance check).
    Deduplicates by match_id across parsed dirs (later dirs win, matching
    `find_match_dirs`)."""
    by_id: dict[int, dict] = {}
    for pd in parsed_dirs:
        errors_dir = pd / "errors"
        if not errors_dir.is_dir():
            continue
        for marker in sorted(errors_dir.glob("*.gone")):
            mid = int(marker.stem)
            details_path = errors_dir / f"{mid}.details.json"
            if not details_path.exists():
                by_id[mid] = {"match_id": mid, "missing": True}
                continue
            d = json.loads(details_path.read_text())
            players = d.get("players") or []
            by_id[mid] = {
                "match_id": mid,
                "start_time": d.get("start_time"),
                "duration": d.get("duration"),
                "game_mode": d.get("game_mode"),
                "lobby_type": d.get("lobby_type"),
                "human_players": d.get("human_players"),
                "radiant_win": d.get("radiant_win"),
                "region": d.get("region"),
                "cluster": d.get("cluster"),
                "od_parsed": d.get("version") is not None,
                "leaver_statuses": [p.get("leaver_status") for p in players],
                "rank_tiers": [p.get("rank_tier") for p in players],
            }
    path = output_dir / "gone_matches.jsonl"
    with open(path, "w") as f:
        for mid in sorted(by_id):
            f.write(json.dumps(by_id[mid]) + "\n")
    n_missing = sum(1 for r in by_id.values() if r.get("missing"))
    print(f"Gone (replay unretrievable): {len(by_id)} matches "
          f"({n_missing} unknown to OpenDota) -> {path}")
    return len(by_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build turn-level causal dataset from parsed match bundles"
    )
    parser.add_argument(
        "--parsed-dir", type=Path, action="append", default=None,
        help="Directory containing parsed match dirs (repeatable, default: parsed)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Directory to write dataset (default: <DOTA2AD_ROOT>/dataset)",
    )
    parser.add_argument(
        "--max-matches", type=int, default=None,
        help="Limit to N matches (randomly sampled with fixed seed)",
    )
    args = parser.parse_args()

    paths = default_paths()
    parsed_dirs = args.parsed_dir or [paths.parsed]
    output_dir = args.output_dir or paths.dataset

    match_dirs = find_match_dirs(parsed_dirs)
    if not match_dirs:
        print(f"No match directories found in {parsed_dirs}")
        return 1

    print(f"Found {len(match_dirs)} match(es) across {len(parsed_dirs)} dir(s)")

    if args.max_matches is not None and args.max_matches < len(match_dirs):
        rng = random.Random(42)
        match_dirs = rng.sample(match_dirs, args.max_matches)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "matches.jsonl"

    MAX_RANDOM = 25
    total_matches = 0
    hero_ids: set[int] = set()
    draft_ability_ids: set[int] = set()
    n_random_by_id: dict[int, int] = {}
    excluded_random: set[int] = set()
    excluded_leaver: set[int] = set()
    excluded_swap: set[int] = set()
    patch_counts: dict[int, int] = {}

    _encode = msgspec.json.encode
    with open(output_file, "wb") as fout:
        for match_dir in match_dirs:
            bundle = load_bundle(match_dir)
            patch_counts[bundle.patch] = patch_counts.get(bundle.patch, 0) + 1

            # Collect vocab IDs from bundle pools (one bundle = one match)
            for h in bundle.hero_pool:
                hero_ids.add(h.hero_id)
            for item in bundle.pool_items:
                draft_ability_ids.add(item.draft_ability_id)

            match_row = build_match_row(bundle)
            fout.write(_encode(match_row) + b"\n")

            # Exclusion checks
            n_random = sum(1 for e in match_row.history if e.is_random)
            n_random_by_id[match_row.match_id] = n_random
            if n_random > MAX_RANDOM:
                excluded_random.add(match_row.match_id)

            if any(p.leaver_status != 0 for p in bundle.players):
                excluded_leaver.add(match_row.match_id)

            if bundle.has_swap:
                excluded_swap.add(match_row.match_id)

            total_matches += 1
            print(f"\r  {total_matches}/{len(match_dirs)} matches", end="", flush=True)

    print(f"\rWrote {total_matches} matches to {output_file}")

    # Patch homogeneity precondition: every downstream consumer assumes one patch —
    # the StatsModel (ability→stat), the BC policy (picks/meta), the recommender, and
    # the random-pick causal eval. Ability balance, the draft pool, and the meta all
    # shift across patches, so a mixed-patch corpus confounds all of them. Refuse to
    # build one rather than silently average across balance changes.
    if len(patch_counts) > 1:
        print(
            f"ERROR: dataset spans multiple patches {dict(sorted(patch_counts.items()))} "
            f"— rebuild from a single-patch collection window."
        )
        return 1
    print(f"Patch: {next(iter(patch_counts))} ({total_matches} matches)")

    vocabs = build_vocabs(hero_ids, draft_ability_ids)
    vocabs_path = output_dir / "vocabs.json"
    with open(vocabs_path, "w") as f:
        json.dump({"draft_id_to_index": vocabs.draft_id_to_index}, f, indent=2)
    write_lookups(match_dirs, vocabs, output_dir)
    n = len(vocabs.draft_id_to_index)
    nh = len(hero_ids)
    print(f"Vocab: {n} IDs ({nh} heroes, {n - nh - 1} abilities) -> {vocabs_path}")

    excluded_path = output_dir / "excluded_matches.json"
    with open(excluded_path, "w") as f:
        json.dump({
            ExcludeReason.TOO_MANY_RANDOM_PICKS: sorted(excluded_random),
            ExcludeReason.LEAVERS: sorted(excluded_leaver),
            ExcludeReason.SWAPS: sorted(excluded_swap),
        }, f)
    print(
        f"Excluded: {len(excluded_random)} random, {len(excluded_leaver)} leaver, "
        f"{len(excluded_swap)} swap -> {excluded_path}"
    )

    # Write train/val/test split
    all_match_ids = [int(d.name) for d in match_dirs]
    val_ids, test_ids = make_split(all_match_ids)
    split_path = output_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump({"val_match_ids": sorted(val_ids), "test_match_ids": sorted(test_ids)}, f)
    print(f"Split: {len(val_ids)} val / {len(test_ids)} test / "
          f"{len(all_match_ids) - len(val_ids) - len(test_ids)} train -> {split_path}")

    n_gone = write_gone_matches(parsed_dirs, output_dir)

    excluded_all = excluded_random | excluded_leaver | excluded_swap
    analytic_ids = [mid for mid in all_match_ids if mid not in excluded_all]
    v_set, t_set = set(val_ids), set(test_ids)
    write_results("dataset", {
        "n_raw": total_matches,
        "n_bots": len(excluded_random),
        "n_leavers": len(excluded_leaver),
        "n_swaps": len(excluded_swap),
        "n_analytic": len(analytic_ids),
        "n_train": sum(1 for m in analytic_ids if m not in v_set and m not in t_set),
        "n_val": sum(1 for m in analytic_ids if m in v_set),
        "n_test": sum(1 for m in analytic_ids if m in t_set),
        "n_train_forced": sum(n_random_by_id[m] for m in analytic_ids
                              if m not in v_set and m not in t_set),
        "n_val_forced": sum(n_random_by_id[m] for m in analytic_ids if m in v_set),
        "n_test_forced": sum(n_random_by_id[m] for m in analytic_ids if m in t_set),
        "n_excluded": len(excluded_random | excluded_leaver | excluded_swap),
        "n_forced": sum(n_random_by_id[m] for m in analytic_ids),
        "n_forced_matches": sum(1 for m in analytic_ids if n_random_by_id[m] > 0),
        "forced_share": sum(n_random_by_id[m] for m in analytic_ids) / (len(analytic_ids) * 50),
        "n_gone": n_gone,
    })

    # Build match stats (ability map + per-match statistics)
    build_match_stats(match_dirs, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
