"""Extract per-match statistics from parsed data into match_stats.jsonl.

Also builds the ability key mapping:
downloads ability_ids.json from dotaconstants, compares with ability keys found
in our dataset, and builds the parser->opendota key mapping in memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import msgspec
import requests

from typing import cast

from dota2ad.core import (
    AbilityId,
    Per10,
    StatsRow,
    Turn,
    VocabKey,
    default_paths,
    load_matches,
    turn_to_pick_slot,
)

OPENDOTA_ABILITY_IDS_URL = "https://raw.githubusercontent.com/odota/dotaconstants/refs/heads/master/build/ability_ids.json"
OPENDOTA_HEROES_URL = "https://raw.githubusercontent.com/odota/dotaconstants/refs/heads/master/build/heroes.json"

NUM_PLAYERS = 10
GOLD_REASON_KEYS = [0, 1, 5, 6, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]
XP_REASON_KEYS = [0, 1, 2, 3, 4, 7]



_encoder = msgspec.json.Encoder()


# ---------------------------------------------------------------------------
# Ability/hero key-map helpers
# ---------------------------------------------------------------------------

def _fetch_and_cache(url: str, cache_path: Path) -> dict:
    """Download JSON from URL, caching to disk."""
    if cache_path.exists():
        with open(cache_path) as f:
            print(f"Loaded cached {cache_path.name} from {cache_path}")
            return json.load(f)

    print(f"Fetching {url}...")
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Cached {len(data)} entries to {cache_path}")
    return data


def build_ability_key_map(ability_ids: dict[str, str], matches_path: Path) -> dict[str, str]:
    """Build parser->canonical key map for all ability keys in our dataset.

    The parser's EntityNames keys are the same canonical internal names
    OpenDota uses (including the `_ad` AD-mode variants), so an exact match
    resolves the whole pool. A miss therefore means a genuinely unknown
    ability — new patch content or a parser regression — so fail loudly rather
    than guess.
    """
    od_keys = set(ability_ids.values())
    # Collect ability keys from matches
    matches = load_matches(matches_path, exclude=[])
    our_keys: set[str] = set()
    for m in matches:
        for event in m.history:
            if event.action_key is not None and event.hero_id is None:
                our_keys.add(event.action_key)

    mapping = {key: key for key in our_keys if key in od_keys}
    unresolved = sorted(our_keys - od_keys)

    if unresolved:
        raise ValueError(
            f"Unresolved ability keys ({len(unresolved)}): {unresolved}\n"
            "Each must be an exact OpenDota ability_ids key."
        )

    identity = sum(1 for k, v in mapping.items() if k == v)
    remapped = sum(1 for k, v in mapping.items() if k != v)
    print(f"Ability key map: {len(mapping)} resolved ({identity} exact, {remapped} remapped)")

    return mapping


def build_hero_key_map(heroes: dict, matches_path: Path) -> dict[str, str]:
    """Build parser hero_key -> OpenDota NPC suffix map.

    The parser's hero keys come from the EntityNames string table — the
    npc_dota_hero_* npc name with the prefix stripped — which is exactly
    OpenDota's suffix, so an exact match resolves every hero. A miss means a
    genuinely unknown hero — new patch content or a parser regression — so
    fail loudly rather than guess.
    """
    od_keys: set[str] = set()
    for h in heroes.values():
        od_keys.add(h["name"].removeprefix("npc_dota_hero_"))

    # Collect hero keys from matches
    matches = load_matches(matches_path, exclude=[])
    our_keys: set[str] = set()
    for m in matches:
        for event in m.history:
            if event.hero_id is not None:
                our_keys.add(event.action_key)

    mapping = {key: key for key in our_keys if key in od_keys}
    unresolved = sorted(our_keys - od_keys)

    if unresolved:
        raise ValueError(
            f"Unresolved hero keys ({len(unresolved)}): {unresolved}\n"
            "Each must be an exact npc_dota_hero_* suffix."
        )

    print(f"Hero key map: {len(mapping)} resolved (all exact)")
    return mapping


# ---------------------------------------------------------------------------
# Stats processing
# ---------------------------------------------------------------------------

class UnparsedMatchError(Exception):
    """OpenDota collected the match but didn't run its parse-match job, so
    extended-stats fields (stuns, obs_placed, gold_t, ...) are absent.
    The match still has its draft + win/loss, so it is usable for draft-only
    training like BC; it just lacks the per-player end-game stats the StatsModel
    and stats-DQN (Q) need."""


def _process_match(
    match_dir: Path,
    canonical_key_to_canonical_id: dict[str, int],
    parser_key_to_canonical_key: dict[str, str],
    hero_key_map: dict[str, str],
) -> StatsRow:
    draft_path = match_dir / "draft_details.json"
    match_path = match_dir / "match_details.json"

    with open(draft_path) as f:
        draft = json.load(f)
    with open(match_path) as f:
        match = json.load(f)

    match_id = match["match_id"]
    duration_min = match["duration"] / 60.0

    if any("stuns" not in p for p in match["players"]):
        raise UnparsedMatchError(str(match_id))

    # Sanity check: draft and match details must agree on hero_id per player_slot
    draft_hero_by_slot = {hp["player_slot"]: hp["hero_id"] for hp in draft["hero_picks"]}
    match_hero_by_slot = {p["player_slot"]: p["hero_id"] for p in match["players"]}
    for slot, hid in draft_hero_by_slot.items():
        assert hid == match_hero_by_slot[slot], (
            f"Match {match_id}: hero_id mismatch at slot {slot}: "
            f"draft={hid} match={match_hero_by_slot[slot]}"
        )

    # Build player_slot -> hero_key from draft
    hero_key_by_slot: dict[int, str] = {}
    for hp in draft["hero_picks"]:
        hero_key_by_slot[hp["player_slot"]] = hp["hero_key"]

    # Build player_slot -> list of (ability_key, draft_ability_id)
    player_abilities: dict[int, list[tuple[str, int]]] = {}
    for am in draft["ability_mappings"]:
        ps = am["player_slot"]
        player_abilities.setdefault(ps, []).append(
            (am["ability_key"], am["draft_ability_id"])
        )

    # Build pick_slot ordering from all draft events sorted by tick
    all_events = [(hp["tick"], hp["player_slot"]) for hp in draft["hero_picks"]]
    all_events += [(ap["tick"], ap["player_slot"]) for ap in draft["picks"]]
    all_events.sort(key=lambda e: e[0])
    event_slots = [pslot for _, pslot in all_events]

    # Map player_slot -> pick_slot from first appearance per player_slot
    player_slot_to_pick_slot: dict[int, int] = {}
    for turn, pslot in enumerate(event_slots):
        if pslot not in player_slot_to_pick_slot:
            player_slot_to_pick_slot[pslot] = turn_to_pick_slot(Turn(turn))

    # Sort match_details players by pick_slot (0-9, interleaved radiant/dire).
    players_by_slot = {p["player_slot"]: p for p in match["players"]}
    pick_slot_to_player_slot = {v: k for k, v in player_slot_to_pick_slot.items()}
    players = [players_by_slot[pick_slot_to_player_slot[ps]] for ps in range(NUM_PLAYERS)]

    # Build loadouts and MMR using draft_details
    loadouts: list[list[VocabKey]] = []
    mmr_vals: list[float | None] = []
    for p in players:
        ps = p["player_slot"]
        hp = next(h for h in draft["hero_picks"] if h["player_slot"] == ps)
        loadouts.append(
            [VocabKey(f"h:{hp['hero_id']}")]
            + [VocabKey(f"a:{did}") for _, did in player_abilities[ps]]
        )
        mmr_vals.append(p.get("computed_mmr"))

    # --- Category 1: Scalar stats (22 features, per-minute where noted) ---
    scalar_stats: list[list[float]] = []
    for p in players:
        scalar_stats.append([
            p["kills"] / duration_min,
            p["deaths"] / duration_min,
            p["assists"] / duration_min,
            p["gold_per_min"],
            p["xp_per_min"],
            p["last_hits"] / duration_min,
            p["denies"] / duration_min,
            p["hero_damage"] / duration_min,
            p["tower_damage"] / duration_min,
            p["hero_healing"] / duration_min,
            p["stuns"] / duration_min,
            p["tower_kills"] / duration_min,
            p["obs_placed"] / duration_min,
            p["sen_placed"] / duration_min,
            p["camps_stacked"] / duration_min,
            p["neutral_kills"] / duration_min,
            p["ancient_kills"] / duration_min,
            p["buyback_count"] / duration_min,
            p["teamfight_participation"],
            p["life_state_dead"] / duration_min,
            p["lane_kills"] / duration_min,
            p["rune_pickups"] / duration_min,
        ])

    # --- Category 2: Time-series snapshots at 10/20/30 min ---
    MINUTES = (10, 20, 30)
    gold_t_list: list[list[float]] = []
    xp_t_list: list[list[float]] = []
    lh_t_list: list[list[float]] = []
    time_mask_list: list[list[bool]] = []
    for p in players:
        gold_arr = p["gold_t"]
        xp_arr = p["xp_t"]
        lh_arr = p["lh_t"]
        gold_vals: list[float] = []
        xp_vals: list[float] = []
        lh_vals: list[float] = []
        mask: list[bool] = []
        for minute in MINUTES:
            valid = len(gold_arr) > minute and len(xp_arr) > minute and len(lh_arr) > minute
            mask.append(valid)
            gold_vals.append(float(gold_arr[minute]) if valid else 0.0)
            xp_vals.append(float(xp_arr[minute]) if valid else 0.0)
            lh_vals.append(float(lh_arr[minute]) if valid else 0.0)
        gold_t_list.append(gold_vals)
        xp_t_list.append(xp_vals)
        lh_t_list.append(lh_vals)
        time_mask_list.append(mask)

    # --- Category 3: Kill/death matchups (5 kills + 5 deaths per player) ---
    kill_counts: list[list[float]] = []
    death_counts: list[list[float]] = []
    for i, p in enumerate(players):
        is_radiant = i % 2 == 0  # pick_slot: even=radiant, odd=dire
        enemy_positions = [j for j in range(NUM_PLAYERS) if (j % 2 == 0) != is_radiant]

        killed = p["killed"]
        killed_by = p["killed_by"]

        p_kills: list[float] = []
        p_deaths: list[float] = []
        for enemy_pos in enemy_positions:
            enemy_slot = pick_slot_to_player_slot[enemy_pos]
            hero_key = hero_key_by_slot[enemy_slot]
            npc_key = f"npc_dota_hero_{hero_key_map[hero_key]}"
            # Per-min, consistent with every other target (scalars, damage,
            # gold/xp reasons). Raw counts conflate kill rate with game length.
            p_kills.append(float(killed.get(npc_key, 0)) / duration_min)
            p_deaths.append(float(killed_by.get(npc_key, 0)) / duration_min)
        kill_counts.append(p_kills)
        death_counts.append(p_deaths)

    # Sanity check: kill_counts[i][j] == death_counts[enemy][i's position in enemy's list]
    for i in range(NUM_PLAYERS):
        is_radiant = i % 2 == 0
        enemies = [j for j in range(NUM_PLAYERS) if (j % 2 == 0) != is_radiant]
        for j_idx, enemy in enumerate(enemies):
            # Find i's index in enemy's enemy list
            enemy_enemies = [k for k in range(NUM_PLAYERS) if (k % 2 == 0) != (enemy % 2 == 0)]
            i_idx = enemy_enemies.index(i)
            assert kill_counts[i][j_idx] == death_counts[enemy][i_idx], (
                f"Match {match_id}: kill_counts[{i}][{j_idx}]={kill_counts[i][j_idx]} "
                f"!= death_counts[{enemy}][{i_idx}]={death_counts[enemy][i_idx]}"
            )

    # --- Category 3b: Damage dealt to each enemy (same pairing as matchups) ---
    # Also extract per-(player, ability, enemy) damage from damage_targets,
    # keyed by draft_ability_id (re-sorted to match ability_draft_ids below).
    damage_dealt: list[list[float]] = []
    spell_damage_by_did: list[dict[int, list[float]]] = []
    for i, p in enumerate(players):
        is_radiant = i % 2 == 0
        enemy_positions = [j for j in range(NUM_PLAYERS) if (j % 2 == 0) != is_radiant]
        damage_targets = p["damage_targets"]
        enemy_npc_keys: list[str] = []
        p_damage: list[float] = []
        for enemy_pos in enemy_positions:
            enemy_slot = pick_slot_to_player_slot[enemy_pos]
            hero_key = hero_key_by_slot[enemy_slot]
            npc_key = f"npc_dota_hero_{hero_key_map[hero_key]}"
            enemy_npc_keys.append(npc_key)
            total = sum(damage_targets.get(src, {}).get(npc_key, 0) for src in damage_targets)
            p_damage.append(total / duration_min)
        damage_dealt.append(p_damage)
        ps_for_spells = p["player_slot"]
        p_spell: dict[int, list[float]] = {}
        for ability_key, draft_ability_id in player_abilities[ps_for_spells]:
            p_spell[draft_ability_id] = [
                damage_targets.get(ability_key, {}).get(k, 0) / duration_min
                for k in enemy_npc_keys
            ]
        spell_damage_by_did.append(p_spell)

    # --- Category 3c: Gold/XP reasons ---
    gold_reasons: list[list[float]] = []
    xp_reasons: list[list[float]] = []
    for p in players:
        gold_reasons.append([p["gold_reasons"].get(str(k), 0) / duration_min for k in GOLD_REASON_KEYS])
        xp_reasons.append([p["xp_reasons"].get(str(k), 0) / duration_min for k in XP_REASON_KEYS])

    # --- Category 4: Ability upgrade priority (4 per player) ---
    # Detect abilities upgraded by multiple players (opendota data issue)
    from collections import Counter
    all_upgrade_ids: Counter[int] = Counter()
    for p in players:
        for cid in set(p.get("ability_upgrades_arr", [])):
            all_upgrade_ids[cid] += 1
    shared_upgrade_ids = {cid for cid, count in all_upgrade_ids.items() if count > 1}

    ability_draft_ids: list[list[AbilityId]] = []
    ability_priorities: list[list[float]] = []
    spell_damage_dealt: list[list[list[float]]] = []
    for i, p in enumerate(players):
        ps = pick_slot_to_player_slot[i]
        player_abs = player_abilities[ps]
        ab_draft_ids: list[AbilityId] = []
        ab_priorities: list[float] = []

        upgrades_arr = p.get("ability_upgrades_arr", [])

        # Build canonical_id -> draft_ability_id for this player's 4 abilities
        draft_id_by_canonical: dict[int, int] = {}
        for ability_key, draft_ability_id in player_abs:
            # Find canonical IDs that map to this ability_key
            # Go: ability_key (parser) -> opendota_key -> canonical_id
            # We need to search ability_ids for the opendota_key
            canonical_key = parser_key_to_canonical_key[ability_key]
            draft_id_by_canonical[canonical_key_to_canonical_id[canonical_key]] = draft_ability_id

        # Kez stance-swap aliasing. Kez has four pairs of abilities, each
        # representing one slot in two stance forms (Katana / Sai). When Kez
        # drafts one form, upgrades during play can land on either form's
        # canonical ID depending on which stance was active when the level
        # was spent. Both forms should count as the same drafted slot. Other
        # heroes can't stance-swap, so they only ever upgrade the form they
        # drafted — the aliasing below is a no-op for them.
        if p["hero_id"] == 145:  # Kez
            for kat_key, sai_key in (
                ("kez_echo_slash",     "kez_falcon_rush_ad"),
                ("kez_grappling_claw", "kez_talon_toss_ad"),
                ("kez_kazurai_katana", "kez_shodo_sai_ad"),
                ("kez_raptor_dance",   "kez_ravens_veil_ad"),
            ):
                kat_cid = canonical_key_to_canonical_id[parser_key_to_canonical_key[kat_key]]
                sai_cid = canonical_key_to_canonical_id[parser_key_to_canonical_key[sai_key]]
                kat_in = kat_cid in draft_id_by_canonical
                sai_in = sai_cid in draft_id_by_canonical
                if kat_in and not sai_in:
                    draft_id_by_canonical[sai_cid] = draft_id_by_canonical[kat_cid]
                elif sai_in and not kat_in:
                    draft_id_by_canonical[kat_cid] = draft_id_by_canonical[sai_cid]

        # Sanity check: all upgraded abilities (from match_details) must be in draft loadout
        # Only checks abilities that are in the draftable set (skips talents, sub-abilities, etc.)
        # Warns instead of failing for abilities upgraded by multiple players (opendota data issue)
        canonical_id_to_key = {v: k for k, v in canonical_key_to_canonical_id.items()}
        drafted_canonical_ids = set(draft_id_by_canonical.keys())
        draftable_canonical_keys = set(parser_key_to_canonical_key.values())
        for cid in set(upgrades_arr):
            if cid not in drafted_canonical_ids:
                akey = canonical_id_to_key.get(cid, "")
                if akey in draftable_canonical_keys:
                    if cid in shared_upgrade_ids:
                        print(
                            f"  WARN: Match {match_id} slot {ps}: skipping shared ability "
                            f"{cid} ({akey}) upgraded by multiple players"
                        )
                    else:
                        raise AssertionError(
                            f"Match {match_id} slot {ps}: upgraded ability {cid} ({akey}) "
                            f"not in draft loadout {player_abs}"
                        )

        for _ability_key, draft_ability_id in player_abs:
            ab_draft_ids.append(AbilityId(draft_ability_id))

            # Find positions in upgrades_arr
            canonical_ids_for_this = [
                cid for cid, did in draft_id_by_canonical.items()
                if did == draft_ability_id
            ]

            positions = [pos for pos, cid in enumerate(upgrades_arr) if cid in canonical_ids_for_this]

            if positions and len(upgrades_arr) > 1:
                priority = 1.0 - (sum(positions) / len(positions)) / (len(upgrades_arr) - 1)
            else:
                priority = 0.0

            ab_priorities.append(priority)

        assert len(ab_draft_ids) == 4, (
            f"Match {match_id} player_slot {ps}: expected 4 abilities, got {len(ab_draft_ids)}"
        )

        # Sort by priority descending so per-slot normalization is meaningful
        paired = sorted(zip(ab_priorities, ab_draft_ids, strict=True), key=lambda x: -x[0])
        ability_priorities.append([p for p, _ in paired])
        sorted_dids = [d for _, d in paired]
        ability_draft_ids.append(sorted_dids)
        # Per-spell damage in the same priority-sorted slot order.
        p_spell = spell_damage_by_did[i]
        spell_damage_dealt.append([
            p_spell.get(int(did), [0.0] * 5) for did in sorted_dids
        ])

    return StatsRow(
        match_id=match_id,
        loadouts=cast(Per10[list[VocabKey]], tuple(loadouts)),
        mmr=cast(Per10[float | None], tuple(mmr_vals)),
        ability_draft_ids=cast(Per10[list[AbilityId]], tuple(ability_draft_ids)),
        scalar_stats=cast(Per10[list[float]], tuple(scalar_stats)),
        gold_t=cast(Per10[list[float]], tuple(gold_t_list)),
        xp_t=cast(Per10[list[float]], tuple(xp_t_list)),
        lh_t=cast(Per10[list[float]], tuple(lh_t_list)),
        time_mask=cast(Per10[list[bool]], tuple(time_mask_list)),
        kill_counts=cast(Per10[list[float]], tuple(kill_counts)),
        death_counts=cast(Per10[list[float]], tuple(death_counts)),
        damage_dealt=cast(Per10[list[float]], tuple(damage_dealt)),
        gold_reasons=cast(Per10[list[float]], tuple(gold_reasons)),
        xp_reasons=cast(Per10[list[float]], tuple(xp_reasons)),
        ability_priorities=cast(Per10[list[float]], tuple(ability_priorities)),
        spell_damage_dealt=cast(Per10[list[list[float]]], tuple(spell_damage_dealt)),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_match_stats(match_dirs: list[Path], output_dir: Path) -> None:
    """Build match_stats.jsonl."""
    matches_path = output_dir / "matches.jsonl"
    cache_dir = default_paths().cache
    output_path = output_dir / "match_stats.jsonl"

    ability_ids = _fetch_and_cache(OPENDOTA_ABILITY_IDS_URL, cache_dir / "ability_ids.json")
    heroes = _fetch_and_cache(OPENDOTA_HEROES_URL, cache_dir / "heroes.json")
    parser_key_to_canonical_key = build_ability_key_map(ability_ids, matches_path)
    hero_key_map = build_hero_key_map(heroes, matches_path)
    canonical_key_to_canonical_id = {v: int(k) for k, v in ability_ids.items() if "," not in k}

    print(f"Building match stats from {len(match_dirs)} match dir(s)...")

    rows: list[StatsRow] = []
    skipped_unparsed: list[str] = []
    n_total = len(match_dirs)
    for i, match_dir in enumerate(match_dirs, 1):
        if not match_dir.is_dir():
            continue
        if not (match_dir / "draft_details.json").exists():
            continue
        try:
            rows.append(_process_match(match_dir, canonical_key_to_canonical_id, parser_key_to_canonical_key, hero_key_map))
        except UnparsedMatchError as e:
            skipped_unparsed.append(str(e))
        print(f"\r  {i}/{n_total} matches", end="", flush=True)

    if skipped_unparsed:
        print(f"\nSkipped {len(skipped_unparsed)} matches without OpenDota parse data (no extended stats)")

    with open(output_path, "wb") as out:
        for row in rows:
            out.write(_encoder.encode(row))
            out.write(b"\n")

    print(f"\rWrote {len(rows)} stats rows to {output_path}")


def main():
    paths = default_paths()
    match_dirs = sorted(d for d in paths.parsed.iterdir() if d.is_dir())
    build_match_stats(match_dirs, paths.dataset)


if __name__ == "__main__":
    main()
