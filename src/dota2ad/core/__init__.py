"""Core module: types, encoding, datasets, I/O, normalization, collation."""

from dota2ad.core.collate import labeled_policy_collate, policy_collate, stats_collate
from dota2ad.core.datasets import PolicyDataset
from dota2ad.core.draft_logic import (
    idx,
    make_split,
    pick_slot_team,
    replay_complete,
    replay_to_turn,
    turn_to_pick_slot,
)
from dota2ad.core.encoding import (
    pad_sets,
    encode_labeled_policy_sample,
    encode_loadout,
    encode_mmr,
    encode_policy_sample,
)
from dota2ad.core.io import (
    SplitIds,
    load_excluded_matches,
    load_matches,
    load_split,
    load_stats_rows,
    load_vocabs,
)
from dota2ad.core.mechanism import (
    iw_to_uniform,
    mech_kind_probs,
    mech_propensity,
    sample_mechanism_pick,
)
from dota2ad.core.paths import Paths, default_paths
from dota2ad.core.normalization import (
    build_stats_records,
    compute_mmr_norm,
    compute_stats_norm,
)
from dota2ad.core.types import (
    NUM_PLAYERS,
    AbilityId,
    ExcludeReason,
    HeroId,
    HistoryEvent,
    LabeledPolicyBatch,
    LabeledPolicySample,
    MatchRow,
    Per10,
    Per12,
    Per36,
    per10,
    PickSlot,
    PlayerPickState,
    PolicyBatch,
    PolicySample,
    StatsBatch,
    StatsNormDict,
    StatsRecord,
    StatsRow,
    Turn,
    TurnRow,
    UnifiedIdx,
    Vocabs,
    VocabKey,
)

__all__ = [
    # types
    "NUM_PLAYERS",
    "AbilityId",
    "ExcludeReason",
    "HeroId",
    "HistoryEvent",
    "LabeledPolicyBatch",
    "LabeledPolicySample",
    "MatchRow",
    # paths
    "Paths",
    "Per10",
    "Per12",
    "Per36",
    "PickSlot",
    "PlayerPickState",
    "PolicyBatch",
    # datasets
    "PolicyDataset",
    "PolicySample",
    "SplitIds",
    "StatsBatch",
    "StatsNormDict",
    "StatsRecord",
    "StatsRow",
    "Turn",
    "TurnRow",
    "UnifiedIdx",
    "VocabKey",
    "Vocabs",
    # normalization
    "build_stats_records",
    "compute_mmr_norm",
    "compute_stats_norm",
    "default_paths",
    # encoding
    "encode_labeled_policy_sample",
    "encode_loadout",
    "encode_mmr",
    "encode_policy_sample",
    # draft logic
    "idx",
    # mechanism (forced-random propensity)
    "iw_to_uniform",
    # collate
    "labeled_policy_collate",
    # io
    "load_excluded_matches",
    "load_matches",
    "load_split",
    "load_stats_rows",
    "load_vocabs",
    "make_split",
    "mech_kind_probs",
    "mech_propensity",
    "pad_sets",
    "per10",
    "pick_slot_team",
    "policy_collate",
    "replay_complete",
    "replay_to_turn",
    "sample_mechanism_pick",
    "stats_collate",
    "turn_to_pick_slot",
]
