"""StatsDQNSuggester: vector-Q suggester with user-preference scalarization.

Returns per-(action, stat) Q values from a trained `QNetStats`. The
`score_composite` reducer combines via a caller-supplied weight vector;
the default weight vector is the balanced ±1 composite from training.

"""

from __future__ import annotations

from pathlib import Path

import torch

from dota2ad.core.collate import policy_collate
from dota2ad.core.types import PolicySample, Vocabs
from dota2ad.models import QNetStats


def _scalarize_q(q_vec: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Composite scalar Q = Σ_k w_k · Q_vec[k], preserving -inf at
    infeasible rows. Inlined here (instead of imported from
    training.stats_simulator) to avoid a circular import between
    `dota2ad.suggest` and `dota2ad.training.stats_simulator`. The trainer-side
    copy is kept for training-only paths."""
    infeas = q_vec[..., 0].isneginf()
    scalar = (q_vec * weights).sum(dim=-1)
    return scalar.masked_fill(infeas, float("-inf"))


class StatsDQNSuggester:
    def __init__(
        self,
        qnet: QNetStats,
        vocabs: Vocabs,
        mmr_mean: float,
        mmr_std: float,
        device: torch.device,
        k_stats: int,
        stat_names: list[str],
        stat_norm_mean: torch.Tensor,        # [K], CPU
        stat_norm_std: torch.Tensor,         # [K], CPU
        bc_mask_frac: float = 0.0,
    ) -> None:
        self.qnet = qnet.eval()
        self.vocabs = vocabs
        self.mmr_mean = mmr_mean
        self.mmr_std = mmr_std
        self.device = device
        self.k_stats = k_stats
        self.stat_names = stat_names
        self.stat_norm_mean = stat_norm_mean
        self.stat_norm_std = stat_norm_std
        # Hard BC-plausibility mask used at training time; inference applies the
        # same threshold so rare (now-untrained) actions aren't recommended. 0 = off.
        self.bc_mask_frac = bc_mask_frac

    def score_sample(self, sample: PolicySample) -> torch.Tensor:
        """Per-(action, stat) Q values [vocab_size, K]; infeas rows -inf."""
        batch = policy_collate([sample], device=self.device)
        with torch.no_grad():
            return self.qnet(batch).squeeze(0)

    def score_composite(
        self, sample: PolicySample, weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (q_vec [V, K], composite_q [V]). Composite is
        Σ_k w_k · Q_vec[k] with infeasible rows preserved at -inf."""
        q_vec = self.score_sample(sample)
        weights_dev = weights.to(self.device) if weights.device != q_vec.device else weights
        composite = _scalarize_q(q_vec, weights_dev)
        return q_vec, composite


def load_stats_dqn(
    path: Path | str, vocabs: Vocabs, device: torch.device,
) -> StatsDQNSuggester:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab_size = len(vocabs.draft_id_to_index)
    qnet = QNetStats(
        vocab_size,
        d=ckpt["d"],
        n_heads=ckpt["n_heads"],
        k_stats=ckpt["k_stats"],
    ).to(device)
    qnet.load_state_dict(ckpt["state_dict"], strict=True)
    return StatsDQNSuggester(
        qnet=qnet,
        vocabs=vocabs,
        mmr_mean=float(ckpt["mmr_mean"]),
        mmr_std=float(ckpt["mmr_std"]),
        device=device,
        k_stats=int(ckpt["k_stats"]),
        stat_names=list(ckpt["stat_names"]),
        stat_norm_mean=ckpt["stat_norm_mean"].cpu(),
        stat_norm_std=ckpt["stat_norm_std"].cpu(),
        bc_mask_frac=float(ckpt.get("bc_mask_frac", 0.0)),
    )
