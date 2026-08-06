"""Set Transformer building blocks (Lee et al., 2019).

Extended with pad_mask support for variable-length sets.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block."""

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            Q: [B, Nq, dim_Q]
            K: [B, Nk, dim_K]
            pad_mask: [B, Nk] True = valid key position (optional)
        Returns: [B, Nq, dim_V]
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
        if pad_mask is not None:
            # [B, Nk] -> [B*num_heads, 1, Nk]
            m = (~pad_mask).repeat(self.num_heads, 1).unsqueeze(1)
            A = A.masked_fill(m, float("-inf"))
        A = torch.softmax(A, 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        return O


class SAB(nn.Module):
    """Set Attention Block: self-attention within a set."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads)

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(X, X, pad_mask)


class PMA(nn.Module):
    """Pooling by Multihead Attention: learned seeds attend over the set."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, pad_mask)


class SetTransformer(nn.Module):
    """Permutation-invariant set encoder: proj -> SAB -> PMA(k=1) -> [n_sets, d_out]"""

    def __init__(self, d_in: int, d_out: int, n_heads: int = 4):
        super().__init__()
        self.d_out = d_out
        self.proj = nn.Linear(d_in, d_out)
        self.sab = SAB(d_out, d_out, n_heads)
        self.pma = PMA(d_out, n_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [n_sets, max_len, d_in] pre-embedded set elements
            mask: [n_sets, max_len] True = valid element
        Returns: [n_sets, d_out]
        """
        h = self.proj(x)
        h = self.sab(h, mask)
        return self.pma(h, mask).squeeze(1)  # [n_sets, d_out]
