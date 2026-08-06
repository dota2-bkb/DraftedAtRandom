"""P_mech — the Dota 2 Ability-Draft forced-random (timeout) pick propensity.

The single source of truth for the reverse-engineered timeout mechanism (see
`experiments/random-mechanism/` for the decompilation and verification). On a
pick-timer expiry the server does NOT draw uniformly over the `m` feasible items
(the obvious `1/m` guess). It:

  1. Picks an ability *side* — basics or ults — by a flat 50/50 coin (forced when
     only one side is still needed).
  2. Gathers one bag: all available heroes (if the seat still needs a hero) ∪ all
     available items of the chosen side.
  3. Draws one uniformly from that bag; the call is `RandomInt(0, |heroes|+|side|)`
     and RandomInt is inclusive by contract — an off-by-one in the caller (N+1
     outcomes for an N-entry bag, the extra outcome landing on the ability
     side) that nudges every draw slightly toward abilities.

So the per-item propensity is uniform *within* a kind but non-uniform *across*
kinds whenever a basic and an ult are both still open (the coin caps each side at
½ regardless of how many items it holds). In one-bag states (seat needs
hero+basics, or hero+ult, or a single kind) it collapses to exact `1/m`.
Within-kind uniformity holds at the item level too — the inclusive-RandomInt
overflow does not favor any fixed bag position (random-mechanism T2/T6, verified
on the bot benchmark).

`mech_propensity(cand_type)` returns the per-feasible-item P_mech aligned with the
candidate list; `iw_to_uniform` gives the importance weight `(1/m)/P_mech(A)` that
maps a P_mech sample back onto the uniform estimand; `sample_mechanism_pick` draws
one forced pick (for the episode simulator).
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np


def mech_kind_probs(n_hero: int, n_basic: int, n_ult: int) -> tuple[float, float, float]:
    """P(hero), P(basic), P(ult) for one forced pick given the feasible counts per
    kind. The side-coin averages the hero∪basic and hero∪ult bags; the `+1` is the
    inclusive-RandomInt off-by-one."""
    H, B, U = float(n_hero), float(n_basic), float(n_ult)
    sides = (B > 0) + (U > 0)                 # number of sides the seat still needs
    if sides == 0:                            # hero-only state: the seat must take a hero
        return 1.0, 0.0, 0.0
    ph = pb = pu = 0.0
    if B > 0:
        ph += (1 / sides) * H / (H + B + 1)
        pb = (1 / sides) * (B + 1) / (H + B + 1)
    if U > 0:
        ph += (1 / sides) * H / (H + U + 1)
        pu = (1 / sides) * (U + 1) / (H + U + 1)
    t = ph + pb + pu
    return ph / t, pb / t, pu / t


def mech_propensity(cand_type: Sequence[int]) -> np.ndarray:
    """Per-feasible-item propensity array `P_mech(a | s)`, aligned with `cand_type`
    (0=hero, 1=basic, 2=ult). Sums to 1. Uniform within a kind, coin-split across."""
    ct = np.asarray(cand_type, dtype=int)
    nh, nb, nu = int((ct == 0).sum()), int((ct == 1).sum()), int((ct == 2).sum())
    ph, pb, pu = mech_kind_probs(nh, nb, nu)
    per_kind = (ph / nh if nh else 0.0, pb / nb if nb else 0.0, pu / nu if nu else 0.0)
    return np.array([per_kind[k] for k in ct])


def iw_to_uniform(cand_type: Sequence[int], realized_feasible_index: int) -> float:
    """The importance weight `w = (1/m) / P_mech(A)` for a realized forced pick at
    position `realized_feasible_index` in the candidate list. `E_{A~P_mech}[w·δ·y] =
    (1/m)·Σ_a δ(a)v(a)`, so multiplying each pick's `δ·y` by `w` makes the estimator
    target the uniform-over-feasible estimand the report states. `w = 1` exactly in
    hero-INELIGIBLE one-bag states (P_mech = 1/m there); in hero-eligible one-bag
    states the inclusive `+1` leaves a small skew (w ≈ 0.99–1.02)."""
    p = mech_propensity(cand_type)
    m = len(p)
    return (1.0 / m) / p[realized_feasible_index]


def sample_mechanism_pick(rng: random.Random, cand_idx: Sequence[int],
                          cand_type: Sequence[int]) -> int:
    """Draw one forced pick ~ P_mech from the feasible set — the faithful timeout
    behavior for the episode simulator (replaces a `rng.choice(cand_idx)` uniform
    draw). `rng` is a `random.Random`."""
    p = mech_propensity(cand_type)
    return rng.choices(list(cand_idx), weights=p.tolist(), k=1)[0]
