"""Faithful re-implementation of Valve's `CUniformRandomStream` (tier0.dll),
the RNG behind Dota 2's Ability-Draft forced-random picks.

Decompiled from `game/bin/win64/tier0.dll` (exports, ready without full analysis):

  ?RandomInt@?$CUniformRandomStreamImpl@VCThreadNullMutex@@@@QEAAHHH@Z   @ 0x18015e0f0
  ?GenerateRandomNumber_Locked@...                                       @ 0x18015d9d0
  ?SetSeed@...                                                           @ 0x18015dfa0

The core generator is Numerical Recipes' `ran1`: a Park-Miller minimal-standard
LCG (`idum = 16807 * idum mod 2147483647`, i.e. constants 0x41a7 and 0x7fffffff
visible in the disassembly, computed via Schrage's method) wrapped in a
Bays-Durham shuffle table of NTAB=32 entries, indexed by `j = iy >> 26` (the top
bits of the *previous* output). The shuffle is what decorrelates consecutive
draws — verified empirically in `run.py` (test 4).

`RandomInt(mn, mx)` (disasm at 0x18015e0f0): computes `range = mx - mn + 1`; if
`range <= 1` it returns `mn` **without drawing** (the `cmp ebp,1 ; jbe` fast path)
— so `RandomInt(0,0)` does NOT advance the stream. Otherwise it rejection-samples
`GenerateRandomNumber()` to remove modulo bias, then returns `mn + g % range`.
"""
from __future__ import annotations

IM = 2147483647      # 0x7fffffff  (2**31 - 1, the Park-Miller modulus)
IA = 16807           # 0x41a7      (the Park-Miller multiplier)
NTAB = 32


class UniformRandomStream:
    """Bit-faithful port of CUniformRandomStreamImpl<CThreadNullMutex>."""

    def __init__(self, seed: int = 0):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        # SetSeed (0x18015dfa0): m_idum = -abs(seed); m_iy = 0; table lazily filled.
        self._idum = -abs(int(seed))
        self._iy = 0
        self._iv = [0] * NTAB

    def _generate(self) -> int:
        """GenerateRandomNumber_Locked (0x18015d9d0): one ran1 step, in [1, IM-1]."""
        idum = self._idum
        if idum <= 0 or self._iy == 0:            # first call / (re)seed: warm up + fill table
            idum = 1 if idum == 0 else -idum
            for _ in range(8):                    # NR ran1: 8 warm-up iterations
                idum = (IA * idum) % IM
            for k in range(NTAB - 1, -1, -1):     # fill the shuffle table in reverse
                idum = (IA * idum) % IM
                self._iv[k] = idum
            self._iy = self._iv[0]
        idum = (IA * idum) % IM                   # Park-Miller step
        j = self._iy >> 26                        # Bays-Durham shuffle index (0..31)
        out = self._iv[j]
        self._iv[j] = idum
        self._iy = out
        self._idum = idum
        return out

    def random_int(self, mn: int, mx: int) -> int:
        """RandomInt (0x18015e0f0): uniform int in [mn, mx] inclusive."""
        rng = mx - mn + 1
        if rng <= 1:                              # degenerate: return mn, DO NOT draw
            return mn
        # rejection sampling to kill modulo bias (matches the disasm's threshold loop)
        thresh = 0x7fffffff - (0x80000000 % rng)
        while True:
            g = self._generate()
            if g <= thresh:
                return mn + g % rng
