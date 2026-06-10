"""Synthetic edit operators for robustness checks."""

from __future__ import annotations

import random

BASES = "ACGT"


def substitute(sequence: str, count: int, *, seed: int = 0) -> str:
    """Apply ``count`` substitutions without changing sequence length."""

    if count < 0 or count > len(sequence):
        raise ValueError("substitution count must lie between 0 and sequence length")
    rng = random.Random(seed)
    symbols = list(sequence)
    for position in rng.sample(range(len(symbols)), count):
        original = symbols[position]
        choices = [base for base in BASES if base != original]
        symbols[position] = rng.choice(choices)
    return "".join(symbols)


def insert(sequence: str, count: int, *, seed: int = 0) -> str:
    """Apply ``count`` random insertions."""

    if count < 0:
        raise ValueError("insertion count must be nonnegative")
    rng = random.Random(seed)
    symbols = list(sequence)
    for _ in range(count):
        position = rng.randrange(len(symbols) + 1)
        symbols.insert(position, rng.choice(BASES))
    return "".join(symbols)


def delete(sequence: str, count: int, *, seed: int = 0) -> str:
    """Apply ``count`` deletions while preventing an empty sequence."""

    if count < 0 or count >= len(sequence):
        raise ValueError("deletion count must be nonnegative and smaller than sequence length")
    rng = random.Random(seed)
    symbols = list(sequence)
    for _ in range(count):
        position = rng.randrange(len(symbols))
        del symbols[position]
    return "".join(symbols)
