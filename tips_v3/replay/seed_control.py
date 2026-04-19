"""Canonical seed lists for each stage."""

from __future__ import annotations

SEEDS_GREEDY_INITIAL = [11, 17, 23]
SEEDS_GREEDY_TIEBREAK = [31]
SEEDS_REDUCTION = [11, 17, 23]
SEEDS_FINAL_VERIFY = [11, 17, 23, 31, 41]


def greedy_initial() -> list[int]:
    return list(SEEDS_GREEDY_INITIAL)


def greedy_tiebreak() -> list[int]:
    return list(SEEDS_GREEDY_TIEBREAK)


def reduction() -> list[int]:
    return list(SEEDS_REDUCTION)


def final_verify() -> list[int]:
    return list(SEEDS_FINAL_VERIFY)
