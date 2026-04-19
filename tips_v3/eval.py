"""GAIA scorer (faithful port of
https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py)
plus an opt-in numeric-tolerance extension for research comparisons.

GAIA's official rules:
  - If GT parses as a float: normalize model answer (strip $, %, ,), float-cast,
    EXACT equality vs float(GT).
  - If GT contains ',' or ';': split both on those chars, POSITIONAL element-wise
    compare. For numeric elements, same as above. For string elements, normalize
    WITHOUT removing punctuation.
  - Otherwise: normalize both (strip whitespace, lowercase, drop punctuation),
    exact string equality.

Normalization does NOT drop articles. The entire string is collapsed to
lowercase-no-whitespace-no-punctuation, then compared with `==`.
"""

from __future__ import annotations

import re
import string


def _is_float(x) -> bool:
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def _normalize_number_str(s: str) -> float:
    for ch in ("$", "%", ","):
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _normalize_str(s: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", s or "")
    if remove_punct:
        return no_spaces.lower().translate(str.maketrans("", "", string.punctuation))
    return no_spaces.lower()


def _split_list(s: str) -> list[str]:
    return re.split(r"[,;]", s or "")


def gaia_scorer(model_answer: str, ground_truth: str) -> bool:
    """Faithful port of GAIA's official question_scorer."""
    if model_answer is None:
        model_answer = "None"

    if _is_float(ground_truth):
        return _normalize_number_str(model_answer) == float(ground_truth)

    if any(ch in ground_truth for ch in (",", ";")):
        gt_elems = _split_list(ground_truth)
        ma_elems = _split_list(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        for ma, gt in zip(ma_elems, gt_elems):
            if _is_float(gt):
                if _normalize_number_str(ma) != float(gt):
                    return False
            elif _normalize_str(ma, remove_punct=False) != _normalize_str(gt, remove_punct=False):
                return False
        return True

    return _normalize_str(model_answer) == _normalize_str(ground_truth)


def gaia_match(model_answer: str, ground_truth: str, *, numeric_rtol: float = 0.01) -> bool:
    """Research-friendly variant: GAIA rules + optional 1% numeric tolerance.

    `numeric_rtol=0` gives the exact official GAIA behavior. Default allows
    within-1% near-misses for flip-rate analysis. Set `0` to match leaderboard
    scoring precisely."""
    if gaia_scorer(model_answer, ground_truth):
        return True
    if numeric_rtol <= 0:
        return False

    # Numeric tolerance path: try float-match with rtol.
    if _is_float(ground_truth):
        norm = _normalize_number_str(model_answer or "")
        gt = float(ground_truth)
        if norm == float("inf"):
            return False
        denom = max(abs(gt), 1e-12)
        return abs(norm - gt) / denom <= numeric_rtol
    return False
