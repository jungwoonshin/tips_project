"""GAIA answer evaluation: normalized exact match."""
from __future__ import annotations

import re
import string


def _normalize(text: str) -> str:
    """Lowercase, strip, remove punctuation and articles, collapse whitespace."""
    t = text.lower().strip()
    # Remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _try_numeric(text: str) -> float | None:
    """Try to parse text as a number."""
    cleaned = text.replace(",", "").replace(" ", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def evaluate_answer(prediction: str | None, ground_truth: str) -> bool:
    """Compare predicted answer to ground truth using GAIA-style matching."""
    if prediction is None:
        return False
    if not ground_truth or ground_truth == "?":
        return False

    pred_norm = _normalize(prediction)
    gt_norm = _normalize(ground_truth)

    # Direct string match
    if pred_norm == gt_norm:
        return True

    # Numeric match
    pred_num = _try_numeric(prediction)
    gt_num = _try_numeric(ground_truth)
    if pred_num is not None and gt_num is not None:
        if abs(pred_num - gt_num) < 1e-6:
            return True

    # Check if ground truth appears within prediction (for short answers)
    if len(gt_norm) > 0 and gt_norm in pred_norm and len(gt_norm) >= len(pred_norm) * 0.5:
        return True

    return False
