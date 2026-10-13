"""Greedy ensemble selection (Caruana et al.) on out-of-fold predictions.

Models are added one at a time, with replacement, whichever most improves the
blended score. It stops when nothing improves. The result is a weight per model
summing to 1. Works for classification probabilities and regression values alike.
"""

from __future__ import annotations

import numpy as np

from .metrics import task_score


def greedy_ensemble(oofs: list[np.ndarray], y: np.ndarray, task: str, n_classes: int,
                    max_iter: int = 25) -> np.ndarray:
    counts = np.zeros(len(oofs))
    blend = np.zeros_like(oofs[0], dtype=float)
    total = 0
    prev = -np.inf

    for _ in range(max_iter):
        s, i = max(
            (task_score(y, (blend * total + p) / (total + 1), task, n_classes), i)
            for i, p in enumerate(oofs)
        )
        if s <= prev:
            break
        prev = s
        counts[i] += 1
        blend = (blend * total + oofs[i]) / (total + 1)
        total += 1

    return counts / counts.sum()
