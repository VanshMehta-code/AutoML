"""Ant colony optimisation for feature-subset selection.

Each feature has a pheromone level tau and a heuristic desirability eta (its
relevance from the feature graph). An ant includes feature i with probability
p_i = (tau_i^alpha * eta_i^beta) / (1 + tau_i^alpha * eta_i^beta), clipped to
[0.05, 0.95]. After all ants run, pheromone evaporates and is deposited on the
features used by good subsets. The selector returns the best subset found, or
None if no subset beats using every feature (so ACO never makes things worse).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .evaluation import Evaluator


def aco_select(
    evaluator: Evaluator,
    heuristic: np.ndarray,
    params: dict,
    family: str = "logreg",
    fraction: float = 0.2,
    n_ants: int = 8,
    n_iter: int = 6,
    evaporation: float = 0.2,
    alpha: float = 1.0,
    beta: float = 2.0,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    n = len(heuristic)
    tau = np.ones(n)
    eta = heuristic / heuristic.max() if heuristic.max() > 0 else np.ones(n)
    eta = eta + 1e-6

    baseline = evaluator.evaluate(family, params, fraction, None).score
    best_mask, best_score = None, -np.inf

    for it in range(n_iter):
        masks, scores = [], []
        for _ in range(n_ants):
            weight = (tau**alpha) * (eta**beta)
            p = np.clip(weight / (1.0 + weight), 0.05, 0.95)
            mask = rng.random(n) < p
            if not mask.any():
                mask[rng.integers(n)] = True
            score = evaluator.evaluate(family, params, fraction, mask).score
            masks.append(mask)
            scores.append(score)
            if score > best_score:
                best_score, best_mask = score, mask.copy()

        tau *= 1.0 - evaporation
        scores_arr = np.array(scores)
        spread = np.ptp(scores_arr) + 1e-12
        normalised = (scores_arr - scores_arr.min()) / spread
        for mask, s in zip(masks, normalised):
            tau += s * mask
        tau = np.clip(tau, 0.01, 10.0)

        log(f"[aco] iter {it + 1}/{n_iter}  best_auc={best_score:.4f}  "
            f"baseline_auc={baseline:.4f}  kept={int(best_mask.sum())}/{n}")

    if best_mask is None or best_score <= baseline:
        return None
    return best_mask
