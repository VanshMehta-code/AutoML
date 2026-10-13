"""Evolutionary hyperparameter search with successive-halving fidelity.

Each generation:
  1. Screen every candidate on a small row subsample (cheap).
  2. Promote the best 1/eta to full data (expensive, yields out-of-fold predictions).
  3. Breed the next population from the elite, with some fresh random blood.

The loop stops at the generation limit or when the time budget runs out.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import models
from .evaluation import Evaluation, Evaluator


def evolve(
    evaluator: Evaluator,
    families: list[str],
    cfg,
    seed: int = 0,
    mask: np.ndarray | None = None,
    log: Callable[[str], None] = print,
) -> list[Evaluation]:
    """Return every full-fidelity evaluation found, best first."""
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + cfg.time_budget_seconds

    pop = [
        (fam, models.sample_params(fam, rng))
        for fam in (families[i % len(families)] for i in range(cfg.population))
    ]
    finals: dict[str, Evaluation] = {}

    for gen in range(cfg.generations):
        screened = []
        for fam, params in pop:
            if time.monotonic() > deadline:
                break
            screened.append(evaluator.evaluate(fam, params, cfg.min_fraction, mask))
        if not screened:
            break

        screened.sort(key=lambda e: e.score, reverse=True)
        n_promote = max(cfg.elite, len(screened) // cfg.eta)
        for ev in screened[:n_promote]:
            if time.monotonic() > deadline:
                break
            full = evaluator.evaluate(ev.family, ev.params, 1.0, mask)
            finals[full.key()] = full

        if not finals:
            break
        best = max(finals.values(), key=lambda e: e.score)
        log(f"[evolve] gen {gen + 1}/{cfg.generations}  screened={len(screened)}  "
            f"full={len(finals)}  best={best.family} cv_auc={best.score:.4f}")
        if time.monotonic() > deadline:
            break

        elite = sorted(finals.values(), key=lambda e: e.score, reverse=True)[: cfg.elite]
        pop = _breed(elite, families, cfg, rng)

    return sorted(finals.values(), key=lambda e: e.score, reverse=True)


def _breed(elite: list[Evaluation], families: list[str], cfg, rng: np.random.Generator):
    children = [(e.family, dict(e.params)) for e in elite]
    while len(children) < cfg.population:
        if rng.random() < 0.2:  # fresh random candidate keeps diversity up
            fam = families[rng.integers(len(families))]
            children.append((fam, models.sample_params(fam, rng)))
            continue
        a = elite[rng.integers(len(elite))]
        b = elite[rng.integers(len(elite))]
        params = models.crossover(a.params, b.params, rng) if (a.family == b.family and rng.random() < 0.5) else dict(a.params)
        children.append((a.family, models.mutate(a.family, params, rng)))
    return children
