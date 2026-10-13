"""Feature graph: each feature's relevance to the target and its redundancy with others.

Nodes are features. Relevance (mutual information with the target) is the node
weight. Redundancy (|Spearman| between features) is the edge weight. Redundant
features are dropped, keeping the more relevant member of each pair. The
relevance vector also serves as the heuristic for ACO feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_classif


@dataclass
class FeatureGraph:
    names: list[str]
    relevance: np.ndarray   # MI with target, scaled to [0, 1]
    redundancy: np.ndarray  # |Spearman| matrix, shape (p, p)
    drop: np.ndarray        # boolean, True = removed as redundant

    def top_edges(self, k: int = 10) -> list[tuple[str, str, float]]:
        """The k most redundant feature pairs, for reporting."""
        p = len(self.names)
        pairs = [
            (self.names[i], self.names[j], float(self.redundancy[i, j]))
            for i in range(p)
            for j in range(i + 1, p)
        ]
        return sorted(pairs, key=lambda t: -t[2])[:k]


def build_feature_graph(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    seed: int = 0,
    redundancy_threshold: float = 0.95,
) -> FeatureGraph:
    mi = mutual_info_classif(X, y, discrete_features="auto", random_state=seed)
    relevance = mi / mi.max() if mi.max() > 0 else np.zeros_like(mi)

    p = X.shape[1]
    if p == 1:
        corr = np.zeros((1, 1))
    else:
        ranks = rankdata(X, axis=0)
        with np.errstate(invalid="ignore"):
            corr = np.abs(np.corrcoef(ranks, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)

    drop = np.zeros(len(names), dtype=bool)
    for i in np.argsort(-relevance):  # visit most relevant first; it survives its pairs
        if drop[i]:
            continue
        for j in np.flatnonzero(corr[i] >= redundancy_threshold):
            if not drop[j] and relevance[j] <= relevance[i]:
                drop[j] = True

    return FeatureGraph(names=list(names), relevance=relevance, redundancy=corr, drop=drop)
