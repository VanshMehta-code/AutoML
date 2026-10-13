"""End-to-end run. The stages execute in this order (details in docs/03_stages.md):

  1. Split off a test set (never touched during search)
  2. Extract metadata from the training split
  3. Fit the Preprocessor on training data and encode it
  4. Build the feature graph and drop redundant features
  5. ACO feature-subset selection (optional, returns None if it doesn't help)
  6. Evolutionary hyperparameter search with successive halving
  7. Greedy ensemble on out-of-fold predictions; refit members on all training rows
  8. Score on the held-out test set and package a FittedModel
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import models
from .aco import aco_select
from .cleaning import Preprocessor
from .config import Config
from .ensemble import greedy_ensemble
from .evaluation import Evaluator
from .evolution import evolve
from .feature_graph import build_feature_graph
from .metadata import CONSTANT, IDENTIFIER, extract_metadata
from .metrics import auc_score
from .model import FittedModel


class AutoML:
    def __init__(self, config: Config, log: Callable[[str], None] = print):
        if not config.target:
            raise ValueError("config.target must be set (in the YAML or with --target).")
        self.config = config
        self.log = log
        self.model_: FittedModel | None = None

    def fit(self, df: pd.DataFrame) -> "AutoML":
        cfg, target, s = self.config, self.config.target, self.config.search
        start = time.monotonic()
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

        df = df.dropna(subset=[target]).reset_index(drop=True)
        train, test = train_test_split(df, test_size=cfg.test_size, stratify=df[target], random_state=cfg.seed)

        # 1-2. Metadata from training rows only
        meta = extract_metadata(train, target)
        self.log(f"[metadata] rows={meta['n_rows']} features={meta['n_features']} "
                 f"task={meta['task']} classes={meta['n_classes']}")

        X_tr, X_te = train.drop(columns=[target]), test.drop(columns=[target])
        le = LabelEncoder().fit(df[target])
        y_tr, y_te = le.transform(train[target]), le.transform(test[target])
        n_classes = len(le.classes_)

        # 3. Cleaning and encoding, fitted on training data only
        pre = Preprocessor(meta, cfg.cleaning.outlier_k, cfg.cleaning.high_cardinality).fit(X_tr, y_tr)
        X_enc = pre.transform(X_tr)

        # 4. Feature graph: drop redundant features
        graph = build_feature_graph(X_enc, y_tr, pre.feature_names_, cfg.seed, cfg.cleaning.redundancy_threshold)
        graph_mask = ~graph.drop
        X_sel = X_enc[:, graph_mask]
        self.log(f"[graph] encoded={X_enc.shape[1]} redundant_dropped={int(graph.drop.sum())}")

        evaluator = Evaluator(X_sel, y_tr, n_classes, cfg.cv_folds, cfg.seed)

        # 5. ACO feature selection over the graph-filtered columns
        aco_mask = None
        if s.use_aco:
            aco_mask = aco_select(
                evaluator, graph.relevance[graph_mask], params={"C": 1.0}, fraction=s.min_fraction,
                n_ants=s.aco_ants, n_iter=s.aco_iterations, evaporation=s.aco_evaporation,
                alpha=s.aco_alpha, beta=s.aco_beta, seed=cfg.seed, log=self.log,
            )
            self.log("[aco] no subset beat all features; keeping them" if aco_mask is None
                     else f"[aco] selected {int(aco_mask.sum())}/{len(aco_mask)} features")

        # 6. Evolutionary hyperparameter search
        evals = evolve(evaluator, s.families, s, seed=cfg.seed, mask=aco_mask, log=self.log)
        if not evals:
            raise RuntimeError("Search produced no results; increase time_budget_seconds.")
        top = evals[: s.ensemble_size]

        # 7. Ensemble weights from OOF predictions; refit kept members on all training rows
        weights = greedy_ensemble([e.oof for e in top], y_tr, n_classes)
        full_mask = graph_mask.copy()
        if aco_mask is not None:
            full_mask = np.zeros_like(graph_mask)
            full_mask[np.flatnonzero(graph_mask)[aco_mask]] = True
        X_model = X_enc[:, full_mask]

        members, kept = [], []
        for e, w in zip(top, weights):
            if w > 0:
                members.append(models.build(e.family, e.params).fit(X_model, y_tr))
                kept.append(w)
        kept_w = np.array(kept) / np.sum(kept)

        # 8. Honest test score
        X_test = pre.transform(X_te)[:, full_mask]
        proba = sum(w * m.predict_proba(X_test) for w, m in zip(kept_w, members))
        test_auc = auc_score(y_te, proba, n_classes)
        test_acc = float((proba.argmax(axis=1) == y_te).mean())

        self.model_ = FittedModel(
            preprocessor=pre,
            classes=le.classes_,
            mask=full_mask,
            members=members,
            weights=kept_w,
            input_columns=list(X_tr.columns),
            target=target,
            report={
                "task": meta["task"],
                "n_train": len(train),
                "n_test": len(test),
                "dropped_columns": [c for c, i in meta["columns"].items() if i["kind"] in (CONSTANT, IDENTIFIER)],
                "features_encoded": int(X_enc.shape[1]),
                "features_after_graph": int(graph_mask.sum()),
                "features_final": int(full_mask.sum()),
                "candidates_evaluated": len(evals),
                "best_cv_auc": top[0].score,
                "test_auc": test_auc,
                "test_accuracy": test_acc,
                "ensemble": [
                    {"family": m_e.family, "params": m_e.params, "weight": float(w), "cv_auc": m_e.score}
                    for m_e, w in zip([e for e, w in zip(top, weights) if w > 0], kept_w)
                ],
                "seconds": round(time.monotonic() - start, 1),
            },
        )
        self.log(f"[done] cv_auc={top[0].score:.4f}  test_auc={test_auc:.4f}  "
                 f"test_acc={test_acc:.4f}  members={len(members)}")
        return self
