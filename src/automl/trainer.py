"""End-to-end run. Stages in order (details in docs/03_stages.md):

  1. Detect the task and split off a test set (the test set never influences a decision)
  2. Drop columns declared in the config
  3. Screen for leakage on training rows and drop flagged columns
  4. Extract metadata from the reduced training split
  5. Fit the Preprocessor on training data and encode it
  6. Build the feature graph and drop redundant features
  7. ACO feature selection (optional; returns None if it does not help)
  8. Evolutionary hyperparameter search
  9. Ensemble on out-of-fold predictions; refit members on all training rows
 10. Score train and test, and package a FittedModel
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
from .leakage import screen_leakage
from .metadata import CONSTANT, IDENTIFIER, detect_task, extract_metadata
from .metrics import task_score
from .model import FittedModel


def _predict(est, X, task: str):
    return est.predict(X) if task == "regression" else est.predict_proba(X)


def _blend(members, weights, X, task: str):
    return sum(w * _predict(m, X, task) for w, m in zip(weights, members))


def _encode_target(s: pd.Series, le: LabelEncoder | None) -> np.ndarray:
    return s.to_numpy(dtype=float) if le is None else le.transform(s)


def _target_summary(y: np.ndarray, task: str) -> dict:
    if task == "regression":
        return {"mean": float(np.mean(y)), "std": float(np.std(y))}
    return {"min_class_frac": float(np.min(np.bincount(y)) / len(y))}


class AutoML:
    def __init__(self, config: Config, log: Callable[[str], None] = print):
        if not config.target:
            raise ValueError("config.target must be set (in the YAML or with --target).")
        self.config = config
        self.log = log
        self.model_: FittedModel | None = None
        self.train_raw_: pd.DataFrame | None = None
        self.test_raw_: pd.DataFrame | None = None
        self.leakage_ = None

    def fit(self, df: pd.DataFrame) -> "AutoML":
        cfg, target, s = self.config, self.config.target, self.config.search
        start = time.monotonic()
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

        df = df.dropna(subset=[target]).reset_index(drop=True)
        task = detect_task(df[target], cfg.task)
        is_reg = task == "regression"
        self.log(f"[task] {task}")

        # 1. Split. Stratify only for classification.
        train, test = train_test_split(
            df, test_size=cfg.test_size, random_state=cfg.seed,
            stratify=None if is_reg else df[target],
        )
        self.train_raw_ = train.reset_index(drop=True)
        self.test_raw_ = test.reset_index(drop=True)
        train, test = self.train_raw_.copy(), self.test_raw_.copy()

        # 2. Columns declared in the config
        absent = [c for c in cfg.drop_columns if c not in train.columns]
        declared = [c for c in cfg.drop_columns if c in train.columns]
        if absent:
            self.log(f"[warn] drop_columns not in data, ignored: {absent}")
        if declared:
            train, test = train.drop(columns=declared), test.drop(columns=declared)
            self.log(f"[drop] declared in config: {declared}")

        le = None if is_reg else LabelEncoder().fit(df[target])
        y_tr = _encode_target(train[target], le)
        y_te = _encode_target(test[target], le)
        n_classes = 0 if is_reg else len(le.classes_)

        # 3. Leakage screen on training rows only
        meta0 = extract_metadata(train, target, task)
        leak = screen_leakage(
            train.drop(columns=[target]), y_tr, meta0, task,
            seed=cfg.seed, threshold=cfg.leakage.threshold, warn_threshold=cfg.leakage.warn_threshold,
        )
        self.leakage_ = leak
        for col, score in sorted(leak.scores.items(), key=lambda kv: -kv[1])[:5]:
            self.log(f"[leak-screen] {col}: single-feature score {score:.3f}")
        if leak.flagged:
            self.log(f"[leak] flagged (score >= {cfg.leakage.threshold}): {leak.flagged}")
        if leak.suspicious:
            self.log(f"[leak] review manually (score >= {cfg.leakage.warn_threshold}): {leak.suspicious}")
        dropped_for_leakage: list[str] = []
        if cfg.leakage.action == "drop" and leak.flagged:
            dropped_for_leakage = list(leak.flagged)
            train, test = train.drop(columns=dropped_for_leakage), test.drop(columns=dropped_for_leakage)
            self.log("[leak] dropped flagged columns")

        # 4-5. Metadata and encoding from the reduced training split
        meta = extract_metadata(train, target, task)
        X_tr, X_te = train.drop(columns=[target]), test.drop(columns=[target])
        self.log(f"[metadata] rows={meta['n_rows']} features={meta['n_features']} "
                 f"task={task} classes={meta['n_classes']}")
        pre = Preprocessor(meta, cfg.cleaning.outlier_k, cfg.cleaning.high_cardinality).fit(X_tr, y_tr)
        X_enc = pre.transform(X_tr)

        # 6. Feature graph
        graph = build_feature_graph(X_enc, y_tr, pre.feature_names_, task, cfg.seed,
                                    cfg.cleaning.redundancy_threshold)
        graph_mask = ~graph.drop
        X_sel = X_enc[:, graph_mask]
        self.log(f"[graph] encoded={X_enc.shape[1]} redundant_dropped={int(graph.drop.sum())}")

        families = models.families_for(task, s.families)
        evaluator = Evaluator(X_sel, y_tr, task, n_classes, cfg.cv_folds, cfg.seed)

        # 7. ACO feature selection
        aco_mask = None
        if s.use_aco:
            aco_family = "ridge" if is_reg else "logreg"
            aco_mask = aco_select(
                evaluator, graph.relevance[graph_mask], params=models.default_params(aco_family),
                family=aco_family, fraction=s.min_fraction, n_ants=s.aco_ants, n_iter=s.aco_iterations,
                evaporation=s.aco_evaporation, alpha=s.aco_alpha, beta=s.aco_beta,
                seed=cfg.seed, log=self.log,
            )
            self.log("[aco] no subset beat all features; keeping them" if aco_mask is None
                     else f"[aco] selected {int(aco_mask.sum())}/{len(aco_mask)} features")

        # 8. Evolutionary search
        evals = evolve(evaluator, families, s, seed=cfg.seed, mask=aco_mask, log=self.log)
        if not evals:
            raise RuntimeError("Search produced no results; increase time_budget_seconds.")
        top = evals[: s.ensemble_size]

        # 9. Ensemble weights from out-of-fold predictions; refit kept members on all training rows
        weights = greedy_ensemble([e.oof for e in top], y_tr, task, n_classes)
        full_mask = graph_mask.copy()
        if aco_mask is not None:
            full_mask = np.zeros_like(graph_mask)
            full_mask[np.flatnonzero(graph_mask)[aco_mask]] = True
        X_model = X_enc[:, full_mask]

        kept = [(e, w) for e, w in zip(top, weights) if w > 0]
        members = [models.build(e.family, e.params, task).fit(X_model, y_tr) for e, _ in kept]
        kept_w = np.array([w for _, w in kept])
        kept_w = kept_w / kept_w.sum()

        # 10. Train and test scores, so overfitting is visible
        X_test = pre.transform(X_te)[:, full_mask]
        train_pred = _blend(members, kept_w, X_model, task)
        test_pred = _blend(members, kept_w, X_test, task)
        train_score = task_score(y_tr, train_pred, task, n_classes)
        test_score = task_score(y_te, test_pred, task, n_classes)
        score_name = "R2" if is_reg else "AUC"

        metrics = {
            "score_name": score_name,
            "train_score": train_score,
            "test_score": test_score,
            "train_test_gap": train_score - test_score,
        }
        if is_reg:
            metrics["test_rmse"] = float(np.sqrt(np.mean((test_pred - y_te) ** 2)))
        else:
            metrics["test_accuracy"] = float((test_pred.argmax(axis=1) == y_te).mean())

        self.model_ = FittedModel(
            preprocessor=pre,
            task=task,
            classes=None if is_reg else le.classes_,
            mask=full_mask,
            members=members,
            weights=kept_w,
            input_columns=list(X_tr.columns),
            target=target,
            report={
                "task": task,
                "n_train": len(train),
                "n_test": len(test),
                "declared_drops": declared,
                "dropped_for_leakage": dropped_for_leakage,
                "review_for_leakage": leak.suspicious,
                "leak_scores": {k: round(v, 4) for k, v in leak.scores.items()},
                "dropped_constant_or_id": [c for c, i in meta0["columns"].items()
                                           if i["kind"] in (CONSTANT, IDENTIFIER)],
                "features_encoded": int(X_enc.shape[1]),
                "features_after_graph": int(graph_mask.sum()),
                "features_final": int(full_mask.sum()),
                "candidates_evaluated": len(evals),
                "best_cv_score": top[0].score,
                "metrics": metrics,
                "target_train": _target_summary(y_tr, task),
                "target_test": _target_summary(y_te, task),
                "ensemble": [
                    {"family": e.family, "params": e.params, "weight": float(w), "cv_score": e.score}
                    for e, w in kept
                ],
                "seconds": round(time.monotonic() - start, 1),
            },
        )
        self.log(f"[done] {score_name} cv={top[0].score:.4f}  train={train_score:.4f}  "
                 f"test={test_score:.4f}  members={len(members)}")
        return self
