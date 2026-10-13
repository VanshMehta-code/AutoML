"""Tests use a small synthetic dataset with the messy cases the pipeline must handle."""

import numpy as np
import pandas as pd
import pytest

from automl.cleaning import Preprocessor
from automl.config import Config, SearchConfig
from automl.ensemble import greedy_ensemble
from automl.evaluation import Evaluator
from automl.evolution import evolve
from automl.feature_graph import build_feature_graph
from automl.metadata import extract_metadata
from automl.model import FittedModel
from automl.trainer import AutoML


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x1[rng.choice(n, 40, replace=False)] = np.nan
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x2_copy": x2 + rng.normal(0, 1e-4, n),         # redundant with x2
        "cat": rng.choice([f"c{i}" for i in range(30)], n),  # high-cardinality categorical
        "small_cat": rng.choice(["a", "b", "c"], n),
        "when": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "const": 1,
        "uid": [f"id{i}" for i in range(n)],            # identifier
    })
    df["target"] = ((np.nan_to_num(x1) + 0.5 * x2 + rng.normal(0, 0.5, n)) > 0).astype(int)
    return df


def test_metadata_column_kinds():
    meta = extract_metadata(make_df(), "target")
    kinds = {c: i["kind"] for c, i in meta["columns"].items()}
    assert kinds["x1"] == "numeric"
    assert kinds["cat"] == "categorical"
    assert kinds["when"] == "datetime"
    assert kinds["const"] == "constant"
    assert kinds["uid"] == "identifier"
    assert meta["task"] == "binary"


def test_preprocessor_output_is_finite_and_consistent():
    df = make_df()
    meta = extract_metadata(df, "target")
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    pre = Preprocessor(meta).fit(X, y)
    out = pre.transform(X)
    assert out.shape[0] == len(df)
    assert np.isfinite(out).all()
    assert len(pre.feature_names_) == out.shape[1]
    assert "const" in pre.dropped_ and "uid" in pre.dropped_


def test_unseen_categories_do_not_crash():
    df = make_df()
    meta = extract_metadata(df, "target")
    X = df.drop(columns="target")
    pre = Preprocessor(meta).fit(X, df["target"].to_numpy())
    new = X.head(5).copy()
    new["small_cat"] = "never_seen"
    new["cat"] = "brand_new"
    assert np.isfinite(pre.transform(new)).all()


def test_feature_graph_flags_redundant_feature():
    df = make_df()
    meta = extract_metadata(df, "target")
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    pre = Preprocessor(meta).fit(X, y)
    graph = build_feature_graph(pre.transform(X), y, pre.feature_names_, seed=0)
    names = np.array(pre.feature_names_)
    dropped = set(names[graph.drop])
    assert ("x2" in dropped) != ("x2_copy" in dropped)  # exactly one of the pair is removed


def test_evolve_returns_sorted_full_fidelity_results():
    df = make_df(300)
    meta = extract_metadata(df, "target")
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    pre = Preprocessor(meta).fit(X, y)
    evaluator = Evaluator(pre.transform(X), y, n_classes=2, cv_folds=3, seed=0)
    cfg = SearchConfig(families=["logreg", "rf"], population=4, generations=2, elite=2,
                       min_fraction=0.5, eta=2, time_budget_seconds=120)
    results = evolve(evaluator, cfg.families, cfg, seed=0, log=lambda _: None)
    assert results and all(r.fraction == 1.0 and r.oof is not None for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_greedy_ensemble_weights_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good = np.column_stack([1 - y * 0.8 - rng.random(200) * 0.1, y * 0.8 + rng.random(200) * 0.1])
    noise = rng.random((200, 2))
    noise /= noise.sum(axis=1, keepdims=True)
    w = greedy_ensemble([good, noise], y, n_classes=2)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]


def test_end_to_end_training_and_inference(tmp_path):
    df = make_df(400)
    cfg = Config(target="target", cv_folds=3)
    cfg.search = SearchConfig(families=["logreg", "hgb"], population=4, generations=2, elite=2,
                              min_fraction=0.5, eta=2, time_budget_seconds=120,
                              use_aco=True, aco_ants=3, aco_iterations=2, ensemble_size=3)
    automl = AutoML(cfg, log=lambda _: None).fit(df)
    report = automl.model_.report
    assert report["test_auc"] > 0.6
    assert report["features_final"] <= report["features_encoded"]

    path = tmp_path / "model.joblib"
    automl.model_.save(path)
    loaded = FittedModel.load(path)
    proba = loaded.predict_proba(df.drop(columns="target").head(20))
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_missing_input_column_raises(tmp_path):
    df = make_df(200)
    cfg = Config(target="target", cv_folds=3)
    cfg.search = SearchConfig(families=["logreg"], population=2, generations=1, elite=1,
                              min_fraction=0.5, eta=2, time_budget_seconds=60, use_aco=False)
    model = AutoML(cfg, log=lambda _: None).fit(df).model_
    with pytest.raises(ValueError, match="missing required columns"):
        model.predict_proba(df.drop(columns=["x1"]))


def test_missing_target_config_raises():
    with pytest.raises(ValueError, match="config.target"):
        AutoML(Config(target=None))


def test_end_to_end_multiclass(tmp_path):
    df = make_df(450)
    df["target"] = pd.cut(df["x2"] + 0.3 * np.nan_to_num(df["x1"]), bins=3, labels=False)
    cfg = Config(target="target", cv_folds=3)
    cfg.search = SearchConfig(families=["logreg", "hgb", "rf"], population=6, generations=2, elite=2,
                              min_fraction=0.5, eta=2, time_budget_seconds=120,
                              use_aco=False, ensemble_size=3)
    model = AutoML(cfg, log=lambda _: None).fit(df).model_
    assert model.report["task"] == "multiclass"
    proba = model.predict_proba(df.drop(columns="target").head(15))
    assert proba.shape == (15, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert model.report["test_auc"] > 0.6
