"""Tests use a small synthetic dataset with the messy cases the pipeline must handle."""

import numpy as np
import pandas as pd
import pytest

from automl.aco import aco_select
from automl.cleaning import Preprocessor
from automl.config import Config, SearchConfig
from automl.ensemble import greedy_ensemble
from automl.evaluation import Evaluator
from automl.evolution import evolve
from automl.feature_graph import build_feature_graph
from automl.leakage import screen_leakage
from automl.metadata import detect_task, extract_metadata
from automl.model import FittedModel
from automl.models import families_for
from automl.trainer import AutoML


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x1[rng.choice(n, 40, replace=False)] = np.nan
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x2_copy": x2 + rng.normal(0, 1e-4, n),                 # redundant with x2
        "cat": rng.choice([f"c{i}" for i in range(30)], n),     # high-cardinality categorical
        "small_cat": rng.choice(["a", "b", "c"], n),
        "when": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "const": 1,
        "uid": [f"id{i}" for i in range(n)],                    # identifier
    })
    df["target"] = ((np.nan_to_num(x1) + 0.5 * x2 + rng.normal(0, 0.5, n)) > 0).astype(int)
    return df


def search(families, **kw):
    base = dict(families=families, population=4, generations=2, elite=2, min_fraction=0.5,
                eta=2, time_budget_seconds=120, use_aco=False, ensemble_size=3)
    base.update(kw)
    return SearchConfig(**base)


def quiet_config(target, families, **kw):
    cfg = Config(target=target, cv_folds=3, **{k: v for k, v in kw.items() if k in ("task", "drop_columns")})
    cfg.search = search(families, **kw.get("search", {}))
    return cfg


def fit(df, cfg):
    return AutoML(cfg, log=lambda _: None).fit(df).model_


# ----- task and metadata -------------------------------------------------

def test_detect_task():
    rng = np.random.default_rng(0)
    assert detect_task(pd.Series(rng.integers(0, 2, 100))) == "binary"
    assert detect_task(pd.Series(rng.integers(0, 3, 100))) == "multiclass"
    assert detect_task(pd.Series(rng.normal(size=100))) == "regression"
    assert detect_task(pd.Series(rng.integers(0, 2, 100)), "regression") == "regression"
    with pytest.raises(ValueError):
        detect_task(pd.Series(["a", "b", "c"] * 10), "regression")


def test_metadata_column_kinds():
    meta = extract_metadata(make_df(), "target", "binary")
    kinds = {c: i["kind"] for c, i in meta["columns"].items()}
    assert kinds["x1"] == "numeric"
    assert kinds["cat"] == "categorical"
    assert kinds["when"] == "datetime"
    assert kinds["const"] == "constant"
    assert kinds["uid"] == "identifier"


def test_families_for_respects_task():
    assert families_for("regression", ["hgb", "logreg", "ridge", "mlp"]) == ["hgb", "ridge", "mlp"]
    assert families_for("binary", ["logreg", "ridge", "mlp"]) == ["logreg", "mlp"]
    with pytest.raises(ValueError):
        families_for("regression", ["logreg"])


# ----- cleaning, graph, search, ensemble ---------------------------------

def test_preprocessor_output_is_finite_and_consistent():
    df = make_df()
    meta = extract_metadata(df, "target", "binary")
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    pre = Preprocessor(meta).fit(X, y)
    out = pre.transform(X)
    assert out.shape[0] == len(df)
    assert np.isfinite(out).all()
    assert len(pre.feature_names_) == out.shape[1]
    assert "const" in pre.dropped_ and "uid" in pre.dropped_


def test_unseen_categories_do_not_crash():
    df = make_df()
    meta = extract_metadata(df, "target", "binary")
    X = df.drop(columns="target")
    pre = Preprocessor(meta).fit(X, df["target"].to_numpy())
    new = X.head(5).copy()
    new["small_cat"] = "never_seen"
    new["cat"] = "brand_new"
    assert np.isfinite(pre.transform(new)).all()


def test_feature_graph_flags_redundant_feature():
    df = make_df()
    meta = extract_metadata(df, "target", "binary")
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    pre = Preprocessor(meta).fit(X, y)
    graph = build_feature_graph(pre.transform(X), y, pre.feature_names_, "binary", seed=0)
    dropped = set(np.array(pre.feature_names_)[graph.drop])
    assert ("x2" in dropped) != ("x2_copy" in dropped)


def test_evolve_returns_sorted_full_fidelity_results():
    df = make_df(300)
    meta = extract_metadata(df, "target", "binary")
    X, y = df.drop(columns="target"), df["target"].to_numpy()
    pre = Preprocessor(meta).fit(X, y)
    evaluator = Evaluator(pre.transform(X), y, "binary", 2, cv_folds=3, seed=0)
    cfg = search(["logreg", "rf"], population=4, generations=2, elite=2, min_fraction=0.5, eta=2)
    results = evolve(evaluator, cfg.families, cfg, seed=0, log=lambda _: None)
    assert results and all(r.fraction == 1.0 and r.oof is not None for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_aco_runs_on_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = 3 * X[:, 0] + rng.normal(0, 0.3, 200)
    evaluator = Evaluator(X, y, "regression", 0, cv_folds=3, seed=0)
    mask = aco_select(evaluator, np.array([1.0, .1, .1, .1, .1, .1]),
                      params={"alpha": 1.0}, family="ridge", fraction=1.0,
                      n_ants=3, n_iter=2, log=lambda _: None)
    assert mask is None or mask[0]  # the informative feature must survive if a subset is chosen


def test_greedy_ensemble_weights_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good = np.column_stack([1 - y * 0.8 - rng.random(200) * 0.1, y * 0.8 + rng.random(200) * 0.1])
    noise = rng.random((200, 2))
    noise /= noise.sum(axis=1, keepdims=True)
    w = greedy_ensemble([good, noise], y, "binary", 2)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]


# ----- leakage -----------------------------------------------------------

def test_leak_is_detected_and_dropped_automatically():
    df = make_df(400)
    rng = np.random.default_rng(1)
    df["leak"] = df["target"] + rng.normal(0, 0.01, len(df))  # computed from the outcome
    model = fit(df, quiet_config("target", ["logreg"]))
    assert "leak" in model.report["dropped_for_leakage"]
    assert "leak" not in model.input_columns


def test_leak_kept_when_action_is_warn():
    df = make_df(400)
    rng = np.random.default_rng(1)
    df["leak"] = df["target"] + rng.normal(0, 0.01, len(df))
    cfg = quiet_config("target", ["logreg"])
    cfg.leakage.action = "warn"
    model = fit(df, cfg)
    assert "leak" in model.input_columns
    assert model.report["dropped_for_leakage"] == []
    assert "leak" in model.report["leak_scores"]


def test_screen_leakage_scores_each_feature():
    df = make_df(300)
    meta = extract_metadata(df, "target", "binary")
    report = screen_leakage(df.drop(columns="target"), df["target"].to_numpy(), meta, "binary")
    assert "x2" in report.scores and "const" not in report.scores


def test_declared_drop_columns_are_removed():
    df = make_df(300)
    cfg = quiet_config("target", ["logreg"], drop_columns=["x2_copy"])
    model = fit(df, cfg)
    assert "x2_copy" not in model.input_columns
    assert model.report["declared_drops"] == ["x2_copy"]


# ----- end-to-end --------------------------------------------------------

def test_binary_end_to_end_reports_train_and_test(tmp_path):
    df = make_df(400)
    cfg = quiet_config("target", ["logreg", "hgb", "mlp"])
    cfg.search.use_aco = True
    cfg.search.aco_ants, cfg.search.aco_iterations = 3, 2
    model = fit(df, cfg)
    m = model.report["metrics"]
    assert m["test_score"] > 0.6
    assert {"train_score", "test_score", "train_test_gap"} <= set(m)
    assert model.report["features_final"] <= model.report["features_encoded"]

    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = FittedModel.load(path)
    proba = loaded.predict_proba(df.drop(columns="target").head(20))
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_multiclass_end_to_end():
    df = make_df(450)
    df["target"] = pd.cut(df["x2"] + 0.3 * np.nan_to_num(df["x1"]), bins=3, labels=False)
    model = fit(df, quiet_config("target", ["hgb", "rf", "mlp"], ensemble_size=3))
    assert model.report["task"] == "multiclass"
    proba = model.predict_proba(df.drop(columns="target").head(15))
    assert proba.shape == (15, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert model.report["metrics"]["test_score"] > 0.6


def test_regression_end_to_end():
    df = make_df(400)
    rng = np.random.default_rng(3)
    df["target"] = 2 * np.nan_to_num(df["x2"]) + np.nan_to_num(df["x1"]) + rng.normal(0, 0.5, len(df))
    model = fit(df, quiet_config("target", ["ridge", "hgb", "mlp"]))
    assert model.report["task"] == "regression"
    assert model.report["metrics"]["score_name"] == "R2"
    assert model.report["metrics"]["test_score"] > 0.8
    preds = model.predict(df.drop(columns="target").head(10))
    assert preds.shape == (10,)
    with pytest.raises(ValueError):
        model.predict_proba(df.drop(columns="target").head(3))


def test_missing_input_column_raises():
    df = make_df(200)
    model = fit(df, quiet_config("target", ["logreg"]))
    with pytest.raises(ValueError, match="missing required columns"):
        model.predict_proba(df.drop(columns=["x1"]))


def test_missing_target_config_raises():
    with pytest.raises(ValueError, match="config.target"):
        AutoML(Config(target=None))


def test_invalid_task_in_config_raises():
    with pytest.raises(ValueError, match="task must be one of"):
        Config(target="target", task="clustering")
