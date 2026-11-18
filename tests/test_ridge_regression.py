"""Unit tests for the RidgeRegression example (numerical correctness, not just 'runs').

Run from the repo root with:  pytest tests/test_ridge_regression.py -v
"""
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ridge_regression", ROOT / "02-least-squares" / "examples" / "ridge_regression.py"
)
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    w = np.array([1.5, -2.0, 0.5])
    y = X @ w + 4.0 + rng.normal(scale=0.01, size=100)
    return X, y, w


def test_lambda_zero_recovers_ols(linear_data):
    X, y, w = linear_data
    model = rr.RidgeRegression(lambda_=0.0).fit(X, y)
    np.testing.assert_allclose(model.weights, w, atol=0.05)
    assert model.intercept == pytest.approx(4.0, abs=0.05)
    assert model.score(X, y) > 0.999


def test_matches_closed_form_without_intercept(linear_data):
    X, y, _ = linear_data
    lam = 2.5
    model = rr.RidgeRegression(lambda_=lam, fit_intercept=False).fit(X, y)
    expected = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)
    np.testing.assert_allclose(model.weights, expected)
    assert model.intercept == 0


def test_intercept_is_not_regularized(linear_data):
    X, y, _ = linear_data
    # Shifting y by a constant should shift only the intercept, at any lambda.
    a = rr.RidgeRegression(lambda_=10.0).fit(X, y)
    b = rr.RidgeRegression(lambda_=10.0).fit(X, y + 100.0)
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-8)
    assert b.intercept - a.intercept == pytest.approx(100.0, abs=1e-6)


def test_weight_norm_shrinks_as_lambda_grows(linear_data):
    X, y, _ = linear_data
    norms = [
        np.linalg.norm(rr.RidgeRegression(lambda_=lam).fit(X, y).weights)
        for lam in (0.0, 1.0, 100.0, 1e6)
    ]
    assert norms == sorted(norms, reverse=True)
    assert norms[-1] < 1e-2


def test_predict_shape_and_returns_self(linear_data):
    X, y, _ = linear_data
    model = rr.RidgeRegression()
    assert model.fit(X, y) is model
    assert model.predict(X).shape == y.shape


def test_score_is_zero_for_mean_predictor():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.arange(10, dtype=float)
    model = rr.RidgeRegression(lambda_=1e12).fit(X, y)  # weights ~ 0 -> predicts mean
    assert model.score(X, y) == pytest.approx(0.0, abs=1e-6)


def test_create_polynomial_features_matches_docstring_example():
    out = rr.create_polynomial_features(np.array([1, 2, 3]), degree=2)
    np.testing.assert_array_equal(out, [[1, 1], [2, 4], [3, 9]])


def test_generate_polynomial_data_shapes_and_determinism():
    X1, y1, y_true = rr.generate_polynomial_data(n_samples=20, noise=0.1, seed=7)
    X2, y2, _ = rr.generate_polynomial_data(n_samples=20, noise=0.1, seed=7)
    assert X1.shape == (20, 1) and y1.shape == y_true.shape == (20,)
    np.testing.assert_array_equal(y1, y2)
    assert np.all(np.diff(X1.ravel()) >= 0)  # sorted


def test_cross_validate_lambda_returns_value_from_grid():
    X, y, _ = rr.generate_polynomial_data(n_samples=50, noise=0.2)
    feats = rr.create_polynomial_features(X, degree=5)
    lambdas = np.array([1e-4, 1e-2, 1.0, 100.0])
    scores, best = rr.cross_validate_lambda(feats, y, lambdas, n_folds=5)
    assert scores.shape == lambdas.shape
    assert best in lambdas
    assert best == lambdas[np.argmax(scores)]
