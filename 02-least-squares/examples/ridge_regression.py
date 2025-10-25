"""
Ridge Regression: Least Squares with L2 Regularization
-------------------------------------------------------
Prevents overfitting by penalizing large weights.

Compares:
1. Ordinary Least Squares (OLS)
2. Ridge Regression (with different λ values)
3. Shows how to choose λ using cross-validation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class RidgeRegression:
    """
    Ridge Regression: minimize ||Xw - y||² + λ||w||²

    The λ term prevents overfitting by penalizing large weights.
    """

    def __init__(self, lambda_: float = 1.0, fit_intercept: bool = True):
        """
        Args:
            lambda_: Regularization strength (λ ≥ 0)
                    λ = 0 → Ordinary Least Squares
                    λ → ∞ → All weights → 0
            fit_intercept: Whether to add intercept term
        """
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.weights = None
        self.intercept = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit ridge regression using closed-form solution.

        Solution: w = (X^T X + λI)^(-1) X^T y

        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) target vector

        Returns:
            self (for method chaining)
        """
        n_samples, n_features = X.shape

        # Add intercept column if needed
        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])

        # Compute closed-form solution
        # (X^T X + λI)^(-1) X^T y
        XtX = X.T @ X
        identity = np.eye(X.shape[1])

        # Don't regularize the intercept term
        if self.fit_intercept:
            identity[0, 0] = 0

        # Solve using stable method (better than explicit inverse)
        self.weights = np.linalg.solve(XtX + self.lambda_ * identity, X.T @ y)

        # Extract intercept
        if self.fit_intercept:
            self.intercept = self.weights[0]
            self.weights = self.weights[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: (n_samples, n_features) feature matrix

        Returns:
            y_pred: (n_samples,) predictions
        """
        return X @ self.weights + self.intercept

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.

        Args:
            X: Feature matrix
            y: True targets

        Returns:
            R² score (1.0 = perfect, 0.0 = predicts mean, <0 = worse than mean)
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


def generate_polynomial_data(
    n_samples: int = 50,
    noise: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for polynomial regression.

    True function: y = 2x - 3x² + noise
    """
    np.random.seed(seed)

    X = np.sort(np.random.uniform(-1, 1, n_samples))
    y_true = 2 * X - 3 * X**2
    y = y_true + np.random.normal(0, noise, n_samples)

    return X.reshape(-1, 1), y, y_true


def create_polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Create polynomial features: [x, x², x³, ..., x^degree]

    Example:
        X = [1, 2, 3]
        degree = 2
        Output = [[1, 1], [2, 4], [3, 9]]
    """
    n_samples = len(X)
    features = np.zeros((n_samples, degree))

    for d in range(1, degree + 1):
        features[:, d-1] = X.ravel() ** d

    return features


def cross_validate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    n_folds: int = 5
) -> Tuple[np.ndarray, float]:
    """
    Find best λ using k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Targets
        lambdas: Array of λ values to try
        n_folds: Number of CV folds

    Returns:
        cv_scores: Mean CV score for each λ
        best_lambda: λ with best CV score
    """
    n_samples = len(X)
    fold_size = n_samples // n_folds
    cv_scores = []

    for lambda_ in lambdas:
        fold_scores = []

        for fold in range(n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size

            val_idx = list(range(val_start, val_end))
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Train and evaluate
            model = RidgeRegression(lambda_=lambda_)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            fold_scores.append(score)

        cv_scores.append(np.mean(fold_scores))

    best_idx = np.argmax(cv_scores)
    best_lambda = lambdas[best_idx]

    return np.array(cv_scores), best_lambda


def main():
    """Demo: Compare OLS vs Ridge on polynomial regression."""

    print("=" * 70)
    print("Ridge Regression vs Ordinary Least Squares")
    print("=" * 70)

    # Generate data
    X, y, y_true = generate_polynomial_data(n_samples=30, noise=0.3)
    print(f"\nGenerated {len(X)} data points")
    print(f"True function: y = 2x - 3x²")

    # Create polynomial features (up to degree 10 - intentionally overfit!)
    degree = 10
    X_poly = create_polynomial_features(X, degree)
    print(f"Using polynomial features up to degree {degree}")

    # Split train/test
    split = int(0.7 * len(X))
    X_train, X_test = X_poly[:split], X_poly[split:]
    y_train, y_test = y[:split], y[split:]

    # Train models with different λ
    lambdas = [0, 0.01, 0.1, 1.0, 10.0]
    models = {}

    print("\n" + "=" * 70)
    print("Training Models")
    print("=" * 70)

    for lambda_ in lambdas:
        model = RidgeRegression(lambda_=lambda_)
        model.fit(X_train, y_train)
        models[lambda_] = model

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        name = "OLS" if lambda_ == 0 else f"Ridge (λ={lambda_})"
        print(f"\n{name}:")
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²:  {test_score:.4f}")
        print(f"  Weight norm: {np.linalg.norm(model.weights):.2f}")

        if lambda_ == 0:
            print(f"  ⚠️  Overfitting! (Train >> Test)")

    # Find best λ using cross-validation
    print("\n" + "=" * 70)
    print("Cross-Validation to Find Best λ")
    print("=" * 70)

    lambda_range = np.logspace(-3, 2, 20)
    cv_scores, best_lambda = cross_validate_lambda(X_train, y_train, lambda_range)

    print(f"\nBest λ found: {best_lambda:.4f}")
    print(f"CV score: {cv_scores[np.argmax(cv_scores)]:.4f}")

    # Train final model with best λ
    final_model = RidgeRegression(lambda_=best_lambda)
    final_model.fit(X_train, y_train)

    print(f"\nFinal model test R²: {final_model.score(X_test, y_test):.4f}")

    # Visualize results
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("""
1. OLS (λ=0) overfits: Perfect on training, poor on test
   → Large weights trying to fit noise

2. Ridge (λ>0) regularizes: Smaller weights, better generalization
   → Trades training accuracy for test performance

3. Too much regularization (λ=10) underfits
   → Weights too constrained, can't fit signal

4. Cross-validation finds the sweet spot
   → Balances bias-variance tradeoff

When to use Ridge:
- More features than samples (p > n)
- Multicollinearity (correlated features)
- Preventing overfitting
- Stable predictions

When NOT to use Ridge:
- Feature selection needed (use LASSO instead)
- You have lots of data and few features
- Interpretability matters (regularization shrinks all weights)
    """)


if __name__ == "__main__":
    main()
