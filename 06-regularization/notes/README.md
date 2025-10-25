# Regularization: The Art of Not Overfitting

> **Status:** Coming soon! Check back for detailed notes.

## Preview

Regularization adds a penalty term to prevent overfitting:

**L2 (Ridge):** `||Xw - y||² + λ||w||²`
- Shrinks all weights
- Smooth, differentiable

**L1 (LASSO):** `||Xw - y||² + λ||w||₁`
- Sparse solutions (sets weights to exactly 0)
- Feature selection

**Elastic Net:** `||Xw - y||² + λ₁||w||₁ + λ₂||w||²`
- Best of both worlds

## When to Use What

| Problem | Use |
|---------|-----|
| Many correlated features | Ridge or Elastic Net |
| Feature selection needed | LASSO |
| Prediction only | Ridge (faster) |
| Interpretation needed | LASSO (sparse) |

## Quick Comparison

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge: shrinks all coefficients
ridge = Ridge(alpha=1.0).fit(X, y)
print(f"Non-zero: {np.sum(ridge.coef_ != 0)}")  # Usually all

# LASSO: sets many to zero
lasso = Lasso(alpha=1.0).fit(X, y)
print(f"Non-zero: {np.sum(lasso.coef_ != 0)}")  # Sparse!

# Elastic Net: balance
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X, y)
```

## The Bayesian Interpretation

Regularization = Prior belief about parameters!

- **L2** = Gaussian prior: "Weights should be small"
- **L1** = Laplace prior: "Weights should be sparse"

## Resources
- [StatQuest: Ridge, LASSO, Elastic Net](https://www.youtube.com/watch?v=Q81RR3yKn30)
- [Elements of Statistical Learning Ch. 3](https://hastie.su.domains/ElemStatLearn/)
