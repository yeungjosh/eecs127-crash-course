# Least Squares: The Foundation of Regression

## Why This Matters for ML/SWE

**ML Perspective:**
- Linear regression is the simplest ML model
- Understanding least squares helps you debug any regression problem
- Ridge/Lasso regression are just least squares + penalties
- Even neural networks use least squares in the final layer for regression tasks

**SWE Perspective:**
- Curve fitting for performance prediction
- Sensor calibration and noise reduction
- Time series forecasting
- A/B test analysis

## The Core Idea

**Problem:** Given data points, find the "best fit" line/curve.

**Mathematical Formulation:**
```
minimize ||Ax - b||²
```

Where:
- `A` is your data matrix (features)
- `x` is the parameters you're solving for (weights)
- `b` is your target values
- `||·||²` is the squared error

**Intuition:** Find the line that minimizes the sum of squared distances from data points.

## Why Squared Error?

You might ask: "Why square the errors? Why not absolute value?"

**Historical Reason:** Gauss chose it because it has a closed-form solution!

**Practical Reasons:**
1. **Differentiable everywhere** (important for optimization)
2. **Penalizes large errors more** (outlier sensitivity)
3. **Statistical optimality** (Maximum Likelihood Estimator under Gaussian noise)
4. **Closed-form solution exists!**

## The Magic Formula

The least squares solution is:

```
x = (A^T A)^(-1) A^T b
```

This is called the **normal equations**.

**Breakdown:**
- `A^T A` is a square matrix (n_features × n_features)
- `A^T b` is a vector (n_features × 1)
- The inverse `(A^T A)^(-1)` exists if columns of A are linearly independent

### Geometric Intuition

The solution `Ax` is the **projection** of `b` onto the column space of `A`.

Think of it as: "Find the point in the subspace spanned by A that's closest to b."

## Linear Regression From Scratch

```python
import numpy as np

def least_squares(X, y):
    """
    Solve least squares: minimize ||Xw - y||²

    Args:
        X: (n_samples, n_features) data matrix
        y: (n_samples,) target vector

    Returns:
        w: (n_features,) optimal weights
    """
    # Method 1: Normal equations (fast for small problems)
    w = np.linalg.inv(X.T @ X) @ X.T @ y

    # Method 2: More numerically stable
    w = np.linalg.solve(X.T @ X, X.T @ y)

    # Method 3: SVD (most stable, works even if X^T X is singular)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    w = Vt.T @ np.diag(1/s) @ U.T @ y

    return w
```

## When Least Squares Fails

### Problem 1: Singular Matrix (Not Enough Data)

```python
# More features than data points
X = np.random.randn(10, 50)  # 10 samples, 50 features
y = np.random.randn(10)

# This will fail! X^T X is not invertible
w = np.linalg.inv(X.T @ X) @ X.T @ y  # ERROR
```

**Solution:** Use **Ridge Regression** (add regularization)

### Problem 2: Multicollinearity (Features are Correlated)

```python
# Feature 2 = 2 * Feature 1 (perfectly correlated)
X = np.array([[1, 2], [2, 4], [3, 6]])

# X^T X is singular (not invertible)
```

**Solution:** Use **Ridge Regression** or remove correlated features

### Problem 3: Outliers (Bad Data)

```python
# One outlier can ruin the fit
X = np.array([1, 2, 3, 4, 100])  # 100 is an outlier
y = np.array([2, 4, 6, 8, 1000])

# Least squares tries too hard to fit the outlier
```

**Solution:** Use **Huber loss** or **RANSAC** (robust regression)

## Ridge Regression: Least Squares + Regularization

**Problem with plain least squares:** Overfitting when you have many features.

**Solution:** Penalize large weights.

```
minimize ||Ax - b||² + λ||x||²
```

The regularization term `λ||x||²` encourages smaller weights.

**Closed-form solution:**
```python
w = np.linalg.inv(X.T @ X + λ * np.eye(n_features)) @ X.T @ y
```

Notice the `+ λ * I`? This ensures the matrix is always invertible!

**How to choose λ?**
- Small λ → More overfitting (trusts data)
- Large λ → More underfitting (trusts regularization)
- Use cross-validation to find the sweet spot

## Real-World Example: House Price Prediction

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
X = load_features()  # [sq_ft, bedrooms, bathrooms, age, ...]
y = load_prices()

# Preprocessing (IMPORTANT!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept term (bias)
X_scaled = np.hstack([np.ones((len(X), 1)), X_scaled])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Train with Ridge regression
λ = 0.1
w = np.linalg.inv(X_train.T @ X_train + λ * np.eye(X_train.shape[1])) @ X_train.T @ y_train

# Predict
y_pred = X_test @ w

# Evaluate
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error: ${mse**0.5:.2f}")
```

## Weighted Least Squares: When Some Data is More Reliable

Sometimes you trust some data points more than others.

**Example:** Recent data is more reliable than old data in time series.

```
minimize ||W^(1/2) (Ax - b)||²
```

Where `W` is a diagonal matrix of weights.

**Solution:**
```python
# Higher weight = more important
weights = np.array([1, 1, 1, 10, 10])  # Last two points are more important
W = np.diag(weights)

w = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
```

## Recursive Least Squares: Online Learning

What if data arrives in a stream? Re-computing from scratch is wasteful.

**Recursive update:**
```python
# Initialize
w = initial_guess
P = np.eye(n_features)

# For each new data point (x_new, y_new):
K = (P @ x_new) / (1 + x_new.T @ P @ x_new)  # Kalman gain
w = w + K * (y_new - x_new.T @ w)  # Update weights
P = P - K @ x_new.T @ P  # Update covariance
```

This is used in:
- Kalman filters (robotics, GPS)
- Online regression (streaming data)
- Adaptive systems (tracking changing patterns)

## Least Squares vs. Gradient Descent

**Least Squares (Closed-form):**
- ✅ Exact solution in one step
- ✅ No hyperparameters (no learning rate)
- ❌ Slow for large problems (matrix inversion is O(n³))
- ❌ Requires all data in memory

**Gradient Descent (Iterative):**
- ✅ Scalable to billions of samples
- ✅ Works with streaming data (SGD)
- ✅ Memory efficient
- ❌ Requires tuning (learning rate, iterations)
- ❌ Only finds approximate solution

**Rule of thumb:**
- n < 10,000 features → Use closed-form
- n > 10,000 features → Use gradient descent

## Common Mistakes and How to Fix Them

### Mistake 1: Forgetting the Intercept
```python
# Wrong: No intercept term
w = np.linalg.solve(X.T @ X, X.T @ y)

# Right: Add column of ones
X_with_intercept = np.hstack([np.ones((len(X), 1)), X])
w = np.linalg.solve(X_with_intercept.T @ X_with_intercept, X_with_intercept.T @ y)
```

### Mistake 2: Not Scaling Features
```python
# Feature 1: square footage (1000-5000)
# Feature 2: number of bedrooms (1-5)
# The scale difference will dominate!

# Fix: Standardize
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

### Mistake 3: Using Normal Equations for Tall Matrices
```python
# If n_samples >> n_features, X^T X is cheaper
# If n_samples << n_features, use SVD instead

if n_samples > 10 * n_features:
    w = np.linalg.solve(X.T @ X, X.T @ y)  # O(n_features³)
else:
    w = np.linalg.lstsq(X, y)[0]  # Uses SVD, more stable
```

## Key Takeaways

1. **Least squares minimizes squared error** - closed-form solution exists
2. **Normal equations:** `x = (A^T A)^(-1) A^T b`
3. **Ridge regression** adds `λ||x||²` to prevent overfitting
4. **Always scale features** before regression
5. **Use `np.linalg.solve` instead of `np.linalg.inv`** (more stable)
6. **For large problems**, use gradient descent instead
7. **Least squares = projection** onto column space of A

## Next Steps

- **Practice**: See `examples/ridge_regression.py` for a complete implementation
- **Experiment**: Try `notebooks/least_squares_visual.ipynb` to visualize fits
- **Challenge**: Implement LASSO from scratch using iterative methods

## Resources

- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) (Section on least squares)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Numerical Recipes: Linear Least Squares](http://numerical.recipes/)
