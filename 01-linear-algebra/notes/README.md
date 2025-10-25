# Linear Algebra: The Foundation of ML

## Why This Matters for ML/SWE

**ML Perspective:**
- Every neural network operation is matrix multiplication
- Data is always represented as vectors/matrices
- Understanding dimensionality is crucial for debugging model issues

**SWE Perspective:**
- Graph algorithms use adjacency matrices
- Search engines use PageRank (eigenvectors)
- Image compression uses SVD
- Database queries can be optimized with linear algebra

## Core Concepts

### 1. Vectors: Your Data Points

**Intuition:** A vector is just a list of numbers. Each data point in your dataset is a vector.

```python
# A user in a recommendation system
user = [age, income, time_on_site, num_purchases]
# This is a 4-dimensional vector
```

**Key Operations:**
- **Dot product**: Measures similarity (cosine similarity in ML)
- **Norm**: Measures size/length (L2 regularization uses this)

**Real Example:** In word embeddings (Word2Vec), each word is a vector. Similar words have similar vectors (high dot product).

### 2. Matrices: Collections of Data

**Intuition:** A matrix is a spreadsheet. Each row is a data point, each column is a feature.

```python
# User database (n_users × n_features)
users = [
    [25, 50000, 30, 5],   # User 1
    [35, 75000, 45, 12],  # User 2
    [28, 60000, 20, 3],   # User 3
]
```

**Key Concepts:**
- **Matrix multiplication**: Transforms data (like neural network layers)
- **Inverse**: Solves systems of equations (least squares)
- **Rank**: Number of "independent" features (dimensionality)

### 3. Eigenvalues & Eigenvectors: Special Directions

**Intuition:** Eigenvectors are directions that don't change when you apply a transformation, they only get scaled.

**Real Example: PageRank**
- Web pages are nodes, links are edges
- PageRank finds the eigenvector of the adjacency matrix
- This tells you the "importance" of each page

**Real Example: Principal Component Analysis (PCA)**
- Find directions (eigenvectors) with maximum variance
- Project data onto these directions to reduce dimensions
- Used in: image compression, feature extraction, visualization

```python
# PCA in one line
from sklearn.decomposition import PCA
reduced_data = PCA(n_components=2).fit_transform(high_dim_data)
```

### 4. Singular Value Decomposition (SVD): The Swiss Army Knife

**What it does:** Breaks any matrix into three simpler matrices:
```
A = U Σ V^T
```

**Intuition:**
- **U**: Left singular vectors (row patterns)
- **Σ**: Singular values (importance of each pattern)
- **V**: Right singular vectors (column patterns)

**Real Applications:**

#### a) Recommender Systems (Netflix Prize)
```
Users × Movies → (Users × Features) × (Importance) × (Features × Movies)
```
- Missing ratings? Use SVD to fill them in!
- Reduces millions of user-movie pairs to ~50 "latent features"

#### b) Image Compression
```python
# Keep only top k singular values
U, s, Vt = np.linalg.svd(image)
compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
# 90% smaller file, 95% same quality
```

#### c) Latent Semantic Analysis (LSA)
- Find topics in documents
- Documents × Words → Topics
- "King - Man + Woman ≈ Queen" uses this

### 5. Positive Definite Matrices: Convexity Guarantees

**Intuition:** A matrix A is positive definite if the function f(x) = x^T A x is always "bowl-shaped" (convex).

**Why it matters:**
- Guarantees your optimization has a unique solution
- Hessian matrix in Newton's method
- Covariance matrices should be positive definite

**Real Example: Training Neural Networks**
```python
# If loss landscape is convex (positive definite Hessian):
# → Gradient descent always finds global minimum
# → No local minima to get stuck in

# Reality: Deep learning is NOT convex
# → Multiple local minima
# → Need tricks like momentum, learning rate scheduling
```

## The ML Pipeline Connection

Let's connect this to a real ML workflow:

```python
# 1. Load data (each row is a vector)
X = load_data()  # n_samples × n_features matrix

# 2. Normalize (vector operations)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# 3. Reduce dimensions (SVD/PCA)
U, s, Vt = np.linalg.svd(X_normalized)
X_reduced = U[:, :50] @ np.diag(s[:50])  # Keep top 50 components

# 4. Train model (matrix multiplication)
W = (X_reduced.T @ X_reduced + λ*I)^(-1) @ X_reduced.T @ y  # Ridge regression

# 5. Predict (dot product)
y_pred = X_new @ W
```

Every single step uses linear algebra!

## Common Pitfalls (and How to Debug)

### Issue 1: Dimension Mismatch
```python
# Error: shapes (100,50) and (60,10) not aligned
X @ W  # Won't work!

# Fix: Check shapes
print(f"X: {X.shape}, W: {W.shape}")
# W should be (50, output_dim)
```

### Issue 2: Singular Matrix (Can't Invert)
```python
# Error: Matrix is singular
W = np.linalg.inv(X.T @ X) @ X.T @ y

# Fix: Add regularization
W = np.linalg.inv(X.T @ X + λ*np.eye(n_features)) @ X.T @ y
```

### Issue 3: Numerical Instability
```python
# Bad: Direct inversion (numerically unstable)
W = np.linalg.inv(A) @ b

# Good: Use solve (more stable)
W = np.linalg.solve(A, b)
```

## Key Takeaways

1. **Vectors = Data points**, **Matrices = Datasets**
2. **SVD** is the most important matrix factorization in ML
3. **Eigenvalues/eigenvectors** power PCA, PageRank, and spectral clustering
4. **Positive definite matrices** guarantee convex optimization problems
5. **Matrix multiplication** is how neural networks work
6. Always check **dimensions** first when debugging

## Next Steps

- **Practice**: See `examples/svd_recommender.py` for a real recommender system
- **Experiment**: Open `notebooks/linear_algebra_visual.ipynb` for interactive visualizations
- **Apply**: Try implementing PCA from scratch using only NumPy

## Resources

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (Best intuitive explanations)
- [Matrix Calculus Cheat Sheet](http://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf)
- [Numerical Linear Algebra by Trefethen](https://people.maths.ox.ac.uk/trefethen/text.html)
