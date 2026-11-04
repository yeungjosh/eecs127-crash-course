# EECS 127 Crash Course for ML/SWE

A hands-on crash course covering key optimization concepts from UC Berkeley's EECS 127 (Optimization Models in Engineering), tailored for Machine Learning Engineers and Software Engineers.

## Course Overview

This crash course focuses on practical implementations of optimization techniques that are fundamental to modern machine learning and data science. Through 7 coding projects, you'll learn how optimization powers everything from image compression to recommender systems.

### What You'll Learn

- **Linear Algebra for ML**: SVD, PCA, matrix factorization
- **Convex Optimization**: Least-squares, L1/L2 regularization, gradient descent
- **Optimization Problems**: Linear programming, quadratic programming
- **Real-world Applications**: Image processing, recommender systems, portfolio optimization, signal processing

## Prerequisites

- **Python** programming (intermediate level)
- **Linear Algebra** basics (matrices, vectors, dot products)
- **Calculus** fundamentals (derivatives, gradients)
- **NumPy** familiarity (helpful but not required)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

### 3. Start Learning!

Follow the recommended order below, or jump to projects that interest you.

## Repository Layout

Besides the notebooks, the repo has per-topic folders with written notes and runnable scripts. See [QUICKSTART.md](QUICKSTART.md) for a 30-minute path through them.

| Folder | Notes | Example script |
|---|---|---|
| `01-linear-algebra/` | Ready | `examples/svd_recommender.py` |
| `02-least-squares/` | Ready | `examples/ridge_regression.py` |
| `03-linear-programming/` | Coming soon | - |
| `04-convex-optimization/` | Ready | `examples/portfolio_optimization.py` |
| `05-duality-kkt/` | Coming soon | - |
| `06-regularization/` | Coming soon | - |
| `07-algorithms/` | Coming soon | - |
| `08-applications/` | Coming soon | - |

Run a script from the repo root, e.g. `python 01-linear-algebra/examples/svd_recommender.py`.

## Projects

### 1. Image Compression with SVD
**File**: `01_image_compression_svd.ipynb`

Learn how Singular Value Decomposition (SVD) enables image compression through low-rank matrix approximation.

**Key Concepts**:
- Singular Value Decomposition (SVD)
- Low-rank approximation
- Compression-quality tradeoff

**ML Relevance**: Foundation for dimensionality reduction, matrix factorization techniques used throughout ML.

---

### 2. PCA for Dimensionality Reduction
**File**: `02_pca_dimensionality_reduction.ipynb`

Implement Principal Component Analysis from scratch and apply it to real datasets.

**Key Concepts**:
- Principal Component Analysis (PCA)
- Eigenvalue decomposition
- Variance maximization

**ML Relevance**: Essential preprocessing technique for high-dimensional data, feature extraction, visualization.

---

### 3. Recommender System with Matrix Factorization
**File**: `03_recommender_matrix_factorization.ipynb`

Build a movie/product recommendation system using collaborative filtering.

**Key Concepts**:
- Matrix factorization
- Alternating Least Squares (ALS)
- Collaborative filtering

**ML Relevance**: Powers Netflix, Amazon, Spotify recommendations. Core technique in recommendation systems.

---

### 4. LASSO Regression & Feature Selection
**File**: `04_lasso_feature_selection.ipynb`

Explore sparse optimization with L1 regularization for automatic feature selection.

**Key Concepts**:
- L1 regularization (LASSO)
- Sparse optimization
- Feature selection vs Ridge regression (L2)

**ML Relevance**: Critical for high-dimensional data, prevents overfitting, automates feature engineering.

---

### 5. Logistic Regression from Scratch
**File**: `05_logistic_regression_gradient_descent.ipynb`

Implement logistic regression using gradient descent for binary classification.

**Key Concepts**:
- Gradient descent (batch vs stochastic)
- Logistic regression
- Convex optimization

**ML Relevance**: Foundation of deep learning optimizers (SGD, Adam), essential classification algorithm.

---

### 6. Portfolio Optimization
**File**: `06_portfolio_optimization.ipynb`

Apply quadratic programming to the classic Markowitz portfolio optimization problem.

**Key Concepts**:
- Quadratic programming (QP)
- Risk-return tradeoff
- Constrained optimization with CVXPy

**ML Relevance**: Demonstrates QP applications, constraint handling, used in resource allocation problems.

---

### 7. Compressed Sensing for Signal Recovery
**File**: `07_compressed_sensing.ipynb`

Recover signals and images from incomplete measurements using L1 minimization.

**Key Concepts**:
- Compressed sensing
- Sparse signal recovery
- L1 minimization

**ML Relevance**: Medical imaging (MRI), signal processing, modern technique for underdetermined systems.

---

## Recommended Learning Path

### For Complete Beginners
Start with projects in order (1 → 7)

### For ML Engineers
Focus on: 2 (PCA) → 3 (Recommender) → 4 (LASSO) → 5 (Logistic Regression) → 7 (Compressed Sensing)

### For Understanding Optimization
Focus on: 5 (Gradient Descent) → 4 (LASSO) → 6 (QP with CVXPy) → 7 (Compressed Sensing)

### For Computer Vision / Signal Processing
Focus on: 1 (SVD) → 2 (PCA) → 7 (Compressed Sensing)

## Tools & Libraries

- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **CVXPy**: Convex optimization modeling
- **scikit-learn**: Reference implementations and datasets
- **Jupyter**: Interactive notebooks

## Additional Resources

### UC Berkeley EECS 127 Official Materials
- [Course Website](https://eecs127.github.io/)
- [Professor El Ghaoui's Page](https://people.eecs.berkeley.edu/~elghaoui/Teaching/EECS127/)

### Textbooks
- **"Optimization Models"** by Calafiore & El Ghaoui
- **"Convex Optimization"** by Boyd & Vandenberghe (free PDF available)

### Online Resources
- [CVXPy Documentation](https://www.cvxpy.org/)
- [Stanford CS229 Notes](http://cs229.stanford.edu/notes/) (ML perspective)
- [Boyd's Convex Optimization Course](https://web.stanford.edu/~boyd/cvxbook/)

## Contributing

Found a bug or have suggestions? Feel free to open an issue or submit a pull request!

## License

This educational material is provided for learning purposes.

---

**Happy Optimizing!** 🚀

*Generated with Claude Code - An AI-powered crash course for mastering optimization fundamentals essential to modern ML engineering.*
