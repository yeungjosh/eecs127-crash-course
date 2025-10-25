# EECS 127 Crash Course: Optimization for ML Engineers & Software Developers

> A practical, intuition-first guide to Berkeley's EECS 127 (Optimization Models in Engineering) from an ML/SWE perspective

## Overview

This repository provides a crash course for **EECS 127: Optimization Models in Engineering** at UC Berkeley, designed specifically for Machine Learning Engineers and Software Developers who want to understand optimization through practical applications and intuitive examples.

**Why this matters for ML/SWE:**
- **ML Perspective**: Every ML model you've trained uses optimization (gradient descent, Adam, SGD). Understanding the theory helps you debug training, choose better hyperparameters, and design custom loss functions.
- **SWE Perspective**: Resource allocation, constraint solving, recommendation systems, and even compiler optimizations rely on these techniques.

## Course Structure

### 🎯 Core Topics

1. **[Linear Algebra Foundations](./01-linear-algebra/)** - The building blocks
   - Vectors, matrices, and why they matter in ML
   - SVD and PCA (dimensionality reduction in practice)
   - Eigenvalues and eigenvectors (PageRank, spectral clustering)

2. **[Least Squares Optimization](./02-least-squares/)** - Your first optimizer
   - Linear regression from scratch
   - Ridge regression and why it prevents overfitting
   - Applications: recommendation systems, curve fitting

3. **[Linear Programming](./03-linear-programming/)** - Constraint solving
   - Resource allocation (AWS cost optimization)
   - Network flow problems (routing, matching)
   - The Simplex algorithm

4. **[Convex Optimization](./04-convex-optimization/)** - The sweet spot
   - What makes a problem "easy" to solve
   - Convex sets, functions, and why they guarantee global optima
   - LP, QP, SOCP, SDP - the optimization zoo

5. **[Duality & KKT Conditions](./05-duality-kkt/)** - The theory behind SVMs
   - Lagrange multipliers and why they work
   - Understanding SVM dual formulation
   - Constraint handling in practice

6. **[Regularization](./06-regularization/)** - Preventing overfitting
   - L1 (LASSO) vs L2 (Ridge) regularization
   - Feature selection with LASSO
   - Elastic net and beyond

7. **[Optimization Algorithms](./07-algorithms/)** - How optimizers work
   - Gradient descent and its variants (SGD, Adam, RMSprop)
   - Newton's method and second-order optimization
   - When to use what

8. **[Real-World Applications](./08-applications/)** - Putting it all together
   - Training neural networks
   - Portfolio optimization
   - Hyperparameter tuning
   - A/B testing and experimental design

## 🚀 Quick Start

### Prerequisites

**Math Background:**
- Basic linear algebra (matrix multiplication, inverses)
- Calculus (derivatives, gradients)
- Probability (optional but helpful)

**Programming:**
- Python proficiency
- Familiarity with NumPy
- Experience with ML frameworks (PyTorch/TensorFlow) is a plus

### Installation

```bash
git clone https://github.com/yourusername/eecs127-crash-course.git
cd eecs127-crash-course
pip install -r requirements.txt
jupyter lab
```

### Recommended Learning Path

**For ML Engineers:**
1. Start with 01-linear-algebra (focus on SVD/PCA)
2. Jump to 06-regularization (LASSO/Ridge)
3. Study 07-algorithms (gradient descent deep dive)
4. Explore 05-duality-kkt (understand SVMs)
5. Apply in 08-applications

**For Software Engineers:**
1. Start with 03-linear-programming (constraint problems)
2. Move to 04-convex-optimization (problem formulation)
3. Study 07-algorithms (implementation details)
4. Explore 08-applications (system design problems)

**Full Course (Recommended):**
Follow the numerical order (01 → 08) for a comprehensive understanding.

## 📚 What Makes This Different

### Intuition First, Theory Second
- Every concept explained with real-world analogies
- Visual intuitions before mathematical rigor
- "Why does this matter?" for each topic

### Code-Driven Learning
- Every concept implemented from scratch in Python
- Production-ready examples using CVX, SciPy, PyTorch
- Jupyter notebooks for interactive exploration

### ML/SWE Context
- Connect optimization to ML training loops
- Show how to debug convergence issues
- Explain design decisions in popular frameworks

## 🛠️ Tech Stack

- **Python 3.8+**: Core language
- **NumPy**: Numerical computing
- **CVXPY**: Convex optimization modeling
- **SciPy**: Scientific computing
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive notebooks
- **PyTorch** (optional): Deep learning examples

## 📖 Each Topic Includes

```
XX-topic-name/
├── notes/
│   └── README.md          # Conceptual explanation with intuition
├── examples/
│   └── simple_example.py  # Minimal, runnable code
│   └── real_world.py      # Production-style implementation
└── notebooks/
    └── interactive.ipynb  # Step-by-step Jupyter walkthrough
```

## 🎓 Learning Outcomes

After completing this crash course, you will:

- **Recognize** when a problem can be formulated as an optimization problem
- **Choose** the right optimization technique for your use case
- **Implement** optimizers from scratch and use production libraries
- **Debug** training issues in ML models
- **Understand** the theory behind popular ML algorithms (SVMs, LASSO, PCA)
- **Apply** optimization to real-world engineering problems

## 🤝 Contributing

Contributions welcome! Areas we'd love help with:
- Additional real-world examples
- Visualizations and interactive demos
- Exercises with solutions
- Debugging common pitfalls sections

## 📚 References

**Official Course Resources:**
- [EECS 127 Official Page](https://people.eecs.berkeley.edu/~elghaoui/Teaching/EECS127/)
- [Course Livebook](https://livebooklabs.com)

**Recommended Textbooks:**
- Boyd & Vandenberghe - [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
- Strang - Linear Algebra and Learning from Data

**Online Resources:**
- [CVX Users Guide](https://www.cvxpy.org/)
- [Distill.pub - Momentum](https://distill.pub/2017/momentum/)

## 📝 License

MIT License - feel free to use for learning and teaching!

## ⭐ Star This Repo

If you find this helpful, please star the repo and share with others who might benefit!

---

**Built for engineers who want to understand optimization, not just use it.**
