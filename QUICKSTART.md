# Quick Start Guide

## 30-Minute Crash Course

Short on time? Here's the fastest path to understanding EECS 127.

### Core Concepts (Read These First)

1. **Linear Algebra** [`01-linear-algebra/notes/README.md`]
   - 10 min read
   - Key takeaway: SVD is your Swiss Army knife for ML

2. **Least Squares** [`02-least-squares/notes/README.md`]
   - 10 min read
   - Key takeaway: Ridge regression prevents overfitting

3. **Convex Optimization** [`04-convex-optimization/notes/README.md`]
   - 10 min read
   - Key takeaway: Convex = guaranteed global optimum

### Hands-On Examples (Run These)

```bash
# 1. SVD-based recommender system (5 min)
python 01-linear-algebra/examples/svd_recommender.py

# 2. Ridge regression with cross-validation (5 min)
python 02-least-squares/examples/ridge_regression.py

# 3. Portfolio optimization (5 min)
python 04-convex-optimization/examples/portfolio_optimization.py
```

## Learning Paths

### Path 1: ML Engineer (Focus on Applications)

**Goal:** Understand the math behind scikit-learn and PyTorch

**Timeline:** 2-3 weeks

1. Week 1: Linear algebra + Least squares
   - Read notes
   - Implement PCA from scratch
   - Implement ridge regression from scratch

2. Week 2: Regularization + Algorithms
   - Understand LASSO vs Ridge
   - Implement gradient descent variants
   - Compare with PyTorch optimizers

3. Week 3: Applications
   - SVM dual formulation
   - Hyperparameter tuning as optimization
   - Neural network training dynamics

**Success metric:** Can explain why Adam works better than SGD

### Path 2: Software Engineer (Focus on Problem Solving)

**Goal:** Use optimization to solve real systems problems

**Timeline:** 2-3 weeks

1. Week 1: Linear programming
   - Formulate constraint problems
   - Use CVXPY to solve resource allocation
   - Apply to real work problem

2. Week 2: Convex optimization
   - Learn problem formulations (LP, QP, SOCP)
   - Understand when problems are tractable
   - Build optimization-based API

3. Week 3: Algorithms
   - Implement solvers
   - Understand complexity
   - Profile and optimize

**Success metric:** Can model a work problem as convex optimization

### Path 3: Full Course (Comprehensive Understanding)

**Goal:** Master the entire course

**Timeline:** 6-8 weeks

Follow the modules in order (01 → 08), spending 1 week per major topic.

For each module:
- [ ] Read notes (1-2 hours)
- [ ] Run examples (30 min)
- [ ] Complete exercises (2-3 hours)
- [ ] Implement one concept from scratch (3-4 hours)
- [ ] Apply to a real problem from your work (2-3 hours)

**Success metric:** Can teach each concept to a colleague

## Module Overview

### 01 - Linear Algebra
**Why:** Foundation for everything in ML
**Key concepts:** SVD, PCA, eigenvalues
**Time:** 1 week
**Prerequisites:** Basic matrix multiplication

### 02 - Least Squares
**Why:** Simplest ML model, understand regression
**Key concepts:** Normal equations, Ridge regression
**Time:** 3-4 days
**Prerequisites:** Module 01

### 03 - Linear Programming
**Why:** Resource allocation, constraint solving
**Key concepts:** Simplex, duality
**Time:** 3-4 days
**Prerequisites:** Basic linear algebra

### 04 - Convex Optimization
**Why:** Understand when problems are "easy"
**Key concepts:** Convexity, LP/QP/SOCP/SDP
**Time:** 1 week
**Prerequisites:** Modules 01-03

### 05 - Duality & KKT
**Why:** Theory behind SVMs, constraint handling
**Key concepts:** Lagrange multipliers, KKT conditions
**Time:** 4-5 days
**Prerequisites:** Module 04

### 06 - Regularization
**Why:** Prevent overfitting in ML models
**Key concepts:** L1/L2 regularization, LASSO
**Time:** 3-4 days
**Prerequisites:** Module 02

### 07 - Algorithms
**Why:** How optimizers actually work
**Key concepts:** Gradient descent, Newton's method
**Time:** 1 week
**Prerequisites:** Modules 01-04

### 08 - Applications
**Why:** Tie everything together
**Key concepts:** Real-world case studies
**Time:** 1 week
**Prerequisites:** All previous modules

## Study Tips

### Active Learning
- **Don't just read** - implement every concept
- **Break things** - modify examples to see what fails
- **Visualize** - plot loss landscapes, gradients, convergence

### Debugging Practice
Common issues you'll encounter:
- Dimension mismatch → Check shapes first
- Singular matrix → Add regularization
- Slow convergence → Check learning rate, feature scaling
- Poor test performance → Regularize more

### Connect to Real Work
For each concept, ask:
1. What problem at work could this solve?
2. What tool/library uses this under the hood?
3. How would I debug this in production?

### Resources by Learning Style

**Visual learners:**
- 3Blue1Brown YouTube series
- Distill.pub interactive articles
- Jupyter notebooks in this repo

**Math-focused:**
- Boyd & Vandenberghe textbook
- Course notes (linked in each module)
- Proof exercises

**Code-focused:**
- Implement from scratch (no libraries)
- Read source code (NumPy, scikit-learn)
- Build real projects

## Progress Tracking

Create a checklist for yourself:

```markdown
# My EECS 127 Journey

## Week 1
- [x] Read linear algebra notes
- [x] Run SVD recommender example
- [ ] Implement PCA from scratch
- [ ] Apply to image compression

## Week 2
- [ ] ...
```

## Getting Help

**Stuck on math?**
- Check the references in each module
- Ask on Math StackExchange with the [optimization] tag
- Watch 3Blue1Brown for intuition

**Stuck on code?**
- Read docstrings in examples
- Print intermediate values
- Visualize what's happening

**Stuck on concepts?**
- Explain it to a rubber duck
- Draw diagrams
- Find analogies to things you know

## Success Metrics

You'll know you've mastered EECS 127 when you can:

1. ✅ Look at an ML paper and recognize the optimization problem
2. ✅ Debug training issues by analyzing the loss landscape
3. ✅ Choose the right optimizer for a new problem
4. ✅ Explain why neural networks work despite being non-convex
5. ✅ Formulate a work problem as convex optimization
6. ✅ Implement an optimizer from scratch
7. ✅ Read scikit-learn source code and understand it

## Next Steps After This Course

**For ML:**
- CS 189: Machine Learning (uses EECS 127)
- Deep Learning (CS 285)
- Probabilistic ML

**For Optimization:**
- EECS 227C: Advanced Convex Optimization
- Numerical Optimization
- Optimal Control

**For Applications:**
- Reinforcement Learning
- Computer Vision
- Natural Language Processing

---

**Remember:** Understanding beats memorization. Take your time, implement things, break them, and rebuild. That's how you truly learn.

Good luck! 🚀
