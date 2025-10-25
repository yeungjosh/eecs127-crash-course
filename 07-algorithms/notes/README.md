# Optimization Algorithms: How Optimizers Really Work

> **Status:** Coming soon! Check back for detailed notes.

## Preview

How do we actually solve optimization problems? Algorithms!

### Gradient Descent
```python
for iteration in range(max_iters):
    gradient = compute_gradient(x)
    x = x - learning_rate * gradient
```

**Pros:** Simple, works for any differentiable function
**Cons:** Can be slow, sensitive to learning rate

### Variants

**SGD (Stochastic):**
- Use mini-batches instead of full dataset
- Faster, noisier

**Momentum:**
- Add velocity term
- Smooths oscillations

**Adam:**
- Adaptive learning rates
- First and second moment estimates
- Most popular in deep learning

**Newton's Method:**
- Uses second derivatives (Hessian)
- Faster convergence
- More expensive per iteration

## Convergence Guarantees

| Algorithm | Problem Type | Convergence Rate |
|-----------|--------------|------------------|
| Gradient Descent | Convex, smooth | O(1/k) |
| GD + Strong convexity | Strongly convex | O(exp(-k)) |
| Newton | Strongly convex | O(exp(-2^k)) |
| SGD | Convex | O(1/√k) |
| Adam | Non-convex | No guarantees |

## Quick Comparison

```python
import torch.optim as optim

# SGD: Simple, robust
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD + Momentum: Faster convergence
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam: Adaptive, works well by default
optimizer = optim.Adam(model.parameters(), lr=0.001)

# LBFGS: Second-order, for smaller problems
optimizer = optim.LBFGS(model.parameters())
```

## Resources
- [Why Momentum Really Works (Distill)](https://distill.pub/2017/momentum/)
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
- [The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292)
