# Duality & KKT Conditions: The Theory Behind SVMs

> **Status:** Coming soon! Check back for detailed notes.

## Preview

**Duality** lets you solve hard problems by solving their "dual" problem instead.

**KKT conditions** are the optimality conditions for constrained optimization.

**Why this matters:**
- SVMs use the dual formulation (kernel trick only works in dual)
- Understanding constraints and their multipliers
- Sensitivity analysis (how much constraints matter)

## The KKT Conditions

For a constrained problem:
```
minimize    f(x)
subject to  g_i(x) ≤ 0
            h_j(x) = 0
```

A point x* is optimal if:
1. **Stationarity:** ∇f(x*) + Σ λ_i ∇g_i(x*) + Σ ν_j ∇h_j(x*) = 0
2. **Primal feasibility:** g_i(x*) ≤ 0, h_j(x*) = 0
3. **Dual feasibility:** λ_i ≥ 0
4. **Complementary slackness:** λ_i g_i(x*) = 0

## Quick Example: SVM Dual

The SVM primal:
```
minimize    (1/2)||w||² + C Σ ξ_i
subject to  y_i(w^T x_i + b) ≥ 1 - ξ_i
```

Becomes the dual:
```
maximize    Σ α_i - (1/2) Σ Σ α_i α_j y_i y_j (x_i^T x_j)
subject to  0 ≤ α_i ≤ C
            Σ α_i y_i = 0
```

This is where the kernel trick comes in: replace x_i^T x_j with K(x_i, x_j)!

## Resources
- [Boyd & Vandenberghe Chapter 5: Duality](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- [Understanding SVMs: A Complete Guide](https://arxiv.org/abs/1804.05512)
