# Convex Optimization: The Sweet Spot

## Why This Matters for ML/SWE

**ML Perspective:**
- SVMs, logistic regression, LASSO - all convex!
- Convex = guaranteed global optimum (no local minima)
- Most of classical ML is convex optimization
- Even in deep learning, we study convex relaxations

**SWE Perspective:**
- Resource allocation (AWS/cloud optimization)
- Network flow and routing
- Portfolio optimization (finance)
- Circuit design and control systems

## The Big Idea: Convexity = Tractability

**Non-convex problems:** Multiple local minima, NP-hard, no guarantees
**Convex problems:** Single global minimum, polynomial-time solvable, guaranteed convergence

This is why we care so much about convexity!

## What Makes a Problem Convex?

A convex optimization problem has the form:

```
minimize    f(x)
subject to  g_i(x) ≤ 0    (inequality constraints)
            h_j(x) = 0    (equality constraints)
```

Where:
- `f(x)` is **convex** (objective function)
- `g_i(x)` are **convex** (inequality constraints)
- `h_j(x)` are **affine** (linear equality constraints)

**Key insight:** If all these conditions hold, any local minimum is a global minimum!

## Visualizing Convexity

### Convex Set

A set C is convex if for any two points x, y in C, the line segment between them is also in C.

```
Convex:     Not convex:
  ████          ██  ██
 ██████        ██    ██
██████████     ████████
```

**Examples:**
- ✅ Hyperplanes, half-spaces
- ✅ Balls, ellipsoids
- ✅ Polyhedra
- ❌ Union of disjoint sets
- ❌ Non-convex polygons

### Convex Function

A function f is convex if its epigraph (area above the curve) is a convex set.

**Intuition:** A bowl shape. Any two points, the line connecting them is above the curve.

**Mathematical test:**
```
f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)  for all θ ∈ [0,1]
```

Translation: "The function value at the midpoint is less than the midpoint of function values."

**Examples:**
- ✅ x² (parabola)
- ✅ e^x (exponential)
- ✅ -log(x) (negative log)
- ✅ ||x|| (norms)
- ✅ max(x₁, x₂, ..., xₙ)
- ❌ sin(x), cos(x)
- ❌ x³

## Common Convex Optimization Problems

### 1. Linear Programming (LP)

**Form:**
```
minimize    c^T x
subject to  Ax ≤ b
            x ≥ 0
```

**Everything is linear!** The objective and all constraints.

**Applications:**
- Resource allocation
- Diet problem (minimize cost, meet nutrition requirements)
- Network flow
- Production planning

**Example: Cloud cost optimization**
```python
import cvxpy as cp

# Variables: number of each instance type
x = cp.Variable(3, nonneg=True)  # [small, medium, large]

# Objective: minimize cost
cost = cp.array([10, 30, 80])  # $/hour
objective = cp.Minimize(cost @ x)

# Constraints: meet demand
cpu_per_instance = [1, 4, 16]  # cores
memory_per_instance = [2, 8, 32]  # GB

constraints = [
    cpu_per_instance @ x >= 100,  # Need 100 CPU cores
    memory_per_instance @ x >= 200,  # Need 200 GB RAM
]

problem = cp.Problem(objective, constraints)
problem.solve()

print(f"Optimal cost: ${problem.value:.2f}/hour")
print(f"Instances: {x.value}")
```

### 2. Quadratic Programming (QP)

**Form:**
```
minimize    (1/2) x^T Q x + c^T x
subject to  Ax ≤ b
```

Where Q is positive semidefinite (makes it convex).

**Applications:**
- Portfolio optimization
- Model predictive control
- Support Vector Machines (SVM)

**Example: Portfolio optimization**
```python
# Variables: allocation to each stock
weights = cp.Variable(n_stocks)

# Objective: minimize risk (variance)
risk = cp.quad_form(weights, covariance_matrix)

# Constraints
constraints = [
    cp.sum(weights) == 1,  # Fully invested
    weights >= 0,  # No short selling
    expected_returns @ weights >= target_return  # Meet return goal
]

problem = cp.Problem(cp.Minimize(risk), constraints)
problem.solve()
```

### 3. Second-Order Cone Programming (SOCP)

**Form:**
```
minimize    c^T x
subject to  ||A_i x + b_i|| ≤ c_i^T x + d_i
```

**Applications:**
- Robust optimization
- Antenna array design
- Filter design

**Why it matters:** More expressive than LP/QP, still polynomial-time solvable.

### 4. Semidefinite Programming (SDP)

**Form:**
```
minimize    c^T x
subject to  F(x) ⪰ 0  (positive semidefinite matrix)
```

**Applications:**
- Eigenvalue optimization
- Relaxations of NP-hard problems
- Quantum information theory

**Example: Max-cut approximation**
```python
# Approximate solution to NP-hard max-cut problem
X = cp.Variable((n, n), PSD=True)  # Positive semidefinite

objective = cp.Maximize(cp.sum(cp.multiply(W, (1 - X))))
constraints = [cp.diag(X) == 1]

problem = cp.Problem(objective, constraints)
```

## Checking Convexity

### Method 1: Second Derivative Test

For a twice-differentiable function f:
- f is convex if Hessian (∇²f) is positive semidefinite everywhere

**Example:**
```python
# Is f(x) = x^2 + 2xy + 3y^2 convex?

# Hessian:
H = [[2, 2],
     [2, 6]]

# Check eigenvalues
eigenvalues = [1.17, 6.83]  # Both positive → PSD → Convex!
```

### Method 2: Composition Rules

Build complex convex functions from simple ones:
- **Sum:** f + g is convex if f and g are convex
- **Scaling:** αf is convex if f is convex and α ≥ 0
- **Max:** max(f, g) is convex if f and g are convex
- **Composition:** f(g(x)) is convex if f is convex & non-decreasing, g is convex

**Example:**
```python
# Is f(x) = e^(x^2) convex?
# g(x) = x^2 is convex
# f(u) = e^u is convex and increasing
# → f(g(x)) is convex ✓
```

## Common Pitfalls

### ❌ Mistake 1: Assuming smooth = convex
```python
f(x) = sin(x)  # Smooth but NOT convex
```

### ❌ Mistake 2: Multiplying convex functions
```python
f(x) = x²  # Convex
g(x) = x²  # Convex
f(x) * g(x) = x⁴  # Still convex, but generally:
# Product of convex ≠ convex!
```

### ❌ Mistake 3: Negating constraints
```python
# Wrong:
constraints = [-f(x) ≥ 0]  # -f is concave, not convex!

# Right:
constraints = [f(x) ≤ 0]  # f is convex
```

## CVXPY: Solving Convex Problems in Python

CVXPY uses **Disciplined Convex Programming (DCP)** to verify convexity.

**Basic template:**
```python
import cvxpy as cp

# 1. Define variables
x = cp.Variable(n)

# 2. Define objective
objective = cp.Minimize(expression)

# 3. Define constraints
constraints = [
    constraint1,
    constraint2,
]

# 4. Solve
problem = cp.Problem(objective, constraints)
problem.solve()

# 5. Extract solution
print(f"Optimal value: {problem.value}")
print(f"Optimal x: {x.value}")
```

**DCP Rules:**
- Minimize convex or maximize concave
- Constraints: convex ≤ concave
- Only use DCP-compliant operations

**Example: LASSO regression**
```python
# minimize ||Ax - b||₂² + λ||x||₁

x = cp.Variable(n)
objective = cp.Minimize(
    cp.sum_squares(A @ x - b) +  # Convex
    lambda_ * cp.norm(x, 1)  # Convex
)
problem = cp.Problem(objective)
problem.solve()
```

## Convex vs Non-Convex in Deep Learning

**Classical ML (Convex):**
- Linear regression
- Logistic regression
- SVMs
- LASSO/Ridge

→ Guaranteed global optimum!

**Deep Learning (Non-Convex):**
- Neural networks (non-linear activations)
- Multiple local minima
- No convergence guarantees

→ Why does it still work?
- Overparameterization (many good local minima)
- Implicit regularization
- Clever initialization (Xavier, He)
- Adaptive optimizers (Adam)

## Key Takeaways

1. **Convex problems are tractable** - polynomial-time solvable
2. **Convexity = no local minima** - any local min is global
3. **Check convexity:** Hessian test or composition rules
4. **Problem hierarchy:** LP ⊂ QP ⊂ SOCP ⊂ SDP
5. **Use CVXPY** for modeling - it verifies convexity
6. **Most classical ML is convex** (SVMs, logistic, LASSO)
7. **Deep learning is NOT convex** (but works anyway)

## Real-World Checklist

Before using convex optimization:

1. ✅ Can you formulate as minimize f(x) subject to constraints?
2. ✅ Is f(x) convex? (Check second derivative or composition)
3. ✅ Are inequality constraints g(x) ≤ 0 convex?
4. ✅ Are equality constraints h(x) = 0 affine?
5. ✅ Is the feasible set non-empty?

If yes to all → Use convex solver (CVXPY, Mosek, Gurobi)
If no → Consider relaxation or heuristics

## Next Steps

- **Practice**: Solve `examples/portfolio_optimization.py`
- **Experiment**: Try `notebooks/cvxpy_tutorial.ipynb`
- **Challenge**: Formulate a real problem from your work as convex optimization

## Resources

- [Boyd & Vandenberghe: Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) (THE textbook)
- [CVXPY Documentation](https://www.cvxpy.org/)
- [Stanford EE364a Lectures](https://www.youtube.com/playlist?list=PL3940DD956CDF0622)
