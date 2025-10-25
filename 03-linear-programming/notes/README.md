# Linear Programming: Solving Constraint Problems

> **Status:** Coming soon! Check back for detailed notes.

## Preview

Linear Programming (LP) solves problems of the form:

```
minimize    c^T x
subject to  Ax ≤ b
            x ≥ 0
```

**Real-world applications:**
- Resource allocation (minimize cloud costs)
- Supply chain optimization
- Network flow problems
- Diet problem (minimize cost, meet nutrition)

**Key concepts to be covered:**
- Simplex algorithm
- Duality in LP
- Sensitivity analysis
- Interior point methods

## Quick Example

```python
import cvxpy as cp

# Minimize cost of cloud instances while meeting CPU/RAM requirements
x = cp.Variable(3, nonneg=True)  # [small, medium, large]

cost = cp.array([10, 30, 80])
objective = cp.Minimize(cost @ x)

constraints = [
    cp.array([1, 4, 16]) @ x >= 100,  # CPU requirement
    cp.array([2, 8, 32]) @ x >= 200,  # RAM requirement
]

cp.Problem(objective, constraints).solve()
```

## Resources
- [Introduction to Linear Optimization - Bertsimas & Tsitsiklis](https://www.amazon.com/Introduction-Linear-Optimization-Scientific-Computation/dp/1886529191)
- [CVXPY LP Examples](https://www.cvxpy.org/examples/basic/linear_program.html)
