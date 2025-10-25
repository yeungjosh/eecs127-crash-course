"""
Portfolio Optimization using Convex Optimization
-------------------------------------------------
Classic Markowitz mean-variance portfolio optimization.

Problem: Allocate money across stocks to minimize risk while meeting return target.

This is a Quadratic Programming (QP) problem:
    minimize    (1/2) w^T Σ w         (portfolio variance)
    subject to  μ^T w ≥ r_target      (expected return)
                1^T w = 1             (fully invested)
                w ≥ 0                 (no short selling)
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, List, Optional


class PortfolioOptimizer:
    """
    Modern Portfolio Theory (Markowitz) optimizer.

    Finds the optimal allocation that minimizes risk for a target return.
    """

    def __init__(
        self,
        returns: np.ndarray,
        stock_names: Optional[List[str]] = None
    ):
        """
        Args:
            returns: (n_periods, n_stocks) historical returns matrix
            stock_names: List of stock names (optional)
        """
        self.returns = returns
        self.n_stocks = returns.shape[1]
        self.stock_names = stock_names or [f"Stock_{i}" for i in range(self.n_stocks)]

        # Estimate parameters from historical data
        self.mean_returns = returns.mean(axis=0)
        self.cov_matrix = np.cov(returns.T)

    def optimize(
        self,
        target_return: float,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Find optimal portfolio weights.

        Args:
            target_return: Minimum expected return (e.g., 0.10 = 10% annual)
            allow_short: Allow negative weights (short selling)

        Returns:
            weights: Optimal allocation (sums to 1)
            risk: Portfolio standard deviation (volatility)
        """
        # Define optimization variable
        w = cp.Variable(self.n_stocks)

        # Objective: minimize variance (risk)
        # variance = w^T Σ w
        portfolio_variance = cp.quad_form(w, self.cov_matrix)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            self.mean_returns @ w >= target_return,  # Meet return target
        ]

        if not allow_short:
            constraints.append(w >= 0)  # No short selling

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != 'optimal':
            raise ValueError(f"Optimization failed: {problem.status}")

        weights = w.value
        risk = np.sqrt(problem.value)  # std dev

        return weights, risk

    def efficient_frontier(
        self,
        n_points: int = 50,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the efficient frontier (risk-return tradeoff curve).

        Args:
            n_points: Number of points on the frontier
            allow_short: Allow short selling

        Returns:
            risks: Array of portfolio risks
            returns: Array of portfolio returns
        """
        # Compute range of feasible returns
        if allow_short:
            # With short selling, can achieve any return
            min_return = self.mean_returns.min()
            max_return = self.mean_returns.max()
        else:
            # Without short selling, limited by long-only constraint
            # Min: least risky portfolio (minimum variance)
            w_min_var, _ = self.optimize_min_variance(allow_short=False)
            min_return = self.mean_returns @ w_min_var

            # Max: put everything in highest return stock
            max_return = self.mean_returns.max()

        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)

        risks = []
        actual_returns = []

        for target in target_returns:
            try:
                weights, risk = self.optimize(target, allow_short)
                risks.append(risk)
                actual_returns.append(self.mean_returns @ weights)
            except ValueError:
                # Infeasible target return
                continue

        return np.array(risks), np.array(actual_returns)

    def optimize_min_variance(
        self,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Find minimum variance portfolio (least risky).

        Args:
            allow_short: Allow short selling

        Returns:
            weights: Optimal allocation
            risk: Portfolio volatility
        """
        w = cp.Variable(self.n_stocks)
        objective = cp.Minimize(cp.quad_form(w, self.cov_matrix))

        constraints = [cp.sum(w) == 1]
        if not allow_short:
            constraints.append(w >= 0)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value, np.sqrt(problem.value)

    def optimize_max_sharpe(
        self,
        risk_free_rate: float = 0.02,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Find maximum Sharpe ratio portfolio.

        Sharpe ratio = (return - risk_free_rate) / risk

        Args:
            risk_free_rate: Risk-free rate (e.g., 0.02 = 2%)
            allow_short: Allow short selling

        Returns:
            weights: Optimal allocation
            sharpe: Sharpe ratio
        """
        # Trick: Sharpe maximization is a QP problem
        # See: https://web.stanford.edu/~boyd/papers/pdf/max_sharpe.pdf

        w = cp.Variable(self.n_stocks)
        excess_returns = self.mean_returns - risk_free_rate

        # This formulation avoids non-convex division
        objective = cp.Maximize(excess_returns @ w)
        constraints = [
            cp.quad_form(w, self.cov_matrix) <= 1,
            cp.sum(w) == 1,
        ]

        if not allow_short:
            constraints.append(w >= 0)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Normalize weights
        weights = w.value / np.sum(w.value)

        # Compute Sharpe
        portfolio_return = self.mean_returns @ weights
        portfolio_risk = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_risk

        return weights, sharpe


def generate_sample_returns(
    n_stocks: int = 5,
    n_periods: int = 252,  # One year of daily returns
    seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """Generate synthetic stock returns."""
    np.random.seed(seed)

    stock_names = ["TECH", "FINANCE", "ENERGY", "HEALTHCARE", "CONSUMER"][:n_stocks]

    # Generate correlated returns
    # Mean returns: 5% to 15% annually
    annual_means = np.random.uniform(0.05, 0.15, n_stocks)
    daily_means = annual_means / 252

    # Volatilities: 15% to 40% annually
    annual_vols = np.random.uniform(0.15, 0.40, n_stocks)
    daily_vols = annual_vols / np.sqrt(252)

    # Generate correlation matrix
    L = np.random.randn(n_stocks, n_stocks)
    corr = L @ L.T
    corr = corr / np.sqrt(np.diag(corr)[:, None] @ np.diag(corr)[None, :])

    # Convert to covariance
    D = np.diag(daily_vols)
    cov = D @ corr @ D

    # Generate returns
    returns = np.random.multivariate_normal(daily_means, cov, n_periods)

    return returns, stock_names


def main():
    """Demo: Portfolio optimization."""

    print("=" * 70)
    print("Portfolio Optimization with Convex Programming")
    print("=" * 70)

    # Generate sample data
    returns, stock_names = generate_sample_returns(n_stocks=5)
    print(f"\nSimulated {returns.shape[0]} days of returns for {len(stock_names)} stocks")

    # Create optimizer
    optimizer = PortfolioOptimizer(returns, stock_names)

    # Show statistics
    print("\n" + "=" * 70)
    print("Stock Statistics (Annualized)")
    print("=" * 70)

    for i, name in enumerate(stock_names):
        annual_return = optimizer.mean_returns[i] * 252
        annual_vol = np.sqrt(optimizer.cov_matrix[i, i] * 252)
        print(f"{name:12} Return: {annual_return:6.2%}  Risk: {annual_vol:6.2%}")

    # Optimize for different target returns
    print("\n" + "=" * 70)
    print("Optimal Portfolios for Different Return Targets")
    print("=" * 70)

    target_returns = [0.08, 0.10, 0.12]  # 8%, 10%, 12% annual

    for target in target_returns:
        print(f"\nTarget return: {target:.1%} annually")

        weights, risk = optimizer.optimize(target / 252 * 1, allow_short=False)  # Daily
        annual_risk = risk * np.sqrt(252)

        print(f"Portfolio risk: {annual_risk:.2%} (std dev)")
        print("Allocation:")
        for name, weight in zip(stock_names, weights):
            if weight > 0.01:  # Show only significant allocations
                print(f"  {name:12} {weight:6.2%}")

    # Find special portfolios
    print("\n" + "=" * 70)
    print("Special Portfolios")
    print("=" * 70)

    # Minimum variance portfolio
    weights_min_var, risk_min_var = optimizer.optimize_min_variance(allow_short=False)
    return_min_var = optimizer.mean_returns @ weights_min_var
    print("\n1. Minimum Variance Portfolio (Least Risky):")
    print(f"   Return: {return_min_var * 252:.2%}  Risk: {risk_min_var * np.sqrt(252):.2%}")
    print("   Allocation:", {name: f"{w:.1%}" for name, w in zip(stock_names, weights_min_var) if w > 0.01})

    # Maximum Sharpe portfolio
    weights_sharpe, sharpe = optimizer.optimize_max_sharpe(
        risk_free_rate=0.02 / 252,  # 2% annually
        allow_short=False
    )
    return_sharpe = optimizer.mean_returns @ weights_sharpe
    risk_sharpe = np.sqrt(weights_sharpe @ optimizer.cov_matrix @ weights_sharpe)
    print("\n2. Maximum Sharpe Ratio Portfolio (Best Risk-Adjusted):")
    print(f"   Sharpe: {sharpe:.2f}")
    print(f"   Return: {return_sharpe * 252:.2%}  Risk: {risk_sharpe * np.sqrt(252):.2%}")
    print("   Allocation:", {name: f"{w:.1%}" for name, w in zip(stock_names, weights_sharpe) if w > 0.01})

    # Compare with naive strategies
    print("\n" + "=" * 70)
    print("Comparison with Naive Strategies")
    print("=" * 70)

    # Equal weights
    equal_weights = np.ones(len(stock_names)) / len(stock_names)
    equal_return = optimizer.mean_returns @ equal_weights
    equal_risk = np.sqrt(equal_weights @ optimizer.cov_matrix @ equal_weights)

    print("\n3. Equal Weight (1/N) Portfolio:")
    print(f"   Return: {equal_return * 252:.2%}  Risk: {equal_risk * np.sqrt(252):.2%}")

    # Best single stock
    best_stock_idx = np.argmax(optimizer.mean_returns)
    best_stock_return = optimizer.mean_returns[best_stock_idx]
    best_stock_risk = np.sqrt(optimizer.cov_matrix[best_stock_idx, best_stock_idx])

    print(f"\n4. Best Single Stock ({stock_names[best_stock_idx]}):")
    print(f"   Return: {best_stock_return * 252:.2%}  Risk: {best_stock_risk * np.sqrt(252):.2%}")

    # Key insights
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("""
1. Diversification reduces risk!
   - Min variance portfolio has lower risk than any single stock
   - Risk reduction comes from negative/low correlations

2. Risk-return tradeoff is fundamental
   - Higher returns require accepting higher risk
   - Efficient frontier shows best possible tradeoff

3. Sharpe ratio finds the "sweet spot"
   - Maximizes return per unit of risk
   - Better metric than raw return

4. Equal weighting is often suboptimal
   - Ignores correlations and individual risks
   - But surprisingly robust in practice (estimation error!)

5. Convex optimization guarantees the global optimum
   - No local minima to worry about
   - Scales to hundreds of stocks
    """)


if __name__ == "__main__":
    main()
