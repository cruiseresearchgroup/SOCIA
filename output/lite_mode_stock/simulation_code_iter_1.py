import os
import sys
import numpy as np
import pandas as pd
import matplotlib
from typing import List, Optional, Dict, Any

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_data_dir() -> str:
    """
    Retrieve the data directory using environment variables.
    Returns:
        str: The full path to the data directory.
    Raises:
        EnvironmentError: If PROJECT_ROOT or DATA_PATH is not set.
        OSError: If directory cannot be created.
    """
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    if not PROJECT_ROOT or not DATA_PATH:
        raise EnvironmentError(
            "Both PROJECT_ROOT and DATA_PATH environment variables must be set."
        )
    data_dir = os.path.join(PROJECT_ROOT, DATA_PATH)
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create data directory '{data_dir}': {e}")
    return data_dir

DATA_DIR = get_data_dir()
result_path = os.path.join(DATA_DIR, "results.csv")
picture_path = os.path.join(DATA_DIR, "figure.png")

class Investor:
    """
    Investor agent in the stock market simulation.

    Attributes:
        strategy (str): Trading strategy ('fundamentalist', 'trend', or 'noise').
        cash (float): Cash holdings.
        shares (float): Number of shares held.
    """
    def __init__(self, strategy: str, init_cash: float = 10000.0, init_shares: float = 100.0) -> None:
        """
        Args:
            strategy (str): Trading strategy.
            init_cash (float): Initial cash holdings.
            init_shares (float): Initial number of shares.
        """
        self.strategy = strategy
        self.cash = float(init_cash)
        self.shares = float(init_shares)

    def decide_order(
        self, price: float, fundamental_value: float, price_history: List[float]
    ) -> float:
        """
        Decide on the order (buy/sell amount) for this time step.

        Args:
            price (float): Current stock price.
            fundamental_value (float): Estimated fundamental value.
            price_history (List[float]): List of previous prices.

        Returns:
            float: The number of shares to buy (>0) or sell (<0).
        """
        order = 0.0
        if self.strategy == "fundamentalist":
            diff = fundamental_value - price
            order = 0.01 * diff * np.random.normal(1, 0.05)
        elif self.strategy == "trend":
            if len(price_history) >= 2:
                trend = price_history[-1] - price_history[-2]
            else:
                trend = 0.0
            order = 0.05 * trend * np.random.normal(1, 0.05)
        elif self.strategy == "noise":
            order = np.random.normal(0, 5)
        max_buy = self.cash / price if price > 0 else 0.0
        max_sell = self.shares
        if order > 0:
            order = min(order, max_buy)
        else:
            order = max(order, -max_sell)
        return float(order)

    def execute_order(self, order: float, price: float) -> None:
        """
        Execute the buy/sell order, updating holdings.

        Args:
            order (float): Shares to buy (>0) or sell (<0).
            price (float): Current stock price.
        """
        cost = order * price
        self.cash -= cost
        self.shares += order

class StockMarketSimulation:
    """
    Main class for simulating the stock market with heterogeneous investors.

    Attributes:
        n_investors (int): Number of investors.
        n_steps (int): Number of time steps.
        price (float): Current stock price.
        prices (List[float]): History of stock prices.
        fundamental_value (float): Current fundamental value.
        fundamental_values (List[float]): History of fundamental values.
        volatility_history (List[float]): Rolling volatility (std of returns, np.nan if not enough data).
        investors (List[Investor]): List of Investor objects.
        time (int): Current time step.
        strategies (List[str]): Strategy assignment for each investor.
    """
    def __init__(
        self,
        n_investors: int = 1000,
        n_steps: int = 100,
        init_price: float = 100.0,
        init_fundamental: float = 100.0,
        mu: float = 0.000,
        sigma_fundamental: float = 0.2,
    ) -> None:
        """
        Args:
            n_investors (int): Number of investors.
            n_steps (int): Number of time steps.
            init_price (float): Initial stock price.
            init_fundamental (float): Initial fundamental value.
            mu (float): Drift of fundamental value.
            sigma_fundamental (float): Volatility of fundamental value.
        """
        self.n_investors = n_investors
        self.n_steps = n_steps
        self.init_price = init_price
        self.price = init_price
        self.prices: List[float] = [init_price]
        self.init_fundamental = init_fundamental
        self.fundamental_value = init_fundamental
        self.mu = mu
        self.sigma_fundamental = sigma_fundamental
        self.fundamental_values: List[float] = [init_fundamental]
        self.volatility_history: List[float] = []
        self.strategies: List[str] = self.assign_strategies()
        self.investors: List[Investor] = self.init_investors()
        self.time = 0
        self._last_orders: Optional[List[float]] = None  # For order consistency

    def assign_strategies(self) -> List[str]:
        """
        Randomly assign strategies to investors.

        Returns:
            List[str]: A list of strategy strings for each investor.
        """
        n_fundamentalist = self.n_investors // 3
        n_trend = self.n_investors // 3
        n_noise = self.n_investors - n_fundamentalist - n_trend
        strategies = (
            ["fundamentalist"] * n_fundamentalist +
            ["trend"] * n_trend +
            ["noise"] * n_noise
        )
        np.random.shuffle(strategies)
        return strategies

    def init_investors(self) -> List[Investor]:
        """
        Initialize all investor agents.

        Returns:
            List[Investor]: List of Investor objects.
        """
        return [Investor(strategy=strat) for strat in self.strategies]

    def update_fundamental_value(self) -> None:
        """
        Update the fundamental value as a geometric Brownian motion.
        """
        dt = 1
        shock = np.random.normal(0, self.sigma_fundamental * np.sqrt(dt))
        self.fundamental_value *= np.exp(self.mu * dt + shock)
        self.fundamental_values.append(self.fundamental_value)

    def aggregate_demand(self) -> float:
        """
        Aggregate demand from all investors and store orders for consistency.

        Returns:
            float: Total net demand (buy orders - sell orders).
        """
        price_history = self.prices
        total_order = 0.0
        orders: List[float] = []
        for investor in self.investors:
            order = investor.decide_order(self.price, self.fundamental_value, price_history)
            orders.append(order)
            total_order += order
        self._last_orders = orders  # Store for use in execute_trades
        return total_order

    def clear_market(self, net_demand: float) -> float:
        """
        Update the price based on excess demand using a log-linear price impact function.

        Args:
            net_demand (float): Aggregate net demand.

        Returns:
            float: New price.
        """
        market_depth = 0.01
        noise = np.random.normal(0, 0.2)
        log_price = np.log(self.price) + market_depth * (net_demand / self.n_investors) + noise / 100
        new_price = max(0.01, np.exp(log_price))
        return new_price

    def execute_trades(self, price: float) -> None:
        """
        Let each investor execute their order at the given price.

        Args:
            price (float): The transaction price.
        """
        if self._last_orders is None:
            self._last_orders = [
                investor.decide_order(price, self.fundamental_value, self.prices)
                for investor in self.investors
            ]
        for investor, order in zip(self.investors, self._last_orders):
            investor.execute_order(order, price)
        self._last_orders = None  # Clear for next step

    def record_volatility(self, window: int = 10) -> None:
        """
        Compute and record rolling volatility (sample std of returns over window).

        Args:
            window (int): Rolling window size.
        """
        t = len(self.prices) - 1
        if t >= window:
            returns = np.diff(np.log(self.prices[t - window + 1: t + 1]))
            vol = float(np.std(returns, ddof=1))
        else:
            vol = np.nan
        self.volatility_history.append(vol)

    def save_results(self, result_path: str) -> None:
        """
        Save simulation results to CSV.

        Args:
            result_path (str): Path to save results.
        """
        try:
            n_rows = len(self.prices)
            result_df = pd.DataFrame({
                "time": np.arange(n_rows),
                "price": self.prices,
                "fundamental_value": self.fundamental_values[:n_rows],
                "volatility": self.volatility_history + [np.nan] * (n_rows - len(self.volatility_history))
            })
            result_df.to_csv(result_path, index=False)
        except Exception as e:
            print(f"Error saving results to '{result_path}': {e}", file=sys.stderr)

    def plot_results(self, picture_path: str) -> None:
        """
        Plot price and volatility, saving the figure.

        Args:
            picture_path (str): Path to save plot.
        """
        try:
            n_rows = len(self.prices)
            times = np.arange(n_rows)
            # Ensure volatility matches prices in length
            vol_hist = self.volatility_history + [np.nan] * (n_rows - len(self.volatility_history))
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(times, self.prices, label="Price", color='blue')
            ax1.plot(times, self.fundamental_values[:n_rows], label="Fundamental Value", color='green', linestyle='--')
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Price", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2 = ax1.twinx()
            ax2.plot(times, vol_hist, label="Volatility", color='red', linestyle=':')
            ax2.set_ylabel("Volatility", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")
            plt.title("Stock Market Simulation: Price, Fundamental Value & Volatility")
            plt.tight_layout()
            plt.savefig(picture_path)
            plt.close()
        except Exception as e:
            print(f"Error saving plot to '{picture_path}': {e}", file=sys.stderr)

    def run(self) -> None:
        """
        Run the full simulation loop and store results.
        """
        self.record_volatility(window=10)
        for t in range(self.n_steps):
            self.time = t
            self.update_fundamental_value()
            net_demand = self.aggregate_demand()
            new_price = self.clear_market(net_demand)
            self.price = new_price
            self.prices.append(new_price)
            self.record_volatility(window=10)
            self.execute_trades(new_price)

def main() -> None:
    """
    Main runner for the simulation: initializes, runs, saves, and plots.
    """
    try:
        sim = StockMarketSimulation(n_investors=1000, n_steps=100)
        sim.run()
        sim.save_results(result_path)
        sim.plot_results(picture_path)
    except (EnvironmentError, OSError) as e:
        print(f"Error initializing simulation: {e}", file=sys.stderr)

main()