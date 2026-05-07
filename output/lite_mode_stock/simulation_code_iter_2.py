import os
import sys
import numpy as np
import pandas as pd
import matplotlib
from typing import List, Optional, Dict, Any

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_data_paths() -> Dict[str, str]:
    """
    Determine and validate the data directory and output file paths based on environment variables.

    Returns:
        Dict[str, str]: Dictionary containing 'DATA_DIR', 'RESULT_PATH', and 'PICTURE_PATH'.

    Raises:
        EnvironmentError: If required environment variables are missing or invalid.
        OSError: If the data directory cannot be created.
    """
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    if not PROJECT_ROOT or not DATA_PATH:
        raise EnvironmentError("Both environment variables 'PROJECT_ROOT' and 'DATA_PATH' must be set and non-empty.")
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
    if not DATA_DIR or not isinstance(DATA_DIR, str) or DATA_DIR.strip() == "":
        raise EnvironmentError(f"Invalid data directory path: {DATA_DIR!r}. Check PROJECT_ROOT and DATA_PATH.")
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create data directory '{DATA_DIR}': {e}")
    result_path = os.path.join(DATA_DIR, "results.csv")
    picture_path = os.path.join(DATA_DIR, "figure.png")
    return {"DATA_DIR": DATA_DIR, "RESULT_PATH": result_path, "PICTURE_PATH": picture_path}

class Investor:
    """
    Investor agent in the stock market simulation.

    Attributes:
        strategy (str): Trading strategy ('fundamentalist', 'trend-follower', or 'noise').
        cash (float): Cash holdings.
        shares (float): Number of shares held.
        strategy_id (int): Index of the investor in the initialization list.
        trade_history (List[float]): History of trades (positive for buy, negative for sell).
    """
    def __init__(self, strategy: str, strategy_id: int, init_cash: float = 10000.0, init_shares: float = 100.0) -> None:
        """
        Initialize an Investor.

        Args:
            strategy (str): Trading strategy. Must be one of 'fundamentalist', 'trend-follower', or 'noise'.
            strategy_id (int): Investor's index in the initialization list.
            init_cash (float): Initial cash holdings.
            init_shares (float): Initial number of shares.
        """
        self.strategy = strategy
        self.strategy_id = strategy_id
        self.cash = float(init_cash)
        self.shares = float(init_shares)
        self.trade_history: List[float] = [0.0]  # Initialize with 0.0 for the initial state

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
        if price is None or price <= 0:
            # Prevent division by zero or nonsensical prices
            return 0.0

        order = 0.0
        # Fundamentalists: Buy when price < fundamental, sell when price > fundamental
        if self.strategy == "fundamentalist":
            diff = (fundamental_value - price) / price
            aggressiveness = 200  # scaling factor for order size
            noise = np.random.normal(1, 0.03)
            order = aggressiveness * diff * noise
        # Trend-followers: Buy if price is going up, sell if going down (momentum based)
        elif self.strategy == "trend-follower":
            window = min(5, len(price_history))
            if window >= 2:
                recent_trend = price_history[-1] - price_history[-window]
                order = 12 * np.sign(recent_trend) * np.abs(recent_trend / price) * np.random.normal(1, 0.08)
            else:
                order = 0.0
        # Noise traders: Random buy/sell, independent of price/fundamental
        elif self.strategy == "noise":
            order = np.random.normal(0, 25)
        # Constrain orders by portfolio
        max_buy = self.cash // price if price > 0 else 0.0
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
            order (float): Number of shares to buy (>0) or sell (<0).
            price (float): Current stock price.
        """
        cost = order * price
        self.cash -= cost
        self.shares += order
        self.trade_history.append(order)

class StockMarketSimulation:
    """
    Simulates a stock market with 1,000 investors of three strategies over 100 time steps,
    tracks stock price, volatility, and agent group behaviors.

    Attributes:
        n_investors (int): Number of investors.
        n_steps (int): Number of time steps.
        price (float): Current stock price.
        prices (List[float]): History of stock prices.
        fundamental_value (float): Current fundamental value.
        fundamental_values (List[float]): History of fundamental values.
        volatility_history (List[float]): Rolling volatility (std of returns).
        investors (List[Investor]): List of Investor objects.
        group_trades (Dict[str, List[float]]): Per-strategy mean trade history for impact visualization.
        group_holdings (Dict[str, List[float]]): Per-strategy mean share holdings.
        time (int): Current time step.
        strategies (List[str]): Strategy assignment for each investor.
    """
    n_investors: int
    n_steps: int
    init_price: float
    price: float
    prices: List[float]
    init_fundamental: float
    fundamental_value: float
    mu: float
    sigma_fundamental: float
    fundamental_values: List[float]
    volatility_history: List[float]
    strategies: List[str]
    investors: List[Investor]
    group_trades: Dict[str, List[float]]
    group_holdings: Dict[str, List[float]]
    time: int
    _last_orders: Optional[List[float]]

    def __init__(
        self,
        n_investors: int = 1000,
        n_steps: int = 100,
        init_price: float = 100.0,
        init_fundamental: float = 100.0,
        mu: float = 0.0,
        sigma_fundamental: float = 0.2,
    ) -> None:
        """
        Initialize the stock market simulation.

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
        self._last_orders: Optional[List[float]] = None
        self.group_trades: Dict[str, List[float]] = {"fundamentalist": [], "trend-follower": [], "noise": []}
        self.group_holdings: Dict[str, List[float]] = {"fundamentalist": [], "trend-follower": [], "noise": []}

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
            ["trend-follower"] * n_trend +
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
        investors = []
        for idx, strat in enumerate(self.strategies):
            investors.append(Investor(strategy=strat, strategy_id=idx))
        return investors

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
        self._last_orders = orders
        return total_order

    def clear_market(self, net_demand: float) -> float:
        """
        Update the price based on excess demand using a log-linear price impact function.

        Args:
            net_demand (float): Aggregate net demand.

        Returns:
            float: New price.
        """
        market_depth = 0.012  # Slightly more sensitive than before
        noise = np.random.normal(0, 0.18)
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
        orders = self._last_orders
        if not isinstance(orders, list) or len(orders) != len(self.investors):
            raise RuntimeError("Internal error: _last_orders is invalid.")
        for investor, order in zip(self.investors, orders):
            investor.execute_order(order, price)
        self._last_orders = None

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

    def record_group_behavior(self) -> None:
        """
        Calculate and record mean trade and holdings per strategy group for this time step.
        """
        group_trades = {"fundamentalist": [], "trend-follower": [], "noise": []}
        group_holdings = {"fundamentalist": [], "trend-follower": [], "noise": []}
        # This could be optimized for very large N, but for N=1000 is fine.
        for investor in self.investors:
            trade = investor.trade_history[-1] if investor.trade_history else 0.0
            group_trades[investor.strategy].append(trade)
            group_holdings[investor.strategy].append(investor.shares)
        for strat in self.group_trades.keys():
            self.group_trades[strat].append(np.mean(group_trades[strat]) if group_trades[strat] else 0.0)
            self.group_holdings[strat].append(np.mean(group_holdings[strat]) if group_holdings[strat] else 0.0)

    def _pad_list(self, lst: List[float], length: int) -> List[float]:
        """Pad a list with np.nan to reach the desired length."""
        pad_len = length - len(lst)
        if pad_len > 0:
            return lst + [np.nan] * pad_len
        else:
            return lst[:length]

    def save_results(self, result_path: str) -> None:
        """
        Save simulation results to CSV.

        Args:
            result_path (str): Path to save results.

        Raises:
            Exception: If writing to result_path fails.
        """
        n_rows = len(self.prices)
        # Precompute padded lists only once
        volatility = self._pad_list(self.volatility_history, n_rows)
        f_trades = self._pad_list(self.group_trades["fundamentalist"], n_rows)
        t_trades = self._pad_list(self.group_trades["trend-follower"], n_rows)
        n_trades = self._pad_list(self.group_trades["noise"], n_rows)
        f_holdings = self._pad_list(self.group_holdings["fundamentalist"], n_rows)
        t_holdings = self._pad_list(self.group_holdings["trend-follower"], n_rows)
        n_holdings = self._pad_list(self.group_holdings["noise"], n_rows)
        try:
            result_df = pd.DataFrame({
                "time": np.arange(n_rows),
                "price": self.prices,
                "fundamental_value": self.fundamental_values[:n_rows],
                "volatility": volatility,
                "fundamentalist_trade": f_trades,
                "trend_follower_trade": t_trades,
                "noise_trade": n_trades,
                "fundamentalist_holding": f_holdings,
                "trend_follower_holding": t_holdings,
                "noise_holding": n_holdings,
            })
            result_df.to_csv(result_path, index=False)
        except Exception as e:
            print(f"Error saving results to '{result_path}': {e}", file=sys.stderr)
            raise

    def plot_results(self, picture_path: str) -> None:
        """
        Plot price, volatility, and mean group trades, showing the impact of strategies.

        Args:
            picture_path (str): Path to save plot.

        Raises:
            Exception: If writing to picture_path fails.
        """
        n_rows = len(self.prices)
        times = np.arange(n_rows)
        # Precompute padded lists only once
        vol_hist = self._pad_list(self.volatility_history, n_rows)
        f_trades = self._pad_list(self.group_trades["fundamentalist"], n_rows)
        t_trades = self._pad_list(self.group_trades["trend-follower"], n_rows)
        n_trades = self._pad_list(self.group_trades["noise"], n_rows)
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

            # Price and fundamental value
            ax1.plot(times, self.prices, label="Price", color='blue')
            ax1.plot(times, self.fundamental_values[:n_rows], label="Fundamental Value", color='green', linestyle='--')
            ax1.set_ylabel("Price", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.legend(loc="upper left")
            ax1.set_title("Stock Price and Fundamental Value")

            # Volatility
            ax2.plot(times, vol_hist, label="Volatility", color='red', linestyle='-')
            ax2.set_ylabel("Volatility", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc="upper left")
            ax2.set_title("Market Volatility (Rolling Std of Log Returns)")

            # Mean group trades
            ax3.plot(times, f_trades, label="Fundamentalist (mean trade)", color='navy', linewidth=1.1)
            ax3.plot(times, t_trades, label="Trend-follower (mean trade)", color='orange', linewidth=1.1)
            ax3.plot(times, n_trades, label="Noise (mean trade)", color='purple', linewidth=1.1)
            ax3.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax3.set_ylabel("Mean Net Trade")
            ax3.set_xlabel("Time Step")
            ax3.legend(loc="upper left")
            ax3.set_title("Mean Net Trade Per Strategy (Impact of Strategies)")

            plt.tight_layout()
            plt.savefig(picture_path)
            plt.close()
        except Exception as e:
            print(f"Error saving plot to '{picture_path}': {e}", file=sys.stderr)
            raise

    def run(self) -> None:
        """
        Run the full simulation loop and store results.
        """
        self.record_volatility(window=10)
        self.record_group_behavior()
        for t in range(self.n_steps):
            self.time = t
            self.update_fundamental_value()
            net_demand = self.aggregate_demand()
            new_price = self.clear_market(net_demand)
            self.price = new_price
            self.prices.append(new_price)
            self.record_volatility(window=10)
            self.execute_trades(new_price)
            self.record_group_behavior()

def main() -> None:
    """
    Main runner for the simulation: initializes, runs, saves, and plots.
    """
    try:
        paths = get_data_paths()
        result_path = paths["RESULT_PATH"]
        picture_path = paths["PICTURE_PATH"]
        sim = StockMarketSimulation(n_investors=1000, n_steps=100)
        sim.run()
        try:
            sim.save_results(result_path)
        except Exception as e:
            print(f"Critical error: could not save simulation results: {e}", file=sys.stderr)
            sys.exit(2)
        try:
            sim.plot_results(picture_path)
        except Exception as e:
            print(f"Critical error: could not save simulation plot: {e}", file=sys.stderr)
            sys.exit(3)
    except (EnvironmentError, OSError) as e:
        print(f"Error initializing simulation or data paths: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(4)

main()