"""
Simple SIR Epidemic Simulation

This module provides a command-line interface and programmatic API to simulate an epidemic using
the classic SIR (Susceptible-Infected-Recovered) model. Users can configure simulation parameters
via command-line arguments or environment variables. The simulation supports visualization and
saving results to CSV, with options to disable these features for headless/server environments.

Usage:
    python sir_simulation.py [options]

Options:
    --population INT           Total population size (default: 1000 or SIM_POPULATION)
    --beta FLOAT               Infection rate beta (default: 0.3 or SIM_BETA)
    --gamma FLOAT              Recovery rate gamma (default: 0.1 or SIM_GAMMA)
    --initial-infected INT     Initial number of infected individuals (default: 1 or SIM_INITIAL_INFECTED)
    --initial-recovered INT    Initial number of recovered individuals (default: 0 or SIM_INITIAL_RECOVERED)
    --n-steps INT              Number of time steps to simulate (default: 160 or SIM_N_STEPS)
    --output PATH              Path to save simulation results CSV (default: results.csv)
    --no-visualize             Disable visualization (useful in headless/server environments)
    --no-save                  Do not save results to file
"""

import os
import sys
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import traceback

def get_env_or_default(key: str, default: Optional[str] = None) -> str:
    """
    Get the value of an environment variable or return a default value.

    Args:
        key (str): Environment variable name.
        default (Optional[str]): Default value if variable is not set.

    Returns:
        str: The environment variable value or the default.

    Raises:
        EnvironmentError: If the variable is not set and no default is provided.
    """
    value = os.environ.get(key)
    if value is None:
        if default is not None:
            return default
        raise EnvironmentError(f"Environment variable '{key}' is not set.")
    return value

# --- Environment Setup ---
# Documentation for users:
# The simulation stores results in a directory constructed as PROJECT_ROOT/DATA_PATH.
# You can set the environment variables PROJECT_ROOT and DATA_PATH, or the defaults will be used.
# Defaults: PROJECT_ROOT=Current working directory, DATA_PATH='data'

PROJECT_ROOT = get_env_or_default("PROJECT_ROOT", default=os.getcwd())
DATA_PATH = get_env_or_default("DATA_PATH", default="data")

DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except PermissionError as e:
    print(f"Permission denied while creating data directory '{DATA_DIR}': {e}", file=sys.stderr)
    sys.exit(1)
except OSError as e:
    print(f"OS error while creating data directory '{DATA_DIR}': {e}", file=sys.stderr)
    sys.exit(1)
RESULT_PATH = os.path.join(DATA_DIR, "results.csv")

class SIRModel:
    """
    Implements the SIR (Susceptible-Infected-Recovered) model for epidemic simulation.

    Main methods:
        - run(n_steps): Simulate the SIR model for a number of time steps and return a DataFrame.
        - S, I, R properties: Access the latest simulation results as lists (empty if not run).
    
    This class does not support stepwise simulation (step/reset) for simplicity and efficiency.
    """

    def __init__(
        self,
        population: int,
        beta: float,
        gamma: float,
        initial_infected: int = 1,
        initial_recovered: int = 0
    ) -> None:
        """
        Initialize the SIR model.

        Args:
            population (int): Total population.
            beta (float): Infection rate.
            gamma (float): Recovery rate.
            initial_infected (int): Initial number of infected individuals.
            initial_recovered (int): Initial number of recovered individuals.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if population <= 0:
            raise ValueError('Population must be positive.')
        if beta < 0 or gamma < 0:
            raise ValueError('Beta and gamma must be non-negative.')
        if initial_infected < 0 or initial_recovered < 0:
            raise ValueError('Initial infected/recovered must be non-negative.')
        if initial_infected + initial_recovered > population:
            raise ValueError('Initial infected + recovered cannot exceed population.')

        self.population: int = population
        self.beta: float = beta
        self.gamma: float = gamma
        self.initial_infected: int = initial_infected
        self.initial_recovered: int = initial_recovered

        # Only store NumPy arrays for results after run()
        self._S: Optional[np.ndarray] = None
        self._I: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None

    def run(self, n_steps: int) -> pd.DataFrame:
        """
        Run the SIR simulation for a given number of steps.

        Args:
            n_steps (int): Number of simulation steps.

        Returns:
            pd.DataFrame: DataFrame containing time series of S, I, R.

        Raises:
            ValueError: If n_steps is not positive.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        susceptible: int = self.population - self.initial_infected - self.initial_recovered
        S = np.empty(n_steps + 1)
        I = np.empty(n_steps + 1)
        R = np.empty(n_steps + 1)
        S[0] = float(susceptible)
        I[0] = float(self.initial_infected)
        R[0] = float(self.initial_recovered)
        for t in range(n_steps):
            new_infections = self.beta * S[t] * I[t] / self.population
            new_recoveries = self.gamma * I[t]
            S[t + 1] = max(S[t] - new_infections, 0)
            I[t + 1] = max(I[t] + new_infections - new_recoveries, 0)
            R[t + 1] = max(R[t] + new_recoveries, 0)
        self._S = S
        self._I = I
        self._R = R
        df = pd.DataFrame({
            'Susceptible': S,
            'Infected': I,
            'Recovered': R
        })
        return df

    @property
    def S(self) -> List[float]:
        """
        Returns the time series of susceptible individuals after the simulation.

        Returns:
            List[float]: List of susceptible counts at each time step, or empty list if simulation has not run.
        """
        if self._S is not None:
            return self._S.tolist()
        else:
            return []

    @property
    def I(self) -> List[float]:
        """
        Returns the time series of infected individuals after the simulation.

        Returns:
            List[float]: List of infected counts at each time step, or empty list if simulation has not run.
        """
        if self._I is not None:
            return self._I.tolist()
        else:
            return []

    @property
    def R(self) -> List[float]:
        """
        Returns the time series of recovered individuals after the simulation.

        Returns:
            List[float]: List of recovered counts at each time step, or empty list if simulation has not run.
        """
        if self._R is not None:
            return self._R.tolist()
        else:
            return []

class EpidemicSimulation:
    """
    Main class for running the SIR epidemic simulation, visualizing, and saving results.

    Typical usage:
        sim = EpidemicSimulation(...)
        sim.run()
        sim.visualize()
        sim.save_results("results.csv")
    """

    def __init__(
        self,
        population: int = 1000,
        beta: float = 0.3,
        gamma: float = 0.1,
        initial_infected: int = 1,
        initial_recovered: int = 0,
        n_steps: int = 160
    ) -> None:
        """
        Initialize the epidemic simulation.

        Args:
            population (int): Total population.
            beta (float): Infection rate.
            gamma (float): Recovery rate.
            initial_infected (int): Initial infected individuals.
            initial_recovered (int): Initial recovered individuals.
            n_steps (int): Number of time steps to simulate.

        Raises:
            ValueError: If parameters are invalid.
        """
        if population <= 0:
            raise ValueError('Population must be positive.')
        if beta < 0 or gamma < 0:
            raise ValueError('Beta and gamma must be non-negative.')
        if initial_infected < 0 or initial_recovered < 0:
            raise ValueError('Initial infected/recovered must be non-negative.')
        if initial_infected + initial_recovered > population:
            raise ValueError('Initial infected + recovered cannot exceed population.')
        if n_steps <= 0:
            raise ValueError('n_steps must be positive.')

        self.sir_model: SIRModel = SIRModel(population, beta, gamma, initial_infected, initial_recovered)
        self.n_steps: int = n_steps
        self.result_df: Optional[pd.DataFrame] = None

    def run(self) -> None:
        """
        Run the SIR simulation and print summary statistics.

        Side effects:
            - Sets self.result_df to the simulation result DataFrame.
            - Prints summary statistics to stdout.
        """
        self.result_df = self.sir_model.run(self.n_steps)
        self.report_summary()

    def report_summary(self) -> None:
        """
        Print summary statistics of the simulation (peak infected, time to peak, total infected).

        Side effects:
            - Prints summary to stdout.
        """
        if self.result_df is None or self.result_df.empty:
            print('No results to summarize.')
            return
        peak_infected = self.result_df['Infected'].max()
        time_to_peak = self.result_df['Infected'].idxmax()
        total_infected = self.result_df['Recovered'].iloc[-1]
        print(f'Peak infected: {peak_infected:.0f} at time step {time_to_peak}')
        print(f'Total infected (final recovered): {total_infected:.0f}')

    def visualize(self) -> None:
        """
        Visualize the SIR simulation results using matplotlib.

        Side effects:
            - Displays a plot window if results are available.
        """
        if self.result_df is None or self.result_df.empty:
            print("No results to visualize. Please run the simulation first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.result_df['Susceptible'], label='Susceptible')
        plt.plot(self.result_df['Infected'], label='Infected')
        plt.plot(self.result_df['Recovered'], label='Recovered')
        plt.title('SIR Model Epidemic Simulation')
        plt.xlabel('Time Steps')
        plt.ylabel('Number of Individuals')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_results(self, path: Union[str, Path]) -> None:
        """
        Save the simulation results to a CSV file.

        Args:
            path (Union[str, Path]): The path to save the results CSV.

        Raises:
            ValueError: If results are not available to save.
            PermissionError: If file cannot be saved due to permissions.
            Exception: For other unexpected errors.
        """
        if self.result_df is not None and not self.result_df.empty:
            try:
                self.result_df.to_csv(path, index=False)
            except PermissionError as e:
                print(f"Permission denied when saving results to '{path}': {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"Unexpected error saving results to '{path}': {e}", file=sys.stderr)
                raise
        else:
            raise ValueError("No results available to save. Please run the simulation first.")

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for epidemic simulation parameters.

    Parameters/Defaults:
        - Reads from environment variables if set, else uses hardcoded defaults.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Simple SIR epidemic simulation. "
                    "Parameters may be set via environment variables SIM_POPULATION, SIM_BETA, SIM_GAMMA, "
                    "SIM_INITIAL_INFECTED, SIM_INITIAL_RECOVERED, SIM_N_STEPS or via command line."
    )
    parser.add_argument("--population", type=int, default=int(os.environ.get('SIM_POPULATION', 1000)),
                        help="Total population size (default: 1000 or environment variable SIM_POPULATION)")
    parser.add_argument("--beta", type=float, default=float(os.environ.get('SIM_BETA', 0.3)),
                        help="Infection rate beta (default: 0.3 or SIM_BETA)")
    parser.add_argument("--gamma", type=float, default=float(os.environ.get('SIM_GAMMA', 0.1)),
                        help="Recovery rate gamma (default: 0.1 or SIM_GAMMA)")
    parser.add_argument("--initial-infected", type=int, default=int(os.environ.get('SIM_INITIAL_INFECTED', 1)),
                        help="Initial number of infected individuals (default: 1 or SIM_INITIAL_INFECTED)")
    parser.add_argument("--initial-recovered", type=int, default=int(os.environ.get('SIM_INITIAL_RECOVERED', 0)),
                        help="Initial number of recovered individuals (default: 0 or SIM_INITIAL_RECOVERED)")
    parser.add_argument("--n-steps", type=int, default=int(os.environ.get('SIM_N_STEPS', 160)),
                        help="Number of time steps to simulate (default: 160 or SIM_N_STEPS)")
    parser.add_argument("--output", type=str, default=RESULT_PATH,
                        help=f"Path to save simulation results CSV (default: {RESULT_PATH})")
    parser.add_argument("--no-visualize", action='store_true',
                        help="Disable visualization (for headless/server environments)")
    parser.add_argument("--no-save", action='store_true',
                        help="Do not save results to file")
    return parser.parse_args()

def main() -> None:
    """
    Run the epidemic simulation, visualize results, and save to a file.

    Side effects:
        - May print errors to stderr and exit the process on failure.
        - Intended for CLI use only. If importing as a module, do not call main().
    """
    args = parse_args()
    try:
        sim = EpidemicSimulation(
            population=args.population,
            beta=args.beta,
            gamma=args.gamma,
            initial_infected=args.initial_infected,
            initial_recovered=args.initial_recovered,
            n_steps=args.n_steps
        )
        sim.run()
        if not args.no_visualize:
            sim.visualize()
        if not args.no_save:
            try:
                sim.save_results(args.output)
            except Exception as e:
                print(f"Error saving results: {e}", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        print(f"Simulation failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

main()