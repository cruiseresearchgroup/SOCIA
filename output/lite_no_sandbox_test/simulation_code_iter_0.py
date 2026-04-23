import os
import sys
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        """
        self.population: int = population
        self.beta: float = beta
        self.gamma: float = gamma
        self.initial_infected: int = initial_infected
        self.initial_recovered: int = initial_recovered
        self.S: List[float] = []
        self.I: List[float] = []
        self.R: List[float] = []

    def reset(self) -> None:
        """
        Reset the model to the initial state.
        """
        susceptible: int = self.population - self.initial_infected - self.initial_recovered
        self.S = [float(susceptible)]
        self.I = [float(self.initial_infected)]
        self.R = [float(self.initial_recovered)]

    def step(self) -> None:
        """
        Advance the simulation by one time step using the SIR equations.

        At each step:
            - New infections are calculated as beta * S * I / N.
            - New recoveries are calculated as gamma * I.
            - S, I, R are updated accordingly, ensuring values do not fall below zero.
        """
        S_prev = self.S[-1]
        I_prev = self.I[-1]
        R_prev = self.R[-1]

        new_infections = self.beta * S_prev * I_prev / self.population
        new_recoveries = self.gamma * I_prev

        S_next = S_prev - new_infections
        I_next = I_prev + new_infections - new_recoveries
        R_next = R_prev + new_recoveries

        S_next = max(S_next, 0)
        I_next = max(I_next, 0)
        R_next = max(R_next, 0)

        self.S.append(S_next)
        self.I.append(I_next)
        self.R.append(R_next)

    def run(self, n_steps: int) -> pd.DataFrame:
        """
        Run the SIR simulation for a given number of steps.

        Args:
            n_steps (int): Number of simulation steps.

        Returns:
            pd.DataFrame: DataFrame containing time series of S, I, R.
        """
        # Use numpy arrays for efficient storage if n_steps is large
        if n_steps > 1000:
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
            df = pd.DataFrame({
                'Susceptible': S,
                'Infected': I,
                'Recovered': R
            })
            # Also update the object's lists for compatibility
            self.S = S.tolist()
            self.I = I.tolist()
            self.R = R.tolist()
            return df
        else:
            self.reset()
            for _ in range(n_steps):
                self.step()
            df = pd.DataFrame({
                'Susceptible': self.S,
                'Infected': self.I,
                'Recovered': self.R
            })
            return df

class EpidemicSimulation:
    """
    Main class for running the SIR epidemic simulation, visualizing, and saving results.
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
        """
        self.sir_model: SIRModel = SIRModel(population, beta, gamma, initial_infected, initial_recovered)
        self.n_steps: int = n_steps
        self.result_df: Optional[pd.DataFrame] = None

    def run(self) -> None:
        """
        Run the SIR simulation.
        """
        self.result_df = self.sir_model.run(self.n_steps)

    def visualize(self) -> None:
        """
        Visualize the SIR simulation results using matplotlib.
        """
        if self.result_df is None:
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
        """
        if self.result_df is not None:
            try:
                self.result_df.to_csv(path, index=False)
            except Exception as e:
                print(f"Failed to save results to '{path}': {e}", file=sys.stderr)
        else:
            print("No results available to save. Please run the simulation first.")

def main() -> None:
    """
    Run the epidemic simulation, visualize results, and save to a file.
    """
    population = 1000
    beta = 0.3
    gamma = 0.1
    initial_infected = 1
    initial_recovered = 0
    n_steps = 160

    sim = EpidemicSimulation(
        population=population,
        beta=beta,
        gamma=gamma,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        n_steps=n_steps
    )
    sim.run()
    sim.visualize()
    sim.save_results(RESULT_PATH)


# Execute main for both direct execution and sandbox wrapper invocation
main()