import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import logging
from typing import List, Tuple, Dict, Optional

def get_data_dir() -> str:
    """
    Retrieve and construct the data directory from environment variables.

    Returns:
        str: The path to the data directory.

    Raises:
        EnvironmentError: If PROJECT_ROOT or DATA_PATH is not set.
        OSError: If directory creation fails.
    """
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")
    if not project_root or not data_path:
        raise EnvironmentError(
            "Both PROJECT_ROOT and DATA_PATH environment variables must be set.\n"
            "Example usage:\n"
            "  export PROJECT_ROOT=/path/to/project\n"
            "  export DATA_PATH=data"
        )
    data_dir = os.path.join(project_root, data_path)
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating data directory '{data_dir}': {e}")
        raise
    return data_dir

class SIRSimulation:
    """
    A class to simulate the spread of an epidemic using the SIR model.

    Attributes:
        population_size (int): Total size of the population (must be >= 1).
        initial_infected (int): Number of initially infected individuals (must be in [1, population_size]).
        beta (float): Infection rate per susceptible-infected contact per time step (must be >= 0).
        gamma (float): Recovery rate per infected per time step (must be >= 0).
        max_steps (int): Maximum number of simulation steps (must be >= 1).
        S (List[int]): List tracking susceptible individuals over time.
        I (List[int]): List tracking infected individuals over time.
        R (List[int]): List tracking recovered individuals over time.
        t (List[int]): List tracking time steps.
    """
    def __init__(
        self,
        population_size: int = 1000,
        initial_infected: int = 1,
        beta: float = 0.3,
        gamma: float = 0.1,
        max_steps: int = 160
    ) -> None:
        """
        Initialize the SIR simulation parameters.

        Args:
            population_size (int): Total population. Must be >= 1.
            initial_infected (int): Initial number of infected individuals. Must be >=1 and <= population_size.
            beta (float): Infection rate per susceptible-infected contact per time step. Must be >= 0.
            gamma (float): Recovery rate per infected per time step. Must be >= 0.
            max_steps (int): Number of time steps to simulate. Must be >=1.
        """
        if population_size < 1:
            raise ValueError("population_size must be >= 1")
        if initial_infected < 1 or initial_infected > population_size:
            raise ValueError("initial_infected must be in [1, population_size]")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.population_size: int = population_size
        self.initial_infected: int = initial_infected
        self.beta: float = beta
        self.gamma: float = gamma
        self.max_steps: int = max_steps
        self.S: List[int] = []
        self.I: List[int] = []
        self.R: List[int] = []
        self.t: List[int] = []

    def step(self, S: int, I: int, R: int) -> Tuple[int, int, int]:
        """
        Perform one time step of the SIR model using deterministic equations.

        Args:
            S (int): Number of susceptible individuals.
            I (int): Number of infected individuals.
            R (int): Number of recovered individuals.

        Returns:
            Tuple[int, int, int]: Updated (S, I, R) values as integers, after rounding and adjustment to conserve population.

        Notes:
            Internally, calculations are performed in float, but values are rounded to int for the returned result.
            The method adjusts the compartment with the largest rounding error to ensure population consistency.
        """
        N = self.population_size
        # Calculate new values as floats
        new_infected = self.beta * S * I / N
        new_recovered = self.gamma * I

        S_new_f = S - new_infected
        I_new_f = I + new_infected - new_recovered
        R_new_f = R + new_recovered

        # Rounding strategy: round all, then adjust the compartment with largest fractional part to ensure sum is N
        srf = [S_new_f, I_new_f, R_new_f]
        int_parts = [int(np.floor(x)) for x in srf]
        fracs = [x - np.floor(x) for x in srf]

        remainder = N - sum(int_parts)
        # Distribute the remainder to the compartments with largest fractions
        if remainder > 0:
            # assign +1 to 'remainder' compartments with largest fractions
            indices = np.argsort(fracs)[::-1]
            for idx in indices[:remainder]:
                int_parts[idx] += 1

        S_new, I_new, R_new = int_parts

        # Ensure no negative values
        S_new = max(S_new, 0)
        I_new = max(I_new, 0)
        R_new = max(R_new, 0)

        # Final check/correction: if due to rounding we lost/gained 1 person, adjust S
        total = S_new + I_new + R_new
        if total > N:
            # Remove from S
            S_new = max(S_new - (total - N), 0)
        elif total < N:
            S_new = min(S_new + (N - total), N)

        return S_new, I_new, R_new

    def run(self) -> None:
        """
        Run the SIR simulation over the specified number of time steps.
        Initializes state and iteratively updates S, I, R.
        Stops early if there are no more infected individuals.
        """
        S = self.population_size - self.initial_infected
        I = self.initial_infected
        R = 0

        self.S = [S]
        self.I = [I]
        self.R = [R]
        self.t = [0]

        for step in range(1, self.max_steps + 1):
            S, I, R = self.step(S, I, R)
            self.S.append(S)
            self.I.append(I)
            self.R.append(R)
            self.t.append(step)
            # Stop early if no more infected individuals
            if I == 0:
                break

    def get_results(self) -> List[Dict[str, int]]:
        """
        Get the simulation results as a list of dicts.

        Returns:
            List[Dict[str, int]]: List of dictionaries with simulation results per time step.
        """
        return [
            {"time": t, "susceptible": s, "infected": i, "recovered": r}
            for t, s, i, r in zip(self.t, self.S, self.I, self.R)
        ]

    def save_results(self, filename: str) -> None:
        """
        Save the simulation results to a CSV file.

        Args:
            filename (str): Path to the results CSV file.

        Raises:
            OSError: If the file cannot be written.
        """
        results = self.get_results()
        try:
            with open(filename, "w", newline="") as csvfile:
                fieldnames = ["time", "susceptible", "infected", "recovered"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
        except OSError as e:
            logging.error(f"Error saving results to {filename}: {e}", exc_info=True)
            print(f"Error saving results to {filename}: {e}")
            raise

    def visualize(self) -> None:
        """
        Visualize the simulation results using matplotlib.
        Produces a line plot for Susceptible, Infected, and Recovered over time.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.t, self.S, label="Susceptible")
        plt.plot(self.t, self.I, label="Infected")
        plt.plot(self.t, self.R, label="Recovered")
        plt.xlabel("Time Steps")
        plt.ylabel("Number of Individuals")
        plt.title("SIR Epidemic Simulation")
        plt.legend()
        plt.tight_layout()
        plt.show()

def main() -> None:
    """
    Main function to run the SIR epidemic simulation.

    Steps performed:
        1. Determines the data directory from environment variables.
        2. Initializes the SIRSimulation with default parameters.
        3. Runs the simulation.
        4. Visualizes the simulation results.
        5. Saves the results to a CSV file in the data directory.

    Note:
        Handles EnvironmentError internally and does not propagate it.
    """
    try:
        data_dir = get_data_dir()
        result_path = os.path.join(data_dir, "results.csv")
    except EnvironmentError as e:
        print(f"Configuration error: {e}")
        print("Please set the environment variables 'PROJECT_ROOT' and 'DATA_PATH' and try again.")
        return
    except OSError as e:
        print(f"Could not create data directory: {e}")
        return

    sim = SIRSimulation()
    sim.run()
    sim.visualize()
    try:
        sim.save_results(result_path)
    except OSError as e:
        print(f"Failed to save simulation results. Please check your file path and permissions. Error: {e}")


# Execute main for both direct execution and sandbox wrapper invocation
main()