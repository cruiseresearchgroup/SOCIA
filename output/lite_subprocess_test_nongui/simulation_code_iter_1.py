import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for servers
import matplotlib.pyplot as plt
import csv
from typing import List, Tuple, Optional

class SIRModel:
    """
    Implements a simple SIR (Susceptible-Infected-Recovered) epidemic model.

    Attributes:
        population (int): Total number of individuals in the simulation.
        beta (float): Infection rate per contact per time step.
        gamma (float): Recovery rate per infected per time step.
        S (int): Number of susceptible individuals.
        I (int): Number of infected individuals.
        R (int): Number of recovered individuals.
        history (List[Tuple[int, int, int]]): Stores the history of S, I, R over time.
    """
    def __init__(
        self,
        population: int = 1000,
        infected_init: int = 1,
        beta: float = 0.3,
        gamma: float = 0.1
    ) -> None:
        """
        Initialize the SIR model with population size, initial infections, infection and recovery rates.

        Args:
            population (int): Total population size.
            infected_init (int): Initial number of infected individuals.
            beta (float): Infection probability per susceptible-infected pair per time step.
            gamma (float): Recovery probability per infected per time step.
        """
        self.population: int = population
        self.beta: float = beta
        self.gamma: float = gamma
        self.S: int = population - infected_init
        self.I: int = infected_init
        self.R: int = 0
        self.history: List[Tuple[int, int, int]] = []
        self._record_state()

    def step(self) -> None:
        """
        Executes a single time step of the SIR model, updating S, I, and R counts.

        Returns:
            None
        """
        prob_infection: float = self.beta * self.I / self.population
        prob_infection = min(max(prob_infection, 0.0), 1.0)
        new_infections: int = np.random.binomial(self.S, prob_infection)
        new_recoveries: int = np.random.binomial(self.I, self.gamma)

        self.S -= new_infections
        self.I += new_infections - new_recoveries
        self.R += new_recoveries
        # Guarantee no negative numbers
        self.S = max(self.S, 0)
        self.I = max(self.I, 0)
        self.R = max(self.R, 0)
        self._record_state()

    def _record_state(self) -> None:
        """
        Records the current S, I, R counts into the history.

        Returns:
            None
        """
        self.history.append((self.S, self.I, self.R))

    def run(self, steps: int = 100) -> None:
        """
        Runs the SIR simulation for the given number of steps.

        Args:
            steps (int): Number of time steps to simulate.

        Returns:
            None
        """
        for _ in range(steps):
            self.step()

    def get_history(self) -> List[Tuple[int, int, int]]:
        """
        Returns the recorded S, I, R history.

        Returns:
            List[Tuple[int, int, int]]: List of tuples (S, I, R) for each time step.
        """
        return self.history

class EpidemicSimulation:
    """
    Main simulation class that manages the SIR model, results saving, and visualization.

    Attributes:
        model (SIRModel): The SIR epidemic model instance.
        steps (int): Number of simulation steps.
        history_np (Optional[np.ndarray]): Cached NumPy array of the simulation history.
    """
    def __init__(
        self,
        population: int = 1000,
        infected_init: int = 1,
        beta: float = 0.3,
        gamma: float = 0.1,
        steps: int = 100
    ) -> None:
        """
        Initializes the epidemic simulation.

        Args:
            population (int): Total population size.
            infected_init (int): Initial number of infected individuals.
            beta (float): Infection rate.
            gamma (float): Recovery rate.
            steps (int): Number of simulation steps.
        """
        self.model: SIRModel = SIRModel(population, infected_init, beta, gamma)
        self.steps: int = steps
        self._history_np: Optional[np.ndarray] = None

    def run(self) -> None:
        """
        Runs the epidemic simulation for the configured number of steps.

        Returns:
            None
        """
        self.model.run(self.steps)
        self._history_np = np.array(self.model.get_history())

    def save_results(self, result_path: str) -> None:
        """
        Saves the simulation results (history of S, I, R) to a CSV file.

        Args:
            result_path (str): File path for saving the results.

        Returns:
            None

        Raises:
            OSError: If the file cannot be written.
        """
        history = self.model.get_history()
        try:
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Step', 'Susceptible', 'Infected', 'Recovered'])
                for step, (S, I, R) in enumerate(history):
                    writer.writerow([step, S, I, R])
        except OSError as e:
            print(f"Error saving results to '{result_path}': {e}", file=sys.stderr)
            raise

    def plot(self, picture_path: str) -> None:
        """
        Plots the SIR curves and saves the figure to disk.

        Args:
            picture_path (str): File path to save the plot.

        Returns:
            None

        Raises:
            OSError: If the figure cannot be saved.
        """
        history_np = self._history_np
        if history_np is None:
            history_np = np.array(self.model.get_history())
        try:
            os.makedirs(os.path.dirname(picture_path), exist_ok=True)
            plt.figure(figsize=(8, 5))
            plt.plot(history_np[:, 0], label='Susceptible')
            plt.plot(history_np[:, 1], label='Infected')
            plt.plot(history_np[:, 2], label='Recovered')
            plt.title("SIR Epidemic Simulation")
            plt.xlabel("Time Step")
            plt.ylabel("Number of Individuals")
            plt.legend()
            plt.tight_layout()
            plt.savefig(picture_path)
            plt.close()
        except OSError as e:
            print(f"Error saving plot to '{picture_path}': {e}", file=sys.stderr)
            raise

def main() -> None:
    """
    Entry point for the SIR epidemic simulation.

    Sets up paths, checks environment variables, creates the simulation,
    runs it, saves results, and makes a plot.

    Environment Variables:
        PROJECT_ROOT (str): Root directory of the project. Must be set.
        DATA_PATH (str): Relative path under PROJECT_ROOT to save outputs. Must be set.

    Raises:
        EnvironmentError: If required environment variables are not set.
        OSError: If there are issues with file or directory creation/saving.
    """
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    if PROJECT_ROOT is None:
        print("Environment variable 'PROJECT_ROOT' is not set.", file=sys.stderr)
        raise EnvironmentError("Environment variable 'PROJECT_ROOT' is not set.")
    if DATA_PATH is None:
        print("Environment variable 'DATA_PATH' is not set.", file=sys.stderr)
        raise EnvironmentError("Environment variable 'DATA_PATH' is not set.")
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating data directory '{DATA_DIR}': {e}", file=sys.stderr)
        raise
    result_path = os.path.join(DATA_DIR, "results.csv")
    picture_path = os.path.join(DATA_DIR, "figure.png")

    # Simulation parameters (can be adjusted as needed)
    POPULATION = 1000
    INFECTED_INIT = 10
    BETA = 0.2
    GAMMA = 0.05
    STEPS = 160

    # Run simulation
    sim = EpidemicSimulation(
        population=POPULATION,
        infected_init=INFECTED_INIT,
        beta=BETA,
        gamma=GAMMA,
        steps=STEPS
    )
    sim.run()
    try:
        sim.save_results(result_path)  # Save results
    except OSError:
        print("Failed to save simulation results.", file=sys.stderr)
        return
    try:
        sim.plot(picture_path)         # Save plot
    except OSError:
        print("Failed to save simulation plot.", file=sys.stderr)
        return

if __name__ == "__main__" or True:
    main()


# Execute main for both direct execution and sandbox wrapper invocation
main()