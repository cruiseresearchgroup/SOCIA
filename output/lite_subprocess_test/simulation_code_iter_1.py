import os
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import List, Dict, Optional

def get_data_dir() -> str:
    """
    Constructs the data directory path using environment variables.
    Checks for required environment variables and raises descriptive errors if missing.

    Returns:
        str: The absolute path to the data directory.

    Raises:
        EnvironmentError: If PROJECT_ROOT or DATA_PATH environment variables are not set.
    """
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")
    if project_root is None:
        raise EnvironmentError("Environment variable PROJECT_ROOT is not set.")
    if data_path is None:
        raise EnvironmentError("Environment variable DATA_PATH is not set.")
    data_dir = os.path.join(project_root, data_path)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_result_path() -> str:
    """
    Returns the path to the results CSV file, ensuring the data directory exists.

    Returns:
        str: The absolute path to the results CSV file.
    """
    data_dir = get_data_dir()
    return os.path.join(data_dir, "results.csv")

class SIRAgent:
    """
    Represents an individual in the SIR epidemic simulation.

    Attributes:
        state (str): The current state of the agent ('S', 'I', or 'R').
    """
    def __init__(self, state: str = 'S') -> None:
        """
        Initialize a new agent with the given state.

        Args:
            state (str): Initial state, 'S' (susceptible), 'I' (infected), or 'R' (recovered).
        """
        self.state: str = state

class SIRSimulation:
    """
    Main class to run the SIR (Susceptible-Infected-Recovered) epidemic simulation.

    Attributes:
        population_size (int): Total number of agents in the simulation.
        initial_infected (int): Number of initially infected agents.
        beta (float): Infection probability per contact.
        gamma (float): Recovery probability per time step.
        steps (int): Number of time steps to simulate.
        seed (int): Seed for random number generation.
        agents (List[SIRAgent]): List of SIRAgent instances in the simulation.
        history (Dict[str, List[int]]): History of S, I, R counts over time.
        curr_counts (Dict[str, int]): Current counts of S, I, R agents.
    """
    def __init__(
        self,
        population_size: int = 1000,
        initial_infected: int = 10,
        beta: float = 0.3,
        gamma: float = 0.1,
        steps: int = 100,
        seed: Optional[int] = 42
    ) -> None:
        """
        Initialize the simulation with model parameters.

        Args:
            population_size (int): Total number of agents.
            initial_infected (int): Number of initially infected agents.
            beta (float): Infection probability per contact.
            gamma (float): Recovery probability per time step.
            steps (int): Number of time steps to simulate.
            seed (Optional[int]): Seed for random number generation.
        """
        self.population_size: int = population_size
        self.initial_infected: int = initial_infected
        self.beta: float = beta
        self.gamma: float = gamma
        self.steps: int = steps
        self.seed: Optional[int] = seed
        self.agents: List[SIRAgent] = []
        self.history: Dict[str, List[int]] = {'S': [], 'I': [], 'R': []}
        self.curr_counts: Dict[str, int] = {'S': 0, 'I': 0, 'R': 0}
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._initialize_population()

    def _initialize_population(self) -> None:
        """
        Set up the initial population of agents with appropriate states.
        """
        self.agents = [SIRAgent('S') for _ in range(self.population_size)]
        infected_indices = random.sample(range(self.population_size), self.initial_infected)
        for idx in infected_indices:
            self.agents[idx].state = 'I'
        # Initialize counts
        self.curr_counts = {
            'S': self.population_size - self.initial_infected,
            'I': self.initial_infected,
            'R': 0
        }

    def step(self) -> None:
        """
        Perform a single time step of the simulation: infections and recoveries.
        Maintains running counts for efficiency.
        """
        # For efficiency, use running counts
        new_states: List[str] = []
        num_infected: int = self.curr_counts['I']
        num_susceptible: int = self.curr_counts['S']
        num_recovered: int = self.curr_counts['R']
        updated_counts: Dict[str, int] = {'S': 0, 'I': 0, 'R': 0}
        for agent in self.agents:
            if agent.state == 'S':
                prob_avoid = (1 - self.beta) ** num_infected
                prob_infection = 1 - prob_avoid
                if random.random() < prob_infection:
                    new_states.append('I')
                    updated_counts['I'] += 1
                else:
                    new_states.append('S')
                    updated_counts['S'] += 1
            elif agent.state == 'I':
                if random.random() < self.gamma:
                    new_states.append('R')
                    updated_counts['R'] += 1
                else:
                    new_states.append('I')
                    updated_counts['I'] += 1
            else:
                new_states.append('R')
                updated_counts['R'] += 1
        for agent, state in zip(self.agents, new_states):
            agent.state = state
        self.curr_counts = updated_counts

    def record(self) -> None:
        """
        Record the current counts of S, I, and R in the population.
        """
        self.history['S'].append(self.curr_counts['S'])
        self.history['I'].append(self.curr_counts['I'])
        self.history['R'].append(self.curr_counts['R'])

    def run(self) -> None:
        """
        Execute the simulation for the specified number of steps.
        """
        self.history = {'S': [], 'I': [], 'R': []}
        # Recalculate initial counts in case run() is called multiple times
        self.curr_counts = {'S': 0, 'I': 0, 'R': 0}
        for agent in self.agents:
            self.curr_counts[agent.state] += 1
        self.record()
        for _ in range(self.steps):
            self.step()
            self.record()

    def save_results(self, path: str) -> None:
        """
        Save the simulation results to a CSV file.

        Args:
            path (str): Path to the CSV file.

        Raises:
            IOError: If unable to write to the file.
        """
        try:
            with open(path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['step', 'S', 'I', 'R'])
                for t in range(self.steps + 1):
                    writer.writerow([t, self.history['S'][t], self.history['I'][t], self.history['R'][t]])
        except Exception as e:
            print(f"Error saving results to {path}: {e}")
            raise

    def visualize(self) -> None:
        """
        Plot the SIR curves over time.
        """
        steps = range(self.steps + 1)
        plt.figure(figsize=(10,6))
        plt.plot(steps, self.history['S'], label='Susceptible')
        plt.plot(steps, self.history['I'], label='Infected')
        plt.plot(steps, self.history['R'], label='Recovered')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Agents')
        plt.title('SIR Epidemic Simulation')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main() -> None:
    """
    Main function to initialize, run, visualize, and save the SIR simulation.
    """
    try:
        result_path = get_result_path()
    except EnvironmentError as e:
        print(f"Environment configuration error: {e}")
        return
    sim = SIRSimulation(
        population_size=1000,
        initial_infected=10,
        beta=0.3,
        gamma=0.1,
        steps=100,
        seed=42
    )
    sim.run()
    sim.visualize()
    try:
        sim.save_results(result_path)
    except Exception:
        pass

main()