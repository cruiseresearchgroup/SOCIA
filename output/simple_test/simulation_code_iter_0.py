import os
import random
import math
import logging
from typing import List, Tuple

# Setup data file paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Agent:
    """
    Represents an agent in the simulation with basic properties.
    
    Attributes:
        id (int): The unique identifier of the agent.
        position (Tuple[int, int]): The (x, y) position of the agent on the grid.
        state (str): The current state of the agent.
    """

    def __init__(self, agent_id: int, position: Tuple[int, int], state: str = "idle"):
        """
        Initialize an Agent with a unique id, position, and state.
        
        Args:
            agent_id (int): Unique identifier for the agent.
            position (Tuple[int, int]): Initial position of the agent on the grid.
            state (str, optional): Initial state of the agent. Defaults to "idle".
        """
        self.id = agent_id
        self.position = position
        self.state = state

    def move(self, grid_size: Tuple[int, int], move_probability: float) -> None:
        """
        Move the agent to a random adjacent cell with a given probability.
        
        Args:
            grid_size (Tuple[int, int]): The dimensions of the grid (width, height).
            move_probability (float): Probability of the agent moving.
        """
        if random.random() < move_probability:
            x, y = self.position
            # Determine possible moves: up, down, left, right
            possible_moves = [
                (x, y - 1), (x, y + 1),  # Up, Down
                (x - 1, y), (x + 1, y)   # Left, Right
            ]
            # Filter valid moves within grid boundaries
            valid_moves = [
                (nx, ny) for nx, ny in possible_moves
                if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]
            ]
            # Choose a new position randomly from valid moves
            self.position = random.choice(valid_moves)

    def interact(self, agents: List['Agent'], interaction_radius: int) -> None:
        """
        Interact with nearby agents and potentially change state.
        
        Args:
            agents (List[Agent]): List of all agents in the simulation.
            interaction_radius (int): Radius within which interaction occurs.
        """
        for other_agent in agents:
            if other_agent.id != self.id:
                if self._is_within_radius(other_agent, interaction_radius):
                    self.state = "active"  # Example state change

    def _is_within_radius(self, other_agent, radius: int) -> bool:
        """
        Check if another agent is within a certain radius using Euclidean distance.
        
        Args:
            other_agent (Agent): The agent to check distance against.
            radius (int): The interaction radius.
        
        Returns:
            bool: True if within radius, False otherwise.
        """
        return math.dist(self.position, other_agent.position) <= radius


class Environment:
    """
    Manages the simulation environment, including the grid and agents.
    
    Attributes:
        grid_size (Tuple[int, int]): Dimensions of the simulation grid.
        agents (List[Agent]): List of agents in the environment.
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), population_size: int = 50, random_seed: int = 42):
        """
        Initialize the environment with a grid and a set of agents.
        
        Args:
            grid_size (Tuple[int, int], optional): Size of the grid. Defaults to (10, 10).
            population_size (int, optional): Number of agents. Defaults to 50.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.grid_size = grid_size
        self.agents = []
        random.seed(random_seed)
        self._initialize_agents(population_size)

    def _initialize_agents(self, population_size: int) -> None:
        """
        Initialize agents with random positions and idle state.
        
        Args:
            population_size (int): Number of agents to initialize.
        """
        for agent_id in range(population_size):
            position = (random.randint(0, self.grid_size[0] - 1),
                        random.randint(0, self.grid_size[1] - 1))
            agent = Agent(agent_id, position)
            self.agents.append(agent)


class Simulation:
    """
    Manages the simulation execution and time steps.
    
    Attributes:
        environment (Environment): The simulation environment.
        parameters (dict): Parameters for agent behaviors.
    """

    def __init__(self, environment: Environment, parameters: dict):
        """
        Initialize the simulation with an environment and parameters.
        
        Args:
            environment (Environment): The simulation environment.
            parameters (dict): Parameters for simulation behaviors.
        """
        self.environment = environment
        self.parameters = parameters
        self.time_step = 0

    def run(self, start_day: int = 0, end_day: int = 10) -> None:
        """
        Execute the simulation over a given period.
        
        Args:
            start_day (int, optional): Start day of the simulation. Defaults to 0.
            end_day (int, optional): End day of the simulation. Defaults to 10.
        """
        for day in range(start_day, end_day):
            self.time_step += 1
            self._simulate_step()

    def _simulate_step(self) -> None:
        """
        Perform a single time step of the simulation.
        """
        for agent in self.environment.agents:
            agent.move(self.environment.grid_size, self.parameters['move_probability'])
            agent.interact(self.environment.agents, self.parameters['interaction_radius'])

    def evaluate(self) -> dict:
        """
        Evaluate the simulation results based on predefined metrics.
        
        Returns:
            dict: Evaluation results with metric names as keys.
        """
        number_of_interactions = 0
        state_changes = 0
        for agent in self.environment.agents:
            if agent.state == "active":
                state_changes += 1
        # Assuming each state change was due to an interaction
        number_of_interactions = state_changes

        results = {
            'number_of_interactions': number_of_interactions,
            'average_state_change': state_changes / len(self.environment.agents) if self.environment.agents else 0.0
        }
        return results

    def save_results(self, filename: str = "results.csv") -> None:
        """
        Save the simulation results to a file.
        
        Args:
            filename (str, optional): Filename to save results. Defaults to "results.csv".
        """
        results = self.evaluate()
        try:
            with open(filename, 'w') as f:
                for key, value in results.items():
                    f.write(f"{key},{value}\n")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            print(f"Failed to save results to {filename}. Please check the file path.")
        except PermissionError as e:
            logging.error(f"Permission error: {e}")
            print(f"Permission denied when trying to save to {filename}.")


def visualize(simulation: Simulation) -> None:
    """
    Optionally visualize the simulation results (not implemented).
    
    This function is intended for future development to include visualization capabilities such as plotting
    agent positions and states over time. It may involve using libraries like matplotlib to create visual
    representations of the simulation data.
    
    Args:
        simulation (Simulation): The simulation to visualize.
    """
    print("Visualization is not implemented yet. Future implementation may include plotting agent positions and states.")


def main() -> None:
    """
    Main function to initialize, run, and visualize the simulation.
    
    It initializes the environment and simulation with specific parameters,
    runs the simulation, evaluates, and saves the results.
    
    Parameters:
    - move_probability: Probability of agents moving to an adjacent cell.
    - interaction_radius: Radius within which agents can interact with each other.
    
    Expected Results:
    - The simulation will run for a predefined number of days, updating agent positions and states.
    - Results are evaluated based on the number of interactions and average state change, and saved to a file.
    """
    # Initialize environment and simulation
    environment = Environment()
    parameters = {
        'move_probability': 0.5,
        'interaction_radius': 1
    }
    simulation = Simulation(environment, parameters)

    # Run the simulation
    simulation.run()

    # Evaluate and save results
    simulation.save_results("results.csv")

    # Optional visualization
    visualize(simulation)

# Execute main for both direct execution and sandbox wrapper invocation
main()