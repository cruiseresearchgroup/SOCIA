import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture
import logging
from typing import Tuple, List, Dict

# Set up logging for error tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path handling setup
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '.')
DATA_PATH = os.environ.get('DATA_PATH', 'data')
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Environment:
    """
    Manages the grid environment where agents move and interact.
    """
    def __init__(self, dimensions: Tuple[int, int] = (100, 100)):
        """
        Initializes the environment with a given dimension.
        
        :param dimensions: Tuple representing the size of the grid.
        """
        self.dimensions = dimensions
        self.tree = None

    def update_tree(self, positions: np.ndarray) -> None:
        """
        Updates the KDTree with new positions.

        :param positions: Array of positions for all agents.
        """
        self.tree = KDTree(positions)

    def get_agents_within_radius(self, position: np.ndarray, radius: float) -> List[int]:
        """
        Retrieves indices of agents within a specified radius from a position.

        :param position: Position to query around.
        :param radius: Radius within which to look for other agents.
        :return: List of indices of agents within the radius.
        """
        if self.tree is None:
            return []
        return self.tree.query_ball_point(position, r=radius)

class Person:
    """
    Represents an individual in the simulation with specific attributes and behaviors.
    """
    def __init__(self, infected_status: str, infection_chance: float, recovery_chance: float, interaction_rate: float,
                 interaction_radius: float, infection_duration_range: Tuple[int, int], transmission_probability: float,
                 step_size: float = 1.0):
        """
        Initializes a person with attributes related to infection and movement.

        :param infected_status: Status of infection ('infected', 'susceptible', or 'recovered').
        :param infection_chance: Chance of getting infected during interaction.
        :param recovery_chance: Chance of recovery after the infection duration.
        :param interaction_rate: Rate at which interactions occur.
        :param interaction_radius: Radius within which interactions happen.
        :param infection_duration_range: Range of infection duration.
        :param transmission_probability: Probability of transmitting the infection.
        :param step_size: Movement step size for random walk.
        """
        self.infected_status = infected_status
        self.infection_chance = infection_chance
        self.recovery_chance = recovery_chance
        self.interaction_rate = interaction_rate
        self.interaction_radius = interaction_radius
        self.position = np.array([0.0, 0.0])  # Initialized to a valid numpy array
        self.infection_duration = 0
        self.infection_duration_range = infection_duration_range
        self.immune_status = 'not_immune'
        self.transmission_probability = transmission_probability
        self.infection_time = 0  # Added attribute to track infection time
        self.step_size = step_size

    def move(self) -> None:
        """
        Simulates the random movement of the person within the environment.
        """
        step = np.random.uniform(-self.step_size, self.step_size, 2)
        self.position = np.clip(self.position + step, 0, 100)  # Clipped to environment grid

    def interact(self, other: 'Person') -> None:
        """
        Defines interaction between people that may lead to infection.

        :param other: The other person with whom this person interacts.
        """
        if self.infected_status == 'infected' and other.infected_status == 'susceptible':
            self.infect(other)

    def infect(self, other: 'Person') -> None:
        """
        Attempts to infect another person based on infection chance and environmental factors.

        :param other: The person to potentially infect.
        """
        environment_factor = np.random.uniform(0.8, 1.2)
        if random.random() < np.random.normal(self.transmission_probability, 0.02) * self.infection_chance * environment_factor:
            other.infected_status = 'infected'
            other.infection_duration = random.randint(*self.infection_duration_range)
            other.infection_time = 0  # Reset infection time upon infection

    def recover(self) -> None:
        """
        Simulates the recovery process of an infected person.
        """
        if self.infected_status == 'infected':
            self.infection_duration -= 1
            self.infection_time += 1  # Track infection time
            if self.infection_duration <= 0:
                if random.random() < self.recovery_chance:
                    self.infected_status = 'recovered'
                    self.immune_status = 'immune'

class Simulation:
    """
    Manages the simulation of the epidemic spread, including evaluation and visualization of results.
    """
    def __init__(self, population_size: int, initial_infected: int, transmission_probability: float,
                 recovery_chance: float, recovery_time: int = 14, step_size: float = 1.0):
        """
        Initializes the simulation with a specified population size, number of initially infected people,
        transmission probability, and recovery chance.

        :param population_size: The total number of people in the simulation.
        :param initial_infected: The number of initially infected people.
        :param transmission_probability: The probability of transmission per interaction.
        :param recovery_chance: The chance of recovery after the infection duration.
        :param recovery_time: The average time for recovery.
        :param step_size: The movement step size for each person.
        """
        if population_size <= 0:
            raise ValueError("Population size must be greater than zero.")
        if initial_infected < 0 or initial_infected > population_size:
            raise ValueError("Initial infected count must be between 0 and the population size.")
        
        random.seed(42)
        self.people: List[Person] = []
        self.transmission_probability = transmission_probability
        self.recovery_chance = recovery_chance
        self.time_step = 0
        self.infection_counts = []
        self.environment = Environment()

        positions = self.generate_clustered_positions(population_size)

        for i in range(population_size):
            infected_status = 'susceptible'
            infection_chance = random.uniform(0.05, 0.15)
            interaction_rate = random.uniform(0.1, 0.3)
            interaction_radius = 1.0
            infection_duration_range = (recovery_time - 5, recovery_time + 5)
            person = Person(infected_status, infection_chance, recovery_chance, interaction_rate, interaction_radius,
                            infection_duration_range, transmission_probability, step_size)
            person.position = positions[i]
            self.people.append(person)

        initial_infected_indices = np.random.choice(population_size, initial_infected, replace=False)
        for idx in initial_infected_indices:
            self.people[idx].infected_status = 'infected'
            self.people[idx].infection_duration = random.randint(5, 15)

    def generate_clustered_positions(self, population_size: int) -> np.ndarray:
        """
        Generates initial positions for the population using a clustering approach.

        :param population_size: The total number of people in the simulation.
        :return: An array of initial positions for the population.
        """
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(np.random.rand(100, 2))
        positions = gmm.sample(population_size)[0]
        return np.clip(positions * 100, 0, 100)  # Scale and clip to fit the environment grid

    def run(self, days: int) -> None:
        """
        Executes the simulation over a specified number of days.

        :param days: The number of days to run the simulation.
        """
        for _ in range(days):
            self.time_step += 1

            for person in self.people:
                person.move()

            positions = np.array([person.position for person in self.people])
            self.environment.update_tree(positions)

            for person in self.people:
                if person.infected_status == 'infected':
                    indices = self.environment.get_agents_within_radius(person.position, person.interaction_radius)
                    for idx in indices:
                        other = self.people[idx]
                        if person != other:
                            person.interact(other)
                person.recover()

            infected_count = sum(p.infected_status == 'infected' for p in self.people)
            self.infection_counts.append(infected_count)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the simulation metrics.

        :return: A dictionary containing infection_rate, recovery_rate, and peak_infection_day.
        """
        peak_infection_day = self.infection_counts.index(max(self.infection_counts)) if self.infection_counts else -1
        metrics = {
            'infection_rate': sum(1 for p in self.people if p.infected_status == 'infected') / len(self.people),
            'recovery_rate': sum(1 for p in self.people if p.infected_status == 'recovered') / len(self.people),
            'peak_infection_day': peak_infection_day
        }
        return metrics

    def visualize(self) -> None:
        """
        Visualizes the results of the simulation.

        Displays a bar chart of the health status distribution of the population at the end of the simulation
        and a line graph showing the number of infections over time.
        """
        # Visualize health status distribution
        statuses = [p.infected_status for p in self.people]
        labels, counts = np.unique(statuses, return_counts=True)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(labels, counts)
        plt.title('Simulation Results')
        plt.xlabel('Health Status')
        plt.ylabel('Count')
        
        # Visualize infection curve over time
        plt.subplot(1, 2, 2)
        plt.plot(self.infection_counts, label='Infected')
        plt.title('Infection Over Time')
        plt.xlabel('Days')
        plt.ylabel('Number of Infected People')
        plt.tight_layout()
        plt.show()

    def save_results(self, filename: str) -> None:
        """
        Saves the simulation results to a file.

        :param filename: The name of the file to save the results.
        """
        if not filename:
            logging.error('Invalid filename: Filename cannot be empty.')
            return
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as file:
                file.write('infected_status,infection_chance,recovery_chance,interaction_rate\n')
                for person in self.people:
                    file.write(f"{person.infected_status},{person.infection_chance},{person.recovery_chance},{person.interaction_rate}\n")
        except PermissionError as e:
            logging.error(f'Permission error while writing the file: {e}')
        except IOError as e:
            logging.error(f'File I/O error: {e}')
        except Exception as e:
            logging.error(f'An unexpected error occurred while writing to the file: {e}')

    def reset(self) -> None:
        """
        Resets or reinitializes the simulation, allowing repeated runs without re-instantiating the class.
        """
        self.__init__(len(self.people), sum(p.infected_status == 'infected' for p in self.people), self.transmission_probability, self.recovery_chance)

def main():
    # Example simulation setup
    population_size = 1000
    initial_infected = int(population_size * 0.01)  # 1% initially infected
    transmission_probability = 0.05
    recovery_chance = 0.99
    recovery_time = 14

    sim = Simulation(population_size, initial_infected, transmission_probability, recovery_chance, recovery_time)
    sim.run(30)
    results = sim.evaluate()
    logging.info(f"Simulation results: {results}")
    sim.visualize()
    sim.save_results('simulation_results.txt')


# Execute main for both direct execution and sandbox wrapper invocation
main()