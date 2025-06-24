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
    def __init__(self, dimensions: Tuple[int, int] = (50, 50)):
        self.dimensions = dimensions
        self.tree = None

    def update_tree(self, positions: np.ndarray) -> None:
        self.tree = KDTree(positions)

    def get_agents_within_radius(self, position: np.ndarray, radius: float) -> List[int]:
        if self.tree is None:
            return []
        return self.tree.query_ball_point(position, r=radius)

class Person:
    """
    Represents an individual in the simulation with specific attributes and behaviors.
    """
    def __init__(self, infected_status: str, transmission_probability: float, recovery_chance: float, interaction_rate: float,
                 interaction_radius: float, infection_duration_range: Tuple[int, int], step_size: float = 1.0):
        self.infected_status = infected_status
        self.transmission_probability = transmission_probability
        self.recovery_chance = recovery_chance
        self.interaction_rate = interaction_rate
        self.interaction_radius = interaction_radius
        self.position = np.array([0.0, 0.0])
        self.infection_duration = 0
        self.infection_duration_range = infection_duration_range
        self.immune_status = 'not_immune'
        self.infection_time = 0
        self.step_size = step_size

    def move(self) -> None:
        step = np.random.uniform(-self.step_size, self.step_size, 2)
        self.position = np.clip(self.position + step, 0, 50)

    def interact(self, other: 'Person') -> None:
        if self.infected_status == 'infected' and other.infected_status == 'susceptible':
            self.infect(other)

    def infect(self, other: 'Person') -> None:
        if other.infected_status == 'susceptible':
            environment_factor = np.random.uniform(0.8, 1.2)
            probability = min(max(np.random.normal(self.transmission_probability, 0.02), 0), 1)
            if random.random() < probability * environment_factor:
                other.get_infected()

    def get_infected(self) -> None:
        self.infected_status = 'infected'
        self.infection_duration = random.randint(*self.infection_duration_range)
        self.infection_time = 0

    def recover(self) -> None:
        if self.infected_status == 'infected':
            self.infection_duration -= 1
            self.infection_time += 1
            if self.infection_duration <= 0:
                if random.random() < self.recovery_chance:
                    self.infected_status = 'recovered'
                    self.update_immune_status()

    def update_immune_status(self) -> None:
        self.immune_status = 'immune'

class Simulation:
    """
    Manages the simulation of the epidemic spread, including evaluation and visualization of results.
    """
    def __init__(self, population_size: int, initial_infected: int, transmission_probability: float,
                 recovery_chance: float, recovery_time: int = 20, step_size: float = 1.0):
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
            infection_chance = random.uniform(0.2, 0.5)
            interaction_rate = random.uniform(0.1, 0.3)
            interaction_radius = random.uniform(5.0, 10.0)
            infection_duration_range = (recovery_time - 5, recovery_time + 5)
            person = Person(infected_status, transmission_probability, recovery_chance, interaction_rate, interaction_radius,
                            infection_duration_range, step_size)
            person.position = positions[i]
            self.people.append(person)

        initial_infected_indices = random.sample(range(population_size), initial_infected)
        for idx in initial_infected_indices:
            self.people[idx].infected_status = 'infected'
            self.people[idx].infection_duration = random.randint(10, 30)

    def generate_clustered_positions(self, population_size: int) -> np.ndarray:
        """
        Generates clustered positions for the population using Gaussian Mixture Model.
        
        :param population_size: Number of positions to generate
        :return: numpy array of positions
        """
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(np.random.rand(100, 2))
        positions = gmm.sample(population_size)[0]
        return np.clip(positions * 50, 0, 50)

    def run(self, days: int) -> None:
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
        peak_infection_day = self.infection_counts.index(max(self.infection_counts)) if self.infection_counts else -1
        steady_state_infection = self.check_steady_state_infection()
        metrics = {
            'infection_rate': sum(1 for p in self.people if p.infected_status == 'infected') / len(self.people),
            'recovery_rate': sum(1 for p in self.people if p.infected_status == 'recovered') / len(self.people),
            'peak_infection_day': peak_infection_day,
            'steady_state_infection': steady_state_infection
        }
        logging.info(f"Steady state infection: {steady_state_infection}")
        return metrics
    
    def check_steady_state_infection(self, threshold: float = 0.01) -> bool:
        if len(self.infection_counts) < 2:
            return False
        recent_changes = np.abs(np.diff(self.infection_counts[-5:]))
        return np.all(recent_changes / np.maximum(1, self.infection_counts[-6:-1]) < threshold)

    def visualize(self) -> None:
        statuses = [p.infected_status for p in self.people]
        labels, counts = np.unique(statuses, return_counts=True)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(labels, counts)
        plt.title('Simulation Results')
        plt.xlabel('Health Status')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.infection_counts, label='Infected')
        plt.title('Infection Over Time')
        plt.xlabel('Days')
        plt.ylabel('Number of Infected People')
        plt.tight_layout()
        plt.show()

    def save_results(self, filename: str) -> None:
        if not filename or not os.path.isdir(os.path.dirname(filename)):
            logging.error('Invalid filename or directory: Ensure the directory exists and filename is not empty.')
            return
        try:
            with open(filename, 'w') as file:
                file.write('infected_status,infection_chance,recovery_chance,interaction_rate\n')
                for person in self.people:
                    file.write(f"{person.infected_status},{person.transmission_probability},{person.recovery_chance},{person.interaction_rate}\n")
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
    population_size = 1000
    initial_infected = int(population_size * 0.05)
    transmission_probability = 0.03
    recovery_chance = 0.9
    recovery_time = 20

    sim = Simulation(population_size, initial_infected, transmission_probability, recovery_chance, recovery_time)
    sim.run(100)
    results = sim.evaluate()
    logging.info(f"Simulation results: {results}")
    sim.visualize()
    sim.save_results('simulation_results.txt')


# Execute main for both direct execution and sandbox wrapper invocation
main()