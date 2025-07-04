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


class Person:
    """
    Represents an individual in the simulation with specific attributes and behaviors.
    """
    def __init__(self, health_status: str, infection_chance: float, recovery_chance: float, interaction_rate: float,
                 interaction_radius: float, infection_duration_range: Tuple[int, int], transmission_probability: float,
                 step_size: float = 1.0):
        self.health_status = health_status
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

    def random_walk(self) -> None:
        """
        Simulates the random movement of the person within the environment.
        """
        step = np.random.uniform(-self.step_size, self.step_size, 2)
        self.position = np.clip(self.position + step, 0, 50)  # Clipped to environment grid

    def interact(self, other: 'Person') -> None:
        """
        Defines interaction between people that may lead to infection.

        :param other: The other person with whom this person interacts.
        """
        if self.health_status == 'infected' and other.health_status == 'susceptible':
            self.become_infected(other)

    def become_infected(self, other: 'Person') -> None:
        """
        Attempts to infect another person based on infection chance and environmental factors.

        :param other: The person to potentially infect.
        """
        environment_factor = np.random.uniform(0.8, 1.2)
        if random.random() < self.transmission_probability * self.infection_chance * environment_factor:
            other.health_status = 'infected'
            other.infection_duration = random.randint(*self.infection_duration_range)
            other.infection_time = 0  # Reset infection time upon infection

    def recover(self) -> None:
        """
        Simulates the recovery process of an infected person.
        """
        if self.health_status == 'infected':
            self.infection_duration -= 1
            self.infection_time += 1  # Track infection time
            if self.infection_duration <= 0:
                if random.random() < self.recovery_chance:
                    self.health_status = 'recovered'
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
        random.seed(42)
        self.people: List[Person] = []
        self.transmission_probability = transmission_probability
        self.recovery_chance = recovery_chance
        self.time_step = 0
        self.infection_counts = []

        # Use Gaussian Mixture Model for initial clustering of agents
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)

        # Fit the GaussianMixture model with random data before sampling
        gmm.fit(np.random.rand(100, 2))
        positions: np.ndarray = gmm.sample(population_size)[0]

        # Ensure positions have the correct shape
        assert positions.shape == (population_size, 2), "Position array has an incorrect shape."

        positions = positions * 50 + 50

        for i in range(population_size):
            health_status = 'susceptible'
            infection_chance = random.uniform(0.05, 0.15)
            interaction_rate = random.uniform(0.1, 0.3)
            interaction_radius = 1.0
            infection_duration_range = (recovery_time - 5, recovery_time + 5)  # Set realistic infection duration range based on recovery time
            person = Person(health_status, infection_chance, recovery_chance, interaction_rate, interaction_radius,
                            infection_duration_range, transmission_probability, step_size)
            person.position = positions[i]
            self.people.append(person)

        # Random selection for initial infections
        for person in random.sample(self.people, initial_infected):
            person.health_status = 'infected'
            person.infection_duration = random.randint(5, 15)

    def run(self, days: int) -> None:
        """
        Executes the simulation over a specified number of days.

        :param days: The number of days to run the simulation.
        """
        for _ in range(days):
            self.time_step += 1

            for person in self.people:
                person.random_walk()

            positions = np.array([person.position for person in self.people])
            tree = KDTree(positions)

            for person in self.people:
                if person.health_status == 'infected':
                    indices = tree.query_ball_point(person.position, r=person.interaction_radius)
                    for idx in indices:
                        other = self.people[idx]
                        if person != other:
                            person.interact(other)
                person.recover()

            infected_count = sum(p.health_status == 'infected' for p in self.people)
            self.infection_counts.append(infected_count)

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the simulation metrics.

        :return: A dictionary containing infection_rate, recovery_rate, and peak_infection_day.
        """
        peak_infection_day = self.infection_counts.index(max(self.infection_counts)) if self.infection_counts else -1
        metrics = {
            'infection_rate': sum(1 for p in self.people if p.health_status == 'infected') / len(self.people),
            'recovery_rate': sum(1 for p in self.people if p.health_status == 'recovered') / len(self.people),
            'peak_infection_day': peak_infection_day
        }
        return metrics

    def visualize(self) -> None:
        """
        Visualizes the results of the simulation.

        Displays a bar chart of the health status distribution of the population at the end of the simulation.
        """
        statuses = [p.health_status for p in self.people]
        labels, counts = np.unique(statuses, return_counts=True)
        plt.bar(labels, counts)
        plt.title('Simulation Results')
        plt.xlabel('Health Status')
        plt.ylabel('Count')
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
                file.write('health_status,infection_chance,recovery_chance,interaction_rate\n')
                for person in self.people:
                    file.write(f"{person.health_status},{person.infection_chance},{person.recovery_chance},{person.interaction_rate}\n")
        except IOError as e:
            logging.error(f'File I/O error: {e}')
        except Exception as e:
            logging.error(f'An unexpected error occurred while writing to the file: {e}')

    def reset(self) -> None:
        """
        Resets or reinitializes the simulation, allowing repeated runs without re-instantiating the class.
        """
        self.__init__(len(self.people), sum(p.health_status == 'infected' for p in self.people), self.transmission_probability, self.recovery_chance)


def main() -> None:
    """
    Initializes and runs the epidemic simulation, then visualizes and saves the results.
    """
    sim = Simulation(population_size=1000, initial_infected=1, transmission_probability=0.1, recovery_chance=0.05)
    sim.run(days=100)
    sim.visualize()
    sim.save_results('results.csv')


# Execute main for both direct execution and sandbox wrapper invocation
main()