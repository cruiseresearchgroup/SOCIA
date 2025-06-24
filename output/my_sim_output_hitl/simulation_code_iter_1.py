import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture
import logging

# Set up logging for error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path handling setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Person:
    """
    Represents an individual in the simulation with specific attributes and behaviors.
    """
    def __init__(self, health_status: str, infection_chance: float, recovery_chance: float, interaction_rate: float, interaction_radius: float, infection_duration_range: tuple):
        self.health_status = health_status
        self.infection_chance = infection_chance
        self.recovery_chance = recovery_chance
        self.interaction_rate = interaction_rate
        self.interaction_radius = interaction_radius
        self.position = None
        self.infection_duration = 0
        self.infection_duration_range = infection_duration_range
        self.immune_status = "not_immune"

    def random_walk(self) -> None:
        """
        Simulates the random movement of the person within the environment.
        """
        step = np.random.uniform(-1, 1, 2)
        self.position = np.clip(self.position + step, 0, 100)

    def interact(self, other: 'Person') -> None:
        """
        Defines interaction between people that may lead to infection.
        """
        if self.health_status == "infected" and other.health_status == "susceptible":
            self.infect(other)

    def infect(self, other: 'Person') -> None:
        """
        Attempts to infect another person based on infection chance and environmental factors.
        """
        environment_factor = np.random.uniform(0.8, 1.2)
        if random.random() < self.infection_chance * environment_factor:
            other.health_status = "infected"
            other.infection_duration = random.randint(*self.infection_duration_range)

    def recover(self) -> None:
        """
        Simulates the recovery process of an infected person.
        """
        if self.health_status == "infected":
            self.infection_duration -= 1
            if self.infection_duration <= 0:
                if random.random() < self.recovery_chance:
                    self.health_status = "recovered"
                    self.immune_status = "immune"

class Simulation:
    """
    Manages the simulation of the epidemic spread.
    """
    def __init__(self, population_size: int, initial_infected: int, transmission_probability: float, recovery_chance: float):
        """
        Initializes the simulation with a specified population size, number of initially infected people,
        transmission probability, and recovery chance.
        """
        random.seed(42)
        self.people = []
        self.transmission_probability = transmission_probability
        self.recovery_chance = recovery_chance
        self.time_step = 0
        self.infection_counts = []

        # Use Gaussian Mixture Model for initial clustering of agents
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        positions = gmm.sample(population_size)[0]
        
        # Ensure positions have the correct shape
        assert positions.shape == (population_size, 2), "Position array has an incorrect shape."

        positions = positions * 50 + 50

        for i in range(population_size):
            health_status = "susceptible"
            infection_chance = random.uniform(0.05, 0.15)
            interaction_rate = random.uniform(0.1, 0.3)
            interaction_radius = 1.0
            infection_duration_range = (5, 15)
            person = Person(health_status, infection_chance, recovery_chance, interaction_rate, interaction_radius, infection_duration_range)
            person.position = positions[i]
            self.people.append(person)

        for person in random.sample(self.people, initial_infected):
            person.health_status = "infected"
            person.infection_duration = random.randint(5, 15)

    def run(self, days: int) -> None:
        """
        Executes the simulation over a specified number of days.
        """
        for _ in range(days):
            self.time_step += 1

            for person in self.people:
                person.random_walk()

            positions = np.array([person.position for person in self.people])
            tree = KDTree(positions)

            for person in self.people:
                if person.health_status == "infected":
                    indices = tree.query_ball_point(person.position, r=person.interaction_radius)
                    for idx in indices:
                        other = self.people[idx]
                        if person != other:
                            person.interact(other)
                person.recover()

            infected_count = sum(p.health_status == "infected" for p in self.people)
            self.infection_counts.append(infected_count)

    def evaluate(self) -> dict:
        """
        Evaluates the simulation metrics.
        """
        peak_infection_day = self.infection_counts.index(max(self.infection_counts)) if self.infection_counts else -1
        metrics = {
            "infection_rate": sum(1 for p in self.people if p.health_status == "infected") / len(self.people),
            "recovery_rate": sum(1 for p in self.people if p.health_status == "recovered") / len(self.people),
            "peak_infection_day": peak_infection_day
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
        plt.title("Simulation Results")
        plt.xlabel("Health Status")
        plt.ylabel("Count")
        plt.show()

    def save_results(self, filename: str) -> None:
        """
        Saves the simulation results to a file.
        """
        try:
            with open(filename, "w") as file:
                file.write("health_status,infection_chance,recovery_chance,interaction_rate\n")
                for person in self.people:
                    file.write(f"{person.health_status},{person.infection_chance},{person.recovery_chance},{person.interaction_rate}\n")
        except IOError as e:
            logging.error(f"File I/O error: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while writing to the file: {e}")
            raise

def main() -> None:
    sim = Simulation(population_size=1000, initial_infected=1, transmission_probability=0.1, recovery_chance=0.05)
    sim.run(days=100)
    sim.visualize()
    sim.save_results("results.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()