import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Path handling setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Person:
    """
    Represents an individual in the simulation with specific attributes and behaviors.
    """
    def __init__(self, health_status: str, infection_probability: float, recovery_rate: float, interaction_rate: float):
        self.health_status = health_status
        self.infection_probability = infection_probability
        self.recovery_rate = recovery_rate
        self.interaction_rate = interaction_rate
        self.position = np.random.uniform(0, 100, 2)  # Random initial position

    def random_walk(self) -> None:
        """
        Simulates the random movement of the person within the environment.
        """
        step = np.random.uniform(-1, 1, 2)  # Random step
        self.position = np.clip(self.position + step, 0, 100)  # Ensure within bounds

    def interact(self, other: 'Person') -> bool:
        """
        Defines interaction between people that may lead to infection.
        """
        distance = np.linalg.norm(self.position - other.position)
        if distance < 1.0:  # Defined interaction radius
            return True
        return False

    def infect(self, other: 'Person', transmission_probability: float) -> None:
        """
        Simulates the infection process when an infected person interacts with a susceptible person.
        """
        if self.health_status == "infected" and other.health_status == "susceptible":
            if random.random() < self.infection_probability * transmission_probability:
                other.health_status = "infected"

    def recover(self) -> None:
        """
        Simulates the recovery process of an infected person.
        """
        if self.health_status == "infected":
            if random.random() < self.recovery_rate:
                self.health_status = "recovered"

class Simulation:
    """
    Manages the simulation of the epidemic spread.
    """
    def __init__(self, population_size: int, initial_infected: int, transmission_probability: float, recovery_rate: float):
        """
        Initializes the simulation with a specified population size, number of initially infected people,
        transmission probability, and recovery rate.

        Parameters:
        - population_size: The total number of people in the simulation.
        - initial_infected: The initial number of infected individuals.
        - transmission_probability: The probability that a susceptible person becomes infected when interacting with an infected person.
        - recovery_rate: The probability that an infected person recovers in a given time step.
        """
        random.seed(42)  # Ensures reproducibility
        self.people = []
        for _ in range(population_size):
            health_status = "susceptible"
            infection_probability = random.uniform(0.05, 0.15)
            interaction_rate = random.uniform(0.1, 0.3)
            person = Person(health_status, infection_probability, recovery_rate, interaction_rate)
            self.people.append(person)
        
        # Infect the initial set of people
        for person in random.sample(self.people, initial_infected):
            person.health_status = "infected"

        self.transmission_probability = transmission_probability
        self.recovery_rate = recovery_rate
        self.time_step = 0
        self.infection_counts = []

    def run(self, days: int) -> None:
        """
        Executes the simulation over a specified number of days.
        """
        for _ in range(days):
            self.time_step += 1

            # Move all people
            for person in self.people:
                person.random_walk()

            # Build a spatial index
            positions = np.array([person.position for person in self.people])
            tree = KDTree(positions)

            # Check for interactions and infections
            for person in self.people:
                if person.health_status == "infected":
                    indices = tree.query_ball_point(person.position, r=1.0)
                    for idx in indices:
                        other = self.people[idx]
                        if person != other and person.interact(other):
                            person.infect(other, self.transmission_probability)
                person.recover()

            # Track infection counts
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
                file.write("health_status,infection_probability,recovery_rate,interaction_rate\n")
                for person in self.people:
                    file.write(f"{person.health_status},{person.infection_probability},{person.recovery_rate},{person.interaction_rate}\n")
        except (PermissionError, OSError) as e:
            print(f"An error occurred while writing to the file: {e}")

def main() -> None:
    sim = Simulation(population_size=1000, initial_infected=1, transmission_probability=0.1, recovery_rate=0.05)
    sim.run(days=100)
    sim.visualize()
    sim.save_results("results.csv")

main()