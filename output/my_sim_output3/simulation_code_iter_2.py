import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

# Constants for health statuses
SUSCEPTIBLE = "susceptible"
INFECTED = "infected"
RECOVERED = "recovered"

class Person:
    """
    Class representing an individual in the simulation.
    """

    def __init__(self, health_status: str, infection_probability: float, recovery_time: int, location: Tuple[int, int]):
        self.health_status = health_status
        self.infection_probability = infection_probability
        self.recovery_time = recovery_time
        self.location = location

    def move(self, grid_size: Tuple[int, int]):
        """
        Move the person to a new location within the grid bounds, restricted to orthogonal directions.
        """
        x, y = self.location
        movement_choice = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        new_x = min(max(x + movement_choice[0], 0), grid_size[0] - 1)
        new_y = min(max(y + movement_choice[1], 0), grid_size[1] - 1)
        self.location = (new_x, new_y)

    def interact(self, other: 'Person', transmission_rate: float):
        """
        Interact with another person and potentially transmit the infection.
        """
        if self.health_status == INFECTED and other.health_status == SUSCEPTIBLE:
            if random.random() < transmission_rate:
                other.health_status = INFECTED
                other.recovery_time = int(np.random.exponential(1 / transmission_rate))
        elif self.health_status == SUSCEPTIBLE and other.health_status == INFECTED:
            if random.random() < transmission_rate:
                self.health_status = INFECTED
                self.recovery_time = int(np.random.exponential(1 / transmission_rate))

    def recover(self):
        """
        Handle the recovery process of the person.
        """
        if self.health_status == INFECTED:
            self.recovery_time -= 1
            if self.recovery_time <= 0:
                self.health_status = RECOVERED

class Environment:
    """
    Class to manage the simulation environment.
    """

    def __init__(self, grid_size: Tuple[int, int], population: List[Person]):
        self.grid_size = grid_size
        self.population = population
        self.infection_counts = []
        self.recovered_counts = []

    def step(self, transmission_rate: float):
        """
        Perform a simulation step, updating the state of the environment.
        """
        # Spatial partitioning using a dictionary to manage interactions
        location_map = {}
        
        for person in self.population:
            person.move(self.grid_size)
            loc = person.location
            if loc not in location_map:
                location_map[loc] = []
            location_map[loc].append(person)

        # Interactions and possible infection
        for people in location_map.values():
            if len(people) > 1:
                for i, person in enumerate(people):
                    for other in people[i + 1:]:
                        person.interact(other, transmission_rate)

        # Recovery process
        for person in self.population:
            person.recover()

        # Track metrics
        self.infection_counts.append(sum(1 for p in self.population if p.health_status == INFECTED))
        self.recovered_counts.append(sum(1 for p in self.population if p.health_status == RECOVERED))

    def get_metrics(self):
        """
        Calculate and return the metrics of the simulation.
        """
        peak_infection = max(self.infection_counts)
        return {
            "infection_count": self.infection_counts,
            "recovered_count": self.recovered_counts,
            "peak_infection": peak_infection
        }

class Simulation:
    """
    Class to control the simulation process.
    """

    def __init__(self, population_size: int, initial_infected: int, grid_size: Tuple[int, int],
                 transmission_rate: float, recovery_rate: float, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Initialize population
        self.population = []
        for _ in range(population_size):
            health_status = SUSCEPTIBLE
            infection_probability = transmission_rate
            recovery_time = 0
            location = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
            self.population.append(Person(health_status, infection_probability, recovery_time, location))

        # Infect initial individuals
        for person in random.sample(self.population, initial_infected):
            person.health_status = INFECTED
            person.recovery_time = int(np.random.exponential(1 / recovery_rate))

        self.environment = Environment(grid_size, self.population)
        self.transmission_rate = transmission_rate
        self.grid_size = grid_size

    def run(self, steps: int):
        """
        Run the simulation for a specified number of steps.
        """
        for _ in range(steps):
            self.environment.step(self.transmission_rate)

    def visualize(self):
        """
        Visualize the results of the simulation.
        """
        metrics = self.environment.get_metrics()
        plt.figure(figsize=(12, 6))
        plt.plot(metrics["infection_count"], label="Infected")
        plt.plot(metrics["recovered_count"], label="Recovered")
        plt.title("Epidemic Simulation")
        plt.xlabel("Days")
        plt.ylabel("Number of Individuals")
        plt.legend()
        plt.show()

def main():
    """
    Main function to run the epidemic simulation.
    """
    # Simulation parameters
    population_size = 1000
    initial_infected = 10
    grid_size = (100, 100)
    transmission_rate = 0.1
    recovery_rate = 0.05
    simulation_steps = 100

    # Create and run the simulation
    sim = Simulation(population_size, initial_infected, grid_size, transmission_rate, recovery_rate)
    sim.run(simulation_steps)
    sim.visualize()

if __name__ == "__main__":
    main()