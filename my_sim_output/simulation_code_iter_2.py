import numpy as np
from matplotlib import pyplot as plt
from enum import Enum
from collections import Counter
import random

# Enum for health status
class HealthStatus(Enum):
    HEALTHY = 0
    INFECTED = 1
    RECOVERED = 2
    DECEASED = 3

# Class for Person entity
class Person:
    """Person class represents a person in the simulation."""

    def __init__(self, location):
        """Initialize a person with a location and health status."""
        self.location = location
        self.health_status = HealthStatus.HEALTHY
        self.days_infected = 0

    def move(self, dimensions):
        """Move the person randomly within the given dimensions."""
        self.location = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]

    def interact(self, other):
        """Interact with another person, potentially spreading or contracting the epidemic."""
        if other.health_status == HealthStatus.INFECTED and self.health_status == HealthStatus.HEALTHY:
            if np.random.random() < infection_probability:
                self.health_status = HealthStatus.INFECTED

    def update_health_status(self, recovery_time, death_probability):
        """Update health status of a person based on recovery_time and death_probability."""
        if self.health_status == HealthStatus.INFECTED:
            self.days_infected += 1
            if np.random.random() < death_probability:
                self.health_status = HealthStatus.DECEASED
            elif self.days_infected >= recovery_time:
                self.health_status = HealthStatus.RECOVERED

# Class for Simulation
class EpidemicSimulation:
    """EpidemicSimulation class represents the simulation environment."""

    def __init__(self, dimensions, population_size, initial_infected_count, infection_probability, recovery_time, death_probability):
        """Initialize the simulation environment."""
        self.dimensions = dimensions
        self.population_size = population_size
        self.initial_infected_count = initial_infected_count
        self.infection_probability = infection_probability
        self.recovery_time = recovery_time
        self.death_probability = death_probability
        self.population = []
        self.total_infected = initial_infected_count
        self.total_recovered = 0
        self.total_deceased = 0

        # Initialize population
        for _ in range(population_size - initial_infected_count):
            location = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]
            self.population.append(Person(location))
        
        # Initialize infected individuals
        for _ in range(initial_infected_count):
            location = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]
            infected_person = Person(location)
            infected_person.health_status = HealthStatus.INFECTED
            self.population.append(infected_person)

    def step(self):
        """Simulate one time step of the epidemic."""
        # Move all people
        for person in self.population:
            if person.health_status != HealthStatus.DECEASED:
                person.move(self.dimensions)

        # Interactions and health status updates
        for person in self.population:
            if person.health_status != HealthStatus.DECEASED:
                # Interact with a random other person
                other = random.choice(self.population)
                if other is not person:
                    person.interact(other)

                # Update health status
                prev_status = person.health_status
                person.update_health_status(self.recovery_time, self.death_probability)
                new_status = person.health_status

                # Update counters
                if prev_status != new_status:
                    if new_status == HealthStatus.INFECTED:
                        self.total_infected += 1
                    elif new_status == HealthStatus.RECOVERED:
                        self.total_recovered += 1
                    elif new_status == HealthStatus.DECEASED:
                        self.total_deceased += 1

    def run(self, steps):
        """Run the simulation for a given number of steps."""
        for _ in range(steps):
            self.step()

    def plot(self):
        """Plot the simulation results."""
        health_status_counter = Counter([p.health_status for p in self.population])
        plt.bar(HealthStatus.keys(), health_status_counter.values())
        plt.show()

# Main function
def main():
    """Run the epidemic simulation."""
    simulation = EpidemicSimulation(
        dimensions=[100, 100],
        population_size=1000,
        initial_infected_count=1,
        infection_probability=0.2,
        recovery_time=14,
        death_probability=0.02
    )
    simulation.run(steps=100)
    simulation.plot()

if __name__ == "__main__":
    main()