import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List

# Follow PEP 8 style guidelines and name the class as CitySimulation instead of 'City'
class CitySimulation:
    """
    Class representing the simulation of a city during an epidemic.
    """

    class HealthStatus(Enum):
        """
        Enum representing the health status of a person.
        """
        HEALTHY = 1
        INFECTED = 2
        RECOVERED = 3
        DEAD = 4

    class Person:
        """
        Class representing a person in the city.
        """

        def __init__(self, location, mobility):
            self.health_status = CitySimulation.HealthStatus.HEALTHY
            self.location = location
            self.mobility = mobility

        def move_to_different_location(self, new_location):
            self.location = new_location

        def interact_with_others(self, people):
            for person in people:
                if person.health_status == CitySimulation.HealthStatus.INFECTED:
                    self.fall_ill()

        def recover_from_illness(self):
            if np.random.rand() < parameters['recovery_rate']:
                self.health_status = CitySimulation.HealthStatus.RECOVERED

        def fall_ill(self):
            self.health_status = CitySimulation.HealthStatus.INFECTED

    class Location:
        """
        Class representing a location in the city.
        """

        def __init__(self, population_density, infection_rate):
            self.population_density = population_density
            self.infection_rate = infection_rate

        def increase_infection_rate(self):
            self.infection_rate += 0.01

        def decrease_infection_rate(self):
            self.infection_rate -= 0.01

    def __init__(self, population, initial_infection_rate, recovery_rate):
        self.population = [self.Person(np.random.randint(100, size=2), np.random.randint(5)) for _ in range(population)]
        self.locations = [self.Location(np.random.randint(50), np.random.rand()) for _ in range(10000)]
        self.initial_infection_rate = initial_infection_rate
        self.recovery_rate = recovery_rate
        self.total_infected = 0
        self.total_recovered = 0
        self.total_deaths = 0

    def simulate_day(self):
        for person in self.population:
            if person.health_status == self.HealthStatus.INFECTED:
                person.recover_from_illness()
                self.total_recovered += 1

            new_location = (person.location + np.random.randint(-person.mobility, person.mobility, size=2)) % 100
            person.move_to_different_location(new_location)
            location = self.locations[new_location[0]*100 + new_location[1]]
            if np.random.rand() < location.infection_rate:
                person.fall_ill()
                self.total_infected += 1

    def simulate(self, days):
        for _ in range(days):
            self.simulate_day()

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.total_infected, label='Total Infected')
        plt.plot(self.total_recovered, label='Total Recovered')
        plt.plot(self.total_deaths, label='Total Deaths')
        plt.legend()
        plt.title('Epidemic Simulation Results')
        plt.xlabel('Days')
        plt.ylabel('Number of People')
        plt.show()

def main():
    # Initialize the simulation with the specified parameters
    simulation = CitySimulation(population=1000, initial_infection_rate=0.1, recovery_rate=0.05)

    # Run the simulation for a specified number of days
    simulation.simulate(days=100)

    # Plot the results of the simulation
    simulation.plot_results()

if __name__ == "__main__":
    main()