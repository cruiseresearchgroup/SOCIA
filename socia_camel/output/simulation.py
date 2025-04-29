# -*- coding: utf-8 -*-
"""
This script simulates a simplified epidemic model (SIRD) with a fixed population size.
The model tracks the number of Susceptible, Infected, Recovered, and Deceased individuals over a specified number of days.
It uses a Monte Carlo approach to simulate the spread of infection based on contact rates, infection rate, recovery rate, and death rate.
The simulation results are then plotted to visualize the epidemic's progression.
"""

import random
import matplotlib.pyplot as plt
import numpy as np

class Model:
    """
    Represents the SIRD epidemic model.

    Attributes:
        population_size (int): The total size of the population.
        infection_rate (float): The probability of infection upon contact with an infected individual.
        recovery_rate (float): The probability of an infected individual recovering on a given day.
        death_rate (float): The probability of an infected individual dying on a given day.
        simulation_days (int): The number of days to simulate.
        population (numpy.ndarray): A 1D array representing the state of each individual ('S', 'I', 'R', 'D').
        S (numpy.ndarray): Number of susceptible individuals over time.
        I (numpy.ndarray): Number of infected individuals over time.
        R (numpy.ndarray): Number of recovered individuals over time.
        D (numpy.ndarray): Number of deceased individuals over time.
    """
    def __init__(self, population_size=1000, infection_rate=0.05, recovery_rate=0.1,
                 death_rate=0.01, initial_infected=1, simulation_days=100):
        """
        Initializes the model with given parameters.

        Args:
            population_size (int): The total population size.
            infection_rate (float): Probability of infection upon contact.
            recovery_rate (float): Probability of recovery per day.
            death_rate (float): Probability of death per day.
            initial_infected (int): Initial number of infected individuals.
            simulation_days (int): Number of days to simulate.
        """
        self.population_size = population_size
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.death_rate = death_rate
        self.simulation_days = simulation_days

        # Initialize the population array: 'S' for Susceptible, 'I' for Infected
        self.population = np.array(['S'] * population_size)
        self.population[:initial_infected] = 'I'

        # Initialize arrays to store the counts of each state over time
        self.S = np.zeros(simulation_days)
        self.I = np.zeros(simulation_days)
        self.R = np.zeros(simulation_days)
        self.D = np.zeros(simulation_days)

    def run(self):
        """
        Runs the simulation for the specified number of days.
        """
        for day in range(self.simulation_days):
            # Find indices of infected individuals
            infected_indices = np.where(self.population == 'I')[0]
            num_infected = len(infected_indices)

            # Determine recoveries and deaths among infected individuals
            recovered = np.random.rand(num_infected) < self.recovery_rate
            deceased = np.logical_and(~recovered, np.random.rand(num_infected) < self.death_rate)

            # Update the population status
            self.population[infected_indices[recovered]] = 'R' # Mark recovered individuals
            self.population[infected_indices[deceased]] = 'D' # Mark deceased individuals

            # Get indices of remaining infected individuals
            remaining_infected_indices = infected_indices[np.logical_and(~recovered, ~deceased)]

            # Simulate infection spread
            for i in remaining_infected_indices:
                # Each infected individual contacts 5 random others
                contacts = random.sample(range(self.population_size), 5)
                for contact in contacts:
                    # Infect susceptible contacts with a certain probability
                    if self.population[contact] == 'S' and random.random() < self.infection_rate:
                        self.population[contact] = 'I'

            # Record the number of individuals in each state for the current day
            self.S[day] = np.sum(self.population == 'S')
            self.I[day] = np.sum(self.population == 'I')
            self.R[day] = np.sum(self.population == 'R')
            self.D[day] = np.sum(self.population == 'D')

    def plot(self):
        """
        Plots the simulation results (S, I, R, D) over time.
        """
        days = range(self.simulation_days)
        plt.plot(days, self.S, label="Susceptible")
        plt.plot(days, self.I, label="Infected")
        plt.plot(days, self.R, label="Recovered")
        plt.plot(days, self.D, label="Deceased")
        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.title("Epidemic Simulation")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    model = Model()
    model.run()
    model.plot()