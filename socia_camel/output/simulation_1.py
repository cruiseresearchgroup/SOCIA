# -*- coding: utf-8 -*-
"""
This script simulates the SEIR (Susceptible-Exposed-Infected-Recovered-Dead) epidemiological model. 
It uses a deterministic approach, implemented with numerical integration using the Euler method.
The model tracks the number of individuals in each compartment (S, E, I, R, D) over time.

The script defines the SEIRModel class, which encapsulates the model's parameters and methods.
It also includes an example usage demonstrating how to initialize, run, and plot the simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt

class SEIRModel:
    """
    A class representing the SEIR epidemiological model with a death compartment.

    Attributes:
        N (int): Total population size.
        beta (float): Transmission rate.
        sigma (float): Incubation rate (1/latent period).
        gamma (float): Recovery rate (1/infectious period).
        mu (float): Mortality rate.
        S0 (int): Initial number of susceptible individuals.
        simulation_duration (int): Duration of the simulation in time steps.
        t (numpy.ndarray): Time array for the simulation.
        S (numpy.ndarray): Array to store the number of susceptible individuals at each time step.
        E (numpy.ndarray): Array to store the number of exposed individuals at each time step.
        I (numpy.ndarray): Array to store the number of infected individuals at each time step.
        R (numpy.ndarray): Array to store the number of recovered individuals at each time step.
        D (numpy.ndarray): Array to store the number of dead individuals at each time step.


    """
    def __init__(self, N, beta, sigma, gamma, mu, I0=1, E0=0, R0=0, D0=0, simulation_duration=100):
        """
        Initializes the SEIR model with given parameters.

        Args:
            N (int): Total population size.
            beta (float): Transmission rate.
            sigma (float): Incubation rate.
            gamma (float): Recovery rate.
            mu (float): Mortality rate.
            I0 (int, optional): Initial number of infected individuals. Defaults to 1.
            E0 (int, optional): Initial number of exposed individuals. Defaults to 0.
            R0 (int, optional): Initial number of recovered individuals. Defaults to 0.
            D0 (int, optional): Initial number of dead individuals. Defaults to 0.
            simulation_duration (int, optional): Duration of the simulation. Defaults to 100.
        """
        self.N = N
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.mu = mu
        self.S0 = N - I0 - E0 - R0 - D0  # Calculate initial susceptible population
        self.simulation_duration = simulation_duration
        self.t = np.arange(simulation_duration + 1)  # Create time array
        self.S = np.zeros(simulation_duration + 1)  # Initialize arrays to store results
        self.E = np.zeros(simulation_duration + 1)
        self.I = np.zeros(simulation_duration + 1)
        self.R = np.zeros(simulation_duration + 1)
        self.D = np.zeros(simulation_duration + 1)

        # Initialize the arrays with initial values
        self.S[0] = self.S0
        self.E[0] = E0
        self.I[0] = I0
        self.R[0] = R0
        self.D[0] = D0

    def run(self):
        """
        Runs the SEIR simulation.

        The method uses a numerical integration (Euler method) to update the number of 
        individuals in each compartment at each time step. The core SEIR equations are implemented 
        using vectorized NumPy operations for efficiency.
        """
        # Precompute constant values for performance
        beta_over_N = self.beta / self.N
        gamma_plus_mu = self.gamma + self.mu

        # Use NumPy vectorized operations for speed
        for i in range(self.simulation_duration):
            self.S[i+1] = self.S[i] - beta_over_N * self.S[i] * self.I[i]  # Update Susceptible
            self.E[i+1] = self.E[i] + beta_over_N * self.S[i] * self.I[i] - self.sigma * self.E[i]  # Update Exposed
            self.I[i+1] = self.I[i] + self.sigma * self.E[i] - gamma_plus_mu * self.I[i]  # Update Infected
            self.R[i+1] = self.R[i] + self.gamma * self.I[i] # Update Recovered
            self.D[i+1] = self.D[i] + self.mu * self.I[i]  # Update Dead


    def plot_results(self):
        """
        Plots the simulation results.

        This method generates a plot showing the number of individuals in each compartment (S, E, I, R, D)
        over time. The plot is labeled and displayed using Matplotlib.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.t, self.S, label='Susceptible')
        plt.plot(self.t, self.E, label='Exposed')
        plt.plot(self.t, self.I, label='Infected')
        plt.plot(self.t, self.R, label='Recovered')
        plt.plot(self.t, self.D, label='Dead')
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('SEIR Model Simulation')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
N = 1000  # Total population
beta = 0.2  # Transmission rate
sigma = 0.1  # Incubation rate
gamma = 0.05  # Recovery rate
mu = 0.01  # Mortality rate
simulation_duration = 100  # Simulation duration

model = SEIRModel(N, beta, sigma, gamma, mu, simulation_duration=simulation_duration)
model.run()
model.plot_results()