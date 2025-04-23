# main.py

import numpy as np
import random
import matplotlib.pyplot as plt
from agent import Person
from environment import Environment
from metrics import Metrics

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Simulation parameters
    population_size = 1000
    grid_dimensions = (50, 50)
    initial_infected_percentage = 0.01
    transmission_probability = 0.05
    recovery_probability = 0.1
    time_steps = 100

    # Initialize the environment
    env = Environment(grid_dimensions, population_size, initial_infected_percentage,
                      transmission_probability, recovery_probability)

    # Metrics to track
    metrics = Metrics(env)

    # Run the simulation
    for _ in range(time_steps):
        env.step()
        metrics.update()

    # Plot results
    metrics.plot_results()

if __name__ == "__main__":
    main()