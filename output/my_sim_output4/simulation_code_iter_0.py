# main.py
import numpy as np
import random
from person import Person
from environment import Environment
from metrics import Metrics
import matplotlib.pyplot as plt

def main():
    # Initialize the random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Setup simulation parameters
    population_size = 1000
    initial_infected = 10
    infection_probability = 0.05
    recovery_time_mean = 14
    grid_dimensions = (50, 50)
    num_days = 100

    # Initialize environment and population
    env = Environment(grid_dimensions)
    people = [Person(env, infection_probability, recovery_time_mean) for _ in range(population_size)]
    
    # Infect some initial individuals
    for i in range(initial_infected):
        people[i].health_status = "Infected"

    # Run the simulation
    metrics = Metrics()
    for day in range(num_days):
        print(f"Day {day}")
        for person in people:
            person.move()
            person.interact(people)
            person.recover()
        
        # Record metrics
        metrics.record(people, day)

    # Visualize results
    metrics.plot_results()

if __name__ == "__main__":
    main()