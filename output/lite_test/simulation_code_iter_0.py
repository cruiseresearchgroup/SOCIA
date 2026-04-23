import os
import numpy as np
import matplotlib.pyplot as plt
import csv

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Individual:
    """
    Represents an individual in the SIR model population.

    Attributes:
        state (str): The epidemiological state, one of 'S', 'I', or 'R'.
    """
    def __init__(self, state):
        """
        Initialize an Individual with a given state.

        Args:
            state (str): Initial state, must be 'S', 'I', or 'R'.
        """
        self.state = state
        pass

class SIRSimulation:
    """
    Implements a simple SIR (Susceptible-Infected-Recovered) epidemic simulation.

    Attributes:
        population_size (int): Total number of individuals.
        initial_infected (int): Initial number of infected individuals.
        beta (float): Infection probability per contact per time step.
        gamma (float): Recovery probability per infected individual per time step.
        individuals (list of Individual): The population.
        history (dict): Tracks the number of S, I, R at each time step.
        steps (int): Number of time steps to simulate.
    """
    def __init__(self, population_size=1000, initial_infected=10, beta=0.3, gamma=0.1, steps=100):
        """
        Initializes the SIR simulation with specified parameters.

        Args:
            population_size (int): Number of individuals.
            initial_infected (int): Initially infected individuals.
            beta (float): Probability of infection per contact.
            gamma (float): Probability of recovery per time step.
            steps (int): Number of time steps.
        """
        self.population_size = population_size
        self.initial_infected = initial_infected
        self.beta = beta
        self.gamma = gamma
        self.steps = steps
        self.individuals = []
        self.history = {"S": [], "I": [], "R": []}
        self._initialize_population()
        pass

    def _initialize_population(self):
        """
        Initializes the population with the specified number of susceptible and infected individuals.
        """
        for _ in range(self.initial_infected):
            self.individuals.append(Individual('I'))
        for _ in range(self.population_size - self.initial_infected):
            self.individuals.append(Individual('S'))
        np.random.shuffle(self.individuals)
        pass

    def step(self):
        """
        Performs a single time step in the simulation, updating infection and recovery states.
        """
        new_infections = []
        new_recoveries = []

        num_infected = sum(1 for indiv in self.individuals if indiv.state == 'I')
        num_susceptible = sum(1 for indiv in self.individuals if indiv.state == 'S')

        infection_prob = 1 - (1 - self.beta) ** num_infected if self.population_size > 0 else 0

        for indiv in self.individuals:
            if indiv.state == 'S':
                if np.random.random() < infection_prob:
                    new_infections.append(indiv)
            elif indiv.state == 'I':
                if np.random.random() < self.gamma:
                    new_recoveries.append(indiv)

        for indiv in new_infections:
            indiv.state = 'I'
        for indiv in new_recoveries:
            indiv.state = 'R'
        pass

    def record_history(self):
        """
        Records the current counts of S, I, and R in the population.
        """
        s_count = sum(1 for indiv in self.individuals if indiv.state == 'S')
        i_count = sum(1 for indiv in self.individuals if indiv.state == 'I')
        r_count = sum(1 for indiv in self.individuals if indiv.state == 'R')
        self.history["S"].append(s_count)
        self.history["I"].append(i_count)
        self.history["R"].append(r_count)
        pass

    def run(self):
        """
        Runs the full simulation loop for the specified number of time steps.
        """
        self.record_history()
        for _ in range(self.steps):
            self.step()
            self.record_history()
        pass

    def visualize(self):
        """
        Visualizes the SIR time series using matplotlib.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.history["S"], label="Susceptible (S)")
        plt.plot(self.history["I"], label="Infected (I)")
        plt.plot(self.history["R"], label="Recovered (R)")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Individuals")
        plt.title("SIR Epidemic Simulation")
        plt.legend()
        plt.tight_layout()
        plt.show()
        pass

    def save_results(self, filename):
        """
        Saves the simulation's SIR time series to a CSV file.

        Args:
            filename (str): The path to the output CSV file.
        """
        out_path = os.path.join(DATA_DIR, filename)
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time_step", "susceptible", "infected", "recovered"])
            for t, (s, i, r) in enumerate(zip(self.history["S"], self.history["I"], self.history["R"])):
                writer.writerow([t, s, i, r])
        pass

def main():
    """
    Driver function for the SIR epidemic simulation.

    Demonstrates initialization, running, visualization, and saving of results.
    """
    sim = SIRSimulation(
        population_size=1000,
        initial_infected=10,
        beta=0.3,
        gamma=0.1,
        steps=100
    )
    sim.run()
    sim.visualize()
    sim.save_results("results.csv")  # Demonstration of saving results
    pass

# Execute main for both direct execution and sandbox wrapper invocation
main()