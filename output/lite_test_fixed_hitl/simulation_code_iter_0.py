import os
import numpy as np
import matplotlib.pyplot as plt
import csv

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
os.makedirs(DATA_DIR, exist_ok=True)
result_path = os.path.join(DATA_DIR, "results.csv")

class SIRSimulation:
    """
    A simple SIR (Susceptible-Infected-Recovered) epidemic simulation.

    Attributes:
        population_size (int): Total number of individuals in the population.
        initial_infected (int): Number of initially infected individuals.
        beta (float): Infection rate per susceptible-infected contact per time step.
        gamma (float): Recovery rate per infected individual per time step.
        n_steps (int): Number of time steps for the simulation.
        S_history (list): History of susceptible counts.
        I_history (list): History of infected counts.
        R_history (list): History of recovered counts.
    """
    def __init__(self, population_size=1000, initial_infected=10, beta=0.3, gamma=0.1, n_steps=100):
        """
        Initialize the SIRSimulation.

        Args:
            population_size (int): Total population.
            initial_infected (int): Number of initially infected individuals.
            beta (float): Infection rate.
            gamma (float): Recovery rate.
            n_steps (int): Number of time steps to simulate.
        """
        self.population_size = population_size
        self.initial_infected = initial_infected
        self.beta = beta
        self.gamma = gamma
        self.n_steps = n_steps

        self.S = population_size - initial_infected
        self.I = initial_infected
        self.R = 0

        self.S_history = []
        self.I_history = []
        self.R_history = []
        pass

    def step(self):
        """
        Advance the simulation by one time step using the SIR model equations.

        Uses simple stochastic transitions based on binomial sampling.

        Returns:
            None
        """
        # Number of new infections
        prob_infection = self.beta * self.I / self.population_size if self.population_size > 0 else 0
        new_infections = np.random.binomial(self.S, prob_infection)
        new_recoveries = np.random.binomial(self.I, self.gamma)

        # Update compartments
        self.S = max(self.S - new_infections, 0)
        self.I = max(self.I + new_infections - new_recoveries, 0)
        self.R = min(self.population_size - self.S - self.I, self.population_size)

        # Save history
        self.S_history.append(self.S)
        self.I_history.append(self.I)
        self.R_history.append(self.R)
        pass

    def run(self):
        """
        Run the SIR simulation for the specified number of steps.

        Returns:
            None
        """
        # Record initial state
        self.S_history = [self.S]
        self.I_history = [self.I]
        self.R_history = [self.R]
        for _ in range(self.n_steps):
            self.step()
        pass

    def get_results(self):
        """
        Get the time series results of the simulation.

        Returns:
            times (list): List of time steps.
            S_history (list): Susceptible count over time.
            I_history (list): Infected count over time.
            R_history (list): Recovered count over time.
        """
        times = list(range(len(self.S_history)))
        return times, self.S_history, self.I_history, self.R_history
        pass

    def save_results(self, filepath):
        """
        Save the SIR simulation results to a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            None
        """
        times, S, I, R = self.get_results()
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'susceptible', 'infected', 'recovered'])
            for t, s, i, r in zip(times, S, I, R):
                writer.writerow([t, s, i, r])
        pass

    def visualize(self):
        """
        Visualize the SIR simulation over time.

        Returns:
            None
        """
        times, S, I, R = self.get_results()
        plt.figure(figsize=(10, 6))
        plt.plot(times, S, label='Susceptible')
        plt.plot(times, I, label='Infected')
        plt.plot(times, R, label='Recovered')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Individuals')
        plt.title('SIR Model Simulation')
        plt.legend()
        plt.tight_layout()
        plt.show()
        pass

def main():
    """
    Main function to run the SIR simulation, visualize, and save results.

    Returns:
        None
    """
    sim = SIRSimulation(
        population_size=1000,
        initial_infected=10,
        beta=0.3,
        gamma=0.1,
        n_steps=100
    )
    sim.run()
    sim.visualize()
    sim.save_results(result_path)
    pass

# Execute main for both direct execution and sandbox wrapper invocation
main()