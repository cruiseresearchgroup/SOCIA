import os
import random
import csv
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


class Person:
    """
    Represents an individual in the SIR model.
    Attributes:
        state (str): The epidemiological state ('S', 'I', or 'R').
    """
    def __init__(self, state='S'):
        """
        Initialize a Person with a given state.
        Args:
            state (str): Initial state ('S', 'I', or 'R').
        """
        self.state = state
        pass


class SIRSimulation:
    """
    Simulates the spread of an epidemic using the SIR model.
    Attributes:
        population_size (int): Number of individuals in the simulation.
        beta (float): Probability of infection per contact per time step.
        gamma (float): Probability of recovery per infected per time step.
        initial_infected (int): Initial number of infected individuals.
        population (list): List of Person instances.
        history (list): Time series of (S, I, R) counts.
        steps (int): Number of simulation steps.
    """
    def __init__(self, population_size=1000, beta=0.3, gamma=0.1, initial_infected=10, steps=100):
        """
        Initialize the SIR simulation with parameters.
        Args:
            population_size (int): Size of the population.
            beta (float): Infection rate.
            gamma (float): Recovery rate.
            initial_infected (int): Number of initially infected individuals.
            steps (int): Number of simulation steps.
        """
        self.population_size = population_size
        self.beta = beta
        self.gamma = gamma
        self.initial_infected = initial_infected
        self.steps = steps
        self.population = []
        self.history = []
        self._initialize_population()
        pass

    def _initialize_population(self):
        """
        Set up the initial population with susceptible and infected individuals.
        """
        self.population = []
        # Start with all susceptible
        for _ in range(self.population_size):
            self.population.append(Person(state='S'))
        # Infect initial individuals
        infected_indices = random.sample(range(self.population_size), self.initial_infected)
        for idx in infected_indices:
            self.population[idx].state = 'I'
        pass

    def step(self):
        """
        Perform one time step of the simulation:
        - Susceptible individuals may become infected with probability based on prevalence.
        - Infected individuals may recover.
        """
        new_states = [person.state for person in self.population]
        num_infected = sum(1 for p in self.population if p.state == 'I')

        # Calculate infection probability for each susceptible person
        for idx, person in enumerate(self.population):
            if person.state == 'S':
                # Probability of being infected by any infected individual
                p_infection = 1 - ((1 - self.beta) ** num_infected)
                if random.random() < p_infection:
                    new_states[idx] = 'I'
            elif person.state == 'I':
                if random.random() < self.gamma:
                    new_states[idx] = 'R'
        # Update states
        for idx, person in enumerate(self.population):
            person.state = new_states[idx]
        pass

    def record(self):
        """
        Record the current counts of S, I, and R in the population.
        """
        count_S = sum(1 for p in self.population if p.state == 'S')
        count_I = sum(1 for p in self.population if p.state == 'I')
        count_R = sum(1 for p in self.population if p.state == 'R')
        self.history.append((count_S, count_I, count_R))
        pass

    def run(self):
        """
        Run the simulation for the specified number of steps.
        """
        self.history = []
        self.record()
        for _ in range(self.steps):
            self.step()
            self.record()
        pass

    def visualize(self):
        """
        Plot the SIR curves over time.
        """
        if not self.history:
            print("No data to visualize.")
            return
        S, I, R = zip(*self.history)
        plt.figure(figsize=(10, 6))
        plt.plot(S, label='Susceptible')
        plt.plot(I, label='Infected')
        plt.plot(R, label='Recovered')
        plt.xlabel("Time Steps")
        plt.ylabel("Number of Individuals")
        plt.title("SIR Epidemic Simulation")
        plt.legend()
        plt.tight_layout()
        plt.show()
        pass

    def save_results(self, filename):
        """
        Save the results of the simulation to a CSV file.
        Args:
            filename (str): The filename to save the results to.
        """
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Susceptible', 'Infected', 'Recovered'])
            for i, (s, inf, r) in enumerate(self.history):
                writer.writerow([i, s, inf, r])
        print(f"Results saved to {file_path}")
        pass


def main():
    """
    Main function to run the SIR epidemic simulation.
    Demonstrates initialization, execution, visualization, and result saving.
    """
    # Simulation parameters (can be adjusted)
    population_size = 1000
    beta = 0.3
    gamma = 0.1
    initial_infected = 10
    steps = 100

    # Initialize simulation
    sim = SIRSimulation(
        population_size=population_size,
        beta=beta,
        gamma=gamma,
        initial_infected=initial_infected,
        steps=steps
    )

    # Run simulation
    sim.run()

    # Visualize results
    sim.visualize()

    # Save results
    sim.save_results("results.csv")
    pass


# Execute main for both direct execution and sandbox wrapper invocation
main()