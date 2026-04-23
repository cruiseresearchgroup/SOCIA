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
        state (str): The epidemiological state of the individual ('S', 'I', or 'R').
    """
    def __init__(self, state='S'):
        """
        Initialize a Person instance.

        Args:
            state (str): Initial state. 'S' for susceptible, 'I' for infected, 'R' for recovered.
        """
        self.state = state

    def __repr__(self):
        """
        String representation of the Person.

        Returns:
            str: The state of the person.
        """
        return f"Person(state={self.state})"


class SIRSimulation:
    """
    Main simulation class for the SIR epidemic model.

    Attributes:
        population_size (int): Total number of individuals.
        beta (float): Transmission rate.
        gamma (float): Recovery rate.
        initial_infected (int): Number of initially infected individuals.
        n_steps (int): Number of simulation steps.
        k (int): Number of contacts per infected per step.
        random_seed (int or None): Seed for random number generator.
        people (list of Person): The population.
        history (dict): Tracks S, I, and R counts over time.
    """
    def __init__(self, population_size=1000, beta=0.3, gamma=0.1, initial_infected=10, n_steps=100, k=10, random_seed=None):
        """
        Initializes the SIRSimulation.

        Args:
            population_size (int): Number of people in the simulation.
            beta (float): Transmission rate per contact per step.
            gamma (float): Recovery rate per infected per step.
            initial_infected (int): Initial number of infected individuals.
            n_steps (int): Number of time steps to simulate.
            k (int): Number of contacts per infected individual per step.
            random_seed (int or None): Random seed for reproducibility.
        """
        self.population_size = population_size
        self.beta = beta
        self.gamma = gamma
        self.initial_infected = initial_infected
        self.n_steps = n_steps
        self.k = k  # FIXED: Make contact rate configurable per feedback
        self.random_seed = random_seed  # FIXED: Add random_seed parameter per feedback
        if self.random_seed is not None:
            random.seed(self.random_seed)  # FIXED: Set random seed for reproducibility
        self.people = []
        self.history = {'S': [], 'I': [], 'R': []}
        self._initialize_population()

    def _initialize_population(self):
        """
        Sets up the initial population with susceptible and infected individuals.
        """
        self.people = [Person('S') for _ in range(self.population_size)]
        infected_indices = random.sample(range(self.population_size), self.initial_infected)
        for idx in infected_indices:
            self.people[idx].state = 'I'

    def step(self):
        """
        Advances the simulation by one time step: processes infections and recoveries.
        """
        new_infections = set()
        new_recoveries = set()
        infected_indices = [i for i, p in enumerate(self.people) if p.state == 'I']
        susceptible_indices = [i for i, p in enumerate(self.people) if p.state == 'S']

        # Each infected person contacts k unique random others per step
        for inf_idx in infected_indices:
            # FIXED: Use self.k as parameter and ensure unique contacts per step
            contacts = random.sample(range(self.population_size), min(self.k, self.population_size))
            for contact_idx in contacts:
                if self.people[contact_idx].state == 'S':
                    if random.random() < self.beta:
                        new_infections.add(contact_idx)

        for inf_idx in infected_indices:
            if random.random() < self.gamma:
                new_recoveries.add(inf_idx)

        for idx in new_infections:
            self.people[idx].state = 'I'
        for idx in new_recoveries:
            self.people[idx].state = 'R'

    def record(self):
        """
        Records the current counts of S, I, R.
        """
        counts = {'S': 0, 'I': 0, 'R': 0}
        for p in self.people:
            counts[p.state] += 1
        self.history['S'].append(counts['S'])
        self.history['I'].append(counts['I'])
        self.history['R'].append(counts['R'])

    def run(self):
        """
        Executes the simulation loop for n_steps and records the results.
        """
        self.history = {'S': [], 'I': [], 'R': []}
        self.record()
        for _ in range(self.n_steps):
            self.step()
            self.record()

    def visualize(self):
        """
        Plots the SIR curves over time.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.history['S'], label='Susceptible')
        plt.plot(self.history['I'], label='Infected')
        plt.plot(self.history['R'], label='Recovered')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Individuals')
        plt.title('SIR Model Simulation')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_results(self, filename):
        """
        Saves the simulation history to a CSV file.

        Args:
            filename (str): The file to save results to.
        """
        try:
            os.makedirs(DATA_DIR, exist_ok=True)  # FIXED: Ensure output directory exists before writing file
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['time', 'S', 'I', 'R']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for t in range(len(self.history['S'])):
                    writer.writerow({
                        'time': t,
                        'S': self.history['S'][t],
                        'I': self.history['I'][t],
                        'R': self.history['R'][t]
                    })
        except Exception as e:
            print(f"Failed to save results: {e}")  # FIXED: Catch exceptions and provide helpful message

def main():
    """
    Demonstrates initialization, execution, visualization, and saving of the SIR simulation.
    """
    # Example parameters; user can adjust as needed
    sim = SIRSimulation(
        population_size=1000,
        beta=0.3,
        gamma=0.1,
        initial_infected=10,
        n_steps=100,
        k=10,  # FIXED: Contact rate now parameterized
        random_seed=42  # FIXED: Set random seed for reproducibility
    )
    sim.run()
    sim.visualize()
    sim.save_results("results.csv")

# Execute main for both direct execution and sandbox wrapper invocation
main()