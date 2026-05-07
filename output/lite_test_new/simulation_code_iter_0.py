import os
import random
import csv
import matplotlib.pyplot as plt

# Path handling setup
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Person:
    """
    Represents an individual in the SIR epidemic simulation.

    Attributes:
        state (str): The current health state of the individual ('S', 'I', or 'R').
    """
    def __init__(self, state='S'):
        """
        Initialize a Person with a given state.

        Args:
            state (str): Initial state, 'S' for susceptible, 'I' for infected, 'R' for recovered.
        """
        self.state = state

    def __repr__(self):
        """
        String representation of the Person instance.

        Returns:
            str: The state of the person.
        """
        return self.state

    pass

class SIRSimulation:
    """
    Simulates an epidemic using the SIR model.

    Attributes:
        population_size (int): Number of individuals in the simulation.
        initial_infected (int): Number of initially infected individuals.
        beta (float): Infection rate (probability of disease transmission per contact per time step).
        gamma (float): Recovery rate (probability of recovery per infected individual per time step).
        people (list of Person): List of Person objects in the simulation.
        history (dict): Dictionary storing time series of S, I, R counts.
        steps (int): Number of simulation steps already executed.
    """
    def __init__(self, population_size=1000, initial_infected=10, beta=0.3, gamma=0.1):
        """
        Initializes the SIR simulation.

        Args:
            population_size (int): Total number of individuals.
            initial_infected (int): Number of initially infected individuals.
            beta (float): Infection probability.
            gamma (float): Recovery probability.
        """
        self.population_size = population_size
        self.initial_infected = initial_infected
        self.beta = beta
        self.gamma = gamma
        self.people = []
        self.history = {'S': [], 'I': [], 'R': []}
        self.steps = 0

        self._initialize_population()

    def _initialize_population(self):
        """
        Initializes the population with the given number of susceptible and infected individuals.
        """
        self.people = []
        for _ in range(self.initial_infected):
            self.people.append(Person('I'))
        for _ in range(self.population_size - self.initial_infected):
            self.people.append(Person('S'))
        random.shuffle(self.people)
        self._record_history()
        pass

    def _record_history(self):
        """
        Records the current state counts into the history.
        """
        states = [p.state for p in self.people]
        S = states.count('S')
        I = states.count('I')
        R = states.count('R')
        self.history['S'].append(S)
        self.history['I'].append(I)
        self.history['R'].append(R)
        pass

    def step(self):
        """
        Executes a single time step of the simulation, updating infections and recoveries.
        """
        new_infections = []
        new_recoveries = []
        # For each infected person, try to infect susceptible individuals
        infected_indices = [i for i, p in enumerate(self.people) if p.state == 'I']
        susceptible_indices = [i for i, p in enumerate(self.people) if p.state == 'S']

        # Infect susceptible people
        for i in infected_indices:
            for j in susceptible_indices:
                # Each infected individual meets each susceptible once per step
                if random.random() < self.beta:
                    new_infections.append(j)

        # Remove duplicates
        new_infections = list(set(new_infections))

        # Update the states for new infections
        for idx in new_infections:
            if self.people[idx].state == 'S':
                self.people[idx].state = 'I'

        # Recovery step
        for i in infected_indices:
            if random.random() < self.gamma:
                new_recoveries.append(i)

        for idx in new_recoveries:
            self.people[idx].state = 'R'

        self._record_history()
        self.steps += 1
        pass

    def run(self, max_steps=100):
        """
        Runs the simulation for the specified number of steps or until no infected remain.

        Args:
            max_steps (int): Maximum number of steps to simulate.
        """
        for _ in range(max_steps):
            if self.history['I'][-1] == 0:
                break
            self.step()
        pass

    def visualize(self):
        """
        Visualizes the time series of S, I, R counts using matplotlib.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['S'], label='Susceptible')
        plt.plot(self.history['I'], label='Infected')
        plt.plot(self.history['R'], label='Recovered')
        plt.xlabel('Time step')
        plt.ylabel('Number of individuals')
        plt.title('SIR Epidemic Simulation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        pass

    def save_results(self, filename):
        """
        Saves the simulation results (history) to a CSV file.

        Args:
            filename (str): Filename for the results CSV file.
        """
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Susceptible', 'Infected', 'Recovered'])
            for step in range(len(self.history['S'])):
                writer.writerow([step, self.history['S'][step], self.history['I'][step], self.history['R'][step]])
        pass

def main():
    """
    Main entry point for the SIR epidemic simulation.

    Initializes the simulation, runs it, visualizes results, and saves to a file.
    """
    sim = SIRSimulation(
        population_size=500,
        initial_infected=5,
        beta=0.2,
        gamma=0.05
    )
    sim.run(max_steps=160)
    sim.visualize()
    sim.save_results("results.csv")
    pass

# Execute main for both direct execution and sandbox wrapper invocation
main()