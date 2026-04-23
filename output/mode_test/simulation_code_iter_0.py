import os
import random
import csv
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class SIRAgent:
    """
    Represents an individual in the SIR epidemic model.
    States:
        - 'S': Susceptible
        - 'I': Infected
        - 'R': Recovered
    """
    def __init__(self, state='S'):
        """
        Initialize agent with a given state.
        Args:
            state (str): The initial state ('S', 'I', or 'R').
        """
        self.state = state
        pass

class SIRSimulation:
    """
    Simulates the SIR epidemic model in a well-mixed population.
    """
    def __init__(self, population_size=1000, initial_infected=10, beta=0.3, gamma=0.1, time_steps=100):
        """
        Initialize the SIR simulation.
        Args:
            population_size (int): Number of individuals in the simulation.
            initial_infected (int): Number of initially infected individuals.
            beta (float): Infection rate (probability of transmission per contact per step).
            gamma (float): Recovery rate (probability of recovery per infected agent per step).
            time_steps (int): Number of steps to run the simulation.
        """
        self.population_size = population_size
        self.initial_infected = initial_infected
        self.beta = beta
        self.gamma = gamma
        self.time_steps = time_steps
        self.agents = []
        self.history = []
        self._initialize_agents()
        pass

    def _initialize_agents(self):
        """
        Initialize the population with susceptible and infected agents.
        """
        self.agents = []
        for i in range(self.population_size):
            if i < self.initial_infected:
                self.agents.append(SIRAgent(state='I'))
            else:
                self.agents.append(SIRAgent(state='S'))
        random.shuffle(self.agents)
        pass

    def _count_states(self):
        """
        Count the number of agents in each state.
        Returns:
            dict: Counts of 'S', 'I', and 'R' agents.
        """
        counts = {'S': 0, 'I': 0, 'R': 0}
        for agent in self.agents:
            counts[agent.state] += 1
        return counts

    def step(self):
        """
        Perform one simulation time step: infections and recoveries.
        """
        new_infections = set()
        new_recoveries = set()
        num_infected = sum(1 for agent in self.agents if agent.state == 'I')
        num_susceptible = sum(1 for agent in self.agents if agent.state == 'S')
        if num_infected == 0:
            # No more infections possible
            self.history.append(self._count_states())
            return

        infection_prob = 1 - (1 - self.beta) ** num_infected  # Well-mixed approximation

        for idx, agent in enumerate(self.agents):
            if agent.state == 'S':
                if random.random() < infection_prob:
                    new_infections.add(idx)
            elif agent.state == 'I':
                if random.random() < self.gamma:
                    new_recoveries.add(idx)

        # Update states
        for idx in new_infections:
            self.agents[idx].state = 'I'
        for idx in new_recoveries:
            self.agents[idx].state = 'R'

        self.history.append(self._count_states())
        pass

    def run(self):
        """
        Run the simulation for the set number of time steps.
        """
        self.history = []
        # Initial state
        self.history.append(self._count_states())
        for t in range(self.time_steps):
            self.step()
        pass

    def visualize(self):
        """
        Visualize the simulation results using matplotlib.
        """
        s_list = [h['S'] for h in self.history]
        i_list = [h['I'] for h in self.history]
        r_list = [h['R'] for h in self.history]
        plt.figure(figsize=(10,6))
        plt.plot(s_list, label='Susceptible', color='blue')
        plt.plot(i_list, label='Infected', color='red')
        plt.plot(r_list, label='Recovered', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Individuals')
        plt.title('SIR Epidemic Simulation')
        plt.legend()
        plt.tight_layout()
        plt.show()
        pass

    def save_results(self, filename):
        """
        Save the simulation results to a CSV file.
        Args:
            filename (str): Name of the file to save results to.
        """
        results_path = os.path.join(DATA_DIR, filename)
        with open(results_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time_step', 'susceptible', 'infected', 'recovered'])
            for t, h in enumerate(self.history):
                writer.writerow([t, h['S'], h['I'], h['R']])
        pass

def main():
    """
    Main function to initialize, run, visualize, and save the SIR simulation.
    """
    population_size = 1000
    initial_infected = 10
    beta = 0.3
    gamma = 0.1
    time_steps = 100

    sim = SIRSimulation(population_size=population_size,
                        initial_infected=initial_infected,
                        beta=beta,
                        gamma=gamma,
                        time_steps=time_steps)
    sim.run()
    sim.visualize()
    sim.save_results("results.csv")
    pass

# Execute main for both direct execution and sandbox wrapper invocation
main()