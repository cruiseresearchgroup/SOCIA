import os
import random
import csv
import math
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Person:
    """
    Represents an individual in the SEIR/SIR model on a spatial grid or contact network.

    Attributes:
        state (str): The epidemiological state ('S', 'E', 'I', or 'R').
        susceptibility (float): Individual's susceptibility to infection.
        contact_rate (int): Number of contacts made per step.
        position (tuple): (x, y) grid coordinates.
        latent_period (int): Steps remaining in the exposed state ('E'). 0 if not exposed.
    """
    def __init__(self, state='S', susceptibility=1.0, contact_rate=10, position=None, latent_period=0):
        """
        Initializes a Person instance.

        Args:
            state (str): Epidemiological state ('S', 'E', 'I', 'R').
            susceptibility (float): Susceptibility to infection (default 1.0).
            contact_rate (int): Number of contacts made per time step.
            position (tuple): (x, y) position on the grid.
            latent_period (int): Number of steps in the exposed ('E') state.
        """
        self.state = state
        self.susceptibility = susceptibility
        self.contact_rate = contact_rate
        self.position = position
        self.latent_period = latent_period

    def __repr__(self):
        """
        String representation of the Person.

        Returns:
            str: The state and properties of the person.
        """
        return (f"Person(state={self.state}, sus={self.susceptibility:.2f}, "
                f"contact_rate={self.contact_rate}, pos={self.position}, "
                f"latent={self.latent_period})")

class SIRSimulation:
    """
    Simulation class for the (optionally spatial/network) SEIR/SIR epidemic model.

    Attributes:
        population_size (int): Number of individuals.
        beta (float): Transmission probability per contact per step.
        gamma (float): Recovery probability per infected per step.
        sigma (float): Probability of exposed becoming infectious per step (1/latent period).
        initial_infected (int): Number of initially infected.
        initial_exposed (int): Number of initially exposed (for SEIR).
        n_steps (int): Simulation steps.
        k (int): Default contact rate per person per step.
        random_seed (int or None): Random seed.
        grid_size (tuple): (rows, cols) of the spatial grid.
        grid (list): 2D grid of indices mapping to people.
        people (list): List of Person objects.
        history (dict): Time series of S, E, I, R counts.
        metrics (dict): Summary statistics.
    """
    def __init__(self,
                 population_size=120,
                 beta=0.3,
                 gamma=0.1,
                 sigma=0.25,
                 initial_infected=10,
                 initial_exposed=0,
                 n_steps=100,
                 k=10,
                 random_seed=None,
                 grid_shape=None):
        """
        Initializes the simulation.

        Args:
            population_size (int): Number of agents.
            beta (float): Probability of transmission per contact.
            gamma (float): Recovery probability per infected per step.
            sigma (float): Probability of E->I transition per exposed per step (SEIR).
            initial_infected (int): Number of initially infected.
            initial_exposed (int): Number of initially exposed.
            n_steps (int): Number of steps to simulate.
            k (int): Default contact rate per agent.
            random_seed (int or None): Random seed.
            grid_shape (tuple or None): (rows, cols) for the spatial grid. If None, uses square grid.
        """
        self.population_size = population_size
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma  # For SEIR: rate from E->I
        self.initial_infected = initial_infected
        self.initial_exposed = initial_exposed
        self.n_steps = n_steps
        self.k = k
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
        self.grid_size = grid_shape or self._default_grid_shape()
        self.grid = None
        self.people = []
        self.history = {}
        self.metrics = {}
        self._initialize_population()

    def _default_grid_shape(self):
        """
        Suggests a nearly square grid shape to fit the population.

        Returns:
            tuple: (rows, cols)
        """
        side = math.ceil(self.population_size ** 0.5)
        return (side, side)

    def _initialize_population(self):
        """
        Set up the spatial grid, create people with heterogeneity, and assign states.
        """
        rows, cols = self.grid_size
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.people = []
        available_cells = [(i, j) for i in range(rows) for j in range(cols)]
        random.shuffle(available_cells)
        for idx in range(self.population_size):
            pos = available_cells[idx]
            # Heterogeneity: random susceptibility and contact_rate
            susceptibility = random.uniform(0.7, 1.3)
            contact_rate = max(1, int(random.gauss(self.k, 2)))  # Some variation
            p = Person(state='S', susceptibility=susceptibility, contact_rate=contact_rate, position=pos)
            self.people.append(p)
            self.grid[pos[0]][pos[1]] = idx  # Map position to person index
        # Infect a random subset
        infected_indices = random.sample(range(self.population_size), self.initial_infected)
        for idx in infected_indices:
            self.people[idx].state = 'I'
        # Optionally add exposed individuals (SEIR)
        if self.initial_exposed > 0:
            susceptible_indices = [i for i in range(self.population_size) if self.people[i].state == 'S']
            exposed_indices = random.sample(susceptible_indices, min(self.initial_exposed, len(susceptible_indices)))
            for idx in exposed_indices:
                self.people[idx].state = 'E'
                self.people[idx].latent_period = max(1, int(random.gauss(1/self.sigma, 1)))  # Draw latent period

    def _get_neighbor_indices(self, idx):
        """
        Returns indices of spatial neighbors (Moore neighborhood) for the agent at idx.

        Args:
            idx (int): Index of the focal agent.

        Returns:
            list of int: Neighbor indices (excluding self).
        """
        rows, cols = self.grid_size
        x, y = self.people[idx].position
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and self.grid[nx][ny] is not None:
                    nidx = self.grid[nx][ny]
                    neighbors.append(nidx)
        return neighbors

    def get_contacts(self, idx):
        """
        Efficiently selects contacts for agent idx (spatial + global contacts).

        Args:
            idx (int): Index of focal agent.

        Returns:
            list of int: Indices of contacted individuals.
        """
        # FIXED: Use spatial structure—neighbors plus random (for long-range contacts)
        contacts = set()
        p = self.people[idx]
        n_spatial = min(5, p.contact_rate)  # Up to 5 close neighbors
        n_random = max(0, p.contact_rate - n_spatial)
        neighbors = self._get_neighbor_indices(idx)
        if neighbors:
            chosen = random.sample(neighbors, min(n_spatial, len(neighbors)))
            contacts.update(chosen)
        # Add a few random long-range contacts for realism
        population_indices = list(range(self.population_size))
        population_indices.remove(idx)
        if neighbors:
            for n in neighbors:
                if n in population_indices:
                    population_indices.remove(n)
        if population_indices and n_random > 0:
            contacts.update(random.sample(population_indices, min(n_random, len(population_indices))))
        return list(contacts)

    def step(self):
        """
        Advances the simulation by one step: infections, E->I transitions, recoveries.
        """
        # FIXED: Add SEIR logic and heterogeneity in contact selection
        new_infections = set()
        new_exposures = set()
        new_recoveries = set()
        # E->I transitions
        for idx, p in enumerate(self.people):
            if p.state == 'E':
                p.latent_period -= 1
                if p.latent_period <= 0:
                    p.state = 'I'
        # Infection process (I contacts S)
        infected_indices = [i for i, p in enumerate(self.people) if p.state == 'I']
        for inf_idx in infected_indices:
            p_inf = self.people[inf_idx]
            contacts = self.get_contacts(inf_idx)
            for contact_idx in contacts:
                p_contact = self.people[contact_idx]
                if p_contact.state == 'S':
                    # Transmission with agent susceptibility
                    if random.random() < self.beta * p_contact.susceptibility:
                        # If using SEIR, move to E, else directly to I
                        if self.sigma > 0:
                            if p_contact.state == 'S':
                                new_exposures.add(contact_idx)
                        else:
                            new_infections.add(contact_idx)
        # Recovery process
        for inf_idx in infected_indices:
            if random.random() < self.gamma:
                new_recoveries.add(inf_idx)
        # Apply new exposures/infections/recoveries
        for idx in new_exposures:
            if self.people[idx].state == 'S':
                self.people[idx].state = 'E'
                # Assign latent period (heterogeneous)
                self.people[idx].latent_period = max(1, int(random.gauss(1/self.sigma, 1)))
        for idx in new_infections:
            if self.people[idx].state == 'S':
                self.people[idx].state = 'I'
        for idx in new_recoveries:
            if self.people[idx].state == 'I':
                self.people[idx].state = 'R'

    def record(self):
        """
        Records the number of S, E, I, R at current time.

        Returns:
            None
        """
        states = {'S': 0, 'E': 0, 'I': 0, 'R': 0}
        for p in self.people:
            states[p.state] += 1
        for key in states:
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(states[key])

    def calculate_metrics(self):
        """
        Calculates summary statistics and validation metrics for the simulation.

        Metrics:
            - peak_infected: Maximum number of infected at any time.
            - time_to_peak: Step at which peak infected occurs.
            - duration: Number of steps until no infected remain.
            - basic_reproduction_number (approximate): R0 estimate.
        """
        I_hist = self.history.get('I', [])
        E_hist = self.history.get('E', [])
        S_hist = self.history.get('S', [])
        peak_infected = max(I_hist) if I_hist else 0
        time_to_peak = I_hist.index(peak_infected) if peak_infected > 0 else None
        duration = next((i for i, val in enumerate(I_hist) if val == 0 and i > 0), len(I_hist)-1)
        # Estimate R0: average secondary infections per infected (approximate)
        if len(S_hist) > 1 and (self.population_size - S_hist[0]) > 0:
            # Using S0 and S1
            r0_est = (S_hist[0] - S_hist[1]) / (self.population_size - S_hist[0])
        else:
            r0_est = None
        self.metrics = {
            'peak_infected': peak_infected,
            'time_to_peak': time_to_peak,
            'duration': duration,
            'estimated_R0': r0_est
        }

    def run(self):
        """
        Executes the simulation loop for n_steps and computes metrics.
        """
        self.history = {'S': [], 'E': [], 'I': [], 'R': []}
        self.metrics = {}
        self.record()
        for _ in range(self.n_steps):
            self.step()
            self.record()
        self.calculate_metrics()

    def visualize(self):
        """
        Plots the epidemic curves for S, E, I, R over time and prints metrics.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.history['S'], label='Susceptible')
        if any(self.history['E']):
            plt.plot(self.history['E'], label='Exposed')
        plt.plot(self.history['I'], label='Infected')
        plt.plot(self.history['R'], label='Recovered')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Individuals')
        plt.title('SEIR/SIR Model Simulation')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # Print metrics
        if hasattr(self, 'metrics'):
            print('Peak infected:', self.metrics.get('peak_infected', 'n/a'))
            print('Time to peak:', self.metrics.get('time_to_peak', 'n/a'))
            print('Duration:', self.metrics.get('duration', 'n/a'))
            print('Estimated R0:', self.metrics.get('estimated_R0', 'n/a'))

    def save_results(self, filename):
        """
        Saves the simulation history and metrics to a CSV file.

        Args:
            filename (str): The file to save results to.
        """
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['time', 'S', 'E', 'I', 'R']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for t in range(len(self.history['S'])):
                    writer.writerow({
                        'time': t,
                        'S': self.history['S'][t],
                        'E': self.history['E'][t],
                        'I': self.history['I'][t],
                        'R': self.history['R'][t]
                    })
            # Optionally save metrics as a separate file
            metrics_path = os.path.join(DATA_DIR, filename.replace('.csv', '_metrics.csv'))
            with open(metrics_path, 'w', newline='') as mfile:
                writer = csv.writer(mfile)
                writer.writerow(['metric', 'value'])
                for k, v in self.metrics.items():
                    writer.writerow([k, v])
        except Exception as e:
            print(f"Failed to save results: {e}")

def main():
    """
    Demonstrates initialization, execution, visualization, and saving of the SEIR/SIR simulation.
    """
    # Example parameters reflecting feedback: spatial grid, heterogeneity, SEIR, metrics, optimized contacts
    sim = SIRSimulation(
        population_size=120,        # FIXED: Match real-world entity count
        beta=0.3,
        gamma=0.1,
        sigma=0.25,                 # Latent period: 1/sigma steps (SEIR)
        initial_infected=10,
        initial_exposed=5,          # Add some initially exposed if desired
        n_steps=100,
        k=10,                       # Default mean contact rate, individual variation added
        random_seed=42
    )
    sim.run()
    sim.visualize()
    sim.save_results("results.csv")

# Execute main for both direct execution and sandbox wrapper invocation
main()