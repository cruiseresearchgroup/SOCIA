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
    Represents an individual in the spatial SIR model with heterogeneity.
    Attributes:
        state (str): The epidemiological state ('S', 'I', or 'R').
        x (int): X-coordinate on the spatial grid.
        y (int): Y-coordinate on the spatial grid.
        susceptibility (float): Individual susceptibility multiplier (default 1.0).
        infectivity (float): Individual infectivity multiplier (default 1.0).
        infected_time (int or None): Number of steps since infection (None if not infected).
        infectious_period (int or None): How many steps this agent will remain infectious.
    """
    def __init__(self, state='S', x=None, y=None, susceptibility=1.0, infectivity=1.0, infectious_period=None):
        """
        Initialize a Person with a given state and properties.
        Args:
            state (str): Initial state ('S', 'I', or 'R').
            x (int): X-coordinate.
            y (int): Y-coordinate.
            susceptibility (float): Susceptibility multiplier.
            infectivity (float): Infectivity multiplier.
            infectious_period (int or None): Steps agent will remain infectious.
        """
        self.state = state
        self.x = x
        self.y = y
        self.susceptibility = susceptibility
        self.infectivity = infectivity
        self.infected_time = 0 if state == 'I' else None
        self.infectious_period = infectious_period if state == 'I' else None


class SIRSimulation:
    """
    Simulates the spread of an epidemic using a spatial, heterogeneous SIR model.
    Attributes:
        population_size (int): Number of individuals in the simulation.
        beta (float): Baseline probability of infection per neighbor per time step.
        gamma (float): Probability of recovery per infected per time step (if not using fixed period).
        initial_infected (int): Initial number of infected individuals.
        grid_size (int): Side length of the square grid.
        population (list): List of Person instances.
        grid (list of lists): 2D grid mapping positions to Person indices.
        steps (int): Number of simulation steps.
        history (list): Time series of (S, I, R, R_t, spatial_cluster_metric) tuples.
        fixed_infectious_period (int or None): If set, infectious period is fixed, otherwise Markovian recovery.
    """
    def __init__(self, population_size=120, beta=0.3, gamma=0.1, initial_infected=10, steps=100,
                 grid_size=None, fixed_infectious_period=8, susceptibility_mu=1.0, susceptibility_sigma=0.15,
                 infectivity_mu=1.0, infectivity_sigma=0.15, seed=42):
        """
        Initialize the SIR simulation with parameters.
        Args:
            population_size (int): Size of the population.
            beta (float): Baseline infection rate.
            gamma (float): Recovery rate (if not using fixed infectious period).
            initial_infected (int): Number of initially infected individuals.
            steps (int): Number of simulation steps.
            grid_size (int or None): Size of grid to use; if None, determined automatically.
            fixed_infectious_period (int or None): If set, agents have fixed infectious duration.
            susceptibility_mu (float): Mean susceptibility.
            susceptibility_sigma (float): Stddev of susceptibility.
            infectivity_mu (float): Mean infectivity.
            infectivity_sigma (float): Stddev of infectivity.
            seed (int): Random seed for reproducibility.
        """
        random.seed(seed)
        self.population_size = population_size
        self.beta = beta
        self.gamma = gamma
        self.initial_infected = initial_infected
        self.steps = steps
        self.fixed_infectious_period = fixed_infectious_period
        if grid_size is None:
            self.grid_size = math.ceil(math.sqrt(population_size))
        else:
            self.grid_size = grid_size
        self.population = []
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.history = []
        self.susceptibility_mu = susceptibility_mu
        self.susceptibility_sigma = susceptibility_sigma
        self.infectivity_mu = infectivity_mu
        self.infectivity_sigma = infectivity_sigma
        self._initialize_population()

    def _initialize_population(self):
        """
        Set up the initial population with spatial clustering, heterogeneity, and assignment to grid positions.
        """
        # To create clustering: assign agents to grid positions in blocks
        cluster_size = max(2, self.grid_size // 6)
        clusters_per_side = self.grid_size // cluster_size
        cluster_centers = []
        for i in range(clusters_per_side):
            for j in range(clusters_per_side):
                center_x = i * cluster_size + cluster_size // 2
                center_y = j * cluster_size + cluster_size // 2
                cluster_centers.append((center_x, center_y))

        assigned = 0
        person_positions = []
        while assigned < self.population_size:
            # Pick a cluster center, then random offset within cluster
            cluster = random.choice(cluster_centers)
            offset_x = random.randint(-cluster_size//2, cluster_size//2)
            offset_y = random.randint(-cluster_size//2, cluster_size//2)
            x = min(max(cluster[0] + offset_x, 0), self.grid_size-1)
            y = min(max(cluster[1] + offset_y, 0), self.grid_size-1)
            if self.grid[x][y] is None:
                person_positions.append((x, y))
                assigned += 1
                self.grid[x][y] = -1  # placeholder

        # Assign heterogeneity
        suscepts = [max(0.1, random.gauss(self.susceptibility_mu, self.susceptibility_sigma)) for _ in range(self.population_size)]
        infectivs = [max(0.1, random.gauss(self.infectivity_mu, self.infectivity_sigma)) for _ in range(self.population_size)]

        # Sample infectious periods if needed
        if self.fixed_infectious_period is not None:
            infectious_periods = [self.fixed_infectious_period for _ in range(self.population_size)]
        else:
            infectious_periods = [None for _ in range(self.population_size)]

        # Assign states
        infected_indices = set(random.sample(range(self.population_size), min(self.initial_infected, self.population_size)))
        self.population = []
        for idx in range(self.population_size):
            state = 'I' if idx in infected_indices else 'S'
            infectious_period = infectious_periods[idx]
            person = Person(
                state=state,
                x=person_positions[idx][0],
                y=person_positions[idx][1],
                susceptibility=suscepts[idx],
                infectivity=infectivs[idx],
                infectious_period=infectious_period
            )
            self.population.append(person)

        # Place references in grid
        for idx, (x, y) in enumerate(person_positions):
            self.grid[x][y] = idx

    def get_neighbors(self, idx):
        """
        Get Person objects representing 8 neighbors of the agent at index idx (Moore neighborhood, toroidal grid).
        Args:
            idx (int): Index into self.population.
        Returns:
            list of Person: Neighboring agents.
        """
        person = self.population[idx]
        neighbors = []
        deltas = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in deltas:
            nx = (person.x + dx) % self.grid_size
            ny = (person.y + dy) % self.grid_size
            neighbor_idx = self.grid[nx][ny]
            if neighbor_idx is not None and neighbor_idx != idx:
                neighbors.append(self.population[neighbor_idx])
        return neighbors

    def step(self):
        """
        Perform one time step of the simulation with spatial and heterogeneous interactions.
        Implements infection transmission among neighbors and recovery process.
        """
        new_states = [p.state for p in self.population]
        new_infectious_periods = [p.infectious_period for p in self.population]
        new_infected_time = [p.infected_time for p in self.population]

        for idx, person in enumerate(self.population):
            if person.state == 'S':
                neighbors = self.get_neighbors(idx)
                infected_neighbors = [n for n in neighbors if n.state == 'I']
                p_infection = 1.0
                for n in infected_neighbors:
                    # Each neighbor tries to infect, combined probability
                    # (1 - p_total) = product_{i} (1 - beta * susceptibility * infectivity)
                    p_contact = self.beta * person.susceptibility * n.infectivity
                    p_infection *= (1 - min(p_contact, 1.0))
                p_infection = 1 - p_infection  # Total probability at least one infection
                if random.random() < p_infection:
                    new_states[idx] = 'I'
                    if self.fixed_infectious_period is not None:
                        new_infectious_periods[idx] = self.fixed_infectious_period
                    new_infected_time[idx] = 0
            elif person.state == 'I':
                # Non-Markovian: infected for a fixed period
                if self.fixed_infectious_period is not None:
                    # Increase infected_time counter
                    infected_time = person.infected_time
                    if infected_time is None:
                        infected_time = 0
                    infected_time += 1
                    new_infected_time[idx] = infected_time
                    if infected_time >= person.infectious_period:
                        new_states[idx] = 'R'
                        new_infectious_periods[idx] = None
                        new_infected_time[idx] = None
                else:
                    # Markovian: recover with prob gamma per step
                    if random.random() < self.gamma:
                        new_states[idx] = 'R'
                        new_infectious_periods[idx] = None
                        new_infected_time[idx] = None

        # Update states
        for idx, person in enumerate(self.population):
            person.state = new_states[idx]
            person.infectious_period = new_infectious_periods[idx]
            person.infected_time = new_infected_time[idx]

    def record(self):
        """
        Record the current counts of S, I, R, compute R_t and spatial statistics.
        """
        count_S = sum(1 for p in self.population if p.state == 'S')
        count_I = sum(1 for p in self.population if p.state == 'I')
        count_R = sum(1 for p in self.population if p.state == 'R')

        # Estimate effective reproductive number R_t
        prev_I = None
        if self.history:
            prev_I = self.history[-1][1]
        else:
            prev_I = count_I
        if prev_I > 0:
            R_t = count_I / prev_I
        else:
            R_t = 0

        # Compute spatial clustering metric: average infected neighbor count for infected agents
        infected_idxs = [idx for idx, p in enumerate(self.population) if p.state == 'I']
        avg_infected_neighbors = 0
        if infected_idxs:
            total = 0
            for idx in infected_idxs:
                neighbors = self.get_neighbors(idx)
                infected_neighbors = sum(1 for n in neighbors if n.state == 'I')
                total += infected_neighbors
            avg_infected_neighbors = total / len(infected_idxs)
        self.history.append((count_S, count_I, count_R, R_t, avg_infected_neighbors))

    def run(self):
        """
        Run the simulation for the specified number of steps.
        """
        self.history = []
        self.record()
        for _ in range(self.steps):
            self.step()
            self.record()

    def visualize(self):
        """
        Plot the SIR curves and metrics over time.
        """
        if not self.history:
            print("No data to visualize.")
            return

        S, I, R, R_t, cluster_metric = zip(*self.history)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(S, label='Susceptible')
        axs[0].plot(I, label='Infected')
        axs[0].plot(R, label='Recovered')
        axs[0].set_ylabel("Number of Individuals")
        axs[0].set_title("SIR Epidemic Simulation with Spatial Clustering")
        axs[0].legend()

        axs[1].plot(R_t, label='Estimated $R_t$')
        axs[1].plot(cluster_metric, label='Avg. Infected Neighbors (I)')
        axs[1].set_xlabel("Time Steps")
        axs[1].set_ylabel("Metric Value")
        axs[1].set_title("Validation Metrics")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def save_results(self, filename):
        """
        Save the results of the simulation to a CSV file.
        Args:
            filename (str): The filename to save the results to.
        """
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Susceptible', 'Infected', 'Recovered', 'R_t', 'Avg_Infected_Neighbors'])
            for i, (s, inf, r, rt, cmet) in enumerate(self.history):
                writer.writerow([i, s, inf, r, rt, cmet])
        print(f"Results saved to {file_path}")

    def save_agent_attributes(self, filename="agent_attributes.csv"):
        """
        Save the agent attributes (heterogeneity and positions) for analysis.
        Args:
            filename (str): The filename to save agent attribute data.
        """
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'InitialState', 'X', 'Y', 'Susceptibility', 'Infectivity'])
            for idx, person in enumerate(self.population):
                writer.writerow([
                    idx,
                    person.state,
                    person.x,
                    person.y,
                    person.susceptibility,
                    person.infectivity
                ])
        print(f"Agent attributes saved to {file_path}")

    def plot_grid(self, step_idx=None):
        """
        Visualize the spatial grid at a specific step (by default, final).
        Args:
            step_idx (int or None): Step to visualize (None = final).
        """
        plt.figure(figsize=(7, 7))
        color_map = {'S': 'blue', 'I': 'red', 'R': 'green'}
        for person in self.population:
            plt.scatter(person.x, person.y, color=color_map.get(person.state, 'gray'), alpha=0.7, s=24)
        plt.title("Agent States on Grid" + (f" (step {step_idx})" if step_idx is not None else " (final)"))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-1, self.grid_size)
        plt.ylim(-1, self.grid_size)
        plt.grid(True, alpha=0.2)
        plt.show()


def main():
    """
    Main function to run the enhanced spatial SIR epidemic simulation.
    Demonstrates initialization, execution, visualization, and result saving.
    """
    # Simulation parameters
    population_size = 120  # Match real-world entity count # FIXED: Parameterized population size
    beta = 0.23
    gamma = 0.1
    initial_infected = 8
    steps = 80
    fixed_infectious_period = 8  # Use fixed infectious period (non-Markovian) # FIXED: Advanced temporal modeling

    # Initialize simulation
    sim = SIRSimulation(
        population_size=population_size,
        beta=beta,
        gamma=gamma,
        initial_infected=initial_infected,
        steps=steps,
        fixed_infectious_period=fixed_infectious_period
    )

    # Run simulation
    sim.run()

    # Visualize results
    sim.visualize()
    sim.plot_grid()

    # Save results
    sim.save_results("results.csv")
    sim.save_agent_attributes("agent_attributes.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()