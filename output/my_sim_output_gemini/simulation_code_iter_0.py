import random
import enum
import collections
import matplotlib.pyplot as plt
import math

# --- config.py ---

class SimulationConfig:
    """
    Defines parameters for the Simple Epidemic Simulation.
    """
    def __init__(self,
                 population_size: int = 1000,
                 initial_infected_count: int = 1,
                 transmission_probability: float = 0.05,
                 infection_duration_steps: int = 10,
                 contact_rate_per_person_per_step: float = 5.0,
                 simulation_steps: int = 100,
                 random_seed: int = None):

        if not isinstance(population_size, int) or population_size <= 0:
            raise ValueError("population_size must be a positive integer.")
        if not isinstance(initial_infected_count, int) or initial_infected_count < 0:
            raise ValueError("initial_infected_count must be a non-negative integer.")
        if initial_infected_count > population_size:
             raise ValueError("initial_infected_count cannot exceed population_size.")
        if not isinstance(transmission_probability, (int, float)) or not (0.0 <= transmission_probability <= 1.0):
            raise ValueError("transmission_probability must be a float between 0.0 and 1.0.")
        if not isinstance(infection_duration_steps, int) or infection_duration_steps <= 0:
            raise ValueError("infection_duration_steps must be a positive integer.")
        if not isinstance(contact_rate_per_person_per_step, (int, float)) or contact_rate_per_person_per_step < 0:
             raise ValueError("contact_rate_per_person_per_step must be a non-negative number.")
        if not isinstance(simulation_steps, int) or simulation_steps <= 0:
            raise ValueError("simulation_steps must be a positive integer.")
        if random_seed is not None and not isinstance(random_seed, int):
             raise ValueError("random_seed must be an integer or None.")

        self.population_size = population_size
        self.initial_infected_count = initial_infected_count
        self.transmission_probability = transmission_probability
        self.infection_duration_steps = infection_duration_steps
        self.contact_rate_per_person_per_step = contact_rate_per_person_per_step
        self.simulation_steps = simulation_steps
        self.random_seed = random_seed

    def __str__(self):
        return f"SimulationConfig(\n" \
               f"  population_size={self.population_size},\n" \
               f"  initial_infected_count={self.initial_infected_count},\n" \
               f"  transmission_probability={self.transmission_probability},\n" \
               f"  infection_duration_steps={self.infection_duration_steps},\n" \
               f"  contact_rate_per_person_per_step={self.contact_rate_per_person_per_step},\n" \
               f"  simulation_steps={self.simulation_steps},\n" \
               f"  random_seed={self.random_seed}\n" \
               f")"

# --- agent.py ---

class HealthState(enum.Enum):
    """
    Represents the possible health states of a Person agent.
    """
    SUSCEPTIBLE = "Susceptible"
    INFECTED = "Infected"
    RECOVERED = "Recovered"

class Person:
    """
    Represents an individual agent in the epidemic simulation.
    """
    def __init__(self, agent_id: int, initial_state: HealthState, current_step: int):
        """
        Initializes a Person agent.

        Args:
            agent_id: A unique identifier for the agent.
            initial_state: The starting HealthState of the agent.
            current_step: The current simulation step (used to record infection time).
        """
        self.id = agent_id
        self.health_state = initial_state
        self.infection_start_step = current_step if initial_state == HealthState.INFECTED else None
        self._marked_for_infection = False # Temporary flag set during interactions

    def __repr__(self):
        return f"Person(id={self.id}, state={self.health_state.value}, infected_at={self.infection_start_step})"

    def step(self, current_step: int, environment):
        """
        Executes the agent's behaviors for a single simulation step.

        Args:
            current_step: The current simulation step number.
            environment: The simulation environment containing other agents and parameters.
        """
        # Reset temporary flags at the start of the step processing
        self._marked_for_infection = False

        # Behavior: interact_with_others
        self._interact_with_others(environment)

        # Behavior: transition_state_based_on_rules
        # This needs to happen AFTER all potential interactions have marked agents
        # within this step. The actual state transition logic is separate to ensure
        # all interactions for the step are considered before anyone changes state.
        # The transition logic is called by the environment/simulation runner
        # after all agents have performed their interactions.

    def _interact_with_others(self, environment):
        """
        Attempts to make contacts with other agents in the environment.

        Args:
            environment: The simulation environment.
        """
        if self.health_state == HealthState.RECOVERED:
            # Recovered individuals do not interact in terms of spreading/getting infected
            # (though they might still have contacts, those contacts have no effect)
            return

        # Determine number of contacts to attempt
        num_contacts = round(environment.params.contact_rate_per_person_per_step)

        if environment.params.population_size <= 1:
             # Cannot interact with others if population is 1 or less
             return

        # List of other agents
        other_agents = [agent for agent in environment.agents if agent.id != self.id]

        for _ in range(num_contacts):
            # Select a random agent to interact with (excluding self)
            if not other_agents: # Should not happen if population_size > 1, but good check
                break
            other_agent = environment.rng.choice(other_agents)

            # Interaction: potential_infection_contact
            Person.potential_infection_contact(self, other_agent, environment.params.transmission_probability, environment.rng)

    @staticmethod
    def potential_infection_contact(agent1, agent2, transmission_probability: float, rng):
        """
        Handles a potential infection contact between two agents.
        Marks a susceptible agent for infection if contact conditions are met.

        Args:
            agent1: The first agent involved in the contact.
            agent2: The second agent involved in the contact.
            transmission_probability: The probability of transmission.
            rng: The random number generator instance.
        """
        susceptible_agent = None
        infected_agent = None

        if agent1.health_state == HealthState.SUSCEPTIBLE and agent2.health_state == HealthState.INFECTED:
            susceptible_agent = agent1
            infected_agent = agent2
        elif agent1.health_state == HealthState.INFECTED and agent2.health_state == HealthState.SUSCEPTIBLE:
            susceptible_agent = agent2
            infected_agent = agent1

        # If a susceptible agent and an infected agent are involved
        if susceptible_agent and infected_agent:
            # Check for transmission
            if rng.random() < transmission_probability:
                # Mark the susceptible agent for infection. The state change
                # happens later in the transition_state method.
                susceptible_agent._marked_for_infection = True

    def transition_state(self, current_step: int, infection_duration_steps: int) -> bool:
        """
        Updates the agent's health state based on rules and current conditions.

        Args:
            current_step: The current simulation step number.
            infection_duration_steps: The number of steps an agent stays infected.

        Returns:
            True if the agent transitioned from Susceptible to Infected in this step,
            False otherwise. (Used for counting new cases).
        """
        transitioned_to_infected = False

        if self.health_state == HealthState.SUSCEPTIBLE:
            # If marked for infection during interactions in this step
            if self._marked_for_infection:
                self.health_state = HealthState.INFECTED
                self.infection_start_step = current_step
                transitioned_to_infected = True
        elif self.health_state == HealthState.INFECTED:
            # Check if it's time to recover
            if current_step - self.infection_start_step >= infection_duration_steps:
                self.health_state = HealthState.RECOVERED
                self.infection_start_step = None # Recovered agents don't have a start step
        # Recovered agents remain Recovered

        # Reset the flag after processing transitions
        self._marked_for_infection = False # Ensure it's reset whether transition happened or not

        return transitioned_to_infected


# --- metrics.py ---

class SimulationMetrics:
    """
    Collects and stores simulation metrics over time.
    """
    def __init__(self, initial_infected_count: int):
        """
        Initializes the metrics storage.

        Args:
            initial_infected_count: The number of agents initially infected.
        """
        self.steps = []
        self.susceptible_counts = []
        self.infected_counts = []
        self.recovered_counts = []
        self.total_cases = [initial_infected_count] # Cumulative count

    def collect(self, current_step: int, agents: list, new_infections_in_step: int):
        """
        Collects the current state counts and updates metrics.

        Args:
            current_step: The current simulation step.
            agents: The list of all agents in the simulation.
            new_infections_in_step: The number of agents who transitioned from S to I
                                    in the current step.
        """
        self.steps.append(current_step)

        susceptible_count = sum(1 for agent in agents if agent.health_state == HealthState.SUSCEPTIBLE)
        infected_count = sum(1 for agent in agents if agent.health_state == HealthState.INFECTED)
        recovered_count = sum(1 for agent in agents if agent.health_state == HealthState.RECOVERED)

        self.susceptible_counts.append(susceptible_count)
        self.infected_counts.append(infected_count)
        self.recovered_counts.append(recovered_count)

        # Update total cases (cumulative S->I transitions)
        # The initial total cases is the initial infected count (step 0).
        # For steps > 0, add the new infections from the *previous* step's transitions
        # (since transitions happen *after* interactions). Or, more simply,
        # the total cases at step T is the sum of initial infected + all S->I transitions up to step T.
        # The `new_infections_in_step` correctly represents S->I transitions that *just* happened.
        current_total_cases = self.total_cases[-1] + new_infections_in_step
        self.total_cases.append(current_total_cases)

        # Optional: Add validation checks here during collection
        total_population = susceptible_count + infected_count + recovered_count
        if total_population != len(agents):
             print(f"Warning: Population conservation violated at step {current_step}! "
                   f"Current sum: {total_population}, Expected: {len(agents)}")


# --- environment.py ---

class Environment:
    """
    Manages the simulation environment and the collection of agents.
    Orchestrates the simulation steps.
    """
    def __init__(self, params: SimulationConfig):
        """
        Initializes the environment.

        Args:
            params: The simulation configuration parameters.
        """
        self.params = params
        self.agents = []
        self.current_step = 0
        self.rng = random.Random(params.random_seed) # Use a dedicated RNG for reproducibility

    def setup(self):
        """
        Initializes the agents based on the configuration parameters.
        """
        self.agents = []
        initial_infected_count = self.params.initial_infected_count
        population_size = self.params.population_size

        if initial_infected_count > population_size:
             raise ValueError("Initial infected count cannot be greater than population size.")

        # Create agents and assign initial states
        # Ensure unique IDs
        agent_ids = list(range(population_size))
        self.rng.shuffle(agent_ids) # Shuffle IDs just in case order matters later

        for i in range(population_size):
            agent_id = i # Use simple sequential ID for now
            if i < initial_infected_count:
                # Initially infected agents start at step 0
                agent = Person(agent_id, HealthState.INFECTED, self.current_step)
            else:
                agent = Person(agent_id, HealthState.SUSCEPTIBLE, self.current_step)
            self.agents.append(agent)

        # Optional: Shuffle agents list to remove any initial state ordering bias
        self.rng.shuffle(self.agents)


    def step(self) -> int:
        """
        Executes a single simulation step.

        Returns:
            The number of new infections (S->I transitions) that occurred in this step.
        """
        self.current_step += 1
        new_infections_this_step = 0

        # Phase 1: Agents perform interactions (which *mark* susceptible agents)
        # Shuffle agents each step to randomize interaction order
        self.rng.shuffle(self.agents)
        for agent in self.agents:
            agent.step(self.current_step, self) # agent._interact_with_others is called here

        # Phase 2: Agents transition states based on their state and marks from interactions
        # We need to iterate through agents again to process transitions after all
        # interactions for the step are complete.
        for agent in self.agents:
            # agent.transition_state is called here
            if agent.transition_state(self.current_step, self.params.infection_duration_steps):
                 new_infections_this_step += 1

        return new_infections_this_step

# --- visualization.py ---

def plot_sir_curves(metrics: SimulationMetrics, population_size: int):
    """
    Plots the Susceptible, Infected, and Recovered counts over time.

    Args:
        metrics: The SimulationMetrics object containing the collected data.
        population_size: The total population size for normalization.
    """
    if not metrics.steps:
        print("No metrics collected to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Plot counts
    plt.plot(metrics.steps, metrics.susceptible_counts, label=HealthState.SUSCEPTIBLE.value, color='blue')
    plt.plot(metrics.steps, metrics.infected_counts, label=HealthState.INFECTED.value, color='red')
    plt.plot(metrics.steps, metrics.recovered_counts, label=HealthState.RECOVERED.value, color='green')

    # Plot total cases (optional, can be on a secondary axis or separate plot)
    # plt.plot(metrics.steps, metrics.total_cases, label='Total Cases (Cumulative)', color='purple', linestyle='--')


    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Individuals")
    plt.title("Simple SIR Epidemic Simulation")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, population_size + 10) # Ensure y-axis starts at 0 and goes slightly above pop size
    plt.xlim(0, metrics.steps[-1]) # Ensure x-axis starts at 0

    plt.show()

def plot_total_cases(metrics: SimulationMetrics):
    """
    Plots the cumulative total cases over time.

    Args:
        metrics: The SimulationMetrics object containing the collected data.
    """
    if not metrics.steps:
        print("No metrics collected to plot.")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(metrics.steps, metrics.total_cases, label='Total Cases (Cumulative)', color='purple')

    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Individuals")
    plt.title("Simple SIR Epidemic Simulation: Total Cases")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(metrics.total_cases) * 1.1 if metrics.total_cases else 10)
    plt.xlim(0, metrics.steps[-1])

    plt.show()


# --- simulation.py ---

def run_simulation(params: SimulationConfig) -> SimulationMetrics:
    """
    Sets up and runs the epidemic simulation.

    Args:
        params: The simulation configuration parameters.

    Returns:
        A SimulationMetrics object containing the results.
    """
    print("Setting up simulation with parameters:")
    print(params)

    environment = Environment(params)
    environment.setup()

    # Initial metric collection
    initial_infected_count = sum(1 for agent in environment.agents if agent.health_state == HealthState.INFECTED)
    metrics = SimulationMetrics(initial_infected_count=initial_infected_count)
    metrics.collect(environment.current_step, environment.agents, new_infections_in_step=0) # Collect initial state at step 0

    print(f"Running simulation for {params.simulation_steps} steps...")
    for step in range(1, params.simulation_steps + 1):
        if step % 10 == 0 or step == 1 or step == params.simulation_steps:
            print(f"  Step {step}/{params.simulation_steps}")

        new_infections = environment.step()
        metrics.collect(environment.current_step, environment.agents, new_infections)

    print("Simulation finished.")
    return metrics

def main():
    """
    Main function to run the simulation and display results.
    """
    # Define simulation parameters
    # You can change these values to explore different scenarios
    config = SimulationConfig(
        population_size=1000,
        initial_infected_count=5,
        transmission_probability=0.08,
        infection_duration_steps=14,
        contact_rate_per_person_per_step=10.0,
        simulation_steps=200,
        random_seed=42 # Set to None for a different run each time
    )

    # Run the simulation
    simulation_results = run_simulation(config)

    # Display results
    plot_sir_curves(simulation_results, config.population_size)
    plot_total_cases(simulation_results)

    # Optional: Print final state counts
    print("\nFinal State Counts:")
    final_s = simulation_results.susceptible_counts[-1]
    final_i = simulation_results.infected_counts[-1]
    final_r = simulation_results.recovered_counts[-1]
    final_total = final_s + final_i + final_r
    print(f"  Susceptible: {final_s}")
    print(f"  Infected: {final_i}")
    print(f"  Recovered: {final_r}")
    print(f"  Total Population: {final_total} (Expected: {config.population_size})")
    print(f"  Total Cases (Cumulative): {simulation_results.total_cases[-1]}")

    # Basic Validation Check: Population Conservation
    if final_total == config.population_size:
        print("Validation Check: Population Conservation - PASSED")
    else:
        print("Validation Check: Population Conservation - FAILED")

if __name__ == "__main__":
    main()