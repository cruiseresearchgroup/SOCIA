import random
import matplotlib.pyplot as plt
import enum

# Define Health Status states using an Enum for clarity
class HealthStatus(enum.Enum):
    SUSCEPTIBLE = "Susceptible"
    INFECTED = "Infected"
    RECOVERED = "Recovered"

class Person:
    """
    Represents an individual agent in the simulation.
    """
    def __init__(self, id, initial_status):
        """
        Initializes a Person agent.

        Args:
            id (int): Unique identifier for the agent.
            initial_status (HealthStatus): The initial health status of the agent.
        """
        if initial_status not in HealthStatus:
             raise ValueError(f"Invalid initial_status: {initial_status}")

        self.id = id
        self.health_status = initial_status
        self._ever_infected = (initial_status == HealthStatus.INFECTED) # Track for total_infections metric

    def potentially_recover(self, recovery_probability):
        """
        Behavior: An infected individual has a chance to recover.

        Args:
            recovery_probability (float): The probability of recovering in this step.

        Returns:
            bool: True if the agent recovered, False otherwise.
        """
        if self.health_status == HealthStatus.INFECTED:
            if random.random() < recovery_probability:
                self.health_status = HealthStatus.RECOVERED
                return True
        return False

    def attempt_infection_from(self, infected_person, transmission_probability):
        """
        Interaction: A susceptible individual attempts to get infected by an infected person.
        This method is called *on* the susceptible person.

        Args:
            infected_person (Person): The infected person attempting transmission.
            transmission_probability (float): The probability of transmission during this interaction.

        Returns:
            bool: True if the agent became infected, False otherwise.
        """
        if self.health_status == HealthStatus.SUSCEPTIBLE and \
           infected_person.health_status == HealthStatus.INFECTED:
            if random.random() < transmission_probability:
                # Note: State change is typically applied *after* all interactions
                # are evaluated in a step. This method just indicates if infection *would* occur.
                # The actual state change is handled in the simulation loop.
                return True
        return False

    def __repr__(self):
        return f"Person(id={self.id}, status={self.health_status.value})"

class Simulation:
    """
    Manages the simulation environment, agents, parameters, and simulation loop.
    """
    def __init__(self, parameters):
        """
        Initializes the simulation.

        Args:
            parameters (dict): A dictionary containing simulation parameters.
        """
        self.parameters = parameters
        self.population_size = parameters.get("population_size", 1000)
        self.initial_infected_count = parameters.get("initial_infected_count", 1)
        self.transmission_probability = parameters.get("transmission_probability", 0.1)
        self.recovery_probability_per_step = parameters.get("recovery_probability_per_step", 0.05)
        self.simulation_steps = parameters.get("simulation_steps", 100)
        self.random_seed = parameters.get("random_seed", None)

        if self.random_seed is not None:
            random.seed(self.random_seed)

        self.agents = []
        self._initialize_population()

        # Metrics history
        self._metrics_history = {
            "step": [],
            HealthStatus.SUSCEPTIBLE.value: [],
            HealthStatus.INFECTED.value: [],
            HealthStatus.RECOVERED.value: [],
            "Total Infections": []
        }

        # Calculate initial metrics
        self._current_metrics = self._calculate_metrics()
        self._total_infections = self.initial_infected_count # Cumulative count

        self._record_metrics(0) # Record initial state

    def _initialize_population(self):
        """
        Creates the agents and sets their initial health statuses.
        """
        if self.initial_infected_count > self.population_size:
            raise ValueError("Initial infected count cannot exceed population size.")

        # Create all agents as Susceptible initially
        self.agents = [Person(id=i, initial_status=HealthStatus.SUSCEPTIBLE)
                       for i in range(self.population_size)]

        # Randomly select initial infected agents
        infected_indices = random.sample(range(self.population_size), self.initial_infected_count)
        for i in infected_indices:
            self.agents[i].health_status = HealthStatus.INFECTED
            self.agents[i]._ever_infected = True # Mark as ever infected

        print(f"Initialized population: {self.population_size} agents, "
              f"{self.initial_infected_count} initially infected.")

    def _calculate_metrics(self):
        """
        Calculates the current counts of agents in each health status.

        Returns:
            dict: A dictionary with current counts.
        """
        counts = {status: 0 for status in HealthStatus}
        for agent in self.agents:
            counts[agent.health_status] += 1
        return counts

    def _record_metrics(self, step):
        """
        Records the current metrics to the history.

        Args:
            step (int): The current simulation step number.
        """
        current_counts = self._calculate_metrics()
        self._metrics_history["step"].append(step)
        self._metrics_history[HealthStatus.SUSCEPTIBLE.value].append(current_counts[HealthStatus.SUSCEPTIBLE])
        self._metrics_history[HealthStatus.INFECTED.value].append(current_counts[HealthStatus.INFECTED])
        self._metrics_history[HealthStatus.RECOVERED.value].append(current_counts[HealthStatus.RECOVERED])
        self._metrics_history["Total Infections"].append(self._total_infections) # Record cumulative total

        # Basic validation check: Population conservation
        total_agents = sum(current_counts.values())
        if total_agents != self.population_size:
            print(f"WARNING: Population size mismatch at step {step}! Expected {self.population_size}, got {total_agents}")

    def run(self):
        """
        Runs the main simulation loop for the specified number of steps.
        """
        print("Starting simulation...")
        for step in range(1, self.simulation_steps + 1):
            # print(f"Step {step}/{self.simulation_steps}")

            # Lists to track state changes for this step
            agents_to_infect_ids = set()
            agents_to_recover_ids = set()

            # --- Step 2: Interactions (Virus Transmission) ---
            # Perform 'population_size' random pairings
            for _ in range(self.population_size):
                # Select two distinct agents randomly
                if self.population_size < 2:
                    break # Cannot pair if less than 2 agents
                agent1, agent2 = random.sample(self.agents, 2)

                # Check for potential transmission
                if agent1.health_status == HealthStatus.INFECTED and agent2.health_status == HealthStatus.SUSCEPTIBLE:
                    if agent2.attempt_infection_from(agent1, self.transmission_probability):
                        agents_to_infect_ids.add(agent2.id)
                elif agent2.health_status == HealthStatus.INFECTED and agent1.health_status == HealthStatus.SUSCEPTIBLE:
                     if agent1.attempt_infection_from(agent2, self.transmission_probability):
                        agents_to_infect_ids.add(agent1.id)

            # --- Step 3: Behaviors (Potentially Recover) ---
            for agent in self.agents:
                if agent.health_status == HealthStatus.INFECTED:
                    if agent.potentially_recover(self.recovery_probability_per_step):
                        agents_to_recover_ids.add(agent.id)

            # --- Step 4: Apply State Changes ---
            # It's important to iterate through the original agent list
            # and apply changes based on the collected IDs/flags.
            for agent in self.agents:
                # Apply infection first (Susceptible -> Infected)
                if agent.id in agents_to_infect_ids:
                    # Ensure the agent is still Susceptible before infecting
                    if agent.health_status == HealthStatus.SUSCEPTIBLE:
                        agent.health_status = HealthStatus.INFECTED
                        # Increment total infections only when a SUSCEPTIBLE person becomes INFECTED
                        if not agent._ever_infected:
                            self._total_infections += 1
                            agent._ever_infected = True # Mark as ever infected

                # Apply recovery (Infected -> Recovered)
                # Only apply recovery if the agent is currently Infected
                # (Could have just been infected in this step, or was already infected)
                if agent.id in agents_to_recover_ids:
                    if agent.health_status == HealthStatus.INFECTED:
                         agent.health_status = HealthStatus.RECOVERED

                # Validation check: Ensure state transitions are valid
                if agent.health_status == HealthStatus.RECOVERED and agent.id in agents_to_infect_ids:
                     print(f"WARNING: Invalid state transition attempt for agent {agent.id} at step {step}: Recovered -> Infected")
                if agent.health_status == HealthStatus.SUSCEPTIBLE and agent.id in agents_to_recover_ids:
                     print(f"WARNING: Invalid state transition attempt for agent {agent.id} at step {step}: Susceptible -> Recovered")


            # --- Step 5: Update Metrics ---
            self._current_metrics = self._calculate_metrics()
            self._record_metrics(step)

            # Validation check: Infected count limit
            if self._current_metrics[HealthStatus.INFECTED] > self.population_size:
                 print(f"WARNING: Infected count ({self._current_metrics[HealthStatus.INFECTED]}) exceeds population size ({self.population_size}) at step {step}!")


        print("Simulation finished.")
        print("\nFinal Metrics:")
        print(f"Susceptible: {self._current_metrics[HealthStatus.SUSCEPTIBLE]}")
        print(f"Infected: {self._current_metrics[HealthStatus.INFECTED]}")
        print(f"Recovered: {self._current_metrics[HealthStatus.RECOVERED]}")
        print(f"Total Individuals Ever Infected: {self._total_infections}")


    def get_metrics_history(self):
        """
        Returns the collected metrics history.

        Returns:
            dict: A dictionary where keys are metric names and values are lists of values over time.
        """
        return self._metrics_history

def visualize_results(metrics_history):
    """
    Plots the simulation results over time.

    Args:
        metrics_history (dict): The metrics history from the simulation.
    """
    steps = metrics_history["step"]
    susceptible = metrics_history[HealthStatus.SUSCEPTIBLE.value]
    infected = metrics_history[HealthStatus.INFECTED.value]
    recovered = metrics_history[HealthStatus.RECOVERED.value]
    total_infected = metrics_history["Total Infections"]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, susceptible, label=HealthStatus.SUSCEPTIBLE.value, color='blue')
    plt.plot(steps, infected, label=HealthStatus.INFECTED.value, color='red')
    plt.plot(steps, recovered, label=HealthStatus.RECOVERED.value, color='green')
    # plt.plot(steps, total_infected, label="Total Ever Infected", color='purple', linestyle='--') # Optional: plot cumulative

    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Individuals")
    plt.title("Simple Epidemic Spread Over Time (SIR Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Sets up parameters, runs the simulation, and visualizes results.
    """
    simulation_parameters = {
        "population_size": 1000,
        "initial_infected_count": 1,
        "transmission_probability": 0.1,
        "recovery_probability_per_step": 0.05,
        "simulation_steps": 200, # Increased steps to see more dynamics
        "random_seed": 42 # Use a seed for reproducibility
    }

    # Basic parameter validation
    if simulation_parameters["initial_infected_count"] > simulation_parameters["population_size"]:
        print("Error: initial_infected_count cannot be greater than population_size.")
        return
    if not (0 <= simulation_parameters["transmission_probability"] <= 1):
         print("Error: transmission_probability must be between 0 and 1.")
         return
    if not (0 <= simulation_parameters["recovery_probability_per_step"] <= 1):
         print("Error: recovery_probability_per_step must be between 0 and 1.")
         return
    if simulation_parameters["simulation_steps"] < 0:
         print("Error: simulation_steps cannot be negative.")
         return


    sim = Simulation(simulation_parameters)
    sim.run()
    metrics = sim.get_metrics_history()
    visualize_results(metrics)

if __name__ == "__main__":
    main()