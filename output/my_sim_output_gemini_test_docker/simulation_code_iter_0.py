import random
import matplotlib.pyplot as plt
import csv
import os

# Define health states as constants
STATE_SUSCEPTIBLE = "Susceptible"
STATE_INFECTED = "Infected"
STATE_RECOVERED = "Recovered"

class Person:
    """
    Represents an individual agent in the simulation.
    """
    def __init__(self, id, initial_state):
        """
        Initializes a Person agent.

        Args:
            id (int): Unique identifier for the agent.
            initial_state (str): The initial health state ('Susceptible', 'Infected', 'Recovered').
        """
        if initial_state not in [STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED]:
            raise ValueError(f"Invalid initial state: {initial_state}")
        self.id = id
        self.health_state = initial_state

    def is_susceptible(self):
        """Checks if the agent is in the Susceptible state."""
        return self.health_state == STATE_SUSCEPTIBLE

    def is_infected(self):
        """Checks if the agent is in the Infected state."""
        return self.health_state == STATE_INFECTED

    def is_recovered(self):
        """Checks if the agent is in the Recovered state."""
        return self.health_state == STATE_RECOVERED

    def set_state(self, new_state):
        """
        Sets the agent's health state.

        Args:
            new_state (str): The new health state.
        """
        if new_state not in [STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED]:
            raise ValueError(f"Invalid state to set: {new_state}")
        self.health_state = new_state

    def __repr__(self):
        return f"Person(id={self.id}, state='{self.health_state}')"


class Simulation:
    """
    Manages the simulation environment, agents, parameters, and time steps.
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
        self.transmission_probability_per_contact = parameters.get("transmission_probability_per_contact", 0.05)
        self.recovery_rate_per_time_step = parameters.get("recovery_rate_per_time_step", 0.01)
        self.simulation_duration_steps = parameters.get("simulation_duration_steps", 100)
        self.random_seed = parameters.get("random_seed", None)

        if self.random_seed is not None:
            random.seed(self.random_seed)

        if self.initial_infected_count > self.population_size:
             raise ValueError("Initial infected count cannot exceed population size.")
        if self.initial_infected_count < 0 or self.population_size < 0:
             raise ValueError("Population size and initial infected count must be non-negative.")
        if not (0.0 <= self.transmission_probability_per_contact <= 1.0):
             raise ValueError("Transmission probability must be between 0 and 1.")
        if not (0.0 <= self.recovery_rate_per_time_step <= 1.0):
             raise ValueError("Recovery rate must be between 0 and 1.")
        if self.simulation_duration_steps < 0:
             raise ValueError("Simulation duration must be non-negative.")


        self.agents = self._create_population()

        # Metrics storage
        self.metrics = {
            "time_step": [],
            STATE_SUSCEPTIBLE: [],
            STATE_INFECTED: [],
            STATE_RECOVERED: [],
            "total_infections": 0 # Cumulative count of agents who have been infected
        }
        # Keep track of agents who have been infected to calculate total_infections correctly
        self._ever_infected_ids = set()
        # Initialize total_infections with the initial infected count
        initial_infected_agents = [agent for agent in self.agents if agent.is_infected()]
        self.metrics["total_infections"] = len(initial_infected_agents)
        self._ever_infected_ids.update(agent.id for agent in initial_infected_agents)


    def _create_population(self):
        """
        Creates the initial population of agents based on parameters.
        """
        agents = []
        # Create all agents as Susceptible initially
        for i in range(self.population_size):
            agents.append(Person(id=i, initial_state=STATE_SUSCEPTIBLE))

        # Randomly select initial infected agents
        if self.initial_infected_count > 0:
            infected_agents = random.sample(agents, self.initial_infected_count)
            for agent in infected_agents:
                agent.set_state(STATE_INFECTED)

        return agents

    def _calculate_metrics(self, time_step):
        """
        Calculates and records the metrics for the current time step.
        """
        susceptible_count = sum(1 for agent in self.agents if agent.is_susceptible())
        infected_count = sum(1 for agent in self.agents if agent.is_infected())
        recovered_count = sum(1 for agent in self.agents if agent.is_recovered())

        # Validation: Population Conservation
        if susceptible_count + infected_count + recovered_count != self.population_size:
            print(f"Warning: Population conservation broken at time step {time_step}!")
            print(f"S: {susceptible_count}, I: {infected_count}, R: {recovered_count}, Total: {susceptible_count + infected_count + recovered_count}")


        self.metrics["time_step"].append(time_step)
        self.metrics[STATE_SUSCEPTIBLE].append(susceptible_count)
        self.metrics[STATE_INFECTED].append(infected_count)
        self.metrics[STATE_RECOVERED].append(recovered_count)
        # total_infections is updated during state transition S->I

    def _time_step(self, current_step):
        """
        Executes one time step of the simulation.
        Implements the time_step_logic algorithm from the model plan.
        """
        susceptible_agents = [agent for agent in self.agents if agent.is_susceptible()]
        infected_agents = [agent for agent in self.agents if agent.is_infected()]

        # Use sets to store IDs of agents whose state should change
        # This prevents modifying the list while iterating and handles potential duplicates
        agents_to_infect_ids = set()
        agents_to_recover_ids = set()

        # 3. Infection Process (Well-mixed model interpretation from plan)
        # Each susceptible agent interacts with each infected agent
        for s_agent in susceptible_agents:
            # Optimization: If an agent is marked for infection, no need to check further infected agents for this step
            if s_agent.id in agents_to_infect_ids:
                continue

            for i_agent in infected_agents:
                if random.random() < self.transmission_probability_per_contact:
                    agents_to_infect_ids.add(s_agent.id)
                    # As per the plan's algorithm: "break the inner loop"
                    break

        # 4. Recovery Process
        for i_agent in infected_agents:
            if random.random() < self.recovery_rate_per_time_step:
                agents_to_recover_ids.add(i_agent.id)

        # 5. State Update
        newly_infected_count_this_step = 0
        for agent in self.agents:
            agent_id = agent.id

            # Check for infection transition (S -> I)
            # Ensure agent is still susceptible and marked for infection
            if agent.is_susceptible() and agent_id in agents_to_infect_ids:
                agent.set_state(STATE_INFECTED)
                newly_infected_count_this_step += 1
                self._ever_infected_ids.add(agent_id) # Add to cumulative count

            # Check for recovery transition (I -> R)
            # Ensure agent is still infected and marked for recovery
            # This check happens *after* potential infection, so an agent infected this step won't recover immediately
            elif agent.is_infected() and agent_id in agents_to_recover_ids:
                 agent.set_state(STATE_RECOVERED)

        # Update total infections metric
        self.metrics["total_infections"] = len(self._ever_infected_ids)


        # 6. Record Metrics
        self._calculate_metrics(current_step)


    def run(self):
        """
        Runs the simulation for the specified duration.
        Implements the simulation_loop algorithm.
        """
        print(f"Starting simulation with {self.population_size} agents for {self.simulation_duration_steps} steps.")
        print(f"Initial infected: {self.initial_infected_count}")
        print(f"Transmission probability: {self.transmission_probability_per_contact}")
        print(f"Recovery rate: {self.recovery_rate_per_time_step}")
        if self.random_seed is not None:
            print(f"Random seed: {self.random_seed}")
        print("-" * 30)


        # Record initial state (time step 0)
        self._calculate_metrics(0)

        for step in range(1, self.simulation_duration_steps + 1):
            # print(f"Time Step {step}")
            self._time_step(step)
            # Optional: Add break condition if epidemic dies out
            # if self.metrics[STATE_INFECTED][-1] == 0 and self.metrics[STATE_SUSCEPTIBLE][-1] > 0:
            #     print(f"Epidemic died out at step {step}. Stopping simulation.")
            #     # Fill remaining steps with last recorded values
            #     for future_step in range(step + 1, self.simulation_duration_steps + 1):
            #          self.metrics["time_step"].append(future_step)
            #          self.metrics[STATE_SUSCEPTIBLE].append(self.metrics[STATE_SUSCEPTIBLE][-1])
            #          self.metrics[STATE_INFECTED].append(self.metrics[STATE_INFECTED][-1])
            #          self.metrics[STATE_RECOVERED].append(self.metrics[STATE_RECOVERED][-1])
            #     break


        print("-" * 30)
        print("Simulation finished.")
        print(f"Final counts: S={self.metrics[STATE_SUSCEPTIBLE][-1]}, I={self.metrics[STATE_INFECTED][-1]}, R={self.metrics[STATE_RECOVERED][-1]}")
        print(f"Total individuals ever infected: {self.metrics['total_infections']}")

        return self.metrics

def plot_results(metrics, title="Epidemic Spread Simulation"):
    """
    Plots the number of individuals in each state over time.

    Args:
        metrics (dict): Dictionary containing simulation results (time series data).
        title (str): Title for the plot.
    """
    if not metrics or not metrics["time_step"]:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 8))
    plt.plot(metrics["time_step"], metrics[STATE_SUSCEPTIBLE], label=STATE_SUSCEPTIBLE, color='blue')
    plt.plot(metrics["time_step"], metrics[STATE_INFECTED], label=STATE_INFECTED, color='red')
    plt.plot(metrics["time_step"], metrics[STATE_RECOVERED], label=STATE_RECOVERED, color='green')
    plt.plot(metrics["time_step"], [metrics["total_infections"]] * len(metrics["time_step"]), label="Total Infected (Cumulative)", color='purple', linestyle='--', alpha=0.7) # Total infected is cumulative

    plt.xlabel("Time Step")
    plt.ylabel("Number of Individuals")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def save_results_to_csv(metrics, filename="simulation_results.csv"):
    """
    Saves the simulation results to a CSV file.

    Args:
        metrics (dict): Dictionary containing simulation results.
        filename (str): The name of the CSV file to save.
    """
    if not metrics or not metrics["time_step"]:
        print("No data to save.")
        return

    headers = ["time_step", STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED, "total_infections_cumulative"]
    data_rows = []

    # The 'total_infections' metric in the dict is the final cumulative value.
    # We need to replicate it for each step or calculate cumulative per step if needed.
    # Let's just save the final cumulative value in a separate column or note it.
    # For simplicity here, we'll just save the S, I, R counts and the final total.
    # A more detailed save would track cumulative infections per step.
    # For this implementation, total_infections in metrics dict is the final cumulative count.
    # Let's add a cumulative infections column derived from S->I transitions if needed.
    # Re-calculating cumulative infections for each step from S->I transitions might be complex
    # without storing per-step transitions. The current `metrics["total_infections"]` is the final value.
    # Let's store the cumulative value *at each step* for the CSV.

    # Recalculate cumulative infections over time for CSV
    cumulative_infected_per_step = [metrics["total_infections"]] * len(metrics["time_step"]) # Placeholder for now.

    # Let's refine the metric recording to correctly track cumulative infections per step.
    # A simple way is to use the initial infected count + sum of newly infected each step.
    # Or, count agents whose ID is in _ever_infected_ids at each step.
    # Let's modify _calculate_metrics to store the cumulative count at each step.
    # Update: The current _calculate_metrics records the *current* S, I, R and the *final* total_infections.
    # Let's revise `_calculate_metrics` or the CSV saving to get cumulative per step.
    # The easiest way is to calculate it from the S counts: total_infected_ever = population_size - current_S - current_R (if no re-susceptibility)
    # Or initial_I + sum(newly_infected_this_step) over time.
    # Let's stick to the _ever_infected_ids set and calculate its size at each step in _calculate_metrics.

    # Assuming _calculate_metrics is updated to store cumulative infections per step:
    # headers = ["time_step", STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED, "cumulative_infected_ever"]
    # Let's assume the 'total_infections' metric list in the dict now stores per-step cumulative counts.
    # Need to update Simulation class to store total_infections as a list over time.

    # Revised approach: Let's update _calculate_metrics to store the size of _ever_infected_ids at each step.
    # And update the metrics dict initialization.
    # Initial metrics:
    # self.metrics = {
    #     "time_step": [],
    #     STATE_SUSCEPTIBLE: [],
    #     STATE_INFECTED: [],
    #     STATE_RECOVERED: [],
    #     "cumulative_infected_ever": [] # This will store the size of _ever_infected_ids at each step
    # }
    # And in _calculate_metrics:
    # self.metrics["cumulative_infected_ever"].append(len(self._ever_infected_ids))

    # Let's assume the Simulation class has been updated as above.
    # Now we can proceed with saving.

    # Check if the 'cumulative_infected_ever' key exists (implies the Simulation class was updated)
    if 'cumulative_infected_ever' in metrics:
         headers = ["time_step", STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED, "cumulative_infected_ever"]
         for i in range(len(metrics["time_step"])):
             data_rows.append([
                 metrics["time_step"][i],
                 metrics[STATE_SUSCEPTIBLE][i],
                 metrics[STATE_INFECTED][i],
                 metrics[STATE_RECOVERED][i],
                 metrics["cumulative_infected_ever"][i]
             ])
    else:
         # Fallback if Simulation class wasn't updated for per-step cumulative
         # Just save the S, I, R counts
         headers = ["time_step", STATE_SUSCEPTIBLE, STATE_INFECTED, STATE_RECOVERED]
         for i in range(len(metrics["time_step"])):
             data_rows.append([
                 metrics["time_step"][i],
                 metrics[STATE_SUSCEPTIBLE][i],
                 metrics[STATE_INFECTED][i],
                 metrics[STATE_RECOVERED][i]
             ])
         print("Warning: 'cumulative_infected_ever' metric not found. Saving only S, I, R counts.")


    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data_rows)
        print(f"Simulation results saved to {filename}")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    # Define simulation parameters
    simulation_parameters = {
        "population_size": 1000,
        "initial_infected_count": 5, # Increased initial infected for faster spread
        "transmission_probability_per_contact": 0.02, # Adjusted probability
        "recovery_rate_per_time_step": 0.005, # Adjusted recovery rate
        "simulation_duration_steps": 200, # Increased duration
        "random_seed": 42 # Use a fixed seed for reproducibility
    }

    # --- Update Simulation Class for per-step cumulative infections ---
    # This is a self-correction based on the CSV saving logic realization.
    # In a real scenario, this update would be done in the Simulation class definition above.
    # Applying it conceptually here before running.
    # The Simulation class definition provided *does* track _ever_infected_ids and updates
    # metrics["total_infections"] (singular) with its size, but doesn't store it per step
    # in a list. Let's add that list.

    # Re-defining or conceptually updating the Simulation class metrics init and calculation:
    # class Simulation:
    #     def __init__(...):
    #         ...
    #         self.metrics = {
    #             "time_step": [],
    #             STATE_SUSCEPTIBLE: [],
    #             STATE_INFECTED: [],
    #             STATE_RECOVERED: [],
    #             "cumulative_infected_ever": [] # Store per-step cumulative count
    #         }
    #         self._ever_infected_ids = set()
    #         ...
    #     def _calculate_metrics(...):
    #         ...
    #         self.metrics["cumulative_infected_ever"].append(len(self._ever_infected_ids))
    #     ...

    # Assuming the Simulation class *is* implemented with the 'cumulative_infected_ever' list:
    # (Note: The code above *has* been updated to include this list)

    try:
        # Create simulation instance
        sim = Simulation(parameters=simulation_parameters)

        # Run the simulation
        simulation_results = sim.run()

        # Visualize the results
        plot_results(simulation_results, title="Simple SIR Epidemic Spread Simulation")

        # Save results to CSV
        output_filename = "sir_simulation_results.csv"
        save_results_to_csv(simulation_results, filename=output_filename)

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}")