# main.py
import random
import matplotlib.pyplot as plt
from simulation import Simulation
from agent import Person
from environment import Environment
from metrics import MetricsTracker

def main():
    """
    Main function to set up, run, and visualize the simulation.
    """
    # Define simulation parameters based on the model plan
    parameters = {
        "population_size": 1000,
        "initial_infected_count": 1,
        "transmission_probability": 0.05,
        "recovery_time": 14, # in simulation steps
        "interactions_per_person_per_step": 5,
        "simulation_steps": 200,
        "random_seed": 42 # Use a fixed seed for reproducibility, set to None for random
    }

    # Set random seed if specified
    if parameters["random_seed"] is not None:
        random.seed(parameters["random_seed"])
        print(f"Using random seed: {parameters['random_seed']}")
    else:
        print("Using system time for random seed.")

    # --- Input Validation (Basic) ---
    if parameters["initial_infected_count"] > parameters["population_size"]:
        print("Error: Initial infected count cannot exceed population size.")
        return
    if parameters["initial_infected_count"] < 0 or parameters["population_size"] <= 0:
         print("Error: Population size and initial infected count must be non-negative.")
         return
    if parameters["transmission_probability"] < 0 or parameters["transmission_probability"] > 1:
         print("Error: Transmission probability must be between 0 and 1.")
         return
    if parameters["recovery_time"] <= 0:
        print("Warning: Recovery time is zero or negative. Infected individuals may not recover.")
    if parameters["interactions_per_person_per_step"] < 0:
        print("Warning: Interactions per person per step is negative. No interactions will occur.")
    if parameters["simulation_steps"] <= 0:
        print("Warning: Simulation steps is zero or negative. Simulation will not run.")


    print("Initializing simulation...")
    print(f"Parameters: {parameters}")

    # Create and run the simulation
    simulation = Simulation(parameters)
    print("Running simulation...")
    simulation.run(parameters["simulation_steps"])
    print("Simulation finished.")

    # Get results
    metrics = simulation.get_metrics()

    # --- Visualization ---
    print("Generating plots...")
    time_steps = range(len(metrics['susceptible_counts']))

    plt.figure(figsize=(12, 8))

    # Plot SIR curves
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, metrics['susceptible_counts'], label='Susceptible', color='blue')
    plt.plot(time_steps, metrics['infected_counts'], label='Infected', color='red')
    plt.plot(time_steps, metrics['recovered_counts'], label='Recovered', color='green')
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Individuals")
    plt.title("Epidemic Spread Over Time (SIR Curves)")
    plt.legend()
    plt.grid(True)

    # Plot Cumulative Infections
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, metrics['cumulative_infections'], label='Cumulative Infections', color='purple')
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Individuals")
    plt.title("Cumulative Infections Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- Basic Validation Check (Conservation of Population) ---
    print("\n--- Basic Validation ---")
    initial_pop = parameters['population_size']
    final_s = metrics['susceptible_counts'][-1]
    final_i = metrics['infected_counts'][-1]
    final_r = metrics['recovered_counts'][-1]
    final_pop = final_s + final_i + final_r

    print(f"Initial Population: {initial_pop}")
    print(f"Final S: {final_s}, Final I: {final_i}, Final R: {final_r}")
    print(f"Final S + I + R: {final_pop}")

    if final_pop == initial_pop:
        print("Validation Check: Population size conserved. (Passed)")
    else:
        print(f"Validation Check: Population size changed! Expected {initial_pop}, got {final_pop}. (Failed)")

    # --- Basic Validation Check (Epidemic Extinction) ---
    if metrics['infected_counts'][-1] == 0:
        print("Validation Check: Epidemic reached extinction. (Passed)")
    elif metrics['infected_counts'][-1] > 0 and parameters['simulation_steps'] > parameters['recovery_time']:
         print(f"Validation Check: Epidemic did not reach extinction (Infected count > 0 at end: {metrics['infected_counts'][-1]}). (Warning/Check)")
    else:
         print("Validation Check: Epidemic extinction not fully evaluated (simulation might be too short).")

if __name__ == "__main__":
    main()