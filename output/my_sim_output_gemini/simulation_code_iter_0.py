# main.py
"""
Main entry point for the Simple Epidemic Simulation.

Sets up the simulation environment, runs the simulation for a specified number of steps,
collects metrics, and visualizes the results.
"""

import random
import matplotlib.pyplot as plt
from collections import defaultdict

# Import custom modules
from config import SimulationConfig
from utils import set_random_seed, get_random_float
from person import Person
from simulation_env import SimulationEnv
from metrics import MetricsTracker

def main():
    """
    Initializes and runs the epidemic simulation.
    """
    print("Starting Simple Epidemic Simulation...")

    # Load configuration
    config = SimulationConfig()
    params = config.get_params()
    print(f"Simulation Parameters: {params}")

    # Set random seed if specified
    if params["random_seed"] is not None:
        set_random_seed(params["random_seed"])
        print(f"Random seed set to {params['random_seed']}")
    else:
        print("No random seed specified, using default randomness.")

    # Initialize simulation environment and agents
    env = SimulationEnv(params)
    print(f"Population initialized with {params['population_size']} agents.")
    print(f"Initial state: Susceptible={env.count_status('Susceptible')}, Infected={env.count_status('Infected')}, Recovered={env.count_status('Recovered')}, Dead={env.count_status('Dead')}")

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()

    # Run simulation loop
    print(f"Running simulation for {params['simulation_steps']} steps...")
    for step in range(params["simulation_steps"]):
        # Record metrics at the beginning of the step
        metrics_tracker.record_step(env.get_agents())

        # Run one simulation step
        env.run_step()

        # Optional: Print progress
        if (step + 1) % 10 == 0 or step == 0 or step == params["simulation_steps"] - 1:
            print(f"Step {step + 1}/{params['simulation_steps']} completed.")

    print("Simulation finished.")

    # Record final metrics
    metrics_tracker.record_step(env.get_agents())

    # Get collected data
    simulation_data = metrics_tracker.get_data()

    # --- Validation Checks ---
    print("\nPerforming validation checks...")

    # 1. Population Conservation
    initial_pop = params['population_size']
    for i, step_data in enumerate(simulation_data):
        current_pop = step_data['Susceptible'] + step_data['Infected'] + step_data['Recovered'] + step_data['Dead']
        if current_pop != initial_pop:
            print(f"Validation Warning: Population not conserved at step {i}. Expected {initial_pop}, got {current_pop}")
    print("Population Conservation check completed (warnings printed if any).")

    # 2. Valid State Transitions (Implicitly handled by code logic)
    print("Valid State Transitions check relies on correct logic implementation (S->I, I->R, I->D).")

    # 3. Epidemic Curve Shape (Visual inspection after plotting)
    print("Epidemic Curve Shape will be visible in the plot.")


    # Visualize results
    print("Generating plots...")
    plot_simulation_results(simulation_data)
    print("Plots generated. Close plot window to exit.")


def plot_simulation_results(data):
    """
    Plots the simulation results (counts of S, I, R, D) over time.

    Args:
        data (list): A list of dictionaries, where each dictionary contains
                     the counts for a simulation step.
    """
    steps = list(range(len(data)))
    susceptible_counts = [d['Susceptible'] for d in data]
    infected_counts = [d['Infected'] for d in data]
    recovered_counts = [d['Recovered'] for d in data]
    dead_counts = [d['Dead'] for d in data]
    total_cases_counts = [d['Total Cases'] for d in data]

    plt.figure(figsize=(12, 8))

    plt.plot(steps, susceptible_counts, label='Susceptible', color='blue')
    plt.plot(steps, infected_counts, label='Infected', color='red')
    plt.plot(steps, recovered_counts, label='Recovered', color='green')
    plt.plot(steps, dead_counts, label='Dead', color='black')
    plt.plot(steps, total_cases_counts, label='Total Cases (Cumulative)', color='purple', linestyle='--')


    plt.xlabel("Simulation Steps")
    plt.ylabel("Number of People")
    plt.title("Simple Epidemic Simulation (SIR+D Model)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()