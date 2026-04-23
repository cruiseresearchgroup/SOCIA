# main.py
import matplotlib.pyplot as plt
import numpy as np
import random

from simulation import Simulation
from agent import Agent
from environment import Environment
from utils import euclidean_distance # Assuming utils has this


def main():
    """
    Sets up and runs the social simulation based on the model plan.
    """
    print("Setting up simulation...")

    # --- Simulation Parameters ---
    # Using parameters from the model plan
    params = {
        "population_size": 100,
        "simulation_steps": 1000,
        "interaction_range": 5.0,
        "movement_speed": 1.0,
        "opinion_influence_factor": 0.1,
        "energy_cost_per_move": 1,
        "energy_gain_per_step": 0.5,
        "environment_dimensions": [100, 100],
        "initial_state_distribution": {"state_A": 0.5, "state_B": 0.5},
        "initial_opinion_range": [0, 1],
        "initial_energy": 100,
        "random_seed": None # Set to an integer for reproducible runs
    }

    # Set random seed if specified
    if params["random_seed"] is not None:
        random.seed(params["random_seed"])
        np.random.seed(params["random_seed"]) # For numpy operations

    # --- Environment Setup ---
    environment = Environment(dimensions=params["environment_dimensions"])

    # --- Agent Initialization ---
    agents = []
    for i in range(params["population_size"]):
        initial_position = environment.get_random_position()
        # Randomly assign initial state based on distribution
        initial_state = random.choices(
            list(params["initial_state_distribution"].keys()),
            weights=list(params["initial_state_distribution"].values()),
            k=1
        )[0]
        # Randomly assign initial opinion within range
        initial_opinion = random.uniform(*params["initial_opinion_range"])
        initial_energy = params["initial_energy"]

        agent = Agent(
            id=i,
            position=initial_position,
            state=initial_state,
            opinion=initial_opinion,
            energy=initial_energy,
            environment=environment, # Pass environment for interaction/movement checks
            params={
                "movement_speed": params["movement_speed"],
                "interaction_range": params["interaction_range"],
                "opinion_influence_factor": params["opinion_influence_factor"],
                "energy_cost_per_move": params["energy_cost_per_move"],
                "energy_gain_per_step": params["energy_gain_per_step"],
            }
        )
        agents.append(agent)

    # Add agents to the environment
    environment.add_agents(agents)

    # --- Simulation Setup ---
    simulation = Simulation(
        agents=agents,
        environment=environment,
        params=params
    )

    # --- Run Simulation ---
    print(f"Running simulation for {params['simulation_steps']} steps...")
    simulation.run(params["simulation_steps"])
    print("Simulation finished.")

    # --- Data Analysis and Visualization ---
    print("Generating visualizations...")
    plot_results(simulation.history)
    print("Done.")


def plot_results(history):
    """
    Plots key simulation metrics over time.

    Args:
        history (dict): A dictionary containing historical data from the simulation.
                        Expected keys: 'step', 'state_counts', 'average_opinion'.
    """
    steps = history['step']
    state_counts = history['state_counts'] # Dict of state -> list of counts
    avg_opinion = history['average_opinion']

    if not steps:
        print("No simulation history data to plot.")
        return

    # Plot State Counts
    plt.figure(figsize=(12, 6))
    for state, counts in state_counts.items():
        plt.plot(steps, counts, label=f'State {state}')
    plt.xlabel("Simulation Step")
    plt.ylabel("Number of Agents")
    plt.title("Agent State Distribution Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Average Opinion
    plt.figure(figsize=(12, 6))
    plt.plot(steps, avg_opinion)
    plt.xlabel("Simulation Step")
    plt.ylabel("Average Opinion")
    plt.title("Average Agent Opinion Over Time")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()