# main.py

import random
import sys
import os
from datetime import datetime

# Add the current directory to the path to allow importing other modules
sys.path.append(os.path.dirname(__file__))

from config import SIMULATION_PARAMETERS
from simulation import Simulation
from utils import plot_results, save_results_to_csv

def main():
    """
    Main function to set up and run the SIR simulation.
    """
    print("Starting Simple Epidemic Simulation (SIR Model)")

    # Load parameters
    params = SIMULATION_PARAMETERS
    population_size = params["population_size"]
    initial_infected_count = params["initial_infected_count"]
    simulation_duration_steps = params["simulation_duration_steps"]
    random_seed = params["random_seed"]

    # --- Parameter Validation ---
    if population_size <= 0:
        print("Error: Population size must be positive.")
        sys.exit(1)
    if initial_infected_count < 0 or initial_infected_count > population_size:
        print(f"Error: Initial infected count ({initial_infected_count}) must be between 0 and population size ({population_size}).")
        sys.exit(1)
    if simulation_duration_steps <= 0:
        print("Error: Simulation duration must be positive.")
        sys.exit(1)
    for param, value in params.items():
         if isinstance(value, (int, float)) and value < 0 and param not in ["random_seed"]:
             print(f"Warning: Parameter '{param}' is negative ({value}). Ensure this is intended.")

    # Set random seed
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using specified random seed: {random_seed}")
    else:
        # Use current time if no seed is specified
        seed = int(datetime.now().timestamp())
        random.seed(seed)
        print(f"Using random seed from current time: {seed}")

    # Create and run the simulation
    print(f"Initializing simulation with {population_size} individuals, {initial_infected_count} initially infected.")
    simulation = Simulation(params)
    simulation.run()

    print("Simulation finished.")

    # Get results
    metrics = simulation.get_metrics()

    # --- Data Analysis and Output ---
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Plot results
    plot_filename = os.path.join(output_dir, "sir_simulation_plot.png")
    print(f"Saving plot to {plot_filename}")
    try:
        plot_results(metrics, plot_filename)
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot: {e}")

    # Save results to CSV
    csv_filename = os.path.join(output_dir, "sir_simulation_data.csv")
    print(f"Saving data to {csv_filename}")
    try:
        save_results_to_csv(metrics, csv_filename)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

    # Basic validation checks (optional, but good practice)
    print("\n--- Validation Checks ---")
    last_step_metrics = {key: values[-1] if isinstance(values, list) and values else None
                         for key, values in metrics.items() if key != 'time_step'}
    last_step_metrics['time_step'] = metrics['time_step'][-1] if metrics['time_step'] else 0

    # Population Conservation Check
    if last_step_metrics['susceptible_count'] is not None and \
       last_step_metrics['infected_count'] is not None and \
       last_step_metrics['recovered_count'] is not None:
        total_population_at_end = last_step_metrics['susceptible_count'] + \
                                  last_step_metrics['infected_count'] + \
                                  last_step_metrics['recovered_count']
        if total_population_at_end == population_size:
            print(f"Validation: Population Conservation PASSED (Total at end: {total_population_at_end})")
        else:
            print(f"Validation: Population Conservation FAILED (Total at end: {total_population_at_end}, Expected: {population_size})")
    else:
         print("Validation: Population Conservation check skipped due to missing metric data.")


    # Epidemic Curve Shape Check (basic)
    infected_counts = metrics.get('infected_count', [])
    if infected_counts:
        # Check if infected count ever rose above initial and then fell
        initial_infected = infected_counts[0] if infected_counts else 0
        peak_infected = max(infected_counts) if infected_counts else 0
        final_infected = infected_counts[-1] if infected_counts else 0

        if peak_infected > initial_infected and final_infected <= initial_infected:
             print("Validation: Epidemic Curve Shape (basic) PASSED (Infected count rose and fell)")
        elif peak_infected == initial_infected and initial_infected > 0 and final_infected == initial_infected:
             print("Validation: Epidemic Curve Shape (basic) - Possible endemic or no spread (Infected count stayed constant)")
        elif peak_infected == initial_infected and initial_infected > 0 and final_infected < initial_infected:
             print("Validation: Epidemic Curve Shape (basic) - Possible outbreak died out quickly (Infected count only fell)")
        elif peak_infected > initial_infected and final_infected > initial_infected:
             print("Validation: Epidemic Curve Shape (basic) - Possible ongoing outbreak or plateaued (Infected count rose and stayed high)")
        else:
             print("Validation: Epidemic Curve Shape (basic) - No significant change in infected count.")
    else:
        print("Validation: Epidemic Curve Shape check skipped due to missing infected count data.")

    # Final State Plausibility Check (basic)
    if last_step_metrics['infected_count'] is not None:
        if last_step_metrics['infected_count'] <= initial_infected_count and last_step_metrics['infected_count'] < population_size * 0.01: # Consider near zero
             print(f"Validation: Final State Plausibility (basic) PASSED (Infected count at end is low: {last_step_metrics['infected_count']})")
        else:
             print(f"Validation: Final State Plausibility (basic) FAILED (Infected count at end is high: {last_step_metrics['infected_count']})")
    else:
         print("Validation: Final State Plausibility check skipped due to missing infected count data.")

    print("--- End of Validation Checks ---")
    print("Simulation finished successfully.")


if __name__ == "__main__":
    main()