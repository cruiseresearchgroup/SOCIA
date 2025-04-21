#!/usr/bin/env python3
"""
Example script to run an epidemic simulation.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.epidemic_model import create_epidemic_simulation


def main():
    """Run an epidemic simulation with configurable parameters."""
    parser = argparse.ArgumentParser(description='Run an epidemic simulation')
    
    # Simulation parameters
    parser.add_argument('--population-size', type=int, default=1000,
                        help='Number of people in the simulation')
    parser.add_argument('--initial-infected', type=int, default=5,
                        help='Number of initially infected people')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of simulation steps')
    parser.add_argument('--transmission-rate', type=float, default=0.05,
                        help='Rate of transmission (0.0-1.0)')
    parser.add_argument('--recovery-rate', type=float, default=0.1,
                        help='Rate of recovery (0.0-1.0)')
    parser.add_argument('--contact-radius', type=float, default=0.05,
                        help='Radius for contact between people')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Configure simulation
    config = {
        'population_size': args.population_size,
        'initial_infected': args.initial_infected,
        'transmission_rate': args.transmission_rate,
        'recovery_rate': args.recovery_rate,
        'contact_radius': args.contact_radius,
        'seed': args.seed
    }
    
    # Print configuration
    print("Epidemic Simulation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create and initialize simulation
    print("\nInitializing simulation...")
    simulation = create_epidemic_simulation(config)
    simulation.initialize()
    
    # Run simulation
    print(f"Running simulation for {args.steps} steps...")
    time_series = []
    
    for step in range(args.steps):
        metrics = simulation.environment.step()
        time_series.append(metrics.copy())
        
        # Print progress every 10 steps
        if step % 10 == 0 or step == args.steps - 1:
            print(f"Step {step + 1}/{args.steps}:")
            print(f"  Susceptible: {metrics['susceptible_count']}")
            print(f"  Infected: {metrics['infected_count']}")
            print(f"  Recovered: {metrics['recovered_count']}")
            print(f"  New Infections: {metrics['new_infections']}")
    
    # Save results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.output_dir, f'epidemic_results_{timestamp}.json')
    
    results = {
        'config': config,
        'final_metrics': simulation.environment.metrics,
        'time_series': time_series
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Visualize the results
    print("\nGenerating visualizations...")
    simulation.visualize()
    
    # Save SIR plot to output directory
    sir_plot_path = os.path.join(args.output_dir, f'epidemic_sir_{timestamp}.png')
    pop_plot_path = os.path.join(args.output_dir, f'epidemic_population_{timestamp}.png')
    
    # Rename the default visualization files
    if os.path.exists('epidemic_sir.png'):
        os.rename('epidemic_sir.png', sir_plot_path)
    
    if os.path.exists('epidemic_population.png'):
        os.rename('epidemic_population.png', pop_plot_path)
    
    print(f"SIR curve visualization saved to {sir_plot_path}")
    print(f"Population state visualization saved to {pop_plot_path}")
    
    # Create an R0 estimation plot
    create_r0_estimation_plot(time_series, os.path.join(args.output_dir, f'r0_estimation_{timestamp}.png'))
    
    # Print final results summary
    print("\nSimulation Complete!")
    print(f"Final state after {args.steps} steps:")
    print(f"  Susceptible: {simulation.environment.metrics['susceptible_count']}")
    print(f"  Infected: {simulation.environment.metrics['infected_count']}")
    print(f"  Recovered: {simulation.environment.metrics['recovered_count']}")
    print(f"  Peak Infections: {simulation.environment.metrics['peak_infections']}")
    
    # Calculate basic reproduction number R0 (simplified estimate)
    total_infected = simulation.environment.metrics['infected_count'] + simulation.environment.metrics['recovered_count']
    initial_susceptible = args.population_size - args.initial_infected
    
    if total_infected > args.initial_infected and initial_susceptible > 0:
        r0_estimate = np.log(total_infected / args.initial_infected) / (1 - (total_infected / args.population_size))
        print(f"  Estimated basic reproduction number (R0): {r0_estimate:.2f}")


def create_r0_estimation_plot(time_series, output_path):
    """
    Create a plot estimating the effective reproduction number over time.
    
    Args:
        time_series: The time series data from the simulation
        output_path: Where to save the plot
    """
    # Extract data
    steps = len(time_series)
    new_infections = [ts['new_infections'] for ts in time_series]
    infected = [ts['infected_count'] for ts in time_series]
    
    # Calculate effective reproduction number (simplified)
    # Re = new infections / (infected * transmission_rate)
    re_values = []
    window_size = 5  # Use a window to smooth the curve
    
    for i in range(window_size, steps):
        avg_new_infections = sum(new_infections[i-window_size:i]) / window_size
        avg_infected = sum(infected[i-window_size:i]) / window_size
        
        if avg_infected > 0:
            re = avg_new_infections / avg_infected
            re_values.append(re)
        else:
            re_values.append(0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(window_size, steps), re_values, 'r-')
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.7)
    
    plt.title('Estimated Effective Reproduction Number (Re) Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Re')
    plt.grid(True, alpha=0.3)
    
    # Add annotation explaining Re
    plt.figtext(0.5, 0.01, 
                'Re > 1: Epidemic growing | Re < 1: Epidemic declining', 
                ha='center', fontsize=10, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Reproduction number estimation plot saved to {output_path}")


if __name__ == "__main__":
    main() 