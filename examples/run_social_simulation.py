#!/usr/bin/env python3
"""
Example script to run a social network simulation.
"""

import os
import sys
import json
import argparse

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.social_network_model import create_social_network_simulation
from utils.visualization import create_network_plot, create_time_series_plot, create_summary_dashboard

def main():
    """Run the social network simulation and visualize results."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a social network simulation')
    parser.add_argument('--num_people', type=int, default=100, help='Number of people in the simulation')
    parser.add_argument('--steps', type=int, default=50, help='Number of simulation steps to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = {
        'num_people': args.num_people,
        'simulation_steps': args.steps,
        'seed': args.seed
    }
    
    # Create and run simulation
    print(f"Creating simulation with {args.num_people} people")
    simulation = create_social_network_simulation(config)
    
    print(f"Running simulation for {args.steps} steps")
    results = simulation.run(steps=args.steps)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'simulation_results.json')
    with open(results_file, 'w') as f:
        # Convert results to serializable format (e.g., convert numpy values to Python types)
        serializable_results = json.loads(json.dumps({
            'config': results['config'],
            'metrics': {k: float(v) if hasattr(v, 'item') else v for k, v in results['metrics'].items()},
            'time_series': results['time_series'],
            'run_info': results['run_info']
        }, default=str))
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to {results_file}")
    
    # Create visualizations
    print("Creating visualizations")
    
    # Network visualization
    node_positions = {}
    for person_id, person in simulation.environment.entities.items():
        node_positions[person_id] = person.attributes['location']
    
    create_network_plot(
        simulation.environment.network,
        node_positions=node_positions,
        title='Social Network Visualization',
        output_path=os.path.join(args.output_dir, 'network_plot.png')
    )
    
    # Time series visualization
    metric_names = list(simulation.environment.metrics.keys())
    create_time_series_plot(
        results['time_series'],
        metric_names,
        title='Simulation Metrics Over Time',
        output_path=os.path.join(args.output_dir, 'time_series.png')
    )
    
    # Create summary dashboard
    create_summary_dashboard(results, args.output_dir)
    
    print(f"Created visualizations in {args.output_dir}")
    print(f"View summary dashboard at {os.path.join(args.output_dir, 'summary.html')}")
    
    # Print final metrics
    print("\nFinal Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value}")

if __name__ == '__main__':
    main() 