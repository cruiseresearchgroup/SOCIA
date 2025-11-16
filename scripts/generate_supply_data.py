#!/usr/bin/env python3
"""
Generate SUPPLY trajectory data using the generative-simulations code.
This script generates fixed training, validation, and test datasets for SUPPLY task.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Add generative-simulations to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'generative-simulations', 'libs', 'SUPPLY'))

from env import load_data, to_dot_dict, SimulatorStep, simulate
from copy import deepcopy

def generate_supply_data(
    n_trajectories=100,
    seed=0,  # Use default seed=0 to match original code
    output_dir='data_fitting/supply_data'
):
    """
    Generate SUPPLY trajectory data and save as CSV files.
    
    Args:
        n_trajectories: Number of trajectories to generate for each set
        seed: Random seed for reproducibility
        output_dir: Output directory for data files
    """
    print(f"Generating SUPPLY data with {n_trajectories} trajectories per set (seed={seed})...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for data generation
    config = {
        "run": {
            "optimizer": "SBI",
            "optimize_params": True,
            "mmd_sigma": 1.0,
            "sbi_num_simulations": 1000,
            "sbi_num_samples_posterior": 5000,
            "sbi_sampling_timeout": 60
        }
    }
    config = to_dot_dict(config)
    env_name = "SUPPLY"
    
    # Load data using the SUPPLY environment code (lead_time=2, default)
    train_set, val_set, test_set, description, train_set_np, val_set_np, test_set_np = load_data(
        n=n_trajectories,
        config=config,
        seed=seed,
        env_name=env_name
    )
    
    # Generate OOD test data with lead_time=5
    print("Generating OOD test data with lead_time=5...")
    np.random.seed(seed + 10000)  # Use different seed for OOD data
    
    def simulate_ood(n=100, T=60):
        """Generate trajectories with lead_time=5 for OOD evaluation."""
        sim = SimulatorStep()
        sim.set_parameters(np.array([5.0, 1.0, 2.0, 5.0], dtype=float))  # lead_time=5 (last param)
        trajectories = []
        for _ in range(n):
            init_inv = np.random.randint(15, 51)
            state = {"inventory": init_inv, "pipeline": [], "backlog": 0, "t": 0}
            states = [deepcopy(state)]
            action = 4  # Constant action
            for _ in range(T):
                state = sim.step(state=state, action=action)
                states.append(state)
            trajectories.append(states)
        return (trajectories, None)
    
    test_set_ood = simulate_ood(n=n_trajectories, T=60)
    
    # Convert trajectories to CSV format
    def trajectories_to_dataframe(trajectories, set_name):
        """Convert trajectory list to DataFrame."""
        rows = []
        for traj_id, trajectory in enumerate(trajectories[0]):
            for t, state in enumerate(trajectory):
                row = {
                    'trajectory_id': traj_id,
                    'time_step': t,
                    'inventory': state.get('inventory', 0),
                    'backlog': state.get('backlog', 0),
                    'pipeline_len': len(state.get('pipeline', [])),
                    'pipeline_items': json.dumps(state.get('pipeline', [])),
                    'action': 4,  # Constant action used in data generation
                    't': state.get('t', t)
                }
                rows.append(row)
        return pd.DataFrame(rows)
    
    # Convert to DataFrames
    print("Converting trajectories to DataFrames...")
    train_df = trajectories_to_dataframe(train_set, 'train')
    val_df = trajectories_to_dataframe(val_set, 'val')
    test_df = trajectories_to_dataframe(test_set, 'test')
    test_df_ood = trajectories_to_dataframe(test_set_ood, 'test_ood')
    
    # Save as CSV (using naming convention consistent with other tasks)
    train_csv_path = os.path.join(output_dir, 'train_data.csv')
    val_csv_path = os.path.join(output_dir, 'val_data.csv')
    test_csv_path = os.path.join(output_dir, 'test_data.csv')
    test_ood_csv_path = os.path.join(output_dir, 'test_ood_data.csv')
    
    print(f"Saving training data to {train_csv_path}...")
    train_df.to_csv(train_csv_path, index=False)
    
    print(f"Saving validation data to {val_csv_path}...")
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"Saving test data (ID, lead_time=2) to {test_csv_path}...")
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"Saving OOD test data (lead_time=5) to {test_ood_csv_path}...")
    test_df_ood.to_csv(test_ood_csv_path, index=False)
    
    # Save metadata
    metadata = {
        'description': description,
        'n_trajectories': n_trajectories,
        'trajectory_length': 61,  # T=60 means 61 states (t=0 to t=60)
        'seed': seed,
        'state_variables': ['inventory', 'backlog', 'pipeline_len', 't'],
        'action': 4,
        'data_files': {
            'train': 'train_data.csv',
            'val': 'val_data.csv',
            'test': 'test_data.csv',
            'test_ood': 'test_ood_data.csv'
        },
        'lead_times': {
            'train': 2,
            'val': 2,
            'test': 2,
            'test_ood': 5
        },
        'statistics': {
            'train': {
                'n_trajectories': len(train_set[0]),
                'total_rows': len(train_df),
                'inventory_mean': float(train_df['inventory'].mean()),
                'inventory_std': float(train_df['inventory'].std()),
                'backlog_mean': float(train_df['backlog'].mean()),
                'backlog_std': float(train_df['backlog'].std()),
            },
            'val': {
                'n_trajectories': len(val_set[0]),
                'total_rows': len(val_df),
                'inventory_mean': float(val_df['inventory'].mean()),
                'inventory_std': float(val_df['inventory'].std()),
                'backlog_mean': float(val_df['backlog'].mean()),
                'backlog_std': float(val_df['backlog'].std()),
            },
            'test': {
                'n_trajectories': len(test_set[0]),
                'total_rows': len(test_df),
                'inventory_mean': float(test_df['inventory'].mean()),
                'inventory_std': float(test_df['inventory'].std()),
                'backlog_mean': float(test_df['backlog'].mean()),
                'backlog_std': float(test_df['backlog'].std()),
            },
            'test_ood': {
                'n_trajectories': len(test_set_ood[0]),
                'total_rows': len(test_df_ood),
                'inventory_mean': float(test_df_ood['inventory'].mean()),
                'inventory_std': float(test_df_ood['inventory'].std()),
                'backlog_mean': float(test_df_ood['backlog'].mean()),
                'backlog_std': float(test_df_ood['backlog'].std()),
            }
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Data Generation Summary")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Training trajectories: {len(train_set[0])} (lead_time=2)")
    print(f"Validation trajectories: {len(val_set[0])} (lead_time=2)")
    print(f"Test trajectories (ID): {len(test_set[0])} (lead_time=2)")
    print(f"Test trajectories (OOD): {len(test_set_ood[0])} (lead_time=5)")
    print(f"Trajectory length: 61 time steps (t=0 to t=60)")
    print(f"Seed: {seed} (ID), {seed + 10000} (OOD)")
    print("\nFiles generated:")
    print(f"  - {train_csv_path}")
    print(f"  - {val_csv_path}")
    print(f"  - {test_csv_path} (In-Distribution)")
    print(f"  - {test_ood_csv_path} (Out-of-Distribution)")
    print(f"  - {metadata_path}")
    print("="*60)
    
    return train_df, val_df, test_df, metadata

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SUPPLY trajectory data')
    parser.add_argument('--n-trajectories', type=int, default=100, help='Number of trajectories per set')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default=0 to match original code)')
    parser.add_argument('--output-dir', type=str, default='data_fitting/supply_data', help='Output directory')
    
    args = parser.parse_args()
    
    generate_supply_data(
        n_trajectories=args.n_trajectories,
        seed=args.seed,
        output_dir=args.output_dir
    )

