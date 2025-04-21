"""
Utilities for visualizing simulation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional

def create_time_series_plot(
    data: List[Dict[str, Any]],
    metrics: List[str],
    title: str = 'Simulation Metrics Over Time',
    output_path: str = 'time_series_plot.png'
) -> None:
    """
    Create a time series plot of simulation metrics.
    
    Args:
        data: List of time step data dictionaries
        metrics: List of metric names to plot
        title: Title of the plot
        output_path: Path to save the plot to
    """
    if not data:
        return
    
    num_metrics = len(metrics)
    fig_rows = (num_metrics + 1) // 2  # Ceil division
    
    plt.figure(figsize=(12, 4 * fig_rows))
    
    # Plot time series metrics
    time_points = [entry.get('time', i) for i, entry in enumerate(data)]
    
    for i, metric in enumerate(metrics):
        plt.subplot(fig_rows, 2, i+1)
        
        values = [entry.get('metrics', {}).get(metric, 0) for entry in data]
        plt.plot(time_points, values)
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path)
    plt.close()

def create_network_plot(
    network: nx.Graph,
    node_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    node_colors: Optional[Dict[str, str]] = None,
    node_sizes: Optional[Dict[str, float]] = None,
    title: str = 'Network Visualization',
    output_path: str = 'network_plot.png'
) -> None:
    """
    Create a network visualization.
    
    Args:
        network: NetworkX graph to visualize
        node_positions: Dictionary mapping node IDs to positions (optional)
        node_colors: Dictionary mapping node IDs to colors (optional)
        node_sizes: Dictionary mapping node IDs to sizes (optional)
        title: Title of the plot
        output_path: Path to save the plot to
    """
    plt.figure(figsize=(12, 8))
    
    # Set node positions
    if node_positions is None:
        pos = nx.spring_layout(network)
    else:
        pos = node_positions
    
    # Set node colors
    if node_colors is None:
        colors = 'blue'
    else:
        colors = [node_colors.get(node, 'blue') for node in network.nodes()]
    
    # Set node sizes
    if node_sizes is None:
        sizes = 50
    else:
        sizes = [node_sizes.get(node, 50) for node in network.nodes()]
    
    # Draw network
    nx.draw_networkx(
        network,
        pos=pos,
        with_labels=False,
        node_size=sizes,
        node_color=colors,
        alpha=0.7
    )
    
    plt.title(title)
    plt.axis('off')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path)
    plt.close()

def create_heatmap(
    data: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = 'Heatmap',
    output_path: str = 'heatmap.png'
) -> None:
    """
    Create a heatmap visualization.
    
    Args:
        data: 2D array of values
        x_labels: Labels for the x-axis
        y_labels: Labels for the y-axis
        title: Title of the plot
        output_path: Path to save the plot to
    """
    plt.figure(figsize=(10, 8))
    
    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Value')
    
    # Set labels
    if x_labels:
        plt.xticks(range(len(x_labels)), x_labels, rotation=45)
    if y_labels:
        plt.yticks(range(len(y_labels)), y_labels)
    
    plt.title(title)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path)
    plt.close()

def create_agent_trajectories_plot(
    trajectories: Dict[str, List[Tuple[float, float]]],
    title: str = 'Agent Trajectories',
    output_path: str = 'agent_trajectories.png'
) -> None:
    """
    Create a visualization of agent trajectories.
    
    Args:
        trajectories: Dictionary mapping agent IDs to lists of positions
        title: Title of the plot
        output_path: Path to save the plot to
    """
    plt.figure(figsize=(10, 8))
    
    # Plot trajectories
    for agent_id, positions in trajectories.items():
        if positions:
            x_values = [pos[0] for pos in positions]
            y_values = [pos[1] for pos in positions]
            plt.plot(x_values, y_values, alpha=0.7, label=agent_id if len(trajectories) <= 10 else None)
            
            # Mark start and end positions
            plt.scatter(x_values[0], y_values[0], color='green', s=50, alpha=0.7)
            plt.scatter(x_values[-1], y_values[-1], color='red', s=50, alpha=0.7)
    
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    if len(trajectories) <= 10:
        plt.legend()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path)
    plt.close()

def create_distribution_comparison_plot(
    sim_data: List[float],
    real_data: List[float],
    num_bins: int = 20,
    title: str = 'Distribution Comparison',
    x_label: str = 'Value',
    y_label: str = 'Frequency',
    output_path: str = 'distribution_comparison.png'
) -> None:
    """
    Create a comparison of simulated and real data distributions.
    
    Args:
        sim_data: Simulated data values
        real_data: Real data values
        num_bins: Number of histogram bins
        title: Title of the plot
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        output_path: Path to save the plot to
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate the range for the bins
    min_val = min(min(sim_data) if sim_data else 0, min(real_data) if real_data else 0)
    max_val = max(max(sim_data) if sim_data else 1, max(real_data) if real_data else 1)
    bin_range = (min_val, max_val)
    
    # Plot histograms
    plt.hist(sim_data, bins=num_bins, alpha=0.5, label='Simulation', range=bin_range)
    plt.hist(real_data, bins=num_bins, alpha=0.5, label='Real Data', range=bin_range)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path)
    plt.close()

def create_summary_dashboard(
    simulation_results: Dict[str, Any],
    output_dir: str = 'output'
) -> None:
    """
    Create a summary dashboard of simulation results.
    
    Args:
        simulation_results: Dictionary containing simulation results
        output_dir: Directory to save the dashboard to
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics and time series data
    metrics = simulation_results.get('metrics', {})
    time_series = simulation_results.get('time_series', [])
    
    # Create time series plot
    if time_series and 'metrics' in time_series[0]:
        metric_names = list(time_series[0]['metrics'].keys())
        create_time_series_plot(
            time_series,
            metric_names,
            title='Simulation Metrics Over Time',
            output_path=os.path.join(output_dir, 'time_series.png')
        )
    
    # Create summary HTML
    with open(os.path.join(output_dir, 'summary.html'), 'w') as f:
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('  <title>Simulation Results Summary</title>\n')
        f.write('  <style>\n')
        f.write('    body { font-family: Arial, sans-serif; margin: 20px; }\n')
        f.write('    h1 { color: #333; }\n')
        f.write('    h2 { color: #555; }\n')
        f.write('    table { border-collapse: collapse; width: 50%; }\n')
        f.write('    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
        f.write('    th { background-color: #f2f2f2; }\n')
        f.write('    img { max-width: 100%; }\n')
        f.write('  </style>\n')
        f.write('</head>\n')
        f.write('<body>\n')
        
        f.write('  <h1>Simulation Results Summary</h1>\n')
        
        # Simulation configuration
        f.write('  <h2>Configuration</h2>\n')
        f.write('  <table>\n')
        f.write('    <tr><th>Parameter</th><th>Value</th></tr>\n')
        
        for param, value in simulation_results.get('config', {}).items():
            f.write(f'    <tr><td>{param}</td><td>{value}</td></tr>\n')
        
        f.write('  </table>\n')
        
        # Final metrics
        f.write('  <h2>Final Metrics</h2>\n')
        f.write('  <table>\n')
        f.write('    <tr><th>Metric</th><th>Value</th></tr>\n')
        
        for metric, value in metrics.items():
            f.write(f'    <tr><td>{metric}</td><td>{value}</td></tr>\n')
        
        f.write('  </table>\n')
        
        # Images
        f.write('  <h2>Visualizations</h2>\n')
        
        # Time series plot
        if os.path.exists(os.path.join(output_dir, 'time_series.png')):
            f.write('  <h3>Metrics Over Time</h3>\n')
            f.write('  <img src="time_series.png" alt="Time Series Plot">\n')
        
        # Add other images if they exist
        for img_name in ['network_plot.png', 'agent_trajectories.png', 'distribution_comparison.png', 'heatmap.png']:
            if os.path.exists(os.path.join(output_dir, img_name)):
                f.write(f'  <h3>{img_name.split(".")[0].replace("_", " ").title()}</h3>\n')
                f.write(f'  <img src="{img_name}" alt="{img_name.split(".")[0].replace("_", " ").title()}">\n')
        
        # Run information
        f.write('  <h2>Run Information</h2>\n')
        f.write('  <table>\n')
        f.write('    <tr><th>Parameter</th><th>Value</th></tr>\n')
        
        run_info = simulation_results.get('run_info', {})
        for param, value in run_info.items():
            if param == 'duration' and value is not None:
                value = f"{value:.2f} seconds"
            f.write(f'    <tr><td>{param}</td><td>{value}</td></tr>\n')
        
        f.write('  </table>\n')
        
        f.write('</body>\n')
        f.write('</html>\n') 