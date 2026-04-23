import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import subprocess
import matplotlib.patheffects as path_effects

# Constants
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
EVAL_DIR = os.path.join(PROJECT_ROOT, "output/evaluation_results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_fitting/mask_adoption_data")
TEST_DATA_FILE = os.path.join(DATA_DIR, "test_data.csv")

def run_evaluations():
    """Run all three evaluation scripts if results don't exist"""
    # Check if evaluation results exist
    iter3_results = os.path.exists(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter3.csv'))
    iter4_results = os.path.exists(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter4.csv'))
    iter5_results = os.path.exists(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter5.csv'))
    
    # Run evaluations if needed
    if not iter3_results:
        print("Running Iter3 ground truth evaluation...")
        subprocess.run(['python', os.path.join(OUTPUT_DIR, 'evaluate_daily_data.py')])
    
    if not iter4_results:
        print("Running Iter4 ground truth evaluation...")
        subprocess.run(['python', os.path.join(OUTPUT_DIR, 'evaluate_daily_data_iter4.py')])
    
    if not iter5_results:
        print("Running Iter5 ground truth evaluation...")
        subprocess.run(['python', os.path.join(OUTPUT_DIR, 'evaluate_daily_data_iter5.py')])

def load_ground_truth_data():
    """Load ground truth data and calculate average mask wearing rate for the last 10 days"""
    try:
        ground_truth = pd.read_csv(TEST_DATA_FILE)
        print(f"Loaded ground truth data with {len(ground_truth)} records")
        
        # Get unique days and sort
        days = sorted(ground_truth['day'].unique())
        
        # Calculate overall average
        overall_avg = ground_truth['wearing_mask'].mean() * 100
        print(f"Overall ground truth mask wearing rate: {overall_avg:.2f}%")
        
        # Calculate average for last 10 days (or all days if less than 10)
        last_days = days[-10:] if len(days) > 10 else days
        last_days_data = ground_truth[ground_truth['day'].isin(last_days)]
        last_days_avg = last_days_data['wearing_mask'].mean() * 100
        print(f"Last {len(last_days)} days average ground truth mask wearing rate: {last_days_avg:.2f}%")
        
        return {
            'overall_avg': overall_avg,
            'last_days_avg': last_days_avg,
            'all_days': days,
            'last_days': last_days
        }
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Ground truth data file not found at {TEST_DATA_FILE}")
        return None

def load_evaluation_results():
    """Load evaluation results for all three iterations"""
    results = {}
    
    try:
        results['iter3'] = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter3.csv'))
        print("Loaded Iter3 evaluation results")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Iter3 evaluation results not found or empty")
        results['iter3'] = None
    
    try:
        results['iter4'] = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter4.csv'))
        print("Loaded Iter4 evaluation results")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Iter4 evaluation results not found or empty")
        results['iter4'] = None
    
    try:
        results['iter5'] = pd.read_csv(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter5.csv'))
        print("Loaded Iter5 evaluation results")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Iter5 evaluation results not found or empty")
        results['iter5'] = None
    
    return results

def recalculate_composite_score(comparison_df):
    """Recalculate composite score to include RMSE in the calculation"""
    # Iterate through rows and recalculate the composite score
    updated_scores = []
    for i, row in comparison_df.iterrows():
        # Extract metrics
        avg_accuracy = row['avg_accuracy']
        avg_f1 = row['avg_f1']
        curve_corr = row['curve_correlation']
        r2 = row['r_squared']
        avg_error = row['avg_error'] / 100  # Convert from percentage
        rmse = row['time_series_rmse'] / 100  # Convert from percentage
        
        # Calculate updated composite score
        # Higher is better: accuracy, f1, correlation, r2
        # Lower is better: error, rmse
        updated_score = (
            avg_accuracy * 0.15 + 
            avg_f1 * 0.15 + 
            curve_corr * 0.15 + 
            r2 * 0.15 - 
            avg_error * 0.15 - 
            rmse * 0.25  # Giving more weight to RMSE for temporal accuracy
        )
        
        updated_scores.append(updated_score)
    
    # Add updated scores to dataframe
    comparison_df['updated_composite_score'] = updated_scores
    
    return comparison_df

def compare_metrics_across_iterations(results):
    """Compare key metrics across all iterations and scenarios"""
    # Check if all results are available
    missing_results = [key for key, value in results.items() if value is None]
    if missing_results:
        print(f"Missing evaluation results for: {', '.join(missing_results)}. Cannot create comprehensive comparison.")
        return None
    
    # Combine results into a single dataframe
    comparison_data = []
    
    for iteration, df in results.items():
        for _, row in df.iterrows():
            comparison_data.append({
                'iteration': iteration,
                'scenario': row['scenario'],
                'avg_sim_rate': row['avg_sim_rate'],
                'avg_gt_rate': row['avg_gt_rate'],
                'avg_accuracy': row['avg_accuracy'],
                'avg_f1': row['avg_f1'],
                'avg_error': row['avg_error'],
                'time_series_rmse': row['time_series_rmse'],
                'curve_correlation': row['curve_correlation'],
                'r_squared': row['r_squared'],
                'composite_score': row['composite_score']
            })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Recalculate composite score to include RMSE
    comparison_df = recalculate_composite_score(comparison_df)
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'all_iterations_ground_truth_comparison.csv'), index=False)
    print(f"All iterations comparison data saved to: {os.path.join(OUTPUT_DIR, 'all_iterations_ground_truth_comparison.csv')}")
    
    return comparison_df

def create_comparison_visualizations(comparison_df, ground_truth_stats=None):
    """Create visualizations comparing all iterations and scenarios"""
    if comparison_df is None or comparison_df.empty:
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # Create color mapping
    iteration_colors = {
        'iter3': 'lightblue',
        'iter4': 'lightcoral',
        'iter5': 'lightgreen'
    }
    
    # 1. Create grouped bar chart for mask wearing rates and other metrics
    plt.figure(figsize=(18, 12))
    
    # Create subplot for each metric
    metrics_to_plot = ['avg_sim_rate', 'avg_error', 'time_series_rmse', 'avg_f1', 'curve_correlation', 'updated_composite_score']
    metric_titles = [
        'Average Mask Wearing Rate (%)',
        'Average Error (%)',
        'Time Series RMSE (%)',
        'F1 Score vs Ground Truth',
        'Curve Correlation',
        'Updated Composite Score'
    ]
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        plt.subplot(2, 3, i+1)
        
        # Restructure data for bar chart
        pivot_data = pd.pivot_table(
            comparison_df,
            values=metric,
            index='scenario',
            columns='iteration'
        )
        
        # Plot bar chart
        ax = pivot_data.plot(kind='bar', ax=plt.gca(), color=[iteration_colors[i] for i in pivot_data.columns])
        
        # Add target line for avg_sim_rate if ground truth data available
        if metric == 'avg_sim_rate' and ground_truth_stats is not None:
            # Add horizontal line for last 10 days average
            ax.axhline(y=ground_truth_stats['last_days_avg'], color='black', linestyle='--', linewidth=2)
            
            # Add annotation for the line
            ax.text(
                -0.5, 
                ground_truth_stats['last_days_avg'] + 1, 
                f"Ground Truth: {ground_truth_stats['last_days_avg']:.1f}%", 
                ha='left', 
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
            )
        
        # Add labels
        plt.title(title)
        plt.xlabel('Scenario')
        plt.ylabel(title)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # Add value labels
        for p, iteration in zip(ax.patches, np.tile(pivot_data.columns, len(pivot_data.index))):
            height = p.get_height()
            if not np.isnan(height):
                if metric in ['avg_sim_rate', 'avg_error', 'time_series_rmse']:
                    label = f'{height:.1f}%'
                else:
                    label = f'{height:.3f}'
                ax.annotate(
                    label,
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=8
                )
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_iterations_metrics_comparison.png'), dpi=300)
    print(f"All iterations metrics comparison chart saved to: {os.path.join(OUTPUT_DIR, 'all_iterations_metrics_comparison.png')}")
    plt.close()
    
    # 2. Create a dedicated time series RMSE comparison chart
    plt.figure(figsize=(14, 8))
    
    # Restructure data for bar chart
    rmse_pivot_data = pd.pivot_table(
        comparison_df,
        values='time_series_rmse',
        index='scenario',
        columns='iteration'
    )
    
    # Create bar plot
    ax = rmse_pivot_data.plot(kind='bar', figsize=(12, 6), color=[iteration_colors[i] for i in rmse_pivot_data.columns])
    
    # Add labels and styling
    plt.title('Time Series RMSE Comparison Across Iterations', fontsize=14)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('RMSE (%)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.legend(title='Iteration')
    
    # Add value labels
    for p, iteration in zip(ax.patches, np.tile(rmse_pivot_data.columns, len(rmse_pivot_data.index))):
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=10
            )
    
    # Highlight the best (lowest) RMSE for each scenario
    for idx, row in rmse_pivot_data.iterrows():
        min_val = row.min()
        min_col = row.idxmin()
        x_pos = rmse_pivot_data.columns.get_loc(min_col)
        bar_pos = ax.patches[len(rmse_pivot_data.columns) * rmse_pivot_data.index.get_loc(idx) + x_pos]
        ax.add_patch(plt.Rectangle(
            (bar_pos.get_x(), bar_pos.get_y()),
            bar_pos.get_width(), bar_pos.get_height(),
            fill=False, edgecolor='black', linestyle='--', linewidth=2
        ))
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
                "RMSE measures the average magnitude of prediction errors over time.\nLower values indicate better temporal prediction accuracy.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_rmse_comparison.png'), dpi=300)
    print(f"Time Series RMSE comparison chart saved to: {os.path.join(OUTPUT_DIR, 'time_series_rmse_comparison.png')}")
    plt.close()
    
    # Create line plot for time series RMSE trends
    plt.figure(figsize=(12, 6))
    
    # Get iterations and scenarios as ordered lists
    iterations = sorted(comparison_df['iteration'].unique())
    scenarios = sorted(comparison_df['scenario'].unique())
    scenario_colors = {
        'baseline': '#1f77b4',  # Muted Blue
        'high_risk': '#ff7f0e', # Safety Orange
        'intervention': '#FF1493', # Bright Pink (was #2ca02c)
    }
    
    # Plot RMSE lines
    for scenario in scenarios:
        scenario_data = comparison_df[comparison_df['scenario'] == scenario]
        if not scenario_data.empty:
            plt.plot(
                scenario_data['iteration'],
                scenario_data['time_series_rmse'],
                marker='o', markersize=8,
                label=scenario,
                color=scenario_colors.get(scenario, 'gray'),
                linewidth=2
            )
            
            # Add value labels
            for i, row in scenario_data.iterrows():
                plt.text(
                    row['iteration'], row['time_series_rmse'] + 0.3,
                    f"{row['time_series_rmse']:.1f}%",
                    ha='center', va='bottom',
                    fontsize=9
                )
    
    plt.title('Time Series RMSE Trend Across Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('RMSE (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Scenario')
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
                "RMSE measures how accurately each iteration predicts mask adoption over time.\nDecreasing trend indicates improved temporal prediction accuracy.",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_rmse_trend.png'), dpi=300)
    print(f"Time Series RMSE trend chart saved to: {os.path.join(OUTPUT_DIR, 'time_series_rmse_trend.png')}")
    plt.close()
    
    # 3. Create radar chart for best scenarios
    plt.figure(figsize=(15, 12))
    
    # Find best scenario for each iteration using the updated composite score
    best_scenarios = {}
    for iteration in comparison_df['iteration'].unique():
        iter_data = comparison_df[comparison_df['iteration'] == iteration]
        best_scenario = iter_data.loc[iter_data['updated_composite_score'].idxmax()]['scenario']
        best_scenarios[iteration] = best_scenario
    
    # Extract data for best scenarios
    best_data = comparison_df[
        comparison_df.apply(
            lambda row: row['scenario'] == best_scenarios[row['iteration']], 
            axis=1
        )
    ]
    
    # Set radar chart metrics
    radar_metrics = ['avg_accuracy', 'avg_f1', 'curve_correlation', 'r_squared', 'time_series_rmse']
    radar_labels = ['Accuracy', 'F1 Score', 'Curve Correlation', 'R²', 'Time Series RMSE (inverted)']
    
    # Calculate angles
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create radar chart
    ax = plt.subplot(121, polar=True)
    
    # Set label positions
    plt.xticks(angles[:-1], radar_labels, size=12)
    
    # Draw radar chart for each iteration's best scenario
    for iteration, scenario in best_scenarios.items():
        # Get data for this iteration and scenario
        data = best_data[
            (best_data['iteration'] == iteration) & 
            (best_data['scenario'] == scenario)
        ]
        
        if data.empty:
            continue
        
        # Get values
        values = []
        for i, metric in enumerate(radar_metrics):
            if metric == 'time_series_rmse':
                # Invert RMSE so lower is better
                max_rmse = comparison_df['time_series_rmse'].max() / 100  # Convert to 0-1 scale
                rmse_val = data['time_series_rmse'].values[0] / 100  # Convert to 0-1 scale
                # Normalize to 0-1 range where 1 is best (lowest RMSE)
                values.append(1 - (rmse_val / max_rmse) if max_rmse > 0 else 0)
            elif metric in ['curve_correlation', 'r_squared']:
                # Normalize correlation from [-1,1] to [0,1]
                val = data[metric].values[0]
                values.append((val + 1) / 2 if metric == 'curve_correlation' else val)
            else:
                values.append(data[metric].values[0])
        
        # Close the loop
        values += values[:1]
        
        # Draw radar chart
        label = f"{iteration} - {scenario}"
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.1))
    plt.title('Best Scenarios Across Iterations', size=15)
    
    # Create bar chart for best scenarios' updated composite scores
    ax2 = plt.subplot(122)
    
    # Prepare data
    best_iter_data = []
    for iteration, scenario in best_scenarios.items():
        data = best_data[
            (best_data['iteration'] == iteration) & 
            (best_data['scenario'] == scenario)
        ]
        if not data.empty:
            best_iter_data.append({
                'iteration': iteration,
                'scenario': scenario,
                'updated_composite_score': data['updated_composite_score'].values[0]
            })
    
    best_iter_df = pd.DataFrame(best_iter_data)
    
    # Plot bar chart
    colors = [iteration_colors[i] for i in best_iter_df['iteration']]
    bars = plt.bar(best_iter_df['iteration'], best_iter_df['updated_composite_score'], color=colors)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.3f}',
            ha='center', va='bottom'
        )
    
    # Add scenario labels above bars
    for i, (_, row) in enumerate(best_iter_df.iterrows()):
        plt.text(
            i,
            row['updated_composite_score'] + 0.02,
            f"{row['scenario']}",
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.title('Best Scenario Updated Composite Scores')
    plt.ylabel('Updated Composite Score')
    plt.ylim(min(best_iter_df['updated_composite_score']) * 1.2 if min(best_iter_df['updated_composite_score']) < 0 
             else 0, 
             max(best_iter_df['updated_composite_score']) * 1.2 if max(best_iter_df['updated_composite_score']) > 0 
             else 0.5)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'best_scenarios_comparison.png'), dpi=300)
    print(f"Best scenarios comparison chart saved to: {os.path.join(OUTPUT_DIR, 'best_scenarios_comparison.png')}")
    plt.close()
    
    # 4. Create heatmap for improvement across iterations
    plt.figure(figsize=(16, 10))
    
    # Create improvement metrics
    improvement_metrics = ['avg_accuracy', 'avg_f1', 'avg_error', 'time_series_rmse', 'curve_correlation', 'updated_composite_score']
    improvement_titles = ['Accuracy', 'F1 Score', 'Error (%)', 'RMSE (%)', 'Curve Correlation', 'Updated Composite Score']
    
    for i, (metric, title) in enumerate(zip(improvement_metrics, improvement_titles)):
        plt.subplot(2, 3, i+1)
        
        # Create pivot table
        pivot_data = pd.pivot_table(
            comparison_df,
            values=metric,
            index='scenario',
            columns='iteration'
        )
        
        # Choose colormap
        cmap = 'RdYlGn' if metric not in ['avg_error', 'time_series_rmse'] else 'RdYlGn_r'
        
        # Format for display
        fmt = '.1f' if metric in ['avg_error', 'time_series_rmse'] else '.3f'
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, cmap=cmap, fmt=fmt)
        
        plt.title(f'{title} Across Iterations')
        plt.ylabel('Scenario')
        plt.xlabel('Iteration')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iteration_improvement_heatmap.png'), dpi=300)
    print(f"Iteration improvement heatmap saved to: {os.path.join(OUTPUT_DIR, 'iteration_improvement_heatmap.png')}")
    plt.close()
    
    # Create a dedicated RMSE heatmap
    plt.figure(figsize=(10, 6))
    
    # Create pivot table for RMSE
    rmse_pivot = pd.pivot_table(
        comparison_df,
        values='time_series_rmse',
        index='scenario',
        columns='iteration'
    )
    
    # Create heatmap
    sns.heatmap(rmse_pivot, annot=True, cmap='RdYlGn_r', fmt='.1f')
    
    plt.title('Time Series RMSE (%) Across Iterations')
    plt.ylabel('Scenario')
    plt.xlabel('Iteration')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_rmse_heatmap.png'), dpi=300)
    print(f"Time Series RMSE heatmap saved to: {os.path.join(OUTPUT_DIR, 'time_series_rmse_heatmap.png')}")
    plt.close()
    
    # 4. Create line plot for trajectory of improvement - NeurIPS style
    
    # Apply NeurIPS-like styling
    plt.style.use('seaborn-v0_8-whitegrid') # Start with a clean base
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['STIXGeneral'] # Or 'Times New Roman', 'Computer Modern'
    plt.rcParams['axes.labelsize'] = 16 # Increased from 14
    plt.rcParams['xtick.labelsize'] = 14 # Increased from 12
    plt.rcParams['ytick.labelsize'] = 14 # Increased from 12
    plt.rcParams['legend.fontsize'] = 12 # Kept at 12
    plt.rcParams['figure.titlesize'] = 18 # Kept at 18
    plt.rcParams['axes.titlesize'] = 16 # Increased from 14
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['lines.markersize'] = 6
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)) # Adjusted figsize
    axes = axes.flatten() # Flatten to easily iterate

    comparison_metrics = ['avg_sim_rate', 'avg_error', 'time_series_rmse', 'avg_f1', 'curve_correlation', 'updated_composite_score']
    comparison_titles = [
        'Avg. Mask Rate', # Shortened
        'Avg. Error',      # Shortened
        'Time Series RMSE',# Shortened
        'F1 Score',
        'Curve Correlation',
        'Composite Score'  # Shortened
    ]
    # Y-axis labels (can be different from titles for conciseness)
    y_labels = [
        'Rate (%)', 
        'Error (%)', 
        'RMSE (%)', 
        'F1 Score', 
        'Correlation', 
        'Score'
    ]

    iterations = sorted(comparison_df['iteration'].unique())
    scenarios = sorted(comparison_df['scenario'].unique())
    
    # Define a professional color palette & specific colors for key scenarios
    # Base palette for any other scenarios
    base_palette = sns.color_palette("deep", n_colors=len(scenarios))
    
    scenario_colors = {
        'baseline': '#1f77b4',  # Muted Blue
        'high_risk': '#ff7f0e', # Safety Orange
        'intervention': '#FF1493', # Bright Pink (was #2ca02c)
    }
    # Assign colors to other scenarios if they exist
    other_scenarios = [s for s in scenarios if s not in scenario_colors]
    for i, scn in enumerate(other_scenarios):
        scenario_colors[scn] = base_palette[i % len(base_palette)] # Cycle through base palette

    # Define markers for better differentiation, specific for key scenarios
    scenario_markers = {
        'baseline': 'o',    # Circle
        'high_risk': 's',   # Square
        'intervention': '^', # Triangle up
    }
    # Default markers for any other scenarios
    default_markers = ['D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    other_marker_idx = 0
    for scn in scenarios:
        if scn not in scenario_markers:
            scenario_markers[scn] = default_markers[other_marker_idx % len(default_markers)]
            other_marker_idx += 1
    

    for i, (metric, title) in enumerate(zip(comparison_metrics, comparison_titles)):
        ax = axes[i]
        
        # Sort scenarios to have a consistent plotting order for legend (e.g., baseline, high_risk, intervention first)
        # Custom sort order for legend and plotting
        preferred_order = ['baseline', 'high_risk', 'intervention']
        sorted_scenarios_for_plot = [s for s in preferred_order if s in scenarios] + \
                                    [s for s in scenarios if s not in preferred_order]

        for scenario in sorted_scenarios_for_plot: # Use the sorted list for plotting
            scenario_data = comparison_df[comparison_df['scenario'] == scenario]
            if not scenario_data.empty:
                # Ensure iterations are sorted for consistent line plotting
                scenario_data = scenario_data.sort_values(by='iteration')
                
                current_linewidth = plt.rcParams['lines.linewidth']
                current_path_effects = None
                zorder = 1 # Default zorder

                if scenario in ['baseline', 'high_risk', 'intervention']:
                    current_linewidth = 2.5 # Thicker line
                    current_path_effects = [path_effects.Stroke(linewidth=current_linewidth + 2, foreground='black', alpha=0.3),
                                            path_effects.Normal()]
                    zorder = 2 # Ensure these lines are on top

                ax.plot(
                    scenario_data['iteration'],
                    scenario_data[metric],
                    marker=scenario_markers.get(scenario, 'x'), 
                    label=scenario if i == 0 else "_nolegend_", 
                    color=scenario_colors.get(scenario, 'gray'),
                    linewidth=current_linewidth,
                    path_effects=current_path_effects,
                    zorder=zorder                )

        if metric == 'avg_sim_rate' and ground_truth_stats is not None:
            ax.axhline(y=ground_truth_stats['last_days_avg'], color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.text(
                # Position text carefully to avoid overlap, adjust x based on iteration labels
                0.95, # x position in axis coordinates (0 to 1)
                ground_truth_stats['last_days_avg'],
                f" GT: {ground_truth_stats['last_days_avg']:.1f}%",
                ha='right', va='bottom', # Align text relative to the point
                fontsize=9,
                color='black',
                alpha=0.8,
                # bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2') 
                # simpler text, no bbox
                transform=ax.get_yaxis_transform() # Use yaxis transform for y, data for x if iterations are numeric
            )

        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(y_labels[i]) # Use concise y-labels
        ax.grid(True) # Grid is already styled by rcParams

    # Add a single figure-level legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(scenarios), bbox_to_anchor=(0.5, -0.02)) # Adjust position

    fig.suptitle('Iteration Improvement Trends Across Scenarios', fontsize=plt.rcParams['figure.titlesize'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and legend
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'iteration_improvement_trend.png'), dpi=300, bbox_inches='tight')
    print(f"Iteration improvement trend chart saved to: {os.path.join(OUTPUT_DIR, 'iteration_improvement_trend.png')}")
    plt.close(fig) # Close the specific figure

def main():
    """Main function to run evaluations and create comparison visualizations"""
    print("\n===== Starting All Iterations Ground Truth Comparison =====\n")
    
    # Run evaluations if needed
    run_evaluations()
    
    # Load ground truth data stats
    ground_truth_stats = load_ground_truth_data()
    
    # Load results
    results = load_evaluation_results()
    
    # Compare across iterations
    comparison_df = compare_metrics_across_iterations(results)
    
    # Create visualizations
    create_comparison_visualizations(comparison_df, ground_truth_stats)
    
    # Determine overall best scenario across all iterations based on updated composite score
    if comparison_df is not None and not comparison_df.empty:
        best_row = comparison_df.loc[comparison_df['updated_composite_score'].idxmax()]
        print("\n===== Overall Best Scenario (Using Updated Composite Score) =====")
        print(f"Iteration: {best_row['iteration']}")
        print(f"Scenario: {best_row['scenario']}")
        print(f"Updated Composite Score: {best_row['updated_composite_score']:.4f}")
        print(f"Original Composite Score: {best_row['composite_score']:.4f}")
        print(f"Accuracy: {best_row['avg_accuracy']:.4f}")
        print(f"F1 Score: {best_row['avg_f1']:.4f}")
        print(f"Error: {best_row['avg_error']:.2f}%")
        print(f"Time Series RMSE: {best_row['time_series_rmse']:.2f}%")
        print(f"Curve Correlation: {best_row['curve_correlation']:.4f}")
        print(f"R-squared: {best_row['r_squared']:.4f}")
    
        # Find best scenario for time series prediction accuracy (lowest RMSE)
        best_rmse_row = comparison_df.loc[comparison_df['time_series_rmse'].idxmin()]
        print("\n===== Best Time Series Prediction Scenario =====")
        print(f"Iteration: {best_rmse_row['iteration']}")
        print(f"Scenario: {best_rmse_row['scenario']}")
        print(f"Time Series RMSE: {best_rmse_row['time_series_rmse']:.2f}%")
        print(f"Average Error: {best_rmse_row['avg_error']:.2f}%")
        print(f"Curve Correlation: {best_rmse_row['curve_correlation']:.4f}")
        print(f"Updated Composite Score: {best_rmse_row['updated_composite_score']:.4f}")
    
    print("\n===== Comparison Complete =====\n")

if __name__ == "__main__":
    main() 