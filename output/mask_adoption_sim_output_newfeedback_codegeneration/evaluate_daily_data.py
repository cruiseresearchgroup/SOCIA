import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Constants
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
EVAL_DIR = os.path.join(PROJECT_ROOT, "output/evaluation_results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_fitting/mask_adoption_data")
TEST_DATA_FILE = os.path.join(DATA_DIR, "test_data.csv")

def load_daily_data():
    """Load all scenario data for iter_3"""
    try:
        all_data = pd.read_csv(os.path.join(EVAL_DIR, "all_scenarios_all_days.csv"))
        print(f"Successfully loaded iter_3 daily data, {len(all_data)} records")
        return all_data
    except FileNotFoundError:
        print("Iter_3 daily data file not found. Please run simulation_code_iter_3.py first to generate the data.")
        return None

def load_ground_truth_data():
    """Load ground truth data from test_data.csv"""
    try:
        ground_truth = pd.read_csv(TEST_DATA_FILE)
        print(f"Successfully loaded ground truth data, {len(ground_truth)} records")
        
        # Convert wearing_mask to boolean for consistency
        ground_truth['wearing_mask'] = ground_truth['wearing_mask'].astype(bool)
        
        return ground_truth
    except FileNotFoundError:
        print(f"Ground truth data file not found at {TEST_DATA_FILE}")
        return None

def calculate_daily_metrics(data, ground_truth):
    """Calculate metrics for each day and scenario using ground truth data"""
    if data is None or ground_truth is None:
        return None
    
    # Create list to store daily metrics
    daily_metrics = []
    
    # Unique scenarios and days
    scenarios = data['scenario'].unique()
    days = data['day'].unique()
    
    # Calculate target rate from ground truth for reference
    target_rate = ground_truth['wearing_mask'].mean()
    print(f"Ground truth average mask wearing rate: {target_rate*100:.2f}%")
    
    for scenario in scenarios:
        for day in days:
            # Get simulation data for current scenario and day
            sim_day_data = data[(data['scenario'] == scenario) & (data['day'] == day)]
            
            # Get ground truth data for the same day
            gt_day_data = ground_truth[ground_truth['day'] == day]
            
            # Skip if no ground truth data for this day
            if len(gt_day_data) == 0:
                print(f"No ground truth data for day {day}, skipping...")
                continue
            
            # Merge simulation and ground truth data on agent_id
            merged_data = sim_day_data.merge(
                gt_day_data[['day', 'agent_id', 'wearing_mask']], 
                left_on=['day', 'agent_id'], 
                right_on=['day', 'agent_id'],
                suffixes=('_sim', '_gt')
            )
            
            # Calculate mask wearing rates
            sim_mask_rate = merged_data['mask_wearing_status'].mean()
            gt_mask_rate = merged_data['wearing_mask'].mean()
            
            # Calculate metrics comparing simulation to ground truth
            accuracy = accuracy_score(merged_data['wearing_mask'], merged_data['mask_wearing_status'])
            precision = precision_score(merged_data['wearing_mask'], merged_data['mask_wearing_status'], zero_division=0)
            recall = recall_score(merged_data['wearing_mask'], merged_data['mask_wearing_status'], zero_division=0)
            f1 = f1_score(merged_data['wearing_mask'], merged_data['mask_wearing_status'], zero_division=0)
            
            # Calculate error metrics
            error = abs(sim_mask_rate - gt_mask_rate)
            mae = mean_absolute_error(merged_data['wearing_mask'].astype(int), merged_data['mask_wearing_status'].astype(int))
            mse = mean_squared_error(merged_data['wearing_mask'].astype(int), merged_data['mask_wearing_status'].astype(int))
            rmse = np.sqrt(mse)
            
            # Add to results
            daily_metrics.append({
                'scenario': scenario,
                'day': day,
                'sim_mask_rate': sim_mask_rate,
                'gt_mask_rate': gt_mask_rate,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'error': error,
                'mae': mae,
                'mse': mse,
                'rmse': rmse
            })
    
    # Convert to DataFrame
    return pd.DataFrame(daily_metrics)

def visualize_metrics_over_time(metrics_df):
    """Create visualizations of metrics over time vs ground truth"""
    if metrics_df is None:
        return
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # Create scenario color mapping
    scenario_colors = {
        'baseline': 'blue',
        'high_risk': 'red',
        'intervention': 'green'
    }
    
    # Create multiple charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mask wearing rate over time compared to ground truth
    ax1 = axes[0, 0]
    
    # Plot ground truth line first
    gt_rates = metrics_df.drop_duplicates('day')[['day', 'gt_mask_rate']]
    ax1.plot(gt_rates['day'], gt_rates['gt_mask_rate'] * 100, 
            marker='*', markersize=10, linestyle='-', label='Ground Truth', color='black', linewidth=2)
    
    # Plot simulation lines
    for scenario in metrics_df['scenario'].unique():
        scenario_data = metrics_df[metrics_df['scenario'] == scenario]
        ax1.plot(scenario_data['day'], scenario_data['sim_mask_rate'] * 100, 
                marker='o', linestyle='-', label=scenario, color=scenario_colors[scenario])
    
    ax1.set_title('Mask Wearing Rate Comparison (Iter3)')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Mask Wearing Rate (%)')
    ax1.set_xticks(metrics_df['day'].unique())
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Prediction error over time
    ax2 = axes[0, 1]
    for scenario in metrics_df['scenario'].unique():
        scenario_data = metrics_df[metrics_df['scenario'] == scenario]
        ax2.plot(scenario_data['day'], scenario_data['error'] * 100, 
                marker='o', linestyle='-', label=scenario, color=scenario_colors[scenario])
    
    ax2.set_title('Prediction Error vs Ground Truth (Iter3)')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Absolute Error (%)')
    ax2.set_xticks(metrics_df['day'].unique())
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. F1 score over time
    ax3 = axes[1, 0]
    for scenario in metrics_df['scenario'].unique():
        scenario_data = metrics_df[metrics_df['scenario'] == scenario]
        ax3.plot(scenario_data['day'], scenario_data['f1'], 
                marker='o', linestyle='-', label=scenario, color=scenario_colors[scenario])
    
    ax3.set_title('F1 Score vs Ground Truth (Iter3)')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('F1 Score')
    ax3.set_xticks(metrics_df['day'].unique())
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. RMSE over time
    ax4 = axes[1, 1]
    for scenario in metrics_df['scenario'].unique():
        scenario_data = metrics_df[metrics_df['scenario'] == scenario]
        ax4.plot(scenario_data['day'], scenario_data['rmse'], 
                marker='o', linestyle='-', label=scenario, color=scenario_colors[scenario])
    
    ax4.set_title('RMSE vs Ground Truth (Iter3)')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('RMSE')
    ax4.set_xticks(metrics_df['day'].unique())
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'daily_metrics_vs_ground_truth_iter3.png'), dpi=300)
    print(f"Time series metrics chart saved to: {os.path.join(OUTPUT_DIR, 'daily_metrics_vs_ground_truth_iter3.png')}")
    plt.close()
    
    # Create heatmap showing performance of each scenario across days
    plt.figure(figsize=(14, 18))
    
    metrics_to_plot = ['sim_mask_rate', 'gt_mask_rate', 'error', 'f1']
    subplot_positions = [1, 2, 3, 4]
    
    for metric, pos in zip(metrics_to_plot, subplot_positions):
        plt.subplot(4, 1, pos)
        
        # Create heatmap data
        if metric == 'gt_mask_rate':
            # For ground truth, create a constant value across scenarios
            pivot_data = pd.pivot_table(
                metrics_df, 
                values=metric, 
                index='scenario',
                columns='day',
                aggfunc='mean'
            )
        else:
            pivot_data = pd.pivot_table(
                metrics_df, 
                values=metric, 
                index='scenario',
                columns='day',
                aggfunc='mean'
            )
        
        # Adjust display format
        if metric in ['sim_mask_rate', 'gt_mask_rate', 'error']:
            pivot_data = pivot_data * 100  # Convert to percentage
        
        # Choose appropriate colormap
        cmap = 'RdYlGn' if metric != 'error' else 'RdYlGn_r'
        
        # Draw heatmap
        sns.heatmap(pivot_data, annot=True, cmap=cmap, fmt='.1f' if metric in ['sim_mask_rate', 'gt_mask_rate', 'error'] else '.3f')
        
        # Set title and labels
        metric_titles = {
            'sim_mask_rate': 'Simulation Mask Wearing Rate (%)',
            'gt_mask_rate': 'Ground Truth Mask Wearing Rate (%)',
            'error': 'Prediction Error vs Ground Truth (%)',
            'f1': 'F1 Score vs Ground Truth'
        }
        plt.title(metric_titles[metric] + ' (Iter3)')
        plt.ylabel('Scenario')
        plt.xlabel('Day')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'daily_metrics_vs_ground_truth_heatmap_iter3.png'), dpi=300)
    print(f"Daily metrics heatmap saved to: {os.path.join(OUTPUT_DIR, 'daily_metrics_vs_ground_truth_heatmap_iter3.png')}")
    plt.close()

    # Create time series similarity visualization
    plt.figure(figsize=(15, 8))
    
    # Plot mask wearing rates as time series
    days = sorted(metrics_df['day'].unique())
    
    # Ground truth as a reference line
    gt_rates = metrics_df.drop_duplicates('day')[['day', 'gt_mask_rate']].sort_values('day')
    plt.plot(gt_rates['day'], gt_rates['gt_mask_rate'] * 100, 
            marker='*', markersize=10, linestyle='-', label='Ground Truth', color='black', linewidth=2)
    
    # Plot each scenario
    for scenario in metrics_df['scenario'].unique():
        scenario_data = metrics_df[metrics_df['scenario'] == scenario].sort_values('day')
        plt.plot(scenario_data['day'], scenario_data['sim_mask_rate'] * 100, 
                marker='o', linestyle='-', label=scenario, color=scenario_colors[scenario])
    
    plt.title('Behavior Propagation Curves: Simulation vs Ground Truth (Iter3)', fontsize=14)
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Mask Wearing Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xticks(days)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'propagation_curve_comparison_iter3.png'), dpi=300)
    print(f"Propagation curve comparison saved to: {os.path.join(OUTPUT_DIR, 'propagation_curve_comparison_iter3.png')}")
    plt.close()

def calculate_scenario_performance(metrics_df):
    """Calculate the overall performance of each scenario compared to ground truth"""
    if metrics_df is None:
        return
    
    # Create a new DataFrame to store summary metrics
    summary = []
    
    # Create scenario color mapping
    scenario_colors = {
        'baseline': 'blue',
        'high_risk': 'red',
        'intervention': 'green'
    }
    
    for scenario in metrics_df['scenario'].unique():
        scenario_data = metrics_df[metrics_df['scenario'] == scenario]
        
        # Calculate average metrics
        avg_sim_rate = scenario_data['sim_mask_rate'].mean()
        avg_gt_rate = scenario_data['gt_mask_rate'].mean()
        avg_accuracy = scenario_data['accuracy'].mean()
        avg_f1 = scenario_data['f1'].mean()
        avg_error = scenario_data['error'].mean()
        avg_rmse = scenario_data['rmse'].mean()
        
        # Group by day for time series analysis
        day_data = scenario_data.sort_values('day')
        
        # Calculate curve similarity (correlation)
        corr, _ = pearsonr(day_data['sim_mask_rate'], day_data['gt_mask_rate'])
        
        # Calculate R-squared
        r2 = r2_score(day_data['gt_mask_rate'], day_data['sim_mask_rate'])
        
        # Calculate weighted accuracy (higher weight for later days)
        days = day_data['day'].values
        weights = np.linspace(0.5, 1.0, len(days))
        weighted_accuracy = np.average(day_data['accuracy'], weights=weights)
        
        # Calculate behavior change (first day to last day)
        first_day = min(days)
        last_day = max(days)
        first_sim = day_data[day_data['day'] == first_day]['sim_mask_rate'].values[0]
        last_sim = day_data[day_data['day'] == last_day]['sim_mask_rate'].values[0]
        first_gt = day_data[day_data['day'] == first_day]['gt_mask_rate'].values[0]
        last_gt = day_data[day_data['day'] == last_day]['gt_mask_rate'].values[0]
        
        sim_change = last_sim - first_sim
        gt_change = last_gt - first_gt
        change_error = abs(sim_change - gt_change)
        
        # Calculate weighted F1 score (higher weight for later days)
        weighted_f1 = np.average(day_data['f1'], weights=weights)
        
        # Calculate RMSE over time
        ts_rmse = np.sqrt(mean_squared_error(
            day_data['gt_mask_rate'], 
            day_data['sim_mask_rate']
        ))
        
        # Calculate composite score (weighted average of metrics)
        # Higher is better: accuracy, f1, correlation, r2, weighted metrics
        # Lower is better: error, rmse, change_error
        composite_score = (
            avg_accuracy * 0.15 + 
            avg_f1 * 0.15 + 
            corr * 0.15 + 
            r2 * 0.15 - 
            avg_error * 0.1 - 
            avg_rmse * 0.1 - 
            change_error * 0.1 +
            weighted_accuracy * 0.1 +
            weighted_f1 * 0.1
        )
        
        summary.append({
            'scenario': scenario,
            'avg_sim_rate': avg_sim_rate * 100,  # Convert to percentage
            'avg_gt_rate': avg_gt_rate * 100,  # Convert to percentage
            'avg_accuracy': avg_accuracy,
            'avg_f1': avg_f1,
            'avg_error': avg_error * 100,  # Convert to percentage
            'time_series_rmse': ts_rmse * 100,  # Convert to percentage
            'curve_correlation': corr,
            'r_squared': r2,
            'weighted_accuracy': weighted_accuracy,
            'weighted_f1': weighted_f1,
            'sim_change': sim_change * 100,  # Convert to percentage
            'gt_change': gt_change * 100,  # Convert to percentage
            'change_error': change_error * 100,  # Convert to percentage
            'composite_score': composite_score
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Save summary data
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter3.csv'), index=False)
    print(f"Scenario vs ground truth evaluation summary saved to: {os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_evaluation_iter3.csv')}")
    
    # Create performance radar chart
    plt.figure(figsize=(10, 8))
    
    # Set radar chart style
    metrics_to_plot = ['avg_accuracy', 'avg_f1', 'curve_correlation', 'r_squared', 'weighted_accuracy', 'weighted_f1']
    metrics_labels = ['Accuracy', 'F1 Score', 'Curve Correlation', 'R²', 'Weighted Accuracy', 'Weighted F1']
    
    # Calculate angles
    N = len(metrics_to_plot)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the radar chart
    
    # Initialize radar chart
    ax = plt.subplot(111, polar=True)
    
    # Set label positions
    plt.xticks(angles[:-1], metrics_labels, size=12)
    
    # Draw radar chart for each scenario
    for scenario in summary_df['scenario']:
        values = summary_df[summary_df['scenario'] == scenario][metrics_to_plot].values.flatten().tolist()
        
        # Normalize correlation and r-squared from [-1,1] to [0,1]
        normalized_values = []
        for i, val in enumerate(values):
            if metrics_to_plot[i] in ['curve_correlation', 'r_squared']:
                # Convert from [-1,1] to [0,1] for correlation
                normalized_values.append((val + 1) / 2 if metrics_to_plot[i] == 'curve_correlation' else val)
            else:
                normalized_values.append(val)
        
        # Close the radar chart
        normalized_values += normalized_values[:1]
        
        # Draw radar chart
        ax.plot(angles, normalized_values, linewidth=2, label=scenario, color=scenario_colors[scenario])
        ax.fill(angles, normalized_values, alpha=0.1, color=scenario_colors[scenario])
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Scenario Performance vs Ground Truth (Iter3)', size=15)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_radar_iter3.png'), dpi=300)
    print(f"Scenario performance radar chart saved to: {os.path.join(OUTPUT_DIR, 'scenario_vs_ground_truth_radar_iter3.png')}")
    plt.close()
    
    # Create bar chart showing key metrics
    plt.figure(figsize=(14, 10))
    
    # Metrics to display
    display_metrics = ['avg_accuracy', 'avg_f1', 'time_series_rmse', 'curve_correlation', 'composite_score']
    display_titles = ['Accuracy', 'F1 Score', 'RMSE (%)', 'Curve Correlation', 'Composite Score']
    
    # Create subplot for each metric
    for i, (metric, title) in enumerate(zip(display_metrics, display_titles)):
        plt.subplot(2, 3, i+1)
        
        # Get values
        scenarios = summary_df['scenario']
        values = summary_df[metric]
        
        # Create bar colors
        colors = [scenario_colors[s] for s in scenarios]
        
        # Create bar chart
        bars = plt.bar(scenarios, values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}' if metric in ['curve_correlation', 'r_squared', 'composite_score', 'avg_accuracy', 'avg_f1'] 
                    else f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Add titles and labels
        plt.title(title)
        plt.ylabel(title)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scenario_metrics_comparison_iter3.png'), dpi=300)
    print(f"Scenario metrics comparison chart saved to: {os.path.join(OUTPUT_DIR, 'scenario_metrics_comparison_iter3.png')}")
    plt.close()
    
    return summary_df

def main():
    """Main function comparing simulation results with ground truth data"""
    print("\n===== Starting Iter3 Ground Truth Evaluation =====\n")
    
    # Load daily data from simulation
    daily_data = load_daily_data()
    if daily_data is None:
        return
    
    # Load ground truth data
    ground_truth = load_ground_truth_data()
    if ground_truth is None:
        return
    
    # Calculate daily metrics compared to ground truth
    daily_metrics = calculate_daily_metrics(daily_data, ground_truth)
    if daily_metrics is None:
        return
    
    # Visualize metrics over time
    visualize_metrics_over_time(daily_metrics)
    
    # Calculate scenario overall performance
    scenario_performance = calculate_scenario_performance(daily_metrics)
    
    # Print evaluation results
    print("\n===== Iter3 Scenario Evaluation vs Ground Truth =====")
    for _, row in scenario_performance.iterrows():
        scenario = row['scenario']
        print(f"\nScenario: {scenario}")
        print(f"  Average Simulation Rate: {row['avg_sim_rate']:.2f}%")
        print(f"  Average Ground Truth Rate: {row['avg_gt_rate']:.2f}%")
        print(f"  Average Accuracy: {row['avg_accuracy']:.4f}")
        print(f"  Average F1 Score: {row['avg_f1']:.4f}")
        print(f"  Average Error: {row['avg_error']:.2f}%")
        print(f"  Time Series RMSE: {row['time_series_rmse']:.2f}%")
        print(f"  Curve Correlation: {row['curve_correlation']:.4f}")
        print(f"  R-squared: {row['r_squared']:.4f}")
        print(f"  Simulation Behavior Change: {row['sim_change']:.2f}%")
        print(f"  Ground Truth Behavior Change: {row['gt_change']:.2f}%")
        print(f"  Composite Score: {row['composite_score']:.4f}")
    
    # Determine best scenario
    best_scenario = scenario_performance.loc[scenario_performance['composite_score'].idxmax()]['scenario']
    print(f"\nBased on ground truth evaluation for Iter3, the best scenario is: {best_scenario}")
    
    print("\n===== Iter3 Ground Truth Evaluation Complete =====\n")

if __name__ == "__main__":
    main()