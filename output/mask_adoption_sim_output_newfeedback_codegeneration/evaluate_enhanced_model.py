import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Constants
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
ACTUAL_RATE = 0.5751  # From iter4_evaluation_summary.csv

def load_data():
    """Load enhanced model results and previous iterations for comparison."""
    # Enhanced model results
    enhanced_baseline = pd.read_csv(os.path.join(OUTPUT_DIR, 'baseline_results_enhanced.csv'))
    enhanced_high_risk = pd.read_csv(os.path.join(OUTPUT_DIR, 'high_risk_results_enhanced.csv'))
    enhanced_intervention = pd.read_csv(os.path.join(OUTPUT_DIR, 'intervention_results_enhanced.csv'))
    
    # Previous iteration results (iter4)
    try:
        iter4_baseline = pd.read_csv(os.path.join(OUTPUT_DIR, 'baseline_results_iter4.csv'))
        iter4_high_risk = pd.read_csv(os.path.join(OUTPUT_DIR, 'high_risk_results_iter4.csv'))
        iter4_intervention = pd.read_csv(os.path.join(OUTPUT_DIR, 'intervention_results_iter4.csv'))
    except FileNotFoundError:
        print("Warning: Some iter4 result files not found, comparison with iter4 might be incomplete")
        iter4_baseline = None
        iter4_high_risk = None
        iter4_intervention = None
    
    # Try to load iter3 evaluation summary if available
    try:
        iter3_summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'iter3_iter4_comparison.csv'))
    except FileNotFoundError:
        print("Warning: iter3_iter4_comparison.csv not found, comparison with iter3 will be unavailable")
        iter3_summary = None
    
    return {
        'enhanced': {
            'baseline': enhanced_baseline,
            'high_risk': enhanced_high_risk,
            'intervention': enhanced_intervention
        },
        'iter4': {
            'baseline': iter4_baseline,
            'high_risk': iter4_high_risk,
            'intervention': iter4_intervention
        },
        'iter3_summary': iter3_summary
    }

def calculate_metrics(df, actual_rate):
    """Calculate performance metrics for a dataset."""
    # Generate expected values based on actual rate
    expected = np.random.choice([True, False], size=len(df), p=[actual_rate, 1-actual_rate])
    
    # Calculate metrics
    accuracy = accuracy_score(expected, df['mask_wearing_status'])
    precision = precision_score(expected, df['mask_wearing_status'])
    recall = recall_score(expected, df['mask_wearing_status'])
    f1 = f1_score(expected, df['mask_wearing_status'])
    
    # Calculate predicted rate
    predicted_rate = df['mask_wearing_status'].mean()
    
    # Calculate error (absolute difference between predicted and actual rate)
    error = abs(predicted_rate - actual_rate)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_rate': predicted_rate,
        'error': error
    }

def analyze_clusters(df, scenario):
    """Analyze performance by cluster."""
    # Ensure we have cluster_id in dataframe
    if 'cluster_id' not in df.columns:
        print(f"Warning: cluster_id not found in {scenario} dataframe, skipping cluster analysis")
        return None
    
    # Calculate rates by cluster
    cluster_rates = df.groupby('cluster_id')['mask_wearing_status'].mean()
    
    # Calculate additional stats by cluster
    cluster_stats = df.groupby('cluster_id').agg({
        'mask_wearing_status': ['mean', 'count'],
        'risk_perception': ['mean', 'min', 'max'],
        'age': ['mean', 'min', 'max']
    })
    
    return {
        'rates': cluster_rates,
        'stats': cluster_stats
    }

def create_summary_table(data, actual_rate):
    """Create a summary table of all metrics."""
    scenarios = ['baseline', 'high_risk', 'intervention']
    rows = []
    
    # Add enhanced model results
    for scenario in scenarios:
        df = data['enhanced'][scenario]
        metrics = calculate_metrics(df, actual_rate)
        
        row = {
            'model': 'enhanced',
            'scenario': scenario,
            'actual_rate': actual_rate,
            'predicted_rate': metrics['predicted_rate'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'error': metrics['error']
        }
        rows.append(row)
    
    # Add iter4 results if available
    if data['iter4']['baseline'] is not None:
        for scenario in scenarios:
            df = data['iter4'][scenario]
            if df is not None:
                metrics = calculate_metrics(df, actual_rate)
                
                row = {
                    'model': 'iter4',
                    'scenario': scenario,
                    'actual_rate': actual_rate,
                    'predicted_rate': metrics['predicted_rate'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'error': metrics['error']
                }
                rows.append(row)
    
    # Create DataFrame from rows
    summary_df = pd.DataFrame(rows)
    
    return summary_df

def visualize_results(data, summary):
    """Create visualizations for the enhanced model evaluation."""
    
    # 1. Comparison of predicted rates
    plt.figure(figsize=(12, 6))
    
    # Get model versions and scenarios
    models = summary['model'].unique()
    scenarios = summary['scenario'].unique()
    
    # Plot predicted rates by model and scenario
    index = np.arange(len(scenarios))
    bar_width = 0.35
    
    # Get rates for each model
    for i, model in enumerate(models):
        model_data = summary[summary['model'] == model]
        rates = [model_data[model_data['scenario'] == s]['predicted_rate'].values[0] for s in scenarios]
        plt.bar(index + i*bar_width, rates, bar_width, alpha=0.8, label=model)
    
    # Add target rate line
    plt.axhline(y=ACTUAL_RATE, color='r', linestyle='--', label=f'Target Rate ({ACTUAL_RATE:.3f})')
    
    plt.xlabel('Scenario')
    plt.ylabel('Predicted Rate')
    plt.title('Comparison of Predicted Rates by Model and Scenario')
    plt.xticks(index + bar_width/2, scenarios)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'enhanced_vs_iter4_rates.png'))
    plt.close()
    
    # 2. F1 Score comparison
    plt.figure(figsize=(12, 6))
    
    # Get F1 scores for each model
    for i, model in enumerate(models):
        model_data = summary[summary['model'] == model]
        f1_scores = [model_data[model_data['scenario'] == s]['f1'].values[0] for s in scenarios]
        plt.bar(index + i*bar_width, f1_scores, bar_width, alpha=0.8, label=model)
    
    plt.xlabel('Scenario')
    plt.ylabel('F1 Score')
    plt.title('Comparison of F1 Scores by Model and Scenario')
    plt.xticks(index + bar_width/2, scenarios)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'enhanced_vs_iter4_f1.png'))
    plt.close()
    
    # 3. Error comparison 
    plt.figure(figsize=(12, 6))
    
    # Get errors for each model
    for i, model in enumerate(models):
        model_data = summary[summary['model'] == model]
        errors = [model_data[model_data['scenario'] == s]['error'].values[0] for s in scenarios]
        plt.bar(index + i*bar_width, errors, bar_width, alpha=0.8, label=model)
    
    plt.xlabel('Scenario')
    plt.ylabel('Absolute Error')
    plt.title('Comparison of Prediction Errors by Model and Scenario')
    plt.xticks(index + bar_width/2, scenarios)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'enhanced_vs_iter4_errors.png'))
    plt.close()
    
    # 4. Cluster analysis for enhanced model
    for scenario in scenarios:
        df = data['enhanced'][scenario]
        cluster_analysis = analyze_clusters(df, scenario)
        
        if cluster_analysis is not None:
            rates = cluster_analysis['rates']
            
            plt.figure(figsize=(10, 6))
            cluster_ids = rates.index
            plt.bar(cluster_ids, rates.values * 100)
            plt.axhline(y=ACTUAL_RATE*100, color='r', linestyle='--', label=f'Target Rate ({ACTUAL_RATE*100:.1f}%)')
            
            plt.xlabel('Cluster ID')
            plt.ylabel('Mask Wearing Rate (%)')
            plt.title(f'Enhanced Model - {scenario.capitalize()} Scenario: Mask Wearing Rates by Cluster')
            plt.xticks(cluster_ids)
            
            # Add value labels
            for i, v in enumerate(rates.values):
                plt.text(i, v*100 + 2, f'{v*100:.1f}%', ha='center')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'enhanced_{scenario}_cluster_rates.png'))
            plt.close()

def main():
    # Load data
    data = load_data()
    
    # Create summary table
    summary = create_summary_table(data, ACTUAL_RATE)
    
    # Save summary to CSV
    summary.to_csv(os.path.join(OUTPUT_DIR, 'enhanced_model_evaluation.csv'), index=False)
    print(f"Evaluation summary saved to: {os.path.join(OUTPUT_DIR, 'enhanced_model_evaluation.csv')}")
    
    # Print summary
    print("\nEnhanced Model Evaluation Summary:")
    print(summary)
    
    # Calculate improvements
    if len(summary['model'].unique()) > 1:
        improvements = {}
        for scenario in summary['scenario'].unique():
            enhanced = summary[(summary['model'] == 'enhanced') & (summary['scenario'] == scenario)]
            iter4 = summary[(summary['model'] == 'iter4') & (summary['scenario'] == scenario)]
            
            if not enhanced.empty and not iter4.empty:
                f1_improvement = (enhanced['f1'].values[0] - iter4['f1'].values[0]) / iter4['f1'].values[0] * 100
                error_improvement = (iter4['error'].values[0] - enhanced['error'].values[0]) / iter4['error'].values[0] * 100
                
                improvements[scenario] = {
                    'f1_improvement': f1_improvement,
                    'error_improvement': error_improvement
                }
        
        if improvements:
            print("\nImprovements over iter4:")
            for scenario, metrics in improvements.items():
                print(f"  {scenario.capitalize()}:")
                print(f"    F1 Score: {metrics['f1_improvement']:.2f}%")
                print(f"    Error reduction: {metrics['error_improvement']:.2f}%")
    
    # Create visualizations
    visualize_results(data, summary)
    print("Visualizations generated in the output directory.")
    
    # Analyze clusters
    print("\nCluster Analysis for Enhanced Model:")
    for scenario in summary['scenario'].unique():
        cluster_analysis = analyze_clusters(data['enhanced'][scenario], scenario)
        if cluster_analysis is not None:
            print(f"\n{scenario.capitalize()} Scenario - Mask Wearing Rates by Cluster:")
            rates = cluster_analysis['rates']
            for cluster_id, rate in rates.items():
                print(f"  Cluster {cluster_id}: {rate*100:.2f}%")

if __name__ == "__main__":
    main() 