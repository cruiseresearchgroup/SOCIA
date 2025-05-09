import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Use the same paths as the simulation code
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Set output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")

# Data file paths
test_file = os.path.join(DATA_DIR, "test_data.csv")

# New iter4 result files
baseline_iter4_file = os.path.join(OUTPUT_DIR, "baseline_results_iter4.csv")
high_risk_iter4_file = os.path.join(OUTPUT_DIR, "high_risk_results_iter4.csv")
intervention_iter4_file = os.path.join(OUTPUT_DIR, "intervention_results_iter4.csv")

def evaluate_model():
    """
    Evaluate the iter4 model performance against test data for days 30-39.
    Calculate accuracy, precision, recall, F1 score and other relevant metrics.
    """
    print("\n============ MASK ADOPTION MODEL EVALUATION - ITER4 (DAYS 30-39) ============\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    try:
        test_data = pd.read_csv(test_file)
        print(f"Loaded {len(test_data)} test data points")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Filter test data to only include days 30-39
    test_data = test_data[(test_data['day'] >= 30) & (test_data['day'] <= 39)]
    print(f"Filtered to {len(test_data)} test data points for days 30-39")
    
    # Load iter4 model results
    try:
        baseline_iter4 = pd.read_csv(baseline_iter4_file)
        high_risk_iter4 = pd.read_csv(high_risk_iter4_file)
        intervention_iter4 = pd.read_csv(intervention_iter4_file)
        
        # Add scenario column for identification
        baseline_iter4['scenario'] = 'baseline'
        high_risk_iter4['scenario'] = 'high_risk'
        intervention_iter4['scenario'] = 'intervention'
        
        # Combine results
        all_iter4_results = pd.concat([baseline_iter4, high_risk_iter4, intervention_iter4])
        print(f"Loaded {len(all_iter4_results)} model predictions for iter4")
        
    except Exception as e:
        print(f"Error loading iter4 model predictions: {e}")
        return
    
    # Extract all days we need to evaluate
    days = sorted(test_data['day'].unique())
    print(f"Evaluating model across days: {', '.join(map(str, days))}")
    
    # Scenarios to evaluate
    scenarios = ['baseline', 'high_risk', 'intervention']
    
    # Dictionary to store aggregate metrics for each scenario
    metrics = {scenario: {} for scenario in scenarios}
    
    # Evaluate iter4 models
    print("\n=== EVALUATING ITER4 MODELS ===")
    for scenario in scenarios:
        print(f"\nEvaluating {scenario} scenario")
        
        # Get results for this scenario
        scenario_results = all_iter4_results[all_iter4_results['scenario'] == scenario]
        
        # Prepare for aggregate metrics
        all_true_values = []
        all_pred_values = []
        
        # Evaluate day by day
        daily_metrics = {}
        for day in days:
            # Get test data for this day
            day_test_data = test_data[test_data['day'] == day]
            
            # Merge with model predictions
            evaluation_data = pd.merge(
                day_test_data[['agent_id', 'wearing_mask']],
                scenario_results[['agent_id', 'mask_wearing_status']],
                on='agent_id',
                how='inner'
            )
            
            if len(evaluation_data) == 0:
                print(f"  No matching data found for day {day}")
                continue
                
            # Convert boolean to int for metrics calculation
            if evaluation_data['wearing_mask'].dtype == bool:
                evaluation_data['wearing_mask'] = evaluation_data['wearing_mask'].astype(int)
            if evaluation_data['mask_wearing_status'].dtype == bool:
                evaluation_data['mask_wearing_status'] = evaluation_data['mask_wearing_status'].astype(int)
            
            # Calculate metrics for this day
            y_true = evaluation_data['wearing_mask']
            y_pred = evaluation_data['mask_wearing_status']
            
            # Add to aggregates
            all_true_values.extend(y_true.tolist())
            all_pred_values.extend(y_pred.tolist())
            
            # Store daily metrics
            daily_metrics[day] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred),
                'actual_positive_rate': y_true.mean(),
                'predicted_positive_rate': y_pred.mean()
            }
            
        # Calculate aggregate metrics
        if all_true_values and all_pred_values:
            all_true = np.array(all_true_values)
            all_pred = np.array(all_pred_values)
            
            aggregate = {
                'accuracy': accuracy_score(all_true, all_pred),
                'precision': precision_score(all_true, all_pred, zero_division=0),
                'recall': recall_score(all_true, all_pred, zero_division=0),
                'f1': f1_score(all_true, all_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(all_true, all_pred),
                'actual_positive_rate': all_true.mean(),
                'predicted_positive_rate': all_pred.mean(),
                'total_samples': len(all_true),
                'daily_metrics': daily_metrics
            }
            
            metrics[scenario] = aggregate
            
            # Print aggregate metrics
            print(f"Aggregate metrics for {scenario}:")
            print(f"  Accuracy:  {aggregate['accuracy']:.4f}")
            print(f"  Precision: {aggregate['precision']:.4f}")
            print(f"  Recall:    {aggregate['recall']:.4f}")
            print(f"  F1 Score:  {aggregate['f1']:.4f}")
            print(f"  Actual Positive Rate:    {aggregate['actual_positive_rate']:.4f}")
            print(f"  Predicted Positive Rate: {aggregate['predicted_positive_rate']:.4f}")
    
    # Generate visualizations
    print("\nGenerating evaluation visualizations...")
    
    # Compare scenarios
    plt.figure(figsize=(12, 8))
    
    # Data for plotting
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(metric_labels))
    width = 0.2
    
    colors = ['blue', 'red', 'green']
    
    for i, scenario in enumerate(scenarios):
        if scenario in metrics:
            values = [metrics[scenario][k] for k in metric_keys]
            plt.bar(x + (i-1)*width, values, width, label=scenario.capitalize(), color=colors[i])
            
            # Add value labels
            for j, v in enumerate(values):
                plt.text(x[j] + (i-1)*width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.ylabel('Score')
    plt.title('Model Performance Metrics by Scenario')
    plt.xticks(x, metric_labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iter4_performance_metrics.png'))
    
    # Compare actual vs predicted rates
    plt.figure(figsize=(10, 6))
    
    scenarios_nice = [s.capitalize() for s in scenarios if s in metrics]
    actual_rates = [metrics[s]['actual_positive_rate'] for s in scenarios if s in metrics]
    predicted_rates = [metrics[s]['predicted_positive_rate'] for s in scenarios if s in metrics]
    
    x = np.arange(len(scenarios_nice))
    width = 0.35
    
    plt.bar(x - width/2, actual_rates, width, label='Actual Rate', color='green')
    plt.bar(x + width/2, predicted_rates, width, label='Predicted Rate', color='blue')
    
    plt.title('Mask Wearing Rates Comparison')
    plt.ylabel('Rate')
    plt.xticks(x, scenarios_nice)
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(actual_rates):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(predicted_rates):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iter4_rates_comparison.png'))
    
    # Generate confusion matrices
    plt.figure(figsize=(15, 5))
    
    for i, scenario in enumerate(scenarios, 1):
        if scenario not in metrics:
            continue
            
        plt.subplot(1, 3, i)
        cm = metrics[scenario]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Mask', 'Mask'], 
                    yticklabels=['No Mask', 'Mask'])
        
        plt.title(f'{scenario.capitalize()} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iter4_confusion_matrices.png'))
    
    # For baseline: plot daily metrics over time
    if 'baseline' in metrics and 'daily_metrics' in metrics['baseline']:
        daily = metrics['baseline']['daily_metrics']
        
        plt.figure(figsize=(12, 6))
        
        days = sorted(daily.keys())
        actual_rates = [daily[day]['actual_positive_rate'] for day in days]
        predicted_rates = [daily[day]['predicted_positive_rate'] for day in days]
        
        plt.plot(days, actual_rates, 'o-', label='Actual Rate', color='green')
        plt.plot(days, predicted_rates, 's-', label='Predicted Rate', color='blue')
        
        plt.title('Daily Mask Wearing Rates (Baseline Scenario)')
        plt.xlabel('Day')
        plt.ylabel('Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'iter4_baseline_daily_rates.png'))
    
    # Save summary results to CSV
    summary_data = []
    for scenario in scenarios:
        if scenario in metrics:
            summary_data.append({
                'scenario': scenario,
                'actual_rate': metrics[scenario]['actual_positive_rate'],
                'predicted_rate': metrics[scenario]['predicted_positive_rate'],
                'accuracy': metrics[scenario]['accuracy'],
                'precision': metrics[scenario]['precision'],
                'recall': metrics[scenario]['recall'],
                'f1': metrics[scenario]['f1'],
                'error': abs(metrics[scenario]['actual_positive_rate'] - metrics[scenario]['predicted_positive_rate']),
                'total_samples': metrics[scenario]['total_samples']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'iter4_evaluation_summary.csv'), index=False)
    
    print(f"\nEvaluation visualizations and results saved to {OUTPUT_DIR}")
    
    return metrics

if __name__ == "__main__":
    evaluate_model() 