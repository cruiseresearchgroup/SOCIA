import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.getcwd())

# Import the DataAnalyzer class to inspect its methods
try:
    from utils.data_loader import DataAnalyzer
except ImportError:
    print("Could not import DataAnalyzer. Using a simplified version for testing.")
    
    class DataAnalyzer:
        @staticmethod
        def analyze_numeric_distribution(data):
            return {"mean": data.mean(), "std": data.std()}
            
        @staticmethod
        def analyze_categorical_distribution(data):
            return {"categories": data.unique().tolist(), "counts": data.value_counts().to_dict()}
            
        @staticmethod
        def extract_patterns(data, target_col, feature_cols):
            return {"correlations": {}}

# Define the paths to the CSV files
agent_attributes_path = 'data_fitting/mask_adoption_data/agent_attributes.csv'
train_data_path = 'data_fitting/mask_adoption_data/train_data.csv'

def simulate_data_analysis(df, file_name):
    """Simulate the data analysis process to identify the boolean subtraction error."""
    print(f"\n--- Analyzing {file_name} to identify boolean subtraction errors ---")
    
    # Check for boolean columns
    bool_columns = df.select_dtypes(include=['bool']).columns.tolist()
    print(f"Boolean columns found: {bool_columns}")
    
    # For each column in the dataframe
    for col in df.columns:
        col_data = df[col]
        print(f"\nAnalyzing column: {col} (type: {col_data.dtype})")
        
        # Check if it's a boolean column
        if col_data.dtype == 'bool':
            try:
                # Try to apply numeric operations that might cause errors
                print("Testing operations that might cause boolean subtract errors:")
                
                # Test 1: Try to use subtraction on boolean values directly
                try:
                    result = col_data - False
                    print("Test 1 (col_data - False): Passed (no error)")
                except Exception as e:
                    print(f"Test 1 (col_data - False): Failed with error: {str(e)}")
                
                # Test 2: Try to calculate differences (which might use subtraction internally)
                try:
                    diff = col_data.diff()
                    print("Test 2 (col_data.diff()): Passed (no error)")
                except Exception as e:
                    print(f"Test 2 (col_data.diff()): Failed with error: {str(e)}")
                
                # Test 3: Test statistical functions that might use subtraction
                try:
                    # Simulate what DataAnalyzer might be doing
                    result = DataAnalyzer.analyze_numeric_distribution(col_data)
                    print("Test 3 (DataAnalyzer.analyze_numeric_distribution): Passed (no error)")
                except Exception as e:
                    print(f"Test 3 (DataAnalyzer.analyze_numeric_distribution): Failed with error: {str(e)}")
                    
            except Exception as e:
                print(f"Error during testing: {str(e)}")
        
        # For numeric columns, also test if there might be boolean values causing issues
        elif pd.api.types.is_numeric_dtype(col_data):
            # Check if there are only 0s and 1s which might be treated as booleans
            unique_values = col_data.unique()
            if set(unique_values).issubset({0, 1, np.nan}):
                print(f"Warning: Column {col} contains only 0s and 1s, which might be implicitly converted to booleans.")
    
    print(f"\n--- Analysis complete for {file_name} ---")
    return df

def fix_data_analyzer():
    """Create a fixed version of the DataAnalyzer class that handles boolean data correctly."""
    print("\n--- Creating Fixed DataAnalyzer Class ---")
    
    class FixedDataAnalyzer:
        @staticmethod
        def analyze_numeric_distribution(data: pd.Series) -> dict:
            """
            Fixed version that properly handles boolean data by converting to int before analysis.
            """
            # Check if data is boolean and convert to int if needed
            if hasattr(data, 'dtype') and data.dtype == 'bool':
                print("Converting boolean data to integer for analysis")
                # Convert boolean to integer (True=1, False=0)
                data = data.astype(int)
            
            if isinstance(data, pd.Series):
                data = data.values
            elif isinstance(data, list):
                data = np.array(data)
            
            return {
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "q25": float(np.percentile(data, 25)),
                "q75": float(np.percentile(data, 75))
            }
    
    # Test the fixed version on boolean data
    print("Testing fixed version of analyze_numeric_distribution...")
    
    # Create a test boolean series
    test_bool_series = pd.Series([True, False, True, True, False])
    
    try:
        # Test the fixed version
        result = FixedDataAnalyzer.analyze_numeric_distribution(test_bool_series)
        print("Fixed version works correctly! Result:", result)
        
        # Output recommended code fix
        print("\nRECOMMENDED FIX:")
        print("Modify the analyze_numeric_distribution method in utils/data_loader.py:")
        print("""
@staticmethod
def analyze_numeric_distribution(data: Union[np.ndarray, pd.Series, List[float]]) -> Dict[str, float]:
    # Check if data is boolean and convert to int if needed
    if hasattr(data, 'dtype') and data.dtype == 'bool':
        # Convert boolean to integer (True=1, False=0) 
        data = data.astype(int)
        
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data)
    
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75))
    }
""")
    except Exception as e:
        print(f"Fixed version still has an error: {str(e)}")

def main():
    # Read the agent_attributes.csv file
    try:
        agent_attributes_df = pd.read_csv(agent_attributes_path)
        print("Contents of agent_attributes.csv:")
        print(agent_attributes_df.head())
        agent_attributes_df = simulate_data_analysis(agent_attributes_df, 'agent_attributes.csv')
    except Exception as e:
        print(f"Error reading agent_attributes.csv: {e}")

    # Read the train_data.csv file
    try:
        train_data_df = pd.read_csv(train_data_path)
        print("\nContents of train_data.csv:")
        print(train_data_df.head())
        train_data_df = simulate_data_analysis(train_data_df, 'train_data.csv')
    except Exception as e:
        print(f"Error reading train_data.csv: {e}")
    
    # Create and test the fixed DataAnalyzer
    fix_data_analyzer()

if __name__ == "__main__":
    # main()
    import numpy as np

    data = [0.4801, 0.4820, 0.4815, 0.4805, 0.4823]  # 5次实验结果

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # 标准差
    sem = std_dev / np.sqrt(len(data))  # 标准误

    print("Mean:", mean)
    print("Standard Deviation (SD):", std_dev)
    print("Standard Error (SEM):", sem)
