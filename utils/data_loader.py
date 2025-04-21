"""
Utilities for loading and analyzing data for simulations.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple

class DataLoader:
    """
    Utility class for loading and preprocessing data for simulations.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data directory or file
        """
        self.data_path = data_path
        self.loaded_data = {}
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pandas.read_csv
        
        Returns:
            Pandas DataFrame containing the data
        """
        full_path = os.path.join(self.data_path, file_path)
        df = pd.read_csv(full_path, **kwargs)
        self.loaded_data[file_path] = df
        return df
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            Dictionary containing the data
        """
        full_path = os.path.join(self.data_path, file_path)
        with open(full_path, 'r') as f:
            data = json.load(f)
        self.loaded_data[file_path] = data
        return data
    
    def load_geojson(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from a GeoJSON file.
        
        Args:
            file_path: Path to the GeoJSON file
        
        Returns:
            Dictionary containing the data
        """
        return self.load_json(file_path)
    
    def get_data(self, file_path: str) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Get previously loaded data.
        
        Args:
            file_path: Path to the data file
        
        Returns:
            Previously loaded data
        """
        if file_path in self.loaded_data:
            return self.loaded_data[file_path]
        else:
            raise KeyError(f"Data from {file_path} has not been loaded")

class DataAnalyzer:
    """
    Utility class for analyzing data for simulations.
    """
    
    @staticmethod
    def analyze_numeric_distribution(data: Union[np.ndarray, pd.Series, List[float]]) -> Dict[str, float]:
        """
        Analyze the distribution of numeric data.
        
        Args:
            data: Numeric data to analyze
        
        Returns:
            Dictionary containing distribution statistics
        """
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
    
    @staticmethod
    def analyze_categorical_distribution(data: Union[np.ndarray, pd.Series, List[str]]) -> Dict[str, int]:
        """
        Analyze the distribution of categorical data.
        
        Args:
            data: Categorical data to analyze
        
        Returns:
            Dictionary containing category counts
        """
        if isinstance(data, np.ndarray) or isinstance(data, list):
            data = pd.Series(data)
        
        counts = data.value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}
    
    @staticmethod
    def analyze_time_series(
        data: Union[np.ndarray, pd.Series, List[float]],
        time: Optional[Union[np.ndarray, pd.Series, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze time series data.
        
        Args:
            data: Time series data to analyze
            time: Time points (optional)
        
        Returns:
            Dictionary containing time series statistics
        """
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        if time is None:
            time = np.arange(len(data))
        elif isinstance(time, pd.Series):
            time = time.values
        elif isinstance(time, list):
            time = np.array(time)
        
        # Basic time series statistics
        result = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "trend": float(np.polyfit(time, data, 1)[0]),
            "seasonal": False,
            "length": len(data)
        }
        
        # Check for seasonality using autocorrelation
        if len(data) >= 10:
            autocorr = np.correlate(data, data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Look for peaks in autocorrelation
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                    peaks.append((i, autocorr[i]))
            
            if peaks:
                result["seasonal"] = True
                result["seasonal_period"] = peaks[0][0]
        
        return result
    
    @staticmethod
    def extract_patterns(
        data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Extract patterns from data using simple correlations.
        
        Args:
            data: DataFrame containing the data
            target_column: Name of the target column
            feature_columns: Names of the feature columns
        
        Returns:
            Dictionary containing pattern information
        """
        result = {"correlations": {}}
        
        # Calculate correlations
        for feature in feature_columns:
            if pd.api.types.is_numeric_dtype(data[feature]) and pd.api.types.is_numeric_dtype(data[target_column]):
                corr = data[feature].corr(data[target_column])
                result["correlations"][feature] = float(corr)
        
        # Sort correlations
        result["top_features"] = sorted(
            result["correlations"].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Extract top 3 features
        result["top_features"] = [(k, v) for k, v in result["top_features"][:3]]
        
        return result 