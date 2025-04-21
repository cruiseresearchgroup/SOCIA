"""
DataAnalysisAgent: Analyzes input data to extract patterns and calibration parameters.
"""

import logging
import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional

from agents.base_agent import BaseAgent
from utils.data_loader import DataLoader, DataAnalyzer

class DataAnalysisAgent(BaseAgent):
    """
    Data Analysis Agent analyzes input data to extract patterns, distributions,
    and parameters that can be used to calibrate simulation models.
    
    This agent is responsible for:
    1. Loading and preprocessing the provided data
    2. Identifying key distributions and patterns in the data
    3. Extracting parameters that can be used to configure and calibrate simulations
    4. Providing insights about the data that inform model selection and design
    """
    
    def process(
        self,
        data_path: str,
        task_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the input data and extract insights.
        
        Args:
            data_path: Path to the input data
            task_spec: Task specification from the Task Understanding Agent
        
        Returns:
            Dictionary containing data analysis results
        """
        self.logger.info("Processing input data")
        
        # Create data loader
        data_loader = DataLoader(data_path)
        
        # Identify available data files
        available_files = self._list_available_files(data_path)
        
        # Determine which files to analyze based on task specification
        files_to_analyze = self._select_files_to_analyze(available_files, task_spec)
        
        # Load and analyze each file
        analysis_results = {}
        
        for file_info in files_to_analyze:
            file_path = file_info["path"]
            file_type = file_info["type"]
            
            try:
                # Load data based on file type
                if file_type == "csv":
                    data = data_loader.load_csv(file_path)
                    analysis_results[file_path] = self._analyze_tabular_data(data, task_spec)
                elif file_type == "json":
                    data = data_loader.load_json(file_path)
                    analysis_results[file_path] = self._analyze_json_data(data, task_spec)
                elif file_type == "geojson":
                    data = data_loader.load_geojson(file_path)
                    analysis_results[file_path] = self._analyze_geojson_data(data, task_spec)
                else:
                    self.logger.warning(f"Unsupported file type: {file_type}")
                    continue
            except Exception as e:
                self.logger.error(f"Error analyzing file {file_path}: {e}")
                continue
        
        # Build prompt for LLM to summarize findings
        summary_prompt = self._build_prompt(
            task_spec=task_spec,
            analysis_results=analysis_results
        )
        
        # Call LLM to summarize findings
        llm_response = self._call_llm(summary_prompt)
        
        # Parse LLM response
        summary = self._parse_llm_response(llm_response)
        
        # If summary is not a dictionary (e.g., text response), convert it to one
        if not isinstance(summary, dict):
            summary = {"summary": summary}
        
        # Combine raw analysis results with LLM summary
        result = {
            "summary": summary,
            "raw_analysis": analysis_results,
            "analyzed_files": [f["path"] for f in files_to_analyze]
        }
        
        self.logger.info("Data analysis completed")
        return result
    
    def _list_available_files(self, data_path: str) -> List[Dict[str, str]]:
        """List available files in the data directory."""
        result = []
        
        for root, _, files in os.walk(data_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_path)
                
                if file.endswith('.csv'):
                    result.append({"path": rel_path, "type": "csv"})
                elif file.endswith('.json'):
                    if self._is_geojson(full_path):
                        result.append({"path": rel_path, "type": "geojson"})
                    else:
                        result.append({"path": rel_path, "type": "json"})
                elif file.endswith('.geojson'):
                    result.append({"path": rel_path, "type": "geojson"})
        
        return result
    
    def _is_geojson(self, file_path: str) -> bool:
        """Check if a JSON file is a GeoJSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return "type" in data and "features" in data
        except:
            return False
    
    def _select_files_to_analyze(
        self,
        available_files: List[Dict[str, str]],
        task_spec: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Select which files to analyze based on task specification."""
        # In a real implementation, this would use the task spec to determine
        # which files are relevant. For now, we'll just return all files.
        return available_files
    
    def _analyze_tabular_data(
        self,
        data: pd.DataFrame,
        task_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze tabular data (CSV)."""
        result = {
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "columns": {},
            "relationships": {}
        }
        
        # Analyze each column
        for col in data.columns:
            col_data = data[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                result["columns"][col] = {
                    "type": "numeric",
                    "stats": DataAnalyzer.analyze_numeric_distribution(col_data)
                }
                
                # Check if it's a time series
                if col.lower() in ["time", "date", "timestamp", "datetime"]:
                    result["columns"][col]["time_series"] = True
            
            elif pd.api.types.is_string_dtype(col_data):
                result["columns"][col] = {
                    "type": "categorical",
                    "stats": DataAnalyzer.analyze_categorical_distribution(col_data)
                }
        
        # Identify potential target variables based on task specification
        target_columns = []
        
        # Extract metrics from task specification
        if "metrics" in task_spec:
            for metric in task_spec["metrics"]:
                metric_name = metric["name"]
                # Look for columns that might correspond to this metric
                for col in data.columns:
                    if metric_name.lower() in col.lower():
                        target_columns.append(col)
        
        # Analyze relationships for potential target variables
        for target_col in target_columns:
            if target_col in data.columns and pd.api.types.is_numeric_dtype(data[target_col]):
                feature_cols = [col for col in data.columns if col != target_col and pd.api.types.is_numeric_dtype(data[col])]
                if feature_cols:
                    result["relationships"][target_col] = DataAnalyzer.extract_patterns(
                        data, target_col, feature_cols
                    )
        
        return result
    
    def _analyze_json_data(
        self,
        data: Dict[str, Any],
        task_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze JSON data."""
        # Basic analysis of JSON structure
        result = {
            "structure": self._analyze_json_structure(data),
            "fields": {}
        }
        
        # Analyze specific fields if they can be mapped to task specification
        # This is a simplified implementation
        return result
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Analyze the structure of a JSON object."""
        if current_depth >= max_depth:
            return {"type": type(data).__name__}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "num_fields": len(data),
                "fields": {k: self._analyze_json_structure(v, max_depth, current_depth + 1) for k, v in list(data.items())[:10]}
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample": self._analyze_json_structure(data[0], max_depth, current_depth + 1) if data else None
            }
        else:
            return {"type": type(data).__name__}
    
    def _analyze_geojson_data(
        self,
        data: Dict[str, Any],
        task_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze GeoJSON data."""
        result = {
            "type": data.get("type", "unknown"),
            "num_features": len(data.get("features", [])),
            "properties": {}
        }
        
        # Analyze properties of the first few features
        features = data.get("features", [])
        if features:
            property_names = set()
            for feature in features[:10]:
                properties = feature.get("properties", {})
                property_names.update(properties.keys())
            
            result["property_names"] = list(property_names)
        
        return result 