"""
DataAnalysisAgent: Analyzes input data to extract patterns and calibration parameters.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import jsonschema
from jsonschema import validate, ValidationError

from agents.base_agent import BaseAgent
from utils.data_loader import DataLoader

class DataAnalysisAgent(BaseAgent):
    """
    Data Analysis Agent leverages LLM capabilities to analyze data, understand patterns,
    and extract parameters that can be used to calibrate simulation models.
    
    This agent is responsible for:
    1. Loading and integrity-checking of data (missing values, outliers) without modification
    2. Identifying key distributions and patterns in the data
    3. Extracting parameters that can be used to configure and calibrate simulations
    4. Using LLM to provide insights about how the data should inform model design and calibration
    """
    
    def __init__(self, config: Any, output_path: Optional[str] = None):
        super().__init__(config)
        # Base output path for persisting processed data
        self.output_path = output_path or os.getcwd()
    
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
            Dictionary containing data analysis results and calibration recommendations
        """
        self.logger.info(f"Processing input data from path: {data_path}")
        
        # Check if data path exists
        if not os.path.isdir(data_path):
            error_msg = f"Data path invalid or missing: {data_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        self.logger.info(f"Successfully verified data path exists: {data_path}")
        
        # Capture task description for semantic summaries
        task_description = task_spec.get("description", "No task description provided")
        
        # Create data loader
        data_loader = DataLoader(data_path)
        
        # Identify available data files
        available_files = self._list_available_files(data_path)
        self.logger.info(f"Found {len(available_files)} files in data path")
        
        # Determine which files to analyze based on task specification
        files_to_analyze = self._select_files_to_analyze(available_files, task_spec)
        self.logger.info(f"Selected {len(files_to_analyze)} files for analysis")
        
        # Check if data_files is specified in task_spec
        if task_spec and "data_files" in task_spec:
            expected_files = set(task_spec["data_files"].keys())
            found_files = {os.path.basename(f["path"]) for f in available_files}
            self.logger.info(f"Expected files: {expected_files}")
            self.logger.info(f"Found files (basename): {found_files}")
            missing_files = expected_files - found_files
            
            if missing_files:
                error_msg = f"Expected data files missing: {missing_files}"
                self.logger.error(error_msg)
                # Stop processing immediately if required files are missing
                raise FileNotFoundError(error_msg)
            else:
                self.logger.info(f"All expected files found in data directory")
                
            # Extract schema information from task_spec if available
            self.schemas = self._extract_schemas_from_task_spec(task_spec)
        else:
            self.schemas = {}
        
        # Prepare to collect semantic summaries for each file
        file_summaries = []
        
        for file_info in files_to_analyze:
            file_path = file_info["path"]
            file_type = file_info["type"]
            full_path = os.path.join(data_path, file_path)
            basename = os.path.basename(file_path)
            
            self.logger.info(f"Loading file: {file_path} (type: {file_type})")
            
            try:
                # Ensure raw file can load
                if not os.path.exists(full_path):
                    self.logger.error(f"Required file missing: {full_path}")
                    raise FileNotFoundError(full_path)
                
                # Load and integrity-check data based on file type, then generate semantic summary
                if file_type == "csv":
                    raw_data = data_loader.load_csv(file_path)
                    # Validate CSV data (missing values, outliers)
                    self._check_csv(raw_data, basename)
                    summary = self._get_semantic_summary(basename, raw_data, 'csv', task_description)
                    file_summaries.append(summary)
                
                elif file_type == "json":
                    raw_data = data_loader.load_json(file_path)
                    # Validate JSON structure using schema-based approach
                    self._check_json(raw_data, basename, task_spec)
                    summary = self._get_semantic_summary(basename, raw_data, 'json', task_description)
                    file_summaries.append(summary)
                
                elif file_type == "pkl":
                    data = data_loader.load_pickle(file_path)
                    # Validate pickle data structure using schema-based approach
                    self._check_pickle(data, basename, task_spec)
                    summary = self._get_semantic_summary(basename, data, 'pkl', task_description)
                    file_summaries.append(summary)
                
                self.logger.info(f"Successfully processed file: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                # Abort further processing on integrity failure
                self.logger.error("Data integrity check failed, aborting analysis.")
                raise
                continue
        
        # Build prompt for LLM analysis
        metrics = task_spec.get("metrics", [])
        metrics_description = json.dumps(metrics, indent=2) if metrics else "No metrics specified"
        
        # Create context about simulation calibration
        calibration_context = self._create_calibration_context(task_spec, {})
        
        # Build comprehensive prompt
        analysis_prompt = self._build_analysis_prompt(
            task_description=task_description,
            metrics_description=metrics_description,
            file_summaries=file_summaries,
            calibration_context=calibration_context
        )
        
        # Call LLM to analyze data and provide insights
        self.logger.info("Calling LLM to analyze data and provide calibration insights")
        llm_response = self._call_llm(analysis_prompt)
        
        # Parse LLM response to extract structured analysis and recommendations
        analysis_results = self._parse_llm_analysis(llm_response)
        
        # Combine all information into the final result
        result = {
            "data_summary": analysis_results.get("data_summary", {}),
            "simulation_parameters": analysis_results.get("simulation_parameters", {}),
            "calibration_strategy": analysis_results.get("calibration_strategy", {}),
            "file_summaries": file_summaries
        }
        
        self.logger.info("Data analysis completed")
        return result
    
    def _extract_schemas_from_task_spec(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract schema information from task specification if available.
        
        Args:
            task_spec: Task specification which may contain schema information
            
        Returns:
            Dictionary mapping file names to their schemas
        """
        schemas = {}
        
        # Check if schemas are directly provided in task_spec
        if "schemas" in task_spec:
            return task_spec["schemas"]
        
        # Try to infer schemas from data_files descriptions
        if "data_files" in task_spec:
            for file_name, description in task_spec["data_files"].items():
                # If description is a string, try to infer schema from it using LLM
                if isinstance(description, str):
                    schema = self._infer_schema_from_description(file_name, description)
                    if schema:
                        schemas[file_name] = schema
                # If description is already a structured object with schema
                elif isinstance(description, dict) and "schema" in description:
                    schemas[file_name] = description["schema"]
                    
        return schemas
    
    def _infer_schema_from_description(self, file_name: str, description: str) -> Dict[str, Any]:
        """
        Use LLM to infer a basic schema from a file description.
        
        Args:
            file_name: Name of the file
            description: Description of the file content
            
        Returns:
            Inferred JSON schema or None if inference failed
        """
        self.logger.info(f"Inferring schema for {file_name} from description")
        
        # Always use a flexible schema by default - no file name hardcoding
        # Create a simple, permissive schema based on file description keywords
        if "trajectories" in description.lower() or "activities" in description.lower():
            return {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        elif "category" in description.lower() or "categories" in description.lower():
            return {
                "type": "object", 
                "additionalProperties": {"type": "string"}
            }
        elif "coordinates" in description.lower() or "geographic" in description.lower() or "longitude" in description.lower():
            return {
                "type": "object",
                "additionalProperties": {"type": "array"}
            }
        else:
            # Provide a universal permissive schema
            return {"type": "object", "additionalProperties": True}
    
    def _list_available_files(self, data_path: str) -> List[Dict[str, str]]:
        """List available files in the data directory."""
        result = []
        
        self.logger.info(f"Scanning directory for files: {data_path}")
        for root, dirs, files in os.walk(data_path):
            self.logger.info(f"Examining directory: {root}, contains {len(files)} files")
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, data_path)
                
                self.logger.debug(f"Checking file: {full_path}")
                
                if file.endswith('.csv'):
                    result.append({"path": rel_path, "type": "csv"})
                elif file.endswith('.json'):
                    if self._is_geojson(full_path):
                        result.append({"path": rel_path, "type": "geojson"})
                    else:
                        result.append({"path": rel_path, "type": "json"})
                elif file.endswith('.geojson'):
                    result.append({"path": rel_path, "type": "geojson"})
                elif file.lower().endswith('.pkl'):
                    # Pickle files (e.g., network data) - ensure case insensitive matching
                    self.logger.info(f"Found pickle file: {full_path}")
                    result.append({"path": rel_path, "type": "pkl"})
        
        self.logger.info(f"Total files found: {len(result)} in path {data_path}")
        for f in result:
            self.logger.info(f"Found file in result: {f['path']} (type: {f['type']})")
        
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
        # Only analyze files specified in task_spec["data_files"] if provided.
        if task_spec and "data_files" in task_spec:
            expected_files = set(task_spec["data_files"].keys())
            selected = [f for f in available_files if os.path.basename(f["path"]) in expected_files]
            # Log files that are skipped because they are not in the spec
            skipped = [f["path"] for f in available_files if os.path.basename(f["path"]) not in expected_files]
            for skip in skipped:
                self.logger.info(f"Skipping file not in task_spec data_files: {skip}")
            return selected
        # Default: return all files if no data_files key in task_spec
        return available_files
    
    def _create_file_summary(
        self, 
        file_name: str, 
        data: pd.DataFrame,
        file_info: Dict[str, Any]
    ) -> str:
        """
        Create a concise summary of the file for inclusion in the LLM prompt.
        """
        column_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        column_descriptions = file_info.get("column_descriptions", {})
        transformations = file_info.get("transformations", {})
        
        # Create statistical summaries for numeric columns
        stats = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                try:
                    stats[col] = {
                        "min": float(data[col].min()),
                        "max": float(data[col].max()),
                        "mean": float(data[col].mean()),
                        "median": float(data[col].median()),
                        "std": float(data[col].std())
                    }
                except:
                    # Skip if we can't compute statistics
                    pass
        
        # For boolean columns, calculate proportion of True values
        bool_props = {}
        for col in data.columns:
            if pd.api.types.is_bool_dtype(data[col]):
                try:
                    bool_props[col] = float(data[col].mean())  # Proportion of True values
                except:
                    pass
        
        # Combine all information
        summary = {
            "file_name": file_name,
            "purpose": file_info.get("purpose", "Unknown purpose"),
            "num_rows": len(data),
            "num_columns": len(data.columns),
            "column_types": column_types,
            "column_descriptions": column_descriptions,
            "transformations": transformations,
            "statistics": stats,
            "boolean_proportions": bool_props,
            "key_insights": file_info.get("key_insights", [])
        }
        
        return json.dumps(summary, indent=2)
    
    def _get_json_structure(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a simplified representation of the JSON structure.
        """
        if isinstance(json_data, dict):
            return {
                "type": "object",
                "keys": list(json_data.keys()),
                "sample_values": {
                    k: type(v).__name__ for k, v in list(json_data.items())[:5]
                }
            }
        elif isinstance(json_data, list) and json_data:
            return {
                "type": "array",
                "length": len(json_data),
                "sample_item_type": type(json_data[0]).__name__
            }
        else:
            return {"type": type(json_data).__name__}
    
    def _create_json_summary(
        self, 
        file_name: str, 
        json_data: Dict[str, Any],
        file_info: Dict[str, Any]
    ) -> str:
        """
        Create a summary of a JSON file.
        """
        structure = file_info.get("structure", {})
        
        # Create a simplified summary
        summary = {
            "file_name": file_name,
            "type": "json",
            "structure": structure
        }
        
        return json.dumps(summary, indent=2)
    
    def _get_pickle_info(self, data: Any, file_name: str) -> Dict[str, Any]:
        """
        Get information about pickle data.
        """
        data_type = type(data).__name__
        
        if hasattr(data, "shape"):  # For numpy arrays
            info = {
                "type": "numpy_array",
                "shape": str(data.shape),
                "dtype": str(data.dtype)
            }
        elif hasattr(data, "nodes"):  # For networkx graphs
            info = {
                "type": "graph",
                "num_nodes": len(data.nodes),
                "num_edges": len(data.edges)
            }
        elif isinstance(data, dict):
            info = {
                "type": "dictionary",
                "num_keys": len(data),
                "key_types": list(set(type(k).__name__ for k in data.keys()))
            }
        elif isinstance(data, list):
            info = {
                "type": "list",
                "length": len(data)
            }
        else:
            info = {
                "type": data_type
            }
        
        return info
    
    def _create_pickle_summary(
        self, 
        file_name: str, 
        data: Any,
        file_info: Dict[str, Any]
    ) -> str:
        """
        Create a summary of a pickle file.
        """
        # Create a simplified summary
        summary = {
            "file_name": file_name,
            "type": "pickle",
            "info": file_info
        }
        
        return json.dumps(summary, indent=2)
    
    def _create_calibration_context(
        self,
        task_spec: Dict[str, Any],
        file_info_dict: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Create context about simulation calibration for the LLM.
        """
        # Extract relevant information from task specification
        task_description = task_spec.get("description", "")
        metrics = task_spec.get("metrics", [])
        
        # Extract file purposes
        file_purposes = {
            file_name: info.get("purpose", "Unknown purpose")
            for file_name, info in file_info_dict.items()
        }
        
        # Create context
        context = {
            "task_description": task_description,
            "metrics": metrics,
            "file_purposes": file_purposes
        }
        
        return json.dumps(context, indent=2)
    
    def _build_analysis_prompt(
        self,
        task_description: str,
        metrics_description: str,
        file_summaries: List[str],
        calibration_context: str
    ) -> str:
        """
        Build a comprehensive prompt for the LLM to analyze the data and provide insights.
        """
        # Join file summaries with line breaks
        file_summaries_text = "\n\n".join(file_summaries)
        
        prompt = f"""
You are an expert data scientist and simulation modeler. Your task is to analyze data for calibrating a simulation model.

TASK DESCRIPTION:
{task_description}

EVALUATION METRICS:
{metrics_description}

DATA SUMMARIES:
{file_summaries_text}

CALIBRATION CONTEXT:
{calibration_context}

Based on the provided data summaries and task description, please analyze the data and provide insights for simulation model calibration.
Your analysis should cover:

1. What key patterns, distributions, and relationships exist in the data?
2. How should this data be used to calibrate the simulation model?
3. What parameters can be extracted from the data for configuring the simulation?
4. What simulation design recommendations would you make based on this data?

Provide your response in the following JSON format:
```json
{{
  "data_summary": {{
    "key_patterns": [
      {{"name": "Pattern Name", "description": "Description of the pattern", "relevance": "Why this matters for the simulation"}}
    ],
    "key_distributions": [
      {{"name": "Distribution Name", "description": "Description of the distribution", "parameters": "Parameters that define this distribution"}}
    ],
    "key_relationships": [
      {{"variables": ["var1", "var2"], "relationship": "Description of the relationship", "strength": "Description of strength"}}
    ]
  }},
  "simulation_parameters": {{
    "parameter_category_1": {{
      "parameter_name_1": {{
        "value": "Extracted or recommended value",
        "source": "Which data file and features this comes from",
        "confidence": "High/Medium/Low",
        "notes": "Any additional notes about this parameter"
      }}
    }}
  }},
  "calibration_strategy": {{
    "preprocessing_steps": [
      {{"step": "Step description", "purpose": "Why this step is necessary"}}
    ],
    "calibration_approach": "Description of overall approach to calibration",
    "validation_strategy": "How to validate the calibrated model",
    "key_variables_to_calibrate": ["var1", "var2", "var3"]
  }}
}}
```

Provide only valid JSON that can be parsed. Don't include any other explanation or text outside the JSON.
"""
        return prompt
    
    def _parse_llm_analysis(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM's analysis response into a structured format.
        """
        try:
            # Try to extract JSON from the response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                analysis = json.loads(json_str)
                return analysis
            else:
                self.logger.warning("Could not extract JSON from analysis response")
                return {
                    "data_summary": {
                        "key_patterns": [],
                        "key_distributions": [],
                        "key_relationships": []
                    },
                    "simulation_parameters": {},
                    "calibration_strategy": {
                        "preprocessing_steps": [],
                        "calibration_approach": "Error extracting analysis",
                        "validation_strategy": "Error extracting analysis",
                        "key_variables_to_calibrate": []
                    }
                }
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing analysis response: {e}")
            return {
                "data_summary": {
                    "key_patterns": [],
                    "key_distributions": [],
                    "key_relationships": []
                },
                "simulation_parameters": {},
                "calibration_strategy": {
                    "preprocessing_steps": [],
                    "calibration_approach": "Error parsing JSON response",
                    "validation_strategy": "Error parsing JSON response",
                    "key_variables_to_calibrate": []
                }
            }
    
    def _convert_df_to_serializable(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert a pandas DataFrame to a serializable format.
        """
        # For datetime columns, convert to strings
        for col in df.columns:
            if pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        # Convert to dictionary
        return df.to_dict(orient='list')
    
    def _get_semantic_summary(self, file_name: str, data: Any, file_type: str, task_description: str) -> str:
        """
        Use LLM to generate a concise semantic metadata summary for the given data file.
        """
        # Prepare a small sample of the data
        if file_type == 'csv' and hasattr(data, 'head'):
            sample = data.head(5).to_dict(orient='records')
        elif file_type == 'json':
            sample_keys = list(data.keys())[:5]
            sample = {k: data[k] for k in sample_keys}
        elif file_type == 'pkl' and isinstance(data, dict):
            sample_keys = list(data.keys())[:5]
            sample = {k: data[k] for k in sample_keys}
        else:
            sample = str(data)[:500]
        sample_str = json.dumps(sample, indent=2)
        prompt = (
            f"Task Description: {task_description}\n\n"
            f"File: {file_name} (type: {file_type})\n"
            f"Data sample:\n{sample_str}\n\n"
            "Please provide a concise semantic metadata summary of this file in the context of the task, addressing:\n"
            "- Overall data structure and type\n"
            "- Meaning of keys or columns\n"
            "- Relationships or nested elements\n"
            "- How this data should inform simulation entities or interactions\n"
        )
        self.logger.info(f"Generating semantic summary for {file_name}")
        llm_response = self._call_llm(prompt)
        return llm_response.strip()
    
    def _check_csv(self, df: pd.DataFrame, file_name: str) -> None:
        """
        Check CSV for missing values and numeric outliers.
        
        Args:
            df: DataFrame to check
            file_name: Name of the file being checked
        """
        # Missing values
        missing = df.isnull().any()
        missing_cols = [col for col, has in missing.items() if has]
        if missing_cols:
            self.logger.error(f"Missing values in {file_name} columns: {missing_cols}")
            raise ValueError(f"Missing values in columns: {missing_cols}")
            
        # Numeric outliers (3-sigma rule)
        for col in df.select_dtypes(include=[np.number]).columns:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
            if not outliers.empty:
                rows = outliers.index.tolist()[:5]
                self.logger.warning(f"Outliers detected in {file_name} column '{col}' at rows {rows}, continuing analysis.")
                # Do not abort on outliers; proceed with analysis
                
        # Also use LLM to perform more intelligent integrity assessment
        self._llm_integrity_check(df, file_name, "csv")
    
    def _check_json(self, data: Any, file_name: str, task_spec: Dict[str, Any]) -> None:
        """
        Check JSON structure using a schema-based approach.
        
        Args:
            data: JSON data to check
            file_name: Name of the file being checked
            task_spec: Task specification that may contain schema information
        """
        if not isinstance(data, dict) and not isinstance(data, list):
            self.logger.error(f"Expected dict or list in {file_name}, got {type(data).__name__}")
            raise ValueError("Invalid JSON structure")
        
        # First, perform basic structure validation - this applies to all files
        self._basic_json_structure_check(data, file_name)
        
        # Use schema validation in a flexible manner
        try:
            schema = self._get_appropriate_schema(data, file_name)
            self.logger.info(f"Using appropriate schema for {file_name}")
            validate(instance=data, schema=schema)
            self.logger.info(f"Basic schema validation passed for {file_name}")
        except ValidationError as e:
            # Log the warning but continue with basic checks
            self.logger.warning(f"Schema validation issue for {file_name}: {e}")
            self.logger.info(f"Continuing with basic structure checks only")
        
        # Use LLM for integrity check only on small files to avoid rate limiting
        if self._is_small_enough_for_llm(data):
            self._llm_integrity_check(data, file_name, "json")
        else:
            self.logger.info(f"File {file_name} too large for LLM integrity check, skipping")
    
    def _basic_json_structure_check(self, data: Any, file_name: str) -> None:
        """
        Perform basic structure check on JSON data.
        
        Args:
            data: JSON data to check
            file_name: Name of the file being checked
        """
        # Check for empty data
        if isinstance(data, dict) and not data:
            self.logger.warning(f"Empty dictionary in {file_name}")
        elif isinstance(data, list) and not data:
            self.logger.warning(f"Empty list in {file_name}")
            
        # Check for consistency in list items
        if isinstance(data, list) and len(data) > 1:
            first_type = type(data[0])
            if not all(isinstance(item, first_type) for item in data):
                self.logger.warning(f"Inconsistent types in list items in {file_name}")
                
        # For dictionaries, check key types and basic value types
        if isinstance(data, dict):
            # Check if all keys are strings (best practice for JSON)
            if not all(isinstance(k, str) for k in data.keys()):
                self.logger.warning(f"Non-string keys found in {file_name}")
            
            # Sample a few values to check for consistency
            values = list(data.values())[:5]
            if values and all(isinstance(v, type(values[0])) for v in values):
                self.logger.info(f"Consistent value types in sampled keys for {file_name}")
            else:
                self.logger.info(f"Mixed value types in {file_name} (may be expected)")
    
    def _get_appropriate_schema(self, data: Any, file_name: str) -> Dict[str, Any]:
        """
        Determine the appropriate schema based on data structure, not file name.
        
        Args:
            data: The data to analyze
            file_name: The name of the file (used only for logging)
            
        Returns:
            An appropriate JSON schema
        """
        # Analyze data structure to determine appropriate schema
        if isinstance(data, dict):
            # Check what kind of dictionary we're dealing with by examining values
            value_samples = list(data.values())[:5]
            
            # If empty dict, use generic schema
            if not value_samples:
                return {"type": "object", "additionalProperties": True}
                
            # If values are primarily strings, it's likely a mapping or category file
            if all(isinstance(v, str) for v in value_samples):
                self.logger.info(f"Detected dictionary with string values in {file_name}")
                return {
                    "type": "object",
                    "additionalProperties": {"type": "string"}
                }
                
            # If values are primarily arrays/lists, it's likely a collection file
            elif all(isinstance(v, list) for v in value_samples):
                self.logger.info(f"Detected dictionary with array values in {file_name}")
                return {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array"
                    }
                }
                
            # If values are nested dictionaries
            elif all(isinstance(v, dict) for v in value_samples):
                self.logger.info(f"Detected dictionary with nested objects in {file_name}")
                return {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object"
                    }
                }
                
            # Mixed content or other structures
            else:
                self.logger.info(f"Detected dictionary with mixed value types in {file_name}")
                return {
                    "type": "object",
                    "additionalProperties": True
                }
                
        elif isinstance(data, list):
            # For lists, check the type of items
            if not data:
                return {"type": "array", "items": {}}
                
            # If items are dictionaries
            if all(isinstance(item, dict) for item in data[:5]):
                return {"type": "array", "items": {"type": "object"}}
                
            # If items are strings
            elif all(isinstance(item, str) for item in data[:5]):
                return {"type": "array", "items": {"type": "string"}}
                
            # If items are arrays
            elif all(isinstance(item, list) for item in data[:5]):
                return {"type": "array", "items": {"type": "array"}}
                
            # Mixed or other
            else:
                return {"type": "array", "items": {}}
        else:
            # For other types, use a generic schema
            return {"type": [type(data).__name__]}
    
    def _check_pickle(self, data: Any, file_name: str, task_spec: Dict[str, Any]) -> None:
        """
        Check pickle data structure using a schema-based approach.
        
        Args:
            data: Pickle data to check
            file_name: Name of the file being checked
            task_spec: Task specification that may contain schema information
        """
        # For pickle files, perform checks based on the data structure, not file name
        if isinstance(data, dict):
            # Apply the same generic dict validation
            self._basic_json_structure_check(data, file_name)
        elif hasattr(data, "nodes") and hasattr(data, "edges"):  # NetworkX graph
            self.logger.info(f"NetworkX graph detected in {file_name}")
            self.logger.info(f"Graph has {len(data.nodes)} nodes and {len(data.edges)} edges")
        elif hasattr(data, "shape") and hasattr(data, "dtype"):  # Numpy array
            self.logger.info(f"Numpy array detected in {file_name}")
            self.logger.info(f"Array shape: {data.shape}, dtype: {data.dtype}")
        elif isinstance(data, list):
            self.logger.info(f"List with {len(data)} items detected in {file_name}")
        else:
            self.logger.info(f"Pickle contains {type(data).__name__} object in {file_name}")
            
        # Use LLM to check only small data 
        if self._is_small_enough_for_llm(data):
            self._llm_integrity_check(data, file_name, "pickle")
        else:
            self.logger.info(f"Pickle file {file_name} too large for LLM integrity check, skipping")
            
    def _is_small_enough_for_llm(self, data: Any) -> bool:
        """
        Check if the data is small enough to be processed by LLM.
        
        Args:
            data: Data to check
            
        Returns:
            True if the data is small enough, False otherwise
        """
        # Convert to JSON string to estimate size
        try:
            json_str = json.dumps(data)
            # If data is bigger than 10KB, consider it too large for LLM
            if len(json_str) > 10240:
                return False
                
            # If data is a large dictionary or list, check sample instead
            if isinstance(data, dict) and len(data) > 50:
                return False
            elif isinstance(data, list) and len(data) > 50:
                return False
                
            return True
        except:
            # If we can't convert to JSON, assume it's too large
            return False
    
    def _llm_integrity_check(self, data: Any, file_name: str, file_type: str) -> None:
        """
        Use LLM to perform a more intelligent data integrity check.
        
        Args:
            data: Data to check
            file_name: Name of the file being checked
            file_type: Type of the file (csv, json, pickle)
        """
        self.logger.info(f"Performing LLM-based integrity check for {file_name}")
        
        # Prepare data sample for LLM - use smaller samples to avoid rate limits
        if file_type == "csv" and hasattr(data, "head"):
            sample = data.head(3).to_dict(orient="records")
            row_count = len(data)
            column_info = {col: str(dtype) for col, dtype in data.dtypes.items()}
        elif file_type == "json" and isinstance(data, dict):
            # Take just a few keys to avoid large payloads
            sample_keys = list(data.keys())[:3]
            sample = {k: data[k] for k in sample_keys}
            if len(data) > 3:
                sample["__note__"] = f"Sample of {len(data)} total keys"
        elif file_type == "pickle":
            # For pickle, just provide type information
            sample = f"Data type: {type(data).__name__}"
            if isinstance(data, dict):
                sample_keys = list(data.keys())[:3]
                sample = {f"Key type: {type(k).__name__}": f"Value type: {type(v).__name__}" 
                        for k, v in [(k, data[k]) for k in sample_keys]}
        else:
            # Provide basic string representation, with a tight limit
            sample = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
        
        # Convert sample to string with a size limit
        try:
            sample_str = json.dumps(sample, indent=2, default=str)
            # Truncate if too large
            if len(sample_str) > 1000:
                sample_str = sample_str[:1000] + "...(truncated)"
        except:
            sample_str = str(sample)[:200] + "...(truncated)"
        
        prompt = f"""
You are an expert data scientist performing basic data integrity checks. 
Analyze this small data sample and identify any CRITICAL integrity issues only.

File: {file_name} (type: {file_type})
Data sample (limited excerpt):
{sample_str}

Only report CRITICAL issues that would prevent proper data usage.
Ignore minor formatting or style issues.

Provide your assessment in this JSON format:
{{
  "has_critical_issues": true/false,
  "issues": [
    {{
      "description": "brief description of critical issue",
      "recommendation": "how to address this issue"
    }}
  ]
}}

If no critical issues found, return has_critical_issues: false with empty issues array.
"""
        
        # Use retry logic for LLM calls
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                response = self._call_llm(prompt)
                
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    assessment_str = response[json_start:json_end]
                    assessment = json.loads(assessment_str)
                    
                    # Log the assessment results
                    if assessment.get("has_critical_issues", False):
                        for issue in assessment.get("issues", []):
                            description = issue.get("description", "No description")
                            self.logger.warning(f"LLM found critical data issue in {file_name}: {description}")
                            
                        # Only raise exception for critical issues if specified in assessment
                        if assessment.get("has_critical_issues") and len(assessment.get("issues", [])) > 0:
                            # Log warning but don't abort
                            self.logger.warning(f"Critical data issues found in {file_name}, but continuing")
                    else:
                        self.logger.info(f"LLM integrity check passed for {file_name}")
                    
                    # Successfully processed, break out of retry loop
                    break
                else:
                    self.logger.warning(f"Could not extract valid assessment for {file_name}")
                    # Try again or continue after last retry
                    retry_count += 1
                    if retry_count <= max_retries:
                        self.logger.info(f"Retrying LLM integrity check for {file_name} (attempt {retry_count})")
                    else:
                        self.logger.warning(f"Failed to get valid assessment after {max_retries} retries")
            except Exception as e:
                # Log the error but don't fail the process
                self.logger.error(f"Error in LLM integrity check for {file_name}: {e}")
                retry_count += 1
                
                if retry_count <= max_retries:
                    self.logger.info(f"Retrying LLM integrity check for {file_name} (attempt {retry_count})")
                else:
                    self.logger.info("Continuing with standard integrity checks")
                    break 