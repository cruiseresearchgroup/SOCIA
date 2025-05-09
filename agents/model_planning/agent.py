"""
ModelPlanningAgent: Designs the simulation model based on task requirements and data analysis.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import pickle
import os
import json
import numpy as np

from agents.base_agent import BaseAgent

class ModelPlanningAgent(BaseAgent):
    """
    Model Planning Agent designs the structure and approach for the simulation
    based on task requirements and data analysis.
    
    This agent is responsible for:
    1. Selecting appropriate modeling approaches (agent-based, system dynamics, etc.)
    2. Defining the structure of the simulation (entities, behaviors, interactions)
    3. Specifying parameters and configuration options
    4. Creating a detailed plan that the Code Generation Agent can implement
    """
    
    def process(
        self,
        task_spec: Dict[str, Any],
        data_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process the task specification and data analysis to create a simulation plan.
        
        Args:
            task_spec: Task specification from the Task Understanding Agent
            data_analysis: Data analysis results from the Data Analysis Agent (optional)
        
        Returns:
            Dictionary containing the simulation model plan
        """
        self.logger.info("Creating simulation model plan")
        
        # If processed data references are provided, log them and optionally load
        if data_analysis and "file_references" in data_analysis:
            refs = data_analysis.get("file_references", {})
            self.logger.info(f"Processed data file references: {refs}")
            # Example: load small sample if needed
            loaded = {}
            for name, path in refs.items():
                try:
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                        # 将DataFrame转换为JSON可序列化的字典格式
                        loaded[name] = {
                            "sample": df.head(5).to_dict('records'),
                            "columns": df.columns.tolist(),
                            "shape": list(df.shape),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                        }
                    elif path.endswith('.pkl'):
                        with open(path, 'rb') as f:
                            obj = pickle.load(f)
                        # 提供对象的更详细描述，确保可序列化
                        obj_type = str(type(obj))
                        obj_info = {"type": obj_type}
                        
                        # 根据对象类型提供不同的描述
                        if isinstance(obj, dict):
                            obj_info["keys"] = list(obj.keys())[:10]  # 最多包含前10个键
                            obj_info["value_types"] = {k: str(type(v)) for k, v in list(obj.items())[:5]}
                            obj_info["size"] = len(obj)
                        elif isinstance(obj, (list, tuple)):
                            obj_info["length"] = len(obj)
                            obj_info["sample_types"] = [str(type(x)) for x in obj[:5]] if obj else []
                        elif hasattr(obj, 'shape'):  # numpy arrays, matrices等
                            obj_info["shape"] = list(map(int, obj.shape))
                            obj_info["dtype"] = str(obj.dtype)
                        elif hasattr(obj, 'nodes'):  # 可能是networkx图
                            obj_info["nodes"] = len(obj.nodes)
                            obj_info["edges"] = len(obj.edges) if hasattr(obj, 'edges') else "unknown"
                            
                        loaded[name] = obj_info
                except Exception as e:
                    self.logger.warning(f"Could not load processed file {name}: {e}")
                    loaded[name] = {"error": str(e)}
            data_analysis['loaded_samples'] = loaded
        
        # Override task_spec data_files to point to processed data files
        if data_analysis and "file_references" in data_analysis:
            self.logger.info("Overriding task_spec data_files with processed file paths")
            task_spec = dict(task_spec)
            task_spec["data_files"] = data_analysis["file_references"]
        
        # Build prompt from template (includes file_summaries, calibration context, and optionally loaded samples)
        prompt = self._build_prompt(
            task_spec=task_spec,
            data_analysis=data_analysis
        )
        
        # Call LLM to design the model
        llm_response = self._call_llm(prompt)
        
        # Parse the response
        model_plan = self._parse_llm_response(llm_response)
        
        # If the response wasn't parsed as JSON, provide a placeholder structure
        if isinstance(model_plan, str):
            model_plan = self._create_default_model_plan(task_spec)
        
        self.logger.info("Model planning completed")
        return model_plan
    
    def _create_default_model_plan(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a default model plan based on the task specification."""
        # Extract simulation type from task spec, default to agent-based
        simulation_type = task_spec.get("simulation_type", "agent_based")
        
        # Extract entities from task spec
        entities = task_spec.get("entities", [])
        
        # Create a default model plan
        default_plan = {
            "model_type": simulation_type,
            "description": f"Default {simulation_type} model for {task_spec.get('title', 'simulation task')}",
            "entities": [],
            "behaviors": [],
            "interactions": [],
            "environment": {
                "type": "grid",
                "dimensions": [100, 100],
                "time_step": 1.0,
                "time_unit": task_spec.get("parameters", {}).get("time_unit", "days")
            },
            "parameters": task_spec.get("parameters", {}),
            "initialization": {
                "population_size": task_spec.get("parameters", {}).get("population_size", 1000),
                "random_seed": 42
            },
            "algorithms": {
                "movement": "random_walk",
                "interaction": "proximity_based"
            },
            "data_sources": [],
            "code_structure": {
                "files": [
                    {"name": "simulation.py", "description": "Main simulation class"},
                    {"name": "entities.py", "description": "Entity classes"},
                    {"name": "environment.py", "description": "Environment class"},
                    {"name": "behaviors.py", "description": "Behavior implementations"},
                    {"name": "utils.py", "description": "Utility functions"},
                    {"name": "config.py", "description": "Configuration constants"},
                    {"name": "visualization.py", "description": "Visualization utilities"},
                    {"name": "main.py", "description": "Entry point script"}
                ],
                "dependencies": [
                    "numpy",
                    "pandas",
                    "matplotlib"
                ]
            }
        }
        
        # Add entities from task spec
        for entity_spec in entities:
            entity_name = entity_spec.get("name", "")
            if entity_name:
                entity_plan = {
                    "name": entity_name,
                    "attributes": entity_spec.get("attributes", []),
                    "behaviors": entity_spec.get("behaviors", []),
                    "initialization": {"method": "random"}
                }
                default_plan["entities"].append(entity_plan)
        
        # Add behaviors based on entities
        for entity in default_plan["entities"]:
            for behavior_name in entity.get("behaviors", []):
                behavior_plan = {
                    "name": behavior_name,
                    "description": f"{behavior_name} behavior for {entity['name']}",
                    "applicable_to": [entity["name"]],
                    "parameters": {},
                    "algorithm": "rule_based"
                }
                default_plan["behaviors"].append(behavior_plan)
        
        # Add interactions from task spec
        for interaction_spec in task_spec.get("interactions", []):
            interaction_name = interaction_spec.get("name", "")
            if interaction_name:
                interaction_plan = {
                    "name": interaction_name,
                    "description": interaction_spec.get("description", ""),
                    "entities_involved": interaction_spec.get("entities_involved", []),
                    "trigger": "proximity",
                    "effect": "state_change"
                }
                default_plan["interactions"].append(interaction_plan)
        
        return default_plan 