"""
Base classes for simulation models in the SOCIA system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import os
import time

class Entity(ABC):
    """Base class for all simulation entities."""
    
    def __init__(self, entity_id: str, attributes: Dict[str, Any] = None):
        """
        Initialize an entity.
        
        Args:
            entity_id: Unique identifier for the entity
            attributes: Dictionary of attributes for the entity
        """
        self.id = entity_id
        self.attributes = attributes or {}
    
    def get_attribute(self, name: str) -> Any:
        """Get the value of an attribute."""
        return self.attributes.get(name)
    
    def set_attribute(self, name: str, value: Any) -> None:
        """Set the value of an attribute."""
        self.attributes[name] = value
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the entity."""
        return {
            "id": self.id,
            "attributes": self.attributes
        }
    
    @abstractmethod
    def update(self, environment: 'Environment', time_step: float) -> None:
        """
        Update the entity's state for the current time step.
        
        Args:
            environment: The simulation environment
            time_step: The current time step
        """
        pass


class Environment(ABC):
    """Base class for simulation environments."""
    
    def __init__(self):
        """Initialize the environment."""
        self.entities = {}
        self.time = 0.0
        self.metrics = {}
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the environment."""
        self.entities[entity.id] = entity
    
    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the environment."""
        if entity_id in self.entities:
            del self.entities[entity_id]
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities in the environment."""
        return list(self.entities.values())
    
    def get_entities_by_type(self, entity_type: type) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if isinstance(e, entity_type)]
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update the environment's metrics."""
        for key, value in metrics.items():
            if key in self.metrics:
                if isinstance(value, list) and isinstance(self.metrics[key], list):
                    self.metrics[key].extend(value)
                elif isinstance(value, dict) and isinstance(self.metrics[key], dict):
                    self.metrics[key].update(value)
                elif isinstance(value, (int, float)) and isinstance(self.metrics[key], (int, float)):
                    self.metrics[key] += value
                else:
                    self.metrics[key] = value
            else:
                self.metrics[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the environment's metrics."""
        return self.metrics
    
    @abstractmethod
    def step(self, time_step: float = 1.0) -> Dict[str, Any]:
        """
        Advance the simulation by one time step.
        
        Args:
            time_step: The time step to advance by
        
        Returns:
            Updated metrics for the current step
        """
        pass


class Simulation(ABC):
    """Base class for simulations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simulation.
        
        Args:
            config: Configuration dictionary for the simulation
        """
        self.config = config or {}
        self.environment = self._create_environment()
        self.logger = logging.getLogger("SOCIA.Simulation")
        self.results = {
            "config": self.config,
            "metrics": {},
            "time_series": [],
            "run_info": {
                "start_time": None,
                "end_time": None,
                "duration": None
            }
        }
    
    @abstractmethod
    def _create_environment(self) -> Environment:
        """Create the simulation environment."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the simulation."""
        pass
    
    def run(self, steps: int = 100, time_step: float = 1.0) -> Dict[str, Any]:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of steps to run
            time_step: Time step size
        
        Returns:
            Simulation results
        """
        self.logger.info(f"Starting simulation with {steps} steps")
        self.initialize()
        
        start_time = time.time()
        self.results["run_info"]["start_time"] = start_time
        
        for step in range(steps):
            self.logger.debug(f"Running step {step+1}/{steps}")
            metrics = self.environment.step(time_step)
            self.results["time_series"].append({
                "step": step,
                "time": self.environment.time,
                "metrics": metrics
            })
        
        end_time = time.time()
        self.results["run_info"]["end_time"] = end_time
        self.results["run_info"]["duration"] = end_time - start_time
        
        # Aggregate metrics
        self.results["metrics"] = self.environment.get_metrics()
        
        self.logger.info(f"Simulation completed in {self.results['run_info']['duration']:.2f} seconds")
        return self.results
    
    def save_results(self, path: str) -> str:
        """
        Save simulation results to a file.
        
        Args:
            path: Path to save the results to
        
        Returns:
            Path to the saved file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Saved simulation results to {path}")
        return path 