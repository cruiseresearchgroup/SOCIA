"""
A template for a social network simulation model.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional

from core.simulation import Entity, Environment, Simulation

class Person(Entity):
    """
    Represents a person in a social network simulation.
    """
    
    def __init__(self, entity_id: str, attributes: Dict[str, Any] = None):
        """
        Initialize a person entity.
        
        Args:
            entity_id: Unique identifier for the person
            attributes: Dictionary of attributes for the person
        """
        super().__init__(entity_id, attributes)
        
        # Default attributes if not provided
        if 'age' not in self.attributes:
            self.attributes['age'] = np.random.randint(18, 80)
        if 'interests' not in self.attributes:
            self.attributes['interests'] = np.random.choice(
                ['sports', 'music', 'art', 'science', 'technology', 'politics'],
                size=np.random.randint(1, 4),
                replace=False
            ).tolist()
        if 'location' not in self.attributes:
            self.attributes['location'] = (np.random.random(), np.random.random())
        if 'extroversion' not in self.attributes:
            self.attributes['extroversion'] = np.random.random()
    
    def update(self, environment: 'SocialNetworkEnvironment', time_step: float) -> None:
        """
        Update the person's state for the current time step.
        
        Args:
            environment: The simulation environment
            time_step: The current time step
        """
        # Example behavior: interact with other people
        self._interact_with_others(environment)
        
        # Example behavior: move randomly
        self._move_randomly(environment, time_step)
    
    def _interact_with_others(self, environment: 'SocialNetworkEnvironment') -> None:
        """
        Interact with other people in the environment.
        
        Args:
            environment: The simulation environment
        """
        # Get all other people
        other_people = [e for e in environment.get_all_entities() if isinstance(e, Person) and e.id != self.id]
        
        # For each person, calculate interaction probability based on:
        # 1. Physical distance
        # 2. Similarity of interests
        # 3. Extroversion
        for other in other_people:
            # Skip if already connected
            if environment.network.has_edge(self.id, other.id):
                continue
            
            # Calculate distance
            my_loc = self.attributes['location']
            other_loc = other.attributes['location']
            distance = np.sqrt((my_loc[0] - other_loc[0])**2 + (my_loc[1] - other_loc[1])**2)
            
            # Calculate interest similarity
            my_interests = set(self.attributes['interests'])
            other_interests = set(other.attributes['interests'])
            interest_similarity = len(my_interests.intersection(other_interests)) / len(my_interests.union(other_interests))
            
            # Calculate interaction probability
            extroversion_factor = (self.attributes['extroversion'] + other.attributes['extroversion']) / 2
            interaction_prob = extroversion_factor * interest_similarity * (1 - distance)
            
            # Decide whether to interact
            if np.random.random() < interaction_prob:
                environment.add_connection(self.id, other.id)
    
    def _move_randomly(self, environment: 'SocialNetworkEnvironment', time_step: float) -> None:
        """
        Move randomly in the environment.
        
        Args:
            environment: The simulation environment
            time_step: The current time step
        """
        # Simple random movement
        x, y = self.attributes['location']
        dx = (np.random.random() - 0.5) * time_step * 0.1
        dy = (np.random.random() - 0.5) * time_step * 0.1
        
        # Update location, ensuring it stays within bounds [0, 1]
        new_x = max(0, min(1, x + dx))
        new_y = max(0, min(1, y + dy))
        self.attributes['location'] = (new_x, new_y)


class SocialNetworkEnvironment(Environment):
    """
    Environment for a social network simulation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the social network environment.
        
        Args:
            config: Configuration for the environment
        """
        super().__init__()
        self.config = config or {}
        
        # Initialize network
        self.network = nx.Graph()
        
        # Initialize metrics
        self.metrics = {
            'num_connections': 0,
            'avg_connections_per_person': 0,
            'clustering_coefficient': 0,
            'communities': 0
        }
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the environment.
        
        Args:
            entity: The entity to add
        """
        super().add_entity(entity)
        self.network.add_node(entity.id, attributes=entity.attributes)
    
    def remove_entity(self, entity_id: str) -> None:
        """
        Remove an entity from the environment.
        
        Args:
            entity_id: ID of the entity to remove
        """
        super().remove_entity(entity_id)
        if entity_id in self.network:
            self.network.remove_node(entity_id)
    
    def add_connection(self, entity1_id: str, entity2_id: str, attributes: Dict[str, Any] = None) -> None:
        """
        Add a connection between two entities.
        
        Args:
            entity1_id: ID of the first entity
            entity2_id: ID of the second entity
            attributes: Attributes of the connection
        """
        if entity1_id in self.network and entity2_id in self.network:
            self.network.add_edge(entity1_id, entity2_id, attributes=attributes or {})
    
    def remove_connection(self, entity1_id: str, entity2_id: str) -> None:
        """
        Remove a connection between two entities.
        
        Args:
            entity1_id: ID of the first entity
            entity2_id: ID of the second entity
        """
        if self.network.has_edge(entity1_id, entity2_id):
            self.network.remove_edge(entity1_id, entity2_id)
    
    def step(self, time_step: float = 1.0) -> Dict[str, Any]:
        """
        Advance the simulation by one time step.
        
        Args:
            time_step: The time step to advance by
        
        Returns:
            Updated metrics for the current step
        """
        # Update all entities
        for entity in self.entities.values():
            entity.update(self, time_step)
        
        # Update time
        self.time += time_step
        
        # Update metrics
        self._update_metrics()
        
        return self.metrics
    
    def _update_metrics(self) -> None:
        """Update metrics for the current state of the network."""
        # Basic network metrics
        num_nodes = self.network.number_of_nodes()
        num_edges = self.network.number_of_edges()
        
        self.metrics['num_connections'] = num_edges
        
        if num_nodes > 0:
            self.metrics['avg_connections_per_person'] = 2 * num_edges / num_nodes
        else:
            self.metrics['avg_connections_per_person'] = 0
        
        # Advanced network metrics (if network is not empty)
        if num_nodes > 0:
            self.metrics['clustering_coefficient'] = nx.average_clustering(self.network)
            
            # Community detection
            if num_nodes > 1:
                communities = list(nx.community.greedy_modularity_communities(self.network))
                self.metrics['communities'] = len(communities)
            else:
                self.metrics['communities'] = 1
        else:
            self.metrics['clustering_coefficient'] = 0
            self.metrics['communities'] = 0


class SocialNetworkSimulation(Simulation):
    """
    Simulation of a social network.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the social network simulation.
        
        Args:
            config: Configuration for the simulation
        """
        self.config = config or {}
        self.environment = self._create_environment()
        self.logger = None  # Will be set by the parent class
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
    
    def _create_environment(self) -> Environment:
        """Create the simulation environment."""
        return SocialNetworkEnvironment(self.config)
    
    def initialize(self) -> None:
        """Initialize the simulation."""
        # Create population
        num_people = self.config.get('num_people', 100)
        
        for i in range(num_people):
            person = Person(f"person_{i}")
            self.environment.add_entity(person)
    
    def visualize(self) -> None:
        """Visualize the social network."""
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot network
        pos = {}
        for person_id in self.environment.network.nodes():
            person = self.environment.get_entity(person_id)
            if person and 'location' in person.attributes:
                pos[person_id] = person.attributes['location']
        
        # Draw network
        nx.draw_networkx(
            self.environment.network,
            pos=pos,
            with_labels=False,
            node_size=50,
            node_color='blue',
            alpha=0.7
        )
        
        plt.title('Social Network Visualization')
        plt.axis('off')
        
        # Save figure
        plt.savefig('social_network.png')
        plt.close()
        
        # Plot metrics over time
        if self.results['time_series']:
            plt.figure(figsize=(12, 8))
            
            # Plot time series metrics
            time_points = [entry['time'] for entry in self.results['time_series']]
            metrics = ['num_connections', 'avg_connections_per_person', 'clustering_coefficient', 'communities']
            
            for i, metric in enumerate(metrics):
                values = [entry['metrics'].get(metric, 0) for entry in self.results['time_series']]
                plt.subplot(2, 2, i+1)
                plt.plot(time_points, values)
                plt.title(metric.replace('_', ' ').title())
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('social_network_metrics.png')
            plt.close()


def create_social_network_simulation(config: Dict[str, Any] = None) -> SocialNetworkSimulation:
    """
    Create a social network simulation with the given configuration.
    
    Args:
        config: Configuration for the simulation
    
    Returns:
        A social network simulation
    """
    default_config = {
        'num_people': 100,
        'simulation_steps': 100,
        'time_step': 1.0,
        'seed': 42
    }
    
    # Merge default config with provided config
    if config:
        config = {**default_config, **config}
    else:
        config = default_config
    
    # Set random seed
    np.random.seed(config['seed'])
    
    return SocialNetworkSimulation(config) 