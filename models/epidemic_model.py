"""
A template for an epidemic simulation model (SIR model).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional

from core.simulation import Entity, Environment, Simulation

class Person:
    """
    Class representing an individual in the epidemic simulation.
    
    Attributes:
        age (int): Age of the person
        health_state (str): Current health state ('susceptible', 'infected', or 'recovered')
        location (tuple): (x, y) coordinates in the simulation space
        mobility (float): Factor determining how far a person can move in each step
        infection_time (int): Time step when the person was infected (None if never infected)
    """
    
    def __init__(self, age: int, health_state: str, location: Tuple[float, float], 
                 mobility: float, infection_time: Optional[int] = None):
        """Initialize a person with given attributes."""
        self.age = age
        self.health_state = health_state
        self.location = location
        self.mobility = mobility
        self.infection_time = infection_time
    
    def update_health_state(self, new_state: str, infection_time: Optional[int] = None):
        """
        Update the health state of the person.
        
        Args:
            new_state (str): New health state
            infection_time (int, optional): Time step of infection (only needed if new_state is 'infected')
        """
        self.health_state = new_state
        if new_state == 'infected':
            self.infection_time = infection_time
    
    def move(self, bounds: Tuple[float, float, float, float]):
        """
        Move the person randomly within the given bounds based on their mobility.
        
        Args:
            bounds (tuple): (min_x, min_y, max_x, max_y) simulation space boundaries
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Calculate random step based on mobility
        dx = (np.random.random() - 0.5) * self.mobility
        dy = (np.random.random() - 0.5) * self.mobility
        
        # Update location ensuring it stays within bounds
        new_x = max(min_x, min(max_x, self.location[0] + dx))
        new_y = max(min_y, min(max_y, self.location[1] + dy))
        
        self.location = (new_x, new_y)


class EpidemicEnvironment:
    """
    Environment for the epidemic simulation.
    
    Handles the population, disease dynamics, and metrics tracking.
    
    Attributes:
        population (list): List of Person objects
        transmission_rate (float): Probability of disease transmission upon contact
        recovery_rate (float): Probability of recovery at each time step
        contact_radius (float): Distance within which people can infect each other
        time_step (int): Current time step of the simulation
        metrics (dict): Current metrics of the simulation
        metrics_history (list): History of metrics for all time steps
    """
    
    def __init__(self, population: List[Person], transmission_rate: float, 
                 recovery_rate: float, contact_radius: float, time_step: int = 0):
        """Initialize the epidemic environment with given parameters."""
        self.population = population
        self.transmission_rate = transmission_rate
        self.recovery_rate = recovery_rate
        self.contact_radius = contact_radius
        self.time_step = time_step
        
        # Initialize metrics
        self.metrics = self._calculate_metrics()
        self.metrics_history = []
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate current metrics of the simulation."""
        susceptible_count = sum(1 for p in self.population if p.health_state == 'susceptible')
        infected_count = sum(1 for p in self.population if p.health_state == 'infected')
        recovered_count = sum(1 for p in self.population if p.health_state == 'recovered')
        
        return {
            'time_step': self.time_step,
            'susceptible_count': susceptible_count,
            'infected_count': infected_count,
            'recovered_count': recovered_count,
            'new_infections': 0  # Will be updated during the step
        }
    
    def step(self):
        """Advance the simulation by one time step."""
        self.time_step += 1
        
        # Track new infections in this step
        new_infections = 0
        
        # Process disease transmission
        for person in self.population:
            if person.health_state == 'susceptible':
                # Check for contacts with infected people
                for other in self.population:
                    if (other.health_state == 'infected' and
                            self._distance(person.location, other.location) <= self.contact_radius):
                        
                        # Determine if transmission occurs
                        if np.random.random() < self.transmission_rate:
                            person.update_health_state('infected', infection_time=self.time_step)
                            new_infections += 1
                            break  # Person is now infected, stop checking contacts
            
            elif person.health_state == 'infected':
                # Determine if recovery occurs
                if np.random.random() < self.recovery_rate:
                    person.update_health_state('recovered')
        
        # Allow people to move
        for person in self.population:
            person.move(bounds=(0, 0, 1, 1))  # Assuming a unit square simulation space
        
        # Update metrics
        self.metrics = self._calculate_metrics()
        self.metrics['new_infections'] = new_infections
        
        # Append to history
        self.metrics_history.append(self.metrics.copy())
        
        return self.metrics
    
    def _distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations."""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


class EpidemicSimulation:
    """
    Main class for running epidemic simulations.
    
    Attributes:
        population_size (int): Number of people in the simulation
        initial_infected (int): Number of initially infected people
        environment (EpidemicEnvironment): The environment instance for this simulation
    """
    
    def __init__(self, population_size: int, initial_infected: int, transmission_rate: float,
                 recovery_rate: float, contact_radius: float, seed: Optional[int] = None):
        """Initialize the simulation with given parameters."""
        self.population_size = population_size
        self.initial_infected = initial_infected
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Set up environment
        self.environment = EpidemicEnvironment(
            population=[],  # Will be populated in initialize()
            transmission_rate=transmission_rate,
            recovery_rate=recovery_rate,
            contact_radius=contact_radius
        )
    
    def initialize(self):
        """Initialize the population and set up the simulation."""
        # Create population
        population = []
        
        # Create susceptible people
        for i in range(self.population_size - self.initial_infected):
            person = Person(
                age=np.random.randint(1, 100),
                health_state='susceptible',
                location=(np.random.random(), np.random.random()),
                mobility=0.02 + 0.03 * np.random.random()  # Mobility between 0.02 and 0.05
            )
            population.append(person)
        
        # Create infected people
        for i in range(self.initial_infected):
            person = Person(
                age=np.random.randint(1, 100),
                health_state='infected',
                location=(np.random.random(), np.random.random()),
                mobility=0.02 + 0.03 * np.random.random(),  # Mobility between 0.02 and 0.05
                infection_time=0  # Infected at the start of simulation
            )
            population.append(person)
        
        # Add population to environment
        self.environment.population = population
    
    def run(self, num_steps: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run the simulation for the specified number of steps.
        
        Args:
            num_steps (int): Number of time steps to run
        
        Returns:
            dict: Metrics history and final state
        """
        # Run simulation for specified number of steps
        for _ in range(num_steps):
            self.environment.step()
        
        # Return results
        return {
            'metrics_history': self.environment.metrics_history,
            'final_state': self.environment.metrics
        }


def create_epidemic_simulation(config: Dict[str, Any]) -> EpidemicSimulation:
    """
    Factory function to create an epidemic simulation with the given configuration.
    
    Args:
        config (dict): Configuration parameters for the simulation
    
    Returns:
        EpidemicSimulation: Configured simulation instance
    """
    # Extract configuration parameters
    population_size = config.get('population_size', 1000)
    initial_infected = config.get('initial_infected', 10)
    transmission_rate = config.get('transmission_rate', 0.1)
    recovery_rate = config.get('recovery_rate', 0.05)
    contact_radius = config.get('contact_radius', 0.02)
    seed = config.get('seed', None)
    
    # Create the simulation
    simulation = EpidemicSimulation(
        population_size=population_size,
        initial_infected=initial_infected,
        transmission_rate=transmission_rate,
        recovery_rate=recovery_rate,
        contact_radius=contact_radius,
        seed=seed
    )
    
    return simulation 