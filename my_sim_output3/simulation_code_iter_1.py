# Import necessary libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional
import logging

# Set up logging for error handling and tracing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Person class and its behaviors
@dataclass
class Person:
    health_status: str
    infection_time: Optional[int] = None
    recovery_time: Optional[int] = None
    immune_status: bool = False
    x: int = field(default_factory=lambda: random.randint(0, 49))
    y: int = field(default_factory=lambda: random.randint(0, 49))

    def move(self, grid_size: int):
        """Simulates the movement of the person within the environment."""
        self.x = (self.x + random.choice([-1, 0, 1])) % grid_size
        self.y = (self.y + random.choice([-1, 0, 1])) % grid_size

    def interact(self, other, infection_probability: float, current_time: int):
        """Enables interaction between agents when they are in proximity."""
        if self.health_status == 'susceptible' and other.health_status == 'infected':
            if not self.immune_status and random.random() < infection_probability:
                self.become_infected(current_time)
                
    def become_infected(self, current_time: int):
        """Changes the health status of a susceptible person to infected."""
        self.health_status = 'infected'
        self.infection_time = current_time
        self.recovery_time = current_time + random.randint(10, 18)  # Average recovery time ± some variation

    def recover(self, current_time: int):
        """Transitions an infected person to a recovered state after a certain period."""
        if self.health_status == 'infected' and current_time >= self.recovery_time:
            self.health_status = 'recovered'
            self.immune_status = True

# Define the Environment class
class Environment:
    def __init__(self, population_size: int, initial_infected: int, infection_probability: float, grid_size: int):
        self.grid_size = grid_size
        self.infection_probability = infection_probability
        self.population = self.initialize_population(population_size, initial_infected)
        self.time_step = 0

    def initialize_population(self, population_size: int, initial_infected: int) -> List[Person]:
        """Initialize the population with a given number of initially infected individuals."""
        population = [Person(health_status='susceptible') for _ in range(population_size)]
        for person in random.sample(population, initial_infected):
            person.become_infected(self.time_step)
        return population

    def step(self):
        """Simulate one time step in the environment."""
        try:
            for person in self.population:
                person.move(self.grid_size)
                
            for i, person in enumerate(self.population):
                for other in self.population[i + 1:]:
                    if abs(person.x - other.x) <= 1 and abs(person.y - other.y) <= 1:
                        person.interact(other, self.infection_probability, self.time_step)
                        other.interact(person, self.infection_probability, self.time_step)
            
            for person in self.population:
                person.recover(self.time_step)
            
            self.time_step += 1
        except Exception as e:
            logging.error(f"Error during simulation step: {e}")

    def run_simulation(self, duration: int):
        """Run the simulation for a specified duration."""
        for _ in range(duration):
            self.step()

    def get_metrics(self):
        """Calculate and return the metrics of the simulation."""
        total_infected = sum(1 for person in self.population if person.immune_status)
        peak_infection = max(self.time_step)
        recovery_rate = sum(1 for person in self.population if person.health_status == 'recovered') / total_infected
        return total_infected, peak_infection, recovery_rate

# Visualization functions for displaying results
def visualize_simulation(environment: Environment):
    susceptible = [person for person in environment.population if person.health_status == 'susceptible']
    infected = [person for person in environment.population if person.health_status == 'infected']
    recovered = [person for person in environment.population if person.health_status == 'recovered']

    plt.figure(figsize=(10, 6))
    plt.scatter([p.x for p in susceptible], [p.y for p in susceptible], color='blue', label='Susceptible', alpha=0.6)
    plt.scatter([p.x for p in infected], [p.y for p in infected], color='red', label='Infected', alpha=0.6)
    plt.scatter([p.x for p in recovered], [p.y for p in recovered], color='green', label='Recovered', alpha=0.6)
    plt.title('Simulation State')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the simulation
def main():
    try:
        # Parameters from the model plan
        population_size = 1000
        initial_infected = 10
        infection_probability = 0.05
        average_recovery_time = 14
        simulation_duration = 100
        grid_size = 50

        # Initialize and run the simulation
        environment = Environment(population_size, initial_infected, infection_probability, grid_size)
        environment.run_simulation(simulation_duration)

        # Visualize the final state of the simulation
        visualize_simulation(environment)

        # Print metrics
        total_infected, peak_infection, recovery_rate = environment.get_metrics()
        logging.info(f'Total Infected: {total_infected}')
        logging.info(f'Peak Infection: {peak_infection}')
        logging.info(f'Recovery Rate: {recovery_rate}')

    except Exception as e:
        logging.error(f"Error in main simulation: {e}")

# Execute the simulation
if __name__ == '__main__':
    main()