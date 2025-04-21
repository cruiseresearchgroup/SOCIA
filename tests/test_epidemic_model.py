import os
import sys
import unittest
import numpy as np

# Add the project root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.epidemic_model import Person, EpidemicEnvironment, EpidemicSimulation, create_epidemic_simulation


class TestEpidemicModel(unittest.TestCase):
    """Test cases for the epidemic simulation model."""
    
    def setUp(self):
        """Set up a basic simulation environment for testing."""
        self.config = {
            'population_size': 100,
            'initial_infected': 5,
            'transmission_rate': 0.1,
            'recovery_rate': 0.05,
            'contact_radius': 0.1,
            'seed': 42
        }
        
        # Create the simulation
        self.simulation = create_epidemic_simulation(self.config)
        self.simulation.initialize()
    
    def test_person_initialization(self):
        """Test that a person is initialized correctly."""
        person = Person(age=30, health_state='susceptible', location=(0.5, 0.5), mobility=0.1)
        
        self.assertEqual(person.age, 30)
        self.assertEqual(person.health_state, 'susceptible')
        self.assertTupleEqual(person.location, (0.5, 0.5))
        self.assertEqual(person.mobility, 0.1)
        self.assertIsNone(person.infection_time)
    
    def test_person_update_health_state(self):
        """Test updating a person's health state."""
        person = Person(age=30, health_state='susceptible', location=(0.5, 0.5), mobility=0.1)
        
        # Infect the person
        person.update_health_state('infected', infection_time=10)
        self.assertEqual(person.health_state, 'infected')
        self.assertEqual(person.infection_time, 10)
        
        # Recover the person
        person.update_health_state('recovered')
        self.assertEqual(person.health_state, 'recovered')
    
    def test_person_move(self):
        """Test that a person can move within bounds."""
        person = Person(age=30, health_state='susceptible', location=(0.5, 0.5), mobility=0.1)
        
        # Set a random seed for reproducibility
        np.random.seed(42)
        
        # Move the person
        old_location = person.location
        person.move(bounds=(0, 0, 1, 1))
        
        # Check that the person has moved
        self.assertNotEqual(old_location, person.location)
        
        # Check that the new location is within bounds
        self.assertTrue(0 <= person.location[0] <= 1)
        self.assertTrue(0 <= person.location[1] <= 1)
    
    def test_environment_initialization(self):
        """Test that the environment is initialized correctly."""
        env = self.simulation.environment
        
        # Check population size
        self.assertEqual(len(env.population), self.config['population_size'])
        
        # Count initial infected
        infected_count = sum(1 for p in env.population if p.health_state == 'infected')
        self.assertEqual(infected_count, self.config['initial_infected'])
        
        # Check initial metrics
        self.assertEqual(env.metrics['susceptible_count'], self.config['population_size'] - self.config['initial_infected'])
        self.assertEqual(env.metrics['infected_count'], self.config['initial_infected'])
        self.assertEqual(env.metrics['recovered_count'], 0)
    
    def test_environment_step(self):
        """Test the environment stepping through one iteration."""
        env = self.simulation.environment
        
        # Get initial counts
        initial_susceptible = env.metrics['susceptible_count']
        initial_infected = env.metrics['infected_count']
        initial_recovered = env.metrics['recovered_count']
        
        # Step the environment
        env.step()
        
        # Check that metrics have been updated
        self.assertTrue('time_step' in env.metrics)
        self.assertTrue('susceptible_count' in env.metrics)
        self.assertTrue('infected_count' in env.metrics)
        self.assertTrue('recovered_count' in env.metrics)
        self.assertTrue('new_infections' in env.metrics)
        
        # Verify population counts
        total_population = env.metrics['susceptible_count'] + env.metrics['infected_count'] + env.metrics['recovered_count']
        self.assertEqual(total_population, self.config['population_size'])
    
    def test_disease_transmission(self):
        """Test that the disease can be transmitted."""
        env = self.simulation.environment
        
        # Place a susceptible person and an infected person close to each other
        susceptible = Person(age=30, health_state='susceptible', location=(0.5, 0.5), mobility=0)
        infected = Person(age=30, health_state='infected', location=(0.51, 0.51), mobility=0, infection_time=0)
        
        # Create a test environment with just these two people
        test_env = EpidemicEnvironment(
            population=[susceptible, infected],
            transmission_rate=1.0,  # 100% transmission rate
            recovery_rate=0.0,  # No recovery
            contact_radius=0.1,  # Large enough to ensure contact
            time_step=0
        )
        
        # Step the environment
        test_env.step()
        
        # Check that the susceptible person is now infected
        self.assertEqual(susceptible.health_state, 'infected')
    
    def test_recovery(self):
        """Test that infected people can recover."""
        # Create a test environment with infected people and high recovery rate
        infected = Person(age=30, health_state='infected', location=(0.5, 0.5), mobility=0, infection_time=0)
        
        test_env = EpidemicEnvironment(
            population=[infected],
            transmission_rate=0.0,  # No transmission
            recovery_rate=1.0,  # 100% recovery rate
            contact_radius=0.1,
            time_step=0
        )
        
        # Step the environment
        test_env.step()
        
        # Check that the infected person is now recovered
        self.assertEqual(infected.health_state, 'recovered')
    
    def test_simulation_run(self):
        """Test running the simulation for multiple steps."""
        # Run the simulation for 10 steps
        for _ in range(10):
            self.simulation.environment.step()
        
        # Check that the time step has been incremented
        self.assertEqual(self.simulation.environment.time_step, 10)
        
        # Verify that the metrics history has 10 entries
        self.assertEqual(len(self.simulation.environment.metrics_history), 10)
    
    def test_create_epidemic_simulation(self):
        """Test creating a simulation with the factory function."""
        sim = create_epidemic_simulation(self.config)
        
        # Check that the simulation has been created with the correct configuration
        self.assertEqual(sim.population_size, self.config['population_size'])
        self.assertEqual(sim.initial_infected, self.config['initial_infected'])
        self.assertEqual(sim.environment.transmission_rate, self.config['transmission_rate'])
        self.assertEqual(sim.environment.recovery_rate, self.config['recovery_rate'])
        self.assertEqual(sim.environment.contact_radius, self.config['contact_radius'])


if __name__ == '__main__':
    unittest.main() 