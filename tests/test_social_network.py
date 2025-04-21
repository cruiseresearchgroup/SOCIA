"""
Test file for the social network simulation model.
"""

import unittest
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.social_network_model import Person, SocialNetworkEnvironment, SocialNetworkSimulation, create_social_network_simulation

class TestSocialNetworkModel(unittest.TestCase):
    """Test case for the social network simulation model."""
    
    def test_person_creation(self):
        """Test creating a person entity."""
        person = Person("test_person")
        
        # Check that the ID is set correctly
        self.assertEqual(person.id, "test_person")
        
        # Check that default attributes are created
        self.assertIn('age', person.attributes)
        self.assertIn('interests', person.attributes)
        self.assertIn('location', person.attributes)
        self.assertIn('extroversion', person.attributes)
        
        # Check that custom attributes are respected
        person = Person("custom_person", {
            'age': 30,
            'interests': ['coding', 'testing'],
            'location': (0.5, 0.5),
            'extroversion': 0.8
        })
        
        self.assertEqual(person.attributes['age'], 30)
        self.assertEqual(person.attributes['interests'], ['coding', 'testing'])
        self.assertEqual(person.attributes['location'], (0.5, 0.5))
        self.assertEqual(person.attributes['extroversion'], 0.8)
    
    def test_environment_creation(self):
        """Test creating a social network environment."""
        env = SocialNetworkEnvironment()
        
        # Check that the network is initialized
        self.assertEqual(env.network.number_of_nodes(), 0)
        self.assertEqual(env.network.number_of_edges(), 0)
        
        # Check that metrics are initialized
        self.assertIn('num_connections', env.metrics)
        self.assertIn('avg_connections_per_person', env.metrics)
        self.assertIn('clustering_coefficient', env.metrics)
        self.assertIn('communities', env.metrics)
    
    def test_adding_entities(self):
        """Test adding entities to the environment."""
        env = SocialNetworkEnvironment()
        person1 = Person("person1")
        person2 = Person("person2")
        
        env.add_entity(person1)
        env.add_entity(person2)
        
        # Check that entities are added to the environment
        self.assertEqual(len(env.entities), 2)
        self.assertIn("person1", env.entities)
        self.assertIn("person2", env.entities)
        
        # Check that entities are added to the network
        self.assertEqual(env.network.number_of_nodes(), 2)
        self.assertIn("person1", env.network.nodes())
        self.assertIn("person2", env.network.nodes())
    
    def test_adding_connections(self):
        """Test adding connections between entities."""
        env = SocialNetworkEnvironment()
        person1 = Person("person1")
        person2 = Person("person2")
        
        env.add_entity(person1)
        env.add_entity(person2)
        env.add_connection("person1", "person2")
        
        # Check that the connection is added to the network
        self.assertEqual(env.network.number_of_edges(), 1)
        self.assertTrue(env.network.has_edge("person1", "person2"))
    
    def test_environment_step(self):
        """Test stepping the environment."""
        env = SocialNetworkEnvironment()
        person1 = Person("person1")
        person2 = Person("person2")
        
        env.add_entity(person1)
        env.add_entity(person2)
        
        # Record initial state
        initial_location1 = person1.attributes['location']
        initial_location2 = person2.attributes['location']
        
        # Step the environment
        metrics = env.step()
        
        # Check that the time is updated
        self.assertEqual(env.time, 1.0)
        
        # Check that metrics are returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('num_connections', metrics)
        
        # Check that entities are updated (e.g., locations changed)
        self.assertNotEqual(person1.attributes['location'], initial_location1)
        self.assertNotEqual(person2.attributes['location'], initial_location2)
    
    def test_simulation_creation(self):
        """Test creating a social network simulation."""
        config = {'num_people': 10, 'simulation_steps': 5}
        sim = create_social_network_simulation(config)
        
        # Check that the simulation is created with the correct configuration
        self.assertEqual(sim.config['num_people'], 10)
        self.assertEqual(sim.config['simulation_steps'], 5)
        
        # Check that the environment is created
        self.assertIsInstance(sim.environment, SocialNetworkEnvironment)
    
    def test_simulation_run(self):
        """Test running a social network simulation."""
        config = {'num_people': 10, 'simulation_steps': 5}
        sim = create_social_network_simulation(config)
        
        # Run the simulation
        results = sim.run(steps=5)
        
        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn('metrics', results)
        self.assertIn('time_series', results)
        
        # Check that time series data is collected
        self.assertEqual(len(results['time_series']), 5)
        
        # Check that entities were created and added to the environment
        self.assertEqual(len(sim.environment.entities), 10)


if __name__ == '__main__':
    unittest.main() 