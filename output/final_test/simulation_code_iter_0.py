import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Setup data file paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
network_file = os.path.join(DATA_DIR, "social_network.json")
train_data_file = os.path.join(DATA_DIR, "train_data.csv")

class Person:
    """
    Represents an individual in the simulation with attributes relevant to mask-wearing behavior.
    """
    def __init__(self, id, mask_wearing_status, social_influence, network_connections, risk_perception):
        self.id = id
        self.mask_wearing_status = mask_wearing_status
        self.social_influence = social_influence
        self.network_connections = network_connections
        self.risk_perception = risk_perception

    def adopt_mask_wearing(self):
        """
        Decides to adopt mask-wearing based on risk perception and social influence.
        """
        # Algorithm: If risk_perception > threshold or influenced by connections, adopt mask-wearing
        risk_perception_threshold = 0.5
        if self.risk_perception > risk_perception_threshold or np.random.rand() < self.social_influence:
            self.mask_wearing_status = True
        pass

    def influence_others(self, network, influence_probability):
        """
        Influences other agents in the network to adopt mask-wearing behavior.
        """
        for connection in self.network_connections:
            if np.random.rand() < influence_probability:
                network[connection].mask_wearing_status = True
        pass

class Network:
    """
    Represents the network of connections among agents.
    """
    def __init__(self, connections, structure_type):
        self.connections = connections
        self.structure_type = structure_type

    def propagate_information(self):
        """
        Propagates information through the network.
        """
        pass

    def update_connections(self):
        """
        Updates the network connections.
        """
        pass

class Simulation:
    """
    Manages the entire simulation process, coordinating agents and network interactions.
    """
    def __init__(self, population_size=1000, random_seed=42):
        np.random.seed(random_seed)
        self.population_size = population_size
        self.agents = []
        self.network = None
        self.load_agents()
        self.load_network()

    def load_agents(self):
        """
        Loads agents from the data file.
        """
        agent_data = pd.read_csv(agent_file)
        for _, row in agent_data.iterrows():
            agent = Person(
                id=row['id'],
                mask_wearing_status=row['initial_mask_wearing'],
                social_influence=0,  # Placeholder for social influence calculation
                network_connections=[],  # Will be set after network is loaded
                risk_perception=row['risk_perception']
            )
            self.agents.append(agent)

    def load_network(self):
        """
        Loads the network structure from the JSON file.
        """
        with open(network_file, 'r') as file:
            network_data = json.load(file)
            connections = {int(k): v['all'] for k, v in network_data.items()}
            self.network = Network(connections=connections, structure_type="small_world")

        for agent in self.agents:
            agent.network_connections = self.network.connections.get(agent.id, [])

    def run(self):
        """
        Runs the simulation for the specified number of days.
        """
        start_day = 30
        end_day = 39
        influence_probability = 0.05

        for day in range(start_day, end_day + 1):
            for agent in self.agents:
                agent.adopt_mask_wearing()
                agent.influence_others(self.network.connections, influence_probability)

    def evaluate(self):
        """
        Evaluates the simulation results based on specified metrics.
        """
        pass

    def visualize(self):
        """
        Visualizes the simulation results.
        """
        adoption_rates = [agent.mask_wearing_status for agent in self.agents]
        plt.hist(adoption_rates, bins=10)
        plt.xlabel("Mask Wearing Status")
        plt.ylabel("Number of Agents")
        plt.title("Distribution of Mask Wearing Status")
        plt.show()

    def save_results(self, filename):
        """
        Saves the simulation results to a file.
        """
        results = {'id': [], 'mask_wearing_status': []}
        for agent in self.agents:
            results['id'].append(agent.id)
            results['mask_wearing_status'].append(agent.mask_wearing_status)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)

def main():
    """
    Main function to run the simulation.
    """
    simulation = Simulation()
    simulation.run()
    simulation.visualize()
    simulation.save_results("results.csv")

if __name__ == "__main__":
    main()