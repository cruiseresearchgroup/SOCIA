import os
import numpy as np
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict

# Constants for data paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Data file paths
agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
network_file = os.path.join(DATA_DIR, "social_network.json")
train_file = os.path.join(DATA_DIR, "train_data.csv")

class Person:
    """
    Represents an individual agent in the simulation with attributes and behaviors related to mask-wearing.
    """
    influence_probability = 0.05
    risk_perception_effect = 1.0
    social_influence_decay = 0.01
    decision_threshold = 0.5
    
    def __init__(self, agent_id: int, mask_wearing_status: bool, risk_perception: float, network_connections: List[int]):
        self.agent_id = agent_id
        self.mask_wearing_status = mask_wearing_status
        self.risk_perception = risk_perception
        self.network_connections = network_connections
        self.social_influence = 0.0
    
    def decide_to_wear_mask(self):
        """
        Decide whether to wear a mask based on social influence and risk perception.
        """
        influence_factor = self.social_influence * Person.influence_probability
        risk_factor = self.risk_perception * Person.risk_perception_effect
        decision_value = influence_factor + risk_factor
        self.mask_wearing_status = decision_value > Person.decision_threshold
        self.social_influence = max(0, self.social_influence - Person.social_influence_decay)
    
    def influence_others(self, network: nx.Graph, influence_weight: float):
        """
        Influence connected agents to adopt mask-wearing behavior.
        """
        for connection in self.network_connections:
            if network.nodes[connection]['social_influence'] == 0.0:
                network.nodes[connection]['social_influence'] += influence_weight

class SocialNetwork:
    """
    Represents the social network structure and facilitates behavior propagation.
    
    Attributes:
        network (nx.Graph): The graph representing the social network.
    """
    def __init__(self, network_structure: Dict[int, Dict[str, List[int]]]):
        self.network = nx.Graph()
        self.network.add_nodes_from(network_structure.keys())
        for person_id, connections in network_structure.items():
            for connection_id in connections['all']:
                self.network.add_edge(person_id, connection_id)
                # Initialize social influence to 0.0
                self.network.nodes[connection_id]['social_influence'] = 0.0

    def propagate_behavior(self, information_spread_rate: float):
        """
        Propagate mask-wearing behavior through the network.
        """
        for node in self.network.nodes:
            person = self.network.nodes[node].get('person')
            if person:
                person_status = person.mask_wearing_status
                person.social_influence = sum(
                    self.network.nodes[neighbor]['person'].mask_wearing_status != person_status
                    for neighbor in self.network.neighbors(node)
                    if 'person' in self.network.nodes[neighbor]
                )
                person.decide_to_wear_mask()

class Simulation:
    """
    Main simulation class to manage the setup, execution, and evaluation of the mask-wearing behavior model.
    """
    def __init__(self):
        self.agents = self.load_agents()
        self.social_network = self.load_social_network()
    
    def load_agents(self) -> List[Person]:
        """
        Load agent data from the CSV file and initialize Person instances.
        """
        agents = []
        try:
            data = pd.read_csv(agent_file)
            if not {'agent_id', 'initial_mask_wearing', 'risk_perception'}.issubset(data.columns):
                raise ValueError("Missing required columns in agent data file.")
            for _, row in data.iterrows():
                if not isinstance(row['agent_id'], int) or not isinstance(row['risk_perception'], (int, float)):
                    raise ValueError("Invalid data type in agent data file.")
                agent = Person(
                    agent_id=row['agent_id'],
                    mask_wearing_status=row['initial_mask_wearing'],
                    risk_perception=row['risk_perception'],
                    network_connections=[]
                )
                agents.append(agent)
        except (FileNotFoundError, pd.errors.ParserError, ValueError) as e:
            raise RuntimeError(f"Error loading agent file: {e}")
        return agents

    def load_social_network(self) -> SocialNetwork:
        """
        Load social network data from JSON file and initialize SocialNetwork.
        """
        try:
            with open(network_file, 'r') as file:
                network_structure = json.load(file)
            if not all(str(agent.agent_id) in network_structure for agent in self.agents):
                raise KeyError("Missing agent entries in network structure.")
            social_network = SocialNetwork(network_structure)
            for agent in self.agents:
                agent.network_connections = network_structure[str(agent.agent_id)]['all']
                social_network.network.nodes[agent.agent_id]['person'] = agent
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Error loading network file: {e}")
        return social_network

    def run(self, start_day: int = 30, end_day: int = 39):
        """
        Execute the simulation loop over the specified prediction period.
        
        Args:
            start_day (int): The starting day for the simulation.
            end_day (int): The ending day for the simulation.
        """
        for day in range(start_day, end_day + 1):
            self.social_network.propagate_behavior(0.05)

    def visualize(self):
        """
        Visualize the results of the simulation.
        """
        mask_wearers = [agent.mask_wearing_status for agent in self.agents]
        plt.hist(mask_wearers, bins=2)
        plt.title('Mask Wearing Distribution')
        plt.xlabel('Mask Wearing Status')
        plt.ylabel('Number of Agents')
        plt.show()

    def save_results(self, filename: str):
        """
        Save the results of the simulation to a file.
        
        Args:
            filename (str): The name of the file where results will be saved.
        """
        results = pd.DataFrame({
            'agent_id': [agent.agent_id for agent in self.agents],
            'mask_wearing_status': [agent.mask_wearing_status for agent in self.agents]
        })
        results.to_csv(filename, index=False)

def main():
    """
    Main function to initialize, run, visualize, and save the simulation.
    """
    simulation = Simulation()
    simulation.run()
    simulation.visualize()
    simulation.save_results("results.csv")

# Execute main for both direct execution and sandbox wrapper invocation
main()