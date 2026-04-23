import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
from math import sqrt

# Define constants and paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Data file paths
AGENT_ATTRIBUTES_FILE = os.path.join(DATA_DIR, "agent_attributes.csv")
SOCIAL_NETWORK_FILE = os.path.join(DATA_DIR, "social_network.json")
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "train_data.csv")

class Person:
    """
    Represents an individual agent in the simulation.
    
    Attributes:
        agent_id (int): Unique identifier for the agent.
        mask_wearing_status (bool): Whether the agent is wearing a mask.
        social_influence (float): The influence score based on social connections.
        network_connections (list): IDs of connected agents in the network.
        risk_perception (float): Perception of risk influencing behavior.
        received_information (bool): Indicates if the agent received information about mask-wearing.
    """
    def __init__(self, agent_id, initial_mask_wearing, risk_perception, network_connections):
        self.agent_id = agent_id
        self.mask_wearing_status = initial_mask_wearing
        self.social_influence = 0.0
        self.network_connections = network_connections
        self.risk_perception = risk_perception
        self.received_information = False

    def influence_others(self, network, influence_strength, family_weight, work_school_weight, community_weight):
        """
        Influence other agents in the network to adopt or change mask-wearing behavior.
        
        Args:
            network (Network): The network in which the agent exists.
            influence_strength (float): The strength of influence exerted by the agent.
            family_weight (float): Weight for family connections influence.
            work_school_weight (float): Weight for work/school connections influence.
            community_weight (float): Weight for community connections influence.
        """
        for connection_id in self.network_connections:
            connection = network.get_person(connection_id)
            if connection:
                # Calculate influence based on connection type weights
                influence = influence_strength * self.social_influence
                if connection_id in network.get_family_connections(self.agent_id):
                    influence *= family_weight
                elif connection_id in network.get_work_school_connections(self.agent_id):
                    influence *= work_school_weight
                else:
                    influence *= community_weight
                
                connection.receive_influence(influence)

    def change_mask_wearing_status(self, threshold):
        """
        Change the mask-wearing status based on received information and social influence.
        
        Args:
            threshold (float): The risk perception threshold for adopting mask-wearing.
        """
        if self.received_information and self.social_influence > threshold:
            self.mask_wearing_status = True

    def receive_influence(self, influence):
        """
        Receive influence from another agent.
        
        Args:
            influence (float): The amount of influence received.
        """
        self.social_influence += influence

class Network:
    """
    Represents a social network of agents in the simulation.
    
    Attributes:
        network_structure (dict): The structure of the network.
        graph (nx.Graph): NetworkX graph object.
        average_degree (float): The average degree of the network.
    """
    def __init__(self, network_structure):
        self.network_structure = network_structure
        self.graph = self._build_graph(network_structure)
        self.average_degree = self._calculate_average_degree()

    def _build_graph(self, network_structure):
        """
        Builds a graph representation of the network using NetworkX.
        
        Args:
            network_structure (dict): The social network structure.
        
        Returns:
            nx.Graph: NetworkX graph object.
        """
        graph = nx.Graph()
        for agent_id, connections in network_structure.items():
            for connection_type, connection_list in connections.items():
                for connection_id in connection_list:
                    graph.add_edge(agent_id, connection_id)
        return graph

    def _calculate_average_degree(self):
        """
        Calculate the average degree of the network.
        
        Returns:
            float: Average degree of the network.
        """
        degrees = [degree for _, degree in self.graph.degree()]
        return np.mean(degrees)

    def propagate_influence(self, influence_strength, family_weight, work_school_weight, community_weight):
        """
        Propagate influence through the network.
        
        Args:
            influence_strength (float): The strength of influence propagation.
            family_weight (float): Weight for family connections influence.
            work_school_weight (float): Weight for work/school connections influence.
            community_weight (float): Weight for community connections influence.
        """
        for agent_id in self.network_structure.keys():
            person = self.get_person(agent_id)
            if person:
                person.influence_others(self, influence_strength, family_weight, work_school_weight, community_weight)

    def get_person(self, agent_id):
        """
        Get a person by their agent ID.
        
        Args:
            agent_id (int): The ID of the agent.
        
        Returns:
            Person: The person object with the given ID.
        """
        # Placeholder: Implement retrieval of Person object based on agent ID
        pass

    def get_family_connections(self, agent_id):
        """
        Get family connections for a given agent.
        
        Args:
            agent_id (int): The ID of the agent.
        
        Returns:
            list: List of family connection IDs.
        """
        return self.network_structure.get(agent_id, {}).get('family', [])

    def get_work_school_connections(self, agent_id):
        """
        Get work/school connections for a given agent.
        
        Args:
            agent_id (int): The ID of the agent.
        
        Returns:
            list: List of work/school connection IDs.
        """
        return self.network_structure.get(agent_id, {}).get('work_school', [])

class Simulation:
    """
    Main simulation class that coordinates the agent-based model.
    
    Attributes:
        agents (list of Person): List of agents in the simulation.
        network (Network): The social network of agents.
        days (int): Number of days to simulate.
    """
    def __init__(self, agents, network, days):
        self.agents = agents
        self.network = network
        self.days = days

    def run(self):
        """
        Run the simulation over the specified number of days.
        """
        for day in range(self.days):
            self.network.propagate_influence(0.5, 1.0, 0.75, 0.5)  # Example influence strengths
            for agent in self.agents:
                agent.change_mask_wearing_status(0.5)  # Example risk threshold

    def evaluate(self):
        """
        Evaluate the simulation results using specified metrics.
        
        Returns:
            dict: Evaluation metrics and their values.
        """
        # Placeholder: Implement evaluation logic
        return {}

    def visualize(self):
        """
        Visualize the results of the simulation.
        """
        # Placeholder: Implement visualization logic
        pass

    def save_results(self, filename):
        """
        Save the simulation results to a file.
        
        Args:
            filename (str): The name of the file to save results to.
        """
        # Placeholder: Implement saving logic
        pass

def load_data():
    """
    Load and preprocess data from CSV and JSON files.
    
    Returns:
        tuple: Tuple containing agent attributes and network structure.
    """
    try:
        # Load agent attributes
        agent_df = pd.read_csv(AGENT_ATTRIBUTES_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file path and ensure the file exists.")
        return None, None

    try:
        # Load social network structure
        with open(SOCIAL_NETWORK_FILE, 'r') as f:
            network_structure = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file path and ensure the file exists.")
        return None, None

    return agent_df, network_structure

def main():
    """
    Main function to initialize, run, visualize, and save the simulation.
    """
    # Load data
    agent_df, network_structure = load_data()
    if agent_df is None or network_structure is None:
        return

    # Initialize agents and network
    agents = [Person(row['agent_id'], row['initial_mask_wearing'], 
                     row['risk_perception'], network_structure[row['agent_id']]['all']) 
              for _, row in agent_df.iterrows()]
    network = Network(network_structure)

    # Initialize and run simulation
    simulation = Simulation(agents, network, days=39)
    simulation.run()

    # Evaluate and visualize results
    results = simulation.evaluate()
    simulation.visualize()

    # Save results
    simulation.save_results("results.csv")

if __name__ == "__main__":
    main()