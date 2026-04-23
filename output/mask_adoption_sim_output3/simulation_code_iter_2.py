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

def adjust_influence_strength(current_day, intervention_days):
    """
    Adjust the influence strength dynamically based on the current day and intervention days.
    
    Args:
        current_day (int): The current day of the simulation.
        intervention_days (list): List of days when government interventions occur.
    
    Returns:
        float: Adjusted influence strength.
    """
    base_strength = 0.5
    if current_day in intervention_days:
        return base_strength * 1.5  # Increase influence strength during intervention
    return base_strength

def adjust_propagation_speed(current_day, intervention_days):
    """
    Adjust the propagation speed dynamically based on the current day and intervention days.
    
    Args:
        current_day (int): The current day of the simulation.
        intervention_days (list): List of days when government interventions occur.
    
    Returns:
        float: Adjusted propagation speed.
    """
    base_speed = 1.0
    if current_day in intervention_days:
        return base_speed * 1.2  # Increase propagation speed during intervention
    return base_speed

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
        # Only agents wearing masks can influence others
        if not self.mask_wearing_status:
            return
            
        # Influence based on connection type
        for connection_id in self.network_connections:
            connection = network.get_person(connection_id)
            if connection:
                # Calculate influence based on connection type weights
                influence = influence_strength
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
        # Only non-mask wearers can change their status
        if not self.mask_wearing_status:
            # Combined influence of risk perception and social influence
            combined_influence = self.risk_perception * 0.5 + self.social_influence * 0.5
            
            if combined_influence > threshold:
                self.mask_wearing_status = True
                # Reset social influence after decision
                self.social_influence = 0.0

    def receive_influence(self, influence):
        """
        Receive influence from another agent.
        
        Args:
            influence (float): The amount of influence received.
        """
        self.social_influence += influence
        # Set received_information to true once influenced
        self.received_information = True

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
        if not hasattr(self, 'simulation'):
            print("Warning: Network has no reference to simulation, cannot propagate influence")
            return
            
        mask_wearers = [agent for agent in self.simulation.agents if agent.mask_wearing_status]
        print(f"Propagating influence from {len(mask_wearers)} mask wearers with strength {influence_strength}")
        
        # Only propagate from mask wearers
        for agent in mask_wearers:
            agent.influence_others(self, influence_strength, family_weight, work_school_weight, community_weight)

    def get_person(self, agent_id):
        """
        Get a person by their agent ID.
        
        Args:
            agent_id (int): The ID of the agent.
        
        Returns:
            Person: The person object with the given ID.
        """
        if hasattr(self, 'simulation'):
            return self.simulation.agent_dict.get(agent_id)
        return None

    def get_family_connections(self, agent_id):
        """
        Get family connections for a given agent.
        
        Args:
            agent_id (int): The ID of the agent.
        
        Returns:
            list: List of family connection IDs.
        """
        return self.network_structure.get(str(agent_id), {}).get('family', [])

    def get_work_school_connections(self, agent_id):
        """
        Get work/school connections for a given agent.
        
        Args:
            agent_id (int): The ID of the agent.
        
        Returns:
            list: List of work/school connection IDs.
        """
        return self.network_structure.get(str(agent_id), {}).get('work_school', [])

class Simulation:
    """
    Main simulation class that coordinates the agent-based model.
    
    Attributes:
        agents (list of Person): List of agents in the simulation.
        agent_dict (dict): Dictionary mapping agent_id to Person objects.
        network (Network): The social network of agents.
        days (int): Number of days to simulate.
        intervention_days (list): List of days when government interventions occur.
    """
    def __init__(self, agents, network, days, intervention_days):
        self.agents = agents
        self.agent_dict = {agent.agent_id: agent for agent in agents}  # Create map of agent_id to agent
        self.network = network
        self.days = days
        self.intervention_days = intervention_days
        
        # Store reference to this simulation in the network for agent lookup
        self.network.simulation = self

    def run(self):
        """
        Run the simulation over the specified number of days.
        """
        print(f"Starting simulation with {len(self.agents)} agents over {self.days} days...")
        print(f"Initial mask wearers: {sum(1 for agent in self.agents if agent.mask_wearing_status)}")
        
        # Set a lower threshold to make adoption easier
        mask_adoption_threshold = 0.3
        
        for day in range(self.days):
            influence_strength = adjust_influence_strength(day, self.intervention_days)
            propagation_speed = adjust_propagation_speed(day, self.intervention_days)
            
            print(f"\nDay {day} - Influence strength: {influence_strength}, Propagation speed: {propagation_speed}")
            
            # Propagate influence with adjusted influence strength
            self.network.propagate_influence(influence_strength, 1.0, 0.75, 0.5)  # Example weights
            
            # Count before changes
            before_count = sum(1 for agent in self.agents if agent.mask_wearing_status)
            
            # Apply changes based on accumulated influence
            for agent in self.agents:
                agent.change_mask_wearing_status(mask_adoption_threshold)
                
            # Calculate and print daily stats
            mask_wearers = sum(1 for agent in self.agents if agent.mask_wearing_status)
            new_adopters = mask_wearers - before_count
            print(f"Day {day}: {mask_wearers} agents wearing masks ({mask_wearers/len(self.agents)*100:.2f}%), New adopters: {new_adopters}")
            
            # If no new adopters for 5 consecutive days, break early
            if new_adopters == 0 and day > 5:
                print(f"No new mask adopters. Ending simulation early at day {day}.")
                break

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
                     row['risk_perception'], network_structure[str(row['agent_id'])]['all']) 
              for _, row in agent_df.iterrows()]
    network = Network(network_structure)

    # Define intervention days for dynamic influence adjustments
    intervention_days = [10, 20, 30]  # Example intervention days

    # Initialize and run simulation
    simulation = Simulation(agents, network, days=39, intervention_days=intervention_days)
    simulation.run()

    # Evaluate and visualize results
    results = simulation.evaluate()
    simulation.visualize()

    # Save results
    simulation.save_results("results.csv")

if __name__ == "__main__":
    main()