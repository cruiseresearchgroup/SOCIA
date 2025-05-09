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
    network_influence_weight = 0.05  # Define network influence weight

    def __init__(self, agent_id: int, mask_wearing_status: bool, risk_perception: float, network_connections: List[int]):
        """
        Initializes a Person instance.

        Args:
            agent_id (int): Unique identifier for the agent.
            mask_wearing_status (bool): Initial mask-wearing status.
            risk_perception (float): Perception of risk influencing mask-wearing decision.
            network_connections (List[int]): List of connected agent IDs.
        """
        self.agent_id = agent_id
        self.mask_wearing_status = bool(mask_wearing_status)
        self.risk_perception = risk_perception
        self.network_connections = network_connections
        self.social_influence = np.random.normal(loc=0.5, scale=0.1)

    def decide_to_wear_mask(self):
        """
        Decide whether to wear a mask based on social influence and risk perception.
        """
        influence_factor = self.social_influence * Person.influence_probability
        risk_factor = self.risk_perception * Person.risk_perception_effect
        decision_value = influence_factor + risk_factor
        self.mask_wearing_status = decision_value > Person.decision_threshold
        dynamic_decay = 0.01 + (0.1 * len(self.network_connections))  # Dynamic decay based on connections
        self.social_influence = max(0, self.social_influence - dynamic_decay)

    def influence_others(self, network: nx.Graph):
        """
        Influence connected agents to adopt mask-wearing behavior.
        """
        influence_changes = {}
        for connection in self.network_connections:
            current_influence = network.nodes[connection].get('social_influence', 0.0)
            influence_changes[connection] = current_influence + Person.network_influence_weight
        nx.set_node_attributes(network, influence_changes, 'social_influence')

class SocialNetwork:
    """
    Represents the social network structure and facilitates behavior propagation.
    
    Attributes:
        network (nx.Graph): The graph representing the social network.
    """
    def __init__(self, network_structure: Dict[int, Dict[str, List[int]]]):
        """
        Initializes the social network from the given structure.

        Args:
            network_structure (Dict[int, Dict[str, List[int]]]): Structure of the network.
        """
        self.network = nx.Graph()
        self.network.add_nodes_from(network_structure.keys())
        for person_id, connections in network_structure.items():
            for connection_id in connections['all']:
                self.network.add_edge(person_id, connection_id)
                self.network.nodes[connection_id].setdefault('social_influence', 0.0)

    def propagate_behavior(self, information_spread_rate: float):
        """
        Propagate mask-wearing behavior through the network.

        Args:
            information_spread_rate (float): Rate of information spread.
        """
        for node in self.network.nodes:
            person = self.network.nodes[node].get('person')
            if person:
                person_status = int(person.mask_wearing_status)
                influence_sum = sum(
                    information_spread_rate * (int(person_status != self.network.nodes[neighbor].get('person', Person(0, False, 0, [])).mask_wearing_status))
                    for neighbor in self.network.neighbors(node)
                    if 'person' in self.network.nodes[neighbor]
                )
                person.social_influence += influence_sum
                person.decide_to_wear_mask()

class Simulation:
    """
    Main simulation class to manage the setup, execution, and evaluation of the mask-wearing behavior model.
    """
    def __init__(self):
        """
        Initializes the simulation by loading agents and the social network.
        """
        self.agents = self.load_agents()
        self.social_network = self.load_social_network()
        self.calibrate_parameters()

    def load_agents(self) -> List[Person]:
        """
        Load agent data from the CSV file and initialize Person instances.

        Returns:
            List[Person]: List of initialized agents.
        """
        agents = []
        try:
            data = pd.read_csv(agent_file)
            if data.empty or not {'agent_id', 'initial_mask_wearing', 'risk_perception'}.issubset(data.columns):
                raise ValueError("Missing required columns or empty agent data file.")
            for _, row in data.iterrows():
                if not isinstance(row['agent_id'], int) or not isinstance(row['risk_perception'], (int, float)):
                    raise ValueError("Invalid data type in agent data file.")
                if not isinstance(row['initial_mask_wearing'], (bool, np.bool_)):
                    raise ValueError("Initial mask wearing status must be boolean.")
                agent = Person(
                    agent_id=row['agent_id'],
                    mask_wearing_status=row['initial_mask_wearing'],
                    risk_perception=row['risk_perception'],
                    network_connections=[]
                )
                agents.append(agent)
            if not agents:
                raise RuntimeError("No agents loaded from the file.")
        except (FileNotFoundError, pd.errors.ParserError, ValueError, PermissionError) as e:
            raise RuntimeError(f"Error loading agent file: {e}")
        return agents

    def load_social_network(self) -> SocialNetwork:
        """
        Load social network data from JSON file and initialize SocialNetwork.

        Returns:
            SocialNetwork: Initialized social network.
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
        except (FileNotFoundError, json.JSONDecodeError, KeyError, PermissionError) as e:
            raise RuntimeError(f"Error loading network file: {e}")
        return social_network

    def calibrate_parameters(self):
        """
        Calibrate the model parameters using the train_data.csv file.
        """
        try:
            train_data = pd.read_csv(train_file)
            # Example calibration logic (to be replaced with actual logic)
            Person.risk_perception_effect = 0.8
            SocialNetwork.information_spread_rate = 0.06
            Person.network_influence_weight = 0.05
        except (FileNotFoundError, pd.errors.ParserError, PermissionError) as e:
            raise RuntimeError(f"Error calibrating parameters from training data: {e}")

    def run(self, start_day: int = 30, end_day: int = 39) -> None:
        """
        Execute the simulation loop over the specified prediction period.

        Args:
            start_day (int): The starting day for the simulation.
            end_day (int): The ending day for the simulation.
        """
        for day in range(start_day, end_day + 1):
            self.social_network.propagate_behavior(0.05)

    def visualize(self) -> None:
        """
        Visualize the results of the simulation and save to file.
        """
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Mask Wearing Distribution histogram
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        mask_wearers = [int(agent.mask_wearing_status) for agent in self.agents]
        plt.hist(mask_wearers, bins=2)
        plt.title('Mask Wearing Distribution')
        plt.xlabel('Mask Wearing Status (0=No, 1=Yes)')
        plt.ylabel('Number of Agents')
        
        # 2. Risk Perception histogram
        plt.subplot(2, 2, 2)
        risk_perceptions = [agent.risk_perception for agent in self.agents]
        plt.hist(risk_perceptions, bins=10)
        plt.title('Risk Perception Distribution')
        plt.xlabel('Risk Perception Level')
        plt.ylabel('Number of Agents')
        
        # 3. Risk Perception vs Mask Wearing scatter plot
        plt.subplot(2, 2, 3)
        x = [agent.risk_perception for agent in self.agents]
        y = [int(agent.mask_wearing_status) for agent in self.agents]
        plt.scatter(x, y, alpha=0.5)
        plt.title('Risk Perception vs Mask Wearing')
        plt.xlabel('Risk Perception Level')
        plt.ylabel('Mask Wearing Status (0=No, 1=Yes)')
        
        # 4. Social Influence vs Mask Wearing scatter plot
        plt.subplot(2, 2, 4)
        social_influences = [agent.social_influence for agent in self.agents]
        plt.scatter(social_influences, y, alpha=0.5)
        plt.title('Social Influence vs Mask Wearing')
        plt.xlabel('Social Influence')
        plt.ylabel('Mask Wearing Status (0=No, 1=Yes)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'simulation_analysis.png'))
        plt.close()
        print(f"Comprehensive analysis chart saved to: {os.path.join(output_dir, 'simulation_analysis.png')}")
        
        # 5. Risk Perception Group mask wearing rates
        plt.figure(figsize=(10, 6))
        # Group data
        risk_groups = ['Low (<0.3)', 'Medium (0.3-0.7)', 'High (>0.7)']
        low_risk = [agent for agent in self.agents if agent.risk_perception < 0.3]
        med_risk = [agent for agent in self.agents if 0.3 <= agent.risk_perception < 0.7]
        high_risk = [agent for agent in self.agents if agent.risk_perception >= 0.7]
        
        mask_rates = []
        for group in [low_risk, med_risk, high_risk]:
            if group:
                rate = sum(int(agent.mask_wearing_status) for agent in group) / len(group)
                mask_rates.append(rate * 100)
            else:
                mask_rates.append(0)
        
        plt.bar(risk_groups, mask_rates)
        plt.title('Mask Wearing Rates by Risk Perception Group')
        plt.xlabel('Risk Perception Group')
        plt.ylabel('Mask Wearing Rate (%)')
        plt.ylim(0, 100)
        
        for i, rate in enumerate(mask_rates):
            plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_mask_relationship.png'))
        plt.close()
        print(f"Risk perception and mask wearing relationship chart saved to: {os.path.join(output_dir, 'risk_mask_relationship.png')}")

    def save_results(self, filename: str) -> None:
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

    def implement_government_intervention(self, intervention_strength: float) -> None:
        """
        Implement a government intervention that affects mask-wearing behavior.

        Args:
            intervention_strength (float): The strength of the intervention on influencing mask-wearing behavior.
        """
        for agent in self.agents:
            agent.social_influence += intervention_strength

def main():
    """
    Main function to initialize, run, visualize, and save the simulation.
    """
    simulation = Simulation()
    
    # Output initial state
    initial_mask_wearers = sum(int(agent.mask_wearing_status) for agent in simulation.agents)
    total_agents = len(simulation.agents)
    print(f"Initial state: {initial_mask_wearers}/{total_agents} agents wearing masks ({initial_mask_wearers/total_agents*100:.2f}%)")
    
    # Run simulation
    print(f"Starting simulation...")
    simulation.run()
    
    # Output final state
    final_mask_wearers = sum(int(agent.mask_wearing_status) for agent in simulation.agents)
    print(f"Final state: {final_mask_wearers}/{total_agents} agents wearing masks ({final_mask_wearers/total_agents*100:.2f}%)")
    print(f"Change: {final_mask_wearers - initial_mask_wearers} agents ({(final_mask_wearers - initial_mask_wearers)/total_agents*100:.2f}%)")
    
    # Visualize and save results
    simulation.visualize()
    simulation.save_results("results.csv")
    print(f"Simulation results saved to: {os.path.abspath('results.csv')}")
    
    # Analyze social network influence
    high_influence = sum(1 for agent in simulation.agents if agent.social_influence > 0.7)
    print(f"Agents with high social influence (>0.7): {high_influence}/{total_agents} ({high_influence/total_agents*100:.2f}%)")
    
    # Analyze mask wearing by different risk perception levels
    low_risk_agents = [agent for agent in simulation.agents if agent.risk_perception < 0.3]
    med_risk_agents = [agent for agent in simulation.agents if 0.3 <= agent.risk_perception < 0.7]
    high_risk_agents = [agent for agent in simulation.agents if agent.risk_perception >= 0.7]
    
    low_risk_wearing = sum(int(agent.mask_wearing_status) for agent in low_risk_agents)
    med_risk_wearing = sum(int(agent.mask_wearing_status) for agent in med_risk_agents)
    high_risk_wearing = sum(int(agent.mask_wearing_status) for agent in high_risk_agents)
    
    print("\nMask wearing by risk perception level:")
    if low_risk_agents:
        print(f"  Low risk group (<0.3): {low_risk_wearing}/{len(low_risk_agents)} wearing masks ({low_risk_wearing/len(low_risk_agents)*100:.2f}%)")
    if med_risk_agents:
        print(f"  Medium risk group (0.3-0.7): {med_risk_wearing}/{len(med_risk_agents)} wearing masks ({med_risk_wearing/len(med_risk_agents)*100:.2f}%)")
    if high_risk_agents:
        print(f"  High risk group (>0.7): {high_risk_wearing}/{len(high_risk_agents)} wearing masks ({high_risk_wearing/len(high_risk_agents)*100:.2f}%)")

main()