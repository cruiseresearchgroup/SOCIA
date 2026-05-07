import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

class Person:
    """
    Represents an individual in the simulation with attributes and behaviors related to mask-wearing.
    """

    def __init__(self, id, mask_wearing_status, social_influence, network_connections, risk_perception, received_info):
        self.id = id
        self.mask_wearing_status = mask_wearing_status
        self.social_influence = social_influence
        self.network_connections = network_connections
        self.risk_perception = risk_perception
        self.received_info = received_info

    def decide_to_wear_mask(self, influence_threshold):
        """
        Determines if the person decides to wear a mask based on social influence, risk perception, and received information.
        """
        if self.social_influence > influence_threshold or self.risk_perception > influence_threshold or self.received_info:
            self.mask_wearing_status = True

    def influence_others(self, influence_factor):
        """
        Influences connected individuals to consider wearing masks based on personal mask-wearing status.
        """
        if self.mask_wearing_status:
            for connection in self.network_connections:
                connection.social_influence += influence_factor


class SocialNetwork:
    """
    Represents the social network of individuals in the simulation, facilitating the propagation of influence.
    """

    def __init__(self, network_structure, connection_strength):
        self.network_structure = network_structure
        self.connection_strength = connection_strength

    def propagate_influence(self):
        """
        Simulates the spread of information and influence through the network.
        """
        for node in self.network_structure.nodes:
            for neighbor in self.network_structure.neighbors(node):
                # Increase social influence based on connection strength
                self.network_structure.nodes[neighbor]['social_influence'] += self.connection_strength


class Simulation:
    """
    Main simulation class that coordinates the execution of the simulation.
    """

    def __init__(self, agents, social_network, simulation_steps, influence_threshold):
        self.agents = agents
        self.social_network = social_network
        self.simulation_steps = simulation_steps
        self.influence_threshold = influence_threshold
        self.results = []

    def run(self):
        """
        Executes the simulation loop for the specified number of steps.
        """
        for step in range(self.simulation_steps):
            for agent in self.agents:
                agent.decide_to_wear_mask(self.influence_threshold)
                agent.influence_others(influence_factor=0.1)  # Example influence factor
            self.social_network.propagate_influence()
            # Collect and store results for analysis
            self.results.append(self.collect_data())

    def collect_data(self):
        """
        Collects data from the current simulation step.
        """
        return {
            "step": len(self.results),
            "adoption_rate": sum(agent.mask_wearing_status for agent in self.agents) / len(self.agents)
        }

    def visualize(self):
        """
        Visualizes the simulation results.
        """
        steps = [result['step'] for result in self.results]
        adoption_rates = [result['adoption_rate'] for result in self.results]
        plt.plot(steps, adoption_rates)
        plt.xlabel("Simulation Step")
        plt.ylabel("Mask Adoption Rate")
        plt.title("Mask Adoption Over Time")
        plt.show()

    def save_results(self, filename):
        """
        Saves the simulation results to a CSV file.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)


def load_data():
    """
    Loads data from CSV and JSON files to initialize the simulation.
    """
    agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
    network_file = os.path.join(DATA_DIR, "social_network.json")

    agent_data = pd.read_csv(agent_file)
    with open(network_file, 'r') as f:
        network_data = json.load(f)

    agents = []
    for _, row in agent_data.iterrows():
        person = Person(
            id=row['agent_id'],
            mask_wearing_status=row['initial_mask_wearing'],
            social_influence=0.0,
            network_connections=[],
            risk_perception=row['risk_perception'],
            received_info=False
        )
        agents.append(person)

    G = nx.Graph()
    for person_id, connections in network_data.items():
        G.add_node(person_id)
        for conn_type in ['family', 'work_school', 'community']:
            for neighbor in connections[conn_type]:
                G.add_edge(person_id, neighbor)

    social_network = SocialNetwork(G, connection_strength=0.1)  # Example connection strength

    return agents, social_network


def main():
    """
    Main function to initialize, run, visualize, and save the simulation.
    """
    agents, social_network = load_data()
    simulation = Simulation(agents, social_network, simulation_steps=100, influence_threshold=0.5)
    simulation.run()
    simulation.visualize()
    simulation.save_results("results.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()