import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from json.decoder import JSONDecodeError

# Define the paths for data files using environment variables
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Data file paths
agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
network_file = os.path.join(DATA_DIR, "social_network.json")
train_data_file = os.path.join(DATA_DIR, "train_data.csv")


class Person:
    """
    Represents an individual in the simulation with attributes
    and behaviors related to mask-wearing.

    Attributes:
        mask_wearing_status (bool): Indicates if the person is currently wearing a mask.
        social_influence (float): Influence from social connections on mask-wearing.
        susceptibility (float): Likelihood to adopt mask-wearing based on risk perception.
        network_connections (list): IDs of connected individuals in the social network.
    """
    def __init__(self, agent_id: int, age: int, occupation: str, risk_perception: float, initial_mask_wearing: bool, network_connections: list):
        self.agent_id = agent_id
        self.age = age
        self.occupation = occupation
        self.risk_perception = risk_perception
        self.mask_wearing_status = initial_mask_wearing
        self.social_influence = 0.0
        self.susceptibility = self.calculate_susceptibility()
        self.network_connections = network_connections
    
    def calculate_susceptibility(self) -> float:
        """
        Calculates susceptibility based on personal attributes.

        Returns:
            float: Calculated susceptibility.
        """
        return self.risk_perception * 0.5  # Example calculation

    def adopt_mask_wearing(self, threshold: float):
        """
        Determines if the person adopts mask-wearing behavior.

        Args:
            threshold (float): Threshold calculated from risk perception and social influence.
        """
        if (self.social_influence + self.risk_perception + self.susceptibility) > threshold:
            self.mask_wearing_status = True

    def influence_others(self, influence_factor: float) -> float:
        """
        Influences connected individuals to adopt mask-wearing behavior.

        Args:
            influence_factor (float): Strength of influence on others.
        Returns:
            float: Influence factor if wearing a mask, otherwise 0.
        """
        if self.mask_wearing_status:
            return influence_factor
        return 0.0


class SocialNetwork:
    """
    Represents the social network structure and interactions.

    Attributes:
        network_structure (dict): Structure of the network with connections.
        connection_strengths (dict): Strengths of connections based on type.
    """
    def __init__(self, network_structure: dict, connection_strengths: dict):
        self.network_structure = network_structure
        self.connection_strengths = connection_strengths

    def propagate_influence(self, persons: dict, influence_factor: float):
        """
        Propagates influence through the network.

        Args:
            persons (dict): Dictionary of Person objects indexed by agent_id.
            influence_factor (float): Influence factor for propagation.
        """
        for agent_id, person in persons.items():
            total_influence = 0.0
            for connection_id in person.network_connections:
                if connection_id in persons:
                    connected_person = persons[connection_id]
                    influence = connected_person.influence_others(influence_factor)
                    connection_type = self.get_connection_type(agent_id, connection_id)
                    weight = self.connection_strengths.get(connection_type, 1)
                    total_influence += influence * weight
            person.social_influence += total_influence

    def get_connection_type(self, agent_id: int, connection_id: int) -> str:
        """
        Determines the type of connection between two agents.

        Args:
            agent_id (int): ID of the agent.
            connection_id (int): ID of the connected agent.
        
        Returns:
            str: The type of connection (e.g., 'family', 'work_school', 'community').
        """
        if agent_id in self.network_structure:
            for connection_type, connections in self.network_structure[agent_id].items():
                if connection_id in connections:
                    return connection_type
        return "all"


class Simulation:
    """
    Main simulation class coordinating the execution of the model.

    Attributes:
        persons (dict): Dictionary of Person entities.
        network (SocialNetwork): Social network managing connections.
        parameters (dict): Simulation parameters.
    """
    def __init__(self):
        self.persons = self.load_agents()
        self.network = self.load_network()
        self.parameters = {
            "initial_mask_wearing_percentage": 0.3,
            "influence_factor": 0.1,
            "network_density": 0.05
        }

    def load_agents(self) -> dict:
        """
        Loads agents from the agent attributes file.

        Returns:
            dict: Dictionary of Person entities indexed by agent_id.
        """
        try:
            df = pd.read_csv(agent_file)
            persons = {}
            for _, row in df.iterrows():
                agent_id = row['agent_id']
                persons[agent_id] = Person(
                    agent_id,
                    row['age'],
                    row['occupation'],
                    row['risk_perception'],
                    row['initial_mask_wearing'],
                    []  # Connections to be filled
                )
            return persons
        except FileNotFoundError:
            print(f"Error: {agent_file} not found.")
            return {}
        except KeyError as ke:
            print(f"Missing expected column {ke} in {agent_file}.")
            return {}
        except pd.errors.EmptyDataError:
            print(f"Error: {agent_file} is empty.")
            return {}
        except Exception as e:
            print(f"An error occurred while loading agents: {e}")
            return {}

    def load_network(self) -> SocialNetwork:
        """
        Loads the social network from the JSON file.

        Returns:
            SocialNetwork: Initialized SocialNetwork object.
        """
        try:
            with open(network_file, 'r') as f:
                network_data = json.load(f)
            for agent_id, connections in network_data.items():
                if agent_id in self.persons:
                    self.persons[agent_id].network_connections = connections.get('all', [])
            connection_strengths = {
                "family": 1.5,
                "work_school": 1.0,
                "community": 0.5
            }
            return SocialNetwork(network_data, connection_strengths)
        except FileNotFoundError:
            print(f"Error: {network_file} not found.")
            return SocialNetwork({}, {})
        except JSONDecodeError:
            print(f"Error: Failed to decode JSON from {network_file}.")
            return SocialNetwork({}, {})
        except KeyError as ke:
            print(f"Missing expected key {ke} in network data.")
            return SocialNetwork({}, {})
        except Exception as e:
            print(f"An error occurred while loading the network: {e}")
            return SocialNetwork({}, {})

    def run(self):
        """
        Executes the simulation loop for the specified prediction period.
        """
        prediction_period = range(30, 40)
        intervention_day = 10
        for day in prediction_period:
            if day >= intervention_day:
                self.apply_government_intervention()
            self.network.propagate_influence(self.persons, self.parameters['influence_factor'])
            for person in self.persons.values():
                threshold = self.parameters['initial_mask_wearing_percentage']
                person.adopt_mask_wearing(threshold)

    def apply_government_intervention(self):
        """
        Applies government intervention effects to the simulation.
        """
        for person in self.persons.values():
            person.risk_perception += 0.1  # Example effect of intervention

    def evaluate(self) -> dict:
        """
        Evaluates the simulation results based on specified metrics.

        Returns:
            dict: Dictionary of evaluation metric results.
        """
        metrics = ["RMSE", "Peak Adoption Rate Error", "Time-to-Peak Error"]
        results = {}
        try:
            df = pd.read_csv(train_data_file)
            actual_rates = df.groupby('day')['wearing_mask'].mean().tolist()
            predicted_rates = [
                sum(1 for p in self.persons.values() if p.mask_wearing_status) / len(self.persons)
                for day in range(30, 40)
            ]
            results["RMSE"] = np.sqrt(np.mean((np.array(predicted_rates) - np.array(actual_rates[-10:])) ** 2))
            results["Peak Adoption Rate Error"] = abs(max(predicted_rates) - max(actual_rates[-10:]))
            results["Time-to-Peak Error"] = abs(np.argmax(predicted_rates) - np.argmax(actual_rates[-10:]))
        except FileNotFoundError:
            print(f"Error: {train_data_file} not found.")
            for metric in metrics:
                results[metric] = None
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            for metric in metrics:
                results[metric] = None
        return results

    def visualize(self):
        """
        Visualizes the simulation results using matplotlib.
        """
        try:
            days = list(range(30, 40))
            adoption_rates = [
                sum(1 for p in self.persons.values() if p.mask_wearing_status) / len(self.persons)
                for _ in days
            ]
            plt.plot(days, adoption_rates, label='Predicted Adoption Rates')
            plt.xlabel('Day')
            plt.ylabel('Adoption Rate')
            plt.title('Mask-Wearing Adoption Rates Over Time')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"An error occurred during visualization: {e}")

    def save_results(self, filename: str):
        """
        Saves simulation results to a file.

        Args:
            filename (str): Path to the file where results will be saved.
        """
        try:
            results = self.evaluate()
            df_results = pd.DataFrame([results], columns=results.keys())
            df_results.to_csv(filename, index=False)
        except Exception as e:
            print(f"An error occurred while saving results: {e}")


def main():
    """
    Main function to initialize, run, visualize, and save the simulation.
    """
    sim = Simulation()
    sim.run()
    sim.visualize()
    sim.save_results("results.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()