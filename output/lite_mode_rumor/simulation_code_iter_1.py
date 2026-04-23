import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend suitable for servers
import matplotlib.pyplot as plt
import csv
import networkx as nx
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class InfoHistoryEntry:
    time_step: int
    num_informed: int

class Agent:
    """
    Represents an agent in the social network.

    Attributes:
        agent_id (int): Unique identifier for the agent.
        has_information (bool): Whether the agent has received the information/rumor.
        influence (float): Influence score of the agent (affects probability to share).
    """
    def __init__(self, agent_id: int, influence: float) -> None:
        """
        Initialize an agent.

        Args:
            agent_id (int): Agent's unique identifier.
            influence (float): Agent's influence value.
        """
        self.agent_id: int = agent_id
        self.has_information: bool = False
        self.influence: float = influence

    def decide_to_share(self, base_share_prob: float) -> bool:
        """
        Decide probabilistically whether to share the information.

        Args:
            base_share_prob (float): Base probability to share.

        Returns:
            bool: True if the agent decides to share, False otherwise.
        """
        prob: float = base_share_prob * self.influence
        decision: bool = random.random() < prob
        return decision

class SocialNetworkSimulation:
    """
    Simulates the spread of information/rumor in a social network.
    """

    def __init__(
        self,
        num_agents: int = 500,
        network_type: str = 'watts_strogatz',
        k: int = 8,
        p: float = 0.1,
        time_steps: int = 100,
        base_share_prob: float = 0.2,
    ) -> None:
        """
        Initialize the simulation.

        Args:
            num_agents (int): Number of agents in the simulation.
            network_type (str): Type of network ('watts_strogatz', 'barabasi_albert', etc.).
            k (int): Each node is joined with its k nearest neighbors in ring topology (for Watts-Strogatz).
            p (float): The probability of rewiring each edge (for Watts-Strogatz).
            time_steps (int): Number of simulation time steps.
            base_share_prob (float): Base probability for an agent to share the information.
        """
        self.num_agents: int = num_agents
        self.time_steps: int = time_steps
        self.base_share_prob: float = base_share_prob
        self.network: nx.Graph = self._generate_network(network_type, k, p)
        self.agents: Dict[int, Agent] = self._init_agents()
        self.information_history: List[InfoHistoryEntry] = []

    def _generate_network(self, network_type: str, k: int, p: float) -> nx.Graph:
        """
        Generate the social network graph.

        Args:
            network_type (str): Type of network to generate. Supported: 'watts_strogatz'. 
                If not 'watts_strogatz', falls back to 'erdos_renyi' with p=0.02.
            k (int): Each node is joined with its k nearest neighbors in ring topology (for Watts-Strogatz).
            p (float): The probability of rewiring each edge (for Watts-Strogatz).

        Returns:
            networkx.Graph: The generated network.
        """
        if network_type == 'watts_strogatz':
            G = nx.watts_strogatz_graph(self.num_agents, k, p)
        else:
            # For unsupported types, fallback to Erdos-Renyi
            G = nx.erdos_renyi_graph(self.num_agents, 0.02)
        return G

    def _init_agents(self) -> Dict[int, Agent]:
        """
        Initialize agents with random influence.

        Returns:
            dict: Mapping from agent_id to Agent instance.
        """
        influences: np.ndarray = np.clip(np.random.normal(loc=1.0, scale=0.2, size=self.num_agents), 0.5, 1.5)
        agents: Dict[int, Agent] = {i: Agent(i, float(influences[i])) for i in range(self.num_agents)}
        return agents

    def _seed_information(self) -> None:
        """
        Seed the information/rumor with one or more initial agents.
        """
        initial_seed: int = random.randint(0, self.num_agents - 1)
        self.agents[initial_seed].has_information = True

    def run(self) -> None:
        """
        Run the simulation.

        Simulates the progression of information/rumor spread over the network for the specified number of time steps.
        Results are stored internally.
        """
        self._seed_information()
        informed_set: set = {agent_id for agent_id, agent in self.agents.items() if agent.has_information}
        for t in range(self.time_steps):
            newly_informed: set = set()
            for agent_id in informed_set:
                agent = self.agents[agent_id]
                if agent.has_information:
                    if agent.decide_to_share(self.base_share_prob):
                        neighbors = list(self.network.neighbors(agent_id))
                        for neighbor_id in neighbors:
                            neighbor = self.agents[neighbor_id]
                            if not neighbor.has_information:
                                accept_prob = min(1.0, 0.4 + 0.4 * agent.influence)
                                if random.random() < accept_prob:
                                    newly_informed.add(neighbor_id)
            for nid in newly_informed:
                self.agents[nid].has_information = True
            informed_set.update(newly_informed)
            count_informed: int = len(informed_set)
            self.information_history.append(InfoHistoryEntry(time_step=t, num_informed=count_informed))

    def _plot_results(self, informed_counts: List[int], picture_path: Optional[str] = None) -> None:
        """
        Plot the number of informed agents over time and save the plot to a file.

        Args:
            informed_counts (list): List of counts of informed agents at each time step.
            picture_path (Optional[str]): Path to save the plot PNG file. If None, no plot is saved.

        Side Effects:
            Writes a PNG image of the results to the specified file.
        """
        if picture_path is None:
            return
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(informed_counts) + 1), informed_counts, marker='o', linestyle='-', color='b')
            plt.xlabel('Time Step')
            plt.ylabel('Number of Informed Agents')
            plt.title('Spread of Information in Social Network')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(picture_path)
            plt.close()
        except (IOError, OSError) as e:
            print(f"Error saving plot to {picture_path}: {e}")

    def save_results(self, result_path: str) -> None:
        """
        Save simulation results to a CSV file.

        Args:
            result_path (str): Path to save the CSV file.

        Side Effects:
            Writes simulation history to a CSV file at the specified path.
        """
        try:
            with open(result_path, mode='w', newline='') as csvfile:
                fieldnames = ['time_step', 'num_informed']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for entry in self.information_history:
                    writer.writerow({'time_step': entry.time_step, 'num_informed': entry.num_informed})
        except (IOError, OSError) as e:
            print(f"Error saving results to {result_path}: {e}")

def main() -> None:
    """
    Entry point for running the simulation, saving results, and generating visualization.

    Side Effects:
        - Creates output directory if not present.
        - Writes results to a CSV file.
        - Writes a PNG plot of the simulation results.
    """
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")

    if not project_root or not data_path:
        print("Error: Both PROJECT_ROOT and DATA_PATH environment variables must be set.")
        print("Exiting without running simulation.")
        return

    data_dir = os.path.join(project_root, data_path)
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating data directory {data_dir}: {e}")
        print("Exiting without running simulation.")
        return

    result_path = os.path.join(data_dir, "results.csv")
    picture_path = os.path.join(data_dir, "figure.png")

    sim = SocialNetworkSimulation()
    sim.run()
    sim.save_results(result_path)
    informed_counts = [entry.num_informed for entry in sim.information_history]
    sim._plot_results(
        informed_counts,
        picture_path=picture_path
    )

main()