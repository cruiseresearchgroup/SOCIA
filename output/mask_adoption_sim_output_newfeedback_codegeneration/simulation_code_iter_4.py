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

def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1 / (1 + np.exp(-x))

class Person:
    """Agent representing individual mask-wearing behaviour."""

    # These class-level parameters will be fine-tuned in Simulation.calibrate_parameters()
    influence_probability = 0.05
    risk_perception_effect = 1.0
    social_influence_decay = 0.01
    decision_threshold = 0.5  # This acts as a bias term in decision_value
    network_influence_weight = 0.05
    min_risk_threshold = 0.7
    environmental_risk = 0.15

    def __init__(self, agent_id: int, mask_wearing_status: bool, risk_perception: float, network_connections: List[int]):
        self.agent_id = agent_id
        self.mask_wearing_status = bool(mask_wearing_status)
        self.risk_perception = float(risk_perception)
        self.network_connections = network_connections
        self.social_influence = np.random.normal(loc=0.5, scale=0.1)
        self.consistent_behavior_days = 0
        # Individual variation
        self.personal_threshold = Person.decision_threshold * np.random.normal(loc=1.0, scale=0.1)

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    def decide_to_wear_mask(self):
        """Decide mask-wearing based on risk, social influence and environment."""
        # 1. Social influence (diminishing returns via cap)
        influence_factor = min(0.8, self.social_influence * Person.influence_probability)

        # 2. Risk perception (with floor for high-risk individuals)
        if self.risk_perception >= Person.min_risk_threshold:
            risk_factor = max(0.6, self.risk_perception * Person.risk_perception_effect)
        else:
            risk_factor = self.risk_perception * Person.risk_perception_effect

        # 3. Environmental risk (global & possibly time-varying)
        environmental_factor = Person.environmental_risk

        # 4. Habit persistence
        habit_strength = min(0.25, self.consistent_behavior_days * 0.02)
        habit_adjustment = habit_strength if self.mask_wearing_status else -habit_strength

        # Final decision value
        decision_value = influence_factor + risk_factor + environmental_factor + habit_adjustment

        previous_status = self.mask_wearing_status
        self.mask_wearing_status = decision_value >= self.personal_threshold

        # Update habit counter
        if self.mask_wearing_status == previous_status:
            self.consistent_behavior_days += 1
        else:
            self.consistent_behavior_days = 0

        # Decay social influence slightly each day
        dynamic_decay = 0.005 + (0.04 * len(self.network_connections))
        self.social_influence = max(0.0, self.social_influence - dynamic_decay)

    # ------------------------------------------------------------------
    # Influence neighbours
    # ------------------------------------------------------------------
    def influence_others(self, network: nx.Graph):
        influence_changes = {}
        for neighbour in self.network_connections:
            current = network.nodes[neighbour].get('social_influence', 0.0)
            influence_changes[neighbour] = current + Person.network_influence_weight
        nx.set_node_attributes(network, influence_changes, 'social_influence')

class SocialNetwork:
    """Wrapper around networkx graph with propagation logic."""

    information_spread_rate = 0.06
    reinforcement_factor = 0.02

    def __init__(self, structure: Dict[int, Dict[str, List[int]]]):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(structure.keys())
        for node_id, conn in structure.items():
            for neighbour in conn['all']:
                self.graph.add_edge(node_id, neighbour)
                self.graph.nodes[neighbour].setdefault('social_influence', 0.0)

    def propagate_behavior(self):
        influence_changes = {}
        for node in self.graph.nodes:
            person: Person = self.graph.nodes[node].get('person')
            if person is None:
                continue
            person_status = int(person.mask_wearing_status)
            neighbours = list(self.graph.neighbors(node))
            if not neighbours:
                continue
            same, diff = 0, 0
            for n in neighbours:
                neigh_person: Person = self.graph.nodes[n].get('person')
                if neigh_person is None:
                    continue
                if person_status == int(neigh_person.mask_wearing_status):
                    same += 1
                else:
                    diff += 1
            diff_influence = SocialNetwork.information_spread_rate * diff
            same_influence = SocialNetwork.reinforcement_factor * same
            random_influence = np.random.normal(loc=0.0, scale=0.04)
            influence_changes[node] = diff_influence + same_influence + random_influence

        # Apply influence and let agents decide again
        for node, change in influence_changes.items():
            person: Person = self.graph.nodes[node].get('person')
            if person is None:
                continue
            person.social_influence = np.clip(person.social_influence + change, 0, 1.0)
            person.decide_to_wear_mask()

class Simulation:
    """Run the mask-wearing adoption simulation."""

    def __init__(self):
        self.agents: List[Person] = self._load_agents()
        self.social_network: SocialNetwork = self._load_network()
        self._calibrate_parameters()

    # --------------------------------------------------------------
    # Data loading helpers
    # --------------------------------------------------------------
    def _load_agents(self) -> List[Person]:
        agents: List[Person] = []
        data = pd.read_csv(agent_file)
        for _, row in data.iterrows():
            agents.append(
                Person(
                    agent_id=int(row['agent_id']),
                    mask_wearing_status=bool(row['initial_mask_wearing']),
                    risk_perception=float(row['risk_perception']),
                    network_connections=[]
                )
            )
        return agents

    def _load_network(self) -> SocialNetwork:
        with open(network_file, 'r') as f:
            structure = json.load(f)
        sn = SocialNetwork(structure)
        for ag in self.agents:
            ag.network_connections = structure[str(ag.agent_id)]['all']
            sn.graph.nodes[ag.agent_id]['person'] = ag
        return sn

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------
    def _calibrate_parameters(self):
        print("Calibrating model parameters using training data…")
        train_df = pd.read_csv(train_file)

        # -------------------- Risk perception effect --------------------
        init_df = train_df[train_df['day'] == 0]
        if 'risk_perception' in init_df.columns:
            corr = init_df['risk_perception'].corr(init_df['wearing_mask'].astype(int))
        else:
            # Merge with agent attributes to obtain risk perception values
            agent_attrs = pd.read_csv(agent_file)[['agent_id', 'risk_perception']]
            merged = init_df.merge(agent_attrs, on='agent_id', how='left')
            corr = merged['risk_perception'].corr(merged['wearing_mask'].astype(int))
        Person.risk_perception_effect = 0.6 + abs(corr)  # 0.6-1.6
        print(f"  risk_perception_effect -> {Person.risk_perception_effect:.3f}")

        # -------------------- Information spread rate -------------------
        daily_rates = train_df.groupby('day')['wearing_mask'].mean()
        daily_change = daily_rates.diff().abs().mean()
        SocialNetwork.information_spread_rate = np.clip(0.03 + daily_change * 1.0, 0.03, 0.12)
        print(f"  information_spread_rate -> {SocialNetwork.information_spread_rate:.3f}")

        # -------------------- Network influence weight ------------------
        transitions = train_df.groupby(['agent_id']).apply(lambda df: df['wearing_mask'].diff().abs().sum())
        transition_ratio = transitions.mean() / 10  # rough scaling
        Person.network_influence_weight = np.clip(0.02 + transition_ratio, 0.02, 0.12)
        print(f"  network_influence_weight -> {Person.network_influence_weight:.3f}")

        # -------------------- Influence probability & decision threshold --------
        Person.influence_probability = np.clip(0.05 + daily_change * 3, 0.05, 0.15)
        Person.decision_threshold = max(0.3, 0.5 - daily_change * 1.5)
        print(f"  influence_probability    -> {Person.influence_probability:.3f}")
        print(f"  decision_threshold       -> {Person.decision_threshold:.3f}")

        # -------------------- Reset personal thresholds with new global value ----
        for ag in self.agents:
            ag.personal_threshold = Person.decision_threshold * np.random.normal(loc=1.0, scale=0.1)

    # --------------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------------
    def run(self, start_day: int = 30, end_day: int = 39, env_risk_growth_rate: float = 0.0):
        print(f"Running simulation from day {start_day} to {end_day} (env growth {env_risk_growth_rate:.3f})…")
        total_days = end_day - start_day + 1
        for day_offset in range(total_days):
            current_day = start_day + day_offset
            # Adjust environmental risk if a growth rate is provided
            if env_risk_growth_rate != 0.0:
                Person.environmental_risk = np.clip(Person.environmental_risk + env_risk_growth_rate, 0.0, 0.45)
            # Propagate behaviour in the network
            self.social_network.propagate_behavior()
            if day_offset % 2 == 0 or current_day == end_day:
                wearing = sum(int(a.mask_wearing_status) for a in self.agents)
                print(f"  Day {current_day}: {wearing}/{len(self.agents)} wearing ({wearing/len(self.agents)*100:.2f}%)  envRisk={Person.environmental_risk:.3f}")

    # --------------------------------------------------------------
    # Utility – save final state
    # --------------------------------------------------------------
    def save_results(self, path: str):
        # Create output directory if it doesn't exist
        output_dir = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to the specified directory
        output_path = os.path.join(output_dir, path)
        
        df = pd.DataFrame({
            'agent_id': [ag.agent_id for ag in self.agents],
            'mask_wearing_status': [ag.mask_wearing_status for ag in self.agents]
        })
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

# ----------------------------------------------------------------------
# Main entry: run baseline, high-risk and intervention scenarios
# ----------------------------------------------------------------------

def main():
    print("\n============= MASK-WEARING BEHAVIOUR SIMULATION – ITER 4 =============\n")
    
    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Scenario 1: Baseline with dynamic risk --------------------
    print("SCENARIO 1 – Baseline (dynamic environmental risk)")
    baseline_sim = Simulation()
    initial_env_risk = Person.environmental_risk  # usually 0.15
    # Growth to 0.35 over 10 days
    growth_rate = (0.35 - initial_env_risk) / 9
    baseline_sim.run(env_risk_growth_rate=growth_rate)
    baseline_sim.save_results("baseline_results_iter4.csv")

    # -------------------- Scenario 2: High environmental risk (static) --------------
    print("\nSCENARIO 2 – High Environmental Risk (static 0.35)")
    Person.environmental_risk = 0.35
    high_risk_sim = Simulation()  # re-instantiate to reset agents with new class params
    high_risk_sim.run(env_risk_growth_rate=0.0)
    high_risk_sim.save_results("high_risk_results_iter4.csv")

    # -------------------- Scenario 3: Targeted intervention -------------------------
    print("\nSCENARIO 3 – Targeted Intervention + dynamic risk")
    Person.environmental_risk = initial_env_risk
    intervention_sim = Simulation()
    # Identify top 5% most connected agents
    topN = int(0.05 * len(intervention_sim.agents))
    influential = sorted(intervention_sim.agents, key=lambda a: len(a.network_connections), reverse=True)[:topN]
    for ag in influential:
        ag.social_influence += 0.6  # large boost
    intervention_sim.run(env_risk_growth_rate=growth_rate)
    intervention_sim.save_results("intervention_results_iter4.csv")
    
    # Save comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Plot mask wearing rates comparison
    scenarios = ['Initial', 'Baseline', 'High Risk', 'Intervention']
    initial_rate = sum(agent.mask_wearing_status for agent in baseline_sim.agents) / len(baseline_sim.agents) * 100
    baseline_rate = sum(agent.mask_wearing_status for agent in baseline_sim.agents) / len(baseline_sim.agents) * 100
    high_risk_rate = sum(agent.mask_wearing_status for agent in high_risk_sim.agents) / len(high_risk_sim.agents) * 100
    intervention_rate = sum(agent.mask_wearing_status for agent in intervention_sim.agents) / len(intervention_sim.agents) * 100
    
    rates = [initial_rate, baseline_rate, high_risk_rate, intervention_rate]
    
    plt.bar(scenarios, rates, color=['gray', 'blue', 'red', 'green'])
    plt.title('Mask Wearing Rates by Scenario (Iter4)')
    plt.ylabel('Percentage of Population (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for i, rate in enumerate(rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.tight_layout()
    scenario_comparison_path = os.path.join(output_dir, 'scenario_comparison_iter4.png')
    plt.savefig(scenario_comparison_path)
    plt.close()
    print(f"\nScenario comparison chart saved to: {scenario_comparison_path}")

    print("\nSimulation finished – results saved for all scenarios (iter4).")

if __name__ == "__main__":
    main() 