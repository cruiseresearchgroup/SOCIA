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
    # Add minimum risk threshold to ensure some agents still wear masks
    min_risk_threshold = 0.7
    # Add environmental risk that affects everyone
    environmental_risk = 0.15

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
        # Track days of consistent behavior
        self.consistent_behavior_days = 0
        # Personal threshold varies slightly for each individual
        self.personal_threshold = Person.decision_threshold * np.random.normal(loc=1.0, scale=0.1)

    def decide_to_wear_mask(self):
        """
        Decide whether to wear a mask based on social influence, risk perception, and environmental factors.
        
        The decision model now includes:
        1. Social influence with diminishing returns
        2. Risk perception with a minimum threshold for high-risk individuals
        3. Environmental risk that affects everyone
        4. Individual variation in decision threshold
        5. Behavioral consistency (habits tend to persist)
        """
        # Calculate social influence factor with diminishing returns
        influence_factor = min(0.7, self.social_influence * Person.influence_probability)
        
        # Risk perception factor with minimum threshold for high-risk individuals
        if self.risk_perception >= Person.min_risk_threshold:
            # High risk individuals maintain higher motivation to wear masks
            risk_factor = max(0.6, self.risk_perception * Person.risk_perception_effect)
        else:
            risk_factor = self.risk_perception * Person.risk_perception_effect
        
        # Add environmental risk factor
        environmental_factor = Person.environmental_risk
        
        # Calculate decision value with all factors
        decision_value = influence_factor + risk_factor + environmental_factor
        
        # Add behavioral consistency (habit) factor
        # If behavior has been consistent, it's harder to change
        if self.consistent_behavior_days > 0:
            habit_strength = min(0.2, self.consistent_behavior_days * 0.02)
            if self.mask_wearing_status:
                decision_value += habit_strength
            else:
                decision_value -= habit_strength
        
        # Make decision using personal threshold
        previous_status = self.mask_wearing_status
        self.mask_wearing_status = decision_value > self.personal_threshold
        
        # Update consistency counter
        if self.mask_wearing_status == previous_status:
            self.consistent_behavior_days += 1
        else:
            self.consistent_behavior_days = 0
        
        # Dynamic decay based on connections with a lower minimum
        dynamic_decay = 0.005 + (0.05 * len(self.network_connections))
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
    # Default information spread rate
    information_spread_rate = 0.06
    # Add reinforcement factor for similar behaviors
    reinforcement_factor = 0.02
    
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
        
        The propagation model now includes:
        1. Information spread based on behavior differences
        2. Reinforcement of similar behaviors (echo chamber effect)
        3. Weighted influence based on connection strength
        4. Random environmental influences

        Args:
            information_spread_rate (float): Rate of information spread.
        """
        # First pass: calculate influence changes for all nodes
        influence_changes = {}
        
        for node in self.network.nodes:
            person = self.network.nodes[node].get('person')
            if not person:
                continue
                
            person_status = int(person.mask_wearing_status)
            neighbors = list(self.network.neighbors(node))
            
            # Skip if no neighbors
            if not neighbors:
                continue
                
            # Count neighbors with same and different behavior
            same_behavior = 0
            diff_behavior = 0
            
            for neighbor in neighbors:
                if 'person' not in self.network.nodes[neighbor]:
                    continue
                    
                neighbor_person = self.network.nodes[neighbor]['person']
                neighbor_status = int(neighbor_person.mask_wearing_status)
                
                if person_status == neighbor_status:
                    same_behavior += 1
                else:
                    diff_behavior += 1
            
            # Calculate total influence change
            # 1. Information spread from different behaviors
            diff_influence = information_spread_rate * diff_behavior
            
            # 2. Reinforcement from similar behaviors (echo chamber)
            same_influence = SocialNetwork.reinforcement_factor * same_behavior
            
            # 3. Random environmental factor (news, policy changes, etc.)
            random_influence = np.random.normal(loc=0.0, scale=0.05)
            
            # Combined influence change
            total_change = diff_influence + same_influence + random_influence
            
            # Store for later application
            influence_changes[node] = total_change
        
        # Second pass: apply all changes at once
        for node, change in influence_changes.items():
            person = self.network.nodes[node].get('person')
            if person:
                person.social_influence += change
                # Ensure social influence stays within reasonable bounds
                person.social_influence = max(0, min(1.0, person.social_influence))
                # Make decision based on updated influence
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
        This method analyzes historical data to determine appropriate values for:
        - risk_perception_effect: How strongly risk perception influences mask wearing decision
        - information_spread_rate: How quickly information spreads through the network
        - network_influence_weight: How much social connections influence decisions
        """
        try:
            print("Calibrating model parameters using training data...")
            train_data = pd.read_csv(train_file)
            
            # Extract agent attributes for reference
            agent_attrs = pd.read_csv(agent_file)
            agent_attrs_dict = {row['agent_id']: {
                'risk_perception': row['risk_perception'],
                'total_connections': row['total_connections']
            } for _, row in agent_attrs.iterrows()}
            
            # 1. Calculate the correlation between risk_perception and mask wearing
            # Create a dataset of day 0 (initial state)
            initial_wearing = train_data[train_data['day'] == 0]
            risk_wear_data = []
            
            # Combine with risk perception data
            for _, row in initial_wearing.iterrows():
                agent_id = row['agent_id']
                if agent_id in agent_attrs_dict:
                    risk_wear_data.append({
                        'agent_id': agent_id,
                        'risk_perception': agent_attrs_dict[agent_id]['risk_perception'],
                        'wearing_mask': row['wearing_mask']
                    })
            
            # Convert to DataFrame for analysis
            risk_wear_df = pd.DataFrame(risk_wear_data)
            
            # Calculate correlation coefficient
            # Convert boolean to int for correlation calculation
            risk_wear_df['wearing_mask_int'] = risk_wear_df['wearing_mask'].astype(int)
            risk_perception_corr = risk_wear_df['risk_perception'].corr(risk_wear_df['wearing_mask_int'])
            
            # The stronger the correlation, the stronger the effect should be
            # Scale to a reasonable range (0.5-1.5)
            Person.risk_perception_effect = 0.5 + abs(risk_perception_corr)
            print(f"Calibrated risk_perception_effect: {Person.risk_perception_effect:.4f}")
            
            # 2. Calculate information spread rate based on how mask wearing changes day-to-day
            # Group by day and calculate the percentage of mask wearers each day
            daily_rates = train_data.groupby('day')['wearing_mask'].mean()
            
            # Calculate the average daily change
            daily_changes = daily_rates.diff().abs().mean()
            
            # Scale to a reasonable range (0.02-0.1)
            # If changes are larger, information spreads faster
            information_spread_rate = 0.02 + (daily_changes * 0.8)
            information_spread_rate = min(0.1, max(0.02, information_spread_rate))
            print(f"Calibrated information_spread_rate: {information_spread_rate:.4f}")
            
            # 3. Calibrate network influence weight based on how "received_info" correlates 
            # with changes in mask wearing behavior
            transition_data = []
            
            # For each agent, detect transitions in mask wearing status
            for agent_id in train_data['agent_id'].unique():
                agent_data = train_data[train_data['agent_id'] == agent_id].sort_values('day')
                if len(agent_data) < 2:
                    continue
                
                # Look for transitions
                for i in range(1, len(agent_data)):
                    prev_state = agent_data.iloc[i-1]['wearing_mask']
                    curr_state = agent_data.iloc[i]['wearing_mask']
                    received_info = agent_data.iloc[i]['received_info']
                    
                    if prev_state != curr_state:
                        # Record this transition with the information received flag
                        transition_data.append({
                            'agent_id': agent_id,
                            'prev_state': prev_state,
                            'curr_state': curr_state,
                            'received_info': received_info
                        })
            
            # Convert to DataFrame
            transitions_df = pd.DataFrame(transition_data)
            
            # Calculate proportion of transitions associated with received information
            if len(transitions_df) > 0:
                info_impact = transitions_df['received_info'].mean()
                # Scale to appropriate range (0.01-0.1)
                network_influence_weight = 0.01 + (info_impact * 0.09)
            else:
                # Default if no transitions found
                network_influence_weight = 0.05
                
            Person.network_influence_weight = network_influence_weight
            print(f"Calibrated network_influence_weight: {network_influence_weight:.4f}")
            
            # Store the information_spread_rate for use in propagate_behavior
            SocialNetwork.information_spread_rate = information_spread_rate
            
        except (FileNotFoundError, pd.errors.ParserError, PermissionError) as e:
            print(f"Error calibrating parameters from training data: {e}")
            print("Using default parameter values instead.")
            # Fallback to default values if calibration fails
            Person.risk_perception_effect = 0.8
            SocialNetwork.information_spread_rate = 0.06
            Person.network_influence_weight = 0.05

    def run(self, start_day: int = 30, end_day: int = 39) -> None:
        """
        Execute the simulation loop over the specified prediction period.

        Args:
            start_day (int): The starting day for the simulation.
            end_day (int): The ending day for the simulation.
        """
        print(f"Running simulation from day {start_day} to day {end_day}...")
        
        # Use the calibrated information spread rate from SocialNetwork class
        information_spread_rate = getattr(SocialNetwork, 'information_spread_rate', 0.05)
        
        for day in range(start_day, end_day + 1):
            self.social_network.propagate_behavior(information_spread_rate)
            # Print progress every 2 days
            if (day - start_day) % 2 == 0 or day == end_day:
                mask_wearers = sum(int(agent.mask_wearing_status) for agent in self.agents)
                total_agents = len(self.agents)
                print(f"  Day {day}: {mask_wearers}/{total_agents} agents wearing masks ({mask_wearers/total_agents*100:.2f}%)")

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
        output_dir = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        results = pd.DataFrame({
            'agent_id': [agent.agent_id for agent in self.agents],
            'mask_wearing_status': [agent.mask_wearing_status for agent in self.agents]
        })
        results.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

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
    Main function to initialize, run and analyze different simulation scenarios.
    """
    print("\n============ MASK ADOPTION BEHAVIOR SIMULATION ============\n")
    
    # Scenario 1: Baseline simulation with calibrated parameters
    print("SCENARIO 1: Baseline Simulation (Calibrated Parameters)")
    print("-------------------------------------------------------")
    baseline_sim = Simulation()
    
    # Output initial state
    initial_mask_wearers = sum(int(agent.mask_wearing_status) for agent in baseline_sim.agents)
    total_agents = len(baseline_sim.agents)
    print(f"Initial state: {initial_mask_wearers}/{total_agents} agents wearing masks ({initial_mask_wearers/total_agents*100:.2f}%)")
    
    # Store initial mask wearing state for reference
    initial_status = {agent.agent_id: agent.mask_wearing_status for agent in baseline_sim.agents}
    
    # Run baseline simulation
    print("Starting baseline simulation...")
    baseline_sim.run()
    
    # Output final state
    baseline_mask_wearers = sum(int(agent.mask_wearing_status) for agent in baseline_sim.agents)
    print(f"Final state: {baseline_mask_wearers}/{total_agents} agents wearing masks ({baseline_mask_wearers/total_agents*100:.2f}%)")
    print(f"Change: {baseline_mask_wearers - initial_mask_wearers} agents ({(baseline_mask_wearers - initial_mask_wearers)/total_agents*100:.2f}%)")
    
    # Visualize and save baseline results
    baseline_sim.visualize()
    baseline_sim.save_results("baseline_results.csv")
    
    # Save day 39 results for evaluation
    results_day39 = pd.DataFrame({
        'agent_id': [agent.agent_id for agent in baseline_sim.agents],
        'mask_wearing_status': [agent.mask_wearing_status for agent in baseline_sim.agents],
        'scenario': ['baseline'] * len(baseline_sim.agents),
        'day': [39] * len(baseline_sim.agents)
    })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(PROJECT_ROOT, "output/evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    results_day39.to_csv(os.path.join(output_dir, "baseline_day39.csv"), index=False)
    
    # Scenario 2: High Environmental Risk (e.g., pandemic peak)
    print("\nSCENARIO 2: High Environmental Risk (pandemic peak)")
    print("-------------------------------------------------------")
    # Create a new simulation but modify the environmental risk
    high_risk_sim = Simulation()
    
    # Reset to initial state for fair comparison
    for agent in high_risk_sim.agents:
        agent.mask_wearing_status = initial_status[agent.agent_id]
    
    # Increase environmental risk
    original_env_risk = Person.environmental_risk
    Person.environmental_risk = 0.35  # Significantly higher risk
    
    print(f"Environmental risk increased from {original_env_risk} to {Person.environmental_risk}")
    print("Starting high risk simulation...")
    high_risk_sim.run()
    
    # Output final state
    high_risk_mask_wearers = sum(int(agent.mask_wearing_status) for agent in high_risk_sim.agents)
    print(f"Final state: {high_risk_mask_wearers}/{total_agents} agents wearing masks ({high_risk_mask_wearers/total_agents*100:.2f}%)")
    print(f"Change from baseline: {high_risk_mask_wearers - baseline_mask_wearers} agents ({(high_risk_mask_wearers - baseline_mask_wearers)/total_agents*100:.2f}%)")
    
    # Save high risk results for evaluation
    high_risk_sim.save_results("high_risk_results.csv")
    
    # Save day 39 results for high risk scenario
    high_risk_day39 = pd.DataFrame({
        'agent_id': [agent.agent_id for agent in high_risk_sim.agents],
        'mask_wearing_status': [agent.mask_wearing_status for agent in high_risk_sim.agents],
        'scenario': ['high_risk'] * len(high_risk_sim.agents),
        'day': [39] * len(high_risk_sim.agents)
    })
    high_risk_day39.to_csv(os.path.join(output_dir, "high_risk_day39.csv"), index=False)
    
    # Scenario 3: Targeted Intervention (government policy)
    print("\nSCENARIO 3: Targeted Intervention (governmental policy)")
    print("-------------------------------------------------------")
    intervention_sim = Simulation()
    
    # Reset to initial state
    for agent in intervention_sim.agents:
        agent.mask_wearing_status = initial_status[agent.agent_id]
    
    # Reset environmental risk to original
    Person.environmental_risk = original_env_risk
    
    # Identify influential agents
    influential_agents = sorted(intervention_sim.agents, 
                               key=lambda a: len(a.network_connections) * a.risk_perception,
                               reverse=True)[:50]  # Target top 50 influential agents
    
    # Apply intervention to influential agents
    for agent in influential_agents:
        agent.social_influence += 0.5
    
    print(f"Applied targeted intervention to {len(influential_agents)} most influential agents")
    print("Starting intervention simulation...")
    intervention_sim.run()
    
    # Output final state
    intervention_mask_wearers = sum(int(agent.mask_wearing_status) for agent in intervention_sim.agents)
    print(f"Final state: {intervention_mask_wearers}/{total_agents} agents wearing masks ({intervention_mask_wearers/total_agents*100:.2f}%)")
    print(f"Change from baseline: {intervention_mask_wearers - baseline_mask_wearers} agents ({(intervention_mask_wearers - baseline_mask_wearers)/total_agents*100:.2f}%)")
    
    # Save intervention results for evaluation
    intervention_sim.save_results("intervention_results.csv")
    
    # Save day 39 results for intervention scenario
    intervention_day39 = pd.DataFrame({
        'agent_id': [agent.agent_id for agent in intervention_sim.agents],
        'mask_wearing_status': [agent.mask_wearing_status for agent in intervention_sim.agents],
        'scenario': ['intervention'] * len(intervention_sim.agents),
        'day': [39] * len(intervention_sim.agents)
    })
    intervention_day39.to_csv(os.path.join(output_dir, "intervention_day39.csv"), index=False)
    
    # Combined day 39 results for all scenarios
    combined_results = pd.concat([results_day39, high_risk_day39, intervention_day39])
    combined_results.to_csv(os.path.join(output_dir, "all_scenarios_day39.csv"), index=False)
    
    # Compare all scenarios
    print("\n============ SIMULATION COMPARISON ============\n")
    print(f"{'Scenario':<30} {'Mask Wearers':<15} {'Rate (%)':<10} {'Change from Initial'}")
    print(f"{'-'*30} {'-'*15} {'-'*10} {'-'*20}")
    print(f"{'Initial State':<30} {initial_mask_wearers:<15} {initial_mask_wearers/total_agents*100:<10.2f} {'N/A'}")
    print(f"{'1: Baseline':<30} {baseline_mask_wearers:<15} {baseline_mask_wearers/total_agents*100:<10.2f} {(baseline_mask_wearers-initial_mask_wearers)/total_agents*100:+.2f}%")
    print(f"{'2: High Environmental Risk':<30} {high_risk_mask_wearers:<15} {high_risk_mask_wearers/total_agents*100:<10.2f} {(high_risk_mask_wearers-initial_mask_wearers)/total_agents*100:+.2f}%")
    print(f"{'3: Targeted Intervention':<30} {intervention_mask_wearers:<15} {intervention_mask_wearers/total_agents*100:<10.2f} {(intervention_mask_wearers-initial_mask_wearers)/total_agents*100:+.2f}%")
    
    # Create and save comparative visualization
    plt.figure(figsize=(12, 8))
    
    # Plot mask wearing rates for different scenarios
    scenarios = ['Initial', 'Baseline', 'High Risk', 'Intervention']
    rates = [
        initial_mask_wearers/total_agents*100,
        baseline_mask_wearers/total_agents*100,
        high_risk_mask_wearers/total_agents*100,
        intervention_mask_wearers/total_agents*100
    ]
    
    # Bar chart for overall comparison
    plt.subplot(2, 2, 1)
    plt.bar(scenarios, rates, color=['gray', 'blue', 'red', 'green'])
    plt.title('Mask Wearing Rates by Scenario')
    plt.ylabel('Percentage of Population (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for i, rate in enumerate(rates):
        plt.text(i, rate + 2, f'{rate:.1f}%', ha='center')
    
    # Risk group comparison across scenarios
    plt.subplot(2, 2, 2)
    
    # Calculate rates for each risk group in each scenario
    def get_risk_group_rates(simulation):
        low_risk = [agent for agent in simulation.agents if agent.risk_perception < 0.3]
        med_risk = [agent for agent in simulation.agents if 0.3 <= agent.risk_perception < 0.7]
        high_risk = [agent for agent in simulation.agents if agent.risk_perception >= 0.7]
        
        low_rate = sum(int(a.mask_wearing_status) for a in low_risk) / len(low_risk) * 100 if low_risk else 0
        med_rate = sum(int(a.mask_wearing_status) for a in med_risk) / len(med_risk) * 100 if med_risk else 0
        high_rate = sum(int(a.mask_wearing_status) for a in high_risk) / len(high_risk) * 100 if high_risk else 0
        
        return [low_rate, med_rate, high_rate]
    
    # Initial rates (from baseline_sim since we only stored status)
    initial_rates = get_risk_group_rates(baseline_sim)
    baseline_rates = get_risk_group_rates(baseline_sim)
    high_risk_rates = get_risk_group_rates(high_risk_sim)
    intervention_rates = get_risk_group_rates(intervention_sim)
    
    risk_groups = ['Low Risk', 'Medium Risk', 'High Risk']
    x = np.arange(len(risk_groups))
    width = 0.2
    
    plt.bar(x - width*1.5, initial_rates, width, label='Initial', color='gray')
    plt.bar(x - width/2, baseline_rates, width, label='Baseline', color='blue')
    plt.bar(x + width/2, high_risk_rates, width, label='High Risk', color='red')
    plt.bar(x + width*1.5, intervention_rates, width, label='Intervention', color='green')
    
    plt.title('Mask Wearing by Risk Group')
    plt.xlabel('Risk Perception Group')
    plt.ylabel('Percentage (%)')
    plt.xticks(x, risk_groups)
    plt.legend()
    plt.ylim(0, 100)
    
    # Effectiveness comparison
    plt.subplot(2, 2, 3)
    changes = [
        0,  # Initial (reference)
        baseline_mask_wearers - initial_mask_wearers,
        high_risk_mask_wearers - initial_mask_wearers,
        intervention_mask_wearers - initial_mask_wearers
    ]
    
    bars = plt.bar(scenarios, changes, color=['gray', 'blue', 'red', 'green'])
    
    # Color negative and positive changes differently
    for i, bar in enumerate(bars):
        if changes[i] < 0:
            bar.set_color('firebrick')
        elif changes[i] > 0:
            bar.set_color('forestgreen')
    
    plt.title('Change in Mask Wearing from Initial State')
    plt.ylabel('Change in Number of Agents')
    
    # Add value labels
    for i, change in enumerate(changes):
        plt.text(i, change + (5 if change >= 0 else -5), f'{change:+d}', ha='center')
    
    # Add a simple network diagram indicating intervention structure
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, 'Model Parameters:\n' +
                      f'- Risk Effect: {Person.risk_perception_effect:.3f}\n' +
                      f'- Spread Rate: {SocialNetwork.information_spread_rate:.3f}\n' +
                      f'- Network Influence: {Person.network_influence_weight:.3f}\n' +
                      f'- Environmental Risk: {original_env_risk:.3f}\n',
            horizontalalignment='center',
            verticalalignment='center',
            transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('scenario_comparison.png')
    print(f"\nScenario comparison chart saved to: {os.path.abspath('scenario_comparison.png')}")
    
    return baseline_sim, high_risk_sim, intervention_sim

main()