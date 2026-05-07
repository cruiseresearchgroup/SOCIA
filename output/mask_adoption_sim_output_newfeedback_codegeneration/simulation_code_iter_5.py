import os
import numpy as np
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from collections import deque

# Constants for data paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Data file paths
agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
network_file = os.path.join(DATA_DIR, "social_network.json")
train_file = os.path.join(DATA_DIR, "train_data.csv")

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x * 3))  # Multiplier to steepen the curve

class Person:
    """Agent representing individual mask-wearing behaviour with enhanced decision model."""

    # Base parameters (will be calibrated per cluster)
    influence_probability = 0.05
    risk_perception_effect = 1.0
    social_influence_decay = 0.01
    decision_threshold = 0.5
    network_influence_weight = 0.05
    min_risk_threshold = 0.7
    environmental_risk = 0.15
    
    # New parameters for enhanced model
    family_weight = 2.0      # Family connections have strongest influence
    work_school_weight = 1.5 # Work/school connections have medium influence
    community_weight = 1.0   # Community connections have baseline influence
    memory_length = 5        # How many days of past decisions to remember
    memory_decay = 0.8       # How quickly memory influence decays
    intervention_fatigue = 0.005  # Reduction in intervention effect over time
    
    # Cluster-specific parameters (will be set during calibration)
    cluster_specific_params = {}

    def __init__(self, agent_id: int, mask_wearing_status: bool, risk_perception: float, 
                 age: int, occupation: str, network_connections: Dict[str, List[int]]):
        self.agent_id = agent_id
        self.mask_wearing_status = bool(mask_wearing_status)
        self.risk_perception = float(risk_perception)
        self.age = age
        self.occupation = occupation
        
        # Store connections by type
        self.family_connections = network_connections.get('family', [])
        self.work_school_connections = network_connections.get('work_school', [])
        self.community_connections = network_connections.get('community', [])
        self.all_connections = network_connections.get('all', [])
        
        # Initialize influence and behavior states
        self.social_influence = np.random.normal(loc=0.5, scale=0.1)
        self.consistent_behavior_days = 0
        self.personal_threshold = Person.decision_threshold * np.random.normal(loc=1.0, scale=0.1)
        
        # Memory of recent decisions (for memory effect)
        self.decision_history = deque(maxlen=Person.memory_length)
        for _ in range(Person.memory_length):
            self.decision_history.append(int(mask_wearing_status))
        
        # Store original risk perception for decay calculations
        self.original_risk_perception = risk_perception
        
        # Assign to cluster (will be populated during calibration)
        self.cluster_id = -1

    def _calculate_decision_value(self, day_since_intervention: int = 0):
        """Calculate decision value with all factors including the new enhancements."""
        # 1. Social influence (with diminishing returns)
        influence_factor = min(0.8, self.social_influence * Person.influence_probability)

        # 2. Risk perception (with floor for high-risk individuals)
        if self.risk_perception >= Person.min_risk_threshold:
            risk_factor = max(0.6, self.risk_perception * Person.risk_perception_effect)
        else:
            risk_factor = self.risk_perception * Person.risk_perception_effect
        
        # 3. Environmental risk (global & time-varying)
        environmental_factor = Person.environmental_risk
        
        # 4. Memory effect (weighted sum of recent decisions)
        memory_weight = 0
        for i, past_decision in enumerate(self.decision_history):
            # More recent decisions have higher weight
            memory_weight += past_decision * (Person.memory_decay ** i)
        # Normalize by max possible weight
        max_memory_weight = sum(Person.memory_decay ** i for i in range(len(self.decision_history)))
        memory_factor = 0.2 * (memory_weight / max_memory_weight if max_memory_weight > 0 else 0)
        
        # 5. Intervention fatigue (reduces effectiveness over time)
        fatigue_factor = min(0.4, day_since_intervention * Person.intervention_fatigue)
        
        # 6. Habit persistence
        habit_strength = min(0.25, self.consistent_behavior_days * 0.02)
        habit_adjustment = habit_strength if self.mask_wearing_status else -habit_strength

        # 7. Cluster-specific adjustments if available
        cluster_adjustment = 0
        if self.cluster_id in Person.cluster_specific_params:
            cluster_params = Person.cluster_specific_params[self.cluster_id]
            # Apply cluster-specific risk boost/reduction
            risk_multiplier = cluster_params.get('risk_multiplier', 1.0)
            risk_factor *= risk_multiplier
            # Apply cluster-specific environment sensitivity
            env_multiplier = cluster_params.get('env_multiplier', 1.0)
            environmental_factor *= env_multiplier
            # Apply cluster-specific influence sensitivity
            influence_multiplier = cluster_params.get('influence_multiplier', 1.0)
            influence_factor *= influence_multiplier
            # Apply general cluster adjustment
            cluster_adjustment = cluster_params.get('bias', 0.0)
        
        # Calculate final decision value
        decision_value = (
            influence_factor + 
            risk_factor + 
            environmental_factor + 
            memory_factor + 
            habit_adjustment -
            fatigue_factor +
            cluster_adjustment
        )
        
        return decision_value

    def decide_to_wear_mask(self, day_since_intervention: int = 0):
        """Decide mask-wearing using sigmoid-based probabilistic decision."""
        previous_status = self.mask_wearing_status
        
        # Calculate decision value
        decision_value = self._calculate_decision_value(day_since_intervention)
        
        # Use sigmoid function for probabilistic decision
        probability = sigmoid(decision_value - self.personal_threshold)
        
        # Make probabilistic decision
        self.mask_wearing_status = np.random.random() < probability
        
        # Update decision history
        self.decision_history.appendleft(int(self.mask_wearing_status))

        # Update habit counter
        if self.mask_wearing_status == previous_status:
            self.consistent_behavior_days += 1
        else:
            self.consistent_behavior_days = 0

        # Decay social influence based on risk perception
        # High-risk individuals have slower decay (more persistent)
        decay_modifier = 1.0 - min(0.5, self.risk_perception)  # Lower decay for higher risk
        dynamic_decay = (0.005 + (0.04 * len(self.all_connections))) * decay_modifier
        self.social_influence = max(0.0, self.social_influence - dynamic_decay)

    def influence_others(self, network: nx.Graph):
        """Influence others with differentiated weights by connection type."""
        influence_changes = {}
        
        # Family connections (strongest influence)
        for neighbour in self.family_connections:
            current = network.nodes[neighbour].get('social_influence', 0.0)
            influence_changes[neighbour] = current + (Person.network_influence_weight * Person.family_weight)
        
        # Work/school connections (medium influence)
        for neighbour in self.work_school_connections:
            if neighbour in influence_changes:
                # Already influenced through family, don't double count
                continue
            current = network.nodes[neighbour].get('social_influence', 0.0)
            influence_changes[neighbour] = current + (Person.network_influence_weight * Person.work_school_weight)
        
        # Community connections (baseline influence)
        for neighbour in self.community_connections:
            if neighbour in influence_changes:
                # Already influenced through other channels, don't double count
                continue
            current = network.nodes[neighbour].get('social_influence', 0.0)
            influence_changes[neighbour] = current + (Person.network_influence_weight * Person.community_weight)
        
        # Apply influence changes
        nx.set_node_attributes(network, influence_changes, 'social_influence')

class SocialNetwork:
    """Enhanced social network with differentiated link strengths and advanced propagation."""

    information_spread_rate = 0.06
    reinforcement_factor = 0.02
    
    # New parameters
    family_propagation_multiplier = 1.8    # Family connections spread information faster
    work_school_propagation_multiplier = 1.4 # Work/school connections have medium spread
    community_propagation_multiplier = 1.0  # Community connections have baseline spread

    def __init__(self, structure: Dict[int, Dict[str, List[int]]]):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(structure.keys())
        
        # Add edges with type attributes
        for node_id, conn in structure.items():
            # Add family edges
            for family_member in conn.get('family', []):
                self.graph.add_edge(node_id, family_member, 
                                    type='family', 
                                    weight=SocialNetwork.family_propagation_multiplier)
            
            # Add work/school edges
            for colleague in conn.get('work_school', []):
                if not self.graph.has_edge(node_id, colleague):
                    self.graph.add_edge(node_id, colleague, 
                                        type='work_school', 
                                        weight=SocialNetwork.work_school_propagation_multiplier)
            
            # Add community edges
            for neighbor in conn.get('community', []):
                if not self.graph.has_edge(node_id, neighbor):
                    self.graph.add_edge(node_id, neighbor, 
                                        type='community', 
                                        weight=SocialNetwork.community_propagation_multiplier)
            
            # Initialize social influence
            self.graph.nodes[node_id].setdefault('social_influence', 0.0)

    def propagate_behavior(self, day_since_intervention: int = 0):
        """Propagate behavior with enhanced link-specific weights."""
        influence_changes = {}
        
        for node in self.graph.nodes:
            person = self.graph.nodes[node].get('person')
            if person is None:
                continue
                
            person_status = int(person.mask_wearing_status)
            neighbors = list(self.graph.neighbors(node))
            
            if not neighbors:
                continue
                
            # Initialize influence types
            same_behavior_family = 0
            diff_behavior_family = 0
            same_behavior_work = 0
            diff_behavior_work = 0
            same_behavior_community = 0
            diff_behavior_community = 0
            
            for neighbor in neighbors:
                neighbor_person = self.graph.nodes[neighbor].get('person')
                if neighbor_person is None:
                    continue
                    
                neighbor_status = int(neighbor_person.mask_wearing_status)
                edge_type = self.graph[node][neighbor].get('type', 'community')  # Default to community
                
                # Count by type and behavior
                if edge_type == 'family':
                    if person_status == neighbor_status:
                        same_behavior_family += 1
                    else:
                        diff_behavior_family += 1
                elif edge_type == 'work_school':
                    if person_status == neighbor_status:
                        same_behavior_work += 1
                    else:
                        diff_behavior_work += 1
                else:  # community
                    if person_status == neighbor_status:
                        same_behavior_community += 1
                    else:
                        diff_behavior_community += 1
            
            # Calculate influence by type with differentiated weights
            family_influence = (SocialNetwork.information_spread_rate * diff_behavior_family * SocialNetwork.family_propagation_multiplier +
                               SocialNetwork.reinforcement_factor * same_behavior_family * SocialNetwork.family_propagation_multiplier)
            
            work_influence = (SocialNetwork.information_spread_rate * diff_behavior_work * SocialNetwork.work_school_propagation_multiplier +
                             SocialNetwork.reinforcement_factor * same_behavior_work * SocialNetwork.work_school_propagation_multiplier)
            
            community_influence = (SocialNetwork.information_spread_rate * diff_behavior_community * SocialNetwork.community_propagation_multiplier +
                                  SocialNetwork.reinforcement_factor * same_behavior_community * SocialNetwork.community_propagation_multiplier)
            
            # Random environmental factor
            random_influence = np.random.normal(loc=0.0, scale=0.04)
            
            # Total combined influence
            total_influence = family_influence + work_influence + community_influence + random_influence
            
            # Apply intervention fatigue if applicable
            if day_since_intervention > 0:
                fatigue_modifier = max(0.5, 1.0 - (day_since_intervention * 0.01))  # Gradual reduction in effect
                total_influence *= fatigue_modifier
            
            influence_changes[node] = total_influence

        # Apply influence and let agents decide again
        for node, change in influence_changes.items():
            person = self.graph.nodes[node].get('person')
            if person is None:
                continue
                
            person.social_influence = np.clip(person.social_influence + change, 0, 1.0)
            person.decide_to_wear_mask(day_since_intervention)

class Simulation:
    """Enhanced simulation with clustering, time-dependent effects, and adaptive interventions."""

    def __init__(self):
        self.agents = self._load_agents()
        self.social_network = self._load_network()
        self.cluster_agents()
        self.intervention_day = 10  # Day when intervention started
        self._calibrate_parameters()

    def _load_agents(self) -> List[Person]:
        """Load agents with additional attributes."""
        agents = []
        data = pd.read_csv(agent_file)
        
        for _, row in data.iterrows():
            agents.append(
                Person(
                    agent_id=int(row['agent_id']),
                    mask_wearing_status=bool(row['initial_mask_wearing']),
                    risk_perception=float(row['risk_perception']),
                    age=int(row['age']),
                    occupation=str(row['occupation']),
                    network_connections={}  # Will be populated in _load_network
                )
            )
        return agents
    
    def _load_network(self) -> SocialNetwork:
        """Load the social network with connection types."""
        with open(network_file, 'r') as f:
            structure = json.load(f)
            
        sn = SocialNetwork(structure)
        
        # Set up connections for each agent
        for agent in self.agents:
            agent_connections = structure[str(agent.agent_id)]
            agent.family_connections = agent_connections.get('family', [])
            agent.work_school_connections = agent_connections.get('work_school', [])
            agent.community_connections = agent_connections.get('community', [])
            agent.all_connections = agent_connections.get('all', [])
            
            # Add agent reference to network
            sn.graph.nodes[agent.agent_id]['person'] = agent
                
        return sn

    def cluster_agents(self, n_clusters=4):
        """Cluster agents based on attributes for personalized parameters."""
        print(f"Clustering agents into {n_clusters} groups...")
        
        # Extract features for clustering
        features = []
        for agent in self.agents:
            # Age normalized to 0-1
            age_norm = min(1.0, agent.age / 100)
            
            # Occupation as one-hot encoded
            occupation_map = {
                'Student': [1, 0, 0, 0],
                'White Collar': [0, 1, 0, 0],
                'Blue Collar': [0, 0, 1, 0],
                'Retired': [0, 0, 0, 1]
            }
            occ_vector = occupation_map.get(agent.occupation, [0, 0, 0, 0])
            
            # Network size features
            family_size_norm = len(agent.family_connections) / 10  # Normalize
            work_size_norm = len(agent.work_school_connections) / 10
            community_size_norm = len(agent.community_connections) / 30
            
            # Risk perception
            risk = agent.risk_perception
            
            # Combine features
            feature_vector = [age_norm, risk, family_size_norm, work_size_norm, community_size_norm] + occ_vector
            features.append(feature_vector)
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Assign cluster labels to agents
        for i, agent in enumerate(self.agents):
            agent.cluster_id = int(clusters[i])
        
        # Analyze clusters
        cluster_counts = {}
        for i in range(n_clusters):
            count = sum(1 for agent in self.agents if agent.cluster_id == i)
            cluster_counts[i] = count
            print(f"  Cluster {i}: {count} agents ({count/len(self.agents)*100:.1f}%)")
        
        return clusters

    def _calculate_cluster_parameters(self):
        """Calculate cluster-specific parameters based on agent attributes."""
        cluster_params = {}
        
        for cluster_id in range(max(agent.cluster_id for agent in self.agents) + 1):
            # Get agents in this cluster
            cluster_agents = [a for a in self.agents if a.cluster_id == cluster_id]
            
            if not cluster_agents:
                continue
                
            # Calculate average risk perception
            avg_risk = sum(a.risk_perception for a in cluster_agents) / len(cluster_agents)
            
            # Calculate average age
            avg_age = sum(a.age for a in cluster_agents) / len(cluster_agents)
            
            # Calculate average connections
            avg_connections = sum(len(a.all_connections) for a in cluster_agents) / len(cluster_agents)
            
            # Calculate cluster-specific parameters
            # Risk multiplier: higher for elderly or high-risk clusters
            risk_multiplier = 1.0
            if avg_age > 60:  # Elderly cluster
                risk_multiplier = 1.3
            elif avg_risk > 0.5:  # High-risk perception cluster
                risk_multiplier = 1.2
            elif avg_age < 20:  # Youth cluster
                risk_multiplier = 0.8
                
            # Environment multiplier: based on age and connections
            env_multiplier = 1.0
            if avg_age > 50:  # Older people more concerned about environment
                env_multiplier = 1.2
            elif avg_connections > 25:  # Highly connected more affected by environment
                env_multiplier = 1.1
            
            # Influence multiplier: based on connections and age
            influence_multiplier = 1.0
            if avg_connections > 25:  # Highly connected more susceptible to influence
                influence_multiplier = 1.15
            elif avg_age < 25:  # Young people more susceptible to peer influence
                influence_multiplier = 1.2
                
            # Bias term: slight adjustment based on cluster characteristics
            bias = 0.0
            if "Student" in [a.occupation for a in cluster_agents]:
                if avg_age < 18:  # School students
                    bias = -0.05  # Less likely to wear masks
            elif "Retired" in [a.occupation for a in cluster_agents]:
                bias = 0.05  # More likely to wear masks
            
            # Store parameters
            cluster_params[cluster_id] = {
                'risk_multiplier': risk_multiplier,
                'env_multiplier': env_multiplier,
                'influence_multiplier': influence_multiplier,
                'bias': bias,
                'avg_risk': avg_risk,
                'avg_age': avg_age,
                'avg_connections': avg_connections
            }
            
        return cluster_params

    def _calibrate_parameters(self):
        """Calibrate parameters with cluster-specific adjustments."""
        print("Calibrating model parameters using training data...")
        train_df = pd.read_csv(train_file)

        # -------------------- Risk perception effect --------------------
        init_df = train_df[train_df['day'] == 0]
        agent_attrs = pd.read_csv(agent_file)[['agent_id', 'risk_perception']]
            merged = init_df.merge(agent_attrs, on='agent_id', how='left')
            corr = merged['risk_perception'].corr(merged['wearing_mask'].astype(int))
        Person.risk_perception_effect = 0.6 + abs(corr)  # 0.6-1.6
        print(f"  risk_perception_effect -> {Person.risk_perception_effect:.3f}")

        # -------------------- Information spread rate -------------------
        daily_rates = train_df.groupby('day')['wearing_mask'].mean()
        
        # Look at rates before and after intervention day
        pre_intervention = daily_rates[:self.intervention_day].values
        post_intervention = daily_rates[self.intervention_day:].values
        
        # Calculate rates of change
        pre_change = np.abs(np.diff(pre_intervention)).mean() if len(pre_intervention) > 1 else 0
        post_change = np.abs(np.diff(post_intervention)).mean() if len(post_intervention) > 1 else 0
        
        # Set parameters based on pre/post intervention dynamics
        SocialNetwork.information_spread_rate = np.clip(0.03 + post_change * 1.5, 0.03, 0.12)
        print(f"  information_spread_rate -> {SocialNetwork.information_spread_rate:.3f}")

        # -------------------- Network influence weights ------------------
        transitions = train_df.groupby(['agent_id']).apply(lambda df: df['wearing_mask'].diff().abs().sum())
        transition_ratio = transitions.mean() / 10  # rough scaling
        base_influence = np.clip(0.02 + transition_ratio, 0.02, 0.12)
        Person.network_influence_weight = base_influence
        print(f"  network_influence_weight -> {Person.network_influence_weight:.3f}")
        
        # Set differentiated connection weights
        family_multiplier = 1.8
        work_school_multiplier = 1.4
        Person.family_weight = family_multiplier
        Person.work_school_weight = work_school_multiplier
        Person.community_weight = 1.0
        print(f"  family_weight -> {Person.family_weight:.3f}")
        print(f"  work_school_weight -> {Person.work_school_weight:.3f}")

        # -------------------- Influence probability & decision threshold --------
        daily_change = daily_rates.diff().abs().mean()
        Person.influence_probability = np.clip(0.05 + daily_change * 3, 0.05, 0.15)
        Person.decision_threshold = max(0.3, 0.5 - daily_change * 1.5)
        print(f"  influence_probability    -> {Person.influence_probability:.3f}")
        print(f"  decision_threshold       -> {Person.decision_threshold:.3f}")
        
        # -------------------- Memory effect parameters --------
        # Analyze how consistent agent behavior is to set memory parameters
        consistency = train_df.groupby('agent_id')['wearing_mask'].apply(
            lambda x: (x == x.shift()).mean()
        ).mean()
        
        Person.memory_decay = np.clip(0.6 + consistency * 0.3, 0.6, 0.9)
        Person.memory_length = int(5 + consistency * 5)
        print(f"  memory_decay -> {Person.memory_decay:.3f}")
        print(f"  memory_length -> {Person.memory_length}")
        
        # -------------------- Intervention fatigue --------
        # Measure if there's fatigue effect in the training data
        if len(post_intervention) > 15:  # Need enough days to detect fatigue
            early_post = post_intervention[:7].mean()
            late_post = post_intervention[-7:].mean()
            fatigue_effect = max(0, early_post - late_post)
            Person.intervention_fatigue = np.clip(fatigue_effect / 100, 0.001, 0.01)
            print(f"  intervention_fatigue -> {Person.intervention_fatigue:.5f}")
        
        # -------------------- Calculate cluster parameters --------------------
        Person.cluster_specific_params = self._calculate_cluster_parameters()
        for cluster_id, params in Person.cluster_specific_params.items():
            print(f"  Cluster {cluster_id} parameters: risk_mult={params['risk_multiplier']:.2f}, "
                  f"env_mult={params['env_multiplier']:.2f}, infl_mult={params['influence_multiplier']:.2f}, "
                  f"bias={params['bias']:.2f}")

        # -------------------- Reset personal thresholds with new global value ----
        for agent in self.agents:
            agent.personal_threshold = Person.decision_threshold * np.random.normal(loc=1.0, scale=0.1)
            
            # Initialize memory with appropriate length
            agent.decision_history = deque(maxlen=Person.memory_length)
            for _ in range(Person.memory_length):
                agent.decision_history.append(int(agent.mask_wearing_status))

    def _calculate_environmental_risk(self, day: int, base_risk: float, growth_rate: float) -> float:
        """
        Calculate time-varying environmental risk with wave pattern.
        Combines linear growth with periodic oscillations to model natural epidemic waves.
        """
        # Base growth component
        linear_component = base_risk + day * growth_rate
        
        # Add wave pattern (subtle oscillation with 10-day period)
        wave_component = 0.02 * np.sin(2 * np.pi * day / 10)
        
        # Combine and ensure within bounds
        risk = np.clip(linear_component + wave_component, 0.05, 0.45)
        return risk

    def run(self, start_day: int = 30, end_day: int = 39, env_risk_growth_rate: float = 0.0):
        """Run simulation with enhanced time-dependent effects and adaptive intervention."""
        print(f"Running enhanced simulation from day {start_day} to {end_day} (env growth {env_risk_growth_rate:.3f})…")
        
        # Store results for each day
        results = {
            'day': [],
            'wearing_mask_count': [],
            'wearing_mask_rate': [],
            'env_risk': []
        }
        
        # Create a list to store daily states
        self.daily_states = []
        
        # Set initial environmental risk
        base_env_risk = Person.environmental_risk
        
        # Calculate days since intervention started
        days_since_start = start_day - self.intervention_day
        
        # Run simulation for each day
        for day_offset in range(end_day - start_day + 1):
            current_day = start_day + day_offset
            days_since_intervention = max(0, current_day - self.intervention_day)
            
            # Update environmental risk with wave pattern
            Person.environmental_risk = self._calculate_environmental_risk(
                day_offset, base_env_risk, env_risk_growth_rate
            )
            
            # Propagate behavior with time-dependent effects
            self.social_network.propagate_behavior(days_since_intervention)
            
            # Record daily state
            daily_state = {
                'day': current_day,
                'agents': [
                    {
                        'agent_id': agent.agent_id,
                        'mask_wearing_status': agent.mask_wearing_status,
                        'risk_perception': agent.risk_perception,
                        'social_influence': agent.social_influence,
                        'cluster_id': agent.cluster_id,
                        'age': agent.age,
                        'occupation': agent.occupation
                    }
                    for agent in self.agents
                ]
            }
            self.daily_states.append(daily_state)
            
            # Collect results for this day
            wearing_count = sum(int(a.mask_wearing_status) for a in self.agents)
            wearing_rate = wearing_count / len(self.agents)
            
            results['day'].append(current_day)
            results['wearing_mask_count'].append(wearing_count)
            results['wearing_mask_rate'].append(wearing_rate)
            results['env_risk'].append(Person.environmental_risk)
            
            # Adaptive intervention (adjust parameters based on current rate)
            if wearing_rate < 0.4 and env_risk_growth_rate > 0:
                # If wearing rate is too low, boost influence probability temporarily
                Person.influence_probability *= 1.05
                
            elif wearing_rate > 0.7:
                # If wearing rate is very high, slightly increase threshold (harder to convince)
                Person.decision_threshold *= 1.01
            
            # Log progress
            if day_offset % 2 == 0 or current_day == end_day:
                print(f"  Day {current_day}: {wearing_count}/{len(self.agents)} wearing "
                      f"({wearing_rate*100:.2f}%)  envRisk={Person.environmental_risk:.3f}")
        
        # Return results for analysis
        return pd.DataFrame(results)

    def save_results(self, path: str):
        """Save final agent states and daily states to CSV."""
        # Create output directory if it doesn't exist
        output_dir = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to the specified directory
        output_path = os.path.join(output_dir, path)
        
        # Save final state for backward compatibility
        df = pd.DataFrame({
            'agent_id': [ag.agent_id for ag in self.agents],
            'mask_wearing_status': [ag.mask_wearing_status for ag in self.agents],
            'risk_perception': [ag.risk_perception for ag in self.agents],
            'cluster_id': [ag.cluster_id for ag in self.agents],
            'age': [ag.age for ag in self.agents],
            'occupation': [ag.occupation for ag in self.agents],
            'social_influence': [ag.social_influence for ag in self.agents],
        })
        df.to_csv(output_path, index=False)
        print(f"Final state results saved to: {output_path}")
        
        # Save daily states if available
        if hasattr(self, 'daily_states'):
            # Create a new dataframe for all daily data
            daily_data = []
            
            for state in self.daily_states:
                day = state['day']
                
                for agent_data in state['agents']:
                    daily_data.append({
                        'day': day,
                        'agent_id': agent_data['agent_id'],
                        'mask_wearing_status': agent_data['mask_wearing_status'],
                        'risk_perception': agent_data['risk_perception'],
                        'social_influence': agent_data['social_influence'],
                        'cluster_id': agent_data['cluster_id'],
                        'age': agent_data['age'],
                        'occupation': agent_data['occupation']
                    })
            
            # Save to CSV
            daily_filename = os.path.splitext(path)[0] + "_daily.csv"
            daily_output_path = os.path.join(output_dir, daily_filename)
            pd.DataFrame(daily_data).to_csv(daily_output_path, index=False)
            print(f"Daily states saved to: {daily_output_path}")

    def visualize_clusters(self, output_filename='cluster_analysis.png'):
        """Visualize mask wearing rates by cluster."""
        # Create output directory if it doesn't exist
        output_dir = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze mask wearing by cluster
        clusters = sorted(set(agent.cluster_id for agent in self.agents))
        cluster_names = [f"Cluster {i}" for i in clusters]
        cluster_rates = []
        
        for cluster_id in clusters:
            cluster_agents = [agent for agent in self.agents if agent.cluster_id == cluster_id]
            if cluster_agents:
                rate = sum(int(agent.mask_wearing_status) for agent in cluster_agents) / len(cluster_agents) * 100
                cluster_rates.append(rate)
            else:
                cluster_rates.append(0)
                
        # Create visualization
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cluster_names, cluster_rates, color=plt.cm.viridis(np.linspace(0, 1, len(clusters))))
        plt.title('Mask Wearing Rates by Cluster')
        plt.xlabel('Agent Cluster')
        plt.ylabel('Mask Wearing Rate (%)')
        plt.ylim(0, 100)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, cluster_rates)):
            plt.text(bar.get_x() + bar.get_width()/2, rate + 2, f'{rate:.1f}%', 
                     ha='center', va='bottom')
            
        # Add annotations for cluster characteristics
        for i, cluster_id in enumerate(clusters):
            if cluster_id in Person.cluster_specific_params:
                params = Person.cluster_specific_params[cluster_id]
                avg_age = params.get('avg_age', 0)
                avg_risk = params.get('avg_risk', 0)
                plt.annotate(f"Age: {avg_age:.1f}, Risk: {avg_risk:.2f}", 
                            xy=(i, 10), ha='center', va='bottom', rotation=0,
                            color='darkblue', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, output_filename))
        plt.close()
        print(f"Cluster analysis saved to: {os.path.join(output_dir, output_filename)}")

# ----------------------------------------------------------------------
# Main entry: run baseline, high-risk and intervention scenarios
# ----------------------------------------------------------------------

def main():
    print("\n============= ENHANCED MASK-WEARING BEHAVIOUR SIMULATION =============\n")
    
    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")
    os.makedirs(output_dir, exist_ok=True)

    # Create evaluation directory
    eval_dir = os.path.join(PROJECT_ROOT, "output/evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    # -------------------- Scenario 1: Enhanced Baseline with wave pattern --------------------
    print("SCENARIO 1 – Enhanced Baseline (wave environmental risk)")
    baseline_sim = Simulation()
    initial_env_risk = Person.environmental_risk  # usually 0.15
    
    # Use wave pattern with growth
    growth_rate = 0.01  # Slower growth with wave pattern
    baseline_results = baseline_sim.run(env_risk_growth_rate=growth_rate)
    baseline_sim.save_results("baseline_results_enhanced.csv")
    
    # Save daily data for baseline scenario
    if hasattr(baseline_sim, 'daily_states'):
        daily_data_baseline = []
        for state in baseline_sim.daily_states:
            day = state['day']
            for agent_data in state['agents']:
                daily_data_baseline.append({
                    'day': day,
                    'agent_id': agent_data['agent_id'],
                    'mask_wearing_status': agent_data['mask_wearing_status'],
                    'scenario': 'baseline',
                    'cluster_id': agent_data['cluster_id']
                })
        pd.DataFrame(daily_data_baseline).to_csv(os.path.join(eval_dir, "baseline_all_days_iter5.csv"), index=False)
        print(f"Baseline daily data saved to evaluation directory")
    
    # Visualize clusters
    baseline_sim.visualize_clusters('baseline_clusters.png')

    # -------------------- Scenario 2: High Risk Environment with Clusters --------------------
    print("\nSCENARIO 2 – High Risk Environment with Cluster-specific Responses")
    Person.environmental_risk = 0.30  # Start higher
    # Boost high-risk clusters' sensitivity further
    for cluster_id, params in Person.cluster_specific_params.items():
        if params['avg_age'] > 60 or params['avg_risk'] > 0.5:
            params['env_multiplier'] *= 1.2
    
    high_risk_sim = Simulation()  # re-instantiate to reset agents with new class params
    high_risk_results = high_risk_sim.run(env_risk_growth_rate=0.015)  # Faster growth
    high_risk_sim.save_results("high_risk_results_enhanced.csv")
    
    # Save daily data for high risk scenario
    if hasattr(high_risk_sim, 'daily_states'):
        daily_data_high_risk = []
        for state in high_risk_sim.daily_states:
            day = state['day']
            for agent_data in state['agents']:
                daily_data_high_risk.append({
                    'day': day,
                    'agent_id': agent_data['agent_id'],
                    'mask_wearing_status': agent_data['mask_wearing_status'],
                    'scenario': 'high_risk',
                    'cluster_id': agent_data['cluster_id']
                })
        pd.DataFrame(daily_data_high_risk).to_csv(os.path.join(eval_dir, "high_risk_all_days_iter5.csv"), index=False)
        print(f"High risk daily data saved to evaluation directory")
    
    high_risk_sim.visualize_clusters('high_risk_clusters.png')
    
    # Reset cluster parameters
    Person.environmental_risk = initial_env_risk

    # -------------------- Scenario 3: Targeted Adaptive Intervention -------------------------
    print("\nSCENARIO 3 – Targeted Adaptive Intervention")
    intervention_sim = Simulation()
    
    # Identify influential agents in each cluster
    cluster_influencers = {}
    for cluster_id in range(max(agent.cluster_id for agent in intervention_sim.agents) + 1):
        cluster_agents = [a for a in intervention_sim.agents if a.cluster_id == cluster_id]
        # Take top 10% most connected in each cluster
        if cluster_agents:
            topN = max(1, int(0.1 * len(cluster_agents)))
            influential = sorted(cluster_agents, key=lambda a: len(a.all_connections), reverse=True)[:topN]
            cluster_influencers[cluster_id] = influential
    
    # Apply targeted intervention to influencers
    targeted_count = 0
    for cluster_id, influencers in cluster_influencers.items():
        # Adjust boost based on cluster properties
        boost = 0.6  # Default boost
        if cluster_id in Person.cluster_specific_params:
            # Higher boost for clusters less likely to wear masks
            if Person.cluster_specific_params[cluster_id].get('avg_risk', 0) < 0.3:
                boost = 0.8
        
        for agent in influencers:
            agent.social_influence += boost
            agent.risk_perception = min(1.0, agent.risk_perception * 1.5)  # Boost risk perception
            targeted_count += 1
    
    print(f"  Applied targeted intervention to {targeted_count} influential agents across clusters")
    
    # Run with adaptive intervention
    intervention_results = intervention_sim.run(env_risk_growth_rate=growth_rate)
    intervention_sim.save_results("intervention_results_enhanced.csv")
    
    # Save daily data for intervention scenario
    if hasattr(intervention_sim, 'daily_states'):
        daily_data_intervention = []
        for state in intervention_sim.daily_states:
            day = state['day']
            for agent_data in state['agents']:
                daily_data_intervention.append({
                    'day': day,
                    'agent_id': agent_data['agent_id'],
                    'mask_wearing_status': agent_data['mask_wearing_status'],
                    'scenario': 'intervention',
                    'cluster_id': agent_data['cluster_id']
                })
        pd.DataFrame(daily_data_intervention).to_csv(os.path.join(eval_dir, "intervention_all_days_iter5.csv"), index=False)
        print(f"Intervention daily data saved to evaluation directory")
    
    intervention_sim.visualize_clusters('intervention_clusters.png')
    
    # Combine all daily results
    if all(hasattr(sim, 'daily_states') for sim in [baseline_sim, high_risk_sim, intervention_sim]):
        all_daily_data = pd.concat([
            pd.DataFrame(daily_data_baseline),
            pd.DataFrame(daily_data_high_risk),
            pd.DataFrame(daily_data_intervention)
        ])
        all_daily_data.to_csv(os.path.join(eval_dir, "all_scenarios_all_days_iter5.csv"), index=False)
        print(f"Combined daily data for all scenarios saved to evaluation directory")
    
    # -------------------- Visualize Results Comparison --------------------
    # Create comparison plot of all scenarios
    plt.figure(figsize=(12, 8))
    
    # Compare mask wearing rates by day
    plt.subplot(2, 1, 1)
    plt.plot(baseline_results['day'], baseline_results['wearing_mask_rate'], 'b-', label='Baseline')
    plt.plot(high_risk_results['day'], high_risk_results['wearing_mask_rate'], 'r-', label='High Risk')
    plt.plot(intervention_results['day'], intervention_results['wearing_mask_rate'], 'g-', label='Intervention')
    plt.title('Mask Wearing Rates by Scenario Over Time')
    plt.xlabel('Day')
    plt.ylabel('Percentage Wearing Masks')
    plt.ylim(0, 1)
    plt.legend()
    
    # Compare environmental risk
    plt.subplot(2, 1, 2)
    plt.plot(baseline_results['day'], baseline_results['env_risk'], 'b--', label='Baseline')
    plt.plot(high_risk_results['day'], high_risk_results['env_risk'], 'r--', label='High Risk')
    plt.plot(intervention_results['day'], intervention_results['env_risk'], 'g--', label='Intervention')
    plt.title('Environmental Risk by Scenario Over Time')
    plt.xlabel('Day')
    plt.ylabel('Environmental Risk')
    plt.ylim(0, 0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_scenario_comparison.png'))
    plt.close()
    
    # Final bar chart comparison of final values
    plt.figure(figsize=(12, 8))
    
    # Plot mask wearing rates comparison
    scenarios = ['Baseline', 'High Risk', 'Intervention']
    final_rates = [
        baseline_results['wearing_mask_rate'].iloc[-1] * 100,
        high_risk_results['wearing_mask_rate'].iloc[-1] * 100,
        intervention_results['wearing_mask_rate'].iloc[-1] * 100
    ]
    
    colors = ['blue', 'red', 'green']
    bars = plt.bar(scenarios, final_rates, color=colors)
    plt.title('Final Mask Wearing Rates by Scenario (Enhanced Model)')
    plt.ylabel('Percentage of Population (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, final_rates)):
        plt.text(bar.get_x() + bar.get_width()/2, rate + 2, f'{rate:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_final_comparison.png'))
    plt.close()
    
    # Create visualization of daily rates 
    if all(hasattr(sim, 'daily_states') for sim in [baseline_sim, high_risk_sim, intervention_sim]):
        plt.figure(figsize=(15, 10))
        
        # Plot daily mask wearing rates over time
        days = range(30, 40)
        
        # Calculate daily mask-wearing rates for each simulation
        baseline_daily_rates = []
        high_risk_daily_rates = []
        intervention_daily_rates = []
        
        for i, day in enumerate(days):
            # Baseline
            baseline_state = baseline_sim.daily_states[i]
            baseline_rate = sum(agent['mask_wearing_status'] for agent in baseline_state['agents']) / len(baseline_state['agents']) * 100
            baseline_daily_rates.append(baseline_rate)
            
            # High risk
            high_risk_state = high_risk_sim.daily_states[i]
            high_risk_rate = sum(agent['mask_wearing_status'] for agent in high_risk_state['agents']) / len(high_risk_state['agents']) * 100
            high_risk_daily_rates.append(high_risk_rate)
            
            # Intervention
            intervention_state = intervention_sim.daily_states[i]
            intervention_rate = sum(agent['mask_wearing_status'] for agent in intervention_state['agents']) / len(intervention_state['agents']) * 100
            intervention_daily_rates.append(intervention_rate)
        
        # Plot the data
        plt.plot(days, baseline_daily_rates, 'o-', color='blue', label='Baseline')
        plt.plot(days, high_risk_daily_rates, 's-', color='red', label='High Risk')
        plt.plot(days, intervention_daily_rates, '^-', color='green', label='Intervention')
        
        # Add target rate line (using same target as in evaluate_daily_data.py)
        plt.axhline(y=57.51, color='black', linestyle='--', label='Target Rate (57.51%)')
        
        plt.title('Daily Mask Wearing Rates Over Time (Iter5)')
        plt.xlabel('Day')
        plt.ylabel('Mask Wearing Rate (%)')
    plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(days)
        
        # Save the time series visualization
    plt.tight_layout()
        time_series_path = os.path.join(output_dir, 'daily_rates_comparison_iter5.png')
        plt.savefig(time_series_path)
        plt.close()
        print(f"Daily rates comparison chart saved to: {time_series_path}")
    
    print("\nEnhanced simulation finished – results saved for all scenarios.")
    print(f"Final wearing rates: Baseline={final_rates[0]:.1f}%, High Risk={final_rates[1]:.1f}%, Intervention={final_rates[2]:.1f}%")

if __name__ == "__main__":
    main() 