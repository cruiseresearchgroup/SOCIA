#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
import os
from tqdm import tqdm
from collections import defaultdict

# Set random seed to ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class Agent:
    """Represents an individual agent in the population"""
    
    def __init__(self, agent_id, age, occupation, initial_mask_wearing=False):
        self.id = agent_id
        self.age = age
        self.occupation = occupation
        
        # Demographic attributes
        self.age_group = self._determine_age_group()
        
        # Social network connections
        self.family_connections = []      # Strong connections
        self.work_school_connections = [] # Medium connections
        self.community_connections = []   # Weak connections
        self.all_connections = []         # Combined list of all connections
        
        # Behavior and risk perception
        self.risk_perception = np.random.beta(2, 5)  # Conservative risk perception distribution
        self.wearing_mask = initial_mask_wearing  # Current mask-wearing status
        self.wearing_mask_history = [initial_mask_wearing]  # Historical mask-wearing status
        
        # Information reception status
        self.received_info = False  # Whether received government promotion or intervention information
        self.info_history = [False]  # Historical information reception status
        
    def _determine_age_group(self):
        """Determine age group based on age"""
        if self.age <= 18:
            return "Youth"
        elif self.age <= 40:
            return "Young Adult"
        elif self.age <= 65:
            return "Middle Age"
        else:
            return "Elderly"
    
    def update_connections(self, family=None, work_school=None, community=None):
        """Update the agent's social connections"""
        if family:
            self.family_connections = family
        if work_school:
            self.work_school_connections = work_school
        if community:
            self.community_connections = community
            
        # Update all connections list
        self.all_connections = (self.family_connections + 
                               self.work_school_connections + 
                               self.community_connections)
    
    def update_mask_wearing(self, population, intervention_active=False, government_info=0.0):
        """
        Update the agent's mask-wearing status
        
        Args:
            population: List of all agents
            intervention_active: Whether intervention measures are active
            government_info: Government information influence strength (0-1)
        """
        # Get connected neighbors
        family_neighbors = [population[i] for i in self.family_connections]
        work_neighbors = [population[i] for i in self.work_school_connections]
        community_neighbors = [population[i] for i in self.community_connections]
        
        # Calculate proportion of mask-wearing among different types of neighbors
        family_mask_ratio = sum(n.wearing_mask for n in family_neighbors) / max(1, len(family_neighbors))
        work_mask_ratio = sum(n.wearing_mask for n in work_neighbors) / max(1, len(work_neighbors))
        community_mask_ratio = sum(n.wearing_mask for n in community_neighbors) / max(1, len(community_neighbors))
        
        # Weighted average, with family having the most influence, community the least
        social_influence = (0.5 * family_mask_ratio + 
                           0.3 * work_mask_ratio + 
                           0.2 * community_mask_ratio)
        
        # Base probability + risk perception influence + social influence + government information influence
        base_probability = 0.05  # 5% base probability
        risk_influence = self.risk_perception * 0.4  # Risk perception impact
        
        # Government intervention influence
        gov_influence = 0.0
        if intervention_active and self.received_info:
            gov_influence = government_info * 0.3  # Government information impact
        
        # Social influence (proportion of neighbors wearing masks)
        social_factor = social_influence * 0.3  # Social influence weight
        
        # Total probability
        total_probability = min(0.95, base_probability + risk_influence + social_factor + gov_influence)
        
        # Decide whether to wear a mask
        self.wearing_mask = random.random() < total_probability
        self.wearing_mask_history.append(self.wearing_mask)
        
    def receive_information(self, information_sources):
        """
        Receive and transmit information
        
        Args:
            information_sources: List of agent IDs who have already received information
        
        Returns:
            bool: Whether new information was received
        """
        # Check if connected neighbors are information sources
        connections_with_info = [c for c in self.all_connections if c in information_sources]
        
        # Information transmission probabilities for different connection types
        family_info = [c for c in self.family_connections if c in information_sources]
        work_info = [c for c in self.work_school_connections if c in information_sources]
        community_info = [c for c in self.community_connections if c in information_sources]
        
        # Probabilities of receiving information from each type of connection
        p_family = 0.8 if family_info else 0.0
        p_work = 0.5 if work_info else 0.0
        p_community = 0.3 if community_info else 0.0
        
        # Take the maximum probability as the chance to receive information
        p_receive = max(p_family, p_work, p_community)
        
        # Those who have already received information will not receive again
        if not self.received_info and random.random() < p_receive:
            self.received_info = True
        
        self.info_history.append(self.received_info)
        return self.received_info

def generate_population(size=1000, initial_mask_rate=0.1):
    """
    Generate a population with diverse attributes
    
    Args:
        size: Population size
        initial_mask_rate: Initial mask-wearing rate
    
    Returns:
        list: List of agent objects
    """
    population = []
    
    # Age distribution
    age_distribution = {
        "Youth": (0, 18, 0.2),   # (min, max, proportion)
        "Young Adult": (19, 40, 0.4),
        "Middle Age": (41, 65, 0.3),
        "Elderly": (66, 100, 0.1)
    }
    
    # Occupation distribution
    occupation_distribution = {
        "Student": 0.25,
        "White Collar": 0.35,
        "Blue Collar": 0.3,
        "Retired": 0.1
    }
    
    # Generate agents
    for i in range(size):
        # Assign age
        r = random.random()
        cumulative = 0
        for age_group, (min_age, max_age, prop) in age_distribution.items():
            cumulative += prop
            if r <= cumulative:
                age = random.randint(min_age, max_age)
                break
        
        # Assign occupation
        r = random.random()
        cumulative = 0
        for occupation, prop in occupation_distribution.items():
            cumulative += prop
            if r <= cumulative:
                break
        
        # Assign initial mask status
        is_wearing_mask = random.random() < initial_mask_rate
        
        # Create agent
        agent = Agent(i, age, occupation, is_wearing_mask)
        population.append(agent)
    
    return population

def create_social_network(population):
    """
    Create a multi-layered social network for the population
    
    Args:
        population: List of agent objects
    
    Returns:
        networkx.Graph: Social network graph
    """
    size = len(population)
    G = nx.Graph()
    
    # Add nodes
    for agent in population:
        G.add_node(agent.id, 
                  age=agent.age, 
                  occupation=agent.occupation,
                  wearing_mask=agent.wearing_mask,
                  risk_perception=agent.risk_perception)
    
    # Create family network (strong connections)
    # Average family size is 4 people
    families = []
    remaining_agents = list(range(size))
    random.shuffle(remaining_agents)
    
    while remaining_agents:
        # Random family size (1-6 people)
        family_size = min(random.choices([1, 2, 3, 4, 5, 6], 
                                         weights=[0.1, 0.15, 0.2, 0.3, 0.15, 0.1])[0], 
                          len(remaining_agents))
        family = remaining_agents[:family_size]
        families.append(family)
        remaining_agents = remaining_agents[family_size:]
    
    # Create family connections
    for family in families:
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                G.add_edge(family[i], family[j], type='family', weight=0.8)
                
                # Update agent's family connections
                population[family[i]].family_connections.append(family[j])
                population[family[j]].family_connections.append(family[i])
    
    # Create work/school networks (medium-strength connections)
    # Group by occupation and age
    occupation_groups = defaultdict(list)
    for agent in population:
        if agent.occupation == "Student":
            if agent.age <= 18:
                group = "Student_Youth"
            else:
                group = "Student_Adult"
        else:
            group = agent.occupation
        
        occupation_groups[group].append(agent.id)
    
    # Create small-world networks for each group
    for group, members in occupation_groups.items():
        if len(members) <= 1:
            continue
            
        # Create small-world network, with each person connected to 5-10 colleagues/classmates on average
        k = min(5, len(members) - 1)  # Ensure k doesn't exceed number of nodes-1
        if len(members) > 10:
            group_graph = nx.watts_strogatz_graph(len(members), k, 0.1)
            
            # Map small-world network connections to agents
            for i, j in group_graph.edges():
                agent_i = members[i]
                agent_j = members[j]
                G.add_edge(agent_i, agent_j, type='work_school', weight=0.5)
                
                # Update agent's work/school connections
                population[agent_i].work_school_connections.append(agent_j)
                population[agent_j].work_school_connections.append(agent_i)
        else:
            # For small groups, create a complete graph
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    G.add_edge(members[i], members[j], type='work_school', weight=0.5)
                    
                    # Update agent's work/school connections
                    population[members[i]].work_school_connections.append(members[j])
                    population[members[j]].work_school_connections.append(members[i])
    
    # Create community network (weak connections)
    # Each agent randomly connects to 5-10 community members
    for agent in population:
        # Possible number of connections
        num_connections = random.randint(5, 10)
        
        # Potential connections
        potential_connections = [i for i in range(size) 
                              if i != agent.id and not G.has_edge(agent.id, i)]
        
        # Randomly select connections
        if potential_connections and num_connections > 0:
            actual_connections = random.sample(
                potential_connections, 
                min(num_connections, len(potential_connections))
            )
            
            for connection in actual_connections:
                G.add_edge(agent.id, connection, type='community', weight=0.3)
                
                # Update agent's community connections
                agent.community_connections.append(connection)
                population[connection].community_connections.append(agent.id)
    
    # Update all agents' connection lists
    for agent in population:
        agent.update_connections()
    
    return G, families

def run_simulation(population, social_network, families, days=40, intervention_day=10):
    """
    Run mask adoption rate simulation
    
    Args:
        population: List of agent objects
        social_network: Social network graph
        families: List of family groupings
        days: Number of simulation days
        intervention_day: Day when intervention is implemented
    
    Returns:
        tuple: (daily mask wearing rates, information reception rates)
    """
    size = len(population)
    
    # Track daily mask-wearing rate and information spread rate
    daily_mask_rates = [sum(agent.wearing_mask for agent in population) / size]
    daily_info_rates = [0.0]  # No one has received intervention information on initial day
    
    # Information sources (list of agent IDs who have received information)
    info_sources = []
    
    # Community leaders (nodes with highest centrality)
    community_leaders = []
    if intervention_day > 0:
        # Calculate node degree centrality
        centrality = nx.degree_centrality(social_network)
        # Choose top 5% of nodes as community leaders
        num_leaders = max(1, int(size * 0.05))
        community_leaders = [n for n, _ in sorted(centrality.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)[:num_leaders]]
    
    # Run simulation
    print(f"Running {days} days of mask adoption simulation...")
    for day in tqdm(range(1, days)):
        # Government intervention (starting from a specific day)
        intervention_active = day >= intervention_day
        intervention_strength = 0.0
        
        if intervention_active:
            # Intensify intervention strength over time, gradually reaching maximum value
            intervention_strength = min(0.8, (day - intervention_day + 1) * 0.1)
            
            # After intervention begins, community leaders serve as information sources
            if day == intervention_day:
                info_sources = community_leaders.copy()
                
                # Mark community leaders as having received information
                for leader_id in community_leaders:
                    population[leader_id].received_info = True
                
                print(f"Intervention started! Initial sources: {len(info_sources)} community leaders")
        
        # Spread information through social network
        new_info_sources = []
        
        # Update agents in random order to avoid update order bias
        update_order = list(range(size))
        random.shuffle(update_order)
        
        for agent_id in update_order:
            # Record information reception status for each agent (appends to info_history)
            received = population[agent_id].receive_information(info_sources)
            # If intervention is active and agent newly received information, add to sources
            if intervention_active and received:
                new_info_sources.append(agent_id)
        
        # Update information sources
        info_sources.extend(new_info_sources)
        
        # Update mask-wearing behavior
        for agent_id in update_order:
            population[agent_id].update_mask_wearing(
                population, 
                intervention_active, 
                intervention_strength
            )
        
        # Record data for the day
        mask_rate = sum(agent.wearing_mask for agent in population) / size
        info_rate = sum(agent.received_info for agent in population) / size
        
        daily_mask_rates.append(mask_rate)
        daily_info_rates.append(info_rate)
        
        # Output current status
        if day % 5 == 0 or day == days - 1:
            print(f"Day {day}: Mask wearing rate = {mask_rate:.2%}, Information reception rate = {info_rate:.2%}")
    
    return daily_mask_rates, daily_info_rates

def export_data(population, families, social_network, daily_mask_rates, daily_info_rates, days=40, train_days=30):
    """
    Export simulation data to multiple files
    
    Args:
        population: List of agent objects
        families: List of family groupings
        social_network: Social network graph
        daily_mask_rates: Daily mask-wearing rates
        daily_info_rates: Daily information reception rates
        days: Total simulation days
        train_days: Number of days for training data
    """
    # Create output directory
    output_dir = "mask_adoption_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Export agent attributes table
    agent_data = []
    for agent in population:
        agent_data.append({
            "agent_id": agent.id,
            "age": agent.age,
            "age_group": agent.age_group,
            "occupation": agent.occupation,
            "risk_perception": agent.risk_perception,
            "initial_mask_wearing": agent.wearing_mask_history[0],
            "family_connections": len(agent.family_connections),
            "work_school_connections": len(agent.work_school_connections),
            "community_connections": len(agent.community_connections),
            "total_connections": len(agent.all_connections)
        })
    
    agent_df = pd.DataFrame(agent_data)
    agent_df.to_csv(f"{output_dir}/agent_attributes.csv", index=False)
    print(f"Exported agent attributes table to {output_dir}/agent_attributes.csv")
    
    # 2. Export social network data
    network_data = {}
    for agent in population:
        network_data[agent.id] = {
            "family": agent.family_connections,
            "work_school": agent.work_school_connections,
            "community": agent.community_connections,
            "all": agent.all_connections
        }
    
    with open(f"{output_dir}/social_network.pkl", "wb") as f:
        pickle.dump(network_data, f)
    
    # Export network graph
    nx.write_gexf(social_network, f"{output_dir}/social_network.gexf")
    print(f"Exported social network data to {output_dir}/social_network.pkl and {output_dir}/social_network.gexf")
    
    # 3. Export time series data
    time_series_data = []
    for agent in population:
        # Ensure history record lists are consistent in length
        mask_history_len = len(agent.wearing_mask_history)
        info_history_len = len(agent.info_history)
        
        for day in range(days):
            # Safely access historical data; use the last day's status if out of range
            if day < mask_history_len:
                wearing_mask = agent.wearing_mask_history[day]
            else:
                wearing_mask = agent.wearing_mask_history[-1] if mask_history_len > 0 else False
                
            if day < info_history_len:
                received_info = agent.info_history[day]
            else:
                received_info = agent.info_history[-1] if info_history_len > 0 else False
            
            time_series_data.append({
                "day": day,
                "agent_id": agent.id,
                "wearing_mask": wearing_mask,
                "received_info": received_info
            })
    
    time_df = pd.DataFrame(time_series_data)
    time_df.to_csv(f"{output_dir}/time_series_data.csv", index=False)
    print(f"Exported time series data to {output_dir}/time_series_data.csv")
    
    # 4. Export aggregate data (daily summary)
    aggregate_data = []
    for day in range(days):
        aggregate_data.append({
            "day": day,
            "mask_rate": daily_mask_rates[day],
            "info_rate": daily_info_rates[day],
            "dataset": "train" if day < train_days else "test"
        })
    
    agg_df = pd.DataFrame(aggregate_data)
    agg_df.to_csv(f"{output_dir}/daily_aggregate_data.csv", index=False)
    print(f"Exported aggregate data to {output_dir}/daily_aggregate_data.csv")
    
    # 5. Split into training and testing sets
    train_df = time_df[time_df["day"] < train_days]
    test_df = time_df[time_df["day"] >= train_days]
    
    train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data.csv", index=False)
    print(f"Split data into training set ({train_days} days) and testing set ({days - train_days} days)")

def visualize_results(daily_mask_rates, daily_info_rates, train_days=30, intervention_day=10):
    """
    Visualize simulation results
    
    Args:
        daily_mask_rates: Daily mask-wearing rates
        daily_info_rates: Daily information reception rates
        train_days: Number of days for training data
        intervention_day: Day when intervention is implemented
    """
    days = len(daily_mask_rates)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(days), daily_mask_rates, 'b-', linewidth=2, label='Mask Wearing Rate')
    plt.plot(range(days), daily_info_rates, 'r-', linewidth=2, label='Information Reception Rate')
    
    # Draw intervention line
    plt.axvline(x=intervention_day, color='green', linestyle='--', label='Intervention Start')
    
    # Draw training/testing split line
    plt.axvline(x=train_days, color='purple', linestyle='--', label='Train/Test Split')
    
    # Fill training and testing regions
    plt.fill_between(range(train_days), 0, 1, alpha=0.1, color='blue', label='Training Set')
    plt.fill_between(range(train_days, days), 0, 1, alpha=0.1, color='red', label='Testing Set')
    
    plt.xlabel('Days')
    plt.ylabel('Rate')
    plt.title('Mask Wearing and Information Spread Simulation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Save chart
    output_dir = "mask_adoption_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f"{output_dir}/simulation_results.png", dpi=300, bbox_inches='tight')
    print(f"Visualization results saved to {output_dir}/simulation_results.png")
    
    plt.show()

def visualize_network(social_network, population, day=0):
    """
    Visualize social network and mask-wearing status
    
    Args:
        social_network: Social network graph
        population: List of agent objects
        day: Day to visualize
    """
    G = social_network.copy()
    
    # Update node attributes to reflect current mask-wearing status
    for agent in population:
        # Check both lengths to avoid index errors
        if day < len(agent.wearing_mask_history):
            G.nodes[agent.id]['wearing_mask'] = agent.wearing_mask_history[day]
        else:
            G.nodes[agent.id]['wearing_mask'] = agent.wearing_mask_history[-1] if agent.wearing_mask_history else False
            
        if day < len(agent.info_history):
            G.nodes[agent.id]['received_info'] = agent.info_history[day]
        else:
            G.nodes[agent.id]['received_info'] = agent.info_history[-1] if agent.info_history else False
    
    # Set node colors
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['wearing_mask']:
            if G.nodes[node]['received_info']:
                color = 'green'  # Wearing mask and received information
            else:
                color = 'blue'   # Wearing mask but not received information
        else:
            if G.nodes[node]['received_info']:
                color = 'orange' # Not wearing mask but received information
            else:
                color = 'red'    # Not wearing mask and not received information
        node_colors.append(color)
    
    # Set edge colors
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data['type'] == 'family':
            edge_colors.append('darkgreen')
        elif data['type'] == 'work_school':
            edge_colors.append('navy')
        else:  # community
            edge_colors.append('gray')
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Use spring_layout to show network structure
    pos = nx.spring_layout(G, seed=RANDOM_SEED)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color=edge_colors)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Mask+Info'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Mask Only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Info Only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='No Mask No Info'),
        Line2D([0], [0], color='darkgreen', lw=2, label='Family Relation'),
        Line2D([0], [0], color='navy', lw=2, label='Work/School Relation'),
        Line2D([0], [0], color='gray', lw=2, label='Community Relation')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f'Social Network and Mask Wearing Status on Day {day}')
    plt.axis('off')
    
    # Save chart
    output_dir = "mask_adoption_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f"{output_dir}/network_day_{day}.png", dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {output_dir}/network_day_{day}.png")
    
    plt.show()

def main():
    # Simulation parameters
    population_size = 1000
    initial_mask_rate = 0.1  # Initial 10% of people wear masks
    simulation_days = 40
    intervention_day = 10
    train_days = 30  # First 30 days as training set
    
    print(f"Generating population of {population_size} people...")
    # Generate population
    population = generate_population(size=population_size, initial_mask_rate=initial_mask_rate)
    
    print("Creating social network...")
    # Create social network
    social_network, families = create_social_network(population)
    
    # Output basic network information
    print(f"Network info: Nodes={len(social_network.nodes())}, Edges={len(social_network.edges())}")
    print(f"Average connections: {sum(len(agent.all_connections) for agent in population) / population_size:.2f}")
    
    # Run simulation
    daily_mask_rates, daily_info_rates = run_simulation(
        population, 
        social_network, 
        families, 
        days=simulation_days, 
        intervention_day=intervention_day
    )
    
    # Export data
    export_data(
        population, 
        families, 
        social_network, 
        daily_mask_rates, 
        daily_info_rates, 
        days=simulation_days, 
        train_days=train_days
    )
    
    # Visualize results
    visualize_results(
        daily_mask_rates, 
        daily_info_rates, 
        train_days=train_days, 
        intervention_day=intervention_day
    )
    
    # Visualize social network at different time points
    visualize_network(social_network, population, day=0)  # Initial state
    visualize_network(social_network, population, day=intervention_day)  # When intervention begins
    visualize_network(social_network, population, day=simulation_days-1)  # Final state

if __name__ == "__main__":
    main() 