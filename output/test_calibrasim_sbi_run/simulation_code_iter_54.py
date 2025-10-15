def main():
    pass

import sys
import os
import json
import argparse
import logging
import shutil
import random
import math
from collections import defaultdict, deque

# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
# NOTE: DATA_DIR is defined per path-handling requirements but not used in this minimal simulation.


def clamp(x, lo, hi):
    """
    Clamp a value x to the interval [lo, hi].

    Args:
        x (float): Input value.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: Value clamped to [lo, hi].
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def sigmoid(x):
    """
    Compute the logistic sigmoid function.

    Args:
        x (float): Input value.

    Returns:
        float: Sigmoid of x in [0, 1].
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)
    except OverflowError:
        # For very large |x|, use limits
        return 0.0 if x < 0 else 1.0


def moving_average(values, window):
    """
    Compute the moving average over a specified window size.

    Args:
        values (list[float]): Sequence of values.
        window (int): Window size (>=1).

    Returns:
        list[float]: Moving average series of the same length as values.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    if window <= 1:
        return list(values)
    out = []
    running = 0.0
    q = deque()
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.popleft()
        out.append(running / len(q))
    return out


def poisson_sample(lmbda, rng):
    """
    Sample from a Poisson distribution using Knuth's algorithm.

    Args:
        lmbda (float): Expected value (lambda), >= 0.
        rng (random.Random): RNG instance.

    Returns:
        int: A non-negative integer sampled from Poisson(lmbda).
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    if lmbda <= 0:
        return 0
    # Knuth's algorithm
    L = math.exp(-lmbda)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def weighted_sample_without_replacement(items, weights, k, rng):
    """
    Sample k distinct items without replacement with probability proportional to given weights.

    Args:
        items (list): Items to sample from.
        weights (list[float]): Non-negative weights corresponding to items.
        k (int): Number of items to sample.
        rng (random.Random): RNG instance.

    Returns:
        list: List of sampled items (length <= k if not enough items).
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    # FIXED: Optimized weighted sampling by maintaining a running total to avoid repeated full sums.
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(items)
    items_copy = list(items)
    weights_copy = [max(0.0, float(w)) for w in weights]
    chosen = []
    total = sum(weights_copy)
    for _ in range(k):
        if not items_copy:
            break
        if total <= 0:
            idx = rng.randrange(len(items_copy))
        else:
            threshold = rng.random() * total
            csum = 0.0
            idx = 0
            for i, w in enumerate(weights_copy):
                csum += w
                if csum >= threshold:
                    idx = i
                    break
        total -= weights_copy[idx]
        weights_copy.pop(idx)
        chosen.append(items_copy.pop(idx))
    return chosen


class Person:
    """
    Represents an individual agent with behavioral attributes and state.

    Attributes:
        idx (int): Unique identifier.
        age (float): Age in years.
        community_id (int): Community identifier.
        income_level (float): Income proxy (annual).
        compliance_propensity (float): Propensity to comply with mandates [0,1].
        attitude_toward_masks (float): Attitude scale [-1,1].
        mask_inventory (int): Count of masks owned.
        perceived_cost (float): Perceived disutility or cost of wearing/purchasing.
        household_id (int): Household id.
        workplace_id (int or None): Workplace location id.
        daily_contacts (int): Average daily contact target.
        media_consumption_level (float): Propensity to receive media [0,1].
        risk_perception (float): Risk perception in [0, 1].
        susceptibility_to_influence (float): Susceptibility to peer influence in [0, 1].
        trust_in_authorities (float): Trust in authorities in [0, 1].
        baseline_adoption_propensity (float): Baseline propensity in [0, 1].
        adoption_threshold (float): Personal threshold in [0, 1].
        habit_strength (float): Habit strength in [0, 1].
        fatigue (float): Fatigue level in [0, 1].
        is_adopting (bool): Current mask-wearing state.
        trust_modifier (float): Multiplicative modifier for trust effect on campaign.

    Methods:
        to_dict(): Convert state to a dictionary.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(
        self,
        idx,
        age,
        community_id,
        risk_perception,
        susceptibility_to_influence,
        trust_in_authorities,
        baseline_adoption_propensity,
        adoption_threshold,
        habit_strength,
        fatigue,
        is_adopting,
        trust_modifier,
        income_level=0.0,
        compliance_propensity=0.5,
        attitude_toward_masks=0.0,
        mask_inventory=0,
        perceived_cost=1.0,
        household_id=None,
        workplace_id=None,
        daily_contacts=10,
        media_consumption_level=0.5,
    ):
        """
        Initialize a Person instance.

        Args:
            idx (int): Agent id.
            age (float): Age in years.
            community_id (int): Community id.
            risk_perception (float): Risk perception.
            susceptibility_to_influence (float): Susceptibility to influence.
            trust_in_authorities (float): Trust in authorities.
            baseline_adoption_propensity (float): Baseline adoption propensity.
            adoption_threshold (float): Adoption threshold.
            habit_strength (float): Habit strength.
            fatigue (float): Fatigue.
            is_adopting (bool): Initial mask-wearing state.
            trust_modifier (float): Trust multiplicative modifier for policy effect.
            income_level (float): Income proxy.
            compliance_propensity (float): Compliance propensity in [0,1].
            attitude_toward_masks (float): Attitude in [-1,1].
            mask_inventory (int): Mask stock.
            perceived_cost (float): Perceived disutility/cost.
            household_id (int or None): Household id.
            workplace_id (int or None): Work location id.
            daily_contacts (int): Expected contact count per day.
            media_consumption_level (float): Media consumption propensity [0,1].
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.idx = idx
        self.age = age
        self.community_id = community_id
        self.income_level = max(0.0, float(income_level))
        self.compliance_propensity = clamp(float(compliance_propensity), 0.0, 1.0)
        self.attitude_toward_masks = clamp(float(attitude_toward_masks), -1.0, 1.0)
        self.mask_inventory = int(max(0, mask_inventory))
        self.perceived_cost = max(0.0, float(perceived_cost))
        self.household_id = household_id
        self.workplace_id = workplace_id
        self.daily_contacts = int(max(0, daily_contacts))
        self.media_consumption_level = clamp(float(media_consumption_level), 0.0, 1.0)
        self.risk_perception = clamp(risk_perception, 0.0, 1.0)
        self.susceptibility_to_influence = clamp(susceptibility_to_influence, 0.0, 1.0)
        self.trust_in_authorities = clamp(trust_in_authorities, 0.0, 1.0)
        self.baseline_adoption_propensity = clamp(baseline_adoption_propensity, 0.0, 1.0)
        self.adoption_threshold = clamp(adoption_threshold, 0.0, 1.0)
        self.habit_strength = clamp(habit_strength, 0.0, 1.0)
        self.fatigue = clamp(fatigue, 0.0, 1.0)
        self.is_adopting = bool(is_adopting)
        self.trust_modifier = float(trust_modifier)

    def to_dict(self):
        """
        Convert the person's state to a dictionary for serialization.

        Returns:
            dict: Dictionary representation of the person's state.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
        return {
            "idx": self.idx,
            "age": self.age,
            "community_id": self.community_id,
            "income_level": self.income_level,
            "compliance_propensity": self.compliance_propensity,
            "attitude_toward_masks": self.attitude_toward_masks,
            "mask_inventory": self.mask_inventory,
            "perceived_cost": self.perceived_cost,
            "household_id": self.household_id,
            "workplace_id": self.workplace_id,
            "daily_contacts": self.daily_contacts,
            "media_consumption_level": self.media_consumption_level,
            "risk_perception": self.risk_perception,
            "susceptibility_to_influence": self.susceptibility_to_influence,
            "trust_in_authorities": self.trust_in_authorities,
            "baseline_adoption_propensity": self.baseline_adoption_propensity,
            "adoption_threshold": self.adoption_threshold,
            "habit_strength": self.habit_strength,
            "fatigue": self.fatigue,
            "is_adopting": self.is_adopting,
            "trust_modifier": self.trust_modifier,
        }


class Network:
    """
    Lightweight undirected weighted graph for social network representation.

    Attributes:
        n (int): Number of nodes.
        adj (dict[int, list[tuple[int, float, bool]]]): Adjacency list mapping node -> list of (neighbor, weight, is_strong).
        rng (random.Random): RNG instance.

    Methods:
        add_edge(i, j, weight, is_strong)
        neighbors(i)
        degree(i)
        to_edge_list()
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, n, rng):
        """
        Initialize a Network.

        Args:
            n (int): Number of nodes.
            rng (random.Random): RNG instance.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.n = n
        self.adj = {i: [] for i in range(n)}
        self.rng = rng

    def add_edge(self, i, j, weight=1.0, is_strong=False):
        """
        Add an undirected edge between i and j.

        Args:
            i (int): Node index.
            j (int): Node index.
            weight (float): Edge weight (>0).
            is_strong (bool): Whether the tie is designated strong.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        if i == j:
            return
        # Avoid duplicate edges
        if any(nb == j for nb, _, _ in self.adj[i]):
            return
        if any(nb == i for nb, _, _ in self.adj[j]):
            return
        w = float(max(1e-6, weight))
        self.adj[i].append((j, w, bool(is_strong)))
        self.adj[j].append((i, w, bool(is_strong)))

    def neighbors(self, i):
        """
        Get neighbors of node i.

        Args:
            i (int): Node index.

        Returns:
            list[tuple[int, float, bool]]: List of (neighbor, weight, is_strong) tuples.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        return list(self.adj.get(i, []))

    def degree(self, i):
        """
        Get degree of node i.

        Args:
            i (int): Node index.

        Returns:
            int: Degree of node i.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        return len(self.adj.get(i, []))

    def to_edge_list(self):
        """
        Return a list of unique undirected edges.

        Returns:
            list[tuple[int, int, float, bool]]: Unique edges with (i, j, weight, is_strong) and i < j.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        edges = []
        seen = set()
        for i in range(self.n):
            for j, w, s in self.adj[i]:
                if i < j and (i, j) not in seen:
                    edges.append((i, j, w, s))
                    seen.add((i, j))
        return edges


class NetworkBuilder:
    """
    Builds a weighted, community-structured small-world network with homophily and strong ties.

    Methods:
        build_network(population, communities, params, rng) -> Network
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    @staticmethod
    def build_network(population, communities, params, rng):
        """
        Construct the network according to the model plan.

        Args:
            population (int): Number of agents.
            communities (list[int]): Community id for each agent.
            params (dict): Parameters including:
                - net_avg_degree
                - net_rewiring_prob
                - net_homophily_strength
                - net_weight_mean
                - net_weight_std
                - net_fraction_strong_ties
            rng (random.Random): RNG instance.

        Returns:
            Network: Constructed network.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        n = int(population)
        G = Network(n, rng)
        k = max(2, int(round(float(params.get("net_avg_degree", 10.0)))))
        if k % 2 == 1:
            k += 1
        p_rewire = float(params.get("net_rewiring_prob", 0.05))
        homophily = clamp(float(params.get("net_homophily_strength", 0.5)), 0.0, 1.0)
        w_mu = float(params.get("net_weight_mean", 1.0))
        w_sigma = float(params.get("net_weight_std", 0.3))
        frac_strong = clamp(float(params.get("net_fraction_strong_ties", 0.2)), 0.0, 0.8)

        # Base ring lattice edges (static list for degree-preserving rewiring)
        base_edges = [(i, (i + offset) % n) for i in range(n) for offset in range(1, k // 2 + 1) if i < (i + offset) % n]
        # Apply degree-preserving rewiring using the static base edge list
        # FIXED: Degree-preserving rewiring that rewires only edges (i, j) as directed from i, preserving degree of i and avoiding reciprocal rewiring.
        for (i, j) in base_edges:
            if rng.random() < p_rewire:
                # Remove edge if present (safe to call remove semantics by filtering)
                G.adj[i] = [(nb, w2, s2) for (nb, w2, s2) in G.adj[i] if nb != j]
                G.adj[j] = [(nb, w2, s2) for (nb, w2, s2) in G.adj[j] if nb != i]
                excluded = {i} | {nb for nb, _, _ in G.neighbors(i)}
                candidates = [u for u in range(n) if u not in excluded]
                u = rng.choice(candidates) if candidates else j
                G.add_edge(i, u, weight=1.0, is_strong=False)
            else:
                if not any(nb == j for nb, _, _ in G.neighbors(i)):
                    G.add_edge(i, j, weight=1.0, is_strong=False)

        # Assign weights log-normal and apply homophily bias
        edges = G.to_edge_list()
        # Select strong ties
        num_strong = int(round(frac_strong * len(edges)))
        strong_indices = set(rng.sample(range(len(edges)), num_strong)) if len(edges) > 0 and num_strong > 0 else set()
        for idx, (i, j, _, _) in enumerate(edges):
            same_comm = communities[i] == communities[j]
            base_weight = math.exp(rng.gauss(w_mu, w_sigma))
            # Homophily: increase within-community weight, slightly reduce cross-community
            if same_comm:
                base_weight *= (1.0 + 0.5 * homophily)
            else:
                base_weight *= (1.0 - 0.25 * homophily)
            is_strong = idx in strong_indices
            if is_strong:
                base_weight *= 2.0
            # Update in adjacency
            G.adj[i] = [(nb, w, s) for (nb, w, s) in G.adj[i] if nb != j]
            G.adj[j] = [(nb, w, s) for (nb, w, s) in G.adj[j] if nb != i]
            G.add_edge(i, j, weight=base_weight, is_strong=is_strong)

        return G


class PolicyCampaign:
    """
    Produces a time-varying campaign intensity signal with ramp, pulses, and decay.

    Methods:
        intensity(day) -> float
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, params):
        """
        Initialize the policy campaign schedule from parameters.

        Args:
            params (dict): Parameters:
                - campaign_base_intensity
                - campaign_start_day
                - campaign_ramp_days
                - campaign_max_intensity
                - campaign_decay_days
                - campaign_pulse_interval
                - campaign_pulse_magnitude
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.base = float(params.get("campaign_base_intensity", 0.2))
        self.start = int(params.get("campaign_start_day", 5))
        self.ramp = int(params.get("campaign_ramp_days", 10))
        self.max_int = float(params.get("campaign_max_intensity", 0.8))
        self.decay_days = max(1, int(params.get("campaign_decay_days", 30)))
        self.pulse_interval = int(params.get("campaign_pulse_interval", 14))
        self.pulse_magnitude = float(params.get("campaign_pulse_magnitude", 0.1))

    def intensity(self, day):
        """
        Compute the campaign intensity for a given day.

        Args:
            day (int): Day index starting at 0.

        Returns:
            float: Intensity in [0, 1].
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        if day < self.start:
            I = self.base
        elif self.start <= day < self.start + self.ramp:
            r = (day - self.start) / max(1, float(self.ramp))
            I = self.base + r * (self.max_int - self.base)
        else:
            t_since_peak = day - (self.start + self.ramp)
            I = self.max_int * math.exp(-t_since_peak / float(self.decay_days))

        if self.pulse_interval > 0 and day > 0 and (day % self.pulse_interval == 0):
            I = min(1.0, I + self.pulse_magnitude)
        return clamp(I, 0.0, 1.0)


class MobilityContact:
    """
    Generates daily peer exposure per person via stochastic contact sampling biased by tie strength and day-type.

    Methods:
        compute_exposure(day, network, persons, params, rng, history) -> list[float]
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, params):
        """
        Initialize mobility/contact module with parameters.

        Args:
            params (dict): Parameters:
                - contact_rate_per_day
                - contact_bias_toward_strong_ties
                - mobility_variance
                - weekend_multiplier
                - shock_probability
                - shock_magnitude
                - peer_window_days (for smoothing exposures)
                - contact_sample_cap
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.rate = float(params.get("contact_rate_per_day", 12.0))
        self.bias = float(params.get("contact_bias_toward_strong_ties", 0.6))
        self.mobility_var = float(params.get("mobility_variance", 0.2))
        self.weekend_mult = float(params.get("weekend_multiplier", 0.8))
        self.shock_prob = float(params.get("shock_probability", 0.01))
        self.shock_mag = float(params.get("shock_magnitude", 0.5))
        self.peer_window_days = max(1, int(params.get("peer_window_days", 3)))
        # FIXED: Added contact_sample_cap to limit per-agent sampling for performance.
        self.contact_sample_cap = max(1, int(params.get("contact_sample_cap", 12)))

    def compute_exposure(self, day, network, persons, rng, history):
        """
        Compute smoothed peer exposure for each person.

        Args:
            day (int): Day index.
            network (Network): Social network.
            persons (list[Person]): Agents list.
            rng (random.Random): RNG.
            history (list[deque]): Exposure history per person.

        Returns:
            list[float]: Smoothed exposure per person.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        n = len(persons)
        is_weekend = (day % 7 in (5, 6))
        mult = self.weekend_mult if is_weekend else 1.0
        if rng.random() < self.shock_prob:
            mult *= self.shock_mag
        # Day-level mobility noise multiplier using log-normal approx via exp(N(0, var))
        day_mult = math.exp(rng.gauss(0.0, self.mobility_var))
        base_rate = max(0.1, self.rate) * mult * day_mult

        exposures = [0.0] * n
        for i, person in enumerate(persons):
            neighbors = network.neighbors(i)
            deg = len(neighbors)
            if deg == 0:
                exposures[i] = 0.0
                history[i].append(0.0)
                while len(history[i]) > self.peer_window_days:
                    history[i].popleft()
                continue
            lam = base_rate
            k = poisson_sample(lam, rng)
            # Weighted sampling: weight^(1 + bias) and add strong tie effect
            items = []
            weights = []
            for nb, w, is_strong in neighbors:
                w_eff = w ** (1.0 + self.bias)
                if is_strong:
                    w_eff *= (1.0 + 0.5 * self.bias)
                items.append(nb)
                weights.append(w_eff)
            if k > 0:
                sampled = weighted_sample_without_replacement(items, weights, min(min(k, deg), self.contact_sample_cap), rng)
                if sampled:
                    # FIXED: Weighted exposure using tie weights (weight-normalized average).
                    w_map = {nb: (w ** (1.0 + self.bias)) * (1.0 + 0.5 * self.bias if s else 1.0) for nb, w, s in neighbors}
                    num = sum((1.0 if persons[j].is_adopting else 0.0) * w_map.get(j, 1.0) for j in sampled)
                    den = sum(w_map.get(j, 1.0) for j in sampled)
                    exposures[i] = num / den if den > 0 else 0.0
                else:
                    exposures[i] = sum(1.0 if persons[j].is_adopting else 0.0 for j, _, _ in neighbors) / float(deg)
            else:
                exposures[i] = sum(1.0 if persons[j].is_adopting else 0.0 for j, _, _ in neighbors) / float(deg)
            # Update history and smooth
            history[i].append(exposures[i])
            while len(history[i]) > self.peer_window_days:
                history[i].popleft()

        smoothed = [sum(h) / float(len(h)) if len(h) > 0 else 0.0 for h in history]
        return smoothed


class BehaviorAdoption:
    """
    Updates agent adoption states based on social exposure, campaign intensity, personal propensity, habit, and fatigue.

    Methods:
        step(persons, peer_exposure, campaign_intensity, params, rng) -> list[int]
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, params):
        """
        Initialize behavior adoption module with parameters.

        Args:
            params (dict): Parameters:
                - influence_weight_social
                - influence_weight_policy
                - influence_weight_personal
                - weight_attitude
                - weight_compliance
                - logistic_slope
                - logistic_intercept
                - noise_scale
                - habit_formation_rate
                - fatigue_rate
                - forgetting_rate
                - dropout_bias
                - baseline_retention
                - retention_weight
                - fatigue_weight
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.w_social = float(params.get("influence_weight_social", 0.6))
        self.w_policy = float(params.get("influence_weight_policy", 0.4))
        self.w_personal = float(params.get("influence_weight_personal", 0.3))
        # FIXED: Add weights for attitude and compliance to align with spec attributes.
        self.w_attitude = float(params.get("weight_attitude", 0.5))
        self.w_compliance = float(params.get("weight_compliance", 0.4))
        self.slope = float(params.get("logistic_slope", 4.0))
        self.intercept = float(params.get("logistic_intercept", -2.0))
        self.noise = float(params.get("noise_scale", 0.05))
        self.habit_rate = float(params.get("habit_formation_rate", 0.1))
        self.fatigue_rate = float(params.get("fatigue_rate", 0.02))
        self.forgetting = float(params.get("forgetting_rate", 0.05))
        self.dropout_bias = float(params.get("dropout_bias", 0.1))
        # FIXED: Calibrated dropout model with baseline retention, habit retention, and fatigue dropout effects.
        self.baseline_retention = float(params.get("baseline_retention", 0.85))
        self.retention_weight = float(params.get("retention_weight", 0.8))
        self.fatigue_weight = float(params.get("fatigue_weight", 0.6))

    def step(self, persons, peer_exposure, campaign_intensity, rng):
        """
        Advance adoption states by one day.

        Args:
            persons (list[Person]): List of agents.
            peer_exposure (list[float]): Smoothed peer exposure per agent.
            campaign_intensity (float): Policy campaign intensity in [0,1].
            rng (random.Random): RNG.

        Returns:
            list[int]: Adoption event per agent: +1 if newly adopted, -1 if dropped, 0 otherwise.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        events = [0] * len(persons)
        for i, p in enumerate(persons):
            social_term = self.w_social * p.susceptibility_to_influence * (peer_exposure[i] - p.adoption_threshold)
            policy_term = self.w_policy * p.trust_in_authorities * p.trust_modifier * campaign_intensity
            personal_term = self.w_personal * (p.baseline_adoption_propensity + p.risk_perception - 0.5)
            attitude_term = self.w_attitude * p.attitude_toward_masks
            compliance_term = self.w_compliance * p.compliance_propensity
            habit_term = p.habit_strength
            fatigue_term = -p.fatigue
            linear = social_term + policy_term + personal_term + attitude_term + compliance_term + habit_term + fatigue_term + self.intercept + rng.gauss(0.0, self.noise)
            p_adopt = clamp(sigmoid(self.slope * linear), 0.0, 1.0)

            if not p.is_adopting:
                new_state = rng.random() < p_adopt
                if new_state:
                    events[i] = 1
            else:
                # FIXED: Dropout probability uses baseline retention, reduced by habit, increased by fatigue.
                pos_drivers = social_term + policy_term + personal_term + attitude_term + compliance_term
                eps = 1e-6
                base_p_drop = max(0.0, 1.0 - self.baseline_retention)
                base_logit = math.log((base_p_drop + eps) / (1.0 - base_p_drop + eps))
                lin_drop = (-pos_drivers + self.dropout_bias) - self.retention_weight * p.habit_strength + self.fatigue_weight * p.fatigue + base_logit
                p_drop = clamp(sigmoid(self.slope * lin_drop), 0.0, 1.0)
                new_state = not (rng.random() < p_drop)
                if not new_state:
                    events[i] = -1

            # Update habit and fatigue
            if new_state:
                p.habit_strength = clamp(p.habit_strength + self.habit_rate * (1.0 - p.habit_strength), 0.0, 1.0)
                p.fatigue = clamp(p.fatigue + self.fatigue_rate, 0.0, 1.0)
            else:
                p.habit_strength = clamp(p.habit_strength - self.forgetting, 0.0, 1.0)
                p.fatigue = clamp(p.fatigue - self.fatigue_rate, 0.0, 1.0)

            p.is_adopting = bool(new_state)
        return events


class Government:
    """
    Government authority controlling mandates, messaging, and fines.

    Attributes:
        mandate_start_day (int): Start day of mandate.
        mandate_end_day (int): End day of mandate (inclusive).
        fine_amount (float): Fine applied upon enforcement when non-compliant.
        policy_stringency (float): Degree of policy emphasis.
        enforcement_resources (float): Resources scaling enforcement.
        messaging_intensity (float): Base intensity for messaging.

    Methods:
        is_mandate_active(day) -> bool
        issue_fine(person) -> float
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, params):
        """
        Initialize Government with configuration parameters.

        Args:
            params (dict): Dictionary with keys (aliased as needed):
                - policy_mandate_start_day
                - policy_mandate_end_day
                - policy_fine_amount
                - policy_stringency
                - enforcement_resources
                - messaging_intensity
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.mandate_start_day = int(params.get("policy_mandate_start_day", params.get("policy_start_day", 0)))
        self.mandate_end_day = int(params.get("policy_mandate_end_day", params.get("sim_days", 365)))
        self.fine_amount = float(params.get("policy_fine_amount", params.get("penalty_amount", 50.0)))
        self.policy_stringency = float(params.get("policy_stringency", 1.0))
        self.enforcement_resources = float(params.get("enforcement_resources", 1.0))
        self.messaging_intensity = float(params.get("messaging_intensity", params.get("campaign_base_intensity", 0.2)))

    def is_mandate_active(self, day):
        """
        Check if the mandate is active on a given day.

        Args:
            day (int): Day index.

        Returns:
            bool: True if active, else False.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        return self.mandate_start_day <= day <= self.mandate_end_day

    def issue_fine(self, person):
        """
        Apply fine effects to a person (behavioral adjustment).

        Args:
            person (Person): The person being fined.

        Returns:
            float: The fine amount applied.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        # Increase compliance and risk perception slightly
        person.compliance_propensity = clamp(person.compliance_propensity + 0.05 * self.policy_stringency, 0.0, 1.0)
        person.risk_perception = clamp(person.risk_perception + 0.02 * self.policy_stringency, 0.0, 1.0)
        return self.fine_amount


class Location:
    """
    Physical location where visits occur and policy may be enforced.

    Attributes:
        loc_id (int): Identifier.
        loc_type (str): 'home','work','public'.
        capacity (int): Capacity of the location.
        base_contact_rate (float): Baseline contacts at this location.
        mask_policy (bool): Whether masks are required here by local policy (excluding government toggles).
        enforcement_level (float): Level of enforcement at this location [0,1].
        foot_traffic_profile (float): Not used in minimal implementation.

    Methods:
        enforce_policy(persons, visitors, mandate_active, rng, government, enforcement_prob) -> int
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, loc_id, loc_type, capacity, base_contact_rate, mask_policy, enforcement_level, foot_traffic_profile=1.0):
        """
        Initialize a Location.

        Args:
            loc_id (int): Location id.
            loc_type (str): Type of location.
            capacity (int): Capacity.
            base_contact_rate (float): Base contact rate.
            mask_policy (bool): Local mask policy requirement.
            enforcement_level (float): Enforcement level [0,1].
            foot_traffic_profile (float): Foot traffic weight (unused placeholder).
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.loc_id = loc_id
        self.loc_type = loc_type
        self.capacity = int(max(1, capacity))
        self.base_contact_rate = float(base_contact_rate)
        self.mask_policy = bool(mask_policy)
        self.enforcement_level = clamp(float(enforcement_level), 0.0, 1.0)
        self.foot_traffic_profile = float(foot_traffic_profile)

    def enforce_policy(self, persons, visitors, mandate_active, rng, government, base_enforcement_prob):
        """
        Enforce mask policy for visits at this location, potentially issuing fines.

        Args:
            persons (list[Person]): Population list.
            visitors (list[int]): Indexes of persons visiting this location.
            mandate_active (bool): Whether government mandate is active.
            rng (random.Random): RNG.
            government (Government): Government authority for fines and stringency.
            base_enforcement_prob (float): Base probability of enforcement checks.

        Returns:
            int: Number of enforcement events triggered (fines issued).
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        events = 0
        requires_mask = self.mask_policy or (mandate_active and self.loc_type in ("work", "public"))
        if not requires_mask:
            return 0
        # Effective enforcement probability
        eff_prob = clamp(base_enforcement_prob * self.enforcement_level * (government.enforcement_resources / max(1e-6, government.policy_stringency)), 0.0, 1.0)
        for pid in visitors:
            p = persons[pid]
            if rng.random() < eff_prob:
                # Enforcement check
                if not p.is_adopting:
                    government.issue_fine(p)
                    events += 1
                else:
                    # Reinforce compliance slightly on successful checks
                    p.compliance_propensity = clamp(p.compliance_propensity + 0.01 * government.policy_stringency, 0.0, 1.0)
            else:
                # Decay compliance slightly when no enforcement
                p.compliance_propensity = clamp(p.compliance_propensity - 0.005, 0.0, 1.0)
        return events


class Retailer:
    """
    Retailer selling masks with inventory, price, restocking, and rationing.

    Attributes:
        retailer_id (int): Identifier.
        inventory (int): Current inventory level.
        restock_rate (int): Masks added each day.
        ration_limit (int): Max masks per purchase.
        price_mean (float): Mean price per unit.
        price_std (float): Std dev for price sampling.
        current_price (float): Current price per unit.

    Methods:
        restock()
        set_price(rng)
        sell_masks(max_qty) -> (qty, price)
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, retailer_id, initial_inventory, restock_rate, ration_limit, price_mean, price_std):
        """
        Initialize a Retailer instance.

        Args:
            retailer_id (int): Id.
            initial_inventory (int): Starting stock.
            restock_rate (int): Daily restock increment.
            ration_limit (int): Per-purchase maximum.
            price_mean (float): Price mean.
            price_std (float): Price standard deviation.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.retailer_id = retailer_id
        self.inventory = int(max(0, initial_inventory))
        self.restock_rate = int(max(0, restock_rate))
        self.ration_limit = int(max(1, ration_limit))
        self.price_mean = float(price_mean)
        self.price_std = float(price_std)
        self.current_price = max(0.1, float(price_mean))

    def restock(self):
        """
        Restock inventory by the restock rate.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        self.inventory += self.restock_rate

    def set_price(self, rng):
        """
        Update daily price by sampling around mean with normal noise.

        Args:
            rng (random.Random): RNG for sampling.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        p = rng.gauss(self.price_mean, self.price_std)
        self.current_price = max(0.1, p)

    def sell_masks(self, max_qty):
        """
        Sell up to max_qty masks to a buyer.

        Args:
            max_qty (int): Max desired units.

        Returns:
            tuple[int, float]: (quantity sold, unit price)
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        if self.inventory <= 0:
            return 0, self.current_price
        qty = min(int(max(0, max_qty)), self.ration_limit, self.inventory)
        self.inventory -= qty
        return qty, self.current_price


class MediaChannel:
    """
    Media channel broadcasting messages that affect risk perception and attitudes.

    Attributes:
        channel_id (int): Identifier.
        message_type (str): 'pro', 'neutral', 'anti'.
        message_intensity (float): Intensity factor.
        reach (float): Fraction of population reached per day [0,1].
        credibility (float): Effectiveness weight [0,1].

    Methods:
        broadcast_message(persons, rng)
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, channel_id, message_type, message_intensity, reach, credibility):
        """
        Initialize a MediaChannel.

        Args:
            channel_id (int): Channel id.
            message_type (str): Type of messaging.
            message_intensity (float): Intensity.
            reach (float): Daily reach fraction.
            credibility (float): Credibility.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.channel_id = channel_id
        self.message_type = message_type
        self.message_intensity = float(message_intensity)
        self.reach = clamp(float(reach), 0.0, 1.0)
        self.credibility = clamp(float(credibility), 0.0, 1.0)

    def broadcast_message(self, persons, rng):
        """
        Broadcast message to a subset of persons to update beliefs.

        Args:
            persons (list[Person]): Population.
            rng (random.Random): RNG for sampling recipients.

        Returns:
            int: Number of people reached.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        reached = 0
        direction = 0.0
        if self.message_type == "pro":
            direction = 1.0
        elif self.message_type == "anti":
            direction = -1.0
        else:
            direction = 0.0
        for p in persons:
            # Probability to receive based on channel reach and person's consumption level
            prob = clamp(self.reach * p.media_consumption_level, 0.0, 1.0)
            if rng.random() < prob:
                reached += 1
                # Update risk perception and attitude based on message intensity, credibility, and trust
                trust_adj = 1.0 + 0.5 * p.trust_in_authorities
                delta_risk = 0.05 * self.message_intensity * self.credibility * direction * trust_adj
                delta_att = 0.04 * self.message_intensity * self.credibility * direction
                p.risk_perception = clamp(p.risk_perception + delta_risk, 0.0, 1.0)
                p.attitude_toward_masks = clamp(p.attitude_toward_masks + delta_att, -1.0, 1.0)
        return reached


class Household:
    """
    Household entity for norms and mask allocation.

    Attributes:
        household_id (int): Identifier.
        member_ids (list[int]): Person ids in the household.
        norm_strength (float): Strength of household norm [0,1].

    Methods:
        share_norms(persons)
        allocate_masks(persons)
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, household_id, member_ids, norm_strength=0.3):
        """
        Initialize a Household.

        Args:
            household_id (int): Household id.
            member_ids (list[int]): Members.
            norm_strength (float): Norm strength.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.household_id = household_id
        self.member_ids = list(member_ids)
        self.norm_strength = clamp(float(norm_strength), 0.0, 1.0)

    def share_norms(self, persons):
        """
        Adjust member attitudes/compliance toward household average to model norms.

        Args:
            persons (list[Person]): Population list.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        if not self.member_ids:
            return
        attitudes = [persons[i].attitude_toward_masks for i in self.member_ids]
        compliance = [persons[i].compliance_propensity for i in self.member_ids]
        mean_att = sum(attitudes) / len(attitudes)
        mean_comp = sum(compliance) / len(compliance)
        for pid in self.member_ids:
            p = persons[pid]
            p.attitude_toward_masks = clamp(p.attitude_toward_masks + self.norm_strength * (mean_att - p.attitude_toward_masks) * 0.2, -1.0, 1.0)
            p.compliance_propensity = clamp(p.compliance_propensity + self.norm_strength * (mean_comp - p.compliance_propensity) * 0.2, 0.0, 1.0)

    def allocate_masks(self, persons):
        """
        Reallocate masks among members to reduce shortages within household.

        Args:
            persons (list[Person]): Population list.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        # Simple heuristic: if any member has zero and another has > 3, transfer 1
        if not self.member_ids:
            return
        needers = [i for i in self.member_ids if persons[i].mask_inventory <= 0]
        donors = [i for i in self.member_ids if persons[i].mask_inventory >= 3]
        for n in needers:
            if not donors:
                break
            d = donors.pop(0)
            if persons[d].mask_inventory >= 2:
                persons[d].mask_inventory -= 1
                persons[n].mask_inventory += 1
                if persons[d].mask_inventory >= 3:
                    donors.append(d)


class ObservationAggregator:
    """
    Computes daily observables including adoption rates, market metrics, and policy compliance.

    Methods:
        record(day, persons, adoption_events, campaign_intensity)
        record_market(day, retailers, purchases_today)
        record_enforcement(count)
        record_visits(day, visits_map, mandated_compliance_rate)
        results() -> dict
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, params, community_count):
        """
        Initialize aggregator.

        Args:
            params (dict): Parameters including:
                - smoothing_window_days
                - community_reporting_subset (unused placeholder)
            community_count (int): Number of communities.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.smooth_window = max(1, int(params.get("smoothing_window_days", 1)))
        # Placeholder for future use
        self.community_reporting_subset = float(params.get("community_reporting_subset", 1.0))
        self.series_overall = []
        self.series_new_adopt = []
        self.series_churn = []
        self.series_campaign = []
        self.series_by_comm = []
        self.community_count = community_count
        # FIXED: Added trackers for required metrics (prices, stockouts, enforcement, purchases, compliance).
        self.enforcement_events = 0
        self.daily_stockout_flag = []
        self.total_price_paid = 0.0
        self.total_masks_bought = 0
        self.daily_purchases_per_capita = []
        self.daily_avg_price = []
        self.daily_compliance_mandated = []
        self.adoption_by_income_quintile_daily = []

    def record(self, day, persons, adoption_events, campaign_intensity):
        """
        Record daily observables.

        Args:
            day (int): Day index.
            persons (list[Person]): Agents.
            adoption_events (list[int]): Adoption events for the day.
            campaign_intensity (float): Campaign intensity.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        n = len(persons)
        if n == 0:
            return
        adopters = sum(1 for p in persons if p.is_adopting)
        overall_rate = adopters / float(n)
        churn = sum(1 for e in adoption_events if e == -1) / float(n)
        new_adopt = sum(1 for e in adoption_events if e == 1) / float(n)
        # By community
        by_comm_count = defaultdict(int)
        by_comm_adopt = defaultdict(int)
        for p in persons:
            by_comm_count[p.community_id] += 1
            if p.is_adopting:
                by_comm_adopt[p.community_id] += 1
        by_comm_rate = {cid: (by_comm_adopt[cid] / float(by_comm_count[cid]) if by_comm_count[cid] > 0 else 0.0) for cid in by_comm_count.keys()}
        self.series_overall.append(overall_rate)
        self.series_new_adopt.append(new_adopt)
        self.series_churn.append(churn)
        self.series_campaign.append(campaign_intensity)
        self.series_by_comm.append(by_comm_rate)
        # Income quintiles adoption
        incomes = [(p.income_level, 1.0 if p.is_adopting else 0.0) for p in persons]
        incomes_sorted = sorted(incomes, key=lambda x: x[0])
        quintiles = []
        if n > 0:
            qsize = max(1, n // 5)
            for qi in range(5):
                start = qi * qsize
                end = (qi + 1) * qsize if qi < 4 else n
                qchunk = incomes_sorted[start:end]
                if len(qchunk) == 0:
                    quintiles.append(0.0)
                else:
                    quintiles.append(sum(v for _, v in qchunk) / float(len(qchunk)))
        self.adoption_by_income_quintile_daily.append({f"Q{qi+1}": quintiles[qi] for qi in range(len(quintiles))})

    def record_market(self, day, retailers, purchases_today, population_size=None):
        """
        Record daily market outcomes: stockouts, purchase volumes, and prices.

        Args:
            day (int): Day index.
            retailers (list[Retailer]): Retailers list.
            purchases_today (list[tuple[float,int]]): List of (unit price, quantity) sales.
            population_size (int or None): Population size for per-capita rate (optional).
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        stockout = any(r.inventory <= 0 for r in retailers)
        self.daily_stockout_flag.append(bool(stockout))
        total_qty = 0
        price_sum = 0.0
        count_sales = 0
        for price, qty in purchases_today:
            self.total_price_paid += price * qty
            self.total_masks_bought += qty
            total_qty += qty
            price_sum += price
            count_sales += 1
        per_capita = (total_qty / float(population_size)) if population_size and population_size > 0 else (total_qty)
        self.daily_purchases_per_capita.append(per_capita)
        avg_price_today = (price_sum / float(count_sales)) if count_sales > 0 else (sum(r.current_price for r in retailers) / float(len(retailers)) if retailers else 0.0)
        self.daily_avg_price.append(avg_price_today)

    def record_enforcement(self, count):
        """
        Record enforcement events.

        Args:
            count (int): Number of enforcement events today.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        self.enforcement_events += int(count)

    def record_visits(self, day, visits_map, mandated_compliance_rate):
        """
        Record visits and compliance rate in mandated locations.

        Args:
            day (int): Day index.
            visits_map (dict[int, list[int]]): Location id -> visitors list (unused in aggregator).
            mandated_compliance_rate (float): Compliance rate in mandated locations for this day.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        self.daily_compliance_mandated.append(float(mandated_compliance_rate))

    def results(self):
        """
        Get aggregated results with smoothing applied to selected series.

        Returns:
            dict: Contains time series and summary metrics including prices, stockouts, and enforcement.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        smoothed_overall = moving_average(self.series_overall, self.smooth_window)
        # For by-community smoothing, apply moving average per community id over time
        all_comm_ids = set()
        for d in self.series_by_comm:
            all_comm_ids.update(d.keys())
        comm_series = {cid: [] for cid in sorted(all_comm_ids)}
        for d in self.series_by_comm:
            for cid in comm_series.keys():
                comm_series[cid].append(float(d.get(cid, 0.0)))
        comm_series_smoothed = {cid: moving_average(vals, self.smooth_window) for cid, vals in comm_series.items()}
        days = len(self.series_overall)
        by_comm_daily = []
        for t in range(days):
            day_dict = {cid: comm_series_smoothed[cid][t] for cid in comm_series_smoothed.keys()}
            by_comm_daily.append(day_dict)

        avg_price_overall = (self.total_price_paid / self.total_masks_bought) if self.total_masks_bought > 0 else 0.0
        stockout_days = sum(1 for f in self.daily_stockout_flag if f)
        return {
            "overall_adoption_rate_over_time": smoothed_overall,
            "adoption_rate_by_community_over_time": by_comm_daily,
            "adoption_churn_daily": list(self.series_churn),
            "new_adoptions_daily": list(self.series_new_adopt),
            "campaign_intensity_daily": list(self.series_campaign),
            "average_price_paid": avg_price_overall,
            "stockout_days": stockout_days,
            "enforcement_events_count": self.enforcement_events,
            "purchases_per_capita_daily": list(self.daily_purchases_per_capita),
            "avg_price_daily": list(self.daily_avg_price),
            "compliance_rate_mandated_daily": list(self.daily_compliance_mandated),
            "adoption_by_income_quintile_daily": list(self.adoption_by_income_quintile_daily),
        }


class Simulation:
    """
    Main simulation orchestrator that initializes agents, builds the network, and runs the daily loop.

    Methods:
        run()
        get_result_json() -> dict
        save_results(filename)
        visualize(filename=None, show=False)
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; class logic follows.

    def __init__(self, cfg):
        """
        Initialize the Simulation with configuration.

        Args:
            cfg (dict): Configuration dictionary including parameters and options.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.cfg = cfg or {}
        self.params = self._merge_defaults(self.cfg)
        seed = int(self.params.get("random_seed", self.params.get("seed", 42)))
        self.rng = random.Random(seed)
        # FIXED: Use sim_days aliased from time_horizon_days if provided.
        self.population = int(self.params.get("population_size", 500))
        self.sim_days = int(self.params.get("sim_days", int(self.params.get("simulation_days", 60))))
        self.community_count = int(self.params.get("net_community_count", 8))
        # Assign households first for consistent home location mapping
        self.household_assignments, self.household_count = self._assign_households(self.population, float(self.params.get("household_size_mean", 2.5)))
        # Initialize agents
        self.persons = self._initialize_population()

        # Build network
        self.network = NetworkBuilder.build_network(
            self.population,
            [p.community_id for p in self.persons],
            self.params,
            self.rng,
        )

        # Entities per feedback/spec
        self.government = Government(self.params)

        # Locations setup
        self.locations, self.home_location_ids, self.work_location_ids, self.public_location_ids = self._initialize_locations()
        # Assign work locations to persons
        for p in self.persons:
            if self.work_location_ids:
                p.workplace_id = self.rng.choice(self.work_location_ids)
            # Map home location from household id (one-to-one)
            if self.home_location_ids:
                p.home_location_id = self.home_location_ids[p.household_id] if p.household_id is not None else None
            else:
                p.home_location_id = None

        # Households with member lists
        hh_members = defaultdict(list)
        for p in self.persons:
            if p.household_id is not None:
                hh_members[p.household_id].append(p.idx)
        self.households = [Household(hid, members, norm_strength=0.3) for hid, members in sorted(hh_members.items(), key=lambda kv: kv[0])]

        # Retailers (market)
        self.retailers = self._initialize_retailers()

        # Media channels
        self.media_channels = self._initialize_media_channels()

        # Modules
        self.campaign = PolicyCampaign(self.params)  # still used as generic signal
        self.mobility = MobilityContact(self.params)
        self.behavior = BehaviorAdoption(self.params)
        self.aggregator = ObservationAggregator(self.params, self.community_count)
        self.peer_history = [deque() for _ in range(self.population)]
        self.metadata = {
            "random_seed": seed,
            "population": self.population,
            "sim_days": self.sim_days,
            "community_count": self.community_count,
        }
        self.result = None

    def _merge_defaults(self, user_cfg):
        """
        Merge user configuration with defaults from the model plan and apply aliasing.

        Args:
            user_cfg (dict): User-supplied config.

        Returns:
            dict: Resolved parameters dictionary.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        defaults = {
            "population_size": 500,
            "sim_days": 60,
            "random_seed": 42,
            "initial_adoption_fraction": 0.15,
            "age_mean": 40.0,
            "age_std": 12.0,
            "risk_perception_mean": 0.5,
            "risk_perception_std": 0.2,
            "trust_in_authorities_mean": 0.6,
            "trust_in_authorities_std": 0.25,
            "susceptibility_to_influence_mean": 0.5,
            "susceptibility_to_influence_std": 0.2,
            "baseline_adoption_propensity_mean": 0.3,
            "baseline_adoption_propensity_std": 0.15,
            "threshold_mean": 0.5,
            "threshold_std": 0.15,
            "net_avg_degree": 10.0,
            "net_rewiring_prob": 0.05,
            "net_homophily_strength": 0.5,
            "net_community_count": 8,
            "net_weight_mean": 1.0,
            "net_weight_std": 0.3,
            "net_fraction_strong_ties": 0.2,
            "campaign_base_intensity": 0.2,
            "campaign_start_day": 5,
            "campaign_ramp_days": 10,
            "campaign_max_intensity": 0.8,
            "campaign_decay_days": 30,
            "campaign_pulse_interval": 14,
            "campaign_pulse_magnitude": 0.1,
            "trust_modifier_mean": 1.0,
            "trust_modifier_std": 0.2,
            "contact_rate_per_day": 12.0,
            "contact_bias_toward_strong_ties": 0.6,
            "mobility_variance": 0.2,
            "weekend_multiplier": 0.8,
            "shock_probability": 0.01,
            "shock_magnitude": 0.5,
            "influence_weight_social": 0.6,
            "influence_weight_policy": 0.4,
            "influence_weight_personal": 0.3,
            "weight_attitude": 0.5,
            "weight_compliance": 0.4,
            "logistic_slope": 4.0,
            "logistic_intercept": -2.0,
            "noise_scale": 0.05,
            "habit_formation_rate": 0.1,
            "fatigue_rate": 0.02,
            "forgetting_rate": 0.05,
            "dropout_bias": 0.1,
            "baseline_retention": 0.85,  # FIXED: Added baseline retention for calibrated dropout.
            "retention_weight": 0.8,     # FIXED: Habit reduces dropout.
            "fatigue_weight": 0.6,       # FIXED: Fatigue increases dropout.
            "peer_window_days": 3,
            "smoothing_window_days": 1,
            "community_reporting_subset": 1.0,
            # Additional parameters for new entities
            "policy_mandate_start_day": 14,
            "policy_mandate_end_day": 365,
            "policy_fine_amount": 50.0,
            "policy_stringency": 1.0,
            "enforcement_resources": 1.0,
            "location_enforcement_prob": 0.6,
            "household_size_mean": 2.5,
            "income_log_mean": 10.0,
            "income_log_sd": 0.7,
            "attitude_mean": 0.0,
            "attitude_std": 0.5,
            "compliance_propensity_mean": 0.6,
            "compliance_propensity_std": 0.2,
            "media_consumption_mean": 0.5,
            "media_consumption_std": 0.25,
            "market_retailer_count": 5,
            "market_initial_inventory": 1000,
            "market_restock_rate": 200,
            "market_ration_limit": 10,
            "market_price_mean": 1.5,
            "market_price_std": 0.2,
            "masks_used_per_day": 0.2,
            "max_mask_inventory_per_person": 50,
            "work_travel_fraction": 0.6,
            "public_space_visit_rate": 0.3,
            "base_contact_rate_home": 2.0,
            "base_contact_rate_work": 6.0,
            "base_contact_rate_public": 8.0,
            "media_message_split": [0.5, 0.3, 0.2],  # [pro, neutral, anti]
            "media_channel_count": 3,
            "media_reach_pro": 0.6,
            "media_reach_neutral": 0.4,
            "media_reach_anti": 0.3,
            "media_credibility_pro": 0.9,
            "media_credibility_neutral": 0.7,
            "media_credibility_anti": 0.6,
            "contact_sample_cap": 12,
        }
        merged = dict(defaults)
        # Flatten nested "parameters" dict if present
        if "parameters" in user_cfg and isinstance(user_cfg["parameters"], dict):
            merged.update(user_cfg["parameters"])
        # Also apply top-level overrides
        for k, v in user_cfg.items():
            if k not in ("entities", "modules", "observables", "metrics", "environment", "initialization", "algorithms", "data_sources", "code_structure", "prediction_period", "evaluation_metrics"):
                merged[k] = v
            else:
                pass
        # FIXED: Alias task-spec parameters to internal names
        alias = {
            "time_horizon_days": "sim_days",
            "initial_adoption_rate": "initial_adoption_fraction",
            "campaign_intensity": "campaign_base_intensity",
            "mandate_start_day": "policy_mandate_start_day",
            "mandate_end_day": "policy_mandate_end_day",
            "policy_enforcement_probability": "location_enforcement_prob",
            "fine_amount": "policy_fine_amount",
            "retailer_count": "market_retailer_count",
            "initial_retailer_inventory": "market_initial_inventory",
            "restock_rate_per_day": "market_restock_rate",
            "rationing_limit_per_purchase": "market_ration_limit",
            "mask_price_mean": "market_price_mean",
            "mask_price_std": "market_price_std",
            "media_message_split": "media_message_split",
            # Task spec naming
            "simulation_days": "sim_days",
            "policy_start_day": "policy_mandate_start_day",
            "enforcement_probability": "location_enforcement_prob",
            "penalty_amount": "policy_fine_amount",
            "mask_price": "market_price_mean",
            "supply_capacity_per_day": "market_restock_rate",
            "distribution_delay_days": "market_distribution_delay_days",  # not used in minimal version
        }
        for src, dst in alias.items():
            if src in merged:
                merged[dst] = merged[src]
        # Default sim_days from time_horizon_days if provided
        if "sim_days" not in merged and "time_horizon_days" in merged:
            merged["sim_days"] = int(merged["time_horizon_days"])
        # If prediction_period specified, optionally override sim_days
        if "prediction_period" in user_cfg and isinstance(user_cfg["prediction_period"], dict):
            end = user_cfg["prediction_period"].get("end_day")
            if end is not None:
                try:
                    merged["sim_days"] = int(end) + 1
                except Exception:
                    pass
        return merged

    def _assign_households(self, N, hh_size_mean):
        """
        Assign persons to households approximately matching a mean size.

        Args:
            N (int): Population size.
            hh_size_mean (float): Mean household size.

        Returns:
            tuple[list[int], int]: (household_id per person, number of households)
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        if N <= 0:
            return [], 0
        target_households = max(1, int(round(N / max(1.0, hh_size_mean))))
        household_sizes = [1] * target_households
        # Distribute remaining members randomly to households
        remaining = N - target_households
        for _ in range(remaining):
            household_sizes[self.rng.randrange(target_households)] += 1
        # Assign persons to households
        assignments = [-1] * N
        pid = 0
        for hid, size in enumerate(household_sizes):
            for _ in range(size):
                if pid < N:
                    assignments[pid] = hid
                    pid += 1
        return assignments, target_households

    def _initialize_population(self):
        """
        Initialize the population with attributes drawn from specified distributions.

        Returns:
            list[Person]: Initialized list of agents.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        N = self.population
        p = self.params
        persons = []
        # Assign communities uniformly
        communities = [i % self.community_count for i in range(N)]
        # Initial adoption sample
        init_frac = clamp(float(p.get("initial_adoption_fraction", 0.15)), 0.0, 1.0)
        num_init = int(round(init_frac * N))
        init_adopters = set(self.rng.sample(range(N), num_init)) if num_init > 0 else set()

        # Trust modifier per person
        t_mu = float(p.get("trust_modifier_mean", 1.0))
        t_sigma = float(p.get("trust_modifier_std", 0.2))

        for i in range(N):
            age = max(0.0, self.rng.gauss(float(p.get("age_mean", 40.0)), float(p.get("age_std", 12.0))))
            risk = clamp(self.rng.gauss(float(p.get("risk_perception_mean", 0.5)), float(p.get("risk_perception_std", 0.2))), 0.0, 1.0)
            trust = clamp(self.rng.gauss(float(p.get("trust_in_authorities_mean", 0.6)), float(p.get("trust_in_authorities_std", 0.25))), 0.0, 1.0)
            suscept = clamp(self.rng.gauss(float(p.get("susceptibility_to_influence_mean", 0.5)), float(p.get("susceptibility_to_influence_std", 0.2))), 0.0, 1.0)
            base_prop = clamp(self.rng.gauss(float(p.get("baseline_adoption_propensity_mean", 0.3)), float(p.get("baseline_adoption_propensity_std", 0.15))), 0.0, 1.0)
            threshold = clamp(self.rng.gauss(float(p.get("threshold_mean", 0.5)), float(p.get("threshold_std", 0.15))), 0.0, 1.0)
            is_adopt = (i in init_adopters)
            habit = 0.4 if is_adopt else 0.0
            fatigue = 0.0
            trust_mod = max(0.0, self.rng.gauss(t_mu, t_sigma))
            # New attributes from spec
            income = math.exp(self.rng.gauss(float(p.get("income_log_mean", 10.0)), float(p.get("income_log_sd", 0.7))))
            compliance = clamp(self.rng.gauss(float(p.get("compliance_propensity_mean", 0.6)), float(p.get("compliance_propensity_std", 0.2))), 0.0, 1.0)
            attitude = clamp(self.rng.gauss(float(p.get("attitude_mean", 0.0)), float(p.get("attitude_std", 0.5))), -1.0, 1.0)
            media_consume = clamp(self.rng.gauss(float(p.get("media_consumption_mean", 0.5)), float(p.get("media_consumption_std", 0.25))), 0.0, 1.0)
            mask_inventory = poisson_sample(2.0, self.rng) if is_adopt else 0
            persons.append(Person(
                idx=i,
                age=age,
                community_id=communities[i],
                risk_perception=risk,
                susceptibility_to_influence=suscept,
                trust_in_authorities=trust,
                baseline_adoption_propensity=base_prop,
                adoption_threshold=threshold,
                habit_strength=habit,
                fatigue=fatigue,
                is_adopting=is_adopt,
                trust_modifier=trust_mod,
                income_level=income,
                compliance_propensity=compliance,
                attitude_toward_masks=attitude,
                mask_inventory=mask_inventory,
                perceived_cost=1.0,
                household_id=self.household_assignments[i],
                workplace_id=None,
                daily_contacts=int(self.params.get("contact_rate_per_day", 12.0)),
                media_consumption_level=media_consume,
            ))
        return persons

    def _initialize_locations(self):
        """
        Initialize locations: home (one per household), work, and public.

        Returns:
            tuple: (locations list, home_ids, work_ids, public_ids)
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        locs = []
        home_ids = []
        work_ids = []
        public_ids = []
        # Home locations
        base_contact_home = float(self.params.get("base_contact_rate_home", 2.0))
        for hid in range(self.household_count):
            loc_id = len(locs)
            locs.append(Location(
                loc_id=loc_id,
                loc_type="home",
                capacity=10,
                base_contact_rate=base_contact_home,
                mask_policy=False,
                enforcement_level=0.1
            ))
            home_ids.append(loc_id)

        # Work locations
        base_contact_work = float(self.params.get("base_contact_rate_work", 6.0))
        # Heuristic number of work locations: one per 50 people
        num_work = max(1, self.population // 50)
        for _ in range(num_work):
            loc_id = len(locs)
            locs.append(Location(
                loc_id=loc_id,
                loc_type="work",
                capacity=200,
                base_contact_rate=base_contact_work,
                mask_policy=True,
                enforcement_level=0.7
            ))
            work_ids.append(loc_id)

        # Public locations
        base_contact_public = float(self.params.get("base_contact_rate_public", 8.0))
        num_public = max(3, self.population // 100)
        for _ in range(num_public):
            loc_id = len(locs)
            locs.append(Location(
                loc_id=loc_id,
                loc_type="public",
                capacity=500,
                base_contact_rate=base_contact_public,
                mask_policy=True,
                enforcement_level=0.5
            ))
            public_ids.append(loc_id)

        return locs, home_ids, work_ids, public_ids

    def _initialize_retailers(self):
        """
        Initialize retailers/market.

        Returns:
            list[Retailer]: Retailers.
        """
        count = int(self.params.get("market_retailer_count", 5))
        init_inv = int(self.params.get("market_initial_inventory", 1000))
        restock = int(self.params.get("market_restock_rate", 200))
        ration = int(self.params.get("market_ration_limit", 10))
        price_mean = float(self.params.get("market_price_mean", 1.5))
        price_std = float(self.params.get("market_price_std", 0.2))
        retailers = []
        for rid in range(count):
            retailers.append(Retailer(
                retailer_id=rid,
                initial_inventory=init_inv,
                restock_rate=restock,
                ration_limit=ration,
                price_mean=price_mean,
                price_std=price_std
            ))
        return retailers

    def _initialize_media_channels(self):
        """
        Initialize media channels based on split and counts.

        Returns:
            list[MediaChannel]: Channels.
        """
        channels = []
        total_count = int(self.params.get("media_channel_count", 3))
        split = self.params.get("media_message_split", [0.5, 0.3, 0.2])
        types = ["pro", "neutral", "anti"]
        # Determine count per type
        counts = [int(round(frac * total_count)) for frac in split]
        # Adjust to match total_count
        while sum(counts) < total_count:
            counts[0] += 1
        while sum(counts) > total_count:
            for i in range(3):
                if counts[i] > 0 and sum(counts) > total_count:
                    counts[i] -= 1
        # Per-type params
        reach_map = {
            "pro": float(self.params.get("media_reach_pro", 0.6)),
            "neutral": float(self.params.get("media_reach_neutral", 0.4)),
            "anti": float(self.params.get("media_reach_anti", 0.3)),
        }
        cred_map = {
            "pro": float(self.params.get("media_credibility_pro", 0.9)),
            "neutral": float(self.params.get("media_credibility_neutral", 0.7)),
            "anti": float(self.params.get("media_credibility_anti", 0.6)),
        }
        cid = 0
        for t, c in zip(types, counts):
            for _ in range(c):
                channels.append(MediaChannel(
                    channel_id=cid,
                    message_type=t,
                    message_intensity=1.0,
                    reach=reach_map[t],
                    credibility=cred_map[t]
                ))
                cid += 1
        return channels

    def run(self):
        """
        Run the simulation loop.
        """
        results = None
        for day in range(self.sim_days):
            # Retailers daily update
            for r in self.retailers:
                r.restock()
                r.set_price(self.rng)

            # Media broadcasts
            for ch in self.media_channels:
                ch.broadcast_message(self.persons, self.rng)

            # Policy campaign intensity
            campaign_intensity = self.campaign.intensity(day)

            # Peer exposure
            peer_exposure = self.mobility.compute_exposure(day, self.network, self.persons, self.rng, self.peer_history)

            # Behavior update
            adoption_events = self.behavior.step(self.persons, peer_exposure, campaign_intensity, self.rng)

            # Households share norms and allocate masks
            for hh in self.households:
                hh.share_norms(self.persons)
                hh.allocate_masks(self.persons)

            # Visits and enforcement
            visits_map = defaultdict(list)
            mandate_active = self.government.is_mandate_active(day)
            base_enf_prob = float(self.params.get("location_enforcement_prob", 0.6))

            # Work visits (weekdays only)
            if day % 7 not in (5, 6):
                for p in self.persons:
                    if self.work_location_ids and self.rng.random() < float(self.params.get("work_travel_fraction", 0.6)):
                        if p.workplace_id is not None:
                            visits_map[p.workplace_id].append(p.idx)

            # Public visits
            for p in self.persons:
                if self.public_location_ids and self.rng.random() < float(self.params.get("public_space_visit_rate", 0.3)):
                    loc_id = self.rng.choice(self.public_location_ids)
                    visits_map[loc_id].append(p.idx)

            # Enforce policy
            enforcement_count = 0
            mandated_visits = 0
            mandated_compliant = 0
            for loc in self.locations:
                visitors = visits_map.get(loc.loc_id, [])
                if visitors:
                    enforcement_count += loc.enforce_policy(
                        self.persons,
                        visitors,
                        mandate_active,
                        self.rng,
                        self.government,
                        base_enf_prob
                    )
                    if mandate_active and loc.loc_type in ("work", "public"):
                        mandated_visits += len(visitors)
                        mandated_compliant += sum(1 for pid in visitors if self.persons[pid].is_adopting)
            compliance_rate = (mandated_compliant / mandated_visits) if mandated_visits > 0 else 0.0
            self.aggregator.record_enforcement(enforcement_count)
            self.aggregator.record_visits(day, visits_map, compliance_rate)

            # Market purchases
            purchases_today = []
            max_inv = int(self.params.get("max_mask_inventory_per_person", 50))
            for p in self.persons:
                # Simple purchase rule: if adopting and inventory low, or high risk+compliance, try to buy
                need = 0
                if p.is_adopting and p.mask_inventory < 5:
                    need = 5 - p.mask_inventory
                elif (p.risk_perception + p.compliance_propensity) > 1.2 and p.mask_inventory < 3:
                    need = 3 - p.mask_inventory
                need = max(0, min(need, max_inv - p.mask_inventory))
                if need <= 0:
                    continue
                # Pick retailer with lowest price
                if not self.retailers:
                    continue
                retailer = min(self.retailers, key=lambda r: r.current_price)
                qty, price = retailer.sell_masks(need)
                if qty > 0:
                    p.mask_inventory += qty
                    purchases_today.append((price, qty))

            # Mask usage
            use_rate = float(self.params.get("masks_used_per_day", 0.2))
            for p in self.persons:
                if p.is_adopting and p.mask_inventory > 0:
                    # Use ~use_rate masks per day (stochastic integer)
                    use = 1 if self.rng.random() < use_rate else 0
                    p.mask_inventory = max(0, p.mask_inventory - use)

            # Record daily metrics
            self.aggregator.record(day, self.persons, adoption_events, campaign_intensity)
            self.aggregator.record_market(day, self.retailers, purchases_today, population_size=self.population)

        self.result = {
            "metadata": self.metadata,
            "results": self.aggregator.results()
        }
        return self.result

    def get_result_json(self):
        """
        Get results as a JSON-serializable dict.
        """
        if self.result is None:
            self.run()
        return self.result

    def save_results(self, filename):
        """
        Save results to a JSON file.
        """
        res = self.get_result_json()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    def visualize(self, filename=None, show=False):
        """
        Placeholder for visualization (not implemented to keep minimal deps).
        """
        return None


# Execute main for both direct execution and sandbox wrapper invocation
main()