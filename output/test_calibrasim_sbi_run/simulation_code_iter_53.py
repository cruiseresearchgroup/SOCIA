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
        risk_perception (float): Risk perception in [0, 1].
        susceptibility_to_influence (float): Susceptibility to peer influence in [0, 1].
        trust_in_authorities (float): Trust in authorities in [0, 1].
        baseline_adoption_propensity (float): Baseline propensity in [0, 1].
        adoption_threshold (float): Personal threshold in [0, 1].
        habit_strength (float): Habit strength in [0, 1].
        fatigue (float): Fatigue level in [0, 1].
        is_adopting (bool): Current adoption state.
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
            is_adopting (bool): Initial adoption state.
            trust_modifier (float): Trust multiplicative modifier for policy effect.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.idx = idx
        self.age = age
        self.community_id = community_id
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

        # Base ring lattice
        for i in range(n):
            for offset in range(1, k // 2 + 1):
                j = (i + offset) % n
                if i < j:
                    G.add_edge(i, j, weight=1.0, is_strong=False)

        # FIXED: Rewiring now selects candidates from explicit non-neighbors and restores original edge if none available to preserve degree.
        for i in range(n):
            neighbors = list(G.neighbors(i))
            for nb, _, _ in neighbors:
                if i < nb and rng.random() < p_rewire:
                    # remove original edge
                    G.adj[i] = [(j, w2, s2) for (j, w2, s2) in G.adj[i] if j != nb]
                    G.adj[nb] = [(j, w2, s2) for (j, w2, s2) in G.adj[nb] if j != i]
                    excluded = {i} | {j for j, _, _ in G.adj[i]}
                    candidates = [u for u in range(n) if u not in excluded]
                    if candidates:
                        candidate = rng.choice(candidates)
                        G.add_edge(i, candidate, weight=1.0, is_strong=False)
                    else:
                        # restore original edge to preserve degree
                        G.add_edge(i, nb, weight=1.0, is_strong=False)

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
            # First remove existing simple weight
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
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; initializer logic follows.
        self.rate = float(params.get("contact_rate_per_day", 12.0))
        self.bias = float(params.get("contact_bias_toward_strong_ties", 0.6))
        self.mobility_var = float(params.get("mobility_variance", 0.2))
        self.weekend_mult = float(params.get("weekend_multiplier", 0.8))
        self.shock_prob = float(params.get("shock_probability", 0.01))
        self.shock_mag = float(params.get("shock_magnitude", 0.5))
        self.peer_window_days = max(1, int(params.get("peer_window_days", 3)))

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
                # Update history and smooth (ensure consistent length)
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
                sampled = weighted_sample_without_replacement(items, weights, min(k, deg), rng)
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
            habit_term = p.habit_strength
            fatigue_term = -p.fatigue
            linear = social_term + policy_term + personal_term + habit_term + fatigue_term + self.intercept + rng.gauss(0.0, self.noise)
            p_adopt = clamp(sigmoid(self.slope * linear), 0.0, 1.0)

            if not p.is_adopting:
                new_state = rng.random() < p_adopt
                if new_state:
                    events[i] = 1
            else:
                # FIXED: Dropout probability uses baseline retention, reduced by habit, increased by fatigue.
                pos_drivers = social_term + policy_term + personal_term
                # Map baseline retention to a logit bias for dropout; small epsilon to avoid div by zero
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


class ObservationAggregator:
    """
    Computes daily observables including adoption rates and campaign intensity.

    Methods:
        record(day, persons, adoption_events, campaign_intensity)
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
        # Store as a dictionary mapping community_id to rate
        self.series_by_comm.append(by_comm_rate)

    def results(self):
        """
        Get aggregated results with smoothing applied to selected series.

        Returns:
            dict: Contains time series for overall adoption rate, community rates, churn, new adoptions, and campaign intensity.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        smoothed_overall = moving_average(self.series_overall, self.smooth_window)
        # For by-community smoothing, apply moving average per community id over time
        # First collect all community ids observed
        all_comm_ids = set()
        for d in self.series_by_comm:
            all_comm_ids.update(d.keys())
        # Build per-community series
        comm_series = {cid: [] for cid in sorted(all_comm_ids)}
        for d in self.series_by_comm:
            for cid in comm_series.keys():
                comm_series[cid].append(float(d.get(cid, 0.0)))
        comm_series_smoothed = {cid: moving_average(vals, self.smooth_window) for cid, vals in comm_series.items()}
        # Transpose back to list of dicts per day
        days = len(self.series_overall)
        by_comm_daily = []
        for t in range(days):
            day_dict = {cid: comm_series_smoothed[cid][t] for cid in comm_series_smoothed.keys()}
            by_comm_daily.append(day_dict)

        return {
            "overall_adoption_rate_over_time": smoothed_overall,
            "adoption_rate_by_community_over_time": by_comm_daily,
            "adoption_churn_daily": list(self.series_churn),
            "new_adoptions_daily": list(self.series_new_adopt),
            "campaign_intensity_daily": list(self.series_campaign),
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
        seed = int(self.params.get("random_seed", 42))
        self.rng = random.Random(seed)
        self.population = int(self.params.get("population_size", 500))
        self.sim_days = int(self.params.get("sim_days", int(self.params.get("simulation_days", 60))))
        self.community_count = int(self.params.get("net_community_count", 8))

        # Initialize agents
        self.persons = self._initialize_population()

        # Build network
        self.network = NetworkBuilder.build_network(
            self.population,
            [p.community_id for p in self.persons],
            self.params,
            self.rng,
        )

        # Modules
        self.campaign = PolicyCampaign(self.params)
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

    def _merge_defaults(self, user_cfg):
        """
        Merge user configuration with defaults from the model plan.

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
        }
        merged = dict(defaults)
        # Flatten any nested "parameters" dict if present
        if "parameters" in user_cfg and isinstance(user_cfg["parameters"], dict):
            merged.update(user_cfg["parameters"])
        # Also apply top-level overrides
        for k, v in user_cfg.items():
            # Avoid overriding when complex structures included in model_plan
            if k not in ("entities", "modules", "observables", "metrics", "environment", "initialization", "algorithms", "data_sources", "code_structure", "prediction_period", "evaluation_metrics"):
                merged[k] = v
            else:
                # Keep as is for future use
                pass
        return merged

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
            ))
        return persons

    def run(self):
        """
        Execute the simulation loop over the configured number of days.

        Side effects:
            Populates the aggregator with recorded observables.

        Returns:
            None
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        for day in range(self.sim_days):
            intensity = self.campaign.intensity(day)
            exposure = self.mobility.compute_exposure(day, self.network, self.persons, self.rng, self.peer_history)
            events = self.behavior.step(self.persons, exposure, intensity, self.rng)
            self.aggregator.record(day, self.persons, events, intensity)

    def get_result_json(self):
        """
        Prepare structured JSON results including series, summary, and metadata.

        Returns:
            dict: Results dictionary ready to be JSON-serialized.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        series = self.aggregator.results()
        overall = series.get("overall_adoption_rate_over_time", [])
        peak_val = max(overall) if overall else 0.0
        peak_day = overall.index(peak_val) if overall else None
        avg_rate = sum(overall) / float(len(overall)) if overall else 0.0
        final_rate = overall[-1] if overall else 0.0
        result = {
            "series": series,
            "summary": {
                "final_adoption_rate": final_rate,
                "average_adoption_rate": avg_rate,
                "peak_adoption_rate": peak_val,
                "peak_adoption_day": peak_day,
            },
            "metadata": self.metadata,
        }
        return result

    def save_results(self, filename):
        """
        Save daily results to a CSV file.

        Args:
            filename (str): Path to the CSV file to write.

        Returns:
            str: The filename written.
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        series = self.aggregator.results()
        days = len(series.get("overall_adoption_rate_over_time", []))
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("day,overall_adoption,new_adoptions,churn,campaign_intensity\n")
                for d in range(days):
                    overall = series["overall_adoption_rate_over_time"][d]
                    new_adopt = series["new_adoptions_daily"][d]
                    churn = series["adoption_churn_daily"][d]
                    camp = series["campaign_intensity_daily"][d]
                    f.write(f"{d},{overall:.6f},{new_adopt:.6f},{churn:.6f},{camp:.6f}\n")
            return filename
        except Exception as e:
            logging.error(f"Failed to save results to {filename}: {e}")
            return filename

    def visualize(self, filename=None, show=False):
        """
        Create a simple visualization of the overall adoption rate over time.

        Args:
            filename (str or None): If provided, save plot to this file.
            show (bool): If True, display the plot interactively (may not work in headless environments).

        Returns:
            None
        """
        pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; method logic follows.
        # FIXED: Use non-interactive Agg backend and ensure warnings go to stderr via logging only.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            logging.warning(f"Visualization skipped (matplotlib not available): {e}")
            return

        series = self.aggregator.results()
        overall = series.get("overall_adoption_rate_over_time", [])
        if not overall:
            logging.info("No data to visualize.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(range(len(overall)), overall, label="Overall adoption rate")
        plt.xlabel("Day")
        plt.ylabel("Adoption rate")
        plt.title("Adoption over time")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(loc="best")
        if filename:
            try:
                plt.savefig(filename, bbox_inches="tight")
            except Exception as e:
                logging.warning(f"Failed to save plot to {filename}: {e}")
        if show:
            try:
                plt.show()
            except Exception as e:
                logging.warning(f"Failed to display plot: {e}")
        plt.close()


def create_simulation(cfg):
    """
    Factory to create a Simulation instance from a config dictionary.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        Simulation: Initialized simulation instance.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    sim = Simulation(cfg)
    return sim


def evaluate(sim, cfg):
    """
    Evaluate the simulation according to dynamic configuration.

    Args:
        sim (Simulation): Completed simulation instance.
        cfg (dict): Configuration including 'evaluation_metrics' and possibly target data.

    Returns:
        dict: Dictionary of evaluation metric results.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    results = {}
    # FIXED: Support evaluation metrics from either top-level cfg or cfg['parameters'].
    eval_list = cfg.get("evaluation_metrics") or cfg.get("parameters", {}).get("evaluation_metrics", [])
    series = sim.aggregator.results()
    # Placeholders for external target data usage; not provided, so we compute simple self-based metrics
    for metric in eval_list:
        if metric == "TimeToPeak_overall":
            overall = series.get("overall_adoption_rate_over_time", [])
            if overall:
                peak_val = max(overall)
                peak_day = overall.index(peak_val)
                results[metric] = {"time_to_peak": peak_day, "peak_value": peak_val}
            else:
                results[metric] = {"time_to_peak": None, "peak_value": 0.0}
        elif metric == "PeakError_overall":
            # Without target, report simulated peak
            overall = series.get("overall_adoption_rate_over_time", [])
            peak_val = max(overall) if overall else 0.0
            results[metric] = {"simulated_peak": peak_val, "note": "No target provided"}
        else:
            # Unsupported without target
            results[metric] = {"supported": False, "reason": "Target data not provided"}
    return results


def parse_args(argv=None):
    """
    Parse command-line arguments for the simulation runner.

    Args:
        argv (list[str] or None): Argument vector.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    ap = argparse.ArgumentParser(description="Mask adoption simulation runner")
    ap.add_argument("--input", "-i", help="Path to JSON config (default: stdin if piped, else defaults)")
    ap.add_argument("--output", "-o", help="Path to write JSON results (default: stdout)")
    ap.add_argument("--smoke", action="store_true", help="Smoke test: emit status ok and exit")
    ap.add_argument("--use-docker", action="store_true", help="Optional: require Docker for specific paths")
    ap.add_argument("--save-csv", default="results.csv", help="Path to save CSV of daily results (default: results.csv)")
    ap.add_argument("--save-plot", default="results.png", help="Path to save plot image (default: results.png)")
    return ap.parse_args(argv)


def have_docker():
    """
    Check whether Docker is available on PATH.

    Returns:
        bool: True if docker executable is found, False otherwise.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    return shutil.which("docker") is not None


def read_config_from_stdin():
    """
    Attempt to read JSON configuration from stdin if data is available.

    Returns:
        dict: Parsed configuration or empty dict if none available.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    try:
        if sys.stdin is not None and not sys.stdin.isatty():
            data = sys.stdin.read()
            if data.strip() == "":
                return {}
            return json.loads(data)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Invalid JSON input from stdin: {e.msg} at line {e.lineno} column {e.colno}\n")
        return {}
    except Exception as e:
        sys.stderr.write(f"Error reading stdin: {e}\n")
        return {}
    return {}


def run_simulation(cfg):
    """
    Run the simulation using the provided configuration and return results.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        dict: Result dictionary suitable for JSON serialization.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    # FIXED: Reintegrated a functional simulation engine per feedback.
    sim = create_simulation(cfg)
    sim.run()
    result = sim.get_result_json()
    # Attach evaluation results if requested
    result["evaluation"] = evaluate(sim, cfg)
    return result


def main(argv=None):
    """
    Main entry point: parse CLI, handle smoke-mode, read config, run simulation, and emit JSON.

    Notes:
        - Unconditionally called at end of file to satisfy sandbox requirement.
        - Writes JSON to stdout or a file specified by --output.
        - Saves CSV and plot as a demonstration.
    """
    pass  # NOTE: 'pass' retained to satisfy strict syntactic requirement; function logic follows.
    # FIXED: Implemented real CLI with argparse, JSON parsing, smoke-mode, and Docker gating.
    # FIXED: Logging directed to stderr to avoid stdout contamination for JSON consumers.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    args = parse_args(argv)

    if args.smoke:
        # FIXED: Provide a smoke test path that requires no external dependencies.
        out = {"status": "ok"}
        s = json.dumps(out)
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(s)
            except Exception as e:
                sys.stderr.write(f"Failed to write smoke output file: {e}\n")
                sys.exit(1)
        else:
            sys.stdout.write(s)
        return

    if args.use_docker and not have_docker():
        # FIXED: Docker gating with explicit error and exit code 2 as requested.
        sys.stderr.write("Docker requested but not found in PATH.\n")
        sys.exit(2)

    cfg = {}
    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Invalid JSON input: {e.msg} at line {e.lineno} column {e.colno}\n")
            sys.exit(1)
        except FileNotFoundError:
            sys.stderr.write(f"Input file not found: {args.input}\n")
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"Error reading input file: {e}\n")
            sys.exit(1)
    else:
        # FIXED: Avoid blocking on stdin by using defaults when no input is piped.
        cfg = read_config_from_stdin()

    # Run simulation
    sim = create_simulation(cfg)
    sim.run()
    result = sim.get_result_json()
    result["evaluation"] = evaluate(sim, cfg)

    # Output JSON via json.dumps to avoid malformed output
    out_str = json.dumps(result)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_str)
        except Exception as e:
            sys.stderr.write(f"Failed to write output JSON: {e}\n")
            sys.exit(1)
    else:
        # Only JSON written to stdout to prevent contamination
        sys.stdout.write(out_str)

    # Demonstration: save CSV and plot (any messages/warnings go to stderr via logging)
    try:
        sim.save_results(args.save_csv)
    except Exception as e:
        logging.warning(f"Failed to save CSV results: {e}")
    try:
        sim.visualize(filename=args.save_plot, show=False)
    except Exception as e:
        logging.warning(f"Failed to generate visualization: {e}")


# Execute main for both direct execution and sandbox wrapper invocation
main()