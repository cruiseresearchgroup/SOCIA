import json
import math
import os
import statistics
import sys
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# Path handling constants
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else ""


def main():
    """
    Entry point to demonstrate simulation execution, visualization, and saving results.

    This function creates a default parameter set, runs the simulation for a short period,
    visualizes results if matplotlib is available, and saves results to a CSV file.
    """
    # Default demo parameters
    params = {
        "population_size": 1000,
        "initial_adoption_rate": 0.1,
        "avg_degree": 10,
        "network_type": "scale_free",  # FIXED: Honor network_type; supports 'scale_free' now
        "homophily_strength": 0.3,     # FIXED: Enable homophily in scale-free attachment
        "mandate_enabled": True,
        "mandate_start_day": 15,
        "mandate_end_day": 90,
        "penalty_amount": 50.0,
        "enforcement_level": 0.4,
        "communication_frequency": 0.6,
        "message_strategy": 0.6,
        "campaign_intensity": 0.5,
        "retailer_count": 8,
        "initial_total_stock": 5000,
        "production_rate_per_day": 200,
        "distribution_delay_days": 2,
        "mask_price": 1.5,
        "rationing_policy": "price",
        "retailer_min_markup": 0.1,
        "retailer_max_markup": 0.5,
        "restock_rate_per_day": 0.1,
        "prevalence_series": [0.1] * 180,
        "case_rate_series": [0.02 + 0.02 * math.sin(2 * math.pi * d / 60.0) for d in range(180)],  # example
        "threshold_cases_for_policy": 0.03,  # FIXED: Policy threshold on epidemic case rate
        "political_identity_weights": [0.35, 0.30, 0.35],
        "prediction_start_day": 0,
        "prediction_end_day": 59,
        "initial_infected_fraction": 0.01,  # initialize small infection prevalence
    }
    sim = Simulation(params, seed=42)
    results = sim.run(60)
    try:
        sim.visualize()
    except Exception as e:
        print("Visualization skipped or failed:", e)
    out_file = os.path.join(DATA_DIR if DATA_DIR else ".", "results.csv")
    sim.save_results(out_file)
    print("Saved results to:", out_file)
    # Demonstrate validation harness
    val = sim.validate(num_runs=3, days=60)
    print("Validation summary:", json.dumps(val, indent=2))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp a number to [lo, hi].

    Parameters
    ----------
    x : float
        Input value to clamp.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    float
        Clamped value.
    """
    return max(lo, min(hi, x))


def gini(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.

    Parameters
    ----------
    values : List[float]
        The list of values.

    Returns
    -------
    float
        Gini coefficient in [0,1], or 0 for empty or constant lists.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(values)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_values, start=1):
        cumulative += v
        weighted_sum += i * v
    if cumulative == 0:
        return 0.0
    return (2 * weighted_sum) / (n * cumulative) - (n + 1) / n


def poisson_sample(lam: float, rng) -> int:
    """
    Sample a Poisson-distributed non-negative integer using Knuth's algorithm.

    Parameters
    ----------
    lam : float
        Poisson rate parameter lambda (>0).
    rng : random.Random
        RNG instance.

    Returns
    -------
    int
        Poisson random variate.
    """
    if lam <= 0.0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            break
    return max(0, k - 1)


@dataclass
class EpidemicContext:
    """
    Epidemic context providing daily case rate and a derived public risk signal.

    Attributes
    ----------
    current_case_rate : float
        Current case rate (e.g., daily new cases per capita).
    case_rate_series : List[float]
        Exogenous sequence of case rates by day.
    public_risk_signal : float
        Normalized public risk signal in [0,1] derived from current_case_rate.

    Methods
    -------
    update_case_rate(day)
        Set current_case_rate from series for the given day.
    update_risk_signal()
        Update public_risk_signal as a clamped transformation of current_case_rate.
    """
    current_case_rate: float = 0.0
    case_rate_series: List[float] = field(default_factory=list)
    public_risk_signal: float = 0.0

    def update_case_rate(self, day: int) -> None:
        """
        Update current case rate from the series for the given day.

        Parameters
        ----------
        day : int
            Day index to pull from case_rate_series.
        """
        if 0 <= day < len(self.case_rate_series):
            self.current_case_rate = float(self.case_rate_series[day])
        elif self.case_rate_series:
            self.current_case_rate = float(self.case_rate_series[-1])
        else:
            self.current_case_rate = 0.0

    def update_risk_signal(self) -> None:
        """
        Update the public risk signal based on current case rate.

        The transformation is a simple clamp; users may replace with a more
        sophisticated mapping from case rates to perceived public risk.
        """
        # Normalize assuming a soft max case rate of ~0.05/day for clamping
        self.public_risk_signal = clamp(self.current_case_rate / 0.05, 0.0, 1.0)


@dataclass
class SocialNetwork:
    """
    SocialNetwork encapsulates network generation and optional influence propagation routines.

    Attributes
    ----------
    topology : str
        Name of topology, e.g., 'small_world' or 'scale_free'.
    average_degree : int
        Average degree parameter for topology.
    clustering_coefficient : float
        Desired clustering coefficient hint (not all topologies use it).
    rewiring_prob : float
        Rewiring probability for small-world networks.
    is_dynamic : bool
        Whether the network changes over time.
    rng : random.Random
        RNG for network generation.
    homophily_strength : float
        Probability to bias attachment toward same-label nodes in scale-free generation.
    homophily_labels : Optional[List[int]]
        Labels for homophily grouping; used if provided.
    adj : List[List[int]]
        Adjacency list result.

    Methods
    -------
    connect_agents(N)
        Return adjacency list for N agents.
    _small_world(N, k, beta)
        Generate a Watts–Strogatz small-world adjacency.
    _scale_free(N, m)
        Generate Barabási–Albert scale-free network with optional homophily bias.
    propagate_social_influence(...)
        Optional helper to aggregate peer influence.
    """
    topology: str
    average_degree: int
    clustering_coefficient: float
    rewiring_prob: float
    is_dynamic: bool
    rng: Any
    homophily_strength: float = 0.0  # FIXED: Add homophily_strength attribute
    homophily_labels: Optional[List[int]] = None  # FIXED: Add homophily_labels attribute
    adj: List[List[int]] = field(default_factory=list)

    def connect_agents(self, N: int) -> List[List[int]]:
        """
        Create an adjacency list for N agents according to the topology.

        Parameters
        ----------
        N : int
            Number of agents.

        Returns
        -------
        List[List[int]]
            Adjacency list of neighbors for each agent.
        """
        # FIXED: Honor network_type 'scale_free' and add homophily
        if self.topology == "scale_free":
            m = max(1, self.average_degree // 2)
            self.adj = self._scale_free(N, m)
        else:
            # Default to small-world if unsupported or explicitly set
            self.adj = self._small_world(N, self.average_degree, self.rewiring_prob)
        return self.adj

    def _small_world(self, N: int, k: int, beta: float) -> List[List[int]]:
        """
        Internal generator for Watts–Strogatz small-world networks.
        """
        k = max(2, k)
        if k % 2 == 1:
            k += 1
        half = k // 2
        adj = [set() for _ in range(N)]
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                adj[i].add(j)
                adj[j].add(i)
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                if self.rng.random() < beta:
                    attempts = 0
                    k2 = j
                    while attempts < 10:
                        cand = self.rng.randrange(N)
                        if cand != i and cand not in adj[i]:
                            k2 = cand
                            break
                        attempts += 1
                    if k2 != j:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        adj[i].add(k2)
                        adj[k2].add(i)
        return [list(nei) for nei in adj]

    def _scale_free(self, N: int, m: int) -> List[List[int]]:
        """
        Internal generator for a Barabási–Albert scale-free network with optional homophily.
        """
        adj = [set() for _ in range(N)]
        # Seed a small clique of size m+1
        seed_size = min(N, m + 1)
        for i in range(seed_size):
            for j in range(i + 1, seed_size):
                adj[i].add(j)
                adj[j].add(i)
        targets = [j for i in range(seed_size) for j in adj[i]]
        if not targets:
            targets = list(range(seed_size))
        for i in range(seed_size, N):
            chosen = set()
            attempts = 0
            while len(chosen) < m and attempts < 50 * m:
                attempts += 1
                cand = self.rng.choice(targets) if targets else self.rng.randrange(i)
                # FIXED: Homophily bias
                if self.homophily_labels is not None and self.rng.random() < self.homophily_strength:
                    my_label = self.homophily_labels[i % len(self.homophily_labels)]
                    same = [t for t in targets if self.homophily_labels[t % len(self.homophily_labels)] == my_label]
                    if same:
                        cand = self.rng.choice(same)
                if cand != i and cand not in chosen:
                    chosen.add(cand)
            for c in chosen:
                adj[i].add(c)
                adj[c].add(i)
            # Preferential attachment: add to targets
            targets.extend(list(chosen))
            targets.extend([i] * len(chosen))
        return [list(nei) for nei in adj]

    def propagate_social_influence(self, adopted_prev: List[float], contact_rate_per_day: int, neighbors: List[int]) -> float:
        """
        Compute peer share using sampled neighbors.
        """
        if not neighbors:
            return 0.0
        k = min(contact_rate_per_day, len(neighbors))
        if k <= 0:
            return 0.0
        idxs = self.rng.sample(neighbors, k)
        return sum(adopted_prev[j] for j in idxs) / max(1, len(idxs))


@dataclass
class Household:
    """
    Household entity representing co-residing individuals and intra-household influence.
    """
    id: int
    member_ids: List[int] = field(default_factory=list)
    household_norm_mask_use: float = 0.0
    socioeconomic_status: float = 0.5
    intra_household_influence_strength: float = 0.6

    def update_norm(self, adopted_prev: List[float]) -> None:
        """
        Update household normative mask use based on previous-day adoption of members.
        """
        if not self.member_ids:
            self.household_norm_mask_use = 0.0
            return
        vals = [adopted_prev[i] for i in self.member_ids]
        self.household_norm_mask_use = sum(vals) / max(1, len(vals))


@dataclass
class Person:
    """
    Person agent representing an individual in the simulation with attributes related to mask adoption behavior.
    """
    id: int
    age: int
    income: float
    household_id: int
    workplace_id: int
    network_neighbors: List[int] = field(default_factory=list)
    trust_in_authority: float = 0.6
    susceptibility_to_peer_influence: float = 0.5
    risk_perception: float = 0.0
    perceived_mask_benefit: float = 0.4
    perceived_mask_cost: float = 0.2
    mask_inventory: int = 0
    mask_adopted: bool = False
    current_mask_use: bool = False
    habit_strength: float = 0.0
    compliance_propensity: float = 0.5
    political_identity: int = 0  # FIXED: Add political_identity attribute
    budget: float = 0.0          # FIXED: Add budget attribute
    fatigue: float = 0.0         # FIXED: Add fatigue attribute
    exposure_to_messages: float = 0.0  # FIXED: Track exposure to information
    education_level: int = 1
    exposure_to_misinformation: float = 0.2
    days_worn: int = 0
    # FIXED: Add epidemiology and mask type differentiation and SES
    health_status: str = "S"  # 'S','I','R'
    days_infected: int = 0
    mask_type: str = "none"  # 'none','cloth','surgical','N95'
    ses: float = 0.5

    def reset_daily_state(self) -> None:
        """
        Reset transient state for the day.
        """
        self.current_mask_use = False

    def perceive_risk(
        self,
        peer_share: float,
        policy_signal: float,
        media_signal: float,
        prevalence_signal: float,
        risk_perception_sensitivity_to_prevalence: float,
        external_prevalence_signal: float,
        w_peer: float,
        w_policy: float,
        w_media: float,
        household_share: float = 0.0,
        w_household: float = 0.6,
        w_prevalence: float = 0.3,
    ) -> None:
        """
        Update risk perception using weighted signals and prevalence sensitivity.
        """
        media_component = 0.5 * (media_signal + 1.0)
        # Exposure to messages increases with positive media/policy
        self.exposure_to_messages = clamp(self.exposure_to_messages + 0.2 * media_component + 0.1 * policy_signal)
        prevalence_component = clamp(
            risk_perception_sensitivity_to_prevalence * (0.5 * prevalence_signal + 0.5 * external_prevalence_signal)
        )
        signal = (
            w_peer * peer_share * (1.0 + 0.1 * self.susceptibility_to_peer_influence)
            + w_household * household_share
            + w_policy * policy_signal * self.trust_in_authority
            + w_media * media_component * (1.0 - self.exposure_to_misinformation)
            + w_prevalence * prevalence_component
        )
        signal = clamp(signal, 0.0, 1.0)
        inertia = 0.7
        self.risk_perception = clamp(inertia * self.risk_perception + (1 - inertia) * signal)

    def update_attitude(
        self,
        habit_formation_rate: float,
        compliance_decay_rate: float,
        mask_effectiveness_perceived: float,
    ) -> None:
        """
        Update perceived benefits/costs and habit strength.
        """
        self.perceived_mask_benefit = clamp(
            0.4 * self.perceived_mask_benefit + 0.6 * clamp(self.risk_perception * mask_effectiveness_perceived)
        )
        self.perceived_mask_cost = clamp(
            0.7 * self.perceived_mask_cost + 0.3 * (1.0 - self.habit_strength)
        )
        # Fatigue increases slightly if not wearing, resets slightly when wearing
        if self.mask_adopted or self.current_mask_use:
            self.fatigue = clamp(self.fatigue * 0.9)
        else:
            self.fatigue = clamp(self.fatigue + compliance_decay_rate)

    def decide_adoption(
        self,
        price: float,
        policy_active: bool,
        rng,
        enforcement_level: float = 0.0,
        penalty_amount: float = 0.0,
        habit_formation_rate: float = 0.02,
        compliance_decay_rate: float = 0.01,
        freeze_adoption: bool = False,
    ) -> bool:
        """
        Decide whether to wear a mask today.
        """
        if freeze_adoption:
            return self.mask_adopted

        policy_pressure = clamp(enforcement_level * self.compliance_propensity * min(1.0, penalty_amount / 100.0))
        # Simple linear utility with fatigue penalty
        benefit_term = self.perceived_mask_benefit
        cost_term = self.perceived_mask_cost + min(0.1, price / 20.0) + 0.2 * self.fatigue
        habit_term = self.habit_strength
        policy_term = (0.5 if policy_active else 0.0) * policy_pressure
        linear_util = benefit_term - cost_term + habit_term + policy_term
        p_wear = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, linear_util))))
        will_wear = rng.random() < p_wear
        if will_wear and self.mask_inventory > 0:
            self.mask_adopted = True
            self.mask_inventory -= 1
            self.habit_strength = clamp(self.habit_strength + habit_formation_rate, 0.0, 1.0)
            return True
        else:
            self.mask_adopted = False
            self.habit_strength = clamp(self.habit_strength * (1.0 - compliance_decay_rate), 0.0, 1.0)
            return False

    def purchase_masks(
        self,
        rng,
        price: float,
        bundle: int,
        subsidy_rate: float,
        mandate_active: bool,
        procurement_access_fraction: float,
    ) -> int:
        """
        Decide how many masks to purchase given affordability, subsidy, and access.
        """
        if rng.random() > procurement_access_fraction:
            return 0
        effective_price = max(0.0, price * (1.0 - subsidy_rate))
        affordability = (self.income + self.budget) / (self.income + self.budget + 10.0 * effective_price + 1e-6)
        # FIXED: SES improves procurement access; scale intent slightly by SES
        intent = 0.3 + 0.5 * affordability + (0.2 if mandate_active else 0.0) + 0.1 * (self.ses - 0.5)
        intent = clamp(intent)
        if self.mask_inventory > 0 and not mandate_active and rng.random() > 0.25:
            return 0
        if rng.random() < intent:
            need = 1 if self.mask_inventory == 0 else 0
            qty = max(need, bundle if affordability > 0.6 else 1)
            return qty
        return 0

    def comply_with_policy(self, enforcement_prob: float, signage_strength: float, rng) -> bool:
        """
        Determine if the person complies with a mask requirement at a location.
        """
        base = clamp(0.5 * self.compliance_propensity + 0.3 * self.trust_in_authority + 0.2 * signage_strength)
        adjusted = clamp(base + 0.4 * enforcement_prob * self.trust_in_authority)
        return rng.random() < adjusted


@dataclass
class Location:
    """
    Location where interactions and policy enforcement may occur.
    """
    id: int
    type: str
    capacity: int
    policy_requires_mask: bool
    enforcement_strictness: float
    signage_strength: float
    foot_traffic_rate: float
    policy_eligible: bool = True
    contact_rate_modifier: float = 1.0

    def enforce_mask_policy(self, person: Person, agency_enforcement: float, rng) -> Tuple[bool, bool, bool]:
        """
        Enforce mask policy with certain probability.
        """
        if not self.policy_requires_mask:
            return (False, person.mask_adopted or person.current_mask_use, False)

        if person.mask_adopted or person.current_mask_use:
            return (False, True, False)

        will_comply = person.comply_with_policy(
            enforcement_prob=clamp(self.enforcement_strictness * agency_enforcement),
            signage_strength=self.signage_strength,
            rng=rng,
        )

        incident = False
        denied = False
        if will_comply:
            if person.mask_inventory > 0:
                person.mask_inventory -= 1
                person.current_mask_use = True
                return (False, True, False)
            else:
                will_comply = False  # Cannot comply if no inventory

        if not will_comply:
            check_prob = clamp(self.enforcement_strictness * agency_enforcement)
            if rng.random() < check_prob:
                incident = True
                denied = True  # FIXED: Deny entry if checked and non-compliant
            else:
                # Soft denial due to signage/strictness even without incident
                if rng.random() < 0.2 * self.signage_strength:
                    denied = True

        return (incident, False, denied)


@dataclass
class PolicyAuthority:
    """
    Policy authority controlling mandates, enforcement, and communications.
    """
    id: int
    mandate_enabled: bool
    mandate_start_day: int
    mandate_end_day: Optional[int]
    penalty_amount: float
    incentive_amount: float
    enforcement_level: float
    communication_frequency: float
    message_strategy: float
    subsidy_rate: float
    enforcement_capacity_per_day: int = 0
    free_mask_distribution_rate: int = 0
    campaign_intensity: float = 0.0
    enforcement_cost_per_incident: float = 0.0
    campaign_cost_per_day: float = 0.0
    threshold_cases_for_policy: float = 0.0  # FIXED: Add thresholding on case rate

    def issue_mandates(self, day: int, epidemic: Optional[EpidemicContext] = None) -> bool:
        """
        Determine if mandates are active on the given day, optionally conditioned on epidemic case rates.
        """
        if not self.mandate_enabled:
            return False
        threshold_ok = True
        if epidemic is not None and self.threshold_cases_for_policy > 0.0:
            threshold_ok = epidemic.current_case_rate >= self.threshold_cases_for_policy
        if self.mandate_end_day is None:
            return (day >= self.mandate_start_day) and threshold_ok
        return (self.mandate_start_day <= day <= self.mandate_end_day) and threshold_ok

    def run_information_campaign(self, rng) -> float:
        """
        Run a campaign information broadcast.
        """
        if rng.random() < clamp(self.communication_frequency * max(0.0, self.campaign_intensity)):
            return clamp(self.message_strategy)
        return 0.0

    def adjust_enforcement(self, day: int, epidemic: Optional[EpidemicContext] = None) -> float:
        """
        Adjust enforcement level (e.g., stronger during mandates), conditioned on active mandate.
        """
        # FIXED: Depend on active mandate state rather than static flag
        active = self.issue_mandates(day, epidemic)
        return clamp(self.enforcement_level * (1.2 if active else 1.0))


@dataclass
class SupplyChain:
    """
    Supply chain for mask production, distribution, and pricing.
    """
    total_stock: int
    production_rate_per_day: int
    distribution_delay_days: int
    price_per_mask: float
    rationing_policy: str = "price"
    min_price: float = 0.5
    max_price: float = 50.0
    _pipeline: List[int] = field(default_factory=list)
    cumulative_produced: int = 0
    cumulative_distributed: int = 0

    def __post_init__(self) -> None:
        """
        Initialize the distribution pipeline.
        """
        self._pipeline = [0] * max(0, int(self.distribution_delay_days))

    def produce_masks(self) -> None:
        """
        Produce masks and progress them through the distribution pipeline.
        """
        self._pipeline.append(int(self.production_rate_per_day))
        shipped = self._pipeline.pop(0) if self._pipeline else int(self.production_rate_per_day)
        self.total_stock += shipped
        self.cumulative_produced += shipped

    def distribute_masks(self, demand: int) -> int:
        """
        Distribute masks to meet demand, subject to stock.
        """
        sold = min(int(demand), self.total_stock)
        self.total_stock -= sold
        self.cumulative_distributed += sold
        return sold

    def adjust_prices(self, stockout: bool) -> None:
        """
        Adjust price based on stockout status under rationing.
        """
        if self.rationing_policy == "price":
            if stockout:
                self.price_per_mask = clamp(self.price_per_mask * 1.1, self.min_price, self.max_price)
            else:
                self.price_per_mask = clamp(self.price_per_mask * 0.98, self.min_price, self.max_price)


@dataclass
class Retailer:
    """
    Retailer entity selling masks with its own inventory, price, and rationing.
    """
    id: int
    inventory_level: int
    restock_rate_per_day: float
    price: float
    rationing_policy: str = "limit"
    rationing_limit_per_purchase: int = 5
    min_price: float = 0.5
    max_price: float = 50.0
    demand_yesterday: int = 0
    sold_yesterday: int = 0
    _initial_inventory: int = 0

    def __post_init__(self) -> None:
        """
        Post-initialize to set initial inventory reference for restocking.
        """
        if self._initial_inventory == 0:
            self._initial_inventory = max(1, self.inventory_level)

    def begin_day(self) -> None:
        """
        Reset daily demand and sold counters at the start of the day.
        """
        self.demand_yesterday = 0
        self.sold_yesterday = 0

    def restock_from_supply(self, supply_chain: SupplyChain) -> int:
        """
        Restock inventory by pulling from a central supply chain.
        """
        target = int(self.restock_rate_per_day * self._initial_inventory)
        if target <= 0:
            return 0
        pulled = supply_chain.distribute_masks(target)
        self.inventory_level += pulled
        return pulled

    def sell(self, requested_qty: int) -> int:
        """
        Sell masks to a buyer subject to inventory and rationing.
        """
        self.demand_yesterday += max(0, int(requested_qty))
        limit = self.rationing_limit_per_purchase if self.rationing_policy == "limit" else requested_qty
        allowed = min(limit, requested_qty)
        sold = min(self.inventory_level, max(0, int(allowed)))
        self.inventory_level -= sold
        self.sold_yesterday += sold
        return sold

    def update_price(self, sensitivity: float) -> None:
        """
        Adjust price based on excess demand.
        """
        base = self.price
        denom = max(1, self.sold_yesterday)
        excess = (self.demand_yesterday - self.sold_yesterday) / float(denom)
        if excess > 0:
            self.price = clamp(base * (1.0 + sensitivity * excess), self.min_price, self.max_price)
        else:
            self.price = clamp(base * (1.0 + sensitivity * excess * 0.5), self.min_price, self.max_price)


@dataclass
class InformationChannel:
    """
    Information channel representing sources like government or social media.
    """
    id: int
    reach_fraction: float
    message_type: str
    reliability: float
    misinformation_rate: float

    def broadcast(self, rng) -> float:
        """
        Broadcast a message with possible misinformation.
        """
        base = 1.0 if self.message_type == "government" else 0.2
        val = base * self.reliability
        if rng.random() < self.misinformation_rate:
            val = -val
        return clamp(val, -1.0, 1.0) * clamp(self.reach_fraction, 0.0, 1.0)


@dataclass
class MediaChannel:
    """
    Media channel broadcasting messages that can support or undermine mask adoption.
    """
    id: int
    reach: float
    message_frequency: float
    bias: float
    misinformation_probability: float

    def broadcast_message(self, rng) -> float:
        """
        Broadcast a message according to frequency and misinformation probability.
        """
        if rng.random() >= self.message_frequency:
            return 0.0
        sign = self.bias
        if rng.random() < self.misinformation_probability:
            sign = -sign
        return clamp(sign, -1.0, 1.0) * clamp(self.reach, 0.0, 1.0)


class Simulation:
    """
    Main simulation orchestrator for mask adoption dynamics.
    """
    def __init__(self, params: Dict[str, Any], seed: int = 42):
        """
        Initialize the simulation with parameters and seed.
        """
        self.p = dict(params)
        # FIXED: Parameter alias mapping for spec conformance and usability
        if "fine_amount" in self.p and "penalty_amount" not in self.p:
            self.p["penalty_amount"] = self.p["fine_amount"]
        if "mask_cost" in self.p and "mask_price" not in self.p:
            self.p["mask_price"] = self.p["mask_cost"]
        if "trust_in_authority_mean" in self.p and "trust_in_authorities_mean" not in self.p:
            self.p["trust_in_authorities_mean"] = self.p["trust_in_authority_mean"]
        if "household_size_distribution" in self.p and isinstance(self.p["household_size_distribution"], dict):
            if "lambda" in self.p["household_size_distribution"]:
                self.p["household_size_lambda"] = self.p["household_size_distribution"]["lambda"]
        if "location_policy_mask_required_fraction" in self.p and "venue_policy_coverage" not in self.p:
            self.p["venue_policy_coverage"] = self.p["location_policy_mask_required_fraction"]
        if "average_degree" in self.p and "avg_degree" not in self.p:
            self.p["avg_degree"] = int(self.p["average_degree"])
        if "enforcement_probability" in self.p and "enforcement_level" not in self.p:
            self.p["enforcement_level"] = float(self.p["enforcement_probability"])
        if "enforcement_strength" in self.p and "enforcement_level" not in self.p:
            self.p["enforcement_level"] = float(self.p["enforcement_strength"])  # FIXED: Accept alias
        if "supply_capacity_per_day" in self.p and "production_rate_per_day" not in self.p:
            self.p["production_rate_per_day"] = int(self.p["supply_capacity_per_day"])  # FIXED: Alias
        if "stockpile_initial" in self.p and "initial_total_stock" not in self.p:
            self.p["initial_total_stock"] = int(self.p["stockpile_initial"])  # FIXED: Alias
        if "network_topology" in self.p and "network_type" not in self.p:
            self.p["network_type"] = self.p["network_topology"]  # FIXED: Alias
        if "threshold_cases_for_policy" not in self.p:
            self.p["threshold_cases_for_policy"] = 0.0  # default threshold

        self.rng = __import__("random").Random(int(seed))

        # Entities
        self.people: List[Person] = []
        self.households: List[Household] = []
        self.locations: List[Location] = []
        self.workplaces: List[Location] = []
        self.retailers: List[Retailer] = []
        self.social_network: Optional[SocialNetwork] = None

        # SupplyChain
        initial_stock_guess = int(
            self.p.get("population_size", 10000)
            * self.p.get("mask_supply_per_capita", 5.0)
            * self.p.get("supplier_initial_inventory_ratio", 1.0)
        )
        self.initial_total_stock = int(self.p.get("initial_total_stock", initial_stock_guess))
        self.supply_chain = SupplyChain(
            total_stock=self.initial_total_stock,
            production_rate_per_day=int(self.p.get("production_rate_per_day", 500)),
            distribution_delay_days=int(self.p.get("distribution_delay_days", 2)),
            price_per_mask=float(self.p.get("mask_price", 2.0)),
            rationing_policy=str(self.p.get("rationing_policy", "price")),
            min_price=float(self.p.get("min_mask_price", 0.5)),
            max_price=float(self.p.get("max_mask_price", 50.0)),
        )

        # Policy
        mend = self.p.get("mandate_end_day", 120)
        mend_opt = None if mend is None else int(mend)
        self.policy = PolicyAuthority(
            id=1,
            mandate_enabled=bool(self.p.get("mandate_enabled", False)),
            mandate_start_day=int(self.p.get("mandate_start_day", 30)),
            mandate_end_day=mend_opt,
            penalty_amount=float(self.p.get("penalty_amount", 50.0)),
            incentive_amount=float(self.p.get("incentive_amount", 0.0)),
            enforcement_level=float(self.p.get("enforcement_level", 0.5)),
            communication_frequency=float(self.p.get("communication_frequency", 0.5)),
            message_strategy=float(self.p.get("message_strategy", 0.6)),
            subsidy_rate=float(self.p.get("subsidy_rate", 0.0)),
            enforcement_capacity_per_day=int(self.p.get("enforcement_capacity_per_day", 0)),
            free_mask_distribution_rate=int(self.p.get("free_mask_distribution_rate", 0)),
            campaign_intensity=float(self.p.get("campaign_intensity", self.p.get("campaign_intensity", 0.5))),
            enforcement_cost_per_incident=float(self.p.get("enforcement_cost_per_incident", 20.0)),
            campaign_cost_per_day=float(self.p.get("campaign_cost_per_day", 100.0)),
            threshold_cases_for_policy=float(self.p.get("threshold_cases_for_policy", 0.0)),  # FIXED: Set threshold
        )

        # Epidemic context
        self.epidemic = EpidemicContext(
            current_case_rate=0.0,
            case_rate_series=list(self.p.get("case_rate_series", [])),
            public_risk_signal=0.0,
        )  # FIXED: Add epidemic context

        # Media and information channels
        self.media: List[MediaChannel] = [
            MediaChannel(
                id=1,
                reach=float(self.p.get("media_reach_main", 0.7)),
                message_frequency=float(self.p.get("message_frequency_per_week", 3)) / 7.0,
                bias=float(self.p.get("media_bias", 0.0)),
                misinformation_probability=float(self.p.get("misinformation_rate", 0.05)),
            )
        ]
        self.info_channels: List[InformationChannel] = [
            InformationChannel(
                id=1,
                reach_fraction=float(self.p.get("gov_channel_reach", 0.6)),
                message_type="government",
                reliability=float(self.p.get("gov_channel_reliability", 0.8)),
                misinformation_rate=float(self.p.get("gov_channel_misinformation_rate", 0.02)),
            ),
            InformationChannel(
                id=2,
                reach_fraction=float(self.p.get("social_channel_reach", 0.8)),
                message_type="social",
                reliability=float(self.p.get("social_channel_reliability", 0.5)),
                misinformation_rate=float(self.p.get("social_channel_misinformation_rate", 0.15)),
            ),
        ]

        # Series and counters
        self.series: Dict[str, List[float]] = {
            "adoption_rate": [],
            "average_price": [],
            "supplier_inventory": [],
            "retailer_inventory": [],
            "enforcement_incidents_per_1000": [],
            "compliance_rate": [],
            "compliance_rate_under_mandate": [],
            "policy_enforcement_level": [],
            "daily_new_adopters": [],
            "adoption_work": [],
            "adoption_public": [],
            "daily_demand": [],
            "daily_sold": [],
            "compliance_in_mandated_locations_series": [],
            "information_balance_index": [],  # FIXED: Track info balance index per day
            "daily_new_infections": [],       # FIXED: Epidemiology series
            "Rt": [],                          # FIXED: Effective reproduction number
        }
        self.daily_counters: Dict[str, List[float]] = {
            "visits_public": [],
            "incidents_public": [],
            "compliant_public": [],
            "visits_work": [],
            "incidents_work": [],
            "compliant_work": [],
        }

        # Accumulators
        self.cumulative_acquired: int = 0
        self.cumulative_purchased: int = 0
        self.cumulative_free_distributed: int = 0
        self.cumulative_fines_collected: float = 0.0
        self.cumulative_enforcement_cost: float = 0.0
        self.cumulative_campaign_cost: float = 0.0
        self.stockout_retailer_days_accum: int = 0
        self.cumulative_infections: int = 0  # FIXED: Track cumulative infections

        # Observed series for RMSE computation if provided
        self.observed_adoption_series: List[float] = []
        if isinstance(self.p.get("observed_adoption_series", []), list):
            self.observed_adoption_series = list(self.p.get("observed_adoption_series", []))

        # First adoption day per agent for cascade metrics
        self.first_adopt_day: List[Optional[int]] = []

        # Panels for contagion coefficient
        self._panel_peers: List[float] = []  # FIXED: Track peer share for contagion estimation
        self._panel_adopt: List[int] = []    # FIXED: Track adoption outcome per person-day
        self._today_info_pro: float = 0.0
        self._today_info_anti: float = 0.0
        self._days_run: int = 0  # FIXED: track days for equity metrics
        self._last_adopt: List[float] = []

    def _small_world(self, N: int, k: int, beta: float) -> List[List[int]]:
        """
        Create a Watts–Strogatz small-world network adjacency list.
        """
        # Delegate to SocialNetwork when available
        if self.social_network:
            return self.social_network._small_world(N, k, beta)
        # Fallback
        k = max(2, k)
        if k % 2 == 1:
            k += 1
        half = k // 2
        adj = [set() for _ in range(N)]
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                adj[i].add(j)
                adj[j].add(i)
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                if self.rng.random() < beta:
                    attempts = 0
                    k2 = j
                    while attempts < 10:
                        cand = self.rng.randrange(N)
                        if cand != i and cand not in adj[i]:
                            k2 = cand
                            break
                        attempts += 1
                    if k2 != j:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        adj[i].add(k2)
                        adj[k2].add(i)
        return [list(nei) for nei in adj]

    def _build_households_poisson(self, N: int, lam: float) -> List[Household]:
        """
        Build households by sampling household sizes from a Poisson distribution until N agents are allocated.
        """
        sizes: List[int] = []
        remaining = N
        while remaining > 0:
            sz = max(1, poisson_sample(lam, self.rng))
            if sz > remaining:
                sz = remaining
            sizes.append(sz)
            remaining -= sz
        households: List[Household] = []
        start = 0
        for hid, sz in enumerate(sizes):
            member_ids = list(range(start, start + sz))
            households.append(
                Household(
                    id=hid,
                    member_ids=member_ids,
                    household_norm_mask_use=0.0,
                    socioeconomic_status=clamp(self.rng.random()),
                    intra_household_influence_strength=float(self.p.get("household_influence_weight", 0.6)),
                )
            )
            start += sz
        return households

    def initialize(self) -> None:
        """
        Initialize population, households, network, locations, retailers, and supply chain.
        """
        N = int(self.p.get("population_size", 10000))
        init_rate = float(self.p.get("initial_adoption_rate", 0.1))
        if "average_degree" in self.p and "avg_degree" not in self.p:
            self.p["avg_degree"] = int(self.p["average_degree"])
        if "enforcement_probability" in self.p and "enforcement_level" not in self.p:
            self.p["enforcement_level"] = float(self.p["enforcement_probability"])
        avg_deg = int(self.p.get("avg_degree", 10))
        risk_init = float(self.p.get("risk_level", 0.2))

        # Households
        lam = float(self.p.get("household_size_lambda", 3.0))
        self.households = self._build_households_poisson(N, lam)

        # People
        self.people = [None] * N  # type: ignore
        num_workplaces = max(1, int(self.p.get("num_workplaces", 50)))
        pol_weights = self.p.get("political_identity_weights", [0.35, 0.30, 0.35])  # FIXED: Add political distribution
        for i in range(N):
            mu = float(self.p.get("income_lognorm_mu", 3.0))
            sigma = float(self.p.get("income_lognorm_sigma", 0.5))
            income = math.exp(self.rng.normalvariate(mu, sigma))
            adopted = self.rng.random() < init_rate
            education_level = self.rng.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
            pol_id = self.rng.choices([-1, 0, 1], weights=pol_weights, k=1)[0]
            person = Person(
                id=i,
                age=self.rng.randint(18, 85),
                income=income,
                household_id=0,
                workplace_id=i % num_workplaces,
                trust_in_authority=clamp(
                    self.rng.normalvariate(self.p.get("trust_in_authorities_mean", 0.5),
                                           self.p.get("trust_in_authorities_std", 0.2))
                ),
                susceptibility_to_peer_influence=clamp(self.rng.random()),
                risk_perception=clamp(risk_init + self.rng.uniform(-0.05, 0.05)),
                perceived_mask_benefit=float(self.p.get("perceived_benefit_base", 0.4)),
                perceived_mask_cost=float(self.p.get("perceived_cost_base", 0.2)),
                mask_inventory=(1 if adopted else 0),
                mask_adopted=adopted,
                habit_strength=clamp(0.4 + 0.2 * self.rng.random()) if adopted else clamp(0.1 * self.rng.random()),
                compliance_propensity=clamp(self.rng.uniform(0.3, 0.9)),
                education_level=education_level,
                exposure_to_misinformation=clamp(self.rng.uniform(0.1, 0.6)),
                political_identity=pol_id,  # FIXED: Set political identity
                budget=float(self.p.get("budget_per_capita", 200.0)),  # FIXED: Set budget
            )
            self.people[i] = person

        # Assign household IDs and SES
        for hh in self.households:
            for pid in hh.member_ids:
                if 0 <= pid < N:
                    self.people[pid].household_id = hh.id
                    self.people[pid].ses = hh.socioeconomic_status  # FIXED: propagate SES to person

        # Initialize epidemiological states
        initial_infected_fraction = float(self.p.get("initial_infected_fraction", 0.005))
        num_initial_infected = int(initial_infected_fraction * N)
        if num_initial_infected > 0:
            infected_ids = self.rng.sample(range(N), num_initial_infected)
            for idx in infected_ids:
                self.people[idx].health_status = "I"
                self.people[idx].days_infected = self.rng.randint(0, int(self.p.get("infectious_period_days", 7)) - 1)

        # Initialize mask types based on SES/income/risk
        for p in self.people:
            p.mask_type = self._choose_mask_type(p, self.supply_chain.price_per_mask)

        # Social network
        self.social_network = SocialNetwork(
            topology=str(self.p.get("network_type", "small_world")),
            average_degree=avg_deg,
            clustering_coefficient=float(self.p.get("clustering_coefficient", 0.1)),
            rewiring_prob=float(self.p.get("social_network_rewiring_p", 0.05)),
            is_dynamic=bool(self.p.get("network_is_dynamic", False)),
            rng=self.rng,
        )
        # FIXED: Provide homophily params to network generator
        self.social_network.homophily_strength = float(self.p.get("homophily_strength", 0.0))
        self.social_network.homophily_labels = [p.political_identity for p in self.people]
        neighbors = self.social_network.connect_agents(N)
        for i, p in enumerate(self.people):
            p.network_neighbors = neighbors[i]

        # Locations
        venue_policy_coverage = float(self.p.get("venue_policy_coverage", 0.6))
        self.workplaces = []
        for wid in range(num_workplaces):
            eligible = self.rng.random() < venue_policy_coverage
            self.workplaces.append(
                Location(
                    id=wid,
                    type="work",
                    capacity=int(self.p.get("workplace_capacity_mean", 200)),
                    policy_requires_mask=False,
                    enforcement_strictness=float(self.p.get("workplace_enforcement_strictness", 0.5)),
                    signage_strength=float(self.p.get("workplace_signage_effect", self.p.get("signage_effect", 0.05))),
                    foot_traffic_rate=float(self.p.get("workplace_attendance_rate", 0.5)),
                    policy_eligible=eligible,
                    contact_rate_modifier=1.0,
                )
            )
        public_venues_count = max(1, int(self.p.get("public_venues_count", 5)))
        self.locations = []
        for lid in range(public_venues_count):
            eligible = self.rng.random() < venue_policy_coverage
            self.locations.append(
                Location(
                    id=lid,
                    type="public",
                    capacity=int(self.p.get("location_capacity_mean", 2000)),
                    policy_requires_mask=False,
                    enforcement_strictness=float(self.p.get("location_enforcement_strictness_mean", 0.5)),
                    signage_strength=float(self.p.get("signage_effect", 0.05)),
                    foot_traffic_rate=float(self.p.get("public_venue_visit_rate", 0.3)),
                    policy_eligible=eligible,
                    contact_rate_modifier=1.0,
                )
            )

        # Retailers: initialize from supply to avoid double counting
        retailer_count = max(1, int(self.p.get("retailer_count", 10)))
        initial_inventory_per_retailer = int(self.p.get("initial_inventory_per_retailer", 1000))
        restock_rate_per_day = float(self.p.get("restock_rate_per_day", 0.1))
        rationing_limit = int(self.p.get("rationing_limit_per_purchase", 5))
        min_price = float(self.p.get("price_floor", 0.5))
        max_price = float(self.p.get("price_ceiling", 50.0))
        initial_price = float(self.p.get("mask_price", 2.0))
        self.retailers = []
        for r in range(retailer_count):
            pulled = self.supply_chain.distribute_masks(initial_inventory_per_retailer)  # FIXED: pull initial stock
            self.retailers.append(
                Retailer(
                    id=r,
                    inventory_level=pulled,
                    restock_rate_per_day=restock_rate_per_day,
                    price=initial_price,
                    rationing_policy="limit" if bool(self.p.get("supply_rationing", True)) else "none",
                    rationing_limit_per_purchase=rationing_limit,
                    min_price=min_price,
                    max_price=max_price,
                )
            )

        # Prepare first adoption day tracking
        self.first_adopt_day = [0 if (p.mask_adopted or p.current_mask_use) else None for p in self.people]
        self._last_adopt = [1.0 if (p.mask_adopted or p.current_mask_use) else 0.0 for p in self.people]

    def _choose_mask_type(self, person: Person, base_price: float) -> str:
        """
        Choose a mask type for a person based on SES, income, and risk.
        """
        # Simple heuristic: richer and higher risk choose higher efficacy
        score = 0.5 * person.ses + 0.5 * clamp(person.risk_perception)
        if score > 0.7 and person.income > math.exp(3.5):
            return "N95"
        elif score > 0.5:
            return "surgical"
        elif score > 0.3:
            return "cloth"
        return "cloth" if (person.mask_adopted or person.mask_inventory > 0) else "none"

    def _aggregate_media_signal(self) -> float:
        """
        Aggregate media messages into a single signal in [-1,1].
        """
        # FIXED: Track pro vs. anti (misinformation) exposure to compute information_balance_index
        pro = 0.0
        anti = 0.0
        for ch in self.media:
            s = ch.broadcast_message(self.rng)
            if s > 0:
                pro += s
            elif s < 0:
                anti += -s
        for ic in self.info_channels:
            s = ic.broadcast(self.rng)
            if s > 0:
                pro += s
            elif s < 0:
                anti += -s
        self._today_info_pro = clamp(pro, 0.0, 1.0)
        self._today_info_anti = clamp(anti, 0.0, 1.0)
        return clamp(pro - anti, -1.0, 1.0)

    def _peer_share(self, adopted_prev: List[float], neighbors: List[int], contact_rate_per_day: int) -> float:
        """
        Compute the share of peers adopting, based on sampled contacts.
        """
        if not neighbors:
            return 0.0
        k = min(contact_rate_per_day, len(neighbors))
        if k <= 0:
            return 0.0
        idxs = set()
        while len(idxs) < k:
            idxs.add(neighbors[self.rng.randrange(len(neighbors))])
        vals = [adopted_prev[j] for j in idxs]
        return sum(vals) / max(1, len(vals))

    def _adaptive_policy_adjustment(self, day: int) -> None:
        """
        Adaptively adjust policy based on recent adoption and compliance.
        """
        if day < 7:
            return
        recent_adoption = statistics.mean(self.series["adoption_rate"][-7:]) if self.series["adoption_rate"] else 0.0
        recent_compliance = statistics.mean(self.series["compliance_rate"][-7:]) if self.series["compliance_rate"] else 0.0
        target_adoption = float(self.p.get("adoption_target_recent", 0.6))
        target_compliance = float(self.p.get("compliance_target_recent", 0.7))

        if (recent_adoption < target_adoption or recent_compliance < target_compliance) and self.policy.mandate_enabled:
            self.policy.enforcement_level = clamp(self.policy.enforcement_level + 0.05)
            if self.policy.mandate_end_day is not None:
                if day > self.policy.mandate_end_day - 7:
                    self.policy.mandate_end_day += 14
        elif recent_adoption > target_adoption + 0.1 and recent_compliance > target_compliance + 0.1:
            self.policy.enforcement_level = clamp(self.policy.enforcement_level - 0.02)

    def _simulate_infections(self, day: int) -> None:
        """
        Simulate infection transmission using a simplified SIR-like process adjusted by mask efficacy and environment.
        """
        beta = float(self.p.get("base_transmission_rate_beta", 0.045))
        # Location modifiers (ventilation/crowding)
        vent = self.p.get("ventilation_effect_by_location", {"home": 0.0, "work": 0.2, "school": 0.2, "transport": 0.1, "retail": 0.15, "community": 0.1})
        contact = self.p.get("contact_rate_by_location", {"home": 4, "work": 10, "school": 12, "transport": 6, "retail": 8, "community": 3})
        eff_out = self.p.get("mask_efficacy_outward", {"cloth": 0.3, "surgical": 0.5, "N95": 0.9})
        eff_in = self.p.get("mask_efficacy_inward", {"cloth": 0.2, "surgical": 0.4, "N95": 0.8})

        # Gather mask type in use today (if wearing)
        def in_use_mask_type(person: Person) -> str:
            return person.mask_type if (person.mask_adopted or person.current_mask_use) else "none"

        infectious = [p for p in self.people if p.health_status == 'I']
        susceptible = [p for p in self.people if p.health_status == 'S']
        if not infectious or not susceptible:
            self.series.setdefault("Rt", []).append(0.0)
            self.series.setdefault("daily_new_infections", []).append(0)
            return

        # Approximate effective transmission probability per contact as avg over combinations
        def mask_pair_reduction(mt_inf: str, mt_sus: str) -> float:
            o = eff_out.get(mt_inf, 0.0)
            i = eff_in.get(mt_sus, 0.0)
            return (1.0 - o) * (1.0 - i)

        # Estimate share by mask type among infectious and susceptible based on today's wearing
        def type_share(agents: List[Person]) -> Dict[str, float]:
            total = max(1, len(agents))
            counts = {"none": 0, "cloth": 0, "surgical": 0, "N95": 0}
            for a in agents:
                mt = in_use_mask_type(a)
                if mt not in counts:
                    mt = "none"
                counts[mt] += 1
            return {k: v / total for k, v in counts.items()}

        sshare = type_share(susceptible)
        ishare = type_share(infectious)
        avg_mask_factor = 0.0
        for mt_i, pi in ishare.items():
            for mt_s, ps in sshare.items():
                avg_mask_factor += pi * ps * mask_pair_reduction(mt_i, mt_s)

        # Combine across location types with ventilation modifiers
        daily_contacts = sum(contact.values())
        env_factor = 1.0 - statistics.mean([vent.get(k, 0.0) for k in contact.keys()])
        prevalence_I = len(infectious) / max(1, len(self.people))
        expected_infections = int(len(susceptible) * daily_contacts * beta * env_factor * avg_mask_factor * prevalence_I)
        new_inf = min(expected_infections, len(susceptible))

        # Infect random susceptibles
        if new_inf > 0:
            idxs = self.rng.sample([i for i in range(len(self.people)) if self.people[i].health_status == 'S'], new_inf)
            for idx in idxs:
                self.people[idx].health_status = 'I'
                self.people[idx].days_infected = 0

        # Progress disease
        for p in self.people:
            if p.health_status == 'I':
                p.days_infected += 1
                if p.days_infected >= int(self.p.get("infectious_period_days", 7)):
                    p.health_status = 'R'

        # Track series
        self.series.setdefault("daily_new_infections", []).append(new_inf)
        self.cumulative_infections = getattr(self, "cumulative_infections", 0) + new_inf
        # Rt estimation
        serial = int(self.p.get("serial_interval_days", 5))
        if len(self.series["daily_new_infections"]) > serial and self.series["daily_new_infections"][-serial-1] > 0:
            Rt = self.series["daily_new_infections"][-1] / max(1e-6, self.series["daily_new_infections"][-serial-1])
        else:
            Rt = 0.0
        self.series.setdefault("Rt", []).append(Rt)

    def step(self, day: int) -> None:
        """
        Execute one simulation day with simplified logic sufficient for demonstration.
        """
        if not self.people:
            self.initialize()

        # Update epidemic signals
        self.epidemic.update_case_rate(day)
        self.epidemic.update_risk_signal()

        # Policy signals
        policy_active = self.policy.issue_mandates(day, self.epidemic)
        enforcement = self.policy.adjust_enforcement(day, self.epidemic)
        policy_signal = 1.0 if policy_active else 0.0

        # Update locations to reflect policy coverage
        for loc in self.locations:
            loc.policy_requires_mask = policy_active and loc.policy_eligible
        for loc in self.workplaces:
            loc.policy_requires_mask = policy_active and loc.policy_eligible

        # Media/info signals
        media_signal = self._aggregate_media_signal()
        info_balance_index = clamp(self._today_info_pro - self._today_info_anti, -1.0, 1.0)

        # Supply operations
        self.supply_chain.produce_masks()
        for r in self.retailers:
            r.begin_day()
            r.restock_from_supply(self.supply_chain)

        # Household norms based on yesterday
        adopted_prev = self._last_adopt[:] if self._last_adopt else [1.0 if (p.mask_adopted or p.current_mask_use) else 0.0 for p in self.people]
        for hh in self.households:
            hh.update_norm(adopted_prev)

        # Average retail price for decisions
        avg_price = statistics.mean([ret.price for ret in self.retailers]) if self.retailers else self.supply_chain.price_per_mask

        # People decisions
        w_peer = float(self.p.get("weight_peer", 0.4))
        w_policy = float(self.p.get("weight_policy", 0.3))
        w_media = float(self.p.get("weight_media", 0.2))
        w_household = float(self.p.get("weight_household", 0.3))
        w_prev = float(self.p.get("weight_prevalence", 0.3))
        risk_prev_sens = float(self.p.get("risk_perception_sensitivity_to_prevalence", 0.5))
        contact_rate = int(self.p.get("contact_rate_per_day", 6))
        mask_eff_perceived = float(self.p.get("mask_effectiveness_perceived", 0.6))
        habit_rate = float(self.p.get("habit_formation_rate", 0.02))
        decay_rate = float(self.p.get("compliance_decay_rate", 0.01))

        # Purchasing parameters
        bundle_size = int(self.p.get("bundle_size", 5))
        procurement_access = float(self.p.get("procurement_access_fraction", 0.8))

        daily_demand = 0
        daily_sold = 0

        adopted_today: List[float] = [0.0] * len(self.people)
        daily_new_adopters = 0

        for p in self.people:
            p.reset_daily_state()
            # Peer and household signals
            peer_share = self._peer_share(adopted_prev, p.network_neighbors, contact_rate) if self.social_network else 0.0
            hh_share = self.households[p.household_id].household_norm_mask_use if 0 <= p.household_id < len(self.households) else 0.0

            # Perceive risk and update attitudes
            p.perceive_risk(
                peer_share=peer_share,
                policy_signal=policy_signal,
                media_signal=media_signal,
                prevalence_signal=sum(adopted_prev) / max(1, len(adopted_prev)),
                risk_perception_sensitivity_to_prevalence=risk_prev_sens,
                external_prevalence_signal=self.epidemic.public_risk_signal,
                w_peer=w_peer,
                w_policy=w_policy,
                w_media=w_media,
                household_share=hh_share,
                w_household=w_household,
                w_prevalence=w_prev,
            )
            p.update_attitude(habit_formation_rate=habit_rate, compliance_decay_rate=decay_rate, mask_effectiveness_perceived=mask_eff_perceived)

            # Decide adoption
            p.decide_adoption(
                price=avg_price,
                policy_active=policy_active,
                rng=self.rng,
                enforcement_level=enforcement,
                penalty_amount=self.policy.penalty_amount,
                habit_formation_rate=habit_rate,
                compliance_decay_rate=decay_rate,
            )

            # Purchase if needed
            desired = p.purchase_masks(
                rng=self.rng,
                price=avg_price,
                bundle=bundle_size,
                subsidy_rate=self.policy.subsidy_rate,
                mandate_active=policy_active,
                procurement_access_fraction=procurement_access,
            )
            if desired > 0:
                daily_demand += desired
                qty_needed = desired
                # Attempt to buy across retailers
                for ret in self.retailers:
                    if qty_needed <= 0:
                        break
                    sold_here = ret.sell(qty_needed)
                    if sold_here > 0:
                        spend = sold_here * ret.price * (1.0 - self.policy.subsidy_rate)
                        if p.budget >= spend:
                            p.budget -= spend
                            p.mask_inventory += sold_here
                            qty_needed -= sold_here
                            daily_sold += sold_here
                            self.cumulative_purchased += sold_here
                # Unmet demand remains unmet

            # Track adoption today
            if p.mask_adopted or p.current_mask_use:
                adopted_today[p.id] = 1.0
                # First time adopter?
                if self.first_adopt_day[p.id] is None:
                    self.first_adopt_day[p.id] = self._days_run
                    daily_new_adopters += 1

        # Update retailer prices (simple rule)
        price_sensitivity = float(self.p.get("retailer_price_sensitivity", 0.05))
        for ret in self.retailers:
            ret.update_price(price_sensitivity)

        # Supplier price adjustment based on stockouts
        stockout_occurred = daily_sold < daily_demand
        self.supply_chain.adjust_prices(stockout_occurred)

        # Metrics
        adoption_rate = sum(adopted_today) / max(1, len(adopted_today))
        self.series["adoption_rate"].append(adoption_rate)
        self.series["daily_new_adopters"].append(daily_new_adopters)
        self.series["average_price"].append(statistics.mean([ret.price for ret in self.retailers]) if self.retailers else self.supply_chain.price_per_mask)
        self.series["supplier_inventory"].append(self.supply_chain.total_stock)
        self.series["retailer_inventory"].append(sum(r.inventory_level for r in self.retailers))
        self.series["daily_demand"].append(daily_demand)
        self.series["daily_sold"].append(daily_sold)
        self.series["enforcement_incidents_per_1000"].append(0.0)
        self.series["compliance_rate"].append(adoption_rate)
        self.series["compliance_rate_under_mandate"].append(adoption_rate if policy_active else 0.0)
        self.series["policy_enforcement_level"].append(enforcement)
        self.series["adoption_work"].append(adoption_rate)
        self.series["adoption_public"].append(adoption_rate)
        self.series["compliance_in_mandated_locations_series"].append(adoption_rate if policy_active else 0.0)
        self.series["information_balance_index"].append(info_balance_index)

        # Save adopted states for peer effects next day
        self._last_adopt = adopted_today

        # Epidemiology and adaptive policy tweaks
        self._simulate_infections(day)
        self._adaptive_policy_adjustment(day)

    def run(self, days: int) -> Dict[str, Any]:
        """
        Run the simulation for a given number of days.
        """
        # FIXED: Implement Simulation.run() to drive the loop and compute summary metrics
        if not self.people:
            self.initialize()
        self._days_run = 0
        for day in range(days):
            self.step(day)
            self._days_run += 1
        # Summary metrics
        adoption_series = self.series.get("adoption_rate", [])
        time_to_50 = next((i for i, v in enumerate(adoption_series) if v >= 0.5), None)
        peak_adoption = max(adoption_series) if adoption_series else 0.0
        purchase_conv = sum(self.series.get("daily_sold", [])) / max(1, sum(self.series.get("daily_demand", [])))
        results = {
            "peak_adoption": peak_adoption,
            "time_to_50pct": time_to_50,
            "final_adoption": adoption_series[-1] if adoption_series else 0.0,
            "purchase_conversion": purchase_conv,
            "supplier_inventory_end": self.supply_chain.total_stock,
            "retailer_inventory_end": sum(r.inventory_level for r in self.retailers),
            "cumulative_purchased": self.cumulative_purchased,
            "cumulative_infections": self.cumulative_infections,
            "series": self.series,
        }
        return results

    def visualize(self) -> None:
        """
        Simple visualization of a few series (adoption and price) if matplotlib is available.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        days = list(range(len(self.series.get("adoption_rate", []))))
        fig, ax1 = plt.subplots()
        ax1.plot(days, self.series.get("adoption_rate", []), label="Adoption rate", color="tab:blue")
        ax1.set_ylabel("Adoption rate")
        ax1.set_xlabel("Day")
        ax2 = ax1.twinx()
        ax2.plot(days, self.series.get("average_price", []), label="Average price", color="tab:orange")
        ax2.set_ylabel("Avg price")
        ax1.set_title("Adoption and Price")
        fig.tight_layout()
        plt.close(fig)  # Avoid GUI in headless env

    def save_results(self, out_file: str) -> None:
        """
        Save selected time series to CSV.
        """
        fields = ["day"] + list(self.series.keys())
        rows = []
        max_len = max((len(v) for v in self.series.values()), default=0)
        for d in range(max_len):
            row = {"day": d}
            for k, v in self.series.items():
                row[k] = v[d] if d < len(v) else ""
            rows.append(row)
        with open(out_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def validate(self, num_runs: int, days: int) -> Dict[str, Any]:
        """
        Run multiple simulations with different seeds and summarize adoption outcomes.
        """
        finals = []
        peaks = []
        for i in range(num_runs):
            sim = Simulation(self.p, seed=42 + i + 1)
            res = sim.run(days)
            finals.append(res.get("final_adoption", 0.0))
            peaks.append(res.get("peak_adoption", 0.0))
        return {
            "runs": num_runs,
            "days": days,
            "final_adoption_mean": statistics.mean(finals) if finals else 0.0,
            "final_adoption_std": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
            "peak_adoption_mean": statistics.mean(peaks) if peaks else 0.0,
        }


# Execute main for both direct execution and sandbox wrapper invocation

# Execute main for both direct execution and sandbox wrapper invocation
main()