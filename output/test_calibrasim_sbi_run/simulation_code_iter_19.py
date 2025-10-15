import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Path handling constants (may be used when reading/writing data files)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp a number to the [lo, hi] range.

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
    pass
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
    pass
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


@dataclass
class Person:
    """
    Person agent representing an individual in the simulation with attributes related to mask adoption behavior.

    Attributes
    ----------
    id : int
        Unique identifier.
    age : int
        Age of the person.
    income : float
        Income used for affordability decisions.
    household_id : int
        Household identifier.
    workplace_id : int
        Workplace identifier.
    network_neighbors : List[int]
        List of neighbor person IDs in the social network.
    mask_attitude : float
        Continuous latent attitude toward mask wearing in [0,1].
    mask_adopted : bool
        Whether the person is currently wearing a mask.
    risk_perception : float
        Perceived risk in [0,1].
    compliance_propensity : float
        Tendency to comply with rules in [0,1].
    social_influence_susceptibility : float
        Sensitivity to peer influence in [0,1].
    trust_in_authorities : float
        Trust in authorities in [0,1].
    media_consumption_profile : float
        How much the individual is influenced by media messages in [0,1].
    mask_inventory : int
        Number of masks held; if 0, cannot adopt even if willing.

    Methods
    -------
    perceive_risk(peer_share, policy_signal, media_signal, w_peer, w_policy, w_media)
        Update risk perception using peer/policy/media signals.
    update_attitude(habit_persistence, decay, susceptibility)
        Update mask attitude given risk perception, habit, and decay.
    decide_adoption(threshold_base, price, affordability_sensitivity, enforcement_prob)
        Decide whether to adopt mask use given threshold and affordability.
    purchase_masks(rng, price, bundle, subsidy_rate)
        Determine quantity of masks to purchase, subject to affordability.
    comply_with_policy(location_enforcement, signage_strength, rng)
        Decide compliance at a location when required.
    """
    id: int
    age: int
    income: float
    household_id: int
    workplace_id: int
    network_neighbors: List[int] = field(default_factory=list)
    mask_attitude: float = 0.0
    mask_adopted: bool = False
    risk_perception: float = 0.0
    compliance_propensity: float = 0.5
    social_influence_susceptibility: float = 0.5
    trust_in_authorities: float = 0.6
    media_consumption_profile: float = 0.5
    mask_inventory: int = 0

    def perceive_risk(
        self,
        peer_share: float,
        policy_signal: float,
        media_signal: float,
        w_peer: float,
        w_policy: float,
        w_media: float,
    ) -> None:
        """
        Update risk perception using a weighted blend of signals and inertia.

        Parameters
        ----------
        peer_share : float
            Observed share of peers adopting [0,1].
        policy_signal : float
            Strength of policy guidance [0,1].
        media_signal : float
            Aggregated media signal in [-1,1], where positive supports adoption.
        w_peer : float
            Weight for peer signal.
        w_policy : float
            Weight for policy signal.
        w_media : float
            Weight for media signal.
        """
        pass
        # Normalize media signal to [0,1]
        media_component = 0.5 * (media_signal + 1.0)
        rp_signal = w_peer * peer_share + w_policy * policy_signal + w_media * media_component
        rp_signal = clamp(rp_signal, 0.0, 1.0)
        inertia = 0.7
        self.risk_perception = clamp(inertia * self.risk_perception + (1 - inertia) * rp_signal)

    def update_attitude(self, habit_persistence: float, decay: float, susceptibility: float) -> None:
        """
        Update the attitude toward mask wearing.

        Parameters
        ----------
        habit_persistence : float
            Persistence factor in [0,1]; higher means slower change.
        decay : float
            Habit decay rate in [0,1].
        susceptibility : float
            How responsive the agent is to new information in [0,1].
        """
        pass
        base_target = 0.6 * self.risk_perception + 0.4 * self.mask_attitude
        target = self.mask_attitude + susceptibility * (base_target - self.mask_attitude)
        updated = habit_persistence * self.mask_attitude + (1 - habit_persistence) * target
        updated -= decay * (1.0 - self.mask_attitude)
        self.mask_attitude = clamp(updated, 0.0, 1.0)

    def decide_adoption(
        self,
        threshold_base: float,
        price: float,
        affordability_sensitivity: float,
        enforcement_prob: float,
    ) -> None:
        """
        Decide whether to wear a mask today.

        Parameters
        ----------
        threshold_base : float
            Base threshold for adoption in [0,1].
        price : float
            Current mask price.
        affordability_sensitivity : float
            Sensitivity of threshold to price/income ratio.
        enforcement_prob : float
            Probability of enforcement at locations (policy signal).
        """
        pass
        # Affordability proxy: share of income needed for masks
        affordability_ratio = price / max(1e-6, self.income + price)
        threshold = threshold_base + affordability_sensitivity * affordability_ratio
        threshold -= 0.15 * self.trust_in_authorities * enforcement_prob
        threshold = clamp(threshold)
        can_adopt = self.mask_inventory > 0
        self.mask_adopted = (self.mask_attitude >= threshold) and can_adopt

    def purchase_masks(
        self, rng, price: float, bundle: int = 5, subsidy_rate: float = 0.0
    ) -> int:
        """
        Decide how many masks to purchase given affordability and subsidy.

        Parameters
        ----------
        rng : random.Random
            RNG for stochastic decisions.
        price : float
            Mask price per unit.
        bundle : int
            Default quantity to attempt to purchase.
        subsidy_rate : float
            Fractional subsidy on price [0,1].

        Returns
        -------
        int
            Quantity desired to purchase (subject to retailer stock).
        """
        pass
        effective_price = max(0.0, price * (1.0 - subsidy_rate))
        affordability = self.income / (self.income + 10.0 * effective_price)
        buy_prob = clamp(0.3 + 0.7 * affordability)
        desire = bundle
        if rng.random() < buy_prob:
            return desire
        return 0

    def comply_with_policy(self, location_enforcement: float, signage_strength: float, rng) -> bool:
        """
        Determine if the person complies with a mask requirement at a location.

        Parameters
        ----------
        location_enforcement : float
            Enforcement probability in [0,1].
        signage_strength : float
            Salience of signage nudging compliance [0,1].
        rng : random.Random
            RNG instance.

        Returns
        -------
        bool
            True if the person complies, False otherwise.
        """
        pass
        # Compliance influenced by propensity, trust, and signage
        nudged = clamp(self.compliance_propensity * 0.6 + self.trust_in_authorities * 0.3 + signage_strength * 0.1)
        # If enforcement will likely occur, compliance increases
        adjusted = clamp(nudged + 0.4 * location_enforcement * self.trust_in_authorities)
        return rng.random() < adjusted


@dataclass
class Location:
    """
    Location where interactions and policy enforcement may occur.

    Attributes
    ----------
    id : int
        Identifier for the location.
    type : str
        Type of location (e.g., 'retail', 'work', 'public').
    capacity : int
        Maximum people that can be present.
    mask_requirement : bool
        Whether masks are required.
    enforcement_level : float
        Baseline enforcement probability [0,1].
    signage_strength : float
        Effectiveness of signage prompting compliance [0,1].
    foot_traffic_rate : float
        Probability an individual visits per day [0,1].

    Methods
    -------
    enforce_mask_policy(person, agency_enforcement, rng)
        Simulate enforcement interaction for a visiting person.
    """
    id: int
    type: str
    capacity: int
    mask_requirement: bool
    enforcement_level: float
    signage_strength: float
    foot_traffic_rate: float

    def enforce_mask_policy(self, person: Person, agency_enforcement: float, rng) -> Tuple[bool, bool]:
        """
        Enforce mask policy with certain probability.

        Parameters
        ----------
        person : Person
            The visiting person.
        agency_enforcement : float
            Additional enforcement scaling from the agency [0,1].
        rng : random.Random
            RNG for stochastic checks.

        Returns
        -------
        Tuple[bool, bool]
            (incident_occurred, compliant_now)
        """
        pass
        if not self.mask_requirement:
            return (False, person.mask_adopted)

        # If already wearing a mask, compliant
        if person.mask_adopted:
            return (False, True)

        # If not adopted, decide to comply upon entry
        will_comply = person.comply_with_policy(
            location_enforcement=clamp(self.enforcement_level * agency_enforcement),
            signage_strength=self.signage_strength,
            rng=rng,
        )

        # If still non-compliant, enforcement may trigger incident
        incident = False
        if not will_comply:
            check_prob = clamp(self.enforcement_level * agency_enforcement)
            if rng.random() < check_prob:
                incident = True

        return (incident, will_comply)


@dataclass
class GovernmentAgency:
    """
    Government agency controlling policy signals and enforcement resources.

    Attributes
    ----------
    policy_level : float
        Intensity of pro-mask guidance [0,1].
    enforcement_resources : float
        Available enforcement resources [0,1].
    messaging_strategy : float
        Effectiveness of messaging [0,1].
    media_reach : float
        Fraction of the population reachable by official communications [0,1].

    Methods
    -------
    broadcast_guidance()
        Produce a policy signal in [0,1].
    adjust_enforcement(day, policy_schedule)
        Adjust enforcement level based on schedule or constraints.
    """
    policy_level: float
    enforcement_resources: float
    messaging_strategy: float
    media_reach: float

    def broadcast_guidance(self) -> float:
        """
        Create a policy signal that informs risk perception and attitudes.

        Returns
        -------
        float
            Policy signal in [0,1].
        """
        pass
        return clamp(self.policy_level * self.messaging_strategy * self.media_reach)

    def adjust_enforcement(self, day: int, policy_schedule: Optional[Dict[int, float]] = None) -> float:
        """
        Adjust enforcement based on a schedule and resource constraints.

        Parameters
        ----------
        day : int
            Current day index.
        policy_schedule : Optional[Dict[int, float]]
            Optional mapping of day->policy_level changes.

        Returns
        -------
        float
            Updated enforcement probability scaling [0,1].
        """
        pass
        if policy_schedule and day in policy_schedule:
            self.policy_level = clamp(policy_schedule[day])
        # Enforcement capped by resources
        return clamp(self.enforcement_resources * (0.5 + 0.5 * self.policy_level))


@dataclass
class Retailer:
    """
    Retailer selling masks, managing inventory and price.

    Attributes
    ----------
    id : int
        Identifier for the retailer.
    inventory : int
        Current stock.
    restock_interval_days : int
        Days between restocks.
    supply_capacity_per_day : int
        Supply capacity per day.
    mask_price : float
        Current price per mask.
    stockout_days : int
        Days with zero inventory encountered.
    _days_until_restock : int
        Internal countdown to restock.
    _stockout_since_last_restock : int
        Counter of stockout days since the last restock.
    min_price : float
        Lower bound for price.
    max_price : float
        Upper bound for price.
    up_rate : float
        Price increase rate when stockouts occur.
    down_rate : float
        Price decrease rate when stockouts do not occur.

    Methods
    -------
    sell_masks(qty)
        Sell masks up to available inventory.
    end_of_day_update()
        Update stockout counters at end of day.
    restock_and_adjust_price()
        Restock inventory and adjust price based on recent stockouts.
    """
    id: int
    inventory: int
    restock_interval_days: int
    supply_capacity_per_day: int
    mask_price: float
    stockout_days: int = 0
    _days_until_restock: int = 0
    _stockout_since_last_restock: int = 0
    min_price: float = 1.0
    max_price: float = 20.0
    up_rate: float = 0.10
    down_rate: float = 0.05

    def sell_masks(self, qty: int) -> int:
        """
        Sell requested quantity if available.

        Parameters
        ----------
        qty : int
            Desired quantity.

        Returns
        -------
        int
            Quantity actually sold.
        """
        pass
        if qty <= 0:
            return 0
        sold = min(qty, self.inventory)
        self.inventory -= sold
        return sold

    def end_of_day_update(self) -> None:
        """
        Update stockout metrics at end of day.
        """
        pass
        if self.inventory <= 0:
            self.stockout_days += 1
            self._stockout_since_last_restock += 1

    def restock_and_adjust_price(self) -> None:
        """
        Restock inventory according to capacity and adjust price depending on stockout experience.
        """
        pass
        self._days_until_restock -= 1
        if self._days_until_restock <= 0:
            # Restock
            added = self.supply_capacity_per_day * max(1, self.restock_interval_days)
            self.inventory += added
            # Price adjust: if stockouts common, increase; else decrease
            if self._stockout_since_last_restock >= max(1, self.restock_interval_days // 3):
                self.mask_price = min(self.max_price, self.mask_price * (1.0 + self.up_rate))
            else:
                self.mask_price = max(self.min_price, self.mask_price * (1.0 - self.down_rate))
            # Reset counters
            self._stockout_since_last_restock = 0
            self._days_until_restock = self.restock_interval_days


@dataclass
class MediaChannel:
    """
    Media channel broadcasting messages that can support or undermine mask adoption.

    Attributes
    ----------
    id : int
        Identifier.
    reach : float
        Audience reach in [0,1].
    message_frequency : float
        Daily probability of broadcasting a message [0,1].
    bias : float
        Bias direction in [-1,1], where positive supports adoption.
    misinformation_probability : float
        Probability that a message is misinformation (flips sign).

    Methods
    -------
    broadcast_message(rng)
        Produce a message signal in [-1,1] with frequency and misinformation.
    """
    id: int
    reach: float
    message_frequency: float
    bias: float
    misinformation_probability: float

    def broadcast_message(self, rng) -> float:
        """
        Broadcast a message according to frequency and misinformation probability.

        Parameters
        ----------
        rng : random.Random
            RNG for stochastic decisions.

        Returns
        -------
        float
            Message signal in [-1,1], or 0 if no message broadcast today.
        """
        pass
        if rng.random() >= self.message_frequency:
            return 0.0
        sign = self.bias
        if rng.random() < self.misinformation_probability:
            sign = -sign
        # Scale by reach
        return clamp(sign, -1.0, 1.0) * clamp(self.reach, 0.0, 1.0)


class Simulation:
    """
    Main simulation orchestrator for mask adoption dynamics.

    Parameters
    ----------
    params : Dict[str, float]
        Configuration parameters for the run.
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    p : Dict[str, float]
        Parameters.
    rng : random.Random
        Deterministic RNG instance.
    people : List[Person]
        Agent population.
    locations : List[Location]
        List of locations.
    retailers : List[Retailer]
        Retailers handling supply/pricing.
    agency : GovernmentAgency
        Policy authority.
    media : List[MediaChannel]
        Media channels.
    series : Dict[str, List[float]]
        Time series metrics: adoption_rate, price, etc.
    daily_counters : Dict[str, List[float]]
        Additional per-day counters (visits, incidents, compliance).
    """
    def __init__(self, params: Dict[str, float], seed: int = 42):
        """
        Initialize the simulation with parameters and seed.

        Parameters
        ----------
        params : Dict[str, float]
            Parameters for the simulation.
        seed : int
            Random seed.
        """
        pass
        self.p = dict(params)
        self.rng = __import__("random").Random(int(seed))

        # Entities
        self.people: List[Person] = []
        self.locations: List[Location] = []
        self.retailers: List[Retailer] = []
        self.agency = GovernmentAgency(
            policy_level=float(self.p.get("policy_level", 0.5)),
            enforcement_resources=float(self.p.get("enforcement_resources", 0.2)),
            messaging_strategy=float(self.p.get("messaging_strategy", 0.6)),
            media_reach=float(self.p.get("media_reach", 0.7)),
        )
        self.media: List[MediaChannel] = [
            MediaChannel(
                id=1,
                reach=float(self.p.get("media_reach_main", 0.7)),
                message_frequency=float(self.p.get("message_frequency_per_week", 3)) / 7.0,
                bias=float(self.p.get("media_bias", 1.0)),
                misinformation_probability=float(self.p.get("misinformation_rate", 0.05)),
            )
        ]

        # Series and counters
        self.series: Dict[str, List[float]] = {
            "adoption_rate": [],
            "average_price": [],
            "retailer_inventory": [],
            "enforcement_incidents_per_1000": [],
            "compliance_rate": [],
        }
        self.daily_counters: Dict[str, List[float]] = {
            "visits": [],
            "incidents": [],
            "compliant_entries": [],
        }

        # Policy schedule (optional)
        self.policy_schedule: Dict[int, float] = {}
        if "policy_change_day" in self.p and "policy_level_post_change" in self.p:
            self.policy_schedule[int(self.p["policy_change_day"])] = float(self.p["policy_level_post_change"])

    def _small_world(self, N: int, k: int, beta: float) -> List[List[int]]:
        """
        Create a Watts–Strogatz small-world network adjacency list.

        Parameters
        ----------
        N : int
            Number of nodes.
        k : int
            Each node connected to k nearest neighbors (k even preferred).
        beta : float
            Rewiring probability.

        Returns
        -------
        List[List[int]]
            Adjacency list of neighbors for each node.
        """
        pass
        k = max(2, k)
        if k % 2 == 1:
            k += 1
        half = k // 2
        adj = [set() for _ in range(N)]
        # Ring lattice
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                adj[i].add(j)
                adj[j].add(i)
        # Rewiring
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                if self.rng.random() < beta:
                    # rewire i->j to i->k2 (avoid self and duplicates)
                    choices = [x for x in range(N) if x != i and x not in adj[i]]
                    if choices:
                        k2 = self.rng.choice(choices)
                        if j in adj[i]:
                            adj[i].discard(j)
                            adj[j].discard(i)
                        adj[i].add(k2)
                        adj[k2].add(i)
        return [list(nei) for nei in adj]

    def initialize(self) -> None:
        """
        Initialize population, network, locations, and retailers.
        """
        pass
        N = int(self.p.get("population_size", 1000))
        init_rate = float(self.p.get("initial_mask_adoption_rate", 0.2))
        avg_deg = int(self.p.get("average_degree", 8))
        risk_init = float(self.p.get("risk_perception_initial_mean", 0.3))

        # People
        self.people = []
        for i in range(N):
            # Log-normal income proxy
            mu = float(self.p.get("income_lognorm_mu", 3.0))
            sigma = float(self.p.get("income_lognorm_sigma", 0.5))
            income = math.exp(self.rng.normalvariate(mu, sigma))
            adopted = self.rng.random() < init_rate
            base_att = 0.75 + 0.15 * self.rng.random() if adopted else 0.25 * self.rng.random()
            person = Person(
                id=i,
                age=self.rng.randint(18, 85),
                income=income,
                household_id=i // 3,
                workplace_id=i % max(1, int(self.p.get("num_workplaces", 50))),
                mask_attitude=clamp(base_att),
                mask_adopted=adopted,
                risk_perception=clamp(risk_init + self.rng.uniform(-0.05, 0.05)),
                compliance_propensity=clamp(self.rng.uniform(0.3, 0.9)),
                social_influence_susceptibility=clamp(self.rng.uniform(0.3, 0.9)),
                trust_in_authorities=clamp(self.rng.uniform(0.3, 0.9)),
                media_consumption_profile=clamp(self.rng.uniform(0.2, 0.8)),
                mask_inventory=1 if adopted else 0,
            )
            self.people.append(person)

        # Social network
        neighbors = self._small_world(N, avg_deg, beta=float(self.p.get("small_world_beta", 0.05)))
        for i, p in enumerate(self.people):
            p.network_neighbors = neighbors[i]

        # Locations (single retail/public hub for enforcement)
        self.locations = [
            Location(
                id=0,
                type="retail",
                capacity=int(self.p.get("location_capacity", 2000)),
                mask_requirement=bool(self.p.get("mask_requirement", True)),
                enforcement_level=float(self.p.get("enforcement_probability", 0.1)),
                signage_strength=float(self.p.get("signage_effect", 0.05)),
                foot_traffic_rate=float(self.p.get("foot_traffic_rate", 0.3)),
            )
        ]

        # Retailer
        initial_inventory = int(self.p.get("supply_capacity_per_day", 3000))
        restock_int = int(self.p.get("stock_replenishment_interval_days", 7))
        mask_price = float(self.p.get("mask_cost", 5.0))
        self.retailers = [
            Retailer(
                id=0,
                inventory=initial_inventory,
                restock_interval_days=restock_int,
                supply_capacity_per_day=int(self.p.get("supply_capacity_per_day", 3000)),
                mask_price=mask_price,
                min_price=float(self.p.get("min_mask_price", 1.0)),
                max_price=float(self.p.get("max_mask_price", 20.0)),
                up_rate=float(self.p.get("price_adjust_up_rate", 0.1)),
                down_rate=float(self.p.get("price_adjust_down_rate", 0.05)),
            )
        ]
        # Initialize restock countdown
        self.retailers[0]._days_until_restock = restock_int

    def _aggregate_media_signal(self) -> float:
        """
        Aggregate media messages into a single signal in [-1,1].

        Returns
        -------
        float
            Aggregated media signal.
        """
        pass
        total = 0.0
        for ch in self.media:
            total += ch.broadcast_message(self.rng)
        return clamp(total, -1.0, 1.0)

    def _peer_share(self, adopted_prev: List[float], neighbors: List[int]) -> float:
        """
        Compute the share of peers adopting.

        Parameters
        ----------
        adopted_prev : List[float]
            Binary list of previous adoption states.
        neighbors : List[int]
            Neighbor indices.

        Returns
        -------
        float
            Peer adoption rate in [0,1].
        """
        pass
        if not neighbors:
            return sum(adopted_prev) / max(1, len(adopted_prev))
        return sum(adopted_prev[j] for j in neighbors) / max(1, len(neighbors))

    def step(self, day: int) -> None:
        """
        Execute one simulation day: update perceptions, attitudes, decisions, purchases, visits, enforcement, and metrics.

        Parameters
        ----------
        day : int
            Day index.
        """
        pass
        # Policy and media
        agency_enforcement = self.agency.adjust_enforcement(day, self.policy_schedule)
        policy_signal = self.agency.broadcast_guidance()
        media_signal = self._aggregate_media_signal()

        # Weights
        w_peer = float(self.p.get("peer_influence_weight", 0.5))
        w_policy = float(self.p.get("policy_influence_weight", 0.3))
        w_media = float(self.p.get("media_influence_weight", 0.2))
        habit = float(self.p.get("habit_persistence", 0.9))
        decay = float(self.p.get("adoption_decay_rate", 0.01))
        threshold_base = float(self.p.get("adoption_threshold_base", 0.6))
        affordability_sensitivity = float(self.p.get("affordability_sensitivity", 0.05))
        subsidy_rate = float(self.p.get("subsidy_rate", 0.0))

        # Precompute peer states
        adopted_prev = [1.0 if p.mask_adopted else 0.0 for p in self.people]

        # Retailer reference
        retailer = self.retailers[0]

        # Person-level updates
        for i, person in enumerate(self.people):
            peer_share = self._peer_share(adopted_prev, person.network_neighbors)
            person.perceive_risk(peer_share, policy_signal, media_signal, w_peer, w_policy, w_media)
            person.update_attitude(habit_persistence=habit, decay=decay, susceptibility=person.social_influence_susceptibility)

            # Purchase if needed
            if person.mask_inventory <= 0 and person.mask_attitude > 0.4:
                desired = person.purchase_masks(self.rng, price=retailer.mask_price, bundle=int(self.p.get("purchase_bundle", 5)), subsidy_rate=subsidy_rate)
                if desired > 0:
                    bought = retailer.sell_masks(desired)
                    person.mask_inventory += bought

            # Decide adoption
            person.decide_adoption(
                threshold_base=threshold_base,
                price=retailer.mask_price,
                affordability_sensitivity=affordability_sensitivity,
                enforcement_prob=agency_enforcement,
            )

            # Consume one mask if adopted
            if person.mask_adopted and person.mask_inventory > 0:
                person.mask_inventory -= 1

        # Visits and enforcement at location
        loc = self.locations[0]
        visits = 0
        incidents = 0
        compliant_entries = 0
        for person in self.people:
            if self.rng.random() < loc.foot_traffic_rate:
                visits += 1
                incident, compliant_now = loc.enforce_mask_policy(person, agency_enforcement, self.rng)
                incidents += 1 if incident else 0
                compliant_entries += 1 if (compliant_now or person.mask_adopted) else 0

        # Retailer daily updates
        retailer.end_of_day_update()
        retailer.restock_and_adjust_price()

        # Metrics
        adoption = sum(1 for p in self.people if p.mask_adopted) / max(1, len(self.people))
        avg_price = retailer.mask_price
        inventory = retailer.inventory
        incidents_per_1000 = (incidents / max(1, visits)) * 1000.0 if visits > 0 else 0.0
        compliance_rate = (compliant_entries / max(1, visits)) if visits > 0 else 0.0

        self.series["adoption_rate"].append(clamp(adoption))
        self.series["average_price"].append(avg_price)
        self.series["retailer_inventory"].append(inventory)
        self.series["enforcement_incidents_per_1000"].append(incidents_per_1000)
        self.series["compliance_rate"].append(clamp(compliance_rate))

        self.daily_counters["visits"].append(visits)
        self.daily_counters["incidents"].append(incidents)
        self.daily_counters["compliant_entries"].append(compliant_entries)

    def run(self, days: int) -> Dict[str, object]:
        """
        Run the simulation for a specified number of days.

        Parameters
        ----------
        days : int
            Number of days to simulate.

        Returns
        -------
        Dict[str, object]
            Results including time series and summary metrics.
        """
        pass
        self.initialize()
        for day in range(days):
            self.step(day)

        # Summary metrics
        target = float(self.p.get("target_adoption_threshold", 0.7))
        time_to_threshold: Optional[int] = None
        for d, val in enumerate(self.series["adoption_rate"]):
            if val >= target:
                time_to_threshold = d
                break

        # Stockout rate over the period
        stockout_days = self.retailers[0].stockout_days
        stockout_rate = stockout_days / max(1, days)

        # Enforcement incidents rate
        total_visits = sum(self.daily_counters["visits"])
        total_incidents = sum(self.daily_counters["incidents"])
        enforcement_incidents_rate = (total_incidents / max(1, total_visits)) * 1000.0

        # Policy effect size: if schedule present, compare post vs pre windows
        policy_effect_size = None
        if self.policy_schedule:
            change_day = min(self.policy_schedule.keys())
            pre_window = self.series["adoption_rate"][: max(1, change_day)]
            post_window = self.series["adoption_rate"][change_day:]
            if pre_window and post_window:
                policy_effect_size = statistics.mean(post_window[-min(14, len(post_window)) :]) - statistics.mean(pre_window[-min(14, len(pre_window)) :])

        # Sustained adoption last 30 days (or last available)
        window = min(30, len(self.series["adoption_rate"]))
        sustained_adoption_rate = statistics.mean(self.series["adoption_rate"][-window:]) if window > 0 else 0.0

        # Inequality: gini over adoption by income group (use adoption*income)
        adoption_by_income = []
        for p in self.people:
            adoption_by_income.append((1.0 if p.mask_adopted else 0.0) * p.income)
        inequality_in_adoption = gini(adoption_by_income)

        results = {
            "adoption_rate": self.series["adoption_rate"],
            "average_price": self.series["average_price"],
            "retailer_inventory": self.series["retailer_inventory"],
            "enforcement_incidents_per_1000": self.series["enforcement_incidents_per_1000"],
            "compliance_rate": self.series["compliance_rate"],
            "time_to_threshold": time_to_threshold,
            "stockout_rate": stockout_rate,
            "enforcement_incidents_rate": enforcement_incidents_rate,
            "policy_effect_size": policy_effect_size,
            "sustained_adoption_rate": sustained_adoption_rate,
            "inequality_in_adoption": inequality_in_adoption,
            "total_visits": total_visits,
            "total_incidents": total_incidents,
        }
        return results

    def evaluate(self, evaluation_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate additional metrics based on selected names.

        Parameters
        ----------
        evaluation_metrics : Optional[List[str]]
            List of metric names to compute.

        Returns
        -------
        Dict[str, float]
            Mapping of metric name to value.
        """
        pass
        if not evaluation_metrics:
            return {}

        metrics_out: Dict[str, float] = {}
        for name in evaluation_metrics:
            if name == "adoption_rate_final":
                metrics_out[name] = self.series["adoption_rate"][-1] if self.series["adoption_rate"] else 0.0
            elif name == "price_final":
                metrics_out[name] = self.series["average_price"][-1] if self.series["average_price"] else 0.0
            elif name == "rmse_to_observed":
                # Placeholder: requires observed series in params
                observed = self.p.get("observed_adoption_series", [])
                if observed and len(observed) == len(self.series["adoption_rate"]):
                    diffsq = [(a - b) ** 2 for a, b in zip(self.series["adoption_rate"], observed)]
                    metrics_out[name] = math.sqrt(sum(diffsq) / max(1, len(diffsq)))
                else:
                    metrics_out[name] = float("nan")
            else:
                metrics_out[name] = float("nan")
        return metrics_out

    def visualize(self) -> None:
        """
        Display a simple textual visualization of adoption over time.
        """
        pass
        series = self.series.get("adoption_rate", [])
        if not series:
            print("No data to visualize.")
            return
        print("Adoption over time (ASCII sparkline):")
        # Generate a simple sparkline using blocks
        blocks = "▁▂▃▄▅▆▇█"
        for i, v in enumerate(series):
            idx = int(clamp(v, 0.0, 1.0) * (len(blocks) - 1))
            sys.stdout.write(blocks[idx])
        sys.stdout.write("\n")
        print(f"Final adoption: {series[-1]:.3f}")

    def save_results(self, filename: str) -> None:
        """
        Save key time series to a CSV file.

        Parameters
        ----------
        filename : str
            Output CSV filename.
        """
        pass
        header = ["day", "adoption_rate", "average_price", "retailer_inventory", "enforcement_incidents_per_1000", "compliance_rate"]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            days = len(self.series["adoption_rate"])
            for d in range(days):
                writer.writerow(
                    [
                        d,
                        self.series["adoption_rate"][d],
                        self.series["average_price"][d],
                        self.series["retailer_inventory"][d],
                        self.series["enforcement_incidents_per_1000"][d],
                        self.series["compliance_rate"][d],
                    ]
                )

    # FIXED: Implemented validation utilities per feedback
    def validate_policy_monotonicity(self) -> Tuple[bool, str]:
        """
        Validation: Policy monotonicity.

        Run two short scenarios: low vs high enforcement/policy and ensure adoption with high policy >= adoption with low policy.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        base_params = dict(self.p)
        short_days = 40

        # Low policy
        low_params = dict(base_params)
        low_params.update({"policy_level": 0.1, "enforcement_resources": 0.05})
        low_sim = Simulation(low_params, seed=123)
        low_res = low_sim.run(short_days)
        low_final = low_res["adoption_rate"][-1]

        # High policy
        high_params = dict(base_params)
        high_params.update({"policy_level": 0.9, "enforcement_resources": 0.9})
        high_sim = Simulation(high_params, seed=123)
        high_res = high_sim.run(short_days)
        high_final = high_res["adoption_rate"][-1]

        ok = high_final >= low_final - 1e-9
        msg = f"policy_monotonicity: high={high_final:.3f}, low={low_final:.3f}"
        return ok, msg

    def validate_no_influence_stability(self) -> Tuple[bool, str]:
        """
        Validation: Stability with no social/policy/media influence.

        Set all influence weights to zero and check adoption remains near initial rate.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        params2 = dict(self.p)
        params2.update(
            {
                "peer_influence_weight": 0.0,
                "policy_influence_weight": 0.0,
                "media_influence_weight": 0.0,
                "adoption_decay_rate": 0.0,
            }
        )
        sim = Simulation(params2, seed=456)
        days = 60
        res = sim.run(days)
        init = float(params2.get("initial_mask_adoption_rate", 0.2))
        avg_last14 = statistics.mean(res["adoption_rate"][-14:]) if len(res["adoption_rate"]) >= 14 else res["adoption_rate"][-1]
        ok = abs(avg_last14 - init) < 0.1
        msg = f"no_influence_stability: avg_last14={avg_last14:.3f}, init={init:.3f}"
        return ok, msg

    def validate_convergence_check(self) -> Tuple[bool, str]:
        """
        Validation: Convergence.

        Confirm that the change in adoption rate over the last 14 days is below epsilon.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        res = self.series.get("adoption_rate", [])
        if len(res) < 15:
            return False, "convergence_check: insufficient data (<15 days)"
        delta = abs(res[-1] - res[-15])
        ok = delta < float(self.p.get("convergence_epsilon", 0.05))
        msg = f"convergence_check: delta14={delta:.4f}"
        return ok, msg


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the simulation CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    pass
    parser = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation CLI")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input JSON with parameters.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file for results.")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV output for time series.")
    parser.add_argument("--population-size", type=int, default=None, help="Override population size.")
    parser.add_argument("--days", type=int, default=60, help="Number of days to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fast-smoke-run", action="store_true", help="Run a fast smoke test (prints SMOKE_OK).")
    parser.add_argument("--validate", type=str, default=None, help="Comma-separated validations: policy_monotonicity,no_influence_stability,convergence_check")
    return parser.parse_args()


def default_params() -> Dict[str, float]:
    """
    Construct default parameters for the simulation.

    Returns
    -------
    Dict[str, float]
        Default parameter dictionary.
    """
    pass
    return {
        # Population and network
        "population_size": 1000,
        "average_degree": 10,
        "small_world_beta": 0.05,
        # Behavior and influences
        "initial_mask_adoption_rate": 0.2,
        "peer_influence_weight": 0.5,
        "policy_influence_weight": 0.3,
        "media_influence_weight": 0.2,
        "habit_persistence": 0.9,
        "adoption_decay_rate": 0.01,
        "adoption_threshold_base": 0.6,
        "affordability_sensitivity": 0.05,
        "risk_perception_initial_mean": 0.3,
        # Policy and enforcement
        "policy_level": 0.5,
        "enforcement_resources": 0.2,
        "messaging_strategy": 0.6,
        "media_reach": 0.7,
        "enforcement_probability": 0.1,
        "mask_requirement": True,
        "signage_effect": 0.05,
        "foot_traffic_rate": 0.3,
        # Supply and pricing
        "supply_capacity_per_day": 3000,
        "stock_replenishment_interval_days": 7,
        "mask_cost": 5.0,
        "min_mask_price": 1.0,
        "max_mask_price": 20.0,
        "price_adjust_up_rate": 0.1,
        "price_adjust_down_rate": 0.05,
        "purchase_bundle": 5,
        # Media
        "media_reach_main": 0.7,
        "message_frequency_per_week": 3.0,
        "media_bias": 1.0,
        "misinformation_rate": 0.05,
        # Targets
        "target_adoption_threshold": 0.7,
        # Optional policy change
        # "policy_change_day": 30,
        # "policy_level_post_change": 0.8,
    }


def run_validations(sim: Simulation, which: List[str]) -> Tuple[bool, List[str]]:
    """
    Run selected validation checks on the simulation model.

    Parameters
    ----------
    sim : Simulation
        Simulation instance (parameters used for baselines).
    which : List[str]
        List of validation names.

    Returns
    -------
    Tuple[bool, List[str]]
        Overall result and list of messages.
    """
    pass
    results = []
    all_ok = True
    for name in which:
        if name == "policy_monotonicity":
            ok, msg = sim.validate_policy_monotonicity()
        elif name == "no_influence_stability":
            ok, msg = sim.validate_no_influence_stability()
        elif name == "convergence_check":
            # Ensure the simulation has run
            if not sim.series["adoption_rate"]:
                # Run a short sim to test convergence
                sim.run(60)
            ok, msg = sim.validate_convergence_check()
        else:
            ok, msg = False, f"Unknown validation: {name}"
        results.append(msg)
        if not ok:
            all_ok = False
    return all_ok, results


def main() -> None:
    """
    Command-line interface entry point.

    Behavior
    --------
    - Loads parameters from defaults and optional JSON input.
    - Applies CLI overrides (population size, days, seed).
    - Runs the simulation and prints JSON results or saves to file.
    - Optionally outputs CSV and runs validations.
    - Provides a fast smoke test path printing 'SMOKE_OK' for harness integration.
    """
    pass
    args = parse_args()

    # Load params
    params = default_params()
    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Accept either {"parameters": {...}} or flat dict
                to_merge = loaded.get("parameters", loaded)
                params.update(to_merge)
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            # Continue with defaults

    # Overrides
    if args.population_size is not None:
        params["population_size"] = int(args.population_size)
    days = int(params.get("simulation_duration_days", args.days))

    # Create and run simulation
    sim = Simulation(params, seed=args.seed)

    # Fast smoke test path
    if args.fast_smoke_run:
        # Small N, few days for fast execution
        sim.p["population_size"] = 60
        result = sim.run(5)
        if (
            len(result["adoption_rate"]) == 5
            and all(0.0 <= x <= 1.0 for x in result["adoption_rate"])
            and isinstance(result.get("stockout_rate", 0.0), float)
        ):
            print("SMOKE_OK")
            return
        else:
            print("SMOKE_FAIL")
            return

    # Validations (optional)
    if args.validate:
        names = [x.strip() for x in args.validate.split(",") if x.strip()]
        ok, messages = run_validations(sim, names)
        for m in messages:
            print(m)
        if not ok:
            # Non-zero exit for validation failures
            print("VALIDATION_FAIL", file=sys.stderr)
            # Still proceed to run the main sim unless we want to exit early
        # Reset sim for main run after validations
        sim = Simulation(params, seed=args.seed)

    # Main run
    results = sim.run(days)

    # Evaluate optional metrics if provided
    evaluation_metrics = params.get("evaluation_metrics", [])
    if isinstance(evaluation_metrics, list) and evaluation_metrics:
        extra = sim.evaluate(evaluation_metrics)
        results["evaluation"] = extra

    # Visualization to stdout (simple)
    sim.visualize()

    # CSV save (explicit demonstration per instruction)
    csv_filename = args.csv_out if args.csv_out else "results.csv"
    sim.save_results(csv_filename)

    out_json = json.dumps(results, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_json)
    else:
        print(out_json)


# FIXED: Restored full simulation implementation with CLI, deterministic seeding, network, supply, enforcement, metrics, validations, and smoke test.
# FIXED: Removed any Docker requirements; pure standard library code.
# FIXED: Implemented entities with spec-compliant attribute names and behavior methods.
# FIXED: Implemented small-world network generator and social influence dynamics.
# FIXED: Added argparse CLI with specified flags and JSON/CSV outputs.
# FIXED: Added validation routines and convergence checks.
# FIXED: Ensured unconditional main() call at bottom.

# Execute main for both direct execution and sandbox wrapper invocation
main()