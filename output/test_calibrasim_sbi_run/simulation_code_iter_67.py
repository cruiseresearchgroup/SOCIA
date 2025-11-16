import os
import json
import csv
import argparse
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# Path Handling Instructions compliance
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a numeric value between lower and upper bounds.

    Parameters:
        x (float): The value to clamp.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: The clamped value.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.

    Parameters:
        x (float): Input value.

    Returns:
        float: Sigmoid(x).
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def gini(values: List[float]) -> float:
    """
    Compute Gini coefficient for a list of non-negative values.

    Parameters:
        values (List[float]): The values to compute Gini coefficient on.

    Returns:
        float: Gini coefficient in [0, 1].
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    xs = sorted(max(0.0, v) for v in values)
    s = sum(xs)
    n = len(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(xs, 1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


@dataclass
class Person:
    """
    Represents an individual in the simulation with attributes for mask adoption, socio-economic factors,
    and household membership.

    Attributes:
        id (int): Unique identifier for the person.
        income (float): Person's income level impacting affordability and inequality metrics.
        risk_perception (float): Perceived risk in [0, 1].
        trust_in_authorities (float): Trust level in [0, 1].
        susceptibility_to_messaging (float): Susceptibility to campaign/media messages [0, 1].
        misinformation_susceptibility (float): Susceptibility to misinformation [0, 1].
        social_influence_weight (float): Weight of peer influence on attitude [0, 1].
        mask_attitude (float): Latent attitude towards mask wearing [-1, 1].
        mask_adoption_state (bool): Whether the person is a current adopter (sticky state).
        compliance_probability (float): Latent compliance propensity in [0, 1].
        access_to_masks (bool): Whether masks are accessible to the person.
        inventory_masks (int): Number of masks available in personal inventory.
        cost_sensitivity (float): Sensitivity to price when considering mask purchases.
        budget (float): Available budget for purchasing masks.
        adoption_score (float): Rolling score representing consistency of wearing.
        age_group (str): Age group label (e.g., 'child', 'adult', 'senior').
        health_risk_level (float): Underlying risk level factor (0-1).
        household_id (int): ID of the household the person belongs to.
        network_neighbors (List[int]): IDs of neighbors in social network.
        past_enforcement_events (int): Count of enforcement events experienced.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    id: int
    income: float
    risk_perception: float
    trust_in_authorities: float
    susceptibility_to_messaging: float
    misinformation_susceptibility: float
    social_influence_weight: float
    mask_attitude: float
    mask_adoption_state: bool
    compliance_probability: float
    access_to_masks: bool
    inventory_masks: int
    cost_sensitivity: float = 0.5  # FIXED: Added economic factor
    budget: float = 100.0  # FIXED: Added budget tracking
    adoption_score: float = 0.0  # FIXED: Added rolling adoption score
    age_group: str = "adult"
    health_risk_level: float = 0.5
    household_id: int = -1
    network_neighbors: List[int] = field(default_factory=list)
    past_enforcement_events: int = 0


@dataclass
class Household:
    """
    Represents a household grouping of persons. Used for sharing masks and norm reinforcement.

    Attributes:
        id (int): Household ID.
        member_ids (List[int]): IDs of members in the household.
        norm_strength (float): Strength of within-household norm reinforcement [0, 1].
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    id: int
    member_ids: List[int]
    norm_strength: float = 0.3


@dataclass
class Location:
    """
    Represents a location type where individuals may visit and observe norms.

    Attributes:
        id (int): Unique identifier.
        type (str): Type of location (e.g., workplace, retail, transit, school, park).
        capacity (int): Maximum capacity; used loosely for realism.
        mask_requirement_policy (bool): Whether masks are required at the location.
        enforcement_strictness (float): Location-specific enforcement strictness [0, 1].
        foot_traffic (float): Relative foot traffic weight for sampling visits.
        observability_factor (float): Observability factor in [0, 1], how visible mask use is in this location.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    id: int
    type: str
    capacity: int
    mask_requirement_policy: bool
    enforcement_strictness: float
    foot_traffic: float
    observability_factor: float = 0.5


@dataclass
class PolicyAuthority:
    """
    Policy authority controlling mandates and enforcement with scope.

    Attributes:
        id (int): Identifier.
        mandate_status (bool): Whether a mandate is currently active.
        mandate_start_day (int): Day on which mandate begins.
        mandate_scope (str): Scope of mandate (e.g., 'indoor_public').
        enforcement_probability (float): Base enforcement probability per visit [0, 1].
        fine_amount (float): Fine amount for noncompliance.
        communication_strategy (str): Message framing for risk and norms.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    id: int
    mandate_status: bool
    mandate_start_day: int
    mandate_scope: str
    enforcement_probability: float
    fine_amount: float
    communication_strategy: str = "risk_and_norms"


@dataclass
class MediaChannel:
    """
    Media channel broadcasting messages that influence risk and attitudes and may carry misinformation.

    Attributes:
        id (int): Unique identifier.
        message_bias (float): Positive promotes masks; negative discourages.
        credibility (float): Credibility of the channel [0, 1].
        reach (float): Fraction of population reached per day [0, 1].
        misinformation_rate (float): Probability that content carries misinformation [0, 1].
        message_frequency (float): Frequency scaling [0, 1].
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    id: int
    message_bias: float
    credibility: float
    reach: float
    misinformation_rate: float
    message_frequency: float


@dataclass
class Retailer:
    """
    Retailer selling masks, with inventory, restocking policy, and price.

    Attributes:
        inventory (int): Current inventory level.
        restock_quantity (int): Quantity added on each restock event.
        restock_interval (int): Interval in days between planned restocks.
        price (float): Current mask unit price.
        max_purchase_per_customer (int): Cap on purchase per customer.
        supply_variability (float): Relative standard deviation for restock noise.
        stockout_days (int): Days where inventory is zero.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    inventory: int
    restock_quantity: int
    restock_interval: int
    price: float
    max_purchase_per_customer: int
    supply_variability: float
    stockout_days: int = 0

    def restock(self, day: int, rng: random.Random) -> int:
        """
        Restock inventory on schedule with variability.

        Parameters:
            day (int): Current simulation day (0-indexed).
            rng (random.Random): Seeded RNG.

        Returns:
            int: Quantity added to inventory.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        # FIXED: Implement retailer restock with variability and cap negative quantities
        added = 0
        if self.restock_interval > 0 and (day % self.restock_interval == 0):
            noise = int(self.restock_quantity * rng.gauss(0, self.supply_variability))
            qty = max(0, self.restock_quantity + noise)
            self.inventory += qty
            added = qty
        return added


def ring_small_world(n: int, k: int, p: float, rng: random.Random) -> Dict[int, List[int]]:
    """
    Build a Watts-Strogatz-like small-world network as an adjacency list.

    Parameters:
        n (int): Number of nodes.
        k (int): Average degree (must be even ideally; we will approximate).
        p (float): Rewiring probability in [0, 1].
        rng (random.Random): Seeded RNG.

    Returns:
        Dict[int, List[int]]: Adjacency list mapping node -> sorted list of neighbors.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    adj = {i: set() for i in range(n)}
    if n <= 1 or k <= 0:
        return {i: [] for i in range(n)}
    half = max(1, min(k // 2, (n - 1) // 2))
    # Start with ring lattice
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            adj[i].add(j)
            adj[j].add(i)
    # Rewire edges with probability p
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if j in adj[i] and rng.random() < p:
                adj[i].discard(j)
                adj[j].discard(i)
                for _ in range(50):
                    u = rng.randrange(n)
                    if u != i and u not in adj[i]:
                        adj[i].add(u)
                        adj[u].add(i)
                        break
                else:
                    adj[i].add(j)
                    adj[j].add(i)
    return {i: sorted(list(neigh)) for i, neigh in adj.items()}


class Simulation:
    """
    Main simulation engine coordinating entities, network, messaging, policy, market, households, and norms.

    This class implements a pure-Python agent-based simulation that:
    - Builds agents, households, locations, policy, media, retailer, and a small-world network.
    - Runs a daily loop including messaging (with misinformation), peer and household influence, purchasing, visits,
      scoped enforcement, and norm observation by location.
    - Aggregates and outputs required metrics, including evaluation with pass/fail flags.

    Notes:
        - Designed to be Docker-independent and directly executable.
        - Uses a single Retailer entity for simplicity.
    """
    pass  # FIXED: Retained 'pass' per interface requirement

    def __init__(self, params: Dict[str, Any], smoke: bool = False) -> None:
        """
        Initialize the simulation with parameters and optional smoke (fast) mode.

        Parameters:
            params (Dict[str, Any]): Simulation parameters.
            smoke (bool): If True, run a small, fast simulation for CI.

        Raises:
            ValueError: If any parameters are invalid.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        # FIXED: Added RNG seeding for reproducibility
        self.params = params
        self.rng = random.Random(params.get('random_seed', 42))
        self.days = 5 if smoke else int(params.get('simulation_horizon_days', 120))
        self.N = 200 if smoke else int(params.get('population_size', 5000))
        self.smoke = smoke

        # Parameter validation and clamping
        self._validate_and_default_params()

        # Build network
        self.network = ring_small_world(self.N, int(self.params.get('avg_degree', 8)), float(self.params.get('network_rewiring_prob', 0.1)), self.rng)

        # Initialize containers
        self.people: List[Person] = []
        self.households: List[Household] = []
        self.locations: List[Location] = []

        # FIXED: Replaced HealthAuthority with PolicyAuthority with scope and enforcement probability
        self.authority = PolicyAuthority(
            id=1,
            mandate_status=False,
            mandate_start_day=int(self.params.get('policy_mandate_day', 20)),
            mandate_scope=str(self.params.get('mandate_scope', 'indoor_public')),
            enforcement_probability=float(self.params.get('enforcement_probability', 0.4)),
            fine_amount=float(self.params.get('fine_amount', 50.0)),
            communication_strategy=str(self.params.get('communication_strategy', 'risk_and_norms')),
        )

        # FIXED: Extended Media to multiple channels with misinformation
        self.media_channels: List[MediaChannel] = [
            MediaChannel(id=1, message_bias=1.0, credibility=0.6, reach=0.6, misinformation_rate=0.05, message_frequency=1.0),
            MediaChannel(id=2, message_bias=-0.6, credibility=0.5, reach=0.4, misinformation_rate=float(self.params.get('misinformation_rate', 0.2)), message_frequency=0.8),
        ]

        self.retailer = Retailer(
            inventory=int(self.params.get('retailer_initial_inventory', 10000 if not smoke else 1000)),
            restock_quantity=int(self.params.get('restock_quantity', 8000 if not smoke else 400)),
            restock_interval=int(self.params.get('restock_interval_days', 7)),
            price=float(self.params.get('mask_price', 1.0)),
            max_purchase_per_customer=int(self.params.get('max_purchase_per_customer', 10)),
            supply_variability=float(self.params.get('supply_variability', 0.1)),
        )
        self._build_entities()

        # Time series for metrics
        self.overall_adoption_series: List[float] = []
        self.adoption_by_loc_series: Dict[str, List[float]] = {}
        self.policy_violations_per_day: List[int] = []
        self.avg_perceived_risk_series: List[float] = []
        self.prev_loc_mask_rate: Dict[str, float] = {}  # FIXED: Added location-based observed norms

        # Market tracking
        self.total_purchased: int = 0  # retained legacy
        self.total_units_demanded: int = 0  # FIXED: Track demand
        self.total_units_supplied: int = 0  # FIXED: Track supply
        self.total_spent: float = 0.0  # FIXED: Track prices paid
        self.stockout_days: int = 0

        # Dynamic model plan structure (minimal to support evaluation)
        self.model_plan: Dict[str, Any] = {
            "evaluation_metrics": [
                "adoption_curve_convergence",
                "mandate_effect_direction",
                "bounded_stockouts"
            ],
        }

    def _validate_and_default_params(self) -> None:
        """
        Validate and clamp key parameters into valid ranges. Provide sensible defaults where necessary.

        Raises:
            ValueError: If critical parameters are invalid after clamping.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        # Probabilities
        prob_keys = [
            'policy_enforcement_strictness', 'campaign_intensity', 'authority_credibility',
            'observation_effect_size', 'enforcement_probability'
        ]
        for k in prob_keys:
            if k in self.params:
                self.params[k] = clamp(float(self.params[k]), 0.0, 1.0)
        # Non-negative values
        nonneg_keys = ['mask_price', 'fine_amount', 'restock_quantity', 'retailer_initial_inventory', 'max_purchase_per_customer']
        for k in nonneg_keys:
            if k in self.params:
                self.params[k] = max(0.0, float(self.params[k]))
        # Location mix distribution default
        if 'location_mix_distribution' not in self.params:
            self.params['location_mix_distribution'] = {
                'workplace': 0.3,
                'transit': 0.1,
                'retail': 0.15,
                'school': 0.1,
                'park': 0.35
            }
        # Contact rates by location default
        if 'contact_rate_by_location' not in self.params:
            self.params['contact_rate_by_location'] = {
                'workplace': 8,
                'transit': 12,
                'retail': 6,
                'school': 10,
                'park': 3
            }

    def _build_entities(self) -> None:
        """
        Construct initial Persons, Households, and Locations.

        Notes:
            - Individuals initialized with random attributes using seeded RNG.
            - Households created with a simple size distribution.
            - Locations include a variety of types with different enforcement and mask policies.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        init_inv = int(self.params.get('initial_inventory_per_person', 2))
        init_adopt = float(self.params.get('initial_adoption_rate', 0.2))
        peer_w = float(self.params.get('peer_influence_weight', 0.2))

        # Build households with sizes drawn from a simple distribution
        sizes = []
        remaining = self.N
        hh_id = 0
        while remaining > 0:
            size = min(remaining, max(1, int(self.rng.choice([1, 2, 3, 4, 5], ) if hasattr(self.rng, 'choice') else random.choice([1, 2, 3, 4, 5]))))
            # emulate weighted: more 2-4 sized by random tweak
            if self.rng.random() < 0.5:
                size = min(remaining, max(1, int(self.rng.gauss(3, 1))))
            sizes.append(size)
            remaining -= size
        # Construct persons and assign to households
        person_idx = 0
        for size in sizes:
            member_ids = []
            hid = hh_id
            for _ in range(size):
                if person_idx >= self.N:
                    break
                income = self.rng.uniform(15000, 150000)
                risk = clamp(self.rng.random(), 0.0, 1.0)
                trust = clamp(self.rng.random(), 0.0, 1.0)
                susc = clamp(self.rng.random(), 0.0, 1.0)
                mis_susc = clamp(self.rng.gauss(0.6, 0.15), 0.0, 1.0)
                attitude = clamp(self.rng.gauss(0, 0.5), -1.0, 1.0)
                adopt = self.rng.random() < init_adopt
                inv = int(init_inv)
                age_group = self.rng.choice(['child', 'adult', 'senior']) if hasattr(self.rng, 'choice') else random.choice(['child', 'adult', 'senior'])
                health_risk_level = clamp(self.rng.gauss(0.5, 0.2), 0.0, 1.0)
                budget = self.rng.uniform(20, 300)
                cost_sensitivity = clamp(self.rng.gauss(0.5, 0.2), 0.0, 1.0)
                person = Person(
                    id=person_idx,
                    income=income,
                    risk_perception=risk,
                    trust_in_authorities=trust,
                    susceptibility_to_messaging=susc,
                    misinformation_susceptibility=mis_susc,
                    social_influence_weight=peer_w,
                    mask_attitude=attitude,
                    mask_adoption_state=adopt,
                    compliance_probability=0.5,
                    access_to_masks=inv > 0,
                    inventory_masks=inv,
                    cost_sensitivity=cost_sensitivity,
                    budget=budget,
                    adoption_score=1.0 if adopt else 0.0,
                    age_group=age_group,
                    health_risk_level=health_risk_level,
                    household_id=hid,
                    network_neighbors=[],  # filled later
                    past_enforcement_events=0,
                )
                self.people.append(person)
                member_ids.append(person_idx)
                person_idx += 1
            self.households.append(Household(id=hid, member_ids=member_ids, norm_strength=0.3))
            hh_id += 1

        # Attach network neighbors
        for i, p in enumerate(self.people):
            p.network_neighbors = self.network.get(i, [])

        # Build a set of diverse locations
        location_mix = self.params.get('location_mix_distribution', {})
        loc_types_config = [
            ('workplace', True, 0.6, 0.6),
            ('transit', True, 0.7, 0.7),
            ('retail', True, 0.6, 0.7),
            ('school', True, 0.5, 0.7),
            ('park', False, 0.1, 0.4),
        ]
        lid = 0
        for ltype, req, enf, obs in loc_types_config:
            share = location_mix.get(ltype, 0.2)
            count = max(1, int(share * 20))
            cap = int(50 + 100 * share)
            for _ in range(count):
                self.locations.append(
                    Location(
                        id=lid,
                        type=ltype,
                        capacity=cap,
                        mask_requirement_policy=req,
                        enforcement_strictness=enf,
                        foot_traffic=share,
                        observability_factor=obs,
                    )
                )
                lid += 1

    def _message_effect(self, person: Person) -> None:
        """
        Apply the effects of PolicyAuthority communication and MediaChannel exposures on an individual's risk
        and attitude, including misinformation influence and individual susceptibility.

        Parameters:
            person (Person): The individual to update.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        m_eff = float(self.params.get('message_effect_size', 0.15))
        # Policy/authority communication: increase perceived risk and trust modestly
        comm_intensity = float(self.params.get('campaign_intensity', 0.7))
        auth_cred = float(self.params.get('authority_credibility', 0.7))
        delta_risk = comm_intensity * m_eff * person.susceptibility_to_messaging * auth_cred
        person.risk_perception = clamp(person.risk_perception + delta_risk, 0.0, 1.0)
        person.trust_in_authorities = clamp(person.trust_in_authorities + 0.1 * comm_intensity * auth_cred, 0.0, 1.0)

        # Media channels: pro and anti effects; misinformation reduces risk and attitude depending on susceptibility
        for ch in self.media_channels:
            exposed = (self.rng.random() < (ch.reach * ch.message_frequency))
            if not exposed:
                continue
            mis = (self.rng.random() < ch.misinformation_rate)
            if mis:
                # misinformation effect scaled by misinformation susceptibility
                mis_effect = m_eff * (0.5 + 0.5 * person.misinformation_susceptibility) * ch.credibility
                person.risk_perception = clamp(person.risk_perception - 0.5 * mis_effect, 0.0, 1.0)
                person.mask_attitude = clamp(person.mask_attitude - 0.4 * mis_effect, -1.0, 1.0)
            else:
                # truthful/pro-mask leaning exposure
                media_delta = ch.credibility * m_eff * person.susceptibility_to_messaging * ch.message_bias
                person.mask_attitude = clamp(person.mask_attitude + 0.5 * media_delta, -1.0, 1.0)
                person.risk_perception = clamp(person.risk_perception + 0.3 * media_delta, 0.0, 1.0)

    def _peer_influence(self, person: Person) -> None:
        """
        Update mask attitude based on neighbors' adoption state.

        Parameters:
            person (Person): The individual to update.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        neighbors = person.network_neighbors
        if not neighbors:
            return
        neigh_adopt = sum(1 for j in neighbors if self.people[j].mask_adoption_state)
        share = neigh_adopt / max(1, len(neighbors))
        # Move attitude towards +1 as neighbor adoption increases above 0.5
        person.mask_attitude = clamp(person.mask_attitude + person.social_influence_weight * (share - 0.5), -1.0, 1.0)

    def _reinforce_household_norms(self, household: Household) -> None:
        """
        Apply within-household norm reinforcement by nudging member attitudes towards the household mean.

        Parameters:
            household (Household): The household to process.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        if not household.member_ids:
            return
        # Mean adoption of household
        adopt_vals = [1.0 if self.people[i].mask_adoption_state else 0.0 for i in household.member_ids]
        mean_adopt = sum(adopt_vals) / max(1, len(adopt_vals))
        # Nudge attitudes towards consistency with household behavior
        for pid in household.member_ids:
            p = self.people[pid]
            target = 2 * mean_adopt - 1.0  # map [0,1] -> [-1,1]
            p.mask_attitude = clamp(p.mask_attitude + household.norm_strength * 0.1 * (target - p.mask_attitude), -1.0, 1.0)

    def _share_masks_among_members(self, household: Household) -> None:
        """
        Share masks among household members to minimize zero-inventory cases,
        moving single units from members with >1 inventory to those with 0 up to 1.

        Parameters:
            household (Household): The household from which to share.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        if not household.member_ids:
            return
        members = household.member_ids[:]
        # Gather needs and donors
        needy = [pid for pid in members if self.people[pid].inventory_masks == 0]
        donors = [pid for pid in members if self.people[pid].inventory_masks > 1]
        # Shuffle for fairness
        self.rng.shuffle(needy)
        self.rng.shuffle(donors)
        for n_pid in needy:
            if not donors:
                break
            d_pid = donors[0]
            if self.people[d_pid].inventory_masks > 1:
                self.people[d_pid].inventory_masks -= 1
                self.people[n_pid].inventory_masks += 1
                # Re-evaluate donor eligibility
                if self.people[d_pid].inventory_masks <= 1:
                    donors.pop(0)
            else:
                donors.pop(0)

    def _decide_wear(self, person: Person, mandate: bool, loc_enf: float, obs_factor: float, loc_type: str) -> bool:
        """
        Decide whether the person wears a mask during a visit considering risk, policy, enforcement, and observed norms.

        Parameters:
            person (Person): The individual.
            mandate (bool): Whether a mask mandate applies in this location.
            loc_enf (float): Location enforcement strictness.
            obs_factor (float): Observability factor contributing to perceived social norms.
            loc_type (str): The type of location.

        Returns:
            bool: True if the person chooses to wear a mask.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        # Base propensity from risk and attitude
        attitude_scaled = (person.mask_attitude + 1) / 2  # [-1,1] -> [0,1]
        base = 0.5 * (person.risk_perception + 0.5 * attitude_scaled)

        # Policy effect
        policy = 0.0
        if mandate:
            policy = 0.3 * float(self.params.get('policy_enforcement_strictness', 0.6)) + 0.2 * sigmoid(self.authority.fine_amount / 100.0)

        # Individual traits: trust and past enforcement
        policy += 0.2 * (person.trust_in_authorities - 0.5)
        policy += 0.05 * person.past_enforcement_events

        # Observed norm for location type
        norm_obs = self.prev_loc_mask_rate.get(loc_type, 0.5)
        norm_term = float(self.params.get('observation_effect_size', 0.12)) * obs_factor * (norm_obs - 0.5)
        p = clamp(base + policy + 0.2 * loc_enf + norm_term, 0.0, 1.0)
        return person.inventory_masks > 0 and (self.rng.random() < p)

    def _purchase(self, person: Person) -> int:
        """
        Attempt to purchase masks if inventory is below threshold using affordability and cost sensitivity.

        Parameters:
            person (Person): The individual.

        Returns:
            int: Number of masks purchased.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        # Buy if below threshold and utility exceeds purchase_threshold
        if person.inventory_masks >= 1:
            return 0
        desired = 2
        price = self.retailer.price
        affordability = clamp(person.budget / max(price, 1e-6), 0.0, 100.0)
        disutility = person.cost_sensitivity * (price / float(self.params.get('mask_price', 1.0)))
        threshold = float(self.params.get('purchase_threshold', 0.6))
        # FIXED: Track demand regardless of supply outcome
        self.total_units_demanded += desired
        purchase_score = sigmoid(affordability - disutility)
        if purchase_score < threshold or self.retailer.inventory <= 0:
            return 0
        sold = min(desired, self.retailer.inventory, self.retailer.max_purchase_per_customer)
        self.retailer.inventory -= sold
        person.inventory_masks += sold
        person.budget = max(0.0, person.budget - sold * price)
        self.total_units_supplied += sold
        self.total_spent += sold * price
        return sold

    def _is_indoor_public(self, loc_type: str) -> bool:
        """
        Determine if a location type qualifies as indoor public per mandate_scope.

        Parameters:
            loc_type (str): Location type.

        Returns:
            bool: True if considered indoor public.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        return loc_type in {'workplace', 'transit', 'retail', 'school'}

    def step(self, day: int) -> None:
        """
        Execute one simulation day: policy update, messaging, restock, household sharing, visits, enforcement, and metrics.

        Parameters:
            day (int): Current day index.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        # Update policy state
        self.authority.mandate_status = (day >= self.authority.mandate_start_day)

        # Messaging and peer/household influence
        for p in self.people:
            self._message_effect(p)

        for hh in self.households:
            self._reinforce_household_norms(hh)

        for p in self.people:
            self._peer_influence(p)

        # Retail restock
        added = self.retailer.restock(day, self.rng)
        _ = added  # reserved for potential use

        # Stockout tally for today before sales (we also allow later count if remains zero)
        if self.retailer.inventory == 0:
            self.stockout_days += 1

        # Household sharing before visits
        for hh in self.households:
            self._share_masks_among_members(hh)

        # Track counts for metrics
        violations = 0
        total_attend = 0
        total_masked = 0
        by_type_counts: Dict[str, Tuple[int, int]] = {}  # type -> (masked, total)
        for loc in self.locations:
            by_type_counts.setdefault(loc.type, (0, 0))

        # Visits and decisions
        for p in self.people:
            # Purchase attempt before attending
            purchased = self._purchase(p)
            self.total_purchased += purchased  # legacy count

            wore_any = False  # FIXED: Track if person actually wore a mask today
            # Sample number of visits and which locations based on location mix distribution
            visit_prob = float(self.params.get('daily_outing_probability', 0.6))
            visits_today = 1 if self.rng.random() < visit_prob else 0
            if visits_today > 0:
                # Choose a location type weighted by mix distribution
                mix = self.params.get('location_mix_distribution', {})
                types = list(mix.keys())
                weights = [max(1e-6, mix[t]) for t in types]
                # Normalize weights
                total_w = sum(weights)
                weights = [w / total_w for w in weights] if total_w > 0 else [1.0 / len(weights)] * len(weights)
                # Draw a location type
                r = self.rng.random()
                cum = 0.0
                chosen_type = types[0]
                for t, w in zip(types, weights):
                    cum += w
                    if r <= cum:
                        chosen_type = t
                        break
                # Choose a specific location of that type
                locs_of_type = [loc for loc in self.locations if loc.type == chosen_type]
                if not locs_of_type:
                    locs_of_type = self.locations
                loc = self.rng.choice(locs_of_type) if hasattr(self.rng, 'choice') else random.choice(locs_of_type)
                mandate_here = False
                if self.authority.mandate_status:
                    if self.authority.mandate_scope == 'indoor_public':
                        mandate_here = self._is_indoor_public(loc.type) and loc.mask_requirement_policy
                    else:
                        mandate_here = loc.mask_requirement_policy

                wear = self._decide_wear(p, mandate_here, loc.enforcement_strictness, loc.observability_factor, loc.type)
                masked, tot = by_type_counts[loc.type]
                if wear:
                    wore_any = True
                    total_masked += 1
                    masked += 1
                    if p.inventory_masks > 0:
                        # Consume a mask unit for the visit
                        p.inventory_masks -= 1
                else:
                    # Potential enforcement under mandate
                    if mandate_here and (self.rng.random() < (self.authority.enforcement_probability * loc.enforcement_strictness)):
                        violations += 1
                        p.past_enforcement_events += 1
                tot += 1
                total_attend += 1
                by_type_counts[loc.type] = (masked, tot)
                # Update access flag
                p.access_to_masks = p.inventory_masks > 0

            # FIXED: Tie adoption state to actual wearing via rolling adoption score
            p.adoption_score = 0.8 * getattr(p, 'adoption_score', 0.0) + 0.2 * (1.0 if wore_any else 0.0)
            p.mask_adoption_state = p.adoption_score >= 0.6

        # Stockout tally if inventory ended zero after day
        if self.retailer.inventory == 0:
            # Count this day again only if it was not counted earlier; to avoid double counting
            # we conservatively keep as-is since we already marked at start-of-day.
            pass

        # Aggregate metrics for today
        overall = sum(1 for p in self.people if p.mask_adoption_state) / max(1, self.N)
        self.overall_adoption_series.append(overall)

        # FIXED: Update observed norms for next day decisions by location type
        self.prev_loc_mask_rate = {}
# FIXED: Applied feedback snippet from simulation_code_iter_65.py
        # Update observed norms for next day decisions
        self.prev_loc_mask_rate = {}
        for t, (m, tot) in by_type_counts.items():
            rate = (m / tot) if tot > 0 else 0.0
            self.adoption_by_loc_series.setdefault(t, []).append(rate)
            self.prev_loc_mask_rate[t] = rate if tot > 0 else self.prev_loc_mask_rate.get(t, 0.5)
            self.prev_loc_mask_rate[t] = rate if tot > 0 else self.prev_loc_mask_rate.get(t, 0.5)

        # Average perceived risk
        mean_risk = sum(p.risk_perception for p in self.people) / max(1, len(self.people))
        self.avg_perceived_risk_series.append(mean_risk)

        # Policy violations
        self.policy_violations_per_day.append(violations)

    def run(self) -> None:
        """
        Run the simulation over the configured horizon.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        for d in range(self.days):
            self.step(d)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Compute and return the metrics required by the Task Specification and feedback.

        Returns:
            Dict[str, Any]: A dictionary of metrics including time series and scalars.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        # location-specific metrics
        adoption_by_location_type = {k: v for k, v in self.adoption_by_loc_series.items()}
        # Required metrics per spec
        time_to_50 = next((i for i, v in enumerate(self.overall_adoption_series) if v >= 0.5), None)
        peak_adoption = max(self.overall_adoption_series) if self.overall_adoption_series else 0.0
        stockout_frequency = self.stockout_days / max(1, self.days)
        enforcement_events = sum(self.policy_violations_per_day)
        average_price_paid = (self.total_spent / max(1, self.total_units_supplied)) if self.total_units_supplied > 0 else 0.0
        mask_demand_supply_ratio = (self.total_units_demanded / max(1, self.total_units_supplied)) if self.total_units_supplied > 0 else float('inf')
        avg_perceived_risk_series = self.avg_perceived_risk_series

        # Adoption disparity index by income quartiles
        incomes = [p.income for p in self.people]
        final_adopt = [1.0 if p.mask_adoption_state else 0.0 for p in self.people]
        q = sorted(zip(incomes, final_adopt), key=lambda x: x[0])
        n = len(q)
        q1 = sum(ad for _, ad in q[:max(1, n // 4)]) / max(1, n // 4)
        q4 = sum(ad for _, ad in q[-max(1, n // 4):]) / max(1, n // 4)
        adoption_disparity_index = q4 - q1

        metrics = {
            "overall_adoption_rate": self.overall_adoption_series,
            "time_to_50_percent_adoption": time_to_50,
            "peak_adoption_rate": peak_adoption,
            "location_specific_adoption": adoption_by_location_type,
            "adoption_disparity_index": adoption_disparity_index,
            "enforcement_events": enforcement_events,
            "stockout_frequency": stockout_frequency,
            "average_price_paid": average_price_paid,
            "mask_demand_supply_ratio": mask_demand_supply_ratio,
            "average_perceived_risk": avg_perceived_risk_series,
        }
        return metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the run against evaluation metrics if configured in the model plan.

        Supported metrics:
            - adoption_curve_convergence: mean absolute change over the final 14 days < 0.005
            - mandate_effect_direction: post-mandate 14-day mean minus pre-mandate 14-day mean > 0.0
            - bounded_stockouts: stockout days fraction < 0.2

        Returns:
            Dict[str, Any]: Evaluation results keyed by metric name with value and pass/fail flag.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        results: Dict[str, Any] = {}
        evals = self.model_plan.get("evaluation_metrics", [])
        series = self.overall_adoption_series

        if "adoption_curve_convergence" in evals:
            window = min(14, len(series) - 1) if len(series) > 1 else 0
            diffs = [abs(series[i] - series[i - 1]) for i in range(len(series) - window + 1, len(series))] if window > 0 else []
            mean_change = sum(diffs) / max(1, len(diffs))
            results["adoption_curve_convergence"] = {"value": mean_change, "pass": mean_change < 0.005}

        if "mandate_effect_direction" in evals:
            mday = self.authority.mandate_start_day
            pre = series[max(0, mday - 14):mday] if mday > 0 else []
            post = series[mday:mday + 14] if mday < len(series) else []
            pre_mean = sum(pre) / max(1, len(pre))
            post_mean = sum(post) / max(1, len(post))
            results["mandate_effect_direction"] = {"value": post_mean - pre_mean, "pass": (post_mean - pre_mean) > 0.0}

        if "bounded_stockouts" in evals:
            frac = self.stockout_days / max(1, self.days)
            results["bounded_stockouts"] = {"value": frac, "pass": frac < 0.2}

        return results

    def visualize(self) -> None:
        """
        Visualize basic time series of overall adoption and policy violations.

        Notes:
            - Attempts to use matplotlib if available; otherwise prints a message.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("Visualization skipped (matplotlib not available):", e)
            return

        days = list(range(len(self.overall_adoption_series)))
        fig, ax1 = plt.subplots()
        ax1.plot(days, self.overall_adoption_series, label="Overall Adoption", color="blue")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Adoption Rate", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(days, self.policy_violations_per_day, label="Policy Violations", color="red", alpha=0.6)
        ax2.set_ylabel("Violations", color="red")
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Mask Adoption and Policy Violations Over Time")
        fig.tight_layout()
        plt.show()

    def save_results(self, filename: str) -> None:
        """
        Save primary daily time series to a CSV file.

        Parameters:
            filename (str): Output CSV filename.
        """
        pass  # FIXED: Retained 'pass' per interface requirement
        fieldnames = ["day", "overall_adoption_rate", "policy_violations", "avg_perceived_risk"]
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.overall_adoption_series)):
                    writer.writerow({
                        "day": i,
                        "overall_adoption_rate": self.overall_adoption_series[i],
                        "policy_violations": self.policy_violations_per_day[i] if i < len(self.policy_violations_per_day) else 0,
                        "avg_perceived_risk": self.avg_perceived_risk_series[i] if i < len(self.avg_perceived_risk_series) else 0.0,
                    })
        except Exception as e:
            print("Error saving results:", e)


def build_default_params(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Construct default simulation parameters from CLI arguments, mapping specification-aligned names.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Dict[str, Any]: Parameter dictionary for Simulation.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    params = {
        "population_size": args.population,
        "avg_degree": 8,
        "network_rewiring_prob": 0.1,
        "initial_adoption_rate": 0.2,
        "initial_inventory_per_person": 2,
        "policy_mandate_day": 20,
        "policy_enforcement_strictness": 0.6,
        "fine_amount": 50.0,
        "campaign_intensity": 0.7,
        "message_effect_size": 0.15,
        "observation_effect_size": 0.12,
        "peer_influence_weight": 0.2,
        "risk_perception_sensitivity": 0.3,
        "purchase_threshold": 0.6,  # FIXED: Added
        "misinformation_rate": 0.2,  # FIXED: Added
        "retailer_initial_inventory": 10000,
        "restock_interval_days": 7,
        "restock_quantity": 8000,
        "mask_price": 1.0,
        "max_purchase_per_customer": 10,
        "simulation_horizon_days": args.days,
        "time_step_days": 1,
        "random_seed": args.seed,
        "supply_variability": 0.1,
        "authority_credibility": 0.7,
        "mandate_scope": "indoor_public",  # FIXED: Added scope
        "enforcement_probability": 0.4,  # FIXED: Added enforcement probability
        "communication_strategy": "risk_and_norms",
        "daily_outing_probability": 0.6,
        "location_mix_distribution": {
            "workplace": 0.3,
            "transit": 0.1,
            "retail": 0.15,
            "school": 0.1,
            "park": 0.35
        },
        "contact_rate_by_location": {
            "workplace": 8,
            "transit": 12,
            "retail": 6,
            "school": 10,
            "park": 3
        }
    }
    return params


def main() -> None:
    """
    Program entry point: parse arguments, run simulation, visualize, and save results.

    Behavior:
        - Supports a fast --smoke mode for CI.
        - Writes metrics.json with required metrics.
        - Saves CSV series and optionally visualizes results.

    Notes:
        - Pure-Python; no Docker dependency.
    """
    pass  # FIXED: Retained 'pass' per interface requirement
    parser = argparse.ArgumentParser(description="Mask Adoption Simulation with Households, Policy Scope, and Misinformation (pure Python)")
    parser.add_argument("--days", type=int, default=120, help="Simulation horizon in days")
    parser.add_argument("--population", type=int, default=5000, help="Population size")
    parser.add_argument("--smoke", action="store_true", help="Run a small fast simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--metrics-file", type=str, default="metrics.json", help="Output metrics JSON filename")
    parser.add_argument("--results-file", type=str, default="results.csv", help="Output results CSV filename")
    args = parser.parse_args()

    params = build_default_params(args)
    sim = Simulation(params, smoke=args.smoke)
    sim.run()

    metrics = sim.get_metrics()
    try:
        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        print("Error writing metrics.json:", e)

    # Print brief summary (truncate lists in smoke mode)
    preview_metrics = {k: (v if not isinstance(v, list) else (v[:5] if args.smoke else v)) for k, v in metrics.items()}
    print(json.dumps(preview_metrics, indent=2))

    # Evaluate and print evaluation metrics
    evaluation = sim.evaluate()
    print("Evaluation:", json.dumps(evaluation, indent=2))

    # Save CSV results
    sim.save_results(args.results_file)

    # Visualize
    if not args.no_visualize:
        sim.visualize()


# Execute main for both direct execution and sandbox wrapper invocation
main()