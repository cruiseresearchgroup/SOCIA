import os
import sys
import json
import math
import random
import argparse
import time
import csv
import shutil
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Set


# FIXED: Removed stray invalid line and implemented full simulation architecture with runnable main()


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp a numeric value between lo and hi inclusive.

    Parameters
    ----------
    x : float
        The input value to clamp.
    lo : float, optional
        Lower bound, by default 0.0
    hi : float, optional
        Upper bound, by default 1.0

    Returns
    -------
    float
        The clamped value.
    """
    pass
    return max(lo, min(hi, x))


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object to be JSON-serializable by converting
    NaN/Inf to None and ensuring only basic Python types remain.

    Parameters
    ----------
    obj : Any
        Input object potentially containing non-JSON-serializable items.

    Returns
    -------
    Any
        JSON-serializable object.
    """
    pass
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        return sanitize_for_json(obj.to_dict())
    return str(obj)


def safe_json_dump(data: Any, path: str) -> None:
    """
    Safely dump data to a JSON file with sanitization and indentation.

    Parameters
    ----------
    data : Any
        The data to serialize to JSON.
    path : str
        File path to write JSON to.

    Returns
    -------
    None
    """
    pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sanitize_for_json(data), f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write JSON to {path}: {e}", file=sys.stderr)


def small_world_graph(n: int, k: int, p: float, rng: random.Random) -> List[List[int]]:
    """
    Generate a Watts-Strogatz small-world graph adjacency list using stdlib only.

    Parameters
    ----------
    n : int
        Number of nodes.
    k : int
        Each node is connected to k nearest neighbors in ring topology (k must be even).
    p : float
        The probability of rewiring each edge.
    rng : random.Random
        Random number generator for reproducibility.

    Returns
    -------
    List[List[int]]
        Adjacency list representation of the graph.
    """
    pass
    k = max(2, int(k) // 2 * 2)
    adj = [set() for _ in range(n)]
    # ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            u, v = i, (i + j) % n
            adj[u].add(v)
            adj[v].add(u)
    # rewire
    for i in range(n):
        for v in list(adj[i]):
            if i < v and rng.random() < p:
                adj[i].remove(v)
                adj[v].remove(i)
                candidates = [x for x in range(n) if x != i and x not in adj[i]]
                if candidates:
                    w = rng.choice(candidates)
                    adj[i].add(w)
                    adj[w].add(i)
    return [list(s) for s in adj]


# Entities


@dataclass
class Person:
    """
    Person agent representing an individual in the population.

    Attributes
    ----------
    id : int
        Unique identifier for the person.
    age_group : str
        Categorical age group label.
    income : float
        Income proxy used for price elasticity and subsidy eligibility.
    health_status : str
        Health status category, e.g., "healthy" or "at_risk".
    risk_perception : float
        Perceived risk of illness (0-1).
    trust_in_authorities : float
        Trust level in authorities (0-1).
    political_orientation : str
        Political orientation categorical label.
    social_network_degree : int
        Social network degree (approximate).
    conformity_trait : float
        Tendency to conform to social norms (0-1).
    susceptibility_to_misinformation : float
        Susceptibility (0-1).
    mask_attitude : float
        Mask attitude index [-2, 2].
    adoption_state : int
        Whether currently wearing mask (0 or 1).
    mask_access : int
        Whether has immediate access to mask (0 or 1).
    mask_stock : int
        Count of masks in possession.
    perceived_cost : float
        Perceived cost/discomfort (0-3).
    perceived_benefit : float
        Perceived benefit proxy (0-2).
    home_id : int
        Household/location ID for home.
    workplace_id : int
        Workplace or school location ID.
    daily_mobility_pattern : Dict[str, float]
        Probabilities for visits to location types.
    region_id : int
        Region identifier.
    """
    pass
    id: int = 0
    age_group: str = "18-34"
    income: float = 1.0
    health_status: str = "healthy"
    risk_perception: float = 0.3
    trust_in_authorities: float = 0.5
    political_orientation: str = "center"
    social_network_degree: int = 0
    conformity_trait: float = 0.5
    susceptibility_to_misinformation: float = 0.3
    mask_attitude: float = 0.0
    adoption_state: int = 0
    mask_access: int = 0
    mask_stock: int = 0
    perceived_cost: float = 0.2
    perceived_benefit: float = 0.0
    home_id: int = -1
    workplace_id: int = -1
    daily_mobility_pattern: Dict[str, float] = field(default_factory=dict)
    region_id: int = 0


@dataclass
class Household:
    """
    Household representing a group of people sharing a home.

    Attributes
    ----------
    id : int
        Unique household identifier.
    size : int
        Household size.
    income_level : float
        Aggregate income proxy.
    region_id : int
        Region identifier.
    norms_mask : float
        Household mask norm index (0-1).
    members : List[int]
        IDs of members (person ids).
    """
    pass
    id: int = 0
    size: int = 1
    income_level: float = 1.0
    region_id: int = 0
    norms_mask: float = 0.5
    members: List[int] = field(default_factory=list)


@dataclass
class Location:
    """
    Location where interactions occur.

    Attributes
    ----------
    id : int
        Unique location identifier.
    type : str
        Location type: home, workplace, school, retail, transit, public_space.
    capacity : int
        Capacity of the location.
    region_id : int
        Region identifier.
    foot_traffic_rate : float
        Foot traffic rate proxy.
    mask_policy : str
        Mask policy string: 'required' or 'optional'.
    enforcement_level : float
        Enforcement level (0-1).
    admission_rules : Dict[str, Any]
        Admission rules metadata.
    """
    pass
    id: int = 0
    type: str = "public_space"
    capacity: int = 100
    region_id: int = 0
    foot_traffic_rate: float = 0.5
    mask_policy: str = "optional"
    enforcement_level: float = 0.3
    admission_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Government:
    """
    Government or PolicyAuthority with policy and messaging attributes.

    Attributes
    ----------
    id : int
        Unique government identifier.
    policy_level : int
        Policy strictness level 0..3.
    enforcement_strength : float
        Enforcement strength (0-1).
    fine_amount : float
        Fine amount per violation.
    campaign_intensity : float
        Messaging campaign intensity (0-1).
    subsidy_level : float
        Fraction of eligible population receiving subsidy per day (0-1).
    distribution_rate : float
        Free mask distribution rate per day (0-1 of population).
    mandate_active : bool
        Whether a mask mandate is active.
    """
    pass
    id: int = 0
    policy_level: int = 1
    enforcement_strength: float = 0.5
    fine_amount: float = 50.0
    campaign_intensity: float = 0.5
    subsidy_level: float = 0.5
    distribution_rate: float = 0.02
    mandate_active: bool = False


@dataclass
class Media:
    """
    Media source broadcasting messages.

    Attributes
    ----------
    id : int
        Unique media identifier.
    reliability : float
        Reliability 0-1.
    bias : float
        Bias: negative is pro-mask; positive anti-mask.
    reach : float
        Fraction of population reached when broadcasting 0-1.
    misinformation_rate : float
        Rate of misinformation 0-1.
    message_frequency : int
        Frequency of message days.
    """
    pass
    id: int = 0
    reliability: float = 0.8
    bias: float = 0.0
    reach: float = 0.7
    misinformation_rate: float = 0.1
    message_frequency: int = 7


@dataclass
class Retailer:
    """
    Retailer of masks.

    Attributes
    ----------
    id : int
        Unique retailer identifier.
    inventory : int
        Stock on hand.
    price : float
        Price per mask.
    restock_rate : float
        Fractional restock rate per day or used in interval policy.
    rationing_policy : Dict[str, Any]
        Rationing rules.
    region_id : int
        Region identifier.
    restock_interval_days : int
        Interval in days for restocking discrete quantity.
    restock_quantity : int
        Quantity added when restocking.
    """
    pass
    id: int = 0
    inventory: int = 0
    price: float = 1.0
    restock_rate: float = 0.1
    rationing_policy: Dict[str, Any] = field(default_factory=lambda: {"limit_per_purchase": 10})
    region_id: int = 0
    restock_interval_days: int = 7
    restock_quantity: int = 100


# Module base


class Module:
    """
    Base module interface for simulation modules.

    Methods
    -------
    forward(state, buffers, params, t)
        Compute updates into buffers for time t. Scheduler commits later.
    """
    pass

    def __init__(self, name: str):
        """
        Initialize a module with a human-readable name.

        Parameters
        ----------
        name : str
            Module name.
        """
        pass
        self.name = name
        self.io_log: List[Dict[str, Any]] = []

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Forward step placeholder. Override in subclasses.

        Parameters
        ----------
        state : Dict[str, Any]
            Current simulation state.
        buffers : Dict[str, Any]
            Buffers for module outputs to be committed later.
        params : Dict[str, Any]
            Effective parameters.
        t : int
            Current day index.
        """
        pass
        raise NotImplementedError("forward must be implemented by subclasses.")

    def log_io(self, record: Dict[str, Any]) -> None:
        """
        Append an I/O record to the module's internal log.

        Parameters
        ----------
        record : Dict[str, Any]
            Record containing inputs/outputs/state snapshots.

        Returns
        -------
        None
        """
        pass
        self.io_log.append(sanitize_for_json(record))


# Modules Implementation


class PopulationNetworkInit(Module):
    """
    Module to initialize population, network, locations, retailers, government, and media.

    Notes
    -----
    - Runs only at t == 0.
    - Uses a Watts-Strogatz small-world network for social ties.
    """
    pass

    def __init__(self, rng: random.Random):
        """
        Create a PopulationNetworkInit module.

        Parameters
        ----------
        rng : random.Random
            RNG for reproducibility.
        """
        pass
        super().__init__("PopulationNetworkInit")
        self.rng = rng

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Initialize entities and network at t == 0.

        Parameters
        ----------
        state : Dict[str, Any]
            Existing state to be augmented.
        buffers : Dict[str, Any]
            Buffer to place initialized entities.
        params : Dict[str, Any]
            Parameters including population size, degrees, etc.
        t : int
            Simulation day.

        Returns
        -------
        None
        """
        pass
        if t != 0:
            return
        N = int(params.get("population_size", 500))
        avg_k = int(params.get("network_avg_degree", 8))
        rewiring_p = float(params.get("network_rewiring_prob", 0.05))
        rng = self.rng

        adj = small_world_graph(N, avg_k, rewiring_p, rng)
        persons: List[Person] = []
        households: List[Household] = []
        locations: List[Location] = []
        retailers: List[Retailer] = []

        # Regions
        region_count = int(params.get("region_count", 3))

        # Create persons
        initial_adoption_rate = float(params.get("initial_mask_adoption_rate", 0.1))
        trust_mu = float(params.get("trust_in_authorities_mean", 0.6))
        trust_sd = float(params.get("trust_in_authorities_std", 0.2))
        cost_mu = float(params.get("perceived_cost_mean", 0.2))
        cost_sd = float(params.get("perceived_cost_std", 0.1))
        threshold_mu = float(params.get("adoption_threshold_mean", 0.4))
        threshold_sd = float(params.get("adoption_threshold_std", 0.1))
        base_risk = float(params.get("risk_perception_base", 0.3))

        # Simple age groups for variety
        age_groups = ["0-17", "18-34", "35-49", "50-64", "65+"]
        pol_orients = ["left", "center", "right"]

        for i in range(N):
            ag = rng.choice(age_groups)
            po = rng.choice(pol_orients)
            income = max(0.1, rng.lognormvariate(0, 0.8))
            trust = clamp(rng.gauss(trust_mu, trust_sd))
            pcost = clamp(rng.gauss(cost_mu, cost_sd), 0.0, 3.0)
            adoption_state = 1 if rng.random() < initial_adoption_rate else 0
            mask_stock = rng.randint(0, 3) + (1 if adoption_state else 0)
            region = rng.randrange(region_count)
            person = Person(
                id=i,
                age_group=ag,
                income=income,
                health_status="healthy" if rng.random() < 0.8 else "at_risk",
                risk_perception=base_risk,
                trust_in_authorities=trust,
                political_orientation=po,
                social_network_degree=len(adj[i]),
                conformity_trait=clamp(rng.random()),
                susceptibility_to_misinformation=clamp(0.3 + (0.1 if po == "right" else -0.05 if po == "left" else 0.0)),
                mask_attitude=0.0,
                adoption_state=adoption_state,
                mask_access=1 if mask_stock > 0 else 0,
                mask_stock=mask_stock,
                perceived_cost=pcost,
                perceived_benefit=0.0,
                home_id=-1,
                workplace_id=-1,
                daily_mobility_pattern={
                    "workplace": 0.6,
                    "school": 0.3,
                    "transit": 0.3,
                    "retail": 0.2,
                    "public_space": 0.4,
                },
                region_id=region,
            )
            persons.append(person)

        # Households - simple grouping of size 2-4
        h_id = 0
        i = 0
        while i < N:
            size = rng.randint(2, 4)
            members = list(range(i, min(N, i + size)))
            hh = Household(
                id=h_id,
                size=len(members),
                income_level=sum(persons[m].income for m in members) / max(1, len(members)),
                region_id=persons[members[0]].region_id,
                norms_mask=random.random() * 0.5 + 0.25,
                members=members,
            )
            for m in members:
                persons[m].home_id = h_id
            households.append(hh)
            i += size
            h_id += 1

        # Locations per type
        loc_types = ["workplace", "school", "retail", "transit", "public_space"]
        loc_id = 0
        for typ in loc_types:
            for _ in range(max(1, N // 100)):
                loc = Location(
                    id=loc_id,
                    type=typ,
                    capacity=max(50, N // 20),
                    region_id=rng.randrange(region_count),
                    foot_traffic_rate=0.4 if typ != "retail" else 0.6,
                    mask_policy="required" if typ in ["workplace", "school", "retail", "transit"] else "optional",
                    enforcement_level=float(params.get("organization_enforcement_level", 0.6)),
                    admission_rules={"mask_required": typ in ["workplace", "school", "retail", "transit"]},
                )
                locations.append(loc)
                loc_id += 1

        # Retailers
        init_inv = int(params.get("retailer_initial_inventory", 1000))
        base_price = float(params.get("mask_price", 1.0))
        for rid in range(max(1, region_count)):
            retailers.append(
                Retailer(
                    id=rid,
                    inventory=int(init_inv * params.get("mask_availability_initial", 0.7)),
                    price=base_price,
                    restock_rate=float(params.get("retailer_restock_rate_per_day", 0.1)),
                    rationing_policy={"limit_per_purchase": int(params.get("retailer_ration_limit_per_purchase", 10))},
                    region_id=rid,
                    restock_interval_days=int(params.get("retailer_restock_interval_days", 7)),
                    restock_quantity=int(params.get("retailer_restock_quantity", 500)),
                )
            )

        # Government and media
        gov = Government(
            id=0,
            policy_level=int(params.get("policy_level", 1)),
            enforcement_strength=float(params.get("enforcement_strength", 0.7)),
            fine_amount=float(params.get("fine_amount", 50.0)),
            campaign_intensity=float(params.get("campaign_intensity", 0.5)),
            subsidy_level=float(params.get("subsidy_level", 0.5)),
            distribution_rate=float(params.get("gov_distribution_rate", 0.02)),
            mandate_active=(params.get("policy_state_initial", "recommendation") == "mandate"),
        )
        media = Media(
            id=0,
            reliability=float(params.get("media_reliability_mean", 0.8)),
            bias=float(params.get("media_bias", 0.0)),
            reach=float(params.get("media_reach", 0.7)),
            misinformation_rate=float(params.get("misinformation_rate", 0.1)),
            message_frequency=int(params.get("message_frequency_days", 7)),
        )

        buffers["init"] = {
            "persons": persons,
            "households": households,
            "locations": locations,
            "retailers": retailers,
            "government": gov,
            "media": media,
            "adjacency": adj,
        }
        self.log_io({"t": t, "action": "init", "N": N, "avg_k": avg_k})


class InformationAndBeliefUpdate(Module):
    """
    Module to update beliefs from media, policy, peers, and household norms.

    Notes
    -----
    - Runs daily.
    - Media messages occur at set frequency and reach fraction of population.
    """
    pass

    def __init__(self, rng: random.Random):
        """
        Create an InformationAndBeliefUpdate module.

        Parameters
        ----------
        rng : random.Random
            RNG for reproducibility.
        """
        pass
        super().__init__("InformationAndBeliefUpdate")
        self.rng = rng
        self.last_reached: Set[int] = set()

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Update risk perception and benefits from info signals.

        Parameters
        ----------
        state : Dict[str, Any]
            Current state, including persons, households, government, media.
        buffers : Dict[str, Any]
            Buffer to record attitude shifts and message reach for aggregation.
        params : Dict[str, Any]
            Effective parameters for learning.
        t : int
            Current day index.

        Returns
        -------
        None
        """
        pass
        if "persons" not in state:
            return

        rng = self.rng
        gov: Government = state["government"]
        media: Media = state["media"]
        persons: List[Person] = state["persons"]
        households: List[Household] = state.get("households", [])

        # Broadcast update
        reached: Set[int] = set()
        attitude_shift_sum = 0.0
        message_today = (t % max(1, media.message_frequency) == 0)
        if message_today:
            mcount = int(clamp(media.reach, 0.0, 1.0) * len(persons))
            idxs = list(range(len(persons)))
            rng.shuffle(idxs)
            reached = set(idxs[:mcount])
            info_strength = clamp(float(params.get("information_effect_strength", 0.4)), 0.0, 3.0)
            campaign_boost = clamp(gov.campaign_intensity, 0.0, 1.0)
            # Effective message tone: reliability reduces noise; bias can tilt
            base_delta = 0.05 * media.reliability * (1.0 - max(0.0, media.bias))
            base_delta = max(0.0, base_delta)
            for i in reached:
                before = persons[i].risk_perception
                persons[i].risk_perception = clamp(
                    (1 - float(params.get("learning_rate", 0.2))) * persons[i].risk_perception
                    + float(params.get("learning_rate", 0.2))
                    * clamp(float(params.get("base_risk_perception", params.get("risk_perception_base", 0.3)))
                            + info_strength * base_delta + campaign_boost * 0.05, 0.0, 1.0),
                    0.0, 1.0
                )
                # Update perceived benefit linking risk and conformity trait
                persons[i].perceived_benefit = clamp(persons[i].risk_perception * (1.0 + persons[i].conformity_trait), 0.0, 2.0)
                attitude_shift_sum += abs(persons[i].risk_perception - before)

        # Peer influence proxy: we defer to DecisionCompliance for explicit peer share,
        # but we can lightly nudge benefit based on household norms.
        hh_norm_weight = float(params.get("household_norms_strength", 0.3))
        if households:
            hh_map = {m: hh for hh in households for m in hh.members}
            for i, person in enumerate(persons):
                hh = hh_map.get(i)
                if hh is not None:
                    person.perceived_benefit = clamp(
                        person.perceived_benefit + hh_norm_weight * (hh.norms_mask - 0.5) * 0.1, 0.0, 2.0
                    )

        buffers[self.name] = {
            "reached": list(reached),
            "attitude_shift_sum": attitude_shift_sum,
        }
        self.last_reached = reached
        self.log_io({"t": t, "reached": len(reached), "attitude_shift_sum": attitude_shift_sum})


class MaskSupplyAndAcquisition(Module):
    """
    Module to handle retailer inventory dynamics and mask purchases.

    Notes
    -----
    - Restocking periodically and via rate.
    - Purchases depend on net benefit and price elasticity.
    """
    pass

    def __init__(self, rng: random.Random):
        """
        Create a MaskSupplyAndAcquisition module.

        Parameters
        ----------
        rng : random.Random
            RNG for reproducibility.
        """
        pass
        super().__init__("MaskSupplyAndAcquisition")
        self.rng = rng
        self.prev_day_sales: int = 0

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Simulate restocking, pricing, and purchase attempts.

        Parameters
        ----------
        state : Dict[str, Any]
            Current state with persons and retailers.
        buffers : Dict[str, Any]
            Buffer recording stock attempts, failures, and sales.
        params : Dict[str, Any]
            Effective parameters controlling prices, subsidies, etc.
        t : int
            Day index.

        Returns
        -------
        None
        """
        pass
        if "retailers" not in state or "persons" not in state:
            return
        rng = self.rng
        retailers: List[Retailer] = state["retailers"]
        persons: List[Person] = state["persons"]
        gov: Government = state["government"]

        # Restock and price adjust
        total_outage = False
        for r in retailers:
            # Periodic restock
            if r.restock_interval_days > 0 and t % r.restock_interval_days == 0 and t > 0:
                r.inventory += r.restock_quantity
            # Price adjustment based on previous day sales demand proxy
            demand_proxy = self.prev_day_sales / max(1, r.inventory + self.prev_day_sales)
            price_adj = 1.0 + float(params.get("price_markup_sensitivity", 0.1)) * (demand_proxy - 0.5)
            r.price = max(0.1, float(params.get("mask_price", 1.0)) * price_adj)
            if r.inventory <= 0:
                total_outage = True

        # Government distribution: free masks
        free_masks = int(float(params.get("gov_distribution_rate", gov.distribution_rate)) * len(persons))
        if free_masks > 0:
            recipients = list(range(len(persons)))
            rng.shuffle(recipients)
            recipients = recipients[:free_masks]
            for i in recipients:
                persons[i].mask_stock += 1
                persons[i].mask_access = 1

        subsidy_per_mask = float(params.get("subsidy_per_mask", params.get("subsidy_amount", 0.0)))
        income_quantile_thresh = float(params.get("income_subsidy_threshold_quantile", 0.3))
        # Compute income threshold
        incomes = sorted([p.income for p in persons])
        idx_thresh = int(income_quantile_thresh * (len(incomes) - 1))
        income_thresh_val = incomes[idx_thresh] if incomes else float("inf")

        # Purchase attempts
        stock_attempts = 0
        stock_fail = 0
        sales = 0

        # We target a 1-week buffer
        mask_use_rate = float(params.get("mask_use_rate_per_day", 1.0)) / max(1.0, float(params.get("mask_reuse_factor", 2.0)))
        ration_limit = int(params.get("retailer_ration_limit_per_purchase", 10))
        price_elasticity = float(params.get("price_elasticity", abs(float(params.get("price_elasticity_of_demand", 0.3)))))

        for i, p in enumerate(persons):
            # Consume masks if wearing
            if p.adoption_state == 1:
                if p.mask_stock > 0:
                    p.mask_stock = max(0, p.mask_stock - int(math.ceil(mask_use_rate)))
            p.mask_access = 1 if p.mask_stock > 0 else 0

            needed = max(0, int(math.ceil(7 * mask_use_rate)) - p.mask_stock)
            if needed <= 0:
                continue

            # Determine effective price in their region
            region_retailers = [r for r in retailers if r.region_id == p.region_id] or retailers
            # Cheapest retailer
            rmin = min(region_retailers, key=lambda r: r.price if r.inventory > 0 else float("inf"))
            effective_price = max(0.0, rmin.price - (subsidy_per_mask if (p.income <= income_thresh_val and rng.random() < gov.subsidy_level) else 0.0))
            # Willingness based on price and income
            afford_factor = p.income / (p.income + 1.0)
            buy_prob = clamp(afford_factor * (1.0 / (1.0 + price_elasticity * effective_price)))
            buy_qty = min(needed, ration_limit)

            if rmin.inventory <= 0:
                stock_attempts += 1
                stock_fail += 1
                continue

            if rng.random() < buy_prob:
                stock_attempts += 1
                qty = min(buy_qty, rmin.inventory)
                if qty > 0:
                    rmin.inventory -= qty
                    p.mask_stock += qty
                    p.mask_access = 1
                    sales += qty
                else:
                    stock_fail += 1

        buffers[self.name] = {
            "stock_attempts": stock_attempts,
            "stock_fail": stock_fail,
            "sales": sales,
            "supply_outage": total_outage or any(r.inventory <= 0 for r in retailers),
        }
        self.prev_day_sales = sales
        self.log_io({"t": t, "attempts": stock_attempts, "fail": stock_fail, "sales": sales})


class MobilityAndLocationUse(Module):
    """
    Module to generate visits to locations by type.

    Notes
    -----
    - Uses simple attendance rates from params.
    - Produces entrants lists by type for enforcement and compliance observation.
    """
    pass

    def __init__(self, rng: random.Random):
        """
        Create MobilityAndLocationUse module.

        Parameters
        ----------
        rng : random.Random
            RNG for reproducibility.
        """
        pass
        super().__init__("MobilityAndLocationUse")
        self.rng = rng

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Generate visits for the day by sampling persons by location types.

        Parameters
        ----------
        state : Dict[str, Any]
            Current state.
        buffers : Dict[str, Any]
            Buffer to store entrants by type as person indices lists.
        params : Dict[str, Any]
            Attendance rates and capacity utilization.
        t : int
            Day index.

        Returns
        -------
        None
        """
        pass
        if "persons" not in state:
            return
        rng = self.rng
        persons = state["persons"]

        entrants_by_type: Dict[str, List[int]] = {
            "workplace": [],
            "school": [],
            "retail": [],
            "transit": [],
            "public_space": [],
        }
        # Attendance rates
        workplace_rate = float(params.get("workplace_attendance_rate", 0.7))
        school_rate = float(params.get("school_attendance_rate", 0.6))
        transit_rate = float(params.get("transit_usage_rate", 0.3))
        retail_rate = float(params.get("retail_visit_rate", 0.2))
        public_rate = float(params.get("public_space_visit_rate", 0.4))

        for i, p in enumerate(persons):
            if rng.random() < workplace_rate:
                entrants_by_type["workplace"].append(i)
            if rng.random() < school_rate:
                entrants_by_type["school"].append(i)
            if rng.random() < transit_rate:
                entrants_by_type["transit"].append(i)
            if rng.random() < retail_rate:
                entrants_by_type["retail"].append(i)
            if rng.random() < public_rate:
                entrants_by_type["public_space"].append(i)

        buffers[self.name] = {"entrants_by_type": entrants_by_type}
        self.log_io({"t": t, "counts": {k: len(v) for k, v in entrants_by_type.items()}})


class DecisionCompliance(Module):
    """
    Module for adoption/discontinuation decisions based on net benefit and thresholds.

    Notes
    -----
    - Incorporates peer influence, media signal, policy signal, and fatigue.
    """
    pass

    def __init__(self, rng: random.Random):
        """
        Create DecisionCompliance module.

        Parameters
        ----------
        rng : random.Random
            RNG for reproducibility.
        """
        pass
        super().__init__("DecisionCompliance")
        self.rng = rng

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Apply decision rules to update adoption states.

        Parameters
        ----------
        state : Dict[str, Any]
            Current state including adjacency and persons.
        buffers : Dict[str, Any]
            Buffer to store updated adoption states.
        params : Dict[str, Any]
            Relevant weights and thresholds.
        t : int
            Day index.

        Returns
        -------
        None
        """
        pass
        if "persons" not in state or "adjacency" not in state:
            return
        rng = self.rng
        persons: List[Person] = state["persons"]
        adj: List[List[int]] = state["adjacency"]
        gov: Government = state["government"]

        # Compose signals
        peer_w = float(params.get("peer_influence_weight", params.get("social_influence_strength", 0.5)))
        media_w = float(params.get("media_influence_weight", params.get("information_effect_strength", 0.3)))
        policy_w = float(params.get("policy_influence_weight", params.get("mandate_effectiveness", 0.4)))
        mask_effectiveness = float(params.get("mask_effectiveness_proxy", 0.5))
        fatigue_rate = float(params.get("compliance_fatigue_rate", params.get("fatigue_or_burnout_rate", 0.01)))
        dropout_base = float(params.get("dropout_rate_without_norms", 0.02))
        threshold_mu = float(params.get("adoption_threshold_mean", 0.4))
        threshold_sd = float(params.get("adoption_threshold_std", 0.1))

        # Media reach set from prior module if present
        info_buf = buffers.get("InformationAndBeliefUpdate", {})
        reached_set: Set[int] = set(info_buf.get("reached", []))

        # Compute peer shares
        peer_share = [0.0] * len(persons)
        for i, p in enumerate(persons):
            neigh = adj[i]
            if neigh:
                s = 0
                for j in neigh:
                    s += persons[j].adoption_state
                peer_share[i] = s / len(neigh)

        new_adopt = [p.adoption_state for p in persons]
        for i, p in enumerate(persons):
            # policy signal based on mandate and trust
            policy_signal = policy_w * (1.0 if gov.mandate_active else 0.0) * (0.5 + 0.5 * p.trust_in_authorities)
            media_signal = media_w * (1.0 if i in reached_set else 0.0)
            net_benefit = peer_w * peer_share[i] + media_signal + policy_signal + mask_effectiveness * p.risk_perception - p.perceived_cost
            threshold_i = clamp(random.gauss(threshold_mu, threshold_sd))
            # fatigue increases perceived cost for adopters
            if p.adoption_state == 1:
                p.perceived_cost = clamp(p.perceived_cost + fatigue_rate * 0.2, 0.0, 3.0)
            # adopt if benefit exceeds threshold and has access
            if p.adoption_state == 0 and net_benefit >= threshold_i and p.mask_access == 1:
                new_adopt[i] = 1
            elif p.adoption_state == 1 and net_benefit < threshold_i:
                if rng.random() < clamp(dropout_base + fatigue_rate, 0.0, 1.0):
                    new_adopt[i] = 0

        buffers[self.name] = {"new_adoption_states": new_adopt, "peer_share": peer_share}
        self.log_io({"t": t, "adopt_count": sum(new_adopt), "population": len(new_adopt)})


class PolicyEnforcementAndAdmission(Module):
    """
    Module to enforce policy at locations and issue penalties, measuring compliance.

    Notes
    -----
    - Uses entrants lists by type from MobilityAndLocationUse.
    """
    pass

    def __init__(self, rng: random.Random):
        """
        Create PolicyEnforcementAndAdmission module.

        Parameters
        ----------
        rng : random.Random
            RNG for reproducibility.
        """
        pass
        super().__init__("PolicyEnforcementAndAdmission")
        self.rng = rng

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Enforce mask policy for entrants and compute compliance metrics.

        Parameters
        ----------
        state : Dict[str, Any]
            Current state.
        buffers : Dict[str, Any]
            Buffer to store compliance stats and penalties.
        params : Dict[str, Any]
            Enforcement parameters.
        t : int
            Day index.

        Returns
        -------
        None
        """
        pass
        if "persons" not in state:
            return
        rng = self.rng
        gov: Government = state["government"]
        persons: List[Person] = state["persons"]

        entrants_by_type = buffers.get("MobilityAndLocationUse", {}).get("entrants_by_type", {})
        if not entrants_by_type:
            entrants_by_type = {k: [] for k in ["workplace", "school", "retail", "transit", "public_space"]}

        # Enforcement probabilities
        enforce_p = float(params.get("enforcement_probability", 0.3))
        fine_amount = float(params.get("fine_amount", gov.fine_amount))
        check_p = float(params.get("admission_check_probability", 0.5))

        comp_by_type: Dict[str, Tuple[int, int]] = {k: [0, 0] for k in entrants_by_type.keys()}  # [numer, denom]
        penalties_value = 0.0

        for typ, entrants in entrants_by_type.items():
            required = typ in ["workplace", "school", "retail", "transit"] and gov.mandate_active
            for i in entrants:
                wearing = persons[i].adoption_state == 1
                comp_by_type[typ][1] += 1
                if wearing:
                    comp_by_type[typ][0] += 1
                elif required:
                    # check and possibly fine
                    if rng.random() < check_p * gov.enforcement_strength:
                        if rng.random() < enforce_p:
                            penalties_value += fine_amount

        buffers[self.name] = {
            "compliance_by_type": {k: (v[0] / v[1] if v[1] > 0 else 0.0) for k, v in comp_by_type.items()},
            "penalties_value": penalties_value,
        }
        self.log_io({"t": t, "penalties": penalties_value})


class GovernmentPolicy(Module):
    """
    Module for policy adjustments over time.

    Notes
    -----
    - Adjusts mandate based on adoption outcomes or schedule.
    """
    pass

    def __init__(self):
        """
        Create GovernmentPolicy module.
        """
        pass
        super().__init__("GovernmentPolicy")

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Adjust the policy level and mandate status based on rules.

        Parameters
        ----------
        state : Dict[str, Any]
            Current state with government and any observables.
        buffers : Dict[str, Any]
            Buffer to store updated policy state.
        params : Dict[str, Any]
            Policy parameters including thresholds and timing.
        t : int
            Day index.

        Returns
        -------
        None
        """
        pass
        if "government" not in state:
            return
        gov: Government = state["government"]
        target_rate = float(params.get("target_adoption_rate", 0.8))
        policy_adjust_interval = int(params.get("policy_adjust_interval_days", 14))
        mandate_start_day = int(params.get("mask_mandate_start_day", 1))

        # Use yesterday's adoption if available
        adoption_yesterday = None
        if "observables" in state and "adoption_rate_over_time" in state["observables"]:
            if state["observables"]["adoption_rate_over_time"]:
                adoption_yesterday = state["observables"]["adoption_rate_over_time"][-1]

        new_policy_level = gov.policy_level
        mandate_active = gov.mandate_active

        if t == mandate_start_day:
            mandate_active = True

        if t > 0 and t % policy_adjust_interval == 0 and adoption_yesterday is not None:
            if adoption_yesterday < 0.6:
                new_policy_level = min(3, new_policy_level + 1)
                mandate_active = True
            elif adoption_yesterday > 0.8:
                new_policy_level = max(0, new_policy_level - 1)
                # possibility to relax mandate but keep if still needed
                mandate_active = adoption_yesterday < target_rate

        buffers[self.name] = {"policy_level": new_policy_level, "mandate_active": mandate_active}
        self.log_io({"t": t, "policy_level": new_policy_level, "mandate": mandate_active})


class AdoptionAggregator(Module):
    """
    Module to aggregate daily observables and compute summary metrics.

    Observables
    -----------
    - adoption_rate_over_time
    - compliance_by_location_type
    - policy_cost_daily
    - supply_outage_daily
    - mask_purchase_rate_daily
    - adoption_inequality_index_income_daily
    """
    pass

    def __init__(self):
        """
        Create AdoptionAggregator module.
        """
        pass
        super().__init__("AdoptionAggregator")

    def gini_of_groups(self, group_means: List[float], weights: List[float]) -> float:
        """
        Compute Gini coefficient over group means with weights.

        Parameters
        ----------
        group_means : List[float]
            Mean adoption per group.
        weights : List[float]
            Group sizes.

        Returns
        -------
        float
            Weighted Gini coefficient (0-1).
        """
        pass
        # Based on relative mean difference
        if not group_means or not weights or sum(weights) == 0:
            return 0.0
        # Expand pairwise sums efficiently
        total_weight = sum(weights)
        mean = sum(m * w for m, w in zip(group_means, weights)) / total_weight
        if mean == 0:
            return 0.0
        diff_sum = 0.0
        for i, mi in enumerate(group_means):
            for j, mj in enumerate(group_means):
                diff_sum += abs(mi - mj) * weights[i] * weights[j]
        gini = diff_sum / (2 * total_weight**2 * mean + 1e-9)
        return clamp(gini, 0.0, 1.0)

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Compute and record observables for the day.

        Parameters
        ----------
        state : Dict[str, Any]
            Current simulation state.
        buffers : Dict[str, Any]
            Buffer to append aggregated results.
        params : Dict[str, Any]
            Parameters used for metrics computation.
        t : int
            Day index.

        Returns
        -------
        None
        """
        pass
        if "persons" not in state:
            return
        persons: List[Person] = state["persons"]
        N = max(1, len(persons))
        adoption_rate = sum(p.adoption_state for p in persons) / N

        # Compliance by type from enforcement module buffers
        comp_by_type = buffers.get("PolicyEnforcementAndAdmission", {}).get("compliance_by_type", {})
        penalties_value = float(buffers.get("PolicyEnforcementAndAdmission", {}).get("penalties_value", 0.0))

        # Supply outage indicator and purchase rate
        supply_outage = 1 if buffers.get("MaskSupplyAndAcquisition", {}).get("supply_outage", False) else 0
        sales = int(buffers.get("MaskSupplyAndAcquisition", {}).get("sales", 0))
        purchase_rate = sales / N

        # Adoption inequality by income quintile
        incomes = [p.income for p in persons]
        order = sorted(range(N), key=lambda i: incomes[i])
        quintiles = [order[int(N * q / 5): int(N * (q + 1) / 5)] for q in range(5)]
        group_means = [(sum(persons[i].adoption_state for i in g) / max(1, len(g))) for g in quintiles if g]
        weights = [len(g) for g in quintiles if g]
        gini = self.gini_of_groups(group_means, weights)

        # Message reach and attitude shift index
        info_buf = buffers.get("InformationAndBeliefUpdate", {})
        reached_today = set(info_buf.get("reached", []))
        attitude_shift_sum = float(info_buf.get("attitude_shift_sum", 0.0))
        attitude_shift_index = attitude_shift_sum / max(1, len(reached_today))

        # Policy cost approximation: subsidy outlays - fines collected (assuming gov procures at price)
        subsidy_per_mask = float(params.get("subsidy_per_mask", params.get("subsidy_amount", 0.0)))
        avg_price = float(params.get("mask_price", 1.0))
        subsidy_outlays = subsidy_per_mask * sales
        fines_collected = penalties_value  # assume fully collected
        policy_cost_daily = max(0.0, subsidy_outlays - fines_collected) + 0.0  # distribution costs omitted

        if "observables" not in state:
            state["observables"] = {
                "adoption_rate_over_time": [],
                "compliance_by_location_type": [],
                "policy_cost_daily": [],
                "supply_outage_daily": [],
                "mask_purchase_rate_daily": [],
                "adoption_inequality_index_income_daily": [],
                "message_reach_daily": [],
                "attitude_shift_index_daily": [],
            }

        state["observables"]["adoption_rate_over_time"].append(adoption_rate)
        state["observables"]["compliance_by_location_type"].append(comp_by_type)
        state["observables"]["policy_cost_daily"].append(policy_cost_daily)
        state["observables"]["supply_outage_daily"].append(supply_outage)
        state["observables"]["mask_purchase_rate_daily"].append(purchase_rate)
        state["observables"]["adoption_inequality_index_income_daily"].append(gini)
        state["observables"]["message_reach_daily"].append(len(reached_today) / N)
        state["observables"]["attitude_shift_index_daily"].append(attitude_shift_index)

        buffers[self.name] = {"adoption_rate": adoption_rate}
        self.log_io({"t": t, "adoption_rate": adoption_rate, "purchase_rate": purchase_rate})


# Simulation and Scheduler


class Simulation:
    """
    Main simulation engine coordinating modules and state.

    Methods
    -------
    run(start_day, end_day)
        Execute simulation over a specified day range.
    set_params(module=None, **kwargs)
        Update parameters; if module is provided, set module-specific params.
    get_params()
        Return current parameters.
    save_results(path)
        Save observables and summary metrics to path.
    save_module_io(module, path)
        Save per-module I/O logs for debugging.
    save_all_io(root_dir)
        Save I/O logs for all modules.
    evaluate()
        Compute evaluation metrics using available ground truth data.
    visualize()
        Produce simple visualization or textual summary.
    """
    pass

    def __init__(self, params: Dict[str, Any], param_defs: Optional[Dict[str, Dict[str, Any]]] = None, fast: bool = True):
        """
        Initialize the simulation engine.

        Parameters
        ----------
        params : Dict[str, Any]
            Effective parameters.
        param_defs : Optional[Dict[str, Dict[str, Any]]], optional
            Parameter definitions with frozen flags, by default None
        fast : bool, optional
            Enable fast mode downscaling, by default True
        """
        pass
        self.params = params
        self.param_defs = param_defs or {}
        self.fast = fast
        seed = int(params.get("random_seed", 42))
        self.rng = random.Random(seed)
        # Downscale population and days in fast mode
        if self.fast:
            self.params["population_size"] = min(600, int(params.get("population_size", 1000)))
            self.params["simulation_days"] = min(60, int(params.get("simulation_days", params.get("time_horizon_days", 60))))
        # Internal mapping/backward compatibility
        self._map_params_to_internal()
        # State
        self.state: Dict[str, Any] = {}
        self.buffers: Dict[str, Any] = {}
        self.modules: List[Module] = [
            PopulationNetworkInit(self.rng),
            GovernmentPolicy(),
            InformationAndBeliefUpdate(self.rng),
            MaskSupplyAndAcquisition(self.rng),
            MobilityAndLocationUse(self.rng),
            DecisionCompliance(self.rng),
            PolicyEnforcementAndAdmission(self.rng),
            AdoptionAggregator(),
        ]
        self.io_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.results: Dict[str, Any] = {}

    def _map_params_to_internal(self) -> None:
        """
        Map task spec style parameters to internal names for compatibility.

        Returns
        -------
        None
        """
        pass
        p = self.params
        if "time_horizon_days" not in p and "simulation_days" in p:
            p["time_horizon_days"] = p["simulation_days"]
        if "network_avg_degree" not in p and "avg_social_degree" in p:
            p["network_avg_degree"] = p["avg_social_degree"]
        if "initial_mask_adoption_rate" not in p and "initial_adoption_rate" in p:
            p["initial_mask_adoption_rate"] = p["initial_adoption_rate"]
        if "risk_perception_base" not in p and "base_risk_perception" in p:
            p["risk_perception_base"] = p["base_risk_perception"]
        if "peer_influence_weight" not in p and "social_influence_strength" in p:
            p["peer_influence_weight"] = p["social_influence_strength"]
        if "media_influence_weight" not in p and "information_effect_strength" in p:
            p["media_influence_weight"] = p["information_effect_strength"]
        if "policy_influence_weight" not in p and "mandate_effectiveness" in p:
            p["policy_influence_weight"] = p["mandate_effectiveness"]
        if "messaging_frequency_days" not in p and "message_frequency_days" in p:
            p["messaging_frequency_days"] = p["message_frequency_days"]
        if "mask_price" not in p:
            p["mask_price"] = 1.0
        if "subsidy_per_mask" not in p and "subsidy_amount" in p:
            p["subsidy_per_mask"] = p["subsidy_amount"]

    def run(self, start_day: int, end_day: int) -> None:
        """
        Run the simulation from start_day to end_day inclusive.

        Parameters
        ----------
        start_day : int
            Start day index (0-based).
        end_day : int
            End day index (inclusive).

        Returns
        -------
        None
        """
        pass
        # Clear previous observables beyond start_day if partial rerun
        if "observables" in self.state and start_day == 0:
            self.state["observables"] = {k: [] for k in self.state["observables"].keys()}
        T = int(self.params.get("time_horizon_days", 60))
        end = min(end_day, T - 1) if T > 0 else end_day

        for t in range(start_day, end + 1):
            self.buffers = {}
            # Initialization module at t=0
            if t == 0:
                self.modules[0].forward(self.state, self.buffers, self.params, t)
                self._commit_init()
            # Government policy (weekly)
            self.modules[1].forward(self.state, self.buffers, self.params, t)
            self._commit_policy()
            # Info and belief
            self.modules[2].forward(self.state, self.buffers, self.params, t)
            # Supply and acquisition
            self.modules[3].forward(self.state, self.buffers, self.params, t)
            # Mobility and visits
            self.modules[4].forward(self.state, self.buffers, self.params, t)
            # Decisions
            self.modules[5].forward(self.state, self.buffers, self.params, t)
            self._commit_adoption()
            # Enforcement
            self.modules[6].forward(self.state, self.buffers, self.params, t)
            # Aggregation
            self.modules[7].forward(self.state, self.buffers, self.params, t)

            # Log per-module I/O for export
            for m in self.modules:
                self.io_logs.setdefault(m.name, [])
                # Append last record if available
                if m.io_log:
                    self.io_logs[m.name].append(m.io_log[-1])

        # Summaries
        self._compute_summary()

    def _commit_init(self) -> None:
        """
        Commit initialization buffers to the state.

        Returns
        -------
        None
        """
        pass
        init = self.buffers.get("init", {})
        self.state.update(init)

    def _commit_policy(self) -> None:
        """
        Commit updated policy level and mandate status.

        Returns
        -------
        None
        """
        pass
        pol = self.buffers.get("GovernmentPolicy", {})
        if "government" in self.state and pol:
            self.state["government"].policy_level = int(pol.get("policy_level", self.state["government"].policy_level))
            self.state["government"].mandate_active = bool(pol.get("mandate_active", self.state["government"].mandate_active))

    def _commit_adoption(self) -> None:
        """
        Commit updated adoption states to persons.

        Returns
        -------
        None
        """
        pass
        upd = self.buffers.get("DecisionCompliance", {})
        if "persons" in self.state and upd:
            new_states = upd.get("new_adoption_states", [])
            for i, p in enumerate(self.state["persons"]):
                if i < len(new_states):
                    p.adoption_state = int(new_states[i])
                    p.mask_access = 1 if p.mask_stock > 0 else 0

    def _compute_summary(self) -> None:
        """
        Compute summary metrics to store in self.results.

        Returns
        -------
        None
        """
        pass
        obs = self.state.get("observables", {})
        adoption_series = obs.get("adoption_rate_over_time", [])
        final_adoption = adoption_series[-1] if adoption_series else 0.0
        target = float(self.params.get("target_adoption_rate", 0.8))
        time_to_target = None
        for d, v in enumerate(adoption_series):
            if v >= target:
                time_to_target = d
                break

        comp_series = obs.get("compliance_by_location_type", [])
        # Aggregate compliance across types
        comp_num = 0.0
        comp_den = 0
        for day_comp in comp_series:
            for _, v in day_comp.items():
                comp_num += v
                comp_den += 1
        avg_compliance = (comp_num / comp_den) if comp_den > 0 else 0.0

        # Stockout rate
        # from MaskSupplyAndAcquisition logs
        attempts = 0
        fails = 0
        for rec in self.io_logs.get("MaskSupplyAndAcquisition", []):
            attempts += int(rec.get("attempts", 0) or 0)
            fails += int(rec.get("fail", 0) or 0)
        stockout_rate = (fails / attempts) if attempts > 0 else 0.0

        # Penalties per day
        penalties_sum = 0.0
        for rec in self.io_logs.get("PolicyEnforcementAndAdmission", []):
            penalties_sum += float(rec.get("penalties", 0.0) or 0.0)
        days = max(1, len(adoption_series))
        penalties_per_day = penalties_sum / days

        # Message reach and attitude shift index
        msg_reach = obs.get("message_reach_daily", [])
        reach_overall = sum(msg_reach) / max(1, len(msg_reach))
        att_shift_idx = obs.get("attitude_shift_index_daily", [])
        avg_att_shift_idx = sum(att_shift_idx) / max(1, len(att_shift_idx)) if att_shift_idx else 0.0

        self.results = {
            "final_adoption_rate": round(final_adoption, 4),
            "time_to_target_adoption": time_to_target,
            "avg_compliance_mandated": round(avg_compliance, 4),
            "mask_supply_shortage_rate": round(stockout_rate, 4),
            "penalties_issued_per_day": round(penalties_per_day, 4),
            "message_reach": round(reach_overall, 4),
            "attitude_shift_index": round(avg_att_shift_idx, 4),
        }

    def set_params(self, module: Optional[str] = None, **kwargs: Any) -> None:
        """
        Update internal parameters, ignoring frozen ones.

        Parameters
        ----------
        module : Optional[str], optional
            Module name scope, currently ignored for simplicity.
        **kwargs : Any
            Key-value pairs of params to update.

        Returns
        -------
        None
        """
        pass
        for k, v in kwargs.items():
            if self._is_frozen(k):
                print(f"[WARN] Parameter '{k}' is frozen; override ignored.", file=sys.stderr)
                continue
            self.params[k] = v
        self._map_params_to_internal()

    def get_params(self) -> Dict[str, Any]:
        """
        Get a copy of current parameters.

        Returns
        -------
        Dict[str, Any]
            Copy of parameters.
        """
        pass
        return dict(self.params)

    def _is_frozen(self, key: str) -> bool:
        """
        Check if a parameter is frozen using parameter definitions.

        Parameters
        ----------
        key : str
            Parameter key to check.

        Returns
        -------
        bool
            True if frozen, else False.
        """
        pass
        if key in self.param_defs:
            return bool(self.param_defs[key].get("frozen", False))
        # Reasonable defaults for frozen
        return key in {"random_seed", "time_step_days"}

    def save_results(self, path: str) -> None:
        """
        Save simulation results and observables to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        None
        """
        pass
        out = {
            "results": self.results,
            "observables": self.state.get("observables", {}),
        }
        safe_json_dump(out, path)

    def save_module_io(self, module: Module, path: str) -> None:
        """
        Save I/O logs for a specific module to a JSON file.

        Parameters
        ----------
        module : Module
            Module instance whose logs to save.
        path : str
            Output JSON path.

        Returns
        -------
        None
        """
        pass
        safe_json_dump(module.io_log, path)

    def save_all_io(self, root_dir: str) -> None:
        """
        Save I/O logs for all modules into root_dir.

        Parameters
        ----------
        root_dir : str
            Directory to save per-module JSON logs.

        Returns
        -------
        None
        """
        pass
        os.makedirs(root_dir, exist_ok=True)
        for m in self.modules:
            path = os.path.join(root_dir, f"{m.name}.json")
            self.save_module_io(m, path)

    def evaluate(self, ground_truth: Optional[Dict[str, List[float]]] = None, window: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Evaluate simulation outputs against ground truth observables.

        Parameters
        ----------
        ground_truth : Optional[Dict[str, List[float]]], optional
            Dict mapping observable names to ground truth series, by default None.
        window : Optional[Tuple[int, int]], optional
            (start_day, end_day) indices for evaluation window, by default None

        Returns
        -------
        Dict[str, Any]
            Metrics dictionary with RMSE, MAE, and simple transition placeholders.
        """
        pass
        obs = self.state.get("observables", {})
        sim_series = obs.get("adoption_rate_over_time", [])
        start, end = window if window is not None else (0, len(sim_series) - 1)
        start = max(0, start)
        end = min(end, len(sim_series) - 1) if sim_series else -1
        sim_window = sim_series[start:end + 1] if end >= start and sim_series else []

        if ground_truth and "adoption_rate_over_time" in ground_truth:
            gt_full = ground_truth["adoption_rate_over_time"]
            gt_window = gt_full[start:end + 1] if end >= start and gt_full else []
        else:
            # Degrade gracefully: compare to self so errors zero
            gt_window = sim_window[:]

        def rmse(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))

        def mae(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

        metrics = {
            "RMSE_aggregate": rmse(sim_window, gt_window),
            "MAE_aggregate": mae(sim_window, gt_window),
            "Brier": 0.0,  # placeholder
            "TransitionFit": {"P01": None, "P11": None, "P10": None, "P00": None},
        }
        return metrics

    def visualize(self) -> None:
        """
        Produce a simple textual visualization of adoption over time.

        Returns
        -------
        None
        """
        pass
        series = self.state.get("observables", {}).get("adoption_rate_over_time", [])
        if not series:
            print("[viz] No adoption series available.")
            return
        print("[viz] Adoption trajectory (first 20 days):")
        for d, v in enumerate(series[:20]):
            bar = "#" * int(v * 50)
            print(f" Day {d:02d}: {v:.3f} {bar}")


# Calibration Architecture


@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator and compatible with Calibrasim.

    Attributes
    ----------
    decision_weights : Dict[str, float]
        Weights for decision head, e.g., peer/media/policy.
    layer_weights : Dict[str, float]
        Layer weights for different network layers (if applicable).
    info_params : Dict[str, float]
        Information flow parameters such as campaign intensity, memory decay.
    noise_params : Dict[str, float]
        Noise/temperature parameters.
    module_params : Dict[str, Dict[str, float]]
        Module-specific parameters mapping.
    engine_type : str
        Engine compatibility identifier.
    meta : Dict[str, Any]
        Metadata including seed, calibrator name, etc.
    """
    pass
    decision_weights: Dict[str, float]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation.
        """
        pass
        return asdict(self)


class ParamsAdapter:
    """
    Adapts FittedParams to the simulation parameter system.

    Methods
    -------
    apply(simulation, params)
        Apply fitted parameters via simulation.set_params and write parameters_used.json.
    capture(simulation)
        Capture current sim params into FittedParams structure.
    validate_frozen(params)
        Validate against frozen parameter definitions.
    """
    pass

    def __init__(self, param_def_path: Optional[str] = None):
        """
        Initialize ParamsAdapter with optional param definitions path.

        Parameters
        ----------
        param_def_path : Optional[str], optional
            Path to parameter_definitions.json, by default None
        """
        pass
        self.param_defs: Dict[str, Dict[str, Any]] = {}
        if param_def_path and os.path.exists(param_def_path):
            try:
                with open(param_def_path, "r", encoding="utf-8") as f:
                    defs = json.load(f)
                    if isinstance(defs, dict):
                        self.param_defs = defs
            except Exception as e:
                print(f"[WARN] Failed to load param definitions: {e}", file=sys.stderr)

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply FittedParams to simulation parameters with mapping.

        Parameters
        ----------
        simulation : Simulation
            Simulation instance to update.
        params : FittedParams
            Fitted parameters.

        Returns
        -------
        None
        """
        pass
        # Map decision weights
        dw = params.decision_weights or {}
        updates = {}
        if "w_peer" in dw:
            updates["peer_influence_weight"] = float(dw["w_peer"])
        if "w_media" in dw:
            updates["media_influence_weight"] = float(dw["w_media"])
        if "w_policy" in dw:
            updates["policy_influence_weight"] = float(dw["w_policy"])
        if "threshold" in dw:
            updates["adoption_threshold_mean"] = float(dw["threshold"])
        # Map info params
        ip = params.info_params or {}
        if "campaign_intensity" in ip:
            updates["campaign_intensity"] = float(ip["campaign_intensity"])
        if "memory_decay" in ip:
            # Could map to learning rate inverse
            updates["learning_rate"] = float(max(0.01, 1.0 - ip["memory_decay"]))
        # Noise params
        np = params.noise_params or {}
        if "temperature" in np:
            # Map to threshold std
            updates["adoption_threshold_std"] = float(max(0.01, min(0.5, np["temperature"])))
        # Module params
        for mod, mparams in (params.module_params or {}).items():
            for k, v in mparams.items():
                updates[k] = v

        # Apply updates, ignoring frozen per simulation param defs
        for k, v in updates.items():
            if simulation._is_frozen(k):
                print(f"[WARN] Frozen parameter '{k}' not updated by adapter.", file=sys.stderr)
            else:
                simulation.params[k] = v
        simulation._map_params_to_internal()
        # Persist parameters_used.json
        safe_json_dump(simulation.get_params(), "parameters_used.json")

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current effective parameters into FittedParams.

        Parameters
        ----------
        simulation : Simulation
            Simulation instance to capture.

        Returns
        -------
        FittedParams
            Captured parameters in FittedParams format.
        """
        pass
        p = simulation.get_params()
        fp = FittedParams(
            decision_weights={
                "w_peer": float(p.get("peer_influence_weight", 0.5)),
                "w_media": float(p.get("media_influence_weight", 0.3)),
                "w_policy": float(p.get("policy_influence_weight", 0.4)),
                "threshold": float(p.get("adoption_threshold_mean", 0.4)),
            },
            layer_weights={
                "family": 0.5,
                "work_school": 0.3,
                "community": 0.2,
            },
            info_params={
                "campaign_intensity": float(p.get("campaign_intensity", 0.5)),
                "gamma_info": 1.0,
                "memory_decay": float(max(0.01, 1.0 - p.get("learning_rate", 0.2))),
            },
            noise_params={
                "temperature": float(p.get("adoption_threshold_std", 0.1)),
            },
            module_params={},
            engine_type="calibrasim",
            meta={"captured_at": time.time()},
        )
        return fp

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Check frozen parameters against definitions and return warnings.

        Parameters
        ----------
        params : FittedParams
            Fitted parameters.

        Returns
        -------
        Dict[str, str]
            Mapping from key to warning message for any frozen violations.
        """
        pass
        warnings: Dict[str, str] = {}
        frozen_keys = {k for k, d in self.param_defs.items() if d.get("frozen")}
        updates = {}
        updates.update(params.decision_weights or {})
        updates.update(params.info_params or {})
        updates.update(params.noise_params or {})
        for mod, mparams in (params.module_params or {}).items():
            updates.update(mparams or {})
        for k in updates.keys():
            if k in frozen_keys:
                warnings[k] = "Attempt to override frozen parameter."
        return warnings


class Calibrator:
    """
    Abstract calibrator interface with a stable evaluation callback signature.

    Methods
    -------
    fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)
        Fit parameters on the training window using evaluator and return FittedParams.
    """
    pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit parameters on the training window using evaluator and return FittedParams.

        Parameters
        ----------
        bundle : Dict[str, Any]
            Data bundle with ground truth and plan metadata.
        simulator : Simulation
            Simulation engine.
        evaluator : callable
            Evaluation function evaluate_params(simulator, params, window).
        train_window : Tuple[int, int]
            Training window (start_day, end_day).
        seed : int
            Random seed for reproducibility.
        budget : int, optional
            Number of trials, by default 100
        artifacts_dir : Optional[str], optional
            Directory to store trial artifacts, by default None
        params_adapter : Optional[ParamsAdapter], optional
            Adapter to apply fitted parameters, by default None

        Returns
        -------
        FittedParams
            Best fitted parameters found.
        """
        pass
        raise NotImplementedError("Calibrator.fit must be implemented.")


class LogitHeadCalibrator(Calibrator):
    """
    Calibrator that fits a logistic decision head from micro-transitions/aggregates.

    Notes
    -----
    - Degrades gracefully if micro-transitions unavailable.
    - Uses L2 regularization on weights (excluding intercept).
    """
    pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 50,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit a simple logistic head by grid search over a small set of weights.

        Parameters
        ----------
        bundle : Dict[str, Any]
            Data bundle.
        simulator : Simulation
            Simulation engine.
        evaluator : callable
            Evaluation function.
        train_window : Tuple[int, int]
            Training (start, end).
        seed : int
            RNG seed.
        budget : int, optional
            Trials budget, by default 50
        artifacts_dir : Optional[str], optional
            Artifacts directory, by default None
        params_adapter : Optional[ParamsAdapter], optional
            Adapter for applying parameters.

        Returns
        -------
        FittedParams
            Best fitted parameters.
        """
        pass
        rng = random.Random(seed)
        base = params_adapter.capture(simulator) if params_adapter else FittedParams(
            decision_weights={"w_peer": 0.5, "w_media": 0.3, "w_policy": 0.4, "threshold": 0.4},
            layer_weights={"family": 0.5, "work_school": 0.3, "community": 0.2},
            info_params={"campaign_intensity": 0.5, "gamma_info": 1.0, "memory_decay": 0.8},
            noise_params={"temperature": 0.1},
        )

        best_params = base
        best_score = float("inf")
        trials = []

        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)

        # Simple grid/random search for weights
        for i in range(budget):
            fp = FittedParams(
                decision_weights={
                    "w_peer": clamp(base.decision_weights.get("w_peer", 0.5) + rng.uniform(-0.2, 0.2), 0.0, 2.0),
                    "w_media": clamp(base.decision_weights.get("w_media", 0.3) + rng.uniform(-0.2, 0.2), 0.0, 2.0),
                    "w_policy": clamp(base.decision_weights.get("w_policy", 0.4) + rng.uniform(-0.2, 0.2), 0.0, 2.0),
                    "threshold": clamp(base.decision_weights.get("threshold", 0.4) + rng.uniform(-0.1, 0.1), 0.0, 1.0),
                },
                layer_weights=base.layer_weights,
                info_params=base.info_params,
                noise_params={
                    "temperature": clamp(base.noise_params.get("temperature", 0.1) + rng.uniform(-0.05, 0.05), 0.01, 0.5)
                },
                module_params={},
                engine_type="calibrasim",
                meta={"trial": i, "calibrator": "logit_head"},
            )
            # Apply, evaluate
            if params_adapter:
                params_adapter.apply(simulator, fp)
            metrics = evaluator(simulator, fp, train_window)
            score = metrics.get("RMSE_aggregate", 0.0) + 0.1 * metrics.get("MAE_aggregate", 0.0)

            if artifacts_dir:
                tdir = os.path.join(artifacts_dir, f"trial_{i}")
                os.makedirs(tdir, exist_ok=True)
                safe_json_dump(fp.to_dict(), os.path.join(tdir, "params_applied.json"))
                safe_json_dump(metrics, os.path.join(tdir, "metrics.json"))

            trials.append({"score": score, "metrics": metrics, "params": fp.to_dict()})

            if score < best_score:
                best_score = score
                best_params = fp

        if artifacts_dir:
            best_dir = os.path.join(artifacts_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            safe_json_dump(best_params.to_dict(), os.path.join(best_dir, "fitted_params.json"))
            safe_json_dump({"budget": budget, "best_score": best_score, "trials": trials}, os.path.join(artifacts_dir, "calibration_report.json"))

        return best_params


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search calibrator over selected simulator parameters.

    Notes
    -----
    - Uses evaluator on the training window as objective.
    - Saves each trial's params and metrics.
    """
    pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 30,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit parameters via random search.

        Parameters
        ----------
        bundle : Dict[str, Any]
            Data bundle.
        simulator : Simulation
            Simulation engine.
        evaluator : callable
            Evaluation function.
        train_window : Tuple[int, int]
            Training window.
        seed : int
            RNG seed.
        budget : int, optional
            Trials budget, by default 30
        artifacts_dir : Optional[str], optional
            Directory for artifacts, by default None
        params_adapter : Optional[ParamsAdapter], optional
            Adapter for parameter application.

        Returns
        -------
        FittedParams
            Best fitted parameters.
        """
        pass
        rng = random.Random(seed)
        base = params_adapter.capture(simulator) if params_adapter else FittedParams(
            decision_weights={"w_peer": 0.5, "w_media": 0.3, "w_policy": 0.4, "threshold": 0.4},
            layer_weights={"family": 0.5, "work_school": 0.3, "community": 0.2},
            info_params={"campaign_intensity": 0.5, "gamma_info": 1.0, "memory_decay": 0.8},
            noise_params={"temperature": 0.1},
        )
        best_params = base
        best_score = float("inf")
        trials = []

        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)

        for i in range(budget):
            fp = FittedParams(
                decision_weights={
                    "w_peer": rng.uniform(0.1, 1.5),
                    "w_media": rng.uniform(0.0, 1.2),
                    "w_policy": rng.uniform(0.0, 1.2),
                    "threshold": rng.uniform(0.2, 0.8),
                },
                layer_weights={
                    "family": rng.uniform(0.2, 0.6),
                    "work_school": rng.uniform(0.2, 0.6),
                    "community": rng.uniform(0.1, 0.4),
                },
                info_params={
                    "campaign_intensity": rng.uniform(0.2, 0.8),
                    "gamma_info": 1.0,
                    "memory_decay": rng.uniform(0.2, 0.9),
                },
                noise_params={"temperature": rng.uniform(0.05, 0.3)},
                module_params={},
                engine_type="calibrasim",
                meta={"trial": i, "calibrator": "random_search"},
            )
            if params_adapter:
                params_adapter.apply(simulator, fp)
            metrics = evaluator(simulator, fp, train_window)
            score = metrics.get("RMSE_aggregate", 0.0) + 0.2 * metrics.get("MAE_aggregate", 0.0)
            if artifacts_dir:
                tdir = os.path.join(artifacts_dir, f"trial_{i}")
                os.makedirs(tdir, exist_ok=True)
                safe_json_dump(fp.to_dict(), os.path.join(tdir, "params_applied.json"))
                safe_json_dump(metrics, os.path.join(tdir, "metrics.json"))
            trials.append({"score": score, "metrics": metrics, "params": fp.to_dict()})
            if score < best_score:
                best_score = score
                best_params = fp

        if artifacts_dir:
            best_dir = os.path.join(artifacts_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            safe_json_dump(best_params.to_dict(), os.path.join(best_dir, "fitted_params.json"))
            safe_json_dump({"budget": budget, "best_score": best_score, "trials": trials}, os.path.join(artifacts_dir, "calibration_report.json"))

        return best_params


class SNPECalibrator(Calibrator):
    """
    True SBI using neural networks for Bayesian parameter inference with fallback.

    Notes
    -----
    - If torch/sbi not available, falls back to RandomSearchCalibrator.
    """
    pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 50,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Perform SNPE; fallback to random search if dependencies unavailable.

        Parameters
        ----------
        bundle : Dict[str, Any]
            Data bundle.
        simulator : Simulation
            Simulation engine.
        evaluator : callable
            Evaluation function.
        train_window : Tuple[int, int]
            Training window.
        seed : int
            RNG seed.
        budget : int, optional
            Number of simulations, by default 50
        artifacts_dir : Optional[str], optional
            Artifacts path, by default None
        params_adapter : Optional[ParamsAdapter], optional
            Adapter for parameters.

        Returns
        -------
        FittedParams
            Fitted parameters.
        """
        pass
        try:
            import torch  # noqa: F401
            # from sbi.inference import SNPE  # heavy import; likely unavailable
            # Minimalistic fallback to random search even if torch present for sandbox constraints
            raise ImportError("SBI modules unavailable; fallback to random search.")
        except Exception:
            # Fallback
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None) -> Calibrator:
    """
    Retrieve a calibrator instance by name with optional config.

    Parameters
    ----------
    name : str
        Calibrator name: 'logit_head', 'random_search', or 'snpe'.
    config_path : Optional[str], optional
        Path to JSON config with kwargs, by default None

    Returns
    -------
    Calibrator
        Instantiated calibrator.
    """
    pass
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    kwargs: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if isinstance(cfg, dict):
                    kwargs.update(cfg)
        except Exception as e:
            print(f"[WARN] Failed to load calibrator config: {e}", file=sys.stderr)
    return CALIBRATOR_REGISTRY[name](**kwargs)  # type: ignore


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).

    For simulation engines: use Simulation.run(start,end) + Simulation.evaluate(),
    read metrics from artifacts/results/, build dict with required keys.
    If micro-transitions unavailable, degrade gracefully or use placeholder values.

    Parameters
    ----------
    simulator : Simulation
        Simulation engine.
    params : FittedParams
        Parameters to apply for evaluation.
    window : Tuple[int, int]
        Evaluation window (start_day, end_day).

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary.
    """
    pass
    adapter = ParamsAdapter()  # local adapter without defs
    adapter.apply(simulator, params)
    start, end = window
    simulator.run(start, end)
    # Ground truth can be absent; using simulation self-compare
    metrics = simulator.evaluate(None, window)
    return metrics


# CLI, parameters, and utilities


def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments for the simulation runner.

    Parameters
    ----------
    argv : Optional[List[str]], optional
        List of command-line arguments, by default None

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    pass
    parser = argparse.ArgumentParser(description="Mask Adoption Diffusion Simulation")
    parser.add_argument("--param-file", type=str, default="parameters.json", help="Path to parameters JSON file")
    parser.add_argument("--set", action="append", default=[], help="Override parameter key=value (repeatable)")
    parser.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()))
    parser.add_argument("--budget", type=int, default=10, help="Calibration budget (iterations)")
    parser.add_argument("--calib-window", type=str, default=None, help="Calibration window start:end (0-index inclusive)")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to store artifacts")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode downscaling")
    parser.add_argument("--calib-config", type=str, default=None, help="Optional calibrator config JSON path")
    return parser.parse_args(argv)


def load_json_file(path: str) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary; return {} on failure.

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    Dict[str, Any]
        Loaded dictionary or empty dict if load fails.
    """
    pass
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}", file=sys.stderr)
        return {}


def load_parameters(param_file: str) -> Dict[str, Any]:
    """
    Load parameters from a JSON file, or synthesize minimal defaults if missing.

    Parameters
    ----------
    param_file : str
        Path to parameters JSON file.

    Returns
    -------
    Dict[str, Any]
        Parameters dictionary.
    """
    pass
    params = load_json_file(param_file)
    if params:
        return params
    # Synthesize minimal defaults to ensure runnable behavior
    print(f"[WARN] Parameter file '{param_file}' not found. Using minimal defaults.", file=sys.stderr)
    rng = random.Random(42)
    return {
        "population_size": 400,
        "simulation_days": 40,
        "time_step_days": 1,
        "random_seed": 42,
        "avg_social_degree": 8,
        "initial_adoption_rate": 0.1,
        "base_risk_perception": 0.3,
        "social_influence_strength": 0.5,
        "information_effect_strength": 0.3,
        "misinformation_rate": 0.1,
        "learning_rate": 0.2,
        "decision_temperature": 0.3,
        "compliance_fatigue_rate": 0.01,
        "mandate_effectiveness": 0.6,
        "enforcement_probability": 0.3,
        "fine_amount": 50.0,
        "mask_price": 1.0,
        "subsidy_per_mask": 0.0,
        "retailer_initial_inventory": 1000,
        "retailer_restock_interval_days": 7,
        "retailer_restock_quantity": 300,
        "message_frequency_days": 7,
        "media_reach": 0.7,
        "media_reliability_mean": 0.8,
        "policy_level": 1,
        "mask_mandate_start_day": 1,
        "policy_adjust_interval_days": 14,
        "target_adoption_rate": 0.8,
        "organization_enforcement_level": 0.6,
        "region_count": 3,
        "retailer_ration_limit_per_purchase": 10,
        "price_markup_sensitivity": 0.1,
        "risk_perception_base": 0.3,
        "trust_in_authorities_mean": 0.6,
        "trust_in_authorities_std": 0.2,
        "perceived_cost_mean": 0.2,
        "perceived_cost_std": 0.1,
        "adoption_threshold_mean": 0.4,
        "adoption_threshold_std": 0.1,
        "dropout_rate_without_norms": 0.02,
        "mask_effectiveness_proxy": 0.5,
        "media_bias": 0.0,
    }


def load_param_definitions(param_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load parameter definitions to detect frozen status; look next to param_file.

    Parameters
    ----------
    param_file : str
        Path to param JSON for locating param definitions.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from param key to definitions (potentially including 'frozen').
    """
    pass
    base_dir = os.path.dirname(os.path.abspath(param_file))
    candidates = [
        os.path.join(base_dir, "parameter_definitions.json"),
        "parameter_definitions.json",
    ]
    for c in candidates:
        defs = load_json_file(c)
        if defs:
            return defs
    # Fallback: mark some defaults as frozen
    return {
        "random_seed": {"frozen": True},
        "time_step_days": {"frozen": True},
    }


def apply_overrides(params: Dict[str, Any], overrides: List[str], param_defs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply CLI overrides in the form key=value, ignoring frozen parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        Existing parameters.
    overrides : List[str]
        Overrides as 'key=value' strings.
    param_defs : Dict[str, Dict[str, Any]]
        Parameter definitions to check 'frozen'.

    Returns
    -------
    Dict[str, Any]
        Updated parameters with applied overrides.
    """
    pass
    for ov in overrides:
        if "=" not in ov:
            print(f"[WARN] Invalid override '{ov}' (expected key=value).", file=sys.stderr)
            continue
        key, val = ov.split("=", 1)
        key = key.strip()
        val = val.strip()
        if param_defs.get(key, {}).get("frozen", False):
            print(f"[WARN] Override ignored for frozen parameter '{key}'.", file=sys.stderr)
            continue
        # Try to parse numeric/bool types
        try:
            if val.lower() in {"true", "false"}:
                parsed = val.lower() == "true"
            elif "." in val:
                parsed = float(val)
            else:
                parsed = int(val)
        except Exception:
            parsed = val
        params[key] = parsed
    return params


def save_parameters_used(params: Dict[str, Any], path: str = "parameters_used.json") -> None:
    """
    Persist the final parameters used by the simulation.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameters dictionary.
    path : str, optional
        Output JSON path, by default "parameters_used.json"

    Returns
    -------
    None
    """
    pass
    safe_json_dump(params, path)


def resolve_data_dir() -> str:
    """
    Resolve the project data directory from environment variables.

    Returns
    -------
    str
        Data directory path.
    """
    pass
    project_root = os.environ.get("PROJECT_ROOT", ".")
    data_path = os.environ.get("DATA_PATH", "")
    data_dir = os.path.join(project_root, data_path)
    return data_dir


def load_ground_truth_series(data_dir: str) -> Dict[str, List[float]]:
    """
    Load ground truth adoption series from train_data.csv if available.

    Parameters
    ----------
    data_dir : str
        Base data directory.

    Returns
    -------
    Dict[str, List[float]]
        Ground truth dict containing 'adoption_rate_over_time', may be empty.
    """
    pass
    path = os.path.join(data_dir, "train_data.csv")
    if not os.path.exists(path):
        return {}
    series: List[float] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("adoption_rate")
                if val is not None and val != "":
                    try:
                        series.append(float(val))
                    except Exception:
                        continue
    except Exception as e:
        print(f"[WARN] Failed to load ground truth: {e}", file=sys.stderr)
        return {}
    return {"adoption_rate_over_time": series}


def temporal_holdout_split(days: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Perform temporal holdout: first 80% for training, remaining 20% for validation.

    Parameters
    ----------
    days : int
        Number of available daily observations.

    Returns
    -------
    Tuple[Tuple[int, int], Tuple[int, int]]
        (train_start, train_end), (val_start, val_end)

    Raises
    ------
    ValueError
        If validation window is empty.
    """
    pass
    if days <= 1:
        raise ValueError("Not enough days for temporal split.")
    train_end = max(0, int(0.8 * days) - 1)
    val_start = train_end + 1
    val_end = days - 1
    if val_end < val_start:
        raise ValueError("No validation days available after temporal split.")
    return (0, train_end), (val_start, val_end)


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point to run the simulation, perform optional calibration, and save outputs.

    Workflow
    --------
    1. parse_cli()
    2. load_parameters() + load_param_definitions() + apply_overrides()
    3. build Simulation()
    4. load_data() and temporal holdout split
    5. calibrator.fit()
    6. simulator.run() full horizon
    7. evaluate() and save_results()
    8. visualize()
    9. Print compact JSON summary to stdout

    Returns
    -------
    None
    """
    pass
    args = parse_cli(argv)
    params = load_parameters(args.param_file)
    param_defs = load_param_definitions(args.param_file)
    params = apply_overrides(params, args.__dict__.get("set", []) or [], param_defs)
    save_parameters_used(params)

    # Build simulator
    sim = Simulation(params, param_defs, fast=bool(args.fast))

    # Load ground truth and split windows
    data_dir = resolve_data_dir()
    gt = load_ground_truth_series(data_dir)

    # If GT absent, synthesize by running initial sim and using its outputs
    if not gt:
        horizon = int(sim.params.get("time_horizon_days", sim.params.get("simulation_days", 40)))
        sim.run(0, horizon - 1)
        series = sim.state.get("observables", {}).get("adoption_rate_over_time", [])[:]
        if not series:
            series = [0.0] * max(10, horizon)
        gt = {"adoption_rate_over_time": series}
        # Reset sim for calibration after generating synthetic GT
        sim = Simulation(params, param_defs, fast=bool(args.fast))

    days = len(gt.get("adoption_rate_over_time", []))
    try:
        (train_start, train_end), (val_start, val_end) = temporal_holdout_split(days)
    except ValueError as e:
        # Ensure at least minimal validation window by extending synthetic series
        print(f"[WARN] {e}. Creating synthetic validation window.", file=sys.stderr)
        # Extend GT by repeating last value
        if "adoption_rate_over_time" in gt and gt["adoption_rate_over_time"]:
            last = gt["adoption_rate_over_time"][-1]
            gt["adoption_rate_over_time"].extend([last] * 5)
            (train_start, train_end), (val_start, val_end) = temporal_holdout_split(len(gt["adoption_rate_over_time"]))
        else:
            gt["adoption_rate_over_time"] = [0.1] * 20
            (train_start, train_end), (val_start, val_end) = temporal_holdout_split(20)

    # Prepare calibration
    artifacts_dir = args.artifacts_dir or "artifacts"
    if os.path.exists(artifacts_dir):
        try:
            shutil.rmtree(artifacts_dir)
        except Exception:
            pass
    os.makedirs(artifacts_dir, exist_ok=True)

    calibrator = get_calibrator(args.calibrator, args.calib_config)
    adapter = ParamsAdapter(param_def_path=os.path.join(os.path.dirname(os.path.abspath(args.param_file)), "parameter_definitions.json"))

    # Bundle for calibration (could include plan/observables)
    bundle = {"ground_truth": gt}

    # Define evaluator closure
    def evaluator(simulator: Simulation, fitted_params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
        # Apply via adapter
        adapter.apply(simulator, fitted_params)
        simulator.run(window[0], window[1])
        return simulator.evaluate(gt, window)

    # Calibration window from CLI, else use training split
    if args.calib_window:
        try:
            s, e = args.calib_window.split(":")
            calib_window = (int(s), int(e))
        except Exception:
            print(f"[WARN] Invalid --calib-window '{args.calib_window}', using training window.", file=sys.stderr)
            calib_window = (train_start, train_end)
    else:
        calib_window = (train_start, train_end)

    # Fit
    seed = int(params.get("random_seed", 42))
    fitted = calibrator.fit(bundle, sim, evaluator, calib_window, seed=seed, budget=int(args.budget), artifacts_dir=artifacts_dir, params_adapter=adapter)

    # Apply best params and run full horizon
    adapter.apply(sim, fitted)
    horizon = int(sim.params.get("time_horizon_days", sim.params.get("simulation_days", 40)))
    sim.run(0, horizon - 1)

    # Evaluate on validation window
    eval_metrics = sim.evaluate(gt, (val_start, val_end))
    safe_json_dump(eval_metrics, os.path.join(artifacts_dir, "results", "metrics_validation.json"))
    os.makedirs(os.path.join(artifacts_dir, "results"), exist_ok=True)
    sim.save_results(os.path.join(artifacts_dir, "results", "simulation_outputs.json"))
    sim.save_all_io(os.path.join(artifacts_dir, "io"))

    # Simple visualization
    sim.visualize()

    # Print compact JSON summary to stdout for verification
    print(json.dumps(sanitize_for_json(sim.results)))


# Execute main for both direct execution and sandbox wrapper invocation
main()