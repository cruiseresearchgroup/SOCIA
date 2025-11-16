def main():
    pass

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
    pass  # Required 'pass' to ensure syntactic correctness
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.

    Parameters:
        x (float): Input value.

    Returns:
        float: Sigmoid(x).
    """
    pass  # Required 'pass' to ensure syntactic correctness
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
    pass  # Required 'pass' to ensure syntactic correctness
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
        political_orientation (str): Political orientation category (e.g., 'left', 'center', 'right').
        workplace_type (str): Workplace type (e.g., 'office', 'service', 'industrial', 'none').
        location_preference (str): Preferred location type to visit.
        habit_strength (float): Habit strength for wearing [0,1].
        fatigue_level (float): Fatigue level [0,1] that reduces propensity to wear.
        mask_quality (float): Quality of mask owned [0,1].
        compliance_cost_sensitivity (float): Sensitivity to compliance/enforcement costs [0,1].
    """
    pass  # Required 'pass' to ensure syntactic correctness
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
    # FIXED: Extend Person with additional attributes per spec feedback
    political_orientation: str = "center"
    workplace_type: str = "office"
    location_preference: str = "retail"
    habit_strength: float = 0.0
    fatigue_level: float = 0.0
    mask_quality: float = 0.5
    compliance_cost_sensitivity: float = 0.5


@dataclass
class Household:
    """
    Represents a household grouping of persons. Used for sharing masks and norm reinforcement.

    Attributes:
        id (int): Household ID.
        member_ids (List[int]): IDs of members in the household.
        norm_strength (float): Strength of within-household norm reinforcement [0, 1].
    """
    pass  # Required 'pass' to ensure syntactic correctness
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
    pass  # Required 'pass' to ensure syntactic correctness
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
    Legacy policy authority (retained for backward compatibility). Not used in the updated simulation.
    """
    pass  # Required 'pass' to ensure syntactic correctness
    id: int = 0
    mandate_status: bool = False
    mandate_start_day: int = 0
    mandate_scope: str = "indoor_public"
    enforcement_probability: float = 0.0
    fine_amount: float = 0.0
    communication_strategy: str = "risk_and_norms"


@dataclass
class PolicyMaker:
    """
    Dynamic policy maker implementing mandate activation/deactivation, enforcement adjustments,
    and information campaign intensity per feedback.

    Attributes:
        mandate_active (bool): Is a mandate currently active.
        enforcement_probability (float): Base enforcement probability [0,1].
        fine_amount (float): Fine amount for violations.
        subsidy_amount (float): Subsidy lowering effective price.
        communication_intensity (float): Intensity of information campaign [0,1].
        activation_threshold (float): Adoption threshold below which mandate is enacted.
        deactivation_threshold (float): Adoption threshold above which mandate is lifted.
    """
    pass  # Required 'pass' to ensure syntactic correctness
    mandate_active: bool = False
    enforcement_probability: float = 0.3
    fine_amount: float = 50.0
    subsidy_amount: float = 0.0
    communication_intensity: float = 0.0
    activation_threshold: float = 0.7
    deactivation_threshold: float = 0.6

    def enact_or_lift_mandate(self, adoption_recent: float) -> None:
        """
        Enact or lift mandate based on recent adoption level.

        Parameters:
            adoption_recent (float): Recent (e.g., 7-day mean) adoption share.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Dynamic mandate logic per feedback
        if not self.mandate_active and adoption_recent < self.activation_threshold:
            self.mandate_active = True
        elif self.mandate_active and adoption_recent > self.deactivation_threshold:
            self.mandate_active = False

    def adjust_enforcement(self, recent_violations: int) -> None:
        """
        Adjust enforcement probability based on recent violations.

        Parameters:
            recent_violations (int): Count of violations yesterday.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Adaptive enforcement
        delta = 0.05 * sigmoid(recent_violations - 5)
        self.enforcement_probability = clamp(self.enforcement_probability + delta, 0.0, 1.0)

    def run_information_campaign(self, observed_risk: float) -> None:
        """
        Update communication intensity based on observed risk and mandate status.

        Parameters:
            observed_risk (float): Current health risk prevalence in [0,1].
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Campaign intensity responds to risk and mandate
        base = 0.5 * observed_risk + 0.2 * (1.0 if self.mandate_active else 0.0)
        self.communication_intensity = clamp(base, 0.0, 1.0)


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
        channel_type (str): Type label (e.g., 'official', 'social').
    """
    pass  # Required 'pass' to ensure syntactic correctness
    id: int
    message_bias: float
    credibility: float
    reach: float
    misinformation_rate: float
    message_frequency: float
    channel_type: str = "official"


@dataclass
class Retailer:
    """
    Retailer selling masks, with inventory, restocking policy, pricing, and rationing.

    Attributes:
        inventory (int): Current inventory level.
        restock_quantity (int): Quantity added on each restock event.
        restock_interval (int): Interval in days between planned restocks.
        price (float): Current mask unit price.
        max_purchase_per_customer (int): Cap on purchase per customer.
        supply_variability (float): Relative std for restock noise.
        stockout_days (int): Number of days with zero inventory (legacy).
        rationing_policy (str): 'none', 'per_customer_cap', 'per_day_cap'.
    """
    pass  # Required 'pass' to ensure syntactic correctness
    inventory: int
    restock_quantity: int
    restock_interval: int
    price: float
    max_purchase_per_customer: int
    supply_variability: float
    stockout_days: int = 0
    rationing_policy: str = "per_customer_cap"

    def restock(self, day: int, rng: random.Random) -> int:
        """
        Restock inventory on schedule with variability.

        Parameters:
            day (int): Current simulation day (0-indexed).
            rng (random.Random): Seeded RNG.

        Returns:
            int: Quantity added to inventory.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Implement retailer restock with variability and cap negative quantities
        added = 0
        if self.restock_interval > 0 and (day % self.restock_interval == 0):
            noise = int(self.restock_quantity * rng.gauss(0, self.supply_variability))
            qty = max(0, self.restock_quantity + noise)
            self.inventory += qty
            added = qty
        return added

    def adjust_price(self, demand_today: int, sold_today: int, price_elasticity: float) -> None:
        """
        Adjust price based on demand-supply stress and price elasticity.

        Parameters:
            demand_today (int): Units demanded today.
            sold_today (int): Units actually sold today.
            price_elasticity (float): Elasticity parameter scaling response.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Basic pricing rule - increase price when unmet demand is high and decrease otherwise
        if demand_today <= 0:
            return
        unmet = max(0, demand_today - sold_today)
        stress = unmet / max(1.0, demand_today)
        # Move price towards 20% higher when stress=1, lower when negative (oversupply)
        target_factor = 1.0 + 0.2 * (2 * stress - 1.0)
        # Smooth adjustment
        self.price = max(0.1, self.price * (1.0 + price_elasticity * (target_factor - 1.0)))


@dataclass
class HealthEnvironment:
    """
    HealthEnvironment encapsulates the exogenous/endogenous risk signal that informs risk perception.

    Attributes:
        risk_signal_series (Optional[List[float]]): Optional exogenous prevalence series in [0,1].
        current_prevalence (float): Current prevalence level in [0,1].
        baseline_growth (float): Baseline daily change when no series is provided.
        seasonality_amplitude (float): Seasonal variation amplitude for synthetic generation.
    """
    pass  # Required 'pass' to ensure syntactic correctness
    risk_signal_series: Optional[List[float]] = None
    current_prevalence: float = 0.1
    baseline_growth: float = 0.0
    seasonality_amplitude: float = 0.0

    def update_risk_signal(self, day: int) -> None:
        """
        Update the current prevalence based on the provided series or a simple synthetic process.

        Parameters:
            day (int): Current day index.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Use exogenous series when available
        if self.risk_signal_series is not None and day < len(self.risk_signal_series):
            self.current_prevalence = clamp(float(self.risk_signal_series[day]), 0.0, 1.0)
            return
        # Otherwise, synthetic low-variance seasonality around 0.1
        seasonal = 0.1 + self.seasonality_amplitude * (0.5 * (1.0 + math.sin(2 * math.pi * day / 30.0)))
        drift = self.baseline_growth * day
        self.current_prevalence = clamp(seasonal + drift, 0.0, 1.0)


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
    pass  # Required 'pass' to ensure syntactic correctness
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
    - Builds agents, households, locations, policy, media, retailers, and a small-world network.
    - Runs a daily loop including messaging (with misinformation), peer and household influence, purchasing, visits,
      scoped enforcement, and norm observation by location, with dynamic health risk feedback.
    - Aggregates and outputs required metrics, including evaluation with pass/fail flags.

    Notes:
        - Designed to be Docker-independent and directly executable.
        - Incorporates HealthEnvironment and dynamic PolicyMaker per feedback.
    """
    pass  # Required 'pass' to ensure syntactic correctness

    def __init__(self, params: Dict[str, Any], smoke: bool = False) -> None:
        """
        Initialize the simulation with parameters and optional smoke (fast) mode.

        Parameters:
            params (Dict[str, Any]): Simulation parameters.
            smoke (bool): If True, run a small, fast simulation for CI.

        Raises:
            ValueError: If any parameters are invalid.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Added RNG seeding for reproducibility
        self.params = dict(params)
        self.rng = random.Random(self.params.get('random_seed', 42))
        # FIXED: Align parameter names per feedback (mapping aliases)
        self._map_parameter_aliases()

        self.days = 5 if smoke else int(self.params.get('simulation_horizon_days', 120))
        self.N = 200 if smoke else int(self.params.get('population_size', 5000))
        self.smoke = smoke

        # Parameter validation and clamping
        self._validate_and_default_params()

        # Build network
        self.network = ring_small_world(
            self.N,
            int(self.params.get('avg_degree', 8)),
            float(self.params.get('network_rewiring_prob', 0.1)),
            self.rng,
        )

        # Initialize containers
        self.people: List[Person] = []
        self.households: List[Household] = []
        self.locations: List[Location] = []
        self.locations_by_type: Dict[str, List[Location]] = {}

        # FIXED: Added HealthEnvironment entity and integrated risk feedback
        self.health_env = HealthEnvironment(
            risk_signal_series=self.params.get('risk_signal_series', None),
            current_prevalence=float(self.params.get('initial_prevalence', 0.1)),
            baseline_growth=float(self.params.get('baseline_growth', 0.0)),
            seasonality_amplitude=float(self.params.get('seasonality_amplitude', 0.0)),
        )

        # FIXED: Replace PolicyAuthority with dynamic PolicyMaker
        self.policymaker = PolicyMaker(
            mandate_active=bool(self.params.get('mandate_active', False)),
            enforcement_probability=float(self.params.get('enforcement_probability', 0.3)),
            fine_amount=float(self.params.get('fine_amount', 50.0)),
            subsidy_amount=float(self.params.get('subsidy_amount', 0.0)),
            communication_intensity=float(self.params.get('info_campaign_intensity', 0.0)),
            activation_threshold=float(self.params.get('activation_threshold', 0.7)),
            deactivation_threshold=float(self.params.get('deactivation_threshold', 0.6)),
        )

        # FIXED: Build Media channels from mix and valence distributions
        self.media_channels: List[MediaChannel] = self._build_media_channels()

        # FIXED: Multiple retailers with rationing and pricing
        self.retailers: List[Retailer] = self._build_retailers()

        # Build population, households, and locations
        self._build_entities()

        # Time series for metrics
        self.overall_adoption_series: List[float] = []
        self.adoption_by_loc_series: Dict[str, List[float]] = {}
        self.policy_violations_per_day: List[int] = []
        self.avg_perceived_risk_series: List[float] = []
        self.prev_loc_mask_rate: Dict[str, float] = {}

        # Market tracking
        self.total_purchased: int = 0  # legacy count
        self.total_units_demanded: int = 0  # Track demand
        self.total_units_supplied: int = 0  # Track supply
        self.total_spent: float = 0.0  # Track prices paid
        self.stockout_events: int = 0  # FIXED: Count stockout events per day/retailer transitions

        # Dynamic model plan structure (minimal to support evaluation)
        self.model_plan: Dict[str, Any] = {
            "evaluation_metrics": [
                "adoption_curve_convergence",
                "mandate_effect_direction",
                "bounded_stockouts"
            ],
        }

    def _map_parameter_aliases(self) -> None:
        """
        Map specification-aligned parameter names to internal ones for backward compatibility.

        This ensures that configurations from external specifications are correctly recognized.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # FIXED: Apply alias mapping per feedback
        alias = {
            'time_horizon_days': 'simulation_horizon_days',
            'average_degree': 'avg_degree',
            'risk_signal_series': 'risk_signal_series',
            'location_policy_rate': 'location_mask_policy_rate',
            'workplace_mask_policy_rate': 'workplace_policy_rate',
            'policy_subsidy_amount': 'subsidy_amount',
            'supply_capacity_per_day': 'supply_capacity_per_day',
            'adoption_target': 'adoption_threshold_target',
        }
        for k_src, k_dst in alias.items():
            if k_src in self.params and k_dst not in self.params:
                self.params[k_dst] = self.params[k_src]

    def _validate_and_default_params(self) -> None:
        """
        Validate and clamp key parameters into valid ranges. Provide sensible defaults where necessary.

        Raises:
            ValueError: If critical parameters are invalid after clamping.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        # Probabilities clamp
        prob_keys = [
            'policy_enforcement_strictness', 'campaign_intensity', 'authority_credibility',
            'observation_effect_size', 'enforcement_probability', 'location_mask_policy_rate',
            'workplace_policy_rate', 'enforcement_effect_on_compliance', 'info_campaign_intensity'
        ]
        for k in prob_keys:
            if k in self.params:
                self.params[k] = clamp(float(self.params[k]), 0.0, 1.0)

        # Non-negative values
        nonneg_keys = ['mask_price', 'fine_amount', 'restock_quantity', 'retailer_initial_inventory',
                       'max_purchase_per_customer', 'subsidy_amount', 'supply_capacity_per_day']
        for k in nonneg_keys:
            if k in self.params:
                self.params[k] = max(0.0, float(self.params[k]))

        # Defaults aligned to spec
        self.params.setdefault('location_mask_policy_rate', 0.5)
        self.params.setdefault('workplace_policy_rate', 0.3)
        self.params.setdefault('adoption_threshold_target', 0.7)
        self.params.setdefault('media_channel_mix', {'official': 0.5, 'social': 0.5})
        self.params.setdefault('message_valence_distribution', {'pro_mask': 0.6, 'neutral': 0.2, 'anti_mask': 0.2})
        self.params.setdefault('enforcement_effect_on_compliance', 0.25)
        self.params.setdefault('norm_smoothing_alpha', 0.5)
        self.params.setdefault('num_retailers', 10)
        self.params.setdefault('supply_capacity_per_day', 1500.0)
        self.params.setdefault('initial_inventory_per_retailer', 500)
        self.params.setdefault('restock_interval_days', 7)
        self.params.setdefault('mask_price', 2.0)
        self.params.setdefault('price_elasticity_of_demand', 0.2)
        self.params.setdefault('rationing_policy', 'per_customer_cap')
        self.params.setdefault('max_purchase_per_customer', 10)
        self.params.setdefault('habit_formation_rate', 0.2)
        self.params.setdefault('fatigue_rate', 0.05)
        self.params.setdefault('forgetting_rate', 0.02)  # used for fatigue/habit decay
        self.params.setdefault('daily_outing_probability', 0.6)

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

    def _build_media_channels(self) -> List[MediaChannel]:
        """
        Construct media channels using the configured media_channel_mix and message_valence_distribution.

        Returns:
            List[MediaChannel]: List of MediaChannel instances.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        mix = self.params.get('media_channel_mix', {'official': 0.5, 'social': 0.5})
        valence = self.params.get('message_valence_distribution', {'pro_mask': 0.6, 'neutral': 0.2, 'anti_mask': 0.2})
        channels: List[MediaChannel] = []
        cid = 1
        for ch_type, weight in mix.items():
            # Pro-mask bias
            pro_share = valence.get('pro_mask', 0.6)
            anti_share = valence.get('anti_mask', 0.2)
            bias = pro_share - anti_share  # [-1,1] approx
            credibility = 0.7 if ch_type == 'official' else 0.5
            reach = clamp(float(weight), 0.0, 1.0)
            misinformation = 0.05 if ch_type == 'official' else clamp(0.2 + 0.3 * anti_share, 0.0, 1.0)
            channels.append(MediaChannel(
                id=cid,
                message_bias=bias,
                credibility=credibility,
                reach=reach,
                misinformation_rate=misinformation,
                message_frequency=1.0,
                channel_type=ch_type
            ))
            cid += 1
        return channels

    def _build_retailers(self) -> List[Retailer]:
        """
        Construct a list of retailers per specification with restocking and rationing.

        Returns:
            List[Retailer]: Retailers in the simulation.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        num = int(self.params.get('num_retailers', 10))
        supply_capacity = float(self.params.get('supply_capacity_per_day', 1500.0))
        per_day_per_retailer = max(1, int(supply_capacity / max(1, num)))
        retailers: List[Retailer] = []
        for _ in range(num):
            retailers.append(Retailer(
                inventory=int(self.params.get('initial_inventory_per_retailer', 500)),
                restock_quantity=per_day_per_retailer * int(self.params.get('restock_interval_days', 7)),
                restock_interval=int(self.params.get('restock_interval_days', 7)),
                price=float(self.params.get('mask_price', 2.0)),
                max_purchase_per_customer=int(self.params.get('max_purchase_per_customer', 10)),
                supply_variability=float(self.params.get('supply_variability', 0.1)),
                rationing_policy=str(self.params.get('rationing_policy', 'per_customer_cap')),
            ))
        return retailers

    def _build_entities(self) -> None:
        """
        Construct initial Persons, Households, and Locations.

        Notes:
            - Individuals initialized with random attributes using seeded RNG.
            - Households created with a simple size distribution.
            - Locations include a variety of types with different enforcement and mask policies.
        """
        pass  # Required 'pass' to ensure syntactic correctness
        init_inv = int(self.params.get('initial_inventory_per_person', 2))
        init_adopt = float(self.params.get('initial_adoption_rate', 0.2))
        peer_w = float(self.params.get('peer_influence_weight', 0.2))

        # Build households with sizes drawn from a simple distribution
        sizes = []
        remaining = self.N
        hh_id = 0
        while remaining > 0:
            size = min(remaining, max(1, int(self.rng.choice([1, 2, 3, 4, 5]))))
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
                age_group = self.rng.choice(['child', 'adult', 'senior'])
                health_risk_level = clamp(self.rng.gauss(0.5, 0.2), 0.0, 1.0)
                budget = self.rng.uniform(20, 300)
                cost_sensitivity = clamp(self.rng.gauss(0.5, 0.2), 0.0, 1.0)
                political_orientation = self.rng.choice(['left', 'center', 'right'])
                workplace_type = self.rng.choice(['office', 'service', 'industrial', 'none'])
                location_preference = self.rng.choice(['retail', 'park', 'transit', 'workplace', 'school'])
                habit_strength = 1.0 if adopt else 0.0
                fatigue_level = 0.0
                mask_quality = clamp(self.rng.gauss(0.6, 0.15), 0.0, 1.0)
                compliance_cost_sensitivity = clamp(self.rng.gauss(0.5, 0.2), 0.0, 1.0)
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
                    political_orientation=political_orientation,
                    workplace_type=workplace_type,
                    location_preference=location_preference,
                    habit_strength=habit_strength,
                    fatigue_level=fatigue_level,
                    mask_quality=mask_quality,
                    compliance_cost_sensitivity=compliance_cost_sensitivity,
                )
                self.people.append(person)
                member_ids.append(person_idx)
                person_idx += 1
            if member_ids:
                self.households.append(Household(id=hid, member_ids=member_ids, norm_strength=0.3))
                hh_id += 1

        # Attach network neighbors
        for i, p in enumerate(self.people):
            p.network_neighbors = self.network.get(i, [])

        # Build a set of diverse locations with policy rates
        location_mix = self.params.get('location_mix_distribution', {})
        loc_types_config = [
            ('workplace', 0.6, 0.6),
            ('transit', 0.7, 0.7),
            ('retail', 0.6, 0.7),
            ('school', 0.5, 0.7),
            ('park', 0.1, 0.4),
        ]
        lid = 0
        for ltype, enf, obs in loc_types_config:
            share = location_mix.get(ltype, 0.2)
            count = max(1, int(share * 20))
            cap = int(50 + 100 * share)
            for _ in range(count):
                # FIXED: Create and register Location instances
                if ltype == 'workplace':
                    policy_rate = float(self.params.get('workplace_policy_rate', 0.3))
                else:
                    policy_rate = float(self.params.get('location_mask_policy_rate', 0.5))
                require_policy = self.rng.random() < clamp(policy_rate, 0.0, 1.0)
                loc = Location(
                    id=lid,
                    type=ltype,
                    capacity=cap,
                    mask_requirement_policy=require_policy,
                    enforcement_strictness=float(enf),
                    foot_traffic=float(share),
                    observability_factor=float(obs),
                )
                self.locations.append(loc)
                self.locations_by_type.setdefault(ltype, []).append(loc)
                lid += 1


# Execute main for both direct execution and sandbox wrapper invocation
main()