import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path handling as instructed
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

logger = logging.getLogger(__name__)


def has_docker() -> bool:
    """
    Check if Docker is available on the system PATH.

    Returns:
        bool: True if Docker is available, False otherwise.
    """
    # FIXED: Added Docker guard to avoid failures in environments without Docker.
    return shutil.which("docker") is not None


def _format_json_error(e: json.JSONDecodeError, text: str, path: str) -> str:
    """
    Format a detailed JSON parsing error message with context snippet.

    Args:
        e (json.JSONDecodeError): The JSON decoding error.
        text (str): The JSON text that failed to parse.
        path (str): The file path or '<stdin>' if from stdin.

    Returns:
        str: A detailed error message string.
    """
    # FIXED: Added robust JSON error formatting with line, column, and snippet context.
    line = e.lineno
    col = e.colno
    start = max(0, e.pos - 40)
    end = min(len(text), e.pos + 40)
    snippet = text[start:end].replace("\n", "\\n")
    return (
        f"Failed to parse JSON in {path or '<stdin>'}: {e.msg} at line {line}, column {col}. "
        f"Context: ...{snippet}..."
    )


def load_json(path: Optional[str]) -> Dict[str, Any]:
    """
    Load JSON from a file path or stdin. Returns an empty dict if path is None or
    file is missing. Exits with code 3 on JSON parsing error.
    """
    if path is None:
        logger.info("No input path provided; using empty configuration.")
        return {}
    if path == "-":
        text = sys.stdin.read()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            msg = _format_json_error(e, text, "<stdin>")
            logger.error(msg)
            raise SystemExit(3)
    p = Path(path)
    if not p.exists():
        logger.warning("Input file not found: %s; proceeding with empty configuration", path)
        return {}
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        msg = _format_json_error(e, text, str(p))
        logger.error(msg)
        raise SystemExit(3)


def save_json(obj: Dict[str, Any], path: Optional[str]) -> None:
    """
    Save a dictionary as pretty-printed JSON to a file or stdout.
    """
    out = json.dumps(obj, ensure_ascii=False, indent=2)
    if path:
        Path(path).write_text(out + "\n", encoding="utf-8")
    else:
        sys.stdout.write(out + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the simulation CLI.
    """
    p = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation (agent-based, networked)")
    p.add_argument("--input", "-i", default=None, help="Path to input JSON (use '-' for stdin)")
    p.add_argument("--output", "-o", default=None, help="Path to write output JSON (default: stdout)")
    p.add_argument("--csv-out", default=None, help="Optional path to write per-day CSV; omit to disable")
    p.add_argument("--steps", type=int, default=None, help="Number of simulation steps (days)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    p.add_argument("--use-docker", action="store_true", help="Attempt to use Docker (guarded; defaults to off)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return p.parse_args(argv)


def stable_hash_dict(d: Dict[str, Any]) -> str:
    """
    Compute a stable SHA-256 hash of a dictionary by canonical JSON serialization.
    """
    try:
        b = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        b = repr(d).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def gini(values: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Compute the weighted Gini coefficient for non-negative values.
    """
    if not values:
        return 0.0
    if weights is None:
        weights = [1.0] * len(values)
    assert len(values) == len(weights)
    vals = [max(0.0, float(v)) for v in values]
    wts = [max(0.0, float(w)) for w in weights]
    pairs = sorted(zip(vals, wts), key=lambda vw: vw[0])
    W = sum(w for _, w in pairs)
    if W <= 0:
        return 0.0
    X = sum(v * w for v, w in pairs)
    if X <= 0:
        return 0.0
    cumX = 0.0
    B = 0.0
    for v, w in pairs:
        prev = cumX
        cumX += v * w
        B += (prev + cumX) * w / (2.0 * X)
    B = B / W
    G = 1.0 - 2.0 * B
    return max(0.0, min(1.0, G))


@dataclass
class Household:
    """
    Represents a household with shared norms and pooled mask stock.
    """
    hid: int
    members: List[int]
    norm_strength: float
    mask_stock: int = 0

    def discuss_norms(self, people: List["Person"], wearing_prev: List[float]) -> None:
        """
        Update household members' attitudes and compliance propensity based on within-household norms.
        """
        if not self.members:
            return
        avg_att = sum(people[i].mask_attitude for i in self.members) / len(self.members)
        avg_wear_prev = sum(wearing_prev[i] for i in self.members) / len(self.members)
        for i in self.members:
            p = people[i]
            # Nudge attitude toward household mean
            p.mask_attitude = max(0.0, min(1.0, p.mask_attitude + self.norm_strength * (avg_att - p.mask_attitude)))
            # Increase compliance propensity if most wore in household yesterday
            delta_c = self.norm_strength * (avg_wear_prev - 0.5) * 0.2
            p.compliance_propensity = max(0.0, min(1.0, p.compliance_propensity + delta_c))

    def share_masks(self, people: List["Person"]) -> None:
        """
        Share pooled masks among household members prioritizing those with higher adoption intent.
        """
        if not self.members:
            return
        total = self.mask_stock + sum(max(0, people[i].mask_inventory) for i in self.members)
        if total <= 0:
            return
        intents = [(i, people[i].adoption_intent) for i in self.members]
        intents.sort(key=lambda x: x[1], reverse=True)
        for i, _intent in intents:
            people[i].mask_inventory = 0
        allocate = total
        for i, _intent in intents:
            if allocate <= 0:
                break
            people[i].mask_inventory += 1
            allocate -= 1
        self.mask_stock = max(0, allocate)


@dataclass
class InformationChannel:
    """
    Represents an information channel (media or campaign) broadcasting messages to the population.
    """
    cid: int
    reach: float
    credibility: float
    bias: float
    misinformation_rate: float
    message_frequency: float
    target_demographics_tag: str = "All"

    def broadcast_message(self) -> None:
        """
        Placeholder for message generation. Effects applied at person-level.
        """
        pass


@dataclass
class Person:
    """
    Represents an individual in the simulation with attributes related to mask adoption.
    """
    pid: int
    income: float
    risk_perception: float
    trust_in_authorities: float
    trust_in_media: float
    conformity_sensitivity: float
    reactance_tendency: float
    mask_attitude: float
    habit_strength: float
    fatigue_level: float
    perceived_mask_cost: float
    mask_adoption_state: str
    mask_type: str
    neighbors: List[int] = field(default_factory=list)
    location_type: str = "workplace"
    exposure_to_misinformation: float = 0.0
    exposure_to_pro_mask_messages: float = 0.0
    mask_inventory: int = 0
    age: int = 30
    health_risk_level: float = 0.2
    social_influence_susceptibility: float = 0.5
    economic_constraint: float = 0.5
    adoption_intent: float = 0.0
    compliance_propensity: float = 0.5
    info_sources: List[str] = field(default_factory=list)
    daily_schedule: Dict[str, float] = field(default_factory=dict)
    perceived_local_cases: Optional[float] = None
    household_id: int = -1
    is_influencer: bool = False
    group: str = "G0"
    retailer_id: int = -1

    def decide_mask_use(self, rng: random.Random, policy_active: bool, peer_share: float, params: Dict[str, float]) -> None:
        """
        Decide whether to wear a mask today based on a simplified utility model.
        """
        policy_influence_weight = float(params.get("policy_influence_weight", 0.3))
        personal_risk_weight = float(params.get("personal_risk_weight", 0.3))
        peer_influence_weight = float(params.get("peer_influence_weight", 0.4))
        habit_formation_rate = float(params.get("habit_formation_rate", 0.05))
        forgetting_rate = float(params.get("forgetting_rate", 0.02))
        fatigue_growth_rate = float(params.get("fatigue_growth_rate", 0.01))
        fatigue_recovery_rate = float(params.get("fatigue_recovery_rate", 0.02))
        disutility_cost_scale = float(params.get("disutility_cost_scale", 1.0))
        trust_weight = float(params.get("trust_weight", 0.5))
        baseline_noise = float(params.get("baseline_compliance_noise_scale", 0.1))
        mask_effectiveness_weight = float(params.get("mask_effectiveness_weight", 0.4))
        observed_norms_influence_weight = float(params.get("observed_norms_influence_weight", 0.2))
        observed_location_norm = float(params.get("observed_location_norm", 0.0))
        penalty_benefit = float(params.get("penalty_benefit", 0.0))
        attitude_weight = float(params.get("attitude_weight", 0.3))
        intent_weight = float(params.get("intent_weight", 0.25))
        comp_weight = float(params.get("compliance_propensity_weight", 0.2))
        health_weight = float(params.get("health_risk_weight", 0.2))
        econ_weight = float(params.get("economic_constraint_weight", 0.2))

        policy_term = policy_influence_weight * (1.0 if policy_active else 0.0) * self.trust_in_authorities
        risk_term = personal_risk_weight * self.risk_perception * mask_effectiveness_weight
        media_term = trust_weight * self.trust_in_media * (self.exposure_to_pro_mask_messages - self.exposure_to_misinformation)
        peer_term = peer_influence_weight * (peer_share - 0.5)
        habit_term = habit_formation_rate * self.habit_strength
        norm_term = observed_norms_influence_weight * (observed_location_norm - 0.5)
        cost_term = disutility_cost_scale * self.perceived_mask_cost + self.fatigue_level + econ_weight * self.economic_constraint
        intent_term = intent_weight * (self.adoption_intent - 0.5)
        comp_term = comp_weight * (self.compliance_propensity - 0.5)
        health_term = health_weight * (self.health_risk_level - 0.2)

        latent = (
            policy_term + risk_term + media_term + peer_term + habit_term + norm_term
            + attitude_weight * (self.mask_attitude - 0.5)
            + intent_term + comp_term + health_term
            - cost_term
        )
        latent += penalty_benefit

        noise = rng.uniform(-baseline_noise, baseline_noise)
        p_wear = sigmoid(latent + noise)
        wearing = rng.random() < p_wear

        if wearing:
            self.mask_adoption_state = "wearing"
            self.habit_strength = min(1.0, self.habit_strength + habit_formation_rate)
            self.fatigue_level = max(0.0, self.fatigue_level - fatigue_recovery_rate)
            if self.risk_perception > 0.6:
                self.mask_type = "N95"
            elif self.income < 50:
                self.mask_type = "cloth"
            else:
                self.mask_type = "surgical"
        else:
            self.mask_adoption_state = "none"
            self.mask_type = "none"
            self.habit_strength = max(0.0, self.habit_strength - forgetting_rate)
            self.fatigue_level = min(1.0, self.fatigue_level + fatigue_growth_rate)

    def update_attitude_from_peers(self, peer_mean_attitude: float, policy_active: bool, params: Dict[str, float]) -> None:
        """
        Update the mask attitude via peer influence with optional reactance.
        """
        peer_influence_weight = float(params.get("peer_influence_weight", 0.4)) * self.social_influence_susceptibility
        reactance_weight = float(params.get("reactance_weight", 0.2))
        delta = peer_influence_weight * (peer_mean_attitude - self.mask_attitude)
        reactance = reactance_weight * self.reactance_tendency * (-1.0 if delta > 0 else 1.0) if policy_active else 0.0
        self.mask_attitude = max(0.0, min(1.0, self.mask_attitude + delta + reactance))
        memory = float(params.get("social_influence_memory_decay", 0.3))
        self.adoption_intent = max(0.0, min(1.0, (1 - memory) * self.adoption_intent + memory * ((peer_mean_attitude + self.mask_attitude) / 2.0)))

    def observe_media(self, rng: random.Random, media_bias_mean: float, misinformation_rate: float, media_message_frequency: float) -> bool:
        """
        Update exposures to media messages.
        """
        exposed = False
        if rng.random() < media_message_frequency:
            tone = media_bias_mean + rng.uniform(-0.05, 0.05)
            exposed = True
            if rng.random() < misinformation_rate:
                self.exposure_to_misinformation = 0.7 * self.exposure_to_misinformation + max(0.0, -tone)
            else:
                self.exposure_to_pro_mask_messages = 0.7 * self.exposure_to_pro_mask_messages + max(0.0, tone)
        return exposed

    def visit_location(self, rng: random.Random, locations: List["Location"], mobility_rate: float, mix: Dict[str, float]) -> Optional[int]:
        """
        Choose a location to visit today based on a mobility rate and location type mix.
        """
        if rng.random() >= mobility_rate:
            return None
        mapping = {
            "home": "home",
            "work": "workplace",
            "retail": "retail",
            "transit": "transit",
            "other": "leisure",
            "outdoor": "outdoor",
            "workplace": "workplace",
            "leisure": "leisure",
        }
        dist_map: Dict[str, float] = {}
        total = 0.0
        for k, v in (mix or {}).items():
            t = mapping.get(k, k)
            dist_map[t] = dist_map.get(t, 0.0) + float(v)
            total += float(v)
        if total <= 0:
            dist_map = {"workplace": 0.2, "retail": 0.2, "leisure": 0.2, "transit": 0.2, "outdoor": 0.2}
        r = rng.random()
        acc = 0.0
        chosen_type = None
        s = sum(dist_map.values())
        if s <= 0:
            dist_map = {"retail": 1.0}
            s = 1.0
        for t, w in dist_map.items():
            acc += w / s
            if r <= acc:
                chosen_type = t
                break
        if chosen_type is None:
            chosen_type = "retail"
        cand = [loc for loc in locations if loc.type == chosen_type]
        if not cand:
            return None
        loc = rng.choice(cand)
        return loc.lid


@dataclass
class Location:
    """
    Represents a location with policy enforcement and crowding parameters.
    """
    lid: int
    type: str
    capacity: int
    mask_requirement: bool
    policy_enforcement_level: float
    signage_visible: float
    crowding_level: float
    observed_norm: float = 0.0
    enforcement_probability: float = 0.5
    fine_amount: float = 50.0
    last_compliance_rate: float = 0.0

    def host_visits(self, agent_ids: List[int]) -> List[int]:
        """
        Admit agents up to capacity.
        """
        return agent_ids[: self.capacity]

    def enforce_policy(self, rng: random.Random, people: List[Person]) -> Tuple[float, int]:
        """
        Enforce mask policy for admitted agents.
        """
        fines_value = 0.0
        denied = 0
        for p in people:
            if self.mask_requirement and p.mask_adoption_state != "wearing":
                if rng.random() < self.enforcement_probability:
                    if rng.random() < 0.5:
                        fines_value += self.fine_amount
                    else:
                        denied += 1
        return fines_value, denied

    def record_compliance(self, people: List[Person]) -> float:
        """
        Record compliance among admitted agents.
        """
        if not people:
            self.last_compliance_rate = 0.0
        else:
            self.last_compliance_rate = sum(1 for p in people if p.mask_adoption_state == "wearing") / len(people)
        return self.last_compliance_rate

    def broadcast_norms(self) -> None:
        """
        Update observed norms for use by entrants on the next day.
        """
        self.observed_norm = self.last_compliance_rate


@dataclass
class PolicyAuthority:
    """
    Represents a policy authority that can activate mandates and publish messaging.
    """
    jurisdiction: str
    mandate_status: bool
    mandate_start_day: int
    enforcement_strength: float
    penalty_amount: float
    exemptions_rate: float = 0.1
    communication_strategy_intensity: float = 0.6
    communication_frequency: float = 0.5
    message_tone: str = "prosocial"

    def issue_mandate(self, day: int, enabled: bool) -> None:
        """
        Activate or keep mandate based on day and configuration.
        """
        self.mandate_status = bool(enabled and (day >= self.mandate_start_day))

    def lift_mandate(self) -> None:
        """
        Lift any active mandate.
        """
        self.mandate_status = False

    def publish_case_counts(self, day: int, external_series: Optional[List[float]], people: List[Person], fallback_prevalence: float) -> None:
        """
        Publish or broadcast perceived case counts to people.
        """
        value = None
        if external_series and 0 <= day < len(external_series):
            try:
                value = float(external_series[day])
            except Exception:
                value = None
        signal = value if value is not None else fallback_prevalence
        for person in people:
            person.perceived_local_cases = signal

    def adjust_messaging(self, people: List[Person], intensity: float) -> None:
        """
        Adjust messaging by nudging individual risk perceptions toward perceived local cases.
        """
        for person in people:
            k = float(intensity) * person.trust_in_authorities
            sig = person.perceived_local_cases if person.perceived_local_cases is not None else person.risk_perception
            person.risk_perception = max(0.0, min(1.0, (1 - k) * person.risk_perception + k * sig))

    def communicate_guidance(self) -> None:
        """
        Placeholder for communication effects on attitudes/trust.
        """
        pass

    def adjust_enforcement(self) -> None:
        """
        Placeholder for adjusting enforcement parameters.
        """
        pass


@dataclass
class Retailer:
    """
    Represents a retailer that sells masks, manages inventory, and adjusts prices.
    """
    rid: int
    region_id: str
    mask_stock: int
    price: float
    restock_rate: int
    supply_lead_time: int
    _incoming: int = 0
    _eta: int = 0
    stockout_days: int = 0

    def sell_masks(self, qty: int) -> int:
        """
        Sell up to qty masks if inventory is available.
        """
        sold = min(qty, self.mask_stock)
        self.mask_stock -= sold
        return sold

    def restock(self) -> None:
        """
        Process restock deliveries or place a new order if inventory is below the daily restock rate.
        """
        if self._eta > 0:
            self._eta -= 1
            if self._eta == 0:
                self.mask_stock += self._incoming
                self._incoming = 0
        elif self.mask_stock < self.restock_rate:
            self._incoming += self.restock_rate
            self._eta = max(1, self.supply_lead_time)

    def adjust_price(self, elasticity: float) -> None:
        """
        Adjust price based on scarcity.
        """
        scarcity = 1.0 / max(1, self.mask_stock)
        self.price = max(0.2, self.price * (1.0 + elasticity * 0.05 * scarcity))


@dataclass
class SupplyChain:
    """
    Represents a simple supply chain for mask availability and pricing.
    """
    available_mask_stock: int
    price_cloth: float
    price_surgical: float
    price_N95: float
    restock_rate: int
    shortage_probability: float

    def update_prices(self) -> None:
        """
        Minimal price update logic responsive to inventory level.
        """
        inv = max(1, self.available_mask_stock)
        scarcity = 1.0 / math.sqrt(inv)
        self.price_cloth = max(0.5, self.price_cloth * (1.0 + 0.01 * scarcity))
        self.price_surgical = max(0.5, self.price_surgical * (1.0 + 0.01 * scarcity))
        self.price_N95 = max(0.5, self.price_N95 * (1.0 + 0.01 * scarcity))

    def release_stock(self) -> None:
        """
        Restock inventory by a fixed daily amount.
        """
        self.available_mask_stock += int(self.restock_rate)

    def apply_shortage_shock(self, rng: random.Random) -> bool:
        """
        Apply a shortage shock probabilistically, reducing available stock.
        """
        if rng.random() < self.shortage_probability:
            reduction = int(0.2 * self.available_mask_stock)
            self.available_mask_stock = max(0, self.available_mask_stock - reduction)
            return True
        return False


@dataclass
class EpidemicEnvironment:
    """
    Represents the epidemic environment providing prevalence/risk signals.
    """
    region_id: str
    prevalence: float
    trend: str

    def update_prevalence(self, rng: random.Random) -> float:
        """
        Update prevalence via a bounded random walk influenced by trend.
        """
        drift = {"increasing": 0.002, "stable": 0.0, "decreasing": -0.002}.get(self.trend, 0.0)
        noise = rng.uniform(-0.001, 0.001)
        self.prevalence = max(0.0, min(1.0, self.prevalence + drift + noise))
        if rng.random() < 0.02:
            self.trend = rng.choice(["increasing", "stable", "decreasing"])
        return self.prevalence


class SimulationEnvironment:
    """
    Coordinates entities and provides global context for the simulation.

    This class is a lightweight container and placeholder for future extensions.
    """
    def __init__(self) -> None:
        self.entities: Dict[str, Any] = {}

    def register(self, name: str, entity: Any) -> None:
        self.entities[name] = entity

    def get(self, name: str) -> Any:
        return self.entities.get(name)


class SocialSimulation:
    """
    Main simulation class for mask adoption dynamics.
    """
    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        self.config = config or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = SimulationEnvironment()
        self.params = self._extract_parameters(self.config)
        self.people: List[Person] = []
        self.locations: List[Location] = []
        self.households: List[Household] = []  # FIXED: Added households entity
        self.information_channels: List[InformationChannel] = []  # FIXED: Replace/align with InformationChannel
        self.policy: Optional[PolicyAuthority] = None
        self.supply: Optional[SupplyChain] = None  # kept for backward compatibility
        self.retailers: List[Retailer] = []  # FIXED: Added Retailer entities
        self.epidemic: Optional[EpidemicEnvironment] = None
        self.daily_adoption: List[float] = []
        self.daily_penalties: List[float] = []
        self.daily_penalties_count: List[int] = []
        self.daily_shortages: List[int] = []
        self.policy_active_daily: List[bool] = []
        self.epidemic_prevalence_daily: List[float] = []
        self.daily_workplace_adoption: List[float] = []
        self.daily_mask_quality: List[float] = []
        self.daily_denied_access: List[int] = []
        self.mask_type_counts: Dict[str, int] = {"cloth": 0, "surgical": 0, "N95": 0, "none": 0}
        self.daily_voluntary_rate: List[float] = []
        self.daily_mandate_rate: List[float] = []
        self.daily_policy_violations: List[int] = []
        self.daily_unmet_demand: List[int] = []
        self.daily_attempted_purchases: List[int] = []
        self.daily_stockout_flag: List[bool] = []
        self.daily_stockout_share: List[float] = []  # FIXED: Track retailer-day stockout share
        self.by_loc_daily_series: List[Dict[str, float]] = []
        self.location_compliance_daily: List[Dict[str, float]] = []  # FIXED: Track per-location compliance
        self.daily_incident_cases: List[int] = []  # FIXED: Added epidemic metric series
        self.daily_Rt: List[float] = []  # FIXED: Added Rt tracking
        self.peak_incidence: int = 0  # FIXED: Track peak daily incidence
        self.cumulative_infections_averted: float = 0.0  # FIXED: Counterfactual comparison
        self._inf_status: List[int] = []  # 0 susceptible, 1 infectious, 2 removed
        self._days_infected_left: List[int] = []
        self.cumulative_masks_purchased: int = 0
        self.daily_masks_purchased: List[int] = []
        self.daily_message_reach: List[float] = []  # FIXED: Track message reach correctly
        # FIXED: Behavioral persistence/volatility tracking
        self._streak: List[int] = []
        self._streak_max: List[int] = []
        self.daily_switching_rate: List[float] = []
        self.daily_peer_exposure_index: List[float] = []
        self._initialized = False
        self._daily_records: List[Dict[str, Any]] = []  # Records for CSV

    def _extract_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters and fallback defaults from the model plan.
        """
        params = {}
        for k, v in config.items():
            if k == "parameters" and isinstance(v, dict):
                params.update(v)
            elif k == "parameters" and isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "key" in item and "default" in item:
                        params[item["key"]] = item["default"]
            elif k == "initialization" and isinstance(v, dict):
                params.setdefault("initialization", v)
            elif k == "prediction_period" and isinstance(v, dict):
                params.setdefault("prediction_period", v)
            elif k == "evaluation_metrics" and isinstance(v, list):
                params.setdefault("evaluation_metrics", v)
            else:
                params.setdefault(k, v)

        # FIXED: Map spec parameter names to internal names
        if "simulation_duration_days" in params:
            params["simulation_days"] = int(params["simulation_duration_days"])
        if "initial_mask_adoption_rate" in params:
            params["initial_adoption_rate"] = float(params["initial_mask_adoption_rate"])
        if "rewiring_rate" in params:
            params["rewire_prob"] = float(params["rewiring_rate"])
        if "enforcement_probability" in params:
            params["enforcement_strength"] = float(params["enforcement_probability"])
        if "mandate_active" in params:
            params["mandate_enabled"] = bool(params["mandate_active"])
        if "retailer_initial_stock_per_1000" in params:
            params["retailer_initial_inventory_per_capita"] = float(params["retailer_initial_stock_per_1000"]) / 1000.0
        if "retail_restock_rate_per_day" in params:
            params["restock_rate_per_day"] = float(params["retail_restock_rate_per_day"])

        # Defaults
        params.setdefault("population_size", 1000)
        params.setdefault("simulation_days", int(params.get("time_horizon_days", 30)))
        params.setdefault("time_step_days", int(params.get("timestep_days", 1)))
        params.setdefault("initial_adoption_rate", 0.2)
        params.setdefault("average_degree", int(params.get("avg_degree", 8)))
        params.setdefault("peer_influence_weight", 0.4)
        params.setdefault("policy_influence_weight", 0.3)
        params.setdefault("personal_risk_weight", 0.3)
        params.setdefault("habit_formation_rate", 0.05)
        params.setdefault("forgetting_rate", 0.02)
        params.setdefault("fatigue_growth_rate", 0.01)
        params.setdefault("fatigue_recovery_rate", 0.02)
        params.setdefault("disutility_cost_scale", 1.0)
        params.setdefault("trust_weight", 0.5)
        params.setdefault("baseline_compliance_noise_scale", 0.1)
        params.setdefault("mask_effectiveness_weight", 0.4)
        params.setdefault("mandate_enabled", True)
        params.setdefault("mandate_start_day", int(params.get("policy_start_day", 30)))
        params.setdefault("mandate_end_day", int(params.get("policy_end_day", 120)))
        params.setdefault("enforcement_strength", 0.5)
        params.setdefault("penalty_amount", float(params.get("fine_amount", 50.0)))
        params.setdefault("media_message_frequency", 1.0)
        params.setdefault("media_bias_mean", 0.0)
        params.setdefault("misinformation_rate", 0.15)
        params.setdefault("target_adoption_rate", float(params.get("target_adoption_threshold", 0.8)))
        params.setdefault("did_pre_window_days", 14)
        params.setdefault("did_post_window_days", 14)
        params.setdefault("rewire_prob", float(params.get("rewiring_prob", params.get("network_rewire_prob", 0.05))))
        params.setdefault("observed_norms_influence_weight", 0.2)
        params.setdefault("location_enforcement_mean", 0.5)
        params.setdefault("exemptions_rate", 0.1)
        params.setdefault("attitude_weight", 0.3)
        params.setdefault("price_elasticity_of_demand", float(params.get("price_elasticity", 0.3)))
        params.setdefault("retailer_initial_inventory_per_capita", float(params.get("retailer_inventory_per_capita", 1.0)))
        params.setdefault("restock_rate_per_day", params.get("retailer_restock_rate", 0.1))
        params.setdefault("mask_price_cloth", float(params.get("mask_price", 2.0)))
        params.setdefault("mask_price_surgical", float(params.get("mask_price", 5.0)))
        params.setdefault("mask_price_respirator", float(params.get("mask_price", 12.0)))
        params.setdefault("income_mean", params.get("income_mean", None))
        params.setdefault("income_std", params.get("income_std", None))
        params.setdefault("mask_availability", params.get("mask_availability", None))
        params.setdefault("initial_infected_fraction", 0.005)
        params.setdefault("infectious_period_days", 7)
        params.setdefault("baseline_transmission_rate", 0.08)
        params.setdefault("asymptomatic_fraction", 0.3)
        params.setdefault("mask_source_control_efficacy", 0.5)
        params.setdefault("mask_wearer_protection_efficacy", 0.3)
        params.setdefault("daily_mobility_rate", 0.6)
        params.setdefault("location_mix_home", 0.5)
        params.setdefault("location_mix_work", 0.2)
        params.setdefault("location_mix_retail", 0.1)
        params.setdefault("location_mix_transit", 0.1)
        params.setdefault("location_mix_other", 0.1)
        params.setdefault("homophily_strength", 0.7)  # FIXED: Parameterize homophily strength
        params.setdefault("household_size_mean", 3)
        params.setdefault("messaging_intensity", 0.6)
        params.setdefault("external_cases_weight", 0.5)
        params.setdefault("risk_perception_sensitivity", 0.6)
        params.setdefault("network_topology", params.get("network_type", "small_world"))
        params.setdefault("retailer_count_per_1000_agents", 5)
        params.setdefault("retailer_price_adjust_elasticity", 0.5)
        params.setdefault("retailer_supply_lead_time_days", 3)
        return params

    def initialize(self, steps_override: Optional[int] = None) -> int:
        """
        Initialize entities and network based on parameters and input data.
        """
        N = int(self.params.get("population_size", 1000))
        avg_deg = int(self.params.get("average_degree", self.params.get("avg_degree", 8)))
        initial_rate = float(self.params.get("initial_adoption_rate", 0.2))
        steps = steps_override if steps_override is not None else int(self.params.get("simulation_days", self.params.get("time_horizon_days", 30)))
        pred = self.params.get("prediction_period") or {}
        if isinstance(pred, dict):
            s = pred.get("start_day")
            e = pred.get("end_day")
            if isinstance(s, int) and isinstance(e, int) and e > s >= 0:
                steps = max(steps, e + 1)

        # External perceived cases series
        # FIXED: Support external perceived_cases_series to override/blend simulated prevalence
        self.params["perceived_cases_series_resolved"] = None
        pcs = self.config.get("perceived_cases_series")
        if isinstance(pcs, list):
            try:
                self.params["perceived_cases_series_resolved"] = [float(x) for x in pcs]
            except Exception:
                self.params["perceived_cases_series_resolved"] = None

        # Create people
        loc_types = ["home", "workplace", "retail", "leisure", "transit", "outdoor"]
        income_mean = self.params.get("income_mean", None)
        income_std = self.params.get("income_std", None)
        incomes: List[float] = []
        for i in range(N):
            if income_mean is not None and income_std is not None:
                try:
                    inc = max(0.1, self.rng.gauss(float(income_mean) / 1000.0, float(income_std) / 1000.0))
                except Exception:
                    inc = self.rng.lognormvariate(3.0, 0.4)
            else:
                inc = self.rng.lognormvariate(3.0, 0.4)
            incomes.append(inc)

        # Assign groups by income terciles for inequality metric
        sorted_incomes = sorted(enumerate(incomes), key=lambda x: x[1])
        tercile_index = [None] * N
        for rank, (idx, _) in enumerate(sorted_incomes):
            if rank < N / 3:
                tercile_index[idx] = "low"
            elif rank < 2 * N / 3:
                tercile_index[idx] = "mid"
            else:
                tercile_index[idx] = "high"

        self.people = []
        for i in range(N):
            inc = incomes[i]
            age = int(self.rng.uniform(18, 80))
            health_risk = max(0.0, min(1.0, self.rng.gauss(0.2 + 0.003 * max(0, age - 40), 0.1)))
            p = Person(
                pid=i,
                income=inc,
                risk_perception=max(0.0, min(1.0, self.rng.betavariate(2.0, 3.0))),
                trust_in_authorities=max(0.0, min(1.0, self.rng.betavariate(2.5, 1.8))),
                trust_in_media=max(0.0, min(1.0, self.rng.betavariate(2.2, 2.2))),
                conformity_sensitivity=max(0.0, min(1.0, self.rng.gauss(0.6, 0.15))),
                reactance_tendency=max(0.0, min(1.0, self.rng.gauss(0.3, 0.15))),
                mask_attitude=max(0.0, min(1.0, self.rng.gauss(0.2, 0.2))),
                habit_strength=0.0,
                fatigue_level=0.0,
                perceived_mask_cost=1.0,
                mask_adoption_state="wearing" if self.rng.random() < initial_rate else "none",
                mask_type="none",
                location_type=self.rng.choice(loc_types),
                mask_inventory=1 if self.rng.random() < 0.5 else 0,
                age=age,
                health_risk_level=health_risk,
                social_influence_susceptibility=max(0.0, min(1.0, self.rng.gauss(0.5, 0.2))),
                economic_constraint=max(0.0, min(1.0, self.rng.gauss(0.5, 0.2))),
                adoption_intent=max(0.0, min(1.0, self.rng.random() * 0.3)),
                compliance_propensity=max(0.0, min(1.0, self.rng.gauss(0.5, 0.15))),
                info_sources=["media", "peers"],
                daily_schedule={
                    "home": float(self.params.get("location_mix_home", 0.5)),
                    "work": float(self.params.get("location_mix_work", 0.2)),
                    "retail": float(self.params.get("location_mix_retail", 0.1)),
                    "transit": float(self.params.get("location_mix_transit", 0.1)),
                    "other": float(self.params.get("location_mix_other", 0.1)),
                },
                perceived_local_cases=None,
                household_id=-1,
                is_influencer=(self.rng.random() < float(self.params.get("influencer_fraction", 0.05))),
                group=tercile_index[i],
                retailer_id=-1,
            )
            if p.mask_adoption_state == "wearing":
                r = self.rng.random()
                if r < 0.5:
                    p.mask_type = "cloth"
                elif r < 0.9:
                    p.mask_type = "surgical"
                else:
                    p.mask_type = "N95"
            else:
                p.mask_type = "none"
            self.people.append(p)

        # Build social network
        topology = str(self.params.get("network_topology", "small_world")).lower()
        if topology == "scale_free":
            # FIXED: Add scale-free network support (Barabási–Albert-like)
            m = max(1, min(avg_deg // 2, N - 1))
            adj: List[set] = [set() for _ in range(N)]
            core = max(2, m + 1)
            for i in range(core):
                for j in range(i + 1, core):
                    adj[i].add(j)
                    adj[j].add(i)
            degree_list = [len(adj[i]) for i in range(N)]
            total_degree = sum(degree_list)
            for new_node in range(core, N):
                targets = set()
                tries = 0
                while len(targets) < m and tries < 10 * N:
                    tries += 1
                    pick = self.rng.randrange(new_node) if new_node > 0 else 0
                    if total_degree > 0:
                        if self.rng.random() < (degree_list[pick] + 1) / (total_degree + new_node):
                            targets.add(pick)
                    else:
                        targets.add(pick)
                for t in targets:
                    if t == new_node:
                        continue
                    adj[new_node].add(t)
                    adj[t].add(new_node)
                    degree_list[new_node] += 1
                    degree_list[t] += 1
                    total_degree += 2
        else:
            # Default: Watts–Strogatz-like ring lattice with rewiring
            k = max(2, avg_deg)
            adj = [set() for _ in range(N)]
            half_k = k // 2
            for i in range(N):
                for d in range(1, half_k + 1):
                    j = (i + d) % N
                    adj[i].add(j)
                    adj[j].add(i)
            if k % 2 == 1:
                for i in range(0, N, 2):
                    j = (i + half_k + 1) % N
                    if j != i:
                        adj[i].add(j)
                        adj[j].add(i)
            rewire_p = float(self.params.get("rewire_prob", self.params.get("rewiring_prob", self.params.get("network_rewire_prob", 0.05))))
            all_nodes = list(range(N))

            def pick_new_neighbor(i: int) -> Optional[int]:
                for _ in range(32):
                    j = self.rng.randrange(N)
                    if j != i and j not in adj[i]:
                        return j
                sample_k = min(64, N)
                cand = [j for j in self.rng.sample(all_nodes, k=sample_k) if j != i and j not in adj[i]]
                return self.rng.choice(cand) if cand else None

            for i in range(N):
                for d in range(1, half_k + 1):
                    j = (i + d) % N
                    if self.rng.random() < rewire_p:
                        if j in adj[i]:
                            adj[i].remove(j)
                        if i in adj[j]:
                            adj[j].remove(i)
                        att_i = self.people[i].mask_attitude
                        new_j = pick_new_neighbor(i)
                        if new_j is None:
                            adj[i].add(j)
                            adj[j].add(i)
                            continue
                        hom = float(self.params.get('homophily_strength', 0.7))
                        if self.rng.random() < hom:
                            best = new_j
                            best_gap = abs(self.people[new_j].mask_attitude - att_i)
                            for _ in range(4):
                                t = pick_new_neighbor(i)
                                if t is None:
                                    break
                                gap = abs(self.people[t].mask_attitude - att_i)
                                if gap < best_gap:
                                    best = t
                                    best_gap = gap
                            new_j = best
                        adj[i].add(new_j)
                        adj[new_j].add(i)
        for i in range(N):
            self.people[i].neighbors = list(adj[i])

        # Initialize policy
        self.policy = PolicyAuthority(
            jurisdiction="region_1",
            mandate_status=False,
            mandate_start_day=int(self.params.get("mandate_start_day", 30)),
            enforcement_strength=float(self.params.get("enforcement_strength", 0.5)),
            penalty_amount=float(self.params.get("penalty_amount", 50.0)),
            exemptions_rate=float(self.params.get("exemptions_rate", 0.1)),
            communication_strategy_intensity=float(self.params.get("communication_strategy_intensity", 0.6)),
            communication_frequency=float(self.params.get("policy_communication_frequency", 0.5)),
            message_tone=str(self.params.get("message_tone", "prosocial")),
        )

        # Backward-compat supply (not primary in purchases anymore)
        pp = self.params
        price_cloth = float(pp.get("mask_price_cloth", pp.get("mask_cost_cloth", 2.0)))
        price_surgical = float(pp.get("mask_price_surgical", pp.get("mask_cost_surgical", 5.0)))
        price_respirator = float(pp.get("mask_price_respirator", pp.get("mask_cost_N95", 12.0)))
        rate = float(pp.get("restock_rate_per_day", pp.get("restock_rate_daily", 0.1)))
        if 0.0 < rate <= 1.0:
            restock_per_day = int(round(rate * N))
        else:
            restock_per_day = int(rate)
        init_inv_per_cap = float(pp.get("retailer_initial_inventory_per_capita", pp.get("retailer_inventory_per_capita", 1.0)))
        initial_stock_calc = int(init_inv_per_cap * N)
        self.supply = SupplyChain(
            available_mask_stock=int(initial_stock_calc),
            price_cloth=price_cloth,
            price_surgical=price_surgical,
            price_N95=price_respirator,
            restock_rate=restock_per_day,
            shortage_probability=float(pp.get("shortage_shock_probability", 0.01)),
        )

        # Initialize retailers
        retailers_per_1000 = int(self.params.get("retailer_count_per_1000_agents", 5))
        num_retailers = max(1, int(round((N / 1000.0) * retailers_per_1000)))
        initial_stock_total = int(init_inv_per_cap * N)
        per_retailer_stock = max(0, initial_stock_total // num_retailers)
        base_price = float(self.params.get("mask_price_surgical", 5.0))
        restock_fraction = float(self.params.get("restock_rate_per_day", 0.1))
        restock_each = max(1, int(round((restock_fraction * initial_stock_total) / max(1, num_retailers))))
        lead_time = int(self.params.get("retailer_supply_lead_time_days", 3))
        self.retailers = []
        for rid in range(num_retailers):
            r = Retailer(
                rid=rid,
                region_id="region_1",
                mask_stock=per_retailer_stock,
                price=base_price,
                restock_rate=restock_each,
                supply_lead_time=lead_time,
            )
            self.retailers.append(r)
        # Assign each person to a retailer (nearest/round-robin)
        for i, p in enumerate(self.people):
            p.retailer_id = i % num_retailers

        # Initialize epidemic
        init_inf_frac = float(self.params.get("initial_infected_fraction", 0.005))
        inf_period = int(self.params.get("infectious_period_days", 7))
        self._inf_status = [1 if self.rng.random() < init_inf_frac else 0 for _ in range(N)]
        self._days_infected_left = [inf_period if s == 1 else 0 for s in self._inf_status]
        self.epidemic = EpidemicEnvironment(region_id="region_1", prevalence=init_inf_frac, trend="stable")

        # Initialize locations with enforcement heterogeneity
        self.locations = []
        loc_enf_mean = float(self.params.get("location_enforcement_mean", 0.5))
        fine_base = float(self.params.get("penalty_amount", 50.0))
        for idx, lt in enumerate(["home", "workplace", "retail", "leisure", "transit", "outdoor"]):
            lev = max(0.0, min(1.0, self.rng.gauss(loc_enf_mean, 0.15)))
            self.locations.append(
                Location(
                    lid=idx,
                    type=lt,
                    capacity=int(self.rng.lognormvariate(4.0, 0.5)),
                    mask_requirement=False,
                    policy_enforcement_level=lev,
                    signage_visible=max(0.0, min(1.0, self.rng.uniform(0.3, 0.9))),
                    crowding_level=max(0.0, min(1.0, self.rng.uniform(0.3, 0.9))),
                    observed_norm=0.0,
                    enforcement_probability=lev,
                    fine_amount=max(0.0, self.rng.gauss(fine_base, 0.1 * fine_base)),
                )
            )

        # Initialize information channels
        self.information_channels = [
            InformationChannel(cid=1, reach=0.6, credibility=0.7, bias=0.2, misinformation_rate=0.05, message_frequency=1.0, target_demographics_tag="All"),
            InformationChannel(cid=2, reach=0.4, credibility=0.5, bias=-0.2, misinformation_rate=0.25, message_frequency=0.8, target_demographics_tag="All"),
        ]

        # Initialize households
        # FIXED: Add Household entity and allocate members
        hh_size_mean = int(self.params.get("household_size_mean", 3))
        household_ids: List[List[int]] = []
        remaining = list(range(N))
        self.rng.shuffle(remaining)
        while remaining:
            size = max(1, int(self.rng.gauss(hh_size_mean, 1)))
            pick = min(size, len(remaining))
            members = [remaining.pop() for _ in range(pick)]
            household_ids.append(members)
        self.households = []
        for i, mem in enumerate(household_ids):
            norm_strength = max(0.0, min(1.0, self.rng.gauss(0.5, 0.2)))
            hh = Household(hid=i, members=mem, norm_strength=norm_strength, mask_stock=int(self.rng.random() < 0.3))
            self.households.append(hh)
            for pid in mem:
                self.people[pid].household_id = i

        # Initialize behavioral streaks
        self._streak = [1 if p.mask_adoption_state == "wearing" else 0 for p in self.people]
        self._streak_max = list(self._streak)

        self._initialized = True
        return steps

    def _compute_location_observed_norms(self, wearing_prev: List[float]) -> Dict[str, float]:
        """
        Compute observed norms per location type from previous-day wearing states.
        """
        loc_groups: Dict[str, List[int]] = {}
        for idx, p in enumerate(self.people):
            loc_groups.setdefault(p.location_type, []).append(idx)
        norms: Dict[str, float] = {}
        for lt, idxs in loc_groups.items():
            if not idxs:
                norms[lt] = 0.0
            else:
                norms[lt] = sum(wearing_prev[i] for i in idxs) / max(1, len(idxs))
        for loc in self.locations:
            if loc.type in norms:
                loc.observed_norm = norms[loc.type]
        return norms

    def _choose_visits(self) -> Dict[int, List[int]]:
        """
        Create a mapping from location id to visitor person ids for the day.
        """
        visits: Dict[int, List[int]] = {loc.lid: [] for loc in self.locations}
        mobility_rate = float(self.params.get("daily_mobility_rate", 0.6))
        for p in self.people:
            lid = p.visit_location(self.rng, self.locations, mobility_rate, p.daily_schedule)
            if lid is not None and lid in visits:
                visits[lid].append(p.pid)
        return visits

    def step(self, day: int) -> Dict[str, Any]:
        """
        Execute one simulation step (one day).
        """
        assert self._initialized, "Simulation must be initialized before stepping."
        # Policy activation
        if self.policy is not None:
            mandate_enabled = bool(self.params.get("mandate_enabled", True))
            end_day = int(self.params.get("mandate_end_day", 120))
            active = mandate_enabled and (day >= int(self.params.get("mandate_start_day", 30))) and (day <= end_day)
            self.policy.issue_mandate(day, enabled=active)
            policy_active = self.policy.mandate_status
        else:
            policy_active = False

        # FIXED: Toggle mandates per location type; include home/outdoor as non-enforced
        for loc in self.locations:
            loc.mask_requirement = bool(policy_active and loc.type in {"workplace", "retail", "leisure", "transit"})

        # SIR prevalence (I/N)
        S_count = sum(1 for s in self._inf_status if s == 0)
        I_count = sum(1 for s in self._inf_status if s == 1)
        Npop = len(self._inf_status) or 1
        current_prevalence_true = I_count / max(1, Npop)

        # Blend external perceived cases if provided and publish via PolicyAuthority
        ext_series = self.params.get('perceived_cases_series_resolved')
        w_ext = float(self.params.get('external_cases_weight', 0.5))
        if isinstance(ext_series, list) and 0 <= day < len(ext_series):
            perceived_signal = (1 - w_ext) * current_prevalence_true + w_ext * float(ext_series[day])
        else:
            perceived_signal = current_prevalence_true
        if self.epidemic is not None:
            self.epidemic.prevalence = perceived_signal
        if self.policy is not None:
            self.policy.publish_case_counts(day, ext_series, self.people, fallback_prevalence=perceived_signal)

        # Update person risk perceptions via messaging adjustment
        alpha = float(self.params.get("risk_perception_sensitivity", 0.6))
        for person in self.people:
            person.risk_perception = max(0.0, min(1.0, (1 - alpha) * person.risk_perception + alpha * (person.perceived_local_cases if person.perceived_local_cases is not None else perceived_signal)))
        if self.policy is not None:
            self.policy.adjust_messaging(self.people, intensity=float(self.params.get("messaging_intensity", 0.6)))
        self.epidemic_prevalence_daily.append(perceived_signal)

        # Compute message reach as fraction of distinct persons exposed at least once
        exposed_ids = set()
        for ch in self.information_channels:
            for person in self.people:
                if self.rng.random() < max(0.0, min(1.0, ch.reach)):
                    if person.observe_media(self.rng, ch.bias, ch.misinformation_rate, ch.message_frequency):
                        exposed_ids.add(person.pid)
        self.daily_message_reach.append(len(exposed_ids) / max(1, len(self.people)))

        # Peer averages from yesterday
        wearing_prev = [1.0 if p.mask_adoption_state == "wearing" else 0.0 for p in self.people]
        attitudes_prev = [p.mask_attitude for p in self.people]

        # Update location observed norms from yesterday
        loc_norms = self._compute_location_observed_norms(wearing_prev)

        # Update attitudes based on peers and compute adoption intent
        peer_exposure_acc = 0.0
        for i, p in enumerate(self.people):
            neigh = p.neighbors or []
            if neigh:
                peer_mean_attitude = sum(attitudes_prev[j] for j in neigh) / len(neigh)
                peer_share_prev = sum(wearing_prev[j] for j in neigh) / len(neigh)
            else:
                peer_mean_attitude = sum(attitudes_prev) / max(1, len(attitudes_prev))
                peer_share_prev = sum(wearing_prev) / max(1, len(wearing_prev))
            peer_exposure_acc += peer_share_prev
            p.update_attitude_from_peers(peer_mean_attitude, policy_active, self.params)
            decay = float(self.params.get("social_influence_memory_decay", 0.3))
            intent_signal = (peer_share_prev * p.conformity_sensitivity + p.risk_perception) / 2.0
            p.adoption_intent = max(0.0, min(1.0, (1 - decay) * p.adoption_intent + decay * intent_signal))
        self.daily_peer_exposure_index.append(peer_exposure_acc / max(1, len(self.people)))

        # Household norms and sharing BEFORE decisions
        for hh in self.households:
            hh.discuss_norms(self.people, wearing_prev)
        for hh in self.households:
            hh.share_masks(self.people)

        # Decisions and supply metrics
        penalties_today_value = 0.0
        penalties_today_count = 0
        shortages_today = 0
        denied_access_today = 0
        policy_violations_today = 0
        attempted_purchases_today = 0
        unmet_demand_today = 0
        purchases_today = 0
        vol_wear_expected_with = 0.0
        vol_wear_expected_without = 0.0

        # Decisions (pre-visit)
        for i, p in enumerate(self.people):
            neigh = p.neighbors or []
            if neigh:
                peer_share = sum(wearing_prev[j] for j in neigh) / len(neigh)
            else:
                peer_share = sum(wearing_prev) / max(1, len(wearing_prev))

            # Base perceived cost (excluding penalties)
            p.perceived_mask_cost = 0.5 + 0.5 * (1.0 - p.income / (p.income + 50.0))
            # Map location norms from yesterday
            observed_location_norm = float(loc_norms.get(p.location_type, 0.0))

            # Expected penalty term for not wearing (approx)
            expected_penalty_value = 0.0
            if policy_active and self.policy is not None:
                loc_enf = 0.5
                for loc in self.locations:
                    if loc.type == p.location_type:
                        loc_enf = loc.policy_enforcement_level
                        break
                expected_penalty_value = (
                    (1.0 - float(self.params.get("exemptions_rate", 0.1)))
                    * float(self.policy.enforcement_strength)
                    * float(loc_enf)
                    * float(self.policy.penalty_amount)
                )

            penalty_benefit = 0.0
            if policy_active and self.policy is not None and self.policy.penalty_amount > 0:
                penalty_benefit = expected_penalty_value / float(self.policy.penalty_amount)

            decision_params = dict(self.params)
            decision_params["penalty_benefit"] = penalty_benefit
            decision_params["observed_location_norm"] = observed_location_norm

            # Deterministic expectation tracking for voluntary vs mandate components
            policy_influence_weight = float(decision_params.get("policy_influence_weight", 0.3))
            personal_risk_weight = float(decision_params.get("personal_risk_weight", 0.3))
            peer_influence_weight = float(decision_params.get("peer_influence_weight", 0.4))
            habit_formation_rate = float(decision_params.get("habit_formation_rate", 0.05))
            disutility_cost_scale = float(decision_params.get("disutility_cost_scale", 1.0))
            trust_weight = float(decision_params.get("trust_weight", 0.5))
            mask_effectiveness_weight = float(decision_params.get("mask_effectiveness_weight", 0.4))
            observed_norms_influence_weight = float(decision_params.get("observed_norms_influence_weight", 0.2))
            attitude_weight = float(decision_params.get("attitude_weight", 0.3))
            intent_weight = float(decision_params.get("intent_weight", 0.25))
            comp_weight = float(decision_params.get("compliance_propensity_weight", 0.2))
            health_weight = float(decision_params.get("health_risk_weight", 0.2))
            econ_weight = float(decision_params.get("economic_constraint_weight", 0.2))

            common_terms = (
                (personal_risk_weight * p.risk_perception * mask_effectiveness_weight) +
                (trust_weight * p.trust_in_media * (p.exposure_to_pro_mask_messages - p.exposure_to_misinformation)) +
                (peer_influence_weight * (peer_share - 0.5)) +
                (habit_formation_rate * p.habit_strength) +
                (observed_norms_influence_weight * (observed_location_norm - 0.5)) +
                (attitude_weight * (p.mask_attitude - 0.5)) +
                (intent_weight * (p.adoption_intent - 0.5)) +
                (comp_weight * (p.compliance_propensity - 0.5)) +
                (health_weight * (p.health_risk_level - 0.2)) -
                (disutility_cost_scale * p.perceived_mask_cost + p.fatigue_level + econ_weight * p.economic_constraint)
            )
            latent_with = (policy_influence_weight * (1.0 if policy_active else 0.0) * p.trust_in_authorities) + common_terms + penalty_benefit
            latent_without = common_terms
            p_wear_with = sigmoid(latent_with)
            p_wear_without = sigmoid(latent_without)
            if policy_active:
                vol_wear_expected_with += p_wear_with
                vol_wear_expected_without += p_wear_without

            # Actual stochastic decision
            p.decide_mask_use(self.rng, policy_active, peer_share, decision_params)

            # If decided to wear but no inventory, attempt purchase via assigned retailer
            if p.mask_adoption_state == "wearing" and p.mask_inventory <= 0:
                attempted_purchases_today += 1
                rid = p.retailer_id if 0 <= p.retailer_id < len(self.retailers) else None
                if rid is not None:
                    retailer = self.retailers[rid]
                    price = retailer.price
                    el = float(self.params.get("price_elasticity_of_demand", 0.3))
                    aff = p.income / (p.income + price * 10.0)
                    ref = float(self.params.get("mask_price_surgical", price))
                    price_factor = (ref / max(0.1, price)) ** abs(el)
                    demand_prob = max(0.0, min(1.0, 0.9 * aff * price_factor))
                    if self.rng.random() < demand_prob:
                        sold = retailer.sell_masks(1)
                        if sold > 0:
                            p.mask_inventory += sold
                            purchases_today += sold
                        else:
                            unmet_demand_today += 1
                            shortages_today += 1
                            p.mask_adoption_state = "none"
                            p.mask_type = "none"
                    else:
                        unmet_demand_today += 1
                        p.mask_adoption_state = "none"
                        p.mask_type = "none"
                else:
                    # No retailer assigned; unmet demand
                    unmet_demand_today += 1
                    shortages_today += 1
                    p.mask_adoption_state = "none"
                    p.mask_type = "none"

            # If wearing, consume one mask unit for the day (simplified)
            if p.mask_adoption_state == "wearing" and p.mask_inventory > 0:
                p.mask_inventory = max(0, p.mask_inventory - 1)

        # Visits and location-level enforcement
        visits_map = self._choose_visits()
        comp_by_loc_type: Dict[str, List[float]] = {}
        for loc in self.locations:
            visitors = visits_map.get(loc.lid, [])
            admitted_ids = loc.host_visits(visitors)
            admitted_people = [self.people[i] for i in admitted_ids]
            # Record compliance
            compl = loc.record_compliance(admitted_people)
            comp_by_loc_type.setdefault(loc.type, []).append(compl)
            # Enforce policy
            fines_value, denied = loc.enforce_policy(self.rng, admitted_people)
            penalties_today_value += fines_value
            penalties_today_count += int(round(fines_value / max(1e-9, loc.fine_amount))) if loc.fine_amount > 0 else 0
            denied_access_today += denied
            policy_violations_today += sum(1 for p in admitted_people if loc.mask_requirement and p.mask_adoption_state != "wearing")
            loc.broadcast_norms()

        # Aggregate metrics
        wearing_now = [1.0 if p.mask_adoption_state == "wearing" else 0.0 for p in self.people]
        adoption_rate = sum(wearing_now) / max(1, len(wearing_now))
        self.daily_adoption.append(adoption_rate)
        self.daily_penalties.append(penalties_today_value)
        self.daily_penalties_count.append(penalties_today_count)
        self.policy_active_daily.append(policy_active)
        self.daily_denied_access.append(denied_access_today)
        if policy_active and len(self.people) > 0:
            n = max(1, len(self.people))
            self.daily_voluntary_rate.append(vol_wear_expected_without / n)
            self.daily_mandate_rate.append(max(0.0, (vol_wear_expected_with - vol_wear_expected_without) / n))
        else:
            self.daily_voluntary_rate.append(adoption_rate)
            self.daily_mandate_rate.append(0.0)
        self.daily_policy_violations.append(policy_violations_today)
        self.daily_unmet_demand.append(unmet_demand_today)
        self.daily_attempted_purchases.append(attempted_purchases_today)
        self.daily_masks_purchased.append(purchases_today)
        self.cumulative_masks_purchased += purchases_today

        # Retailer daily stockout share
        if self.retailers:
            out_count = sum(1 for r in self.retailers if r.mask_stock <= 0)
            share = out_count / max(1, len(self.retailers))
            self.daily_stockout_share.append(share)
            self.daily_stockout_flag.append(share > 0.0)
        else:
            retailer_stockout_today = (self.supply.available_mask_stock == 0) if self.supply else False
            self.daily_stockout_flag.append(bool(retailer_stockout_today))
            self.daily_stockout_share.append(1.0 if retailer_stockout_today else 0.0)

        # Per-day adoption by location type (based on assigned primary location_type)
        loc_groups: Dict[str, List[int]] = {}
        for idx, person in enumerate(self.people):
            loc_groups.setdefault(person.location_type, []).append(idx)
        loc_today: Dict[str, float] = {}
        for lt, idxs in loc_groups.items():
            loc_today[lt] = (sum(wearing_now[i] for i in idxs) / len(idxs)) if idxs else 0.0
        self.by_loc_daily_series.append(loc_today)

        # Location compliance averaged by type
        comp_agg: Dict[str, float] = {}
        for lt, vals in comp_by_loc_type.items():
            comp_agg[lt] = sum(vals) / len(vals) if vals else 0.0
        self.location_compliance_daily.append(comp_agg)

        # Retailers restock and adjust price
        for r in self.retailers:
            if r.mask_stock <= 0:
                r.stockout_days += 1
            r.restock()
            r.adjust_price(float(self.params.get("retailer_price_adjust_elasticity", 0.5)))

        # Count mask types
        self.mask_type_counts = {"cloth": 0, "surgical": 0, "N95": 0, "none": 0}
        for p in self.people:
            self.mask_type_counts[p.mask_type] = self.mask_type_counts.get(p.mask_type, 0) + 1

        # Workplace adoption rate
        workplace_idxs = [i for i, person in enumerate(self.people) if person.location_type == "workplace"]
        wr = (sum(wearing_now[i] for i in workplace_idxs) / len(workplace_idxs)) if workplace_idxs else 0.0
        self.daily_workplace_adoption.append(wr)

        # Average mask quality among those wearing
        quality_map = {"none": 0.0, "cloth": 0.3, "surgical": 0.6, "N95": 0.9}
        wearing_quality_vals = [quality_map.get(self.people[i].mask_type, 0.0) for i in range(len(self.people)) if wearing_now[i] > 0.5]
        avg_quality = (sum(wearing_quality_vals) / len(wearing_quality_vals)) if wearing_quality_vals else 0.0
        self.daily_mask_quality.append(avg_quality)

        # Switching rate and streaks
        switches = sum(1 for i in range(len(self.people)) if wearing_now[i] != wearing_prev[i])
        self.daily_switching_rate.append(switches / max(1, len(self.people)))
        for i in range(len(self.people)):
            if wearing_now[i] > 0.5:
                self._streak[i] += 1
                self._streak_max[i] = max(self._streak_max[i], self._streak[i])
            else:
                self._streak[i] = 0

        # Epidemic update (simple SIR-like with mask effects)
        prev_I = I_count
        beta = float(self.params.get("baseline_transmission_rate", 0.08))
        eff_source = (1.0 - float(self.params.get("mask_source_control_efficacy", 0.5)) * adoption_rate)
        inf_period = int(self.params.get("infectious_period_days", 7))
        incident = 0
        expected_no_mask_incident = 0.0
        for i in range(len(self.people)):
            if self._inf_status[i] == 0:  # susceptible
                p_infect_nomask = min(1.0, beta * current_prevalence_true)
                expected_no_mask_incident += p_infect_nomask
                p_infect = min(1.0, beta * current_prevalence_true * eff_source)
                if wearing_now[i] > 0.5:
                    p_infect *= (1.0 - float(self.params.get("mask_wearer_protection_efficacy", 0.3)))
                if self.rng.random() < p_infect:
                    self._inf_status[i] = 1
                    self._days_infected_left[i] = inf_period
                    incident += 1
            elif self._inf_status[i] == 1:
                self._days_infected_left[i] -= 1
                if self._days_infected_left[i] <= 0:
                    self._inf_status[i] = 2
        self.daily_incident_cases.append(incident)
        self.peak_incidence = max(self.peak_incidence, incident)
        self.daily_Rt.append((incident / prev_I) if prev_I > 0 else 0.0)
        averted = max(0.0, expected_no_mask_incident - incident)
        self.cumulative_infections_averted += averted

        # Daily record (for CSV)
        record = {
            "day": day,
            "adoption_rate": adoption_rate,
            "workplace_adoption": wr,
            "prevalence_signal": perceived_signal,
            "message_reach": self.daily_message_reach[-1],
            "penalties_value": penalties_today_value,
            "penalties_count": penalties_today_count,
            "denied_access": denied_access_today,
            "policy_active": int(policy_active),
            "voluntary_rate": self.daily_voluntary_rate[-1],
            "mandate_rate": self.daily_mandate_rate[-1],
            "policy_violations": policy_violations_today,
            "unmet_demand": unmet_demand_today,
            "attempted_purchases": attempted_purchases_today,
            "purchases": purchases_today,
            "stockout_share": self.daily_stockout_share[-1],
            "incident_cases": incident,
            "Rt": self.daily_Rt[-1],
            "avg_mask_quality": avg_quality,
            "switching_rate": self.daily_switching_rate[-1],
            "peer_exposure_index": self.daily_peer_exposure_index[-1],
        }
        self._daily_records.append(record)

        # Return summary
        return record

    def run(self, steps: int) -> Dict[str, Any]:
        """
        Run the simulation for a fixed number of steps.
        """
        for day in range(steps):
            self.step(day)

        result: Dict[str, Any] = {
            "seed": self.seed,
            "parameters": self.params,
            "series": {
                "adoption_rate": self.daily_adoption,
                "workplace_adoption": self.daily_workplace_adoption,
                "message_reach": self.daily_message_reach,
                "penalties_value": self.daily_penalties,
                "penalties_count": self.daily_penalties_count,
                "denied_access": self.daily_denied_access,
                "policy_active": self.policy_active_daily,
                "voluntary_rate": self.daily_voluntary_rate,
                "mandate_rate": self.daily_mandate_rate,
                "policy_violations": self.daily_policy_violations,
                "unmet_demand": self.daily_unmet_demand,
                "attempted_purchases": self.daily_attempted_purchases,
                "purchases": self.daily_masks_purchased,
                "stockout_share": self.daily_stockout_share,
                "prevalence_signal": self.epidemic_prevalence_daily,
                "incident_cases": self.daily_incident_cases,
                "Rt": self.daily_Rt,
                "avg_mask_quality": self.daily_mask_quality,
                "switching_rate": self.daily_switching_rate,
                "peer_exposure_index": self.daily_peer_exposure_index,
            },
            "by_location_adoption": self.by_loc_daily_series,
            "location_compliance": self.location_compliance_daily,
            "final": {
                "final_adoption_rate": self.daily_adoption[-1] if self.daily_adoption else 0.0,
                "avg_adoption_rate": (sum(self.daily_adoption) / len(self.daily_adoption)) if self.daily_adoption else 0.0,
                "mask_type_counts": self.mask_type_counts,
                "cumulative_masks_purchased": self.cumulative_masks_purchased,
                "peak_incidence": self.peak_incidence,
                "cumulative_infections_averted": self.cumulative_infections_averted,
            },
            "daily_records": self._daily_records,
            "hash": stable_hash_dict(self.params),
        }
        return result


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    cfg = load_json(args.input)
    seed = int(args.seed)
    sim = SocialSimulation(cfg, seed=seed)
    steps = sim.initialize(steps_override=args.steps)
    logger.info("Initialized simulation with %d agents for %d days (seed=%d)", sim.params.get("population_size"), steps, seed)

    result = sim.run(steps)

    # Optional CSV output
    if args.csv_out:
        fieldnames = [
            "day",
            "adoption_rate",
            "workplace_adoption",
            "prevalence_signal",
            "message_reach",
            "penalties_value",
            "penalties_count",
            "denied_access",
            "policy_active",
            "voluntary_rate",
            "mandate_rate",
            "policy_violations",
            "unmet_demand",
            "attempted_purchases",
            "purchases",
            "stockout_share",
            "incident_cases",
            "Rt",
            "avg_mask_quality",
            "switching_rate",
            "peer_exposure_index",
        ]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in result.get("daily_records", []):
                writer.writerow({k: row.get(k) for k in fieldnames})

    save_json(result, args.output)


# Execute main for both direct execution and sandbox wrapper invocation

# Execute main for both direct execution and sandbox wrapper invocation
main()