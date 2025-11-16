import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project data directory setup (path handling instructions)
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
    pass
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
    pass
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

    Args:
        path (Optional[str]): Path to the JSON file or '-' for stdin.

    Returns:
        Dict[str, Any]: Parsed JSON object as a dictionary.
    """
    # FIXED: Implemented robust JSON parsing with detailed diagnostics and specific exit code (3).
    pass
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

    Args:
        obj (Dict[str, Any]): Object to serialize.
        path (Optional[str]): Output path or None for stdout.

    Returns:
        None
    """
    pass
    out = json.dumps(obj, ensure_ascii=False, indent=2)
    if path:
        Path(path).write_text(out + "\n", encoding="utf-8")
    else:
        sys.stdout.write(out + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the simulation CLI.

    Args:
        argv (Optional[List[str]]): Argument vector or None to use sys.argv.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # FIXED: Implemented functional CLI as requested.
    pass
    p = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation (deterministic minimal core)")
    p.add_argument("--input", "-i", default=None, help="Path to input JSON (use '-' for stdin)")
    p.add_argument("--output", "-o", default=None, help="Path to write output JSON (default: stdout)")
    p.add_argument("--steps", type=int, default=None, help="Number of simulation steps (days)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    p.add_argument("--use-docker", action="store_true", help="Attempt to use Docker (guarded; defaults to off)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return p.parse_args(argv)


def stable_hash_dict(d: Dict[str, Any]) -> str:
    """
    Compute a stable SHA-256 hash of a dictionary by canonical JSON serialization.

    Args:
        d (Dict[str, Any]): Input dictionary.

    Returns:
        str: Hex digest of SHA-256.
    """
    pass
    try:
        b = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        b = repr(d).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.

    Args:
        x (float): Input value.

    Returns:
        float: Sigmoid of x.
    """
    pass
    if x >= 0:
        z = pow(2.718281828459045, -x)
        return 1 / (1 + z)
    z = pow(2.718281828459045, x)
    return z / (1 + z)


def gini(values: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Compute Gini coefficient for a list of non-negative values.

    Args:
        values (List[float]): Values list.
        weights (Optional[List[float]]): Optional weights list.

    Returns:
        float: Gini coefficient in [0,1].
    """
    pass
    if not values:
        return 0.0
    if weights is None:
        weights = [1.0] * len(values)
    paired = sorted(zip(values, weights), key=lambda vw: vw[0])
    cumw = 0.0
    cumval = 0.0
    total_w = sum(weights)
    total_v = sum(v * w for v, w in paired)
    if total_v == 0 or total_w == 0:
        return 0.0
    g = 0.0
    for v, w in paired:
        prev_cumw = cumw
        cumw += w
        g += v * w * (cumw + prev_cumw)
        cumval += v * w
    g = 1.0 - g / (total_w * total_v)
    return max(0.0, min(1.0, g))


@dataclass
class Person:
    """
    Represents an individual in the simulation with attributes related to mask adoption.

    Attributes:
        pid (int): Person ID.
        income (float): Daily income proxy for affordability.
        risk_perception (float): Perceived personal risk [0,1].
        trust_in_authorities (float): Trust in authorities [0,1].
        trust_in_media (float): Trust in media [0,1].
        conformity_sensitivity (float): Sensitivity to norms [0,1].
        reactance_tendency (float): Reactance tendency [0,1].
        mask_attitude (float): Attitude toward masks [0,1].
        habit_strength (float): Habit strength [0,1].
        fatigue_level (float): Fatigue level [0,1].
        perceived_mask_cost (float): Perceived cost including penalties.
        mask_adoption_state (str): 'wearing' or 'none'.
        mask_type (str): 'cloth', 'surgical', 'N95', or 'none'.
        neighbors (List[int]): Neighbor indices in the social network.
        location_type (str): Assigned primary location type for grouping metrics.
        exposure_to_misinformation (float): Anti-mask exposure proxy.
        exposure_to_pro_mask_messages (float): Pro-mask exposure proxy.
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

    def decide_mask_use(self, rng: random.Random, policy_active: bool, peer_share: float, params: Dict[str, float]) -> None:
        """
        Decide whether to wear a mask today based on a simplified utility model.

        Args:
            rng (random.Random): Random number generator.
            policy_active (bool): Whether a mandate is active.
            peer_share (float): Share of neighbors wearing (previous day).
            params (Dict[str, float]): Model parameters.

        Returns:
            None
        """
        pass
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

        # Utility benefits and costs (simplified deterministic form with seeded noise)
        policy_term = policy_influence_weight * (1.0 if policy_active else 0.0) * self.trust_in_authorities
        risk_term = personal_risk_weight * self.risk_perception * mask_effectiveness_weight
        media_term = trust_weight * self.trust_in_media * (self.exposure_to_pro_mask_messages - self.exposure_to_misinformation)
        peer_term = peer_influence_weight * (peer_share - 0.5)
        habit_term = habit_formation_rate * self.habit_strength
        cost_term = disutility_cost_scale * self.perceived_mask_cost + self.fatigue_level

        latent = policy_term + risk_term + media_term + peer_term + habit_term - cost_term
        noise = rng.uniform(-baseline_noise, baseline_noise)
        p_wear = sigmoid(latent + noise)
        wearing = rng.random() < p_wear

        if wearing:
            self.mask_adoption_state = "wearing"
            self.habit_strength = min(1.0, self.habit_strength + habit_formation_rate)
            self.fatigue_level = max(0.0, self.fatigue_level - fatigue_recovery_rate)
            # Simple type choice: higher risk prefers N95
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

        Args:
            peer_mean_attitude (float): Average neighbor attitude.
            policy_active (bool): Whether mandate is active.
            params (Dict[str, float]): Model parameters.

        Returns:
            None
        """
        pass
        peer_influence_weight = float(params.get("peer_influence_weight", 0.4))
        reactance_weight = float(params.get("reactance_weight", 0.2))
        delta = peer_influence_weight * (peer_mean_attitude - self.mask_attitude)
        reactance = reactance_weight * self.reactance_tendency * (-1.0 if delta > 0 else 1.0) if policy_active else 0.0
        self.mask_attitude = max(0.0, min(1.0, self.mask_attitude + delta + reactance))

    def observe_media(self, rng: random.Random, media_bias_mean: float, misinformation_rate: float, media_message_frequency: float) -> None:
        """
        Update exposures to media messages.

        Args:
            rng (random.Random): Random source.
            media_bias_mean (float): Mean bias of the media tone.
            misinformation_rate (float): Probability of misinformation.
            media_message_frequency (float): Frequency multiplier.

        Returns:
            None
        """
        pass
        # Simple deterministic exposure increment
        if rng.random() < media_message_frequency:
            tone = media_bias_mean + rng.uniform(-0.05, 0.05)
            if rng.random() < misinformation_rate:
                # Anti-mask message
                self.exposure_to_misinformation = 0.7 * self.exposure_to_misinformation + max(0.0, -tone)
            else:
                # Pro-mask message
                self.exposure_to_pro_mask_messages = 0.7 * self.exposure_to_pro_mask_messages + max(0.0, tone)


@dataclass
class Location:
    """
    Represents a location with basic policy enforcement and crowding parameters.

    Attributes:
        lid (int): Location ID.
        type (str): Type of location (home, workplace, retail, etc.).
        capacity (int): Capacity of the location.
        mask_requirement (bool): Whether masks are required at this location.
        policy_enforcement_level (float): Enforcement strictness [0,1].
        signage_visible (float): Signage visibility [0,1].
        crowding_level (float): Crowding level [0,1].
    """
    lid: int
    type: str
    capacity: int
    mask_requirement: bool
    policy_enforcement_level: float
    signage_visible: float
    crowding_level: float

    def host_visits(self) -> None:
        """
        Placeholder for visit handling logic.
        """
        pass

    def enforce_policy(self) -> None:
        """
        Placeholder for enforcement logic.
        """
        pass

    def broadcast_norms(self) -> None:
        """
        Placeholder for norms broadcasting logic.
        """
        pass


@dataclass
class PolicyAuthority:
    """
    Represents a policy authority that can activate mandates.

    Attributes:
        jurisdiction (str): Jurisdiction identifier.
        mandate_status (bool): Whether mandate is currently active.
        mandate_start_day (int): Day when mandate activates.
        enforcement_strength (float): Enforcement probability [0,1].
        penalty_amount (float): Penalty amount currency units.
        exemptions_rate (float): Fraction exempt.
        communication_strategy_intensity (float): Intensity [0,2].
        communication_frequency (float): Frequency [0,2].
        message_tone (str): Tone ('prosocial', 'authoritative', 'neutral').
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

        Args:
            day (int): Current day (0-indexed).
            enabled (bool): Whether mandates are allowed.

        Returns:
            None
        """
        pass
        self.mandate_status = bool(enabled and (day >= self.mandate_start_day))

    def lift_mandate(self) -> None:
        """
        Lift any active mandate.
        """
        pass
        self.mandate_status = False

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
class MediaOutlet:
    """
    Represents a media outlet broadcasting messages.

    Attributes:
        mid (int): Media outlet ID.
        reach (float): Fraction of population it reaches [0,1].
        bias (float): Bias of messages [-1,1].
        misinformation_rate (float): Share of misinformation messages [0,1].
        message_frequency (float): Daily frequency multiplier [0,3].
        target_demographics_tag (str): Target tag (e.g., 'Left', 'Right', 'All').
    """
    mid: int
    reach: float
    bias: float
    misinformation_rate: float
    message_frequency: float
    target_demographics_tag: str = "All"

    def broadcast_message(self) -> None:
        """
        Placeholder for message generation. Effects applied at person-level.
        """
        pass

    def target_audience(self) -> None:
        """
        Placeholder for targeting logic.
        """
        pass


@dataclass
class SupplyChain:
    """
    Represents a simple supply chain for mask availability and pricing.

    Attributes:
        available_mask_stock (int): Overall stock quantity.
        price_cloth (float): Cloth mask price.
        price_surgical (float): Surgical mask price.
        price_N95 (float): N95 mask price.
        restock_rate (int): Daily restock amount.
        shortage_probability (float): Probability of a shortage shock.
    """
    available_mask_stock: int
    price_cloth: float
    price_surgical: float
    price_N95: float
    restock_rate: int
    shortage_probability: float

    def update_prices(self) -> None:
        """
        Placeholder for price update logic.
        """
        pass

    def release_stock(self) -> None:
        """
        Placeholder for stock release logic.
        """
        pass

    def apply_shortage_shock(self) -> None:
        """
        Placeholder for shortage shock logic.
        """
        pass


@dataclass
class EpidemicEnvironment:
    """
    Represents the epidemic environment providing prevalence/risk signals.

    Attributes:
        region_id (str): Region identifier.
        prevalence (float): Current prevalence [0,1].
        trend (str): Trend descriptor ('increasing','stable','decreasing').
    """
    region_id: str
    prevalence: float
    trend: str

    def update_prevalence(self) -> None:
        """
        Placeholder for prevalence update logic.
        """
        pass


class SimulationEnvironment:
    """
    Coordinates entities and provides global context for the simulation.

    This class is a lightweight container and placeholder for future extensions.
    """
    def __init__(self) -> None:
        """
        Initialize the simulation environment container.
        """
        pass
        self.entities: Dict[str, Any] = {}

    def register(self, name: str, entity: Any) -> None:
        """
        Register an entity with a name.

        Args:
            name (str): Name key.
            entity (Any): Entity instance.

        Returns:
            None
        """
        pass
        self.entities[name] = entity

    def get(self, name: str) -> Any:
        """
        Retrieve a registered entity by name.

        Args:
            name (str): Name key.

        Returns:
            Any: The entity or None.
        """
        pass
        return self.entities.get(name)


class SocialSimulation:
    """
    Main simulation class for mask adoption dynamics.

    Implements a minimal deterministic core consistent with feedback:
    - Deterministic RNG via seed
    - Robust to partial/missing input
    - Produces structured results
    """
    def __init__(self, config: Dict[str, Any], seed: int) -> None:
        """
        Initialize the simulation with configuration and RNG seed.

        Args:
            config (Dict[str, Any]): Configuration/model plan dictionary.
            seed (int): Random seed.

        Returns:
            None
        """
        pass
        self.config = config or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = SimulationEnvironment()
        self.params = self._extract_parameters(self.config)
        self.people: List[Person] = []
        self.locations: List[Location] = []
        self.media_outlets: List[MediaOutlet] = []
        self.policy: Optional[PolicyAuthority] = None
        self.supply: Optional[SupplyChain] = None
        self.epidemic: Optional[EpidemicEnvironment] = None
        self.daily_adoption: List[float] = []
        self.daily_penalties: List[float] = []
        self.daily_shortages: List[int] = []
        self.mask_type_counts: Dict[str, int] = {"cloth": 0, "surgical": 0, "N95": 0, "none": 0}
        self._initialized = False

    def _extract_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parameters and fallback defaults from the model plan.

        Args:
            config (Dict[str, Any]): Model plan or input configuration.

        Returns:
            Dict[str, Any]: Parameters dictionary.
        """
        pass
        params = {}
        # Dynamic configuration extraction
        for k, v in config.items():
            if k == "parameters" and isinstance(v, dict):
                params.update(v)
            elif k == "parameters" and isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "key" in item and "default" in item:
                        params[item["key"]] = item["default"]
            elif k == "initialization" and isinstance(v, dict):
                # Include initialization block for reference
                params.setdefault("initialization", v)
            elif k == "prediction_period" and isinstance(v, dict):
                params.setdefault("prediction_period", v)
            # Include any additional fields for reference
        # Set defaults if missing
        params.setdefault("population_size", 1000)
        params.setdefault("simulation_days", 30)
        params.setdefault("time_step_days", 1)
        params.setdefault("initial_adoption_rate", 0.2)
        params.setdefault("average_degree", 8)
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
        params.setdefault("mandate_start_day", 30)
        params.setdefault("enforcement_strength", 0.5)
        params.setdefault("penalty_amount", 50.0)
        params.setdefault("media_message_frequency", 1.0)
        params.setdefault("media_bias_mean", 0.0)
        params.setdefault("misinformation_rate", 0.15)
        params.setdefault("target_adoption_rate", 0.8)
        params.setdefault("did_pre_window_days", 14)
        params.setdefault("did_post_window_days", 14)
        return params

    def initialize(self, steps_override: Optional[int] = None) -> int:
        """
        Initialize entities and network based on parameters and input data.

        Args:
            steps_override (Optional[int]): Optional override for number of steps.

        Returns:
            int: Number of steps to simulate.
        """
        pass
        N = int(self.params.get("population_size", 1000))
        avg_deg = int(self.params.get("average_degree", 8))
        initial_rate = float(self.params.get("initial_adoption_rate", 0.2))
        # Steps selection with prediction_period override
        steps = steps_override if steps_override is not None else int(self.params.get("simulation_days", 30))
        pred = self.params.get("prediction_period") or {}
        if isinstance(pred, dict):
            s = pred.get("start_day")
            e = pred.get("end_day")
            if isinstance(s, int) and isinstance(e, int) and e > s >= 0:
                steps = max(steps, e + 1)

        # Create people
        loc_types = ["home", "workplace", "retail", "leisure", "transit", "outdoor"]
        for i in range(N):
            income = self.rng.lognormvariate(3.0, 0.4)  # daily income ~ lognormal
            p = Person(
                pid=i,
                income=income,
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
            )
            if p.mask_adoption_state == "wearing":
                # Initialize mask type distribution
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

        # Ring lattice neighbors (small-world-like with no rewiring for minimalism)
        k = max(2, avg_deg)
        for i in range(N):
            neigh = []
            for d in range(1, k // 2 + 1):
                neigh.append((i - d) % N)
                neigh.append((i + d) % N)
            self.people[i].neighbors = neigh

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

        # Initialize supply
        self.supply = SupplyChain(
            available_mask_stock=int(self.params.get("initial_mask_supply", 100000)),
            price_cloth=float(self.params.get("mask_cost_cloth", 2.0)),
            price_surgical=float(self.params.get("mask_cost_surgical", 1.0)),
            price_N95=float(self.params.get("mask_cost_N95", 3.0)),
            restock_rate=int(self.params.get("restock_rate_daily", 10000)),
            shortage_probability=float(self.params.get("shortage_shock_probability", 0.01)),
        )

        # Initialize epidemic
        self.epidemic = EpidemicEnvironment(region_id="region_1", prevalence=0.01, trend="stable")

        self._initialized = True
        return steps

    def step(self, day: int) -> Dict[str, Any]:
        """
        Execute one simulation step (one day).

        Args:
            day (int): Current day index.

        Returns:
            Dict[str, Any]: Dictionary of step metrics.
        """
        pass
        assert self._initialized, "Simulation must be initialized before stepping."
        # Policy activation
        if self.policy is not None:
            self.policy.issue_mandate(day, enabled=bool(self.params.get("mandate_enabled", True)))
            policy_active = self.policy.mandate_status
        else:
            policy_active = False

        # Media exposures
        media_bias_mean = float(self.params.get("media_bias_mean", 0.0))
        misinformation_rate = float(self.params.get("misinformation_rate", 0.15))
        media_message_frequency = float(self.params.get("media_message_frequency", 1.0))
        for person in self.people:
            person.observe_media(self.rng, media_bias_mean, misinformation_rate, media_message_frequency)

        # Peer averages from yesterday
        wearing_prev = [1.0 if p.mask_adoption_state == "wearing" else 0.0 for p in self.people]
        attitudes_prev = [p.mask_attitude for p in self.people]

        # Update attitudes based on peers
        for i, p in enumerate(self.people):
            neigh = p.neighbors or []
            if neigh:
                peer_mean_attitude = sum(attitudes_prev[j] for j in neigh) / len(neigh)
            else:
                peer_mean_attitude = sum(attitudes_prev) / max(1, len(attitudes_prev))
            p.update_attitude_from_peers(peer_mean_attitude, policy_active, self.params)

        # Decide wearing today
        penalties_today = 0.0
        for i, p in enumerate(self.people):
            neigh = p.neighbors or []
            if neigh:
                peer_share = sum(wearing_prev[j] for j in neigh) / len(neigh)
            else:
                peer_share = sum(wearing_prev) / max(1, len(wearing_prev))

            # Adjust perceived cost by expected penalty if policy active
            expected_penalty = 0.0
            if policy_active and self.policy is not None:
                expected_penalty = (1.0 - self.params.get("exemptions_rate", 0.1)) * self.policy.enforcement_strength * self.policy.penalty_amount
            p.perceived_mask_cost = 0.5 + 0.5 * (1.0 - p.income / (p.income + 50.0)) + (expected_penalty / 100.0 if policy_active else 0.0)
            p.decide_mask_use(self.rng, policy_active, peer_share, self.params)
            # Penalty realization for non-wearers under mandate
            if policy_active and p.mask_adoption_state != "wearing" and self.policy is not None:
                if self.rng.random() < self.policy.enforcement_strength:
                    penalties_today += self.policy.penalty_amount

        # Aggregate metrics
        wearing_now = [1.0 if p.mask_adoption_state == "wearing" else 0.0 for p in self.people]
        adoption_rate = sum(wearing_now) / max(1, len(wearing_now))
        self.daily_adoption.append(adoption_rate)
        self.daily_penalties.append(penalties_today)

        # Shortages (simplified: no dynamics; could be based on stock)
        self.daily_shortages.append(0)

        # Count mask types
        self.mask_type_counts = {"cloth": 0, "surgical": 0, "N95": 0, "none": 0}
        for p in self.people:
            self.mask_type_counts[p.mask_type] = self.mask_type_counts.get(p.mask_type, 0) + 1

        return {
            "day": day,
            "adoption_rate": adoption_rate,
            "penalties_today": penalties_today,
            "policy_active": policy_active,
        }

    def run(self, steps: int) -> Dict[str, Any]:
        """
        Run the simulation for the specified number of steps.

        Args:
            steps (int): Number of days to simulate.

        Returns:
            Dict[str, Any]: Summary results.
        """
        pass
        if not self._initialized:
            steps = self.initialize(steps_override=steps)

        for day in range(steps):
            self.step(day)

        # Compute observables
        overall = self.daily_adoption
        target = float(self.params.get("target_adoption_rate", 0.8))
        time_to_target = None
        for i, v in enumerate(overall):
            if v >= target:
                time_to_target = i
                break

        # By location type adoption (approximate: mean among agents assigned to that type)
        by_loc: Dict[str, float] = {}
        loc_groups: Dict[str, List[int]] = {}
        for idx, p in enumerate(self.people):
            loc_groups.setdefault(p.location_type, []).append(idx)
        wearing_last = [1.0 if p.mask_adoption_state == "wearing" else 0.0 for p in self.people]
        for lt, idxs in loc_groups.items():
            if idxs:
                by_loc[lt] = sum(wearing_last[i] for i in idxs) / len(idxs)
            else:
                by_loc[lt] = 0.0

        # Compliance rate under mandate (average wearing when policy active)
        comp_under_mandate = 0.0
        if any(True for d in range(len(self.daily_adoption)) if d >= int(self.params.get("mandate_start_day", 30))):
            post = self.daily_adoption[int(self.params.get("mandate_start_day", 30)):]
            if post:
                comp_under_mandate = sum(post) / len(post)

        # Simple DiD (difference between post and pre averages)
        pre_w = int(self.params.get("did_pre_window_days", 14))
        post_w = int(self.params.get("did_post_window_days", 14))
        mstart = int(self.params.get("mandate_start_day", 30))
        pre_series = self.daily_adoption[max(0, mstart - pre_w):mstart]
        post_series = self.daily_adoption[mstart:mstart + post_w]
        if pre_series and post_series:
            policy_effect_did = (sum(post_series) / len(post_series)) - (sum(pre_series) / len(pre_series))
        else:
            policy_effect_did = 0.0

        # Mask type distribution
        total_people = max(1, len(self.people))
        mask_type_distribution = {k: v / total_people for k, v in self.mask_type_counts.items()}

        # Adoption inequality by income (group by quintile using final wearing state as proxy)
        incomes = sorted([p.income for p in self.people])
        quintiles = [incomes[int(len(incomes) * q / 5)] for q in range(1, 5)] if incomes else []
        group_sums = [0.0] * 5
        group_counts = [0] * 5
        for p in self.people:
            g = 0
            while g < 4 and p.income > quintiles[g]:
                g += 1
            group_sums[g] += 1.0 if p.mask_adoption_state == "wearing" else 0.0
            group_counts[g] += 1
        group_means = [(group_sums[i] / group_counts[i]) if group_counts[i] else 0.0 for i in range(5)]
        adoption_gini_income = gini(group_means, [c if c > 0 else 1.0 for c in group_counts])

        # Media exposure distribution (simple statistics)
        pro_exp = [p.exposure_to_pro_mask_messages for p in self.people]
        anti_exp = [p.exposure_to_misinformation for p in self.people]
        media_exposure_distribution = {
            "pro_mean": sum(pro_exp) / total_people,
            "anti_mean": sum(anti_exp) / total_people,
        }

        # Cumulative shortages and penalties
        cum_shortages = sum(self.daily_shortages)
        cum_penalties = sum(self.daily_penalties)

        results = {
            "observable": {
                "adoption_rate_daily": overall,
                "adoption_rate_by_location_type": by_loc,
                "time_to_target_adoption": time_to_target,
                "mask_type_distribution": mask_type_distribution,
                "compliance_rate_under_mandate": comp_under_mandate,
                "policy_effect_size_DiD": policy_effect_did,
                "adoption_inequality_by_income": adoption_gini_income,
                "media_exposure_distribution": media_exposure_distribution,
                "supply_shortage_incidents": cum_shortages,
                "penalties_assessed": {"count": len([p for p in self.daily_penalties if p > 0]), "value": cum_penalties},
            }
        }
        return results

    def evaluate(self, metric_names: Optional[List[str]]) -> Dict[str, Any]:
        """
        Evaluate the simulation according to specified metric names.
        Provides generic placeholders for minimal testing.

        Args:
            metric_names (Optional[List[str]]): List of metric names.

        Returns:
            Dict[str, Any]: Metric results dictionary.
        """
        pass
        if not metric_names:
            return {}
        # Placeholder evaluation: report basic stats for requested metrics
        obs = {
            "overall_mean_adoption": sum(self.daily_adoption) / max(1, len(self.daily_adoption)),
            "final_adoption": self.daily_adoption[-1] if self.daily_adoption else 0.0,
        }
        res: Dict[str, Any] = {}
        for name in metric_names:
            res[name] = obs  # Same placeholder for all metrics
        return res

    def visualize(self) -> None:
        """
        Produce a simple textual visualization of results.
        """
        pass
        if not self.daily_adoption:
            print("No results to visualize.")
            return
        print("Adoption rate over time (first 10 days):", [round(v, 3) for v in self.daily_adoption[:10]])
        print("... last 5 days:", [round(v, 3) for v in self.daily_adoption[-5:]])
        target = float(self.params.get("target_adoption_rate", 0.8))
        ttt = next((i for i, v in enumerate(self.daily_adoption) if v >= target), None)
        print(f"Time to target adoption ({target}):", ttt)

    def save_results(self, filename: str) -> None:
        """
        Save daily adoption results to a CSV file.

        Args:
            filename (str): Output CSV filename.

        Returns:
            None
        """
        pass
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["day", "adoption_rate", "penalties_today"])
                for i, (adopt, pen) in enumerate(zip(self.daily_adoption, self.daily_penalties)):
                    writer.writerow([i, adopt, pen])
            logger.info("Saved results to %s", filename)
        except Exception as e:
            logger.error("Failed to save results to %s: %s", filename, e)


def run_simulation(data: Dict[str, Any], steps: int, seed: int) -> Dict[str, Any]:
    """
    Run the deterministic minimal social simulation and produce a structured JSON result.

    Args:
        data (Dict[str, Any]): Input data/model plan.
        steps (int): Number of steps to simulate.
        seed (int): RNG seed.

    Returns:
        Dict[str, Any]: Result object with metadata and summary.
    """
    # FIXED: Implemented deterministic minimal simulation core with seed and summary outputs.
    pass
    sim = SocialSimulation(config=data, seed=seed)
    steps_to_run = steps
    try:
        steps_to_run = sim.initialize(steps_override=steps)
    except Exception as exc:
        logger.error("Initialization failed: %s", exc)
        raise
    results = sim.run(steps_to_run)
    # Compose output
    input_hash = stable_hash_dict(data) if data else None
    output = {
        "ok": True,
        "seed": seed,
        "timestamp": int(time.time()),
        "input_hash": input_hash,
        "summary": {
            "steps": steps_to_run,
            "population": len(sim.people),
            "final_adoption": sim.daily_adoption[-1] if sim.daily_adoption else 0.0,
            "cumulative_penalties": sum(sim.daily_penalties),
            "cumulative_shortages": sum(sim.daily_shortages),
        },
        "results": results,
    }
    return output


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entrypoint for running the simulation.

    Args:
        argv (Optional[List[str]]): Command-line arguments.

    Returns:
        int: Exit status code (0 success, 2 invalid usage, 3 JSON parse error, 1 other errors).
    """
    # FIXED: Restored functional main with CLI parsing, logging, and exit code discipline.
    pass
    args: Optional[argparse.Namespace] = None
    try:
        args = parse_args(argv)
    except SystemExit as e:
        # Argparse uses exit code 2 for invalid usage
        return int(getattr(e, "code", 2) or 2)

    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO), format="%(levelname)s: %(message)s")
    logger.info("Starting simulation")

    # Docker guards and env override
    use_docker_requested = bool(args.use_docker)
    skip_docker_env = os.getenv("SIM_SKIP_DOCKER") == "1"
    docker_available = has_docker()
    use_docker = use_docker_requested and not skip_docker_env and docker_available
    # FIXED: Added Docker guards and clear fallback messages.
    if use_docker_requested and not docker_available:
        logger.warning("Docker not found; falling back to native execution.")
    if skip_docker_env:
        logger.info("SIM_SKIP_DOCKER=1 set; using native execution path.")
    if use_docker:
        logger.info("Docker execution requested and available; native path used for this minimal implementation.")

    # Load input JSON
    try:
        data = load_json(args.input)
    except SystemExit as e:
        # load_json already emitted error and used exit code 3
        return int(getattr(e, "code", 3) or 3)
    except Exception as e:
        logger.error("Unexpected error while loading input: %s", e)
        return 1

    # Steps from CLI or data parameters
    steps = args.steps
    if steps is None:
        # Attempt to read from data config
        steps = int(
            (data.get("parameters", {}) or {}).get("simulation_days", 30) if isinstance(data.get("parameters"), dict) else
            (next((p["default"] for p in data.get("parameters", []) if isinstance(p, dict) and p.get("key") == "simulation_days"), 30)
             if isinstance(data.get("parameters"), list) else 30)
        )
    seed = int(args.seed)

    # Run simulation
    try:
        output = run_simulation(data, steps=steps, seed=seed)
    except Exception as exc:
        logger.error("Simulation failed: %s", exc)
        return 1

    # Save JSON output
    try:
        save_json(output, args.output)
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        return 1

    # Demonstrate visualization and saving CSV results
    try:
        # Recreate simulation object to access results for saving and visualization
        sim_demo = SocialSimulation(config=data, seed=seed)
        sim_demo.initialize(steps_override=steps)
        sim_demo.run(steps)
        sim_demo.visualize()
        # FIXED: Demonstrate saving results to CSV as requested.
        sim_demo.save_results("results.csv")
    except Exception as exc:
        logger.warning("Post-processing (visualize/save CSV) failed: %s", exc)

    logger.info("Simulation completed successfully")
    return 0


# Execute main for both direct execution and sandbox wrapper invocation
# FIXED: Direct call to main with deterministic exit handling.
ret_code = main()
try:
    code = int(ret_code) if ret_code is not None else 0
except Exception:
    code = 1
sys.exit(code)

# Execute main for both direct execution and sandbox wrapper invocation
main()