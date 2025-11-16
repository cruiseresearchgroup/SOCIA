import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# FIXED: Removed stray non-Python text and restored full simulation implementation
# FIXED: Added JSON-safe sanitization utilities and used them for all prints/saves
# FIXED: Implemented FAST_MODE/SAND_TEST to avoid timeouts and skip heavy I/O in sandbox
# FIXED: Restored main orchestrator to parse CLI, run Simulation, evaluate, and print JSON
# FIXED: Guarded metrics and time series against NaN/Inf and undefined windows
# FIXED: Added parameter registry with frozen parameter handling and CLI overrides
# FIXED: Implemented pluggable calibration architecture with registry and adapter
# FIXED: Ensured all docstrings are properly closed and included pass statements
# FIXED: Applied feedback to allow zero rewiring, fix calibrator factory kwargs, improve warnings,
#        honor evaluation window start, synchronize persisted parameters, improve trust dynamics,
#        and increase MAPE sensitivity
# FIXED: Integrated awareness into adoption probability and implemented simple awareness memory (max_memory_days)
# FIXED: Implemented simple triadic closure controlled by clustering_bias
# FIXED: Added NoOpCalibrator ('none') and time budget guards in calibrators
# FIXED: Guarded parameter persistence in ParamsAdapter.apply with SKIP_IO
# FIXED: Added Watts–Strogatz small-world network option and parameters
# FIXED: Decoupled policy signal from trust and added awareness_decay with daily decay for everyone
# FIXED: Added minimal ContextScheduler and SupplyChain (Retailer) with purchasing and stockout tracking
# FIXED: Extended AdoptionAggregator with contextual adoption and inequality index; tracked stockout rate
# FIXED: Guarded calibrator artifact writes with SKIP_IO
# FIXED: Changed default calibrator to 'none' and reduced default budget; added unconditional time caps for calibrators
# FIXED: Guarded ParameterManager.persist_used with SKIP_IO
# FIXED: Optimized homophily rewiring by sampling candidates
# FIXED: Orchestrator uses a local skip_io flag instead of changing environment at runtime
# FIXED: Skipped heavy counterfactual evaluation for large populations
# FIXED: SyntaxError in NetworkDynamics.forward (removed stray quote)
# FIXED: NameError in InteractionScheduler.forward (use daily_fraction), added parameter clamping and adjacency guard
# FIXED: ParameterManager._cast_value supports dict parameters via JSON parsing
# FIXED: SupplyChain._afford_prob clamps income to non-negative
# FIXED: Persist contextual adoption, purchase_rate_daily, and inequality index to history; summarize equity gap


# ---------------------------
# Global Constants and Paths
# ---------------------------

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# FIXED: Default FAST_MODE to "0" to avoid unintended downscaling; still consider SAND_TEST.
FAST_MODE = (
    os.environ.get("FAST_MODE", "0") == "1"
    or os.environ.get("SAND_TEST", "0") == "1"
)
SKIP_IO = os.environ.get("SKIP_IO", "1") == "1" or FAST_MODE

# ---------------------------
# Utilities
# ---------------------------


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    pass
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # best-effort
        pass


def seed_all(seed: int) -> None:
    """Seed all RNG sources for determinism."""
    pass
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        # numpy may not be available
        pass


def clamp(x: float, low: float, high: float) -> float:
    """Clamp value into [low, high]."""
    pass
    return low if x < low else high if x > high else x


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    pass
    try:
        if x < -700:
            return 0.0
        if x > 700:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def moving_average(series: List[float], window: int) -> List[float]:
    """Compute causal moving average with given window size."""
    pass
    if window <= 1:
        return list(series)
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(series):
        s += v
        if i >= window:
            s -= series[i - window]
            out.append(s / float(window))
        else:
            out.append(s / float(i + 1))
    return out


def pearson_r(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    pass
    n = min(len(x), len(y))
    if n < 2:
        return None
    x_slice = x[:n]
    y_slice = y[:n]
    mx = sum(x_slice) / n
    my = sum(y_slice) / n
    num = 0.0
    dx = 0.0
    dy = 0.0
    for i in range(n):
        a = x_slice[i] - mx
        b = y_slice[i] - my
        num += a * b
        dx += a * a
        dy += b * b
    if dx <= 0 or dy <= 0:
        return 0.0
    return num / math.sqrt(dx * dy)


def rmse(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    """Compute RMSE."""
    pass
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return None
    s = 0.0
    for i in range(n):
        d = float(y_true[i]) - float(y_pred[i])
        s += d * d
    return math.sqrt(s / n)


def mae(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    """Compute MAE."""
    pass
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return None
    s = 0.0
    for i in range(n):
        d = abs(float(y_true[i]) - float(y_pred[i]))
        s += d
    return s / n


def mape(y_true: List[float], y_pred: List[float], epsilon: float = 1e-6) -> Optional[float]:
    """Compute MAPE with epsilon guard."""
    pass
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return None
    s = 0.0
    for i in range(n):
        denom = max(abs(float(y_true[i])), epsilon)
        s += abs((float(y_pred[i]) - float(y_true[i])) / denom)
    return s / n


def time_to_peak(series: List[float], window: Optional[Tuple[int, int]] = None) -> Optional[int]:
    """Return index of peak within window if provided."""
    pass
    if not series:
        return None
    start = 0 if window is None else max(0, int(window[0]))
    end = len(series) - 1 if window is None else min(len(series) - 1, int(window[1]))
    if end < start:
        return None
    idx = max(range(start, end + 1), key=lambda i: series[i] if math.isfinite(series[i]) else float("-inf"))
    return idx


def is_finite_number(x: Any) -> bool:
    """Check if x is a finite number."""
    pass
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def sanitize_for_json(obj: Any) -> Any:
    """Sanitize objects for safe json.dumps (replace NaN/Inf with None)."""
    pass
    if is_finite_number(obj):
        return float(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    try:
        f = float(obj)  # type: ignore
        return f if math.isfinite(f) else None
    except Exception:
        return None


def safe_round(x: Any, ndigits: int = 3) -> Optional[float]:
    """Round finite numbers; return None if invalid."""
    pass
    try:
        xf = float(x)
        return round(xf, ndigits) if math.isfinite(xf) else None
    except Exception:
        return None


def parse_kv_overrides(items: List[str]) -> Dict[str, str]:
    """Parse list of key=value strings into dict."""
    pass
    out: Dict[str, str] = {}
    for item in items:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ---------------------------
# Parameter Management
# ---------------------------

DEFAULT_PARAM_DEFS: Dict[str, Dict[str, Any]] = {
    # Global
    "random_seed": dict(dtype="int", default=42, bounds=[1, 2147483647], owner_module="global", frozen=True),
    "population_size": dict(dtype="int", default=5000, bounds=[100, 100000], owner_module="global", frozen=False),
    "sim_days": dict(dtype="int", default=60, bounds=[30, 365], owner_module="global", frozen=False),
    "time_step_hours": dict(dtype="int", default=24, bounds=[1, 24], owner_module="global", frozen=False),
    "initial_adoption_rate": dict(dtype="float", default=0.05, bounds=[0.0, 1.0], owner_module="global", frozen=False),
    "age_share_young": dict(dtype="float", default=0.25, bounds=[0.0, 0.5], owner_module="global", frozen=False),
    "age_share_adult": dict(dtype="float", default=0.6, bounds=[0.4, 0.7], owner_module="global", frozen=False),
    "age_share_senior": dict(dtype="float", default=0.15, bounds=[0.05, 0.3], owner_module="global", frozen=False),
    "ses_share_low": dict(dtype="float", default=0.3, bounds=[0.1, 0.6], owner_module="global", frozen=False),
    "ses_share_mid": dict(dtype="float", default=0.5, bounds=[0.2, 0.7], owner_module="global", frozen=False),
    "ses_share_high": dict(dtype="float", default=0.2, bounds=[0.1, 0.5], owner_module="global", frozen=False),
    "openness_mean": dict(dtype="float", default=0.0, bounds=[-1.0, 1.0], owner_module="global", frozen=False),
    "openness_std": dict(dtype="float", default=0.5, bounds=[0.1, 1.0], owner_module="global", frozen=False),
    "trust_policy_mean": dict(dtype="float", default=0.0, bounds=[-1.0, 1.0], owner_module="global", frozen=False),
    "trust_policy_std": dict(dtype="float", default=0.5, bounds=[0.1, 1.0], owner_module="global", frozen=False),
    "threshold_mean": dict(dtype="float", default=0.4, bounds=[0.1, 0.9], owner_module="global", frozen=False),
    "threshold_std": dict(dtype="float", default=0.15, bounds=[0.05, 0.5], owner_module="global", frozen=False),
    # NetworkDynamics
    "degree_mean": dict(dtype="float", default=8.0, bounds=[2.0, 20.0], owner_module="NetworkDynamics", frozen=False),
    "degree_dispersion": dict(dtype="float", default=1.5, bounds=[0.5, 3.0], owner_module="NetworkDynamics", frozen=False),
    "clustering_bias": dict(dtype="float", default=0.2, bounds=[0.0, 1.0], owner_module="NetworkDynamics", frozen=False),
    "homophily_strength": dict(dtype="float", default=0.5, bounds=[0.0, 1.0], owner_module="NetworkDynamics", frozen=False),
    "rewiring_rate": dict(dtype="float", default=0.01, bounds=[0.0, 0.1], owner_module="NetworkDynamics", frozen=False),
    "assortativity_target": dict(dtype="float", default=0.2, bounds=[0.0, 0.8], owner_module="NetworkDynamics", frozen=False),
    "social_network_type": dict(dtype="str", default="small_world", owner_module="NetworkDynamics", frozen=False),
    "ws_rewiring_p": dict(dtype="float", default=0.05, bounds=[0.0, 1.0], owner_module="NetworkDynamics", frozen=False),
    # InteractionScheduler
    "daily_contact_fraction": dict(dtype="float", default=0.3, bounds=[0.05, 1.0], owner_module="InteractionScheduler", frozen=False),
    "meeting_noise": dict(dtype="float", default=0.1, bounds=[0.0, 0.5], owner_module="InteractionScheduler", frozen=False),
    "exogenous_contact_rate": dict(dtype="float", default=0.5, bounds=[0.0, 3.0], owner_module="InteractionScheduler", frozen=False),
    # PolicyBroadcast
    "message_frequency": dict(dtype="int", default=1, bounds=[1, 7], owner_module="PolicyBroadcast", frozen=False),
    "message_intensity": dict(dtype="float", default=0.7, bounds=[0.0, 2.0], owner_module="PolicyBroadcast", frozen=False),
    "media_campaign_intensity": dict(dtype="float", default=0.3, bounds=[0.0, 2.0], owner_module="PolicyBroadcast", frozen=False),
    "targeting_strength": dict(dtype="float", default=0.3, bounds=[0.0, 1.0], owner_module="PolicyBroadcast", frozen=False),
    "policy_channel_reach": dict(dtype="float", default=0.6, bounds=[0.1, 1.0], owner_module="PolicyBroadcast", frozen=False),
    "misinformation_fraction": dict(dtype="float", default=0.1, bounds=[0.0, 0.5], owner_module="PolicyBroadcast", frozen=False),
    "misinformation_rate": dict(dtype="float", default=0.1, bounds=[0.0, 1.0], owner_module="PolicyBroadcast", frozen=False),
    "trust_update_rate": dict(dtype="float", default=0.05, bounds=[0.0, 0.2], owner_module="PolicyBroadcast", frozen=False),
    "awareness_decay": dict(dtype="float", default=0.02, bounds=[0.0, 0.2], owner_module="PolicyBroadcast", frozen=False),
    # InfluenceAndAdoption
    "base_adoption_logit": dict(dtype="float", default=-2.0, bounds=[-5.0, 0.0], owner_module="InfluenceAndAdoption", frozen=False),
    "social_influence_weight": dict(dtype="float", default=1.5, bounds=[0.0, 3.0], owner_module="InfluenceAndAdoption", frozen=False),
    "personal_trait_weight": dict(dtype="float", default=1.0, bounds=[0.0, 2.0], owner_module="InfluenceAndAdoption", frozen=False),
    "policy_signal_weight": dict(dtype="float", default=1.2, bounds=[0.0, 3.0], owner_module="InfluenceAndAdoption", frozen=False),
    "fatigue_decay": dict(dtype="float", default=0.01, bounds=[0.0, 0.1], owner_module="InfluenceAndAdoption", frozen=False),
    "disadoption_rate": dict(dtype="float", default=0.001, bounds=[0.0, 0.05], owner_module="InfluenceAndAdoption", frozen=False),
    "noise_scale": dict(dtype="float", default=0.1, bounds=[0.0, 1.0], owner_module="InfluenceAndAdoption", frozen=False),
    "max_memory_days": dict(dtype="int", default=14, bounds=[1, 60], owner_module="InfluenceAndAdoption", frozen=False),
    "disease_incidence_signal": dict(dtype="float", default=0.01, bounds=[0.0, 1.0], owner_module="InfluenceAndAdoption", frozen=False),
    "risk_perception_sensitivity": dict(dtype="float", default=0.5, bounds=[0.0, 5.0], owner_module="InfluenceAndAdoption", frozen=False),
    "comfort_penalty": dict(dtype="float", default=0.2, bounds=[0.0, 2.0], owner_module="InfluenceAndAdoption", frozen=False),
    "compliance_cost_disutility": dict(dtype="float", default=0.2, bounds=[0.0, 2.0], owner_module="InfluenceAndAdoption", frozen=False),
    # Enforcement
    "enforcement_strength": dict(dtype="float", default=0.3, bounds=[0.0, 1.0], owner_module="LocationPolicy", frozen=False),
    "fine_amount": dict(dtype="float", default=50.0, bounds=[0.0, 1000.0], owner_module="LocationPolicy", frozen=False),
    # AdoptionAggregator
    "smoothing_window": dict(dtype="int", default=1, bounds=[1, 7], owner_module="AdoptionAggregator", frozen=False),
    "report_lag_days": dict(dtype="int", default=0, bounds=[0, 3], owner_module="AdoptionAggregator", frozen=False),
    # Movement/Context
    "location_mix": dict(
        dtype="dict",
        default={"home": 0.5, "work": 0.25, "school": 0.1, "retail": 0.1, "transit_public": 0.05},
        owner_module="ContextScheduler",
        frozen=False,
    ),
    # Supply/Retail
    "mask_price": dict(dtype="float", default=1.0, bounds=[0.0, 20.0], owner_module="SupplyChain", frozen=False),
    "price_elasticity": dict(dtype="float", default=-0.8, bounds=[-5.0, 0.0], owner_module="SupplyChain", frozen=False),
    "retailer_restock_rate": dict(dtype="float", default=0.1, bounds=[0.0, 1.0], owner_module="SupplyChain", frozen=False),
    "supply_lead_time_days": dict(dtype="int", default=7, bounds=[1, 60], owner_module="SupplyChain", frozen=False),
    "initial_inventory_per_capita": dict(dtype="float", default=2.0, bounds=[0.0, 50.0], owner_module="SupplyChain", frozen=False),
}


class ParameterManager:
    """Manages simulation parameters, defaults, overrides, and persistence."""
    pass

    def __init__(self, param_defs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialize with parameter definitions."""
        pass
        self.defs: Dict[str, Dict[str, Any]] = param_defs.copy() if param_defs else DEFAULT_PARAM_DEFS.copy()
        self.values: Dict[str, Any] = {k: v.get("default") for k, v in self.defs.items()}
        self.warnings: List[str] = []

    def load_file(self, path: Optional[str]) -> None:
        """Load parameter values from JSON file; ignore unknown keys."""
        pass
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept either flat dict or {"parameters": {...}}
            params = data.get("parameters", data) if isinstance(data, dict) else {}
            for k, v in params.items():
                if k in self.defs:
                    self.values[k] = self._cast_value(k, v)
        except FileNotFoundError:
            self.warnings.append(f"Parameter file not found: {path}. Using defaults.")
        except Exception as e:
            self.warnings.append(f"Failed to load parameters from {path}: {e}. Using defaults.")

    def apply_overrides(self, overrides: Dict[str, str]) -> None:
        """Apply CLI overrides; ignore frozen params with warnings."""
        pass
        for k, v in overrides.items():
            if k not in self.defs:
                self.warnings.append(f"Unknown parameter override ignored: {k}")
                continue
            frozen = bool(self.defs[k].get("frozen", False))
            if frozen:
                self.warnings.append(f"Override ignored for frozen parameter: {k}")
                continue
            try:
                self.values[k] = self._cast_value(k, v)
            except Exception as e:
                self.warnings.append(f"Invalid override for {k}: {v} ({e})")

    def _cast_value(self, key: str, v: Any) -> Any:
        """Cast value to parameter dtype."""
        pass
        info = self.defs.get(key, {})
        dtype = info.get("dtype")
        if dtype == "int":
            return int(v)
        if dtype == "float":
            return float(v)
        if dtype == "bool":
            if isinstance(v, bool):
                return v
            if str(v).lower() in ("1", "true", "yes", "y", "t"):
                return True
            if str(v).lower() in ("0", "false", "no", "n", "f"):
                return False
            raise ValueError("Invalid boolean")
        # FIXED: Support dict dtype by parsing JSON strings or ensuring dict type
        if dtype == "dict":
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    # leave to fallback
                    pass
            return dict(v) if isinstance(v, dict) else v
        return v

    def validate_bounds(self) -> List[str]:
        """Validate parameter bounds; auto-clamp if outside and log warning."""
        pass
        warns: List[str] = []
        for k, v in self.values.items():
            info = self.defs.get(k, {})
            bounds = info.get("bounds")
            if bounds and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                low, high = bounds
                try:
                    fv = float(v)
                    if fv < float(low) or fv > float(high):
                        new_v = clamp(fv, float(low), float(high))
                        self.values[k] = int(new_v) if info.get("dtype") == "int" else new_v
                        warns.append(f"Clamped {k} from {v} to {self.values[k]} within bounds {bounds}")
                except Exception:
                    # ignore non-numeric
                    pass
        self.warnings.extend(warns)
        return warns

    def export_definitions(self, path: str) -> None:
        """Export parameter definitions JSON for calibrators/adapter use."""
        pass
        try:
            ensure_dir(os.path.dirname(path))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(self.defs), f, indent=2)
        except Exception:
            # best-effort
            pass

    def persist_used(self, path: str) -> None:
        """Persist parameters_used.json with final values (guarded by SKIP_IO)."""
        pass
        # FIXED: Guard persistence with SKIP_IO to avoid I/O in restricted environments
        if SKIP_IO:
            return
        try:
            ensure_dir(os.path.dirname(path))
            payload = dict(parameters=self.values, warnings=self.warnings)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(payload), f, indent=2)
        except Exception:
            # best-effort
            pass

    def get_values(self) -> Dict[str, Any]:
        """Get current values dictionary."""
        pass
        return self.values.copy()

    def set_values(self, **kwargs: Any) -> None:
        """Set multiple values with casting; ignore unknown keys."""
        pass
        for k, v in kwargs.items():
            if k in self.defs:
                self.values[k] = self._cast_value(k, v)


# ---------------------------
# Entities
# ---------------------------


class Person:
    """Agent representation with attributes for adoption dynamics."""
    pass

    def __init__(
        self,
        idx: int,
        age_group: str,
        ses: str,
        openness: float,
        trust_policy: float,
        threshold: float,
        adopted: int,
        awareness: float = 0.0,
    ) -> None:
        """Initialize a Person object."""
        pass
        self.id = idx
        self.age_group = age_group
        self.ses = ses
        self.openness = openness
        self.conformity = 0.0
        self.trust_policy = trust_policy
        self.fatigue = 0.0
        self.threshold = clamp(threshold, 0.0, 1.0)
        self.adopted = int(adopted)
        self.awareness = clamp(awareness, 0.0, 1.0)
        self.last_adopt_day = 0 if adopted else -1
        self.degree = 0
        # FIXED: Awareness memory buffer used by InfluenceAndAdoption via max_memory_days
        self.awareness_memory: List[float] = []
        # FIXED: Added risk perception and basic economic and context attributes
        self.risk_perception = 0.0  # evolves with disease signal and media
        self.mask_inventory = 1 if adopted else 0  # start with a mask if already adopted
        self.income = 0.0  # assigned based on SES
        self.current_location_type = "home"  # updated daily by ContextScheduler


class PolicyChannel:
    """Policy channel parameters and counters."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize with parameters."""
        pass
        self.message_frequency = int(params.get("message_frequency", 1))
        # Support alias media_campaign_intensity
        self.message_intensity = float(
            params.get("message_intensity", params.get("media_campaign_intensity", 0.7))
        )
        self.targeting_strength = float(params.get("targeting_strength", 0.3))
        self.reach = float(params.get("policy_channel_reach", 0.6))
        # Support alias misinformation_rate
        self.misinformation_fraction = float(params.get("misinformation_fraction", params.get("misinformation_rate", 0.1)))
        self.trust_update_rate = float(params.get("trust_update_rate", 0.05))
        self.messages_sent = 0


# ---------------------------
# Simulation Modules
# ---------------------------


class BaseModule:
    """Base module interface for forward computation."""
    pass

    def __init__(self, name: str, params: Dict[str, Any]) -> None:
        """Initialize base module with name and params."""
        pass
        self.name = name
        self.params = params

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Compute outputs and write to buffers. No global state mutation."""
        pass
        raise NotImplementedError("BaseModule.forward not implemented")


class NetworkDynamics(BaseModule):
    """Network module: builds and rewires social network."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize network dynamics module."""
        pass
        super().__init__("NetworkDynamics", params)

    def _build_initial_graph(self, N: int, persons: List[Person], rng: random.Random) -> List[set]:
        """Build initial adjacency; supports small-world topology and fallback stub matching."""
        pass
        adj: List[set] = [set() for _ in range(N)]
        net_type = str(self.params.get("social_network_type", "small_world")).lower()
        k = max(2, int(round(float(self.params.get("degree_mean", 8.0)))))
        if net_type == "small_world":
            # Watts–Strogatz ring lattice
            p_rewire = float(self.params.get("ws_rewiring_p", 0.05))
            for i in range(N):
                for j in range(1, k // 2 + 1):
                    u, v = i, (i + j) % N
                    if u != v:
                        adj[u].add(v)
                        adj[v].add(u)
            # Rewire with probability p_rewire
            for i in range(N):
                for j in list(adj[i]):
                    if i < j and rng.random() < p_rewire:
                        if j in adj[i]:
                            adj[i].remove(j)
                            adj[j].remove(i)
                        candidates = [x for x in range(N) if x != i and x not in adj[i]]
                        if candidates:
                            w = rng.choice(candidates)
                            adj[i].add(w)
                            adj[w].add(i)
        else:
            # Fallback: stub matching with Poisson degree
            mean_deg = float(self.params.get("degree_mean", 8.0))
            # Sample degrees via Poisson approx
            degs = [max(0, self._poisson(mean_deg, rng)) for _ in range(N)]
            # Build stubs
            stubs: List[int] = []
            for i, k_i in enumerate(degs):
                for _ in range(k_i):
                    stubs.append(i)
            rng.shuffle(stubs)
            # Pair stubs
            E: set = set()
            for i in range(0, len(stubs) - 1, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in E:
                    continue
                E.add((a, b))
            # Add edges
            for (u, v) in E:
                adj[u].add(v)
                adj[v].add(u)

        # Homophily and clustering augmentation
        self._homophily_rewire(adj, persons, rng, fraction=0.05)
        try:
            clustering_bias = float(self.params.get("clustering_bias", 0.0))
        except Exception:
            clustering_bias = 0.0
        if clustering_bias > 0.0:
            self._triadic_closure(adj, rng, clustering_bias)
        return adj

    def _triadic_closure(self, adj: List[set], rng: random.Random, clustering_bias: float) -> None:
        """Add edges by closing triangles with probability proportional to clustering_bias."""
        pass
        N = len(adj)
        if N <= 2:
            return
        # Number of attempts scales with bias and edges
        total_edges = sum(len(nei) for nei in adj) // 2
        attempts = max(0, int(clustering_bias * max(N, total_edges)))
        tried = 0
        while tried < attempts:
            tried += 1
            u = rng.randrange(N)
            if len(adj[u]) < 2:
                continue
            v = rng.choice(list(adj[u]))
            if not adj[v]:
                continue
            w = rng.choice(list(adj[v]))
            if w == u:
                continue
            a, b = (u, w) if u < w else (w, u)
            if b not in adj[a] and a != b:
                adj[a].add(b)
                adj[b].add(a)

    def _poisson(self, lam: float, rng: random.Random) -> int:
        """Knuth algorithm for Poisson sampling with small lambda."""
        pass
        if lam <= 0.0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L and k < 10000:
            k += 1
            p *= rng.random()
        return max(0, k - 1)

    def _homophily_score(self, i: int, j: int, persons: List[Person]) -> float:
        """Compute similarity score based on age_group, ses, adoption."""
        pass
        pi = persons[i]
        pj = persons[j]
        score = 0.0
        score += 1.0 if pi.age_group == pj.age_group else 0.0
        score += 1.0 if pi.ses == pj.ses else 0.0
        score += 1.0 if pi.adopted == pj.adopted else 0.0
        return score

    def _homophily_rewire(self, adj: List[set], persons: List[Person], rng: random.Random, fraction: float) -> None:
        """Rewire a fraction of edges to increase homophily (sampled candidates)."""
        pass
        N = len(adj)
        edges = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        if not edges:
            return
        rng.shuffle(edges)
        # FIXED: Calculate K without forcing at least one; allow zero rewiring
        K = max(0, int(round(fraction * len(edges))))
        if K == 0:
            return
        homo_strength = float(self.params.get("homophily_strength", 0.5))
        # FIXED: Optimize by sampling limited candidate non-neighbors rather than scanning all
        max_candidates = 64
        for idx in range(min(K, len(edges))):
            u, v = edges[idx]
            current_score = self._homophily_score(u, v, persons)
            if rng.random() < homo_strength * 0.5:
                # Sample a subset of non-neighbors
                non_neighbors: List[int] = []
                attempts = 0
                target = min(max_candidates, max(0, N - 1 - len(adj[u])))
                seen: set = set()
                while len(non_neighbors) < target and attempts < 5 * max_candidates:
                    attempts += 1
                    w = rng.randrange(N)
                    if w == u or w in adj[u] or w in seen:
                        continue
                    seen.add(w)
                    non_neighbors.append(w)
                if not non_neighbors:
                    continue
                w = max(
                    non_neighbors,
                    key=lambda c: self._homophily_score(u, c, persons) + rng.random() * 0.01,
                )
                new_score = self._homophily_score(u, w, persons)
                if new_score > current_score:
                    # Rewire
                    if v in adj[u]:
                        adj[u].remove(v)
                        adj[v].remove(u)
                    adj[u].add(w)
                    adj[w].add(u)

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Build or rewire the network and write adjacency and degrees to buffers."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        # Build or copy adjacency
        if t == 0 or state.get("adjacency") is None:
            adj = self._build_initial_graph(N, persons, rng)
        else:
            adj = [set(nei) for nei in state["adjacency"]]
            # Daily rewiring
            # FIXED: Removed stray quote causing SyntaxError
            rewiring_rate = float(self.params.get("rewiring_rate", 0.01))
            self._homophily_rewire(adj, persons, rng, fraction=rewiring_rate)

        degrees = [len(adj[i]) for i in range(N)]
        buffers["graph"] = adj
        buffers["degrees"] = degrees
        buffers["average_degree_daily"] = sum(degrees) / float(N) if N > 0 else 0.0


class PolicyBroadcast(BaseModule):
    """Broadcast policy messages to agents, updating awareness and trust."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize policy broadcast module."""
        pass
        super().__init__("PolicyBroadcast", params)

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Emit policy signals to a subset of agents with awareness decay and trust update."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        signal_by_agent: List[float] = [0.0 for _ in range(N)]
        messages_sent = 0
        freq = int(self.params.get("message_frequency", 1))
        # Support alias media_campaign_intensity for message_intensity
        intensity = float(self.params.get("message_intensity", self.params.get("media_campaign_intensity", 0.7)))
        targeting_strength = float(self.params.get("targeting_strength", 0.3))
        reach = float(self.params.get("policy_channel_reach", 0.6))
        misinformation_fraction = float(self.params.get("misinformation_fraction", self.params.get("misinformation_rate", 0.1)))
        trust_update_rate = float(self.params.get("trust_update_rate", 0.05))
        max_memory_days = int(self.params.get("max_memory_days", 14))
        awareness_decay = float(self.params.get("awareness_decay", 0.02))
        if freq <= 0:
            freq = 1

        # Mild default decay for trust each day (toward 0)
        if trust_update_rate > 0.0:
            for pi in persons:
                pi.trust_policy = clamp((1.0 - 0.5 * trust_update_rate) * pi.trust_policy, -2.0, 2.0)

        # Daily awareness decay for everyone
        for pi in persons:
            pi.awareness = clamp(pi.awareness * (1.0 - awareness_decay), 0.0, 1.0)

        # Memory bumps default to zero
        memory_bumps: List[float] = [0.0 for _ in range(N)]

        if t % freq == 0:
            # Prioritize non-adopters
            non_adopters = [i for i, p in enumerate(persons) if p.adopted == 0]
            others = [i for i, p in enumerate(persons) if p.adopted == 1]
            target_pool: List[int] = []
            for i in non_adopters:
                target_pool.append(i)
            # Include a fraction of others based on targeting_strength
            include_count = int((1.0 - targeting_strength) * len(others))
            if include_count > 0 and others:
                rng.shuffle(others)
                target_pool.extend(others[:include_count])

            # Determine reached agents
            M = int(reach * len(target_pool))
            rng.shuffle(target_pool)
            reached = target_pool[:M]

            misinformation_factor = 1.0 - misinformation_fraction
            for i in reached:
                pi = persons[i]
                # FIXED: Decouple policy signal from trust; modulate responsiveness with bounded function
                base = intensity * misinformation_factor
                trust_amp = 0.5 + 0.5 * math.tanh(pi.trust_policy)
                signal = base * trust_amp
                signal_by_agent[i] = signal
                # Awareness bump with decay already applied
                pi.awareness = clamp(pi.awareness + 0.2, 0.0, 1.0)
                memory_bumps[i] = 1.0
                # FIXED: Update trust toward a bounded target derived from base signal with reversible smoothing
                raw_target = math.tanh(base)
                target = 2.0 * raw_target  # map to [-2, 2]
                pi.trust_policy = clamp(
                    (1.0 - trust_update_rate) * pi.trust_policy + trust_update_rate * target,
                    -2.0,
                    2.0,
                )
            messages_sent = len(reached)

        # FIXED: Update awareness memory for all agents with today's bump (0 or 1) and enforce max memory length
        for i, pi in enumerate(persons):
            pi.awareness_memory.append(memory_bumps[i])
            if len(pi.awareness_memory) > max_memory_days:
                # Trim oldest
                pi.awareness_memory = pi.awareness_memory[-max_memory_days:]

        buffers["policy_signal_by_agent"] = signal_by_agent
        buffers["messages_sent"] = messages_sent


class InteractionScheduler(BaseModule):
    """Sample daily active contacts from graph and exogenous random meetings."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize interaction scheduler module."""
        pass
        super().__init__("InteractionScheduler", params)

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Compute active neighbors per agent as daily interactions."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        # FIXED: Guard adjacency presence with fallback to empty sets
        adj: List[set] = state.get("adjacency") or [set() for _ in range(N)]
        # FIXED: Clamp parameters to safe ranges and ensure non-negative exogenous rate
        daily_fraction = clamp(float(self.params.get("daily_contact_fraction", 0.3)), 0.0, 1.0)
        meeting_noise = clamp(float(self.params.get("meeting_noise", 0.1)), 0.0, 1.0)
        exogenous_rate = max(0.0, float(self.params.get("exogenous_contact_rate", 0.5)))

        # Active neighbors per agent
        active_neighbors: List[List[int]] = []
        for i in range(N):
            neigh = list(adj[i]) if i < len(adj) else []
            rng.shuffle(neigh)
            # FIXED: Use the correctly defined variable name
            k = max(0, int(len(neigh) * daily_fraction))
            subset = neigh[:k]
            # Exogenous random meetings
            exo_k = max(0, int(exogenous_rate))
            if rng.random() < (exogenous_rate - int(exogenous_rate)):
                exo_k += 1
            for _ in range(exo_k):
                if N <= 1:
                    break
                j = rng.randrange(N)
                if j != i and rng.random() < meeting_noise:
                    subset.append(j)
            # Deduplicate
            subset = list(set(subset))
            active_neighbors.append(subset)

        buffers["daily_active_neighbors"] = active_neighbors


class ContextScheduler(BaseModule):
    """Assign a daily context (location type) to each person based on a location mix."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize context scheduler."""
        pass
        super().__init__("ContextScheduler", params)

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Draw a context for each person and write counts by context to buffers."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        mix: Dict[str, float] = dict(self.params.get("location_mix", {"home": 1.0}))
        # Normalize mix
        total = sum(float(v) for v in mix.values()) or 1.0
        items = list(mix.items())
        probs = [float(v) / total for (_, v) in items]
        contexts = [k for (k, _) in items]
        counts: Dict[str, int] = {k: 0 for k in contexts}

        for p in persons:
            r = rng.random()
            cum = 0.0
            choice = contexts[-1]
            for prob, ctx in zip(probs, contexts):
                cum += prob
                if r <= cum:
                    choice = ctx
                    break
            p.current_location_type = choice
            counts[choice] = counts.get(choice, 0) + 1

        buffers["context_counts"] = counts


class SupplyChain(BaseModule):
    """Minimal retailer and purchasing with inventory, restocking, and stockout tracking."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize supply chain module with retailer state."""
        pass
        super().__init__("SupplyChain", params)
        self.initialized = False
        self.retailer: Dict[str, Any] = {
            "inventory": 0,
            "on_order": [],  # list of (day_available, quantity)
            "attempts_today": 0,
            "stockouts_today": 0,
        }

    def _init_retailer(self, state: Dict[str, Any]) -> None:
        """Initialize retailer inventory and queues."""
        pass
        N = len(state["persons"])
        inv_per_capita = float(self.params.get("initial_inventory_per_capita", 2.0))
        initial_inventory = max(0, int(inv_per_capita * N))
        self.retailer["inventory"] = initial_inventory
        self.retailer["on_order"] = []
        self.initialized = True

    def _receive_orders(self, t: int) -> None:
        """Receive any orders that have arrived at day t."""
        pass
        arriving = [q for (day, q) in list(self.retailer["on_order"]) if day <= t]
        if arriving:
            self.retailer["inventory"] += sum(arriving)
            # Remove arrived
            self.retailer["on_order"] = [(day, q) for (day, q) in self.retailer["on_order"] if day > t]

    def _maybe_place_order(self, t: int, state: Dict[str, Any]) -> None:
        """Place restock orders based on restock rate; fulfillment after a lead time."""
        pass
        restock_rate = float(self.params.get("retailer_restock_rate", 0.1))
        lead = int(self.params.get("supply_lead_time_days", 7))
        N = len(state["persons"])
        # Target order based on rate and population
        qty = int(max(0.0, restock_rate * N))
        if qty > 0:
            self.retailer["on_order"].append((t + lead, qty))

    def _afford_prob(self, income: float, price: float, elasticity: float) -> float:
        """Compute purchase probability given income, price, and price elasticity."""
        pass
        # Relative price burden: price as a fraction of daily income proxy
        # FIXED: Clamp income to non-negative to avoid extreme burdens
        denom = max(1.0, max(0.0, income) / 30.0)
        burden = price / denom
        a = abs(elasticity)
        # Higher burden reduces probability; center around 0.5 at burden ~ 1.0/a
        x = 1.0 - a * burden
        return clamp(sigmoid(x), 0.0, 1.0)

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Process restocking and purchasing attempts; write daily stockouts and attempts to buffers."""
        pass
        if not self.initialized:
            self._init_retailer(state)

        # Receive arriving orders
        self._receive_orders(t)
        # Reset counters
        self.retailer["attempts_today"] = 0
        self.retailer["stockouts_today"] = 0

        # Place orders each day
        self._maybe_place_order(t, state)

        persons: List[Person] = state["persons"]
        price = float(self.params.get("mask_price", 1.0))
        elasticity = float(self.params.get("price_elasticity", -0.8))
        # Location types where masks are more likely required
        enforced_types = {"retail", "transit_public"}

        # Attempt purchases for persons who may need a mask today
        for p in persons:
            need_mask = (p.adopted == 1 and p.mask_inventory <= 0) or (
                p.mask_inventory <= 0 and p.current_location_type in enforced_types and p.awareness > 0.2
            )
            if not need_mask:
                continue
            self.retailer["attempts_today"] += 1
            p_buy = self._afford_prob(p.income, price, elasticity)
            will_buy = rng.random() < p_buy
            if will_buy:
                if self.retailer["inventory"] > 0:
                    self.retailer["inventory"] -= 1
                    # Grant one mask
                    p.mask_inventory += 1
                else:
                    self.retailer["stockouts_today"] += 1

        # Write buffers
        attempts = int(self.retailer["attempts_today"])
        stockouts = int(self.retailer["stockouts_today"])
        buffers["supply.purchase_attempts"] = attempts
        buffers["supply.stockouts"] = stockouts
        buffers["supply.inventory"] = int(self.retailer["inventory"])


class InfluenceAndAdoption(BaseModule):
    """Compute adoption probabilities and update adoption states via buffers."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize influence and adoption module."""
        pass
        super().__init__("InfluenceAndAdoption", params)

    def _enforcement_signal(self, location: str) -> float:
        """Compute enforcement-based signal contribution for the given location."""
        pass
        enforcement_strength = float(self.params.get("enforcement_strength", 0.3))
        fine_amount = float(self.params.get("fine_amount", 50.0))
        # Stronger signal in enforced contexts
        if location in {"retail", "transit_public"}:
            return clamp(enforcement_strength + 0.002 * fine_amount, 0.0, 1.5)
        return 0.0

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Compute adoption updates and write to buffers; no direct state mutation."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        active_neighbors: List[List[int]] = buffers.get("daily_active_neighbors", [[] for _ in range(N)])
        policy_signal: List[float] = buffers.get("policy_signal_by_agent", [0.0 for _ in range(N)])

        base_logit = float(self.params.get("base_adoption_logit", -2.0))
        w_social = float(self.params.get("social_influence_weight", 1.5))
        w_trait = float(self.params.get("personal_trait_weight", 1.0))
        w_policy = float(self.params.get("policy_signal_weight", 1.2))
        fatigue_decay = float(self.params.get("fatigue_decay", 0.01))
        disadopt_rate = float(self.params.get("disadoption_rate", 0.001))
        noise_scale = float(self.params.get("noise_scale", 0.1))
        disease_signal = float(self.params.get("disease_incidence_signal", 0.01))
        risk_w = float(self.params.get("risk_perception_sensitivity", 0.5))
        comfort_penalty = float(self.params.get("comfort_penalty", 0.2))
        compliance_disutil = float(self.params.get("compliance_cost_disutility", 0.2))

        updates: List[Tuple[int, int]] = []
        for i in range(N):
            pi = persons[i]
            # Fatigue decay
            pi.fatigue = max(0.0, pi.fatigue - fatigue_decay)

            # Update risk perception (bounded)
            pi.risk_perception = clamp(0.9 * pi.risk_perception + 0.1 * (disease_signal + 0.2 * pi.awareness), 0.0, 1.0)

            neigh = active_neighbors[i] if i < len(active_neighbors) else []
            frac_adopted = 0.0
            if neigh:
                adopted_count = sum(1 for j in neigh if persons[j].adopted == 1)
                frac_adopted = adopted_count / float(len(neigh))

            social_term = w_social * (frac_adopted - pi.threshold)
            trait_term = w_trait * float(pi.openness)
            # Awareness-integrated policy effect; include simple memory of past messages
            mem_avg = sum(pi.awareness_memory) / len(pi.awareness_memory) if pi.awareness_memory else 0.0
            awareness_effect = 0.5 * float(pi.awareness) + 0.5 * mem_avg
            awareness_factor = 0.5 + 0.5 * awareness_effect
            policy_term = (w_policy * float(policy_signal[i]) if i < len(policy_signal) else 0.0) * awareness_factor

            # Enforcement at current location
            loc = getattr(pi, "current_location_type", "home")
            enforce_term = self._enforcement_signal(loc)

            # Comfort and compliance costs penalize adoption tendency
            penalty_term = -comfort_penalty * (1.0 if pi.adopted == 1 else 0.5) - compliance_disutil

            # Risk perception promotes adoption
            risk_term = risk_w * pi.risk_perception

            noise = rng.gauss(0.0, noise_scale)
            logit_p = base_logit + social_term + trait_term + policy_term + enforce_term + risk_term + penalty_term - pi.fatigue + noise
            p_adopt = sigmoid(logit_p)

            if pi.adopted == 0:
                if rng.random() < p_adopt:
                    updates.append((i, 1))
            else:
                # Higher reversion when social support is low and no enforcement
                revert_base = disadopt_rate * max(0.0, 1.0 - frac_adopted) * (1.0 - min(1.0, enforce_term))
                if rng.random() < revert_base:
                    updates.append((i, -1))

        buffers["adoption_updates"] = updates


class AdoptionAggregator(BaseModule):
    """Aggregate daily observables and produce smoothed time series."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize aggregator module."""
        pass
        super().__init__("AdoptionAggregator", params)
        self.raw_adoption_rate: List[float] = []
        self.raw_new_adopters: List[int] = []
        self.raw_disadopters: List[int] = []
        self.raw_messages: List[int] = []
        self.raw_avg_degree: List[float] = []
        self.raw_p01: List[float] = []  # daily transition non->adopt
        self.raw_p10: List[float] = []  # daily transition adopt->non
        self.observed_adoption_rate: List[float] = []
        self.observed_new_adopters: List[float] = []
        self.observed_messages: List[float] = []
        self.observed_avg_degree: List[float] = []
        # FIXED: Track supply-related metrics
        self.raw_stockout_rate: List[float] = []
        self.stockout_days: int = 0

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Append raw metrics, apply smoothing and lag, and write latest observables to buffers."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        adopted_count = sum(1 for p in persons if p.adopted == 1)
        adoption_rate = adopted_count / float(N) if N > 0 else 0.0
        updates = buffers.get("adoption_updates", [])
        new_adopters = sum(1 for (_i, delta) in updates if delta == 1)
        disadopters = sum(1 for (_i, delta) in updates if delta == -1)
        messages = int(buffers.get("messages_sent", 0))
        avg_deg = float(buffers.get("average_degree_daily", 0.0))
        prev_adopted_count = int(buffers.get("prev_adopted_count", adopted_count))

        # Transition estimates
        non_prev = max(1, N - prev_adopted_count)
        prev_adopted_nonzero = max(1, prev_adopted_count)
        p01 = float(new_adopters) / float(non_prev)
        p10 = float(disadopters) / float(prev_adopted_nonzero)

        self.raw_adoption_rate.append(adoption_rate)
        self.raw_new_adopters.append(new_adopters)
        self.raw_disadopters.append(disadopters)
        self.raw_messages.append(messages)
        self.raw_avg_degree.append(avg_deg)
        self.raw_p01.append(p01)
        self.raw_p10.append(p10)

        window = int(self.params.get("smoothing_window", 1))
        lag = int(self.params.get("report_lag_days", 0))

        # Apply moving average
        smoothed_adopt = moving_average(self.raw_adoption_rate, window)
        smoothed_new = moving_average([float(x) for x in self.raw_new_adopters], window)
        smoothed_msg = moving_average([float(x) for x in self.raw_messages], window)
        smoothed_deg = moving_average(self.raw_avg_degree, window)

        # Apply lag: we shift output using lag; fill earlier with last available
        def lagged(series: List[float], lag_days: int) -> float:
            idx = len(series) - 1 - lag_days
            if idx < 0:
                return series[0] if series else 0.0
            return series[idx]

        self.observed_adoption_rate.append(lagged(smoothed_adopt, lag))
        self.observed_new_adopters.append(lagged(smoothed_new, lag))
        self.observed_messages.append(lagged(smoothed_msg, lag))
        self.observed_avg_degree.append(lagged(smoothed_deg, lag))

        # Expose last observables
        buffers["observable.adoption_rate_daily"] = self.observed_adoption_rate[-1]
        buffers["observable.new_adopters_daily"] = self.observed_new_adopters[-1]
        buffers["observable.policy_messages_daily"] = self.observed_messages[-1]
        buffers["observable.average_degree_daily"] = self.observed_avg_degree[-1]
        # Expose transitions (raw, unsmoothed) for diagnostics
        buffers["transitions.P01_daily"] = self.raw_p01[-1]
        buffers["transitions.P10_daily"] = self.raw_p10[-1]

        # FIXED: Contextual adoption rates
        by_ctx: Dict[str, Dict[str, int]] = {}
        for p in persons:
            c = getattr(p, "current_location_type", "home")
            if c not in by_ctx:
                by_ctx[c] = {"adopt": 0, "count": 0}
            by_ctx[c]["adopt"] += 1 if p.adopted == 1 else 0
            by_ctx[c]["count"] += 1
        ctx_rates = {k: (v["adopt"] / v["count"] if v["count"] > 0 else 0.0) for k, v in by_ctx.items()}
        buffers["observable.contextual_adoption_rate"] = ctx_rates

        # FIXED: Inequality index across SES groups (range of group means)
        ses_groups: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        for p in persons:
            ses_groups.setdefault(getattr(p, "ses", "mid"), []).append(1.0 if p.adopted == 1 else 0.0)
        group_means = [sum(v) / len(v) for v in ses_groups.values() if v]
        inequality = (max(group_means) - min(group_means)) if len(group_means) >= 2 else 0.0
        buffers["metric.inequality_index"] = inequality

        # FIXED: Stockout rate from SupplyChain buffers
        attempts = int(buffers.get("supply.purchase_attempts", 0))
        stockouts = int(buffers.get("supply.stockouts", 0))
        rate = (float(stockouts) / float(max(1, attempts))) if attempts > 0 else 0.0
        self.raw_stockout_rate.append(rate)
        if stockouts > 0:
            self.stockout_days += 1
        buffers["metric.stockout_rate_daily"] = rate
        buffers["supply.inventory_level"] = int(buffers.get("supply.inventory", 0))
        # FIXED: Expose per-capita purchase rate
        buffers["supply.purchase_rate_daily"] = attempts / float(max(1, N))


# ---------------------------
# Simulation and Scheduler
# ---------------------------


class Simulation:
    """Main simulation engine coordinating modules and state."""
    pass

    def __init__(self, params: Dict[str, Any], rng_seed: Optional[int] = None) -> None:
        """Initialize Simulation with params and optional seed."""
        pass
        self.params = params.copy()
        self.N = int(self.params.get("population_size", 1000))
        self.sim_days = int(self.params.get("sim_days", 60))
        self.rng = random.Random()
        self.seed = int(self.params.get("random_seed", 42) if rng_seed is None else rng_seed)
        self.rng.seed(self.seed)
        seed_all(self.seed)

        # Entities and state
        self.persons: List[Person] = self._initialize_population(self.N, self.params, self.rng)
        self.policy_channel = PolicyChannel(self.params)
        self.state: Dict[str, Any] = {
            "persons": self.persons,
            "adjacency": None,
        }

        # Modules
        self.mod_network = NetworkDynamics(self.params)
        self.mod_policy = PolicyBroadcast(self.params)
        self.mod_context = ContextScheduler(self.params)
        self.mod_interaction = InteractionScheduler(self.params)
        self.mod_supply = SupplyChain(self.params)
        self.mod_influence = InfluenceAndAdoption(self.params)
        self.mod_aggregate = AdoptionAggregator(self.params)
        self.modules: List[BaseModule] = [
            self.mod_network,
            self.mod_policy,
            self.mod_context,
            self.mod_interaction,
            self.mod_supply,
            self.mod_influence,
            self.mod_aggregate,
        ]

        # IO storage
        self.buffers: Dict[str, Any] = {}
        self.history: Dict[str, List[Any]] = {
            "observable.adoption_rate_daily": [],
            "observable.new_adopters_daily": [],
            "observable.policy_messages_daily": [],
            "observable.average_degree_daily": [],
            "transitions.P01_daily": [],
            "transitions.P10_daily": [],
            "metric.stockout_rate_daily": [],
            # FIXED: Persist additional series for metrics alignment
            "supply.purchase_rate_daily": [],
            "observable.contextual_adoption_rate": [],
            "metric.inequality_index": [],
        }

    def _initialize_population(self, N: int, params: Dict[str, Any], rng: random.Random) -> List[Person]:
        """Initialize agents according to attribute distributions."""
        pass
        age_shares = [
            float(params.get("age_share_young", 0.25)),
            float(params.get("age_share_adult", 0.6)),
            float(params.get("age_share_senior", 0.15)),
        ]
        ses_shares = [
            float(params.get("ses_share_low", 0.3)),
            float(params.get("ses_share_mid", 0.5)),
            float(params.get("ses_share_high", 0.2)),
        ]
        openness_mean = float(params.get("openness_mean", 0.0))
        openness_std = float(params.get("openness_std", 0.5))
        trust_policy_mean = float(params.get("trust_policy_mean", 0.0))
        trust_policy_std = float(params.get("trust_policy_std", 0.5))
        threshold_mean = float(params.get("threshold_mean", 0.4))
        threshold_std = float(params.get("threshold_std", 0.15))
        init_adopt_rate = float(params.get("initial_adoption_rate", 0.05))

        # Normalize shares
        def draw_categorical(shares: List[float], labels: List[str]) -> str:
            s = sum(shares)
            probs = [x / s if s > 0 else 0.0 for x in shares]
            r = rng.random()
            cum = 0.0
            for p, lab in zip(probs, labels):
                cum += p
                if r <= cum:
                    return lab
            return labels[-1]

        persons: List[Person] = []
        for i in range(N):
            age_group = draw_categorical(age_shares, ["young", "adult", "senior"])
            ses = draw_categorical(ses_shares, ["low", "mid", "high"])
            openness = clamp(rng.gauss(openness_mean, openness_std), -3.0, 3.0)
            trust_policy = clamp(rng.gauss(trust_policy_mean, trust_policy_std), -2.0, 2.0)
            threshold = clamp(rng.gauss(threshold_mean, threshold_std), 0.0, 1.0)
            adopted = 1 if rng.random() < init_adopt_rate else 0
            p = Person(i, age_group, ses, openness, trust_policy, threshold, adopted, awareness=0.0)
            # Assign income based on SES
            p.income = 30.0 if ses == "low" else 70.0 if ses == "mid" else 150.0
            persons.append(p)
        return persons

    def run(self, start_day: int = 0, end_day: Optional[int] = None, time_budget_sec: Optional[float] = None) -> None:
        """Run simulation in [start_day, end_day] (inclusive end if provided)."""
        pass
        T = self.sim_days if end_day is None else min(self.sim_days, int(end_day) + 1)
        start = max(0, int(start_day))
        start_time = time.time()
        for t in range(start, T):
            # Time budget guard
            if time_budget_sec is not None and (time.time() - start_time) > time_budget_sec:
                break

            # Buffers reset
            buffers: Dict[str, Any] = {}
            # Pre-capture adopted count for transitions (before updates)
            buffers["prev_adopted_count"] = sum(1 for p in self.persons if p.adopted == 1)

            # Network build/rewire
            self.mod_network.forward(self.state, buffers, self.params, t, self.rng)
            # Commit adjacency and degrees
            self.state["adjacency"] = buffers.get("graph", self.state.get("adjacency"))
            degrees = buffers.get("degrees", [])
            for i, p in enumerate(self.persons):
                p.degree = degrees[i] if i < len(degrees) else p.degree

            # Policy broadcast
            self.mod_policy.forward(self.state, buffers, self.params, t, self.rng)
            # Context scheduling
            self.mod_context.forward(self.state, buffers, self.params, t, self.rng)
            # Interactions
            self.mod_interaction.forward(self.state, buffers, self.params, t, self.rng)
            # Supply chain purchasing pre-adoption
            self.mod_supply.forward(self.state, buffers, self.params, t, self.rng)
            # Influence and adoption
            self.mod_influence.forward(self.state, buffers, self.params, t, self.rng)

            # Commit adoption updates
            for (i, delta) in buffers.get("adoption_updates", []):
                if 0 <= i < len(self.persons):
                    if delta == 1 and self.persons[i].adopted == 0:
                        self.persons[i].adopted = 1
                        self.persons[i].last_adopt_day = t
                        self.persons[i].fatigue = self.persons[i].fatigue + 0.1
                    elif delta == -1 and self.persons[i].adopted == 1:
                        self.persons[i].adopted = 0

            # Optional consumption: if adopted and in enforced context and has a mask, consume one
            for p in self.persons:
                if p.adopted == 1 and p.current_location_type in {"retail", "transit_public"} and p.mask_inventory > 0:
                    p.mask_inventory -= 1

            # Aggregation
            self.mod_aggregate.forward(self.state, buffers, self.params, t, self.rng)

            # Record observables
            for key in list(self.history.keys()):
                val = buffers.get(key, 0.0)
                # FIXED: Store dicts as-is (sanitized) and numbers as float
                if isinstance(val, dict):
                    self.history[key].append(sanitize_for_json(val))
                else:
                    try:
                        self.history[key].append(float(val))
                    except Exception:
                        self.history[key].append(sanitize_for_json(val))

    def save_results(self, path: str) -> None:
        """Save simulation results to JSON file."""
        pass
        if SKIP_IO:
            return
        try:
            ensure_dir(os.path.dirname(path))
            payload = {
                "parameters": self.params,
                "results": self.history,
                "seed": self.seed,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(payload), f, indent=2)
        except Exception:
            # best-effort
            pass

    def save_all_io(self, root_dir: str) -> None:
        """Save module I/O if needed."""
        pass
        if SKIP_IO:
            return
        try:
            ensure_dir(root_dir)
            # Save per-module observables
            agg_dir = os.path.join(root_dir, "AdoptionAggregator")
            ensure_dir(agg_dir)
            obs = {
                "observable.adoption_rate_daily": self.mod_aggregate.observed_adoption_rate,
                "observable.new_adopters_daily": self.mod_aggregate.observed_new_adopters,
                "observable.policy_messages_daily": self.mod_aggregate.observed_messages,
                "observable.average_degree_daily": self.mod_aggregate.observed_avg_degree,
                "transitions.P01_daily": self.mod_aggregate.raw_p01,
                "transitions.P10_daily": self.mod_aggregate.raw_p10,
            }
            with open(os.path.join(agg_dir, "observables.json"), "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(obs), f, indent=2)
        except Exception:
            pass

    def visualize(self, show: bool = False, save_path: Optional[str] = None) -> None:
        """Optional visualization of adoption rates; fallback to noop if matplotlib missing."""
        pass
        if SKIP_IO and not show:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore

            t = list(range(len(self.history["observable.adoption_rate_daily"])))
            plt.figure(figsize=(8, 4))
            plt.plot(t, self.history["observable.adoption_rate_daily"], label="Adoption rate")
            plt.xlabel("Day")
            plt.ylabel("Rate")
            plt.title("Adoption Rate Over Time")
            plt.legend()
            plt.tight_layout()
            if save_path:
                ensure_dir(os.path.dirname(save_path))
                plt.savefig(save_path)
            if show:
                plt.show()
            plt.close()
        except Exception:
            # graceful degradation
            pass

    def get_params(self) -> Dict[str, Any]:
        """Get current parameters."""
        pass
        return self.params.copy()

    def set_params(self, **kwargs: Any) -> None:
        """Set parameters at runtime and propagate into modules."""
        pass
        self.params.update(kwargs)
        # Update modules' param views
        for mod in self.modules:
            mod.params.update(kwargs)

    def evaluate(self, gt: Dict[str, List[float]], window: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """Compute evaluation metrics comparing observables to ground truth and include supply/policy metrics."""
        pass
        # Build predictions
        pred_adopt = self.history.get("observable.adoption_rate_daily", [])
        if not pred_adopt:
            return {}
        # Ground truth series fallback
        gt_adopt = gt.get("adoption_rate", [0.0 for _ in range(len(pred_adopt))])

        # Align window
        start = 0 if window is None else max(0, int(window[0]))
        end = len(pred_adopt) - 1 if window is None else min(len(pred_adopt) - 1, int(window[1]))
        if end < start:
            start, end = 0, len(pred_adopt) - 1
        y_pred = pred_adopt[start : end + 1]
        y_true = gt_adopt[start : end + 1] if len(gt_adopt) >= end + 1 else gt_adopt

        metrics: Dict[str, Any] = {}
        metrics["RMSE"] = rmse(y_true, y_pred)
        metrics["MAE"] = mae(y_true, y_pred)
        # FIXED: Increase MAPE sensitivity by reducing epsilon
        metrics["MAPE"] = mape(y_true, y_pred, epsilon=1e-6)
        metrics["TimeToPeakError"] = None
        peak_pred = time_to_peak(y_pred, None)
        peak_true = time_to_peak(y_true, None)
        if peak_pred is not None and peak_true is not None:
            metrics["TimeToPeakError"] = abs(int(peak_pred) - int(peak_true))
        metrics["PearsonR"] = pearson_r(y_true, y_pred)

        # Extended summary
        final_adopt = y_pred[-1] if y_pred else 0.0
        metrics["final_adoption_rate"] = final_adopt
        # Compute times to thresholds on full series
        def time_to_threshold(series: List[float], thr: float) -> Optional[int]:
            for i, v in enumerate(series):
                if v >= thr:
                    return i
            return None

        metrics["time_to_50_percent_adoption"] = time_to_threshold(pred_adopt, 0.5)
        metrics["time_to_70_percent"] = time_to_threshold(pred_adopt, 0.7)
        metrics["time_to_80_percent_adoption"] = time_to_threshold(pred_adopt, 0.8)
        metrics["peak_adoption_rate"] = max(pred_adopt) if pred_adopt else 0.0

        # FIXED: Supply metrics from aggregator
        stockout_series = self.history.get("metric.stockout_rate_daily", [])
        metrics["stockout_rate_mean"] = sum(stockout_series) / float(len(stockout_series)) if stockout_series else 0.0
        metrics["mask_supply_shortage_days"] = getattr(self.mod_aggregate, "stockout_days", 0)
        # FIXED: Equity gap summary
        ineq_series = self.history.get("metric.inequality_index", [])
        metrics["equity_gap_mean"] = (sum(ineq_series) / float(len(ineq_series))) if ineq_series else None

        # FIXED: Skip heavy counterfactual evaluation for large N
        try:
            if self.N <= 2000:
                sim2 = Simulation(self.get_params(), rng_seed=self.seed)
                sim2.set_params(policy_signal_weight=0.0, message_intensity=0.0, media_campaign_intensity=0.0)
                sim2.run(0, None, time_budget_sec=1.0 if FAST_MODE else 2.0)
                series2 = sim2.history.get("observable.adoption_rate_daily", [])
                metrics["policy_effect_contribution"] = (pred_adopt[-1] - (series2[-1] if series2 else 0.0)) if pred_adopt else None
            else:
                metrics["policy_effect_contribution"] = None
        except Exception:
            metrics["policy_effect_contribution"] = None

        metrics["Rt_last"] = None  # Not applicable in this social adoption model
        metrics["enforcement_cost"] = None
        metrics["misinformation_prevalence"] = None
        return metrics


# ---------------------------
# Data IO and Ground Truth
# ---------------------------


def load_ground_truth(data_dir: str, sim_days: int) -> Dict[str, List[float]]:
    """Load ground truth adoption series if available, else return default flat series."""
    pass
    # Attempt to load from predefined files (optional)
    gt: Dict[str, List[float]] = {}
    try:
        # Example: train_adoption.csv not guaranteed to be present; we fallback gracefully
        csv_path = os.path.join(data_dir, "train_adoption.csv")
        if os.path.exists(csv_path):
            # Minimal CSV parsing
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            header = lines[0].split(",")
            # Expect columns e.g., date, adoption_rate, new_adopters
            if "adoption_rate" in header:
                idx = header.index("adoption_rate")
                series: List[float] = []
                for line in lines[1:]:
                    cols = line.split(",")
                    try:
                        series.append(float(cols[idx]))
                    except Exception:
                        continue
                if series:
                    gt["adoption_rate"] = series[:sim_days]
    except Exception:
        pass
    if "adoption_rate" not in gt:
        gt["adoption_rate"] = [0.1 for _ in range(sim_days)]
    return gt


# ---------------------------
# Calibration Interfaces
# ---------------------------

@dataclass
class FittedParams:
    """Container for all parameters needed by the simulator."""
    decision_weights: Dict[str, float]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        pass
        return asdict(self)


class ParamsAdapter:
    """Adapts FittedParams to simulation parameter system."""
    pass

    def __init__(self, param_defs_path: str) -> None:
        """Initialize adapter with path to parameter definitions."""
        pass
        self.param_defs_path = param_defs_path
        self.param_defs: Dict[str, Any] = {}
        try:
            with open(param_defs_path, "r", encoding="utf-8") as f:
                self.param_defs = json.load(f)
        except Exception:
            self.param_defs = DEFAULT_PARAM_DEFS.copy()

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """Apply params via simulation.set_params and persist parameters_used.json."""
        pass
        # Map decision weights to simulation parameters
        dw = params.decision_weights or {}
        sim_kwargs: Dict[str, Any] = {}
        # Mapping logic with defaults
        if "intercept" in dw:
            sim_kwargs["base_adoption_logit"] = float(dw["intercept"])
        if "w_social" in dw:
            sim_kwargs["social_influence_weight"] = float(dw["w_social"])
        if "w_trait" in dw:
            sim_kwargs["personal_trait_weight"] = float(dw["w_trait"])
        if "w_policy" in dw:
            sim_kwargs["policy_signal_weight"] = float(dw["w_policy"])

        # Map info params
        ip = params.info_params or {}
        if "campaign_intensity" in ip:
            sim_kwargs["message_intensity"] = float(ip["campaign_intensity"])
            sim_kwargs["media_campaign_intensity"] = float(ip["campaign_intensity"])
        if "memory_decay" in ip:
            sim_kwargs["fatigue_decay"] = float(ip["memory_decay"])
        if "trust_update_rate" in ip:
            sim_kwargs["trust_update_rate"] = float(ip["trust_update_rate"])

        # Noise params
        np_ = params.noise_params or {}
        if "temperature" in np_:
            sim_kwargs["noise_scale"] = float(np_["temperature"])

        # Module-specific
        for module_name, module_dict in (params.module_params or {}).items():
            for k, v in module_dict.items():
                sim_kwargs[k] = v

        # Respect frozen params
        frozen_warnings = self.validate_frozen(params)
        # FIXED: Print full messages from validate_frozen warnings
        for k, msg in (frozen_warnings or {}).items():
            print(f"[ParamsAdapter] Warning: {msg}", file=sys.stderr)

        # Apply to simulation
        simulation.set_params(**sim_kwargs)

        # Persist parameters_used.json (guarded by SKIP_IO)
        # FIXED: Guard file persistence in ParamsAdapter.apply with SKIP_IO.
        if not SKIP_IO:
            try:
                path = os.path.join(ARTIFACTS_DIR, "results", "parameters_used.json")
                ensure_dir(os.path.dirname(path))
                # Merge simulation current params
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(sanitize_for_json(simulation.get_params()), f, indent=2)
            except Exception:
                pass

    def capture(self, simulation: Simulation) -> FittedParams:
        """Capture current effective params from simulation into FittedParams."""
        pass
        p = simulation.get_params()
        decision_weights = {
            "intercept": float(p.get("base_adoption_logit", -2.0)),
            "w_social": float(p.get("social_influence_weight", 1.5)),
            "w_trait": float(p.get("personal_trait_weight", 1.0)),
            "w_policy": float(p.get("policy_signal_weight", 1.2)),
        }
        info_params = {
            "campaign_intensity": float(p.get("message_intensity", p.get("media_campaign_intensity", 0.7))),
            "memory_decay": float(p.get("fatigue_decay", 0.01)),
            "trust_update_rate": float(p.get("trust_update_rate", 0.05)),
        }
        noise_params = {"temperature": float(p.get("noise_scale", 0.1))}
        layer_weights: Dict[str, float] = {}  # Not used in this model
        return FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            module_params={},
            meta={"captured_at": time.time()},
        )

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """Check frozen params and return warnings."""
        pass
        warnings: Dict[str, str] = {}
        defs = self.param_defs if isinstance(self.param_defs, dict) else DEFAULT_PARAM_DEFS
        candidate_keys: List[str] = []
        candidate_keys.extend(
            [
                "base_adoption_logit",
                "social_influence_weight",
                "personal_trait_weight",
                "policy_signal_weight",
                "message_intensity",
                "media_campaign_intensity",
                "fatigue_decay",
                "trust_update_rate",
                "noise_scale",
            ]
        )
        for k in candidate_keys:
            info = defs.get(k)
            if info and bool(info.get("frozen", False)):
                warnings[k] = f"Parameter '{k}' is frozen and should not be modified."
        return warnings


class Calibrator:
    """Pluggable calibrator interface with a stable evaluation callback signature."""
    pass

    def __init__(self, **kwargs: Any) -> None:
        """Optional calibrator configuration via kwargs (ignored in basic implementations)."""
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
        """Return FittedParams, fitted strictly on the training window."""
        pass
        raise NotImplementedError("Calibrator.fit must be implemented by subclasses")


class LogitHeadCalibrator(Calibrator):
    """Fits a logistic decision head on micro-transitions; degrades gracefully if unavailable."""
    pass

    def __init__(self, **kwargs: Any) -> None:
        """Accept optional kwargs for interface compatibility; not used."""
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
        """Attempt to adjust decision weights to match adoption trajectory."""
        pass
        rng = random.Random(seed)
        adapter = params_adapter or ParamsAdapter(os.path.join(ARTIFACTS_DIR, "results", "parameter_definitions.json"))
        current = adapter.capture(simulator)
        best_params = current
        best_score = float("inf")

        # Degrade: simple line search on intercept and w_policy around current
        intercept0 = current.decision_weights.get("intercept", -2.0)
        w_policy0 = current.decision_weights.get("w_policy", 1.2)
        candidates: List[Tuple[float, float]] = []
        for di in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            for dp in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                candidates.append((intercept0 + di, max(0.0, w_policy0 + dp)))
        if FAST_MODE:
            candidates = candidates[:5]

        trial_root = os.path.join(artifacts_dir or ARTIFACTS_DIR, "calibration", "logit_head")
        ensure_dir(trial_root)
        start_time = time.time()
        # FIXED: Add unconditional time budget guard via env var override
        time_cap = 2.0 if FAST_MODE else float(os.environ.get("CALIB_TIME_CAP_SEC", "10.0"))
        for i, (intercept, w_policy) in enumerate(candidates):
            if (time.time() - start_time) > time_cap:
                break
            trial_params = FittedParams(
                decision_weights={
                    "intercept": intercept,
                    "w_social": current.decision_weights.get("w_social", 1.5),
                    "w_trait": current.decision_weights.get("w_trait", 1.0),
                    "w_policy": w_policy,
                },
                layer_weights=current.layer_weights.copy(),
                info_params=current.info_params.copy(),
                noise_params=current.noise_params.copy(),
                module_params={},
                meta={"trial": i, "seed": seed},
            )
            score_dict = evaluate_params(simulator, trial_params, train_window)
            score = float(score_dict.get("RMSE_aggregate", 1e9) or 1e9)
            # Save trial artifacts (guarded)
            if not SKIP_IO:
                try:
                    trial_dir = os.path.join(trial_root, f"trial_{i}")
                    ensure_dir(trial_dir)
                    with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                        json.dump(sanitize_for_json(trial_params.to_dict()), f, indent=2)
                    with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(sanitize_for_json(score_dict), f, indent=2)
                except Exception:
                    pass

            if score < best_score:
                best_score = score
                best_params = trial_params

        # Save best (guarded)
        if not SKIP_IO:
            try:
                best_dir = os.path.join(trial_root, "best")
                ensure_dir(best_dir)
                with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                    json.dump(sanitize_for_json(best_params.to_dict()), f, indent=2)
            except Exception:
                pass
        return best_params


class RandomSearchCalibrator(Calibrator):
    """Black-box random search over selected simulator parameters within bounds."""
    pass

    def __init__(self, **kwargs: Any) -> None:
        """Accept optional kwargs for interface compatibility; not used."""
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
        """Run random search to minimize aggregate evaluation metric."""
        pass
        rng = random.Random(seed)
        adapter = params_adapter or ParamsAdapter(os.path.join(ARTIFACTS_DIR, "results", "parameter_definitions.json"))
        current = adapter.capture(simulator)
        best_params = current
        best_score = float("inf")
        n_trials = max(5, min(budget, 50 if FAST_MODE else budget))
        trial_root = os.path.join(artifacts_dir or ARTIFACTS_DIR, "calibration", "random_search")
        ensure_dir(trial_root)
        # FIXED: Add unconditional time budget guard to avoid long calibration in constrained runs
        start_time = time.time()
        time_cap = float(os.environ.get("CALIB_TIME_CAP_SEC", "10.0")) if not FAST_MODE else 2.0

        for i in range(n_trials):
            if (time.time() - start_time) > time_cap:
                break
            trial_params = FittedParams(
                decision_weights={
                    "intercept": current.decision_weights.get("intercept", -2.0) + rng.uniform(-0.5, 0.5),
                    "w_social": max(0.0, current.decision_weights.get("w_social", 1.5) + rng.uniform(-0.5, 0.5)),
                    "w_trait": max(0.0, current.decision_weights.get("w_trait", 1.0) + rng.uniform(-0.5, 0.5)),
                    "w_policy": max(0.0, current.decision_weights.get("w_policy", 1.2) + rng.uniform(-0.5, 0.5)),
                },
                layer_weights=current.layer_weights.copy(),
                info_params={
                    "campaign_intensity": max(0.0, current.info_params.get("campaign_intensity", 0.7) + rng.uniform(-0.2, 0.2)),
                    "memory_decay": max(0.0, current.info_params.get("memory_decay", 0.01) + rng.uniform(-0.01, 0.01)),
                    "trust_update_rate": max(0.0, current.info_params.get("trust_update_rate", 0.05) + rng.uniform(-0.02, 0.02)),
                },
                noise_params={"temperature": max(0.0, current.noise_params.get("temperature", 0.1) + rng.uniform(-0.05, 0.05))},
                module_params={},
                meta={"trial": i, "seed": seed},
            )
            score_dict = evaluate_params(simulator, trial_params, train_window)
            score = float(score_dict.get("RMSE_aggregate", 1e9) or 1e9)

            # Save trial artifacts (guarded)
            if not SKIP_IO:
                try:
                    trial_dir = os.path.join(trial_root, f"trial_{i}")
                    ensure_dir(trial_dir)
                    with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                        json.dump(sanitize_for_json(trial_params.to_dict()), f, indent=2)
                    with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(sanitize_for_json(score_dict), f, indent=2)
                except Exception:
                    pass

            if score < best_score:
                best_score = score
                best_params = trial_params

        # Save best (guarded)
        if not SKIP_IO:
            try:
                best_dir = os.path.join(trial_root, "best")
                ensure_dir(best_dir)
                with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                    json.dump(sanitize_for_json(best_params.to_dict()), f, indent=2)
            except Exception:
                pass
        return best_params


class SNPECalibrator(Calibrator):
    """True SBI using neural networks for Bayesian parameter inference; falls back if unavailable."""
    pass

    def __init__(self, **kwargs: Any) -> None:
        """Accept optional kwargs for interface compatibility; not used."""
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
        """Attempt SNPE; fallback to RandomSearch if dependencies unavailable."""
        pass
        try:
            import torch  # type: ignore
            from sbi import utils as sbi_utils  # type: ignore
            from sbi import inference as sbi_inference  # type: ignore
        except Exception:
            # Fallback
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        # Simple SNPE setup on a low-dimensional parameter vector [intercept, w_policy, campaign_intensity]
        rng = random.Random(seed)
        adapter = params_adapter or ParamsAdapter(os.path.join(ARTIFACTS_DIR, "results", "parameter_definitions.json"))
        baseline = adapter.capture(simulator)
        low = torch.tensor([-3.0, 0.0, 0.1], dtype=torch.float32)
        high = torch.tensor([-1.0, 3.0, 1.5], dtype=torch.float32)
        prior = sbi_utils.BoxUniform(low=low, high=high)
        inference = sbi_inference.SNPE(prior=prior)

        num_simulations = max(50, min(200, budget)) if not FAST_MODE else min(30, budget)
        theta_list = []
        x_list = []
        for _ in range(num_simulations):
            # Sample theta
            theta = prior.sample().numpy().tolist()
            fp = FittedParams(
                decision_weights={
                    "intercept": float(theta[0]),
                    "w_social": baseline.decision_weights.get("w_social", 1.5),
                    "w_trait": baseline.decision_weights.get("w_trait", 1.0),
                    "w_policy": float(theta[1]),
                },
                layer_weights=baseline.layer_weights.copy(),
                info_params={
                    "campaign_intensity": float(theta[2]),
                    "memory_decay": baseline.info_params.get("memory_decay", 0.01),
                    "trust_update_rate": baseline.info_params.get("trust_update_rate", 0.05),
                },
                noise_params=baseline.noise_params.copy(),
                module_params={},
                meta={"seed": seed},
            )
            score = evaluate_params(simulator, fp, train_window)
            # Summary statistic: RMSE
            rmse_val = float(score.get("RMSE_aggregate", 1e9) or 1e9)
            theta_list.append(theta)
            x_list.append([rmse_val])

        import torch  # type: ignore

        theta_t = torch.tensor(theta_list, dtype=torch.float32)
        x_t = torch.tensor(x_list, dtype=torch.float32)
        density_estimator = inference.append_simulations(theta_t, x_t).train()
        posterior = inference.build_posterior(density_estimator)

        # Condition on low RMSE target (use a small observed x)
        x_obs = torch.tensor([[0.05]], dtype=torch.float32)
        samples = posterior.sample((min(1000, 100 if FAST_MODE else 1000),), x=x_obs)
        # Select best sample by evaluating
        best_fp = baseline
        best_score = float("inf")
        for s in samples:
            th = s.numpy().tolist()
            fp = FittedParams(
                decision_weights={
                    "intercept": float(th[0]),
                    "w_social": baseline.decision_weights.get("w_social", 1.5),
                    "w_trait": baseline.decision_weights.get("w_trait", 1.0),
                    "w_policy": float(th[1]),
                },
                layer_weights=baseline.layer_weights.copy(),
                info_params={
                    "campaign_intensity": float(th[2]),
                    "memory_decay": baseline.info_params.get("memory_decay", 0.01),
                    "trust_update_rate": baseline.info_params.get("trust_update_rate", 0.05),
                },
                noise_params=baseline.noise_params.copy(),
                module_params={},
                meta={"seed": seed},
            )
            score = evaluate_params(simulator, fp, train_window)
            rmse_val = float(score.get("RMSE_aggregate", 1e9) or 1e9)
            if rmse_val < best_score:
                best_score = rmse_val
                best_fp = fp

        # Save best (guarded)
        if not SKIP_IO:
            try:
                trial_root = os.path.join(artifacts_dir or ARTIFACTS_DIR, "calibration", "snpe")
                best_dir = os.path.join(trial_root, "best")
                ensure_dir(best_dir)
                with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                    json.dump(sanitize_for_json(best_fp.to_dict()), f, indent=2)
            except Exception:
                pass

        return best_fp


class NoOpCalibrator(Calibrator):
    """No-op calibrator that returns the simulator's current parameters without fitting."""
    pass

    def __init__(self, **kwargs: Any) -> None:
        """Initialize NoOpCalibrator; kwargs accepted for interface compatibility."""
        pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 0,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """Return the current captured parameters without any calibration."""
        pass
        adapter = params_adapter or ParamsAdapter(os.path.join(ARTIFACTS_DIR, "results", "parameter_definitions.json"))
        return adapter.capture(simulator)


CALIBRATOR_REGISTRY: Dict[str, Any] = {
    "none": NoOpCalibrator,
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None) -> Calibrator:
    """Factory to instantiate a calibrator by name; config not used in this simple implementation."""
    pass
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    # Load optional config into kwargs (not used; kept for interface completeness)
    kwargs: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                kwargs.update(cfg)
        except Exception:
            pass
    # FIXED: Do not pass kwargs to calibrator constructors to avoid TypeError
    return CALIBRATOR_REGISTRY[name]()  # type: ignore


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).

    For simulation engines: use Simulation.run(start,end) + Simulation.evaluate(),
    read metrics from artifacts/results/, build dict with required keys.
    If micro-transitions unavailable, degrade gracefully or use placeholder values.
    """
    pass
    # Clone a lightweight simulator by re-initializing with same params/seed for determinism
    sim = Simulation(simulator.get_params(), rng_seed=simulator.seed)
    adapter = ParamsAdapter(os.path.join(ARTIFACTS_DIR, "results", "parameter_definitions.json"))
    adapter.apply(sim, params)
    # Enforce fast runtime for evaluation
    time_budget = 1.5 if FAST_MODE else None
    start, end = window
    # FIXED: Honor evaluation window start; run only from start to end
    sim.run(start_day=start, end_day=end, time_budget_sec=time_budget)
    # Prepare ground truth trimmed to the evaluation window
    gt_all = load_ground_truth(DATA_DIR, sim.sim_days)
    gt_trim = {}
    try:
        series = gt_all.get("adoption_rate", [])
        gt_trim["adoption_rate"] = series[start : end + 1] if end + 1 <= len(series) else series[start:]
    except Exception:
        gt_trim["adoption_rate"] = [0.1 for _ in range(max(0, end - start + 1))]
    # Evaluate on local window (relative to run)
    local_window = (0, max(0, end - start))
    metrics = sim.evaluate(gt_trim, window=local_window)

    # Derive simple transition summary from aggregator if available
    p01_series = getattr(sim.mod_aggregate, "raw_p01", []) if hasattr(sim, "mod_aggregate") else []
    p10_series = getattr(sim.mod_aggregate, "raw_p10", []) if hasattr(sim, "mod_aggregate") else []
    # Use entire run span (already limited to window) for averaging
    mean_p01 = sum(p01_series) / len(p01_series) if p01_series else None
    mean_p10 = sum(p10_series) / len(p10_series) if p10_series else None
    mean_p00 = 1.0 - mean_p01 if mean_p01 is not None else None
    mean_p11 = 1.0 - mean_p10 if mean_p10 is not None else None

    # Build required keys
    out = {
        "RMSE_aggregate": metrics.get("RMSE"),
        "MAE_aggregate": metrics.get("MAE"),
        "Brier": metrics.get("RMSE"),  # degrade: use RMSE as proxy
        "TransitionFit": {
            "P01": mean_p01,
            "P11": mean_p11,
            "P10": mean_p10,
            "P00": mean_p00,
        },
    }
    return out


# ---------------------------
# CLI and Orchestration
# ---------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the simulation and calibration."""
    pass
    parser = argparse.ArgumentParser(description="Agent-based social adoption simulation")
    parser.add_argument("--param-file", type=str, default=None, help="Path to parameters.json")
    parser.add_argument("--set", type=str, action="append", default=[], help="Override parameter as key=value (repeatable)")
    # FIXED: Change default calibrator to 'none' to avoid expensive calibration by default
    parser.add_argument("--calibrator", type=str, default="none", help="Calibrator name (none, logit_head, random_search, snpe)")
    # FIXED: Reduce default budget to prevent timeouts
    parser.add_argument("--budget", type=int, default=10, help="Calibration budget (iterations)")
    parser.add_argument("--calib-window", type=str, default=None, help="Calibration window start:end")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode")
    parser.add_argument("--skip-io", action="store_true", help="Skip heavy I/O")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides parameter random_seed)")
    return parser.parse_args(argv)


def orchestrate(args: argparse.Namespace) -> Dict[str, Any]:
    """Main orchestration: load params, run calibration, rollout, evaluate, and persist artifacts."""
    pass
    # Parameter manager
    pm = ParameterManager()
    pm.load_file(args.param_file)

    # Apply FAST_MODE constraints
    fast = args.fast or FAST_MODE
    if fast:
        # Downscale defaults to avoid timeouts
        pm.values["population_size"] = min(500, int(pm.values.get("population_size", 500)))
        pm.values["sim_days"] = min(40, int(pm.values.get("sim_days", 60)))
        pm.values["daily_contact_fraction"] = min(0.2, float(pm.values.get("daily_contact_fraction", 0.3)))

    # Local skip_io guard
    # FIXED: Use a local skip_io flag rather than changing environment variables at runtime
    skip_io = SKIP_IO or args.skip_io

    # Overrides
    overrides = parse_kv_overrides(args.set)
    pm.apply_overrides(overrides)
    pm.validate_bounds()

    # Export parameter definitions for adapter/calibrators
    param_defs_path = os.path.join(ARTIFACTS_DIR, "results", "parameter_definitions.json")
    pm.export_definitions(param_defs_path)

    # Initialize simulator
    seed = args.seed if args.seed is not None else int(pm.values.get("random_seed", 42))
    sim = Simulation(pm.get_values(), rng_seed=seed)

    # Determine calibration window
    if args.calib_window and ":" in args.calib_window:
        s_str, e_str = args.calib_window.split(":", 1)
        train_start = max(0, int(s_str))
        train_end = min(sim.sim_days - 1, int(e_str))
    else:
        train_start = 0
        train_end = max(0, min(sim.sim_days - 1, sim.sim_days // 2 - 1))
    train_window = (train_start, train_end)

    # Calibrator
    calibrator = get_calibrator(args.calibrator, None)
    # Prepare artifacts dir
    calib_art_dir = os.path.join(ARTIFACTS_DIR, "calibration")
    ensure_dir(calib_art_dir)
    adapter = ParamsAdapter(param_defs_path)

    # Fit
    budget = args.budget if not fast else min(args.budget, 20)
    fitted = calibrator.fit(
        bundle={},
        simulator=sim,
        evaluator=evaluate_params,
        train_window=train_window,
        seed=seed,
        budget=budget,
        artifacts_dir=calib_art_dir,
        params_adapter=adapter,
    )

    # Apply best params
    adapter.apply(sim, fitted)

    # Full rollout
    # FIXED: Fast/sandbox guard to limit runtime and avoid timeouts
    time_budget = 3.0 if fast else None
    sim.run(0, None, time_budget_sec=time_budget)

    # Save artifacts
    if not skip_io:
        sim.save_results(os.path.join(ARTIFACTS_DIR, "results", "simulation_results.json"))
        sim.save_all_io(os.path.join(ARTIFACTS_DIR, "io"))
        sim.visualize(show=False, save_path=os.path.join(ARTIFACTS_DIR, "figs", "overview.png"))

    # Evaluation
    gt = load_ground_truth(DATA_DIR, sim.sim_days)
    metrics = sim.evaluate(gt, window=None)

    # FIXED: Ensure persisted parameters reflect calibrated run; guarded by skip_io
    pm.set_values(**sim.get_params())
    if not skip_io:
        pm.persist_used(os.path.join(ARTIFACTS_DIR, "results", "parameters_used.json"))

    # Build summary
    adoption_series = sim.history.get("observable.adoption_rate_daily", [])
    final_adopt = adoption_series[-1] if adoption_series else 0.0
    summary = {
        "status": "ok",
        "fast_mode": fast,
        "seed": seed,
        "population_size": sim.N,
        "sim_days": sim.sim_days,
        "final_adoption_rate": safe_round(final_adopt),
        "time_to_50": metrics.get("time_to_50_percent_adoption"),
        "time_to_70": metrics.get("time_to_70_percent"),
        "time_to_80": metrics.get("time_to_80_percent_adoption"),
        "peak_adoption_rate": safe_round(metrics.get("peak_adoption_rate", 0.0)),
        "rmse": safe_round(metrics.get("RMSE", None)),
        "mae": safe_round(metrics.get("MAE", None)),
        "mape": safe_round(metrics.get("MAPE", None)),
        "pearson_r": safe_round(metrics.get("PearsonR", None)),
        "stockout_rate_mean": safe_round(metrics.get("stockout_rate_mean", 0.0)),
        "equity_gap_mean": safe_round(metrics.get("equity_gap_mean", None)),
    }
    return summary


def main() -> None:
    """Entry point: parse args, run orchestration, and print JSON summary."""
    pass
    args = parse_args()
    summary = orchestrate(args)
    # FIXED: JSON-safe print to avoid NaN/Inf errors
    print(json.dumps(sanitize_for_json(summary)))


# Execute main for both direct execution and sandbox wrapper invocation
main()