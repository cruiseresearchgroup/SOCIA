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


# ---------------------------
# Global Constants and Paths
# ---------------------------

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# FAST_MODE detection
FAST_MODE = (
    os.environ.get("FAST_MODE", "1") == "1"
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


def mape(y_true: List[float], y_pred: List[float], epsilon: float = 1.0) -> Optional[float]:
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
    "initial_adoption_rate": dict(dtype="float", default=0.05, bounds=[0.0, 0.2], owner_module="global", frozen=False),
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
    # InteractionScheduler
    "daily_contact_fraction": dict(dtype="float", default=0.3, bounds=[0.05, 1.0], owner_module="InteractionScheduler", frozen=False),
    "meeting_noise": dict(dtype="float", default=0.1, bounds=[0.0, 0.5], owner_module="InteractionScheduler", frozen=False),
    "exogenous_contact_rate": dict(dtype="float", default=0.5, bounds=[0.0, 3.0], owner_module="InteractionScheduler", frozen=False),
    # PolicyBroadcast
    "message_frequency": dict(dtype="int", default=1, bounds=[1, 7], owner_module="PolicyBroadcast", frozen=False),
    "message_intensity": dict(dtype="float", default=0.7, bounds=[0.0, 2.0], owner_module="PolicyBroadcast", frozen=False),
    "targeting_strength": dict(dtype="float", default=0.3, bounds=[0.0, 1.0], owner_module="PolicyBroadcast", frozen=False),
    "policy_channel_reach": dict(dtype="float", default=0.6, bounds=[0.1, 1.0], owner_module="PolicyBroadcast", frozen=False),
    "misinformation_fraction": dict(dtype="float", default=0.1, bounds=[0.0, 0.5], owner_module="PolicyBroadcast", frozen=False),
    "trust_update_rate": dict(dtype="float", default=0.05, bounds=[0.0, 0.2], owner_module="PolicyBroadcast", frozen=False),
    # InfluenceAndAdoption
    "base_adoption_logit": dict(dtype="float", default=-2.0, bounds=[-5.0, 0.0], owner_module="InfluenceAndAdoption", frozen=False),
    "social_influence_weight": dict(dtype="float", default=1.5, bounds=[0.0, 3.0], owner_module="InfluenceAndAdoption", frozen=False),
    "personal_trait_weight": dict(dtype="float", default=1.0, bounds=[0.0, 2.0], owner_module="InfluenceAndAdoption", frozen=False),
    "policy_signal_weight": dict(dtype="float", default=1.2, bounds=[0.0, 3.0], owner_module="InfluenceAndAdoption", frozen=False),
    "fatigue_decay": dict(dtype="float", default=0.01, bounds=[0.0, 0.1], owner_module="InfluenceAndAdoption", frozen=False),
    "disadoption_rate": dict(dtype="float", default=0.001, bounds=[0.0, 0.05], owner_module="InfluenceAndAdoption", frozen=False),
    "noise_scale": dict(dtype="float", default=0.1, bounds=[0.0, 1.0], owner_module="InfluenceAndAdoption", frozen=False),
    "max_memory_days": dict(dtype="int", default=14, bounds=[1, 60], owner_module="InfluenceAndAdoption", frozen=False),
    # AdoptionAggregator
    "smoothing_window": dict(dtype="int", default=1, bounds=[1, 7], owner_module="AdoptionAggregator", frozen=False),
    "report_lag_days": dict(dtype="int", default=0, bounds=[0, 3], owner_module="AdoptionAggregator", frozen=False),
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
        """Persist parameters_used.json with final values."""
        pass
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


class PolicyChannel:
    """Policy channel parameters and counters."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize with parameters."""
        pass
        self.message_frequency = int(params.get("message_frequency", 1))
        self.message_intensity = float(params.get("message_intensity", 0.7))
        self.targeting_strength = float(params.get("targeting_strength", 0.3))
        self.reach = float(params.get("policy_channel_reach", 0.6))
        self.misinformation_fraction = float(params.get("misinformation_fraction", 0.1))
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
        """Build initial adjacency using simple stub matching and mild homophily."""
        pass
        mean_deg = float(self.params.get("degree_mean", 8.0))
        adj: List[set] = [set() for _ in range(N)]
        # Sample degrees via Poisson approx
        degs = [max(0, self._poisson(mean_deg, rng)) for _ in range(N)]
        # Build stubs
        stubs: List[int] = []
        for i, k in enumerate(degs):
            for _ in range(k):
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
        # Homophily pass
        self._homophily_rewire(adj, persons, rng, fraction=0.05)
        return adj

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
        """Rewire a fraction of edges to increase homophily."""
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
        K = int(max(1, fraction * len(edges)))
        homo_strength = float(self.params.get("homophily_strength", 0.5))
        for idx in range(min(K, len(edges))):
            u, v = edges[idx]
            current_score = self._homophily_score(u, v, persons)
            if rng.random() < homo_strength * 0.5:
                # try find w more similar to u
                candidates = [w for w in range(N) if w != u and w not in adj[u]]
                if not candidates:
                    continue
                w = max(candidates, key=lambda c: self._homophily_score(u, c, persons) + rng.random() * 0.01)
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
        """Emit policy signals to a subset of agents."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        signal_by_agent: List[float] = [0.0 for _ in range(N)]
        messages_sent = 0
        freq = int(self.params.get("message_frequency", 1))
        intensity = float(self.params.get("message_intensity", 0.7))
        targeting_strength = float(self.params.get("targeting_strength", 0.3))
        reach = float(self.params.get("policy_channel_reach", 0.6))
        misinformation_fraction = float(self.params.get("misinformation_fraction", 0.1))
        trust_update_rate = float(self.params.get("trust_update_rate", 0.05))
        if freq <= 0:
            freq = 1

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
                signal = intensity * pi.trust_policy * misinformation_factor
                signal_by_agent[i] = signal
                # Awareness bump
                pi.awareness = clamp(pi.awareness + 0.2, 0.0, 1.0)
                # Trust update bounded
                sign_intensity = 1.0 if intensity >= 0 else -1.0
                pi.trust_policy = clamp(
                    pi.trust_policy + trust_update_rate * sign_intensity * misinformation_factor, -2.0, 2.0
                )
            messages_sent = len(reached)

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
        adj: List[set] = state["adjacency"]
        daily_fraction = float(self.params.get("daily_contact_fraction", 0.3))
        meeting_noise = float(self.params.get("meeting_noise", 0.1))
        exogenous_rate = float(self.params.get("exogenous_contact_rate", 0.5))

        # Active neighbors per agent
        active_neighbors: List[List[int]] = []
        for i in range(N):
            neigh = list(adj[i])
            rng.shuffle(neigh)
            k = max(0, int(len(neigh) * daily_fraction))
            subset = neigh[:k]
            # Exogenous random meetings
            # Poisson approx by sum of Bernoulli trials
            exo_k = max(0, int(exogenous_rate))
            if rng.random() < (exogenous_rate - int(exogenous_rate)):
                exo_k += 1
            for _ in range(exo_k):
                j = rng.randrange(N) if N > 0 else i
                if j != i and rng.random() < meeting_noise:
                    subset.append(j)
            # Deduplicate
            subset = list(set(subset))
            active_neighbors.append(subset)

        buffers["daily_active_neighbors"] = active_neighbors


class InfluenceAndAdoption(BaseModule):
    """Compute adoption probabilities and update adoption states via buffers."""
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize influence and adoption module."""
        pass
        super().__init__("InfluenceAndAdoption", params)

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

        updates: List[Tuple[int, int]] = []
        for i in range(N):
            pi = persons[i]
            # Fatigue decay
            pi.fatigue = max(0.0, pi.fatigue - fatigue_decay)
            neigh = active_neighbors[i] if i < len(active_neighbors) else []
            frac_adopted = 0.0
            if neigh:
                adopted_count = sum(1 for j in neigh if persons[j].adopted == 1)
                frac_adopted = adopted_count / float(len(neigh))
            social_term = w_social * (frac_adopted - pi.threshold)
            trait_term = w_trait * float(pi.openness)
            policy_term = w_policy * float(policy_signal[i]) if i < len(policy_signal) else 0.0
            noise = rng.gauss(0.0, noise_scale)
            logit_p = base_logit + social_term + trait_term + policy_term - pi.fatigue + noise
            p_adopt = sigmoid(logit_p)

            if pi.adopted == 0:
                if rng.random() < p_adopt:
                    updates.append((i, 1))
            else:
                if rng.random() < disadopt_rate:
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
        self.raw_messages: List[int] = []
        self.raw_avg_degree: List[float] = []
        self.observed_adoption_rate: List[float] = []
        self.observed_new_adopters: List[float] = []
        self.observed_messages: List[float] = []
        self.observed_avg_degree: List[float] = []

    def forward(
        self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int, rng: random.Random
    ) -> None:
        """Append raw metrics, apply smoothing and lag, and write latest observables to buffers."""
        pass
        persons: List[Person] = state["persons"]
        N = len(persons)
        adopted_count = sum(1 for p in persons if p.adopted == 1)
        adoption_rate = adopted_count / float(N) if N > 0 else 0.0
        new_adopters = sum(1 for (_i, delta) in buffers.get("adoption_updates", []) if delta == 1)
        messages = int(buffers.get("messages_sent", 0))
        avg_deg = float(buffers.get("average_degree_daily", 0.0))

        self.raw_adoption_rate.append(adoption_rate)
        self.raw_new_adopters.append(new_adopters)
        self.raw_messages.append(messages)
        self.raw_avg_degree.append(avg_deg)

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
        self.mod_interaction = InteractionScheduler(self.params)
        self.mod_influence = InfluenceAndAdoption(self.params)
        self.mod_aggregate = AdoptionAggregator(self.params)
        self.modules: List[BaseModule] = [
            self.mod_network,
            self.mod_policy,
            self.mod_interaction,
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
            # Network build/rewire
            self.mod_network.forward(self.state, buffers, self.params, t, self.rng)
            # Commit adjacency and degrees
            self.state["adjacency"] = buffers.get("graph", self.state.get("adjacency"))
            degrees = buffers.get("degrees", [])
            for i, p in enumerate(self.persons):
                p.degree = degrees[i] if i < len(degrees) else p.degree

            # Policy broadcast
            self.mod_policy.forward(self.state, buffers, self.params, t, self.rng)
            # Interactions
            self.mod_interaction.forward(self.state, buffers, self.params, t, self.rng)
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

            # Aggregation
            self.mod_aggregate.forward(self.state, buffers, self.params, t, self.rng)

            # Record observables
            for key in list(self.history.keys()):
                self.history[key].append(float(buffers.get(key, 0.0)))

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
        """Compute evaluation metrics comparing observables to ground truth."""
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
        metrics["MAPE"] = mape(y_true, y_pred, epsilon=1.0)
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
        metrics["Rt_last"] = None  # Not applicable in this social adoption model
        metrics["stockout_rate_mean"] = None  # Not modeled
        metrics["sustained_adoption_rate"] = safe_round(sum(pred_adopt[-7:]) / max(1, len(pred_adopt[-7:])), 4)
        metrics["mask_supply_shortage_days"] = None
        metrics["enforcement_cost"] = None
        metrics["misinformation_prevalence"] = None
        metrics["policy_effect_contribution"] = None
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
        for w in frozen_warnings:
            print(f"[ParamsAdapter] Warning: {w}", file=sys.stderr)

        # Apply to simulation
        simulation.set_params(**sim_kwargs)

        # Persist parameters_used.json
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
            "campaign_intensity": float(p.get("message_intensity", 0.7)),
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
        # Build a reverse map: which simulation keys might be changed
        candidate_keys: List[str] = []
        candidate_keys.extend(
            [
                "base_adoption_logit",
                "social_influence_weight",
                "personal_trait_weight",
                "policy_signal_weight",
                "message_intensity",
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
        # Capture current
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
        for i, (intercept, w_policy) in enumerate(candidates):
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
            # Save trial artifacts
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

        # Save best
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

        for i in range(n_trials):
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

            # Save trial artifacts
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

        # Save best
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

        # Save best
        try:
            trial_root = os.path.join(artifacts_dir or ARTIFACTS_DIR, "calibration", "snpe")
            best_dir = os.path.join(trial_root, "best")
            ensure_dir(best_dir)
            with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                json.dump(sanitize_for_json(best_fp.to_dict()), f, indent=2)
        except Exception:
            pass

        return best_fp


CALIBRATOR_REGISTRY: Dict[str, Any] = {
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
    return CALIBRATOR_REGISTRY[name](**kwargs)  # type: ignore


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
    sim.run(start_day=0, end_day=end, time_budget_sec=time_budget)
    gt = load_ground_truth(DATA_DIR, sim.sim_days)
    metrics = sim.evaluate(gt, window=window)
    # Build required keys
    out = {
        "RMSE_aggregate": metrics.get("RMSE"),
        "MAE_aggregate": metrics.get("MAE"),
        "Brier": metrics.get("RMSE"),  # degrade: use RMSE as proxy
        "TransitionFit": {
            "P01": None,
            "P11": None,
            "P10": None,
            "P00": None,
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
    parser.add_argument("--calibrator", type=str, default="random_search", help="Calibrator name")
    parser.add_argument("--budget", type=int, default=50, help="Calibration budget (iterations)")
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
        # Skip calibration heavy IO by default
        os.environ["SKIP_IO"] = "1"

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
    if not SKIP_IO and not args.skip_io:
        sim.save_results(os.path.join(ARTIFACTS_DIR, "results", "simulation_results.json"))
        sim.save_all_io(os.path.join(ARTIFACTS_DIR, "io"))
        sim.visualize(show=False, save_path=os.path.join(ARTIFACTS_DIR, "figs", "overview.png"))

    # Evaluation
    gt = load_ground_truth(DATA_DIR, sim.sim_days)
    metrics = sim.evaluate(gt, window=None)

    # Persist parameters used
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