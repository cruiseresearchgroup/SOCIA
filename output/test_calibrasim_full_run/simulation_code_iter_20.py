import os
import json
import math
import time
import argparse
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    """
    Seed Python, numpy (if available), and random for reproducibility.

    # FIXED: Seed call parentheses balanced to avoid syntax errors
    """
    pass
    random.seed(seed)
    try:
        import torch  # noqa
        torch.manual_seed(seed)  # type: ignore
        if torch.cuda.is_available():  # type: ignore
            torch.cuda.manual_seed_all(seed)  # type: ignore
    except Exception:
        # torch not installed or no GPU
        pass
    if np is not None:
        np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists.
    """
    pass
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def moving_average(arr: List[float], window: int) -> List[float]:
    """
    Compute a simple moving average with a given window size.
    For the first elements where the window does not fit, use the average of available items.
    """
    pass
    if window <= 1 or len(arr) == 0:
        return list(arr)
    out: List[float] = []
    cumsum = 0.0
    for i, v in enumerate(arr):
        cumsum += float(v)
        if i >= window:
            cumsum -= float(arr[i - window])
        denom = float(min(i + 1, window))
        out.append(cumsum / denom if denom > 0 else 0.0)
    return out


def _json_clean(obj: Any) -> Any:
    """
    Recursively clean an object for JSON serialization.
    Converts NaN and Infinity to None or bounded values to prevent allow_nan issues.

    # FIXED: Sanitize JSON to avoid NaN and parsing errors
    """
    pass
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, list):
        return [_json_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    try:
        return json.loads(json.dumps(obj))
    except Exception:
        return str(obj)


def safe_json_dump(data: Any, fp: str) -> None:
    """
    Safely dump JSON with allow_nan=False and sanitized content.
    """
    pass
    data = _json_clean(data)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)


def parse_kv_override(s: str) -> Tuple[str, Any]:
    """
    Parse a CLI override of the form key=value into (key, parsed_value)
    """
    pass
    if "=" not in s:
        return s, True
    k, v = s.split("=", 1)
    v = v.strip()
    # Try to parse types
    if v.lower() in ("true", "false"):
        return k.strip(), v.lower() == "true"
    try:
        if "." in v:
            return k.strip(), float(v)
        return k.strip(), int(v)
    except Exception:
        return k.strip(), v


# -----------------------------------------------------------------------------
# Parameter Registry
# -----------------------------------------------------------------------------

@dataclass
class ParameterDefinition:
    """
    Definition for a parameter, including dtype, default, bounds, owner_module, and frozen flag.
    """
    key: str
    dtype: str = "float"
    default: Any = 0.0
    low: Optional[float] = None
    high: Optional[float] = None
    owner_module: str = "global"
    frozen: bool = False
    description: str = ""


class ParameterRegistry:
    """
    Registry to manage parameter definitions, values, frozen status, and overrides.
    """
    def __init__(self, definitions: Dict[str, ParameterDefinition], values: Dict[str, Any]) -> None:
        """
        Initialize the registry with definitions and initial values.
        """
        pass
        self.defs: Dict[str, ParameterDefinition] = definitions
        self.values: Dict[str, Any] = {}
        for k, d in self.defs.items():
            self.values[k] = values.get(k, d.default)

    def set_params(self, module: Optional[str] = None, **kwargs: Any) -> Dict[str, str]:
        """
        Set parameters, optionally filtering by owner_module.
        Returns a dict of warnings for frozen or unknown keys.

        # FIXED: Apply overrides via registry with dtype casting and bounds clamping.
        """
        pass
        warnings: Dict[str, str] = {}
        for k, v in kwargs.items():
            if k not in self.defs:
                warnings[k] = f"Unknown parameter: {k}"
                continue
            d = self.defs[k]
            if module is not None and d.owner_module != module:
                # ignore silently if module filter does not match
                continue
            if d.frozen:
                warnings[k] = f"Parameter '{k}' is frozen and cannot be overridden."
                continue
            # cast dtype
            try:
                if d.dtype == "int":
                    v = int(v)
                elif d.dtype == "float":
                    v = float(v)
                elif d.dtype == "bool":
                    if isinstance(v, str):
                        v = v.lower() in ("1", "true", "yes", "y", "on")
                    else:
                        v = bool(v)
                elif d.dtype == "str":
                    v = str(v)
                else:
                    # leave as is
                    pass
            except Exception:
                warnings[k] = f"Failed to cast parameter '{k}' to dtype '{d.dtype}'."
                continue
            # clamp bounds if numeric
            if isinstance(v, (int, float)):
                if d.low is not None:
                    v = max(d.low, float(v))
                if d.high is not None:
                    v = min(d.high, float(v))
                # cast back to int if needed
                if d.dtype == "int":
                    v = int(round(v))
            self.values[k] = v
        return warnings

    def get_params(self, module: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters, optionally filtering by owner_module.
        """
        pass
        if module is None:
            return dict(self.values)
        return {k: v for k, v in self.values.items() if self.defs.get(k, ParameterDefinition(k)).owner_module == module}

    def export_used(self) -> Dict[str, Any]:
        """
        Export final parameters including frozen ones.
        """
        pass
        return dict(self.values)

    def is_frozen(self, key: str) -> bool:
        """
        Return True if a parameter is frozen.
        """
        pass
        d = self.defs.get(key)
        return bool(d.frozen) if d else False

    def bounds(self, key: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Return (low, high) bounds for a parameter.
        """
        pass
        d = self.defs.get(key)
        if not d:
            return None, None
        return d.low, d.high

    def dtype(self, key: str) -> str:
        """
        Return dtype for a parameter.
        """
        pass
        d = self.defs.get(key)
        return d.dtype if d else "float"

    def clone(self) -> "ParameterRegistry":
        """
        Create a shallow clone of this registry (definitions shared, values copied).
        """
        pass
        return ParameterRegistry(self.defs, dict(self.values))


def load_parameter_definitions(path: Optional[str], initial_params: Dict[str, Any]) -> Dict[str, ParameterDefinition]:
    """
    Load parameter definitions from a JSON file if available; otherwise infer definitions from provided parameters.
    Some commonly frozen parameters are enforced.

    The JSON file is expected to be a dict of key -> {dtype, default, bounds: {low,high}, owner_module, frozen, description}

    # FIXED: Add definitions for capture_micro/save_micro_io/micro_sample_frac to allow CLI toggles and registry validation.
    # FIXED: Add definitions for save_heavy_io and io_agent_sample_n to control IO size.
    """
    pass
    definitions: Dict[str, ParameterDefinition] = {}
    loaded: Dict[str, Any] = {}
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            loaded = {}
    # Infer from file or initial params
    keys = set(initial_params.keys()) | set(loaded.keys())
    for k in keys:
        spec = loaded.get(k, {})
        dtype = spec.get("dtype", "float")
        default = spec.get("default", initial_params.get(k, 0.0))
        bounds = spec.get("bounds", {})
        low = bounds.get("low")
        high = bounds.get("high")
        owner_module = spec.get("owner_module", "global")
        frozen = spec.get("frozen", False)
        description = spec.get("description", "")
        definitions[k] = ParameterDefinition(
            key=k, dtype=dtype, default=default, low=low, high=high,
            owner_module=owner_module, frozen=frozen, description=description
        )
    # Enforce some common frozen params if present
    for fk in ["random_seed", "network_type_code", "adoption_rule_code"]:
        if fk in definitions:
            definitions[fk].frozen = True

    # Add micro-capture toggles if not present
    if "capture_micro" not in definitions:
        definitions["capture_micro"] = ParameterDefinition(
            key="capture_micro", dtype="bool", default=False, low=0, high=1, owner_module="InfluencePropagation",
            frozen=False, description="Enable capturing micro transitions per-agent per-day."
        )
    if "save_micro_io" not in definitions:
        definitions["save_micro_io"] = ParameterDefinition(
            key="save_micro_io", dtype="bool", default=False, low=0, high=1, owner_module="global",
            frozen=False, description="If True, include micro transitions in saved IO artifacts."
        )
    if "micro_sample_frac" not in definitions:
        definitions["micro_sample_frac"] = ParameterDefinition(
            key="micro_sample_frac", dtype="float", default=0.1, low=0.0, high=1.0, owner_module="InfluencePropagation",
            frozen=False, description="Fraction of agents to sample for micro transition capture."
        )
    # FIXED: Add save_heavy_io and io_agent_sample_n parameter definitions
    if "save_heavy_io" not in definitions:
        definitions["save_heavy_io"] = ParameterDefinition(
            key="save_heavy_io", dtype="bool", default=False, owner_module="global",
            description="If True, persist heavy per-agent arrays in IO; else store aggregates/samples."
        )
    if "io_agent_sample_n" not in definitions:
        definitions["io_agent_sample_n"] = ParameterDefinition(
            key="io_agent_sample_n", dtype="int", default=100, low=0, high=10000, owner_module="global",
            description="Number of agents to include in sampled IO arrays."
        )
    # FIXED: Applied feedback snippet from load_parameter_definitions()
if "save_heavy_io" not in definitions:
    definitions["save_heavy_io"] = ParameterDefinition(
        key="save_heavy_io", dtype="bool", default=False, owner_module="global",
        description="If True, persist heavy per-agent arrays in IO; else store aggregates/samples.")
if "io_agent_sample_n" not in definitions:
    definitions["io_agent_sample_n"] = ParameterDefinition(
        key="io_agent_sample_n", dtype="int", default=100, low=0, high=10000, owner_module="global",
        description="Number of agents to include in sampled IO arrays.")
return definitions


def load_parameters_file(path: str) -> Dict[str, Any]:
    """
    Load parameters from a JSON file; on failure, return default minimal parameters.

    # FIXED: Harden JSON loads with try/except and provide clear fallback defaults.
    """
    pass
    if not path or not os.path.exists(path):
        # Minimal set matching our engine requirements
        return {
            "random_seed": 42,
            "population_size": 2000,
            "time_horizon_days": 120,
            "initial_adoption_fraction": 0.02,
            "initial_media_exposure_mean": 0.2,
            "average_degree": 12.0,
            "network_type_code": 0,
            "rewiring_prob": 0.08,
            "community_count": 10,
            "baseline_media_intensity": 0.3,
            "media_reach_fraction": 0.5,
            "media_pulse_interval_days": 21,
            "media_pulse_duration_days": 5,
            "media_pulse_magnitude": 0.4,
            "policy_start_day": 30,
            "policy_end_day": 90,
            "enforcement_level": 0.6,
            "targeting_bias_by_degree": 0.5,
            "top_n_communities_targeted": 3,
            "community_policy_strength": 0.5,
            "social_weight": 1.0,
            "media_weight": 0.3,
            "threshold_alpha": 2.0,
            "threshold_beta": 3.0,
            "adoption_noise_sigma": 0.05,
            "async_fraction": 1.0,
            "neighbor_memory_window_days": 3,
            "influence_decay_lambda": 1.0,
            "adoption_rule_code": 0,
            "decay_rate_daily": 0.02,
            "relapse_prob_daily": 0.01,
            "habit_strength": 0.4,
            "fatigue_accumulation_rate": 0.02,
            "recovery_rate_daily": 0.03,
            "smoothing_window_days": 3,
            "report_lag_days": 1,
            "initial_threshold_correlation_with_degree": -0.2,
            # Micro flags defaults added
            "capture_micro": False,
            "save_micro_io": False,
            "micro_sample_frac": 0.1,
            # FIXED: Heavy IO controls defaults
            "save_heavy_io": False,
            "io_agent_sample_n": 100,
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Fallback defaults on parse error
        return {
            "random_seed": 42,
            "population_size": 2000,
            "time_horizon_days": 120,
            "initial_adoption_fraction": 0.02,
            "initial_media_exposure_mean": 0.2,
            "average_degree": 12.0,
            "network_type_code": 0,
            "rewiring_prob": 0.08,
            "community_count": 10,
            "baseline_media_intensity": 0.3,
            "media_reach_fraction": 0.5,
            "media_pulse_interval_days": 21,
            "media_pulse_duration_days": 5,
            "media_pulse_magnitude": 0.4,
            "policy_start_day": 30,
            "policy_end_day": 90,
            "enforcement_level": 0.6,
            "targeting_bias_by_degree": 0.5,
            "top_n_communities_targeted": 3,
            "community_policy_strength": 0.5,
            "social_weight": 1.0,
            "media_weight": 0.3,
            "threshold_alpha": 2.0,
            "threshold_beta": 3.0,
            "adoption_noise_sigma": 0.05,
            "async_fraction": 1.0,
            "neighbor_memory_window_days": 3,
            "influence_decay_lambda": 1.0,
            "adoption_rule_code": 0,
            "decay_rate_daily": 0.02,
            "relapse_prob_daily": 0.01,
            "habit_strength": 0.4,
            "fatigue_accumulation_rate": 0.02,
            "recovery_rate_daily": 0.03,
            "smoothing_window_days": 3,
            "report_lag_days": 1,
            "initial_threshold_correlation_with_degree": -0.2,
            "capture_micro": False,
            "save_micro_io": False,
            "micro_sample_frac": 0.1,
            "save_heavy_io": False,
            "io_agent_sample_n": 100,
        }


# -----------------------------------------------------------------------------
# Module Base
# -----------------------------------------------------------------------------

class Module:
    """
    Base Module with forward signature and metadata.
    """
    def __init__(self, name: str, dependencies: Optional[List[str]] = None, tick_rate_days: int = 1) -> None:
        """
        Initialize a module with its name, dependencies, and tick rate.
        """
        pass
        self.name = name
        self.dependencies = dependencies or []
        self.tick_rate_days = tick_rate_days

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: ParameterRegistry, t: int) -> Dict[str, Any]:
        """
        Run the module forward for day t and return outputs to write to buffers.
        Implementations should be pure with respect to global state; actual committing happens in the scheduler.
        """
        pass
        return {}


# -----------------------------------------------------------------------------
# Modules Implementations
# -----------------------------------------------------------------------------

class MediaAndPolicy(Module):
    """
    Generates exogenous influence signals from media and policy interventions; can target high-degree nodes and selected communities.
    """
    def __init__(self) -> None:
        """
        Construct the MediaAndPolicy module.
        """
        pass
        super().__init__(name="MediaAndPolicy", dependencies=["NetworkBuilder"], tick_rate_days=1)

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: ParameterRegistry, t: int) -> Dict[str, Any]:
        """
        Compute external influence per agent for day t and whether policy is active.

        Pseudocode mapping implemented closely to the given plan with degree-targeting and community targeting.
        """
        pass
        N = state["N"]
        deg = state["people"]["degree"]
        comm = state["people"]["community"]
        baseline = float(params.values.get("baseline_media_intensity", 0.3))
        reach_fraction = float(params.values.get("media_reach_fraction", 0.5))
        pulse_interval = int(params.values.get("media_pulse_interval_days", 21))
        pulse_duration = int(params.values.get("media_pulse_duration_days", 5))
        pulse_magnitude = float(params.values.get("media_pulse_magnitude", 0.4))
        policy_start = int(params.values.get("policy_start_day", 30))
        policy_end = int(params.values.get("policy_end_day", 90))
        enforcement = float(params.values.get("enforcement_level", 0.6))
        bias_by_degree = float(params.values.get("targeting_bias_by_degree", 0.5))
        top_n_targeted = int(params.values.get("top_n_communities_targeted", 3))
        community_policy_strength = float(params.values.get("community_policy_strength", 0.5))

        policy_active = (t >= policy_start) and (t <= policy_end)
        base_intensity = baseline + (enforcement * 0.5 if policy_active else 0.0)
        # Periodic pulse
        if pulse_interval > 0:
            shift = (t - 1) % pulse_interval
            if 0 <= shift < pulse_duration:
                base_intensity += pulse_magnitude

        max_deg = max(1, int(max(deg)))
        external = [0.0] * N

        # Determine targeted top communities by size if requested
        targeted_set: set = set()
        if top_n_targeted > 0 and policy_active:
            # compute sizes
            counts: Dict[int, int] = {}
            for c in comm:
                counts[int(c)] = counts.get(int(c), 0) + 1
            top_comms = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n_targeted]
            targeted_set = {cid for cid, _ in top_comms}

        # Influence per agent with degree targeting and community multiplier
        for i in range(N):
            # Bernoulli reach mask
            rmask = random.random() < reach_fraction
            if not rmask:
                external[i] = 0.0
                continue
            deg_bias = (deg[i] / float(max_deg)) ** bias_by_degree if bias_by_degree > 0 else 1.0
            comm_target = 1.0 + community_policy_strength if int(comm[i]) in targeted_set else 1.0
            external[i] = base_intensity * (0.5 + 0.5 * deg_bias) * comm_target

        return {
            "signals.external_influence_per_person": external,
            "signals.policy_active": policy_active,
        }


class InfluencePropagation(Module):
    """
    Updates agent adoption based on social influence from neighbors combined with external signals using a mixed threshold/cascade mechanism.
    Also captures micro-transition features for calibration.
    """
    def __init__(self) -> None:
        """
        Construct the InfluencePropagation module.
        """
        pass
        super().__init__(name="InfluencePropagation", dependencies=["MediaAndPolicy"], tick_rate_days=1)

    def _neighbor_social_fraction(self, state: Dict[str, Any], t: int, memory_window: int) -> List[float]:
        """
        Compute per-agent fraction of neighbors adopted using a windowed average.
        Implements a performance-aware fallback by reducing window automatically for large populations.

        # FIXED: Optimize neighbor social fraction by capping window under heavy load and optional numpy usage.
        """
        pass
        N = state["N"]
        neighbors: List[List[int]] = state["neighbors"]
        is_influencer = state["people"]["is_influencer"]
        influencer_boost = float(state.get("params_values", {}).get("influencer_weight_multiplier", 1.0))
        # compute average degree
        deg_arr = state["people"]["degree"]
        avg_deg = float(sum(deg_arr)) / float(max(1, len(deg_arr)))
        # Cap window if too heavy
        window = min(memory_window, len(state.get("adopted_history", [])))
        complexity_est = N * max(1.0, avg_deg) * max(1, window)
        if complexity_est > 5_000_000:
            # Reduce window automatically to keep complexity manageable
            window = max(1, int(5_000_000 / max(1, int(N * max(1.0, avg_deg)))))
        if window <= 0:
            return [0.0] * N

        # Use only the last 'window' days
        adopted_history: List[List[int]] = state.get("adopted_history", [])[-window:]
        # If numpy available, convert to arrays for faster slicing
        if np is not None:
            A = np.array(adopted_history, dtype=float)  # shape (window, N)
            fracs = [0.0] * N
            # Precompute neighbor weights (influencer weighting)
            for i in range(N):
                neigh = neighbors[i]
                if not neigh:
                    fracs[i] = 0.0
                    continue
                if influencer_boost != 1.0:
                    weights = np.array([influencer_boost if is_influencer[j] else 1.0 for j in neigh], dtype=float)
                else:
                    weights = np.ones(len(neigh), dtype=float)
                denom = float(weights.sum()) if weights.size > 0 else 1.0
                # Mean over window of weighted neighbor adoption
                # A[:, neigh] -> shape (window, len(neigh))
                vals = (A[:, neigh] * weights[None, :]).sum(axis=1) / denom
                fracs[i] = float(vals.mean())
            return fracs

        # Fallback pure-Python implementation
        frac = [0.0] * N
        for i in range(N):
            neigh = neighbors[i]
            if not neigh:
                frac[i] = 0.0
                continue
            total = 0.0
            denom_base = 0.0
            # Precompute neighbor weights
            weights = [(influencer_boost if is_influencer[j] else 1.0) for j in neigh]
            denom = sum(weights) if weights else 1.0
            denom_base = denom
            for day_ad in adopted_history:
                cnt = 0.0
                for w, j in zip(weights, neigh):
                    cnt += w * day_ad[j]
                total += (cnt / max(1.0, denom_base))
            frac[i] = total / float(window)
        return frac

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: ParameterRegistry, t: int) -> Dict[str, Any]:
        """
        Run mixed linear threshold / independent cascade decision update.
        """
        pass
        N = state["N"]
        adopted = state["people"]["adopted"]
        thresholds = state["people"]["threshold"]
        external = buffers.get("signals.external_influence_per_person", [0.0] * N)

        social_weight = float(params.values.get("social_weight", 1.0))
        media_weight = float(params.values.get("media_weight", 0.3))
        noise_sigma = float(params.values.get("adoption_noise_sigma", 0.05))
        async_fraction = float(params.values.get("async_fraction", 1.0))
        memory_window = int(params.values.get("neighbor_memory_window_days", 3))
        adoption_rule_code = int(params.values.get("adoption_rule_code", 0))

        # store select params in state for helper
        state["params_values"] = {
            "influencer_weight_multiplier": float(params.values.get("influencer_weight_multiplier", 1.0))
        }

        # social fraction from neighbors
        social_fracs = self._neighbor_social_fraction(state, t, memory_window)
        update_mask = [False] * N
        if async_fraction >= 1.0:
            for i in range(N):
                update_mask[i] = True
        else:
            # Sample subset
            idxs = list(range(N))
            random.shuffle(idxs)
            count = int(max(1, round(async_fraction * N)))
            for i in idxs[:count]:
                update_mask[i] = True

        new_adopted = adopted[:]  # start from yesterday
        new_adoptions_today = 0
        # Collect micro-transitions if enabled; subsample to reduce size
        micro_enabled = state.get("capture_micro", False)
        micro_sample_frac = float(state.get("micro_sample_frac", 0.1))
        micro_records: List[Dict[str, Any]] = []

        for i in range(N):
            if not update_mask[i]:
                continue
            if adopted[i]:
                continue  # skip already adopted for adoption; handled by decay module for relapse
            social_signal = social_weight * social_fracs[i]
            media_signal = media_weight * external[i]
            total_signal = social_signal + media_signal + (random.gauss(0.0, noise_sigma) if noise_sigma > 0 else 0.0)
            will_adopt = False
            if adoption_rule_code == 0:
                will_adopt = total_signal >= thresholds[i]
            else:
                prob = 1.0 - math.exp(-max(0.0, total_signal))
                prob = max(0.0, min(1.0, prob))
                will_adopt = random.random() < prob
            if micro_enabled and random.random() < micro_sample_frac:
                micro_records.append({
                    "i": i,
                    "t": t,
                    "prev": 1 if adopted[i] else 0,
                    "social_frac": social_fracs[i],
                    "external": external[i],
                    "total_signal": total_signal,
                    "threshold": thresholds[i],
                    "will_adopt": 1 if will_adopt else 0,
                })
            if will_adopt:
                new_adopted[i] = 1
                new_adoptions_today += 1

        return {
            "people.adopted.proposed": new_adopted,
            "signals.new_adoptions_today": new_adoptions_today,
            "micro.transitions": micro_records,
        }


class BehaviorDecayAndMemory(Module):
    """
    Models habit reinforcement, fatigue, decay, and possible relapse from adopted to non-adopted states.
    """
    def __init__(self) -> None:
        """
        Construct the BehaviorDecayAndMemory module.
        """
        pass
        super().__init__(name="BehaviorDecayAndMemory", dependencies=["InfluencePropagation"], tick_rate_days=1)

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: ParameterRegistry, t: int) -> Dict[str, Any]:
        """
        Update habit strength and fatigue, and perform relapse transitions based on daily probabilities.
        """
        pass
        N = state["N"]
        adopted_proposed = buffers.get("people.adopted.proposed", state["people"]["adopted"])
        habit = state["people"]["habit_strength"][:]
        fatigue = state["people"]["fatigue"][:]

        decay_rate = float(params.values.get("decay_rate_daily", 0.02))
        relapse_prob_daily = float(params.values.get("relapse_prob_daily", 0.01))
        habit_init = float(params.values.get("habit_strength", 0.4))
        fatigue_accumulation_rate = float(params.values.get("fatigue_accumulation_rate", 0.02))
        recovery_rate_daily = float(params.values.get("recovery_rate_daily", 0.03))

        relapses_today = 0
        adopted_final = adopted_proposed[:]
        ever = state["people"].get("ever_adopted", [False] * N)  # FIXED: Use ever_adopted sentinel to disambiguate never vs relapsed
        for i in range(N):
            if adopted_proposed[i]:
                # reinforce habit and accumulate fatigue
                habit[i] = min(1.0, habit[i] + (1.0 - habit[i]) * 0.1)
                fatigue[i] = min(1.0, fatigue[i] + fatigue_accumulation_rate)
                relapse_prob = relapse_prob_daily * (1.0 - habit[i]) * (0.5 + fatigue[i])
                if random.random() < relapse_prob:
                    adopted_final[i] = 0
                    relapses_today += 1
            else:
                # decay habit and recover fatigue
                habit[i] = max(0.0, habit[i] - decay_rate)
                fatigue[i] = max(0.0, fatigue[i] - recovery_rate_daily)
                # If never adopted, keep initial habit baseline at minimum
                if not ever[i]:
                    habit[i] = max(habit[i], min(1.0, habit_init * 0.5))

        return {
            "people.adopted.final": adopted_final,
            "people.habit_strength.updated": habit,
            "people.fatigue.updated": fatigue,
            "signals.relapses_today": relapses_today,
        }


class ObservationAggregator(Module):
    """
    Aggregates daily observables and applies smoothing/reporting lag to match data characteristics.
    """
    def __init__(self) -> None:
        """
        Construct the ObservationAggregator module.
        """
        pass
        super().__init__(name="ObservationAggregator", dependencies=["BehaviorDecayAndMemory"], tick_rate_days=1)

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: ParameterRegistry, t: int) -> Dict[str, Any]:
        """
        Aggregate observables for day t with smoothing and lag.

        # FIXED: Make forward pure: emit raw values; central commit in Simulation.run appends and smooths.
        """
        pass
        N = state["N"]
        adopted_today = buffers.get("people.adopted.final", state["people"]["adopted"])
        new_adopt = int(buffers.get("signals.new_adoptions_today", 0))

        # adoption rate (raw for the day)
        raw_rate = sum(adopted_today) / float(N) if N > 0 else 0.0

        # by community (snapshot)
        comm = state["people"]["community"]
        by_c: Dict[int, List[int]] = {}
        for i, c in enumerate(comm):
            by_c.setdefault(int(c), []).append(int(adopted_today[i]))
        by_comm_rate: Dict[str, float] = {}
        for c, vals in by_c.items():
            by_comm_rate[str(c)] = sum(vals) / float(len(vals)) if len(vals) > 0 else 0.0

        return {
            # FIXED: Emit raw values only
            "observable.raw_adoption_rate": float(raw_rate),
            "observable.raw_new_adoptions": float(new_adopt),
            "observable.adoption_rate_by_community": by_comm_rate,
        }


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

class Simulation:
    """
    Main simulation engine that coordinates modules, state, buffers, scheduler, and artifacts.
    """
    def __init__(self, params: ParameterRegistry, artifacts_root: str, data_dir: Optional[str] = None) -> None:
        """
        Initialize the Simulation with parameters and artifacts root, build population and network, and set initial state.
        """
        pass
        self.params = params
        self.artifacts_root = artifacts_root
        self.data_dir = data_dir or ""
        # Directories
        self.paths = {
            "root": artifacts_root,
            "results": os.path.join(artifacts_root, "results"),
            "io": os.path.join(artifacts_root, "io"),
            "figs": os.path.join(artifacts_root, "figs"),
            "logs": os.path.join(artifacts_root, "logs"),
            "snapshots": os.path.join(artifacts_root, "snapshots"),
            "calibration": os.path.join(artifacts_root, "calibration"),
        }
        for p in self.paths.values():
            ensure_dir(p)
        # Build network and population
        self.state: Dict[str, Any] = {}
        self.module_daily_io: Dict[str, List[Dict[str, Any]]] = {}
        self._init_population()
        self._init_modules()

    def _init_population(self) -> None:
        """
        Initialize population attributes, communities, households, thresholds, and network.

        # FIXED: Seed NetworkX graph generators for reproducibility.
        # FIXED: Default capture_micro to False and read from params; set micro_sample_frac; set save_micro_io flag.
        # FIXED: Add ever_adopted sentinel to disambiguate 'never adopted' from 'relapsed' agents.
        # FIXED: Prepare cache container for reusable structures.
        """
        pass
        N = int(self.params.values.get("population_size", 2000))
        # If networkx is not available, fallback to trivial neighbor lists
        neighbors: List[List[int]] = [[] for _ in range(N)]
        degree_arr = [0] * N
        communities = [0] * N
        net_type = int(self.params.values.get("network_type_code", 0))
        avg_deg = float(self.params.values.get("average_degree", 12.0))
        community_count = int(self.params.values.get("community_count", 10))
        rewiring_prob = float(self.params.values.get("rewiring_prob", 0.08))
        seed_val = int(self.params.values.get("random_seed", 42))  # FIXED: Seed graph generators

        if nx is not None and N > 1:
            # Build graph per network_type_code
            if net_type == 0:
                # Watts-Strogatz
                k = max(2, int(round(avg_deg)))
                try:
                    G = nx.watts_strogatz_graph(N, k, rewiring_prob, seed=seed_val)
                except Exception:
                    G = nx.erdos_renyi_graph(N, p=min(1.0, avg_deg / max(1.0, N - 1.0)), seed=seed_val)
            elif net_type == 1:
                p = min(1.0, avg_deg / max(1.0, N - 1.0))
                G = nx.erdos_renyi_graph(N, p, seed=seed_val)
            else:
                m = max(1, int(round(avg_deg / 2.0)))
                G = nx.barabasi_albert_graph(N, m, seed=seed_val)
            for i in range(N):
                neigh = list(G.neighbors(i))
                neighbors[i] = neigh
                degree_arr[i] = len(neigh)
        else:
            # trivial line graph fallback
            for i in range(N):
                neigh = []
                if i > 0:
                    neigh.append(i - 1)
                if i < N - 1:
                    neigh.append(i + 1)
                neighbors[i] = neigh
                degree_arr[i] = len(neigh)

        # Assign communities in a round-robin manner
        for i in range(N):
            communities[i] = i % max(1, community_count)

        # Households: sample sizes and assign directly in order
        # FIXED: Respect sampled household sizes when assigning members
        households: List[Dict[str, Any]] = []
        remaining = N
        hid = 0
        mean_size = 3.0
        sizes: List[int] = []
        while remaining > 0:
            if np is not None:
                size = int(max(1, np.random.poisson(mean_size)))
            else:
                size = max(1, int(round(random.random() * 2 + 1)))
            size = min(size, remaining)
            households.append({"id": hid, "members": [], "norm_strength": 0.5 + 0.2 * (random.random() - 0.5)})
            sizes.append(size)
            hid += 1
            remaining -= size
        agent_idx = 0
        for hh_idx, sz in enumerate(sizes):
            for _ in range(sz):
                if agent_idx >= N:
                    break
                households[hh_idx]["members"].append(agent_idx)
                agent_idx += 1

        # People attributes
        initial_adopt_frac = float(self.params.values.get("initial_adoption_fraction", 0.02))
        initial_media_exp_mean = float(self.params.values.get("initial_media_exposure_mean", 0.2))
        threshold_alpha = float(self.params.values.get("threshold_alpha", 2.0))
        threshold_beta = float(self.params.values.get("threshold_beta", 3.0))
        corr_target = float(self.params.values.get("initial_threshold_correlation_with_degree", -0.2))

        adopted = [1 if random.random() < initial_adopt_frac else 0 for _ in range(N)]
        time_since_adoption = [0 if a else -1 for a in adopted]
        habit_strength = [float(self.params.values.get("habit_strength", 0.4)) for _ in range(N)]
        fatigue = [0.0 for _ in range(N)]
        media_exposure = [max(0.0, random.gauss(initial_media_exp_mean, 0.05)) for _ in range(N)]

        # thresholds from Beta with degree correlation by rank mixing
        if np is not None:
            raw_thresh = list(np.random.beta(threshold_alpha, threshold_beta, size=N))
        else:
            # fallback uniform mixing
            raw_thresh = [random.random() for _ in range(N)]
        # enforce correlation target by sorting
        deg_sorted_idx = sorted(range(N), key=lambda i: degree_arr[i])
        thresh_sorted_vals = sorted(raw_thresh)
        # if corr_target negative, assign in reverse order for higher degree -> lower threshold
        if corr_target < 0:
            for rank, idx in enumerate(deg_sorted_idx):
                raw_thresh[idx] = thresh_sorted_vals[-(rank + 1)]
        else:
            for rank, idx in enumerate(deg_sorted_idx):
                raw_thresh[idx] = thresh_sorted_vals[rank]

        # influencers: top percent by degree
        top_k = max(1, int(0.05 * N))
        top_idx = sorted(range(N), key=lambda i: degree_arr[i], reverse=True)[:top_k]
        is_influencer = [0] * N
        for i in top_idx:
            is_influencer[i] = 1

        # Ever adopted flag
        ever_adopted = [bool(a) for a in adopted]  # FIXED: Initialize ever_adopted sentinel

        self.state = {
            "N": N,
            "neighbors": neighbors,
            "people": {
                "adopted": adopted,
                "time_since_adoption": time_since_adoption,
                "habit_strength": habit_strength,
                "fatigue": fatigue,
                "threshold": raw_thresh,
                "media_exposure": media_exposure,
                "susceptibility": [1.0] * N,
                "community": communities,
                "degree": degree_arr,
                "is_influencer": is_influencer,
                "ever_adopted": ever_adopted,  # FIXED: Add sentinel
            },
            "households": households,
            "adopted_history": [adopted[:] for _ in range(max(1, int(self.params.values.get("neighbor_memory_window_days", 3))))],
            "signals": {
                "external_influence": [],
                "policy_active": [],
                "new_adoptions_daily": [],
                "relapses_daily": [],
            },
            "observables": {
                "raw_adoption_rate": [],
                "raw_new_adoptions": [],
                "adoption_rate_daily": [],
                "new_adoptions_daily": [],
                "adoption_rate_by_community": [],
                "cumulative_adoptions": [],
            },
            # toggle micro transitions capture; default off
            "capture_micro": bool(self.params.values.get("capture_micro", False)),  # FIXED: Default off
            "micro_sample_frac": float(self.params.values.get("micro_sample_frac", 0.1)),
            "save_micro_io": bool(self.params.values.get("save_micro_io", False)),
            "cache": {},  # FIXED: Add cache container for potential reuse
        }

    def _init_modules(self) -> None:
        """
        Initialize and register module instances in execution order.
        """
        pass
        self.modules: List[Module] = []
        # NetworkBuilder is implicitly handled at initialization
        self.modules.append(MediaAndPolicy())
        self.modules.append(InfluencePropagation())
        self.modules.append(BehaviorDecayAndMemory())
        self.modules.append(ObservationAggregator())
        for m in self.modules:
            self.module_daily_io[m.name] = []

    def reset(self) -> None:
        """
        Reset simulation state and IO buffers by reinitializing population and modules.

        # FIXED: Provide reset() to allow reuse without state carry-over if needed.
        """
        pass
        self.module_daily_io = {}
        self._init_population()
        self._init_modules()

    def run(self, start_day: int, end_day: int) -> None:
        """
        Execute the simulation from start_day to end_day inclusive.
        Outputs across modules are buffered then committed to global state.

        # FIXED: Removed duplicate 'prev' state updates by centralizing updates once per day.
        # FIXED: Guard saving of micro-transitions in daily IO; drop unless save_micro_io=True.
        # FIXED: Compute cumulative_adoptions in commit using emitted smoothed incident values.
        # FIXED: Make ObservationAggregator.forward pure and handle appends/smoothing here.
        # FIXED: Add heavy IO controls to sample/drop large arrays to reduce artifact size.
        """
        pass
        # Clip to time horizon
        horizon = int(self.params.values.get("time_horizon_days", 120))
        start_day = max(0, start_day)
        end_day = min(horizon - 1, end_day)
        for t in range(start_day, end_day + 1):
            buffers: Dict[str, Any] = {}
            daily_io: Dict[str, Dict[str, Any]] = {}
            # Execute modules
            for m in self.modules:
                out = m.forward(self.state, buffers, self.params, t)
                # Save outputs to buffer
                for k, v in out.items():
                    buffers[k] = v
                # Collect IO, optionally skipping micro transitions and heavy arrays
                io_out = dict(out)
                # Drop micro transitions unless explicitly requested
                if not self.state.get("save_micro_io", False):
                    io_out.pop("micro.transitions", None)
                # Drop or sample heavy arrays unless explicitly requested
                if not bool(self.params.values.get("save_heavy_io", False)):
                    for k in [
                        "signals.external_influence_per_person",
                        "people.adopted.proposed",
                        "people.adopted.final",
                    ]:
                        if k in io_out:
                            v = io_out.pop(k)
                            if isinstance(v, list):
                                # keep only aggregates & sample
                                mean_val = float(sum(v) / max(1, len(v))) if len(v) > 0 else 0.0
                                n = int(self.params.values.get("io_agent_sample_n", 100))
                                n = max(0, min(n, len(v)))
                                io_out[k + ".mean"] = mean_val
                                io_out[k + ".sample"] = v[:n]
                daily_io[m.name] = io_out

            # Commit phase for this day
            # Commit external signals
            ext = buffers.get("signals.external_influence_per_person")
            if ext is not None:
                # store daily average for analysis
                self.state["signals"]["external_influence"].append(float(sum(ext) / max(1, len(ext))))
            self.state["signals"]["policy_active"].append(bool(buffers.get("signals.policy_active", False)))

            # Adoption commit
            new_adopted = buffers.get("people.adopted.final")
            if new_adopted is None:
                # fallback to proposed or existing
                new_adopted = buffers.get("people.adopted.proposed", self.state["people"]["adopted"])
            today_use = [1 if x else 0 for x in new_adopted]
            prev_use = self.state["people"]["adopted"]
            # update time_since_adoption
            ts = self.state["people"]["time_since_adoption"]
            for i in range(self.state["N"]):
                if today_use[i]:
                    ts[i] = 1 if prev_use[i] == 0 else ts[i] + 1
                else:
                    ts[i] = 0 if prev_use[i] == 1 else ts[i] - 1

            # Ever adopted update
            ever = self.state["people"]["ever_adopted"]
            for i in range(self.state["N"]):
                if today_use[i]:
                    ever[i] = True

            # Habit and fatigue commit from buffers
            upd_habit = buffers.get("people.habit_strength.updated")
            upd_fatigue = buffers.get("people.fatigue.updated")
            if upd_habit is not None:
                self.state["people"]["habit_strength"] = list(upd_habit)
            if upd_fatigue is not None:
                self.state["people"]["fatigue"] = list(upd_fatigue)

            # Centralize prev_adopted update for churn calculation
            # FIXED: Centralize update and avoid duplicate assignment
            self.state["people"]["adopted"] = today_use

            # Maintain adopted history ring buffer
            hist = self.state["adopted_history"]
            hist.append(today_use[:])
            # Keep only last window days (use configured window)
            window = max(1, int(self.params.values.get("neighbor_memory_window_days", 3)))
            if len(hist) > window:
                # drop oldest
                self.state["adopted_history"] = hist[-window:]

            # Signals commit
            self.state["signals"]["new_adoptions_daily"].append(int(buffers.get("signals.new_adoptions_today", 0)))
            self.state["signals"]["relapses_daily"].append(int(buffers.get("signals.relapses_today", 0)))

            # Observables commit (append raw, then smooth and lag)
            raw_rate = float(buffers.get("observable.raw_adoption_rate", 0.0))
            raw_new = float(buffers.get("observable.raw_new_adoptions", 0.0))
            self.state["observables"]["raw_adoption_rate"].append(raw_rate)
            self.state["observables"]["raw_new_adoptions"].append(raw_new)
            # Smooth and lag
            sw = int(self.params.values.get("smoothing_window_days", 3))
            lag = int(self.params.values.get("report_lag_days", 1))
            rate_series = moving_average(self.state["observables"]["raw_adoption_rate"], sw)
            new_series = moving_average(self.state["observables"]["raw_new_adoptions"], sw)
            idx_emit = len(rate_series) - 1 - lag
            rate_emit = float(rate_series[idx_emit]) if idx_emit >= 0 and len(rate_series) > 0 else 0.0
            new_emit = float(new_series[idx_emit]) if idx_emit >= 0 and len(new_series) > 0 else 0.0
            self.state["observables"]["adoption_rate_daily"].append(rate_emit)
            self.state["observables"]["new_adoptions_daily"].append(new_emit)
            self.state["observables"]["adoption_rate_by_community"].append(buffers.get("observable.adoption_rate_by_community", {}))
            # FIXED: Compute cumulative based on emitted incident values
            prev_cum = self.state["observables"]["cumulative_adoptions"][-1] if self.state["observables"]["cumulative_adoptions"] else 0.0
            new_cum = prev_cum + new_emit
            self.state["observables"]["cumulative_adoptions"].append(float(new_cum))

            # Save IO
            self.module_daily_io["MediaAndPolicy"].append(daily_io.get("MediaAndPolicy", {}))
            self.module_daily_io["InfluencePropagation"].append(daily_io.get("InfluencePropagation", {}))
            self.module_daily_io["BehaviorDecayAndMemory"].append(daily_io.get("BehaviorDecayAndMemory", {}))
            self.module_daily_io["ObservationAggregator"].append(daily_io.get("ObservationAggregator", {}))

    def save_results(self, filename_prefix: str) -> None:
        """
        Save simulation results to JSON files under artifacts/results.
        """
        pass
        results_dir = self.paths["results"]
        ensure_dir(results_dir)
        # Observables
        observables = self.state["observables"]
        safe_json_dump(observables, os.path.join(results_dir, f"{filename_prefix}_observables.json"))
        # Signals
        safe_json_dump(self.state["signals"], os.path.join(results_dir, f"{filename_prefix}_signals.json"))
        # Population snapshot
        people = self.state["people"].copy()
        # to avoid dumping huge lists, sample a subset for snapshot
        N = self.state["N"]
        idx_sample = list(range(min(N, 100)))
        snapshot = {k: [v[i] for i in idx_sample] if isinstance(v, list) else v for k, v in people.items()}
        safe_json_dump(snapshot, os.path.join(results_dir, f"{filename_prefix}_people_snapshot.json"))

    def save_module_io(self, module_name: str, path: Optional[str] = None) -> None:
        """
        Save module I/O buffers for all days for a specific module.
        """
        pass
        path = path or self.paths["io"]
        ensure_dir(path)
        data = self.module_daily_io.get(module_name, [])
        safe_json_dump(data, os.path.join(path, f"{module_name}_io.json"))

    def save_all_io(self, root_dir: Optional[str] = None) -> None:
        """
        Save all modules' IO buffers to artifacts/io.
        """
        pass
        for m in self.module_daily_io.keys():
            self.save_module_io(m, root_dir or self.paths["io"])

    def evaluate(self, gt: Optional[Dict[str, List[float]]] = None, window: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Compute evaluation metrics from observables against ground-truth if available.
        Metrics: RMSE_overall_adoption (normalized), MAE_new_adoptions (absolute), PeakError_new_adoptions, TimeToPeak_new_adoptions.

        # FIXED: Add window slicing support to compute metrics on a specified window and avoid length mismatches.
        """
        pass
        obs = self.state["observables"]
        results_dir = self.paths["results"]
        y_hat_rate_full = obs.get("adoption_rate_daily", [])
        y_hat_new_full = obs.get("new_adoptions_daily", [])

        # Load ground truth if not given
        if gt is None:
            gt = self._load_ground_truth()

        y_true_rate_full = gt.get("overall_adoption_rate", []) if gt else []
        y_true_new_full = gt.get("new_adoptions", []) if gt else []

        # Slice by window if provided
        def slice_series(series: List[float], w: Optional[Tuple[int, int]]) -> List[float]:
            if not series:
                return []
            if w is None:
                return list(series)
            s, e = w
            s = max(0, s)
            e = max(s, e)
            e = min(e, len(series) - 1)
            if s > e:
                return []
            return list(series[s:e + 1])

        y_hat_rate = slice_series(y_hat_rate_full, window)
        y_hat_new = slice_series(y_hat_new_full, window)
        y_true_rate = slice_series(y_true_rate_full, window) if y_true_rate_full else []
        y_true_new = slice_series(y_true_new_full, window) if y_true_new_full else []

        # Align lengths safely
        def align(a: List[float], b: List[float]) -> Tuple[List[float], List[float]]:
            n = min(len(a), len(b))
            return a[:n], b[:n]

        def rmse(a: List[float], b: List[float], normalize: bool = True) -> float:
            a, b = align(a, b)
            if not a or not b:
                return float("nan")
            err = 0.0
            for x, y in zip(a, b):
                d = (x - y)
                err += d * d
            err = math.sqrt(err / float(len(a)))
            if normalize:
                denom = (max(b) - min(b)) if (b and (max(b) - min(b)) > 0) else 1.0
                err = err / denom
            return err

        def mae(a: List[float], b: List[float]) -> float:
            a, b = align(a, b)
            if not a or not b:
                return float("nan")
            return sum(abs(x - y) for x, y in zip(a, b)) / float(len(a))

        def peak_error(a: List[float], b: List[float]) -> float:
            a, b = align(a, b)
            if not a or not b:
                return float("nan")
            return abs(max(a) - max(b))

        def time_to_peak(seq: List[float]) -> int:
            return int(seq.index(max(seq))) if seq else -1

        metrics = {
            "RMSE_overall_adoption": rmse(y_hat_rate, y_true_rate) if y_true_rate else None,
            "MAE_new_adoptions": mae(y_hat_new, y_true_new) if y_true_new else None,
            "PeakError_new_adoptions": peak_error(y_hat_new, y_true_new) if y_true_new else None,
            "TimeToPeak_new_adoptions": time_to_peak(y_hat_new),
            "window_evaluated": list(window) if window is not None else None,
        }
        safe_json_dump(metrics, os.path.join(results_dir, "metrics.json"))
        return metrics

    def _load_ground_truth(self) -> Dict[str, List[float]]:
        """
        Attempt to load ground-truth data from data directory; if missing, try synthetic target.
        """
        pass
        # Path handling per instructions
        PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
        DATA_PATH = os.environ.get("DATA_PATH", "")
        DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else (self.data_dir or "")
        time_series = os.path.join(DATA_DIR, "time_series.csv") if DATA_DIR else ""
        if pd is not None and time_series and os.path.exists(time_series):
            try:
                df = pd.read_csv(time_series)
                gt = {
                    "overall_adoption_rate": df.get("overall_adoption_rate", pd.Series([], dtype=float)).astype(float).tolist(),
                    "new_adoptions": df.get("new_adoptions", pd.Series([], dtype=float)).astype(float).tolist(),
                    "cumulative_adoptions": df.get("cumulative_adoptions", pd.Series([], dtype=float)).astype(float).tolist(),
                }
                return gt
            except Exception:
                pass
        # Fallback: synthetic ground truth from snapshot if exists
        gt_path = os.path.join(self.paths["results"], "ground_truth.json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def export_micro_transitions(self) -> Optional[pd.DataFrame]:
        """
        Export micro-transition records captured by InfluencePropagation if pandas is available.
        """
        pass
        if pd is None:
            return None
        infl = self.module_daily_io.get("InfluencePropagation", [])
        rows: List[Dict[str, Any]] = []
        for day in infl:
            recs = day.get("micro.transitions", [])
            if isinstance(recs, list):
                rows.extend(recs)
        if not rows:
            return None
        return pd.DataFrame(rows)

    def visualize(self) -> None:
        """
        Visualize key observables if matplotlib is available.
        """
        pass
        if plt is None:
            print("Visualization skipped: matplotlib not available.")
            return
        obs = self.state["observables"]
        x = list(range(len(obs.get("adoption_rate_daily", []))))
        plt.figure(figsize=(10, 6))
        plt.plot(x, obs.get("adoption_rate_daily", []), label="Adoption Rate (Daily)")
        plt.plot(x, obs.get("new_adoptions_daily", []), label="New Adoptions (Daily)")
        plt.xlabel("Day")
        plt.ylabel("Value")
        plt.title("Simulation Observables")
        plt.legend()
        fig_path = os.path.join(self.paths["figs"], "observables.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved figure: {fig_path}")


# -----------------------------------------------------------------------------
# Pluggable Calibration Architecture
# -----------------------------------------------------------------------------

@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator.
    """
    decision_weights: Dict[str, float]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for JSON serialization.
        """
        pass
        return asdict(self)


class ParamsAdapter:
    """
    Adapts FittedParams to simulation parameter system.
    """
    def __init__(self, param_defs_path: Optional[str] = None, registry: Optional[ParameterRegistry] = None) -> None:
        """
        Initialize adapter with optional parameter definitions path for frozen checks and a registry.
        """
        pass
        self.param_defs_path = param_defs_path
        self.registry = registry
        self._defs: Dict[str, ParameterDefinition] = registry.defs if registry else {}

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply params via simulation parameter system: set_params() + parameters_used.json

        # FIXED: Apply via ParameterRegistry.set_params() to respect frozen/unknown and clamp to bounds.
        """
        pass
        if self.registry is None:
            self.registry = simulation.params
        # Map decision weights to InfluencePropagation params
        updates: Dict[str, Any] = {}
        for k, v in params.decision_weights.items():
            if k in ("social_weight", "media_weight", "adoption_noise_sigma"):
                updates[k] = float(v)
        # Layer weights mapping (e.g., community policy strength)
        for k, v in params.layer_weights.items():
            if k == "community":
                updates["community_policy_strength"] = float(v)
        # Info params
        for k, v in params.info_params.items():
            if k in ("baseline_media_intensity", "media_reach_fraction"):
                updates[k] = float(v)
        # Noise params
        for k, v in params.noise_params.items():
            if k == "temperature":
                # map to adoption_noise_sigma as proxy
                updates["adoption_noise_sigma"] = float(v)
        # Module-specific params
        for _, kv in params.module_params.items():
            for k, v in kv.items():
                updates[k] = v

        # Apply updates respecting frozen flags and clamping to bounds
        warnings = self.registry.set_params(**updates)

        # Persist final parameters
        used = self.registry.export_used()
        safe_json_dump(used, os.path.join(simulation.paths["results"], "parameters_used.json"))
        if warnings:
            safe_json_dump(warnings, os.path.join(simulation.paths["results"], "parameters_warnings.json"))

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams.
        """
        pass
        vals = simulation.params.values
        decision_weights = {
            "social_weight": float(vals.get("social_weight", 1.0)),
            "media_weight": float(vals.get("media_weight", 0.3)),
            "adoption_noise_sigma": float(vals.get("adoption_noise_sigma", 0.05)),
        }
        layer_weights = {
            "community": float(vals.get("community_policy_strength", 0.5)),
        }
        info_params = {
            "baseline_media_intensity": float(vals.get("baseline_media_intensity", 0.3)),
            "media_reach_fraction": float(vals.get("media_reach_fraction", 0.5)),
        }
        noise_params = {"temperature": float(vals.get("adoption_noise_sigma", 0.05))}
        return FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            module_params={},
            meta={"captured_at": time.time()},
        )

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Check frozen params and return warnings for any attempted updates on frozen keys.
        """
        pass
        warnings: Dict[str, str] = {}
        if self.registry is None:
            return warnings
        for group in [params.decision_weights, params.layer_weights, params.info_params, params.noise_params]:
            for k in group.keys():
                if self.registry.is_frozen(k):
                    warnings[k] = f"Parameter '{k}' is frozen."
        for _, kv in params.module_params.items():
            for k in kv.keys():
                if self.registry.is_frozen(k):
                    warnings[k] = f"Parameter '{k}' is frozen."
        return warnings


class Calibrator(ABC):
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """
    @abstractmethod
    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Return FittedParams, fitted strictly on the training window.
        """
        pass


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search over selected simulator parameters.
    """
    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Initialize with an optional search space dictionary mapping keys to (low, high).
        """
        pass
        self.search_space = search_space or {
            "social_weight": (0.0, 2.0),
            "media_weight": (0.0, 2.0),
            "baseline_media_intensity": (0.0, 2.0),
            "media_reach_fraction": (0.0, 1.0),
            "adoption_noise_sigma": (0.0, 0.5),
            "enforcement_level": (0.0, 1.0),
            "decay_rate_daily": (0.0, 0.2),
            "relapse_prob_daily": (0.0, 0.1),
        }

    def _sample_params(self, base: FittedParams) -> FittedParams:
        """
        Sample within bounds and return new FittedParams instance.
        """
        pass
        def rand(low: float, high: float) -> float:
            return low + random.random() * (high - low)

        sampled = FittedParams(
            decision_weights={
                "social_weight": rand(*self.search_space["social_weight"]),
                "media_weight": rand(*self.search_space["media_weight"]),
                "adoption_noise_sigma": rand(*self.search_space["adoption_noise_sigma"]),
            },
            layer_weights={"community": base.layer_weights.get("community", 0.5) if base.layer_weights else 0.5},
            info_params={
                "baseline_media_intensity": rand(*self.search_space["baseline_media_intensity"]),
                "media_reach_fraction": rand(*self.search_space["media_reach_fraction"]),
            },
            noise_params={"temperature": rand(*self.search_space["adoption_noise_sigma"])},
            module_params={
                "MediaAndPolicy": {"enforcement_level": rand(*self.search_space["enforcement_level"])},
                "BehaviorDecayAndMemory": {
                    "decay_rate_daily": rand(*self.search_space["decay_rate_daily"]),
                    "relapse_prob_daily": rand(*self.search_space["relapse_prob_daily"]),
                }
            },
            meta={"sampled_at": time.time()}
        )
        return sampled

    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Random search over the provided budget. Uses evaluator(simulator, params, window) as objective.

        # FIXED: Harden calibration score handling: treat None/NaN as +inf to avoid exceptions.
        """
        pass
        random.seed(seed)
        ensure_dir(artifacts_dir or "")
        best_score = float("inf")
        best_params: Optional[FittedParams] = None
        base = params_adapter.capture(simulator) if params_adapter else FittedParams({}, {}, {}, {})
        # QUICK_TEST gating
        quick = os.environ.get("QUICK_TEST", "0") == "1"
        if quick:
            budget = min(budget, 3)

        trials_meta: List[Dict[str, Any]] = []
        for i in range(max(1, budget)):
            fp = self._sample_params(base)
            trial_dir = None
            if artifacts_dir:
                trial_dir = os.path.join(artifacts_dir, f"trial_{i}")
                ensure_dir(trial_dir)
                safe_json_dump(fp.to_dict(), os.path.join(trial_dir, "params_sampled.json"))
                # FIXED: Also save as params_applied.json for compatibility
                safe_json_dump(fp.to_dict(), os.path.join(trial_dir, "params_applied.json"))

            metrics = evaluator(simulator, fp, train_window)
            score_raw = metrics.get("RMSE_aggregate", None)
            score = float(score_raw) if isinstance(score_raw, (int, float)) and math.isfinite(float(score_raw)) else float("inf")
            trials_meta.append({"trial": i, "score": score})
            if trial_dir:
                safe_json_dump(metrics, os.path.join(trial_dir, "metrics.json"))

            if score < best_score:
                best_score = score
                best_params = fp

        # Save final best
        if artifacts_dir and best_params is not None:
            best_dir = os.path.join(artifacts_dir, "best")
            ensure_dir(best_dir)
            safe_json_dump(best_params.to_dict(), os.path.join(best_dir, "fitted_params.json"))
            safe_json_dump({"budget": budget, "trials": trials_meta, "best_score": best_score}, os.path.join(artifacts_dir, "calibration_report.json"))

        return best_params if best_params is not None else base


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head from micro-transitions (if available). Degrades gracefully if unavailable.
    """
    def __init__(self, l2: float = 1.0) -> None:
        """
        Initialize with L2 regularization parameter.
        """
        pass
        self.l2 = l2

    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Fit logistic regression to micro-transition dataset and convert coefficients to decision weights.
        If micro data unavailable, fallback to RandomSearchCalibrator with small budget.
        """
        pass
        random.seed(seed)
        ensure_dir(artifacts_dir or "")
        df = simulator.export_micro_transitions()
        if df is None or df.empty:
            # fallback
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget=min(5, budget), artifacts_dir=artifacts_dir, params_adapter=params_adapter)

        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
        except Exception:
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget=min(5, budget), artifacts_dir=artifacts_dir, params_adapter=params_adapter)

        # Prepare data
        X = df[["social_frac", "external"]].values
        y = df["will_adopt"].values
        # Fit
        clf = LogisticRegression(C=1.0 / max(1e-6, self.l2), fit_intercept=True, max_iter=200)
        clf.fit(X, y)

        # Map coefficients to params: social_weight ~ coef[0], media_weight ~ coef[1], noise -> inverse of coef magnitude
        social_w = float(clf.coef_[0, 0])
        media_w = float(clf.coef_[0, 1])
        noise_sigma = float(min(0.5, max(0.01, 1.0 / (abs(social_w) + abs(media_w) + 1e-6))))

        fp = FittedParams(
            decision_weights={"social_weight": social_w, "media_weight": media_w, "adoption_noise_sigma": noise_sigma},
            layer_weights={"community": float(simulator.params.values.get("community_policy_strength", 0.5))},
            info_params={
                "baseline_media_intensity": float(simulator.params.values.get("baseline_media_intensity", 0.3)),
                "media_reach_fraction": float(simulator.params.values.get("media_reach_fraction", 0.5))
            },
            noise_params={"temperature": noise_sigma},
            module_params={},
            meta={"calibrator": "logit_head", "seed": seed}
        )

        # Score and save
        metrics = evaluator(simulator, fp, train_window)
        if artifacts_dir:
            best_dir = os.path.join(artifacts_dir, "best")
            ensure_dir(best_dir)
            safe_json_dump(fp.to_dict(), os.path.join(best_dir, "fitted_params.json"))
            safe_json_dump(metrics, os.path.join(best_dir, "metrics.json"))
            safe_json_dump({"budget": 1, "trials": [{"trial": 0, "score": metrics.get("RMSE_aggregate", None)}], "best_score": metrics.get("RMSE_aggregate", None)},
                           os.path.join(artifacts_dir, "calibration_report.json"))

        return fp


class SNPECalibrator(Calibrator):
    """
    True SBI using neural networks for Bayesian parameter inference.
    Falls back to RandomSearchCalibrator if dependencies unavailable.
    """
    def __init__(self, rounds: int = 1, num_simulations: int = 100) -> None:
        """
        Initialize SNPE calibrator with rounds and number of simulations.
        """
        pass
        self.rounds = rounds
        self.num_simulations = num_simulations

    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Attempt SNPE; fallback to RandomSearch on ImportError or runtime failure.
        """
        pass
        try:
            import torch  # noqa
            from sbi.inference import SNPE  # type: ignore
            from sbi.utils import BoxUniform  # type: ignore
        except Exception:
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        # Define prior bounds for selected params
        import torch  # type: ignore
        priors_low = torch.tensor([0.0, 0.0, 0.0, 0.0])  # social_weight, media_weight, baseline_media_intensity, media_reach_fraction
        priors_high = torch.tensor([2.0, 2.0, 2.0, 1.0])
        prior = BoxUniform(priors_low, priors_high)

        def simulate(theta):
            # theta: [social_weight, media_weight, baseline_media_intensity, media_reach_fraction]
            sw, mw, bmi, mrf = map(float, theta.tolist())
            fp = FittedParams(
                decision_weights={"social_weight": sw, "media_weight": mw, "adoption_noise_sigma": float(simulator.params.values.get("adoption_noise_sigma", 0.05))},
                layer_weights={"community": float(simulator.params.values.get("community_policy_strength", 0.5))},
                info_params={"baseline_media_intensity": bmi, "media_reach_fraction": mrf},
                noise_params={"temperature": float(simulator.params.values.get("adoption_noise_sigma", 0.05))},
                module_params={},
                meta={}
            )
            metrics = evaluator(simulator, fp, train_window)
            # Return a scalar summary statistic (RMSE) as torch tensor
            rmse_val = metrics.get("RMSE_aggregate", None)
            rmse_val = float(rmse_val) if isinstance(rmse_val, (int, float)) and math.isfinite(float(rmse_val)) else float("inf")
            return torch.tensor([rmse_val], dtype=torch.float32)

        inference = SNPE(prior=prior)
        # Simulations
        num_sim = min(budget, self.num_simulations)
        xs = []
        ys = []
        torch.manual_seed(seed)
        for _ in range(num_sim):
            theta = prior.sample((1,)).squeeze(0)
            y = simulate(theta)
            xs.append(theta)
            ys.append(y)
        xs = torch.stack(xs)
        ys = torch.stack(ys).squeeze(-1)
        density_estimator = inference.append_simulations(xs, ys).train()
        posterior = inference.build_posterior(density_estimator)
        # Sample from posterior mode approximation
        try:
            theta_star = posterior.sample((1,)).squeeze(0)
        except Exception:
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        sw, mw, bmi, mrf = map(float, theta_star.tolist())
        fp = FittedParams(
            decision_weights={"social_weight": sw, "media_weight": mw, "adoption_noise_sigma": float(simulator.params.values.get("adoption_noise_sigma", 0.05))},
            layer_weights={"community": float(simulator.params.values.get("community_policy_strength", 0.5))},
            info_params={"baseline_media_intensity": bmi, "media_reach_fraction": mrf},
            noise_params={"temperature": float(simulator.params.values.get("adoption_noise_sigma", 0.05))},
            module_params={},
            meta={"calibrator": "snpe", "seed": seed}
        )
        if artifacts_dir:
            best_dir = os.path.join(artifacts_dir, "best")
            ensure_dir(best_dir)
            metrics = evaluator(simulator, fp, train_window)
            safe_json_dump(fp.to_dict(), os.path.join(best_dir, "fitted_params.json"))
            safe_json_dump(metrics, os.path.join(best_dir, "metrics.json"))
            safe_json_dump({"budget": budget, "trials": [], "best_score": metrics.get("RMSE_aggregate", None)}, os.path.join(artifacts_dir, "calibration_report.json"))
        return fp


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None):
    """
    Create a calibrator instance by name, optionally loading config from a JSON file for kwargs.

    # FIXED: Validate and filter kwargs using inspect.signature; fallback to default on invalid.
    # FIXED: Harden JSON load with try/except.
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
                    kwargs = cfg
        except Exception:
            kwargs = {}
    Ctor = CALIBRATOR_REGISTRY[name]
    try:
        import inspect
        sig = inspect.signature(Ctor.__init__)
        valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return Ctor(**valid)
    except Exception:
        return Ctor()


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).

    For simulation engines: use Simulation.run(start,end) + Simulation.evaluate(),
    read metrics from artifacts/results/, build dict with required keys.
    If micro-transitions unavailable, degrade gracefully or use placeholder values.

    # FIXED: Preserve and restore DISABLE_EARLY_STOP environment variable using try/finally
    # FIXED: Create a fresh Simulation per evaluation to avoid state carryover across trials.
    # FIXED: Pass evaluation window to Simulation.evaluate() to compute metrics on the requested interval.
    """
    pass
    # Create a fresh simulator to avoid state carryover
    reg_clone = simulator.params.clone()
    sim_eval = Simulation(params=reg_clone, artifacts_root=simulator.artifacts_root, data_dir=simulator.data_dir)
    adapter = ParamsAdapter(registry=sim_eval.params)
    # Apply params (clamped within bounds)
    adapter.apply(sim_eval, params)
    old_early = os.environ.get("DISABLE_EARLY_STOP", "0")
    try:
        os.environ["DISABLE_EARLY_STOP"] = "1"
        start, end = window
        sim_eval.run(start, end)
        metrics = sim_eval.evaluate(window=window)
        # Construct aggregate metrics
        rmse_agg = metrics.get("RMSE_overall_adoption", None)
        mae_agg = metrics.get("MAE_new_adoptions", None)
        # Brier and TransitionFit placeholders
        obs = sim_eval.state["observables"]
        y_hat = obs.get("adoption_rate_daily", [])
        # In absence of GT here, compare to itself -> 0
        brier = sum((p - r) ** 2 for p, r in zip(y_hat, y_hat)) / float(max(1, len(y_hat)))
        # Transition probabilities from micro if available
        transfit = {"P01": None, "P11": None, "P10": None, "P00": None}
        df = sim_eval.export_micro_transitions()
        if df is not None and not df.empty:
            p01 = float((df["prev"] == 0).sum())
            a01 = float(((df["prev"] == 0) & (df["will_adopt"] == 1)).sum())
            p11 = float((df["prev"] == 1).sum())
            a11 = float(((df["prev"] == 1) & (df["will_adopt"] == 1)).sum())
            transfit = {
                "P01": (a01 / p01) if p01 > 0 else None,
                "P11": (a11 / p11) if p11 > 0 else None,
                "P10": None,
                "P00": None,
            }
        out = {
            "RMSE_aggregate": rmse_agg,
            "MAE_aggregate": mae_agg,
            "Brier": brier,
            "TransitionFit": transfit,
        }
        # Persist
        safe_json_dump(out, os.path.join(sim_eval.paths["results"], "evaluation_metrics.json"))
        return out
    finally:
        os.environ["DISABLE_EARLY_STOP"] = old_early


# -----------------------------------------------------------------------------
# CLI and Main
# -----------------------------------------------------------------------------

def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments for the simulation program.
    """
    pass
    parser = argparse.ArgumentParser(description="Agent-based diffusion simulation with pluggable calibration.")
    parser.add_argument("--param-file", type=str, default="parameters.json", help="Path to parameters JSON file.")
    parser.add_argument("--parameter-defs", type=str, default="parameter_definitions.json", help="Path to parameter definitions JSON file.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override parameters: key=value (repeatable).")
    parser.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()), help="Calibrator to use.")
    parser.add_argument("--budget", type=int, default=5, help="Calibration budget (number of trials).")
    parser.add_argument("--calib-window", type=str, default="0:60", help="Training window 'start:end' (inclusive).")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts root directory.")
    parser.add_argument("--calibrator-config", type=str, default=None, help="Path to calibrator config JSON.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization step.")
    # FIXED: Add CLI toggles to control micro capture and IO size.
    parser.add_argument("--capture-micro", action="store_true", help="Enable micro-transition capture (disabled by default).")
    parser.add_argument("--save-micro-io", action="store_true", help="Persist micro transitions in IO artifacts (disabled by default).")
    # FIXED: Applied feedback snippet from parse_cli()
parser.add_argument("--micro-sample-frac", type=float, default=None, help="Sampling fraction for micro transitions (0.0-1.0).")
parser.add_argument("--save-heavy-io", action="store_true", help="Persist heavy per-agent arrays in IO (off by default).")
parser.add_argument("--io-agent-sample-n", type=int, default=None, help="Number of agent entries to include when sampling heavy arrays in IO.")
    # FIXED: Add heavy IO control flags
    parser.add_argument("--save-heavy-io", action="store_true", help="Persist heavy per-agent arrays in IO (off by default).")
    parser.add_argument("--io-agent-sample-n", type=int, default=None, help="Number of agent entries to include when sampling heavy arrays in IO.")
    return parser.parse_args(argv)


def main() -> None:
    """
    Entry point orchestrating the end-to-end workflow:
    parse_cli() -> load parameters -> seed -> instantiate Simulation -> baseline -> holdout split ->
    calibrator.fit() -> apply best -> rollout -> evaluate -> save results.

    # FIXED: Restore full simulation and remove stray non-Python text.
    # FIXED: Apply CLI overrides via ParameterRegistry.set_params() to respect registry safeguards.
    # FIXED: Apply heavy IO flags and persist parameters used.
    """
    pass
    args = parse_cli()

    # Load parameters and definitions
    raw_params = load_parameters_file(args.param_file)
    defs = load_parameter_definitions(args.parameter_defs, raw_params)

    # Initialize registry
    registry = ParameterRegistry(defs, raw_params)

    # CLI overrides
    overrides_dict: Dict[str, Any] = {}
    for s in args.overrides:
        k, v = parse_kv_override(s)
        overrides_dict[k] = v
    # Apply overrides via registry API
    warnings = registry.set_params(**overrides_dict)  # FIXED: Use registry.set_params
    if warnings:
        print("Override warnings:", warnings)

    # Apply micro toggles via registry to centralize config
    micro_updates: Dict[str, Any] = {}
    if args.capture_micro:
        micro_updates["capture_micro"] = True
    if args.save_micro_io:
        micro_updates["save_micro_io"] = True
    if args.micro_sample_frac is not None:
        micro_updates["micro_sample_frac"] = float(args.micro_sample_frac)
    if micro_updates:
        micro_warn = registry.set_params(**micro_updates)
        if micro_warn:
            print("Micro toggle warnings:", micro_warn)

    # FIXED: Heavy IO flags
    io_updates: Dict[str, Any] = {}
    if args.save_heavy_io:
        io_updates["save_heavy_io"] = True
    if args.io_agent_sample_n is not None:
        io_updates["io_agent_sample_n"] = int(args.io_agent_sample_n)
    if io_updates:
        io_warn = registry.set_params(**io_updates)
        if io_warn:
            print("IO toggle warnings:", io_warn)

    # Seed
    # FIXED: Balanced parentheses
    seed_all(int(registry.values.get("random_seed", 42)))

    # Paths
    artifacts_root = args.artifacts_dir
    ensure_dir(artifacts_root)
    params_used_path = os.path.join(artifacts_root, "results", "parameters_used.json")
    ensure_dir(os.path.dirname(params_used_path))

    # Persist final parameters actually used
    safe_json_dump(registry.export_used(), params_used_path)

    # Build Simulation
    # QUICK_TEST gating: keep runtime manageable by default
    quick = os.environ.get("QUICK_TEST", "0") == "1"
    if quick:
        # Reduce heavy parameters for speed; not persisted or frozen, only limited effect
        registry.values["population_size"] = int(min(int(registry.values.get("population_size", 2000)), 800))
        registry.values["time_horizon_days"] = int(min(int(registry.values.get("time_horizon_days", 120)), 60))

    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
    DATA_PATH = os.environ.get("DATA_PATH", "")
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if (PROJECT_ROOT and DATA_PATH) else ""

    sim = Simulation(params=registry, artifacts_root=artifacts_root, data_dir=DATA_DIR)

    # Baseline run to produce synthetic ground truth if external data not present
    gt = sim._load_ground_truth()
    if not gt:
        # Use baseline as ground truth
        horizon = int(registry.values.get("time_horizon_days", 120))
        sim.run(0, horizon - 1)
        obs = sim.state["observables"]
        gt = {
            "overall_adoption_rate": obs.get("adoption_rate_daily", []),
            "new_adoptions": obs.get("new_adoptions_daily", []),
            "cumulative_adoptions": obs.get("cumulative_adoptions", []),
        }
        safe_json_dump(gt, os.path.join(sim.paths["results"], "ground_truth.json"))
        # Reset sim for calibration to avoid data leakage from baseline
        sim = Simulation(params=registry, artifacts_root=artifacts_root, data_dir=DATA_DIR)

    # Parse calibration window
    try:
        start_s, end_s = args.calib_window.split(":")
        calib_window = (int(start_s), int(end_s))
    except Exception:
        calib_window = (0, int(registry.values.get("time_horizon_days", 120)) // 2)

    # Calibrator
    calibrator = get_calibrator(args.calibrator, args.calibrator_config)
    adapter = ParamsAdapter(param_defs_path=args.parameter_defs, registry=registry)
    calib_dir = os.path.join(sim.paths["calibration"], args.calibrator)
    ensure_dir(calib_dir)

    # Calibration (skip or reduce under QUICK_TEST)
    budget = int(args.budget)
    if quick:
        budget = min(budget, 3)
    best_params = calibrator.fit(
        bundle=None,
        simulator=sim,
        evaluator=evaluate_params,
        train_window=calib_window,
        seed=int(registry.values.get("random_seed", 42)),
        budget=budget,
        artifacts_dir=calib_dir,
        params_adapter=adapter,
    )

    # Apply best params
    if best_params is not None:
        adapter.apply(sim, best_params)

    # Final rollout over the full horizon
    horizon = int(registry.values.get("time_horizon_days", 120))
    sim.run(0, horizon - 1)
    # Evaluate vs ground truth
    metrics = sim.evaluate(gt=gt)
    print("Final metrics:", metrics)

    # Save outputs and IO
    sim.save_results("final")
    sim.save_all_io()
    if not args.no_visualize:
        sim.visualize()

    print("Simulation completed. Artifacts at:", artifacts_root)


# Execute main for both direct execution and sandbox wrapper invocation
main()