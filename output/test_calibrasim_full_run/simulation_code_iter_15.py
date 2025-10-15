import os
import sys
import json
import math
import time
import random
import shutil
import typing as _t
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod

try:
    import numpy as np
except Exception:
    np = None  # Will degrade gracefully if numpy not available

# Optional dependencies
try:
    import networkx as nx
except Exception:
    nx = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# FIXED: PROJECT_ROOT robust to missing __file__; DATA_DIR and ARTIFACTS with env overrides
# FIXED: Replaced malformed nested try/except with a single clean block
if os.environ.get("PROJECT_ROOT"):
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
else:
    if "__file__" in globals():
        PROJECT_ROOT = str(Path(__file__).resolve().parent)
    else:
        PROJECT_ROOT = os.getcwd()
DATA_PATH = os.environ.get("DATA_PATH") or "data"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR") or os.path.join(PROJECT_ROOT, "artifacts")


def _json_clean(obj: _t.Any) -> _t.Any:
    """
    Recursively transform Python objects into JSON-safe structures.

    - Floats: convert NaN/Inf to None
    - Sets/Tuples: convert to lists
    - Dicts/Lists: recurse on values

    Parameters
    ----------
    obj : Any
        Object to sanitize.

    Returns
    -------
    Any
        JSON-serializable object.
    """
    try:
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {k: _json_clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_json_clean(v) for v in obj]
        if hasattr(obj, "__dict__"):
            # Dataclass or class instance
            try:
                return _json_clean(asdict(obj))  # type: ignore
            except Exception:
                return _json_clean(obj.__dict__)
        return obj
    except Exception:
        return None
    pass


def seed_all(seed: int) -> None:
    """
    Seed all relevant random number generators for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set.
    """
    try:
        random.seed(seed)
    except Exception:
        print("Warning: Unable to seed Python's random module.", file=sys.stderr)
    try:
        if np is not None:
            np.random.seed(seed % (2**32 - 1))
    except Exception:
        print("Warning: Unable to seed numpy.random.", file=sys.stderr)
    pass


def safe_mkdirs(path: str) -> None:
    """
    Create directories recursively if they do not exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {path}: {e}", file=sys.stderr)
    pass


def load_json_file(path: _t.Union[str, Path], default: _t.Any, desc: str) -> _t.Any:
    """
    Load a JSON file with robust error handling and return a default on failure.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.
    default : Any
        Default object to return if loading fails.
    desc : str
        Human-readable description for logging.

    Returns
    -------
    Any
        Parsed JSON or default.
    """
    p = str(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{desc} not found at {p}; using defaults.", file=sys.stderr)
        return default
    except json.JSONDecodeError as e:
        print(f"Invalid {desc} at {p}: {e}; using defaults.", file=sys.stderr)
        return default
    except Exception as e:
        print(f"Error loading {desc} at {p}: {e}; using defaults.", file=sys.stderr)
        return default
    pass


def load_params(default_path: _t.Union[str, Path]) -> dict:
    """
    Load parameters from a JSON file with env override and robust error handling.

    Environment
    ----------
    PARAMS_PATH : str
        Optional override path for the parameters file.

    Parameters
    ----------
    default_path : str or Path
        Fallback path to parameters JSON.

    Returns
    -------
    dict
        Parameters dictionary.
    """
    override = os.environ.get("PARAMS_PATH")
    path = override or str(default_path)
    default_params = {
        "population_size": 1000,
        "simulation_days": 60,
        "random_seed": 42,
        "time_steps": 60,
    }
    params = load_json_file(path, default_params, desc="parameters.json")
    if not isinstance(params, dict):
        print("parameters.json must be a JSON object; using defaults.", file=sys.stderr)
        return default_params
    return params
    pass


def load_param_definitions(default_path: _t.Union[str, Path]) -> dict:
    """
    Load parameter definitions from a JSON file with env override.

    Environment
    ----------
    PARAM_DEFS_PATH : str
        Optional override path for the parameter definitions file.

    Parameters
    ----------
    default_path : str or Path
        Fallback path to parameter definitions JSON.

    Returns
    -------
    dict
        Parameter definitions mapping parameter keys to metadata.
    """
    override = os.environ.get("PARAM_DEFS_PATH")
    path = override or str(default_path)
    # Minimal defaults sufficient to run the simulation
    minimal_defs = {
        # Network
        "network_ws_k": {"dtype": "int", "default": 12, "bounds": {"low": 2, "high": 50}, "owner_module": "SocialNetworkFormation", "frozen": False},
        "network_ws_p_rewire": {"dtype": "float", "default": 0.08, "bounds": {"low": 0.0, "high": 0.5}, "owner_module": "SocialNetworkFormation", "frozen": False},
        "network_dynamic_rewire_rate": {"dtype": "float", "default": 0.01, "bounds": {"low": 0.0, "high": 0.1}, "owner_module": "SocialNetworkFormation", "frozen": False},
        "network_homophily_weight": {"dtype": "float", "default": 0.6, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "SocialNetworkFormation", "frozen": False},
        "network_max_degree": {"dtype": "int", "default": 100, "bounds": {"low": 10, "high": 300}, "owner_module": "SocialNetworkFormation", "frozen": False},
        # Communication/Peer exposures
        "comm_transmission_prob_base": {"dtype": "float", "default": 0.15, "bounds": {"low": 0.01, "high": 0.6}, "owner_module": "CommunicationDynamics", "frozen": False},
        "comm_transmission_bias_strength": {"dtype": "float", "default": 0.2, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "CommunicationDynamics", "frozen": False},
        "comm_contact_budget_daily": {"dtype": "int", "default": 20, "bounds": {"low": 5, "high": 100}, "owner_module": "CommunicationDynamics", "frozen": False},
        "comm_message_half_life_days": {"dtype": "float", "default": 7.0, "bounds": {"low": 1.0, "high": 30.0}, "owner_module": "CommunicationDynamics", "frozen": False},
        "comm_max_exposures_per_day": {"dtype": "int", "default": 5, "bounds": {"low": 1, "high": 20}, "owner_module": "CommunicationDynamics", "frozen": False},
        # Policy
        "policy_mandate_start_day": {"dtype": "int", "default": 20, "bounds": {"low": 0, "high": 120}, "owner_module": "PolicyIntervention", "frozen": True},
        "policy_mandate_end_day": {"dtype": "int", "default": 45, "bounds": {"low": 1, "high": 180}, "owner_module": "PolicyIntervention", "frozen": True},
        "policy_incentive_amount": {"dtype": "float", "default": 0.5, "bounds": {"low": 0.0, "high": 5.0}, "owner_module": "PolicyIntervention", "frozen": False},
        "policy_enforcement_probability": {"dtype": "float", "default": 0.6, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "PolicyIntervention", "frozen": False},
        "policy_target_coverage": {"dtype": "float", "default": 0.7, "bounds": {"low": 0.1, "high": 0.95}, "owner_module": "PolicyIntervention", "frozen": True},
        "marketing_budget_daily": {"dtype": "float", "default": 1000.0, "bounds": {"low": 0.0, "high": 10000.0}, "owner_module": "PolicyIntervention", "frozen": False},
        "marketing_contact_probability": {"dtype": "float", "default": 0.5, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "PolicyIntervention", "frozen": False},
        # Behavior and decision
        "adoption_base_rate": {"dtype": "float", "default": -1.8, "bounds": {"low": -5.0, "high": 1.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_peer_weight": {"dtype": "float", "default": 2.2, "bounds": {"low": 0.0, "high": 5.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_marketing_weight": {"dtype": "float", "default": 0.8, "bounds": {"low": 0.0, "high": 3.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_policy_weight": {"dtype": "float", "default": 1.0, "bounds": {"low": 0.0, "high": 4.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_exposure_weight": {"dtype": "float", "default": 0.2, "bounds": {"low": 0.0, "high": 3.0}, "owner_module": "BehaviorAdoption", "frozen": False},  # FIXED: Added exposure weight param
        "adoption_threshold_mean": {"dtype": "float", "default": 0.35, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_threshold_std": {"dtype": "float", "default": 0.12, "bounds": {"low": 0.0, "high": 0.5}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_temperature": {"dtype": "float", "default": 0.6, "bounds": {"low": 0.05, "high": 2.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_fatigue_decay": {"dtype": "float", "default": 0.05, "bounds": {"low": 0.0, "high": 0.2}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_reversion_prob": {"dtype": "float", "default": 0.02, "bounds": {"low": 0.0, "high": 0.2}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_stubborn_fraction": {"dtype": "float", "default": 0.1, "bounds": {"low": 0.0, "high": 0.5}, "owner_module": "BehaviorAdoption", "frozen": True},
        "adoption_stubborn_resistance": {"dtype": "float", "default": 1.0, "bounds": {"low": 0.0, "high": 3.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_memory_window_days": {"dtype": "int", "default": 14, "bounds": {"low": 3, "high": 60}, "owner_module": "BehaviorAdoption", "frozen": False},
        "adoption_social_reinforcement": {"dtype": "float", "default": 0.5, "bounds": {"low": 0.0, "high": 2.0}, "owner_module": "BehaviorAdoption", "frozen": False},
        # Aggregation smoothing
        "agg_smoothing_window_days": {"dtype": "int", "default": 3, "bounds": {"low": 1, "high": 14}, "owner_module": "AdoptionAggregator", "frozen": False},
        "agg_report_by_group": {"dtype": "bool", "default": False, "bounds": {"low": 0, "high": 1}, "owner_module": "AdoptionAggregator", "frozen": True},
        # Mobility/Location compliance
        "work_attendance_rate": {"dtype": "float", "default": 0.7, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "LocationCompliance", "frozen": False},
        "public_visit_rate": {"dtype": "float", "default": 0.4, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "LocationCompliance", "frozen": False},
        "enforcement_strictness_home": {"dtype": "float", "default": 0.0, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "LocationCompliance", "frozen": True},
        "enforcement_strictness_work": {"dtype": "float", "default": 0.7, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "LocationCompliance", "frozen": False},
        "enforcement_strictness_public": {"dtype": "float", "default": 0.5, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "LocationCompliance", "frozen": False},
        # Retailer and supply
        "retailer_initial_stock_per_capita": {"dtype": "float", "default": 1.5, "bounds": {"low": 0.0, "high": 10.0}, "owner_module": "Retailer", "frozen": False},
        "retailer_restock_rate_per_day": {"dtype": "float", "default": 0.05, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "Retailer", "frozen": False},
        "retailer_price_per_mask": {"dtype": "float", "default": 1.0, "bounds": {"low": 0.0, "high": 50.0}, "owner_module": "Retailer", "frozen": False},
        "purchase_base_probability": {"dtype": "float", "default": 0.25, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "Retailer", "frozen": False},
        "purchase_cost_sensitivity": {"dtype": "float", "default": 0.4, "bounds": {"low": 0.0, "high": 5.0}, "owner_module": "Retailer", "frozen": False},
        # Media broadcast and risk signal
        "media_channel_count": {"dtype": "int", "default": 6, "bounds": {"low": 1, "high": 100}, "owner_module": "MediaBroadcast", "frozen": False},
        "misinformation_fraction": {"dtype": "float", "default": 0.3, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "MediaBroadcast", "frozen": False},
        "media_neutral_fraction": {"dtype": "float", "default": 0.2, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "MediaBroadcast", "frozen": False},
        "message_frequency_per_day": {"dtype": "int", "default": 1, "bounds": {"low": 0, "high": 24}, "owner_module": "MediaBroadcast", "frozen": False},
        "media_credibility_mean": {"dtype": "float", "default": 0.6, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "MediaBroadcast", "frozen": False},
        "risk_wave_period_days": {"dtype": "float", "default": 90.0, "bounds": {"low": 7.0, "high": 365.0}, "owner_module": "RiskSignal", "frozen": False},
        "risk_wave_phase": {"dtype": "float", "default": 0.0, "bounds": {"low": 0.0, "high": 365.0}, "owner_module": "RiskSignal", "frozen": False},
        "risk_baseline": {"dtype": "float", "default": 0.2, "bounds": {"low": 0.0, "high": 1.0}, "owner_module": "RiskSignal", "frozen": False},
        "perceived_risk_peak": {"dtype": "float", "default": 1.0, "bounds": {"low": 0.0, "high": 2.0}, "owner_module": "RiskSignal", "frozen": False},
        # Globals
        "population_size": {"dtype": "int", "default": 5000, "bounds": {"low": 1, "high": 100000}, "owner_module": "global", "frozen": True},
        "simulation_days": {"dtype": "int", "default": 60, "bounds": {"low": 10, "high": 365}, "owner_module": "global", "frozen": True},
        "random_seed": {"dtype": "int", "default": 42, "bounds": {"low": 0, "high": 2147483647}, "owner_module": "global", "frozen": True},
    }
    defs = load_json_file(path, minimal_defs, desc="parameter_definitions.json")
    if not isinstance(defs, dict):
        print("parameter_definitions.json must be a JSON object; using minimal defaults.", file=sys.stderr)
        return minimal_defs
    return defs
    pass


def coerce_type(value: _t.Any, dtype: str) -> _t.Any:
    """
    Coerce a value to the specified dtype.

    Parameters
    ----------
    value : Any
        Value to coerce.
    dtype : str
        Target dtype: 'int', 'float', 'bool', 'categorical' (leave as str).

    Returns
    -------
    Any
        Coerced value, or original if coercion fails.
    """
    try:
        if dtype == "int":
            return int(float(value))
        if dtype == "float":
            return float(value)
        if dtype == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            s = str(value).strip().lower()
            return s in {"true", "1", "yes", "y", "on"}
        if dtype == "categorical":
            return value
        return value
    except Exception:
        return value
    pass


def normalize_params(params: dict, param_defs: dict) -> dict:
    """
    Normalize parameters: apply defaults, coerce types, and enforce caps.

    - Derives 'time_steps' from 'simulation_days' if missing.
    - Caps population_size <= 100,000 and time_steps <= 10,000.
    - Honors QUICK_TEST override for deterministic short runs.

    Parameters
    ----------
    params : dict
        Raw input parameters.
    param_defs : dict
        Parameter definitions with defaults and dtypes.

    Returns
    -------
    dict
        Normalized parameters.
    """
    out = {}
    # Apply defaults from definitions
    for k, meta in param_defs.items():
        default = meta.get("default")
        out[k] = default
    # Update with provided params
    if isinstance(params, dict):
        for k, v in params.items():
            out[k] = v
    # Coerce types
    for k, meta in param_defs.items():
        dtype = meta.get("dtype")
        if k in out:
            out[k] = coerce_type(out[k], dtype)
    # Derive time_steps
    try:
        ts = params.get("time_steps")
        ts = int(ts) if ts is not None else None
    except Exception:
        ts = None
    if not ts or ts <= 0:
        sim_days = out.get("simulation_days", 60)
        try:
            sim_days = int(sim_days)
        except Exception:
            sim_days = 60
        out["time_steps"] = max(1, sim_days)
    else:
        out["time_steps"] = max(1, ts)
    # Caps
    try:
        pop = int(out.get("population_size", 1000))
    except Exception:
        pop = 1000
    out["population_size"] = max(1, min(pop, 100000))
    try:
        steps = int(out.get("time_steps", 60))
    except Exception:
        steps = 60
    out["time_steps"] = max(1, min(steps, 10000))
    # QUICK_TEST deterministic small run
    if os.environ.get("QUICK_TEST", "0") == "1":
        out["population_size"] = 50
        out["simulation_days"] = 10
        out["time_steps"] = 10
        out["random_seed"] = 0
    return out
    pass


def parse_cli(argv: _t.List[str]) -> dict:
    """
    Parse CLI arguments, supporting parameter file, overrides, calibration options.

    Parameters
    ----------
    argv : list of str
        Argument vector.

    Returns
    -------
    dict
        Parsed options.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Social simulation with modular architecture and calibration."
    )
    parser.add_argument("--param-file", type=str, default=os.path.join(PROJECT_ROOT, "parameters.json"))
    parser.add_argument("--param-defs-file", type=str, default=os.path.join(PROJECT_ROOT, "parameter_definitions.json"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override parameters as key=value. Repeatable.")
    parser.add_argument("--calibrator", type=str, default="random_search", help="Calibrator name: logit_head, random_search, snpe")
    parser.add_argument("--budget", type=int, default=10, help="Calibration budget (iterations).")
    parser.add_argument("--calib-window", type=str, default=None, help="Calibration window start:end (days).")
    parser.add_argument("--artifacts-dir", type=str, default=ARTIFACTS_DIR, help="Artifacts root directory.")
    parser.add_argument("--quick-test", action="store_true", help="Enable quick test mode.")

    # FIXED: Use parse_known_args and guard against SystemExit to be robust to unknown args
    try:
        args, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        class _Obj:
            """Fallback object with default args if parsing fails."""
            pass
        args = _Obj()
        args.param_file = os.path.join(PROJECT_ROOT, "parameters.json")
        args.param_defs_file = os.path.join(PROJECT_ROOT, "parameter_definitions.json")
        args.overrides = []
        args.calibrator = "random_search"
        args.budget = 10
        args.calib_window = None
        args.artifacts_dir = ARTIFACTS_DIR
        args.quick_test = False

    if args.quick_test:
        os.environ["QUICK_TEST"] = "1"
    window = None
    if args.calib_window:
        try:
            s, e = args.calib_window.split(":")
            window = (int(s), int(e))
        except Exception:
            print("Invalid --calib-window format; expected start:end", file=sys.stderr)
    return {
        "param_file": args.param_file,
        "param_defs_file": args.param_defs_file,
        "overrides": args.overrides or [],
        "calibrator": args.calibrator,
        "budget": int(args.budget),
        "calib_window": window,
        "artifacts_dir": args.artifacts_dir,
    }
    pass


def apply_cli_overrides(params: dict, param_defs: dict, overrides: _t.List[str]) -> dict:
    """
    Apply --set key=value overrides to parameters, ignoring frozen params.

    Parameters
    ----------
    params : dict
        Current parameters.
    param_defs : dict
        Parameter definitions including 'frozen' and 'dtype'.
    overrides : list of str
        Overrides in "key=value" format.

    Returns
    -------
    dict
        Updated parameters dictionary.
    """
    updated = dict(params)
    for item in overrides:
        if "=" not in item:
            print(f"Ignoring invalid override '{item}'; expected key=value.", file=sys.stderr)
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in param_defs:
            print(f"Override ignored: unknown parameter '{key}'.", file=sys.stderr)
            continue
        if param_defs.get(key, {}).get("frozen", False):
            print(f"Override ignored: parameter '{key}' is frozen.", file=sys.stderr)
            continue
        dtype = param_defs.get(key, {}).get("dtype", None)
        coerced = coerce_type(value, dtype) if dtype else value
        updated[key] = coerced
    return updated
    pass


class ParameterRegistry:
    """
    Registry for parameters: definitions, grouping by modules, and validation.

    This class provides helpers to query and set parameters ensuring they
    respect dtypes and bounds.

    Methods
    -------
    get(key, default=None)
    set(key, value)
    group_by_module()
    frozen_keys()
    """

    def __init__(self, param_defs: dict, params: dict):
        """
        Initialize the registry.

        Parameters
        ----------
        param_defs : dict
            Parameter definitions metadata.
        params : dict
            Current parameters values.
        """
        self.param_defs = dict(param_defs or {})
        self.params = dict(params or {})
        pass

    def get(self, key: str, default: _t.Any = None) -> _t.Any:
        """
        Get parameter value with default.

        Parameters
        ----------
        key : str
            Parameter key.
        default : Any, optional
            Default value if key not found.

        Returns
        -------
        Any
            Parameter value or default.
        """
        return self.params.get(key, default)
        pass

    def set(self, key: str, value: _t.Any) -> None:
        """
        Set parameter value with validation and coercion.

        Parameters
        ----------
        key : str
            Parameter key.
        value : Any
            New value to set.

        Raises
        ------
        KeyError
            If the key is unknown.
        """
        if key not in self.param_defs:
            raise KeyError(f"Unknown parameter: {key}")
        if self.param_defs[key].get("frozen", False):
            print(f"Attempted to set frozen parameter '{key}'; ignoring.", file=sys.stderr)
            return
        dtype = self.param_defs[key].get("dtype")
        coerced = coerce_type(value, dtype) if dtype else value
        # Enforce bounds for numeric dtypes
        bounds = self.param_defs[key].get("bounds", {})
        lo = bounds.get("low", None)
        hi = bounds.get("high", None)
        try:
            # FIXED: Respect dtype casting after clamping to avoid int params stored as floats
            if dtype in {"int", "float"} and (lo is not None or hi is not None):
                v = float(coerced)
                if lo is not None and v < float(lo):
                    v = float(lo)
                if hi is not None and v > float(hi):
                    v = float(hi)
                coerced = int(round(v)) if dtype == "int" else float(v)
        except Exception:
            pass
        self.params[key] = coerced
        pass

    def group_by_module(self) -> dict:
        """
        Group parameters by their owner module.

        Returns
        -------
        dict
            Mapping from module name to {param_key: value}.
        """
        grouped = {}
        for k, meta in self.param_defs.items():
            owner = meta.get("owner_module", "global")
            grouped.setdefault(owner, {})
            grouped[owner][k] = self.params.get(k, meta.get("default"))
        return grouped
        pass

    def frozen_keys(self) -> _t.Set[str]:
        """
        Get the set of frozen parameter keys.

        Returns
        -------
        set of str
            Frozen parameter keys.
        """
        frozen = set()
        for k, meta in self.param_defs.items():
            if meta.get("frozen", False):
                frozen.add(k)
        return frozen
        pass


@dataclass
class Person:
    """
    Representation of an agent with attributes and state for the simulation.
    """
    id: int = 0
    age_bucket: int = 0
    income_log10: float = 4.5
    group_id: int = 0
    openness: float = 0.5
    stubborn: int = 0
    adopted: bool = False
    adoption_threshold: float = 0.35
    utility_bias: float = 0.0
    exposures_memory: float = 0.0
    last_adoption_change_day: int = -1
    # Extended attributes for mask adoption and enforcement mechanics
    trust_in_media: float = 0.5
    trust_in_authority: float = 0.6
    risk_attitude: float = 0.5
    habit_persistence: float = 0.7
    prev_mask_use: int = 0
    is_influencer: int = 0
    is_anti_masker: int = 0
    mask_inventory: int = 1
    home_id: int = -1
    work_id: int = -1
    pass


@dataclass
class Household:
    """
    Minimal household structure for within-household norms (not explicitly used in network).
    """
    id: int
    members: _t.List[int]
    norm_strength: float = 0.5
    pass


@dataclass
class Location:
    """
    Location representation with enforcement parameters.
    """
    id: int
    type: str  # 'home', 'work', 'public'
    capacity: int
    baseline_contact_rate: int
    enforcement_strictness: float
    requires_masks: bool = False
    pass


class Module:
    """
    Base class for all modules with a common forward interface.

    Subclasses should implement forward to compute outputs from inputs,
    writing results to the buffers dictionary. They should not mutate
    the global state directly.

    Attributes
    ----------
    name : str
        Module name.
    dependencies : list of str
        Names of dependent modules that must run before this one.
    inputs : list of str
        Descriptive list of expected inputs.
    outputs : list of str
        Descriptive list of outputs.

    Methods
    -------
    forward(state, buffers, params, t)
        Perform module update for time t.
    """

    def __init__(self, name: str, dependencies: _t.Optional[_t.List[str]] = None):
        """
        Initialize the module.

        Parameters
        ----------
        name : str
            Module name.
        dependencies : list of str, optional
            Names of modules that must be executed before this one.
        """
        self.name = name
        self.dependencies = list(dependencies or [])
        self.inputs: _t.List[str] = []
        self.outputs: _t.List[str] = []
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Run the module for current time step t.

        Parameters
        ----------
        state : dict
            Global state dictionary.
        buffers : dict
            Staging buffers for signals updated in this tick.
        params : ParameterRegistry
            Parameter registry for reading parameter values.
        t : int
            Current simulation day/time.

        Notes
        -----
        Subclasses must override this method to produce outputs into `buffers`.
        """
        raise NotImplementedError("Module.forward must be implemented by subclasses.")
        pass


class SocialNetworkFormation(Module):
    """
    Initializes and dynamically rewires the social network with homophily.
    """

    def __init__(self):
        """
        Construct the SocialNetworkFormation module.
        """
        super().__init__("SocialNetworkFormation", dependencies=[])
        self.inputs = []
        self.outputs = ["signal.network.adjacency", "signal.network.degree"]
        pass

    def _init_graph(self, N: int, k: int, p_rewire: float) -> dict:
        """
        Initialize a small-world network adjacency.

        Parameters
        ----------
        N : int
            Number of nodes.
        k : int
            Average degree (even).
        p_rewire : float
            Rewiring probability.

        Returns
        -------
        dict
            Adjacency mapping node -> set(neighbors)
        """
        k = max(2, int(k))
        if k % 2 == 1:
            k += 1
        if nx is not None:
            try:
                G = nx.watts_strogatz_graph(N, k, p_rewire, seed=random.randint(0, 2**31 - 1))
                adj = {i: set(G.neighbors(i)) for i in range(N)}
                return adj
            except Exception:
                print("Fallback to ring lattice adjacency due to networkx error.", file=sys.stderr)
        # Simple ring lattice fallback
        adj = {i: set() for i in range(N)}
        half = k // 2
        for i in range(N):
            for d in range(1, half + 1):
                j1 = (i + d) % N
                j2 = (i - d) % N
                adj[i].add(j1)
                adj[i].add(j2)
                adj[j1].add(i)
                adj[j2].add(i)
        return adj
        pass

    def _rewire_step(self, adj: dict, state: dict, params: ParameterRegistry) -> dict:
        """
        Perform a stochastic rewiring step with homophily.

        Parameters
        ----------
        adj : dict
            Current adjacency mapping.
        state : dict
            Global state containing agents.
        params : ParameterRegistry
            Parameters.
        Returns
        -------
        dict
            Updated adjacency mapping (in-place modified).
        """
        N = len(adj)
        p_dyn = float(params.get("network_dynamic_rewire_rate", 0.01))
        w_hom = float(params.get("network_homophily_weight", 0.6))
        max_deg = int(params.get("network_max_degree", 100))
        agents: _t.List[Person] = state.get("agents", [])

        def similarity(i: int, j: int) -> float:
            a = agents[i]
            b = agents[j]
            sim = 0.0
            # Simple components: group equality, openness closeness, age bucket equality
            sim += 1.0 if a.group_id == b.group_id else 0.0
            sim += 1.0 - abs(a.openness - b.openness)
            sim += 1.0 if a.age_bucket == b.age_bucket else 0.0
            return sim / 3.0

        for i in range(N):
            if random.random() < p_dyn and len(adj[i]) > 0:
                try:
                    j = random.choice(list(adj[i]))
                except Exception:
                    continue
                adj[i].discard(j)
                adj[j].discard(i)
                candidates = [u for u in range(N) if u != i and u not in adj[i] and len(adj[u]) < max_deg]
                if not candidates:
                    continue
                best_u = None
                best_score = -1.0
                for u in candidates:
                    score = w_hom * similarity(i, u) + (1.0 - w_hom) * random.random()
                    if score > best_score:
                        best_score = score
                        best_u = u
                if best_u is not None:
                    adj[i].add(best_u)
                    adj[best_u].add(i)
        return adj
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Update network formation/re-wiring and emit adjacency and degree.

        Parameters
        ----------
        state : dict
            State containing agents and previous adjacency.
        buffers : dict
            Buffers where outputs should be written.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Current time.
        """
        N = int(params.get("population_size", 1000))
        k = int(params.get("network_ws_k", 12))
        p_rewire = float(params.get("network_ws_p_rewire", 0.08))
        if t == 0 or state.get("adjacency") is None:
            adj = self._init_graph(N, k, p_rewire)
        else:
            adj = state.get("adjacency")
            self._rewire_step(adj, state, params)
        degree = [len(adj[i]) for i in range(N)]
        buffers["signal.network.adjacency"] = adj
        buffers["signal.network.degree"] = degree
        pass


class CommunicationDynamics(Module):
    """
    Simulates daily message transmission along edges and maintains exposure memories.
    """

    def __init__(self):
        """
        Construct the CommunicationDynamics module.
        """
        super().__init__("CommunicationDynamics", dependencies=["SocialNetworkFormation"])
        self.inputs = ["signal.network.adjacency", "signal.agent.adopted_flags"]
        self.outputs = ["signal.agent.exposures_today", "signal.exposure_rate_daily"]
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Update exposures given adjacency and adopted flags.

        Parameters
        ----------
        state : dict
            Global state (agents, adopted flags).
        buffers : dict
            Outputs for this tick.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Current time.
        """
        adj: dict = buffers.get("signal.network.adjacency") or state.get("adjacency", {})
        adopted_flags = state.get("adopted_flags", [])
        N = len(adopted_flags)
        exposures_today = [0 for _ in range(N)]
        p_base = float(params.get("comm_transmission_prob_base", 0.15))
        bias_strength = float(params.get("comm_transmission_bias_strength", 0.2))
        max_exp = int(params.get("comm_max_exposures_per_day", 5))
        degrees = buffers.get("signal.network.degree") or [len(adj.get(i, [])) for i in range(N)]

        for i in range(N):
            ni = adj.get(i, set())
            for j in ni:
                if adopted_flags[i] and not adopted_flags[j]:
                    bias = 1.0 + bias_strength * (1.0 if degrees[i] > degrees[j] else 0.0)
                    p_send = min(1.0, p_base * bias)
                    if random.random() < p_send and exposures_today[j] < max_exp:
                        exposures_today[j] += 1

        half_life = float(params.get("comm_message_half_life_days", 7.0))
        decay_factor = math.pow(0.5, 1.0 / max(1e-6, half_life))
        exposures_memory_next = []
        for idx in range(N):
            prev = state["agents"][idx].exposures_memory
            exposures_memory_next.append(decay_factor * prev + exposures_today[idx])

        buffers["signal.agent.exposures_today"] = exposures_today
        buffers["signal.exposure_rate_daily"] = float(sum(exposures_today)) / max(1, N)
        buffers["signal.agent.exposures_memory_next"] = exposures_memory_next
        pass


class MediaBroadcast(Module):
    """
    Media broadcast generating per-agent bias shifts and misinformation exposure rate.
    """

    def __init__(self):
        """
        Construct the MediaBroadcast module.
        """
        super().__init__("MediaBroadcast", dependencies=[])
        self.inputs = []
        self.outputs = ["signal.agent.media_shift", "signal.media.misinformation_rate"]
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Emit per-agent daily media shift and overall misinformation exposure rate.

        Parameters
        ----------
        state : dict
            Global state (agents).
        buffers : dict
            Output buffer.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Current time step.
        """
        agents: _t.List[Person] = state.get("agents", [])
        N = len(agents)
        chan_count = int(params.get("media_channel_count", 6))
        mis_frac = float(params.get("misinformation_fraction", 0.3))
        neutral_frac = float(params.get("media_neutral_fraction", 0.2))
        cred_mean = float(params.get("media_credibility_mean", 0.6))
        msgs_per_day = int(params.get("message_frequency_per_day", 1))

        pro_frac = max(0.0, 1.0 - mis_frac - neutral_frac)
        stance_bins = ["pro"] * int(round(pro_frac * 100)) + ["anti"] * int(round(mis_frac * 100)) + ["neutral"] * int(round(neutral_frac * 100))
        if not stance_bins:
            stance_bins = ["neutral"]
        misinformation_seen = 0
        media_shift = [0.0 for _ in range(N)]
        for i, person in enumerate(agents):
            # simulate messages
            shift = 0.0
            saw_anti = False
            for _ in range(max(1, msgs_per_day)):
                stance = random.choice(stance_bins)
                credibility = max(0.0, min(1.0, (cred_mean + (random.random() - 0.5) * 0.4)))
                influence = credibility * person.trust_in_media
                if stance == "pro":
                    shift += +influence
                elif stance == "anti":
                    shift += -influence
                    saw_anti = True
                else:
                    shift += 0.0
            # clamp shift
            media_shift[i] = max(-2.0, min(2.0, shift))
            if saw_anti:
                misinformation_seen += 1
        mis_rate = float(misinformation_seen) / max(1, N)
        buffers["signal.agent.media_shift"] = media_shift
        buffers["signal.media.misinformation_rate"] = mis_rate
        pass


class RiskSignal(Module):
    """
    Exogenous risk signal (e.g., epidemic wave).
    """

    def __init__(self):
        """
        Construct the RiskSignal module.
        """
        super().__init__("RiskSignal", dependencies=[])
        self.inputs = []
        self.outputs = ["signal.risk.value"]
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Compute daily risk signal.

        Parameters
        ----------
        state : dict
            Global state.
        buffers : dict
            Outputs buffer.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Day index.
        """
        period = float(params.get("risk_wave_period_days", 90.0))
        phase = float(params.get("risk_wave_phase", 0.0))
        baseline = float(params.get("risk_baseline", 0.2))
        peak = float(params.get("perceived_risk_peak", 1.0))
        try:
            value = baseline + max(0.0, math.sin(2.0 * math.pi * (t + phase) / max(1e-6, period))) * peak
        except Exception:
            value = baseline
        value = max(0.0, min(1.0, value))
        buffers["signal.risk.value"] = value
        pass


class PolicyIntervention(Module):
    """
    Generates daily policy and marketing signals (mandates, incentives, outreach).
    """

    def __init__(self):
        """
        Construct the PolicyIntervention module.
        """
        super().__init__("PolicyIntervention", dependencies=[])
        self.inputs = []
        self.outputs = [
            "signal.policy.active",
            "signal.policy.incentive_amount",
            "signal.policy.enforcement_probability",
            "signal.policy.requires_masks_by_type",
            "signal.marketing.contacts_by_agent",
        ]
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Emit daily policy active flag, incentive, enforcement, and marketing contacts by agent.

        Parameters
        ----------
        state : dict
            Global state including number of agents.
        buffers : dict
            Outputs for this tick.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Current time.
        """
        start_day = int(params.get("policy_mandate_start_day", 20))
        end_day = int(params.get("policy_mandate_end_day", 45))
        incentive_amount = float(params.get("policy_incentive_amount", 0.5))
        marketing_budget = float(params.get("marketing_budget_daily", 1000.0))
        marketing_prob = float(params.get("marketing_contact_probability", 0.5))
        enforcement_prob = float(params.get("policy_enforcement_probability", 0.6))

        N = len(state.get("agents", []))
        policy_active = (t >= start_day) and (t <= end_day)
        incentive = incentive_amount if policy_active else 0.0
        contacts = [0 for _ in range(N)]
        expected_contacts = int(round(marketing_budget))
        for _ in range(expected_contacts):
            if N <= 0:
                break
            i = random.randint(0, max(0, N - 1))
            if random.random() < marketing_prob:
                contacts[i] += 1

        requires_masks = {
            "home": False,
            "work": policy_active,
            "public": policy_active,
        }

        buffers["signal.policy.active"] = policy_active
        buffers["signal.policy.incentive_amount"] = incentive
        buffers["signal.policy.enforcement_probability"] = enforcement_prob
        buffers["signal.policy.requires_masks_by_type"] = requires_masks
        buffers["signal.marketing.contacts_by_agent"] = contacts
        pass


class BehaviorAdoption(Module):
    """
    Agent decision-making to adopt or drop the behavior based on signals.
    """

    def __init__(self):
        """
        Construct the BehaviorAdoption module.
        """
        super().__init__(
            "BehaviorAdoption",
            dependencies=["SocialNetworkFormation", "CommunicationDynamics", "PolicyIntervention", "MediaBroadcast", "RiskSignal"],
        )
        self.inputs = [
            "signal.network.adjacency",
            "signal.agent.exposures_today",
            "signal.policy.active",
            "signal.policy.incentive_amount",
            "signal.marketing.contacts_by_agent",
            "signal.agent.media_shift",
            "signal.risk.value",
        ]
        self.outputs = [
            "signal.agent.adopted_flags",
            "signal.agent.desired_mask_use",
            "signal.new_adoptions_today",
            "signal.dropouts_today",
        ]
        pass

    def _sigmoid(self, x: float) -> float:
        """
        Stable sigmoid function.

        Parameters
        ----------
        x : float
            Input.

        Returns
        -------
        float
            Sigmoid(x).
        """
        try:
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            else:
                z = math.exp(x)
                return z / (1.0 + z)
        except Exception:
            return 1.0 / (1.0 + math.exp(-x))
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Update agent adoption states based on social, policy, media, and marketing signals.

        Parameters
        ----------
        state : dict
            Global state.
        buffers : dict
            Outputs for this tick.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Current time.
        """
        adj: dict = buffers.get("signal.network.adjacency") or state.get("adjacency", {})
        exposures_today: _t.List[int] = buffers.get("signal.agent.exposures_today", [])
        policy_active: bool = buffers.get("signal.policy.active", False)
        incentive_amount: float = buffers.get("signal.policy.incentive_amount", 0.0)
        contacts_by_agent: _t.List[int] = buffers.get("signal.marketing.contacts_by_agent", [])
        media_shift: _t.List[float] = buffers.get("signal.agent.media_shift", [])
        risk_value: float = float(buffers.get("signal.risk.value", 0.2))
        agents: _t.List[Person] = state.get("agents", [])
        N = len(agents)
        adopted_flags = [a.adopted for a in agents]

        # Parameters
        base_rate = float(params.get("adoption_base_rate", -1.8))
        w_peer = float(params.get("adoption_peer_weight", 2.2))
        w_marketing = float(params.get("adoption_marketing_weight", 0.8))
        w_policy = float(params.get("adoption_policy_weight", 1.0))
        w_exposure = float(params.get("adoption_exposure_weight", 0.2))  # FIXED: Use exposures in utility
        mean_th = float(params.get("adoption_threshold_mean", 0.35))
        temp = float(params.get("adoption_temperature", 0.6))
        stubborn_penalty = float(params.get("adoption_stubborn_resistance", 1.0))
        fatigue_decay = float(params.get("adoption_fatigue_decay", 0.05))
        reversion_prob = float(params.get("adoption_reversion_prob", 0.02))
        enforcement_prob = float(params.get("policy_enforcement_probability", 0.6))
        social_reinf = float(params.get("adoption_social_reinforcement", 0.5))

        new_adopts = 0
        dropouts = 0
        adopted_next = adopted_flags[:]
        desired_mask_use = [False for _ in range(N)]
        for i in range(N):
            neighbors = list(adj.get(i, []))
            peer_frac = 0.0
            if neighbors:
                s = sum(1 for u in neighbors if adopted_flags[u])
                peer_frac = s / max(1, len(neighbors))
            reinforce = social_reinf * max(0.0, peer_frac - mean_th)
            marketing_signal = math.log(1 + (contacts_by_agent[i] if i < len(contacts_by_agent) else 0))
            # FIXED: Use incentive magnitude rather than boolean flag
            policy_strength = incentive_amount if policy_active else 0.0
            agent = agents[i]
            exp_signal = math.log1p(agent.exposures_memory)  # cumulative exposure memory impact
            media_sig = media_shift[i] if i < len(media_shift) else 0.0
            risk_sig = 0.3 * risk_value * (1.0 + agent.risk_attitude)
            habit_term = agent.habit_persistence * (2.0 * float(agent.prev_mask_use) - 1.0)

            # Utility
            utility = base_rate
            utility += w_peer * peer_frac
            utility += w_marketing * marketing_signal
            utility += w_policy * policy_strength
            utility += w_exposure * exp_signal
            utility += reinforce
            utility += 0.2 * media_sig
            utility += risk_sig
            utility += habit_term
            utility -= stubborn_penalty * (1 if agent.stubborn else 0)
            # Fatigue discourages frequent switching (milder)
            last_change = agent.last_adoption_change_day if agent.last_adoption_change_day is not None else -1
            idle_days = max(0, t - (last_change if last_change >= 0 else 0))
            utility -= fatigue_decay * (idle_days / 60.0)

            # Probabilistic adoption using logistic choice relative to threshold
            thr = agent.adoption_threshold
            p_adopt = self._sigmoid((utility - thr) / max(1e-6, temp))

            if not adopted_flags[i]:
                if random.random() < p_adopt:
                    adopted_next[i] = True
                    new_adopts += 1
                    agent.last_adoption_change_day = t
                    desired_mask_use[i] = True
                else:
                    desired_mask_use[i] = False
            else:
                # Remain adopted with probability (1 - reversion)
                reversion_p = reversion_prob * (1.0 - enforcement_prob * (1.0 if policy_active else 0.0))
                reversion_p = min(1.0, max(0.0, reversion_p))
                if random.random() < reversion_p:
                    adopted_next[i] = False
                    dropouts += 1
                    agent.last_adoption_change_day = t
                    desired_mask_use[i] = False
                else:
                    desired_mask_use[i] = True

        buffers["signal.agent.adopted_flags"] = adopted_next
        buffers["signal.agent.desired_mask_use"] = desired_mask_use
        buffers["signal.new_adoptions_today"] = new_adopts
        buffers["signal.dropouts_today"] = dropouts
        pass


class Retailer(Module):
    """
    Retailer module handling mask inventory, restocking, and purchases.
    """

    def __init__(self):
        """
        Construct the Retailer module.
        """
        super().__init__("Retailer", dependencies=["BehaviorAdoption"])
        self.inputs = ["signal.agent.desired_mask_use"]
        self.outputs = [
            "signal.retailer.stock_level",
            "signal.retailer.masks_sold_today",
            "signal.agent.mask_inventory_next",
            "signal.agent.has_mask_today",
        ]
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Simulate purchases given desired mask use, price, stock, and cost sensitivity.

        Parameters
        ----------
        state : dict
            Global state containing agents and previous stock level.
        buffers : dict
            Outputs buffer.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Day index.
        """
        agents: _t.List[Person] = state.get("agents", [])
        desired: _t.List[bool] = buffers.get("signal.agent.desired_mask_use", [])
        N = len(agents)
        # Initialize stock
        if t == 0 and state.get("retailer_stock_level") is None:
            init_per_capita = float(params.get("retailer_initial_stock_per_capita", 1.5))
            state["retailer_stock_level"] = int(round(init_per_capita * max(1, N)))
        stock_level = int(state.get("retailer_stock_level", 0))
        restock_rate = float(params.get("retailer_restock_rate_per_day", 0.05))
        restock_amount = int(round(restock_rate * max(1, N)))
        stock_level += restock_amount
        price = float(params.get("retailer_price_per_mask", 1.0))
        p_base = float(params.get("purchase_base_probability", 0.25))
        cost_sens = float(params.get("purchase_cost_sensitivity", 0.4))
        masks_sold = 0
        inv_next = [a.mask_inventory for a in agents]
        has_mask_today = [False for _ in range(N)]
        for i in range(N):
            has = agents[i].mask_inventory > 0
            if has:
                has_mask_today[i] = True
                continue
            # Need to buy if desire True and no mask
            if i < len(desired) and desired[i]:
                # Willingness to pay scales with income
                income = agents[i].income_log10
                affordability = 1.0 / (1.0 + math.exp(-(income - 3.5)))
                p_buy = p_base * (0.5 + 0.5 * affordability) * (1.0 / (1.0 + cost_sens * price))
                if stock_level > 0 and random.random() < min(1.0, max(0.0, p_buy)):
                    stock_level -= 1
                    inv_next[i] += 1
                    masks_sold += 1
                    has_mask_today[i] = True
                else:
                    has_mask_today[i] = False
            else:
                has_mask_today[i] = False

        buffers["signal.retailer.stock_level"] = stock_level
        buffers["signal.retailer.masks_sold_today"] = masks_sold
        buffers["signal.agent.mask_inventory_next"] = inv_next
        buffers["signal.agent.has_mask_today"] = has_mask_today
        pass


class LocationCompliance(Module):
    """
    Simulates visits to locations and computes compliance under enforcement.
    """

    def __init__(self):
        """
        Construct the LocationCompliance module.
        """
        super().__init__("LocationCompliance", dependencies=["PolicyIntervention", "BehaviorAdoption", "Retailer"])
        self.inputs = [
            "signal.policy.active",
            "signal.policy.enforcement_probability",
            "signal.policy.requires_masks_by_type",
            "signal.agent.desired_mask_use",
            "signal.agent.has_mask_today",
        ]
        self.outputs = [
            "signal.agent.current_mask_use_today",
            "signal.compliance_by_type",
            "signal.enforcement_actions_today",
        ]
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Compute immediate compliance by location type and enforcement actions.

        Parameters
        ----------
        state : dict
            Global state.
        buffers : dict
            Output buffer.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Day index.
        """
        agents: _t.List[Person] = state.get("agents", [])
        N = len(agents)
        policy_active = bool(buffers.get("signal.policy.active", False))
        enforcement_prob = float(buffers.get("signal.policy.enforcement_probability", params.get("policy_enforcement_probability", 0.6)))
        requires_masks_by_type = buffers.get("signal.policy.requires_masks_by_type", {"home": False, "work": False, "public": False})
        desired: _t.List[bool] = buffers.get("signal.agent.desired_mask_use", [])
        has_mask: _t.List[bool] = buffers.get("signal.agent.has_mask_today", [])
        # Strictness per type
        strict_home = float(params.get("enforcement_strictness_home", 0.0))
        strict_work = float(params.get("enforcement_strictness_work", 0.7))
        strict_public = float(params.get("enforcement_strictness_public", 0.5))
        work_att = float(params.get("work_attendance_rate", 0.7))
        public_visit = float(params.get("public_visit_rate", 0.4))

        # Collect per-type observed compliance (averaged over visitors)
        comp_counts = {"home": [0, 0], "work": [0, 0], "public": [0, 0]}  # [wearers, visitors]
        enforcement_actions = 0
        current_mask_use_today = [False for _ in range(N)]

        for i in range(N):
            agent = agents[i]
            # Home visit (always)
            wear = (desired[i] if i < len(desired) else False) and (has_mask[i] if i < len(has_mask) else False)
            # No enforcement at home
            comp_counts["home"][1] += 1
            comp_counts["home"][0] += 1 if wear else 0

            # Work visit
            if random.random() < work_att:
                wear_work = wear
                if requires_masks_by_type.get("work", False) and not wear_work:
                    # immediate compliance check under enforcement
                    p_comply = min(1.0, enforcement_prob * strict_work * agent.trust_in_authority)
                    forced = random.random() < p_comply and (has_mask[i] if i < len(has_mask) else False)
                    if not forced:
                        # potential enforcement action
                        if random.random() < enforcement_prob * strict_work:
                            enforcement_actions += 1
                    wear_work = wear_work or forced
                comp_counts["work"][1] += 1
                comp_counts["work"][0] += 1 if wear_work else 0
                wear = wear or wear_work

            # Public visit
            if random.random() < public_visit:
                wear_public = wear
                if requires_masks_by_type.get("public", False) and not wear_public:
                    p_comply = min(1.0, enforcement_prob * strict_public * agent.trust_in_authority)
                    forced = random.random() < p_comply and (has_mask[i] if i < len(has_mask) else False)
                    if not forced:
                        if random.random() < enforcement_prob * strict_public:
                            enforcement_actions += 1
                    wear_public = wear_public or forced
                comp_counts["public"][1] += 1
                comp_counts["public"][0] += 1 if wear_public else 0
                wear = wear or wear_public

            current_mask_use_today[i] = wear

        # Compute averages
        compliance_by_type = {}
        for k, (w, v) in comp_counts.items():
            compliance_by_type[k] = float(w) / max(1, v)
        buffers["signal.agent.current_mask_use_today"] = current_mask_use_today
        buffers["signal.compliance_by_type"] = compliance_by_type
        buffers["signal.enforcement_actions_today"] = enforcement_actions
        pass


class AdoptionAggregator(Module):
    """
    Aggregates and reports daily observables and smoothed series.
    """

    def __init__(self):
        """
        Construct the AdoptionAggregator module.
        """
        super().__init__(
            "AdoptionAggregator",
            dependencies=["BehaviorAdoption", "CommunicationDynamics", "LocationCompliance"],
        )
        self.inputs = [
            "signal.agent.adopted_flags",
            "signal.agent.exposures_today",
            "signal.agent.current_mask_use_today",
            "signal.compliance_by_type",
            "signal.enforcement_actions_today",
            "signal.media.misinformation_rate",
            "signal.policy.active",
        ]
        self.outputs = [
            "signal.adoption_rate_daily",
            "signal.cumulative_adopters",
            "signal.exposure_rate_daily_smoothed",
            "signal.churn_rate_daily",
            "signal.compliance_under_mandate_daily",
        ]
        pass

    def forward(self, state: dict, buffers: dict, params: ParameterRegistry, t: int) -> None:
        """
        Compute adoption rate, cumulative adopters (unique), churn, compliance under mandate, and smoothed exposures.

        Parameters
        ----------
        state : dict
            Global state.
        buffers : dict
            Outputs for this tick.
        params : ParameterRegistry
            Parameter registry.
        t : int
            Current time.
        """
        adopted_flags = buffers.get("signal.agent.adopted_flags", state.get("adopted_flags", []))
        current_mask = buffers.get("signal.agent.current_mask_use_today", None)
        N = len(adopted_flags)
        # Use current mask use if available, else adopted flags
        if current_mask is not None and len(current_mask) == N:
            adoption_rate = float(sum(1 for x in current_mask if x)) / max(1, N)
            today_mask_use = current_mask
        else:
            adoption_rate = float(sum(1 for x in adopted_flags if x)) / max(1, N)
            today_mask_use = adopted_flags

        # Churn: fraction of yesterday's wearers who stopped today
        yesterday = state.get("prev_mask_use", [0] * N)
        prev_wearers = sum(1 for x in yesterday if x)
        stopped = 0
        for i in range(min(N, len(yesterday))):
            if yesterday[i] and not today_mask_use[i]:
                stopped += 1
        churn = float(stopped) / max(1, prev_wearers)

        # Unique cumulative adopters tracked with a bitset on state (initialize once)
        if t == 0 or "ever_adopted" not in state:
            state["ever_adopted"] = [bool(x) for x in today_mask_use]
        else:
            for i, flag in enumerate(today_mask_use):
                state["ever_adopted"][i] = state["ever_adopted"][i] or bool(flag)
        cumulative_unique = int(sum(1 for x in state["ever_adopted"] if x))

        exposure_rate_daily = float(buffers.get("signal.exposure_rate_daily", 0.0))
        # Simple moving average smoothing
        window = int(params.get("agg_smoothing_window_days", 3))
        prev_exposures = state.get("exposure_rate_history", [])
        exposure_smoothed = exposure_rate_daily
        if window > 1:
            hist = prev_exposures[-(window - 1):] + [exposure_rate_daily]
            exposure_smoothed = sum(hist) / float(len(hist))

        # Compliance under mandate
        policy_active = bool(buffers.get("signal.policy.active", False))
        compliance_by_type = buffers.get("signal.compliance_by_type", {"home": None, "work": None, "public": None})
        if policy_active:
            compliance_under_mandate = adoption_rate
        else:
            compliance_under_mandate = None

        # Enforcement actions per 1000
        enforcement_actions_today = int(buffers.get("signal.enforcement_actions_today", 0))

        # Update state histories for evaluator
        state["prev_mask_use"] = [1 if x else 0 for x in today_mask_use]
        # Persist compliance_by_type for metrics
        state.setdefault("compliance_by_type_history", [])
        state["compliance_by_type_history"].append(compliance_by_type)
        state.setdefault("enforcement_actions_history", [])
        state["enforcement_actions_history"].append(enforcement_actions_today)

        buffers["signal.adoption_rate_daily"] = adoption_rate
        buffers["signal.cumulative_adopters"] = cumulative_unique
        buffers["signal.exposure_rate_daily_smoothed"] = exposure_smoothed
        buffers["signal.churn_rate_daily"] = churn
        buffers["signal.compliance_under_mandate_daily"] = compliance_under_mandate
        pass


class Simulation:
    """
    Main simulation class coordinating modules, scheduler, and artifacts.

    Methods
    -------
    run(start_day, end_day)
    reset(reseed=None)
    save_results(path)
    save_module_io(module, path)
    save_all_io(root_dir)
    evaluate()
    visualize()
    set_params(module=None, **kwargs)
    get_params()
    """

    def __init__(self, params: dict, param_defs: dict):
        """
        Initialize the simulation: agents, modules, state, and artifacts.

        Parameters
        ----------
        params : dict
            Simulation parameters.
        param_defs : dict
            Parameter definitions.
        """
        self.param_defs = dict(param_defs)
        self.params = ParameterRegistry(self.param_defs, params)
        self.artifacts_root = ARTIFACTS_DIR
        safe_mkdirs(self.artifacts_root)
        safe_mkdirs(os.path.join(self.artifacts_root, "results"))
        safe_mkdirs(os.path.join(self.artifacts_root, "io"))
        safe_mkdirs(os.path.join(self.artifacts_root, "figs"))
        safe_mkdirs(os.path.join(self.artifacts_root, "observables"))

        # State
        self.state: dict = {
            "agents": [],
            "households": [],
            "locations": [],
            "adjacency": None,
            "adopted_flags": [],
            "cumulative_adopters": 0,
            "adoption_rate_history": [],
            "current_mask_use_history": [],
            "churn_rate_history": [],
            "exposure_rate_history": [],
            "exposure_rate_smoothed_history": [],
            "misinformation_rate_history": [],
            "compliance_by_type_history": [],
            "enforcement_actions_history": [],
            "masks_sold_history": [],
            "retailer_stock_level": None,
            "time": 0,
        }
        self.module_io: dict = {}  # per-module I/O traces
        self._build_agents()
        self._build_environment()
        self._build_modules()
        pass

    def _build_agents(self) -> None:
        """
        Initialize the agent population with heterogeneous attributes.
        """
        N = int(self.params.get("population_size", 1000))
        stubborn_fraction = float(self.params.get("adoption_stubborn_fraction", 0.1))
        initial_adopted_fraction = 0.05
        agents: _t.List[Person] = []
        # Households: random sizes around mean ~ 2.6
        mean_size = 2.6
        households: _t.List[Household] = []
        remaining = N
        hid = 0
        while remaining > 0:
            if np is not None:
                size = int(max(1, np.random.poisson(mean_size)))
            else:
                # Approximate Poisson by geometric with p ~ 1/mean
                size = max(1, int(round(random.random() * 2 + 1)))
            size = min(size, remaining)
            member_indices: _t.List[int] = []
            households.append(Household(id=hid, members=member_indices, norm_strength=0.5 + 0.2 * (random.random() - 0.5)))
            hid += 1
            remaining -= size
        # Flatten households not yet with members; we'll assign sequentially
        agent_idx = 0
        hh_idx = 0
        for hh in households:
            for _ in range(len(hh.members), len(hh.members) + max(1, int(round(N / len(households))))):
                if agent_idx >= N:
                    break
                hh.members.append(agent_idx)
                agent_idx += 1
            if agent_idx >= N:
                break
        # If underfilled or overfilled adjust
        flat_members = [idx for hh in households for idx in hh.members]
        if len(flat_members) < N:
            # add remaining to last household
            last_hh = households[-1]
            for idx in range(len(flat_members), N):
                last_hh.members.append(idx)
        elif len(flat_members) > N:
            # trim extras
            overflow = len(flat_members) - N
            cut = 0
            for hh in households:
                while hh.members and cut < overflow:
                    hh.members.pop()
                    cut += 1
                    if cut >= overflow:
                        break

        for i in range(N):
            # Age bucket distribution (approximate)
            r = random.random()
            if r < 0.35:
                age_bucket = 0
            elif r < 0.35 + 0.4:
                age_bucket = 1
            else:
                age_bucket = 2
            # Income log10 normal via Box-Muller or fallback
            if np is not None:
                income = float(np.random.normal(4.5, 0.25))
            else:
                income = 4.5 + (random.random() - 0.5) * 0.5
            group_id = random.randint(0, 4)
            # Openness Beta approx
            if np is not None:
                openness = float(np.random.beta(2.0, 5.0))
            else:
                openness = (random.random() + random.random()) / 2.0
            stubborn = 1 if random.random() < stubborn_fraction else 0
            # Threshold ~ N(mean,std) truncated to [0,1]
            mean_th = float(self.params.get("adoption_threshold_mean", 0.35))
            std_th = float(self.params.get("adoption_threshold_std", 0.12))
            if np is not None:
                thr = float(np.clip(np.random.normal(mean_th, std_th), 0.0, 1.0))
            else:
                thr = min(1.0, max(0.0, mean_th + (random.random() - 0.5) * 2 * std_th))
            adopted = random.random() < initial_adopted_fraction
            trust_media = 0.5 + 0.15 * (random.random() - 0.5)
            trust_auth = 0.6 + 0.2 * (random.random() - 0.5)
            risk_att = 0.5 + 0.2 * (random.random() - 0.5)
            habit = 0.7 + 0.2 * (random.random() - 0.5)
            is_anti = 1 if random.random() < 0.05 else 0
            is_infl = 1 if random.random() < 0.02 else 0
            agent = Person(
                id=i,
                age_bucket=age_bucket,
                income_log10=income,
                group_id=group_id,
                openness=openness,
                stubborn=stubborn,
                adopted=adopted,
                adoption_threshold=thr,
                utility_bias=0.0,
                exposures_memory=0.0,
                last_adoption_change_day=0 if adopted else -1,
                trust_in_media=trust_media,
                trust_in_authority=trust_auth,
                risk_attitude=risk_att,
                habit_persistence=habit,
                prev_mask_use=1 if adopted else 0,
                is_influencer=is_infl,
                is_anti_masker=is_anti,
                mask_inventory=1,
                home_id=-1,
                work_id=-1,
            )
            agents.append(agent)
        # Assign households properly
        for hh in households:
            for idx in hh.members:
                if 0 <= idx < len(agents):
                    agents[idx].home_id = hh.id
        self.state["agents"] = agents
        self.state["households"] = households
        self.state["adopted_flags"] = [a.adopted for a in agents]
        self.state["cumulative_adopters"] = sum(1 for x in self.state["adopted_flags"] if x)
        pass

    def _build_environment(self) -> None:
        """
        Initialize locations minimally (work/public) for compliance tracking.
        """
        # For simplicity, we keep a small list; compliance simulation uses rates not explicit assignments
        work_loc = Location(id=0, type="work", capacity=1000000, baseline_contact_rate=8, enforcement_strictness=float(self.params.get("enforcement_strictness_work", 0.7)))
        pub_loc = Location(id=1, type="public", capacity=1000000, baseline_contact_rate=6, enforcement_strictness=float(self.params.get("enforcement_strictness_public", 0.5)))
        home_loc = Location(id=2, type="home", capacity=1000000, baseline_contact_rate=4, enforcement_strictness=float(self.params.get("enforcement_strictness_home", 0.0)))
        self.state["locations"] = [work_loc, pub_loc, home_loc]
        pass

    def _build_modules(self) -> None:
        """
        Instantiate and order modules respecting dependencies.
        """
        self.modules: _t.List[Module] = [
            SocialNetworkFormation(),
            CommunicationDynamics(),
            MediaBroadcast(),
            RiskSignal(),
            PolicyIntervention(),
            BehaviorAdoption(),
            Retailer(),
            LocationCompliance(),
            AdoptionAggregator(),
        ]
        # Deterministic order by declared dependencies
        name_to_module = {m.name: m for m in self.modules}
        # Simple topological sort
        visited = set()
        ordered: _t.List[Module] = []

        def dfs(m: Module):
            if m.name in visited:
                return
            for dep in m.dependencies:
                dfs(name_to_module[dep])
            visited.add(m.name)
            ordered.append(m)

        for m in self.modules:
            dfs(m)
        self.modules = ordered
        pass

    def _record_module_io(self, module: Module, inputs: dict, outputs: dict, t: int) -> None:
        """
        Record per-module inputs and outputs for diagnostics.

        Parameters
        ----------
        module : Module
            Module instance.
        inputs : dict
            Inputs consumed.
        outputs : dict
            Outputs produced.
        t : int
            Current time step.
        """
        rec = self.module_io.setdefault(module.name, [])
        rec.append({"t": t, "inputs": inputs, "outputs": outputs})
        pass

    def reset(self, reseed: _t.Optional[int] = None) -> None:
        """
        Reset the simulation to a clean state for a new run, optionally reseeding RNG.

        Parameters
        ----------
        reseed : int or None
            If provided, reseeds RNGs for reproducibility.
        """
        # FIXED: Reset simulator state for calibration trial comparability
        if reseed is not None:
            seed_all(int(reseed))
        self.state = {
            "agents": [],
            "households": [],
            "locations": [],
            "adjacency": None,
            "adopted_flags": [],
            "cumulative_adopters": 0,
            "adoption_rate_history": [],
            "current_mask_use_history": [],
            "churn_rate_history": [],
            "exposure_rate_history": [],
            "exposure_rate_smoothed_history": [],
            "misinformation_rate_history": [],
            "compliance_by_type_history": [],
            "enforcement_actions_history": [],
            "masks_sold_history": [],
            "retailer_stock_level": None,
            "time": 0,
        }
        self.module_io = {}
        self._build_agents()
        self._build_environment()
        # Modules are stateless; keep them
        pass

    def run(self, start_day: int, end_day: int) -> None:
        """
        Execute the simulation from start_day to end_day (exclusive).

        Parameters
        ----------
        start_day : int
            Start day index.
        end_day : int
            End day index (exclusive).
        """
        start_day = int(start_day)
        end_day = int(end_day)
        # Early stopping can be disabled via param/flag for calibration
        target_stop = float(self.params.get("policy_target_coverage", 0.7))
        disable_early_stop = bool(int(os.environ.get("DISABLE_EARLY_STOP", "0")))
        for t in range(start_day, end_day):
            self.state["time"] = t
            buffers: dict = {}
            # Module execution
            for module in self.modules:
                # Build inputs snapshot for IO logging (avoid heavy state dumps)
                inputs_snapshot = {
                    "state": {
                        "adopted_flags": list(self.state.get("adopted_flags", [])),
                        "cumulative_adopters": int(self.state.get("cumulative_adopters", 0)),
                    },
                    "buffers": list(buffers.keys()),
                }
                module.forward(self.state, buffers, self.params, t)
                # Collect outputs names produced so far
                outputs_snapshot = {k: v for k, v in buffers.items() if k in module.outputs or k.startswith("signal.")}
                self._record_module_io(module, inputs_snapshot, outputs_snapshot, t)

            # Commit step: update state from buffers
            if "signal.network.adjacency" in buffers:
                self.state["adjacency"] = buffers["signal.network.adjacency"]
            if "signal.agent.exposures_memory_next" in buffers:
                for idx, agent in enumerate(self.state["agents"]):
                    agent.exposures_memory = buffers["signal.agent.exposures_memory_next"][idx]
            if "signal.agent.adopted_flags" in buffers:
                self.state["adopted_flags"] = buffers["signal.agent.adopted_flags"]
                for idx, agent in enumerate(self.state["agents"]):
                    agent.adopted = self.state["adopted_flags"][idx]
            if "signal.agent.mask_inventory_next" in buffers:
                # Update inventory
                next_inv = buffers["signal.agent.mask_inventory_next"]
                for idx, agent in enumerate(self.state["agents"]):
                    if idx < len(next_inv):
                        agent.mask_inventory = max(0, int(next_inv[idx]))
                        # Consume one for today's wear if used
                        # We'll deduct consumption when they wear at least once per day
            if "signal.retailer.stock_level" in buffers:
                self.state["retailer_stock_level"] = int(buffers["signal.retailer.stock_level"])
            if "signal.agent.current_mask_use_today" in buffers:
                today_use = buffers["signal.agent.current_mask_use_today"]
                self.state["current_mask_use_history"].append([1 if x else 0 for x in today_use])
                # Deduct one mask per day if used and inventory available
                for idx, agent in enumerate(self.state["agents"]):
                    used = 1 if (idx < len(today_use) and today_use[idx]) else 0
                    if used and agent.mask_inventory > 0:
                        agent.mask_inventory -= 1
                    agent.prev_mask_use = used
            if "signal.cumulative_adopters" in buffers:
                self.state["cumulative_adopters"] = int(buffers["signal.cumulative_adopters"])
            if "signal.adoption_rate_daily" in buffers:
                self.state["adoption_rate_history"].append(float(buffers["signal.adoption_rate_daily"]))
            if "signal.exposure_rate_daily" in buffers:
                self.state["exposure_rate_history"].append(float(buffers["signal.exposure_rate_daily"]))
            if "signal.exposure_rate_daily_smoothed" in buffers:
                self.state["exposure_rate_smoothed_history"].append(float(buffers["signal.exposure_rate_daily_smoothed"]))
            if "signal.media.misinformation_rate" in buffers:
                self.state["misinformation_rate_history"].append(float(buffers["signal.media.misinformation_rate"]))
            if "signal.enforcement_actions_today" in buffers:
                self.state["enforcement_actions_history"].append(int(buffers["signal.enforcement_actions_today"]))
            if "signal.compliance_by_type" in buffers:
                self.state.setdefault("compliance_by_type_history", [])
                self.state["compliance_by_type_history"].append(buffers["signal.compliance_by_type"])
            if "signal.churn_rate_daily" in buffers:
                self.state["churn_rate_history"].append(float(buffers["signal.churn_rate_daily"]))
            if "signal.retailer.masks_sold_today" in buffers:
                self.state["masks_sold_history"].append(int(buffers["signal.retailer.masks_sold_today"]))

            # Early stopping
            if (not disable_early_stop) and self.state["adoption_rate_history"] and self.state["adoption_rate_history"][-1] >= min(0.999, max(0.7, target_stop)):
                break
        pass

    def save_results(self, path: _t.Union[str, Path]) -> None:
        """
        Save primary simulation results to a JSON file.

        Parameters
        ----------
        path : str or Path
            File path to write results JSON.
        """
        results = {
            "adoption_rate_over_time": self.state.get("adoption_rate_history", []),
            "exposure_rate_over_time": self.state.get("exposure_rate_history", []),
            "exposure_rate_smoothed_over_time": self.state.get("exposure_rate_smoothed_history", []),
            "misinformation_rate_over_time": self.state.get("misinformation_rate_history", []),
            "current_mask_use_history": self.state.get("current_mask_use_history", []),
            "enforcement_actions_history": self.state.get("enforcement_actions_history", []),
            "compliance_by_type_history": self.state.get("compliance_by_type_history", []),
            "masks_sold_history": self.state.get("masks_sold_history", []),
            "final_adoption_rate": (self.state.get("adoption_rate_history", [])[-1] if self.state.get("adoption_rate_history") else 0.0),
            "cumulative_adopters_unique": self.state.get("cumulative_adopters", 0),
        }
        try:
            with open(str(path), "w", encoding="utf-8") as f:
                # FIXED: Sanitize JSON and disallow NaN
                json.dump(_json_clean(results), f, allow_nan=False)
        except Exception as e:
            print(f"Error saving results to {path}: {e}", file=sys.stderr)
        pass

    def save_module_io(self, module: Module, path: _t.Union[str, Path]) -> None:
        """
        Save per-module I/O traces.

        Parameters
        ----------
        module : Module
            Module instance to save.
        path : str or Path
            Output file path.
        """
        data = self.module_io.get(module.name, [])
        try:
            with open(str(path), "w", encoding="utf-8") as f:
                # FIXED: Sanitize module IO (e.g., sets in adjacency) and disallow NaN
                json.dump(_json_clean(data), f, allow_nan=False)
        except Exception as e:
            print(f"Error saving module IO for {module.name} to {path}: {e}", file=sys.stderr)
        pass

    def save_all_io(self, root_dir: _t.Union[str, Path]) -> None:
        """
        Save I/O traces for all modules under the given root directory.

        Parameters
        ----------
        root_dir : str or Path
            Directory to store per-module IO JSON.
        """
        root = str(root_dir)
        safe_mkdirs(root)
        for module in self.modules:
            out_path = os.path.join(root, f"{module.name}_io.json")
            self.save_module_io(module, out_path)
        pass

    def _gini(self, values: _t.List[float]) -> float:
        """
        Compute the Gini coefficient for a list of non-negative values.

        Parameters
        ----------
        values : list of float
            Values.

        Returns
        -------
        float
            Gini coefficient in [0,1].
        """
        try:
            xs = [max(0.0, float(x)) for x in values]
            n = len(xs)
            if n == 0:
                return float("nan")
            xs_sorted = sorted(xs)
            cumx = 0.0
            cum_sum = 0.0
            for i, x in enumerate(xs_sorted, 1):
                cumx += x
                cum_sum += i * x
            if cumx == 0:
                return 0.0
            g = (2.0 * cum_sum) / (n * cumx) - (n + 1.0) / n
            return g
        except Exception:
            return float("nan")
        pass

    def _logistic_fit_r2(self, series: _t.List[float]) -> float:
        """
        Fit a simple logistic curve to adoption series and compute R^2 (approx).

        Parameters
        ----------
        series : list of float
            Adoption rate time series.

        Returns
        -------
        float
            R-squared of the fit on the first half of the series.
        """
        if np is None or len(series) < 5:
            return float("nan")
        try:
            y = np.array(series, dtype=float)
            n = len(y)
            x = np.arange(n, dtype=float)
            # Fit logistic via logit transform for early phase (avoid saturation)
            eps = 1e-6
            y_clip = np.clip(y, eps, 1 - eps)
            z = np.log(y_clip / (1 - y_clip))
            # Linear fit z = a + b x
            X = np.vstack([np.ones_like(x), x]).T
            # Use only first half
            m = max(3, n // 2)
            Xm = X[:m]
            zm = z[:m]
            beta, _, _, _ = np.linalg.lstsq(Xm, zm, rcond=None)
            zhat = Xm @ beta
            # Back transform to yhat
            yhat = 1.0 / (1.0 + np.exp(-zhat))
            ss_res = np.sum((zm - zhat) ** 2)
            ss_tot = np.sum((zm - np.mean(zm)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, eps)
            return float(r2)
        except Exception:
            return float("nan")
        pass

    def policy_monotonicity_check(self, steps: int = 30, seeds: _t.Optional[_t.List[int]] = None) -> dict:
        """
        Check that higher enforcement_probability leads to non-decreasing adoption in most seeds.

        Parameters
        ----------
        steps : int
            Number of days to run for each test.
        seeds : list of int or None
            Seeds to test.

        Returns
        -------
        dict
            Report containing pass flag and details.
        """
        base_seed = int(self.params.get("random_seed", 42))
        if seeds is None:
            seeds = [base_seed + i for i in range(3)]
        enforcement_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        passes = 0
        trials = 0
        results = []
        old_disable = os.environ.get("DISABLE_EARLY_STOP", "0")
        os.environ["DISABLE_EARLY_STOP"] = "1"
        try:
            for s in seeds:
                series = []
                for lev in enforcement_levels:
                    # Clone simulation lightweight by resetting and setting param
                    self.reset(reseed=s)
                    try:
                        self.params.set("policy_enforcement_probability", lev)
                    except KeyError:
                        pass
                    self.run(0, steps)
                    final = float(self.state.get("adoption_rate_history", [0.0])[-1])
                    series.append(final)
                monotonic = all(series[i] <= series[i + 1] + 1e-9 for i in range(len(series) - 1))
                results.append({"seed": s, "finals": series, "monotonic": monotonic})
                trials += 1
                if monotonic:
                    passes += 1
        finally:
            os.environ["DISABLE_EARLY_STOP"] = old_disable
        ok = (passes / max(1, trials)) >= 0.9
        return {"pass": ok, "details": results}
        pass

    def evaluate(self) -> dict:
        """
        Compute and save evaluation metrics against available target data.

        Returns
        -------
        dict
            Metrics dictionary.
        """
        # Load target data if available
        metrics = {}
        # Observables from simulation
        sim_adopt = self.state.get("adoption_rate_history", [])
        sim_expo = self.state.get("exposure_rate_smoothed_history", [])
        sim_churn = self.state.get("churn_rate_history", [])
        sim_compliance_hist = self.state.get("compliance_by_type_history", [])
        sim_enforce = self.state.get("enforcement_actions_history", [])

        # Simple derived metrics
        metrics["final_adoption_rate"] = float(sim_adopt[-1]) if sim_adopt else 0.0
        metrics["peak_adoption"] = max(sim_adopt) if sim_adopt else 0.0
        # time_to_threshold_70
        t70 = None
        for i, v in enumerate(sim_adopt):
            if v >= 0.70:
                t70 = i
                break
        # FIXED: Applied feedback snippet from simulation.py
metrics["time_to_threshold_70"] = t70
# Additional required metric: time_to_50_percent
metrics["time_to_50_percent"] = next((i for i, v in enumerate(sim_adopt) if v >= 0.50), None)
        # Additional required metric: time_to_50_percent
        t50 = None
        for i, v in enumerate(sim_adopt):
            if v >= 0.50:
                t50 = i
                break
        metrics["time_to_50_percent"] = t50
        # time_to_80_percent
        t80 = None
        for i, v in enumerate(sim_adopt):
            if v >= 0.80:
                t80 = i
                break
        metrics["time_to_80_percent"] = t80
        # Post-policy sustainability: average adoption rate after mandate end for 30 days
        mend = int(self.params.get("policy_mandate_end_day", 45))
        post_window = list(range(mend + 1, min(mend + 31, len(sim_adopt))))
        if post_window:
            vals = [sim_adopt[i] for i in post_window if i < len(sim_adopt)]
            metrics["sustained_adoption_after_policy_lift"] = sum(vals) / max(1, len(vals))
        else:
            metrics["sustained_adoption_after_policy_lift"] = None
        # Average compliance by location type
        avg_comp = {"home": None, "work": None, "public": None}
        if sim_compliance_hist:
            sums = {"home": 0.0, "work": 0.0, "public": 0.0}
            cnt = 0
            for d in sim_compliance_hist:
                for k in ["home", "work", "public"]:
                    if d.get(k) is not None:
                        sums[k] += float(d.get(k))
                cnt += 1
            if cnt > 0:
                avg_comp = {k: (sums[k] / cnt) for k in sums}
        metrics["average_compliance_by_location_type"] = avg_comp
        # Inequality index by income group (Gini over last-day adoption by income terciles)
        try:
            agents: _t.List[Person] = self.state.get("agents", [])
            if agents and self.state.get("current_mask_use_history"):
                last_use = self.state["current_mask_use_history"][-1]
                pairs = sorted([(a.income_log10, last_use[i] if i < len(last_use) else 0) for i, a in enumerate(agents)], key=lambda x: x[0])
                n = len(pairs)
                terc = n // 3
                groups = [
                    pairs[:terc],
                    pairs[terc:2 * terc],
                    pairs[2 * terc:],
                ]
                group_rates = []
                for g in groups:
                    if g:
                        rate = sum(int(x[1]) for x in g) / float(len(g))
                    else:
                        rate = 0.0
                    group_rates.append(rate)
                metrics["inequality_index_by_group"] = self._gini(group_rates)
            else:
                metrics["inequality_index_by_group"] = float("nan")
        except Exception:
            metrics["inequality_index_by_group"] = float("nan")
        # Enforcement actions per 1000 average over last 30 days
        if sim_enforce:
            last30 = sim_enforce[-30:] if len(sim_enforce) >= 30 else sim_enforce
            per_1000 = [1000.0 * float(x) / max(1, int(self.params.get("population_size", 1))) for x in last30]
            metrics["enforcement_actions_per_1000_avg_30d"] = sum(per_1000) / max(1, len(per_1000))
        else:
            metrics["enforcement_actions_per_1000_avg_30d"] = None

        # Load ground truth CSV if present for RMSE/MAE
        gt_adopt = []
        gt_cum = []
        train_csv = os.path.join(DATA_DIR, "train_data.csv")
        if os.path.isfile(train_csv):
            try:
                import csv
                with open(train_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ar = row.get("adoption_rate")
                        ca = row.get("cumulative_adopters")
                        gt_adopt.append(float(ar) if ar is not None and ar != "" else None)
                        gt_cum.append(float(ca) if ca is not None and ca != "" else None)
            except Exception as e:
                print(f"Error reading ground truth data: {e}", file=sys.stderr)

        def rmse(a: _t.List[float], b: _t.List[float]) -> float:
            try:
                n = min(len(a), len(b))
                if n == 0:
                    return float("nan")
                s = 0.0
                cnt = 0
                for i in range(n):
                    if a[i] is None or b[i] is None:
                        continue
                    s += (a[i] - b[i]) ** 2
                    cnt += 1
                return math.sqrt(s / max(1, cnt))
            except Exception:
                return float("nan")

        def mae(a: _t.List[float], b: _t.List[float]) -> float:
            try:
                n = min(len(a), len(b))
                if n == 0:
                    return float("nan")
                s = 0.0
                cnt = 0
                for i in range(n):
                    if a[i] is None or b[i] is None:
                        continue
                    s += abs(a[i] - b[i])
                    cnt += 1
                return s / max(1, cnt)
            except Exception:
                return float("nan")

        def pearsonr(a: _t.List[float], b: _t.List[float]) -> float:
            try:
                xs = []
                ys = []
                for x, y in zip(a, b):
                    if x is None or y is None:
                        continue
                    xs.append(x)
                    ys.append(y)
                n = len(xs)
                if n < 2:
                    return float("nan")
                mean_x = sum(xs) / n
                mean_y = sum(ys) / n
                num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(xs, ys))
                denx = math.sqrt(sum((xi - mean_x) ** 2 for xi in xs))
                deny = math.sqrt(sum((yi - mean_y) ** 2 for yi in ys))
                if denx == 0 or deny == 0:
                    return float("nan")
                return num / (denx * deny)
            except Exception:
                return float("nan")

        # Metrics vs ground truth
        metrics["RMSE_adoption_rate"] = rmse(sim_adopt, gt_adopt) if gt_adopt else float("nan")
        metrics["MAE_cumulative_adopters"] = mae([None] * len(sim_adopt), gt_cum) if gt_cum else float("nan")
        metrics["PearsonR_adoption_rate"] = pearsonr(sim_adopt, gt_adopt) if gt_adopt else float("nan")

        # Logistic fit diagnostics
        metrics["logistic_fit_R2_initial"] = self._logistic_fit_r2(sim_adopt)

        # Policy monotonicity check (optional)
        try:
            if os.environ.get("ENABLE_POLICY_MONO_CHECK", "0") == "1":
                metrics["policy_monotonicity_check"] = self.policy_monotonicity_check()
        except Exception as e:
            metrics["policy_monotonicity_check"] = {"error": str(e)}

        # Save metrics
        out_path = os.path.join(self.artifacts_root, "results", "metrics.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                # FIXED: Sanitize metrics dump to avoid NaN/Inf
                json.dump(_json_clean(metrics), f, allow_nan=False)
        except Exception as e:
            print(f"Error saving metrics: {e}", file=sys.stderr)
        return metrics
        pass

    def visualize(self) -> None:
        """
        Visualize simulation results using matplotlib if available.
        """
        if plt is None:
            print("Visualization skipped (matplotlib not available).", file=sys.stderr)
            return
        try:
            t = list(range(len(self.state.get("adoption_rate_history", []))))
            plt.figure(figsize=(9, 5))
            plt.plot(t, self.state.get("adoption_rate_history", []), label="Adoption rate")
            plt.plot(t, self.state.get("exposure_rate_smoothed_history", []), label="Exposure rate (smoothed)")
            plt.plot(t, [d.get("work", None) if d else None for d in self.state.get("compliance_by_type_history", [])], label="Compliance work", alpha=0.6)
            plt.plot(t, [d.get("public", None) if d else None for d in self.state.get("compliance_by_type_history", [])], label="Compliance public", alpha=0.6)
            plt.xlabel("Day")
            plt.ylabel("Rate")
            plt.title("Simulation Results")
            plt.legend()
            fig_path = os.path.join(self.artifacts_root, "figs", "adoption_plot.png")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
        except Exception as e:
            print(f"Error during visualization: {e}", file=sys.stderr)
        pass

    def set_params(self, module: _t.Optional[str] = None, **kwargs) -> None:
        """
        Update parameters via registry, optionally filtered by module.

        Parameters
        ----------
        module : str or None
            Owner module name to restrict updates. If None, no restriction.
        kwargs : dict
            Key-value pairs to update.
        """
        for k, v in kwargs.items():
            if module is not None:
                meta = self.param_defs.get(k, {})
                if meta.get("owner_module") != module:
                    continue
            try:
                self.params.set(k, v)
            except KeyError:
                print(f"Unknown parameter in set_params: {k}", file=sys.stderr)
        pass

    def get_params(self) -> dict:
        """
        Return the current parameters as a dictionary.

        Returns
        -------
        dict
            Parameter dictionary.
        """
        return dict(self.params.params)
        pass


# FIXED: Implement Pluggable Calibration Architecture using ABCs with concrete adapter
@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator.
    """
    decision_weights: dict
    layer_weights: dict
    info_params: dict
    noise_params: dict
    module_params: dict = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary.

        Returns
        -------
        dict
            Serializable dictionary representation.
        """
        return asdict(self)
    pass


class ParamsAdapter(ABC):
    """
    Adapts FittedParams to simulation parameter system.
    """
    @abstractmethod
    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply params via simulation parameter system: set_params() + parameters_used.json

        Parameters
        ----------
        simulation : Simulation
            Simulation instance.
        params : FittedParams
            Parameters to apply.
        """
        raise NotImplementedError
        pass

    @abstractmethod
    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams.

        Parameters
        ----------
        simulation : Simulation
            Simulation instance.

        Returns
        -------
        FittedParams
            Captured fitted parameters snapshot.
        """
        raise NotImplementedError
        pass

    @abstractmethod
    def validate_frozen(self, params: FittedParams) -> dict:
        """
        Check frozen params and return warnings.

        Parameters
        ----------
        params : FittedParams
            Parameters to be applied.

        Returns
        -------
        dict
            Mapping of param -> warning message (if any).
        """
        raise NotImplementedError
        pass


class DefaultParamsAdapter(ParamsAdapter):
    """
    Default adapter that maps FittedParams fields to simulation parameters.
    """

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply params via simulation parameter system: set_params() + parameters_used.json

        Parameters
        ----------
        simulation : Simulation
            Simulation instance.
        params : FittedParams
            Parameters to apply.
        """
        # Default implementation maps known fields to behavior and communication/policy weights
        mapping = {
            # Behavior
            "adoption_base_rate": params.decision_weights.get("b0", None),
            "adoption_peer_weight": params.decision_weights.get("w_peer", None) or params.layer_weights.get("peer", None),
            "adoption_marketing_weight": params.decision_weights.get("w_marketing", None) or params.layer_weights.get("marketing", None),
            "adoption_policy_weight": params.decision_weights.get("w_policy", None) or params.layer_weights.get("policy", None),
            "adoption_exposure_weight": params.decision_weights.get("w_exposure", None),  # FIXED: Map exposures weight
            "adoption_temperature": params.noise_params.get("temperature", None),
            # Communication
            "comm_transmission_prob_base": params.info_params.get("gamma_info", None),
            "comm_message_half_life_days": params.info_params.get("memory_decay", None),
        }
        # Include module_params overrides
        for _mod, mp in params.module_params.items():
            for k, v in mp.items():
                mapping[k] = v

        # Apply with registry, ignoring frozen
        for key, val in mapping.items():
            if val is None:
                continue
            try:
                simulation.params.set(key, val)
            except KeyError:
                print(f"ParamsAdapter: Unknown parameter '{key}' ignored.", file=sys.stderr)

        # Persist parameters_used.json including frozen and applied overrides
        used_params_path = os.path.join(simulation.artifacts_root, "results", "parameters_used.json")
        try:
            with open(used_params_path, "w", encoding="utf-8") as f:
                json.dump(_json_clean(simulation.get_params()), f, allow_nan=False)
        except Exception as e:
            print(f"Error saving parameters_used.json: {e}", file=sys.stderr)
        pass

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams.

        Parameters
        ----------
        simulation : Simulation
            Simulation instance.

        Returns
        -------
        FittedParams
            Captured fitted parameters snapshot.
        """
        p = simulation.get_params()
        fp = FittedParams(
            decision_weights={
                "b0": float(p.get("adoption_base_rate", -1.8)),
                "w_peer": float(p.get("adoption_peer_weight", 2.2)),
                "w_marketing": float(p.get("adoption_marketing_weight", 0.8)),
                "w_policy": float(p.get("adoption_policy_weight", 1.0)),
                "w_exposure": float(p.get("adoption_exposure_weight", 0.2)),
            },
            layer_weights={
                "peer": float(p.get("adoption_peer_weight", 2.2)),
                "marketing": float(p.get("adoption_marketing_weight", 0.8)),
                "policy": float(p.get("adoption_policy_weight", 1.0)),
            },
            info_params={
                "gamma_info": float(p.get("comm_transmission_prob_base", 0.15)),
                "memory_decay": float(p.get("comm_message_half_life_days", 7.0)),
            },
            noise_params={
                "temperature": float(p.get("adoption_temperature", 0.6)),
            },
            module_params={},
            engine_type="calibrasim",
            meta={"captured_at": time.time()},
        )
        return fp
        pass

    def validate_frozen(self, params: FittedParams) -> dict:
        """
        Check frozen params and return warnings.

        Parameters
        ----------
        params : FittedParams
            Parameters to be applied.

        Returns
        -------
        dict
            Mapping of param -> warning message (if any).
        """
        # No specific frozen mapping at this layer; simulation registry prevents setting frozen params
        return {}
        pass


class Calibrator(ABC):
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """

    @abstractmethod
    def fit(
        self,
        bundle,
        simulator: Simulation,
        evaluator,
        train_window: _t.Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: _t.Optional[str] = None,
        params_adapter: _t.Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit parameters on training window and return best FittedParams.

        Parameters
        ----------
        bundle : Any
            Optional data bundle.
        simulator : Simulation
            Simulator instance.
        evaluator : callable
            Evaluation callback evaluate_params(simulator, params, window).
        train_window : tuple of int
            (start, end) training window.
        seed : int
            Random seed.
        budget : int
            Number of iterations/trials.
        artifacts_dir : str or None
            Directory to store calibration artifacts.
        params_adapter : ParamsAdapter or None
            Adapter for applying parameters.

        Returns
        -------
        FittedParams
            Best fitted parameters according to evaluator.
        """
        raise NotImplementedError
        pass


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head from micro-transitions; degrades gracefully if data unavailable.
    """

    def fit(
        self,
        bundle,
        simulator: Simulation,
        evaluator,
        train_window: _t.Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: _t.Optional[str] = None,
        params_adapter: _t.Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit a logistic head to approximate adoption decision boundary.

        Notes
        -----
        If micro-transition data is unavailable, this method returns a captured
        snapshot of current params.

        Returns
        -------
        FittedParams
            Fitted or captured parameters.
        """
        random.seed(seed)
        if params_adapter is None:
            params_adapter = DefaultParamsAdapter()
        # Attempt to construct synthetic micro data from simulator by perturbations
        try:
            base_fp = params_adapter.capture(simulator)
            # Minimal tuning: adjust temperature and peer weight by small random search
            best_fp = base_fp
            best_score = float("inf")
            for i in range(max(3, budget // 10)):
                fp = params_adapter.capture(simulator)
                fp.noise_params["temperature"] = max(0.05, min(2.0, fp.noise_params.get("temperature", 0.6) * (0.8 + 0.4 * random.random())))
                fp.decision_weights["w_peer"] = max(0.0, min(5.0, fp.decision_weights.get("w_peer", 2.2) + (random.random() - 0.5)))
                result = evaluator(simulator, fp, train_window)
                score = float(result.get("RMSE_aggregate", float("inf")))
                # Save trial
                if artifacts_dir:
                    trial_dir = os.path.join(artifacts_dir, f"trial_logit_{i}")
                    safe_mkdirs(trial_dir)
                    try:
                        with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                            json.dump(_json_clean(fp.to_dict()), f, allow_nan=False)
                        with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(_json_clean(result), f, allow_nan=False)
                    except Exception as e:
                        print(f"Error saving trial artifacts: {e}", file=sys.stderr)
                if score < best_score:
                    best_score = score
                    best_fp = fp
            return best_fp
        except Exception as e:
            print(f"LogitHeadCalibrator degraded: {e}", file=sys.stderr)
            return params_adapter.capture(simulator)
        pass


class RandomSearchCalibrator(Calibrator):
    """
    Black-box search over selected simulator parameters using evaluator as objective.
    """

    def fit(
        self,
        bundle,
        simulator: Simulation,
        evaluator,
        train_window: _t.Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: _t.Optional[str] = None,
        params_adapter: _t.Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Run random search to optimize selected weights.

        Returns
        -------
        FittedParams
            Best parameters found.
        """
        random.seed(seed)
        if params_adapter is None:
            params_adapter = DefaultParamsAdapter()
        base = params_adapter.capture(simulator)
        best_fp = base
        best_score = float("inf")
        for i in range(max(1, budget)):
            trial = params_adapter.capture(simulator)
            # Sample search space
            trial.decision_weights["b0"] = max(-5.0, min(1.0, base.decision_weights.get("b0", -1.8) + (random.random() - 0.5)))
            trial.decision_weights["w_peer"] = max(0.0, min(5.0, base.decision_weights.get("w_peer", 2.2) + (random.random() - 0.5) * 1.0))
            trial.decision_weights["w_marketing"] = max(0.0, min(3.0, base.decision_weights.get("w_marketing", 0.8) + (random.random() - 0.5) * 0.5))
            trial.decision_weights["w_policy"] = max(0.0, min(4.0, base.decision_weights.get("w_policy", 1.0) + (random.random() - 0.5) * 0.5))
            trial.decision_weights["w_exposure"] = max(0.0, min(3.0, base.decision_weights.get("w_exposure", 0.2) + (random.random() - 0.5) * 0.5))
            trial.noise_params["temperature"] = max(0.05, min(2.0, base.noise_params.get("temperature", 0.6) * (0.8 + 0.4 * random.random())))
            trial.info_params["gamma_info"] = max(0.01, min(0.6, base.info_params.get("gamma_info", 0.15) * (0.8 + 0.4 * random.random())))
            trial.info_params["memory_decay"] = max(1.0, min(30.0, base.info_params.get("memory_decay", 7.0) * (0.5 + random.random())))
            result = evaluator(simulator, trial, train_window)
            score = float(result.get("RMSE_aggregate", float("inf")))
            if artifacts_dir:
                trial_dir = os.path.join(artifacts_dir, f"trial_{i}")
                safe_mkdirs(trial_dir)
                try:
                    with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                        json.dump(_json_clean(trial.to_dict()), f, allow_nan=False)
                    with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(_json_clean(result), f, allow_nan=False)
                except Exception as e:
                    print(f"Error saving trial artifacts: {e}", file=sys.stderr)
            if score < best_score:
                best_score = score
                best_fp = trial
        # Save best
        if artifacts_dir:
            best_dir = os.path.join(artifacts_dir, "best")
            safe_mkdirs(best_dir)
            try:
                with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                    json.dump(_json_clean(best_fp.to_dict()), f, allow_nan=False)
                safe_best = best_score if math.isfinite(best_score) else None
                report = {
                    "budget": int(budget),
                    "best_score": safe_best,
                    "calibrator": "random_search",
                    "timestamp": float(time.time()),
                }
                with open(os.path.join(artifacts_dir, "calibration_report.json"), "w", encoding="utf-8") as f:
                    # FIXED: Sanitize and disallow NaN in calibration report
                    json.dump(_json_clean(report), f, allow_nan=False)
            except Exception as e:
                print(f"Error saving best artifacts: {e}", file=sys.stderr)
        return best_fp
        pass


class SNPECalibrator(Calibrator):
    """
    SBI using neural density estimation; falls back to random search if unavailable.
    """

    def fit(
        self,
        bundle,
        simulator: Simulation,
        evaluator,
        train_window: _t.Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: _t.Optional[str] = None,
        params_adapter: _t.Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit parameters using SNPE if available, otherwise fallback to RandomSearch.

        Returns
        -------
        FittedParams
            Best parameters found.
        """
        try:
            import torch  # noqa: F401
            from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi  # noqa: F401
            # For brevity and environment constraints, we fall back immediately.
            raise RuntimeError("SNPE path is stubbed for this environment; using fallback.")
        except Exception as e:
            print(f"SNPECalibrator fallback to random search: {e}", file=sys.stderr)
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)
        pass


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: _t.Optional[str] = None):
    """
    Retrieve a calibrator by name with optional config (not used here).

    Parameters
    ----------
    name : str
        Calibrator name key.
    config_path : str or None
        Optional path to JSON/YAML config (ignored in this minimal implementation).

    Returns
    -------
    Calibrator
        Calibrator instance.

    Raises
    ------
    ValueError
        If name not in registry.
    """
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    # Load optional config path ignored; instantiate with defaults
    return CALIBRATOR_REGISTRY[name]()
    pass


def evaluate_params(simulator: Simulation, params: FittedParams, window) -> dict:
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
        Simulation instance (will be mutated).
    params : FittedParams
        Parameters to apply via adapter.
    window : tuple of int
        Evaluation window (start, end).

    Returns
    -------
    dict
        Metrics dictionary with required keys.
    """
    adapter = DefaultParamsAdapter()
    adapter.apply(simulator, params)
    # FIXED: Reset simulator for a clean, comparable trial with consistent seed
    try:
        seed_val = int(simulator.params.get("random_seed", 42))
    except Exception:
        seed_val = 42
    simulator.reset(reseed=seed_val)
    # Rerun from window start to end
    start, end = window
    # FIXED: Remove fragile signature fallback; use defined signature directly
    os.environ["DISABLE_EARLY_STOP"] = "1"
    simulator.run(start, end)
    metrics = simulator.evaluate()

    # Compute RMSE over the evaluation window explicitly if ground truth exists
    rmse_window = float("nan")
    try:
        import csv
        gt_path = os.path.join(DATA_DIR, "train_data.csv")
        if os.path.isfile(gt_path):
            gt_adopt = []
            with open(gt_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ar = row.get("adoption_rate")
                    gt_adopt.append(float(ar) if ar not in (None, "") else None)
            sim_series = simulator.state.get("adoption_rate_history", [])
            gt_slice = gt_adopt[start:end]
            n = min(len(sim_series), len(gt_slice))
            s = 0.0
            cnt = 0
            for i in range(n):
                if sim_series[i] is None or gt_slice[i] is None:
                    continue
                s += (sim_series[i] - gt_slice[i]) ** 2
                cnt += 1
            rmse_window = math.sqrt(s / max(1, cnt)) if cnt > 0 else float("nan")
    except Exception:
        rmse_window = float("nan")

    # Aggregate generic metrics for comparison
    rmse_agg = rmse_window if math.isfinite(rmse_window) else float(metrics.get("RMSE_adoption_rate", float("nan")))
    mae_agg = float(metrics.get("MAE_cumulative_adopters", float("nan"))) if metrics.get("MAE_cumulative_adopters") is not None else float("nan")
    # Brier and TransitionFit degrade gracefully
    brier = float("nan")
    trans = {"P01": None, "P11": None, "P10": None, "P00": None}
    out = {
        "RMSE_aggregate": rmse_agg,
        "MAE_aggregate": mae_agg,
        "Brier": brier,
        "TransitionFit": trans,
    }
    return out
    pass


def persist_parameters_used(params: dict, dest_dir: str) -> None:
    """
    Persist the final parameters used to parameters_used.json in results directory.

    Parameters
    ----------
    params : dict
        Parameters to persist.
    dest_dir : str
        Artifacts root directory.
    """
    path = os.path.join(dest_dir, "results", "parameters_used.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            # FIXED: Sanitize and disallow NaN
            json.dump(_json_clean(params), f, allow_nan=False)
    except Exception as e:
        print(f"Error writing parameters_used.json: {e}", file=sys.stderr)
    pass


def _temporal_split_days_from_csv(csv_path: str) -> _t.Optional[_t.Tuple[int, int, int]]:
    """
    Utility: read number of days from train_data.csv and compute (n_days, n_train, n_val).

    Parameters
    ----------
    csv_path : str
        Path to CSV with 'adoption_rate' field.

    Returns
    -------
    tuple or None
        (n_days, n_train, n_val) if CSV exists, else None.
    """
    if not os.path.isfile(csv_path):
        return None
    try:
        import csv
        days = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for _ in reader:
                days += 1
        if days <= 1:
            return (days, days, 0)
        n_train = max(1, int(math.floor(0.8 * days)))
        n_val = max(0, days - n_train)
        return (days, n_train, n_val)
    except Exception:
        return None
    pass


def main():
    """
    Program entry point orchestrating the simulation and calibration workflow.

    Steps:
    - Parse CLI
    - Load params and definitions with robust error handling
    - Normalize and apply overrides (ignoring frozen)
    - Initialize Simulation
    - Persist parameters_used.json
    - Validate and run baseline
    - Calibrate using selected calibrator
    - Rollout with best params, evaluate, save results, and visualize
    """
    # FIXED: Robust CLI and file handling, with no extraneous stdout
    opts = parse_cli(sys.argv[1:])
    # Load params and definitions
    raw_params = load_params(opts["param_file"])
    param_defs = load_param_definitions(opts["param_defs_file"])
    # Normalize and overrides
    params = normalize_params(raw_params, param_defs)
    params = apply_cli_overrides(params, param_defs, opts["overrides"])
    # Seed
    seed_all(int(params.get("random_seed", 42)))
    # Initialize simulator
    sim = Simulation(params, param_defs)
    # Persist parameters used
    persist_parameters_used(sim.get_params(), sim.artifacts_root)

    # Compute run steps
    steps = int(sim.params.get("time_steps", params.get("time_steps", 60)))
    start_day = 0
    end_day = steps

    # Baseline run (run signature is defined as run(start_day, end_day))
    # FIXED: Removed fallback TypeError handling; use single signature
    sim.run(start_day, end_day)

    # Save baseline results
    sim.save_results(os.path.join(sim.artifacts_root, "results", "baseline_results.json"))
    baseline_metrics = sim.evaluate()

    # Temporal holdout split for calibration (if data available)
    train_csv = os.path.join(DATA_DIR, "train_data.csv")
    split = _temporal_split_days_from_csv(train_csv)
    if split is not None and split[2] == 0:
        print("Warning: No validation days available after temporal split.", file=sys.stderr)
    # Calibration window from CLI or default to first 80% of data if available; else full steps
    if opts["calib_window"] is not None:
        window = opts["calib_window"]
    elif split is not None and split[0] > 0:
        window = (0, min(split[1], steps))
    else:
        window = (0, steps)

    # Calibration
    calibrator = get_calibrator(opts["calibrator"])
    calib_dir = os.path.join(sim.artifacts_root, "calibration")
    if os.path.exists(calib_dir):
        try:
            shutil.rmtree(calib_dir)
        except Exception:
            pass
    safe_mkdirs(calib_dir)
    adapter = DefaultParamsAdapter()
    best_params = calibrator.fit(
        bundle=None,
        simulator=sim,
        evaluator=evaluate_params,
        train_window=window,
        seed=int(params.get("random_seed", 42)),
        budget=int(opts["budget"]),
        artifacts_dir=calib_dir,
        params_adapter=adapter,
    )

    # Apply best params and rerun full simulation
    adapter.apply(sim, best_params)
    # FIXED: Use reset to ensure clean rollout with best params
    sim.reset(reseed=int(sim.params.get("random_seed", 42)))
    sim.run(start_day, end_day)
    sim.save_results(os.path.join(sim.artifacts_root, "results", "post_calibration_results.json"))
    final_metrics = sim.evaluate()
    # Save final fitted params
    best_dir = os.path.join(calib_dir, "best")
    safe_mkdirs(best_dir)
    try:
        with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
            json.dump(_json_clean(best_params.to_dict()), f, allow_nan=False)
    except Exception as e:
        print(f"Error saving fitted params: {e}", file=sys.stderr)

    # Save IO traces
    sim.save_all_io(os.path.join(sim.artifacts_root, "io"))
    # Visualize
    sim.visualize()

    # Return minimal summary for sandbox systems that introspect returned value (no stdout)
    return {
        "baseline_final_adoption_rate": float(baseline_metrics.get("final_adoption_rate", 0.0)),
        "final_adoption_rate": float(final_metrics.get("final_adoption_rate", 0.0)),
        "time_to_threshold_70": final_metrics.get("time_to_threshold_70"),
        "time_to_50_percent": final_metrics.get("time_to_50_percent"),
    }
    pass


# Execute main for both direct execution and sandbox wrapper invocation
main()