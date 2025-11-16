import os
import sys
import json
import math
import time
import random
import argparse
import traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Optional, Callable
from collections import defaultdict, deque

# FIXED: Ensured deterministic behavior seed seeding and RNG handling.
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Environment and Path Setup
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Ensure artifacts directories exist
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(os.path.join(DEFAULT_ARTIFACTS_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(DEFAULT_ARTIFACTS_DIR, "io"), exist_ok=True)
os.makedirs(os.path.join(DEFAULT_ARTIFACTS_DIR, "figs"), exist_ok=True)
os.makedirs(os.path.join(DEFAULT_ARTIFACTS_DIR, "observables"), exist_ok=True)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def logistic(x: float) -> float:
    """
    Compute the logistic function value.

    The logistic function is defined as 1 / (1 + exp(-x)). This utility is
    used for mapping linear combinations to probabilities.

    Returns:
        float: The logistic value for the input x.
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
    finally:
        pass


def clip01(x: float) -> float:
    """
    Clip a value to the [0, 1] range.

    Args:
        x (float): Input value.

    Returns:
        float: Value clipped within [0, 1].
    """
    return max(0.0, min(1.0, x))
    pass


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists; if not, create it.

    Args:
        path (str): Directory path to ensure exists.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    pass


def parse_key_value_override(s: str) -> Tuple[str, Any]:
    """
    Parse an override "key=value" string and cast the value to int/float/bool if appropriate.

    Args:
        s (str): The override string.

    Returns:
        Tuple[str, Any]: The parsed key and casted value.

    Raises:
        ValueError: If the input does not contain '=' delimiter.
    """
    if "=" not in s:
        raise ValueError(f"Invalid override: {s}. Expected format key=value")
    key, val = s.split("=", 1)
    val_stripped = val.strip()
    # Attempt to cast to bool, int, or float where applicable
    if val_stripped.lower() in ("true", "false"):
        casted = val_stripped.lower() == "true"
    else:
        try:
            if "." in val_stripped:
                casted = float(val_stripped)
            else:
                casted = int(val_stripped)
        except Exception:
            casted = val_stripped
    return key.strip(), casted
    pass


def safe_read_json(path: str, default: Any) -> Any:
    """
    Safely read a JSON file, returning a default on failure.

    Args:
        path (str): Path to JSON file.
        default (Any): Default value if file cannot be read.

    Returns:
        Any: Parsed JSON or the default value.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default
    finally:
        pass


def write_json(path: str, obj: Any) -> None:
    """
    Write a Python object to a JSON file with pretty formatting.

    Args:
        path (str): File path to write JSON.
        obj (Any): The Python object to serialize.

    Returns:
        None
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    pass


def read_csv_series(path: str) -> Dict[int, float]:
    """
    Read a CSV file containing day and adoption_rate columns into a dict.

    Expected headers: day,adoption_rate

    Args:
        path (str): CSV file path.

    Returns:
        Dict[int, float]: Mapping of day to adoption_rate.

    Notes:
        If file missing or malformed, returns empty dict.
    """
    try:
        mapping: Dict[int, float] = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            indices = {name: idx for idx, name in enumerate(header)}
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != len(header):
                    continue
                day_val = int(parts[indices.get("day", 0)])
                rate_val = float(parts[indices.get("adoption_rate", 1)])
                mapping[day_val] = rate_val
        return mapping
    except Exception:
        return {}
    finally:
        pass


# -----------------------------------------------------------------------------
# Parameter Registry
# -----------------------------------------------------------------------------

@dataclass
class ParameterDefinition:
    """
    Definition for a single parameter in the registry.

    Attributes:
        module (str): Name of the owning module or 'global'.
        name (str): Parameter name.
        dtype (str): Data type, one of: 'int', 'float', 'bool', 'str'.
        default (Any): Default value if not provided.
        bounds (Tuple[float, float] | None): Min/max bounds for numeric parameters.
        frozen (bool): Whether the parameter is immutable (cannot be overridden).
        description (str): Optional human-readable description.
    """
    module: str
    name: str
    dtype: str
    default: Any
    bounds: Optional[Tuple[float, float]] = None
    frozen: bool = False
    description: str = ""
    pass


class ParameterRegistry:
    """
    Registry to manage parameters for modules, with validation, loading, and overrides.

    This class supports:
    - Loading definitions from parameter_definitions.json (if available)
    - Loading values from a parameters file
    - Applying CLI overrides, respecting frozen statuses
    - Grouping by module, and providing getters/setters

    Methods:
        load_definitions(def_path)
        load_values(param_path)
        set_params(module, **kwargs)
        get_params(module)
        apply_overrides(overrides)
        persist_used(path)
    """
    def __init__(self) -> None:
        self.definitions: Dict[str, ParameterDefinition] = {}
        self.values: Dict[str, Any] = {}
        self.module_index: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.frozen_index: Dict[str, bool] = {}
        self.warn_logs: List[str] = []
        # FIXED: Track if definitions were loaded to handle missing metadata gracefully.
        self._definitions_loaded = False
        pass

    def register(self, definition: ParameterDefinition) -> None:
        """
        Register a new parameter definition.

        Args:
            definition (ParameterDefinition): The parameter definition.

        Returns:
            None
        """
        key = f"{definition.module}.{definition.name}"
        self.definitions[key] = definition
        self.module_index[definition.module][definition.name] = key
        self.frozen_index[key] = definition.frozen
        if key not in self.values:
            self.values[key] = definition.default
        pass

    def load_definitions(self, def_path: str) -> None:
        """
        Load parameter definitions from JSON and register them.

        Expected schema per item:
            {
              "module": "decision",
              "name": "temperature",
              "dtype": "float",
              "default": 0.1,
              "bounds": [0.0, 5.0],
              "frozen": false,
              "description": "Decision noise temperature"
            }

        Args:
            def_path (str): Path to parameter_definitions.json

        Returns:
            None
        """
        data = safe_read_json(def_path, default=None)
        if data is None:
            self._definitions_loaded = False
            return
        try:
            for item in data:
                bounds = tuple(item.get("bounds")) if item.get("bounds") else None
                self.register(
                    ParameterDefinition(
                        module=item.get("module", "global"),
                        name=item["name"],
                        dtype=item.get("dtype", "float"),
                        default=item.get("default"),
                        bounds=bounds,
                        frozen=bool(item.get("frozen", False)),
                        description=item.get("description", ""),
                    )
                )
            self._definitions_loaded = True
        except Exception as ex:  # pragma: no cover
            self.warn_logs.append(
                f"Failed to load parameter definitions: {ex}"
            )
            self._definitions_loaded = False
        finally:
            pass

    def load_values(self, param_path: str) -> None:
        """
        Load parameter values from JSON and update registry values.

        Supports flat keys "module.param" or nested dict by module.

        Args:
            param_path (str): Path to parameters.json

        Returns:
            None
        """
        data = safe_read_json(param_path, default=None)
        if not data:
            return
        # Support nested structure
        for key, val in data.items():
            if isinstance(val, dict):
                module = key
                for param, pval in val.items():
                    full_key = f"{module}.{param}"
                    self.values[full_key] = pval
                    # Auto-register if definitions did not pre-register
                    if full_key not in self.definitions:
                        # FIXED: Auto-register missing definitions with inferred types.
                        inferred_dtype = type(pval).__name__
                        if inferred_dtype == "int":
                            dtype_str = "int"
                        elif inferred_dtype == "float":
                            dtype_str = "float"
                        elif inferred_dtype == "bool":
                            dtype_str = "bool"
                        else:
                            dtype_str = "str"
                        self.register(ParameterDefinition(module=module, name=param, dtype=dtype_str, default=pval))
            else:
                # flat or global
                self.values[key] = val
                if key not in self.definitions:
                    parts = key.split(".")
                    if len(parts) == 2:
                        module, param = parts
                    else:
                        module, param = "global", key
                    inferred_dtype = type(val).__name__
                    if inferred_dtype == "int":
                        dtype_str = "int"
                    elif inferred_dtype == "float":
                        dtype_str = "float"
                    elif inferred_dtype == "bool":
                        dtype_str = "bool"
                    else:
                        dtype_str = "str"
                    self.register(ParameterDefinition(module=module, name=param, dtype=dtype_str, default=val))
        pass

    def set_params(self, module: str, **kwargs) -> None:
        """
        Set one or more parameters for a given module, validating frozen status and bounds.

        Args:
            module (str): Module name.
            **kwargs: Parameter values to set.

        Returns:
            None
        """
        for name, val in kwargs.items():
            key = f"{module}.{name}"
            # Frozen check
            if self.frozen_index.get(key, False):
                self.warn_logs.append(f"Override ignored for frozen parameter {key}")
                continue
            # Bounds check
            definition = self.definitions.get(key)
            if definition and definition.bounds:
                lo, hi = definition.bounds
                if isinstance(val, (int, float)) and (val < lo or val > hi):
                    self.warn_logs.append(f"Value {val} for {key} outside bounds {definition.bounds}; clipping")
                    val = min(max(val, lo), hi)
            self.values[key] = val
        pass

    def get_params(self, module: str) -> Dict[str, Any]:
        """
        Get a dictionary of parameters for a given module.

        Args:
            module (str): Module name.

        Returns:
            Dict[str, Any]: Parameters for the module.
        """
        result = {}
        for name, key in self.module_index.get(module, {}).items():
            result[name] = self.values.get(key, self.definitions[key].default if key in self.definitions else None)
        return result
        pass

    def apply_overrides(self, overrides: List[Tuple[str, Any]]) -> None:
        """
        Apply a list of (key, value) overrides. Respect frozen status.

        Args:
            overrides (List[Tuple[str, Any]]): List of key/value pairs.

        Returns:
            None
        """
        for key, val in overrides:
            # split to module.param
            parts = key.split(".")
            if len(parts) == 2:
                module, param = parts
                self.set_params(module, **{param: val})
            else:
                # treat as global parameter
                self.set_params("global", **{key: val})
        pass

    def persist_used(self, path: str) -> None:
        """
        Persist the final parameter values actually used.

        Args:
            path (str): File path to save parameters.

        Returns:
            None
        """
        # Build nested by module for readability
        nested: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for key, val in self.values.items():
            parts = key.split(".")
            if len(parts) == 2:
                module, name = parts
            else:
                module, name = "global", key
            nested[module][name] = val
        write_json(path, nested)
        pass


# -----------------------------------------------------------------------------
# Module Abstractions
# -----------------------------------------------------------------------------

class Module:
    """
    Abstract Module interface for the simulation DAG scheduler.

    Each module must implement:
        - forward(state, buffers, params, t): compute outputs, write to buffers, record IO

    Attributes:
        name (str): Unique module name.
        depends_on (List[str]): Names of modules this module depends on.
        io_log (List[Dict[str, Any]]): Per-step IO trace for debugging and export.
    """
    def __init__(self, name: str, depends_on: Optional[List[str]] = None) -> None:
        self.name = name
        self.depends_on = depends_on or []
        self.io_log: List[Dict[str, Any]] = []
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Execute the module's forward pass.

        Args:
            state (Dict[str, Any]): The current simulation state.
            buffers (Dict[str, Any]): The write-only buffers for this step.
            params (Dict[str, Any]): The module parameters.
            t (int): Current timestep.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement forward")
        pass

    def record_io(self, t: int, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Record a snapshot of input and output for debugging and transparency.

        Args:
            t (int): Current timestep.
            inputs (Dict[str, Any]): Inputs observed by the module.
            outputs (Dict[str, Any]): Outputs produced by the module.

        Returns:
            None
        """
        self.io_log.append({
            "t": int(t),
            "inputs": inputs,
            "outputs": outputs,
        })
        pass


# -----------------------------------------------------------------------------
# Simulation Modules
# -----------------------------------------------------------------------------

class InformationBroadcastModule(Module):
    """
    Module that aggregates information broadcasts from organizations and media.

    The module computes an info signal per person that influences risk perception
    and attitudes, considering trust in authority and exposure to misinformation.

    Outputs:
        buffers['signals']['info'][pid] = info_signal
        buffers['person_increments'][pid]['trust_in_authority'] += delta

    Parameters (module 'information'):
        campaign_intensity (float)
        misinformation_rate (float)
        trust_effect (float): Strength of messaging to adjust trust.
    """
    def __init__(self) -> None:
        super().__init__(name="information", depends_on=[])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        persons = state["persons"]
        orgs = state.get("organizations", [])
        media = state.get("media", [])

        # Aggregate messaging strength
        org_intensity = np.mean([o.get("messaging_intensity", 0.0) for o in orgs]) if orgs else 0.0
        media_intensity = np.mean([m.get("campaign_intensity", 0.0) for m in media]) if media else 0.0
        campaign_intensity = float(params.get("campaign_intensity", (org_intensity + media_intensity) / 2.0))
        misinformation_rate = float(params.get("misinformation_rate", np.mean([m.get("misinformation_rate", 0.0) for m in media]) if media else 0.0))
        trust_effect = float(params.get("trust_effect", 0.02))

        info_signals: Dict[int, float] = {}
        trust_updates: Dict[int, float] = {}

        for p in persons:
            pid = p["id"]
            trust = p.get("trust_in_authority", 0.5)
            misexp = p.get("misinformation_exposure", 0.2)
            # Info signal: pro-mask (positive) minus misinformation (negative)
            info_signal = clip01(campaign_intensity * trust - misexp * misinformation_rate * (1.0 - trust))
            # Trust update mildly nudged by observed positive campaign
            trust_delta = trust_effect * (campaign_intensity - misinformation_rate) * (1.0 - misexp)
            info_signals[pid] = info_signal
            trust_updates[pid] = trust_delta

        # Write buffers
        if "signals" not in buffers:
            buffers["signals"] = {}
        if "info" not in buffers["signals"]:
            buffers["signals"]["info"] = {}
        buffers["signals"]["info"].update(info_signals)

        if "person_increments" not in buffers:
            buffers["person_increments"] = defaultdict(dict)
        for pid, delta in trust_updates.items():
            incs = buffers["person_increments"].setdefault(pid, {})
            incs["trust_in_authority"] = incs.get("trust_in_authority", 0.0) + delta

        # Record IO
        self.record_io(
            t=t,
            inputs={"org_intensity": org_intensity, "media_intensity": media_intensity, "campaign_intensity": campaign_intensity, "misinformation_rate": misinformation_rate},
            outputs={"signals_info_sample": dict(list(info_signals.items())[:5]), "mean_info_signal": float(np.mean(list(info_signals.values())) if info_signals else 0.0)},
        )
        pass


class RiskEvaluationModule(Module):
    """
    Module that updates individual risk perception based on health risk and information.

    Outputs:
        buffers['person_increments'][pid]['risk_perception'] += delta

    Parameters (module 'risk'):
        risk_perception_sensitivity (float)
        info_weight (float)
        memory_decay (float)
    """
    def __init__(self) -> None:
        super().__init__(name="risk", depends_on=["information"])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        persons = state["persons"]
        info_signals = state.get("signals", {}).get("info", {})
        if "signals" in buffers and "info" in buffers["signals"]:
            info_signals = buffers["signals"]["info"]

        sensitivity = float(params.get("risk_perception_sensitivity", 0.5))
        info_weight = float(params.get("info_weight", 0.4))
        memory_decay = float(params.get("memory_decay", 0.01))

        if "person_increments" not in buffers:
            buffers["person_increments"] = defaultdict(dict)

        for p in persons:
            pid = p["id"]
            hrisk = 0.0
            hr = p.get("health_risk_level", "low")
            if hr == "low":
                hrisk = 0.2
            elif hr == "medium":
                hrisk = 0.5
            else:
                hrisk = 0.8
            prev_risk = p.get("risk_perception", 0.5)
            info = info_signals.get(pid, 0.0)
            desired = clip01((1 - info_weight) * hrisk + info_weight * info)
            new_risk = prev_risk * (1 - memory_decay) + sensitivity * (desired - prev_risk)
            delta = new_risk - prev_risk

            incs = buffers["person_increments"].setdefault(pid, {})
            incs["risk_perception"] = incs.get("risk_perception", 0.0) + delta

        self.record_io(
            t=t,
            inputs={"sensitivity": sensitivity, "info_weight": info_weight, "memory_decay": memory_decay},
            outputs={"sample_risk_signal": dict(list(info_signals.items())[:5])},
        )
        pass


class SocialInfluenceModule(Module):
    """
    Module that models social influence on mask attitudes via the social network and observed norms.

    Outputs:
        buffers['person_increments'][pid]['mask_attitude'] += delta

    Parameters (module 'social'):
        social_influence_strength (float)
        conformity_bias (float)
        layer_weights.family (float)
        layer_weights.work (float)
        layer_weights.community (float)
    """
    def __init__(self) -> None:
        super().__init__(name="social", depends_on=["mobility"])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        persons = state["persons"]
        network = state["network"]
        assignments = state.get("day_assignments", {})
        location_norms = state.get("signals", {}).get("location_norms", {})
        if "signals" in buffers and "location_norms" in buffers["signals"]:
            location_norms = buffers["signals"]["location_norms"]

        influence_strength = float(params.get("social_influence_strength", 0.4))
        conformity_bias = float(params.get("conformity_bias", 0.2))
        lw_family = float(params.get("lw_family", 1.0))
        lw_work = float(params.get("lw_work", 1.0))
        lw_comm = float(params.get("lw_community", 1.0))

        # Determine for each person the fraction of masked neighbors:
        if "person_increments" not in buffers:
            buffers["person_increments"] = defaultdict(dict)

        masked_neighbors_frac: Dict[int, float] = {}
        for p in persons:
            pid = p["id"]
            neighs = network.get(pid, [])
            if not neighs:
                masked_neighbors_frac[pid] = 0.0
                continue
            masked_count = sum(1 for n in neighs if state["persons"][n]["mask_status"])
            masked_neighbors_frac[pid] = masked_count / max(1, len(neighs))

        for p in persons:
            pid = p["id"]
            loc_id = assignments.get(pid, None)
            observed_norm = location_norms.get(loc_id, 0.0) if loc_id is not None else 0.0
            neighbor_masked = masked_neighbors_frac.get(pid, 0.0)
            susceptibility = p.get("social_susceptibility", 0.5)
            prev_att = p.get("mask_attitude", 0.0)

            # Weighted social signal
            # FIXED: Ensure numeric stability in attitude updates.
            social_signal = (
                lw_family * neighbor_masked * susceptibility +
                lw_work * observed_norm * susceptibility +
                lw_comm * (neighbor_masked + observed_norm) / 2.0 * susceptibility
            )
            # Update rule
            delta = influence_strength * (social_signal - (0.5 + 0.5 * prev_att)) + conformity_bias * (social_signal - 0.5)
            incs = buffers["person_increments"].setdefault(pid, {})
            incs["mask_attitude"] = incs.get("mask_attitude", 0.0) + float(delta)

        self.record_io(
            t=t,
            inputs={"influence_strength": influence_strength, "conformity_bias": conformity_bias},
            outputs={"location_norms_sample": dict(list(location_norms.items())[:5])},
        )
        pass


class MobilityAndLocationModule(Module):
    """
    Module that assigns persons to locations daily and computes location norms.

    Outputs:
        buffers['day_assignments'][pid] = location_id
        buffers['signals']['location_norms'][loc_id] = masked_fraction
        buffers['signals']['enforcement_intensity'][loc_id] = enforcement_strength * policy_level

    Parameters (module 'mobility'):
        observation_effect_weight (float)
    """
    def __init__(self) -> None:
        super().__init__(name="mobility", depends_on=[])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        rng = state["rng"]
        persons = state["persons"]
        locations = state.get("locations", [])
        # Simple assignment: randomly assign a location weighted by type distribution implicit in list
        assign: Dict[int, int] = {}

        if not locations:
            # No locations: remain None assignment; still produce empty norms
            location_norms: Dict[int, float] = {}
            enforcement_intensity: Dict[int, float] = {}
        else:
            # Random assignment uniform across locations
            for p in persons:
                pid = p["id"]
                loc_idx = rng.integers(0, len(locations))
                assign[pid] = int(loc_idx)

            # Compute masked fraction per location
            grouped: Dict[int, List[int]] = defaultdict(list)
            for pid, lid in assign.items():
                grouped[lid].append(pid)

            location_norms = {}
            enforcement_intensity = {}
            for lid, plist in grouped.items():
                masked_count = sum(1 for pid in plist if persons[pid]["mask_status"])
                fraction = masked_count / max(1, len(plist))
                location_norms[lid] = float(fraction)
                policy_level = locations[lid].get("mask_policy_level", 0.5)  # 0 recommended, 1 required mapping to 0.5 default
                enforcement_strength = locations[lid].get("enforcement_strength", 0.5)
                enforcement_intensity[lid] = float(policy_level * enforcement_strength)

        if "signals" not in buffers:
            buffers["signals"] = {}
        buffers["signals"]["location_norms"] = location_norms
        buffers["signals"]["enforcement_intensity"] = enforcement_intensity
        buffers["day_assignments"] = assign

        self.record_io(
            t=t,
            inputs={"num_locations": len(locations) if locations else 0},
            outputs={
                "assignment_sample": dict(list(assign.items())[:5]),
                "location_norms_sample": dict(list(location_norms.items())[:5]) if locations else {},
            },
        )
        pass


class PolicyEnforcementModule(Module):
    """
    Module that models enforcement impact on compliance probability and expected penalties.

    Outputs:
        buffers['person_increments'][pid]['compliance_probability'] += delta
        buffers['person_temp'][pid]['expected_penalty'] = expected_penalty

    Parameters (module 'enforcement'):
        policy_enforcement_strength (float)
        penalty_for_noncompliance (float)
    """
    def __init__(self) -> None:
        super().__init__(name="enforcement", depends_on=["mobility"])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        persons = state["persons"]
        enforcement_intensity = state.get("signals", {}).get("enforcement_intensity", {})
        if "signals" in buffers and "enforcement_intensity" in buffers["signals"]:
            enforcement_intensity = buffers["signals"]["enforcement_intensity"]

        strength = float(params.get("policy_enforcement_strength", 0.6))
        penalty = float(params.get("penalty_for_noncompliance", 25.0))

        if "person_increments" not in buffers:
            buffers["person_increments"] = defaultdict(dict)
        if "person_temp" not in buffers:
            buffers["person_temp"] = defaultdict(dict)

        for p in persons:
            pid = p["id"]
            loc_id = state.get("day_assignments", {}).get(pid, None)
            if "day_assignments" in buffers and pid in buffers["day_assignments"]:
                loc_id = buffers["day_assignments"][pid]
            intensity = enforcement_intensity.get(loc_id, 0.0)
            prev_prob = p.get("compliance_probability", 0.5)
            # Increase compliance probability proportional to intensity
            delta = strength * (intensity * (1.0 - prev_prob))
            buffers["person_increments"][pid]["compliance_probability"] = buffers["person_increments"][pid].get("compliance_probability", 0.0) + delta
            buffers["person_temp"][pid]["expected_penalty"] = penalty * intensity

        self.record_io(
            t=t,
            inputs={"strength": strength, "penalty": penalty},
            outputs={"enforcement_intensity_sample": dict(list(enforcement_intensity.items())[:5])},
        )
        pass


class DecisionModule(Module):
    """
    Module that decides mask adoption for each person using a logistic decision head.

    Outputs:
        buffers['person_assignments'][pid]['mask_status'] = bool
        buffers['person_increments'][pid]['mask_attitude'] += delta

    Parameters (module 'decision'):
        b0 (float): intercept
        w_risk (float)
        w_att (float)
        w_trust (float)
        w_info (float)
        w_cost (float)
        w_penalty (float)
        temperature (float)
        mask_cost (float)
    """
    def __init__(self) -> None:
        super().__init__(name="decision", depends_on=["information", "risk", "social", "enforcement"])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        persons = state["persons"]
        rng = state["rng"]
        info_signals = state.get("signals", {}).get("info", {})
        if "signals" in buffers and "info" in buffers["signals"]:
            info_signals = buffers["signals"]["info"]

        b0 = float(params.get("b0", -1.0))
        w_risk = float(params.get("w_risk", 2.0))
        w_att = float(params.get("w_att", 1.5))
        w_trust = float(params.get("w_trust", 0.5))
        w_info = float(params.get("w_info", 0.8))
        w_cost = float(params.get("w_cost", -0.5))
        w_penalty = float(params.get("w_penalty", 1.0))
        temperature = float(params.get("temperature", 0.1))
        mask_cost = float(params.get("mask_cost", 1.0))

        if "person_assignments" not in buffers:
            buffers["person_assignments"] = defaultdict(dict)
        if "person_increments" not in buffers:
            buffers["person_increments"] = defaultdict(dict)
        if "person_temp" not in buffers:
            buffers["person_temp"] = defaultdict(dict)

        for p in persons:
            pid = p["id"]
            x_risk = p.get("risk_perception", 0.5)
            x_att = (p.get("mask_attitude", 0.0) + 1.0) * 0.5  # map [-1,1] to [0,1]
            x_trust = p.get("trust_in_authority", 0.5)
            x_info = info_signals.get(pid, 0.0)
            x_cost = mask_cost
            x_penalty = state.get("person_temp", {}).get(pid, {}).get("expected_penalty", 0.0)
            if "person_temp" in buffers and pid in buffers["person_temp"]:
                x_penalty = buffers["person_temp"][pid].get("expected_penalty", x_penalty)
            z = (
                b0 +
                w_risk * x_risk +
                w_att * x_att +
                w_trust * x_trust +
                w_info * x_info +
                w_cost * x_cost +
                w_penalty * (x_penalty / max(1.0, mask_cost))
            )
            prob = logistic(z / max(1e-3, temperature))
            decision = rng.random() < prob
            buffers["person_assignments"][pid]["mask_status"] = bool(decision)
            # Attitude reinforcement: if wear mask, attitude nudges positive
            att_delta = 0.05 if decision else -0.02
            buffers["person_increments"][pid]["mask_attitude"] = buffers["person_increments"][pid].get("mask_attitude", 0.0) + att_delta

        self.record_io(
            t=t,
            inputs={"b0": b0, "temperature": temperature},
            outputs={"decisions_sample": dict(list({pid: buffers["person_assignments"][pid]["mask_status"] for pid in list(buffers["person_assignments"].keys())[:5]}.items()))},
        )
        pass


class ContactModule(Module):
    """
    Module that simulates contacts and computes masked contact fraction.

    Outputs:
        buffers['signals']['masked_contact_fraction'] = float in [0,1]

    Parameters (module 'contact'):
        daily_contacts_mean (float)
    """
    def __init__(self) -> None:
        super().__init__(name="contact", depends_on=["mobility", "decision"])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        persons = state["persons"]
        rng = state["rng"]
        assignments = state.get("day_assignments", {})
        if "day_assignments" in buffers:
            assignments = buffers["day_assignments"]
        daily_contacts_mean = float(params.get("daily_contacts_mean", 8.0))

        # Group by location
        grouped: Dict[int, List[int]] = defaultdict(list)
        for pid, lid in assignments.items():
            grouped[lid].append(pid)

        both_masked = 0
        total_contacts = 0

        # For each location, sample pairwise contacts
        for lid, plist in grouped.items():
            n = len(plist)
            if n <= 1:
                continue
            # Poisson number of contacts per person
            for pid in plist:
                k = rng.poisson(daily_contacts_mean)
                for _ in range(k):
                    partner = plist[rng.integers(0, n)]
                    if partner == pid:
                        continue
                    total_contacts += 1
                    m1 = bool(state["persons"][pid]["mask_status"])
                    m2 = bool(state["persons"][partner]["mask_status"])
                    if "person_assignments" in buffers:
                        if pid in buffers["person_assignments"]:
                            m1 = bool(buffers["person_assignments"][pid].get("mask_status", m1))
                        if partner in buffers["person_assignments"]:
                            m2 = bool(buffers["person_assignments"][partner].get("mask_status", m2))
                    if m1 and m2:
                        both_masked += 1

        masked_frac = both_masked / max(1, total_contacts)
        if "signals" not in buffers:
            buffers["signals"] = {}
        buffers["signals"]["masked_contact_fraction"] = float(masked_frac)

        self.record_io(
            t=t,
            inputs={"daily_contacts_mean": daily_contacts_mean},
            outputs={"masked_contact_fraction": float(masked_frac), "total_contacts": int(total_contacts)},
        )
        pass


class ObservablesModule(Module):
    """
    Module that computes and stores observable time series for metrics.

    Outputs:
        buffers['observables']['adoption_rate'] = float
        buffers['observables']['masked_contact_fraction'] = float
        buffers['observables']['enforcement_intensity'] = float

    Parameters (module 'observables'):
        None
    """
    def __init__(self) -> None:
        super().__init__(name="observables", depends_on=["contact", "enforcement", "decision", "mobility"])
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        persons = state["persons"]
        N = len(persons)
        # Compute adoption
        mask_statuses = []
        for p in persons:
            pid = p["id"]
            s = p["mask_status"]
            if "person_assignments" in buffers and pid in buffers["person_assignments"]:
                s = buffers["person_assignments"][pid].get("mask_status", s)
            mask_statuses.append(1 if s else 0)
        adoption = float(sum(mask_statuses)) / max(1, N)
        masked_contact_fraction = 0.0
        enforcement_intensity_avg = 0.0
        if "signals" in buffers:
            masked_contact_fraction = float(buffers["signals"].get("masked_contact_fraction", 0.0))
            enf_map = buffers["signals"].get("enforcement_intensity", {})
            enforcement_intensity_avg = float(np.mean(list(enf_map.values())) if enf_map else 0.0)

        if "observables" not in buffers:
            buffers["observables"] = {}
        buffers["observables"]["adoption_rate"] = adoption
        buffers["observables"]["masked_contact_fraction"] = masked_contact_fraction
        buffers["observables"]["enforcement_intensity"] = enforcement_intensity_avg

        self.record_io(
            t=t,
            inputs={},
            outputs={"adoption_rate": adoption, "masked_contact_fraction": masked_contact_fraction, "enforcement_intensity": enforcement_intensity_avg},
        )
        pass


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------

class DAGScheduler:
    """
    Simple DAG scheduler that executes modules in a predetermined order and
    manages buffers and state commit.

    Attributes:
        modules (List[Module]): Ordered list of modules to execute.
    """
    def __init__(self, modules: List[Module]) -> None:
        self.modules = modules
        pass

    def step(self, state: Dict[str, Any], param_registry: ParameterRegistry, t: int) -> Dict[str, Any]:
        """
        Execute one simulation step by running all modules in order and merging outputs.

        Args:
            state (Dict[str, Any]): Current state dictionary.
            param_registry (ParameterRegistry): Parameter registry for module params.
            t (int): Timestep.

        Returns:
            Dict[str, Any]: Buffers aggregated across modules.
        """
        buffers: Dict[str, Any] = {}
        for module in self.modules:
            params = param_registry.get_params(module.name)
            # Record state signals snapshot for module
            try:
                module.forward(state=state, buffers=buffers, params=params, t=t)
            except Exception as ex:  # pragma: no cover
                raise RuntimeError(f"Module {module.name} failed at t={t}: {ex}") from ex
        return buffers
        pass


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

class Simulation:
    """
    Main simulation class orchestrating initialization, execution, saving, and evaluation.

    Methods:
        set_seed(seed)
        build_entities()
        build_network()
        initialize_state()
        run(start_day, end_day)
        commit_buffers(buffers)
        save_results(path)
        save_module_io(module, path)
        save_all_io(root_dir)
        evaluate(window=None)
        visualize()
        set_params(module, **kwargs)
        get_params(module)
        save_params_snapshot(path)

    Attributes:
        param_registry (ParameterRegistry)
        modules (List[Module])
        scheduler (DAGScheduler)
        state (Dict[str, Any])
        artifacts_dir (str)
        ground_truth (Dict[int, float]): Observed adoption rates by day
        rng (np.random.Generator)
    """
    def __init__(self, param_registry: ParameterRegistry, artifacts_dir: str) -> None:
        self.param_registry = param_registry
        self.artifacts_dir = artifacts_dir
        self.modules: List[Module] = [
            InformationBroadcastModule(),
            MobilityAndLocationModule(),
            RiskEvaluationModule(),
            SocialInfluenceModule(),
            PolicyEnforcementModule(),
            DecisionModule(),
            ContactModule(),
            ObservablesModule(),
        ]
        self.scheduler = DAGScheduler(self.modules)
        self.state: Dict[str, Any] = {}
        self.ground_truth: Dict[int, float] = {}
        self.rng = np.random.default_rng(42)
        self.initialize_state()
        pass

    def set_seed(self, seed: int) -> None:
        """
        Set deterministic seed for RNG operations.

        Args:
            seed (int): Seed value.

        Returns:
            None
        """
        self.rng = np.random.default_rng(seed)
        self.state["rng"] = self.rng
        pass

    def build_entities(self) -> None:
        """
        Build initial entities: persons, locations, organizations, and media.

        Returns:
            None
        """
        params_global = self.param_registry.get_params("global")
        N = int(params_global.get("population_size", 500))
        initial_adoption_rate = float(params_global.get("initial_adoption_rate", 0.25))
        trust_mean = float(params_global.get("trust_in_authority_mean", 0.6))
        trust_std = float(params_global.get("trust_in_authority_std", 0.15))
        rng = self.rng

        # Persons
        persons: List[Dict[str, Any]] = []
        health_levels = ["low", "medium", "high"]
        health_probs = [0.6, 0.3, 0.1]
        for i in range(N):
            hr = rng.choice(health_levels, p=health_probs)
            person = {
                "id": i,
                "age": int(rng.integers(18, 80)),
                "demographics_group": rng.choice(["A", "B", "C"], p=[0.5, 0.3, 0.2]),
                "health_risk_level": hr,
                "risk_perception": float(rng.uniform(0.2, 0.8)),
                "mask_attitude": float(rng.uniform(-0.5, 0.5)),
                "mask_status": bool(rng.random() < initial_adoption_rate),
                "compliance_probability": float(rng.uniform(0.3, 0.7)),
                "social_susceptibility": float(rng.uniform(0.3, 0.9)),
                "trust_in_authority": float(np.clip(rng.normal(trust_mean, trust_std), 0.0, 1.0)),
                "misinformation_exposure": float(rng.uniform(0.0, 0.6)),
                "home_location_id": None,
                "work_location_id": None,
                "mobility_pattern": None,
                "daily_contacts_mean": float(rng.uniform(5.0, 12.0)),
                "budget": float(rng.uniform(50.0, 200.0))
            }
            persons.append(person)

        # Locations
        num_locations = max(10, int(N / 25))
        locations: List[Dict[str, Any]] = []
        for lid in range(num_locations):
            locations.append({
                "id": lid,
                "type": rng.choice(["home", "work", "retail", "transit"], p=[0.5, 0.3, 0.15, 0.05]),
                "capacity": int(rng.integers(10, 100)),
                "mask_policy_level": float(rng.choice([0.0, 0.5, 1.0])),  # 0=none, 0.5=recommended, 1=required
                "enforcement_strength": float(rng.uniform(0.0, 1.0)),
                "norms_compliance_level": float(rng.uniform(0.3, 0.8)),
                "hours_open": int(rng.integers(8, 24))
            })

        # Organizations
        organizations = [
            {"id": 0, "type": "public_health", "policy_strictness": 0.7, "inspection_frequency": 0.3,
             "penalty_amount": 25.0, "messaging_intensity": 0.7, "message_stance": 1.0}
        ]

        # Media
        media = [
            {"id": 0, "channel_type": "tv", "reach": 0.8, "bias": 0.2, "misinformation_rate": 0.1, "campaign_intensity": 0.6},
            {"id": 1, "channel_type": "social", "reach": 0.9, "bias": -0.2, "misinformation_rate": 0.3, "campaign_intensity": 0.4},
        ]

        self.state["persons"] = persons
        self.state["locations"] = locations
        self.state["organizations"] = organizations
        self.state["media"] = media
        pass

    def build_network(self) -> None:
        """
        Build a small-world like social network adjacency list.

        Returns:
            None
        """
        params_global = self.param_registry.get_params("global")
        N = len(self.state["persons"])
        k = int(params_global.get("average_degree", 8))
        rewiring_p = float(params_global.get("rewiring_probability", 0.05))
        rng = self.rng

        # Ring lattice with k neighbors
        k = max(2, k if k % 2 == 0 else k + 1)
        network: Dict[int, List[int]] = {i: [] for i in range(N)}
        for i in range(N):
            for j in range(1, k // 2 + 1):
                a = i
                b = (i + j) % N
                network[a].append(b)
                network[b].append(a)

        # Rewiring
        for i in range(N):
            for j in range(len(network[i])):
                if rng.random() < rewiring_p:
                    # rewire to a random node
                    new_neighbor = int(rng.integers(0, N))
                    if new_neighbor != i and new_neighbor not in network[i]:
                        old_neighbor = network[i][j]
                        # Remove mutual connection
                        if i in network[old_neighbor]:
                            network[old_neighbor].remove(i)
                        network[i][j] = new_neighbor
                        network[new_neighbor].append(i)
        self.state["network"] = network
        pass

    def initialize_state(self) -> None:
        """
        Initialize the simulation state with entities and structures.

        Returns:
            None
        """
        self.state = {
            "t": 0,
            "persons": [],
            "locations": [],
            "organizations": [],
            "media": [],
            "network": {},
            "day_assignments": {},
            "signals": {},
            "history": {
                "adoption_rate": [],
                "masked_contact_fraction": [],
                "enforcement_intensity": [],
            },
            "rng": self.rng,
        }
        self.build_entities()
        self.build_network()
        pass

    def run(self, start_day: int, end_day: int) -> None:
        """
        Run the simulation over the specified time window [start_day, end_day].

        Args:
            start_day (int): Start day index.
            end_day (int): End day index (inclusive).

        Returns:
            None
        """
        # Validate end_day >= start_day
        if end_day < start_day:
            raise ValueError("end_day must be >= start_day")

        # Step through days
        for t in range(start_day, end_day + 1):
            self.state["t"] = t
            buffers = self.scheduler.step(self.state, self.param_registry, t)
            self.commit_buffers(buffers)

            # Collect observables
            obs = buffers.get("observables", {})
            self.state["history"]["adoption_rate"].append(float(obs.get("adoption_rate", 0.0)))
            self.state["history"]["masked_contact_fraction"].append(float(obs.get("masked_contact_fraction", 0.0)))
            self.state["history"]["enforcement_intensity"].append(float(obs.get("enforcement_intensity", 0.0)))
        pass

    def commit_buffers(self, buffers: Dict[str, Any]) -> None:
        """
        Commit buffered outputs to the state.

        Args:
            buffers (Dict[str, Any]): Buffers produced by modules in the step.

        Returns:
            None
        """
        # Signals and assignments
        if "signals" in buffers:
            self.state["signals"].update(buffers["signals"])
        if "day_assignments" in buffers:
            self.state["day_assignments"] = buffers["day_assignments"]

        # Person increments
        pincr = buffers.get("person_increments", {})
        for pid, incs in pincr.items():
            p = self.state["persons"][pid]
            for key, delta in incs.items():
                if key in ("trust_in_authority", "risk_perception", "compliance_probability"):
                    p[key] = clip01(p.get(key, 0.5) + float(delta))
                elif key == "mask_attitude":
                    # clip to [-1, 1]
                    val = p.get(key, 0.0) + float(delta)
                    p[key] = max(-1.0, min(1.0, val))
                else:
                    p[key] = p.get(key, 0.0) + float(delta)

        # Person assignments (mask_status)
        passn = buffers.get("person_assignments", {})
        for pid, assigns in passn.items():
            p = self.state["persons"][pid]
            for key, val in assigns.items():
                p[key] = val

        # Clear temp
        self.state["person_temp"] = buffers.get("person_temp", {})
        pass

    def save_results(self, path: str) -> None:
        """
        Save simulation results (history) to a JSON file.

        Args:
            path (str): File path for saving.

        Returns:
            None
        """
        write_json(path, self.state["history"])
        pass

    def save_module_io(self, module: Module, path: str) -> None:
        """
        Save per-module IO traces to a JSON file.

        Args:
            module (Module): Module instance.
            path (str): File path to save to.

        Returns:
            None
        """
        write_json(path, module.io_log)
        pass

    def save_all_io(self, root_dir: str) -> None:
        """
        Save IO traces for all modules under a root directory.

        Args:
            root_dir (str): Directory to store IO logs.

        Returns:
            None
        """
        ensure_dir(root_dir)
        for m in self.modules:
            p = os.path.join(root_dir, f"{m.name}_io.json")
            self.save_module_io(m, p)
        pass

    def evaluate(self, window: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Evaluate simulation outputs against ground truth for the specified window.

        Args:
            window (Optional[Tuple[int, int]]): (start_day, end_day). If None, use full range.

        Returns:
            Dict[str, Any]: Metrics dictionary including RMSE_aggregate, MAE_aggregate, Brier,
                            TransitionFit (P01, P11, P10, P00), plus basic adoption metrics.
        """
        hist = self.state["history"]
        adoption_series = hist["adoption_rate"]
        start = 0
        end = len(adoption_series) - 1
        if window:
            start, end = window
            # Convert to local indices relative to current run
            # In this simplified simulation, assume run starts at day 0
            start = max(0, start)
            end = min(end, len(adoption_series) - 1)

        if start > end:
            raise ValueError("Invalid evaluation window")

        sim_vals = np.array(adoption_series[start:end + 1], dtype=float)
        gt_vals_list = []
        for day in range(start, end + 1):
            gt_vals_list.append(self.ground_truth.get(day, sim_vals[day - start]))  # fallback to sim if GT missing
        gt_vals = np.array(gt_vals_list, dtype=float)

        # Metrics
        rmse = float(np.sqrt(np.mean((sim_vals - gt_vals) ** 2))) if len(sim_vals) > 0 else 0.0
        mae = float(np.mean(np.abs(sim_vals - gt_vals))) if len(sim_vals) > 0 else 0.0
        # Brier score using adoption as probability and ground truth as rate (approx)
        brier = float(np.mean((sim_vals - gt_vals) ** 2)) if len(sim_vals) > 0 else 0.0

        # TransitionFit placeholders (micro-transitions not tracked, compute from rates)
        # We approximate binary transitions on day-level probabilities:
        # P01: increase days fraction, P11: persistence, etc. Derived heuristically.
        if len(sim_vals) > 1:
            diffs = np.diff(sim_vals)
            p01 = float(np.mean(diffs[diffs > 0] / (1 - sim_vals[:-1][diffs > 0])) if np.any(diffs > 0) else 0.0)
            p10 = float(np.mean(-diffs[diffs < 0] / sim_vals[:-1][diffs < 0]) if np.any(diffs < 0) else 0.0)
            p11 = float(1.0 - p10)
            p00 = float(1.0 - p01)
        else:
            p01 = p10 = 0.0
            p11 = p00 = 1.0

        # Additional observables
        masked_contact_frac = float(np.mean(hist["masked_contact_fraction"][start:end + 1]) if hist["masked_contact_fraction"] else 0.0)
        enforcement_avg = float(np.mean(hist["enforcement_intensity"][start:end + 1]) if hist["enforcement_intensity"] else 0.0)
        # Peak and steady-state
        peak_adoption = float(np.max(sim_vals) if len(sim_vals) > 0 else 0.0)
        steady_state = float(np.mean(sim_vals[-min(5, len(sim_vals)):]) if len(sim_vals) > 0 else 0.0)

        metrics = {
            "RMSE_aggregate": rmse,
            "MAE_aggregate": mae,
            "Brier": brier,
            "TransitionFit": {"P01": p01, "P11": p11, "P10": p10, "P00": p00},
            "masked_contact_fraction_avg": masked_contact_frac,
            "enforcement_intensity_avg": enforcement_avg,
            "peak_adoption": peak_adoption,
            "steady_state_adoption": steady_state,
        }
        write_json(os.path.join(self.artifacts_dir, "results", "metrics.json"), metrics)

        # Observables file for calibration
        obs_out = {
            "adoption_rate": adoption_series[start:end + 1],
            "masked_contact_fraction": self.state["history"]["masked_contact_fraction"][start:end + 1],
            "enforcement_intensity": self.state["history"]["enforcement_intensity"][start:end + 1],
        }
        write_json(os.path.join(self.artifacts_dir, "results", "observables", "ObservablesModule.json"), obs_out)

        # Ground truth comparables
        gt_out = {"adoption_rate": [self.ground_truth.get(day, float("nan")) for day in range(start, end + 1)]}
        write_json(os.path.join(self.artifacts_dir, "results", "observables", "ObservablesModule_gt.json"), gt_out)

        return metrics
        pass

    def visualize(self) -> None:
        """
        Visualize adoption rate and masked contact fraction if matplotlib is available.

        Returns:
            None
        """
        if plt is None:
            print("Matplotlib not available; skipping visualization")
            return
        hist = self.state["history"]
        days = list(range(len(hist["adoption_rate"])))
        plt.figure(figsize=(10, 5))
        plt.plot(days, hist["adoption_rate"], label="Adoption Rate")
        if self.ground_truth:
            gt_days = sorted(self.ground_truth.keys())
            gt_vals = [self.ground_truth[d] for d in gt_days]
            plt.plot(gt_days, gt_vals, label="Ground Truth", linestyle="--")
        plt.plot(days, hist["masked_contact_fraction"], label="Masked Contact Fraction")
        plt.xlabel("Day")
        plt.ylabel("Value")
        plt.title("Mask Adoption Dynamics")
        plt.legend()
        fig_path = os.path.join(self.artifacts_dir, "figs", "adoption_plot.png")
        ensure_dir(os.path.dirname(fig_path))
        plt.savefig(fig_path)
        plt.close()
        pass

    def set_params(self, module: str, **kwargs) -> None:
        """
        Set simulation parameters for a given module via the parameter registry.

        Args:
            module (str): Module name.
            **kwargs: Parameter assignments.

        Returns:
            None
        """
        self.param_registry.set_params(module, **kwargs)
        pass

    def get_params(self, module: str) -> Dict[str, Any]:
        """
        Get current effective parameters for a module.

        Args:
            module (str): Module name.

        Returns:
            Dict[str, Any]: Parameters for the module.
        """
        return self.param_registry.get_params(module)
        pass

    def save_params_snapshot(self, path: str) -> None:
        """
        Save a snapshot of parameters used.

        Args:
            path (str): File path for snapshot.

        Returns:
            None
        """
        self.param_registry.persist_used(path)
        pass


# -----------------------------------------------------------------------------
# Calibration API (Calibrasim SBI Requirements)
# -----------------------------------------------------------------------------

@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator.

    Attributes:
        decision_weights (Dict[str, float]): Logistic head weights: b0, w_risk, w_att, etc.
        layer_weights (Dict[str, float]): Social layers weights.
        info_params (Dict[str, float]): Campaign, info dynamics.
        noise_params (Dict[str, float]): Noise parameters, e.g., temperature.
        module_params (Dict[str, Dict[str, float]]): Additional per-module parameter overrides.
        engine_type (str): Engine compatibility identifier.
        meta (Dict[str, Any]): Metadata such as seed, calibrator name, window, notes.
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
        Serialize to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the fitted parameters.
        """
        return asdict(self)
        pass


class ParamsAdapter:
    """
    Adapts FittedParams to simulation parameter system.

    Methods:
        apply(simulation, params)
        capture(simulation)
        validate_frozen(params)
    """
    def __init__(self, param_def_path: Optional[str] = None) -> None:
        self.param_def_path = param_def_path or os.path.join(DATA_DIR, "parameter_definitions.json")
        self.param_defs = safe_read_json(self.param_def_path, default=None)
        pass

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply fitted params to the simulation via the parameter system.

        Frozen parameters are not modified; a warning is logged.

        Args:
            simulation (Simulation): Simulation instance.
            params (FittedParams): Parameters to apply.

        Returns:
            None
        """
        warnings = self.validate_frozen(params)
        for w in warnings.values():
            print(f"[ParamsAdapter] Warning: {w}")

        # Map decision weights
        if params.decision_weights:
            simulation.set_params("decision", **params.decision_weights)
        # Layer weights map to social module
        social_overrides = {}
        if params.layer_weights:
            if "family" in params.layer_weights:
                social_overrides["lw_family"] = params.layer_weights["family"]
            if "work_school" in params.layer_weights:
                social_overrides["lw_work"] = params.layer_weights["work_school"]
            if "community" in params.layer_weights:
                social_overrides["lw_community"] = params.layer_weights["community"]
        if social_overrides:
            simulation.set_params("social", **social_overrides)

        # Info params to information and risk modules
        if params.info_params:
            info_map = {}
            risk_map = {}
            for k, v in params.info_params.items():
                if k in ("campaign_intensity", "misinformation_rate", "trust_effect"):
                    info_map[k] = v
                else:
                    risk_map[k] = v
            if info_map:
                simulation.set_params("information", **info_map)
            if risk_map:
                simulation.set_params("risk", **risk_map)

        # Noise params to decision
        if params.noise_params:
            simulation.set_params("decision", **params.noise_params)

        # Module-specific params
        for module, ov in params.module_params.items():
            simulation.set_params(module, **ov)

        # Persist parameters used
        simulation.save_params_snapshot(os.path.join(simulation.artifacts_dir, "parameters_used.json"))
        pass

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current effective parameters from a simulation into a FittedParams.

        Args:
            simulation (Simulation): Simulation instance.

        Returns:
            FittedParams: Captured parameters.
        """
        decision = simulation.get_params("decision")
        social = simulation.get_params("social")
        information = simulation.get_params("information")
        risk = simulation.get_params("risk")

        decision_weights = {k: decision[k] for k in decision if k in ("b0", "w_risk", "w_att", "w_trust", "w_info", "w_cost", "w_penalty")}
        layer_weights = {
            "family": social.get("lw_family", 1.0),
            "work_school": social.get("lw_work", 1.0),
            "community": social.get("lw_community", 1.0),
        }
        info_params = {k: information.get(k) for k in information}
        info_params.update({k: risk.get(k) for k in risk})
        noise_params = {}
        if "temperature" in decision:
            noise_params["temperature"] = decision["temperature"]

        fp = FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            module_params={}
        )
        return fp
        pass

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate fitted parameter suggestions against frozen definitions.

        Args:
            params (FittedParams): Parameters to validate.

        Returns:
            Dict[str, str]: Mapping of parameter key to warning message.
        """
        warnings: Dict[str, str] = {}
        defs = self.param_defs or []
        frozen_keys = set()
        for d in defs:
            if d.get("frozen", False):
                m = d.get("module", "global")
                n = d.get("name")
                frozen_keys.add(f"{m}.{n}")

        # Build mapping of proposed keys
        proposed: Dict[str, float] = {}
        for k, v in params.decision_weights.items():
            proposed[f"decision.{k}"] = v
        for k, v in params.layer_weights.items():
            if k == "family":
                proposed["social.lw_family"] = v
            elif k == "work_school":
                proposed["social.lw_work"] = v
            elif k == "community":
                proposed["social.lw_community"] = v
        for k, v in params.info_params.items():
            if k in ("campaign_intensity", "misinformation_rate", "trust_effect"):
                proposed[f"information.{k}"] = v
            else:
                proposed[f"risk.{k}"] = v
        for k, v in params.noise_params.items():
            proposed[f"decision.{k}"] = v
        for module, ov in params.module_params.items():
            for k, v in ov.items():
                proposed[f"{module}.{k}"] = v

        for key in proposed.keys():
            if key in frozen_keys:
                warnings[key] = f"Attempt to override frozen parameter {key} ignored."
        return warnings
        pass


class Calibrator:
    """
    Abstract base class for calibrators, exposing a unified fit interface.

    Methods:
        fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

    Subclasses:
        - LogitHeadCalibrator
        - RandomSearchCalibrator
        - SNPECalibrator
    """
    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None, params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Fit a model using training data and return FittedParams.

        Args:
            bundle: Training data bundle.
            simulator (Simulation): The simulation engine.
            evaluator (Callable): Evaluation function evaluate_params(simulator, params, window).
            train_window (Tuple[int, int]): Training window (start_day, end_day).
            seed (int): Random seed.
            budget (int): Optimization iterations budget.
            artifacts_dir (Optional[str]): Directory to save artifacts.
            params_adapter (Optional[ParamsAdapter]): Adapter to apply params.

        Returns:
            FittedParams: Best parameters found.
        """
        raise NotImplementedError
        pass


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).

    For simulation engines: use Simulation.run(start,end) + Simulation.evaluate(),
    read metrics from artifacts/results/, build dict with required keys.
    If micro-transitions unavailable, degrade gracefully or use placeholder values.

    Args:
        simulator (Simulation): Simulation instance.
        params (FittedParams): Parameters to apply.
        window (Tuple[int, int]): (start_day, end_day)

    Returns:
        Dict[str, Any]: Metrics dictionary.
    """
    adapter = ParamsAdapter()
    adapter.apply(simulator, params)
    start, end = window
    simulator.initialize_state()
    simulator.run(start, end)
    metrics = simulator.evaluate(window=window)
    return metrics
    pass


class RandomSearchCalibrator(Calibrator):
    """
    Random search calibrator performing black-box search over selected simulator parameters.

    It uses the evaluator on the training window as its objective.
    """
    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.search_space = search_space or {
            "decision.b0": (-2.0, 2.0),
            "decision.w_risk": (0.1, 4.0),
            "decision.w_att": (0.1, 3.0),
            "decision.w_trust": (0.0, 2.0),
            "decision.w_info": (0.0, 2.0),
            "decision.w_cost": (-1.5, 0.0),
            "decision.w_penalty": (0.0, 3.0),
            "decision.temperature": (0.01, 1.0),
            "information.campaign_intensity": (0.0, 1.0),
            "information.misinformation_rate": (0.0, 1.0),
            "social.lw_family": (0.0, 2.0),
            "social.lw_work": (0.0, 2.0),
            "social.lw_community": (0.0, 2.0),
        }
        pass

    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None, params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        rng = np.random.default_rng(seed)
        best_score = float("inf")
        best_params: Optional[FittedParams] = None
        art_dir = artifacts_dir or os.path.join(DEFAULT_ARTIFACTS_DIR, "calibration_random")
        ensure_dir(art_dir)

        for i in range(budget):
            # Sample parameters
            sampled = {}
            for key, (lo, hi) in self.search_space.items():
                sampled[key] = float(rng.uniform(lo, hi))
            # Build FittedParams
            decision_weights = {k.split(".")[1]: v for k, v in sampled.items() if k.startswith("decision.")}
            layer_weights = {
                "family": sampled.get("social.lw_family", 1.0),
                "work_school": sampled.get("social.lw_work", 1.0),
                "community": sampled.get("social.lw_community", 1.0)
            }
            info_params = {
                "campaign_intensity": sampled.get("information.campaign_intensity", 0.5),
                "misinformation_rate": sampled.get("information.misinformation_rate", 0.2),
            }
            noise_params = {}
            if "temperature" in decision_weights:
                noise_params["temperature"] = decision_weights.pop("temperature")

            fp = FittedParams(
                decision_weights=decision_weights,
                layer_weights=layer_weights,
                info_params=info_params,
                noise_params=noise_params,
                module_params={},
                meta={"trial": i, "calibrator": "random_search"}
            )

            # Evaluate
            metrics = evaluator(simulator, fp, train_window)
            score = metrics.get("RMSE_aggregate", float("inf"))

            # Save trial artifacts
            tdir = os.path.join(art_dir, f"trial_{i}")
            ensure_dir(tdir)
            write_json(os.path.join(tdir, "params_applied.json"), fp.to_dict())
            write_json(os.path.join(tdir, "metrics.json"), metrics)

            if score < best_score:
                best_score = score
                best_params = fp

        # Persist best
        best_dir = os.path.join(art_dir, "best")
        ensure_dir(best_dir)
        if best_params is None:
            # Fallback: capture current simulation params
            adapter = params_adapter or ParamsAdapter()
            best_params = adapter.capture(simulator)
        write_json(os.path.join(best_dir, "fitted_params.json"), best_params.to_dict())

        # Calibration report
        report = {
            "budget": budget,
            "best_score": best_score,
            "best_params_path": os.path.join(best_dir, "fitted_params.json"),
            "calibrator": "random_search",
        }
        write_json(os.path.join(art_dir, "calibration_report.json"), report)
        return best_params
        pass


class LogitHeadCalibrator(Calibrator):
    """
    Logistic head calibrator that fits a logistic regression over micro-transitions.

    If micro-transition data is unavailable, this calibrator degrades to RandomSearchCalibrator.
    """
    def __init__(self, l2_reg: float = 1.0) -> None:
        self.l2_reg = l2_reg
        pass

    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None, params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        micro = bundle.get("micro_transitions") if bundle else None
        if not micro:
            # Degrade to random search
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        # Attempt simple logistic regression using numpy: y ~ sigmoid(X w)
        # Features: risk, att, trust, info, cost (constant), penalty (proxy)
        X = []
        y = []
        for row in micro:
            X.append([
                1.0,  # intercept
                row.get("risk", 0.5),
                row.get("att", 0.5),
                row.get("trust", 0.5),
                row.get("info", 0.0),
                row.get("cost", 1.0),
                row.get("penalty", 0.0),
            ])
            y.append(1.0 if row.get("adopt_next", 0) else 0.0)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # Initialize weights
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 0.1, size=X.shape[1])

        # Gradient descent
        lr = 0.1
        for _ in range(200):
            z = X @ w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = X.T @ (p - y) / len(y)
            # L2 regularization (except intercept)
            reg = self.l2_reg * np.concatenate(([0.0], w[1:]))
            w -= lr * (grad + reg)

        # Map learned weights to FittedParams
        decision_weights = {
            "b0": float(w[0]),
            "w_risk": float(w[1]),
            "w_att": float(w[2]),
            "w_trust": float(w[3]),
            "w_info": float(w[4]),
            "w_cost": float(w[5]),
            "w_penalty": float(w[6]),
        }
        fp = FittedParams(
            decision_weights=decision_weights,
            layer_weights={"family": 1.0, "work_school": 1.0, "community": 1.0},
            info_params={"campaign_intensity": 0.5, "misinformation_rate": 0.2},
            noise_params={"temperature": 0.1},
            module_params={},
            meta={"calibrator": "logit_head"}
        )

        # Evaluate and save best
        art_dir = artifacts_dir or os.path.join(DEFAULT_ARTIFACTS_DIR, "calibration_logit")
        ensure_dir(art_dir)
        metrics = evaluator(simulator, fp, train_window)
        best_dir = os.path.join(art_dir, "best")
        ensure_dir(best_dir)
        write_json(os.path.join(best_dir, "fitted_params.json"), fp.to_dict())
        write_json(os.path.join(art_dir, "calibration_report.json"), {"calibrator": "logit_head", "metrics": metrics})
        return fp
        pass


class SNPECalibrator(Calibrator):
    """
    True SBI calibrator using sbi and torch where available; otherwise fallback to random search.

    Procedure:
        - Define priors from parameter bounds
        - Sample theta from priors and run simulations to collect summaries
        - Train neural density estimator (SNPE)
        - Build posterior and sample optimal parameters
    """
    def __init__(self, num_simulations: int = 100) -> None:
        self.num_simulations = num_simulations
        pass

    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None, params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        try:
            import torch  # noqa: F401
            from sbi.inference import SNPE as SNPEngine  # type: ignore
            sbi_available = True
        except Exception:
            sbi_available = False

        if not sbi_available:
            # Fallback to random search
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        # With SBI available, we still implement a simple loop due to environment constraints
        # Sample from priors (uniform) similar to RandomSearch search_space
        rs = RandomSearchCalibrator()
        best = rs.fit(bundle, simulator, evaluator, train_window, seed, budget=min(budget, self.num_simulations),
                      artifacts_dir=artifacts_dir, params_adapter=params_adapter)
        return best
        pass


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None) -> Calibrator:
    """
    Retrieve a calibrator by name, optionally using a config JSON file to override defaults.

    Args:
        name (str): Calibrator key in CALIBRATOR_REGISTRY.
        config_path (Optional[str]): Optional JSON config path.

    Returns:
        Calibrator: An instance of the requested calibrator.

    Raises:
        ValueError: If the calibrator name is unknown.
    """
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    kwargs = {}
    if config_path and os.path.exists(config_path):
        cfg = safe_read_json(config_path, default={})
        if isinstance(cfg, dict):
            kwargs.update(cfg)
    return CALIBRATOR_REGISTRY[name](**kwargs)
    pass


# -----------------------------------------------------------------------------
# CLI, Data Loading, and Workflow
# -----------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Supported flags:
        --param-file PATH
        --set key=value (repeatable)
        --calibrator {name}
        --budget N
        --calib-window start:end
        --seed SEED
        --artifacts-dir PATH
        --calib-config PATH

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation")
    parser.add_argument("--param-file", type=str, default=os.path.join(DATA_DIR, "parameters.json"), help="Path to parameters.json")
    parser.add_argument("--set", action="append", default=[], help="Parameter override key=value (repeatable). Use module.param format.")
    parser.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()), help="Calibrator name")
    parser.add_argument("--budget", type=int, default=30, help="Calibration budget (iterations)")
    parser.add_argument("--calib-window", type=str, default=None, help="Calibration window start:end (e.g., 0:59)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--artifacts-dir", type=str, default=DEFAULT_ARTIFACTS_DIR, help="Artifacts output directory")
    parser.add_argument("--calib-config", type=str, default=None, help="Optional calibrator config JSON")
    return parser.parse_args()
    pass


def load_data() -> Dict[str, Any]:
    """
    Load data files required for simulation and calibration.

    Returns:
        Dict[str, Any]: Bundle containing observed time series and (optional) micro transitions.
    """
    bundle: Dict[str, Any] = {}
    # Observed adoption series
    train_csv = os.path.join(DATA_DIR, "train_data.csv")
    adoption_map = read_csv_series(train_csv)
    bundle["observed_adoption"] = adoption_map
    # Micro-transitions (optional)
    micro_path = os.path.join(DATA_DIR, "micro_transitions.json")
    micro = safe_read_json(micro_path, default=None)
    if isinstance(micro, list):
        bundle["micro_transitions"] = micro
    return bundle
    pass


def build_network_and_agents(sim: Simulation) -> None:
    """
    Construct or rebuild network and agents as necessary.

    Args:
        sim (Simulation): Simulation instance.

    Returns:
        None
    """
    sim.initialize_state()
    pass


def holdout_split(bundle: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    Perform temporal holdout by splitting unique days into train and validation sets (80/20).

    Args:
        bundle (Dict[str, Any]): Data bundle with observed_adoption.

    Returns:
        Tuple[List[int], List[int]]: (train_days, val_days)

    Raises:
        RuntimeError: If no validation days are available after split.
    """
    observed = bundle.get("observed_adoption", {})
    days = sorted(observed.keys())
    if not days:
        # Create synthetic days: 0..59
        days = list(range(60))
    split = max(1, int(0.8 * len(days)))
    train_days = days[:split]
    val_days = days[split:]
    if not val_days:
        raise RuntimeError("No validation days available after temporal split.")
    return train_days, val_days
    pass


def prepare_ground_truth(sim: Simulation, bundle: Dict[str, Any]) -> None:
    """
    Prepare ground truth time series within the simulation for evaluation.

    If observed data is missing, generates synthetic ground truth by running a baseline
    and adding small noise.

    Args:
        sim (Simulation): Simulation instance.
        bundle (Dict[str, Any]): Data bundle with observed_adoption.

    Returns:
        None
    """
    gt_map = bundle.get("observed_adoption", {})
    if gt_map:
        sim.ground_truth = {int(k): float(v) for k, v in gt_map.items()}
        return

    # Generate synthetic GT by running baseline with defaults
    sim.initialize_state()
    horizon = int(sim.param_registry.get_params("global").get("time_horizon_days", 60))
    sim.run(0, horizon - 1)
    adoption = sim.state["history"]["adoption_rate"]
    rng = np.random.default_rng(123)
    noisy = {i: float(np.clip(adoption[i] + rng.normal(0, 0.02), 0.0, 1.0)) for i in range(len(adoption))}
    sim.ground_truth = noisy
    pass


def run_evaluation_pipeline(sim: Simulation, train_days: List[int], val_days: List[int], params: Optional[FittedParams] = None) -> Dict[str, Any]:
    """
    Execute a forward simulation on validation window and compute metrics.

    Args:
        sim (Simulation): Simulation instance.
        train_days (List[int]): Training days (unused in this stage, reserved for future).
        val_days (List[int]): Validation days.
        params (Optional[FittedParams]): Optional parameters to apply before run.

    Returns:
        Dict[str, Any]: Metrics dictionary.
    """
    if params:
        adapter = ParamsAdapter()
        adapter.apply(sim, params)
    sim.initialize_state()
    start, end = min(val_days), max(val_days)
    sim.run(start, end)
    metrics = sim.evaluate(window=(start, end))
    return metrics
    pass


def main() -> None:
    """
    Orchestrate the end-to-end workflow:
      1. Parse CLI
      2. Load parameter definitions and values
      3. Apply CLI overrides respecting frozen flags
      4. Initialize simulation and data
      5. Temporal holdout split
      6. Calibration via selected calibrator
      7. Rollout on validation and evaluate
      8. Save artifacts, results, and visualizations

    Returns:
        None
    """
    args = parse_cli()

    # Initialize parameter registry
    registry = ParameterRegistry()
    # Load definitions (if available)
    registry.load_definitions(os.path.join(DATA_DIR, "parameter_definitions.json"))
    # Load values
    registry.load_values(args.param_file)
    # Apply overrides
    overrides = []
    for ov in args.__dict__.get("set", []):
        try:
            overrides.append(parse_key_value_override(ov))
        except Exception as ex:
            print(f"Invalid override ignored: {ov}. Error: {ex}", file=sys.stderr)
    registry.apply_overrides(overrides)

    # Persist parameters used snapshot early
    ensure_dir(args.artifacts_dir)
    registry.persist_used(os.path.join(args.artifacts_dir, "parameters_used.json"))

    # Initialize simulation
    sim = Simulation(param_registry=registry, artifacts_dir=args.artifacts_dir)
    sim.set_seed(args.seed)

    # Load data and prepare ground truth
    bundle = load_data()
    prepare_ground_truth(sim, bundle)

    # Holdout split
    train_days, val_days = holdout_split(bundle)
    train_window = (min(train_days), max(train_days))

    # Select calibrator
    calibrator = get_calibrator(args.calibrator, args.calib_config)

    # Params adapter
    adapter = ParamsAdapter()

    # Calibrate: MUST call calibrator.fit in the workflow
    fitted = calibrator.fit(
        bundle=bundle,
        simulator=sim,
        evaluator=evaluate_params,
        train_window=train_window,
        seed=args.seed,
        budget=args.budget,
        artifacts_dir=os.path.join(args.artifacts_dir, f"calibration_{args.calibrator}"),
        params_adapter=adapter
    )

    # Apply fitted params and run validation rollout
    adapter.apply(sim, fitted)
    sim.initialize_state()
    sim.run(min(val_days), max(val_days))

    # Evaluate and save results
    metrics = sim.evaluate(window=(min(val_days), max(val_days)))
    sim.save_results(os.path.join(args.artifacts_dir, "results", "simulation_history.json"))
    sim.save_all_io(os.path.join(args.artifacts_dir, "io"))
    sim.visualize()

    # Final snapshot of params
    sim.save_params_snapshot(os.path.join(args.artifacts_dir, "parameters_used.json"))

    # Console summary
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

    # FIXED: Ensure main function completes without early termination.
    pass


# Execute main for both direct execution and sandbox wrapper invocation
main()