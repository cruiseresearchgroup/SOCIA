import os
import sys
import json
import math
import random
import argparse
import ast
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Tuple, List, Optional, Callable, Mapping

import numpy as np
import pandas as pd

try:
    from scipy.stats import wasserstein_distance
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# Global path setup using environment variables
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def set_global_seed(seed: int) -> None:
    """
    Set global seeds for numpy and random for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists; treats empty path as current directory.
    """
    if not path:
        path = "."
    os.makedirs(path, exist_ok=True)


def save_json(obj: Mapping[str, Any], path: str) -> None:
    """
    Save a mapping as JSON to given path.
    """
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load a JSON object from path into a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_kv_override(s: str) -> Tuple[str, str]:
    """
    Parse a key=value string override into key and value strings.
    """
    if "=" not in s:
        raise ValueError(f"Invalid override '{s}', expected key=value")
    key, value = s.split("=", 1)
    return key.strip(), value.strip()


@dataclass
class SimulationConfig:
    seed: int = 42
    gov_intervention_day: int = 10
    output_root: str = os.path.join(PROJECT_ROOT, "artifacts")
    k_runs: int = 10
    forecast_days: int = 10
    verbose: bool = True
    debug_trace: bool = False  # if True, record per-step diagnostics
    init_from_day0: bool = True  # if True, init prev state from observed day0 when start_idx==0


@dataclass
class ParameterDefinition:
    name: str
    default: Any
    bounds: Tuple[float, float]
    frozen: bool = False
    dtype: str = "float"
    module: str = "global"
    description: str = ""


class ParameterRegistry:
    """
    Registry for parameter definitions and current values. Can be initialized from file;
    otherwise uses default definitions. Supports dynamic extension for demographic effects.
    """
    def __init__(self, defs_path: str):
        self.defs_path = defs_path
        self.definitions: Dict[str, ParameterDefinition] = {}
        self.values: Dict[str, Any] = {}
        self._init_or_load_definitions()

    def _default_definitions(self) -> Dict[str, ParameterDefinition]:
        defs: Dict[str, ParameterDefinition] = {}
        add = lambda d: defs.__setitem__(d.name, d)

        # Decision module parameters
        add(ParameterDefinition("Decision.alpha", 0.0, (-5.0, 5.0), False, "float", "Decision", "Intercept utility"))
        add(ParameterDefinition("Decision.gamma", 1.0, (0.0, 5.0), False, "float", "Decision", "Persistence"))
        add(ParameterDefinition("Decision.theta_f", 1.0, (-3.0, 3.0), False, "float", "Decision", "Family influence"))
        add(ParameterDefinition("Decision.theta_w", 1.0, (-3.0, 3.0), False, "float", "Decision", "Work influence"))
        add(ParameterDefinition("Decision.theta_c", 1.0, (-3.0, 3.0), False, "float", "Decision", "Community influence"))
        add(ParameterDefinition("Decision.beta_r", 0.0, (-3.0, 3.0), False, "float", "Decision", "Risk sensitivity"))
        add(ParameterDefinition("Decision.beta_i", 0.5, (-3.0, 3.0), False, "float", "Decision", "Information memory sensitivity"))
        add(ParameterDefinition("Decision.tau", 1.0, (0.1, 5.0), False, "float", "Decision", "Decision noise temperature"))

        # Demographic effects (will be dynamically extended to match dataset)
        for i in range(1):  # placeholder; will be extended after data load
            add(ParameterDefinition(f"Decision.age_effects.{i}", 0.0, (-2.0, 2.0), False, "float", "Decision", "Age group effect"))
        for i in range(1):  # placeholder; will be extended after data load
            add(ParameterDefinition(f"Decision.occ_effects.{i}", 0.0, (-2.0, 2.0), False, "float", "Decision", "Occupation effect"))

        # Info module parameters
        add(ParameterDefinition("Info.phi_family", 0.1, (0.0, 1.0), False, "float", "Info", "Info influence family"))
        add(ParameterDefinition("Info.phi_work", 0.1, (0.0, 1.0), False, "float", "Info", "Info influence work"))
        add(ParameterDefinition("Info.phi_community", 0.1, (0.0, 1.0), False, "float", "Info", "Info influence community"))
        add(ParameterDefinition("Info.lambda_broadcast_base", 0.05, (0.0, 1.0), False, "float", "Info", "Broadcast base"))
        add(ParameterDefinition("Info.lambda_broadcast_factor_after", 1.5, (1.0, 5.0), False, "float", "Info", "Broadcast factor after intervention"))
        add(ParameterDefinition("Info.rho_info_decay", 0.5, (0.0, 1.0), False, "float", "Info", "Info memory decay"))

        # Layer weights module (normalize internally in model)
        add(ParameterDefinition("Layers.family_weight", 1.0, (0.0, 5.0), False, "float", "Layers", "Layer weight family"))
        add(ParameterDefinition("Layers.work_weight", 1.0, (0.0, 5.0), False, "float", "Layers", "Layer weight work"))
        add(ParameterDefinition("Layers.community_weight", 1.0, (0.0, 5.0), False, "float", "Layers", "Layer weight community"))

        # Activity module parameters
        add(ParameterDefinition("Activity.base", 0.6, (0.0, 1.0), False, "float", "Activity", "Base activity level"))
        add(ParameterDefinition("Activity.amplitude", 0.2, (0.0, 1.0), False, "float", "Activity", "Amplitude of weekly cycle"))
        add(ParameterDefinition("Activity.phase", 0.0, (-math.pi, math.pi), False, "float", "Activity", "Phase of weekly cycle"))

        # Global parameters
        add(ParameterDefinition("Global.gov_intervention_day", 10, (0, 1000), True, "int", "global", "Intervention day (frozen)"))
        return defs

    def _init_or_load_definitions(self) -> None:
        if os.path.exists(self.defs_path):
            try:
                data = load_json(self.defs_path)
                for name, meta in data.items():
                    self.definitions[name] = ParameterDefinition(
                        name=name,
                        default=meta.get("default"),
                        bounds=tuple(meta.get("bounds", [float("-inf"), float("inf")])),
                        frozen=bool(meta.get("frozen", False)),
                        dtype=meta.get("dtype", "float"),
                        module=meta.get("module", "global"),
                        description=meta.get("description", ""),
                    )
                if not self.definitions:
                    raise ValueError("Empty parameter_definitions.json")
            except Exception:
                self.definitions = self._default_definitions()
                self._save_definitions()
        else:
            self.definitions = self._default_definitions()
            self._save_definitions()
        self.values = {name: d.default for name, d in self.definitions.items()}

    def _save_definitions(self) -> None:
        payload: Dict[str, Any] = {}
        for name, d in self.definitions.items():
            payload[name] = {
                "default": d.default,
                "bounds": list(d.bounds),
                "frozen": d.frozen,
                "dtype": d.dtype,
                "module": d.module,
                "description": d.description,
            }
        ensure_dir(os.path.dirname(self.defs_path))
        save_json(payload, self.defs_path)

    def set_values(self, new_vals: Dict[str, Any], ignore_frozen: bool = True) -> Dict[str, str]:
        """
        Set parameter values; returns warnings for issues (unknown names, frozen, bounds).
        """
        warnings: Dict[str, str] = {}
        for k, v in new_vals.items():
            if k not in self.definitions:
                warnings[k] = f"Unknown parameter '{k}' ignored."
                continue
            d = self.definitions[k]
            if d.frozen and ignore_frozen:
                warnings[k] = f"Parameter '{k}' is frozen; override ignored."
                continue
            try:
                if d.dtype == "float":
                    val = float(v)
                elif d.dtype == "int":
                    val = int(v)
                elif d.dtype == "bool":
                    if isinstance(v, bool):
                        val = v
                    elif isinstance(v, (int, float, np.integer, np.floating)):
                        val = (int(v) != 0)
                    else:
                        val = (str(v).lower() in ("true", "1", "yes", "y"))
                else:
                    val = v
            except Exception:
                warnings[k] = f"Parameter '{k}' could not be cast to {d.dtype}; override ignored."
                continue
            low, high = d.bounds
            if isinstance(val, (int, float)):
                if val < low:
                    warnings[k] = f"Parameter '{k}' clipped to lower bound {low}."
                    val = low
                if val > high:
                    warnings[k] = f"Parameter '{k}' clipped to upper bound {high}."
                    val = high
            self.values[k] = val
        return warnings

    def to_json(self) -> Dict[str, Any]:
        return dict(self.values)

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def ensure_demographic_params(self, age_dim: int, occ_dim: int) -> None:
        """
        Ensure parameter definitions include demographic effects matching the given dimensions.
        Will add missing keys and remove extra keys beyond the required dimension.
        """
        changed = False
        # Age
        existing_age = [k for k in self.definitions.keys() if k.startswith("Decision.age_effects.")]
        existing_age_idx = set()
        for k in existing_age:
            try:
                idx = int(k.split(".")[-1])
                existing_age_idx.add(idx)
            except Exception:
                continue
        # Add missing
        for j in range(max(0, age_dim)):
            key = f"Decision.age_effects.{j}"
            if key not in self.definitions:
                self.definitions[key] = ParameterDefinition(key, 0.0, (-2.0, 2.0), False, "float", "Decision", "Age group effect")
                self.values[key] = 0.0
                changed = True
        # Remove extras
        for idx in existing_age_idx:
            if idx >= age_dim:
                key = f"Decision.age_effects.{idx}"
                if key in self.definitions:
                    del self.definitions[key]
                    changed = True
                if key in self.values:
                    del self.values[key]

        # Occupation
        existing_occ = [k for k in self.definitions.keys() if k.startswith("Decision.occ_effects.")]
        existing_occ_idx = set()
        for k in existing_occ:
            try:
                idx = int(k.split(".")[-1])
                existing_occ_idx.add(idx)
            except Exception:
                continue
        for j in range(max(0, occ_dim)):
            key = f"Decision.occ_effects.{j}"
            if key not in self.definitions:
                self.definitions[key] = ParameterDefinition(key, 0.0, (-2.0, 2.0), False, "float", "Decision", "Occupation effect")
                self.values[key] = 0.0
                changed = True
        for idx in existing_occ_idx:
            if idx >= occ_dim:
                key = f"Decision.occ_effects.{idx}"
                if key in self.definitions:
                    del self.definitions[key]
                    changed = True
                if key in self.values:
                    del self.values[key]

        if changed:
            self._save_definitions()


@dataclass
class FittedParams:
    """
    Data structure for capturing fitted parameter sets.
    """
    decision_weights: Dict[str, Any]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ParamsAdapter:
    """
    Adapter for mapping FittedParams into the Simulation's ParameterRegistry.
    """
    def __init__(self, registry: ParameterRegistry, used_path: str):
        self.registry = registry
        self.used_path = used_path

    def apply(self, simulation: "Simulation", params: FittedParams) -> Dict[str, str]:
        mapping: Dict[str, Any] = {}

        # Decision weights
        for k, v in params.decision_weights.items():
            if k in ("age_effects", "occ_effects"):
                if isinstance(v, dict):
                    for key in sorted(v.keys(), key=lambda x: int(x)):
                        j = int(key)
                        mapping[f"Decision.{k}.{j}"] = float(v[key])
            else:
                mapping[f"Decision.{k}"] = float(v)

        # Layer weights
        for k, v in params.layer_weights.items():
            if k == "family":
                mapping["Layers.family_weight"] = float(v)
            elif k in ("work", "work_school", "workschool"):
                mapping["Layers.work_weight"] = float(v)
            elif k == "community":
                mapping["Layers.community_weight"] = float(v)

        # Info
        for k, v in params.info_params.items():
            if k in ("phi_family", "phi_work", "phi_community"):
                mapping[f"Info.{k}"] = float(v)
            elif k == "lambda_broadcast_base":
                mapping["Info.lambda_broadcast_base"] = float(v)
            elif k in ("lambda_broadcast_factor_after", "lambda_broadcast_factor_after_day10"):
                mapping["Info.lambda_broadcast_factor_after"] = float(v)
            elif k in ("rho_info_decay",):
                mapping["Info.rho_info_decay"] = float(v)

        # Noise
        for k, v in params.noise_params.items():
            if k == "tau":
                mapping["Decision.tau"] = float(v)

        # Module-specific params
        for mod, sub in params.module_params.items():
            for k, v in sub.items():
                mapping[f"{mod}.{k}"] = float(v)

        warnings = self.validate_frozen(params)
        apply_warnings = simulation.set_params(mapping)
        # Merge warnings
        merged = {}
        merged.update(warnings or {})
        merged.update(apply_warnings or {})
        save_json(simulation.param_registry.to_json(), self.used_path)
        return merged

    def capture(self, simulation: "Simulation") -> FittedParams:
        vals = simulation.param_registry.to_json()
        decision: Dict[str, Any] = {
            "alpha": vals.get("Decision.alpha", 0.0),
            "gamma": vals.get("Decision.gamma", 1.0),
            "theta_f": vals.get("Decision.theta_f", 1.0),
            "theta_w": vals.get("Decision.theta_w", 1.0),
            "theta_c": vals.get("Decision.theta_c", 1.0),
            "beta_r": vals.get("Decision.beta_r", 0.0),
            "beta_i": vals.get("Decision.beta_i", 0.5),
            "tau": vals.get("Decision.tau", 1.0),
        }
        # Recover available demographic effects dynamically
        age_keys = [k for k in vals.keys() if k.startswith("Decision.age_effects.")]
        occ_keys = [k for k in vals.keys() if k.startswith("Decision.occ_effects.")]
        age_effs = {k.split(".")[-1]: vals[k] for k in age_keys}
        occ_effs = {k.split(".")[-1]: vals[k] for k in occ_keys}
        decision["age_effects"] = age_effs
        decision["occ_effects"] = occ_effs

        layer = {
            "family": vals.get("Layers.family_weight", 1.0),
            "work_school": vals.get("Layers.work_weight", 1.0),
            "community": vals.get("Layers.community_weight", 1.0),
        }

        info = {
            "phi_family": vals.get("Info.phi_family", 0.1),
            "phi_work": vals.get("Info.phi_work", 0.1),
            "phi_community": vals.get("Info.phi_community", 0.1),
            "lambda_broadcast_base": vals.get("Info.lambda_broadcast_base", 0.05),
            "lambda_broadcast_factor_after": vals.get("Info.lambda_broadcast_factor_after", 1.5),
            "rho_info_decay": vals.get("Info.rho_info_decay", 0.5),
        }

        noise = {"tau": vals.get("Decision.tau", 1.0)}

        fitted = FittedParams(
            decision_weights=decision,
            layer_weights=layer,
            info_params=info,
            noise_params=noise,
            module_params={
                "Activity": {
                    "base": vals.get("Activity.base", 0.6),
                    "amplitude": vals.get("Activity.amplitude", 0.2),
                    "phase": vals.get("Activity.phase", 0.0),
                }
            },
            engine_type="calibrasim",
            meta={}
        )
        return fitted

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate against frozen parameters in the registry for any overrides present in params.
        """
        warnings: Dict[str, str] = {}
        # Collect mapping of params to fully-qualified keys in registry
        applied: Dict[str, Any] = {}

        for k, v in params.decision_weights.items():
            if k in ("age_effects", "occ_effects") and isinstance(v, dict):
                for key, val in v.items():
                    applied[f"Decision.{k}.{key}"] = val
            else:
                applied[f"Decision.{k}"] = v
        for k, v in params.layer_weights.items():
            if k == "family":
                applied["Layers.family_weight"] = v
            elif k in ("work", "work_school", "workschool"):
                applied["Layers.work_weight"] = v
            elif k == "community":
                applied["Layers.community_weight"] = v
        for k, v in params.info_params.items():
            if k in ("phi_family", "phi_work", "phi_community"):
                applied[f"Info.{k}"] = v
            elif k == "lambda_broadcast_base":
                applied["Info.lambda_broadcast_base"] = v
            elif k in ("lambda_broadcast_factor_after", "lambda_broadcast_factor_after_day10"):
                applied["Info.lambda_broadcast_factor_after"] = v
            elif k == "rho_info_decay":
                applied["Info.rho_info_decay"] = v

        for name, d in self.registry.definitions.items():
            if d.frozen and name in applied:
                warnings[name] = "Attempt to override frozen parameter; ignored."
        return warnings


class Calibrator:
    """
    Base class for calibrators.
    """
    def fit(
        self,
        bundle: Any,
        simulator: "Simulation",
        evaluator: Callable[["Simulation", FittedParams, Tuple[int, int]], Dict[str, Any]],
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        raise NotImplementedError("Calibrator.fit must be implemented by subclasses.")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically safe sigmoid.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def build_multiplex_adjacency(
    social: Dict[str, Dict[str, List[Any]]],
    id2idx: Dict[int, int],
    n: int,
) -> Dict[str, List[Any]]:
    """
    Build multiplex adjacency with optional per-edge weights.

    social format per layer can be:
      - [neighbor_id, ...]
      - [{'id': neighbor_id, 'w': weight}, ...] or keys 'agent_id'/'weight'
    Returns dict layer -> list of (indices_array, weights_array) for each agent.
    """
    layers = ["family", "work_school", "community"]
    adj_maps: Dict[str, List[Dict[int, float]]] = {layer: [dict() for _ in range(n)] for layer in layers}
    for k_str, v in social.items():
        try:
            i = id2idx.get(int(k_str))
        except Exception:
            i = None
        if i is None:
            continue
        for layer in layers:
            entries = v.get(layer, [])
            for nbr in entries:
                if isinstance(nbr, dict):
                    nid = nbr.get("id", nbr.get("agent_id", nbr.get("nbr")))
                    w = float(nbr.get("w", nbr.get("weight", 1.0)))
                else:
                    nid = nbr
                    w = 1.0
                try:
                    j = id2idx.get(int(nid))
                except Exception:
                    j = None
                if j is None or j == i:
                    continue
                prev_w = adj_maps[layer][i].get(j, 0.0)
                adj_maps[layer][i][j] = max(prev_w, w)
                prev_w2 = adj_maps[layer][j].get(i, 0.0)
                adj_maps[layer][j][i] = max(prev_w2, w)

    out: Dict[str, List[Any]] = {}
    for layer in layers:
        arr_list: List[Any] = []
        for i in range(n):
            if not adj_maps[layer][i]:
                arr_list.append((np.array([], dtype=int), np.array([], dtype=np.float64)))
            else:
                items = sorted(adj_maps[layer][i].items(), key=lambda kv: kv[0])
                idxs = np.array([k for k, _ in items], dtype=int)
                ws = np.array([w for _, w in items], dtype=np.float64)
                arr_list.append((idxs, ws))
        out[layer] = arr_list
    return out


def compute_layer_share(states_prev: np.ndarray, neighbors: List[Any]) -> np.ndarray:
    """
    Compute (weighted) neighbor adoption shares for each agent in a layer.

    neighbors[i] can be:
      - np.ndarray of neighbor indices (equal weights)
      - (indices_array, weights_array)
      - {'idx': indices_array, 'w': weights_array}
    """
    N = states_prev.shape[0]
    shares = np.zeros(N, dtype=float)
    for i in range(N):
        nbrs = neighbors[i]
        if isinstance(nbrs, np.ndarray):
            idxs = nbrs
            ws = None
        elif isinstance(nbrs, tuple) and len(nbrs) == 2:
            idxs, ws = nbrs
        elif isinstance(nbrs, dict):
            idxs = nbrs.get("idx", np.array([], dtype=int))
            ws = nbrs.get("w", None)
        else:
            try:
                idxs = np.array(list(nbrs), dtype=int)
                ws = None
            except Exception:
                idxs = np.array([], dtype=int)
                ws = None
        if idxs.size == 0:
            shares[i] = 0.0
            continue
        vals = states_prev[idxs]
        if ws is None or len(ws) == 0:
            shares[i] = float(np.mean(vals))
        else:
            wsum = float(np.sum(ws))
            shares[i] = float(np.sum(vals * ws) / wsum) if wsum > 0 else 0.0
    return shares


def wasserstein_safe(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """
    Compute 1-Wasserstein distance with SciPy if available, else fallback to simple L1 on sorted arrays.
    """
    if _HAVE_SCIPY:
        return float(wasserstein_distance(u_values, v_values))
    else:
        try:
            u = np.sort(u_values)
            v = np.sort(v_values)
            n = min(len(u), len(v))
            if n == 0:
                return 0.0
            return float(np.mean(np.abs(u[:n] - v[:n])))
        except Exception:
            return float(np.mean(np.abs(u_values - v_values)))


def composite_score(metrics: Dict[str, Any], weights: Dict[str, float], target_metric: Optional[str] = None) -> float:
    """
    Compute a composite or targeted score from metrics. Lower is better.
    """
    if target_metric:
        return float(metrics.get(target_metric, float("inf")))
    score = 0.0
    for k, w in weights.items():
        score += w * float(metrics.get(k, 0.0))
    return score


class Module:
    """
    Base class for simulator modules.
    """
    def __init__(self, name: str):
        self.name = name
        self.supports_calibration: bool = True

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        raise NotImplementedError("Module.forward must be implemented by subclasses.")


class ActivityModule(Module):
    """
    Weekly activity dynamics producing activity_level in [0,1] each day.
    """
    def __init__(self, name: str = "Activity"):
        super().__init__(name)

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        base = float(params.get("base", 0.6))
        amp = float(params.get("amplitude", 0.2))
        phase = float(params.get("phase", 0.0))
        # t is the absolute day value; weekly cycle repeats every 7 days
        weekly_pos = (t % 7) / 7.0
        act = base + amp * math.sin(2.0 * math.pi * weekly_pos + phase)
        act = float(max(0.0, min(1.0, act)))
        buffers["activity_level"] = act


class InfoPropagationModule(Module):
    """
    Propagate information via peers and broadcast, maintaining an exponential memory state.
    """
    def __init__(self, neighbors: Dict[str, List[Any]], name: str = "Info"):
        super().__init__(name)
        self.neighbors = neighbors

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        prev_states: np.ndarray = state["prev_state"]
        mem_prev: np.ndarray = state["mem_prev"]
        N = prev_states.shape[0]

        # Compute neighbor shares (raw)
        share_f_raw = compute_layer_share(prev_states, self.neighbors["family"])
        share_w_raw = compute_layer_share(prev_states, self.neighbors["work_school"])
        share_c_raw = compute_layer_share(prev_states, self.neighbors["community"])

        # Apply activity and normalized layer weights
        activity_level = float(buffers.get("activity_level", 1.0))
        lw = state.get("global", {}).get("layer_weights", {})
        wf = float(lw.get("family_weight", 1.0))
        ww = float(lw.get("work_weight", 1.0))
        wc = float(lw.get("community_weight", 1.0))
        s = wf + ww + wc
        if s > 0:
            wf, ww, wc = wf / s, ww / s, wc / s

        share_f = share_f_raw * activity_level * wf
        share_w = share_w_raw * activity_level * ww
        share_c = share_c_raw * activity_level * wc

        # Store for reuse by Decision
        buffers["share_f_raw"] = share_f_raw
        buffers["share_w_raw"] = share_w_raw
        buffers["share_c_raw"] = share_c_raw
        buffers["share_f"] = share_f
        buffers["share_w"] = share_w
        buffers["share_c"] = share_c

        phi_f = float(params.get("phi_family", 0.1))
        phi_w = float(params.get("phi_work", 0.1))
        phi_c = float(params.get("phi_community", 0.1))
        lam_base = float(params.get("lambda_broadcast_base", 0.05))
        lam_factor_after = float(params.get("lambda_broadcast_factor_after", 1.5))
        rho = float(params.get("rho_info_decay", 0.5))
        intervention_day = int(state.get("global", {}).get("gov_intervention_day", 10))

        lam_t = lam_base if t < intervention_day else lam_base * lam_factor_after
        u = phi_f * share_f + phi_w * share_w + phi_c * share_c + lam_t
        p_info = 1.0 - np.exp(-np.clip(u, 0.0, 50.0))
        rec = (np.random.rand(N) < p_info).astype(np.float64)
        mem_next = rho * mem_prev + (1.0 - rho) * rec

        buffers["received_info"] = rec
        buffers["mem_next"] = mem_next


class DecisionModule(Module):
    """
    Stochastic binary adoption decision based on utility logits and temperature tau.
    """
    def __init__(self, neighbors: Dict[str, List[Any]], demographics: Dict[str, np.ndarray], name: str = "Decision"):
        super().__init__(name)
        self.neighbors = neighbors
        self.demographics = demographics

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        prev_states: np.ndarray = state["prev_state"]
        mem_prev: np.ndarray = state["mem_prev"]
        risk: np.ndarray = state["risk"]
        N = prev_states.shape[0]

        # Use precomputed shares if available, else compute and scale with normalized layer weights
        if "share_f" in buffers and "share_w" in buffers and "share_c" in buffers:
            share_f = buffers["share_f"]
            share_w = buffers["share_w"]
            share_c = buffers["share_c"]
        else:
            share_f = compute_layer_share(prev_states, self.neighbors["family"])
            share_w = compute_layer_share(prev_states, self.neighbors["work_school"])
            share_c = compute_layer_share(prev_states, self.neighbors["community"])
            activity_level = float(buffers.get("activity_level", 1.0))
            lw = state.get("global", {}).get("layer_weights", {})
            wf = float(lw.get("family_weight", 1.0))
            ww = float(lw.get("work_weight", 1.0))
            wc = float(lw.get("community_weight", 1.0))
            s = wf + ww + wc
            if s > 0:
                wf, ww, wc = wf / s, ww / s, wc / s
            share_f *= activity_level * wf
            share_w *= activity_level * ww
            share_c *= activity_level * wc

        alpha = float(params.get("alpha", 0.0))
        gamma = float(params.get("gamma", 1.0))
        theta_f = float(params.get("theta_f", 1.0))
        theta_w = float(params.get("theta_w", 1.0))
        theta_c = float(params.get("theta_c", 1.0))
        beta_r = float(params.get("beta_r", 0.0))
        beta_i = float(params.get("beta_i", 0.5))
        tau = float(params.get("tau", 1.0))

        logits = (
            alpha + gamma * prev_states + theta_f * share_f + theta_w * share_w + theta_c * share_c
            + beta_r * risk + beta_i * mem_prev
        )

        age_oh = self.demographics.get("age_oh")
        occ_oh = self.demographics.get("occ_oh")
        if age_oh is not None and age_oh.shape[1] > 0:
            K_age = age_oh.shape[1]
            for j in range(K_age):
                wj = float(params.get(f"age_effects.{j}", 0.0))
                logits += wj * age_oh[:, j]
        if occ_oh is not None and occ_oh.shape[1] > 0:
            K_occ = occ_oh.shape[1]
            for j in range(K_occ):
                wj = float(params.get(f"occ_effects.{j}", 0.0))
                logits += wj * occ_oh[:, j]

        logits = logits / max(1e-6, tau)
        p = sigmoid(logits)
        next_states = (np.random.rand(N) < p).astype(np.float64)

        buffers["p_adopt"] = p
        buffers["state_next"] = next_states


class Simulation:
    """
    Complete simulation engine for mask adoption behavior with information diffusion.

    Usage:
      - load_data(): reads all required files, constructs network and demographics.
      - build_modules(): initializes internal modules.
      - set_params(): applies registry parameter mapping.
      - run(): executes for a given time window and dataset.
      - evaluate(): computes goodness-of-fit metrics.
    """
    def __init__(self, data_dir: str, cfg: SimulationConfig, artifacts_dir: str):
        self.cfg = cfg
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        ensure_dir(self.artifacts_dir)
        self.param_defs_path = os.path.join(self.artifacts_dir, "parameter_definitions.json")
        self.params_used_path = os.path.join(self.artifacts_dir, "parameters_used.json")
        self.param_registry = ParameterRegistry(self.param_defs_path)

        self.agents_df: Optional[pd.DataFrame] = None
        self.social_raw: Optional[Dict[str, Any]] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.common_ids: Optional[np.ndarray] = None
        self.id2idx: Optional[Dict[int, int]] = None

        self.neighbors: Optional[Dict[str, List[Any]]] = None

        self.obs_wearing_train: Optional[np.ndarray] = None
        self.obs_received_train: Optional[np.ndarray] = None
        self.obs_days_train: Optional[List[int]] = None

        self.obs_wearing_val: Optional[np.ndarray] = None
        self.obs_received_val: Optional[np.ndarray] = None
        self.obs_days_val: Optional[List[int]] = None

        self.obs_wearing_test: Optional[np.ndarray] = None
        self.obs_received_test: Optional[np.ndarray] = None
        self.obs_days_test: Optional[List[int]] = None

        self.age_oh: Optional[np.ndarray] = None
        self.occ_oh: Optional[np.ndarray] = None
        self.risk: Optional[np.ndarray] = None

        self.modules: List[Module] = []
        self._module_params_map: Dict[str, Dict[str, Any]] = {}

        self.buffers: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}

        set_global_seed(self.cfg.seed)
        self.metadata: Dict[str, Any] = {}

    def load_data(self) -> None:
        """
        Load datasets, network, demographics, and derive aligned arrays. Dynamically
        update parameter registry to match demographic one-hot dimensions.
        """
        if self.cfg.verbose:
            print("Loading data...")

        agents_path = os.path.join(self.data_dir, "agent_attributes.csv")
        social_path = os.path.join(self.data_dir, "social_network.json")
        train_path = os.path.join(self.data_dir, "train_data.csv")
        val_path = os.path.join(self.data_dir, "val_data.csv")
        test_path = os.path.join(self.data_dir, "test_data.csv")
        meta_path = os.path.join(self.data_dir, "metadata.json")

        try:
            self.agents_df = pd.read_csv(agents_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load agent_attributes.csv: {e}")

        # Validate agent attributes columns
        req_agents = {"agent_id", "age_group", "occupation", "risk_perception"}
        missing_agents = req_agents - set(self.agents_df.columns.astype(str).tolist())
        if missing_agents:
            raise RuntimeError(f"agent_attributes.csv missing required columns: {sorted(list(missing_agents))}")

        try:
            with open(social_path, "r", encoding="utf-8") as f:
                self.social_raw = json.load(f)
            if not isinstance(self.social_raw, dict):
                raise RuntimeError("social_network.json must be an object mapping agent_id to layer lists.")
        except Exception as e:
            raise RuntimeError(f"Failed to load social_network.json: {e}")

        def validate_obs_df(df: pd.DataFrame, name: str) -> None:
            req = {"day", "agent_id", "wearing_mask", "received_info"}
            missing = req - set(df.columns.astype(str).tolist())
            if missing:
                raise RuntimeError(f"{name} missing required columns: {sorted(list(missing))}")

        try:
            self.train_df = pd.read_csv(train_path)
            validate_obs_df(self.train_df, "train_data.csv")
        except Exception as e:
            raise RuntimeError(f"Failed to load train_data.csv: {e}")

        if os.path.exists(val_path):
            try:
                self.val_df = pd.read_csv(val_path)
                validate_obs_df(self.val_df, "val_data.csv")
            except Exception:
                self.val_df = None
        else:
            self.val_df = None

        if os.path.exists(test_path):
            try:
                self.test_df = pd.read_csv(test_path)
                validate_obs_df(self.test_df, "test_data.csv")
            except Exception:
                self.test_df = None
        else:
            self.test_df = None

        if os.path.exists(meta_path):
            try:
                self.metadata = load_json(meta_path)
                if isinstance(self.metadata, dict) and "gov_intervention_day" in self.metadata:
                    # allow setting frozen via explicit metadata for initialization
                    self.param_registry.set_values({"Global.gov_intervention_day": int(self.metadata["gov_intervention_day"])}, ignore_frozen=False)
            except Exception:
                self.metadata = {}

        agent_ids = set(self.agents_df["agent_id"].astype(int).tolist())
        social_ids = set(int(k) for k in self.social_raw.keys())
        train_ids = set(self.train_df["agent_id"].astype(int).tolist())
        common = sorted(list(agent_ids & social_ids & train_ids))
        if not common:
            raise RuntimeError("No common agent IDs across agent_attributes, social_network, and train_data")
        self.common_ids = np.array(common, dtype=int)
        self.id2idx = {aid: i for i, aid in enumerate(common)}

        # Align agents_df to common_ids order
        self.agents_df = self.agents_df[self.agents_df["agent_id"].isin(common)].copy()
        self.agents_df = self.agents_df.set_index("agent_id").loc[self.common_ids].reset_index()

        # Build neighbors adjacency
        self.neighbors = build_multiplex_adjacency(self.social_raw, self.id2idx, len(self.common_ids))

        # Risk vector
        self.risk = self.agents_df.set_index("agent_id").loc[self.common_ids]["risk_perception"].to_numpy(dtype=np.float64)

        # Demographics one-hot (vectorized)
        age_series = self.agents_df.set_index("agent_id").loc[self.common_ids]["age_group"].astype(str)
        occ_series = self.agents_df.set_index("agent_id").loc[self.common_ids]["occupation"].astype(str)
        age_cats = sorted(pd.unique(age_series).tolist())
        occ_cats = sorted(pd.unique(occ_series).tolist())
        N = len(self.common_ids)

        K_age = max(1, len(age_cats) - 1)
        K_occ = max(1, len(occ_cats) - 1)
        self.age_oh = np.zeros((N, K_age), dtype=np.float64)
        self.occ_oh = np.zeros((N, K_occ), dtype=np.float64)
        age_baseline = age_cats[0] if len(age_cats) > 0 else None
        occ_baseline = occ_cats[0] if len(occ_cats) > 0 else None
        age_index_map = {c: j for j, c in enumerate([x for x in age_cats if x != age_baseline])}
        occ_index_map = {c: j for j, c in enumerate([x for x in occ_cats if x != occ_baseline])}
        age_vals = age_series.to_numpy()
        occ_vals = occ_series.to_numpy()

        if age_index_map:
            age_idx_arr = np.array([age_index_map.get(x, -1) for x in age_vals], dtype=int)
            mask = age_idx_arr >= 0
            self.age_oh[np.where(mask)[0], age_idx_arr[mask]] = 1.0
        if occ_index_map:
            occ_idx_arr = np.array([occ_index_map.get(x, -1) for x in occ_vals], dtype=int)
            mask = occ_idx_arr >= 0
            self.occ_oh[np.where(mask)[0], occ_idx_arr[mask]] = 1.0

        # Dynamically ensure parameter definitions match these one-hot dimensions
        self.param_registry.ensure_demographic_params(self.age_oh.shape[1], self.occ_oh.shape[1])

        self.obs_wearing_train, self.obs_received_train, self.obs_days_train = self._pivot_states(self.train_df)

        if self.val_df is not None:
            self.obs_wearing_val, self.obs_received_val, self.obs_days_val = self._pivot_states(self.val_df)
        else:
            self.obs_wearing_val, self.obs_received_val, self.obs_days_val = None, None, None

        if self.test_df is not None:
            self.obs_wearing_test, self.obs_received_test, self.obs_days_test = self._pivot_states(self.test_df)
        else:
            self.obs_wearing_test, self.obs_received_test, self.obs_days_test = None, None, None

        if self.cfg.verbose:
            print(f"Loaded data: N={N} agents, T_train={self.obs_wearing_train.shape[0]} days")

    @staticmethod
    def _parse_bool_value(val: Any) -> float:
        """
        Robust boolean parsing for various representations. Returns 0.0 or 1.0.
        """
        try:
            if pd.isna(val):
                return 0.0
        except Exception:
            pass
        if isinstance(val, (np.integer, int)):
            return 1.0 if int(val) != 0 else 0.0
        if isinstance(val, (np.floating, float)):
            try:
                if math.isnan(float(val)):
                    return 0.0
            except Exception:
                pass
            return 1.0 if float(val) >= 0.5 else 0.0
        if isinstance(val, str):
            s = val.strip().lower()
            if s in {"1", "true", "yes", "y", "t"}:
                return 1.0
            if s in {"0", "false", "no", "n", "f"}:
                return 0.0
            try:
                fv = float(s)
                return 1.0 if fv >= 0.5 else 0.0
            except Exception:
                return 0.0
        return 1.0 if bool(val) else 0.0

    def _pivot_states(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Pivot long-format (day, agent_id, wearing_mask, received_info) into dense matrices [T,N].
        """
        days_sorted = sorted(pd.unique(df["day"].astype(int)))
        N = len(self.common_ids)
        day_to_idx = {d: i for i, d in enumerate(days_sorted)}
        wearing = np.zeros((len(days_sorted), N), dtype=np.float64)
        received = np.zeros((len(days_sorted), N), dtype=np.float64)
        for _, row in df.iterrows():
            d = int(row["day"])
            a = int(row["agent_id"])
            if a not in self.id2idx:
                continue
            i_day = day_to_idx[d]
            i_agent = self.id2idx[a]
            wm_val = row["wearing_mask"]
            ri_val = row["received_info"]
            wearing[i_day, i_agent] = self._parse_bool_value(wm_val)
            received[i_day, i_agent] = self._parse_bool_value(ri_val)
        return wearing, received, days_sorted

    def build_modules(self) -> None:
        """
        Initialize simulator modules.
        """
        demographics = {"age_oh": self.age_oh, "occ_oh": self.occ_oh}
        self.modules = [
            ActivityModule(name="Activity"),
            InfoPropagationModule(neighbors=self.neighbors, name="Info"),
            DecisionModule(neighbors=self.neighbors, demographics=demographics, name="Decision"),
        ]

    def set_params(self, mapping: Dict[str, Any]) -> Dict[str, str]:
        """
        Apply parameter value mapping to registry.
        """
        warnings = self.param_registry.set_values(mapping, ignore_frozen=True)
        save_json(self.param_registry.to_json(), self.params_used_path)
        if self.cfg.verbose and warnings:
            print(f"Parameter warnings: {warnings}")
        return warnings

    def get_params(self) -> Dict[str, Any]:
        """
        Get current parameter values from registry.
        """
        return self.param_registry.to_json()

    def _module_params(self, module_name: str) -> Dict[str, Any]:
        """
        Slice registry values into per-module dictionary.
        """
        p: Dict[str, Any] = {}
        for k, v in self.param_registry.values.items():
            if k.startswith(f"{module_name}."):
                sub = k.split(".", 1)[1]
                p[sub] = v
        return p

    def _select_received_matrix(self, dataset: str) -> Optional[np.ndarray]:
        if dataset == "train":
            return self.obs_received_train
        elif dataset == "val":
            return self.obs_received_val
        elif dataset == "test":
            return self.obs_received_test
        else:
            return None

    def _select_days_list(self, dataset: str) -> Optional[List[int]]:
        if dataset == "train":
            return self.obs_days_train
        elif dataset == "val":
            return self.obs_days_val
        elif dataset == "test":
            return self.obs_days_test
        else:
            return None

    def _init_memory_from_observed(self, dataset: str, start_idx: int) -> Optional[np.ndarray]:
        """
        Initialize memory vector mem_prev using observed received_info up to start_idx-1.
        """
        rec = self._select_received_matrix(dataset)
        if rec is None:
            return None
        rho = float(self.param_registry.get("Info.rho_info_decay", 0.5))
        N = rec.shape[1]
        mem = np.zeros(N, dtype=np.float64)
        for t in range(0, min(start_idx, rec.shape[0])):
            mem = rho * mem + (1.0 - rho) * rec[t, :]
        return mem

    def compute_memory_from_observed(self, dataset: str, upto_idx: int) -> Optional[np.ndarray]:
        """
        Compute memory up to and including upto_idx from observed received_info.
        """
        rec = self._select_received_matrix(dataset)
        if rec is None:
            return None
        upto = min(max(upto_idx, 0), rec.shape[0] - 1)
        rho = float(self.param_registry.get("Info.rho_info_decay", 0.5))
        N = rec.shape[1]
        mem = np.zeros(N, dtype=np.float64)
        for t in range(0, upto + 1):
            mem = rho * mem + (1.0 - rho) * rec[t, :]
        return mem

    def run(self, start_idx: int, end_idx: int, init_prev_states: Optional[np.ndarray] = None, dataset: str = "train", init_mem_prev: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Roll out the simulator on the specified dataset and window [start_idx, end_idx).
        Uses observed last state and memory to initialize if not provided.
        Passes absolute day values to modules for time-dependent effects.
        """
        if self.neighbors is None:
            raise RuntimeError("Neighbors not built. Call load_data() first.")
        if len(self.modules) < 3:
            raise RuntimeError("Modules not built. Call build_modules() first.")

        T = end_idx - start_idx
        if T <= 0:
            raise ValueError("Invalid window for simulation: empty or negative length.")
        N = len(self.common_ids)

        # Determine day values for this dataset to pass to modules
        days_list = self._select_days_list(dataset)
        if days_list is None or len(days_list) == 0:
            # fallback to sequential days starting at 0
            days_seq = list(range(T))
        else:
            # Ensure indices are within range
            if end_idx > len(days_list):
                raise RuntimeError("End index exceeds available days in selected dataset.")
            days_seq = days_list[start_idx:end_idx]

        if init_prev_states is not None:
            prev = init_prev_states.copy().astype(np.float64)
        else:
            obs_matrix = None
            if dataset == "train":
                obs_matrix = self.obs_wearing_train
            elif dataset == "val":
                obs_matrix = self.obs_wearing_val
            elif dataset == "test":
                obs_matrix = self.obs_wearing_test
            if obs_matrix is not None:
                if start_idx > 0 and start_idx - 1 < obs_matrix.shape[0]:
                    prev = obs_matrix[start_idx - 1, :].copy().astype(np.float64)
                elif start_idx == 0 and self.cfg.init_from_day0 and obs_matrix.shape[0] > 0:
                    prev = obs_matrix[0, :].copy().astype(np.float64)
                else:
                    prev = np.zeros(N, dtype=np.float64)
            else:
                prev = np.zeros(N, dtype=np.float64)

        if init_mem_prev is not None:
            mem_prev = init_mem_prev.copy().astype(np.float64)
        else:
            mem_init = self._init_memory_from_observed(dataset, start_idx)
            mem_prev = mem_init if mem_init is not None else np.zeros(N, dtype=np.float64)

        states = np.zeros((T, N), dtype=np.float64)
        probs = np.zeros((T, N), dtype=np.float64)
        info = np.zeros((T, N), dtype=np.float64)
        memory = np.zeros((T, N), dtype=np.float64)

        # Optional tracing for diagnostics
        trace: Dict[str, List[float]] = {} if self.cfg.debug_trace else None

        global_state = {
            "gov_intervention_day": int(self.param_registry.get("Global.gov_intervention_day", 10)),
            "layer_weights": {
                "family_weight": float(self.param_registry.get("Layers.family_weight", 1.0)),
                "work_weight": float(self.param_registry.get("Layers.work_weight", 1.0)),
                "community_weight": float(self.param_registry.get("Layers.community_weight", 1.0)),
            },
        }

        for k in range(T):
            day_value = days_seq[k]
            self.buffers = {}
            p_activity = self._module_params("Activity")
            p_info = self._module_params("Info")
            p_decision = self._module_params("Decision")

            state_view = {
                "prev_state": prev,
                "mem_prev": mem_prev,
                "risk": self.risk,
                "global": global_state,
            }

            # Activity module uses absolute day value
            self.modules[0].forward(state_view, self.buffers, p_activity, day_value)
            # Info propagation uses absolute day value; decision uses mem_prev (causal)
            self.modules[1].forward(state_view, self.buffers, p_info, day_value)
            rec = self.buffers.get("received_info")
            mem_next = self.buffers.get("mem_next")

            # Decision uses previous memory, not the simultaneously updated one
            self.modules[2].forward(state_view, self.buffers, p_decision, day_value)
            p = self.buffers.get("p_adopt")
            nxt = self.buffers.get("state_next")

            info[k, :] = rec
            probs[k, :] = p
            states[k, :] = nxt
            memory[k, :] = mem_next

            # Trace selected diagnostics
            if trace is not None:
                def add_trace(key: str, val: float) -> None:
                    if key not in trace:
                        trace[key] = []
                    trace[key].append(float(val))
                add_trace("activity_level", float(self.buffers.get("activity_level", 0.0)))
                add_trace("share_f_mean", float(np.mean(self.buffers.get("share_f", np.zeros(N)))))
                add_trace("share_w_mean", float(np.mean(self.buffers.get("share_w", np.zeros(N)))))
                add_trace("share_c_mean", float(np.mean(self.buffers.get("share_c", np.zeros(N)))))
                add_trace("received_info_mean", float(np.mean(rec)))
                add_trace("p_adopt_mean", float(np.mean(p)))
                add_trace("day_value", float(day_value))

            prev = nxt
            mem_prev = mem_next

        self.outputs = {
            "states": states,
            "probs": probs,
            "info": info,
            "memory": memory,
            "days": days_seq,
        }
        if trace is not None:
            self.outputs["trace"] = trace
        return self.outputs

    def evaluate(self, observed_wearing: np.ndarray, start_idx: int, end_idx: int, observed_received: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate simulation outputs against observed wearing and optionally received_info.
        """
        if not self.outputs:
            raise RuntimeError("No simulation outputs available. Run the simulation first.")

        sim_states = self.outputs["states"]
        sim_probs = self.outputs["probs"]
        T = end_idx - start_idx
        if T != sim_states.shape[0]:
            raise RuntimeError("Simulation outputs length does not match evaluation window.")

        obs = observed_wearing[start_idx:end_idx, :]
        sim_rates = sim_states.mean(axis=1)
        obs_rates = obs.mean(axis=1)

        rmse = float(np.sqrt(np.mean((sim_rates - obs_rates) ** 2)))
        mae = float(np.mean(np.abs(sim_rates - obs_rates)))
        brier = float(np.mean((sim_probs - obs) ** 2))

        if T >= 2:
            prev_obs = obs[:-1, :].flatten()
            curr_obs = obs[1:, :].flatten()
            prev_sim = sim_states[:-1, :].flatten()
            curr_sim = sim_states[1:, :].flatten()

            def trans_probs(prev_flat: np.ndarray, curr_flat: np.ndarray) -> Dict[str, float]:
                mask_prev0 = (prev_flat == 0.0)
                mask_prev1 = (prev_flat == 1.0)
                denom0 = float(np.sum(mask_prev0))
                denom1 = float(np.sum(mask_prev1))
                p01 = float(np.sum(mask_prev0 & (curr_flat == 1.0)) / denom0) if denom0 > 0 else 0.0
                p00 = float(np.sum(mask_prev0 & (curr_flat == 0.0)) / denom0) if denom0 > 0 else 0.0
                p11 = float(np.sum(mask_prev1 & (curr_flat == 1.0)) / denom1) if denom1 > 0 else 0.0
                p10 = float(np.sum(mask_prev1 & (curr_flat == 0.0)) / denom1) if denom1 > 0 else 0.0
                return {"P01": p01, "P11": p11, "P10": p10, "P00": p00}

            obs_tp = trans_probs(prev_obs, curr_obs)
            sim_tp = trans_probs(prev_sim, curr_sim)
            trans_err = {k: abs(sim_tp[k] - obs_tp[k]) for k in ["P01", "P11", "P10", "P00"]}
            trans_err_mean = float(np.mean(list(trans_err.values())))
        else:
            trans_err = {"P01": 0.0, "P11": 0.0, "P10": 0.0, "P00": 0.0}
            trans_err_mean = 0.0

        # Wasserstein over time-aggregated rates
        wdist_time = wasserstein_safe(sim_rates, obs_rates)

        # Wasserstein per-day over agent-level probabilities vs observations
        wdist_agent_mean = None
        try:
            per_day = []
            for t in range(T):
                per_day.append(wasserstein_safe(self.outputs["probs"][t, :], obs[t, :]))
            if per_day:
                wdist_agent_mean = float(np.mean(per_day))
        except Exception:
            wdist_agent_mean = None

        # Broadcast memory accuracy (optional)
        broadcast_mem_acc = None
        if "memory" in self.outputs:
            sim_mem = self.outputs["memory"].mean(axis=1)
            if observed_received is None and self.obs_received_train is not None:
                observed_received = self.obs_received_train
            if observed_received is not None and observed_received.shape[0] >= end_idx:
                obs_rec = observed_received[start_idx:end_idx, :].mean(axis=1)
                broadcast_mem_acc = float(np.mean(np.abs(sim_mem - obs_rec)))

        metrics = {
            "RMSE_aggregate": rmse,
            "MAE_aggregate": mae,
            "RMSE_aggregate_mean": rmse,
            "MAE_aggregate_mean": mae,
            "Brier": brier,
            "TransitionFit": trans_err,
            "TransitionFit_mean": trans_err_mean,
            "Wasserstein": wdist_time,
            "Wasserstein_time_rate": wdist_time,
            "observed_daily_rates": obs_rates.tolist(),
            "predicted_daily_rates_mean": sim_rates.tolist(),
        }
        if wdist_agent_mean is not None:
            metrics["Wasserstein_agents_mean"] = wdist_agent_mean
        if broadcast_mem_acc is not None:
            metrics["Broadcast_memory_accuracy"] = broadcast_mem_acc
        return metrics

    def save_results(self, results_path: str) -> None:
        """
        Save full simulation outputs to disk as JSON.
        """
        if not self.outputs:
            raise RuntimeError("No outputs to save; run the simulation first.")
        payload: Dict[str, Any] = {
            "days": self.outputs["days"],
            "states": self.outputs["states"].tolist(),
            "probs": self.outputs["probs"].tolist(),
            "info": self.outputs["info"].tolist(),
            "memory": self.outputs["memory"].tolist(),
        }
        if "trace" in self.outputs:
            payload["trace"] = self.outputs["trace"]
        save_json(payload, results_path)

    def save_module_io(self, module: Module, path: str) -> None:
        """
        Save module buffers (last step and optional trace) to disk for diagnostics.
        """
        payload: Dict[str, Any] = {
            "module": module.name,
            "buffers": {k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (int, float)) else v)
                        for k, v in self.buffers.items()}
        }
        if "trace" in self.outputs:
            payload["trace"] = self.outputs["trace"]
        save_json(payload, path)

    def save_all_io(self, root_dir: str) -> None:
        """
        Save IO for all modules and final results.
        """
        ensure_dir(root_dir)
        for mod in self.modules:
            self.save_module_io(mod, os.path.join(root_dir, f"{mod.name}_io.json"))
        self.save_results(os.path.join(root_dir, "simulation_outputs.json"))

    def visualize(self, path_png: str, observed: Optional[np.ndarray] = None, start_idx: int = 0, end_idx: Optional[int] = None) -> None:
        """
        Plot simulated vs observed daily adoption rates when possible.
        """
        try:
            import matplotlib.pyplot as plt
            if not self.outputs:
                raise RuntimeError("Run simulation before visualization.")
            sim_rates = self.outputs["states"].mean(axis=1)
            days = self.outputs["days"]
            if end_idx is None:
                end_idx = start_idx + len(days)
            plt.figure(figsize=(8, 4))
            plt.plot(days[:end_idx - start_idx], sim_rates[:end_idx - start_idx], label="Simulated")
            if observed is not None:
                obs_rates = observed[start_idx:end_idx, :].mean(axis=1)
                plt.plot(days[:end_idx - start_idx], obs_rates, label="Observed", linestyle="--")
            plt.xlabel("Day")
            plt.ylabel("Adoption rate")
            plt.title("Mask Adoption Rates: Simulation vs Observed")
            plt.legend()
            dirp = os.path.dirname(path_png) or "."
            ensure_dir(dirp)
            plt.tight_layout()
            plt.savefig(path_png)
            plt.close()
        except Exception:
            save_json({"note": "Visualization not available in this environment."}, path_png + ".json")

    def run_counterfactual(self, name: str, overrides: Dict[str, Any], window: Tuple[int, int], dataset: str = "train", init_from_observed: bool = True) -> Dict[str, Any]:
        """
        Run a counterfactual scenario by temporarily applying parameter overrides, running the simulation,
        and saving outputs to artifacts/counterfactuals/{name}.

        Special network overrides (handled here, not in parameter registry):
          - network_rescale_weights: {layer_name: factor}
          - network_drop_fraction: {layer_name: fraction_to_drop_in_[0,1]}
          - network_use_social_path: path to alternate social_network.json to use
        """
        cf_dir = os.path.join(self.artifacts_dir, "counterfactuals", name)
        ensure_dir(cf_dir)

        # Separate network overrides from parameter overrides
        net_keys = {"network_rescale_weights", "network_drop_fraction", "network_use_social_path"}
        network_cfg: Dict[str, Any] = {}
        param_overrides = overrides.copy()
        for k in list(param_overrides.keys()):
            if k in net_keys:
                network_cfg[k] = param_overrides.pop(k)
        save_json(param_overrides, os.path.join(cf_dir, "overrides.json"))
        if network_cfg:
            save_json(network_cfg, os.path.join(cf_dir, "network_cfg.json"))

        # Backup params and apply overrides
        original_params = self.get_params()
        self.set_params({**original_params, **param_overrides})

        # Backup neighbors and modules
        original_neighbors = self.neighbors
        original_modules = self.modules

        # Apply optional network modifications
        if network_cfg:
            # prepare social source
            if "network_use_social_path" in network_cfg and os.path.exists(network_cfg["network_use_social_path"]):
                try:
                    with open(network_cfg["network_use_social_path"], "r", encoding="utf-8") as f:
                        social_source = json.load(f)
                except Exception:
                    social_source = self.social_raw
            else:
                social_source = self.social_raw

            # Make a shallow copy for modifications
            social_mod = {}
            for k, v in social_source.items():
                social_mod[str(k)] = {
                    "family": list(v.get("family", [])),
                    "work_school": list(v.get("work_school", [])),
                    "community": list(v.get("community", []))
                }

            # Drop fraction of edges per layer
            drop_cfg = network_cfg.get("network_drop_fraction", {})
            layers = ["family", "work_school", "community"]
            # Build undirected edge sets
            if isinstance(drop_cfg, dict) and any(layer in drop_cfg for layer in layers):
                for layer in layers:
                    frac = float(drop_cfg.get(layer, 0.0))
                    if frac <= 0.0:
                        continue
                    # Build edge set
                    edges = set()
                    for a_str, neighs in social_mod.items():
                        for nbr in neighs.get(layer, []):
                            if isinstance(nbr, dict):
                                b = int(nbr.get("id", nbr.get("agent_id", nbr.get("nbr", -1))))
                            else:
                                b = int(nbr)
                            a = int(a_str)
                            if a == b:
                                continue
                            edge = (min(a, b), max(a, b))
                            edges.add(edge)
                    edges = list(edges)
                    random.shuffle(edges)
                    n_drop = int(len(edges) * frac)
                    drop_set = set(edges[:n_drop])
                    # Remove dropped edges from adjacency
                    for (a, b) in drop_set:
                        for u, v in [(a, b), (b, a)]:
                            lst = social_mod.get(str(u), {}).get(layer, [])
                            new_lst = []
                            for nbr in lst:
                                if isinstance(nbr, dict):
                                    nid = int(nbr.get("id", nbr.get("agent_id", nbr.get("nbr", -1))))
                                else:
                                    nid = int(nbr)
                                if nid != v:
                                    new_lst.append(nbr)
                            social_mod[str(u)][layer] = new_lst

            # Rebuild neighbors
            self.neighbors = build_multiplex_adjacency(social_mod, self.id2idx, len(self.common_ids))

            # Rescale weights per layer
            rescale_cfg = network_cfg.get("network_rescale_weights", {})
            if isinstance(rescale_cfg, dict):
                for layer, factor in rescale_cfg.items():
                    if layer in self.neighbors:
                        try:
                            f = float(factor)
                        except Exception:
                            f = 1.0
                        for i in range(len(self.neighbors[layer])):
                            idxs, ws = self.neighbors[layer][i]
                            if isinstance(ws, np.ndarray) and ws.size > 0:
                                self.neighbors[layer][i] = (idxs, ws * f)

            # Rebuild modules with modified neighbors
            demographics = {"age_oh": self.age_oh, "occ_oh": self.occ_oh}
            self.modules = [
                ActivityModule(name="Activity"),
                InfoPropagationModule(neighbors=self.neighbors, name="Info"),
                DecisionModule(neighbors=self.neighbors, demographics=demographics, name="Decision"),
            ]

        start_idx, end_idx = window
        if init_from_observed:
            if dataset == "train":
                init_prev = self.obs_wearing_train[start_idx - 1, :] if start_idx > 0 else (self.obs_wearing_train[0, :] if self.cfg.init_from_day0 else np.zeros_like(self.obs_wearing_train[0, :]))
                init_mem = self._init_memory_from_observed("train", start_idx)
            elif dataset == "val":
                if self.obs_wearing_val is not None:
                    init_prev = self.obs_wearing_val[start_idx - 1, :] if start_idx > 0 else (self.obs_wearing_val[0, :] if self.cfg.init_from_day0 else np.zeros_like(self.obs_wearing_val[0, :]))
                else:
                    init_prev = None
                init_mem = self._init_memory_from_observed("val", start_idx)
            else:
                if self.obs_wearing_test is not None:
                    init_prev = self.obs_wearing_test[start_idx - 1, :] if start_idx > 0 else (self.obs_wearing_test[0, :] if self.cfg.init_from_day0 else np.zeros_like(self.obs_wearing_test[0, :]))
                else:
                    init_prev = None
                init_mem = self._init_memory_from_observed("test", start_idx)
        else:
            init_prev = None
            init_mem = None

        self.run(start_idx, end_idx, init_prev_states=init_prev, dataset=dataset, init_mem_prev=init_mem)
        self.save_results(os.path.join(cf_dir, "simulation_outputs.json"))
        observed_matrix = (self.obs_wearing_train if dataset == "train" else self.obs_wearing_val if dataset == "val" else self.obs_wearing_test)
        self.visualize(os.path.join(cf_dir, "plot.png"),
                       observed=observed_matrix,
                       start_idx=start_idx, end_idx=end_idx)

        # Restore neighbors and modules
        if network_cfg:
            self.neighbors = original_neighbors
            self.modules = original_modules

        # Restore params
        self.set_params(original_params)
        return self.outputs


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic regression head to infer decision-related weights from observed data.
    """
    def __init__(self, l2_reg: float = 1.0, max_iter: int = 300, learning_rate: float = 0.1, calibrate_tau: bool = False):
        self.l2_reg = float(l2_reg)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.calibrate_tau = bool(calibrate_tau)

    def _build_features(
        self,
        simulator: Simulation,
        start_idx: int,
        end_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        wearing = simulator.obs_wearing_train
        received = simulator.obs_received_train
        neighbors = simulator.neighbors
        age_oh = simulator.age_oh
        occ_oh = simulator.occ_oh
        risk = simulator.risk
        rho = simulator.param_registry.get("Info.rho_info_decay", 0.5)

        T = wearing.shape[0]
        N = wearing.shape[1]
        mem = np.zeros_like(wearing)
        for t in range(1, T):
            mem[t, :] = rho * mem[t - 1, :] + (1.0 - rho) * received[t, :]

        rows = []
        labels = []
        for t in range(max(1, start_idx), end_idx):
            prev = wearing[t - 1, :]
            share_f = compute_layer_share(prev, neighbors["family"])
            share_w = compute_layer_share(prev, neighbors["work_school"])
            share_c = compute_layer_share(prev, neighbors["community"])
            mem_t = mem[t, :]
            base = np.stack([np.ones(N), prev, share_f, share_w, share_c, risk, mem_t], axis=1)
            if age_oh is not None and age_oh.shape[1] > 0:
                base = np.concatenate([base, age_oh], axis=1)
            if occ_oh is not None and occ_oh.shape[1] > 0:
                base = np.concatenate([base, occ_oh], axis=1)
            rows.append(base)
            labels.append(wearing[t, :])
        X = np.vstack(rows)
        y = np.concatenate(labels, axis=0)
        return X, y

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)
        reg_mask = np.ones(n_features, dtype=np.float64)
        reg_mask[0] = 0.0
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for it in range(1, self.max_iter + 1):
            z = X @ w
            p = sigmoid(z)
            grad = X.T @ (p - y) / n_samples + self.l2_reg * reg_mask * w / n_samples
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** it)
            v_hat = v / (1 - beta2 ** it)
            w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return w

    def _calibrate_tau(self, logits: np.ndarray, y: np.ndarray) -> float:
        # simple grid search to minimize Brier score
        taus = np.linspace(0.2, 5.0, 40)
        best_tau = 1.0
        best_score = float("inf")
        for tau in taus:
            p = sigmoid(logits / tau)
            score = float(np.mean((p - y) ** 2))
            if score < best_score:
                best_score = score
                best_tau = float(tau)
        return best_tau

    def fit(
        self,
        bundle: Any,
        simulator: Simulation,
        evaluator: Callable[[Simulation, FittedParams, Tuple[int, int]], Dict[str, Any]],
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        set_global_seed(seed)
        start_idx, end_idx = train_window
        if simulator.cfg.verbose:
            print("Fitting logistic-head calibrator...")

        X, y = self._build_features(simulator, start_idx, end_idx)
        beta = self._fit_logistic(X, y)
        n_base = 7

        decision: Dict[str, Any] = {
            "alpha": float(beta[0]),
            "gamma": float(beta[1]),
            "theta_f": float(beta[2]),
            "theta_w": float(beta[3]),
            "theta_c": float(beta[4]),
            "beta_r": float(beta[5]),
            "beta_i": float(beta[6]),
            "tau": simulator.param_registry.get("Decision.tau", 1.0),
        }

        idx = n_base
        age_effs: Dict[str, float] = {}
        occ_effs: Dict[str, float] = {}
        age_dim = simulator.age_oh.shape[1] if simulator.age_oh is not None else 0
        occ_dim = simulator.occ_oh.shape[1] if simulator.occ_oh is not None else 0
        for j in range(age_dim):
            age_effs[str(j)] = float(beta[idx + j])
        idx += age_dim
        for j in range(occ_dim):
            occ_effs[str(j)] = float(beta[idx + j])
        decision["age_effects"] = age_effs
        decision["occ_effects"] = occ_effs

        if self.calibrate_tau:
            logits = X @ beta
            decision["tau"] = self._calibrate_tau(logits, y)

        layer = {
            "family": simulator.param_registry.get("Layers.family_weight", 1.0),
            "work_school": simulator.param_registry.get("Layers.work_weight", 1.0),
            "community": simulator.param_registry.get("Layers.community_weight", 1.0),
        }
        info = {
            "phi_family": simulator.param_registry.get("Info.phi_family", 0.1),
            "phi_work": simulator.param_registry.get("Info.phi_work", 0.1),
            "phi_community": simulator.param_registry.get("Info.phi_community", 0.1),
            "lambda_broadcast_base": simulator.param_registry.get("Info.lambda_broadcast_base", 0.05),
            "lambda_broadcast_factor_after": simulator.param_registry.get("Info.lambda_broadcast_factor_after", 1.5),
            "rho_info_decay": simulator.param_registry.get("Info.rho_info_decay", 0.5),
        }
        noise = {"tau": decision["tau"]}
        fitted = FittedParams(
            decision_weights=decision,
            layer_weights=layer,
            info_params=info,
            noise_params=noise,
            module_params={
                "Activity": {
                    "base": simulator.param_registry.get("Activity.base", 0.6),
                    "amplitude": simulator.param_registry.get("Activity.amplitude", 0.2),
                    "phase": simulator.param_registry.get("Activity.phase", 0.0),
                }
            },
            engine_type="calibrasim",
            meta={"calibrator": "logit_head", "seed": seed}
        )

        if params_adapter is not None:
            warnings = params_adapter.apply(simulator, fitted)
            if warnings:
                save_json(warnings, os.path.join(artifacts_dir or simulator.artifacts_dir, "logit_head_warnings.json"))

        _ = evaluator(simulator, fitted, train_window)
        return fitted


class RandomSearchCalibrator(Calibrator):
    """
    Random search over parameter space with configurable objective and early stopping.
    """
    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None, target_metric: Optional[str] = None, weights: Optional[Dict[str, float]] = None, patience: int = 10, eval_every: int = 1):
        self.search_space = search_space or {}
        self.target_metric = target_metric  # if provided, minimize this metric
        # Composite metric weights for RMSE, MAE, TransitionFit_mean, Wasserstein
        self.weights = weights or {"RMSE_aggregate": 0.5, "MAE_aggregate": 0.3, "TransitionFit_mean": 0.1, "Wasserstein": 0.1}
        self.patience = int(patience)
        self.eval_every = int(eval_every)

    def _default_search_space(self, simulator: Simulation) -> Dict[str, Tuple[float, float]]:
        space: Dict[str, Tuple[float, float]] = {}
        keys = [
            "Decision.alpha", "Decision.gamma",
            "Decision.theta_f", "Decision.theta_w", "Decision.theta_c",
            "Decision.beta_r", "Decision.beta_i", "Decision.tau",
            "Info.phi_family", "Info.phi_work", "Info.phi_community",
            "Info.lambda_broadcast_base", "Info.lambda_broadcast_factor_after", "Info.rho_info_decay",
            "Layers.family_weight", "Layers.work_weight", "Layers.community_weight",
            "Activity.base", "Activity.amplitude", "Activity.phase",
        ]
        for k in keys:
            d = simulator.param_registry.definitions.get(k)
            if d is not None:
                low, high = d.bounds
                space[k] = (float(low), float(high))
        return space

    def _sample_params(self, space: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        sample = {}
        for k, (low, high) in space.items():
            sample[k] = random.uniform(low, high)
        return sample

    def _evaluate_validation(self, simulator: Simulation, train_window: Tuple[int, int]) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate current simulator parameters on validation data (or temporal split fallback).
        Returns metrics and score.
        """
        if simulator.obs_wearing_val is not None and simulator.obs_wearing_val.size > 0:
            init_prev_val = simulator.obs_wearing_train[-1, :].copy()
            init_mem_val = simulator.compute_memory_from_observed("train", simulator.obs_wearing_train.shape[0] - 1)
            simulator.run(0, simulator.obs_wearing_val.shape[0], init_prev_states=init_prev_val, dataset="val", init_mem_prev=init_mem_val)
            metrics_val = simulator.evaluate(simulator.obs_wearing_val, 0, simulator.obs_wearing_val.shape[0], observed_received=simulator.obs_received_val)
        else:
            # Temporal split fallback on train data
            days_all = simulator.obs_days_train
            days_train, days_val = temporal_holdout_split(days_all, ratio=0.8)
            val_start = days_all.index(days_val[0])
            val_end = days_all.index(days_val[-1]) + 1
            init_prev_val = simulator.obs_wearing_train[val_start - 1, :] if val_start > 0 else (simulator.obs_wearing_train[0, :] if simulator.cfg.init_from_day0 else np.zeros_like(simulator.obs_wearing_train[0, :]))
            init_mem_val = simulator._init_memory_from_observed("train", val_start)
            simulator.run(val_start, val_end, init_prev_states=init_prev_val, dataset="train", init_mem_prev=init_mem_val if init_mem_val is not None else None)
            metrics_val = simulator.evaluate(simulator.obs_wearing_train, val_start, val_end, observed_received=simulator.obs_received_train)
        score_val = composite_score(metrics_val, self.weights, self.target_metric)
        return metrics_val, score_val

    def fit(
        self,
        bundle: Any,
        simulator: Simulation,
        evaluator: Callable[[Simulation, FittedParams, Tuple[int, int]], Dict[str, Any]],
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        set_global_seed(seed)
        art_dir = artifacts_dir or os.path.join(simulator.artifacts_dir, "calibration_random_search")
        ensure_dir(art_dir)
        if simulator.cfg.verbose:
            print(f"Random search calibration: budget={budget}, patience={self.patience}")

        space = self.search_space or self._default_search_space(simulator)
        best_score = float("inf")
        best_params: Optional[Dict[str, Any]] = None
        best_metrics: Optional[Dict[str, Any]] = None

        base_vals = simulator.get_params()
        no_improve = 0

        for i in range(budget):
            set_global_seed(seed + i)
            trial_dir = os.path.join(art_dir, f"trial_{i}")
            ensure_dir(trial_dir)

            cand = base_vals.copy()
            cand.update(self._sample_params(space))
            warnings = simulator.set_params(cand)

            adapter = params_adapter or ParamsAdapter(simulator.param_registry, simulator.params_used_path)
            fitted = adapter.capture(simulator)

            metrics_train = evaluator(simulator, fitted, train_window)
            score_train = composite_score(metrics_train, self.weights, self.target_metric)

            # Periodically evaluate validation metrics and track best via validation
            metrics_val, score_val = self._evaluate_validation(simulator, train_window)

            save_json(adapter.capture(simulator).to_dict(), os.path.join(trial_dir, "params_applied.json"))
            save_json({"train": metrics_train, "validation": metrics_val}, os.path.join(trial_dir, "metrics.json"))
            if warnings:
                save_json(warnings, os.path.join(trial_dir, "warnings.json"))

            current_score = score_val
            if current_score < best_score:
                best_score = current_score
                best_params = cand.copy()
                best_metrics = {"train": metrics_train, "validation": metrics_val}
                no_improve = 0
                if simulator.cfg.verbose:
                    print(f"Trial {i}: New best validation score={best_score:.6f}")
            else:
                no_improve += 1

            if (i + 1) % max(1, self.eval_every) == 0 and simulator.cfg.verbose:
                print(f"Completed trial {i + 1}/{budget}; train score={score_train:.6f}, val score={score_val:.6f}, no_improve={no_improve}")

            if no_improve >= self.patience:
                if simulator.cfg.verbose:
                    print(f"Early stopping at trial {i + 1} due to no improvement for {self.patience} trials.")
                break

        if best_params is None:
            best_params = base_vals.copy()
        simulator.set_params(best_params)
        adapter = params_adapter or ParamsAdapter(simulator.param_registry, simulator.params_used_path)
        best_fitted = adapter.capture(simulator)
        ensure_dir(os.path.join(art_dir, "best"))
        save_json(best_fitted.to_dict(), os.path.join(art_dir, "best", "fitted_params.json"))
        if best_metrics is not None:
            save_json(best_metrics, os.path.join(art_dir, "best", "metrics.json"))
        report = {
            "budget": budget,
            "best_score": best_score,
            "objective": (self.target_metric or "weighted_composite")
        }
        save_json(report, os.path.join(art_dir, "calibration_report.json"))
        return best_fitted


class SNPECalibrator(Calibrator):
    """
    Simulation-Based Inference calibrator using SNPE from sbi library with graceful fallback.

    Includes reproducibility via torch manual seeds and builds uniform box prior from registry bounds.
    """
    def __init__(self, n_simulations: int = 500, max_epochs: int = 50, batch_size: int = 128, target_metric: Optional[str] = None, weights: Optional[Dict[str, float]] = None, posterior_selection_samples: int = 50):
        self.n_simulations = int(n_simulations)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self._theta_keys: List[str] = []
        self.target_metric = target_metric
        self.weights = weights or {"RMSE_aggregate": 0.5, "MAE_aggregate": 0.3, "TransitionFit_mean": 0.1, "Wasserstein": 0.1}
        self.posterior_selection_samples = int(posterior_selection_samples)

    def _try_imports(self):
        try:
            import torch
            from sbi import utils as sbi_utils  # type: ignore
            from sbi.inference import SNPE as NPE  # type: ignore
            return torch, sbi_utils, NPE
        except Exception:
            return None, None, None

    def _build_prior(self, simulator: Simulation, sbi_utils) -> Any:
        import torch  # local import after checking
        keys = [
            "Decision.alpha", "Decision.gamma",
            "Decision.theta_f", "Decision.theta_w", "Decision.theta_c",
            "Decision.beta_r", "Decision.beta_i",
            "Info.phi_family", "Info.phi_work", "Info.phi_community",
            "Info.lambda_broadcast_base", "Info.lambda_broadcast_factor_after", "Info.rho_info_decay",
            "Decision.tau",
        ]
        mins = []
        maxs = []
        self._theta_keys = []
        for k in keys:
            d = simulator.param_registry.definitions.get(k)
            if d is None:
                continue
            low, high = d.bounds
            mins.append(float(low))
            maxs.append(float(high))
            self._theta_keys.append(k)
        low_t = torch.tensor(mins, dtype=torch.float32)
        high_t = torch.tensor(maxs, dtype=torch.float32)
        prior = sbi_utils.BoxUniform(low=low_t, high=high_t)
        return prior

    def _theta_to_params(self, theta: np.ndarray, simulator: Simulation) -> Dict[str, Any]:
        mapping = {}
        for i, k in enumerate(self._theta_keys):
            mapping[k] = float(theta[i])
        return mapping

    def _evaluate_validation(self, simulator: Simulation, train_window: Tuple[int, int]) -> Tuple[Dict[str, Any], float]:
        if simulator.obs_wearing_val is not None and simulator.obs_wearing_val.size > 0:
            init_prev_val = simulator.obs_wearing_train[-1, :].copy()
            init_mem_val = simulator.compute_memory_from_observed("train", simulator.obs_wearing_train.shape[0] - 1)
            simulator.run(0, simulator.obs_wearing_val.shape[0], init_prev_states=init_prev_val, dataset="val", init_mem_prev=init_mem_val)
            metrics_val = simulator.evaluate(simulator.obs_wearing_val, 0, simulator.obs_wearing_val.shape[0], observed_received=simulator.obs_received_val)
        else:
            days_all = simulator.obs_days_train
            days_train, days_val = temporal_holdout_split(days_all, ratio=0.8)
            val_start = days_all.index(days_val[0])
            val_end = days_all.index(days_val[-1]) + 1
            init_prev_val = simulator.obs_wearing_train[val_start - 1, :] if val_start > 0 else (simulator.obs_wearing_train[0, :] if simulator.cfg.init_from_day0 else np.zeros_like(simulator.obs_wearing_train[0, :]))
            init_mem_val = simulator._init_memory_from_observed("train", val_start)
            simulator.run(val_start, val_end, init_prev_states=init_prev_val, dataset="train", init_mem_prev=init_mem_val if init_mem_val is not None else None)
            metrics_val = simulator.evaluate(simulator.obs_wearing_train, val_start, val_end, observed_received=simulator.obs_received_train)
        score_val = composite_score(metrics_val, self.weights, self.target_metric)
        return metrics_val, score_val

    def fit (
        self,
        bundle: Any,
        simulator: Simulation,
        evaluator: Callable[[Simulation, FittedParams, Tuple[int, int]], Dict[str, Any]],
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        set_global_seed(seed)
        torch, sbi_utils, NPE = self._try_imports()
        art_dir = artifacts_dir or os.path.join(simulator.artifacts_dir, "calibration_snpe")
        ensure_dir(art_dir)

        if torch is None or sbi_utils is None or NPE is None:
            rs = RandomSearchCalibrator(weights=self.weights, target_metric=self.target_metric)
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, art_dir, params_adapter)

        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # type: ignore
                torch.cuda.manual_seed_all(seed)  # type: ignore
        except Exception:
            pass

        prior = self._build_prior(simulator, sbi_utils)

        thetas = []
        xs = []
        for i in range(self.n_simulations):
            theta = prior.sample((1,)).squeeze(0).detach().cpu().numpy()
            thetas.append(theta)
            mapping = self._theta_to_params(theta, simulator)
            simulator.set_params(mapping)
            start_idx, end_idx = train_window
            init_prev = simulator.obs_wearing_train[start_idx - 1, :] if start_idx > 0 else (simulator.obs_wearing_train[0, :] if simulator.cfg.init_from_day0 else np.zeros_like(simulator.obs_wearing_train[0, :]))
            init_mem = simulator._init_memory_from_observed("train", start_idx)
            simulator.run(start_idx, end_idx, init_prev_states=init_prev, dataset="train", init_mem_prev=init_mem if init_mem is not None else None)
            sim_states = simulator.outputs["states"]
            xs.append(sim_states.mean(axis=1))

        thetas_np = np.array(thetas, dtype=np.float32)
        xs_np = np.array(xs, dtype=np.float32)
        theta_tensor = torch.tensor(thetas_np, dtype=torch.float32)
        x_tensor = torch.tensor(xs_np, dtype=torch.float32)

        inference = NPE(prior=prior)
        inference = inference.append_simulations(theta_tensor, x_tensor)
        try:
            density_estimator = inference.train(training_batch_size=self.batch_size, max_num_epochs=self.max_epochs, show_train_summary=False)
        except TypeError:
            density_estimator = inference.train(training_batch_size=self.batch_size, max_num_epochs=self.max_epochs)
        posterior = inference.build_posterior(density_estimator)

        obs = simulator.obs_wearing_train[train_window[0]:train_window[1], :].mean(axis=1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        num_samples = max(1, min(self.posterior_selection_samples, 200))
        posterior_samples = posterior.sample((num_samples,), x=obs_tensor).detach().cpu().numpy()

        # Select best posterior sample by validation performance if possible
        best_theta = None
        best_score = float("inf")
        best_metrics_val = None
        for s in range(posterior_samples.shape[0]):
            theta = posterior_samples[s]
            mapping = self._theta_to_params(theta, simulator)
            simulator.set_params(mapping)
            metrics_val, score_val = self._evaluate_validation(simulator, train_window)
            if score_val < best_score:
                best_score = score_val
                best_theta = theta
                best_metrics_val = metrics_val

        if best_theta is None:
            theta_mean = posterior_samples.mean(axis=0)
            mapping = self._theta_to_params(theta_mean, simulator)
        else:
            mapping = self._theta_to_params(best_theta, simulator)

        simulator.set_params(mapping)
        adapter = params_adapter or ParamsAdapter(simulator.param_registry, simulator.params_used_path)
        fitted = adapter.capture(simulator)
        save_json(fitted.to_dict(), os.path.join(art_dir, "fitted_params.json"))
        if best_metrics_val is not None:
            save_json(best_metrics_val, os.path.join(art_dir, "best_validation_metrics.json"))
        return fitted


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None, **kwargs) -> Calibrator:
    """
    Construct a calibrator by name, optionally merging kwargs with JSON config file.
    """
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    ctor_kwargs: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            cfg = load_json(config_path)
            if isinstance(cfg, dict):
                ctor_kwargs.update(cfg)
        except Exception:
            pass
    ctor_kwargs.update(kwargs)
    return CALIBRATOR_REGISTRY[name](**ctor_kwargs)


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply params to simulator, run on train window, and save metrics to artifacts.
    """
    adapter = ParamsAdapter(simulator.param_registry, simulator.params_used_path)
    _ = adapter.apply(simulator, params)

    start_idx, end_idx = window
    init_prev = simulator.obs_wearing_train[start_idx - 1, :] if start_idx > 0 else (simulator.obs_wearing_train[0, :] if simulator.cfg.init_from_day0 else np.zeros_like(simulator.obs_wearing_train[0, :]))
    init_mem = simulator._init_memory_from_observed("train", start_idx)
    simulator.run(start_idx, end_idx, init_prev_states=init_prev, dataset="train", init_mem_prev=init_mem if init_mem is not None else None)

    metrics = simulator.evaluate(simulator.obs_wearing_train, start_idx, end_idx, observed_received=simulator.obs_received_train)
    ensure_dir(os.path.join(simulator.artifacts_dir, "results"))
    save_json(metrics, os.path.join(simulator.artifacts_dir, "results", f"metrics_{start_idx}_{end_idx}.json"))
    return metrics


def temporal_holdout_split(days: List[int], ratio: float = 0.8) -> Tuple[List[int], List[int]]:
    """
    Split days list temporally according to ratio.
    """
    if not days:
        raise ValueError("Empty days list for temporal split.")
    split_idx = max(1, int(math.floor(ratio * len(days))))
    days_train = days[:split_idx]
    days_val = days[split_idx:]
    if not days_val:
        raise ValueError("No validation days available after temporal split.")
    return days_train, days_val


def parse_cli(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask Adoption Simulator with Calibration")
    p.add_argument("--param-file", type=str, default=None, help="JSON file with parameter overrides")
    p.add_argument("--set", action="append", default=[], help="Parameter override key=value (repeatable)")
    p.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()), help="Calibrator to use")
    p.add_argument("--budget", type=int, default=50, help="Calibration budget (iterations)")
    p.add_argument("--calib-window", type=str, default=None, help="Training window 'start:end' (day values or indices)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts output directory")
    p.add_argument("--calibrator-config", type=str, default=None, help="Path to calibrator config JSON")
    return p.parse_args(argv)


def main() -> None:
    args = parse_cli(sys.argv[1:])
    cfg = SimulationConfig(seed=args.seed)
    set_global_seed(cfg.seed)

    artifacts_dir = args.artifacts_dir or os.path.join(PROJECT_ROOT, "artifacts")
    ensure_dir(artifacts_dir)

    sim = Simulation(data_dir=DATA_DIR, cfg=cfg, artifacts_dir=artifacts_dir)
    try:
        sim.load_data()
    except Exception as e:
        print(f"Error during data loading: {e}")
        return

    sim.build_modules()

    param_overrides: Dict[str, Any] = {}
    if args.param_file and os.path.exists(args.param_file):
        try:
            param_overrides.update(load_json(args.param_file))
        except Exception as e:
            print(f"Warning: Failed to load param file '{args.param_file}': {e}")

    overrides_from_cli: Dict[str, Any] = {}
    for s in args.__dict__.get("set", []) or []:
        try:
            k, v = parse_kv_override(s)
            try:
                lit = ast.literal_eval(v)
                overrides_from_cli[k] = lit
            except Exception:
                # Try int first, then float, else string/bool
                try:
                    overrides_from_cli[k] = int(v)
                except Exception:
                    try:
                        overrides_from_cli[k] = float(v)
                    except Exception:
                        vl = v.lower()
                        if vl in ("true", "false"):
                            overrides_from_cli[k] = (vl == "true")
                        else:
                            overrides_from_cli[k] = v
        except Exception as e:
            print(f"Warning: ignoring override '{s}': {e}")

    all_overrides = {}
    all_overrides.update(param_overrides)
    all_overrides.update(overrides_from_cli)
    warnings = sim.set_params(all_overrides)
    if warnings:
        save_json(warnings, os.path.join(artifacts_dir, "override_warnings.json"))

    # Determine training window
    train_days_all = sim.obs_days_train
    train_start = 0
    train_end = sim.obs_wearing_train.shape[0] if sim.obs_wearing_train is not None else 0

    # Optional calib-window parsing: supports day values or index positions
    if args.calib_window:
        try:
            s_str, e_str = args.calib_window.split(":")
            s_val = int(s_str)
            e_val = int(e_str)
            # Try interpret as day values
            if s_val in train_days_all and e_val in train_days_all:
                train_start = train_days_all.index(s_val)
                train_end = train_days_all.index(e_val) + 1
            else:
                # interpret as indices
                train_start = max(0, s_val)
                train_end = min(sim.obs_wearing_train.shape[0], e_val)
            if not (0 <= train_start < train_end <= sim.obs_wearing_train.shape[0]):
                raise ValueError("Calibration window out of range.")
        except Exception:
            print("Warning: invalid --calib-window, using default full train window.")

    calibrator = get_calibrator(args.calibrator, config_path=args.calibrator_config)
    params_adapter = ParamsAdapter(sim.param_registry, sim.params_used_path)

    bundle = {
        "obs_wearing_train": sim.obs_wearing_train,
        "obs_received_train": sim.obs_received_train,
        "neighbors": sim.neighbors,
        "risk": sim.risk,
        "age_oh": sim.age_oh,
        "occ_oh": sim.occ_oh,
        "cfg": asdict(cfg),
    }

    fitted_params = calibrator.fit(
        bundle=bundle,
        simulator=sim,
        evaluator=evaluate_params,
        train_window=(train_start, train_end),
        seed=cfg.seed,
        budget=args.budget,
        artifacts_dir=os.path.join(artifacts_dir, f"calibration_{args.calibrator}"),
        params_adapter=params_adapter,
    )

    # Validation evaluation
    if sim.obs_wearing_val is not None and sim.obs_wearing_val.size > 0:
        # initialize with last train state and memory continued from train
        init_prev_val = sim.obs_wearing_train[-1, :].copy()
        init_mem_val = sim.compute_memory_from_observed("train", sim.obs_wearing_train.shape[0] - 1)
        sim.run(0, sim.obs_wearing_val.shape[0], init_prev_states=init_prev_val, dataset="val", init_mem_prev=init_mem_val)
        metrics_val = sim.evaluate(sim.obs_wearing_val, 0, sim.obs_wearing_val.shape[0], observed_received=sim.obs_received_val)
        ensure_dir(os.path.join(artifacts_dir, "results"))
        save_json(metrics_val, os.path.join(artifacts_dir, "results", "metrics_validation.json"))
        sim.visualize(os.path.join(artifacts_dir, "figs", "validation_plot.png"), observed=sim.obs_wearing_val, start_idx=0, end_idx=sim.obs_wearing_val.shape[0])
    else:
        # Temporal split fallback on train data
        days_all = sim.obs_days_train
        days_train, days_val = temporal_holdout_split(days_all, ratio=0.8)
        val_start = days_all.index(days_val[0])
        val_end = days_all.index(days_val[-1]) + 1
        init_prev_val = sim.obs_wearing_train[val_start - 1, :] if val_start > 0 else (sim.obs_wearing_train[0, :] if sim.cfg.init_from_day0 else np.zeros_like(sim.obs_wearing_train[0, :]))
        init_mem_val = sim._init_memory_from_observed("train", val_start)
        sim.run(val_start, val_end, init_prev_states=init_prev_val, dataset="train", init_mem_prev=init_mem_val if init_mem_val is not None else None)
        metrics_val = sim.evaluate(sim.obs_wearing_train, val_start, val_end, observed_received=sim.obs_received_train)
        ensure_dir(os.path.join(artifacts_dir, "results"))
        save_json(metrics_val, os.path.join(artifacts_dir, "results", "metrics_validation.json"))
        sim.visualize(os.path.join(artifacts_dir, "figs", "validation_plot.png"), observed=sim.obs_wearing_train, start_idx=val_start, end_idx=val_end)

    # Test evaluation
    if sim.obs_wearing_test is not None and sim.obs_wearing_test.size > 0:
        init_prev_test = sim.obs_wearing_train[-1, :].copy()
        init_mem_test = sim.compute_memory_from_observed("train", sim.obs_wearing_train.shape[0] - 1)
        sim.run(0, sim.obs_wearing_test.shape[0], init_prev_states=init_prev_test, dataset="test", init_mem_prev=init_mem_test)
        metrics_test = sim.evaluate(sim.obs_wearing_test, 0, sim.obs_wearing_test.shape[0], observed_received=sim.obs_received_test)
        save_json(metrics_test, os.path.join(artifacts_dir, "results", "metrics_test.json"))

    save_json(fitted_params.to_dict(), os.path.join(artifacts_dir, "fitted_params.json"))
    sim.save_all_io(os.path.join(artifacts_dir, "io"))

    save_json(sim.param_registry.to_json(), os.path.join(artifacts_dir, "parameters_used.json"))

    print("Calibration and evaluation complete.")
    if 'metrics_val' in locals() and "RMSE_aggregate" in metrics_val and "MAE_aggregate" in metrics_val:
        print(f"Validation RMSE: {metrics_val.get('RMSE_aggregate'):.4f}, MAE: {metrics_val.get('MAE_aggregate'):.4f}")


main()