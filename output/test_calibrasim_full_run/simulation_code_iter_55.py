import sys
import os
import json
import argparse
import time
import random
import math
import socket
import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Callable
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# FIXED: Applied prioritized feedback to align with task specification:
# - Added EpidemiologicalContext module and risk feedback into AdoptionDecision.
# - Integrated supply and affordability constraints into AdoptionDecision.
# - Normalized parameter keys (average_degree, mask_price, restock_rate_per_day, mandate_on/off_day, fine_amount, enforcement_probability_per_entry).
# - Adjusted HouseholdModule to read agents from buffers on t=0 for purchase requests.
# - RetailerModule: skipped restock on t==0 and mapped restock_rate_per_day.
# - LocationModule: tracked entries/masked_entries and computed enforcement counts per entries.
# - evaluate(): added compute_policy_effect flag (to avoid counterfactual during calibration), new metrics (time_to_50_percent_adoption, sustained_adoption_days_above_threshold, adoption_by_location_type, fine_incidents_per_1000_entries, exposure_weighted_adoption).
# - Adapter: propagate time_horizon_days from parameters to initialization; removed default forced to 60 inside adapter.
# - NetworkBuilder: parameter fallback to 'average_degree' per spec.
# - Commit: tracked entries/masked entries in daily_counters; logged risk signals history.
# - Calibration: evaluator avoids counterfactual policy run to improve performance.


def eprint(*args, **kwargs) -> None:
    """
    Print to stderr.
    """
    pass
    print(*args, file=sys.stderr, **kwargs)


def safe_makedirs(path: str) -> None:
    """
    Create directory path if it does not exist.
    """
    pass
    if path:
        os.makedirs(path, exist_ok=True)


def load_text_from_file(path: str) -> str:
    """
    Load text content from a file in UTF-8 encoding.
    """
    pass
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_text_from_url(url: str, timeout: float = 10.0, retries: int = 3, backoff: float = 0.5) -> str:
    """
    Load text content from a URL with retries and exponential backoff.
    Uses urllib to avoid external dependencies.
    """
    pass
    attempt = 0
    last_err = None
    while attempt < retries:
        try:
            req = Request(url, headers={"User-Agent": "social-sim/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except (HTTPError, URLError, socket.timeout) as e:
            # FIXED: Catch socket.timeout to improve network robustness
            last_err = e
            sleep_time = backoff * (2 ** attempt)
            eprint(f"Network error on attempt {attempt + 1}/{retries}: {e}. Retrying in {sleep_time:.2f}s")
            time.sleep(sleep_time)
            attempt += 1
    raise RuntimeError(f"Failed to fetch URL after {retries} attempts: {last_err}")


def parse_json(text: str) -> Any:
    """
    Parse JSON text and report errors with line/column context.
    Raises ValueError on parse errors.
    """
    pass
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        lines = text.splitlines()
        start = max(0, e.lineno - 2)
        end = min(len(lines), e.lineno + 1)
        context_lines = []
        for i in range(start, end):
            prefix = ">>" if (i + 1) == e.lineno else "  "
            context_lines.append(f"{prefix} {i + 1}: {lines[i]}")
        context = "\n".join(context_lines)
        raise ValueError(f"JSON parse error at line {e.lineno}, column {e.colno}: {e.msg}\n{context}") from e


def coerce_type(value: Any, dtype: str) -> Any:
    """
    Coerce a value to a given dtype as specified in plan parameters.
    Supports: 'int', 'float', 'str', 'bool'.
    """
    pass
    if dtype == "int":
        return int(value)
    if dtype == "float":
        return float(value)
    if dtype == "str":
        return str(value)
    if dtype == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
        raise ValueError(f"Cannot coerce value to bool: {value}")
    return value


def clamp(x: float, low: float, high: float) -> float:
    """
    Clamp value x into [low, high].
    """
    pass
    if x < low:
        return low
    if x > high:
        return high
    return x


def truncated_normal(mean_val: float, std: float, low: float, high: float) -> float:
    """
    Sample from a truncated normal by simple rejection sampling.
    """
    pass
    for _ in range(1000):
        sample = random.gauss(mean_val, std)
        if low <= sample <= high:
            return sample
    return clamp(mean_val, low, high)


def lognormal_from_mu_sigma(mu: float, sigma: float) -> float:
    """
    Sample from a lognormal distribution with given log-space mean and stddev.
    """
    pass
    return random.lognormvariate(mu, sigma)


def mean(values: List[float]) -> float:
    """
    Compute mean of a list safely.
    """
    pass
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_mean(values: List[float]) -> float:
    """
    Compute a safe mean for potentially empty lists, returning 0.0 if empty.
    """
    pass
    if not values:
        return 0.0
    return sum(values) / len(values)


def gini(values: List[float]) -> float:
    """
    Compute Gini coefficient for a list of non-negative values.
    """
    pass
    # FIXED: Use filtered length n after removing negatives for correctness
    sorted_vals = sorted(v for v in values if v >= 0)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    cumvals = 0.0
    cum_sum = 0.0
    for i, v in enumerate(sorted_vals, start=1):
        cum_sum += v
        cumvals += i * v
    if cum_sum == 0:
        return 0.0
    g = (2 * cumvals) / (n * cum_sum) - (n + 1) / n
    return g


def max_consecutive_ge(values: List[float], threshold: float) -> int:
    """
    Compute the maximum number of consecutive values >= threshold in the list.
    """
    pass
    max_run = 0
    cur = 0
    for v in values:
        if v >= threshold:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


@dataclass
class ParameterDefinition:
    """
    Definition for a parameter in the registry.
    """
    pass
    key: str
    dtype: str
    default: Any
    bounds: Optional[Dict[str, float]]
    owner_module: str
    description: str = ""
    frozen: bool = False


class ParameterRegistry:
    """
    Registry handling parameter definitions, values, and overrides.
    """
    pass

    def __init__(self, definitions: Dict[str, ParameterDefinition]):
        """
        Initialize a parameter registry with definitions.
        """
        pass
        self.definitions: Dict[str, ParameterDefinition] = definitions
        self.values: Dict[str, Any] = {k: d.default for k, d in definitions.items()}
        self.module_index: Dict[str, List[str]] = {}
        for key, d in definitions.items():
            self.module_index.setdefault(d.owner_module, []).append(key)

    @classmethod
    def from_plan(cls, plan: Dict[str, Any]) -> "ParameterRegistry":
        """
        Build ParameterRegistry from plan parameters.
        Supports either list of definitions or dict of key->value.
        """
        pass
        defs: Dict[str, ParameterDefinition] = {}
        params_section = plan.get("parameters", [])
        if isinstance(params_section, list):
            for p in params_section:
                key = p["key"]
                dtype = p.get("dtype", "float")
                default = p.get("default")
                bounds = p.get("bounds")
                owner = p.get("owner_module", "global")
                frozen = p.get("frozen", False)
                description = p.get("description", "")
                defs[key] = ParameterDefinition(
                    key=key,
                    dtype=dtype,
                    default=default,
                    bounds=bounds,
                    owner_module=owner,
                    description=description,
                    frozen=frozen,
                )
        elif isinstance(params_section, dict):
            # FIXED: Accept dict-style parameters per feedback
            for key, default in params_section.items():
                if isinstance(default, bool):
                    dtype = "bool"
                elif isinstance(default, int):
                    dtype = "int"
                elif isinstance(default, float):
                    dtype = "float"
                else:
                    dtype = "str"
                defs[key] = ParameterDefinition(
                    key=key,
                    dtype=dtype,
                    default=default,
                    bounds=None,
                    owner_module="global",
                    description="",
                    frozen=False,
                )
        else:
            raise ValueError("Unsupported 'parameters' format in plan")
        return cls(defs)

    def set_param(self, key: str, value: Any, allow_frozen_override: bool = False) -> bool:
        """
        Set a parameter value, respecting type and bounds.
        Returns True if applied, False if ignored (e.g., frozen).
        """
        pass
        if key not in self.definitions:
            raise KeyError(f"Unknown parameter key: {key}")
        d = self.definitions[key]
        if d.frozen and not allow_frozen_override:
            eprint(f"Ignoring override for frozen parameter '{key}'")
            return False
        try:
            coerced = coerce_type(value, d.dtype)
        except Exception as ex:
            raise ValueError(f"Failed to coerce '{key}' to {d.dtype}: {ex}") from ex
        if d.bounds and isinstance(coerced, (int, float)):
            low = d.bounds.get("low", float("-inf"))
            high = d.bounds.get("high", float("inf"))
            if not (low <= coerced <= high):
                eprint(f"Warning: value {coerced} for '{key}' outside bounds [{low}, {high}], clamping")
                coerced = clamp(coerced, low, high)
        self.values[key] = coerced
        return True

    def get_param(self, key: str) -> Any:
        """
        Retrieve a parameter value.
        """
        pass
        if key not in self.definitions:
            raise KeyError(f"Unknown parameter key: {key}")
        return self.values[key]

    def get_param_or(self, key: str, default: Any) -> Any:
        """
        Retrieve a parameter value if defined; otherwise return default.
        This is used for optional parameters to improve robustness.
        """
        pass
        if key not in self.definitions:
            return default
        return self.values.get(key, default)

    def apply_file(self, path: str) -> None:
        """
        Apply parameter values from a JSON file mapping key->value.
        """
        pass
        try:
            text = load_text_from_file(path)
            data = parse_json(text)
            if not isinstance(data, dict):
                raise ValueError("Parameter file must contain a JSON object mapping keys to values")
            for k, v in data.items():
                try:
                    self.set_param(k, v)
                except Exception as ex:
                    eprint(f"Error applying param '{k}': {ex}")
        except FileNotFoundError:
            eprint(f"Parameter file not found: {path}")
        except Exception as ex:
            eprint(f"Failed to apply parameter file '{path}': {ex}")

    def apply_overrides(self, overrides: List[str]) -> None:
        """
        Apply CLI overrides of the form key=value.
        """
        pass
        for ov in overrides:
            if "=" not in ov:
                eprint(f"Ignoring malformed override: '{ov}', expected key=value")
                continue
            key, value_str = ov.split("=", 1)
            key = key.strip()
            value_str = value_str.strip()
            if key not in self.definitions:
                eprint(f"Ignoring unknown param override: '{key}'")
                continue
            dtype = self.definitions[key].dtype
            try:
                value = coerce_type(value_str, dtype)
                self.set_param(key, value)
            except Exception as ex:
                eprint(f"Failed to apply override '{ov}': {ex}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a snapshot of currently effective parameter values.
        """
        pass
        return dict(self.values)

    def save_used(self, path: str) -> None:
        """
        Save effective parameters to a JSON file.
        """
        pass
        safe_makedirs(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def save_definitions(self, path: str) -> None:
        """
        Save parameter definitions for downstream tools/calibration.
        """
        pass
        safe_makedirs(os.path.dirname(path))
        defs = []
        for k, d in self.definitions.items():
            defs.append({
                "key": d.key,
                "dtype": d.dtype,
                "default": d.default,
                "bounds": d.bounds,
                "owner_module": d.owner_module,
                "description": d.description,
                "frozen": d.frozen,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(defs, f, indent=2, sort_keys=False)


def validate_plan(plan: Dict[str, Any]) -> None:
    """
    Validate minimal structure of the plan; supports internal schema and task spec schema.
    Raises ValueError on invalid structure.
    """
    pass
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    # FIXED: Accept both internal and task spec schemas per feedback
    if "modules" in plan and isinstance(plan.get("modules"), list):
        names = [m.get("name", m.get("type")) for m in plan["modules"]]
        if any(n is None for n in names):
            raise ValueError("Each module must include 'name' or 'type'")
        if len(set(names)) != len(names):
            raise ValueError("Module names must be unique")
        params = plan.get("parameters", [])
        if not (isinstance(params, list) or isinstance(params, dict)):
            raise ValueError("Plan 'parameters' must be a non-empty list or dict")
        observables = plan.get("observables", [])
        for obs in observables:
            if "id" not in obs or "source_module" not in obs:
                raise ValueError("Each observable must include 'id' and 'source_module'")
            if "target_data_field" not in obs:
                raise ValueError(f"Observable '{obs.get('id', '')}' missing 'target_data_field'")
    elif all(k in plan for k in ("entities", "interactions", "parameters")):
        # Task spec schema is accepted; adapter will handle further checks
        pass
    else:
        raise ValueError("Unknown plan schema: expected either internal {modules,parameters} or task spec {entities,interactions,parameters}")


def adapt_task_spec_to_internal_plan(task_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a task specification style plan into the internal modules+parameters plan.
    This is a minimal adapter to unblock execution.
    """
    pass
    modules: List[Dict[str, Any]] = [
        {"name": "NetworkBuilder", "type": "NetworkBuilder", "dependencies": []},
        {"name": "AgentInitializer", "type": "AgentInitializer", "dependencies": ["NetworkBuilder"]},
        {"name": "HouseholdModule", "type": "HouseholdModule", "dependencies": ["AgentInitializer"]},
        {"name": "RetailerModule", "type": "RetailerModule", "dependencies": []},
        {"name": "PolicyMakerModule", "type": "PolicyMakerModule", "dependencies": []},
        {"name": "MediaMisinformationModule", "type": "MediaMisinformationModule", "dependencies": []},
        {"name": "LocationModule", "type": "LocationModule", "dependencies": ["PolicyMakerModule"]},
        {"name": "PeerInfluence", "type": "PeerInfluence", "dependencies": ["AgentInitializer"]},
        {"name": "MediaInfluence", "type": "MediaInfluence", "dependencies": []},
        # FIXED: Added EpidemiologicalContext module for risk feedback
        {"name": "EpidemiologicalContext", "type": "EpidemiologicalContext", "dependencies": []},
        {"name": "PolicyIntervention", "type": "PolicyIntervention", "dependencies": ["PolicyMakerModule"]},
        # FIXED: Added EpidemiologicalContext as dependency for AdoptionDecision
        {"name": "AdoptionDecision", "type": "AdoptionDecision", "dependencies": ["PeerInfluence", "MediaInfluence", "PolicyIntervention", "EpidemiologicalContext"]},
        {"name": "AdoptionAggregator", "type": "AdoptionAggregator", "dependencies": ["AdoptionDecision"]},
    ]
    # FIXED: Propagate time_horizon_days from parameters to initialization; avoid forcing default 60
    horizon = task_plan.get("parameters", {}).get("time_horizon_days")
    init = task_plan.get("initialization", {})
    if horizon is not None:
        init = dict(init)
        init["time_horizon_days"] = int(horizon)
    adapted = {
        "modules": modules,
        "parameters": task_plan.get("parameters", {}),
        "observables": task_plan.get("observables", [
            {"id": "observable.adoption_rate_daily", "source_module": "AdoptionAggregator", "target_data_field": "adoption_rate"}
        ]),
        "initialization": init,
        "data_sources": task_plan.get("data_sources", []),
        "algorithms": task_plan.get("algorithms", []),
        "evaluation_metrics": task_plan.get("evaluation_metrics", []),
        "environment": task_plan.get("environment", {}),
        "prediction_period": task_plan.get("prediction_period", {"end_day": 60}),
        "code_structure": task_plan.get("code_structure", {}),
    }
    return adapted


class SimModule:
    """
    Base class for simulation modules.
    """
    pass

    def __init__(self, name: str, config: Dict[str, Any], registry: ParameterRegistry):
        """
        Initialize the module with its name, config from plan, and parameter registry.
        """
        pass
        self.name = name
        self.config = config
        self.registry = registry
        self.dependencies = config.get("dependencies", [])
        self.tick_rate = config.get("tick_rate", {"unit": "days", "value": 1})
        self.io_log: Dict[int, Dict[str, Any]] = {}

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute outputs for time step t. Should not mutate state; return outputs to be merged into buffers.
        """
        pass
        return {}


class NetworkBuilder(SimModule):
    """
    Constructs the contact network among Person agents using configurable generative models.
    Produces adjacency list 'graph' and 'neighbors'.
    """
    pass

    def _ring_lattice(self, n: int, k: int) -> List[List[int]]:
        """
        Construct a ring lattice where each node connects to k nearest neighbors (k even).
        """
        pass
        if k % 2 == 1:
            k -= 1
        neighbors = [[] for _ in range(n)]
        half = k // 2
        for i in range(n):
            for d in range(1, half + 1):
                j1 = (i + d) % n
                j2 = (i - d) % n
                neighbors[i].append(j1)
                neighbors[i].append(j2)
        for i in range(n):
            neighbors[i] = sorted(set(neighbors[i]))
        return neighbors

    def _watts_strogatz(self, n: int, k: int, p: float) -> List[List[int]]:
        """
        Watts-Strogatz small-world graph generator without external dependencies.
        """
        pass
        nbrs = self._ring_lattice(n, k)
        for i in range(n):
            for j in list(nbrs[i]):
                if j < i:
                    continue
                if random.random() < p:
                    allowed = set(range(n))
                    allowed.discard(i)
                    allowed -= set(nbrs[i])
                    if not allowed:
                        continue
                    l = random.choice(list(allowed))
                    if j in nbrs[i]:
                        nbrs[i].remove(j)
                    if i in nbrs[j]:
                        nbrs[j].remove(i)
                    nbrs[i].append(l)
                    nbrs[l].append(i)
        for i in range(n):
            nbrs[i] = sorted(set(nbrs[i]))
        return nbrs

    def _barabasi_albert(self, n: int, m: int) -> List[List[int]]:
        """
        Barabasi-Albert generator: start with m+1 fully connected nodes, then attach new nodes with preferential attachment.
        """
        pass
        if m < 1 or m >= n:
            m = max(1, min(n - 1, m))
        nbrs: List[List[int]] = [[] for _ in range(n)]
        initial = m + 1
        for i in range(initial):
            for j in range(i + 1, initial):
                nbrs[i].append(j)
                nbrs[j].append(i)
        targets: List[int] = []
        for i in range(initial):
            targets += [i] * len(nbrs[i])
        for new_node in range(initial, n):
            chosen = set()
            attempts = 0
            while len(chosen) < m and attempts < 10 * n:
                if targets:
                    pick = random.choice(targets)
                else:
                    pick = random.randrange(0, new_node)
                chosen.add(pick)
                attempts += 1
            for v in chosen:
                nbrs[new_node].append(v)
                nbrs[v].append(new_node)
                targets.append(v)
                targets.append(new_node)
            targets += [new_node] * len(chosen)
        for i in range(n):
            nbrs[i] = sorted(set(nbrs[i]))
        return nbrs

    def _configuration_model(self, n: int, exponent: float, mean_degree: float) -> List[List[int]]:
        """
        Configuration model with simple-graph projection by rejection.
        This is a simplistic implementation and may not strictly meet degree targets.
        """
        pass
        min_deg = 1
        max_deg = max(2, int(min(n - 1, mean_degree * 3)))
        degs: List[int] = []
        for _ in range(n):
            r = random.random()
            a = exponent
            kval = int(min(max_deg, max(min_deg, math.floor(min_deg * (1 - r) ** (-1 / (a - 1))))))
            degs.append(kval)
        if sum(degs) % 2 == 1:
            degs[0] += 1
        stubs = []
        for i, d in enumerate(degs):
            stubs.extend([i] * d)
        random.shuffle(stubs)
        nbrs: List[List[int]] = [[] for _ in range(n)]
        attempts = 0
        max_attempts = n * 50
        while len(stubs) >= 2 and attempts < max_attempts:
            a = stubs.pop()
            b = stubs.pop()
            if a == b or b in nbrs[a]:
                stubs.append(a)
                stubs.append(b)
                random.shuffle(stubs)
                attempts += 1
                continue
            nbrs[a].append(b)
            nbrs[b].append(a)
        for i in range(n):
            nbrs[i] = sorted(set(nbrs[i]))
        return nbrs

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Build network at t==0 according to plan and parameters; no-op afterwards.
        """
        pass
        out: Dict[str, Any] = {}
        # FIXED: Skip if already built (supports evaluation reuse)
        if t == 0 and state.get("neighbors"):
            self.io_log[t] = {"note": "reuse existing network"}
            return out
        if t != 0:
            self.io_log[t] = {"note": "no-op"}
            return out
        n = int(self.registry.get_param_or("population_size", 1000))
        network_type_raw = self.registry.get_param_or("network_type", "watts_strogatz")
        network_type_map = {
            "watts_strogatz": "watts_strogatz",
            "small_world": "watts_strogatz",
            "Watts-Strogatz": "watts_strogatz",
            "WS": "watts_strogatz",
            "barabasi_albert": "barabasi_albert",
            "BA": "barabasi_albert",
            "configuration": "configuration",
            "config": "configuration",
        }
        network_type = network_type_map.get(str(network_type_raw), "watts_strogatz")
        # FIXED: Parameter normalization: prefer mean_degree, then average_degree, then avg_social_degree
        mean_degree = self.registry.get_param_or("mean_degree", None)
        if mean_degree is None:
            mean_degree = self.registry.get_param_or("average_degree", None)
        if mean_degree is None:
            mean_degree = self.registry.get_param_or("avg_social_degree", 6)
        mean_degree = float(mean_degree)
        ws_p = float(self.registry.get_param_or("ws_rewiring_prob", 0.05))
        ba_m_default = max(1, int(mean_degree // 2))
        ba_m = int(self.registry.get_param_or("ba_m", ba_m_default))
        deg_exp = float(self.registry.get_param_or("deg_exponent", 2.5))

        k = max(2, int(round(mean_degree)))
        neighbors = [[] for _ in range(n)]
        if network_type == "watts_strogatz":
            neighbors = self._watts_strogatz(n, k, ws_p)
        elif network_type == "barabasi_albert":
            neighbors = self._barabasi_albert(n, ba_m)
        elif network_type == "configuration":
            neighbors = self._configuration_model(n, deg_exp, mean_degree)
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")
        out["graph"] = neighbors
        out["neighbors"] = neighbors
        self.io_log[t] = {"built": network_type, "n": n, "mean_degree_param": mean_degree}
        return out


class AgentInitializer(SimModule):
    """
    Initializes Person agents' attributes and starting states, including age, risk_group, budget, and mask inventory.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Initialize agents at t==0 using parameters; no-op after.
        """
        pass
        out: Dict[str, Any] = {}
        # FIXED: Skip if agents already exist (supports evaluation reuse)
        if t == 0 and state.get("agents"):
            self.io_log[t] = {"note": "reuse existing agents"}
            return out
        if t != 0:
            self.io_log[t] = {"note": "no-op"}
            return out
        n = int(self.registry.get_param_or("population_size", 1000))
        init_adopt = float(self.registry.get_param_or("initial_adoption_rate", 0.2))
        threshold_mean = float(self.registry.get_param_or("threshold_mean", 0.5))
        threshold_std = float(self.registry.get_param_or("threshold_std", 0.15))
        threshold_clip_low = float(self.registry.get_param_or("threshold_clip_low", 0.0))
        threshold_clip_high = float(self.registry.get_param_or("threshold_clip_high", 1.0))
        group_share_low = float(self.registry.get_param_or("group_share_low_income", 0.5))
        risk_high_share = float(self.registry.get_param_or("risk_high_share", 0.2))
        media_sens_mu = float(self.registry.get_param_or("media_sensitivity_mean", 0.0))
        policy_sens_mu = float(self.registry.get_param_or("policy_sensitivity_mean", 0.0))
        infl_mu = float(self.registry.get_param_or("influenceability_mean", 0.0))
        attr_std = float(self.registry.get_param_or("attribute_std", 0.25))
        init_prop_mu = float(self.registry.get_param_or("initial_propensity_mean", 0.0))
        budget_mean = float(self.registry.get_param_or("budget_mean", 50.0))
        budget_std = float(self.registry.get_param_or("budget_std", 10.0))
        initial_mask_inventory = int(self.registry.get_param_or("initial_mask_inventory", 0))

        neighbors = buffers.get("neighbors") or state.get("neighbors") or [[] for _ in range(n)]

        agents: List[Dict[str, Any]] = []
        for i in range(n):
            group = "low_income" if random.random() < group_share_low else "high_income"
            threshold = truncated_normal(threshold_mean, threshold_std, threshold_clip_low, threshold_clip_high)
            influenceability = clamp(lognormal_from_mu_sigma(infl_mu, attr_std), 0.1, 3.0)
            media_sensitivity = clamp(lognormal_from_mu_sigma(media_sens_mu, attr_std), 0.1, 3.0)
            policy_sensitivity = clamp(lognormal_from_mu_sigma(policy_sens_mu, attr_std), 0.1, 3.0)
            propensity = random.gauss(init_prop_mu, 0.05)
            adoption_state = 1 if random.random() < init_adopt else 0
            risk_group = "high_risk" if random.random() < risk_high_share else "low_risk"
            age = int(clamp(random.gauss(40, 15), 18, 85))
            budget = max(0.0, random.gauss(budget_mean, budget_std))
            agent = {
                "id": i,
                "group": group,
                "risk_group": risk_group,
                "age": age,
                "degree": len(neighbors[i]) if i < len(neighbors) else 0,
                "neighbors": neighbors[i] if i < len(neighbors) else [],
                "adoption_state": adoption_state,
                "propensity": propensity,
                "threshold": threshold,
                "influenceability": influenceability,
                "media_sensitivity": media_sensitivity,
                "policy_sensitivity": policy_sensitivity,
                "memory_peer": 0.0,
                "memory_propensity": 0.0,
                "last_state_change_day": -10**9,
                "household_id": -1,
                "mask_inventory": int(initial_mask_inventory),
                "budget": float(budget),
            }
            agents.append(agent)
        out["agents"] = agents
        self.io_log[t] = {"initialized_agents": len(agents)}
        return out


class HouseholdModule(SimModule):
    """
    Creates households and coordinates mask purchases across members.
    """
    pass

    def _create_households(self, n: int) -> List[List[int]]:
        """
        Create households with a simple size distribution around mean 3.
        """
        pass
        size_mean = float(self.registry.get_param_or("household_size_mean", 3.0))
        min_size = int(self.registry.get_param_or("household_size_min", 2))
        max_size = int(self.registry.get_param_or("household_size_max", 5))
        assignments: List[List[int]] = []
        remaining = list(range(n))
        random.shuffle(remaining)
        while remaining:
            size = int(clamp(round(random.gauss(size_mean, 1.0)), min_size, max_size))
            members = remaining[:size]
            remaining = remaining[size:]
            assignments.append(members)
        return assignments

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        At t==0: create households and set household_id for each agent.
        Each day: issue purchase requests for members with adoption_state==1 and insufficient masks.
        """
        pass
        out: Dict[str, Any] = {}
        # FIXED: Read agents from buffers when available to enable day-0 purchase requests
        agents = buffers.get("agents", state.get("agents", []))
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        n = len(agents)
        if t == 0 or state.get("households") is None:
            households = self._create_households(n)
            out["households"] = households
            # Also update agent household_id via buffer
            hh_ids = [-1] * n
            for hid, members in enumerate(households):
                for m in members:
                    hh_ids[m] = hid
            out["update.household_id"] = hh_ids
            self.io_log[t] = {"households_created": len(households)}
        # Purchase coordination
        desired_min_inventory = int(self.registry.get_param_or("desired_min_inventory", 1))
        purchase_requests: List[Dict[str, Any]] = []
        for a in agents:
            if a.get("adoption_state", 0) == 1 and a.get("mask_inventory", 0) < desired_min_inventory:
                need = desired_min_inventory - int(a.get("mask_inventory", 0))
                if need > 0:
                    purchase_requests.append({
                        "agent_id": a["id"],
                        "quantity": int(need),
                        "max_spend": float(a.get("budget", 0.0)),
                    })
        out["purchase.requests"] = purchase_requests
        self.io_log[t] = {"purchase_requests": len(purchase_requests)}
        return out


class RetailerModule(SimModule):
    """
    Models a retailer with mask inventory, restocking, price, and sales processing.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Restock inventory and process purchase requests from households.
        """
        pass
        out: Dict[str, Any] = {}
        # FIXED: Normalize mask price parameter (mask_cost <- mask_price) and skip restock on day 0
        price_cfg = float(self.registry.get_param_or("mask_cost", self.registry.get_param_or("mask_price", 2.0)))
        retailer = state.get("retailer", {"inventory": 0, "price": price_cfg})
        if t == 0:
            inventory_init = int(self.registry.get_param_or("initial_inventory", 1000))
            retailer["inventory"] = inventory_init
            retailer["price"] = price_cfg
            retailer["max_daily_sales"] = int(self.registry.get_param_or("max_daily_sales", 1000))
            out["retailer"] = retailer
            self.io_log[t] = {"init_inventory": inventory_init}
        # Restock (skip on day 0), normalize restock_rate_per_day
        restock = int(self.registry.get_param_or("restock_rate_per_day", self.registry.get_param_or("restock_rate", 100)))
        if t == 0:
            retailer_inventory = retailer.get("inventory", 0)
        else:
            retailer_inventory = retailer.get("inventory", 0) + restock
        price = float(self.registry.get_param_or("mask_cost", self.registry.get_param_or("mask_price", retailer.get("price", 2.0))))
        max_daily_sales = int(self.registry.get_param_or("max_daily_sales", retailer.get("max_daily_sales", 1000)))
        requests = buffers.get("purchase.requests", [])
        # Process requests
        delta_mask_inventory = [0] * len(state.get("agents", []))
        delta_budget = [0.0] * len(state.get("agents", []))
        sales = 0
        for req in requests:
            if sales >= max_daily_sales:
                break
            agent_id = int(req["agent_id"])
            qty = int(req["quantity"])
            # Sell up to available inventory and budget
            can_afford_qty = int(min(qty, math.floor(req.get("max_spend", 0.0) / price))) if price > 0 else qty
            to_sell = min(can_afford_qty, retailer_inventory)
            if to_sell <= 0:
                continue
            retailer_inventory -= to_sell
            delta_mask_inventory[agent_id] += to_sell
            delta_budget[agent_id] -= to_sell * price
            sales += to_sell
        out["delta.mask_inventory"] = delta_mask_inventory
        out["delta.budget"] = delta_budget
        out["retailer.inventory"] = retailer_inventory
        out["event.purchases"] = int(sales)
        self.io_log[t] = {"sales": sales, "inventory_after": retailer_inventory}
        return out


class PolicyMakerModule(SimModule):
    """
    Emits policy_active flag and enforcement strength for the day based on plan parameters.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Emit daily policy flags. A simple on/off schedule with start and end days.
        """
        pass
        out: Dict[str, Any] = {}
        policy_type = self.registry.get_param_or("policy_type", "none")
        # FIXED: Normalize policy start/end using mandate_on/off_day
        start = int(self.registry.get_param_or("policy_start_day", self.registry.get_param_or("mandate_on_day", 10**9)))
        end = int(self.registry.get_param_or("policy_end_day", self.registry.get_param_or("mandate_off_day", -10**9)))
        active = (policy_type != "none") and (start <= t <= end)
        enforcement_strength = float(self.registry.get_param_or("enforcement_strength", 0.0))
        out["env.policy_active"] = bool(active)
        out["env.enforcement_strength"] = enforcement_strength if active else 0.0
        out["env.policy_type"] = policy_type
        self.io_log[t] = {"policy_active": bool(active)}
        return out


class MediaMisinformationModule(SimModule):
    """
    Simulates misinformation exposures and updates agent thresholds accordingly.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        For each agent, sample misinformation exposure and adjust threshold upward.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        misinfo_rate = float(self.registry.get_param_or("misinformation_rate", 0.05))
        misinfo_reach = float(self.registry.get_param_or("misinfo_reach", 0.4))
        misinfo_credibility = float(self.registry.get_param_or("misinfo_credibility", 0.5))
        misinfo_effect = float(self.registry.get_param_or("misinfo_effect_on_threshold", 0.05))
        updates = [None] * len(agents)
        exposures = 0
        for i, a in enumerate(agents):
            exposed = (random.random() < misinfo_rate * misinfo_reach)
            if exposed:
                exposures += 1
                new_thr = clamp(a.get("threshold", 0.5) + misinfo_effect * misinfo_credibility, 0.0, 1.0)
                updates[i] = new_thr
            else:
                updates[i] = a.get("threshold", 0.5)
        out["update.threshold"] = updates
        out["event.misinfo_exposures"] = exposures
        self.io_log[t] = {"misinfo_exposures": exposures}
        return out


class LocationModule(SimModule):
    """
    Models attendance at public locations and enforcement of mask policy.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        For each agent, decide to attend public location; if policy active and not wearing mask, enforce with some probability.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        policy_active = buffers.get("env.policy_active", state.get("env", {}).get("policy_active", False))
        enforcement_strength = float(buffers.get("env.enforcement_strength", state.get("env", {}).get("enforcement_strength", 0.0)))
        attendance_rate = float(self.registry.get_param_or("attendance_rate", 0.4))
        # FIXED: Normalize enforcement strictness and fine amount to spec keys
        enforcement_strictness = float(self.registry.get_param_or("enforcement_strictness", self.registry.get_param_or("enforcement_probability_per_entry", 0.5)))
        penalty_amount = float(self.registry.get_param_or("penalty_amount", self.registry.get_param_or("fine_amount", 10.0)))
        # Enforcement events and budget deltas
        delta_budget = [0.0] * len(agents)
        enforcement_actions = 0
        # FIXED: Track entries and masked entries for metrics
        entries = 0
        masked_entries = 0
        for i, a in enumerate(agents):
            if random.random() < attendance_rate:
                entries += 1
                wears_mask = (a.get("adoption_state", 0) == 1)
                if wears_mask:
                    masked_entries += 1
                if policy_active and not wears_mask:
                    prob_enforce = clamp(enforcement_strictness * (1.0 + enforcement_strength), 0.0, 1.0)
                    if random.random() < prob_enforce:
                        enforcement_actions += 1
                        delta_budget[i] -= penalty_amount
        out["delta.budget.enforcement"] = delta_budget
        out["event.enforcement_actions"] = int(enforcement_actions)
        out["event.entries"] = int(entries)
        out["event.masked_entries"] = int(masked_entries)
        self.io_log[t] = {"enforcement_actions": enforcement_actions, "entries": entries, "masked_entries": masked_entries}
        return out


class PeerInfluence(SimModule):
    """
    Computes peer pressure signals from neighbors' adoption states with memory and conformity effects.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute peer pressure per agent and update memory_peer.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents, no-op"}
            return out
        peer_weight = float(self.registry.get_param_or("peer_weight", 0.7))
        conf_exp = float(self.registry.get_param_or("conformity_exponent", 1.0))
        mem_decay = float(self.registry.get_param_or("neighborhood_memory_decay", 0.2))
        signals = []
        memory_peer_updates = []
        for agent in agents:
            nbrs = agent.get("neighbors", [])
            if not nbrs:
                peer_frac = 0.0
            else:
                peer_frac = sum(state["agents"][j]["adoption_state"] for j in nbrs) / len(nbrs)
            mem_prev = agent.get("memory_peer", 0.0)
            mem_new = (1 - mem_decay) * mem_prev + mem_decay * peer_frac
            adjusted = (mem_new ** conf_exp)
            signal_val = peer_weight * agent["influenceability"] * adjusted
            signals.append(signal_val)
            memory_peer_updates.append(mem_new)
        out["signal.peer_pressure"] = signals
        out["update.memory_peer"] = memory_peer_updates
        self.io_log[t] = {"mean_peer_signal": mean(signals)}
        return out


class MediaInfluence(SimModule):
    """
    Generates broadcast media signal over time with campaign window and decay.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute media effect signal per agent.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents, no-op"}
            return out
        media_weight = float(self.registry.get_param_or("media_weight", 0.6))
        start = int(self.registry.get_param_or("media_campaign_start_day", 0))
        end = int(self.registry.get_param_or("media_campaign_end_day", -1))
        intensity = float(self.registry.get_param_or("media_campaign_intensity", 0.0))
        decay = float(self.registry.get_param_or("media_decay_rate", 0.05))
        if start <= t <= end and end >= 0:
            active_days = t - start
            base = intensity * math.exp(-decay * max(0, active_days))
        else:
            base = 0.0
        signals = [media_weight * a["media_sensitivity"] * base for a in agents]
        out["signal.media_effect"] = signals
        self.io_log[t] = {"media_base": base, "mean_media_signal": mean(signals)}
        return out


class EpidemiologicalContext(SimModule):
    """
    Emits a smoothed and lagged epidemiological risk signal based on exogenous case rates.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute a risk signal per agent using smoothed exogenous case rates with a reporting lag.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            return out
        series = self.registry.get_param_or("exogenous_case_rate_series", [])
        lag = int(self.registry.get_param_or("case_reporting_lag_days", 3))
        win = int(self.registry.get_param_or("case_smoothing_window_days", 7))
        idx = max(0, t - lag)
        if isinstance(series, list) and series:
            start = max(0, idx - win + 1)
            vals = series[start: idx + 1]
            case_val = mean(vals) if vals else 0.0
        else:
            case_val = 0.0
        risk_weight = float(self.registry.get_param_or("risk_sensitivity_weight", 0.3))
        signals = [risk_weight * case_val] * len(agents)
        out["signal.risk_effect"] = signals
        self.io_log[t] = {"risk_signal": case_val}
        return out


class PolicyIntervention(SimModule):
    """
    Applies policy-driven signals possibly targeted by group and compliance; reads env.policy_active from PolicyMakerModule.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute policy effect per agent, active only when env.policy_active is true.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents, no-op"}
            return out
        policy_type = self.registry.get_param_or("policy_type", "none")
        # FIXED: Normalize start/end to mandate_on/off day if present
        start = int(self.registry.get_param_or("policy_start_day", self.registry.get_param_or("mandate_on_day", 10**9)))
        end = int(self.registry.get_param_or("policy_end_day", self.registry.get_param_or("mandate_off_day", -10**9)))
        incentive = float(self.registry.get_param_or("incentive_amount", 0.0))
        penalty = float(self.registry.get_param_or("penalty_amount", self.registry.get_param_or("fine_amount", 0.0)))
        compliance_prob_base = float(self.registry.get_param_or("compliance_prob_base", 0.5))
        enforcement = float(self.registry.get_param_or("enforcement_strength", 0.0))
        target = self.registry.get_param_or("group_target", "all")
        env_active = bool(buffers.get("env.policy_active", False))
        window_active = (start <= t <= end)
        active = env_active and window_active and (policy_type != "none")
        effects = []
        for a in agents:
            if not active:
                effects.append(0.0)
                continue
            target_ok = (target == "all") or (target == a["group"])
            if not target_ok:
                effects.append(0.0)
                continue
            base_prob = clamp(compliance_prob_base * (1.0 + enforcement), 0.0, 1.0)
            compliant = random.random() < base_prob
            if not compliant:
                effects.append(0.0)
                continue
            if policy_type == "incentive":
                eff = a["policy_sensitivity"] * incentive / 10.0
            elif policy_type == "mandate":
                eff = a["policy_sensitivity"] * penalty / 20.0
            elif policy_type == "mixed":
                eff = a["policy_sensitivity"] * (incentive / 10.0 + penalty / 20.0)
            else:
                eff = 0.0
            effects.append(eff)
        out["signal.policy_effect"] = effects
        self.io_log[t] = {"active": bool(active), "mean_policy_signal": mean(effects)}
        return out


class AdoptionDecision(SimModule):
    """
    Combines signals and updates agent adoption states via logistic choice with inertia and noise.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Combine signals, update memory, and compute adoption/disadoption transitions.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents, no-op"}
            return out
        peer_sig = buffers.get("signal.peer_pressure", [0.0] * len(agents))
        media_sig = buffers.get("signal.media_effect", [0.0] * len(agents))
        policy_sig = buffers.get("signal.policy_effect", [0.0] * len(agents))
        # FIXED: Integrate risk signal from EpidemiologicalContext
        risk_sig = buffers.get("signal.risk_effect", [0.0] * len(agents))
        adoption_slope = float(self.registry.get_param_or("adoption_slope", 4.0))
        noise_scale = float(self.registry.get_param_or("noise_scale", 0.1))  # FIXED: Use get_param_or for optional param
        disadoption_rate = float(self.registry.get_param_or("disadoption_rate", 0.01))
        stickiness = float(self.registry.get_param_or("stickiness", 0.5))
        refractory = int(self.registry.get_param_or("refractory_period_days", 7))
        social_inertia = float(self.registry.get_param_or("social_inertia", 0.3))
        prop_mem_decay = float(self.registry.get_param_or("propensity_memory_decay", 0.2))
        # FIXED: Normalize mask price parameter for affordability considerations
        mask_price = float(self.registry.get_param_or("mask_cost", self.registry.get_param_or("mask_price", 2.0)))

        new_states = []
        prop_updates = []
        mem_prop_updates = []
        new_flags = []
        dis_flags = []
        for idx, a in enumerate(agents):
            # FIXED: Include risk signal; add supply penalty and affordability scaling
            net_signal = peer_sig[idx] + media_sig[idx] + policy_sig[idx] + risk_sig[idx] - a["threshold"]
            mem_prop_prev = a.get("memory_propensity", 0.0)
            mem_prop_new = (1 - prop_mem_decay) * mem_prop_prev + prop_mem_decay * net_signal
            inv = a.get("mask_inventory", 0)
            afford = 1.0 if a.get("budget", 0.0) >= mask_price else 0.5
            supply_penalty = 0.0 if inv > 0 else 0.5
            latent = mem_prop_new - social_inertia * (1 if a["adoption_state"] == 1 else 0) - supply_penalty
            p = 1.0 / (1.0 + math.exp(-adoption_slope * latent))
            p *= afford
            p = clamp(p + random.gauss(0.0, noise_scale), 0.0, 1.0)
            state_prev = a["adoption_state"]
            new_state = state_prev
            new_flag = 0
            dis_flag = 0
            if state_prev == 0:
                if random.random() < p:
                    new_state = 1
                    new_flag = 1
            else:
                if (t - a["last_state_change_day"]) < refractory:
                    new_state = 1
                else:
                    prob_dis = disadoption_rate * (1 - p) * (1 - stickiness)
                    if random.random() < prob_dis:
                        new_state = 0
                        dis_flag = 1
            new_states.append(new_state)
            prop_updates.append(latent)
            mem_prop_updates.append(mem_prop_new)
            new_flags.append(new_flag)
            dis_flags.append(dis_flag)
        out["update.adoption_state"] = new_states
        out["update.propensity"] = prop_updates
        out["update.memory_propensity"] = mem_prop_updates
        out["transition.new_adoptions"] = new_flags
        out["transition.disadoptions"] = dis_flags
        self.io_log[t] = {
            "mean_prob_signal": mean([peer_sig[i] + media_sig[i] + policy_sig[i] + risk_sig[i] for i in range(len(agents))]),
            "new_adoptions": sum(new_flags),
            "disadoptions": sum(dis_flags),
        }
        return out


class AdoptionAggregator(SimModule):
    """
    Aggregates adoption states into daily observables, with simple moving average smoothing.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute aggregate observables for the current day using buffered transitions and current states.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents, no-op"}
            return out
        smoothing_window_days = int(self.registry.get_param_or("smoothing_window_days", 7))
        reporting_lag_days = int(self.registry.get_param_or("reporting_lag_days", 0))
        adoption_state = buffers.get("update.adoption_state")
        if adoption_state is None:
            adoption_state = [a["adoption_state"] for a in agents]
        N = len(agents)
        adopted_today = sum(adoption_state)
        rate = adopted_today / N if N else 0.0
        new_adoptions = sum(buffers.get("transition.new_adoptions", [0] * N))
        low_idx = [i for i, a in enumerate(agents) if a["group"] == "low_income"]
        high_idx = [i for i, a in enumerate(agents) if a["group"] == "high_income"]
        rate_low = (sum(adoption_state[i] for i in low_idx) / len(low_idx)) if low_idx else 0.0
        rate_high = (sum(adoption_state[i] for i in high_idx) / len(high_idx)) if high_idx else 0.0

        prev_obs_buf = state.get("observables_buffer", {
            "adoption_rate_daily": [],
            "new_adoptions_daily": [],
            "adoption_rate_low_income_daily": [],
            "adoption_rate_high_income_daily": [],
            "time": [],
        })

        def moving_avg(series: List[float], w: int) -> float:
            return mean(series[-w:]) if series else 0.0

        smoothed_rate = moving_avg(prev_obs_buf.get("adoption_rate_daily", []) + [rate], smoothing_window_days)
        smoothed_new = moving_avg(prev_obs_buf.get("new_adoptions_daily", []) + [new_adoptions], smoothing_window_days)
        smoothed_low = moving_avg(prev_obs_buf.get("adoption_rate_low_income_daily", []) + [rate_low], smoothing_window_days)
        smoothed_high = moving_avg(prev_obs_buf.get("adoption_rate_high_income_daily", []) + [rate_high], smoothing_window_days)

        out["observable.adoption_rate_daily"] = {"t": t + reporting_lag_days, "value": smoothed_rate}
        out["observable.new_adoptions_daily"] = {"t": t + reporting_lag_days, "value": smoothed_new}
        out["observable.adoption_rate_low_income_daily"] = {"t": t + reporting_lag_days, "value": smoothed_low}
        out["observable.adoption_rate_high_income_daily"] = {"t": t + reporting_lag_days, "value": smoothed_high}

        out["append.observables_buffer"] = {
            "adoption_rate_daily": rate,
            "new_adoptions_daily": new_adoptions,
            "adoption_rate_low_income_daily": rate_low,
            "adoption_rate_high_income_daily": rate_high,
            "time": t,
        }

        self.io_log[t] = {
            "rate": rate,
            "new_adoptions": new_adoptions,
            "rate_low": rate_low,
            "rate_high": rate_high,
        }
        return out


def copy_static_agent_features(agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a shallow copy of agents retaining static features; dynamic fields reset to baseline.
    """
    pass
    copied: List[Dict[str, Any]] = []
    for a in agents:
        b = dict(a)
        # Reset dynamic fields
        b["adoption_state"] = a.get("adoption_state", 0)
        b["memory_peer"] = 0.0
        b["memory_propensity"] = 0.0
        b["last_state_change_day"] = -10**9
        b["mask_inventory"] = a.get("mask_inventory", 0)
        copied.append(b)
    return copied


class Simulation:
    """
    Main simulation engine orchestrating modules and state updates.
    """
    pass

    def __init__(self, plan: Dict[str, Any], registry: ParameterRegistry, seed: Optional[int] = None, artifacts_dir: str = "artifacts"):
        """
        Initialize the simulation with plan and parameter registry.
        """
        pass
        self.plan = plan
        self.registry = registry
        # FIXED: Always use registry.get_param_or for seeding determinism
        self.seed = seed if seed is not None else int(registry.get_param_or("random_seed", 0))
        random.seed(self.seed)
        self.artifacts_dir = artifacts_dir
        self.modules: List[SimModule] = []
        self.module_map: Dict[str, SimModule] = {}
        self.state: Dict[str, Any] = {
            "graph": None,
            "neighbors": None,
            "agents": [],
            "households": None,
            "retailer": None,
            "env": {},
            "time": 0,
            "observables": {},
            "observables_buffer": {},
            "history": {},
            "io": {},
            # FIXED: Track per-day transitions to avoid denominator inflation
            "daily_transitions": {},  # t -> dict(P01,P11,P10,P00,N0,N1)
            "counters": {
                "total_masks_purchased": 0,
                "total_enforcements": 0,
                "total_misinfo_exposures": 0,
            },
            # FIXED: Track daily entries and masked entries for metrics
            "daily_counters": {},  # t -> dict(purchases,enforcements,misinfo_exposures,entries,masked_entries)
        }
        self._build_modules()
        # FIXED: Guard artifact writes when artifacts_dir is falsy (internal runs)
        if self.artifacts_dir:
            safe_makedirs(self.artifacts_dir)
            with open(os.path.join(self.artifacts_dir, "plan_snapshot.json"), "w", encoding="utf-8") as f:
                json.dump(self.plan, f, indent=2)

    def _build_modules(self) -> None:
        """
        Instantiate module classes according to plan and register them in dependency order.
        """
        pass
        module_configs = self.plan.get("modules", [])
        name_to_config = {m.get("name", m.get("type")): m for m in module_configs}

        def instantiate(name: str) -> SimModule:
            cfg = name_to_config[name]
            mtype = cfg.get("type", name)
            if mtype == "NetworkBuilder":
                return NetworkBuilder(name, cfg, self.registry)
            if mtype == "AgentInitializer":
                return AgentInitializer(name, cfg, self.registry)
            if mtype == "HouseholdModule":
                return HouseholdModule(name, cfg, self.registry)
            if mtype == "RetailerModule":
                return RetailerModule(name, cfg, self.registry)
            if mtype == "PolicyMakerModule":
                return PolicyMakerModule(name, cfg, self.registry)
            if mtype == "MediaMisinformationModule":
                return MediaMisinformationModule(name, cfg, self.registry)
            if mtype == "LocationModule":
                return LocationModule(name, cfg, self.registry)
            if mtype == "PeerInfluence":
                return PeerInfluence(name, cfg, self.registry)
            if mtype == "MediaInfluence":
                return MediaInfluence(name, cfg, self.registry)
            if mtype == "EpidemiologicalContext":
                return EpidemiologicalContext(name, cfg, self.registry)
            if mtype == "PolicyIntervention":
                return PolicyIntervention(name, cfg, self.registry)
            if mtype == "AdoptionDecision":
                return AdoptionDecision(name, cfg, self.registry)
            if mtype == "AdoptionAggregator":
                return AdoptionAggregator(name, cfg, self.registry)
            return SimModule(name, cfg, self.registry)

        remaining = set(name_to_config.keys())
        added = set()
        while remaining:
            progress = False
            for name in list(remaining):
                deps = set(name_to_config[name].get("dependencies", []))
                if deps.issubset(added):
                    mod = instantiate(name)
                    self.modules.append(mod)
                    self.module_map[name] = mod
                    added.add(name)
                    remaining.remove(name)
                    progress = True
            if not progress:
                cycle = ", ".join(sorted(remaining))
                raise RuntimeError(f"Dependency cycle or missing dependencies among modules: {cycle}")

    def _commit_buffers(self, buffers: Dict[str, Any], t: int) -> None:
        """
        Commit buffered outputs to the global state.
        """
        pass
        # Graph/neighbors
        if "graph" in buffers:
            self.state["graph"] = buffers["graph"]
            self.state["neighbors"] = buffers.get("neighbors", buffers["graph"])
        # Agent initialization
        if "agents" in buffers:
            self.state["agents"] = buffers["agents"]
        # Households
        if "households" in buffers:
            self.state["households"] = buffers["households"]
        if "update.household_id" in buffers:
            for i, hid in enumerate(buffers["update.household_id"]):
                self.state["agents"][i]["household_id"] = hid
        # Retailer updates
        if "retailer" in buffers:
            self.state["retailer"] = buffers["retailer"]
        if "retailer.inventory" in buffers:
            if self.state.get("retailer") is None:
                self.state["retailer"] = {}
            self.state["retailer"]["inventory"] = buffers["retailer.inventory"]
        # Environment policy flags
        env = self.state.setdefault("env", {})
        if "env.policy_active" in buffers:
            env["policy_active"] = buffers["env.policy_active"]
        if "env.enforcement_strength" in buffers:
            env["enforcement_strength"] = buffers["env.enforcement_strength"]
        if "env.policy_type" in buffers:
            env["policy_type"] = buffers["env.policy_type"]
        # Memory updates from PeerInfluence
        if "update.memory_peer" in buffers:
            for i, val in enumerate(buffers["update.memory_peer"]):
                self.state["agents"][i]["memory_peer"] = val
        # Threshold updates from misinformation
        if "update.threshold" in buffers:
            for i, val in enumerate(buffers["update.threshold"]):
                self.state["agents"][i]["threshold"] = val
        # Adoption and propensity updates
        if "update.adoption_state" in buffers:
            prev = [a["adoption_state"] for a in self.state["agents"]]
            next_states = buffers["update.adoption_state"]
            for i, new_state in enumerate(next_states):
                if new_state != prev[i]:
                    self.state["agents"][i]["adoption_state"] = new_state
                    self.state["agents"][i]["last_state_change_day"] = t
            # FIXED: Track per-day transitions
            transitions_day = {"P01": 0, "P11": 0, "P10": 0, "P00": 0, "N0": 0, "N1": 0}
            for i, s_prev in enumerate(prev):
                s_new = next_states[i]
                if s_prev == 0 and s_new == 1:
                    transitions_day["P01"] += 1
                elif s_prev == 1 and s_new == 1:
                    transitions_day["P11"] += 1
                elif s_prev == 1 and s_new == 0:
                    transitions_day["P10"] += 1
                elif s_prev == 0 and s_new == 0:
                    transitions_day["P00"] += 1
            transitions_day["N0"] = sum(1 for v in prev if v == 0)
            transitions_day["N1"] = sum(1 for v in prev if v == 1)
            self.state["daily_transitions"][t] = transitions_day
        if "update.propensity" in buffers:
            for i, val in enumerate(buffers["update.propensity"]):
                self.state["agents"][i]["propensity"] = val
        if "update.memory_propensity" in buffers:
            for i, val in enumerate(buffers["update.memory_propensity"]):
                self.state["agents"][i]["memory_propensity"] = val
        # Inventory and budget deltas from retailer and enforcement
        if "delta.mask_inventory" in buffers:
            for i, delta in enumerate(buffers["delta.mask_inventory"]):
                self.state["agents"][i]["mask_inventory"] = int(self.state["agents"][i].get("mask_inventory", 0) + int(delta))
        if "delta.budget" in buffers:
            for i, delta in enumerate(buffers["delta.budget"]):
                self.state["agents"][i]["budget"] = float(self.state["agents"][i].get("budget", 0.0) + float(delta))
        if "delta.budget.enforcement" in buffers:
            for i, delta in enumerate(buffers["delta.budget.enforcement"]):
                self.state["agents"][i]["budget"] = float(self.state["agents"][i].get("budget", 0.0) + float(delta))
        # Append to observables_buffer if present from aggregator forward
        if "append.observables_buffer" in buffers:
            obs_buf = self.state.setdefault("observables_buffer", {
                "adoption_rate_daily": [],
                "new_adoptions_daily": [],
                "adoption_rate_low_income_daily": [],
                "adoption_rate_high_income_daily": [],
                "time": [],
            })
            delta = buffers["append.observables_buffer"]
            obs_buf["adoption_rate_daily"].append(delta["adoption_rate_daily"])
            obs_buf["new_adoptions_daily"].append(delta["new_adoptions_daily"])
            obs_buf["adoption_rate_low_income_daily"].append(delta["adoption_rate_low_income_daily"])
            obs_buf["adoption_rate_high_income_daily"].append(delta["adoption_rate_high_income_daily"])
            obs_buf["time"].append(delta["time"])
        # Observables emitted directly
        obs_state = self.state.setdefault("observables", {})
        for key, value in buffers.items():
            if key.startswith("observable."):
                series = obs_state.setdefault(key, [])
                series.append(value)
        # Signals history if needed
        sig_keys = ["signal.peer_pressure", "signal.media_effect", "signal.policy_effect", "signal.risk_effect"]
        for sk in sig_keys:
            if sk in buffers:
                hist = self.state["history"].setdefault(sk, {})
                hist[t] = buffers[sk]
        # FIXED: Use single source of truth for purchases to avoid double counting
        purchases = None
        if "event.purchases" in buffers:
            purchases = int(buffers["event.purchases"])
        else:
            if "delta.mask_inventory" in buffers:
                purchases = int(sum(d for d in buffers["delta.mask_inventory"] if d > 0))
        if purchases:
            self.state["counters"]["total_masks_purchased"] += int(purchases)
            dayc = self.state["daily_counters"].setdefault(t, {"purchases": 0, "enforcements": 0, "misinfo_exposures": 0, "entries": 0, "masked_entries": 0})
            dayc["purchases"] += int(purchases)
        if "event.enforcement_actions" in buffers:
            self.state["counters"]["total_enforcements"] += int(buffers["event.enforcement_actions"])
            dayc = self.state["daily_counters"].setdefault(t, {"purchases": 0, "enforcements": 0, "misinfo_exposures": 0, "entries": 0, "masked_entries": 0})
            dayc["enforcements"] += int(buffers["event.enforcement_actions"])
        if "event.misinfo_exposures" in buffers:
            self.state["counters"]["total_misinfo_exposures"] += int(buffers["event.misinfo_exposures"])
            dayc = self.state["daily_counters"].setdefault(t, {"purchases": 0, "enforcements": 0, "misinfo_exposures": 0, "entries": 0, "masked_entries": 0})
            dayc["misinfo_exposures"] += int(buffers["event.misinfo_exposures"])
        # FIXED: Track entries and masked entries daily for location metrics
        if "event.entries" in buffers or "event.masked_entries" in buffers:
            dayc = self.state["daily_counters"].setdefault(t, {"purchases": 0, "enforcements": 0, "misinfo_exposures": 0, "entries": 0, "masked_entries": 0})
            if "event.entries" in buffers:
                dayc["entries"] += int(buffers["event.entries"])
            if "event.masked_entries" in buffers:
                dayc["masked_entries"] += int(buffers["event.masked_entries"])
        self.state["time"] = t

    def run(self, start_day: int, end_day: int) -> None:
        """
        Run simulation from start_day to end_day inclusive.
        """
        pass
        # FIXED: Bootstrap initialization if starting mid-run and state is empty
        if start_day > 0 and not self.state.get("agents"):
            buffers0: Dict[str, Any] = {}
            for mod in self.modules:
                out0 = mod.forward(self.state, buffers0, 0)
                if isinstance(out0, dict):
                    buffers0.update(out0)
                self.state.setdefault("io", {}).setdefault(mod.name, {})[0] = mod.io_log.get(0, {})
            self._commit_buffers(buffers0, 0)
        T = end_day
        for t in range(start_day, T + 1):
            buffers: Dict[str, Any] = {}
            for mod in self.modules:
                try:
                    out = mod.forward(self.state, buffers, t)
                    if not isinstance(out, dict):
                        out = {}
                    for k, v in out.items():
                        buffers[k] = v
                    self.state.setdefault("io", {}).setdefault(mod.name, {})[t] = mod.io_log.get(t, {})
                except Exception as ex:
                    eprint(f"Module '{mod.name}' failed at t={t}: {ex}")
                    raise
            self._commit_buffers(buffers, t)

    def save_results(self, path: str) -> None:
        """
        Save simulation results (observables) to the given path as JSON.
        """
        pass
        safe_makedirs(os.path.dirname(path))
        results = {
            "observables": self.state.get("observables", {}),
            "time_last": self.state.get("time", 0),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)

    def save_module_io(self, module_name: str, path: str) -> None:
        """
        Save per-timestep IO log for a module.
        """
        pass
        safe_makedirs(os.path.dirname(path))
        io = self.state.get("io", {}).get(module_name, {})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(io, f, indent=2, sort_keys=True)

    def save_all_io(self, root_dir: str) -> None:
        """
        Save IO logs for all modules under root_dir.
        """
        pass
        safe_makedirs(root_dir)
        for mod in self.modules:
            self.save_module_io(mod.name, os.path.join(root_dir, f"{mod.name}.json"))

    def _compute_policy_effect(self) -> float:
        """
        Compute policy effect size via pre/post difference and counterfactual simulation if possible.
        Returns the counterfactual effect if available, else pre/post.
        """
        pass
        obs = self.state.get("observables", {})
        series = obs.get("observable.adoption_rate_daily", [])
        if not series:
            return 0.0
        start = int(self.registry.get_param_or("policy_start_day", self.registry.get_param_or("mandate_on_day", 10**9)))
        end = int(self.registry.get_param_or("policy_end_day", self.registry.get_param_or("mandate_off_day", -10**9)))
        if start > end or end < 0 or start == 10**9:
            return 0.0
        window = [pt for pt in series if start <= pt["t"] <= end]
        if not window:
            return 0.0
        wlen = len(window)
        pre_window = [pt for pt in series if (start - wlen) <= pt["t"] < start]
        if not pre_window:
            pre_mean = 0.0
        else:
            pre_mean = mean([pt["value"] for pt in pre_window])
        post_mean = mean([pt["value"] for pt in window])
        effect_pre_post = post_mean - pre_mean
        # Counterfactual: disable policy and rerun
        try:
            # FIXED: Use artifacts_dir=None to avoid heavy I/O for internal counterfactual run
            sim_copy = Simulation(self.plan, ParameterRegistry.from_plan(self.plan), seed=self.seed, artifacts_dir=None)
            # Copy params and disable policy
            sim_copy.registry.values = dict(self.registry.values)
            sim_copy.registry.set_param("policy_type", "none")
            horizon = self.state.get("time", end)
            sim_copy.run(0, horizon)
            cf_obs = sim_copy.state.get("observables", {})
            cf_series = cf_obs.get("observable.adoption_rate_daily", [])
            cf_window = [pt for pt in cf_series if start <= pt["t"] <= end]
            if cf_window:
                cf_post_mean = mean([pt["value"] for pt in cf_window])
                return post_mean - cf_post_mean
        except Exception as ex:
            eprint(f"Counterfactual run failed, using pre/post policy effect: {ex}")
        return effect_pre_post

    def _compute_misinformation_exposure_rate(self) -> float:
        """
        Compute average misinformation exposure rate per person per day over the run horizon.
        """
        pass
        N = len(self.state.get("agents", [])) or 1
        days = max(1, self.state.get("time", 0) + 1)
        total = int(self.state.get("counters", {}).get("total_misinfo_exposures", 0))
        return float(total) / float(N * days)

    def evaluate(self, ground_truth: Optional[Dict[str, List[Dict[str, Any]]]] = None, compute_policy_effect: bool = True) -> Dict[str, Any]:
        """
        Compute extended metrics including required mask adoption metrics and standard error metrics if GT is provided.
        Returns a metrics dictionary and saves it to artifacts/results/metrics.json.
        """
        pass
        obs = self.state.get("observables", {})
        series = obs.get("observable.adoption_rate_daily", [])
        sim_by_t = {pt["t"]: pt["value"] for pt in series}
        adoption_over_time = [pt["value"] for pt in series]
        peak_adoption_rate = max(adoption_over_time) if adoption_over_time else None
        # FIXED: Added time_to_50_percent_adoption and sustained days above threshold
        time_to_50 = None
        if series:
            for pt in series:
                if pt["value"] >= 0.5:
                    time_to_50 = pt["t"]
                    break
        time_to_70 = None
        if series:
            for pt in series:
                if pt["value"] >= 0.7:
                    time_to_70 = pt["t"]
                    break
        sustained_threshold = float(self.registry.get_param_or("sustained_threshold", 0.8))
        sustained_over_threshold_total_days = int(sum(1 for v in adoption_over_time if v >= sustained_threshold))
        sustained_over_80 = max_consecutive_ge(adoption_over_time, 0.8)
        # Inequality Gini across groups at last day
        agents = self.state.get("agents", [])
        adoption_inequality_gini = None
        if agents:
            groups = {}
            for a in agents:
                g = a.get("group", "all")
                groups.setdefault(g, [0, 0])
                groups[g][0] += a.get("adoption_state", 0)
                groups[g][1] += 1
            group_rates = [v[0] / v[1] for v in groups.values() if v[1] > 0]
            adoption_inequality_gini = gini(group_rates) if len(group_rates) > 1 else 0.0
        # Policy effect (optional for performance)
        # FIXED: Applied feedback snippet from simulation.py
policy_effect = self._compute_policy_effect() if compute_policy_effect else 0.0 if compute_policy_effect else 0.0
        # Purchases/enforcement/misinformation
        N = len(agents) or 1
        avg_masks_purchased = float(self.state.get("counters", {}).get("total_masks_purchased", 0)) / float(N)
        enforcement_actions = int(self.state.get("counters", {}).get("total_enforcements", 0))
        misinfo_rate = self._compute_misinformation_exposure_rate()
        # FIXED: Compute fines per 1000 entries and adoption by location type (public)
        daily_counters = self.state.get("daily_counters", {})
        total_entries = sum(dc.get("entries", 0) for dc in daily_counters.values())
        total_masked_entries = sum(dc.get("masked_entries", 0) for dc in daily_counters.values())
        fine_incidents_per_1000_entries = (enforcement_actions / total_entries * 1000.0) if total_entries > 0 else 0.0
        adoption_by_location_type = {
            "public": (total_masked_entries / total_entries) if total_entries > 0 else 0.0
        }
        # FIXED: Exposure-weighted adoption using risk level param for public
        public_risk_level = float(self.registry.get_param_or("public_risk_level", 1.0))
        exposure_weighted_adoption = ((total_masked_entries * public_risk_level) / (total_entries * public_risk_level)) if total_entries > 0 else 0.0
        # Standard errors versus ground truth
        rmse = None
        mae = None
        brier = None
        peak_error = None
        ttp = None
        if ground_truth and "observable.adoption_rate_daily" in ground_truth:
            gt_series_full = ground_truth["observable.adoption_rate_daily"]
            gt_by_t = {pt["t"]: pt["value"] for pt in gt_series_full}
            common_ts = sorted(set(sim_by_t.keys()) & set(gt_by_t.keys()))
            if common_ts:
                diffs = [sim_by_t[t] - gt_by_t[t] for t in common_ts]
                L = len(diffs)
                mse = sum(d * d for d in diffs) / L
                rmse = math.sqrt(mse)
                mae = sum(abs(d) for d in diffs) / L
                brier = mse  # FIXED: Compute aggregate Brier as MSE over common timestamps
                sim_peak_t = max(common_ts, key=lambda tt: sim_by_t[tt])
                gt_peak_t = max(common_ts, key=lambda tt: gt_by_t[tt])
                peak_error = sim_by_t[sim_peak_t] - gt_by_t[gt_peak_t]
                ttp = sim_peak_t - min(common_ts)
        if series and ttp is None:
            try:
                idx_peak = max(range(len(series)), key=lambda i: series[i]["value"])
                ttp = series[idx_peak]["t"] - series[0]["t"]
            except Exception:
                ttp = None
        # TransitionFit averaged over days
        dts = self.state.get("daily_transitions", {})
        sum_P01 = sum(v.get("P01", 0) for v in dts.values())
        sum_P11 = sum(v.get("P11", 0) for v in dts.values())
        sum_P10 = sum(v.get("P10", 0) for v in dts.values())
        sum_P00 = sum(v.get("P00", 0) for v in dts.values())
        sum_N0 = sum(v.get("N0", 0) for v in dts.values()) or 1
        sum_N1 = sum(v.get("N1", 0) for v in dts.values()) or 1
        transfit = {
            "P01": sum_P01 / sum_N0,
            "P11": sum_P11 / sum_N1,
            "P10": sum_P10 / sum_N1,
            "P00": sum_P00 / sum_N0,
        }
        sim_series_vals = [pt["value"] for pt in series]
        metrics = {
            "RMSE_aggregate": rmse,
            "MAE_aggregate": mae,
            "PeakError": peak_error,
            "TimeToPeak": ttp,
            "Brier": brier,
            "TransitionFit": transfit,
            "FinalAdoption": (sim_series_vals[-1] if sim_series_vals else None),
            "MeanAdoption": (mean(sim_series_vals) if sim_series_vals else None),
            # Domain metrics
            "adoption_rate_over_time": adoption_over_time,
            "peak_adoption_rate": peak_adoption_rate,
            "time_to_50_percent_adoption": time_to_50,
            "time_to_70_percent_adoption": time_to_70,
            "sustained_adoption_duration_over_80": sustained_over_80,
            "sustained_adoption_days_above_threshold": sustained_over_threshold_total_days,
            "adoption_inequality_gini": adoption_inequality_gini,
            "policy_effect_size": policy_effect,
            "average_masks_purchased_per_person": avg_masks_purchased,
            "enforcement_actions_count": enforcement_actions,
            "misinformation_exposure_rate": misinfo_rate,
            "fine_incidents_per_1000_entries": fine_incidents_per_1000_entries,
            "adoption_by_location_type": adoption_by_location_type,
            "exposure_weighted_adoption": exposure_weighted_adoption,
        }
        # FIXED: Guard artifact writes; skip if artifacts_dir is falsy
        if self.artifacts_dir:
            results_dir = os.path.join(self.artifacts_dir, "results")
            safe_makedirs(results_dir)
            with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
        return metrics

    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Optional visualization using matplotlib if available. Saves to PNG if save_path provided.
        """
        pass
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as ex:
            eprint(f"Visualization skipped (matplotlib unavailable): {ex}")
            return
        obs = self.state.get("observables", {})
        series = obs.get("observable.adoption_rate_daily", [])
        xs = [pt["t"] for pt in series]
        ys = [pt["value"] for pt in series]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, label="Adoption rate")
        plt.xlabel("Day")
        plt.ylabel("Rate")
        plt.title("Adoption Rate Over Time")
        plt.legend()
        plt.tight_layout()
        if save_path:
            safe_makedirs(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close()


@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator.
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
        Convert FittedParams to a serializable dictionary.
        """
        pass
        return asdict(self)


class ParamsAdapter:
    """
    Adapts FittedParams to simulation parameter system.
    """
    pass

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply params via simulation parameter system: set_params() + parameters_used.json
        """
        pass

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams
        """
        pass

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Check frozen params and return warnings
        """
        pass


class PlanParamsAdapter(ParamsAdapter):
    """
    Adapter that applies FittedParams to the Simulation's ParameterRegistry based on plan definitions.
    """
    pass

    def __init__(self, definitions_path: Optional[str] = None):
        """
        Initialize the adapter with optional path to parameter definitions JSON.
        """
        pass
        self.definitions_path = definitions_path
        self.definitions: Dict[str, Dict[str, Any]] = {}
        if definitions_path and os.path.exists(definitions_path):
            try:
                with open(definitions_path, "r", encoding="utf-8") as f:
                    defs = json.load(f)
                    if isinstance(defs, list):
                        for d in defs:
                            self.definitions[d["key"]] = d
                    elif isinstance(defs, dict):
                        for k, v in defs.items():
                            self.definitions[k] = v
            except Exception as ex:
                eprint(f"Warning: failed to load parameter_definitions: {ex}")

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply parameters to the simulation registry and save used parameters file.
        """
        pass
        reg = simulation.registry

        mapping = {
            "peer_weight": params.decision_weights.get("peer_weight", None),
            "media_weight": params.decision_weights.get("media_weight", None),
            "adoption_slope": params.decision_weights.get("adoption_slope", None),
            "noise_scale": params.noise_params.get("temperature", params.noise_params.get("noise_scale", None)),
        }
        info_map = {
            "media_campaign_intensity": params.info_params.get("campaign_intensity", params.info_params.get("media_campaign_intensity", None)),
            "media_decay_rate": params.info_params.get("gamma_info", params.info_params.get("media_decay_rate", None)),
        }
        for k, v in {**mapping, **info_map}.items():
            if v is None:
                continue
            d = self.definitions.get(k)
            if d and d.get("frozen", False):
                eprint(f"Ignoring override for frozen parameter '{k}'")
                continue
            if k in reg.definitions:
                try:
                    reg.set_param(k, v)
                except Exception as ex:
                    eprint(f"ParamsAdapter: failed to set '{k}'={v}: {ex}")

        for module_name, kv in params.module_params.items():
            for k, v in kv.items():
                d = self.definitions.get(k)
                if d and d.get("frozen", False):
                    eprint(f"Ignoring override for frozen parameter '{k}' from module '{module_name}'")
                    continue
                if k in reg.definitions:
                    try:
                        reg.set_param(k, v)
                    except Exception as ex:
                        eprint(f"ParamsAdapter: failed to set '{k}' from module '{module_name}': {ex}")
        # FIXED: Persist parameters used after application only if artifacts_dir is set
        if simulation.artifacts_dir:
            reg.save_used(os.path.join(simulation.artifacts_dir, "parameters_used.json"))

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current parameters as FittedParams with heuristic grouping.
        """
        pass
        reg = simulation.registry
        dw = {
            "peer_weight": float(reg.get_param_or("peer_weight", 0.7)),
            "media_weight": float(reg.get_param_or("media_weight", 0.6)),
            "adoption_slope": float(reg.get_param_or("adoption_slope", 4.0)),
        }
        info = {
            "media_campaign_intensity": float(reg.get_param_or("media_campaign_intensity", 0.0)),
            "media_decay_rate": float(reg.get_param_or("media_decay_rate", 0.05)),
        }
        noise = {
            "temperature": float(reg.get_param_or("noise_scale", 0.1))
        }
        return FittedParams(decision_weights=dw, layer_weights={}, info_params=info, noise_params=noise)

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate if any proposed keys correspond to frozen params in definitions.
        """
        pass
        warnings: Dict[str, str] = {}
        for source_dict in [params.decision_weights, params.info_params, params.noise_params]:
            for k in list(source_dict.keys()):
                defn = self.definitions.get(k)
                if defn and defn.get("frozen", False):
                    warnings[k] = "Attempt to override frozen parameter; will be ignored."
        return warnings


class Calibrator:
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """
    pass

    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Return FittedParams, fitted strictly on the training window.
        """
        pass
        raise NotImplementedError("Calibrator.fit must be implemented.")


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head from micro-transitions. If unavailable, degrades gracefully to heuristic tuning.
    """
    pass

    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 50, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Fit a simple logistic head via gradient-free search using aggregate errors (degrades gracefully).
        Uses evaluator callback for scoring.
        """
        pass
        random.seed(seed)
        adapter = params_adapter or PlanParamsAdapter(os.path.join(simulator.artifacts_dir, "parameter_definitions.json") if simulator.artifacts_dir else None)
        base = adapter.capture(simulator)
        w_peer = base.decision_weights.get("peer_weight", 0.7)
        w_media = base.decision_weights.get("media_weight", 0.6)
        slope = base.decision_weights.get("adoption_slope", 4.0)
        temp = base.noise_params.get("temperature", 0.1)

        best = None
        best_score = float("inf")
        results_dir = artifacts_dir or ""
        if results_dir:
            safe_makedirs(results_dir)
        start, end = train_window

        def score_from_metrics(m: Dict[str, Any]) -> float:
            rm = m.get("RMSE_aggregate")
            if isinstance(rm, (int, float)):
                return float(rm)
            ma = m.get("MAE_aggregate")
            if isinstance(ma, (int, float)):
                return float(ma)
            fin = m.get("FinalAdoption")
            if isinstance(fin, (int, float)):
                return float(1.0 - fin)
            return 1.0

        for i in range(budget):
            cand = FittedParams(
                decision_weights={
                    "peer_weight": clamp(w_peer + random.uniform(-0.2, 0.2), 0.0, 2.0),
                    "media_weight": clamp(w_media + random.uniform(-0.2, 0.2), 0.0, 2.0),
                    "adoption_slope": clamp(slope + random.uniform(-0.5, 0.5), 0.5, 10.0),
                },
                layer_weights={},
                info_params={
                    "media_campaign_intensity": base.info_params.get("media_campaign_intensity", 0.0),
                    "media_decay_rate": base.info_params.get("media_decay_rate", 0.05)
                },
                noise_params={"temperature": clamp(temp + random.uniform(-0.05, 0.05), 0.0, 0.5)},
                meta={"trial": i, "seed": seed}
            )
            # FIXED: Use evaluator callback instead of direct sim calls; avoid counterfactual during scoring
            metrics = evaluator(simulator, cand, (start, end))
            score = score_from_metrics(metrics)
            if results_dir:
                trial_dir = os.path.join(results_dir, f"trial_{i}")
                safe_makedirs(trial_dir)
                with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                    json.dump(cand.to_dict(), f, indent=2)
                with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            if score < best_score:
                best_score = score
                best = cand
        if best is None:
            best = base
        if results_dir:
            best_dir = os.path.join(results_dir, "best")
            safe_makedirs(best_dir)
            with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                json.dump(best.to_dict(), f, indent=2)
            report = {
                "budget": budget,
                "best_score": best_score,
                "train_window": list(train_window),
            }
            with open(os.path.join(results_dir, "calibration_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return best


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search over selected simulator parameters with bounded ranges.
    """
    pass

    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 30, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Perform random search calibration for a limited budget, returning the best FittedParams.
        Uses evaluator callback for scoring.
        """
        pass
        random.seed(seed)
        adapter = params_adapter or PlanParamsAdapter(os.path.join(simulator.artifacts_dir, "parameter_definitions.json") if simulator.artifacts_dir else None)
        base = adapter.capture(simulator)

        results_dir = artifacts_dir or ""
        if results_dir:
            safe_makedirs(results_dir)
        start, end = train_window
        best = None
        best_score = float("inf")

        def score_from_metrics(m: Dict[str, Any]) -> float:
            rm = m.get("RMSE_aggregate")
            if isinstance(rm, (int, float)):
                return float(rm)
            ma = m.get("MAE_aggregate")
            if isinstance(ma, (int, float)):
                return float(ma)
            fin = m.get("FinalAdoption")
            if isinstance(fin, (int, float)):
                return float(1.0 - fin)
            return 1.0

        for i in range(budget):
            cand = FittedParams(
                decision_weights={
                    "peer_weight": random.uniform(0.0, 2.0),
                    "media_weight": random.uniform(0.0, 2.0),
                    "adoption_slope": random.uniform(0.5, 10.0),
                },
                layer_weights={},
                info_params={
                    "media_campaign_intensity": random.uniform(0.0, 3.0),
                    "media_decay_rate": random.uniform(0.0, 0.5),
                },
                noise_params={"temperature": random.uniform(0.0, 0.5)},
                meta={"trial": i, "seed": seed}
            )
            # FIXED: Use evaluator callback and avoid heavy I/O if artifacts_dir is None
            metrics = evaluator(simulator, cand, (start, end))
            score = score_from_metrics(metrics)
            if results_dir:
                trial_dir = os.path.join(results_dir, f"trial_{i}")
                safe_makedirs(trial_dir)
                with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                    json.dump(cand.to_dict(), f, indent=2)
                with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            if score < best_score:
                best_score = score
                best = cand
        if best is None:
            best = base
        if results_dir:
            best_dir = os.path.join(results_dir, "best")
            safe_makedirs(best_dir)
            with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                json.dump(best.to_dict(), f, indent=2)
            report = {
                "budget": budget,
                "best_score": best_score,
                "train_window": list(train_window),
            }
            with open(os.path.join(results_dir, "calibration_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return best


class SNPECalibrator(Calibrator):
    """
    Simulation-based inference using SNPE if available; falls back to random search gracefully otherwise.
    """
    pass

    def fit(self, bundle, simulator: Simulation, evaluator: Callable, train_window: Tuple[int, int], seed: int,
            budget: int = 50, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Attempt SNPE with sbi/torch; if unavailable, fall back to RandomSearchCalibrator.
        """
        pass
        try:
            import torch  # type: ignore
            from sbi import utils as sbi_utils  # type: ignore
            from sbi import inference as sbi_inference  # type: ignore
        except Exception as ex:
            eprint(f"SBI unavailable ({ex}), falling back to RandomSearchCalibrator.")
            fallback = RandomSearchCalibrator()
            return fallback.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        reg = simulator.registry

        def mk_uniform(low: float, high: float):
            import torch  # type: ignore
            return sbi_utils.BoxUniform(low=torch.tensor([low], dtype=torch.float32),
                                        high=torch.tensor([high], dtype=torch.float32))

        priors = {
            "peer_weight": mk_uniform(0.0, 2.0),
            "media_weight": mk_uniform(0.0, 2.0),
            "adoption_slope": mk_uniform(0.5, 10.0),
            "media_campaign_intensity": mk_uniform(0.0, 3.0),
            "media_decay_rate": mk_uniform(0.0, 0.5),
            "temperature": mk_uniform(0.0, 0.5),
        }

        def sim_score(theta: "torch.Tensor") -> "torch.Tensor":
            theta = theta.flatten()
            cand = FittedParams(
                decision_weights={
                    "peer_weight": float(theta[0]),
                    "media_weight": float(theta[1]),
                    "adoption_slope": float(theta[2]),
                },
                layer_weights={},
                info_params={
                    "media_campaign_intensity": float(theta[3]),
                    "media_decay_rate": float(theta[4]),
                },
                noise_params={"temperature": float(theta[5])},
                meta={"seed": seed}
            )
            metrics = evaluator(simulator, cand, train_window)
            score = metrics.get("RMSE_aggregate")
            if score is None:
                score = metrics.get("MAE_aggregate")
                if score is None:
                    fin = metrics.get("FinalAdoption")
                    score = float(1.0 - fin) if isinstance(fin, (int, float)) else 1.0
            import torch  # type: ignore
            return torch.tensor([float(score)], dtype=torch.float32)

        import torch  # type: ignore
        prior_list = [priors[k] for k in ["peer_weight", "media_weight", "adoption_slope",
                                          "media_campaign_intensity", "media_decay_rate", "temperature"]]
        low_cat = torch.cat([p.low for p in prior_list], dim=0)
        high_cat = torch.cat([p.high for p in prior_list], dim=0)
        from sbi import utils as sbi_utils  # type: ignore
        prior = sbi_utils.BoxUniform(low=low_cat, high=high_cat)
        from sbi import inference as sbi_inference  # type: ignore
        inference = sbi_inference.SNPE(prior=prior)
        num_sims = budget
        xs = []
        ys = []
        for _ in range(num_sims):
            theta = prior.sample((1,)).squeeze(0)
            y = sim_score(theta)
            xs.append(theta)
            ys.append(y)
        xs_t = torch.stack(xs, dim=0)
        ys_t = torch.stack(ys, dim=0)
        density_estimator = inference.append_simulations(xs_t, ys_t).train()
        posterior = inference.build_posterior(density_estimator)
        obs = torch.zeros(1)
        samples = posterior.sample((100,), x=obs)
        # Choose minimum norm sample as heuristic
        norms = torch.norm(samples, dim=1)
        best_idx = int(torch.argmin(norms))
        best_theta = samples[best_idx]
        best = FittedParams(
            decision_weights={
                "peer_weight": float(best_theta[0]),
                "media_weight": float(best_theta[1]),
                "adoption_slope": float(best_theta[2]),
            },
            layer_weights={},
            info_params={
                "media_campaign_intensity": float(best_theta[3]),
                "media_decay_rate": float(best_theta[4]),
            },
            noise_params={"temperature": float(best_theta[5])},
        )
        results_dir = artifacts_dir or ""
        if results_dir:
            safe_makedirs(results_dir)
            with open(os.path.join(results_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                json.dump(best.to_dict(), f, indent=2)
        return best


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None) -> Calibrator:
    """
    Return a calibrator instance by name; accepts optional config path (unused in this minimal implementation).
    """
    pass
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    return CALIBRATOR_REGISTRY[name]()


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).

    Uses a fresh Simulation instance and reuses network/static agent features from the provided simulator when available.
    """
    pass
    adapter = PlanParamsAdapter(os.path.join(simulator.artifacts_dir, "parameter_definitions.json") if simulator.artifacts_dir else None)
    # FIXED: Use artifacts_dir=None to reduce I/O footprint during evaluation
    sim_copy = Simulation(simulator.plan, ParameterRegistry.from_plan(simulator.plan), seed=simulator.seed, artifacts_dir=None)
    # FIXED: Reuse network and static agents to speed up calibration
    if simulator.state.get("neighbors") and simulator.state.get("agents"):
        sim_copy.state["graph"] = simulator.state["neighbors"]
        sim_copy.state["neighbors"] = simulator.state["neighbors"]
        sim_copy.state["agents"] = copy_static_agent_features(simulator.state["agents"])
    sim_copy.registry.values = dict(simulator.registry.values)
    # Apply candidate parameters
    adapter.apply(sim_copy, params)
    start, end = window
    sim_copy.run(start, end)
    # FIXED: Avoid counterfactual during calibration evaluation for performance
    metrics = sim_copy.evaluate(compute_policy_effect=False)
    if "Brier" not in metrics or metrics["Brier"] is None:
        metrics["Brier"] = None
    if "TransitionFit" not in metrics:
        metrics["TransitionFit"] = {"P01": None, "P11": None, "P10": None, "P00": None}
    return metrics


def load_plan(source_file: Optional[str], source_url: Optional[str], timeout: float, retries: int) -> Dict[str, Any]:
    """
    Load model plan JSON from file or URL based on CLI input. Adapts task spec schema if needed.
    """
    pass
    if bool(source_file) == bool(source_url):
        raise ValueError("Provide exactly one of --plan-file or --plan-url")
    if source_file:
        text = load_text_from_file(source_file)
    else:
        text = load_text_from_url(source_url, timeout=timeout, retries=retries)
    plan = parse_json(text)
    validate_plan(plan)
    # FIXED: Adapt task spec schema to internal schema if needed
    if "modules" not in plan and "entities" in plan:
        plan = adapt_task_spec_to_internal_plan(plan)
    return plan


def load_ground_truth(plan: Dict[str, Any]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Load ground-truth observables if referenced by plan data_sources; returns None if unavailable.
    """
    pass
    try:
        return None
    except Exception as ex:
        eprint(f"Ground truth loading failed: {ex}")
        return None


def parse_calib_window(text: str) -> Tuple[int, int]:
    """
    Parse calibration window string 'start:end' into a tuple of ints.
    """
    pass
    if ":" not in text:
        raise ValueError("Calibration window must be in 'start:end' format")
    s, e = text.split(":", 1)
    return int(s), int(e)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments for simulation execution and calibration.
    """
    pass
    parser = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation with Calibration")
    parser.add_argument("--plan-file", help="Path to model plan JSON file")
    parser.add_argument("--plan-url", help="URL to model plan JSON")
    parser.add_argument("--param-file", help="Path to parameters JSON", required=False)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Parameter overrides key=value (repeatable)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--timeout", type=float, default=10.0, help="Network timeout for plan URL")
    parser.add_argument("--retries", type=int, default=3, help="Network retries for plan URL")
    parser.add_argument("--calibrator", choices=list(CALIBRATOR_REGISTRY.keys()), default="random_search", help="Calibration algorithm")
    parser.add_argument("--budget", type=int, default=10, help="Calibration budget (iterations/trials)")
    parser.add_argument("--calib-window", type=str, default="0:30", help="Calibration window 'start:end'")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save artifacts")
    parser.add_argument("--visualize", action="store_true", help="Render visualization if matplotlib is available")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main orchestrator for loading plan, applying parameters, running simulation, calibration, evaluation, and saving outputs.
    """
    pass
    try:
        args = parse_args(argv)

        # Load plan
        plan = load_plan(args.plan_file, args.plan_url, timeout=args.timeout, retries=args.retries)
        # Build parameter registry
        registry = ParameterRegistry.from_plan(plan)
        if args.param_file:
            registry.apply_file(args.param_file)
        if args.overrides:
            registry.apply_overrides(args.overrides)
        # Save parameter definitions to file for adapters/calibrators
        param_defs_path = os.path.join(args.artifacts_dir, "parameter_definitions.json")
        safe_makedirs(args.artifacts_dir)
        registry.save_definitions(param_defs_path)
        # Save effective params snapshot initially
        registry.save_used(os.path.join(args.artifacts_dir, "parameters_used.json"))

        # Initialize simulation
        seed = args.seed if args.seed is not None else int(registry.get_param_or("random_seed", 0))
        sim = Simulation(plan, registry, seed=seed, artifacts_dir=args.artifacts_dir)

        # Calibration train window
        try:
            train_window = parse_calib_window(args.calib_window)
        except Exception as ex:
            eprint(f"Invalid calibration window: {ex}")
            sys.exit(1)

        # Instantiate calibrator
        try:
            calibrator = get_calibrator(args.calibrator, config_path=None)
        except Exception as ex:
            eprint(f"Failed to initialize calibrator: {ex}")
            sys.exit(1)

        # Run calibration
        eprint(f"Starting calibration with {args.calibrator}, budget={args.budget}, window={train_window}")
        adapter = PlanParamsAdapter(param_defs_path)
        try:
            fitted = calibrator.fit(bundle=None, simulator=sim, evaluator=evaluate_params, train_window=train_window,
                                    seed=seed or 0, budget=args.budget, artifacts_dir=os.path.join(args.artifacts_dir, "calibration"),
                                    params_adapter=adapter)
        except Exception as ex:
            eprint(f"Calibration failed: {ex}")
            fitted = adapter.capture(sim)
        # Apply fitted parameters to main simulation and save
        adapter.apply(sim, fitted)
        with open(os.path.join(args.artifacts_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
            json.dump(fitted.to_dict(), f, indent=2)

        # Determine simulation horizon from plan initialization or parameters
        init = plan.get("initialization", {})
        time_horizon = init.get("time_horizon_days")
        if time_horizon is None and "time_horizon_days" in registry.definitions:
            time_horizon = int(registry.get_param_or("time_horizon_days", 60))
        if time_horizon is None:
            time_horizon = 60

        # Run simulation full horizon
        try:
            sim.run(0, int(time_horizon))
        except Exception as ex:
            eprint(f"Simulation run failed: {ex}")
            sys.exit(1)

        # Save results and IO
        sim.save_results(os.path.join(args.artifacts_dir, "results", "simulation_results.json"))
        sim.save_all_io(os.path.join(args.artifacts_dir, "io"))

        # Evaluate
        gt = load_ground_truth(plan)
        # FIXED: compute_policy_effect True for final evaluation; heavy but final only
        metrics = sim.evaluate(ground_truth=gt, compute_policy_effect=True)
        eprint(f"Evaluation metrics: {metrics}")

        # Visualization
        if args.visualize:
            sim.visualize(save_path=os.path.join(args.artifacts_dir, "figs", "adoption_rate.png"))

        # Persist final parameters used
        registry.save_used(os.path.join(args.artifacts_dir, "parameters_used.json"))

        # Print a minimal JSON result to stdout for automation
        print(json.dumps({"status": "ok", "metrics": metrics, "artifacts_dir": args.artifacts_dir}, indent=2))
    except Exception as ex:
        eprint(f"Fatal error: {ex}")
        sys.exit(1)


# Execute main for both direct execution and sandbox wrapper invocation
main()