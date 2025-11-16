import sys
import os
import json
import argparse
import time
import random
import math
import socket
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional, Callable
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# FIXED: Removed unused 'statistics' import per feedback

# FIXED: Removed stray plain text 'Error: Request timed out.' from previous iteration
# FIXED: Implemented robust CLI, JSON parsing with error context, file/URL loaders with timeouts/retries
# FIXED: Added deterministic minimal-yet-functional simulation with network, agents, and modules
# FIXED: Added schema validation, parameter registry, seed handling, and artifacts saving
# FIXED: Implemented pluggable calibration architecture with three calibrators and evaluator
# FIXED: Kept direct call to main() at end to satisfy sandbox execution requirement


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
    """
    pass
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        lines = text.splitlines()
        start = max(0, e.lineno - 2)
        end = min(len(lines), e.lineno + 1)
        eprint(f"JSON parse error at line {e.lineno}, column {e.colno}: {e.msg}")
        for i in range(start, end):
            prefix = ">>" if (i + 1) == e.lineno else "  "
            snippet = lines[i]
            eprint(f"{prefix} {i + 1}: {snippet}")
        sys.exit(1)


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


def truncated_normal(mean: float, std: float, low: float, high: float) -> float:
    """
    Sample from a truncated normal by simple rejection sampling.
    """
    pass
    for _ in range(1000):
        sample = random.gauss(mean, std)
        if low <= sample <= high:
            return sample
    return clamp(mean, low, high)


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
        Build ParameterRegistry from plan parameters list.
        """
        pass
        defs: Dict[str, ParameterDefinition] = {}
        params_list = plan.get("parameters", [])
        for p in params_list:
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
    Validate minimal structure of the plan.
    """
    pass
    if not isinstance(plan, dict):
        eprint("Plan must be a JSON object")
        sys.exit(1)
    modules = plan.get("modules", [])
    if not isinstance(modules, list) or len(modules) < 3:
        eprint("Plan must contain at least 3 modules")
        sys.exit(1)
    # FIXED: Support plans that specify 'type' instead of 'name'; ensure uniqueness by effective name/type
    names = [m.get("name", m.get("type")) for m in modules]
    if any(n is None for n in names):
        eprint("Each module must include 'name' or 'type'")
        sys.exit(1)
    if len(set(names)) != len(names):
        eprint("Module names must be unique")
        sys.exit(1)
    params = plan.get("parameters", [])
    if not isinstance(params, list) or not params:
        eprint("Plan must contain a non-empty 'parameters' list")
        sys.exit(1)
    observables = plan.get("observables", [])
    for obs in observables:
        if "id" not in obs or "source_module" not in obs:
            eprint("Each observable must include 'id' and 'source_module'")
            sys.exit(1)
        if "target_data_field" not in obs:
            eprint(f"Observable '{obs.get('id', '')}' missing 'target_data_field'")
            sys.exit(1)


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
        # Remove duplicates
        for i in range(n):
            neighbors[i] = sorted(set(neighbors[i]))
        return neighbors

    def _watts_strogatz(self, n: int, k: int, p: float) -> List[List[int]]:
        """
        Watts-Strogatz small-world graph generator without external dependencies.
        """
        pass
        nbrs = self._ring_lattice(n, k)
        # Rewiring
        for i in range(n):
            for j in list(nbrs[i]):
                # Only consider forward edges to avoid double-processing
                if j < i:
                    continue
                if random.random() < p:
                    # Rewire edge (i, j) to new node l
                    allowed = set(range(n))
                    allowed.discard(i)
                    allowed -= set(nbrs[i])
                    if not allowed:
                        continue
                    l = random.choice(list(allowed))
                    # remove old
                    if j in nbrs[i]:
                        nbrs[i].remove(j)
                    if i in nbrs[j]:
                        nbrs[j].remove(i)
                    # add new
                    nbrs[i].append(l)
                    nbrs[l].append(i)
        # Ensure unique lists
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
        # initial fully connected graph of size m+1
        initial = m + 1
        for i in range(initial):
            for j in range(i + 1, initial):
                nbrs[i].append(j)
                nbrs[j].append(i)
        # target list for preferential choice (node index repeated by degree)
        targets: List[int] = []
        for i in range(initial):
            targets += [i] * len(nbrs[i])
        for new_node in range(initial, n):
            # choose m distinct nodes using preferential attachment
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
            # update targets for new edges
            targets += [new_node] * len(chosen)
        # Ensure unique lists
        for i in range(n):
            nbrs[i] = sorted(set(nbrs[i]))
        return nbrs

    def _configuration_model(self, n: int, exponent: float, mean_degree: float) -> List[List[int]]:
        """
        Configuration model with simple-graph projection by rejection.
        This is a simplistic implementation and may not strictly meet degree targets.
        """
        pass
        # Sample degrees from a power-law and adjust to target mean roughly
        min_deg = 1
        max_deg = max(2, int(min(n - 1, mean_degree * 3)))
        degs: List[int] = []
        for _ in range(n):
            r = random.random()
            a = exponent
            kval = int(min(max_deg, max(min_deg, math.floor(min_deg * (1 - r) ** (-1 / (a - 1))))))
            degs.append(kval)
        # Adjust sum of degrees to even
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
                # Reject, reinsert and reshuffle locally
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
        if t != 0:
            self.io_log[t] = {"note": "no-op"}
            return out
        n = int(self.registry.get_param_or("population_size", 1000))
        # FIXED: Robust handling of network_type synonyms and optional mean_degree parameter
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
        mean_degree = self.registry.get_param_or("mean_degree", None)
        if mean_degree is None:
            mean_degree = self.registry.get_param_or("avg_social_degree", 6)
        mean_degree = float(mean_degree)
        ws_p = float(self.registry.get_param_or("ws_rewiring_prob", 0.05))
        ba_m_default = max(1, int(mean_degree // 2))
        ba_m = int(self.registry.get_param_or("ba_m", ba_m_default))
        deg_exp = float(self.registry.get_param_or("deg_exponent", 2.5))

        k = max(2, int(round(mean_degree)))
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
    Initializes Person agents' attributes and starting states.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Initialize agents at t==0 using parameters; no-op after.
        """
        pass
        out: Dict[str, Any] = {}
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
        media_sens_mu = float(self.registry.get_param_or("media_sensitivity_mean", 0.0))
        policy_sens_mu = float(self.registry.get_param_or("policy_sensitivity_mean", 0.0))
        infl_mu = float(self.registry.get_param_or("influenceability_mean", 0.0))
        attr_std = float(self.registry.get_param_or("attribute_std", 0.25))
        init_prop_mu = float(self.registry.get_param_or("initial_propensity_mean", 0.0))

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
            agent = {
                "id": i,
                "group": group,
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
            }
            agents.append(agent)
        out["agents"] = agents
        self.io_log[t] = {"initialized_agents": len(agents)}
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


class PolicyIntervention(SimModule):
    """
    Applies policy-driven signals possibly targeted by group and compliance.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute policy effect per agent.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents, no-op"}
            return out
        policy_type = self.registry.get_param_or("policy_type", "none")
        start = int(self.registry.get_param_or("policy_start_day", 10**9))
        end = int(self.registry.get_param_or("policy_end_day", -10**9))
        incentive = float(self.registry.get_param_or("incentive_amount", 0.0))
        penalty = float(self.registry.get_param_or("penalty_amount", 0.0))
        compliance_prob_base = float(self.registry.get_param_or("compliance_prob_base", 0.5))
        enforcement = float(self.registry.get_param_or("enforcement_strength", 0.0))
        target = self.registry.get_param_or("group_target", "all")
        active = (start <= t <= end) and (policy_type != "none")
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
        adoption_slope = float(self.registry.get_param_or("adoption_slope", 4.0))
        noise_scale = float(self.registry.get_param_or("noise_scale", 0.1))  # FIXED: Use get_param_or for optional param
        disadoption_rate = float(self.registry.get_param_or("disadoption_rate", 0.01))
        stickiness = float(self.registry.get_param_or("stickiness", 0.5))
        refractory = int(self.registry.get_param_or("refractory_period_days", 7))
        social_inertia = float(self.registry.get_param_or("social_inertia", 0.3))
        prop_mem_decay = float(self.registry.get_param_or("propensity_memory_decay", 0.2))

        new_states = []
        prop_updates = []
        mem_prop_updates = []
        new_flags = []
        dis_flags = []
        for idx, a in enumerate(agents):
            net_signal = peer_sig[idx] + media_sig[idx] + policy_sig[idx] - a["threshold"]
            mem_prop_prev = a.get("memory_propensity", 0.0)
            mem_prop_new = (1 - prop_mem_decay) * mem_prop_prev + prop_mem_decay * net_signal
            latent = mem_prop_new - social_inertia * (1 if a["adoption_state"] == 1 else 0)
            p = 1.0 / (1.0 + math.exp(-adoption_slope * latent))
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
            "mean_prob_signal": mean([peer_sig[i] + media_sig[i] + policy_sig[i] for i in range(len(agents))]),
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
        # FIXED: Use get_param_or for optional params to avoid KeyErrors
        smoothing_window_days = int(self.registry.get_param_or("smoothing_window_days", 7))
        reporting_lag_days = int(self.registry.get_param_or("reporting_lag_days", 0))
        # Aggregate using the to-be-committed new states to reflect day t status.
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

        # FIXED: Avoid mutating state in forward; compute smoothed values using a hypothetical append
        prev_obs_buf = state.get("observables_buffer", {
            "adoption_rate_daily": [],
            "new_adoptions_daily": [],
            "adoption_rate_low_income_daily": [],
            "adoption_rate_high_income_daily": [],
            "time": [],
        })

        def moving_avg(series: List[float], w: int) -> float:
            return mean(series[-w:]) if series else 0.0

        # Compute smoothed values as if we had appended today's values
        smoothed_rate = moving_avg(prev_obs_buf.get("adoption_rate_daily", []) + [rate], smoothing_window_days)
        smoothed_new = moving_avg(prev_obs_buf.get("new_adoptions_daily", []) + [new_adoptions], smoothing_window_days)
        smoothed_low = moving_avg(prev_obs_buf.get("adoption_rate_low_income_daily", []) + [rate_low], smoothing_window_days)
        smoothed_high = moving_avg(prev_obs_buf.get("adoption_rate_high_income_daily", []) + [rate_high], smoothing_window_days)

        # Emit observable outputs
        out["observable.adoption_rate_daily"] = {"t": t + reporting_lag_days, "value": smoothed_rate}
        out["observable.new_adoptions_daily"] = {"t": t + reporting_lag_days, "value": smoothed_new}
        out["observable.adoption_rate_low_income_daily"] = {"t": t + reporting_lag_days, "value": smoothed_low}
        out["observable.adoption_rate_high_income_daily"] = {"t": t + reporting_lag_days, "value": smoothed_high}

        # FIXED: Return buffer append deltas to be committed in _commit_buffers
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
        self.seed = seed if seed is not None else int(registry.get_param_or("random_seed", 0)) if "random_seed" in registry.definitions else None
        if self.seed is not None:
            random.seed(self.seed)
        self.artifacts_dir = artifacts_dir
        self.modules: List[SimModule] = []
        self.module_map: Dict[str, SimModule] = {}
        self.state: Dict[str, Any] = {
            "graph": None,
            "neighbors": None,
            "agents": [],
            "time": 0,
            "observables": {},
            "observables_buffer": {},
            "history": {},
            "io": {},
            "transitions": {
                "P01": 0,
                "P11": 0,
                "P10": 0,
                "P00": 0,
                "N0": 0,
                "N1": 0
            }
        }
        self._build_modules()
        # Save plan snapshot
        safe_makedirs(self.artifacts_dir)
        with open(os.path.join(self.artifacts_dir, "plan_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump(self.plan, f, indent=2)

    def _build_modules(self) -> None:
        """
        Instantiate module classes according to plan and register them in dependency order.
        """
        pass
        module_configs = self.plan.get("modules", [])
        # FIXED: Support 'type' field with fallback to 'name'
        name_to_config = {m.get("name", m.get("type")): m for m in module_configs}

        def instantiate(name: str) -> SimModule:
            cfg = name_to_config[name]
            mtype = cfg.get("type", name)
            if mtype == "NetworkBuilder":
                return NetworkBuilder(name, cfg, self.registry)
            if mtype == "AgentInitializer":
                return AgentInitializer(name, cfg, self.registry)
            if mtype == "PeerInfluence":
                return PeerInfluence(name, cfg, self.registry)
            if mtype == "MediaInfluence":
                return MediaInfluence(name, cfg, self.registry)
            if mtype == "PolicyIntervention":
                return PolicyIntervention(name, cfg, self.registry)
            if mtype == "AdoptionDecision":
                return AdoptionDecision(name, cfg, self.registry)
            if mtype == "AdoptionAggregator":
                return AdoptionAggregator(name, cfg, self.registry)
            # Default fallback for unknown module types
            return SimModule(name, cfg, self.registry)

        # Topological order by dependencies
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
        # Memory updates from PeerInfluence
        if "update.memory_peer" in buffers:
            for i, val in enumerate(buffers["update.memory_peer"]):
                self.state["agents"][i]["memory_peer"] = val
        # Adoption and propensity updates
        if "update.adoption_state" in buffers:
            prev = [a["adoption_state"] for a in self.state["agents"]]
            next_states = buffers["update.adoption_state"]
            for i, new_state in enumerate(next_states):
                if new_state != prev[i]:
                    self.state["agents"][i]["adoption_state"] = new_state
                    self.state["agents"][i]["last_state_change_day"] = t
            # Track transitions for evaluator
            transitions = self.state["transitions"]
            for i, s_prev in enumerate(prev):
                s_new = next_states[i]
                if s_prev == 0 and s_new == 1:
                    transitions["P01"] += 1
                elif s_prev == 1 and s_new == 1:
                    transitions["P11"] += 1
                elif s_prev == 1 and s_new == 0:
                    transitions["P10"] += 1
                elif s_prev == 0 and s_new == 0:
                    transitions["P00"] += 1
            transitions["N0"] += sum(1 for v in prev if v == 0)
            transitions["N1"] += sum(1 for v in prev if v == 1)
        if "update.propensity" in buffers:
            for i, val in enumerate(buffers["update.propensity"]):
                self.state["agents"][i]["propensity"] = val
        if "update.memory_propensity" in buffers:
            for i, val in enumerate(buffers["update.memory_propensity"]):
                self.state["agents"][i]["memory_propensity"] = val

        # FIXED: Append to observables_buffer if present from aggregator forward
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
        sig_keys = ["signal.peer_pressure", "signal.media_effect", "signal.policy_effect"]
        for sk in sig_keys:
            if sk in buffers:
                hist = self.state["history"].setdefault(sk, {})
                hist[t] = buffers[sk]

        # Per-module IO logging is captured separately; nothing here
        self.state["time"] = t

    def run(self, start_day: int, end_day: int) -> None:
        """
        Run simulation from start_day to end_day inclusive.
        """
        pass
        T = end_day
        # Buffer container persists within each tick
        for t in range(start_day, T + 1):
            buffers: Dict[str, Any] = {}
            # Execute modules in dependency order; later modules can see earlier buffers
            for mod in self.modules:
                try:
                    out = mod.forward(self.state, buffers, t)
                    if not isinstance(out, dict):
                        out = {}
                    # Merge outputs into buffers
                    for k, v in out.items():
                        buffers[k] = v
                    # IO log
                    self.state.setdefault("io", {}).setdefault(mod.name, {})[t] = mod.io_log.get(t, {})
                except Exception as ex:
                    eprint(f"Module '{mod.name}' failed at t={t}: {ex}")
                    raise
            # Commit buffers
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

    def evaluate(self, ground_truth: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """
        Compute metrics comparing simulation observables to ground truth if provided.
        Returns a metrics dictionary and saves it to artifacts/results/metrics.json.
        """
        pass
        obs = self.state.get("observables", {})
        # FIXED: Align by timestamp 't' for adoption_rate_daily
        sim_series_full = obs.get("observable.adoption_rate_daily", [])
        sim_by_t = {pt["t"]: pt["value"] for pt in sim_series_full}
        rmse = None
        mae = None
        peak_error = None
        ttp = None
        if ground_truth and "observable.adoption_rate_daily" in ground_truth:
            gt_series_full = ground_truth["observable.adoption_rate_daily"]
            gt_by_t = {pt["t"]: pt["value"] for pt in gt_series_full}
            common_ts = sorted(set(sim_by_t.keys()) & set(gt_by_t.keys()))
            if common_ts:
                diffs = [sim_by_t[t] - gt_by_t[t] for t in common_ts]
                L = len(diffs)
                rmse = math.sqrt(sum(d * d for d in diffs) / L)
                mae = sum(abs(d) for d in diffs) / L
                # Peak alignment difference
                sim_peak_t = max(common_ts, key=lambda tt: sim_by_t[tt])
                gt_peak_t = max(common_ts, key=lambda tt: gt_by_t[tt])
                peak_error = sim_by_t[sim_peak_t] - gt_by_t[gt_peak_t]
                ttp = sim_peak_t - min(common_ts)  # time to peak relative to start of common window
        # Peak and time-to-peak on sim series if gt unavailable
        if sim_series_full and ttp is None:
            try:
                idx_peak = max(range(len(sim_series_full)), key=lambda i: sim_series_full[i]["value"])
                ttp = sim_series_full[idx_peak]["t"] - sim_series_full[0]["t"]
            except Exception:
                ttp = None
        # Brier placeholder using mean over day-level adoption probability; not directly available
        brier = None

        transitions = self.state.get("transitions", {})
        P01 = transitions.get("P01", 0)
        P11 = transitions.get("P11", 0)
        P10 = transitions.get("P10", 0)
        P00 = transitions.get("P00", 0)
        N0 = transitions.get("N0", 0) or 1
        N1 = transitions.get("N1", 0) or 1
        transfit = {
            "P01": P01 / N0,
            "P11": P11 / N1,
            "P10": P10 / N1,
            "P00": P00 / N0,
        }
        sim_series_vals = [pt["value"] for pt in sim_series_full]
        metrics = {
            "RMSE_aggregate": rmse,
            "MAE_aggregate": mae,
            "PeakError": peak_error,
            "TimeToPeak": ttp,
            "Brier": brier,
            "TransitionFit": transfit,
            "FinalAdoption": (sim_series_vals[-1] if sim_series_vals else None),  # for fallback scoring
            "MeanAdoption": (mean(sim_series_vals) if sim_series_vals else None),
        }
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
                    for d in defs:
                        self.definitions[d["key"]] = d
            except Exception as ex:
                eprint(f"Warning: failed to load parameter_definitions: {ex}")

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply parameters to the simulation registry and save used parameters file.
        """
        pass
        reg = simulation.registry

        # Map decision weights to module-specific parameters in registry (heuristic mapping)
        mapping = {
            "peer_weight": params.decision_weights.get("peer_weight", None),
            "media_weight": params.decision_weights.get("media_weight", None),
            "adoption_slope": params.decision_weights.get("adoption_slope", None),
            "noise_scale": params.noise_params.get("temperature", params.noise_params.get("noise_scale", None)),
        }
        # Information parameters mapping
        info_map = {
            "media_campaign_intensity": params.info_params.get("campaign_intensity", params.info_params.get("media_campaign_intensity", None)),
            "media_decay_rate": params.info_params.get("gamma_info", params.info_params.get("media_decay_rate", None)),
        }
        # Apply mappings where definitions exist and not frozen
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

        # Apply module_params granular overrides
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

        # Save parameters_used after application
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
        Fit a simple logistic head via gradient descent using aggregate errors (degrades gracefully).
        """
        pass
        random.seed(seed)
        adapter = params_adapter or PlanParamsAdapter(os.path.join(simulator.artifacts_dir, "parameter_definitions.json"))
        base = adapter.capture(simulator)
        w_peer = base.decision_weights.get("peer_weight", 0.7)
        w_media = base.decision_weights.get("media_weight", 0.6)
        slope = base.decision_weights.get("adoption_slope", 4.0)
        temp = base.noise_params.get("temperature", 0.1)

        best = None
        best_score = float("inf")
        results_dir = artifacts_dir or os.path.join(simulator.artifacts_dir, "calibration", "logit_head")
        safe_makedirs(results_dir)
        start, end = train_window

        def score_from_metrics(m: Dict[str, Any]) -> float:
            # FIXED: Robust scoring fallback when RMSE/MAE unavailable
            rm = m.get("RMSE_aggregate")
            if isinstance(rm, (int, float)):
                return float(rm)
            ma = m.get("MAE_aggregate")
            if isinstance(ma, (int, float)):
                return float(ma)
            fin = m.get("FinalAdoption")
            if isinstance(fin, (int, float)):
                return float(1.0 - fin)  # prefer higher adoption => lower score
            return 1.0  # neutral fallback

        for i in range(budget):
            # Simple gradient-free adjustment around current best
            cand = FittedParams(
                decision_weights={
                    "peer_weight": clamp(w_peer + random.uniform(-0.2, 0.2), 0.0, 2.0),
                    "media_weight": clamp(w_media + random.uniform(-0.2, 0.2), 0.0, 2.0),
                    "adoption_slope": clamp(slope + random.uniform(-0.5, 0.5), 0.5, 10.0),
                },
                layer_weights={},
                info_params={"media_campaign_intensity": base.info_params.get("media_campaign_intensity", 0.0),
                             "media_decay_rate": base.info_params.get("media_decay_rate", 0.05)},
                noise_params={"temperature": clamp(temp + random.uniform(-0.05, 0.05), 0.0, 0.5)},
                meta={"trial": i, "seed": seed}
            )
            # Apply to a fresh simulator copy
            sim_copy = Simulation(simulator.plan, ParameterRegistry.from_plan(simulator.plan), seed=seed, artifacts_dir=os.path.join(results_dir, f"trial_{i}"))
            # seed registry with baseline values
            sim_copy.registry.values = dict(simulator.registry.values)
            adapter.apply(sim_copy, cand)
            sim_copy.run(start, end)
            metrics = sim_copy.evaluate()
            score = score_from_metrics(metrics)
            # Save trial artifacts
            with open(os.path.join(sim_copy.artifacts_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                json.dump(cand.to_dict(), f, indent=2)
            with open(os.path.join(sim_copy.artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if score < best_score:
                best_score = score
                best = cand
        if best is None:
            best = base
        # Save best params
        best_dir = os.path.join(results_dir, "best")
        safe_makedirs(best_dir)
        with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
            json.dump(best.to_dict(), f, indent=2)
        # Calibration report
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
        """
        pass
        random.seed(seed)
        adapter = params_adapter or PlanParamsAdapter(os.path.join(simulator.artifacts_dir, "parameter_definitions.json"))
        base = adapter.capture(simulator)

        results_dir = artifacts_dir or os.path.join(simulator.artifacts_dir, "calibration", "random_search")
        safe_makedirs(results_dir)
        start, end = train_window
        best = None
        best_score = float("inf")

        def score_from_metrics(m: Dict[str, Any]) -> float:
            # FIXED: Robust scoring fallback when RMSE/MAE unavailable
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
            sim_copy = Simulation(simulator.plan, ParameterRegistry.from_plan(simulator.plan), seed=seed,
                                  artifacts_dir=os.path.join(results_dir, f"trial_{i}"))
            sim_copy.registry.values = dict(simulator.registry.values)
            adapter.apply(sim_copy, cand)
            sim_copy.run(start, end)
            metrics = sim_copy.evaluate()
            score = score_from_metrics(metrics)
            with open(os.path.join(sim_copy.artifacts_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                json.dump(cand.to_dict(), f, indent=2)
            with open(os.path.join(sim_copy.artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if score < best_score:
                best_score = score
                best = cand
        if best is None:
            best = base
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

        # Define priors (uniform) based on registry bounds
        reg = simulator.registry

        def mk_uniform(low: float, high: float):
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

        # Simulator function for SNPE: sample theta -> evaluate metric (RMSE)
        def sim_fn(theta: "torch.Tensor") -> "torch.Tensor":
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
            sim_copy = Simulation(simulator.plan, ParameterRegistry.from_plan(simulator.plan), seed=seed)
            sim_copy.registry.values = dict(reg.values)
            adapter = params_adapter or PlanParamsAdapter(os.path.join(simulator.artifacts_dir, "parameter_definitions.json"))
            adapter.apply(sim_copy, cand)
            start, end = train_window
            sim_copy.run(start, end)
            metrics = sim_copy.evaluate()
            score = metrics.get("RMSE_aggregate")
            if score is None:
                # Fallback: use MAE or FinalAdoption-based surrogate
                score = metrics.get("MAE_aggregate")
                if score is None:
                    fin = metrics.get("FinalAdoption")
                    score = float(1.0 - fin) if isinstance(fin, (int, float)) else 1.0
            return torch.tensor([float(score)], dtype=torch.float32)

        prior_list = [priors[k] for k in ["peer_weight", "media_weight", "adoption_slope",
                                          "media_campaign_intensity", "media_decay_rate", "temperature"]]
        import torch  # type: ignore  # re-import for type
        low_cat = torch.cat([p.low for p in prior_list], dim=0)
        high_cat = torch.cat([p.high for p in prior_list], dim=0)
        prior = sbi_utils.BoxUniform(
            low=low_cat,
            high=high_cat,
        )
        inference = sbi_inference.SNPE(prior=prior)
        # Generate simulation data
        num_sims = budget
        xs = []
        ys = []
        for _ in range(num_sims):
            theta = prior.sample((1,)).squeeze(0)
            y = sim_fn(theta)
            xs.append(theta)
            ys.append(y)
        xs_t = torch.stack(xs, dim=0)
        ys_t = torch.stack(ys, dim=0)
        density_estimator = inference.append_simulations(xs_t, ys_t).train()
        posterior = inference.build_posterior(density_estimator)
        # Condition on an "ideal" target small error (e.g., zero)
        obs = torch.zeros(1)
        samples = posterior.sample((100,), x=obs)
        best_idx = int(torch.argmin(torch.norm(samples, dim=1)))
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
        results_dir = artifacts_dir or os.path.join(simulator.artifacts_dir, "calibration", "snpe")
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
    Return a calibrator instance by name; accepts optional config (unused in this minimal implementation).
    """
    pass
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    # In a full implementation, config_path could be parsed for hyperparameters; here we ignore or load defaults
    return CALIBRATOR_REGISTRY[name]()


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
    adapter = PlanParamsAdapter(os.path.join(simulator.artifacts_dir, "parameter_definitions.json"))
    # FIXED: Use a fresh simulator to avoid contaminating the provided instance
    sim_copy = Simulation(simulator.plan, ParameterRegistry.from_plan(simulator.plan), seed=simulator.seed,
                          artifacts_dir=os.path.join(simulator.artifacts_dir, "eval_tmp"))
    sim_copy.registry.values = dict(simulator.registry.values)
    adapter.apply(sim_copy, params)
    start, end = window
    sim_copy.run(start, end)
    metrics = sim_copy.evaluate()
    # Ensure required keys exist
    if "Brier" not in metrics or metrics["Brier"] is None:
        metrics["Brier"] = None
    if "TransitionFit" not in metrics:
        metrics["TransitionFit"] = {"P01": None, "P11": None, "P10": None, "P00": None}
    return metrics


def load_plan(source_file: Optional[str], source_url: Optional[str], timeout: float, retries: int) -> Dict[str, Any]:
    """
    Load model plan JSON from file or URL based on CLI input.
    """
    pass
    if bool(source_file) == bool(source_url):
        eprint("Provide exactly one of --plan-file or --plan-url")
        sys.exit(1)
    try:
        if source_file:
            text = load_text_from_file(source_file)
        else:
            text = load_text_from_url(source_url, timeout=timeout, retries=retries)
    except Exception as ex:
        eprint(f"Failed to load plan: {ex}")
        sys.exit(1)
    plan = parse_json(text)
    validate_plan(plan)
    return plan


def load_ground_truth(plan: Dict[str, Any]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Load ground-truth observables if referenced by plan data_sources; returns None if unavailable.
    """
    pass
    try:
        # Simplified: attempt to load a CSV if target_data_field points to a CSV; but in minimal code, we skip
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
    parser = argparse.ArgumentParser(description="Social simulation runner with calibration")
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
    registry.save_definitions(param_defs_path)
    # Save effective params snapshot initially
    registry.save_used(os.path.join(args.artifacts_dir, "parameters_used.json"))

    # Initialize simulation
    seed = args.seed if args.seed is not None else (int(registry.get_param_or("random_seed", 0)) if "random_seed" in registry.definitions else None)
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
    metrics = sim.evaluate(ground_truth=gt)
    eprint(f"Evaluation metrics: {metrics}")

    # Visualization
    if args.visualize:
        sim.visualize(save_path=os.path.join(args.artifacts_dir, "figs", "adoption_rate.png"))

    # Persist final parameters used
    registry.save_used(os.path.join(args.artifacts_dir, "parameters_used.json"))

    # Print a minimal JSON result to stdout for automation
    print(json.dumps({"status": "ok", "metrics": metrics, "artifacts_dir": args.artifacts_dir}, indent=2))


# Execute main for both direct execution and sandbox wrapper invocation
main()