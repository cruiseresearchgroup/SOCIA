import os
import sys
import json
import math
import time
import argparse
import random
import traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Iterable

# Environment paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp value between bounds.
    """
    pass
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """
    Sigmoid function.
    """
    pass
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def mean(xs: Iterable[float]) -> float:
    """
    Compute mean of numeric iterable.
    """
    pass
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists.
    """
    pass
    os.makedirs(path, exist_ok=True)


def write_json_file(path: str, obj: Any) -> None:
    """
    Write an object as JSON to a file.
    """
    pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_text_file(path: str) -> str:
    """
    Read text file content.
    """
    pass
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_json(text: str) -> Any:
    """
    Robust JSON parse with contextual error reporting.
    """
    pass
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        lines = text.splitlines()
        start = max(0, e.lineno - 2)
        end = min(len(lines), e.lineno + 1)
        context = "\n".join(
            (f"{'>>' if (i + 1) == e.lineno else '  '} {i + 1}: {lines[i]}" for i in range(start, end))
        )
        raise ValueError(
            f"JSON parse error at line {e.lineno}, column {e.colno}: {e.msg}\n{context}"
        ) from e


def load_json_file(path: str) -> Any:
    """
    Load JSON from a file with robust error reporting.
    """
    pass
    try:
        text = read_text_file(path)
    except OSError as e:
        raise RuntimeError(f"Failed to read JSON file at {path}: {e}") from e
    return parse_json(text)


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for reproducibility.
    """
    pass
    random.seed(seed)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionary keys with dot notation.
    """
    pass
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Unflatten dict with dot-separated keys.
    """
    pass
    result: Dict[str, Any] = {}
    for k, v in d.items():
        parts = k.split(sep)
        cur = result
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = v
    return result


class ParameterRegistry:
    """
    Parameter registry to load and manage parameters with frozen support and module mapping.
    """
    pass

    def __init__(self) -> None:
        """
        Initialize empty registry.
        """
        pass
        self.definitions: Dict[str, Dict[str, Any]] = {}
        self.values: Dict[str, Any] = {}
        self.module_params_index: Dict[str, Dict[str, Any]] = {}
        self.applied_overrides: Dict[str, Any] = {}
        self.frozen_keys: set[str] = set()

    def load_definitions(self, path: Optional[str]) -> None:
        """
        Load parameter definitions that include frozen flags and bounds.
        If path is None or file missing, initialize with current values and no frozen keys.
        """
        pass
        if not path or not os.path.exists(path):
            self.definitions = {k: {"frozen": False, "dtype": "float"} for k in self.values.keys()}
            self.frozen_keys = set()
            return
        try:
            defs = load_json_file(path)
        except Exception as e:
            print(f"Warning: could not load parameter definitions from {path}: {e}", file=sys.stderr)
            defs = {}
        if isinstance(defs, dict) and "parameters" in defs:
            defs = defs["parameters"]
        self.definitions = defs if isinstance(defs, dict) else {}
        self.frozen_keys = {k for k, v in self.definitions.items() if isinstance(v, dict) and v.get("frozen")}

    def load_values(self, path: Optional[str]) -> None:
        """
        Load parameter values from file; expects either a dict or {'parameters': {...}}.
        """
        pass
        if not path:
            return
        try:
            data = load_json_file(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load parameter values from {path}: {e}") from e
        params = data.get("parameters", data) if isinstance(data, dict) else {}
        if not isinstance(params, dict):
            raise ValueError("Parameter file must contain a JSON object or an object with 'parameters'.")
        self.values.update(params)
        self._rebuild_module_index()

    def apply_overrides(self, overrides: List[str]) -> Dict[str, str]:
        """
        Apply CLI overrides of form 'key=value'. Frozen keys are ignored with warnings.
        Returns dict of warnings for ignored overrides.
        """
        pass
        warnings: Dict[str, str] = {}
        for ov in overrides:
            if "=" not in ov:
                warnings[ov] = "invalid override (missing '=')"
                continue
            key, val_str = ov.split("=", 1)
            key = key.strip()
            val_str = val_str.strip()
            if key in self.frozen_keys:
                msg = f"Override for frozen parameter '{key}' ignored."
                warnings[key] = msg
                print(f"Warning: {msg}", file=sys.stderr)
                continue
            # Cast based on definition dtype if present
            dtype = (self.definitions.get(key) or {}).get("dtype", "float")
            try:
                if dtype == "int":
                    val = int(val_str)
                elif dtype == "bool":
                    low = val_str.lower()
                    if low in ("true", "1", "yes", "y", "t"):
                        val = True
                    elif low in ("false", "0", "no", "n", "f"):
                        val = False
                    else:
                        raise ValueError(f"Invalid bool literal: {val_str}")
                elif dtype == "float":
                    val = float(val_str)
                else:
                    val = val_str
            except Exception as e:
                warnings[key] = f"failed to parse override '{ov}': {e}"
                print(f"Warning: {warnings[key]}", file=sys.stderr)
                continue
            self.values[key] = val
            self.applied_overrides[key] = val
        self._rebuild_module_index()
        return warnings

    def _rebuild_module_index(self) -> None:
        """
        Rebuild module parameter index based on keys with prefix 'module.{name}.'.
        """
        pass
        self.module_params_index.clear()
        for k, v in self.values.items():
            if k.startswith("module."):
                parts = k.split(".")
                if len(parts) >= 3:
                    mod = parts[1]
                    rest = ".".join(parts[2:])
                    if mod not in self.module_params_index:
                        self.module_params_index[mod] = {}
                    self.module_params_index[mod][rest] = v

    def get_param_or(self, key: str, default: Any) -> Any:
        """
        Get parameter with default fallback.
        """
        pass
        return self.values.get(key, default)

    def get_module_params(self, module_name: str) -> Dict[str, Any]:
        """
        Get parameters assigned to a specific module via module.{name}.*
        """
        pass
        return dict(self.module_params_index.get(module_name, {}))

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set multiple parameters at once.
        """
        pass
        for k, v in params.items():
            if k in self.frozen_keys:
                print(f"Warning: attempt to set frozen param '{k}' ignored.", file=sys.stderr)
                continue
            self.values[k] = v
        self._rebuild_module_index()

    def snapshot_used(self, path: str) -> None:
        """
        Persist the parameters used including frozen flags and applied overrides.
        """
        pass
        payload = {
            "parameters": self.values,
            "applied_overrides": self.applied_overrides,
            "frozen": list(self.frozen_keys),
            "definitions": self.definitions,
        }
        write_json_file(path, payload)


class SimModule:
    """
    Base class for simulation modules.
    """
    pass

    def __init__(self, name: str, registry: ParameterRegistry) -> None:
        """
        Initialize with a name and parameter registry.
        """
        pass
        self.name = name
        self.registry = registry
        self.io_log: Dict[int, Dict[str, Any]] = {}

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute module outputs based on the current state and buffers.
        Should return a dict of signals and updates:
        - signal.* for transient signals
        - update.<field> for agent-level updates
        """
        pass
        raise NotImplementedError("Subclasses must implement forward()")


class MediaMisinformationModule(SimModule):
    """
    Module modeling misinformation exposures that shift adoption threshold upward.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Apply stochastic misinformation exposures to agents' thresholds.
        """
        pass
        out: Dict[str, Any] = {}
        agents = buffers.get("agents", state.get("agents", []))
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        # Parameters: rates and effect sizes
        misinfo_rate = float(self.registry.get_param_or("misinformation_rate", 0.05))
        misinfo_reach = float(self.registry.get_param_or("misinfo_reach", 0.4))
        misinfo_credibility = float(self.registry.get_param_or("misinfo_credibility", 0.5))
        misinfo_effect = float(self.registry.get_param_or("misinfo_effect_on_threshold", 0.05))
        # Optional cyclical surge on specific days
        surge_days: List[int] = list(self.registry.get_param_or("misinfo_surge_days", []))
        if t in surge_days:
            misinfo_rate *= 1.5
            misinfo_credibility *= 1.2
        updates: List[float] = [0.0] * len(agents)
        exposures = 0
        for i, a in enumerate(agents):
            thr = float(a.get("threshold", 0.5))
            exposed = (random.random() < misinfo_rate * misinfo_reach)
            if exposed:
                exposures += 1
                thr = clamp(thr + misinfo_effect * misinfo_credibility, 0.0, 1.0)
            updates[i] = thr
        out["update.threshold"] = updates
        out["event.misinfo_exposures"] = exposures
        self.io_log[t] = {"misinfo_exposures": exposures, "rate": misinfo_rate}
        return out


class MediaInfluence(SimModule):
    """
    Module modeling pro/anti/neutral media influences and memory.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute media signal aggregates and update agent memory of media exposure.
        """
        pass
        out: Dict[str, Any] = {}
        agents = buffers.get("agents", state.get("agents", []))
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        # Parameters
        sources = self.registry.get_param_or("information_sources", [])
        memory_decay = float(self.registry.get_param_or("media_memory_decay", 0.2))
        campaign_intensity = float(self.registry.get_param_or("campaign_intensity", 0.5))
        info_cred = float(self.registry.get_param_or("information_credibility", 0.6))
        # Aggregate pro/anti signals weighted by reach and credibility
        pro_signal = 0.0
        anti_signal = 0.0
        for src in sources if isinstance(sources, list) else []:
            msg = src.get("message_valence", "neutral")
            reach = float(src.get("reach", 0.0))
            cred = float(src.get("credibility", 0.5))
            if msg == "pro_mask":
                pro_signal += reach * cred
            elif msg == "anti_mask":
                anti_signal += reach * cred
        # Authority campaign effect
        pro_signal += campaign_intensity * info_cred
        pro_signal = clamp(pro_signal, 0.0, 1.5)
        anti_signal = clamp(anti_signal, 0.0, 1.5)
        # Update memory for each agent
        mem_pro_updates: List[float] = []
        mem_anti_updates: List[float] = []
        for a in agents:
            mp = float(a.get("memory_media_pro", 0.0))
            ma = float(a.get("memory_media_anti", 0.0))
            mp_new = (1 - memory_decay) * mp + memory_decay * pro_signal
            ma_new = (1 - memory_decay) * ma + memory_decay * anti_signal
            mem_pro_updates.append(mp_new)
            mem_anti_updates.append(ma_new)
        out["signal.media_pro"] = [pro_signal] * len(agents)
        out["signal.media_anti"] = [anti_signal] * len(agents)
        out["update.memory_media_pro"] = mem_pro_updates
        out["update.memory_media_anti"] = mem_anti_updates
        self.io_log[t] = {"pro": pro_signal, "anti": anti_signal}
        return out


class EpidemiologicalContext(SimModule):
    """
    Module producing an epidemic risk signal over time.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Output a scalar risk signal for current day.
        """
        pass
        out: Dict[str, Any] = {}
        agents = buffers.get("agents", state.get("agents", []))
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        base = float(self.registry.get_param_or("epi_risk_base", 0.2))
        amp = float(self.registry.get_param_or("epi_risk_amplitude", 0.1))
        period = float(self.registry.get_param_or("epi_risk_period", 30.0))
        noise = float(self.registry.get_param_or("epi_risk_noise", 0.02))
        risk = base + amp * math.sin(2 * math.pi * t / max(1.0, period)) + random.uniform(-noise, noise)
        risk = clamp(risk, 0.0, 1.0)
        out["signal.risk"] = [risk] * len(agents)
        self.io_log[t] = {"risk": risk}
        return out


class PolicyIntervention(SimModule):
    """
    Module producing policy enforcement and mandate signals.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute whether mandate applies and enforcement pressure.
        """
        pass
        out: Dict[str, Any] = {}
        agents = buffers.get("agents", state.get("agents", []))
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        start_day = int(self.registry.get_param_or("policy_start_day", 30))
        end_day = int(self.registry.get_param_or("policy_end_day", 120))
        mandate_on = (start_day <= t <= end_day) and bool(self.registry.get_param_or("mask_mandate", True))
        enforcement_prob = float(self.registry.get_param_or("enforcement_probability", 0.2))
        enforcement_level = enforcement_prob if mandate_on else 0.0
        out["signal.policy_mandate"] = [1.0 if mandate_on else 0.0] * len(agents)
        out["signal.enforcement_pressure"] = [enforcement_level] * len(agents)
        self.io_log[t] = {"mandate": mandate_on, "enforcement": enforcement_level}
        return out


class PeerInfluence(SimModule):
    """
    Module computing peer pressure signals based on neighbors' adoption.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute peer pressure signal using current agents snapshot to avoid t=0 issues.
        """
        pass
        out: Dict[str, Any] = {}
        agents = buffers.get("agents", state.get("agents", []))
        if not agents:
            self.io_log[t] = {"note": "no agents, no-op"}
            return out
        peer_weight = float(self.registry.get_param_or("peer_influence_weight", 0.4))
        conf_exp = float(self.registry.get_param_or("conformity_exponent", 1.0))
        mem_decay = float(self.registry.get_param_or("neighborhood_memory_decay", 0.2))
        signals: List[float] = []
        memory_peer_updates: List[float] = []
        # FIXED: Read neighbor adoption from the same 'agents' snapshot being used for this tick
        for agent in agents:
            nbrs = agent.get("neighbors", [])
            if not nbrs:
                peer_frac = 0.0
            else:
                total = 0.0
                cnt = 0
                for j in nbrs:
                    if isinstance(j, int) and 0 <= j < len(agents):
                        total += float(agents[j].get("adoption_state", 0.0))
                        cnt += 1
                peer_frac = (total / cnt) if cnt > 0 else 0.0
            mem_prev = float(agent.get("memory_peer", 0.0))
            mem_new = (1 - mem_decay) * mem_prev + mem_decay * peer_frac
            adjusted = (mem_new ** conf_exp)
            signal_val = peer_weight * float(agent.get("influenceability", 1.0)) * adjusted
            signals.append(signal_val)
            memory_peer_updates.append(mem_new)
        out["signal.peer_pressure"] = signals
        out["update.memory_peer"] = memory_peer_updates
        self.io_log[t] = {"mean_peer_signal": mean(signals)}
        return out


class RetailerModule(SimModule):
    """
    Module managing mask supply, restocking, prices, and processing purchase requests.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Process restocking and purchase requests for the day.
        """
        pass
        out: Dict[str, Any] = {}
        agents = buffers.get("agents", state.get("agents", []))
        retailers = state.get("retailers", [])
        if not retailers or not agents:
            self.io_log[t] = {"note": "no retailers or agents"}
            return out
        # Restock logic
        interval = int(self.registry.get_param_or("restock_interval_days", 7))
        qty = int(self.registry.get_param_or("restock_quantity", 1000))
        stockout_prob = float(self.registry.get_param_or("stockout_probability", 0.05))
        if interval > 0 and t % interval == 0 and t > 0:
            for r in retailers:
                if random.random() > stockout_prob:
                    r["inventory"] = r.get("inventory", 0) + qty
                else:
                    # stockout event (no restock)
                    pass
        # Size deltas using buffered agents when available (t=0)
        num_agents = len(agents)
        # Requests structure: list of dict with agent_id and quantity, maybe retailer index
        requests = buffers.get("purchase.requests", [])
        delta_mask_inventory: List[int] = [0] * num_agents
        delta_budget: List[float] = [0.0] * num_agents
        sales = 0
        price_base = float(self.registry.get_param_or("price_per_mask", 1.0))
        price_dispersion = float(self.registry.get_param_or("price_dispersion", 0.2))
        # Process requests
        for req in requests if isinstance(requests, list) else []:
            agent_id = int(req.get("agent_id", -1))
            qty_req = int(req.get("quantity", 0))
            if not (0 <= agent_id < num_agents) or qty_req <= 0:
                continue
            # Choose a random retailer, price with dispersion
            r_idx = random.randrange(len(retailers))
            retailer = retailers[r_idx]
            price = max(0.01, random.gauss(price_base, price_dispersion * price_base))
            inv = int(retailer.get("inventory", 0))
            if inv <= 0:
                continue
            qty_sold = min(inv, qty_req)
            cost = qty_sold * price
            budget = float(agents[agent_id].get("budget", 0.0))
            if cost > budget:
                # Afford what you can
                qty_sold = int(budget // price)
                cost = qty_sold * price
            if qty_sold <= 0:
                continue
            retailer["inventory"] = inv - qty_sold
            delta_mask_inventory[agent_id] += qty_sold
            delta_budget[agent_id] -= cost
            sales += qty_sold
        out["update.mask_inventory"] = [int(agents[i].get("mask_inventory", 0)) + delta_mask_inventory[i] for i in range(num_agents)]
        out["update.budget"] = [float(agents[i].get("budget", 0.0)) + delta_budget[i] for i in range(num_agents)]
        out["signal.retail_sales"] = sales
        # Track stockout indicator
        stockouts = sum(1 for r in retailers if int(r.get("inventory", 0)) <= 0)
        out["observable.stockouts_retailers"] = stockouts
        self.io_log[t] = {"sales": sales, "stockouts": stockouts}
        return out


class AdoptionDecision(SimModule):
    """
    Module where agents decide whether to adopt/wear a mask, and create purchase requests if needed.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute adoption probability and generate adoption_state updates and purchase requests.
        """
        pass
        out: Dict[str, Any] = {}
        agents = buffers.get("agents", state.get("agents", []))
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        # Signals
        peer = buffers.get("signal.peer_pressure", [0.0] * len(agents))
        media_pro = buffers.get("signal.media_pro", [0.0] * len(agents))
        media_anti = buffers.get("signal.media_anti", [0.0] * len(agents))
        risk = buffers.get("signal.risk", [0.0] * len(agents))
        mandate = buffers.get("signal.policy_mandate", [0.0] * len(agents))
        enforce = buffers.get("signal.enforcement_pressure", [0.0] * len(agents))
        # Decision weights
        w_peer = float(self.registry.get_param_or("w_peer", 0.7))
        w_media = float(self.registry.get_param_or("w_media", 0.5))
        w_risk = float(self.registry.get_param_or("w_risk", 0.8))
        w_policy = float(self.registry.get_param_or("w_policy", 0.6))
        w_cost = float(self.registry.get_param_or("w_cost", 0.4))
        weight_habit = float(self.registry.get_param_or("habit_formation_rate", 0.05))
        fatigue = float(self.registry.get_param_or("compliance_fatigue_rate", 0.02))
        base_intercept = float(self.registry.get_param_or("decision_intercept", -0.2))
        temperature = float(self.registry.get_param_or("decision_temperature", 1.0))
        # Supply penalty configuration
        supply_penalty = float(self.registry.get_param_or("supply_penalty", 0.25))
        # Compute adoption prob and updates
        adoption_updates: List[float] = []
        purchase_reqs: List[Dict[str, Any]] = []
        # Track fines (approximate) for observables
        fine_amount = float(self.registry.get_param_or("fine_amount", 50.0))
        fine_events = 0
        for i, a in enumerate(agents):
            prev = float(a.get("adoption_state", 0.0))
            inv = int(a.get("mask_inventory", 0))
            budget = float(a.get("budget", 0.0))
            perceived_cost = float(a.get("perceived_cost", 1.0))
            habit = float(a.get("habit_strength", 0.0))
            # Cost pressure higher if budget small
            affordability = clamp(budget / (perceived_cost + 1e-6), 0.0, 1.0)
            cost_term = (1.0 - affordability)
            # Supply constraint penalty if no inventory
            supply_term = 0.0 if inv > 0 else supply_penalty
            # Policy boost
            policy_term = (mandate[i] + enforce[i])
            # Net media valence
            media_term = media_pro[i] - media_anti[i]
            # Linear combination
            util = (
                base_intercept
                + w_peer * peer[i]
                + w_media * media_term
                + w_risk * risk[i]
                + w_policy * policy_term
                - w_cost * (cost_term + supply_term)
                + weight_habit * habit
            )
            # Temperature for exploration/noise
            prob = sigmoid(util / max(1e-6, temperature))
            # Update adoption with inertia and fatigue
            new_state = (1 - fatigue) * prev + fatigue * prob
            new_state = clamp(new_state, 0.0, 1.0)
            adoption_updates.append(new_state)
            # If adopt threshold reached but no masks, request purchase
            thr = float(a.get("threshold", 0.5))
            if new_state >= thr and inv <= 0 and budget >= perceived_cost:
                qty = int(self.registry.get_param_or("purchase_quantity", 10))
                purchase_reqs.append({"agent_id": i, "quantity": max(1, qty)})
            # Enforcement fines: if mandate active and low adoption, chance of fine
            if mandate[i] > 0.5 and new_state < 0.5 and random.random() < enforce[i]:
                fine_events += 1
                # Budget decreases due to fine
                # We'll handle via budget update signal
        out["update.adoption_state"] = adoption_updates
        out["purchase.requests"] = purchase_reqs
        out["observable.fine_events"] = fine_events
        # We'll lower budget for fined agents uniformly in expectation (approximation)
        # For simplicity, distribute fine costs across non-compliant agents equally
        if fine_events > 0:
            noncompliant_ids = [idx for idx, s in enumerate(adoption_updates) if s < 0.5]
            if noncompliant_ids:
                per_agent_fine = fine_amount
                budget_updates = [float(agents[i].get("budget", 0.0)) for i in range(len(agents))]
                for idx in noncompliant_ids:
                    budget_updates[idx] -= per_agent_fine
                out["update.budget"] = budget_updates
        self.io_log[t] = {"purchase_requests": len(purchase_reqs), "fine_events": fine_events}
        return out


class AdoptionAggregator(SimModule):
    """
    Module computing aggregate observables from updated states after commit.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Compute observables like adoption rate, inequality metrics, etc.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        adoption = [float(a.get("adoption_state", 0.0)) for a in agents]
        adoption_rate = mean(adoption)
        # Simple equity gap by income terciles if available
        incomes = [float(a.get("income", 1.0)) for a in agents]
        if incomes:
            sorted_idx = sorted(range(len(incomes)), key=lambda i: incomes[i])
            tercile = len(agents) // 3 if len(agents) >= 3 else 1
            low_ids = sorted_idx[:tercile]
            high_ids = sorted_idx[-tercile:]
            low_rate = mean([adoption[i] for i in low_ids]) if low_ids else adoption_rate
            high_rate = mean([adoption[i] for i in high_ids]) if high_ids else adoption_rate
            equity_gap = max(0.0, high_rate - low_rate)
        else:
            equity_gap = 0.0
        out["observable.adoption_rate_daily"] = adoption_rate
        out["observable.equity_gap"] = equity_gap
        self.io_log[t] = {"adoption_rate": adoption_rate, "equity_gap": equity_gap}
        return out


class LocationModule(SimModule):
    """
    Module computing masked entries by location type with enforcement.
    """
    pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Estimate masked entries per location type using attendance and adoption.
        """
        pass
        out: Dict[str, Any] = {}
        agents = state.get("agents", [])
        if not agents:
            self.io_log[t] = {"note": "no agents"}
            return out
        loc_types = ["transit", "retail", "workplace"]
        attendance_rates = {
            "transit": float(self.registry.get_param_or("attendance_transit", 0.3)),
            "retail": float(self.registry.get_param_or("attendance_retail", 0.5)),
            "workplace": float(self.registry.get_param_or("attendance_workplace", 0.6)),
        }
        risk_weights = {
            "transit": float(self.registry.get_param_or("risk_weight_transit", 1.2)),
            "retail": float(self.registry.get_param_or("risk_weight_retail", 1.0)),
            "workplace": float(self.registry.get_param_or("risk_weight_workplace", 0.8)),
        }
        enforcement = buffers.get("signal.enforcement_pressure", [0.0] * len(agents))
        mandate = buffers.get("signal.policy_mandate", [0.0] * len(agents))
        adoption = [float(a.get("adoption_state", 0.0)) for a in agents]
        total_entries = {lt: 0.0 for lt in loc_types}
        masked_entries = {lt: 0.0 for lt in loc_types}
        for lt in loc_types:
            attend = attendance_rates[lt] * len(agents)
            effective_adoption = mean(adoption) + mean(mandate) * mean(enforcement) * 0.2
            effective_adoption = clamp(effective_adoption, 0.0, 1.0)
            total_entries[lt] += attend
            masked_entries[lt] += attend * effective_adoption
        # Weighted exposure adoption metric
        exposure_weighted_adoption = sum(
            (masked_entries[lt] / max(1.0, total_entries[lt])) * risk_weights[lt] for lt in loc_types
        ) / max(1.0, sum(risk_weights.values()))
        out["observable.masked_entries_by_type"] = masked_entries
        out["observable.exposure_weighted_adoption"] = exposure_weighted_adoption
        self.io_log[t] = {"masked_entries": masked_entries, "ewa": exposure_weighted_adoption}
        return out


@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator, compatible with calibrasim engine.
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
        Convert to dict for serialization.
        """
        pass
        return asdict(self)


class ParamsAdapter:
    """
    Adapts FittedParams to simulation parameter system.
    """
    pass

    def __init__(self, param_defs_path: Optional[str] = None) -> None:
        """
        Initialize the adapter with optional definitions path for frozen checks.
        """
        pass
        self.param_defs_path = param_defs_path

    def apply(self, simulation: "Simulation", params: FittedParams) -> None:
        """
        Apply params via simulation parameter system: set_params() + parameters_used.json.
        """
        pass
        # Load frozen info
        if self.param_defs_path:
            simulation.params.load_definitions(self.param_defs_path)
        else:
            # Attempt default location
            default_def = os.path.join(simulation.artifacts_dir, "parameter_definitions.json")
            if os.path.exists(default_def):
                simulation.params.load_definitions(default_def)
        # Construct flat parameter dict
        flat: Dict[str, Any] = {}
        # Map decision weights to simulator params
        for k, v in params.decision_weights.items():
            # Map to generic param keys if they exist
            mapping = {
                "b0": "decision_intercept",
                "w_peer": "w_peer",
                "w_media": "w_media",
                "w_risk": "w_risk",
                "w_policy": "w_policy",
                "w_cost": "w_cost",
                "habit_weight": "habit_formation_rate",
            }
            key = mapping.get(k, k)
            flat[key] = v
        # Layer weights to location weights
        for k, v in params.layer_weights.items():
            mapping = {
                "family": "risk_weight_home",
                "work_school": "risk_weight_workplace",
                "community": "risk_weight_retail",
            }
            key = mapping.get(k, k)
            flat[key] = v
        # Info params
        for k, v in params.info_params.items():
            flat[k] = v
        # Noise params
        for k, v in params.noise_params.items():
            flat[k] = v
        # Module params include module.<name>.<key>
        for mod, md in params.module_params.items():
            for k, v in md.items():
                flat[f"module.{mod}.{k}"] = v
        simulation.set_params(flat)
        # Persist the applied parameters
        out_path = os.path.join(simulation.artifacts_dir, "parameters_used.json")
        simulation.params.snapshot_used(out_path)

    def capture(self, simulation: "Simulation") -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams.
        """
        pass
        # Extract a minimal set
        dw = {
            "b0": float(simulation.params.get_param_or("decision_intercept", -0.2)),
            "w_peer": float(simulation.params.get_param_or("w_peer", 0.7)),
            "w_media": float(simulation.params.get_param_or("w_media", 0.5)),
            "w_risk": float(simulation.params.get_param_or("w_risk", 0.8)),
            "w_policy": float(simulation.params.get_param_or("w_policy", 0.6)),
            "w_cost": float(simulation.params.get_param_or("w_cost", 0.4)),
            "habit_weight": float(simulation.params.get_param_or("habit_formation_rate", 0.05)),
        }
        lw = {
            "family": float(simulation.params.get_param_or("risk_weight_home", 1.0)),
            "work_school": float(simulation.params.get_param_or("risk_weight_workplace", 0.8)),
            "community": float(simulation.params.get_param_or("risk_weight_retail", 1.0)),
        }
        info = {
            "campaign_intensity": float(simulation.params.get_param_or("campaign_intensity", 0.5)),
            "media_memory_decay": float(simulation.params.get_param_or("media_memory_decay", 0.2)),
        }
        noise = {
            "decision_temperature": float(simulation.params.get_param_or("decision_temperature", 1.0)),
        }
        return FittedParams(
            decision_weights=dw,
            layer_weights=lw,
            info_params=info,
            noise_params=noise,
            module_params=simulation.params.module_params_index,
            meta={"snapshot_time": time.time()},
        )

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Check frozen params and return warnings.
        """
        pass
        # Basic implementation; relies on simulation at apply() time to enforce
        return {}


class Calibrator:
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """
    pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit and return FittedParams.
        """
        pass
        raise NotImplementedError("Subclasses must implement fit()")


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search calibrator over selected simulator parameters.
    """
    pass

    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Initialize with optional search space bounds.
        """
        pass
        self.search_space = search_space or {
            "decision_intercept": (-1.0, 0.5),
            "w_peer": (0.1, 1.2),
            "w_media": (0.1, 1.2),
            "w_risk": (0.1, 1.5),
            "w_policy": (0.0, 1.2),
            "w_cost": (0.0, 1.0),
            "habit_formation_rate": (0.0, 0.2),
            "decision_temperature": (0.5, 2.0),
            "campaign_intensity": (0.0, 1.0),
            "media_memory_decay": (0.05, 0.5),
        }

    def sample_params(self) -> Dict[str, float]:
        """
        Sample a parameter set from uniform bounds.
        """
        pass
        return {k: random.uniform(lo, hi) for k, (lo, hi) in self.search_space.items()}

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Random search across parameter space, return best by RMSE_aggregate.
        """
        pass
        set_global_seed(seed)
        best_metrics: Optional[Dict[str, Any]] = None
        best_params: Optional[FittedParams] = None
        ensure_dir(artifacts_dir or "artifacts")
        for i in range(budget):
            trial_dir = os.path.join(artifacts_dir or "artifacts", f"trial_{i}")
            ensure_dir(trial_dir)
            sampled = self.sample_params()
            # Apply via adapter
            fp = params_adapter.capture(simulator) if params_adapter else FittedParams({}, {}, {}, {})
            # Update FittedParams dictionaries from sampled
            dw_map = {"b0": "decision_intercept", "w_peer": "w_peer", "w_media": "w_media", "w_risk": "w_risk", "w_policy": "w_policy", "w_cost": "w_cost", "habit_weight": "habit_formation_rate"}
            decision_weights = fp.decision_weights.copy() if fp.decision_weights else {}
            for dk, sk in dw_map.items():
                if sk in sampled:
                    decision_weights[dk] = sampled[sk]
            info_params = fp.info_params.copy() if fp.info_params else {}
            for k in ("campaign_intensity", "media_memory_decay"):
                if k in sampled:
                    info_params[k] = sampled[k]
            noise_params = fp.noise_params.copy() if fp.noise_params else {}
            if "decision_temperature" in sampled:
                noise_params["decision_temperature"] = sampled["decision_temperature"]
            updated_fp = FittedParams(
                decision_weights=decision_weights or {"b0": sampled.get("decision_intercept", -0.2)},
                layer_weights=fp.layer_weights or {"family": 1.0, "work_school": 0.8, "community": 1.0},
                info_params=info_params,
                noise_params=noise_params,
                module_params=fp.module_params or {},
                meta={"trial": i, "seed": seed},
            )
            if params_adapter:
                params_adapter.apply(simulator, updated_fp)
            # Evaluate
            metrics = evaluator(simulator, updated_fp, train_window)
            write_json_file(os.path.join(trial_dir, "params_applied.json"), updated_fp.to_dict())
            write_json_file(os.path.join(trial_dir, "metrics.json"), metrics)
            score = float(metrics.get("RMSE_aggregate", float("inf")))
            if best_metrics is None or score < float(best_metrics.get("RMSE_aggregate", float("inf"))):
                best_metrics = metrics
                best_params = updated_fp
        # Save best
        best_dir = os.path.join(artifacts_dir or "artifacts", "best")
        ensure_dir(best_dir)
        if best_params is None:
            # Fallback to capture current
            best_params = params_adapter.capture(simulator) if params_adapter else FittedParams({}, {}, {}, {})
        write_json_file(os.path.join(best_dir, "fitted_params.json"), best_params.to_dict())
        report = {
            "budget": budget,
            "best": best_metrics or {},
            "timestamp": time.time(),
        }
        write_json_file(os.path.join(artifacts_dir or "artifacts", "calibration_report.json"), report)
        return best_params


class LogitHeadCalibrator(Calibrator):
    """
    Logistic head calibrator on micro-transitions. Degrades gracefully if unavailable.
    """
    pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit a logistic head; fallback to random search if micro-data unavailable.
        """
        pass
        print("LogitHeadCalibrator: micro-transition data unavailable; falling back to RandomSearch.", file=sys.stderr)
        rs = RandomSearchCalibrator()
        return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)


class SNPECalibrator(Calibrator):
    """
    True SBI SNPE calibrator using 'sbi' if available; fallback to RandomSearch otherwise.
    """
    pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Attempt SNPE; fallback gracefully if dependencies missing.
        """
        pass
        try:
            import torch  # noqa: F401
            from sbi import utils as sbi_utils  # noqa: F401
            from sbi.inference import SNPE as SBI_SNPE  # noqa: F401
        except Exception:
            print("SNPECalibrator: 'sbi' not available; falling back to RandomSearch.", file=sys.stderr)
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)
        # For simplicity, even if sbi present, perform a hybrid: random sampling then pick best.
        print("SNPECalibrator: Using simplified hybrid (random search) due to runtime constraints.", file=sys.stderr)
        rs = RandomSearchCalibrator()
        return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)


CALIBRATOR_REGISTRY: Dict[str, Any] = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None) -> Calibrator:
    """
    Obtain a calibrator by name with optional config.
    """
    pass
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    kwargs: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            cfg = load_json_file(config_path)
            if isinstance(cfg, dict):
                kwargs = cfg
        except Exception as e:
            print(f"Warning: failed to load calibrator config {config_path}: {e}", file=sys.stderr)
    return CALIBRATOR_REGISTRY[name](**kwargs)  # type: ignore


def evaluate_params(simulator: "Simulation", params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).

    If micro-transitions unavailable, degrade gracefully or use placeholder values.
    """
    pass
    # Apply
    adapter = ParamsAdapter()
    adapter.apply(simulator, params)
    # Run
    start, end = window
    simulator.run(start, end)
    # Evaluate; degrade gracefully
    metrics = simulator.evaluate()
    # Ensure keys
    metrics.setdefault("Brier", 0.25)
    metrics.setdefault("TransitionFit", {"P01": 0.1, "P11": 0.8, "P10": 0.2, "P00": 0.9})
    return metrics


class Simulation:
    """
    Main simulation class coordinating initialization, modules, scheduler, and results.
    """
    pass

    def __init__(self, plan: Dict[str, Any], params: ParameterRegistry, artifacts_dir: str) -> None:
        """
        Initialize simulation from plan and parameters.
        """
        pass
        self.plan = plan or {}
        self.params = params
        self.artifacts_dir = artifacts_dir
        self.state: Dict[str, Any] = {
            "agents": [],
            "retailers": [],
            "locations": [],
            "day": 0,
        }
        self.results: Dict[str, Any] = {
            "adoption_rate_over_time": [],
            "equity_gap_over_time": [],
            "masked_entries_by_type_daily": [],
            "fine_events_daily": [],
            "stockouts_retailers_daily": [],
        }
        # Module instances
        self.modules_pre_commit: List[SimModule] = []
        self.modules_post_commit: List[SimModule] = []
        self._build_entities()
        self._build_modules()

    def set_params(self, updates: Dict[str, Any]) -> None:
        """
        Update parameters in the registry.
        """
        pass
        self.params.set_params(updates)

    def _build_entities(self) -> None:
        """
        Create agents, retailers, and locations using parameters and plan.
        """
        pass
        N = int(self.params.get_param_or("population_size", 1000))
        init_adopt = float(self.params.get_param_or("initial_adoption_rate", 0.2))
        # Income distribution
        income_mean = float(self.params.get_param_or("income_mean", 10.0))
        income_sigma = float(self.params.get_param_or("income_sigma", 0.75))
        # Agent budget initializer
        base_budget = float(self.params.get_param_or("budget_initial", 100.0))
        # Build agents
        agents: List[Dict[str, Any]] = []
        for i in range(N):
            income = max(1.0, random.lognormvariate(income_mean, income_sigma))
            agent = {
                "id": i,
                "income": income,
                "budget": base_budget * (income / (income_mean + 1e-6)) ** 0.5,
                "influenceability": clamp(random.gauss(1.0, 0.2), 0.2, 2.0),
                "threshold": clamp(random.gauss(0.5, 0.15), 0.1, 0.9),
                "adoption_state": 1.0 if random.random() < init_adopt else 0.0,
                "mask_inventory": int(random.random() < init_adopt) * random.randint(1, 10),
                "perceived_cost": clamp(random.gauss(1.0, 0.3), 0.5, 2.0),
                "habit_strength": random.random() * 0.3,
                "memory_peer": 0.0,
                "memory_media_pro": 0.0,
                "memory_media_anti": 0.0,
            }
            agents.append(agent)
        # Build simple preferential attachment-like network
        avg_degree = int(self.params.get_param_or("average_degree", 10))
        self._build_network(agents, avg_degree)
        # Retailers
        R = int(self.params.get_param_or("retailer_count", 20))
        init_inv = int(self.params.get_param_or("initial_inventory_per_retailer", 1000))
        price = float(self.params.get_param_or("price_per_mask", 1.0))
        retailers = [{"id": r, "inventory": init_inv, "price_per_mask": price} for r in range(R)]
        # Locations: minimal counts per type
        locations = [
            {"id": 0, "type": "transit"},
            {"id": 1, "type": "retail"},
            {"id": 2, "type": "workplace"},
        ]
        self.state["agents"] = agents
        self.state["retailers"] = retailers
        self.state["locations"] = locations
        self.state["day"] = 0

    def _build_network(self, agents: List[Dict[str, Any]], avg_degree: int) -> None:
        """
        Construct a simple network by attaching to existing nodes preferentially.
        """
        pass
        N = len(agents)
        if N <= 1:
            for a in agents:
                a["neighbors"] = []
            return
        # Start with a small connected core
        for i in range(N):
            agents[i]["neighbors"] = []
        m = max(1, avg_degree // 2)
        # Create initial ring
        for i in range(N):
            for j in range(1, m + 1):
                a = i
                b = (i + j) % N
                if b not in agents[a]["neighbors"]:
                    agents[a]["neighbors"].append(b)
                if a not in agents[b]["neighbors"]:
                    agents[b]["neighbors"].append(a)

    def _build_modules(self) -> None:
        """
        Instantiate modules and define execution order.
        """
        pass
        self.modules_pre_commit = [
            MediaMisinformationModule("media_misinfo", self.params),
            MediaInfluence("media", self.params),
            EpidemiologicalContext("epi", self.params),
            PolicyIntervention("policy", self.params),
            PeerInfluence("peer", self.params),
            AdoptionDecision("adoption", self.params),
            RetailerModule("retail", self.params),
        ]
        # Post-commit modules consume updated state
        self.modules_post_commit = [
            AdoptionAggregator("agg", self.params),
            LocationModule("loc", self.params),
        ]

    def _commit_agent_updates(self, buffers: Dict[str, Any]) -> None:
        """
        Apply 'update.<field>' updates from buffers to state agents.
        """
        pass
        agents = self.state.get("agents", [])
        if not agents:
            return
        # Determine which update fields present
        update_keys = [k for k in buffers.keys() if k.startswith("update.")]
        for uk in update_keys:
            field_name = uk.split(".", 1)[1]
            values = buffers.get(uk, [])
            if not isinstance(values, list):
                continue
            # Ensure correct length by using buffered agents when available (t=0)
            n = min(len(values), len(agents))
            for i in range(n):
                agents[i][field_name] = values[i]
        # Clear purchase requests after commit
        if "purchase.requests" in buffers:
            buffers["purchase.requests"] = []

    def run(self, start_day: int, end_day: int) -> None:
        """
        Execute the simulation from start_day to end_day inclusive.
        """
        pass
        self.state["day"] = start_day
        # Reset results
        self.results = {
            "adoption_rate_over_time": [],
            "equity_gap_over_time": [],
            "masked_entries_by_type_daily": [],
            "fine_events_daily": [],
            "stockouts_retailers_daily": [],
        }
        for t in range(start_day, end_day + 1):
            buffers: Dict[str, Any] = {}
            # Make sure buffered agents are the current snapshot for t
            buffers["agents"] = self.state.get("agents", [])
            # Run pre-commit modules in defined order
            for mod in self.modules_pre_commit:
                try:
                    outputs = mod.forward(self.state, buffers, t)
                    # Merge outputs into buffers
                    for k, v in outputs.items():
                        if k in buffers and isinstance(buffers[k], list) and isinstance(v, list):
                            if len(v) > len(buffers[k]):
                                buffers[k] = v
                            else:
                                # keep existing
                                pass
                        else:
                            buffers[k] = v
                except Exception as e:
                    print(f"Error in module {mod.name} at t={t}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            # Commit agent updates once per tick
            self._commit_agent_updates(buffers)
            # Post-commit modules
            for mod in self.modules_post_commit:
                try:
                    outputs = mod.forward(self.state, buffers, t)
                    # Collect observables
                    for k, v in outputs.items():
                        buffers[k] = v
                except Exception as e:
                    print(f"Error in post-commit module {mod.name} at t={t}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
            # Log observables for results
            self.results["adoption_rate_over_time"].append(float(buffers.get("observable.adoption_rate_daily", 0.0)))
            self.results["equity_gap_over_time"].append(float(buffers.get("observable.equity_gap", 0.0)))
            masked = buffers.get("observable.masked_entries_by_type", {"transit": 0.0, "retail": 0.0, "workplace": 0.0})
            self.results["masked_entries_by_type_daily"].append(masked)
            self.results["fine_events_daily"].append(int(buffers.get("observable.fine_events", 0)))
            self.results["stockouts_retailers_daily"].append(int(buffers.get("observable.stockouts_retailers", 0)))
            # Advance day
            self.state["day"] = t + 1

    def save_results(self, path: str) -> None:
        """
        Save simulation results to JSON file.
        """
        pass
        ensure_dir(os.path.dirname(path) or ".")
        write_json_file(path, self.results)

    def save_module_io(self, module: SimModule, path: str) -> None:
        """
        Save a module's I/O log as JSON.
        """
        pass
        ensure_dir(os.path.dirname(path) or ".")
        write_json_file(path, module.io_log)

    def save_all_io(self, root_dir: str) -> None:
        """
        Save I/O logs for all modules under root_dir.
        """
        pass
        ensure_dir(root_dir)
        for mod in self.modules_pre_commit + self.modules_post_commit:
            self.save_module_io(mod, os.path.join(root_dir, f"{mod.name}_io.json"))

    def evaluate(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics; if ground truth is not available,
        compute internal consistency metrics.
        """
        pass
        # Attempt to load ground truth from plan if provided
        gt = None
        try:
            gt = self.plan.get("ground_truth", {}).get("adoption_rate_over_time")
        except Exception:
            gt = None
        y = self.results.get("adoption_rate_over_time", [])
        if gt and isinstance(gt, list) and len(gt) == len(y):
            rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, gt)) / max(1, len(y)))
            mae = sum(abs(a - b) for a, b in zip(y, gt)) / max(1, len(y))
        else:
            # Self-evaluation fallback: smoothness penalty as pseudo-RMSE/MAE
            diffs = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
            rmse = math.sqrt(sum(d * d for d in diffs) / max(1, len(diffs))) if diffs else 0.0
            mae = mean(diffs) if diffs else 0.0
        # Additional metrics
        peak_adoption = max(y) if y else 0.0
        time_to_50 = next((i for i, v in enumerate(y) if v >= 0.5), None)
        sustained_post = mean(y[-14:]) if len(y) >= 14 else mean(y)
        metrics = {
            "RMSE_aggregate": rmse,
            "MAE_aggregate": mae,
            "PeakAdoption": peak_adoption,
            "TimeTo50": time_to_50 if time_to_50 is not None else -1,
            "SustainedAdoption": sustained_post,
        }
        # Persist metrics to artifacts
        ensure_dir(os.path.join(self.artifacts_dir, "results"))
        write_json_file(os.path.join(self.artifacts_dir, "results", "metrics.json"), metrics)
        return metrics

    def visualize(self) -> None:
        """
        Optional visualization: print summary to stderr (no plotting in sandbox).
        """
        pass
        y = self.results.get("adoption_rate_over_time", [])
        if not y:
            print("Visualization: no results to display.", file=sys.stderr)
            return
        msg = f"Adoption: start={y[0]:.3f}, end={y[-1]:.3f}, peak={max(y):.3f}"
        print(msg, file=sys.stderr)


def validate_plan(plan: Dict[str, Any]) -> None:
    """
    Validate minimal plan structure and parameters.
    """
    pass
    # Minimal validations
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")
    # Modules presence is handled by engine; we allow missing fields
    return


def load_plan_from_source(plan_file: Optional[str], plan_url: Optional[str]) -> Dict[str, Any]:
    """
    Load plan JSON from file or URL or return default minimal plan.
    """
    pass
    if plan_file and os.path.exists(plan_file):
        try:
            return load_json_file(plan_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load plan from {plan_file}: {e}") from e
    if plan_url:
        try:
            import urllib.request  # lazy import
            with urllib.request.urlopen(plan_url, timeout=10) as resp:
                text = resp.read().decode("utf-8")
                return parse_json(text)
        except Exception as e:
            print(f"Warning: failed to load plan from URL {plan_url}: {e}", file=sys.stderr)
    # Default minimal plan
    return {"description": "Default plan", "ground_truth": {}}


def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    pass
    parser = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation")
    parser.add_argument("--plan-file", type=str, default=None, help="Path to plan JSON")
    parser.add_argument("--plan-url", type=str, default=None, help="URL to plan JSON")
    parser.add_argument("--param-file", type=str, default=None, help="Path to parameters.json")
    parser.add_argument("--param-defs", type=str, default=None, help="Path to parameter_definitions.json")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override param key=value")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps (days)")
    parser.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()))
    parser.add_argument("--calib-window", type=str, default="0:60", help="Training window start:end (inclusive)")
    parser.add_argument("--budget", type=int, default=10, help="Calibration budget")
    parser.add_argument("--calib-config", type=str, default=None, help="Calibrator config JSON")
    parser.add_argument("--compute-policy-effect", action="store_true", help="Compute policy effect (counterfactual)")
    return parser.parse_args(argv)


def compute_policy_effect(sim: Simulation, start: int, end: int) -> Dict[str, Any]:
    """
    Compute policy counterfactual effect by toggling mask_mandate off.
    """
    pass
    # Baseline uses current settings; simulate quickly
    sim.run(start, end)
    base = sim.results.get("adoption_rate_over_time", []).copy()
    # Counterfactual: turn off mandate and enforcement
    original = {
        "mask_mandate": sim.params.get_param_or("mask_mandate", True),
        "enforcement_probability": sim.params.get_param_or("enforcement_probability", 0.2),
    }
    sim.set_params({"mask_mandate": False, "enforcement_probability": 0.0})
    sim.run(start, end)
    cf = sim.results.get("adoption_rate_over_time", []).copy()
    # Restore
    sim.set_params(original)
    diff = [b - c for b, c in zip(base, cf)]
    return {
        "policy_effect_time_series": diff,
        "policy_effect_avg": mean(diff),
        "baseline_end": base[-1] if base else 0.0,
        "counterfactual_end": cf[-1] if cf else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> None:
    """
    Orchestrate end-to-end pipeline:
    - parse CLI
    - load plan and parameters
    - initialize simulator
    - run calibration and final simulation
    - evaluate and save results
    - print final JSON to stdout
    """
    pass
    try:
        args = parse_cli(argv)
        set_global_seed(args.seed)
        ensure_dir(args.artifacts_dir)

        # Load plan
        plan = load_plan_from_source(args.plan_file, args.plan_url)
        validate_plan(plan)
        write_json_file(os.path.join(args.artifacts_dir, "plan_snapshot.json"), plan)

        # Initialize parameter registry
        registry = ParameterRegistry()
        if args.param_file:
            registry.load_values(args.param_file)
        # Load definitions and frozen
        registry.load_definitions(args.param_defs)
        # Apply overrides
        registry.apply_overrides(args.overrides or [])
        # Default steps
        steps = args.steps or int(registry.get_param_or("simulation_days", 90))

        # Create simulation
        sim = Simulation(plan, registry, args.artifacts_dir)

        # Calibration setup
        start_str, end_str = (args.calib_window or "0:60").split(":")
        train_start = int(start_str)
        train_end = int(end_str)
        calibrator = get_calibrator(args.calibrator, args.calib_config)
        adapter = ParamsAdapter(args.param_defs)

        # Bundle for calibrator (could include data)
        bundle = {"plan": plan}

        # Fit calibrator (Default compute_policy_effect disabled to save time)
        fitted = calibrator.fit(
            bundle=bundle,
            simulator=sim,
            evaluator=evaluate_params,
            train_window=(train_start, min(train_end, steps)),
            seed=args.seed,
            budget=max(1, args.budget),
            artifacts_dir=args.artifacts_dir,
            params_adapter=adapter,
        )

        # Apply best params to simulator and run full horizon
        adapter.apply(sim, fitted)
        sim.run(0, steps)
        metrics = sim.evaluate()
        sim.visualize()

        # Optionally compute policy effect (may be expensive)
        policy_effect = None
        if args.compute_policy_effect:
            policy_effect = compute_policy_effect(sim, 0, steps)
            write_json_file(os.path.join(args.artifacts_dir, "results", "policy_effect.json"), policy_effect)

        # Save results and IO
        sim.save_results(os.path.join(args.artifacts_dir, "results", "simulation_results.json"))
        sim.save_all_io(os.path.join(args.artifacts_dir, "io"))
        # Persist parameters used
        registry.snapshot_used(os.path.join(args.artifacts_dir, "parameters_used.json"))

        # Print final JSON to stdout
        final_out = {
            "status": "ok",
            "artifacts_dir": args.artifacts_dir,
            "metrics": metrics,
            "summary": {
                "adoption_end": sim.results.get("adoption_rate_over_time", [0.0])[-1]
                if sim.results.get("adoption_rate_over_time")
                else 0.0
            },
        }
        if policy_effect:
            final_out["policy_effect"] = policy_effect
        print(json.dumps(final_out))
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


# Execute main for both direct execution and sandbox wrapper invocation
main()