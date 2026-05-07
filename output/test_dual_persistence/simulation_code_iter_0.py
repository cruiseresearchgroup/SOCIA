def main():
    pass

#!/usr/bin/env python3
"""
Multi-agent simulator for mask-wearing policy evaluation on a multiplex social network.

This program:
- Loads agents, multiplex network, and panel training data from CSV/JSON files.
- Builds a multiplex network with layers: family, work_school, community.
- Calibrates role-specific logistic decision policies on a temporal training window.
- Simulates forward on a validation window under two modes:
  (a) observed received_info (forecast-style)
  (b) endogenous info diffusion (policy-style)
- Computes evaluation metrics (RMSE, MAE, Brier, TransitionFit) overall and stratified by role.
- Outputs per-agent time series, aggregate curves, transition matrices, intervention logs, diagnostics, and calibration logs.

Author: Code Generation Agent
"""

import os
import csv
import json
import math
import random
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


def ensure_output_dir(path: str) -> None:
    """Create the output directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def set_random_seed(seed: int) -> None:
    """Set seed for Python's random and NumPy's RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def parse_bool(value: Any, default: int = 0) -> int:
    """Robust boolean parsing to int 0/1."""
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) != 0 else 0
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    if s in {"0", "false", "f", "no", "n"}:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return int(default)


@dataclass
class Agent:
    """Agent with static attributes and derived role."""
    agent_id: int
    age: Optional[int] = None
    age_group: Optional[str] = None
    occupation: Optional[str] = None
    risk_perception: float = 0.5
    role: str = "Nonworker"
    degree_family: int = 0
    degree_work_school: int = 0
    degree_community: int = 0
    degree_total: int = 0
    initial_mask_wearing: int = 0  # Default to 0 if not provided


@dataclass
class MultiplexNetwork:
    """Multiplex network with separate adjacency lists per layer."""
    layers: Dict[str, Dict[int, List[int]]] = field(default_factory=lambda: {
        "family": {},
        "work_school": {},
        "community": {},
        "all": {}
    })

    def enforce_symmetry(self) -> None:
        """Ensure the adjacency is undirected/symmetric on each layer (except 'all' which is derived)."""
        for layer in ["family", "work_school", "community"]:
            adj = self.layers.get(layer, {})
            for i, nbrs in list(adj.items()):
                unique = set(nbrs)
                adj[i] = list(unique)
                for j in unique:
                    if j not in adj:
                        adj[j] = []
                    if i not in adj[j]:
                        adj[j].append(i)
            self.layers[layer] = adj
        self.rebuild_union()

    def rebuild_union(self) -> None:
        """Rebuild the 'all' union layer from other layers."""
        union: Dict[int, List[int]] = {}
        for layer in ["family", "work_school", "community"]:
            for i, nbrs in self.layers.get(layer, {}).items():
                if i not in union:
                    union[i] = []
                union[i].extend(nbrs)
        for i, nbrs in union.items():
            union[i] = list(sorted(set(nbrs)))
        self.layers["all"] = union

    def add_missing_agents(self, agent_ids: List[int]) -> None:
        """Ensure every agent has entries in each layer."""
        for layer in ["family", "work_school", "community", "all"]:
            if layer not in self.layers:
                self.layers[layer] = {}
            for a in agent_ids:
                if a not in self.layers[layer]:
                    self.layers[layer][a] = []

    def degrees(self) -> Dict[str, Dict[int, int]]:
        """Return degrees per agent per layer."""
        degs: Dict[str, Dict[int, int]] = {}
        for layer in ["family", "work_school", "community", "all"]:
            degs[layer] = {i: len(nbrs) for i, nbrs in self.layers[layer].items()}
        return degs

    def diagnostics(self) -> Dict[str, Any]:
        """Compute diagnostics like degree distributions and reciprocity checks."""
        diag: Dict[str, Any] = {}
        for layer in ["family", "work_school", "community"]:
            adj = self.layers.get(layer, {})
            degs = [len(nbrs) for nbrs in adj.values()]
            reciprocity_violations = 0
            edges_checked = 0
            for i, nbrs in adj.items():
                for j in nbrs:
                    edges_checked += 1
                    if i not in adj.get(j, []):
                        reciprocity_violations += 1
            diag[layer] = {
                "num_nodes": len(adj),
                "avg_degree": float(statistics.mean(degs)) if degs else 0.0,
                "max_degree": int(max(degs)) if degs else 0,
                "reciprocity_violations": reciprocity_violations,
                "edges_checked": edges_checked
            }
        return diag

    def neighbor_share(self, layer: str, mask_state_prev: Dict[int, int]) -> Dict[int, float]:
        """Compute perceived neighbor mask share for each agent on a given layer."""
        shares: Dict[int, float] = {}
        adj = self.layers.get(layer, {})
        for i, nbrs in adj.items():
            if not nbrs:
                shares[i] = 0.0
            else:
                total = len(nbrs)
                masked = sum(mask_state_prev.get(j, 0) for j in nbrs)
                shares[i] = masked / total if total > 0 else 0.0
        return shares


@dataclass
class PolicyParams:
    """
    Role-specific logistic policy parameters.
    z = alpha + gamma*mask_prev + wf*neighbor_fam + ww*neighbor_work + wc*neighbor_comm
        + rho*(risk_perception * risk_alert_level) + mu*received_info + pi*mandate_strength
        - kappa*mask_cost
    """
    alpha: float = 0.0
    gamma: float = 1.0
    wf: float = 0.5
    ww: float = 0.5
    wc: float = 0.25
    rho: float = 0.5
    mu: float = 0.5
    pi: float = 0.5
    kappa: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "wf": self.wf,
            "ww": self.ww,
            "wc": self.wc,
            "rho": self.rho,
            "mu": self.mu,
            "pi": self.pi,
            "kappa": self.kappa,
        }


@dataclass
class InfoParams:
    """Parameters for information propagation."""
    beta_family: float = 0.1
    beta_work_school: float = 0.1
    beta_community: float = 0.05
    info_rate_broadcast: float = 0.1
    info_decay: int = 7  # days memory


@dataclass
class ScenarioSignals:
    """Daily schedules for exogenous signals."""
    mandate_strength: Dict[int, float] = field(default_factory=dict)
    media_intensity: Dict[int, float] = field(default_factory=dict)
    mask_cost: Dict[int, float] = field(default_factory=dict)
    risk_alert_level: Dict[int, float] = field(default_factory=dict)

    def get(self, signal: str, day: int, default: float = 0.0) -> float:
        schedule = getattr(self, signal, {})
        return float(schedule.get(day, default))


class DataLoader:
    """Loads agents, social network, and panel training data."""

    @staticmethod
    def load_agents(path: str) -> Dict[int, Agent]:
        """
        Load agent attributes from CSV.
        Required columns: agent_id.
        Optional: age, age_group, occupation, risk_perception, initial_mask_wearing,
                  family_connections, work_school_connections, community_connections, total_connections.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Agent file not found: {path}")
        agents: Dict[int, Agent] = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "agent_id" not in reader.fieldnames:
                raise ValueError("agent_attributes.csv missing required column 'agent_id'")
            for row in reader:
                try:
                    agent_id = int(row["agent_id"])
                except Exception as e:
                    raise ValueError(f"Invalid agent_id in row: {row}") from e
                age = None
                if "age" in row and row.get("age") not in (None, "",):
                    try:
                        age = int(row.get("age"))
                    except Exception:
                        age = None
                age_group = row.get("age_group") or None
                occupation = row.get("occupation") or None
                try:
                    risk_perception = float(row.get("risk_perception", 0.5))
                except Exception:
                    risk_perception = 0.5
                initial_mask_wearing = parse_bool(row.get("initial_mask_wearing", 0), 0)
                deg_fam = int(row.get("family_connections", 0)) if row.get("family_connections") else 0
                deg_work = int(row.get("work_school_connections", 0)) if row.get("work_school_connections") else 0
                deg_comm = int(row.get("community_connections", 0)) if row.get("community_connections") else 0
                deg_tot = int(row.get("total_connections", deg_fam + deg_work + deg_comm)) if row.get("total_connections") else (deg_fam + deg_work + deg_comm)
                role = DataLoader.derive_role(occupation, age_group)
                agents[agent_id] = Agent(
                    agent_id=agent_id,
                    age=age,
                    age_group=age_group,
                    occupation=occupation,
                    risk_perception=risk_perception,
                    role=role,
                    degree_family=deg_fam,
                    degree_work_school=deg_work,
                    degree_community=deg_comm,
                    degree_total=deg_tot,
                    initial_mask_wearing=initial_mask_wearing
                )
        return agents

    @staticmethod
    def derive_role(occupation: Optional[str], age_group: Optional[str]) -> str:
        """Derive role from occupation and age_group."""
        if occupation:
            occ = occupation.strip().lower()
            if occ == "student":
                return "Student"
            if "blue" in occ:
                return "Worker_BlueCollar"
            if "white" in occ:
                return "Worker_WhiteCollar"
        if age_group:
            ag = age_group.strip().lower()
            if ag in {"youth", "elderly"}:
                return "Nonworker"
        return "Nonworker"

    @staticmethod
    def load_network(path: str) -> MultiplexNetwork:
        """
        Load multiplex network from JSON.
        Expected: dict with keys 'family', 'work_school', 'community'. Node IDs as strings.
        """
        if not os.path.exists(path):
            print(f"Warning: network file missing at {path}; using empty network.")
            net = MultiplexNetwork(layers={"family": {}, "work_school": {}, "community": {}, "all": {}})
            return net
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layers = {}
        for layer in ["family", "work_school", "community"]:
            layer_dict = data.get(layer, {}) or {}
            converted: Dict[int, List[int]] = {}
            for k, v in layer_dict.items():
                try:
                    i = int(k)
                    converted[i] = [int(x) for x in v]
                except Exception:
                    continue
            layers[layer] = converted
        net = MultiplexNetwork(layers=layers)
        net.enforce_symmetry()
        return net

    @staticmethod
    def load_panel(path: str) -> Dict[int, Dict[int, Dict[str, int]]]:
        """
        Load panel train_data as nested dict: day -> agent_id -> { 'wearing_mask': 0/1, 'received_info': 0/1 }.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Panel data file not found: {path}")
        panel: Dict[int, Dict[int, Dict[str, int]]] = defaultdict(dict)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"day", "agent_id", "wearing_mask", "received_info"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"train_data.csv missing required columns: {required - set(reader.fieldnames or [])}")
            for row in reader:
                try:
                    day = int(row["day"])
                    agent_id = int(row["agent_id"])
                except Exception as e:
                    raise ValueError(f"Invalid day/agent_id in row: {row}") from e
                mask = parse_bool(row.get("wearing_mask", 0), 0)
                info = parse_bool(row.get("received_info", 0), 0)
                panel[day][agent_id] = {
                    "wearing_mask": mask,
                    "received_info": info
                }
        return panel

    @staticmethod
    def unique_sorted_days(panel: Dict[int, Dict[int, Dict[str, int]]]) -> List[int]:
        """Return sorted list of unique days in the panel data."""
        return sorted(panel.keys())


class Calibrator:
    """
    Calibrates logistic policy parameters per role by maximizing likelihood on training window.
    Uses gradient ascent on log-likelihood with L2 regularization.
    """

    def __init__(self, agents: Dict[int, Agent], network: MultiplexNetwork, panel: Dict[int, Dict[int, Dict[str, int]]],
                 train_days: List[int], scenario: ScenarioSignals, l2: float = 0.01, epochs: int = 200,
                 lr: float = 0.05, seed: int = 42):
        self.agents = agents
        self.network = network
        self.panel = panel
        self.train_days = train_days
        self.scenario = scenario
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.roles = ["Student", "Worker_BlueCollar", "Worker_WhiteCollar", "Nonworker"]
        self.params: Dict[str, PolicyParams] = {r: PolicyParams() for r in self.roles}
        self.training_log: Dict[str, List[float]] = {r: [] for r in self.roles}

    def build_features(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Build feature matrices X and labels y per role using training data.
        Features per role:
        - intercept (1)
        - mask_prev
        - neighbor_family_share
        - neighbor_work_school_share (Nonworker role uses 0)
        - neighbor_community_share
        - risk_perception * risk_alert_level[day]
        - received_info (observed)
        - mandate_strength[day]
        - -mask_cost[day]  (negative for correct sign)
        """
        features_by_role: Dict[str, List[List[float]]] = {r: [] for r in self.roles}
        labels_by_role: Dict[str, List[int]] = {r: [] for r in self.roles}

        days_sorted = sorted(self.train_days)
        if len(days_sorted) < 2:
            return {r: (np.zeros((0, 1)), np.zeros((0,))) for r in self.roles}

        for idx in range(1, len(days_sorted)):
            t_prev = days_sorted[idx - 1]
            t = days_sorted[idx]
            state_prev = {aid: rec["wearing_mask"] for aid, rec in self.panel.get(t_prev, {}).items()}
            state_t = self.panel.get(t, {})
            nf = self.network.neighbor_share("family", state_prev)
            nw = self.network.neighbor_share("work_school", state_prev)
            nc = self.network.neighbor_share("community", state_prev)

            for aid, obs in state_t.items():
                if aid not in self.agents:
                    continue
                if aid not in state_prev:
                    continue
                agent = self.agents[aid]
                role = agent.role if agent.role in self.roles else "Nonworker"
                mask_prev = state_prev.get(aid, 0)
                received_info = int(obs.get("received_info", 0))
                ms = float(self.scenario.get("mandate_strength", t, 0.0))
                mc = float(self.scenario.get("mask_cost", t, 0.0))
                ra = float(self.scenario.get("risk_alert_level", t, 1.0))
                features = [
                    1.0,
                    float(mask_prev),
                    float(nf.get(aid, 0.0)),
                    float(nw.get(aid, 0.0)) if role != "Nonworker" else 0.0,
                    float(nc.get(aid, 0.0)),
                    float(agent.risk_perception) * ra,
                    float(received_info),
                    ms,
                    -mc
                ]
                y = int(obs.get("wearing_mask", 0))
                features_by_role[role].append(features)
                labels_by_role[role].append(y)

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for r in self.roles:
            X = np.array(features_by_role[r], dtype=float) if features_by_role[r] else np.zeros((0, 9), dtype=float)
            y = np.array(labels_by_role[r], dtype=float) if labels_by_role[r] else np.zeros((0,), dtype=float)
            out[r] = (X, y)
        return out

    def fit(self) -> Dict[str, PolicyParams]:
        """Run gradient ascent for each role to fit parameters."""
        set_random_seed(self.seed)
        datasets = self.build_features()
        for role in self.roles:
            X, y = datasets[role]
            if X.shape[0] == 0:
                self.training_log[role].append(float("nan"))
                continue
            init = self.params[role]
            theta = np.array([
                init.alpha, init.gamma, init.wf, init.ww if role != "Nonworker" else 0.0,
                init.wc, init.rho, init.mu, init.pi, init.kappa
            ], dtype=float)
            train_mask = np.ones_like(theta, dtype=float)
            if role == "Nonworker":
                train_mask[3] = 0.0
            for epoch in range(self.epochs):
                z = X.dot(theta)
                z = np.clip(z, -35.0, 35.0)
                p = 1.0 / (1.0 + np.exp(-z))
                # Stable log-likelihood
                # log(sigmoid(z)) = -softplus(-z); log(1-sigmoid(z)) = -softplus(z)
                ll = float(np.sum(y * (-np.log1p(np.exp(-z))) + (1 - y) * (-np.log1p(np.exp(z)))) - 0.5 * self.l2 * np.sum(theta * theta))
                # Gradient
                grad = X.T.dot(y - p) - self.l2 * theta
                grad = grad * train_mask
                theta += self.lr * grad / (X.shape[0] + 1e-9)
                if epoch % 10 == 0 or epoch == self.epochs - 1:
                    self.training_log[role].append(ll)
            self.params[role] = PolicyParams(
                alpha=theta[0],
                gamma=theta[1],
                wf=theta[2],
                ww=theta[3] if role != "Nonworker" else 0.0,
                wc=theta[4],
                rho=theta[5],
                mu=theta[6],
                pi=theta[7],
                kappa=max(0.0, theta[8])
            )
        return self.params

    def export_training_log(self) -> Dict[str, List[float]]:
        return self.training_log


class Simulator:
    """
    Multi-agent simulator for mask-wearing and information diffusion.
    """

    def __init__(self, agents: Dict[int, Agent], network: MultiplexNetwork,
                 policy_params: Dict[str, PolicyParams], info_params: InfoParams,
                 scenario: ScenarioSignals, seed: int = 123):
        self.agents = agents
        self.network = network
        self.policy_params = policy_params
        self.info_params = info_params
        self.scenario = scenario
        self.seed = seed
        set_random_seed(seed)
        self.roles = ["Student", "Worker_BlueCollar", "Worker_WhiteCollar", "Nonworker"]

    def get_params_for_agent(self, agent: Agent) -> PolicyParams:
        role = agent.role if agent.role in self.roles else "Nonworker"
        return self.policy_params.get(role, PolicyParams())

    def simulate(self, days: List[int], init_mask_state: Dict[int, int],
                 observed_received_info: Optional[Dict[int, Dict[int, int]]] = None,
                 use_endogenous_info: bool = False) -> Dict[str, Any]:
        """
        Simulate over provided days (must be ordered).
        init_mask_state: agent_id -> mask at day before the first day in 'days'.
        observed_received_info: dict day -> agent_id -> info (optional).
        use_endogenous_info: if True, generate info via diffusion; otherwise use observed if provided, else broadcast only.
        Returns: dict with per-agent rows, aggregates, transitions.
        """
        if not days:
            return {}
        mask_prev = dict(init_mask_state)  # t-1 state
        info_memory: Dict[int, int] = {aid: 0 for aid in self.agents.keys()}
        per_agent_records: List[Dict[str, Any]] = []
        aggregate_by_day: Dict[int, Dict[str, Any]] = {}
        transitions_by_day: Dict[int, Dict[str, float]] = {}

        for t in days:
            nf = self.network.neighbor_share("family", mask_prev)
            nw = self.network.neighbor_share("work_school", mask_prev)
            nc = self.network.neighbor_share("community", mask_prev)

            info_today: Dict[int, int] = {}
            informed_prev = {aid for aid, mem in info_memory.items() if mem > 0}

            for aid, agent in self.agents.items():
                ms = float(self.scenario.get("mandate_strength", t, 0.0))
                media = float(self.scenario.get("media_intensity", t, self.info_params.info_rate_broadcast))
                obs_info = observed_received_info.get(t, {}).get(aid, 0) if observed_received_info else 0
                if not use_endogenous_info and observed_received_info is not None:
                    info_today[aid] = int(obs_info)
                else:
                    p_broadcast = media if media is not None else self.info_params.info_rate_broadcast
                    received = 1 if random.random() < p_broadcast else 0
                    if received == 0:
                        fam_nbrs = self.network.layers["family"].get(aid, [])
                        informed_fam = sum(1 for j in fam_nbrs if j in informed_prev)
                        p_fam = 1.0 - ((1.0 - self.info_params.beta_family) ** informed_fam) if informed_fam > 0 else 0.0
                        work_nbrs = self.network.layers["work_school"].get(aid, [])
                        informed_work = sum(1 for j in work_nbrs if j in informed_prev)
                        p_work = 1.0 - ((1.0 - self.info_params.beta_work_school) ** informed_work) if informed_work > 0 else 0.0
                        comm_nbrs = self.network.layers["community"].get(aid, [])
                        informed_comm = sum(1 for j in comm_nbrs if j in informed_prev)
                        p_comm = 1.0 - ((1.0 - self.info_params.beta_community) ** informed_comm) if informed_comm > 0 else 0.0
                        p_neighbor = 1.0 - (1.0 - p_fam) * (1.0 - p_work) * (1.0 - p_comm)
                        received = 1 if random.random() < p_neighbor else 0
                    if received == 0 and ms > 0.0:
                        received = 1 if random.random() < ms else 0
                    info_today[aid] = int(received)

            # Update info memory with decay
            for aid in info_memory.keys():
                if info_today.get(aid, 0) == 1:
                    info_memory[aid] = self.info_params.info_decay
                else:
                    info_memory[aid] = max(0, info_memory[aid] - 1)

            # Decision step
            mask_prob: Dict[int, float] = {}
            mask_today: Dict[int, int] = {}
            for aid, agent in self.agents.items():
                params = self.get_params_for_agent(agent)
                ms = float(self.scenario.get("mandate_strength", t, 0.0))
                mc = float(self.scenario.get("mask_cost", t, 0.0))
                ra = float(self.scenario.get("risk_alert_level", t, 1.0))
                mprev = float(mask_prev.get(aid, agent.initial_mask_wearing))
                # Incorporate info memory (boost if memory > 0)
                recinfo = 1.0 if info_memory.get(aid, 0) > 0 else 0.0
                x = params.alpha
                x += params.gamma * mprev
                x += params.wf * float(nf.get(aid, 0.0))
                x += params.ww * float(nw.get(aid, 0.0))
                x += params.wc * float(nc.get(aid, 0.0))
                x += params.rho * float(agent.risk_perception) * ra
                x += params.mu * recinfo
                x += params.pi * ms
                x += -params.kappa * mc
                p = sigmoid(x)
                mask_prob[aid] = p
                mask_today[aid] = 1 if random.random() < p else 0
                per_agent_records.append({
                    "day": t,
                    "agent_id": aid,
                    "role": agent.role,
                    "age_group": agent.age_group if agent.age_group else "",
                    "occupation": agent.occupation if agent.occupation else "",
                    "received_info": info_today.get(aid, 0),
                    "wearing_mask": mask_today[aid],
                    "prob_mask": p,
                    "neighbor_family_share": float(nf.get(aid, 0.0)),
                    "neighbor_work_share": float(nw.get(aid, 0.0)),
                    "neighbor_comm_share": float(nc.get(aid, 0.0)),
                })

            mask_vals = list(mask_today.values())
            overall_rate = sum(mask_vals) / len(mask_vals) if mask_vals else 0.0
            by_role_counts: Dict[str, List[int]] = defaultdict(list)
            for aid, m in mask_today.items():
                by_role_counts[self.agents[aid].role].append(m)
            role_rates = {r: (sum(v) / len(v) if v else 0.0) for r, v in by_role_counts.items()}
            aggregate_by_day[t] = {
                "day": t,
                "overall_rate": overall_rate,
                **{f"rate_{r}": role_rates.get(r, 0.0) for r in self.roles}
            }

            counts = Counter()
            for aid in self.agents.keys():
                prev = mask_prev.get(aid, 0)
                cur = mask_today.get(aid, 0)
                if prev == 1 and cur == 1:
                    counts["P11"] += 1
                elif prev == 1 and cur == 0:
                    counts["P10"] += 1
                elif prev == 0 and cur == 1:
                    counts["P01"] += 1
                else:
                    counts["P00"] += 1
            total_agents = len(self.agents)
            transitions_by_day[t] = {k: counts.get(k, 0) / total_agents for k in ["P11", "P10", "P01", "P00"]}

            mask_prev = mask_today

        result = {
            "per_agent": per_agent_records,
            "aggregate_by_day": aggregate_by_day,
            "transitions_by_day": transitions_by_day,
        }
        return result


class Evaluator:
    """Computes evaluation metrics comparing simulation outputs to observed validation data."""

    def __init__(self, agents: Dict[int, Agent]):
        self.agents = agents
        self.roles = ["Student", "Worker_BlueCollar", "Worker_WhiteCollar", "Nonworker"]

    @staticmethod
    def compute_aggregate_curve(panel_subset: Dict[int, Dict[int, Dict[str, int]]], agents: Dict[int, Agent]) -> Dict[int, float]:
        """Compute aggregate mask-wearing rate per day from panel."""
        agg: Dict[int, float] = {}
        for day, recs in panel_subset.items():
            vals = [int(r["wearing_mask"]) for _, r in recs.items()]
            agg[day] = sum(vals) / len(vals) if vals else 0.0
        return agg

    @staticmethod
    def compute_aggregate_curve_by_role(panel_subset: Dict[int, Dict[int, Dict[str, int]]], agents: Dict[int, Agent]) -> Dict[int, Dict[str, float]]:
        """
        Compute per-role adoption rates per day from panel.
        Returns: dict day -> dict role -> rate in [0,1]
        """
        out: Dict[int, Dict[str, float]] = {}
        for day, recs in panel_subset.items():
            role_counts: Dict[str, List[int]] = defaultdict(list)
            for aid, r in recs.items():
                role = agents.get(aid).role if aid in agents else "Nonworker"
                role_counts[role].append(int(r["wearing_mask"]))
            out[day] = {role: (sum(vals) / len(vals) if vals else 0.0) for role, vals in role_counts.items()}
        return out

    @staticmethod
    def rmse(series_pred: Dict[int, float], series_true: Dict[int, float]) -> float:
        days = sorted(set(series_pred.keys()) & set(series_true.keys()))
        if not days:
            return float("nan")
        se = [(series_pred[d] - series_true[d]) ** 2 for d in days]
        return math.sqrt(sum(se) / len(se))

    @staticmethod
    def mae(series_pred: Dict[int, float], series_true: Dict[int, float]) -> float:
        days = sorted(set(series_pred.keys()) & set(series_true.keys()))
        if not days:
            return float("nan")
        ae = [abs(series_pred[d] - series_true[d]) for d in days]
        return sum(ae) / len(ae)

    def brier(self, per_agent_records: List[Dict[str, Any]], panel_subset: Dict[int, Dict[int, Dict[str, int]]], days: List[int]) -> Tuple[float, Dict[str, float]]:
        """Compute Brier score overall and by role for validation days."""
        # Map day->agent->(p, role)
        records_by_day: Dict[int, Dict[int, Tuple[float, str]]] = defaultdict(dict)
        for rec in per_agent_records:
            d = rec["day"]
            if d in days:
                records_by_day[d][rec["agent_id"]] = (float(rec["prob_mask"]), str(rec["role"]))
        scores: List[float] = []
        scores_by_role: Dict[str, List[float]] = defaultdict(list)
        for d in days:
            obs_day = panel_subset.get(d, {})
            pred_day = records_by_day.get(d, {})
            for aid, obs in obs_day.items():
                if aid in pred_day:
                    p, role = pred_day[aid]
                    y = int(obs["wearing_mask"])
                    s = (p - y) ** 2
                    scores.append(s)
                    scores_by_role[role].append(s)
        overall = float(sum(scores) / len(scores)) if scores else float("nan")
        by_role = {r: (sum(v) / len(v) if v else float("nan")) for r, v in scores_by_role.items()}
        return overall, by_role

    def transition_fit(self, per_agent_records: List[Dict[str, Any]], panel_subset: Dict[int, Dict[int, Dict[str, int]]], days: List[int]) -> Dict[str, Any]:
        """
        Compare simulated vs observed transition proportions per day.
        Returns dict with per-day absolute errors and averages, overall and by role.
        """
        # Build simulated mask by day->aid
        sim_by_day: Dict[int, Dict[int, int]] = defaultdict(dict)
        role_by_aid = {aid: ag.role for aid, ag in self.agents.items()}
        for rec in per_agent_records:
            d = rec["day"]
            if d in days:
                sim_by_day[d][rec["agent_id"]] = int(rec["wearing_mask"])
        # Observed mask by day->aid
        obs_by_day: Dict[int, Dict[int, int]] = {
            d: {aid: int(r["wearing_mask"]) for aid, r in panel_subset.get(d, {}).items()}
            for d in days
        }

        def compute_transitions(prev_states: Dict[int, int], cur_states: Dict[int, int], ids: List[int]) -> Dict[str, float]:
            counts = Counter()
            n = len(ids)
            if n == 0:
                return {k: 0.0 for k in ["P11", "P10", "P01", "P00"]}
            for aid in ids:
                prev = prev_states.get(aid, 0)
                cur = cur_states.get(aid, 0)
                if prev == 1 and cur == 1:
                    counts["P11"] += 1
                elif prev == 1 and cur == 0:
                    counts["P10"] += 1
                elif prev == 0 and cur == 1:
                    counts["P01"] += 1
                else:
                    counts["P00"] += 1
            return {k: counts.get(k, 0) / n for k in ["P11", "P10", "P01", "P00"]}

        per_day_errors: Dict[int, Dict[str, Any]] = {}
        sorted_days = sorted(days)
        for i in range(1, len(sorted_days)):
            t_prev = sorted_days[i - 1]
            t = sorted_days[i]
            sim_prev = sim_by_day.get(t_prev, {})
            sim_cur = sim_by_day.get(t, {})
            obs_prev = obs_by_day.get(t_prev, {})
            obs_cur = obs_by_day.get(t, {})

            common_ids = list(set(sim_prev.keys()) & set(sim_cur.keys()) & set(obs_prev.keys()) & set(obs_cur.keys()))

            sim_overall = compute_transitions(sim_prev, sim_cur, common_ids)
            obs_overall = compute_transitions(obs_prev, obs_cur, common_ids)
            overall_err = {k: abs(sim_overall.get(k, 0.0) - obs_overall.get(k, 0.0)) for k in ["P11", "P10", "P01", "P00"]}

            # By role
            by_role_err: Dict[str, Dict[str, float]] = {}
            for role in self.roles:
                role_ids = [aid for aid in common_ids if role_by_aid.get(aid, "Nonworker") == role]
                sim_r = compute_transitions(sim_prev, sim_cur, role_ids)
                obs_r = compute_transitions(obs_prev, obs_cur, role_ids)
                by_role_err[role] = {k: abs(sim_r.get(k, 0.0) - obs_r.get(k, 0.0)) for k in ["P11", "P10", "P01", "P00"]}

            per_day_errors[t] = {
                "overall": overall_err,
                "by_role": by_role_err
            }

        # Averages across days
        avg_overall = {k: 0.0 for k in ["P11", "P10", "P01", "P00"]}
        avg_by_role: Dict[str, Dict[str, float]] = {role: {k: 0.0 for k in ["P11", "P10", "P01", "P00"]} for role in self.roles}
        num_days = len(per_day_errors)
        if num_days > 0:
            for t, rec in per_day_errors.items():
                for k in avg_overall.keys():
                    avg_overall[k] += rec["overall"].get(k, 0.0)
                for role in self.roles:
                    for k in avg_by_role[role].keys():
                        avg_by_role[role][k] += rec["by_role"].get(role, {}).get(k, 0.0)
            for k in avg_overall.keys():
                avg_overall[k] /= num_days
            for role in self.roles:
                for k in avg_by_role[role].keys():
                    avg_by_role[role][k] /= num_days

        return {
            "per_day": per_day_errors,
            "avg_overall": avg_overall,
            "avg_by_role": avg_by_role
        }


# Execute main for both direct execution and sandbox wrapper invocation
main()