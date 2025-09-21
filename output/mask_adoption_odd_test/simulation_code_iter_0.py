#!/usr/bin/env python3
"""
Simulate and calibrate the diffusion of mask-wearing behavior over a multiplex social
network (family, work/school, community), driven by individual risk perception,
information exposure, and peer influence.

This program:
- Loads agent attributes, a multiplex social network, and panel outcomes.
- Supports both 80/20 temporal validation and multi-day forecast beyond the observed range.
- Calibrates parameters via teacher-forced likelihood on the training window using a
  deterministic, bounded, derivative-free multi-start local search.
- Rolls the simulator forward on the validation/forecast range, producing stochastic replications.
- Computes evaluation metrics (RMSE/MAE for aggregate mask rates, Brier score, and
  transition fit) with 95% CIs from the replications.
- Saves calibrated parameters, approximate confidence intervals, diagnostics, and
  simulated time series to the specified data directory.

CLI path handling:
- --data_dir points to the folder containing the three data files (defaults to task's data_folder).

Data files expected in data_dir:
- agent_attributes.csv with columns: agent_id, age_group, occupation, risk_perception
- social_network.json either of:
  (A) Node-centric dict: {agent_id: {"family": [...], "work_school": [...], "community": [...], "all": [...]}}
  (B) Layer-centric dict: {"family": {node: [neighbors], ...}, "work_school": {...}, "community": {...}}
- train_data.csv with columns: day, agent_id, received_info, wearing_mask

Usage:
    python simulate.py --seed 42 --replications 50 --starts 3 --max_iter 120 \
        --data_dir data_fitting/mask_adoption_data/ --forecast_horizon 10 \
        --policy_step_day 10 --policy_level 1.0 --symmetrize_edges 1 --save_agent_trajectories

Notes:
- Deterministic behavior is ensured via a global random seed.
- The calibration procedure uses a derivative-free multi-start local search with bounds and light regularization.
- Confidence intervals are approximated via univariate curvature and (optional) day-weighted bootstrap.

Author: Code Generation Agent (Improved)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "Missing required dependency 'pandas'. Please install it (e.g., pip install pandas) "
        "to run this program."
    ) from e


# -----------------------
# Utility Functions
# -----------------------

def set_global_seed(seed: int) -> np.random.Generator:
    """
    Set global deterministic seeds for numpy and random.
    """
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer.")
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    return rng


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p_clip / (1.0 - p_clip))


def safe_log(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-12, 1.0)
    return np.log(p_clip)


def ensure_file_exists(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")


def to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Value {x} cannot be converted to int.") from e


# -----------------------
# Data Structures
# -----------------------

LAYER_NAMES = ["family", "work_school", "community"]
AGE_GROUPS = ["Youth", "YoungAdult", "MiddleAge"]
OCC_GROUPS = ["Student", "Blue Collar", "White Collar"]


@dataclass
class DataContainer:
    """
    Container for simulation data and metadata.
    """
    agent_ids: List[int]
    id_to_index: Dict[int, int]
    age_group: np.ndarray
    occupation: np.ndarray
    risk_perception: np.ndarray
    risk_transformed: np.ndarray
    days: List[int]
    day_to_index: Dict[int, int]
    Y_obs: np.ndarray
    I_obs: np.ndarray
    policy_signal: np.ndarray  # can be longer than observed T if forecasting


class Network:
    """
    Multiplex social network supporting layer-specific neighbor lists and multi-context flags.
    """

    def __init__(
        self,
        id_to_index: Dict[int, int],
        layer_adjacency: Dict[str, Dict[int, List[int]]],
        symmetrize: bool = True,
    ):
        self.id_to_index = id_to_index
        self.index_to_id = {v: k for k, v in id_to_index.items()}
        self.n_agents = len(id_to_index)
        self.layers = LAYER_NAMES

        # Normalize and symmetrize adjacency per layer
        self.adj_index: Dict[str, List[np.ndarray]] = {}
        for layer in self.layers:
            adj = layer_adjacency.get(layer, {})
            if not isinstance(adj, dict):
                raise ValueError(f"Layer '{layer}' adjacency must be a dict of node -> neighbors.")
            adj_clean = self._normalize_adjacency(adj, symmetrize=symmetrize)
            self.adj_index[layer] = self._adjacency_to_index_lists(adj_clean)

        # Multi-context flags
        self.multi_flags: Dict[str, List[np.ndarray]] = self._build_multi_context_flags()

        # Degrees per layer
        self.degrees: Dict[str, np.ndarray] = {layer: self._compute_degrees(self.adj_index[layer])
                                               for layer in self.layers}

    @staticmethod
    def _normalize_adjacency(adj: Dict[Any, List[Any]], symmetrize: bool) -> Dict[int, List[int]]:
        clean: Dict[int, set] = {}
        for k, neighs in adj.items():
            try:
                i = int(k)
            except Exception as e:
                raise ValueError(f"Adjacency key '{k}' is not convertible to int.") from e
            if not isinstance(neighs, (list, tuple, set)):
                raise ValueError(f"Adjacency list for node {k} must be a list/tuple/set.")
            s = set()
            for n in neighs:
                try:
                    j = int(n)
                except Exception as e:
                    raise ValueError(f"Neighbor '{n}' for node {k} not convertible to int.") from e
                if j != i:
                    s.add(j)
            clean[i] = s

        if symmetrize:
            for i, neighs in list(clean.items()):
                for j in list(neighs):
                    if j not in clean:
                        clean[j] = set()
                    clean[j].add(i)

        out: Dict[int, List[int]] = {i: sorted(list(neighs)) for i, neighs in clean.items()}
        return out

    def _adjacency_to_index_lists(self, adj: Dict[int, List[int]]) -> List[np.ndarray]:
        N = self.n_agents
        out: List[np.ndarray] = []
        for i_idx in range(N):
            i_id = self.index_to_id[i_idx]
            neigh_ids = adj.get(i_id, [])
            neigh_indices = [self.id_to_index[j] for j in neigh_ids if j in self.id_to_index]
            neigh_indices = sorted(set([j for j in neigh_indices if j != i_idx]))
            out.append(np.array(neigh_indices, dtype=np.int32))
        return out

    @staticmethod
    def _compute_degrees(adj_index: List[np.ndarray]) -> np.ndarray:
        return np.array([len(neigh) for neigh in adj_index], dtype=np.int32)

    def _build_multi_context_flags(self) -> Dict[str, List[np.ndarray]]:
        pair_counts: Dict[Tuple[int, int], int] = {}
        for layer in self.layers:
            lst = self.adj_index[layer]
            for i, neighs in enumerate(lst):
                for j in neighs:
                    u, v = (i, j) if i < j else (j, i)
                    pair_counts[(u, v)] = pair_counts.get((u, v), 0) + 1

        multi: Dict[str, List[np.ndarray]] = {}
        for layer in self.layers:
            flags: List[np.ndarray] = []
            lst = self.adj_index[layer]
            for i, neighs in enumerate(lst):
                flag = np.zeros(len(neighs), dtype=bool)
                for idx, j in enumerate(neighs):
                    u, v = (i, j) if i < j else (j, i)
                    flag[idx] = pair_counts.get((u, v), 0) >= 2
                flags.append(flag)
            multi[layer] = flags
        return multi

    def compute_layer_fraction(
        self, state: np.ndarray, layer: str, multi_context_bonus: float
    ) -> np.ndarray:
        if layer not in self.layers:
            raise ValueError(f"Unknown layer '{layer}'. Expected one of {self.layers}.")
        st = state.copy()
        st[np.isnan(st)] = 0.0

        N = self.n_agents
        out = np.zeros(N, dtype=float)
        deg = self.degrees[layer]
        neigh_lists = self.adj_index[layer]
        flags = self.multi_flags[layer]

        if multi_context_bonus < 0 or multi_context_bonus > 1:
            raise ValueError("multi_context_bonus must be in [0,1].")

        for i in range(N):
            neigh = neigh_lists[i]
            if neigh.size == 0:
                out[i] = 0.0
                continue
            base_sum = st[neigh].sum()
            if multi_context_bonus > 0.0:
                multi_mask = flags[i]
                if multi_mask.size > 0:
                    base_sum += multi_context_bonus * st[neigh[multi_mask]].sum()
            out[i] = base_sum / float(max(1, int(deg[i])))
        return out

    def compute_peer_norms(
        self, wearing_prev: np.ndarray, multi_context_bonus: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pf = self.compute_layer_fraction(wearing_prev, "family", multi_context_bonus)
        pw = self.compute_layer_fraction(wearing_prev, "work_school", multi_context_bonus)
        pc = self.compute_layer_fraction(wearing_prev, "community", multi_context_bonus)
        return pf, pw, pc

    def compute_info_exposure_fractions(
        self, info_prev: np.ndarray, multi_context_bonus: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ef = self.compute_layer_fraction(info_prev, "family", multi_context_bonus)
        ew = self.compute_layer_fraction(info_prev, "work_school", multi_context_bonus)
        ec = self.compute_layer_fraction(info_prev, "community", multi_context_bonus)
        return ef, ew, ec


@dataclass
class ParameterSet:
    """
    Model parameters with bounds and convenience methods for transformation.
    """
    # Peer layer weights
    w_family: float = 0.33
    w_work_school: float = 0.33
    w_community: float = 0.33

    # Mask adoption logistic coefficients
    alpha_peer: float = 1.0
    alpha_inertia: float = 2.0
    alpha_info: float = 1.0
    alpha_risk: float = 0.5
    alpha0: float = 0.0

    # Fixed effects (free parameters)
    fe_age_youth: float = 0.0
    fe_age_youngadult: float = 0.0  # fe_age_middleage = -(youth + youngadult)

    fe_occ_student: float = 0.0
    fe_occ_bluecollar: float = 0.0  # fe_occ_whitecollar = -(student + bluecollar)

    gamma_policy: float = 0.0

    # Info propagation parameters
    beta_family: float = 0.2
    beta_work_school: float = 0.2
    beta_community: float = 0.2
    beta0_segments: np.ndarray = field(default_factory=lambda: np.array([0.05], dtype=float))
    rho_info_decay: float = 0.01

    # Network and noise parameters
    multi_context_bonus: float = 0.2
    temperature_noise: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "w_family": self.w_family,
            "w_work_school": self.w_work_school,
            "w_community": self.w_community,
            "alpha_peer": self.alpha_peer,
            "alpha_inertia": self.alpha_inertia,
            "alpha_info": self.alpha_info,
            "alpha_risk": self.alpha_risk,
            "alpha0": self.alpha0,
            "FE_age_Youth": self.FE_age_dict()["Youth"],
            "FE_age_YoungAdult": self.FE_age_dict()["YoungAdult"],
            "FE_age_MiddleAge": self.FE_age_dict()["MiddleAge"],
            "FE_occ_Student": self.FE_occ_dict()["Student"],
            "FE_occ_BlueCollar": self.FE_occ_dict()["Blue Collar"],
            "FE_occ_WhiteCollar": self.FE_occ_dict()["White Collar"],
            "gamma_policy": self.gamma_policy,
            "beta_family": self.beta_family,
            "beta_work_school": self.beta_work_school,
            "beta_community": self.beta_community,
            "beta0_segments": self.beta0_segments.tolist(),
            "rho_info_decay": self.rho_info_decay,
            "multi_context_bonus": self.multi_context_bonus,
            "temperature_noise": self.temperature_noise,
        }
        return d

    def FE_age_dict(self) -> Dict[str, float]:
        youth = self.fe_age_youth
        young = self.fe_age_youngadult
        middle = -(youth + young)
        return {"Youth": youth, "YoungAdult": young, "MiddleAge": middle}

    def FE_occ_dict(self) -> Dict[str, float]:
        student = self.fe_occ_student
        blue = self.fe_occ_bluecollar
        white = -(student + blue)
        return {"Student": student, "Blue Collar": blue, "White Collar": white}

    def build_fe_vectors(self, data: DataContainer) -> Tuple[np.ndarray, np.ndarray]:
        fe_age_map = self.FE_age_dict()
        fe_occ_map = self.FE_occ_dict()
        N = len(data.agent_ids)
        fe_age = np.zeros(N, dtype=float)
        fe_occ = np.zeros(N, dtype=float)
        for i in range(N):
            key_age = str(data.age_group[i])
            key_occ = str(data.occupation[i])
            fe_age[i] = fe_age_map.get(key_age, fe_age_map.get(key_age.replace(" ", ""), 0.0))
            fe_occ[i] = fe_occ_map.get(key_occ, fe_occ_map.get(key_occ.replace(" ", ""), 0.0))
        return fe_age, fe_occ

    def enforce_bounds(self) -> None:
        self.w_family = float(np.clip(self.w_family, 0.0, 1.0))
        self.w_work_school = float(np.clip(self.w_work_school, 0.0, 1.0))
        self.w_community = float(np.clip(self.w_community, 0.0, 1.0))
        s = self.w_family + self.w_work_school + self.w_community
        if s > 1.0 and s > 0.0:
            self.w_family /= s
            self.w_work_school /= s
            self.w_community /= s

        self.alpha_peer = float(np.clip(self.alpha_peer, 0.0, 6.0))
        self.alpha_inertia = float(np.clip(self.alpha_inertia, 0.0, 6.0))
        self.alpha_info = float(np.clip(self.alpha_info, 0.0, 6.0))
        self.alpha_risk = float(np.clip(self.alpha_risk, -6.0, 6.0))
        self.alpha0 = float(np.clip(self.alpha0, -6.0, 6.0))

        self.fe_age_youth = float(np.clip(self.fe_age_youth, -3.0, 3.0))
        self.fe_age_youngadult = float(np.clip(self.fe_age_youngadult, -3.0, 3.0))
        self.fe_occ_student = float(np.clip(self.fe_occ_student, -3.0, 3.0))
        self.fe_occ_bluecollar = float(np.clip(self.fe_occ_bluecollar, -3.0, 3.0))

        self.gamma_policy = float(np.clip(self.gamma_policy, 0.0, 6.0))
        self.beta_family = float(np.clip(self.beta_family, 0.0, 2.0))
        self.beta_work_school = float(np.clip(self.beta_work_school, 0.0, 2.0))
        self.beta_community = float(np.clip(self.beta_community, 0.0, 2.0))

        self.beta0_segments = np.clip(self.beta0_segments, 0.0, 2.0).astype(float)
        self.rho_info_decay = float(np.clip(self.rho_info_decay, 0.0, 0.5))
        self.multi_context_bonus = float(np.clip(self.multi_context_bonus, 0.0, 1.0))
        self.temperature_noise = float(np.clip(self.temperature_noise, 0.5, 2.0))

    @staticmethod
    def bounds_spec(num_segments: int, include_policy: bool = True) -> List[Tuple[float, float]]:
        bounds = []
        bounds += [(0.0, 1.0)] * 3
        bounds += [(0.0, 6.0), (0.0, 6.0), (0.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)]
        bounds += [(-3.0, 3.0), (-3.0, 3.0)]
        bounds += [(-3.0, 3.0), (-3.0, 3.0)]
        bounds += [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
        bounds += [(0.0, 2.0)] * num_segments
        bounds += [(0.0, 0.5)]
        bounds += [(0.0, 1.0)]
        if include_policy:
            bounds += [(0.0, 6.0)]
        bounds += [(0.5, 2.0)]
        return bounds

    @staticmethod
    def from_vector(vec: np.ndarray, num_segments: int, include_policy: bool = True) -> "ParameterSet":
        idx = 0
        w_f, w_ws, w_c = float(vec[idx]), float(vec[idx + 1]), float(vec[idx + 2])
        idx += 3
        alpha_peer, alpha_inertia, alpha_info, alpha_risk, alpha0 = (
            float(vec[idx]),
            float(vec[idx + 1]),
            float(vec[idx + 2]),
            float(vec[idx + 3]),
            float(vec[idx + 4]),
        )
        idx += 5
        fe_age_youth, fe_age_youngadult = float(vec[idx]), float(vec[idx + 1])
        idx += 2
        fe_occ_student, fe_occ_bluecollar = float(vec[idx]), float(vec[idx + 1])
        idx += 2
        beta_family, beta_work_school, beta_community = float(vec[idx]), float(vec[idx + 1]), float(vec[idx + 2])
        idx += 3
        beta0_segments = np.array(vec[idx: idx + num_segments], dtype=float)
        idx += num_segments
        rho_info_decay = float(vec[idx])
        idx += 1
        multi_context_bonus = float(vec[idx])
        idx += 1
        if include_policy:
            gamma_policy = float(vec[idx])
            idx += 1
        else:
            gamma_policy = 0.0
        temperature_noise = float(vec[idx])

        ps = ParameterSet(
            w_family=w_f,
            w_work_school=w_ws,
            w_community=w_c,
            alpha_peer=alpha_peer,
            alpha_inertia=alpha_inertia,
            alpha_info=alpha_info,
            alpha_risk=alpha_risk,
            alpha0=alpha0,
            fe_age_youth=fe_age_youth,
            fe_age_youngadult=fe_age_youngadult,
            fe_occ_student=fe_occ_student,
            fe_occ_bluecollar=fe_occ_bluecollar,
            gamma_policy=gamma_policy,
            beta_family=beta_family,
            beta_work_school=beta_work_school,
            beta_community=beta_community,
            beta0_segments=beta0_segments,
            rho_info_decay=rho_info_decay,
            multi_context_bonus=multi_context_bonus,
            temperature_noise=temperature_noise,
        )
        ps.enforce_bounds()
        return ps

    def to_vector(self, include_policy: bool = True) -> np.ndarray:
        parts: List[float] = [
            self.w_family, self.w_work_school, self.w_community,
            self.alpha_peer, self.alpha_inertia, self.alpha_info, self.alpha_risk, self.alpha0,
            self.fe_age_youth, self.fe_age_youngadult,
            self.fe_occ_student, self.fe_occ_bluecollar,
            self.beta_family, self.beta_work_school, self.beta_community,
        ]
        parts += list(self.beta0_segments)
        parts += [self.rho_info_decay, self.multi_context_bonus]
        if include_policy:
            parts += [self.gamma_policy]
        parts += [self.temperature_noise]
        return np.array(parts, dtype=float)


# -----------------------
# CLI
# -----------------------

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask-wearing diffusion simulator with calibration and forecast.")
    parser.add_argument("--data_dir", type=str, default="data_fitting/mask_adoption_data/",
                        help="Directory containing agent_attributes.csv, social_network.json, train_data.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic behavior.")
    parser.add_argument("--replications", type=int, default=50, help="Number of stochastic replications for validation/forecast metrics.")
    parser.add_argument("--starts", type=int, default=3, help="Number of multi-start initializations in calibration.")
    parser.add_argument("--max_iter", type=int, default=120, help="Maximum iterations per start in calibration.")
    parser.add_argument("--segments", type=int, default=None, help="Number of beta0_t baseline segments (default: inferred).")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap runs for CI estimation (day-weighted).")
    parser.add_argument("--policy_constant", type=float, default=0.0, help="Constant mask mandate signal in [0,1] if no step schedule provided.")
    parser.add_argument("--policy_step_day", type=int, default=10, help="Day index at which policy steps to --policy_level (default 10). Use negative to disable.")
    parser.add_argument("--policy_level", type=float, default=1.0, help="Policy level after step day in [0,1].")
    parser.add_argument("--policy_end_day", type=int, default=-1, help="Optional end day (exclusive) for policy step; -1 for no end (persistent).")
    parser.add_argument("--forecast_horizon", type=int, default=10, help="Number of days to forecast beyond last observed day.")
    parser.add_argument("--symmetrize_edges", type=int, default=1, help="Whether to symmetrize edges (1) or not (0).")
    parser.add_argument("--save_agent_trajectories", action="store_true", help="Save per-agent deterministic probability paths for validation/forecast days.")
    parser.add_argument("--output_prefix", type=str, default="results", help="Prefix for output files (saved under data_dir).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    # Validate args
    if args.replications <= 0:
        raise ValueError("replications must be positive.")
    if args.starts <= 0:
        raise ValueError("starts must be positive.")
    if args.max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if args.segments is not None and args.segments <= 0:
        raise ValueError("segments must be positive if provided.")
    if not (0.0 <= args.policy_constant <= 1.0):
        raise ValueError("policy_constant must be in [0,1].")
    if not (0.0 <= args.policy_level <= 1.0):
        raise ValueError("policy_level must be in [0,1].")
    if args.forecast_horizon < 0:
        raise ValueError("forecast_horizon must be >= 0.")
    if args.symmetrize_edges not in (0, 1):
        raise ValueError("symmetrize_edges must be 0 or 1.")

    return args


# -----------------------
# Data Loading
# -----------------------

def _normalize_age_group(x: Any) -> str:
    if isinstance(x, str):
        s = x.strip()
        for a in AGE_GROUPS:
            if s.lower().replace(" ", "") == a.lower():
                return a
        if s.lower() in {"youth", "teen", "teenager", "child"}:
            return "Youth"
        if s.lower().replace(" ", "") in {"youngadult", "young"}:
            return "YoungAdult"
    return "MiddleAge"


def _normalize_occupation(x: Any) -> str:
    if isinstance(x, str):
        s = x.strip()
        for o in OCC_GROUPS:
            if s.lower() == o.lower():
                return o
        if s.lower() in {"student", "pupil"}:
            return "Student"
        if s.lower().replace(" ", "") in {"bluecollar", "labor", "laborer", "worker"}:
            return "Blue Collar"
    return "White Collar"


def _transform_risk(risk: np.ndarray) -> np.ndarray:
    r = risk.astype(float)
    if np.nanmin(r) < 0.0 or np.nanmax(r) > 1.0:
        rmin = np.nanmin(r)
        rmax = np.nanmax(r)
        if rmax - rmin < 1e-12:
            r = np.zeros_like(r) + 0.5
        else:
            r = (r - rmin) / (rmax - rmin)
        print("Warning: risk_perception values were outside [0,1]; applied min-max normalization.", file=sys.stderr)
    r = np.clip(r, 1e-6, 1.0 - 1e-6)
    return logit(r)


def _extract_nodes_from_network_json(net_json: Dict[str, Any]) -> set:
    nodes = set()
    if all(k in net_json for k in LAYER_NAMES):
        # Layer-centric
        for layer in LAYER_NAMES:
            layer_obj = net_json.get(layer, {})
            if isinstance(layer_obj, dict):
                for k, neighs in layer_obj.items():
                    try:
                        i = int(k)
                        nodes.add(i)
                    except Exception:
                        continue
                    if isinstance(neighs, (list, tuple, set)):
                        for n in neighs:
                            try:
                                nodes.add(int(n))
                            except Exception:
                                continue
            elif isinstance(layer_obj, list):
                for edge in layer_obj:
                    if isinstance(edge, (list, tuple)) and len(edge) == 2:
                        try:
                            i, j = int(edge[0]), int(edge[1])
                            nodes.add(i); nodes.add(j)
                        except Exception:
                            continue
    else:
        # Node-centric
        for k, v in net_json.items():
            try:
                i = int(k)
                nodes.add(i)
            except Exception:
                continue
            if isinstance(v, dict):
                for layer in LAYER_NAMES:
                    lst = v.get(layer, [])
                    if isinstance(lst, (list, tuple, set)):
                        for n in lst:
                            try:
                                nodes.add(int(n))
                            except Exception:
                                continue
    return nodes


def _standardize_layer_adjacency(layer_data: Any) -> Dict[int, List[int]]:
    if isinstance(layer_data, dict):
        out: Dict[int, List[int]] = {}
        for k, neighs in layer_data.items():
            try:
                i = int(k)
            except Exception as e:
                raise ValueError(f"Invalid node id in network: {k}") from e
            if not isinstance(neighs, (list, tuple, set)):
                raise ValueError(f"Neighbors for node {k} must be list/tuple/set.")
            arr = []
            for n in neighs:
                try:
                    j = int(n)
                except Exception as e:
                    raise ValueError(f"Invalid neighbor id {n} for node {k}") from e
                if j != i:
                    arr.append(j)
            out[i] = sorted(list(set(arr)))
        return out
    elif isinstance(layer_data, list):
        adj: Dict[int, set] = {}
        for edge in layer_data:
            if not (isinstance(edge, (list, tuple)) and len(edge) == 2):
                raise ValueError("Each edge in edge-list must be a pair [i, j].")
            try:
                i, j = int(edge[0]), int(edge[1])
            except Exception as e:
                raise ValueError(f"Invalid edge {edge}; cannot convert to int.") from e
            if i == j:
                continue
            if i not in adj:
                adj[i] = set()
            if j not in adj:
                adj[j] = set()
            adj[i].add(j)
        return {i: sorted(list(neighs)) for i, neighs in adj.items()}
    else:
        raise ValueError("Layer adjacency must be a dict or a list of edges.")


def _parse_network_json_into_layers(net_json: Dict[str, Any]) -> Dict[str, Dict[int, List[int]]]:
    """
    Support both formats:
      - Layer-centric: top-level keys 'family','work_school','community'
      - Node-centric: top-level keys agent_id -> {layer: [neighbors]}
    """
    layer_adjacency_ids: Dict[str, Dict[int, List[int]]] = {layer: {} for layer in LAYER_NAMES}
    if all(k in net_json for k in LAYER_NAMES):
        for layer in LAYER_NAMES:
            layer_adjacency_ids[layer] = _standardize_layer_adjacency(net_json.get(layer, {}))
        return layer_adjacency_ids

    # Node-centric
    for node_key, payload in net_json.items():
        try:
            i = int(node_key)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for layer in LAYER_NAMES:
            neighs = payload.get(layer, [])
            if not isinstance(neighs, (list, tuple, set)):
                continue
            lst: List[int] = []
            for n in neighs:
                try:
                    j = int(n)
                except Exception:
                    continue
                if j != i:
                    lst.append(j)
            if i not in layer_adjacency_ids[layer]:
                layer_adjacency_ids[layer][i] = sorted(list(set(lst)))
            else:
                # Merge if duplicate
                merged = set(layer_adjacency_ids[layer][i]).union(lst)
                layer_adjacency_ids[layer][i] = sorted(list(merged))
    return layer_adjacency_ids


def load_data(args: argparse.Namespace, data_dir: str) -> Tuple[DataContainer, Dict[str, Dict[int, List[int]]]]:
    """
    Load agents, network JSON, and panel outcomes.
    """
    agent_file = os.path.join(data_dir, "agent_attributes.csv")
    network_file = os.path.join(data_dir, "social_network.json")
    train_file = os.path.join(data_dir, "train_data.csv")

    ensure_file_exists(agent_file)
    ensure_file_exists(network_file)
    ensure_file_exists(train_file)

    agents_df = pd.read_csv(agent_file)
    events_df = pd.read_csv(train_file)
    with open(network_file, "r") as f:
        net_json = json.load(f)

    required_agent_cols = {"agent_id", "age_group", "occupation", "risk_perception"}
    if not required_agent_cols.issubset(set(agents_df.columns)):
        missing = required_agent_cols - set(agents_df.columns)
        raise ValueError(f"agent_attributes.csv is missing required columns: {missing}")

    agents_df = agents_df.copy()
    agents_df["agent_id"] = agents_df["agent_id"].apply(to_int)
    agents_df["age_group"] = agents_df["age_group"].apply(_normalize_age_group)
    agents_df["occupation"] = agents_df["occupation"].apply(_normalize_occupation)
    agents_df["risk_perception"] = pd.to_numeric(agents_df["risk_perception"], errors="coerce")

    required_events_cols = {"day", "agent_id", "received_info", "wearing_mask"}
    if not required_events_cols.issubset(set(events_df.columns)):
        missing = required_events_cols - set(events_df.columns)
        raise ValueError(f"train_data.csv is missing required columns: {missing}")
    events_df = events_df.copy()
    events_df["day"] = events_df["day"].apply(to_int)
    events_df["agent_id"] = events_df["agent_id"].apply(to_int)
    events_df["received_info"] = pd.to_numeric(events_df["received_info"], errors="coerce")
    events_df["wearing_mask"] = pd.to_numeric(events_df["wearing_mask"], errors="coerce")

    # Standardize network layers (support node-centric or layer-centric)
    layer_adjacency_ids: Dict[str, Dict[int, List[int]]] = _parse_network_json_into_layers(net_json)

    # Build union of agent IDs across sources
    ids_agents = set(agents_df["agent_id"].unique().tolist())
    ids_events = set(events_df["agent_id"].unique().tolist())
    ids_network = _extract_nodes_from_network_json(net_json)
    all_ids = sorted(list(ids_agents.union(ids_events).union(ids_network)))
    id_to_index: Dict[int, int] = {aid: idx for idx, aid in enumerate(all_ids)}
    N = len(all_ids)

    # Build attributes arrays, filling defaults for agents not present in the attributes CSV
    default_age = "MiddleAge"
    default_occ = "White Collar"
    default_risk = 0.5

    attr_map_age: Dict[int, str] = dict(zip(agents_df["agent_id"], agents_df["age_group"]))
    attr_map_occ: Dict[int, str] = dict(zip(agents_df["agent_id"], agents_df["occupation"]))
    attr_map_risk: Dict[int, float] = dict(zip(agents_df["agent_id"], agents_df["risk_perception"]))

    age_group = np.array([attr_map_age.get(aid, default_age) for aid in all_ids], dtype=object)
    occupation = np.array([attr_map_occ.get(aid, default_occ) for aid in all_ids], dtype=object)
    risk_perception = np.array([attr_map_risk.get(aid, default_risk) for aid in all_ids], dtype=float)

    # Days sorted
    unique_days = sorted(events_df["day"].unique().tolist())
    if len(unique_days) == 0:
        raise ValueError("train_data.csv contains no rows. Cannot proceed.")
    T = len(unique_days)
    day_to_index = {d: i for i, d in enumerate(unique_days)}

    # Observed matrices Y (mask) and I (info)
    Y_obs = np.full((T, N), np.nan, dtype=float)
    I_obs = np.full((T, N), np.nan, dtype=float)

    for _, row in events_df.iterrows():
        d = int(row["day"])
        a = int(row["agent_id"])
        if d not in day_to_index or a not in id_to_index:
            continue
        t = day_to_index[d]
        i = id_to_index[a]
        wm = row["wearing_mask"]
        ri = row["received_info"]
        if not pd.isna(wm):
            Y_obs[t, i] = float(1.0 if wm >= 0.5 else 0.0)
        if not pd.isna(ri):
            I_obs[t, i] = float(1.0 if ri >= 0.5 else 0.0)

    # Default policy signal filled later/extended in main (here initialize with constant up to T)
    policy_signal = np.full(T, float(args.policy_constant), dtype=float)

    # Risk transform
    risk_transformed = _transform_risk(risk_perception)

    data = DataContainer(
        agent_ids=all_ids,
        id_to_index=id_to_index,
        age_group=age_group,
        occupation=occupation,
        risk_perception=risk_perception,
        risk_transformed=risk_transformed,
        days=unique_days,
        day_to_index=day_to_index,
        Y_obs=Y_obs,
        I_obs=I_obs,
        policy_signal=policy_signal,
    )

    # Check if network non-empty after parsing, else raise informative error
    deg_sum_total = 0
    for layer in LAYER_NAMES:
        adj = layer_adjacency_ids.get(layer, {})
        for node, neighs in adj.items():
            deg_sum_total += len(neighs)
    if deg_sum_total == 0:
        raise ValueError(
            "Parsed social_network.json produced an empty network (no edges). "
            "Ensure the file follows one of the supported formats: "
            "(A) Layer-centric with top-level keys 'family','work_school','community', or "
            "(B) Node-centric mapping agent_id -> {'family': [...], 'work_school': [...], 'community': [...]}."
        )

    return data, layer_adjacency_ids


# -----------------------
# Build Network and Agents
# -----------------------

def build_network_and_agents(
    data: DataContainer,
    layer_adjacency_ids: Dict[str, Dict[int, List[int]]],
    symmetrize_edges: bool = True
) -> Network:
    net = Network(data.id_to_index, layer_adjacency_ids, symmetrize=symmetrize_edges)
    return net


# -----------------------
# Holdout Split and Policy
# -----------------------

def holdout_split(data: DataContainer, forecast_horizon: int) -> Dict[str, Any]:
    """
    If forecast_horizon > 0: use all observed days for training; validation/forecast are T..T+H-1.
    Else: strict temporal holdout 80/20 within observed.
    """
    days = data.days
    T = len(days)
    if forecast_horizon > 0:
        train_idx = list(range(T))
        val_idx = list(range(T, T + forecast_horizon))
        train_days = days.copy()
        # Labels for forecast days continue after last observed day label
        last_day_label = days[-1]
        val_day_labels = list(range(last_day_label + 1, last_day_label + 1 + forecast_horizon))
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_days": train_days,
            "val_days": list(range(T, T + forecast_horizon)),
            "val_day_labels": val_day_labels,
        }
    else:
        n_train = int(math.floor(0.8 * T))
        if n_train <= 0 or n_train >= T:
            raise ValueError("No validation days available after temporal split.")
        train_idx = list(range(n_train))
        val_idx = list(range(n_train, T))
        train_days = [days[i] for i in train_idx]
        val_day_labels = [days[i] for i in val_idx]
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "train_days": train_days,
            "val_days": [i for i in val_idx],
            "val_day_labels": val_day_labels,
        }


def build_policy_signal(total_T: int, args: argparse.Namespace) -> np.ndarray:
    """
    Build day-level policy signal of length total_T based on CLI args.
    """
    if args.policy_step_day is not None and args.policy_step_day >= 0:
        sig = np.zeros(total_T, dtype=float)
        start = int(args.policy_step_day)
        end = int(args.policy_end_day) if args.policy_end_day is not None else -1
        level = float(args.policy_level)
        if start < total_T:
            if end is None or end < 0 or end > total_T:
                sig[start:] = level
            else:
                sig[start:end] = level
        return sig
    else:
        return np.full(total_T, float(args.policy_constant), dtype=float)


# -----------------------
# Calibration
# -----------------------

class Calibrator:
    """
    Likelihood-based calibrator with derivative-free multi-start random local search.
    """

    def __init__(
        self,
        segments: int,
        starts: int = 3,
        max_iter: int = 120,
        verbose: bool = False,
        include_policy: bool = True,
    ):
        self.segments = int(segments)
        self.starts = int(starts)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.include_policy = bool(include_policy)
        self.loss_trace: List[float] = []
        self.best_vector: Optional[np.ndarray] = None
        self.best_params: Optional[ParameterSet] = None
        self.segment_map: Optional[np.ndarray] = None  # day-index -> segment index

        # Regularization strengths
        self.l2_alpha = 1e-3
        self.l2_fe = 1e-3
        self.l2_beta0 = 1e-4

    def _compute_segments_map(self, T: int) -> np.ndarray:
        S = self.segments
        if S <= 0:
            raise ValueError("Number of segments must be positive.")
        seg_map = np.zeros(T, dtype=int)
        seg_len = int(math.ceil(T / S))
        for t in range(T):
            seg_map[t] = min(S - 1, t // seg_len)
        return seg_map

    def _initial_params(self, data: DataContainer) -> ParameterSet:
        S = self.segments
        beta0 = np.full(S, 0.05, dtype=float)

        Y = data.Y_obs
        I = data.I_obs
        ym = np.nanmean(Y) if np.isfinite(np.nanmean(Y)) else 0.5
        Y_mean = float(np.clip(ym, 1e-6, 1 - 1e-6))
        I_mean = np.nanmean(I) if np.isfinite(np.nanmean(I)) else 0.1
        alpha0 = float(np.clip(math.log(Y_mean / (1.0 - Y_mean)), -2.0, 2.0))
        beta_guess = float(np.clip(I_mean, 0.01, 0.5))

        p = ParameterSet(
            w_family=0.33,
            w_work_school=0.33,
            w_community=0.34,
            alpha_peer=1.0,
            alpha_inertia=2.0,
            alpha_info=1.0,
            alpha_risk=0.5,
            alpha0=alpha0,
            fe_age_youth=0.0,
            fe_age_youngadult=0.0,
            fe_occ_student=0.0,
            fe_occ_bluecollar=0.0,
            gamma_policy=0.0,
            beta_family=beta_guess,
            beta_work_school=beta_guess,
            beta_community=beta_guess,
            beta0_segments=beta0,
            rho_info_decay=0.01,
            multi_context_bonus=0.2,
            temperature_noise=1.0,
        )
        p.enforce_bounds()
        return p

    def _beta0_for_days(self, day_indices: List[int]) -> np.ndarray:
        seg_map = self.segment_map
        if seg_map is None:
            raise RuntimeError("Segment map is not initialized.")
        return seg_map[np.array(day_indices, dtype=int)]

    def _compute_info_nll(
        self,
        params: ParameterSet,
        data: DataContainer,
        net: Network,
        day_indices: List[int],
        day_weights: Optional[np.ndarray] = None,
    ) -> float:
        if len(day_indices) <= 1:
            return 0.0
        I = data.I_obs
        eps = 1e-9

        seg_map = self.segment_map
        assert seg_map is not None
        nll = 0.0

        for t_idx_pos in range(1, len(day_indices)):
            t_prev = day_indices[t_idx_pos - 1]
            t = day_indices[t_idx_pos]
            if t >= I.shape[0]:
                # No observed info for this day; skip in NLL
                continue
            seg = seg_map[t]
            beta0 = params.beta0_segments[seg]

            I_prev = I[t_prev].copy()
            I_curr = I[t].copy()

            ef, ew, ec = net.compute_info_exposure_fractions(I_prev, params.multi_context_bonus)
            hazard = 1.0 - np.exp(- (beta0 + params.beta_family * ef + params.beta_work_school * ew + params.beta_community * ec))
            hazard = np.clip(hazard, eps, 1.0 - eps)

            rho = params.rho_info_decay
            stay_prob = 1.0 - rho
            P = np.where(np.nan_to_num(I_prev) >= 0.5, stay_prob, hazard)
            P = np.clip(P, eps, 1.0 - eps)

            mask_obs = ~np.isnan(I_curr)
            if np.any(mask_obs):
                y = I_curr[mask_obs]
                p = P[mask_obs]
                contrib = - (y * safe_log(p) + (1.0 - y) * safe_log(1.0 - p)).sum()
                if day_weights is not None and t < len(day_weights):
                    contrib *= float(day_weights[t])
                nll += contrib
        return float(nll)

    def _compute_mask_nll(
        self,
        params: ParameterSet,
        data: DataContainer,
        net: Network,
        day_indices: List[int],
        day_weights: Optional[np.ndarray] = None,
    ) -> float:
        if len(day_indices) <= 1:
            return 0.0
        Y = data.Y_obs
        I = data.I_obs
        eps = 1e-9

        fe_age, fe_occ = params.build_fe_vectors(data)
        risk = data.risk_transformed

        w_f, w_ws, w_c = params.w_family, params.w_work_school, params.w_community

        nll = 0.0
        for t_idx_pos in range(1, len(day_indices)):
            t_prev = day_indices[t_idx_pos - 1]
            t = day_indices[t_idx_pos]
            if t >= Y.shape[0]:
                # No observed mask for this day; skip in NLL
                continue

            Y_prev = Y[t_prev].copy()
            Y_curr = Y[t].copy()
            I_curr = I[t].copy()

            pf, pw, pc = net.compute_peer_norms(Y_prev, params.multi_context_bonus)
            peer_weighted = w_f * pf + w_ws * pw + w_c * pc

            yprev = np.nan_to_num(Y_prev)
            icurr = np.nan_to_num(I_curr)
            logit_val = (
                params.alpha0
                + params.alpha_inertia * yprev
                + params.alpha_info * icurr
                + params.alpha_peer * peer_weighted
                + params.alpha_risk * risk
                + fe_age + fe_occ
                + params.gamma_policy * (data.policy_signal[t] if t < len(data.policy_signal) else data.policy_signal[-1])
            )
            prob = sigmoid(logit_val / max(1e-6, params.temperature_noise))
            prob = np.clip(prob, eps, 1.0 - eps)

            mask_obs = ~np.isnan(Y_curr)
            if np.any(mask_obs):
                y = Y_curr[mask_obs]
                p = prob[mask_obs]
                contrib = - (y * safe_log(p) + (1.0 - y) * safe_log(1.0 - p)).sum()
                if day_weights is not None and t < len(day_weights):
                    contrib *= float(day_weights[t])
                nll += contrib
        return float(nll)

    def _loss(
        self,
        params: ParameterSet,
        data: DataContainer,
        net: Network,
        train_idx: List[int],
        day_weights: Optional[np.ndarray] = None,
    ) -> float:
        nll_info = self._compute_info_nll(params, data, net, train_idx, day_weights)
        nll_mask = self._compute_mask_nll(params, data, net, train_idx, day_weights)

        reg = 0.0
        reg += self.l2_alpha * (
            params.alpha_peer**2 + params.alpha_inertia**2 + params.alpha_info**2 +
            params.alpha_risk**2 + params.alpha0**2
        )
        fe_age = params.FE_age_dict()
        fe_occ = params.FE_occ_dict()
        reg += self.l2_fe * sum(v**2 for v in fe_age.values())
        reg += self.l2_fe * sum(v**2 for v in fe_occ.values())
        reg += self.l2_beta0 * float(np.sum(params.beta0_segments**2))

        total = nll_info + nll_mask + reg
        return float(total)

    def _random_vector_within_bounds(self, bounds: List[Tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
        vec = []
        for lo, hi in bounds:
            vec.append(rng.uniform(lo, hi))
        return np.array(vec, dtype=float)

    def _perturb(self, vec: np.ndarray, bounds: List[Tuple[float, float]], step_scale: float,
                 rng: np.random.Generator) -> np.ndarray:
        new_vec = vec.copy()
        for i, (lo, hi) in enumerate(bounds):
            span = hi - lo
            sd = step_scale * span
            new_vec[i] += rng.normal(0.0, sd)
            new_vec[i] = float(np.clip(new_vec[i], lo, hi))
        return new_vec

    def _param_names(self, include_policy: bool) -> List[str]:
        names = [
            "w_family", "w_work_school", "w_community",
            "alpha_peer", "alpha_inertia", "alpha_info", "alpha_risk", "alpha0",
            "FE_age_Youth", "FE_age_YoungAdult",
            "FE_occ_Student", "FE_occ_BlueCollar",
            "beta_family", "beta_work_school", "beta_community",
        ]
        for s in range(self.segments):
            names.append(f"beta0_seg{s}")
        names += ["rho_info_decay", "multi_context_bonus"]
        if include_policy:
            names += ["gamma_policy"]
        names += ["temperature_noise"]
        return names

    def fit(
        self,
        data: DataContainer,
        net: Network,
        holdout: Dict[str, Any],
        rng: np.random.Generator,
        total_T: Optional[int] = None,
        bootstrap_runs: int = 0
    ) -> Tuple[ParameterSet, Dict[str, Any]]:
        """
        Fit/calibrate the parameter set on the training window; optionally run bootstrap.
        """
        T_obs = len(data.days)
        T_total = int(total_T) if total_T is not None else T_obs
        self.segment_map = self._compute_segments_map(T_total)

        include_policy = self.include_policy and (np.any(data.policy_signal[:T_total] > 0.0))
        bounds = ParameterSet.bounds_spec(num_segments=self.segments, include_policy=include_policy)

        init_params = self._initial_params(data)
        if self.segments != len(init_params.beta0_segments):
            init_params.beta0_segments = np.full(self.segments, 0.05, dtype=float)
            init_params.enforce_bounds()
        init_vec = init_params.to_vector(include_policy=include_policy)

        train_idx = holdout["train_idx"]

        best_vec = None
        best_loss = float("inf")
        loss_trace: List[float] = []

        for s in range(self.starts):
            if self.verbose:
                print(f"[Calibrator] Start {s + 1}/{self.starts}")
            if s == 0:
                vec = init_vec.copy()
            else:
                vec = self._random_vector_within_bounds(bounds, rng)

            step_scale = 0.10
            current_params = ParameterSet.from_vector(vec, num_segments=self.segments, include_policy=include_policy)
            current_loss = self._loss(current_params, data, net, train_idx)
            loss_trace.append(current_loss)
            last_improve_iter = 0

            for it in range(self.max_iter):
                cand_vec = self._perturb(vec, bounds, step_scale=step_scale, rng=rng)
                cand_params = ParameterSet.from_vector(cand_vec, num_segments=self.segments, include_policy=include_policy)
                cand_loss = self._loss(cand_params, data, net, train_idx)

                if cand_loss < current_loss:
                    vec = cand_vec
                    current_loss = cand_loss
                    last_improve_iter = it
                    if self.verbose and (it % 10 == 0):
                        print(f"  iter {it}: improved loss = {current_loss:.4f}")
                else:
                    if rng.uniform() < 0.05:
                        vec = self._random_vector_within_bounds(bounds, rng)
                        current_params = ParameterSet.from_vector(vec, num_segments=self.segments, include_policy=include_policy)
                        current_loss = self._loss(current_params, data, net, train_idx)
                        last_improve_iter = it

                loss_trace.append(current_loss)

                if (it - last_improve_iter) > 20 and step_scale > 0.01:
                    step_scale *= 0.7
                    last_improve_iter = it

            final_params = ParameterSet.from_vector(vec, num_segments=self.segments, include_policy=include_policy)
            final_loss = self._loss(final_params, data, net, train_idx)
            if self.verbose:
                print(f"[Calibrator] Start {s + 1} final loss: {final_loss:.4f}")
            if final_loss < best_loss:
                best_loss = final_loss
                best_vec = vec.copy()

        if best_vec is None:
            raise RuntimeError("Calibration failed to produce any candidate parameters.")

        best_params = ParameterSet.from_vector(best_vec, num_segments=self.segments, include_policy=include_policy)
        self.loss_trace = loss_trace
        self.best_vector = best_vec
        self.best_params = best_params

        # Approximate CIs via univariate curvature (finite differences)
        cis_curv = self._approximate_cis(best_vec, bounds, data, net, train_idx, include_policy)

        diagnostics: Dict[str, Any] = {
            "best_loss": best_loss,
            "loss_trace": [float(x) for x in loss_trace],
            "param_conf_int": cis_curv,
        }

        # Bootstrap day-weights refit
        if bootstrap_runs and bootstrap_runs > 0:
            if self.verbose:
                print(f"[Calibrator] Running bootstrap with {bootstrap_runs} replicates...")
            param_names = self._param_names(include_policy)
            boot_vecs = []
            boot_losses = []
            boot_step_iters = max(10, min(50, self.max_iter // 2))
            for b in range(bootstrap_runs):
                # Bayesian bootstrap: gamma(1,1) weights per day on training indices
                day_weights = np.ones(T_total, dtype=float)
                weights = rng.gamma(shape=1.0, scale=1.0, size=len(train_idx))
                for idx_pos, t in enumerate(train_idx):
                    day_weights[t] = float(weights[idx_pos])

                vec = best_vec.copy()
                step_scale = 0.08
                current_loss = self._loss(ParameterSet.from_vector(vec, self.segments, include_policy),
                                          data, net, train_idx, day_weights=day_weights)
                for it in range(boot_step_iters):
                    cand_vec = self._perturb(vec, bounds, step_scale=step_scale, rng=rng)
                    cand_loss = self._loss(ParameterSet.from_vector(cand_vec, self.segments, include_policy),
                                           data, net, train_idx, day_weights=day_weights)
                    if cand_loss < current_loss:
                        vec = cand_vec
                        current_loss = cand_loss
                    if (it + 1) % 15 == 0:
                        step_scale = max(0.01, step_scale * 0.8)

                boot_vecs.append(vec.copy())
                boot_losses.append(current_loss)

            boot_vecs = np.array(boot_vecs, dtype=float)
            cis_boot: Dict[str, Tuple[float, float]] = {}
            for i, name in enumerate(param_names):
                lower = float(np.percentile(boot_vecs[:, i], 2.5))
                upper = float(np.percentile(boot_vecs[:, i], 97.5))
                cis_boot[name] = (lower, upper)
            diagnostics["bootstrap_param_samples"] = boot_vecs.tolist()
            diagnostics["bootstrap_losses"] = [float(x) for x in boot_losses]
            diagnostics["param_conf_int_bootstrap"] = cis_boot

        return best_params, diagnostics

    def _approximate_cis(
        self,
        vec_opt: np.ndarray,
        bounds: List[Tuple[float, float]],
        data: DataContainer,
        net: Network,
        train_idx: List[int],
        include_policy: bool
    ) -> Dict[str, Tuple[float, float]]:
        names = self._param_names(include_policy)

        params_opt = ParameterSet.from_vector(vec_opt, num_segments=self.segments, include_policy=include_policy)
        base_loss = self._loss(params_opt, data, net, train_idx)

        cis: Dict[str, Tuple[float, float]] = {}
        for i, name in enumerate(names):
            lo, hi = bounds[i]
            span = hi - lo
            h = max(1e-3 * span, 1e-4)

            v_plus = vec_opt.copy()
            v_minus = vec_opt.copy()
            v_plus[i] = float(np.clip(v_plus[i] + h, lo, hi))
            v_minus[i] = float(np.clip(v_minus[i] - h, lo, hi))

            loss_plus = self._loss(
                ParameterSet.from_vector(v_plus, num_segments=self.segments, include_policy=include_policy),
                data, net, train_idx
            )
            loss_minus = self._loss(
                ParameterSet.from_vector(v_minus, num_segments=self.segments, include_policy=include_policy),
                data, net, train_idx
            )
            second_deriv = (loss_plus - 2.0 * base_loss + loss_minus) / (h ** 2)
            curvature = max(1e-6, float(second_deriv))
            se = math.sqrt(1.0 / curvature)
            lower = float(np.clip(vec_opt[i] - 1.96 * se, lo, hi))
            upper = float(np.clip(vec_opt[i] + 1.96 * se, lo, hi))
            cis[name] = (lower, upper)
        return cis


# -----------------------
# Simulator
# -----------------------

@dataclass
class SimulationResult:
    val_days: List[int]
    obs_agg_rates: np.ndarray
    sim_agg_rates_mean: np.ndarray
    sim_agg_rates_ci: np.ndarray
    brier_score: float
    transitions_obs: np.ndarray
    transitions_sim_mean: np.ndarray
    transitions_sim_ci: np.ndarray
    replicated_agg_rates: np.ndarray
    transitions_sim_all: np.ndarray
    pY_deterministic: Optional[np.ndarray] = None  # (T_val, N)
    pI_deterministic: Optional[np.ndarray] = None  # (T_val, N)


class Simulator:
    """
    Forward simulator for validation/forecast rollout with both probabilistic and stochastic paths.
    """

    def __init__(self, net: Network, params: ParameterSet):
        self.net = net
        self.params = params

    def _deterministic_probability_path(
        self,
        data: DataContainer,
        holdout: Dict[str, Any],
        segment_map: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Returns aggregate predicted mask probabilities per day, Brier score, and per-agent prob matrices.
        """
        params = self.params
        net = self.net
        val_idx = holdout["val_idx"]
        train_idx = holdout["train_idx"]
        T_val = len(val_idx)
        N = self.net.n_agents

        if T_val <= 0:
            raise ValueError("Validation/forecast index set is empty.")

        t0 = train_idx[-1]
        Y0 = data.Y_obs[t0].copy()
        I0 = data.I_obs[t0].copy()
        Y0[np.isnan(Y0)] = 0.0
        I0[np.isnan(I0)] = 0.0

        pY_prev = Y0.copy()
        pI_prev = I0.copy()

        fe_age, fe_occ = self.params.build_fe_vectors(data)
        risk = data.risk_transformed
        w_f, w_ws, w_c = params.w_family, params.w_work_school, params.w_community
        eps = 1e-9

        pY_agg_per_day = np.zeros(T_val, dtype=float)
        brier_sum = 0.0
        brier_count = 0

        pY_store = np.zeros((T_val, N), dtype=float)
        pI_store = np.zeros((T_val, N), dtype=float)

        for k, t in enumerate(val_idx):
            ef, ew, ec = net.compute_info_exposure_fractions(pI_prev, params.multi_context_bonus)
            seg = segment_map[t]
            beta0 = params.beta0_segments[seg]
            hazard = 1.0 - np.exp(- (beta0 + params.beta_family * ef + params.beta_work_school * ew + params.beta_community * ec))
            hazard = np.clip(hazard, eps, 1.0 - eps)
            pI_t = pI_prev * (1.0 - params.rho_info_decay) + (1.0 - pI_prev) * hazard

            pf, pw, pc = net.compute_peer_norms(pY_prev, params.multi_context_bonus)
            peer_w = w_f * pf + w_ws * pw + w_c * pc

            policy_val = data.policy_signal[t] if t < len(data.policy_signal) else data.policy_signal[-1]
            logit_val = (
                params.alpha0
                + params.alpha_inertia * pY_prev
                + params.alpha_info * pI_t
                + params.alpha_peer * peer_w
                + params.alpha_risk * risk
                + fe_age + fe_occ
                + params.gamma_policy * policy_val
            )
            pY_t = sigmoid(logit_val / max(1e-6, params.temperature_noise))
            pY_prev = pY_t
            pI_prev = pI_t

            pY_store[k, :] = pY_t
            pI_store[k, :] = pI_t

            pY_agg_per_day[k] = float(np.nanmean(pY_t))

            if t < data.Y_obs.shape[0]:
                Y_obs_t = data.Y_obs[t]
                mask_obs = ~np.isnan(Y_obs_t)
                if np.any(mask_obs):
                    y = Y_obs_t[mask_obs]
                    p = pY_t[mask_obs]
                    brier_sum += float(np.sum((p - y) ** 2))
                    brier_count += int(np.sum(mask_obs))

        brier_score = (brier_sum / brier_count) if brier_count > 0 else float("nan")
        return pY_agg_per_day, brier_score, pY_store, pI_store

    def rollout(
        self,
        data: DataContainer,
        holdout: Dict[str, Any],
        segment_map: np.ndarray,
        replications: int,
        rng: np.random.Generator
    ) -> SimulationResult:
        params = self.params
        net = self.net
        val_idx = holdout["val_idx"]
        T_val = len(val_idx)
        N = net.n_agents
        if T_val <= 0:
            raise ValueError("Validation/forecast index set is empty.")

        # Deterministic probability path for Brier and per-agent storage
        pY_agg_det, brier_score, pY_store, pI_store = self._deterministic_probability_path(data, holdout, segment_map)

        # Observed aggregate rates for days within observed window; forecast days -> NaN
        obs_agg_rates = np.zeros(T_val, dtype=float)
        obs_agg_rates[:] = np.nan
        for k, t in enumerate(val_idx):
            if t < data.Y_obs.shape[0]:
                yt = data.Y_obs[t]
                obs_agg_rates[k] = float(np.nanmean(yt))

        # Stochastic replications
        sim_agg_rates = np.zeros((replications, T_val), dtype=float)
        transitions_sim_all = np.zeros((replications, T_val, 4), dtype=float)  # P01, P11, P10, P00

        # Observed transitions
        transitions_obs = np.zeros((T_val, 4), dtype=float)
        transitions_obs[:] = np.nan
        for k, t in enumerate(val_idx):
            t_prev = t - 1
            if t_prev < 0 or t >= data.Y_obs.shape[0]:
                continue
            Y_prev_obs = data.Y_obs[t_prev]
            Y_curr_obs = data.Y_obs[t]
            mask = (~np.isnan(Y_prev_obs)) & (~np.isnan(Y_curr_obs))
            denom = float(np.sum(mask))
            if denom <= 0:
                transitions_obs[k, :] = np.nan
            else:
                y0 = Y_prev_obs[mask]
                y1 = Y_curr_obs[mask]
                p01 = float(np.sum((y0 < 0.5) & (y1 >= 0.5)) / denom)
                p11 = float(np.sum((y0 >= 0.5) & (y1 >= 0.5)) / denom)
                p10 = float(np.sum((y0 >= 0.5) & (y1 < 0.5)) / denom)
                p00 = float(np.sum((y0 < 0.5) & (y1 < 0.5)) / denom)
                transitions_obs[k, :] = [p01, p11, p10, p00]

        # Initial conditions from last training day
        t0 = holdout["train_idx"][-1]
        Y0 = data.Y_obs[t0].copy()
        I0 = data.I_obs[t0].copy()
        Y0[np.isnan(Y0)] = 0.0
        I0[np.isnan(I0)] = 0.0

        fe_age, fe_occ = self.params.build_fe_vectors(data)
        risk = data.risk_transformed
        w_f, w_ws, w_c = params.w_family, params.w_work_school, params.w_community
        eps = 1e-9

        for r in range(replications):
            Y_prev = Y0.copy()
            I_prev = I0.copy()

            for k, t in enumerate(val_idx):
                seg = segment_map[t]
                beta0 = params.beta0_segments[seg]

                ef, ew, ec = net.compute_info_exposure_fractions(I_prev, params.multi_context_bonus)
                hazard = 1.0 - np.exp(- (beta0 + params.beta_family * ef + params.beta_work_school * ew + params.beta_community * ec))
                hazard = np.clip(hazard, eps, 1.0 - eps)

                P_info = np.where(I_prev >= 0.5, 1.0 - params.rho_info_decay, hazard)
                P_info = np.clip(P_info, eps, 1.0 - eps)
                I_t = (rng.random(N) < P_info).astype(float)

                pf, pw, pc = net.compute_peer_norms(Y_prev, params.multi_context_bonus)
                peer_w = w_f * pf + w_ws * pw + w_c * pc

                policy_val = data.policy_signal[t] if t < len(data.policy_signal) else data.policy_signal[-1]
                logit_val = (
                    params.alpha0
                    + params.alpha_inertia * Y_prev
                    + params.alpha_info * I_t
                    + params.alpha_peer * peer_w
                    + params.alpha_risk * risk
                    + fe_age + fe_occ
                    + params.gamma_policy * policy_val
                )
                P_mask = sigmoid(logit_val / max(1e-6, params.temperature_noise))
                P_mask = np.clip(P_mask, eps, 1.0 - eps)
                Y_t = (rng.random(N) < P_mask).astype(float)

                sim_agg_rates[r, k] = float(np.mean(Y_t))

                y0 = Y_prev
                y1 = Y_t
                denom = float(N)
                p01 = float(np.sum((y0 < 0.5) & (y1 >= 0.5)) / denom)
                p11 = float(np.sum((y0 >= 0.5) & (y1 >= 0.5)) / denom)
                p10 = float(np.sum((y0 >= 0.5) & (y1 < 0.5)) / denom)
                p00 = float(np.sum((y0 < 0.5) & (y1 < 0.5)) / denom)
                transitions_sim_all[r, k, :] = [p01, p11, p10, p00]

                Y_prev = Y_t
                I_prev = I_t

        sim_mean = np.mean(sim_agg_rates, axis=0)
        lower = np.percentile(sim_agg_rates, 2.5, axis=0)
        upper = np.percentile(sim_agg_rates, 97.5, axis=0)
        sim_ci = np.vstack([lower, upper]).T

        transitions_sim_mean = np.mean(transitions_sim_all, axis=0)
        trans_lower = np.percentile(transitions_sim_all, 2.5, axis=0)
        trans_upper = np.percentile(transitions_sim_all, 97.5, axis=0)
        transitions_sim_ci = np.stack([trans_lower, trans_upper], axis=-1)

        result = SimulationResult(
            val_days=holdout["val_day_labels"],
            obs_agg_rates=obs_agg_rates,
            sim_agg_rates_mean=sim_mean,
            sim_agg_rates_ci=sim_ci,
            brier_score=brier_score,
            transitions_obs=transitions_obs,
            transitions_sim_mean=transitions_sim_mean,
            transitions_sim_ci=transitions_sim_ci,
            replicated_agg_rates=sim_agg_rates,
            transitions_sim_all=transitions_sim_all,
            pY_deterministic=pY_store,
            pI_deterministic=pI_store,
        )
        return result


# -----------------------
# Evaluation
# -----------------------

class Evaluator:
    """
    Evaluation of simulator outputs on validation/forecast days with metrics.
    """

    @staticmethod
    def _mean_and_ci(samples: np.ndarray) -> Tuple[float, float, float]:
        mean = float(np.mean(samples))
        lower = float(np.percentile(samples, 2.5))
        upper = float(np.percentile(samples, 97.5))
        return mean, lower, upper

    def compute_metrics(self, sim_res: SimulationResult) -> Dict[str, Any]:
        obs = sim_res.obs_agg_rates  # (T)
        sim_rep = sim_res.replicated_agg_rates  # (R, T)
        R, T = sim_rep.shape

        valid_days = np.isfinite(obs)
        if not np.any(valid_days):
            rmse_mean = float("nan"); rmse_lo = float("nan"); rmse_hi = float("nan")
            mae_mean = float("nan"); mae_lo = float("nan"); mae_hi = float("nan")
        else:
            obs_valid = obs[valid_days]
            sim_rep_valid = sim_rep[:, valid_days]
            rmse_samples = np.sqrt(np.mean((sim_rep_valid - obs_valid[None, :]) ** 2, axis=1))
            mae_samples = np.mean(np.abs(sim_rep_valid - obs_valid[None, :]), axis=1)
            rmse_mean, rmse_lo, rmse_hi = self._mean_and_ci(rmse_samples)
            mae_mean, mae_lo, mae_hi = self._mean_and_ci(mae_samples)

        # TransitionFit metrics using per-replication arrays
        trans_obs = sim_res.transitions_obs  # (T,4)
        valid_trans_days = np.all(np.isfinite(trans_obs), axis=1)
        tf_errors = {}
        if np.any(valid_trans_days):
            trans_sim_all = sim_res.transitions_sim_all  # (R,T,4)
            trans_names = ["P01", "P11", "P10", "P00"]
            for j, name in enumerate(trans_names):
                obs_j = trans_obs[valid_trans_days, j][None, :]  # (1, T_valid)
                sim_j = trans_sim_all[:, valid_trans_days, j]    # (R, T_valid)
                errs = np.mean(np.abs(sim_j - obs_j), axis=1)    # (R,)
                mean, lo, hi = self._mean_and_ci(errs)
                tf_errors[name] = {"mean_abs_error": mean, "ci_lower": lo, "ci_upper": hi}
        else:
            trans_names = ["P01", "P11", "P10", "P00"]
            for name in trans_names:
                tf_errors[name] = {"mean_abs_error": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

        metrics = {
            "RMSE_aggregate": {"mean": rmse_mean, "ci_lower": rmse_lo, "ci_upper": rmse_hi},
            "MAE_aggregate": {"mean": mae_mean, "ci_lower": mae_lo, "ci_upper": mae_hi},
            "Brier": sim_res.brier_score,
            "TransitionFit": tf_errors,
        }
        return metrics


# -----------------------
# Save Results
# -----------------------

def save_results(
    args: argparse.Namespace,
    params: ParameterSet,
    diagnostics: Dict[str, Any],
    sim_res: SimulationResult,
    metrics: Dict[str, Any],
) -> None:
    data_dir = args.data_dir
    prefix = args.output_prefix
    results_json = os.path.join(data_dir, f"{prefix}_results.json")
    timeseries_csv = os.path.join(data_dir, f"{prefix}_timeseries.csv")
    params_csv = os.path.join(data_dir, f"{prefix}_parameters.csv")
    loss_csv = os.path.join(data_dir, f"{prefix}_loss_curve.csv")
    traj_npz = os.path.join(data_dir, f"{prefix}_agent_trajectories.npz")

    out = {
        "parameters": params.to_dict(),
        "diagnostics": diagnostics,
        "metrics": metrics,
        "time_series": {
            "val_days": sim_res.val_days,
            "obs_aggregate_mask_rate": sim_res.obs_agg_rates.tolist(),
            "sim_aggregate_mask_rate_mean": sim_res.sim_agg_rates_mean.tolist(),
            "sim_aggregate_mask_rate_ci_lower": sim_res.sim_agg_rates_ci[:, 0].tolist(),
            "sim_aggregate_mask_rate_ci_upper": sim_res.sim_agg_rates_ci[:, 1].tolist(),
        },
    }
    with open(results_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results JSON: {results_json}")

    df_ts = pd.DataFrame({
        "day": sim_res.val_days,
        "obs_mask_rate": sim_res.obs_agg_rates,
        "sim_mask_rate_mean": sim_res.sim_agg_rates_mean,
        "sim_mask_rate_ci_lower": sim_res.sim_agg_rates_ci[:, 0],
        "sim_mask_rate_ci_upper": sim_res.sim_agg_rates_ci[:, 1],
    })
    df_ts.to_csv(timeseries_csv, index=False)
    print(f"Saved time series CSV: {timeseries_csv}")

    pd.DataFrame(list(params.to_dict().items()), columns=["parameter", "value"]).to_csv(params_csv, index=False)
    print(f"Saved parameters CSV: {params_csv}")

    if "loss_trace" in diagnostics and diagnostics["loss_trace"]:
        pd.DataFrame({"loss": diagnostics["loss_trace"]}).to_csv(loss_csv, index=False)
        print(f"Saved loss curve CSV: {loss_csv}")

    if args.save_agent_trajectories and (sim_res.pY_deterministic is not None):
        np.savez_compressed(traj_npz,
                            val_days=np.array(sim_res.val_days, dtype=int),
                            pY=sim_res.pY_deterministic,
                            pI=sim_res.pI_deterministic)
        print(f"Saved per-agent deterministic trajectories: {traj_npz}")


# -----------------------
# Main Orchestration
# -----------------------

def main() -> None:
    args = parse_cli()
    rng = set_global_seed(args.seed)

    # Load data
    data, layer_adjacency_ids = load_data(args, args.data_dir)
    if args.verbose:
        print(f"Loaded data: N={len(data.agent_ids)} agents, T_obs={len(data.days)} days from {args.data_dir}")

    # Temporal split or forecast
    holdout = holdout_split(data, forecast_horizon=int(args.forecast_horizon))
    if args.verbose:
        if args.forecast_horizon > 0:
            print(f"Forecast mode: training on all {len(holdout['train_days'])} observed days; "
                  f"forecasting {len(holdout['val_idx'])} days ({holdout['val_day_labels'][0]}..{holdout['val_day_labels'][-1]}).")
        else:
            print(f"Temporal validation: train_days={len(holdout['train_days'])}, val_days={len(holdout['val_idx'])}")

    # Determine segments (beta0_t baseline)
    if args.segments is None:
        segments = int(min(4, math.ceil(len(data.days) / 7.0)))
        segments = max(1, segments)
    else:
        segments = int(args.segments)

    # Extend policy_signal to total T (observed + forecast)
    T_total = len(data.days) + int(args.forecast_horizon)
    policy_signal = build_policy_signal(T_total, args)
    data.policy_signal = policy_signal  # may be longer than observed

    # Build network
    symmetrize = bool(args.symmetrize_edges == 1)
    net = build_network_and_agents(data, layer_adjacency_ids, symmetrize_edges=symmetrize)
    if args.verbose:
        degs = {layer: int(np.sum(net.degrees[layer])) for layer in LAYER_NAMES}
        print(f"Network built. Total degrees per layer: {degs}")

    # Calibrator
    include_policy = np.any(data.policy_signal[:T_total] > 0.0)
    calibrator = Calibrator(
        segments=segments,
        starts=args.starts,
        max_iter=args.max_iter,
        verbose=args.verbose,
        include_policy=include_policy
    )

    t0 = time.time()
    best_params, diagnostics = calibrator.fit(data, net, holdout, rng, total_T=T_total, bootstrap_runs=int(args.bootstrap))
    t1 = time.time()
    if args.verbose:
        print(f"Calibration completed in {t1 - t0:.2f}s. Best loss: {diagnostics.get('best_loss')}")

    # Simulator rollout
    segment_map = calibrator.segment_map
    simulator = Simulator(net, best_params)
    sim_res = simulator.rollout(data, holdout, segment_map=segment_map, replications=args.replications, rng=rng)

    # Evaluation
    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(sim_res)

    # Save results
    save_results(args, best_params, diagnostics, sim_res, metrics)

    print("End-to-end run completed successfully.")


# Execute main

# Execute main for both direct execution and sandbox wrapper invocation
main()