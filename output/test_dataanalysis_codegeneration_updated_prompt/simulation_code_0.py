#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mask-Wearing Diffusion Simulator (Multiplex Network)

This program ingests agent attributes, a multiplex social network, and panel
training data; performs a two-stage calibration (behavioral policy via logistic
regression by role, then information-exposure process via bounded gradient
descent with a light joint refinement loop); forward-simulates on a temporal
holdout; and evaluates validation metrics (aggregate RMSE/MAE, Brier, and
transition fit). It saves calibrated parameters, simulated trajectories, and
metrics to disk.

Usage (with sensible defaults):
    PROJECT_ROOT=/abs/project  DATA_PATH=data \
    python simulate.py \
        --agent-file agent_attributes.csv \
        --network-file social_network.json \
        --train-file train_data.csv \
        --output-dir results \
        --seed 42 \
        --k-runs 5

Environment variables (required for data path handling):
    PROJECT_ROOT : absolute path to project root
    DATA_PATH    : path relative to PROJECT_ROOT where data files live

All data files are expected under: os.path.join(PROJECT_ROOT, DATA_PATH, <file>)

Author: Code Generation Agent
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Global deterministic seed handling
# ---------------------------------------------------------------------------

GLOBAL_RNG: Optional[np.random.RandomState] = None


def set_global_seed(seed: int) -> None:
    """
    Set the global deterministic random seed for numpy operations.

    Parameters
    ----------
    seed : int
        Non-negative integer seed within [0, 2**31 - 1].

    Raises
    ------
    ValueError
        If the seed is not within valid range.
    """
    if not (isinstance(seed, int) and 0 <= seed < 2 ** 31):
        raise ValueError("seed must be int in [0, 2**31 - 1].")
    global GLOBAL_RNG
    GLOBAL_RNG = np.random.RandomState(seed)


# ---------------------------------------------------------------------------
# Path handling (as required)
# ---------------------------------------------------------------------------

import os as _os

PROJECT_ROOT = _os.environ.get("PROJECT_ROOT")
DATA_PATH = _os.environ.get("DATA_PATH")
if PROJECT_ROOT is None or DATA_PATH is None:
    raise EnvironmentError(
        "PROJECT_ROOT and DATA_PATH environment variables must be set. "
        "Example: PROJECT_ROOT=/abs/project DATA_PATH=data"
    )
DATA_DIR = _os.path.join(PROJECT_ROOT, DATA_PATH)


def _data_path(filename: str) -> str:
    """
    Build absolute data file path under DATA_DIR.

    Parameters
    ----------
    filename : str
        File name relative to the data directory.

    Returns
    -------
    str
        Absolute path to the data file.
    """
    return _os.path.join(DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

ROLES = ["Student", "Blue Collar", "White Collar", "Other"]
LAYERS = ["family", "work_school", "community"]

def sigmoid(x: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


def ewma(values: NDArray[np.float64], alpha: float = 0.2) -> NDArray[np.float64]:
    """Simple exponentially weighted moving average."""
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Data classes and validation
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for the simulator and calibration."""
    seed: int = 42
    k_runs: int = 5
    max_iter_policy: int = 400
    max_iter_info: int = 400
    lr_policy: float = 0.1
    lr_info: float = 0.05
    l2_reg: float = 0.01
    info_bounds: Dict[str, Tuple[float, float]] = None  # set in __post_init__
    output_dir: str = "results"
    inner_refine_steps: int = 100
    early_stop_patience: int = 20
    memory_decay: float = 1.0  # used for peer averaging if needed

    def __post_init__(self):
        if self.info_bounds is None:
            # Parameter bounds derived from blueprint
            self.info_bounds = {
                "w_family": (0.0, 3.0),
                "w_work_school": (0.0, 3.0),
                "w_community": (0.0, 3.0),
                "p_contact_family": (0.0, 1.0),
                "p_contact_work_school": (0.0, 1.0),
                "p_contact_community": (0.0, 1.0),
                "kappa": (0.0, 10.0),       # slope
                "sig_info0": (-5.0, 5.0),   # intercept
                "b_exo": (0.0, 0.5),        # constant baseline (EWMA-level)
            }


@dataclass
class PolicyParams:
    """Role-specific logistic policy coefficients."""
    beta0: float
    beta_inertia: float
    beta_info: float
    beta_family: float
    beta_work: float
    beta_comm: float
    beta_risk: float
    beta_mandate: float

    def as_vector(self) -> NDArray[np.float64]:
        return np.array([
            self.beta0, self.beta_inertia, self.beta_info,
            self.beta_family, self.beta_work, self.beta_comm,
            self.beta_risk, self.beta_mandate
        ], dtype=np.float64)


@dataclass
class InfoParams:
    """Information process parameters (logistic form)."""
    w_family: float
    w_work_school: float
    w_community: float
    p_contact_family: float
    p_contact_work_school: float
    p_contact_community: float
    kappa: float
    sig_info0: float
    b_exo: float  # constant baseline; time-varying proxy handled by EWMA on training residuals

    def clip_(self, bounds: Dict[str, Tuple[float, float]]) -> None:
        """In-place projection onto parameter bounds."""
        for k, (lo, hi) in bounds.items():
            v = getattr(self, k)
            v = float(np.clip(v, lo, hi))
            setattr(self, k, v)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Network representation
# ---------------------------------------------------------------------------

class MultiplexNetwork:
    """
    Multiplex network holding per-layer symmetric adjacency lists and utilities.

    The input JSON maps stringified node IDs to adjacency per layer (family,
    work_school, community, all). We convert to integers, remove self-loops,
    deduplicate, and symmetrize each layer.
    """

    def __init__(self, layer_adj: Dict[str, Dict[int, List[int]]]):
        """
        Parameters
        ----------
        layer_adj : dict
            Mapping from layer name to {node_id: [neighbors]} adjacency.
        """
        self.layer_adj = layer_adj  # already symmetrized and cleaned
        self.nodes = sorted(set().union(*[set(d.keys()) for d in layer_adj.values()]))

    @staticmethod
    def from_json(path: str) -> "MultiplexNetwork":
        """
        Load and clean multiplex network from JSON.

        Parameters
        ----------
        path : str
            Absolute path to social_network.json

        Returns
        -------
        MultiplexNetwork

        Raises
        ------
        FileNotFoundError, ValueError
        """
        if not os.path.isabs(path):
            raise ValueError("Network path must be absolute.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Network file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        required_layers = ["family", "work_school", "community", "all"]
        # Convert to ints, clean, symmetrize per layer
        layer_adj: Dict[str, Dict[int, List[int]]] = {L: {} for L in required_layers}

        def _clean_list(lst: Iterable[int], self_id: int) -> List[int]:
            s = set(int(x) for x in lst if int(x) != self_id)
            return sorted(s)

        # Initial pass: clean and deduplicate, remove self loops
        for sid, payload in raw.items():
            i = int(sid)
            for L in required_layers:
                if L not in payload:
                    raise ValueError(f"Missing layer '{L}' for node {i} in network JSON.")
                layer_adj[L][i] = _clean_list(payload[L], i)

        # Symmetrize for each layer except 'all' (which we'll recompute as union)
        for L in ["family", "work_school", "community"]:
            for i, nbrs in list(layer_adj[L].items()):
                for j in nbrs:
                    layer_adj[L].setdefault(j, [])
                    if i not in layer_adj[L][j]:
                        layer_adj[L][j].append(i)
            # sort
            for i in layer_adj[L]:
                layer_adj[L][i] = sorted(set(layer_adj[L][i]))

        # Recompute 'all' as union
        union: Dict[int, List[int]] = {}
        all_nodes = sorted(set().union(*[set(d.keys()) for d in layer_adj.values()]))
        for i in all_nodes:
            fam = set(layer_adj["family"].get(i, []))
            wor = set(layer_adj["work_school"].get(i, []))
            com = set(layer_adj["community"].get(i, []))
            union[i] = sorted(fam | wor | com)
        layer_adj["all"] = union

        return MultiplexNetwork(layer_adj)

    def degree(self, node: int, layer: str) -> int:
        """Return degree of node in a given layer."""
        return len(self.layer_adj[layer].get(node, []))

    def neighbors(self, node: int, layer: str) -> List[int]:
        """Return neighbors of node in a given layer."""
        return self.layer_adj[layer].get(node, [])


# ---------------------------------------------------------------------------
# Agents and decision policy
# ---------------------------------------------------------------------------

class Agent:
    """
    Individual agent with static attributes and dynamic state.

    Dynamic state tracked externally in arrays for vectorization: wearing_mask_t,
    received_info_t. This class holds metadata and lookup indices.
    """

    def __init__(self, agent_id: int, age: int, age_group: str, occupation: str,
                 risk_perception: float, initial_mask: bool):
        self.agent_id = int(agent_id)
        self.age = int(age)
        self.age_group = str(age_group)
        self.occupation = str(occupation)
        self.role = self._map_role(self.occupation)
        self.risk_perception = float(risk_perception)
        self.initial_mask = bool(initial_mask)

    @staticmethod
    def _map_role(occupation: str) -> str:
        if occupation in ("Student",):
            return "Student"
        if occupation in ("Blue Collar",):
            return "Blue Collar"
        if occupation in ("White Collar",):
            return "White Collar"
        return "Other"


class DecisionPolicy:
    """
    Logistic decision policy with role-specific coefficients.

    P(wear_t=1) = sigmoid(beta0 + beta_inertia*wear_{t-1} + beta_info*received_info_t
                           + beta_family*peer_share_family + beta_work*peer_share_work_school
                           + beta_comm*peer_share_community + beta_risk*risk_perception
                           + beta_mandate*mandate_t)
    """

    def __init__(self, role_to_params: Dict[str, PolicyParams]):
        for role in ROLES:
            if role not in role_to_params:
                raise ValueError(f"Missing policy parameters for role '{role}'.")
        self.role_to_params = role_to_params

    def prob(self,
             role_idx: NDArray[np.int64],
             wear_prev: NDArray[np.float64],
             info_t: NDArray[np.float64],
             s_family: NDArray[np.float64],
             s_work: NDArray[np.float64],
             s_comm: NDArray[np.float64],
             risk: NDArray[np.float64],
             mandate_t: float = 0.0) -> NDArray[np.float64]:
        """
        Compute adoption probability for all agents at a time step.

        Parameters
        ----------
        role_idx : ndarray[int], shape (N,)
            Indices mapping to ROLES list.
        wear_prev : ndarray[float], shape (N,)
        info_t : ndarray[float], shape (N,)
        s_family, s_work, s_comm : ndarray[float], shape (N,)
            Peer mask shares per layer.
        risk : ndarray[float], shape (N,)
        mandate_t : float
            Optional mandate signal in [0,1].

        Returns
        -------
        ndarray[float], shape (N,)
            Probability of wearing a mask at time t.
        """
        N = wear_prev.shape[0]
        logits = np.zeros(N, dtype=np.float64)
        for r, role in enumerate(ROLES):
            mask = (role_idx == r)
            if not np.any(mask):
                continue
            p = self.role_to_params[role]
            logits[mask] = (
                p.beta0
                + p.beta_inertia * wear_prev[mask]
                + p.beta_info * info_t[mask]
                + p.beta_family * s_family[mask]
                + p.beta_work * s_work[mask]
                + p.beta_comm * s_comm[mask]
                + p.beta_risk * risk[mask]
                + p.beta_mandate * float(mandate_t)
            )
        return sigmoid(logits)


# ---------------------------------------------------------------------------
# Environment signals (exogenous)
# ---------------------------------------------------------------------------

class EnvironmentSignals:
    """
    Exogenous signals container.

    For simplicity, we model a constant b_exo learned on training and allow
    optional daily series via EWMA residual smoothing. If a mandate series were
    provided, it could be injected here (set to zeros otherwise).
    """

    def __init__(self,
                 b_exo: float,
                 daily_b_exo_series: Optional[pd.Series] = None,
                 mandate_series: Optional[pd.Series] = None):
        self.b_exo = float(b_exo)
        self.daily_b_exo_series = daily_b_exo_series  # indexed by day
        self.mandate_series = mandate_series  # indexed by day in [0,1]

    def baseline_for_day(self, day: int) -> float:
        """Return baseline exogenous info for a given day."""
        if self.daily_b_exo_series is not None and day in self.daily_b_exo_series.index:
            return float(self.daily_b_exo_series.loc[day])
        return self.b_exo

    def mandate_for_day(self, day: int) -> float:
        """Return mandate indicator for a given day."""
        if self.mandate_series is not None and day in self.mandate_series.index:
            v = float(self.mandate_series.loc[day])
            return float(np.clip(v, 0.0, 1.0))
        return 0.0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Multi-agent simulator operating on a multiplex network with an information
    exposure process and a role-conditional decision policy.
    """

    def __init__(self,
                 network: MultiplexNetwork,
                 agents_df: pd.DataFrame,
                 role_index: NDArray[np.int64],
                 risk_vec: NDArray[np.float64],
                 policy: DecisionPolicy,
                 info_params: InfoParams,
                 signals: EnvironmentSignals):
        self.network = network
        self.agents_df = agents_df
        self.role_index = role_index
        self.risk_vec = risk_vec
        self.policy = policy
        self.info_params = info_params
        self.signals = signals

    def compute_peer_shares(self,
                            wear_prev: NDArray[np.float64]) -> Tuple[NDArray[np.float64],
                                                                      NDArray[np.float64],
                                                                      NDArray[np.float64]]:
        """
        Compute peer mask shares per layer given previous wearing states.

        Parameters
        ----------
        wear_prev : ndarray[float], shape (N,)
            Previous day wearing states (0/1).

        Returns
        -------
        s_family, s_work, s_comm : ndarray[float], shape (N,)
        """
        N = wear_prev.shape[0]
        idx_to_id = self.agents_df["agent_id"].to_numpy()
        id_to_idx = {aid: i for i, aid in enumerate(idx_to_id)}

        def layer_share(layer: str) -> NDArray[np.float64]:
            out = np.zeros(N, dtype=np.float64)
            for i, aid in enumerate(idx_to_id):
                nbrs = self.network.neighbors(aid, layer)
                if len(nbrs) == 0:
                    out[i] = 0.0
                else:
                    s = 0.0
                    c = 0
                    for j_id in nbrs:
                        j = id_to_idx.get(j_id, None)
                        if j is None:
                            continue
                        s += wear_prev[j]
                        c += 1
                    out[i] = s / c if c > 0 else 0.0
            return out

        s_fam = layer_share("family")
        s_work = layer_share("work_school")
        s_comm = layer_share("community")
        return s_fam, s_work, s_comm

    def info_probability(self,
                         s_family: NDArray[np.float64],
                         s_work: NDArray[np.float64],
                         s_comm: NDArray[np.float64],
                         day: int) -> NDArray[np.float64]:
        """
        Compute per-agent probability of receiving information on a day,
        using logistic(sig_info0 + kappa * sum_L (w_L * p_contact_L * s_iL) + b_exo_day).

        Parameters
        ----------
        s_family, s_work, s_comm : ndarray[float], shape (N,)
        day : int

        Returns
        -------
        ndarray[float], shape (N,)
        """
        P = self.info_params
        intensity = (
            P.w_family * P.p_contact_family * s_family
            + P.w_work_school * P.p_contact_work_school * s_work
            + P.w_community * P.p_contact_community * s_comm
        )
        baseline = self.signals.baseline_for_day(day)
        logits = P.sig_info0 + P.kappa * intensity + baseline
        return sigmoid(logits)

    def step(self,
             day: int,
             wear_prev: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        One synchronous simulation step.

        Parameters
        ----------
        day : int
        wear_prev : ndarray[float], shape (N,)
            Wearing state at t-1 (0 or 1).

        Returns
        -------
        wear_t : ndarray[float], shape (N,)
        info_t : ndarray[float], shape (N,)
        """
        s_fam, s_work, s_comm = self.compute_peer_shares(wear_prev)
        p_info = self.info_probability(s_fam, s_work, s_comm, day)
        info_t = (GLOBAL_RNG.rand(len(wear_prev)) < p_info).astype(np.float64)

        mandate_t = self.signals.mandate_for_day(day)
        p_wear = self.policy.prob(
            role_idx=self.role_index,
            wear_prev=wear_prev,
            info_t=info_t,
            s_family=s_fam,
            s_work=s_work,
            s_comm=s_comm,
            risk=self.risk_vec,
            mandate_t=mandate_t,
        )
        wear_t = (GLOBAL_RNG.rand(len(wear_prev)) < p_wear).astype(np.float64)
        return wear_t, info_t

    def run(self,
            start_day: int,
            end_day: int,
            init_wear: NDArray[np.float64],
            k_runs: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run K stochastic simulations and return averaged probabilities and
        optional sampled trajectories.

        Parameters
        ----------
        start_day : int
            Inclusive start day for forward simulation.
        end_day : int
            Inclusive end day.
        init_wear : ndarray[float], shape (N,)
            Initial wearing state at start_day - 1.
        k_runs : int
            Number of independent runs to average.

        Returns
        -------
        probs_df : DataFrame
            Columns: ['day', 'agent_id', 'p_wear_mean']
        samples_df : DataFrame
            Columns: ['day', 'agent_id', 'sampled_wear_mean'] (mean across runs)
        """
        N = init_wear.shape[0]
        agent_ids = self.agents_df["agent_id"].to_numpy()
        days = list(range(start_day, end_day + 1))

        prob_accum = {d: np.zeros(N, dtype=np.float64) for d in days}
        sample_accum = {d: np.zeros(N, dtype=np.float64) for d in days}

        for r in range(k_runs):
            wear_prev = init_wear.copy()
            for d in days:
                s_fam, s_work, s_comm = self.compute_peer_shares(wear_prev)
                p_info = self.info_probability(s_fam, s_work, s_comm, d)
                info_t = (GLOBAL_RNG.rand(N) < p_info).astype(np.float64)
                p_wear = self.policy.prob(
                    role_idx=self.role_index,
                    wear_prev=wear_prev,
                    info_t=info_t,
                    s_family=s_fam,
                    s_work=s_work,
                    s_comm=s_comm,
                    risk=self.risk_vec,
                    mandate_t=self.signals.mandate_for_day(d),
                )
                wear_t = (GLOBAL_RNG.rand(N) < p_wear).astype(np.float64)
                prob_accum[d] += p_wear
                sample_accum[d] += wear_t
                wear_prev = wear_t

        # Build DataFrames
        rows_probs = []
        rows_samples = []
        for d in days:
            rows_probs.append(pd.DataFrame({
                "day": d,
                "agent_id": agent_ids,
                "p_wear_mean": prob_accum[d] / float(k_runs)
            }))
            rows_samples.append(pd.DataFrame({
                "day": d,
                "agent_id": agent_ids,
                "sampled_wear_mean": sample_accum[d] / float(k_runs)
            }))
        probs_df = pd.concat(rows_probs, ignore_index=True)
        samples_df = pd.concat(rows_samples, ignore_index=True)
        return probs_df, samples_df


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class LogisticRegressor:
    """
    Simple L2-regularized logistic regression fitted via batch gradient descent.

    Implemented to avoid external heavy dependencies; suitable for medium-size
    datasets typical in this simulation context.
    """

    def __init__(self, lr: float = 0.1, max_iter: int = 400, l2: float = 0.01):
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.coef_: Optional[NDArray[np.float64]] = None
        self.loss_curve_: List[float] = []

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64],
            patience: int = 20) -> None:
        """
        Fit logistic regression.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,)
        patience : int
            Early stopping patience on loss.

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must be shapes (n, d) and (n,).")
        n, d = X.shape
        w = np.zeros(d, dtype=np.float64)
        best_loss = np.inf
        best_w = w.copy()
        no_improve = 0

        for it in range(self.max_iter):
            z = X @ w
            p = sigmoid(z)
            # Negative log-likelihood with L2
            eps = 1e-9
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 0.5 * self.l2 * np.sum(w * w)
            self.loss_curve_.append(float(loss))
            # Gradient
            grad = (X.T @ (p - y)) / n + self.l2 * w
            w -= self.lr * grad

            if loss + 1e-9 < best_loss:
                best_loss = loss
                best_w = w.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.coef_ = best_w

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return p(y=1|x)."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted.")
        return sigmoid(X @ self.coef_)

    def get_params(self) -> NDArray[np.float64]:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted.")
        return self.coef_.copy()


class Calibrator:
    """
    Calibrates policy (per-role logistic models) and information process
    parameters, then performs a light joint refinement by simulating on the
    training window and minimizing a composite loss (Brier + aggregate RMSE).
    """

    def __init__(self,
                 cfg: SimulationConfig,
                 network: MultiplexNetwork,
                 agents_df: pd.DataFrame,
                 train_df: pd.DataFrame,
                 role_index: NDArray[np.int64],
                 risk_vec: NDArray[np.float64]):
        self.cfg = cfg
        self.network = network
        self.agents_df = agents_df
        self.train_df = train_df
        self.role_index = role_index
        self.risk_vec = risk_vec

        # Placeholders
        self.policy_params: Dict[str, PolicyParams] = {}
        self.info_params: Optional[InfoParams] = None
        self.daily_b_exo_series: Optional[pd.Series] = None

    # ------------------------- Feature engineering -------------------------

    def _compute_peer_shares_from_observed(self, df_window: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-layer peer mask shares s_i,L(t-1) using observed wearing_mask
        and the network for each (agent, day) in df_window.
        """
        # Prepare fast lookups
        agent_ids = self.agents_df["agent_id"].to_numpy()
        id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        days = sorted(df_window["day"].unique())

        # Build wearing matrix W[day][idx] from observed data
        N = len(agent_ids)
        W_prev = {}
        for d in days:
            sub = df_window[df_window["day"] == d][["agent_id", "wearing_mask"]]
            vec = np.zeros(N, dtype=np.float64)
            for _, row in sub.iterrows():
                i = id_to_idx.get(int(row.agent_id), None)
                if i is not None:
                    vec[i] = float(row.wearing_mask)
            W_prev[d] = vec

        rows = []
        for d in days[1:]:  # need t-1
            prev = W_prev[d - 1]
            # compute shares from prev
            s_fam = np.zeros(N, dtype=np.float64)
            s_work = np.zeros(N, dtype=np.float64)
            s_comm = np.zeros(N, dtype=np.float64)
            for i, aid in enumerate(agent_ids):
                def share(layer: str) -> float:
                    nbrs = self.network.neighbors(int(aid), layer)
                    if not nbrs:
                        return 0.0
                    s = 0.0
                    c = 0
                    for j_id in nbrs:
                        j = id_to_idx.get(j_id, None)
                        if j is None:
                            continue
                        s += prev[j]
                        c += 1
                    return s / c if c > 0 else 0.0

                s_fam[i] = share("family")
                s_work[i] = share("work_school")
                s_comm[i] = share("community")

            df_d = df_window[df_window["day"] == d].copy()
            df_d["s_family_tm1"] = s_fam
            df_d["s_work_tm1"] = s_work
            df_d["s_comm_tm1"] = s_comm
            # add wear_{t-1}
            df_d = df_d.merge(
                df_window[["agent_id", "day", "wearing_mask"]]
                .rename(columns={"day": "day_prev", "wearing_mask": "wear_tm1"}),
                left_on=["agent_id", "day"],
                right_on=["agent_id", "day_prev"],
                how="left"
            )
            # shift to t-1 match
            df_d["wear_tm1"] = df_d["wear_tm1"].shift(1)
            df_d = df_d.drop(columns=["day_prev"])
            rows.append(df_d)

        out = pd.concat(rows, ignore_index=True)
        out = out.dropna(subset=["s_family_tm1", "s_work_tm1", "s_comm_tm1"])
        return out

    # ------------------------- Stage A: policy -----------------------------

    def _fit_policy_by_role(self, df_train: pd.DataFrame) -> Dict[str, PolicyParams]:
        """
        Fit logistic policy per role using observed data on the training window.
        """
        # Compute features
        df_feat = self._compute_peer_shares_from_observed(df_train)
        # Align wear_tm1 explicitly
        # Create role column
        agents_small = self.agents_df[["agent_id", "occupation", "risk_perception"]].copy()
        agents_small["role"] = agents_small["occupation"].apply(Agent._map_role)
        df_feat = df_feat.merge(agents_small, on="agent_id", how="left")

        # Build X, y per role
        role_params: Dict[str, PolicyParams] = {}
        for role in ROLES:
            sub = df_feat[df_feat["role"] == role].copy()
            # Drop first day (no t-1)
            sub = sub.sort_values(["agent_id", "day"])
            # Construct features
            y = sub["wearing_mask"].astype(float).to_numpy()
            wear_tm1 = sub["wear_tm1"].fillna(method="ffill").fillna(0.0).astype(float).to_numpy()
            info_t = sub["received_info"].astype(float).to_numpy()
            s_family = sub["s_family_tm1"].astype(float).to_numpy()
            s_work = sub["s_work_tm1"].astype(float).to_numpy()
            s_comm = sub["s_comm_tm1"].astype(float).to_numpy()
            risk = sub["risk_perception"].astype(float).to_numpy()

            # No mandate series provided -> zeros
            mandate = np.zeros_like(risk)

            X = np.column_stack([
                np.ones_like(wear_tm1),
                wear_tm1,
                info_t,
                s_family,
                s_work,
                s_comm,
                risk,
                mandate
            ])

            if len(X) == 0:
                # Fallback small coefficients if no data
                role_params[role] = PolicyParams(
                    beta0= -1.0, beta_inertia= 1.0, beta_info= 1.0,
                    beta_family= 0.5, beta_work= 0.5, beta_comm= 0.3,
                    beta_risk= 0.5, beta_mandate= 0.5
                )
                continue

            lr = LogisticRegressor(lr=self.cfg.lr_policy,
                                   max_iter=self.cfg.max_iter_policy,
                                   l2=self.cfg.l2_reg)
            lr.fit(X, y, patience=self.cfg.early_stop_patience)
            w = lr.get_params()
            role_params[role] = PolicyParams(
                beta0=float(w[0]),
                beta_inertia=float(w[1]),
                beta_info=float(w[2]),
                beta_family=float(w[3]),
                beta_work=float(w[4]),
                beta_comm=float(w[5]),
                beta_risk=float(w[6]),
                beta_mandate=float(w[7]),
            )

        return role_params

    # ------------------------- Stage B: information ------------------------

    def _fit_information_process(self, df_train: pd.DataFrame) -> Tuple[InfoParams, pd.Series]:
        """
        Fit information exposure parameters to match observed received_info_t.

        We minimize cross-entropy between predicted p_info and observed
        received_info, with parameter bounds enforced by projection.
        """
        df_feat = self._compute_peer_shares_from_observed(df_train).copy()
        df_feat = df_feat.sort_values(["day", "agent_id"])

        y = df_feat["received_info"].astype(float).to_numpy()
        s_family = df_feat["s_family_tm1"].astype(float).to_numpy()
        s_work = df_feat["s_work_tm1"].astype(float).to_numpy()
        s_comm = df_feat["s_comm_tm1"].astype(float).to_numpy()
        days = df_feat["day"].astype(int).to_numpy()

        # Initialize parameters within bounds
        P = InfoParams(
            w_family=1.0, w_work_school=1.0, w_community=0.8,
            p_contact_family=0.5, p_contact_work_school=0.4, p_contact_community=0.3,
            kappa=2.0, sig_info0= -1.0, b_exo=0.1
        )
        P.clip_(self.cfg.info_bounds)

        # Learn a time-varying baseline via EWMA residuals after each epoch
        unique_days = sorted(df_feat["day"].unique())
        day_index = {d: i for i, d in enumerate(unique_days)}
        b_series = pd.Series(index=unique_days, dtype=float)

        def predict_p(P_: InfoParams, b_series_local: Optional[pd.Series]) -> NDArray[np.float64]:
            intensity = (
                P_.w_family * P_.p_contact_family * s_family
                + P_.w_work_school * P_.p_contact_work_school * s_work
                + P_.w_community * P_.p_contact_community * s_comm
            )
            baseline = np.array([b_series_local.get(int(d), P_.b_exo) if b_series_local is not None else P_.b_exo
                                 for d in days], dtype=np.float64)
            logits = P_.sig_info0 + P_.kappa * intensity + baseline
            return sigmoid(logits)

        best_loss = np.inf
        best_params = P
        best_b = b_series.copy()

        no_improve = 0
        for it in range(self.cfg.max_iter_info):
            p = predict_p(P, b_series if it > 0 else None)
            eps = 1e-9
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            # L2 on weights to keep plausible
            loss += 1e-3 * (
                P.w_family**2 + P.w_work_school**2 + P.w_community**2
                + P.kappa**2 + P.sig_info0**2
            )

            # Gradients (approximate; baseline handled separately)
            # dloss/dlogit = p - y
            g_common = (p - y)
            # chain for each parameter
            grad_w_family = np.mean(g_common * (P.kappa * P.p_contact_family * s_family))
            grad_w_work = np.mean(g_common * (P.kappa * P.p_contact_work_school * s_work))
            grad_w_comm = np.mean(g_common * (P.kappa * P.p_contact_community * s_comm))
            grad_p_fam = np.mean(g_common * (P.kappa * P.w_family * s_family))
            grad_p_work = np.mean(g_common * (P.kappa * P.w_work_school * s_work))
            grad_p_comm = np.mean(g_common * (P.kappa * P.w_community * s_comm))
            grad_kappa = np.mean(g_common * (P.w_family * P.p_contact_family * s_family
                                             + P.w_work_school * P.p_contact_work_school * s_work
                                             + P.w_community * P.p_contact_community * s_comm))
            grad_sig0 = np.mean(g_common)

            # L2 regularization gradients
            grad_w_family += 2e-3 * P.w_family
            grad_w_work += 2e-3 * P.w_work_school
            grad_w_comm += 2e-3 * P.w_community
            grad_kappa += 2e-3 * P.kappa
            grad_sig0 += 2e-3 * P.sig_info0

            # Gradient step
            P.w_family -= self.cfg.lr_info * grad_w_family
            P.w_work_school -= self.cfg.lr_info * grad_w_work
            P.w_community -= self.cfg.lr_info * grad_w_comm
            P.p_contact_family -= self.cfg.lr_info * grad_p_fam
            P.p_contact_work_school -= self.cfg.lr_info * grad_p_work
            P.p_contact_community -= self.cfg.lr_info * grad_p_comm
            P.kappa -= self.cfg.lr_info * grad_kappa
            P.sig_info0 -= self.cfg.lr_info * grad_sig0
            # b_exo updated via residual mean
            P.b_exo = float(np.clip(P.b_exo - self.cfg.lr_info * np.mean(g_common), *self.cfg.info_bounds["b_exo"]))

            # Project to bounds
            P.clip_(self.cfg.info_bounds)

            # Update day-wise EWMA baseline from residuals
            resid_by_day = pd.Series(0.0, index=unique_days, dtype=float)
            count_by_day = pd.Series(0.0, index=unique_days, dtype=float)
            for d, r in zip(days, (y - p)):
                resid_by_day.loc[int(d)] += float(r)
                count_by_day.loc[int(d)] += 1.0
            avg_resid = resid_by_day.divide(count_by_day.replace(0.0, np.nan)).fillna(0.0)
            # smooth and shift toward zero to keep within bounds
            smoothed = ewma(avg_resid.to_numpy(), alpha=0.3)
            smoothed = np.clip(smoothed + P.b_exo, *self.cfg.info_bounds["b_exo"])
            b_series = pd.Series(smoothed, index=unique_days)

            if loss + 1e-9 < best_loss:
                best_loss = loss
                best_params = InfoParams(**P.as_dict())
                best_b = b_series.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.early_stop_patience:
                    break

        return best_params, best_b

    # ------------------------- Joint refinement (light) --------------------

    def _joint_refine(self,
                      policy: DecisionPolicy,
                      info_params: InfoParams,
                      daily_b: pd.Series,
                      train_days: List[int]) -> Tuple[InfoParams, pd.Series]:
        """
        Simulate on training window and minimize a composite loss over info params
        (policy fixed). This is a light refinement to improve alignment.
        """
        # Build simulator for refinement
        signals = EnvironmentSignals(b_exo=info_params.b_exo, daily_b_exo_series=daily_b)
        sim = Simulator(
            network=self.network,
            agents_df=self.agents_df,
            role_index=self.role_index,
            risk_vec=self.risk_vec,
            policy=policy,
            info_params=info_params,
            signals=signals
        )

        # Build observed matrices
        obs = self.train_df[self.train_df["day"].isin(train_days)].copy()
        obs = obs.merge(self.agents_df[["agent_id"]], on="agent_id", how="inner")
        obs = obs.sort_values(["day", "agent_id"])

        agent_ids = self.agents_df["agent_id"].to_numpy()
        id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        N = len(agent_ids)

        # Initial previous-wear at day0 from earliest observation
        min_day = min(train_days)
        prev_vec = np.zeros(N, dtype=np.float64)
        first_day_df = self.train_df[self.train_df["day"] == min_day]
        for _, row in first_day_df.iterrows():
            i = id_to_idx.get(int(row.agent_id), None)
            if i is not None:
                prev_vec[i] = float(row.wearing_mask)

        P = info_params
        best_P = InfoParams(**P.as_dict())
        best_b = daily_b.copy()
        best_loss = np.inf
        no_improve = 0

        for step in range(self.cfg.inner_refine_steps):
            sim.info_params = P
            sim.signals = EnvironmentSignals(b_exo=P.b_exo, daily_b_exo_series=daily_b)
            wear_prev = prev_vec.copy()
            pred_rows = []
            for d in train_days[1:]:
                wear_t, info_t = sim.step(d, wear_prev)
                pred_rows.append(pd.DataFrame({
                    "day": d,
                    "agent_id": agent_ids,
                    "p_hat": wear_t,   # sampled; coarse for refinement
                }))
                wear_prev = wear_t

            pred = pd.concat(pred_rows, ignore_index=True)
            merged = obs.merge(pred, on=["day", "agent_id"], how="inner")
            # Composite loss: Brier + aggregate RMSE
            y = merged["wearing_mask"].astype(float).to_numpy()
            yhat = merged["p_hat"].astype(float).to_numpy()
            brier = np.mean((y - yhat) ** 2)
            # aggregate prevalence by day
            agg = merged.groupby("day")[["wearing_mask", "p_hat"]].mean()
            rmse = float(np.sqrt(np.mean((agg["wearing_mask"] - agg["p_hat"]) ** 2)))
            loss = brier + rmse

            if loss + 1e-9 < best_loss:
                best_loss = loss
                best_P = InfoParams(**P.as_dict())
                best_b = daily_b.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.early_stop_patience:
                    break

            # Nudged gradient-like update using residual correlations
            resid = (yhat - y)
            # Approximate gradient signs
            sign = np.sign(np.mean(resid))
            P.sig_info0 -= self.cfg.lr_info * 0.1 * sign
            P.kappa -= self.cfg.lr_info * 0.1 * sign
            P.clip_(self.cfg.info_bounds)
            # Small contraction of p_contact to avoid overfitting
            for name in ["p_contact_family", "p_contact_work_school", "p_contact_community"]:
                val = getattr(P, name)
                setattr(P, name, float(np.clip(val * 0.999, *self.cfg.info_bounds[name])))

        return best_P, best_b

    # ------------------------- Public API ---------------------------------

    def fit(self) -> Tuple[DecisionPolicy, InfoParams, pd.Series]:
        """
        Execute two-stage calibration and light joint refinement.

        Returns
        -------
        policy : DecisionPolicy
        info_params : InfoParams
        daily_b_exo_series : pd.Series
        """
        # Stage A: policy
        policy_params = self._fit_policy_by_role(self.train_df)
        self.policy_params = policy_params
        policy = DecisionPolicy(role_to_params=policy_params)

        # Stage B: information process
        info_params, daily_b = self._fit_information_process(self.train_df)

        # Joint refinement on training days
        train_days = sorted(self.train_df["day"].unique())
        info_params_ref, daily_b_ref = self._joint_refine(policy, info_params, daily_b, train_days)

        self.info_params = info_params_ref
        self.daily_b_exo_series = daily_b_ref
        return policy, info_params_ref, daily_b_ref


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Computes evaluation metrics on the validation window:
    - RMSE_aggregate, MAE_aggregate (daily prevalence)
    - Brier (per-agent probability vs outcome)
    - TransitionFit (P01, P11, P10 absolute error)
    """

    @staticmethod
    def compute_metrics(probs_df: pd.DataFrame,
                        samples_df: pd.DataFrame,
                        valid_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute metrics using averaged probabilities and observed validation data.

        Parameters
        ----------
        probs_df : DataFrame
            ['day', 'agent_id', 'p_wear_mean']
        samples_df : DataFrame
            ['day', 'agent_id', 'sampled_wear_mean']  (not essential for metrics)
        valid_df : DataFrame
            ['day', 'agent_id', 'wearing_mask', 'received_info']

        Returns
        -------
        dict
            Metric name -> value.
        """
        df = valid_df.merge(probs_df, on=["day", "agent_id"], how="inner").copy()
        if df.empty:
            raise ValueError("Validation merge produced empty set; check day ranges and IDs.")

        # Brier score
        y = df["wearing_mask"].astype(float).to_numpy()
        p = df["p_wear_mean"].astype(float).to_numpy()
        brier = float(np.mean((y - p) ** 2))

        # Aggregate prevalence curves
        agg = df.groupby("day")[["wearing_mask", "p_wear_mean"]].mean()
        rmse_agg = float(np.sqrt(np.mean((agg["wearing_mask"] - agg["p_wear_mean"]) ** 2)))
        mae_agg = float(np.mean(np.abs(agg["wearing_mask"] - agg["p_wear_mean"])))

        # TransitionFit: compute observed transitions y_{t-1}->y_t and predicted analog
        # Build (agent, day-1) join
        df_prev = df.copy()
        df_prev["day"] = df_prev["day"] + 1  # shift so that join aligns t with t-1
        merged = df.merge(df_prev[["agent_id", "day", "wearing_mask"]].rename(
            columns={"wearing_mask": "wear_tm1"}),
            on=["agent_id", "day"], how="inner"
        )
        if not merged.empty:
            y_t = merged["wearing_mask"].astype(float).to_numpy()
            y_tm1 = merged["wear_tm1"].astype(float).to_numpy()
            # Observed transition rates
            eps = 1e-9
            P01_obs = float(np.sum((y_tm1 == 0) & (y_t == 1)) / (np.sum(y_tm1 == 0) + eps))
            P11_obs = float(np.sum((y_tm1 == 1) & (y_t == 1)) / (np.sum(y_tm1 == 1) + eps))
            P10_obs = float(np.sum((y_tm1 == 1) & (y_t == 0)) / (np.sum(y_tm1 == 1) + eps))
            # Proxy predicted transitions: threshold p_wear_mean at 0.5 for simplicity
            yhat_t = (merged["p_wear_mean"] >= 0.5).astype(float).to_numpy()
            P01_hat = float(np.sum((y_tm1 == 0) & (yhat_t == 1)) / (np.sum(y_tm1 == 0) + eps))
            P11_hat = float(np.sum((y_tm1 == 1) & (yhat_t == 1)) / (np.sum(y_tm1 == 1) + eps))
            P10_hat = float(np.sum((y_tm1 == 1) & (yhat_t == 0)) / (np.sum(y_tm1 == 1) + eps))
            trans_err = abs(P01_obs - P01_hat) + abs(P11_obs - P11_hat) + abs(P10_obs - P10_hat)
        else:
            trans_err = float("nan")

        return {
            "RMSE_aggregate": rmse_agg,
            "MAE_aggregate": mae_agg,
            "Brier": brier,
            "TransitionFit_abs_err_sum": float(trans_err),
        }


# ---------------------------------------------------------------------------
# Data loading and holdout
# ---------------------------------------------------------------------------

def load_data(agent_file: str, network_file: str, train_file: str) -> Tuple[pd.DataFrame, MultiplexNetwork, pd.DataFrame]:
    """
    Load and validate input datasets.

    Parameters
    ----------
    agent_file : str
        File name under DATA_DIR (e.g., 'agent_attributes.csv').
    network_file : str
        File name under DATA_DIR (e.g., 'social_network.json').
    train_file : str
        File name under DATA_DIR (e.g., 'train_data.csv').

    Returns
    -------
    agents_df, network, train_df

    Raises
    ------
    FileNotFoundError, ValueError
    """
    # Paths (absolute) using required pattern
    agent_path = os.path.join(DATA_DIR, agent_file)
    network_path = os.path.join(DATA_DIR, network_file)
    train_path = os.path.join(DATA_DIR, train_file)

    # Validate files exist
    for pth, name in [(agent_path, "agent_attributes.csv"),
                      (network_path, "social_network.json"),
                      (train_path, "train_data.csv")]:
        if not os.path.isabs(pth):
            raise ValueError(f"Path must be absolute (got {pth}). Ensure PROJECT_ROOT and DATA_PATH are absolute and joined correctly.")
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Required data file missing: {pth} ({name})")

    # Load agents
    agents_df = pd.read_csv(agent_path)
    required_cols = {
        "agent_id", "age", "age_group", "occupation", "risk_perception", "initial_mask_wearing",
        "family_connections", "work_school_connections", "community_connections", "total_connections"
    }
    if not required_cols.issubset(set(agents_df.columns)):
        missing = required_cols - set(agents_df.columns)
        raise ValueError(f"agent_attributes.csv missing columns: {missing}")

    # Load network
    network = MultiplexNetwork.from_json(network_path)

    # Validate coverage
    agent_ids = set(agents_df["agent_id"].astype(int).tolist())
    network_ids = set(network.layer_adj["all"].keys())
    if not agent_ids.issubset(network_ids):
        missing_in_net = sorted(list(agent_ids - network_ids))[:10]
        warnings.warn(f"{len(agent_ids - network_ids)} agents in attributes missing from network. Example: {missing_in_net}. They will be treated as isolated.")
    # Add isolated entries for missing nodes
    for aid in (agent_ids - network_ids):
        for L in ["family", "work_school", "community", "all"]:
            network.layer_adj[L][int(aid)] = []

    # Load train panel
    train_df = pd.read_csv(train_path)
    req_train = {"day", "agent_id", "wearing_mask", "received_info"}
    if not req_train.issubset(set(train_df.columns)):
        missing = req_train - set(train_df.columns)
        raise ValueError(f"train_data.csv missing columns: {missing}")

    # Coerce dtypes
    train_df["day"] = train_df["day"].astype(int)
    train_df["agent_id"] = train_df["agent_id"].astype(int)
    train_df["wearing_mask"] = train_df["wearing_mask"].astype(bool)
    train_df["received_info"] = train_df["received_info"].astype(bool)

    # Initialization from day 0 if present, else from attributes
    earliest = train_df.groupby("agent_id")["day"].min().rename("min_day")
    init_join = train_df.merge(earliest, on="agent_id", how="left")
    init_mask_from_data = init_join[init_join["day"] == init_join["min_day"]][["agent_id", "wearing_mask"]].rename(
        columns={"wearing_mask": "initial_mask_from_data"}
    )
    agents_df = agents_df.merge(init_mask_from_data, on="agent_id", how="left")
    agents_df["initial_mask"] = agents_df["initial_mask_from_data"].fillna(agents_df["initial_mask_wearing"]).astype(bool)
    agents_df = agents_df.drop(columns=["initial_mask_from_data"])

    # Degree validation (soft)
    deg_check = []
    for L, col in [("family", "family_connections"),
                   ("work_school", "work_school_connections"),
                   ("community", "community_connections")]:
        degs = []
        for aid in agents_df["agent_id"]:
            degs.append(len(network.neighbors(int(aid), L)))
        agents_df[f"{L}_degree_from_net"] = degs
        # record discrepancy rate
        diff = np.mean(np.abs(agents_df[col].to_numpy() - agents_df[f"{L}_degree_from_net"].to_numpy()))
        deg_check.append((L, diff))
    for L, diff in deg_check:
        if diff > 5:
            warnings.warn(f"Large discrepancy between declared and network degrees on layer '{L}' (mean abs diff ~ {diff:.2f}).")

    return agents_df, network, train_df


def build_network_and_agents(agents_df: pd.DataFrame,
                             network: MultiplexNetwork) -> Tuple[pd.DataFrame, NDArray[np.int64], NDArray[np.float64]]:
    """
    Prepare agent array order, role indices, and risk vector for simulation.

    Returns
    -------
    agents_df_sorted, role_index, risk_vec
    """
    agents_df = agents_df.sort_values("agent_id").reset_index(drop=True)
    role_index = agents_df["occupation"].apply(Agent._map_role).apply(lambda r: ROLES.index(r)).astype(int).to_numpy()
    risk_vec = agents_df["risk_perception"].astype(float).to_numpy()
    return agents_df, role_index, risk_vec


def holdout_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal holdout: first 80% of unique days for training, last 20% for validation.

    Returns
    -------
    train_win, valid_win
    """
    unique_days = sorted(train_df["day"].unique())
    if len(unique_days) < 5:
        raise ValueError("Not enough days for temporal holdout (need >= 5 unique days).")
    split_idx = int(math.floor(0.8 * len(unique_days))) - 1
    split_idx = max(0, min(split_idx, len(unique_days) - 2))
    train_days = unique_days[:split_idx + 1]
    valid_days = unique_days[split_idx + 1:]

    train_win = train_df[train_df["day"].isin(train_days)].copy()
    valid_win = train_df[train_df["day"].isin(valid_days)].copy()
    return train_win, valid_win


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(output_dir: str,
                 policy_params: Dict[str, PolicyParams],
                 info_params: InfoParams,
                 probs_df: pd.DataFrame,
                 samples_df: pd.DataFrame,
                 metrics: Dict[str, float],
                 seed: int,
                 extra: Optional[Dict] = None) -> None:
    """
    Save calibrated parameters, trajectories, and metrics to disk.

    Parameters
    ----------
    output_dir : str
        Directory to create (under PROJECT_ROOT) for outputs.
    policy_params : dict
    info_params : InfoParams
    probs_df, samples_df : DataFrames
    metrics : dict
    seed : int
    extra : dict, optional
    """
    # Outputs saved under PROJECT_ROOT/<output_dir>
    out_dir_abs = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    # Parameters JSON
    params = {
        "seed": seed,
        "policy_params": {role: asdict(pp) for role, pp in policy_params.items()},
        "info_params": info_params.as_dict(),
    }
    if extra:
        params["extra"] = extra
    with open(os.path.join(out_dir_abs, "calibrated_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Trajectories
    probs_df.to_csv(os.path.join(out_dir_abs, "sim_probs_validation.csv"), index=False)
    samples_df.to_csv(os.path.join(out_dir_abs, "sim_samples_validation.csv"), index=False)

    # Metrics
    mdf = pd.DataFrame([metrics])
    mdf.to_csv(os.path.join(out_dir_abs, "metrics_validation.csv"), index=False)

    # Minimal README
    with open(os.path.join(out_dir_abs, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Outputs generated by Mask-Wearing Diffusion Simulator\n"
            "- calibrated_params.json : Calibrated parameters and seed\n"
            "- sim_probs_validation.csv : Agent-level mean probabilities on validation days\n"
            "- sim_samples_validation.csv : Agent-level mean sampled wearing on validation days\n"
            "- metrics_validation.csv : Evaluation metrics on validation window\n"
        )


# ---------------------------------------------------------------------------
# CLI parsing (optional but included)
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments with sensible defaults. All file paths are
    resolved under DATA_DIR following the required pattern.
    """
    parser = argparse.ArgumentParser(description="Multiplex Mask-Wearing Diffusion Simulator")
    parser.add_argument("--agent-file", type=str, default="agent_attributes.csv",
                        help="Agent attributes CSV filename under DATA_DIR.")
    parser.add_argument("--network-file", type=str, default="social_network.json",
                        help="Multiplex network JSON filename under DATA_DIR.")
    parser.add_argument("--train-file", type=str, default="train_data.csv",
                        help="Training panel CSV filename under DATA_DIR.")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory under PROJECT_ROOT.")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument("--k-runs", type=int, default=5, help="K stochastic forward runs to average.")
    parser.add_argument("--max-iter-policy", type=int, default=400, help="Max iterations for policy calibration.")
    parser.add_argument("--max-iter-info", type=int, default=400, help="Max iterations for info calibration.")
    parser.add_argument("--lr-policy", type=float, default=0.1, help="Learning rate for policy.")
    parser.add_argument("--lr-info", type=float, default=0.05, help="Learning rate for information process.")
    parser.add_argument("--l2-reg", type=float, default=0.01, help="L2 regularization for logistic regression.")
    parser.add_argument("--inner-refine-steps", type=int, default=100, help="Joint refinement steps.")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--memory-decay", type=float, default=1.0, help="Memory decay for peer averaging (0..1).")
    args = parser.parse_args()

    # Validate ranges
    if not (0 <= args.k_runs <= 1000):
        raise ValueError("--k-runs must be between 0 and 1000.")
    if args.seed < 0 or args.seed >= 2 ** 31:
        raise ValueError("--seed must be in [0, 2**31 - 1].")
    if not (0.0 < args.lr_policy <= 1.0) or not (0.0 < args.lr_info <= 1.0):
        raise ValueError("--lr-policy and --lr-info must be in (0, 1].")
    if not (0.0 <= args.l2_reg <= 10.0):
        raise ValueError("--l2-reg must be in [0, 10].")
    if not (0.0 <= args.memory_decay <= 1.0):
        raise ValueError("--memory-decay must be in [0, 1].")

    return args


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the end-to-end pipeline:

    parse_cli() → load_data() → build_network_and_agents() → holdout_split()
    → calibrator.fit() → simulator.rollout() → evaluator.compute_metrics()
    → save_results()
    """
    # 1) Parse CLI
    args = parse_cli()

    # 2) Set seed
    set_global_seed(args.seed)

    # 3) Load data
    agents_df, network, full_df = load_data(
        agent_file=args.agent_file,
        network_file=args.network_file,
        train_file=args.train_file
    )

    # 4) Build network/agents arrays
    agents_df, role_index, risk_vec = build_network_and_agents(agents_df, network)

    # 5) Temporal holdout
    train_win, valid_win = holdout_split(full_df)

    # 6) Calibration
    cfg = SimulationConfig(
        seed=args.seed,
        k_runs=int(args.k_runs),
        max_iter_policy=int(args.max_iter_policy),
        max_iter_info=int(args.max_iter_info),
        lr_policy=float(args.lr_policy),
        lr_info=float(args.lr_info),
        l2_reg=float(args.l2_reg),
        output_dir=str(args.output_dir),
        inner_refine_steps=int(args.inner_refine_steps),
        early_stop_patience=int(args.early_stop_patience),
        memory_decay=float(args.memory_decay),
    )
    calibrator = Calibrator(cfg, network, agents_df, train_win, role_index, risk_vec)
    policy, info_params, daily_b = calibrator.fit()

    # 7) Forward simulation on validation window
    # Initialize wear at last training day for each agent
    last_train_day = max(train_win["day"].unique())
    last_train = train_win[train_win["day"] == last_train_day][["agent_id", "wearing_mask"]]
    # Ensure all agents present
    base = agents_df[["agent_id"]].merge(last_train, on="agent_id", how="left")
    init_wear = base["wearing_mask"].fillna(agents_df["initial_mask"]).astype(bool).astype(float).to_numpy()

    # Simulator with calibrated params
    signals = EnvironmentSignals(b_exo=info_params.b_exo, daily_b_exo_series=daily_b)
    simulator = Simulator(network, agents_df, role_index, risk_vec, policy, info_params, signals)

    valid_days = sorted(valid_win["day"].unique())
    if len(valid_days) == 0:
        raise ValueError("Validation window is empty after split; cannot simulate.")
    start_day, end_day = valid_days[0], valid_days[-1]

    probs_df, samples_df = simulator.run(
        start_day=start_day,
        end_day=end_day,
        init_wear=init_wear,
        k_runs=cfg.k_runs
    )

    # 8) Evaluate metrics
    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(probs_df, samples_df, valid_win)

    # 9) Save results
    save_results(
        output_dir=cfg.output_dir,
        policy_params=policy.role_to_params,
        info_params=info_params,
        probs_df=probs_df,
        samples_df=samples_df,
        metrics=metrics,
        seed=cfg.seed,
        extra={
            "daily_b_exo_series": daily_b.to_dict() if daily_b is not None else None,
            "train_days": sorted(train_win["day"].unique()),
            "valid_days": sorted(valid_win["day"].unique()),
        }
    )

    # 10) Minimal console report
    print("=== Validation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"Results saved under: {os.path.join(PROJECT_ROOT, cfg.output_dir)}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Provide actionable message and exit non-zero
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
