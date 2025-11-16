import json
import math
import os
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings
warnings.filterwarnings("ignore")

# EvoTorch imports for evolutionary algorithm
try:
    import evotorch
    from evotorch import Problem
    from evotorch.algorithms import SNES, CEM, GeneticAlgorithm
    from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver
    EVOTORCH_AVAILABLE = True
except ImportError:
    EVOTORCH_AVAILABLE = False
    print("Warning: EvoTorch not available. EvoCalibrator will not be functional.")


@dataclass
class SimulationConfig:
    data_folder: str = "data_fitting/mask_adoption_data"
    seed: int = 42
    k_runs: int = 20
    l2_reg: float = 1.0
    max_iter: int = 400
    learning_rate: float = 0.1
    val_split_ratio: float = 0.8
    gov_intervention_day: int = 10
    gov_lam_factor_default: float = 1.5
    rho_info_decay_default: float = 0.5
    output_folder: str = "outputs_EvoCalibrator"
    forecast_days: int = 10  # Days 30-39 if training is 0-29
    verbose: bool = True


@dataclass
class Parameters:
    # Decision model
    alpha: float
    gamma: float
    theta_f: float
    theta_w: float
    theta_c: float
    beta_r: float
    beta_i: float
    age_effects: Dict[str, float]
    occ_effects: Dict[str, float]
    tau: float
    # Layer weights derived and normalized
    w_family: float
    w_work: float
    w_community: float
    # Info propagation
    phi_family: float
    phi_work: float
    phi_community: float
    lambda_broadcast_base: float
    lambda_broadcast_factor_after_day10: float
    rho_info_decay: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class FittedParams:
    """Container for all parameters needed by the simulator."""
    decision_weights: Dict[str, float]              # e.g., b0, b_prev, wF, wW, wC, b_info, b_risk, etc.
    layer_weights: Dict[str, float]                 # e.g., family, work_school, community
    info_params: Dict[str, float]                   # e.g., campaign_intensity, gamma_info, memory_decay
    noise_params: Dict[str, float]                  # e.g., temperature
    meta: Dict[str, Any]                            # e.g., seed, calibrator_name, training_window, notes
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_parameters(self) -> Parameters:
        """Convert FittedParams to legacy Parameters format."""
        # Extract values from structured dictionaries
        alpha = self.decision_weights.get('alpha', 0.0)
        gamma = self.decision_weights.get('gamma', 0.0)
        theta_f = self.decision_weights.get('theta_f', 1.0)
        theta_w = self.decision_weights.get('theta_w', 1.0)
        theta_c = self.decision_weights.get('theta_c', 1.0)
        beta_r = self.decision_weights.get('beta_r', 0.0)
        beta_i = self.decision_weights.get('beta_i', 0.0)
        tau = self.noise_params.get('tau', 1.0)
        
        # Layer weights
        w_family = self.layer_weights.get('family', 1.0)
        w_work = self.layer_weights.get('work_school', 1.0)
        w_community = self.layer_weights.get('community', 1.0)
        
        # Info params
        phi_family = self.info_params.get('phi_family', 0.1)
        phi_work = self.info_params.get('phi_work', 0.1)
        phi_community = self.info_params.get('phi_community', 0.1)
        lambda_broadcast_base = self.info_params.get('lambda_broadcast_base', 0.05)
        lambda_broadcast_factor_after_day10 = self.info_params.get('lambda_broadcast_factor_after_day10', 1.5)
        rho_info_decay = self.info_params.get('rho_info_decay', 0.5)
        
        # Age and occupation effects
        age_effects = self.decision_weights.get('age_effects', {})
        occ_effects = self.decision_weights.get('occ_effects', {})
        
        return Parameters(
            alpha=alpha, gamma=gamma, theta_f=theta_f, theta_w=theta_w, theta_c=theta_c,
            beta_r=beta_r, beta_i=beta_i, age_effects=age_effects, occ_effects=occ_effects,
            tau=tau, w_family=w_family, w_work=w_work, w_community=w_community,
            phi_family=phi_family, phi_work=phi_work, phi_community=phi_community,
            lambda_broadcast_base=lambda_broadcast_base, 
            lambda_broadcast_factor_after_day10=lambda_broadcast_factor_after_day10,
            rho_info_decay=rho_info_decay
        )


class Calibrator(ABC):
    """Pluggable calibrator interface with a stable evaluation callback signature."""
    
    @abstractmethod
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """Return FittedParams, fitted strictly on the training window."""
        pass


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_agent_attributes(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load agent_attributes from {path}: {e}") from e


def load_social_network(path: str) -> Dict[str, Dict[str, List[int]]]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load social_network from {path}: {e}") from e


def load_train_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load train_data from {path}: {e}") from e


def align_ids(
    agents_df: pd.DataFrame,
    social: Dict[str, Dict[str, List[int]]],
    train_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[int, int], pd.DataFrame, Dict[str, Dict[str, List[int]]], pd.DataFrame]:
    agents_ids = set(agents_df["agent_id"].astype(int).tolist())
    social_ids = set(int(k) for k in social.keys())
    train_ids = set(train_df["agent_id"].astype(int).unique().tolist())
    common = sorted(list(agents_ids & social_ids & train_ids))
    if len(common) == 0:
        raise RuntimeError("No common agent IDs across agent_attributes.csv, social_network.json, and train_data.csv")
    id2idx = {aid: i for i, aid in enumerate(common)}
    # Filter dataframes
    agents_df_f = agents_df[agents_df["agent_id"].isin(common)].copy()
    train_df_f = train_df[train_df["agent_id"].isin(common)].copy()
    # Filter social to only common ids and neighbors in common
    social_f: Dict[str, Dict[str, List[int]]] = {}
    for k, v in social.items():
        ik = int(k)
        if ik in id2idx:
            social_f[k] = {
                "family": [int(x) for x in v.get("family", []) if int(x) in id2idx],
                "work_school": [int(x) for x in v.get("work_school", []) if int(x) in id2idx],
                "community": [int(x) for x in v.get("community", []) if int(x) in id2idx],
            }
    return np.array(common, dtype=int), id2idx, agents_df_f, social_f, train_df_f


def build_multiplex_adjacency(
    social: Dict[str, Dict[str, List[int]]],
    id2idx: Dict[int, int],
    n: int,
) -> Dict[str, List[np.ndarray]]:
    layers = ["family", "work_school", "community"]
    adj: Dict[str, List[set]] = {layer: [set() for _ in range(n)] for layer in layers}
    # Symmetrize and deduplicate
    for k_str, v in social.items():
        i = id2idx[int(k_str)]
        for layer in layers:
            for nbr in v.get(layer, []):
                if nbr in id2idx:
                    j = id2idx[nbr]
                    if i != j:
                        adj[layer][i].add(j)
                        adj[layer][j].add(i)
    # Convert to numpy arrays
    adj_arrays: Dict[str, List[np.ndarray]] = {}
    for layer in layers:
        arr_list: List[np.ndarray] = []
        for i in range(n):
            if len(adj[layer][i]) == 0:
                arr_list.append(np.array([], dtype=int))
            else:
                arr_list.append(np.fromiter(adj[layer][i], dtype=int))
        adj_arrays[layer] = arr_list
    return adj_arrays


def compute_layer_neighbor_share(states: np.ndarray, neighbors: List[np.ndarray]) -> np.ndarray:
    n = states.shape[0]
    shares = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = neighbors[i]
        if neigh.size == 0:
            shares[i] = 0.0
        else:
            shares[i] = float(np.mean(states[neigh]))
    return shares


def pivot_states(train_df: pd.DataFrame, id2idx: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    # Determine days
    days = sorted(train_df["day"].unique().tolist())
    n_days = len(days)
    n_agents = len(id2idx)
    wearing = np.zeros((n_days, n_agents), dtype=np.float64)
    received = np.zeros((n_days, n_agents), dtype=np.float64)
    # Sort train_df for efficient filling
    df_sorted = train_df.sort_values(["day", "agent_id"])
    day_to_idx = {d: i for i, d in enumerate(days)}
    for _, row in df_sorted.iterrows():
        d = int(row["day"])
        a = int(row["agent_id"])
        i_day = day_to_idx[d]
        i_agent = id2idx[a]
        wearing[i_day, i_agent] = 1.0 if bool(row["wearing_mask"]) else 0.0
        received[i_day, i_agent] = 1.0 if bool(row["received_info"]) else 0.0
    return wearing, received, days


def encode_demographics(agents_df: pd.DataFrame, common_ids: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    # Map to row order as common_ids
    idx_series = pd.Series(np.arange(len(common_ids)), index=common_ids)
    df_sorted = agents_df.set_index("agent_id").loc[common_ids]
    age_groups = df_sorted["age_group"].astype(str).tolist()
    occs = df_sorted["occupation"].astype(str).tolist()
    # Unique categories
    age_cats_all = sorted(list(pd.unique(df_sorted["age_group"].astype(str))))
    occ_cats_all = sorted(list(pd.unique(df_sorted["occupation"].astype(str))))
    # Baselines
    age_baseline = "Middle Age" if "Middle Age" in age_cats_all else age_cats_all[0]
    occ_baseline = "White Collar" if "White Collar" in occ_cats_all else occ_cats_all[0]
    age_cats = [c for c in age_cats_all if c != age_baseline]
    occ_cats = [c for c in occ_cats_all if c != occ_baseline]
    # Build one-hot excluding baseline
    n = len(common_ids)
    age_oh = np.zeros((n, len(age_cats)), dtype=np.float64)
    occ_oh = np.zeros((n, len(occ_cats)), dtype=np.float64)
    age_index_map = {c: idx for idx, c in enumerate(age_cats)}
    occ_index_map = {c: idx for idx, c in enumerate(occ_cats)}
    for i in range(n):
        ag = age_groups[i]
        oc = occs[i]
        if ag in age_index_map:
            age_oh[i, age_index_map[ag]] = 1.0
        if oc in occ_index_map:
            occ_oh[i, occ_index_map[oc]] = 1.0
    age_cat_names = age_cats  # column order
    occ_cat_names = occ_cats
    return age_oh, age_cat_names, occ_oh, occ_cat_names


def compute_mem_info(received_info: np.ndarray, rho: float) -> np.ndarray:
    # received_info: days x agents (0/1)
    T, N = received_info.shape
    mem = np.zeros((T, N), dtype=np.float64)
    for t in range(1, T):
        mem[t, :] = rho * mem[t - 1, :] + (1.0 - rho) * received_info[t, :]
    return mem


def build_train_validation_splits(days: List[int], ratio: float) -> Tuple[int, int, int]:
    # Returns indices for train_end_exclusive, val_start_inclusive, val_end_exclusive
    T = len(days)
    split_idx = int(math.floor(ratio * T))
    train_end = max(1, split_idx)  # at least 1 to allow t-1
    val_start = train_end
    val_end = T
    return train_end, val_start, val_end


def build_feature_matrix(
    wearing: np.ndarray,
    mem_info: np.ndarray,
    share_f_by_day: np.ndarray,
    share_w_by_day: np.ndarray,
    share_c_by_day: np.ndarray,
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    day_start: int,
    day_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Build dataset for days [day_start, day_end) predicting wearing[t] from features at t-1 and mem_info[t]
    # wearing: T x N
    # share_*_by_day indexed by same T x N using wearing[t-1]
    T, N = wearing.shape
    assert 0 <= day_start < day_end <= T
    rows = []
    labels = []
    for t in range(day_start, day_end):
        wear_prev = wearing[t - 1, :]
        share_f = share_f_by_day[t - 1, :]
        share_w = share_w_by_day[t - 1, :]
        share_c = share_c_by_day[t - 1, :]
        mem_t = mem_info[t, :]
        # Constant features
        intercept = np.ones(N, dtype=np.float64)
        # risk is per-agent
        # concatenate: [intercept, wear_prev, share_f, share_w, share_c, risk, mem_t, age_oh, occ_oh]
        base = np.stack([intercept, wear_prev, share_f, share_w, share_c, risk, mem_t], axis=1)  # N x 7
        # Append demographics
        if age_oh.shape[1] > 0:
            base = np.concatenate([base, age_oh], axis=1)
        if occ_oh.shape[1] > 0:
            base = np.concatenate([base, occ_oh], axis=1)
        rows.append(base)
        labels.append(wearing[t, :])
    X = np.vstack(rows)  # (N*(day_end-day_start)) x n_features
    y = np.concatenate(labels, axis=0)  # (N*(day_end-day_start), )
    return X, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1.0,
    max_iter: int = 400,
    lr: float = 0.1,
    verbose: bool = False,
) -> np.ndarray:
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    # Exclude intercept from regularization (index 0)
    reg_mask = np.ones(n_features, dtype=np.float64)
    reg_mask[0] = 0.0
    # Adam optimizer
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for it in range(1, max_iter + 1):
        z = X @ w
        p = sigmoid(z)
        # Gradient of negative log-likelihood with L2
        grad = X.T @ (p - y) / n_samples + l2_reg * reg_mask * w / n_samples
        # Adam updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** it)
        v_hat = v / (1 - beta2 ** it)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if verbose and (it % 20 == 0 or it == max_iter):
            nll = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) / n_samples
            reg_term = 0.5 * l2_reg * np.sum((reg_mask * w) ** 2) / n_samples
            total_loss = nll + reg_term
            print(f"[fit_logistic_l2] Iter {it:3d}: nll={nll:.5f} reg={reg_term:.5f} total={total_loss:.5f}")
    return w


def derive_layer_weights_and_betas(theta_f: float, theta_w: float, theta_c: float) -> Tuple[float, float, float, float, float, float]:
    # Convert layer coefficients into normalized weights and beta magnitudes
    coefs = np.array([theta_f, theta_w, theta_c], dtype=np.float64)
    abs_coefs = np.abs(coefs)
    total = np.sum(abs_coefs)
    if total <= 1e-12:
        # default equal weights
        w = np.array([1/3, 1/3, 1/3], dtype=np.float64)
    else:
        w = abs_coefs / total
    beta_f, beta_w, beta_c = float(abs_coefs[0]), float(abs_coefs[1]), float(abs_coefs[2])
    return float(w[0]), float(w[1]), float(w[2]), beta_f, beta_w, beta_c


def calibrate_info_params_simple(
    received_info: np.ndarray,
    share_f_by_day: np.ndarray,
    share_w_by_day: np.ndarray,
    share_c_by_day: np.ndarray,
    gov_intervention_day: int,
    default_factor: float,
) -> Tuple[float, float, float, float, float]:
    # Use daily population averages to infer lambda_broadcast_base and factor, with heuristic phi's
    T, N = received_info.shape
    p_obs = received_info.mean(axis=1)  # per-day prevalence
    sf = share_f_by_day.mean(axis=1)
    sw = share_w_by_day.mean(axis=1)
    sc = share_c_by_day.mean(axis=1)

    # Heuristic phi values
    phi_f = 0.3
    phi_w = 0.2
    phi_c = 0.1

    # Pre and post intervention means
    pre_mask = np.arange(T) < gov_intervention_day
    post_mask = ~pre_mask
    if pre_mask.sum() == 0:
        pre_mask[:] = True
    # Compute means safely
    p0 = float(p_obs[pre_mask].mean()) if pre_mask.any() else float(p_obs.mean())
    s0 = float((phi_f * sf[pre_mask] + phi_w * sw[pre_mask] + phi_c * sc[pre_mask]).mean()) if pre_mask.any() else float((phi_f * sf + phi_w * sw + phi_c * sc).mean())
    lam_base = max(0.0, -math.log(max(1e-9, 1.0 - p0)) - s0)

    if post_mask.any():
        p1 = float(p_obs[post_mask].mean())
        s1 = float((phi_f * sf[post_mask] + phi_w * sw[post_mask] + phi_c * sc[post_mask]).mean())
        lam1 = max(0.0, -math.log(max(1e-9, 1.0 - p1)) - s1)
        lam_factor = (lam1 / lam_base) if lam_base > 1e-9 else default_factor
        lam_factor = max(1.0, min(5.0, lam_factor))
    else:
        lam_factor = default_factor

    # Clamp lambda
    lam_base = float(max(0.0, min(0.5, lam_base)))
    return phi_f, phi_w, phi_c, lam_base, lam_factor


def simulate_step_info(
    states_prev: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    phi_f: float,
    phi_w: float,
    phi_c: float,
    lambda_broadcast: float,
) -> np.ndarray:
    # Compute neighbor shares
    share_f = compute_layer_neighbor_share(states_prev, neighbors["family"])
    share_w = compute_layer_neighbor_share(states_prev, neighbors["work_school"])
    share_c = compute_layer_neighbor_share(states_prev, neighbors["community"])
    # Probability of receiving info
    u = phi_f * share_f + phi_w * share_w + phi_c * share_c + lambda_broadcast
    p_info = 1.0 - np.exp(-np.clip(u, 0.0, 50.0))
    rec = (np.random.rand(states_prev.shape[0]) < p_info).astype(np.float64)
    return rec


def compute_logit(
    prev_states: np.ndarray,
    share_f: np.ndarray,
    share_w: np.ndarray,
    share_c: np.ndarray,
    risk: np.ndarray,
    mem_info: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    params: Parameters,
    age_cat_names: List[str],
    occ_cat_names: List[str],
) -> np.ndarray:
    # Reconstruct age and occ effect vectors aligned to columns in age_oh and occ_oh
    age_effects_vec = np.array([params.age_effects.get(cat, 0.0) for cat in age_cat_names], dtype=np.float64)
    occ_effects_vec = np.array([params.occ_effects.get(cat, 0.0) for cat in occ_cat_names], dtype=np.float64)
    logits = (
        params.alpha
        + params.gamma * prev_states
        + params.theta_f * share_f
        + params.theta_w * share_w
        + params.theta_c * share_c
        + params.beta_r * risk
        + params.beta_i * mem_info
    )
    if age_oh.shape[1] > 0:
        logits += age_oh @ age_effects_vec
    if occ_oh.shape[1] > 0:
        logits += occ_oh @ occ_effects_vec
    return logits


def simulate_window(
    start_states: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    age_cat_names: List[str],
    occ_cat_names: List[str],
    params: Parameters,
    start_day_index: int,
    end_day_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Returns arrays: states over days [start_day_index+1..end_day_index], info_received, probabilities
    N = start_states.shape[0]
    days_count = end_day_index - start_day_index
    states = np.zeros((days_count, N), dtype=np.float64)
    info = np.zeros((days_count, N), dtype=np.float64)
    probs = np.zeros((days_count, N), dtype=np.float64)
    prev_states = start_states.copy()
    mem = np.zeros(N, dtype=np.float64)
    for d in range(days_count):
        global_day = start_day_index + d + 1
        # Determine lambda with intervention
        lam = params.lambda_broadcast_base * (params.lambda_broadcast_factor_after_day10 if global_day >= 10 else 1.0)
        # Info step
        rec = simulate_step_info(prev_states, neighbors, params.phi_family, params.phi_work, params.phi_community, lam)
        info[d, :] = rec
        mem = params.rho_info_decay * mem + (1.0 - params.rho_info_decay) * rec
        # Neighbor shares for decision
        share_f = compute_layer_neighbor_share(prev_states, neighbors["family"])
        share_w = compute_layer_neighbor_share(prev_states, neighbors["work_school"])
        share_c = compute_layer_neighbor_share(prev_states, neighbors["community"])
        logits = compute_logit(prev_states, share_f, share_w, share_c, risk, mem, age_oh, occ_oh, params, age_cat_names, occ_cat_names)
        if params.tau is not None and params.tau > 0:
            logits = logits / params.tau
        p = sigmoid(logits)
        probs[d, :] = p
        new_states = (np.random.rand(N) < p).astype(np.float64)
        states[d, :] = new_states
        prev_states = new_states
    return states, info, probs


def evaluate_params(simulator, params: FittedParams, window) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).
    """
    # Convert FittedParams to Parameters for compatibility
    legacy_params = params.to_parameters()
    
    # Extract window information
    wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg = window
    train_start, train_end = train_window
    
    # Get age and occupation category names from actual data
    n_age_cats = age_oh.shape[1]
    n_occ_cats = occ_oh.shape[1]
    
    # Use actual category names based on data dimensions
    age_cat_names = [f'age_cat_{i}' for i in range(n_age_cats)]
    occ_cat_names = [f'occ_cat_{i}' for i in range(n_occ_cats)]
    
    # Run simulation on the training window
    result = evaluate_on_validation(
        wearing, neighbors, risk, age_oh, occ_oh,
        age_cat_names, occ_cat_names, legacy_params, 
        train_start, train_end, cfg.k_runs
    )
    
    return result


def evaluate_on_validation(
    wearing: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    age_cat_names: List[str],
    occ_cat_names: List[str],
    params: Parameters,
    val_start_idx: int,
    val_end_idx: int,
    k_runs: int,
) -> Dict[str, Any]:
    # Initial state is wearing[val_start_idx-1, :]
    init_states = wearing[val_start_idx - 1, :]
    T_val = val_end_idx - val_start_idx
    N = wearing.shape[1]
    # Observed daily rates
    obs_rates = wearing[val_start_idx:val_end_idx, :].mean(axis=1)
    # Observed transitions overall across val window
    prev_obs = wearing[val_start_idx - 1:val_end_idx - 1, :]
    curr_obs = wearing[val_start_idx:val_end_idx, :]
    # Construct observed transition probabilities
    def transition_probs(prev: np.ndarray, curr: np.ndarray) -> Dict[str, float]:
        # Flatten
        prev_f = prev.flatten()
        curr_f = curr.flatten()
        n = prev_f.shape[0]
        p01 = np.mean((prev_f == 0.0) & (curr_f == 1.0))
        p11 = np.mean((prev_f == 1.0) & (curr_f == 1.0))
        p10 = np.mean((prev_f == 1.0) & (curr_f == 0.0))
        p00 = np.mean((prev_f == 0.0) & (curr_f == 0.0))
        return {"P01": p01, "P11": p11, "P10": p10, "P00": p00}

    obs_trans = transition_probs(prev_obs, curr_obs)

    run_rmse = []
    run_mae = []
    run_brier = []
    run_trans_err = []

    daily_rate_runs = []

    for r in range(k_runs):
        sim_states, sim_info, sim_probs = simulate_window(
            start_states=init_states,
            neighbors=neighbors,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            params=params,
            start_day_index=val_start_idx - 1,
            end_day_index=val_end_idx - 1,
        )
        # sim_states: T_val x N, sim_probs: T_val x N
        sim_rates = sim_states.mean(axis=1)
        daily_rate_runs.append(sim_rates)
        # RMSE and MAE on aggregate rates
        rmse = math.sqrt(float(np.mean((sim_rates - obs_rates) ** 2)))
        mae = float(np.mean(np.abs(sim_rates - obs_rates)))
        run_rmse.append(rmse)
        run_mae.append(mae)
        # Brier: use predicted probabilities vs observed wearing
        brier = float(np.mean((sim_probs - wearing[val_start_idx:val_end_idx, :]) ** 2))
        run_brier.append(brier)
        # Transitions from simulation
        prev_sim = np.vstack([init_states.reshape(1, -1), sim_states[:-1, :]])
        sim_trans = transition_probs(prev_sim, sim_states)
        trans_err = float(np.mean([abs(sim_trans[k] - obs_trans[k]) for k in ["P01", "P11", "P10", "P00"]]))
        run_trans_err.append(trans_err)

    def mean_ci(arr: List[float]) -> Tuple[float, float]:
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci = 1.96 * s / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return m, ci

    rmse_mean, rmse_ci = mean_ci(run_rmse)
    mae_mean, mae_ci = mean_ci(run_mae)
    brier_mean, brier_ci = mean_ci(run_brier)
    trans_mean, trans_ci = mean_ci(run_trans_err)
    daily_rate_runs_arr = np.vstack(daily_rate_runs)  # k_runs x T_val
    daily_mean = daily_rate_runs_arr.mean(axis=0)
    daily_std = daily_rate_runs_arr.std(axis=0, ddof=1) if k_runs > 1 else np.zeros(T_val)
    daily_ci = 1.96 * daily_std / math.sqrt(k_runs) if k_runs > 1 else daily_std

    metrics = {
        "RMSE_aggregate_mean": rmse_mean,
        "RMSE_aggregate_CI95": rmse_ci,
        "MAE_aggregate_mean": mae_mean,
        "MAE_aggregate_CI95": mae_ci,
        "Brier_mean": brier_mean,
        "Brier_CI95": brier_ci,
        "TransitionFit_mean": trans_mean,
        "TransitionFit_CI95": trans_ci,
        "observed_daily_rates": obs_rates.tolist(),
        "predicted_daily_rates_mean": daily_mean.tolist(),
        "predicted_daily_rates_CI95": daily_ci.tolist(),
        "k_runs": k_runs,
    }
    return metrics


def simulate_forecast(
    wearing: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    age_cat_names: List[str],
    occ_cat_names: List[str],
    params: Parameters,
    last_train_day_idx: int,
    forecast_days: int,
    k_runs: int,
) -> Dict[str, Any]:
    # Start from last_train_day_idx state
    init_states = wearing[last_train_day_idx, :]
    start_day = last_train_day_idx
    end_day = last_train_day_idx + forecast_days
    run_rates = []
    for r in range(k_runs):
        sim_states, sim_info, sim_probs = simulate_window(
            start_states=init_states,
            neighbors=neighbors,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            params=params,
            start_day_index=start_day,
            end_day_index=end_day,
        )
        run_rates.append(sim_states.mean(axis=1))
    run_rates_arr = np.vstack(run_rates)  # k x forecast_days
    mean_rates = run_rates_arr.mean(axis=0)
    std_rates = run_rates_arr.std(axis=0, ddof=1) if k_runs > 1 else np.zeros_like(mean_rates)
    ci95 = 1.96 * std_rates / math.sqrt(k_runs) if k_runs > 1 else std_rates
    days_forecast = [start_day + d + 1 for d in range(forecast_days)]
    forecast = {
        "days": days_forecast,
        "forecast_mean_rates": mean_rates.tolist(),
        "forecast_CI95": ci95.tolist(),
        "k_runs": k_runs,
    }
    return forecast


class LogitHeadCalibrator(Calibrator):
    """Fits a logistic decision head from micro-transitions on days_train (L2 regularized; intercept not regularized)."""
    
    def __init__(self, l2_reg: float = 1.0, max_iter: int = 400, learning_rate: float = 0.1):
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """Return FittedParams, fitted strictly on the training window."""
        wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
        train_start, train_end = train_window
        
        # Set seed for reproducibility
        set_global_seed(seed)
        
        # Compute neighbor shares and memory info for feature matrix
        T = wearing.shape[0]
        N = wearing.shape[1]
        
        # Compute neighbor shares per day
        share_f_by_day = np.zeros_like(wearing)
        share_w_by_day = np.zeros_like(wearing)
        share_c_by_day = np.zeros_like(wearing)
        
        for t in range(T):
            s_prev = wearing[t, :]
            share_f_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["family"])
            share_w_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["work_school"])
            share_c_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["community"])
        
        # Compute memory info (assuming received_info is available or zero)
        received_info = np.zeros_like(wearing)  # Placeholder - should be extracted from bundle if available
        mem_info = compute_mem_info(received_info, cfg.rho_info_decay_default)
        
        # Build feature matrix for logistic regression
        X, y = build_feature_matrix(
            wearing=wearing,
            mem_info=mem_info,
            share_f_by_day=share_f_by_day,
            share_w_by_day=share_w_by_day,
            share_c_by_day=share_c_by_day,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            day_start=train_start,
            day_end=train_end
        )
        
        # Fit logistic regression with L2 regularization
        print(f"LogitHeadCalibrator: Starting optimization (L2={self.l2_reg}, max_iter={self.max_iter}, lr={self.learning_rate})")
        beta = fit_logistic_l2(X, y, self.l2_reg, self.max_iter, self.learning_rate, verbose=True)
        
        # Create FittedParams from beta coefficients
        decision_weights = {
            'alpha': beta[0],  # intercept
            'gamma': beta[1],  # previous state
            'theta_f': 1.0,    # family weight (normalized)
            'theta_w': 1.0,    # work weight (normalized)  
            'theta_c': 1.0,    # community weight (normalized)
            'beta_r': beta[2] if len(beta) > 2 else 0.0,  # risk perception
            'beta_i': beta[3] if len(beta) > 3 else 0.0,  # info received
            'age_effects': {},
            'occ_effects': {}
        }
        
        # Extract demographic effects if available
        if len(beta) > 4:
            n_age = age_oh.shape[1]
            n_occ = occ_oh.shape[1]
            age_effects = {f'age_{i}': beta[4+i] for i in range(n_age)}
            occ_effects = {f'occ_{i}': beta[4+n_age+i] for i in range(n_occ)}
            decision_weights['age_effects'] = age_effects
            decision_weights['occ_effects'] = occ_effects
        
        layer_weights = {
            'family': 1.0,
            'work_school': 1.0, 
            'community': 1.0
        }
        
        info_params = {
            'phi_family': 0.1,
            'phi_work': 0.1,
            'phi_community': 0.1,
            'lambda_broadcast_base': 0.05,
            'lambda_broadcast_factor_after_day10': cfg.gov_lam_factor_default,
            'rho_info_decay': cfg.rho_info_decay_default
        }
        
        noise_params = {
            'tau': 1.0
        }
        
        meta = {
            'seed': seed,
            'calibrator_name': 'logit_head',
            'training_window': train_window,
            'l2_reg': self.l2_reg,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate
        }
        
        # Create fitted parameters
        fitted_params = FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            meta=meta
        )
        
        # Evaluate the fitted parameters for debugging
        print("LogitHeadCalibrator: Evaluating fitted parameters...")
        try:
            result = evaluator(simulator, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
            rmse = result.get('RMSE_aggregate_mean', float('inf'))
            mae = result.get('MAE_aggregate_mean', float('inf'))
            print(f"LogitHeadCalibrator: Training RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        except Exception as e:
            print(f"LogitHeadCalibrator: Evaluation failed: {e}")
        
        return fitted_params


class RandomSearchCalibrator(Calibrator):
    """Black-box search over selected simulator params (e.g., layer weights, info rates, memory, temperature)."""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
    
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """Return FittedParams, fitted strictly on the training window."""
        set_global_seed(seed)
        
        # Extract cfg from bundle
        wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
        
        best_params = None
        best_score = float('inf')
        
        for trial in range(self.n_trials):
            # Random parameter sampling
            trial_seed = seed + trial
            np.random.seed(trial_seed)
            
            # Sample random parameters
            decision_weights = {
                'alpha': np.random.normal(0.0, 1.0),
                'gamma': np.random.uniform(0.5, 3.0),
                'theta_f': np.random.uniform(0.5, 2.0),
                'theta_w': np.random.uniform(0.5, 2.0),
                'theta_c': np.random.uniform(0.5, 2.0),
                'beta_r': np.random.normal(0.0, 0.5),
                'beta_i': np.random.normal(0.0, 0.5),
                'age_effects': {},
                'occ_effects': {}
            }
            
            layer_weights = {
                'family': np.random.uniform(0.1, 2.0),
                'work_school': np.random.uniform(0.1, 2.0),
                'community': np.random.uniform(0.1, 2.0)
            }
            
            info_params = {
                'phi_family': np.random.uniform(0.01, 0.3),
                'phi_work': np.random.uniform(0.01, 0.3),
                'phi_community': np.random.uniform(0.01, 0.3),
                'lambda_broadcast_base': np.random.uniform(0.01, 0.2),
                'lambda_broadcast_factor_after_day10': np.random.uniform(1.0, 3.0),
                'rho_info_decay': np.random.uniform(0.1, 0.9)
            }
            
            noise_params = {
                'tau': np.random.uniform(0.5, 2.0)
            }
            
            meta = {
                'seed': trial_seed,
                'calibrator_name': 'random_search',
                'training_window': train_window,
                'trial': trial
            }
            
            # Create candidate parameters
            candidate_params = FittedParams(
                decision_weights=decision_weights,
                layer_weights=layer_weights,
                info_params=info_params,
                noise_params=noise_params,
                meta=meta
            )
            
            # Evaluate on training window
            try:
                result = evaluator(simulator, candidate_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
                score = result.get('RMSE_aggregate_mean', float('inf'))
                
                if trial % 20 == 0:  # Print every 20th trial
                    print(f"Trial {trial}: RMSE = {score:.4f}, Best so far = {best_score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_params = candidate_params
                    print(f"New best at trial {trial}: RMSE = {score:.4f}")
                    
            except Exception as e:
                # Skip this trial if evaluation fails
                if trial % 20 == 0:
                    print(f"Trial {trial} failed: {e}")
                continue
        
        if best_params is None:
            # Fallback to default parameters
            best_params = self._get_default_params(seed, train_window)
            
        return best_params
    
    def _get_default_params(self, seed: int, train_window: Tuple[int, int]) -> FittedParams:
        """Get default parameters as fallback."""
        return FittedParams(
            decision_weights={
                'alpha': 0.0, 'gamma': 1.0, 'theta_f': 1.0, 'theta_w': 1.0, 'theta_c': 1.0,
                'beta_r': 0.0, 'beta_i': 0.0, 'age_effects': {}, 'occ_effects': {}
            },
            layer_weights={'family': 1.0, 'work_school': 1.0, 'community': 1.0},
            info_params={
                'phi_family': 0.1, 'phi_work': 0.1, 'phi_community': 0.1,
                'lambda_broadcast_base': 0.05, 'lambda_broadcast_factor_after_day10': 1.5,
                'rho_info_decay': 0.5
            },
            noise_params={'tau': 1.0},
            meta={'seed': seed, 'calibrator_name': 'random_search', 'training_window': train_window}
        )


class SBICalibrator(Calibrator):
    """Simulation-Based Inference calibrator using neural posterior estimation."""
    
    def __init__(self, n_simulations: int = 10000, n_rounds: int = 3, neural_net_config: Optional[Dict] = None, k_observables: int = 1):
        """
        Initialize SBI calibrator.
        
        Args:
            n_simulations: Number of simulations per round for training the neural network
            n_rounds: Number of sequential rounds of inference
            neural_net_config: Configuration for the neural network (architecture, training params, etc.)
            k_observables: Number of observables (1 or 5)
                          - K=1: daily average wearing rate only
                          - K=5: daily average wearing rate + 4 transition probabilities
        """
        self.n_simulations = n_simulations
        self.n_rounds = n_rounds
        self.neural_net_config = neural_net_config or self._get_default_nn_config()
        self.k_observables = k_observables
        
        if k_observables not in [1, 5]:
            raise ValueError(f"k_observables must be 1 or 5, got {k_observables}")
        
        # Placeholder for SBI components (will be implemented later)
        self.posterior_estimator = None
        self.prior = None
        self.simulator_wrapper = None
        
    def _get_default_nn_config(self) -> Dict:
        """Get default neural network configuration optimized for M3 chip."""
        return {
            'flow_type': 'maf',  # 'maf' or 'nsf'
            'hidden_features': 48,  # Reduced for M3 chip (was 64)
            'num_transforms': 4,    # Reduced complexity (was 5)
            'num_blocks': 2,
            'learning_rate': 1e-3,
            'batch_size': 128,      # Reduced for M3 chip (was 256)
            'max_epochs': 80,       # Slightly reduced (was 100)
            'early_stopping_patience': 8
        }
    
    def _define_prior(self) -> Dict[str, Tuple[float, float]]:
        """
        Define prior distributions for model parameters as uniform distributions.
        Returns parameter bounds for uniform priors based on RandomSearchCalibrator ranges.
        
        Returns:
            Dict mapping parameter names to (lower_bound, upper_bound) tuples
        """
        # Define parameter bounds based on RandomSearchCalibrator sampling ranges
        prior_bounds = {
            # Decision weights
            'alpha': (-3.0, 3.0),           # Convert from normal(0, 1) to wider uniform range
            'gamma': (0.5, 3.0),
            'theta_f': (0.5, 2.0),
            'theta_w': (0.5, 2.0), 
            'theta_c': (0.5, 2.0),
            'beta_r': (-2.0, 2.0),          # Convert from normal(0, 0.5) to uniform range
            'beta_i': (-2.0, 2.0),          # Convert from normal(0, 0.5) to uniform range
            
            # Layer weights
            'family': (0.1, 2.0),
            'work_school': (0.1, 2.0),
            'community': (0.1, 2.0),
            
            # Info params
            'phi_family': (0.01, 0.3),
            'phi_work': (0.01, 0.3),
            'phi_community': (0.01, 0.3),
            'lambda_broadcast_base': (0.01, 0.2),
            'lambda_broadcast_factor_after_day10': (1.0, 3.0),
            'rho_info_decay': (0.1, 0.9),
            
            # Noise params
            'tau': (0.5, 2.0),
            
            # Age effects (assuming reasonable ranges for demographic effects)
            'age_0': (-2.0, 2.0),           # Youth effect
            'age_1': (-2.0, 2.0),           # Young Adult effect  
            'age_2': (-2.0, 2.0),           # Middle Age effect (baseline may be 0)
            
            # Occupation effects
            'occ_0': (-2.0, 2.0),           # Blue Collar effect
            'occ_1': (-2.0, 2.0),           # White Collar effect
            'occ_2': (-2.0, 2.0),           # Student effect
        }
        
        return prior_bounds
    
    def _sample_from_prior(self, n_samples: int, seed: int = None) -> Tuple[np.ndarray, List[str]]:
        """
        Sample parameters from uniform prior distributions.
        
        Args:
            n_samples: Number of parameter samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (samples array of shape (n_samples, n_parameters), parameter names list)
        """
        if seed is not None:
            np.random.seed(seed)
            
        prior_bounds = self._define_prior()
        param_names = list(prior_bounds.keys())
        n_params = len(param_names)
        
        samples = np.zeros((n_samples, n_params))
        
        for i, param_name in enumerate(param_names):
            lower, upper = prior_bounds[param_name]
            samples[:, i] = np.random.uniform(lower, upper, n_samples)
            
        return samples, param_names
    
    def _samples_to_fitted_params(self, sample: np.ndarray, param_names: List[str], 
                                  seed: int, train_window: Tuple[int, int]) -> FittedParams:
        """
        Convert a parameter sample to FittedParams format.
        
        Args:
            sample: Single parameter sample array
            param_names: List of parameter names corresponding to sample indices
            seed: Random seed for meta information
            train_window: Training window for meta information
            
        Returns:
            FittedParams object
        """
        # Create parameter dictionary from sample
        params_dict = {name: float(sample[i]) for i, name in enumerate(param_names)}
        
        # Organize parameters into FittedParams structure
        decision_weights = {
            'alpha': params_dict.get('alpha', 0.0),
            'gamma': params_dict.get('gamma', 1.0),
            'theta_f': params_dict.get('theta_f', 1.0),
            'theta_w': params_dict.get('theta_w', 1.0),
            'theta_c': params_dict.get('theta_c', 1.0),
            'beta_r': params_dict.get('beta_r', 0.0),
            'beta_i': params_dict.get('beta_i', 0.0),
            'age_effects': {
                'age_0': params_dict.get('age_0', 0.0),
                'age_1': params_dict.get('age_1', 0.0),
                'age_2': params_dict.get('age_2', 0.0),
            },
            'occ_effects': {
                'occ_0': params_dict.get('occ_0', 0.0),
                'occ_1': params_dict.get('occ_1', 0.0),
                'occ_2': params_dict.get('occ_2', 0.0),
            }
        }
        
        layer_weights = {
            'family': params_dict.get('family', 1.0),
            'work_school': params_dict.get('work_school', 1.0),
            'community': params_dict.get('community', 1.0)
        }
        
        info_params = {
            'phi_family': params_dict.get('phi_family', 0.1),
            'phi_work': params_dict.get('phi_work', 0.1),
            'phi_community': params_dict.get('phi_community', 0.1),
            'lambda_broadcast_base': params_dict.get('lambda_broadcast_base', 0.05),
            'lambda_broadcast_factor_after_day10': params_dict.get('lambda_broadcast_factor_after_day10', 1.5),
            'rho_info_decay': params_dict.get('rho_info_decay', 0.5)
        }
        
        noise_params = {
            'tau': params_dict.get('tau', 1.0)
        }
        
        meta = {
            'seed': seed,
            'calibrator_name': 'sbi',
            'training_window': train_window,
            'parameter_source': 'prior_sample'
        }
        
        return FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            meta=meta
        )
    
    def _setup_simulator_wrapper(self, bundle, train_window: Tuple[int, int]) -> None:
        """Setup simulator wrapper for SBI."""
        # Store bundle and training window for later use
        self.bundle = bundle
        self.train_window = train_window
        
        # Extract components from bundle for simulation
        wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
        train_start, train_end = train_window
        
        # Store simulation components
        self.simulation_components = {
            'wearing': wearing,
            'neighbors': neighbors,
            'risk': risk,
            'age_oh': age_oh,
            'occ_oh': occ_oh,
            'cfg': cfg,
            'train_start': train_start,
            'train_end': train_end,
            'T': train_end - train_start,  # Number of time steps
            'N': wearing.shape[1]  # Number of agents
        }
        
        # Age and occupation category names will be set when needed
        self.age_cat_names = None
        self.occ_cat_names = None
        
        print(f"  Simulator wrapper setup complete: T={self.simulation_components['T']} steps, N={self.simulation_components['N']} agents, K={self.k_observables} observables")
    
    def _compute_transition_probabilities(self, prev_states: np.ndarray, curr_states: np.ndarray) -> np.ndarray:
        """
        Compute transition probabilities between two consecutive time steps.
        
        Args:
            prev_states: States at time t-1, shape (N,)
            curr_states: States at time t, shape (N,)
            
        Returns:
            Transition probabilities [P01, P11, P10, P00] as population proportions
            Note: These are population-level transition rates, not conditional probabilities
        """
        # Convert to binary (0/1)
        prev_binary = (prev_states > 0.5).astype(int)
        curr_binary = (curr_states > 0.5).astype(int)
        
        # Count transitions
        n_total = len(prev_binary)
        if n_total == 0:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Population-level transition rates (what fraction of total population makes each transition)
        # P01: fraction of population that transitions from not wearing -> wearing
        p01 = np.mean((prev_binary == 0) & (curr_binary == 1))
        
        # P11: fraction of population that transitions from wearing -> wearing  
        p11 = np.mean((prev_binary == 1) & (curr_binary == 1))
        
        # P10: fraction of population that transitions from wearing -> not wearing
        p10 = np.mean((prev_binary == 1) & (curr_binary == 0))
        
        # P00: fraction of population that transitions from not wearing -> not wearing
        p00 = np.mean((prev_binary == 0) & (curr_binary == 0))
        
        return np.array([p01, p11, p10, p00])
    
    def _run_single_simulation(self, fitted_params: FittedParams, seed: int = None) -> np.ndarray:
        """
        Run a single simulation with given parameters and return trajectory.
        
        Args:
            fitted_params: Parameters to use for simulation
            seed: Random seed for reproducibility
            
        Returns:
            Trajectory array of shape (T, K) where T is time steps and K is observables
            - K=1: daily average wearing rate only
            - K=5: daily average wearing rate + 4 transition probabilities
        """
        if seed is not None:
            set_global_seed(seed)
        
        # Convert to legacy parameters for compatibility
        params = fitted_params.to_parameters()
        
        # Get simulation components
        comp = self.simulation_components
        
        # Initial state: use observed state at train_start-1
        init_states = comp['wearing'][comp['train_start'] - 1, :]
        
        # Run simulation for the training window
        sim_states, sim_info, sim_probs = simulate_window(
            start_states=init_states,
            neighbors=comp['neighbors'],
            risk=comp['risk'],
            age_oh=comp['age_oh'],
            occ_oh=comp['occ_oh'],
            age_cat_names=self.age_cat_names,
            occ_cat_names=self.occ_cat_names,
            params=params,
            start_day_index=comp['train_start'] - 1,
            end_day_index=comp['train_end'] - 1
        )
        
        # sim_states shape: (T, N)
        T = sim_states.shape[0]
        
        if self.k_observables == 1:
            # K=1: Only daily average wearing rate
            daily_rates = np.mean(sim_states, axis=1)  # Shape: (T,)
            trajectory = daily_rates.reshape(-1, 1)  # Shape: (T, 1)
            
        elif self.k_observables == 5:
            # K=5: Daily average wearing rate + 4 transition probabilities
            trajectory = np.zeros((T, 5))  # Shape: (T, 5)
            
            # First column: daily average wearing rate
            trajectory[:, 0] = np.mean(sim_states, axis=1)
            
            # Columns 1-4: transition probabilities (P01, P11, P10, P00)
            # For first day (t=0), we don't have previous state, so fill with 0.0
            trajectory[0, 1:5] = 0.0  # Use 0.0 instead of NaN for neural network compatibility
            
            # For days t=1 to T-1, compute transition probabilities
            for t in range(1, T):
                prev_states = sim_states[t-1, :]  # States at t-1
                curr_states = sim_states[t, :]    # States at t
                transition_probs = self._compute_transition_probabilities(prev_states, curr_states)
                trajectory[t, 1:5] = transition_probs  # [P01, P11, P10, P00]
        
        return trajectory
    
    def _flatten_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Flatten trajectory matrix to vector for neural network input.
        
        Args:
            trajectory: Trajectory array of shape (T, K)
            
        Returns:
            Flattened vector of length T*K
        """
        return trajectory.flatten()
    
    def _generate_training_data(self, n_samples: int = 1000, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for SBI: N pairs of (parameters, trajectory_vectors).
        
        Args:
            n_samples: Number of parameter-trajectory pairs to generate (default: 1000)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (parameter_samples, trajectory_vectors) where:
            - parameter_samples: shape (n_samples, n_parameters)
            - trajectory_vectors: shape (n_samples, T*K) where T*K is flattened trajectory length
        """
        print(f"SBICalibrator: Generating {n_samples} training samples...")
        
        if seed is not None:
            set_global_seed(seed)
        
        # Step 1: Sample N parameter sets from prior
        print("  Step 1: Sampling parameters from prior...")
        param_samples, param_names = self._sample_from_prior(n_samples=n_samples, seed=seed)
        print(f"    Generated {param_samples.shape[0]} parameter samples")
        
        # Step 2: For each parameter set, run simulation and collect trajectory
        print("  Step 2: Running simulations and collecting trajectories...")
        trajectory_vectors = []
        
        for i in range(n_samples):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"    Running simulation {i+1}/{n_samples}...")
            
            # Convert parameter sample to FittedParams
            fitted_params = self._samples_to_fitted_params(
                param_samples[i], param_names, seed + i if seed is not None else None, self.train_window
            )
            
            # Run single simulation to get trajectory
            trajectory = self._run_single_simulation(fitted_params, seed + i if seed is not None else None)
            
            # Flatten trajectory to vector
            trajectory_vector = self._flatten_trajectory(trajectory)
            trajectory_vectors.append(trajectory_vector)
        
        # Convert to numpy array
        trajectory_vectors = np.array(trajectory_vectors)  # Shape: (n_samples, T*K)
        
        print(f"  Training data generation complete:")
        print(f"    Parameter samples shape: {param_samples.shape}")
        print(f"    Trajectory vectors shape: {trajectory_vectors.shape}")
        print(f"    Each trajectory vector length: {trajectory_vectors.shape[1]} (T={self.simulation_components['T']}, K={self.k_observables})")
        
        return param_samples, trajectory_vectors
    
    def _compute_summary_statistics(self, simulated_data: np.ndarray, observed_data: np.ndarray) -> np.ndarray:
        """Compute summary statistics for SBI."""
        # TODO: Implement summary statistics computation
        # Common choices:
        # - Mean adoption rate over time
        # - Final adoption rate
        # - Time to peak adoption
        # - Variance in adoption rates
        # - Network clustering effects
        pass
    
    def _train_neural_posterior_estimator(self, theta_samples: np.ndarray, x_samples: np.ndarray) -> None:
        """
        Train neural network to estimate posterior p(theta|x) using SBI library.
        
        Args:
            theta_samples: Parameter samples of shape (n_samples, n_parameters)
            x_samples: Trajectory vectors of shape (n_samples, trajectory_length)
        """
        print("SBICalibrator: Training neural posterior estimator...")
        
        try:
            # Import SBI library components
            import torch
            from sbi import utils as sbi_utils
            from sbi.inference import NPE
            
            print(f"  Using PyTorch: {torch.__version__}")
            print(f"  Training data: {theta_samples.shape[0]} samples")
            print(f"  Parameter dimension: {theta_samples.shape[1]}")
            print(f"  Observation dimension: {x_samples.shape[1]}")
            
        except ImportError as e:
            print(f"  ✗ SBI library not available: {e}")
            print("  Please install: pip install sbi-dev")
            print("  Falling back to placeholder implementation...")
            self._placeholder_neural_estimator(theta_samples, x_samples)
            return
        
        try:
            # Convert numpy arrays to torch tensors
            theta_tensor = torch.tensor(theta_samples, dtype=torch.float32)
            x_tensor = torch.tensor(x_samples, dtype=torch.float32)
            
            print(f"  Converted to tensors: theta {theta_tensor.shape}, x {x_tensor.shape}")
            
            # Define prior bounds for SBI (convert our uniform priors)
            prior_bounds = self._define_prior()
            prior_min = torch.tensor([bounds[0] for bounds in prior_bounds.values()], dtype=torch.float32)
            prior_max = torch.tensor([bounds[1] for bounds in prior_bounds.values()], dtype=torch.float32)
            
            # Create uniform prior for SBI
            prior = sbi_utils.BoxUniform(low=prior_min, high=prior_max)
            print(f"  Created uniform prior with bounds: [{prior_min.min():.2f}, {prior_max.max():.2f}]")
            
            # Initialize Neural Posterior Estimation
            # Use MAF (Masked Autoregressive Flow) by default
            flow_type = self.neural_net_config.get('flow_type', 'maf')  # 'maf' or 'nsf'
            
            if flow_type.lower() == 'nsf':
                # Neural Spline Flow
                density_estimator = 'nsf'
                print("  Using Neural Spline Flow (NSF)")
            else:
                # Masked Autoregressive Flow (default)
                density_estimator = 'maf'
                print("  Using Masked Autoregressive Flow (MAF)")
            
            # Create NPE inference object
            inference = NPE(
                prior=prior,
                density_estimator=density_estimator,
                device='cpu',  # Use CPU for compatibility
                show_progress_bars=True
            )
            
            print(f"  Created NPE with {density_estimator.upper()} density estimator")
            
            # Configure training parameters
            training_batch_size = self.neural_net_config.get('batch_size', 256)
            max_epochs = self.neural_net_config.get('max_epochs', 100)
            learning_rate = self.neural_net_config.get('learning_rate', 1e-3)
            
            print(f"  Training config: batch_size={training_batch_size}, max_epochs={max_epochs}, lr={learning_rate}")
            
            # Train the neural posterior estimator
            print("  Starting neural network training...")
            
            # Append training data (SBI can handle multiple rounds)
            inference = inference.append_simulations(theta_tensor, x_tensor)
            
            # Train the estimator
            density_estimator = inference.train(
                training_batch_size=training_batch_size,
                max_num_epochs=max_epochs,
                learning_rate=learning_rate,
                show_train_summary=True
            )
            
            # Store the trained estimator
            self.posterior_estimator = inference.build_posterior(density_estimator)
            
            print("  ✓ Neural posterior estimator training completed successfully")
            print(f"  Trained estimator type: {type(self.posterior_estimator).__name__}")
            
        except Exception as e:
            print(f"  ✗ Neural posterior estimator training failed: {e}")
            print("  Falling back to placeholder implementation...")
            self._placeholder_neural_estimator(theta_samples, x_samples)
    
    def _placeholder_neural_estimator(self, theta_samples: np.ndarray, x_samples: np.ndarray) -> None:
        """Placeholder implementation when SBI library is not available."""
        print("  Using placeholder neural estimator:")
        print(f"    - Would train on {theta_samples.shape[0]} parameter-trajectory pairs")
        print(f"    - Parameter dimension: {theta_samples.shape[1]}")
        print(f"    - Observation dimension: {x_samples.shape[1]}")
        print(f"    - Flow type: {self.neural_net_config.get('flow_type', 'maf').upper()}")
        print("  Placeholder training completed (no actual training performed)")
        
        # Store a dummy estimator indicator
        self.posterior_estimator = "placeholder_estimator"
    
    def _sample_from_posterior(self, observed_x: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """
        Sample parameters from the learned posterior given observed data.
        
        Args:
            observed_x: Observed trajectory vector of shape (trajectory_length,)
            n_samples: Number of posterior samples to generate
            
        Returns:
            Posterior parameter samples of shape (n_samples, n_parameters)
        """
        print(f"SBICalibrator: Sampling {n_samples} parameters from posterior...")
        
        if self.posterior_estimator is None:
            print("  ✗ No trained posterior estimator available")
            return self._placeholder_posterior_sampling(observed_x, n_samples)
        
        if self.posterior_estimator == "placeholder_estimator":
            print("  Using placeholder posterior sampling...")
            return self._placeholder_posterior_sampling(observed_x, n_samples)
        
        try:
            import torch
            
            # Convert observed data to tensor
            x_tensor = torch.tensor(observed_x.reshape(1, -1), dtype=torch.float32)
            print(f"  Observation tensor shape: {x_tensor.shape}")
            
            # Sample from the posterior
            print("  Sampling from trained neural posterior...")
            posterior_samples = self.posterior_estimator.sample(
                (n_samples,), 
                x=x_tensor,
                show_progress_bars=True
            )
            
            # Convert back to numpy
            posterior_samples_np = posterior_samples.detach().cpu().numpy()
            
            print(f"  ✓ Posterior sampling completed")
            print(f"  Posterior samples shape: {posterior_samples_np.shape}")
            
            return posterior_samples_np
            
        except Exception as e:
            print(f"  ✗ Posterior sampling failed: {e}")
            print("  Falling back to placeholder sampling...")
            return self._placeholder_posterior_sampling(observed_x, n_samples)
    
    def _placeholder_posterior_sampling(self, observed_x: np.ndarray, n_samples: int) -> np.ndarray:
        """Placeholder posterior sampling when neural estimator is not available."""
        print(f"  Placeholder posterior sampling:")
        print(f"    - Observation dimension: {observed_x.shape}")
        print(f"    - Generating {n_samples} samples from prior (fallback)")
        
        # Fallback: sample from prior (not ideal but functional)
        prior_samples, _ = self._sample_from_prior(n_samples=n_samples, seed=42)
        
        print(f"    - Generated {prior_samples.shape[0]} prior samples as posterior approximation")
        return prior_samples
    
    def _posterior_to_fitted_params(self, posterior_samples: np.ndarray, seed: int, train_window: Tuple[int, int]) -> FittedParams:
        """
        Convert posterior samples to FittedParams format.
        
        Args:
            posterior_samples: Posterior parameter samples of shape (n_samples, n_parameters)
            seed: Random seed for meta information
            train_window: Training window for meta information
            
        Returns:
            FittedParams object with posterior mean parameters
        """
        print(f"SBICalibrator: Converting posterior samples to FittedParams...")
        print(f"  Posterior samples shape: {posterior_samples.shape}")
        
        # Use posterior mean as point estimate
        posterior_mean = np.mean(posterior_samples, axis=0)
        print(f"  Using posterior mean as point estimate")
        
        # Get parameter names
        param_names = list(self._define_prior().keys())
        
        # Convert posterior mean to FittedParams using existing method
        fitted_params = self._samples_to_fitted_params(
            posterior_mean, param_names, seed, train_window
        )
        
        # Update meta information with posterior statistics
        posterior_std = np.std(posterior_samples, axis=0)
        
        fitted_params.meta.update({
            'parameter_estimation_method': 'posterior_mean',
            'n_posterior_samples': posterior_samples.shape[0],
            'posterior_mean': posterior_mean.tolist(),
            'posterior_std': posterior_std.tolist(),
            'posterior_summary_stats': {
                'mean_param_std': float(np.mean(posterior_std)),
                'max_param_std': float(np.max(posterior_std)),
                'min_param_std': float(np.min(posterior_std))
            }
        })
        
        print(f"  ✓ Conversion completed")
        print(f"  Posterior uncertainty (mean std): {np.mean(posterior_std):.4f}")
        
        return fitted_params
    
    def _save_sbi_checkpoint(self, param_samples: np.ndarray, trajectory_vectors: np.ndarray, 
                            train_window: Tuple[int, int], seed: int, cfg) -> str:
        """
        Save SBI checkpoint including trained model and training data.
        
        Args:
            param_samples: Training parameter samples
            trajectory_vectors: Training trajectory vectors
            train_window: Training window
            seed: Random seed
            cfg: Configuration object
            
        Returns:
            Output directory path
        """
        # Create output directory
        output_dir = os.path.join(cfg.data_folder, f"outputs_SBICalibrator_K{self.k_observables}")
        ensure_dir(output_dir)
        
        print(f"SBICalibrator: Saving checkpoint to {output_dir}...")
        
        # 1. Save SBI configuration
        config_data = {
            "sbi_config": {
                "calibrator_type": "sbi",
                "k_observables": self.k_observables,
                "n_simulations": self.n_simulations,
                "n_rounds": self.n_rounds,
                "neural_net_config": self.neural_net_config,
                "training_window": train_window,
                "seed": seed,
                "training_samples": param_samples.shape[0],
                "trajectory_length": trajectory_vectors.shape[1]
            },
            "config": cfg.__dict__  # Include original config for compatibility
        }
        save_json(config_data, os.path.join(output_dir, "config.json"))
        
        # 2. Save training data
        training_data = {
            "parameter_samples": param_samples.tolist(),
            "trajectory_vectors": trajectory_vectors.tolist(),
            "parameter_names": list(self._define_prior().keys()),
            "prior_bounds": self._define_prior(),
            "training_info": {
                "n_samples": param_samples.shape[0],
                "n_parameters": param_samples.shape[1],
                "trajectory_length": trajectory_vectors.shape[1],
                "k_observables": self.k_observables
            }
        }
        save_json(training_data, os.path.join(output_dir, "training_data.json"))
        
        # 3. Save trained neural posterior estimator (if available)
        if hasattr(self, 'posterior_estimator') and self.posterior_estimator is not None:
            if self.posterior_estimator != "placeholder_estimator":
                try:
                    import torch
                    # Save the trained posterior estimator
                    torch.save(self.posterior_estimator, os.path.join(output_dir, "posterior_estimator.pt"))
                    print("  ✓ Saved trained neural posterior estimator")
                except Exception as e:
                    print(f"  ⚠ Could not save posterior estimator: {e}")
            else:
                # Save placeholder indicator
                save_json({"estimator_type": "placeholder"}, os.path.join(output_dir, "posterior_estimator.json"))
                print("  ✓ Saved placeholder estimator info")
        
        # 4. Save prior information for easy reloading
        prior_info = {
            "prior_bounds": self._define_prior(),
            "parameter_names": list(self._define_prior().keys()),
            "n_parameters": len(self._define_prior())
        }
        save_json(prior_info, os.path.join(output_dir, "prior_info.json"))
        
        print(f"  ✓ SBI checkpoint saved successfully")
        return output_dir
    
    def _load_sbi_checkpoint(self, checkpoint_dir: str) -> Dict:
        """
        Load SBI checkpoint from directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        print(f"SBICalibrator: Loading checkpoint from {checkpoint_dir}...")
        
        checkpoint_data = {}
        
        try:
            # Load configuration
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, 'r') as f:
                checkpoint_data['config'] = json.load(f)
            
            # Load training data
            training_data_path = os.path.join(checkpoint_dir, "training_data.json")
            with open(training_data_path, 'r') as f:
                checkpoint_data['training_data'] = json.load(f)
            
            # Load prior information
            prior_info_path = os.path.join(checkpoint_dir, "prior_info.json")
            with open(prior_info_path, 'r') as f:
                checkpoint_data['prior_info'] = json.load(f)
            
            # Try to load trained posterior estimator
            estimator_pt_path = os.path.join(checkpoint_dir, "posterior_estimator.pt")
            estimator_json_path = os.path.join(checkpoint_dir, "posterior_estimator.json")
            
            if os.path.exists(estimator_pt_path):
                try:
                    import torch
                    checkpoint_data['posterior_estimator'] = torch.load(estimator_pt_path, map_location='cpu')
                    print("  ✓ Loaded trained neural posterior estimator")
                except Exception as e:
                    print(f"  ⚠ Could not load posterior estimator: {e}")
                    checkpoint_data['posterior_estimator'] = None
            elif os.path.exists(estimator_json_path):
                with open(estimator_json_path, 'r') as f:
                    estimator_info = json.load(f)
                checkpoint_data['posterior_estimator'] = estimator_info.get('estimator_type', 'placeholder')
                print("  ✓ Loaded placeholder estimator info")
            else:
                checkpoint_data['posterior_estimator'] = None
                print("  ⚠ No posterior estimator found")
            
            print(f"  ✓ Checkpoint loaded successfully")
            return checkpoint_data
            
        except Exception as e:
            print(f"  ✗ Failed to load checkpoint: {e}")
            return None
    
    def sample_from_checkpoint(self, checkpoint_dir: str, observed_x: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """
        Sample parameters from a saved checkpoint given observed data.
        
        Args:
            checkpoint_dir: Directory containing SBI checkpoint
            observed_x: Observed trajectory vector
            n_samples: Number of posterior samples to generate
            
        Returns:
            Posterior parameter samples
        """
        print(f"SBICalibrator: Sampling from checkpoint...")
        
        # Load checkpoint
        checkpoint_data = self._load_sbi_checkpoint(checkpoint_dir)
        if checkpoint_data is None:
            print("  ✗ Failed to load checkpoint")
            return None
        
        # Restore configuration
        sbi_config = checkpoint_data['config']['sbi_config']
        self.k_observables = sbi_config['k_observables']
        self.neural_net_config = sbi_config['neural_net_config']
        
        # Restore posterior estimator
        if checkpoint_data['posterior_estimator'] is not None:
            if isinstance(checkpoint_data['posterior_estimator'], str):
                self.posterior_estimator = checkpoint_data['posterior_estimator']
            else:
                self.posterior_estimator = checkpoint_data['posterior_estimator']
        
        # Sample from posterior
        posterior_samples = self._sample_from_posterior(observed_x, n_samples)
        
        return posterior_samples
    
    def create_fitted_params_from_checkpoint(self, checkpoint_dir: str, observed_x: np.ndarray, 
                                           n_samples: int = 1000) -> FittedParams:
        """
        Create FittedParams from checkpoint by sampling posterior.
        
        Args:
            checkpoint_dir: Directory containing SBI checkpoint
            observed_x: Observed trajectory vector
            n_samples: Number of posterior samples for estimation
            
        Returns:
            FittedParams object with posterior mean parameters
        """
        print(f"SBICalibrator: Creating FittedParams from checkpoint...")
        
        # Sample from checkpoint
        posterior_samples = self.sample_from_checkpoint(checkpoint_dir, observed_x, n_samples)
        if posterior_samples is None:
            return None
        
        # Load checkpoint to get metadata
        checkpoint_data = self._load_sbi_checkpoint(checkpoint_dir)
        sbi_config = checkpoint_data['config']['sbi_config']
        
        # Convert to FittedParams
        fitted_params = self._posterior_to_fitted_params(
            posterior_samples, 
            sbi_config['seed'], 
            tuple(sbi_config['training_window'])
        )
        
        # Update meta with checkpoint info
        fitted_params.meta.update({
            'loaded_from_checkpoint': True,
            'checkpoint_directory': checkpoint_dir,
            'k_observables': sbi_config['k_observables'],
            'original_training_samples': sbi_config['training_samples']
        })
        
        return fitted_params
    
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """
        Fit parameters using Simulation-Based Inference.
        
        The SBI approach:
        1. Define prior distributions over parameters
        2. Run many simulations with parameters sampled from prior
        3. Train neural network to estimate posterior p(theta|x_obs)
        4. Sample from posterior to get parameter estimates
        """
        print(f"SBICalibrator: Starting SBI with {self.n_simulations} simulations and {self.n_rounds} rounds...")
        
        # Set seed for reproducibility
        set_global_seed(seed)
        
        # Extract data from bundle
        wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
        
        # Step 1: Define prior distributions and test sampling
        print("SBICalibrator: Step 1 - Defining prior distributions...")
        prior_bounds = self._define_prior()
        print(f"  Defined priors for {len(prior_bounds)} parameters")
        
        # Step 2: Setup simulator wrapper
        print("SBICalibrator: Step 2 - Setting up simulator wrapper...")
        self._setup_simulator_wrapper(bundle, train_window)
        
        # Get age and occupation category names from main function context
        # We need to extract these from the actual data processing
        # For now, we'll determine them based on the age_oh and occ_oh dimensions
        n_age_cats = age_oh.shape[1] if age_oh.shape[1] > 0 else 0
        n_occ_cats = occ_oh.shape[1] if occ_oh.shape[1] > 0 else 0
        
        # Create generic category names based on dimensions
        self.age_cat_names = [f'age_cat_{i}' for i in range(n_age_cats)]
        self.occ_cat_names = [f'occ_cat_{i}' for i in range(n_occ_cats)]
        
        print(f"  Demographics: {n_age_cats} age categories, {n_occ_cats} occupation categories")
        
        # Step 3: Generate training data
        print("SBICalibrator: Step 3 - Generating training data...")
        n_training_samples = self.n_simulations  # Use full simulation count
        print(f"  Generating {n_training_samples} training samples for SBI...")
        
        param_samples, trajectory_vectors = self._generate_training_data(
            n_samples=n_training_samples, seed=seed
        )
        
        # Step 4: Show training data summary
        print("SBICalibrator: Step 4 - Training data summary:")
        print(f"  Parameter samples shape: {param_samples.shape}")
        print(f"  Trajectory vectors shape: {trajectory_vectors.shape}")
        print(f"  Sample trajectory statistics:")
        print(f"    Mean trajectory value: {np.mean(trajectory_vectors):.4f}")
        print(f"    Std trajectory value: {np.std(trajectory_vectors):.4f}")
        print(f"    Min trajectory value: {np.min(trajectory_vectors):.4f}")
        print(f"    Max trajectory value: {np.max(trajectory_vectors):.4f}")
        
        # Step 5: Train neural posterior estimator
        print("SBICalibrator: Step 5 - Training neural posterior estimator...")
        self._train_neural_posterior_estimator(param_samples, trajectory_vectors)
        
        # Step 6: Generate observed data for testing (use first sample as mock observation)
        print("SBICalibrator: Step 6 - Testing posterior inference...")
        mock_observed_x = trajectory_vectors[0]  # Use first trajectory as mock observation
        print(f"  Using mock observation with shape: {mock_observed_x.shape}")
        
        # Step 7: Sample from posterior
        posterior_samples = self._sample_from_posterior(mock_observed_x, n_samples=100)
        
        # Step 8: Save SBI checkpoint
        output_dir = self._save_sbi_checkpoint(param_samples, trajectory_vectors, train_window, seed, cfg)
        
        # Step 9: Convert posterior samples to FittedParams
        fitted_params = self._posterior_to_fitted_params(posterior_samples, seed, train_window)
        
        # Save calibrated parameters to checkpoint directory (for compatibility)
        calibrated_params_path = os.path.join(output_dir, "calibrated_parameters.json")
        save_json(fitted_params.to_dict(), calibrated_params_path)
        
        # Update meta information
        fitted_params.meta.update({
            'n_simulations': self.n_simulations,
            'n_rounds': self.n_rounds,
            'neural_net_config': self.neural_net_config,
            'status': 'sbi_complete',
            'n_training_samples': n_training_samples,
            'trajectory_vector_length': trajectory_vectors.shape[1],
            'k_observables': self.k_observables,
            'mock_observation_used': True,
            'checkpoint_saved_to': output_dir
        })
        
        print("SBICalibrator: Complete SBI pipeline executed successfully!")
        print(f"  Checkpoint saved to: {output_dir}")
        
        # Return the fitted parameters from posterior inference
        return fitted_params


class BoCalibrator(Calibrator):
    """Bayesian Optimization calibrator using Gaussian Process surrogate models."""
    
    def __init__(self, n_trials: int = 300, acquisition_function: str = 'EI', 
                 kernel_type: str = 'RBF', random_state: int = None,
                 metric_type: str = 'composite', metric_weights: Dict[str, float] = None,
                 normalize_metrics: bool = True, fast_mode_iterations: int = 50,
                 use_turbo: bool = False, turbo_config: Dict[str, Any] = None):
        """
        Initialize Bayesian Optimization calibrator.
        
        Args:
            n_trials: Number of optimization trials/iterations
            acquisition_function: Acquisition function type ('EI', 'PI', 'UCB')
            kernel_type: Gaussian Process kernel type ('RBF', 'Matern', 'WhiteKernel')
            random_state: Random seed for reproducibility
            metric_type: Metric type ('rmse', 'mae', 'brier', 'transition', 'composite', 'adaptive')
            metric_weights: Weights for composite metrics {'rmse': 0.5, 'brier': 0.3, 'transition': 0.2}
            normalize_metrics: Whether to normalize metrics to [0, 1] range
            fast_mode_iterations: Number of iterations to use fast mode (fewer simulations)
            use_turbo: Whether to use TuRBO (Trust Region Bayesian Optimization)
            turbo_config: TuRBO configuration dict {'trust_region_size': 0.8, 'max_cholesky_size': 2000}
        """
        self.n_trials = n_trials
        self.acquisition_function = acquisition_function
        self.kernel_type = kernel_type
        self.random_state = random_state
        self.metric_type = metric_type
        self.normalize_metrics = normalize_metrics
        self.fast_mode_iterations = fast_mode_iterations
        self.use_turbo = use_turbo
        
        # TuRBO configuration
        if turbo_config is None:
            self.turbo_config = {
                'trust_region_size': 0.8,      # Initial trust region size
                'max_cholesky_size': 2000,     # Max Cholesky decomposition size
                'min_trust_region': 1e-8,      # Minimum trust region size
                'max_trust_region': 1.0,       # Maximum trust region size
                'success_tolerance': 3,        # Number of successes to expand TR
                'failure_tolerance': 10,       # Number of failures to shrink TR
                'expansion_factor': 2.0,       # Trust region expansion factor
                'contraction_factor': 0.5      # Trust region contraction factor
            }
        else:
            self.turbo_config = turbo_config
        
        # Set default weights for composite metrics (updated based on analysis)
        if metric_weights is None:
            self.metric_weights = {'rmse': 0.4, 'mae': 0.2, 'brier': 0.2, 'transition': 0.2}
        else:
            self.metric_weights = metric_weights
        
        # Placeholder for BO components (will be implemented later)
        self.gp_model = None
        self.acquisition_optimizer = None
        self.parameter_bounds = None
        self.optimization_history = []
        
        # TuRBO state tracking
        if self.use_turbo:
            self.turbo_state = {
                'trust_region_size': self.turbo_config['trust_region_size'],
                'center': None,  # Will be set to best point
                'successes': 0,
                'failures': 0,
                'best_value': float('inf'),
                'iteration': 0
            }
        
        # Metric normalization parameters (will be computed from initial samples)
        self.metric_ranges = {
            'rmse': (0.0, 1.0),      # RMSE typically in [0, 1] for normalized rates
            'mae': (0.0, 1.0),       # MAE typically in [0, 1] for normalized rates
            'brier': (0.0, 0.25),    # Brier score typically in [0, 0.25] for binary classification
            'transition': (0.0, 1.0) # TransitionFit error typically in [0, 1]
        }
        
        # Min-max normalization parameters (computed from initial samples)
        self.normalization_stats = None
        self.initial_metrics_collected = []
        
        print(f"BoCalibrator initialized:")
        print(f"  - Trials: {self.n_trials}")
        print(f"  - Acquisition function: {self.acquisition_function}")
        print(f"  - Kernel type: {self.kernel_type}")
        print(f"  - Random state: {self.random_state}")
        print(f"  - Metric type: {self.metric_type}")
        print(f"  - Metric weights: {self.metric_weights}")
        print(f"  - Normalize metrics: {self.normalize_metrics}")
        print(f"  - Fast mode iterations: {self.fast_mode_iterations}")
        print(f"  - Use TuRBO: {self.use_turbo}")
        if self.use_turbo:
            print(f"  - TuRBO config: {self.turbo_config}")
        
        # Initialize parameter bounds and names immediately
        self.parameter_bounds = self._define_parameter_bounds()
        self.param_names = list(self.parameter_bounds.keys())
        
        # Convert bounds to BoTorch format
        bounds_list = []
        for param_name in self.param_names:
            lower, upper = self.parameter_bounds[param_name]
            bounds_list.append([lower, upper])
        
        self.bounds = torch.tensor(bounds_list, dtype=torch.float64).T  # Shape: (2, n_params)
    
    def _normalize_metric(self, value: float, metric_name: str) -> float:
        """
        Normalize a metric value to [0, 1] range.
        
        Args:
            value: Raw metric value
            metric_name: Name of the metric ('rmse', 'mae', 'brier', 'transition')
            
        Returns:
            Normalized value in [0, 1] range
        """
        if not self.normalize_metrics:
            return value
        
        if metric_name not in self.metric_ranges:
            return value
        
        min_val, max_val = self.metric_ranges[metric_name]
        
        # Clamp value to expected range
        clamped_value = max(min_val, min(value, max_val))
        
        # Normalize to [0, 1]
        if max_val - min_val > 0:
            normalized = (clamped_value - min_val) / (max_val - min_val)
        else:
            normalized = 0.0
        
        return normalized
    
    def _compute_normalization_stats(self, metrics_list: List[Dict[str, float]]) -> None:
        """
        Compute min-max normalization statistics from initial samples.
        
        Args:
            metrics_list: List of metric dictionaries from initial evaluations
        """
        if not metrics_list:
            return
        
        # Extract metric values
        rmse_values = [m.get('rmse', 0.0) for m in metrics_list]
        mae_values = [m.get('mae', 0.0) for m in metrics_list]
        brier_values = [m.get('brier', 0.0) for m in metrics_list]
        transition_values = [m.get('transition', 0.0) for m in metrics_list]
        
        # Compute min-max for each metric
        self.normalization_stats = {
            'rmse': {
                'min': min(rmse_values),
                'max': max(rmse_values),
                'range': max(rmse_values) - min(rmse_values)
            },
            'mae': {
                'min': min(mae_values),
                'max': max(mae_values),
                'range': max(mae_values) - min(mae_values)
            },
            'brier': {
                'min': min(brier_values),
                'max': max(brier_values),
                'range': max(brier_values) - min(brier_values)
            },
            'transition': {
                'min': min(transition_values),
                'max': max(transition_values),
                'range': max(transition_values) - min(transition_values)
            }
        }
        
        print(f"  Computed normalization statistics from {len(metrics_list)} initial samples:")
        for metric, stats in self.normalization_stats.items():
            print(f"    {metric}: min={stats['min']:.4f}, max={stats['max']:.4f}, range={stats['range']:.4f}")
    
    def _normalize_metric_minmax(self, value: float, metric_name: str) -> float:
        """
        Normalize a metric value using min-max normalization from initial samples.
        
        Args:
            value: Raw metric value
            metric_name: Name of the metric
            
        Returns:
            Normalized value in [0, 1] range
        """
        if self.normalization_stats is None or metric_name not in self.normalization_stats:
            # Fallback to original normalization
            return self._normalize_metric(value, metric_name)
        
        stats = self.normalization_stats[metric_name]
        
        # Avoid division by zero
        if stats['range'] <= 1e-8:
            return 0.0
        
        # Min-max normalization: (value - min) / (max - min)
        normalized = (value - stats['min']) / stats['range']
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, normalized))
    
    def _get_objective_value(self, result: Dict[str, Any], iteration: int = 0) -> float:
        """
        Calculate objective value based on improved metric combination strategy.
        
        Args:
            result: Evaluation result dictionary
            iteration: Current optimization iteration
            
        Returns:
            Objective value (to be minimized)
        """
        # Extract raw metrics
        rmse = result.get('RMSE_aggregate_mean', float('inf'))
        mae = result.get('MAE_aggregate_mean', float('inf'))  
        brier = result.get('Brier_mean', float('inf'))
        transition_fit = result.get('TransitionFit_mean', float('inf'))
        
        # Check for invalid values
        if any(v == float('inf') for v in [rmse, mae, brier, transition_fit]):
            return float('inf')
        
        # Store metrics for initial normalization computation
        if len(self.initial_metrics_collected) < 20:  # Collect first 20 samples for normalization
            metrics_dict = {
                'rmse': rmse,
                'mae': mae,
                'brier': brier,
                'transition': transition_fit
            }
            self.initial_metrics_collected.append(metrics_dict)
            
            # Compute normalization stats after collecting enough samples
            if len(self.initial_metrics_collected) == 20:
                self._compute_normalization_stats(self.initial_metrics_collected)
        
        # Apply unified direction: TransitionFit loss = 1 - TransitionFit (larger is better → smaller is better)
        tf_loss = 1.0 - transition_fit
        
        # Determine metric strategy based on current phase
        if self.metric_type == 'adaptive':
            if iteration < self.fast_mode_iterations:
                current_metric_type = 'composite'
            elif iteration < self.n_trials * 0.7:
                current_metric_type = 'rmse'
            else:
                current_metric_type = 'composite'
        else:
            current_metric_type = self.metric_type
        
        # Calculate objective based on metric type
        if current_metric_type == 'rmse':
            return self._normalize_metric_minmax(rmse, 'rmse')
        
        elif current_metric_type == 'mae':
            return self._normalize_metric_minmax(mae, 'mae')
        
        elif current_metric_type == 'brier':
            return self._normalize_metric_minmax(brier, 'brier')
        
        elif current_metric_type == 'transition':
            return self._normalize_metric_minmax(tf_loss, 'transition')
        
        elif current_metric_type == 'composite':
            # Improved composite metric with min-max normalization
            
            # Normalize individual metrics using min-max from initial samples
            norm_rmse = self._normalize_metric_minmax(rmse, 'rmse')
            norm_mae = self._normalize_metric_minmax(mae, 'mae')
            norm_brier = self._normalize_metric_minmax(brier, 'brier')
            norm_tf_loss = self._normalize_metric_minmax(tf_loss, 'transition')
            
            # Weighted linear combination: y = w1*RMSE~ + w2*MAE~ + w3*Brier~ + w4*TF_loss~
            composite_score = (
                self.metric_weights.get('rmse', 0.4) * norm_rmse +
                self.metric_weights.get('mae', 0.2) * norm_mae +
                self.metric_weights.get('brier', 0.2) * norm_brier +
                self.metric_weights.get('transition', 0.2) * norm_tf_loss
            )
            
            return composite_score
        
        else:
            # Default to RMSE
            return self._normalize_metric_minmax(rmse, 'rmse')
    
    def _define_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Define parameter bounds for Bayesian optimization.
        Uses similar ranges as RandomSearchCalibrator and SBICalibrator.
        
        Returns:
            Dict mapping parameter names to (lower_bound, upper_bound) tuples
        """
        bounds = {
            # Decision weights
            'alpha': (-3.0, 3.0),
            'gamma': (0.5, 3.0),
            'theta_f': (0.5, 2.0),
            'theta_w': (0.5, 2.0),
            'theta_c': (0.5, 2.0),
            'beta_r': (-2.0, 2.0),
            'beta_i': (-2.0, 2.0),
            
            # Layer weights
            'family': (0.1, 2.0),
            'work_school': (0.1, 2.0),
            'community': (0.1, 2.0),
            
            # Info params
            'phi_family': (0.01, 0.3),
            'phi_work': (0.01, 0.3),
            'phi_community': (0.01, 0.3),
            'lambda_broadcast_base': (0.01, 0.2),
            'lambda_broadcast_factor_after_day10': (1.0, 3.0),
            'rho_info_decay': (0.1, 0.9),
            
            # Noise params
            'tau': (0.5, 2.0),
            
            # Age effects
            'age_0': (-2.0, 2.0),
            'age_1': (-2.0, 2.0),
            'age_2': (-2.0, 2.0),
            
            # Occupation effects
            'occ_0': (-2.0, 2.0),
            'occ_1': (-2.0, 2.0),
            'occ_2': (-2.0, 2.0),
        }
        
        return bounds
    
    def _update_turbo_state(self, candidate: np.ndarray, objective_value: float) -> None:
        """
        Update TuRBO trust region state based on the new candidate.
        
        Args:
            candidate: New candidate parameter vector
            objective_value: Objective value for the candidate
        """
        if not self.use_turbo:
            return
        
        self.turbo_state['iteration'] += 1
        
        # Initialize center if first iteration
        if self.turbo_state['center'] is None:
            self.turbo_state['center'] = candidate.copy()
            self.turbo_state['best_value'] = objective_value
            return
        
        # Check if this is an improvement
        if objective_value < self.turbo_state['best_value']:
            # Success: update best point and center
            self.turbo_state['best_value'] = objective_value
            self.turbo_state['center'] = candidate.copy()
            self.turbo_state['successes'] += 1
            self.turbo_state['failures'] = 0  # Reset failure count
            
            # Expand trust region if enough successes
            if self.turbo_state['successes'] >= self.turbo_config['success_tolerance']:
                new_size = min(
                    self.turbo_state['trust_region_size'] * self.turbo_config['expansion_factor'],
                    self.turbo_config['max_trust_region']
                )
                self.turbo_state['trust_region_size'] = new_size
                self.turbo_state['successes'] = 0
                print(f"  TuRBO: Expanding trust region to {new_size:.4f}")
        else:
            # Failure: increment failure count
            self.turbo_state['failures'] += 1
            self.turbo_state['successes'] = 0  # Reset success count
            
            # Shrink trust region if too many failures
            if self.turbo_state['failures'] >= self.turbo_config['failure_tolerance']:
                new_size = max(
                    self.turbo_state['trust_region_size'] * self.turbo_config['contraction_factor'],
                    self.turbo_config['min_trust_region']
                )
                self.turbo_state['trust_region_size'] = new_size
                self.turbo_state['failures'] = 0
                print(f"  TuRBO: Contracting trust region to {new_size:.4f}")
    
    def _generate_turbo_bounds(self) -> torch.Tensor:
        """
        Generate trust region bounds around the current center.
        
        Returns:
            Tensor of bounds for trust region [2, n_params]
        """
        if not self.use_turbo or self.turbo_state['center'] is None:
            # Return original bounds if TuRBO not active or no center
            bounds_list = []
            for param_name in self.param_names:
                lower, upper = self.parameter_bounds[param_name]
                bounds_list.append([lower, upper])
            return torch.tensor(bounds_list, dtype=torch.float64).T
        
        center = self.turbo_state['center']
        tr_size = self.turbo_state['trust_region_size']
        
        # Create trust region bounds
        bounds_list = []
        for i, param_name in enumerate(self.param_names):
            lower_orig, upper_orig = self.parameter_bounds[param_name]
            param_range = upper_orig - lower_orig
            
            # Trust region bounds around center
            tr_lower = max(lower_orig, center[i] - tr_size * param_range / 2)
            tr_upper = min(upper_orig, center[i] + tr_size * param_range / 2)
            
            bounds_list.append([tr_lower, tr_upper])
        
        return torch.tensor(bounds_list, dtype=torch.float64).T
    
    def _save_calibrated_parameters(self, fitted_params: FittedParams, cfg, 
                                   train_window: Tuple[int, int], seed: int) -> None:
        """
        Save calibrated parameters to outputs_BoCalibrator directory.
        
        Args:
            fitted_params: Fitted parameters
            cfg: Simulation configuration
            train_window: Training window
            seed: Random seed
        """
        # Create output directory
        output_dir = os.path.join(cfg.data_folder, "outputs_BoCalibrator")
        ensure_dir(output_dir)
        
        print(f"BoCalibrator: Saving calibrated parameters to {output_dir}")
        
        # Save configuration
        config_data = {
            "config": asdict(cfg),
            "train_window": train_window,
            "seed": seed,
            "calibrator_type": "BoCalibrator"
        }
        config_path = os.path.join(output_dir, "config.json")
        save_json(config_data, config_path)
        
        # Save calibrated parameters
        param_path = os.path.join(output_dir, "calibrated_parameters.json")
        save_json(fitted_params.to_dict(), param_path)
        
        # Save optimization history if available
        if hasattr(self, 'optimization_history') and self.optimization_history:
            history_path = os.path.join(output_dir, "optimization_history.json")
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = []
            for entry in self.optimization_history:
                serializable_entry = {}
                for key, value in entry.items():
                    if isinstance(value, np.ndarray):
                        serializable_entry[key] = value.tolist()
                    else:
                        serializable_entry[key] = value
                serializable_history.append(serializable_entry)
            save_json(serializable_history, history_path)
        
        print(f"BoCalibrator: Parameters saved successfully!")
        print(f"  - Config: {config_path}")
        print(f"  - Parameters: {param_path}")
        if hasattr(self, 'optimization_history') and self.optimization_history:
            print(f"  - History: {history_path}")
    
    def _sample_to_fitted_params(self, sample: np.ndarray, param_names: List[str], 
                                seed: int, train_window: Tuple[int, int], 
                                age_cat_names: List[str] = None, occ_cat_names: List[str] = None) -> FittedParams:
        """
        Convert a parameter sample to FittedParams format.
        
        Args:
            sample: Parameter sample array
            param_names: List of parameter names
            seed: Random seed
            train_window: Training window
            age_cat_names: List of age category names
            occ_cat_names: List of occupation category names
            
        Returns:
            FittedParams object
        """
        # Create parameter dictionary from sample
        params_dict = {name: float(sample[i]) for i, name in enumerate(param_names)}
        
        # Organize parameters into FittedParams structure
        decision_weights = {
            'alpha': params_dict.get('alpha', 0.0),
            'gamma': params_dict.get('gamma', 1.0),
            'theta_f': params_dict.get('theta_f', 1.0),
            'theta_w': params_dict.get('theta_w', 1.0),
            'theta_c': params_dict.get('theta_c', 1.0),
            'beta_r': params_dict.get('beta_r', 0.0),
            'beta_i': params_dict.get('beta_i', 0.0),
            'age_effects': {
                f'age_cat_{i}': params_dict.get(f'age_{i}', 0.0) 
                for i in range(len(age_cat_names) if age_cat_names else 3)
            },
            'occ_effects': {
                f'occ_cat_{i}': params_dict.get(f'occ_{i}', 0.0) 
                for i in range(len(occ_cat_names) if occ_cat_names else 3)
            }
        }
        
        layer_weights = {
            'family': params_dict.get('family', 1.0),
            'work_school': params_dict.get('work_school', 1.0),
            'community': params_dict.get('community', 1.0)
        }
        
        info_params = {
            'phi_family': params_dict.get('phi_family', 0.1),
            'phi_work': params_dict.get('phi_work', 0.1),
            'phi_community': params_dict.get('phi_community', 0.1),
            'lambda_broadcast_base': params_dict.get('lambda_broadcast_base', 0.05),
            'lambda_broadcast_factor_after_day10': params_dict.get('lambda_broadcast_factor_after_day10', 1.5),
            'rho_info_decay': params_dict.get('rho_info_decay', 0.5)
        }
        
        noise_params = {
            'tau': params_dict.get('tau', 1.0)
        }
        
        meta = {
            'seed': seed,
            'calibrator_name': 'bayesian_optimization',
            'training_window': train_window,
            'acquisition_function': self.acquisition_function,
            'kernel_type': self.kernel_type,
            'n_trials': self.n_trials
        }
        
        return FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            meta=meta
        )
    
    def _setup_bo_optimizer(self) -> None:
        """
        Setup Bayesian optimization components using BoTorch.
        """
        print("BoCalibrator: Setting up Bayesian optimization components...")
        
        # Define parameter bounds
        self.parameter_bounds = self._define_parameter_bounds()
        self.param_names = list(self.parameter_bounds.keys())
        n_params = len(self.param_names)
        
        print(f"  Parameter space: {n_params} dimensions")
        print(f"  Parameter bounds: {self.parameter_bounds}")
        
        # Convert bounds to BoTorch format
        # Use TuRBO bounds if enabled, otherwise use original bounds
        if self.use_turbo:
            self.bounds = self._generate_turbo_bounds()
            print(f"  Using TuRBO trust region bounds")
        else:
            bounds_list = []
            for param_name in self.param_names:
                lower, upper = self.parameter_bounds[param_name]
                bounds_list.append([lower, upper])
            self.bounds = torch.tensor(bounds_list, dtype=torch.float64).T  # Shape: (2, n_params)
        
        print(f"  Bounds tensor shape: {self.bounds.shape}")
        
        # Initialize data storage
        self.X_train = None  # Will store parameter samples
        self.Y_train = None  # Will store objective values
        
        # Initialize GP model (will be created when we have data)
        self.gp_model = None
        
        # Initialize acquisition function (keep the one set during __init__)
        # self.acquisition_function is already set in __init__
        
        print("  ✓ BO optimizer setup complete with BoTorch")
    
    def _initialize_random_samples(self, n_init: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize random samples using Sobol sampling.
        
        Args:
            n_init: Number of initial samples
            seed: Random seed
            
        Returns:
            Tuple of (X_init, Y_init) where X_init are parameters and Y_init are objectives
        """
        print(f"  Initializing {n_init} random samples using Sobol sequence...")
        
        # Use Sobol sampling for better space coverage
        X_init = draw_sobol_samples(
            bounds=self.bounds,
            n=1,
            q=n_init,
            seed=seed
        ).squeeze(0)  # Shape: (n_init, n_params)
        
        print(f"  Generated initial parameter samples: {X_init.shape}")
        return X_init
    
    def _fit_gp_model(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        """
        Fit Gaussian Process model to training data.
        
        Args:
            X: Parameter samples, shape (n_samples, n_params)
            Y: Objective values, shape (n_samples, 1)
            
        Returns:
            Fitted GP model
        """
        print(f"  Fitting GP model to {X.shape[0]} training points...")
        
        # Create GP model
        gp = SingleTaskGP(X, Y)
        
        # Create marginal log likelihood
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        
        # Fit the model
        fit_gpytorch_mll(mll)
        
        print(f"  ✓ GP model fitted successfully")
        return gp
    
    def _get_acquisition_function(self, gp_model: SingleTaskGP, best_f: float) -> Any:
        """
        Get acquisition function based on configuration.
        
        Args:
            gp_model: Fitted GP model
            best_f: Best objective value seen so far
            
        Returns:
            Acquisition function
        """
        if self.acquisition_function.lower() == 'ei':
            acq_func = ExpectedImprovement(gp_model, best_f=best_f)
        elif self.acquisition_function.lower() == 'pi':
            acq_func = ProbabilityOfImprovement(gp_model, best_f=best_f)
        elif self.acquisition_function.lower() == 'ucb':
            acq_func = UpperConfidenceBound(gp_model, beta=2.0)
        else:
            print(f"  Warning: Unknown acquisition function '{self.acquisition_function}', using EI")
            acq_func = ExpectedImprovement(gp_model, best_f=best_f)
        
        print(f"  Using acquisition function: {self.acquisition_function.upper()}")
        return acq_func
    
    def _optimize_acquisition(self, acq_func: Any, n_candidates: int = 1) -> torch.Tensor:
        """
        Optimize acquisition function to get next candidate points.
        
        Args:
            acq_func: Acquisition function
            n_candidates: Number of candidate points to generate
            
        Returns:
            Next candidate parameters, shape (n_candidates, n_params)
        """
        print(f"  Optimizing acquisition function for {n_candidates} candidate(s)...")
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_candidates,
            num_restarts=20,  # Number of random restarts
            raw_samples=100,  # Number of raw samples for initialization
        )
        
        print(f"  ✓ Generated {candidates.shape[0]} candidate(s)")
        return candidates
    
    def _objective_function(self, params: np.ndarray, bundle, evaluator, 
                           train_window: Tuple[int, int], seed: int, iteration: int = 0) -> float:
        """
        Objective function for Bayesian optimization with support for composite metrics.
        
        Args:
            params: Parameter vector
            bundle: Data bundle
            evaluator: Evaluation function
            train_window: Training window
            seed: Random seed
            iteration: Current optimization iteration
            
        Returns:
            Objective value (to be minimized, already positive)
        """
        try:
            # Extract age and occupation category names from bundle
            wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
            n_age_cats = age_oh.shape[1]
            n_occ_cats = occ_oh.shape[1]
            age_cat_names = [f'age_cat_{i}' for i in range(n_age_cats)]
            occ_cat_names = [f'occ_cat_{i}' for i in range(n_occ_cats)]
            
            # Convert parameters to FittedParams
            fitted_params = self._sample_to_fitted_params(
                params, self.param_names, seed, train_window, age_cat_names, occ_cat_names
            )
            
            # Determine if we should use fast mode (fewer simulations)
            fast_mode = iteration < self.fast_mode_iterations
            if fast_mode:
                # Temporarily modify k_runs for faster evaluation
                wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
                original_k_runs = cfg.k_runs
                cfg.k_runs = 5  # Fast mode: use only 5 runs instead of 20
                result = evaluator(None, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
                cfg.k_runs = original_k_runs  # Restore original value
            else:
                # Full evaluation mode
                wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
                result = evaluator(None, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
            
            # Calculate objective value using the new metric system
            objective_value = self._get_objective_value(result, iteration)
            
            # Store evaluation history for analysis
            self.optimization_history.append({
                'iteration': iteration,
                'params': params.copy(),
                'objective': objective_value,
                'fast_mode': fast_mode,
                'metrics': {
                    'rmse': result.get('RMSE_aggregate_mean', float('inf')),
                    'mae': result.get('MAE_aggregate_mean', float('inf')),
                    'brier': result.get('Brier_mean', float('inf')),
                    'transition': result.get('TransitionFit_mean', float('inf'))
                }
            })
            
            return objective_value  # Already positive, to be minimized
            
        except Exception as e:
            print(f"BoCalibrator: Objective function evaluation failed: {e}")
            return float('inf')  # Worst possible score
    
    def _run_bayesian_optimization(self, bundle, evaluator, train_window: Tuple[int, int], seed: int) -> Tuple[np.ndarray, float]:
        """
        Run Bayesian optimization to find optimal parameters using BoTorch.
        
        Args:
            bundle: Data bundle
            evaluator: Evaluation function
            train_window: Training window
            seed: Random seed
            
        Returns:
            Tuple of (optimal_parameters, optimal_objective_value)
        """
        print(f"BoCalibrator: Running Bayesian optimization with {self.n_trials} trials...")
        
        # Setup optimizer
        self._setup_bo_optimizer()
        
        # Initialize optimization history
        self.optimization_history = []
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Step 1: Initialize with random samples
        n_init = min(10, self.n_trials // 3)  # Use 10 initial samples or 1/3 of trials
        print(f"Step 1: Initializing with {n_init} random samples...")
        
        X_init = self._initialize_random_samples(n_init, seed)
        
        # Evaluate initial samples
        Y_init = []
        for i in range(n_init):
            params_np = X_init[i].numpy()
            objective = self._objective_function(params_np, bundle, evaluator, train_window, seed, iteration=i)
            Y_init.append(objective)
            print(f"  Sample {i+1}/{n_init}: objective = {objective:.4f}")
        
        Y_init = torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1)  # Shape: (n_init, 1)
        
        # Update training data
        self.X_train = X_init
        self.Y_train = Y_init
        
        # Find best initial point
        best_idx = Y_init.argmin()
        best_params = X_init[best_idx].numpy()
        best_objective = Y_init[best_idx].item()
        
        # Initialize TuRBO state with best initial point
        if self.use_turbo:
            self.turbo_state['center'] = best_params.copy()
            self.turbo_state['best_value'] = best_objective
            print(f"  TuRBO: Initialized trust region center with best initial sample")
        
        print(f"  Best initial sample: objective = {best_objective:.4f}")
        
        # Step 2-5: Main optimization loop
        print(f"Step 2-5: Running main optimization loop for {self.n_trials - n_init} iterations...")
        
        for iteration in range(n_init, self.n_trials):
            print(f"\n--- Iteration {iteration + 1}/{self.n_trials} ---")
            
            # Step 2: Fit GP model
            gp_model = self._fit_gp_model(self.X_train, self.Y_train)
            
            # Step 3: Get acquisition function and optimize it
            acq_func = self._get_acquisition_function(gp_model, best_objective)
            candidates = self._optimize_acquisition(acq_func, n_candidates=1)
            
            # Step 4: Evaluate new candidate
            new_params = candidates[0].numpy()
            new_objective = self._objective_function(new_params, bundle, evaluator, train_window, seed, iteration=iteration)
            
            print(f"  New candidate: objective = {new_objective:.4f}")
            
            # Update training data
            self.X_train = torch.cat([self.X_train, candidates], dim=0)
            self.Y_train = torch.cat([self.Y_train, torch.tensor([[new_objective]], dtype=torch.float64)], dim=0)
            
            # Update TuRBO state
            if self.use_turbo:
                self._update_turbo_state(new_params, new_objective)
                # Update bounds for next iteration
                self.bounds = self._generate_turbo_bounds()
            
            # Update best if improved
            if new_objective < best_objective:
                best_params = new_params.copy()
                best_objective = new_objective
                print(f"  ✓ New best found: objective = {best_objective:.4f}")
            
            # Store iteration info
            self.optimization_history.append({
                'iteration': iteration,
                'best_objective': best_objective,
                'new_objective': new_objective,
                'improvement': best_objective - new_objective if new_objective < best_objective else 0.0
            })
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == self.n_trials - 1:
                print(f"  Progress: {iteration + 1}/{self.n_trials} iterations, Best objective: {best_objective:.4f}")
        
        print(f"\nBoCalibrator: Optimization completed!")
        print(f"  Final best objective: {best_objective:.4f}")
        print(f"  Total evaluations: {len(self.optimization_history) + n_init}")
        if len(self.optimization_history) > 0:
            best_iter_idx = np.where(np.array([h.get('best_objective', h.get('objective', float('inf'))) for h in self.optimization_history]) == best_objective)[0]
            if len(best_iter_idx) > 0:
                print(f"  Best parameters found in iteration: {best_iter_idx[0] + n_init}")
            else:
                print(f"  Best parameters found in iteration: 0")
        else:
            print(f"  Best parameters found in iteration: 0")
        
        return best_params, best_objective
    
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
        """
        Fit parameters using Bayesian Optimization.
        
        Args:
            bundle: Data bundle containing simulation data
            simulator: Simulator object (not used by BoCalibrator)
            evaluator: Evaluation function
            train_window: Training window
            seed: Random seed
            
        Returns:
            FittedParams object with optimized parameters
        """
        print(f"BoCalibrator: Starting Bayesian optimization calibration...")
        
        # Set seed for reproducibility
        if self.random_state is not None:
            set_global_seed(self.random_state)
        else:
            set_global_seed(seed)
        
        # Run Bayesian optimization
        optimal_params, optimal_objective = self._run_bayesian_optimization(
            bundle, evaluator, train_window, seed
        )
        
        # Extract age and occupation category names from bundle
        wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
        n_age_cats = age_oh.shape[1]
        n_occ_cats = occ_oh.shape[1]
        age_cat_names = [f'age_cat_{i}' for i in range(n_age_cats)]
        occ_cat_names = [f'occ_cat_{i}' for i in range(n_occ_cats)]
        
        # Convert optimal parameters to FittedParams
        fitted_params = self._sample_to_fitted_params(
            optimal_params, self.param_names, seed, train_window, age_cat_names, occ_cat_names
        )
        
        # Update meta information
        fitted_params.meta.update({
            'optimal_objective_value': float(optimal_objective),
            'n_optimization_trials': self.n_trials,
            'optimization_history_length': len(self.optimization_history),
            'bo_config': {
                'acquisition_function': self.acquisition_function,
                'kernel_type': self.kernel_type,
                'n_trials': self.n_trials,
                'metric_type': self.metric_type,
                'metric_weights': self.metric_weights,
                'normalize_metrics': self.normalize_metrics,
                'fast_mode_iterations': self.fast_mode_iterations,
                'use_turbo': self.use_turbo,
                'turbo_config': self.turbo_config if self.use_turbo else None,
                'n_parameters': len(self.param_names),
                'parameter_bounds': {name: list(bounds) for name, bounds in self.parameter_bounds.items()},
                'botorch_version': botorch.__version__,
                'pytorch_version': torch.__version__
            },
            'optimization_summary': {
                'n_initial_samples': min(10, self.n_trials // 3),
                'n_bo_iterations': self.n_trials - min(10, self.n_trials // 3),
                'best_improvement': max([h.get('improvement', 0) for h in self.optimization_history], default=0.0),
                'convergence_info': {
                    'final_improvement': self.optimization_history[-1].get('improvement', 0.0) if self.optimization_history else 0.0,
                    'total_improvements': sum(1 for h in self.optimization_history if h.get('improvement', 0) > 0)
                }
            }
        })
        
        # Evaluate final parameters for reporting
        print("BoCalibrator: Evaluating final optimized parameters...")
        try:
            wearing, neighbors, risk, age_oh, occ_oh, cfg = bundle
            result = evaluator(simulator, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
            
            # Report all metrics
            rmse = result.get('RMSE_aggregate_mean', float('inf'))
            mae = result.get('MAE_aggregate_mean', float('inf'))
            brier = result.get('Brier_mean', float('inf'))
            transition = result.get('TransitionFit_mean', float('inf'))
            
            print(f"BoCalibrator: Final metrics:")
            print(f"  - RMSE = {rmse:.4f}")
            print(f"  - MAE = {mae:.4f}")
            print(f"  - Brier = {brier:.4f}")
            print(f"  - TransitionFit = {transition:.4f}")
            
            # Calculate and report composite score
            if self.metric_type in ['composite', 'adaptive']:
                composite_score = self._get_objective_value(result, self.n_trials)
                print(f"  - Composite Score = {composite_score:.4f}")
                
        except Exception as e:
            print(f"BoCalibrator: Final evaluation failed: {e}")
        
        # Save calibrated parameters to outputs_BoCalibrator directory
        self._save_calibrated_parameters(fitted_params, cfg, train_window, seed)
        
        print("BoCalibrator: Bayesian optimization calibration completed!")
        
        return fitted_params


class EvoCalibrator(Calibrator):
    """Evolutionary Algorithm calibrator using EvoTorch for parameter optimization."""
    
    def __init__(self, n_generations: int = 10, population_size: int = 200, 
                 algorithm: str = 'GA', random_state: int = None,
                 metric_type: str = 'rmse', mutation_sigma: float = 0.03,
                 crossover_prob: float = 1.0, crossover_eta: float = 8.0,
                 tournament_size: int = 3, elite_ratio: float = 0.1):
        """
        Initialize Evolutionary Algorithm calibrator using Genetic Algorithm (GA).
        
        Args:
            n_generations: Number of generations to evolve (default: 10)
            population_size: Size of the population in each generation (default: 200)
            algorithm: Evolution algorithm type (fixed to 'GA')
            random_state: Random seed for reproducibility
            metric_type: Metric type to optimize ('rmse', 'mae', 'brier', 'composite')
            mutation_sigma: Standard deviation for Gaussian mutation (default: 0.03)
            crossover_prob: Crossover probability for SBX (default: 1.0)
            crossover_eta: Distribution index for SBX (default: 8.0)
            tournament_size: Tournament size for selection (default: 3)
            elite_ratio: Ratio of elite individuals to preserve each generation
        """
        if not EVOTORCH_AVAILABLE:
            raise ImportError("EvoTorch is not available. Please install evotorch to use EvoCalibrator.")
        
        # Force algorithm to be GA for this calibrator
        if algorithm != 'GA':
            print(f"Warning: EvoCalibrator only supports GA algorithm. Changing from '{algorithm}' to 'GA'")
        
        self.n_generations = n_generations
        self.population_size = population_size
        self.algorithm = 'GA'  # Fixed to GA
        self.random_state = random_state
        self.metric_type = metric_type
        self.mutation_sigma = mutation_sigma
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.tournament_size = tournament_size
        self.elite_ratio = elite_ratio
        
        # Evolution components
        self.problem = None
        self.algorithm_instance = None
        self.parameter_bounds = None
        self.param_names = None
        self.optimization_history = []
        
        print(f"EvoCalibrator initialized with Genetic Algorithm (GA):")
        print(f"  - Generations: {self.n_generations}")
        print(f"  - Population size: {self.population_size}")
        print(f"  - Algorithm: {self.algorithm}")
        print(f"  - Random state: {self.random_state}")
        print(f"  - Metric type: {self.metric_type}")
        print(f"  - Selection: Tournament selection (size={self.tournament_size})")
        print(f"  - Crossover: SBX (prob={self.crossover_prob}, eta={self.crossover_eta})")
        print(f"  - Mutation: Gaussian (sigma={self.mutation_sigma})")
        print(f"  - Elite ratio: {self.elite_ratio}")
        
        # Initialize parameter bounds
        self.parameter_bounds = self._define_parameter_bounds()
        self.param_names = list(self.parameter_bounds.keys())
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
    
    def _define_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define parameter bounds for optimization."""
        return {
            # Decision model parameters
            'alpha': (0.001, 0.999),        # Logit intercept
            'gamma': (0.01, 5.0),           # Risk sensitivity
            'theta_f': (0.01, 5.0),         # Friend influence
            'theta_w': (0.01, 5.0),         # Work influence  
            'theta_c': (0.01, 5.0),         # Community influence
            'beta_r': (0.01, 5.0),          # Risk perception coefficient
            'beta_i': (0.01, 5.0),          # Info coefficient
            
            # Information model parameters
            'gov_lam_factor': (0.5, 3.0),   # Government information strength
            'rho_info_decay': (0.1, 0.9),   # Information decay rate
            
            # Demographic coefficients
            'age_coef_1': (-2.0, 2.0),      # Age group 1
            'age_coef_2': (-2.0, 2.0),      # Age group 2
            'age_coef_3': (-2.0, 2.0),      # Age group 3
            'age_coef_4': (-2.0, 2.0),      # Age group 4
            'age_coef_5': (-2.0, 2.0),      # Age group 5
            'occ_coef_0': (-2.0, 2.0),      # Occupation 0
            'occ_coef_1': (-2.0, 2.0),      # Occupation 1
            'occ_coef_2': (-2.0, 2.0),      # Occupation 2
            'occ_coef_3': (-2.0, 2.0),      # Occupation 3
            'occ_coef_4': (-2.0, 2.0),      # Occupation 4
            'occ_coef_5': (-2.0, 2.0),      # Occupation 5
            'occ_coef_6': (-2.0, 2.0),      # Occupation 6
            'occ_coef_7': (-2.0, 2.0),      # Occupation 7
            'occ_coef_8': (-2.0, 2.0),      # Occupation 8
            'occ_coef_9': (-2.0, 2.0),      # Occupation 9
        }
    
    def _create_fitness_function(self, evaluator, bundle, train_window):
        """Create fitness function for evolutionary optimization."""
        
        def fitness_function(parameters_tensor):
            """
            Fitness function that converts tensor parameters to FittedParams
            and evaluates using the provided evaluator.
            
            Args:
                parameters_tensor: 1D Tensor of shape (n_parameters,) for single solution
                
            Returns:
                Scalar fitness value (negative RMSE for maximization)
            """
            # Convert tensor to parameter dictionary
            param_dict = {}
            for j, param_name in enumerate(self.param_names):
                param_dict[param_name] = float(parameters_tensor[j].item())
            
            # Create FittedParams instance with proper structure
            decision_weights = {
                'alpha': param_dict['alpha'],
                'gamma': param_dict['gamma'],
                'theta_f': param_dict['theta_f'],
                'theta_w': param_dict['theta_w'],
                'theta_c': param_dict['theta_c'],
                'beta_r': param_dict['beta_r'],
                'beta_i': param_dict['beta_i'],
                'age_effects': {
                    '1': param_dict['age_coef_1'],
                    '2': param_dict['age_coef_2'],
                    '3': param_dict['age_coef_3'],
                    '4': param_dict['age_coef_4'],
                    '5': param_dict['age_coef_5']
                },
                'occ_effects': {
                    '0': param_dict['occ_coef_0'],
                    '1': param_dict['occ_coef_1'],
                    '2': param_dict['occ_coef_2'],
                    '3': param_dict['occ_coef_3'],
                    '4': param_dict['occ_coef_4'],
                    '5': param_dict['occ_coef_5'],
                    '6': param_dict['occ_coef_6'],
                    '7': param_dict['occ_coef_7'],
                    '8': param_dict['occ_coef_8'],
                    '9': param_dict['occ_coef_9']
                }
            }
            
            layer_weights = {
                'family': 1.0,
                'work_school': 1.0,
                'community': 1.0
            }
            
            info_params = {
                'phi_family': 0.1,
                'phi_work': 0.1,
                'phi_community': 0.1,
                'lambda_broadcast_base': 0.05,
                'lambda_broadcast_factor_after_day10': param_dict['gov_lam_factor'],
                'rho_info_decay': param_dict['rho_info_decay']
            }
            
            noise_params = {
                'tau': 1.0
            }
            
            meta = {
                'method': "evolutionary",
                'n_trials': self.n_generations * self.population_size,
                'generation': len(self.optimization_history) // self.population_size,
                'individual': len(self.optimization_history) % self.population_size
            }
            
            fitted_params = FittedParams(
                decision_weights=decision_weights,
                layer_weights=layer_weights,
                info_params=info_params,
                noise_params=noise_params,
                meta=meta
            )
            
            try:
                # Evaluate parameters using the correct signature
                # Evaluator expects (simulator, fitted_params, bundle_tuple)
                bundle_tuple = (bundle[0], bundle[1], bundle[2], bundle[3], bundle[4], train_window, bundle[5])
                result = evaluator(None, fitted_params, bundle_tuple)
                
                # Extract fitness based on metric type
                if self.metric_type == 'rmse':
                    fitness = -result.get('RMSE_aggregate_mean', float('inf'))  # Negative for maximization
                elif self.metric_type == 'mae':
                    fitness = -result.get('MAE_aggregate_mean', float('inf'))
                elif self.metric_type == 'brier':
                    fitness = -result.get('Brier_mean', float('inf'))
                else:  # composite or other
                    fitness = -result.get('RMSE_aggregate_mean', float('inf'))
                
                # Handle invalid results
                if not np.isfinite(fitness):
                    fitness = -1e6  # Large negative value for invalid results
                
                # Debug output for first few evaluations
                eval_count = len(self.optimization_history)
                if eval_count < 5:
                    print(f"  Evaluation {eval_count + 1}: fitness = {fitness:.6f}, RMSE = {-fitness:.6f}")
                    
            except Exception as e:
                print(f"Error evaluating parameters: {e}")
                fitness = -1e6  # Large negative value for failed evaluations
            
            # Store optimization history
            self.optimization_history.append({
                'generation': len(self.optimization_history) // self.population_size,
                'individual': len(self.optimization_history) % self.population_size,
                'parameters': param_dict.copy(),
                'fitness': fitness,
                'rmse': -fitness if self.metric_type == 'rmse' else result.get('RMSE_aggregate_mean', float('inf'))
            })
            
            # Ensure fitness is a finite number and convert to tensor
            if not np.isfinite(fitness):
                fitness = -1e6
            
            return torch.tensor(fitness, dtype=torch.float32)
        
        return fitness_function
    
    def _create_problem(self, evaluator, bundle, train_window):
        """Create EvoTorch optimization problem."""
        
        # Get parameter bounds
        bounds_lower = []
        bounds_upper = []
        for param_name in self.param_names:
            lower, upper = self.parameter_bounds[param_name]
            bounds_lower.append(lower)
            bounds_upper.append(upper)
        
        bounds_lower = torch.tensor(bounds_lower, dtype=torch.float32)
        bounds_upper = torch.tensor(bounds_upper, dtype=torch.float32)
        
        # Create fitness function
        fitness_fn = self._create_fitness_function(evaluator, bundle, train_window)
        
        # Create problem
        problem = Problem(
            objective_sense="max",  # We want to maximize fitness (minimize negative RMSE)
            solution_length=len(self.param_names),
            bounds=(bounds_lower, bounds_upper),
            objective_func=fitness_fn,
            dtype=torch.float32
        )
        
        return problem
    
    def _create_algorithm(self, problem):
        """Create Genetic Algorithm instance with proper ES configuration."""
        
        # Create genetic operators for EvoTorch following ES configuration
        # Gaussian mutation with specified standard deviation
        mutation_op = GaussianMutation(
            problem,
            stdev=self.mutation_sigma
        )
        
        # Simulated Binary Crossover (SBX) with specified parameters
        crossover_op = SimulatedBinaryCrossOver(
            problem,
            tournament_size=self.tournament_size,
            eta=self.crossover_eta
        )
        
        # Create GA with both crossover and mutation operators
        algorithm = GeneticAlgorithm(
            problem,
            popsize=self.population_size,
            operators=[crossover_op, mutation_op]
        )
        
        print(f"EvoCalibrator: Created GA with ES configuration:")
        print(f"  - Population size: {self.population_size}")
        print(f"  - Tournament selection size: {self.tournament_size}")
        print(f"  - SBX crossover (eta={self.crossover_eta})")
        print(f"  - Gaussian mutation (sigma={self.mutation_sigma})")
        
        return algorithm
    
    def _save_calibrated_parameters(self, fitted_params: FittedParams, cfg, 
                                   train_window: Tuple[int, int], seed: int) -> None:
        """
        Save calibrated parameters to outputs_EvoCalibrator directory.
        
        Args:
            fitted_params: Fitted parameters
            cfg: Simulation configuration
            train_window: Training window
            seed: Random seed
        """
        # Create output directory with calibrator-specific name
        output_dir = os.path.join(cfg.data_folder, "outputs_EvoCalibrator")
        ensure_dir(output_dir)
        
        print(f"EvoCalibrator: Saving calibrated parameters to {output_dir}")
        
        # Save configuration
        config_data = {
            "config": asdict(cfg),
            "train_window": train_window,
            "seed": seed,
            "calibrator_type": "EvoCalibrator",
            "algorithm": self.algorithm,
            "n_generations": self.n_generations,
            "population_size": self.population_size,
            "mutation_sigma": self.mutation_sigma,
            "crossover_prob": self.crossover_prob,
            "crossover_eta": self.crossover_eta,
            "tournament_size": self.tournament_size,
            "elite_ratio": self.elite_ratio,
            "metric_type": self.metric_type
        }
        config_path = os.path.join(output_dir, "config.json")
        save_json(config_data, config_path)
        
        # Save calibrated parameters
        param_path = os.path.join(output_dir, "calibrated_parameters.json")
        save_json(fitted_params.to_dict(), param_path)
        
        # Save optimization history if available
        if hasattr(self, 'optimization_history') and self.optimization_history:
            history_path = os.path.join(output_dir, "optimization_history.json")
            # Convert numpy arrays and tensors to lists for JSON serialization
            serializable_history = []
            for entry in self.optimization_history:
                serializable_entry = {}
                for key, value in entry.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        serializable_entry[key] = value.tolist() if hasattr(value, 'tolist') else list(value)
                    elif isinstance(value, (np.float32, np.float64)):
                        serializable_entry[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        serializable_entry[key] = int(value)
                    else:
                        serializable_entry[key] = value
                serializable_history.append(serializable_entry)
            
            save_json(serializable_history, history_path)
            print(f"EvoCalibrator: Saved optimization history to {history_path}")
        
        print(f"EvoCalibrator: Parameters saved successfully!")
        print(f"  - Config: {config_path}")
        print(f"  - Parameters: {param_path}")
        if hasattr(self, 'optimization_history') and self.optimization_history:
            print(f"  - History: {history_path}")
    
    def fit(self, bundle, simulator, evaluator, train_window, seed: int = None) -> FittedParams:
        """
        Fit parameters using evolutionary algorithm optimization.
        
        Args:
            bundle: Data bundle containing network and features
            simulator: Simulator function (not used directly)
            evaluator: Evaluation function that takes (fitted_params, bundle, train_window)
            train_window: Training window tuple (start_idx, end_idx)
            seed: Random seed for reproducibility
            
        Returns:
            FittedParams: Optimized parameters
        """
        print(f"EvoCalibrator: Starting evolutionary optimization...")
        print(f"  - Parameters to optimize: {len(self.param_names)}")
        print(f"  - Generations: {self.n_generations}")
        print(f"  - Population size: {self.population_size}")
        print(f"  - Algorithm: {self.algorithm}")
        print(f"  - Metric: {self.metric_type}")
        
        # Set random seed
        if seed is not None:
            self.random_state = seed
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Reset optimization history
        self.optimization_history = []
        
        # Create optimization problem
        self.problem = self._create_problem(evaluator, bundle, train_window)
        
        # Create algorithm
        self.algorithm_instance = self._create_algorithm(self.problem)
        
        # Run evolution
        print(f"  - Running {self.n_generations} generations...")
        
        best_fitness_history = []
        mean_fitness_history = []
        
        for generation in range(self.n_generations):
            # Run one generation
            self.algorithm_instance.step()
            
            # Get population fitness
            population = self.algorithm_instance.population
            fitness_values = population.evals
            
            # Track statistics
            best_fitness = float(torch.max(fitness_values))
            mean_fitness = float(torch.mean(fitness_values))
            
            best_fitness_history.append(best_fitness)
            mean_fitness_history.append(mean_fitness)
            
            # Print progress
            if generation % 10 == 0 or generation == self.n_generations - 1:
                print(f"    Generation {generation + 1:3d}/{self.n_generations}: "
                      f"Best fitness = {best_fitness:.6f}, Mean fitness = {mean_fitness:.6f}")
        
        # Get best solution
        population = self.algorithm_instance.population
        best_idx = torch.argmax(population.evals)
        best_solution = population.values[best_idx]
        best_fitness = float(population.evals[best_idx])
        
        # Convert best solution to parameter dictionary
        best_params_dict = {}
        for i, param_name in enumerate(self.param_names):
            best_params_dict[param_name] = float(best_solution[i].item())
        
        print(f"  - Optimization completed!")
        print(f"  - Best fitness: {best_fitness:.6f}")
        print(f"  - Best RMSE: {-best_fitness:.6f}")
        
        # Create FittedParams with best solution
        decision_weights = {
            'alpha': best_params_dict['alpha'],
            'gamma': best_params_dict['gamma'],
            'theta_f': best_params_dict['theta_f'],
            'theta_w': best_params_dict['theta_w'],
            'theta_c': best_params_dict['theta_c'],
            'beta_r': best_params_dict['beta_r'],
            'beta_i': best_params_dict['beta_i'],
            'age_effects': {
                '1': best_params_dict['age_coef_1'],
                '2': best_params_dict['age_coef_2'],
                '3': best_params_dict['age_coef_3'],
                '4': best_params_dict['age_coef_4'],
                '5': best_params_dict['age_coef_5']
            },
            'occ_effects': {
                '0': best_params_dict['occ_coef_0'],
                '1': best_params_dict['occ_coef_1'],
                '2': best_params_dict['occ_coef_2'],
                '3': best_params_dict['occ_coef_3'],
                '4': best_params_dict['occ_coef_4'],
                '5': best_params_dict['occ_coef_5'],
                '6': best_params_dict['occ_coef_6'],
                '7': best_params_dict['occ_coef_7'],
                '8': best_params_dict['occ_coef_8'],
                '9': best_params_dict['occ_coef_9']
            }
        }
        
        layer_weights = {
            'family': 1.0,
            'work_school': 1.0,
            'community': 1.0
        }
        
        info_params = {
            'phi_family': 0.1,
            'phi_work': 0.1,
            'phi_community': 0.1,
            'lambda_broadcast_base': 0.05,
            'lambda_broadcast_factor_after_day10': best_params_dict['gov_lam_factor'],
            'rho_info_decay': best_params_dict['rho_info_decay']
        }
        
        noise_params = {
            'tau': 1.0
        }
        
        meta = {
            'seed': self.random_state,
            'calibrator_name': 'evolutionary',
            'method': "evolutionary",
            'algorithm': self.algorithm,
            'n_generations': self.n_generations,
            'population_size': self.population_size,
            'n_trials': self.n_generations * self.population_size,
            'convergence_history': best_fitness_history,
            'best_fitness': best_fitness,
            'best_rmse': -best_fitness
        }
        
        fitted_params = FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            meta=meta
        )
        
        # Save calibrated parameters to outputs_EvoCalibrator directory
        # Extract configuration from bundle if available
        if len(bundle) >= 6:
            cfg = bundle[5]  # Configuration is the 6th element in bundle
            self._save_calibrated_parameters(fitted_params, cfg, train_window, seed)
        
        print("EvoCalibrator: Evolutionary algorithm calibration completed!")
        
        return fitted_params


# Calibrator registry
CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "sbi": SBICalibrator,
    "bo": BoCalibrator,
    "evo": EvoCalibrator,
}


def get_calibrator(name: str, config_path: str = None, **kwargs):
    """Get calibrator by name with optional configuration."""
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    
    # For now, use default parameters
    # TODO: Load optional config (JSON/YAML) into kwargs
    # Allow passing kwargs for calibrator-specific parameters
    
    return CALIBRATOR_REGISTRY[name](**kwargs)


def main() -> None:
    """
    Orchestrates the multi-agent simulation:
    - Load data
    - Align IDs and build multiplex network
    - Prepare features and neighbor shares
    - Calibrate info and decision parameters
    - Evaluate on temporal holdout with K-run simulation
    - Forecast days 30-39
    - Save outputs and configuration
    """
    cfg = SimulationConfig()
    set_global_seed(cfg.seed)

    # Prepare output directory
    out_dir = os.path.join(cfg.data_folder, cfg.output_folder)
    ensure_dir(out_dir)

    # Load data
    agents_path = os.path.join(cfg.data_folder, "agent_attributes.csv")
    social_path = os.path.join(cfg.data_folder, "social_network.json")
    train_path = os.path.join(cfg.data_folder, "train_data.csv")
    try:
        agents_df = load_agent_attributes(agents_path)
        social_raw = load_social_network(social_path)
        train_df = load_train_data(train_path)
    except Exception as e:
        # Print error and exit gracefully
        print(str(e))
        return

    # Align IDs
    common_ids, id2idx, agents_df, social_f, train_df = align_ids(agents_df, social_raw, train_df)
    n_agents = len(common_ids)

    # Build network (multiplex)
    neighbors = build_multiplex_adjacency(social_f, id2idx, n_agents)

    # Pivot time series
    wearing, received, days = pivot_states(train_df, id2idx)
    T = wearing.shape[0]

    # Risk perception aligned
    risk_perception = agents_df.set_index("agent_id").loc[common_ids]["risk_perception"].to_numpy(dtype=np.float64)

    # Encode demographics
    age_oh, age_cat_names, occ_oh, occ_cat_names = encode_demographics(agents_df, common_ids)

    # Prepare neighbor shares per day using observed wearing states
    share_f_by_day = np.zeros_like(wearing)
    share_w_by_day = np.zeros_like(wearing)
    share_c_by_day = np.zeros_like(wearing)
    for t in range(T):
        s_prev = wearing[t, :]
        share_f_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["family"])
        share_w_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["work_school"])
        share_c_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["community"])

    # Compute mem_info array using default rho for training features
    rho_train = cfg.rho_info_decay_default
    mem_info = compute_mem_info(received, rho=rho_train)

    # Split into train/validation
    train_end_idx, val_start_idx, val_end_idx = build_train_validation_splits(days, cfg.val_split_ratio)

    # Use new pluggable calibration architecture
    # Create calibrator - uncomment the desired calibrator
    # calibrator = get_calibrator("logit_head")      # Logistic regression calibrator
    # calibrator = get_calibrator("random_search")  # Random search calibrator
    # calibrator = get_calibrator("sbi")            # Simulation-Based Inference calibrator
    # calibrator = get_calibrator("bo")             # Bayesian Optimization calibrator
    # calibrator = get_calibrator("evo")            # Evolutionary Algorithm calibrator
    
    # SBI Standard M3 configuration (K=5, N=1000) - COMMENTED OUT:
    # calibrator = get_calibrator("sbi", k_observables=5, n_simulations=1000,
    #                            neural_net_config={'batch_size': 64, 'hidden_features': 32, 'max_epochs': 50})
    
    # Bayesian Optimization configuration - CURRENTLY ACTIVE (IMPROVED):
    # Standard BO:
    # calibrator = get_calibrator("bo", n_trials=300, acquisition_function='EI', 
    #                            kernel_type='RBF', random_state=42,
    #                            metric_type='composite', 
    #                            metric_weights={'rmse': 0.4, 'mae': 0.2, 'brier': 0.2, 'transition': 0.2},
    #                            normalize_metrics=True, fast_mode_iterations=50)
    
    # TuRBO (Trust Region Bayesian Optimization) - COMMENTED OUT:
    # calibrator = get_calibrator("bo", n_trials=300, acquisition_function='EI', 
    #                            kernel_type='RBF', random_state=42,
    #                            metric_type='composite', 
    #                            metric_weights={'rmse': 0.4, 'mae': 0.2, 'brier': 0.2, 'transition': 0.2},
    #                            normalize_metrics=True, fast_mode_iterations=50,
    #                            use_turbo=True,
    #                            turbo_config={
    #                                'trust_region_size': 0.8, 
    #                                'success_tolerance': 3, 
    #                                'failure_tolerance': 10,
    #                                'expansion_factor': 2.0,
    #                                'contraction_factor': 0.5,
    #                                'min_trust_region': 1e-8,
    #                                'max_trust_region': 1.0
    #                            })
    
    # Evolutionary Algorithm configuration - CURRENTLY ACTIVE:
    # Using ES (Evolution Strategy) configuration for parameter optimization
    # Population size: 200, Generations: 10, Tournament selection, SBX crossover, Gaussian mutation
    calibrator = get_calibrator("evo", n_generations=10, population_size=200, 
                               algorithm='GA', random_state=42, metric_type='rmse',
                               mutation_sigma=0.03, crossover_prob=1.0, crossover_eta=8.0,
                               tournament_size=3, elite_ratio=0.1)
    
    # Evolutionary Algorithm configuration - ALTERNATIVE ES CONFIGURATIONS:
    
    # Standard ES configuration (longer evolution):
    # calibrator = get_calibrator("evo", n_generations=20, population_size=100, 
    #                            algorithm='GA', random_state=42, metric_type='rmse',
    #                            mutation_sigma=0.05, crossover_prob=0.9, crossover_eta=10.0,
    #                            tournament_size=2, elite_ratio=0.05)
    
    # High exploration ES configuration:
    # calibrator = get_calibrator("evo", n_generations=15, population_size=150,
    #                            algorithm='GA', random_state=42, metric_type='rmse',
    #                            mutation_sigma=0.08, crossover_prob=1.0, crossover_eta=5.0,
    #                            tournament_size=4, elite_ratio=0.15)
    
    # Conservative ES configuration (small mutations):
    # calibrator = get_calibrator("evo", n_generations=25, population_size=80,
    #                            algorithm='GA', random_state=42, metric_type='rmse',
    #                            mutation_sigma=0.01, crossover_prob=0.8, crossover_eta=15.0,
    #                            tournament_size=2, elite_ratio=0.2)
    
    # Prepare bundle for calibrator
    bundle = (wearing, neighbors, risk_perception, age_oh, occ_oh, cfg)
    
    # Define training window
    train_window = (1, train_end_idx)
    
    # Fit parameters using calibrator
    fitted_params = calibrator.fit(
        bundle=bundle,
        simulator=None,  # Not used by LogitHeadCalibrator
        evaluator=evaluate_params,
        train_window=train_window,
        seed=cfg.seed
    )
    
    # Convert to legacy parameters for compatibility with existing code
    params = fitted_params.to_parameters()
    
    # Parameters are already available from fitted_params
    # No need for manual mapping - use the converted params directly

    # Evaluate on validation via K-run simulation
    metrics = evaluate_on_validation(
        wearing=wearing,
        neighbors=neighbors,
        risk=risk_perception,
        age_oh=age_oh,
        occ_oh=occ_oh,
        age_cat_names=age_cat_names,
        occ_cat_names=occ_cat_names,
        params=params,
        val_start_idx=val_start_idx,
        val_end_idx=val_end_idx,
        k_runs=cfg.k_runs,
    )

    # Forecast next cfg.forecast_days from last observed day (T-1)
    forecast = simulate_forecast(
        wearing=wearing,
        neighbors=neighbors,
        risk=risk_perception,
        age_oh=age_oh,
        occ_oh=occ_oh,
        age_cat_names=age_cat_names,
        occ_cat_names=occ_cat_names,
        params=params,
        last_train_day_idx=T - 1,
        forecast_days=cfg.forecast_days,
        k_runs=cfg.k_runs,
    )

    # Save outputs
    save_json({"config": asdict(cfg)}, os.path.join(out_dir, "config.json"))
    save_json(fitted_params.to_dict(), os.path.join(out_dir, "calibrated_parameters.json"))
    save_json(metrics, os.path.join(out_dir, "validation_metrics.json"))
    save_json(forecast, os.path.join(out_dir, "forecast.json"))

    # Also save daily predicted rates for validation as CSV
    val_days = days[val_start_idx:val_end_idx]
    df_val = pd.DataFrame({
        "day": val_days,
        "observed_rate": metrics["observed_daily_rates"],
        "predicted_rate_mean": metrics["predicted_daily_rates_mean"],
        "predicted_rate_CI95": metrics["predicted_daily_rates_CI95"],
    })
    df_val.to_csv(os.path.join(out_dir, "validation_daily_rates.csv"), index=False)

    # Save forecast CSV
    df_fore = pd.DataFrame({
        "day": forecast["days"],
        "forecast_mean_rate": forecast["forecast_mean_rates"],
        "forecast_CI95": forecast["forecast_CI95"],
    })
    df_fore.to_csv(os.path.join(out_dir, "forecast_daily_rates.csv"), index=False)

    if cfg.verbose:
        print("Calibration and evaluation complete.")
        print(f"Saved outputs to {out_dir}")


# Execute main for both direct execution and sandbox wrapper invocation
if __name__ == "__main__":
    main()
else:
    # When imported as module
    main()