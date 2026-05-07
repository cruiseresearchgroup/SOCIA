import json
import math
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# ------------------------- Logging Configuration -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ------------------------- Utilities and Data Structures -------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


class NodeIndex:
    """Mapping between agent IDs and contiguous indices [0..N-1]."""
    def __init__(self, ids: List[int]):
        self.ids_sorted = sorted(set(ids))
        self.id_to_idx = {a: i for i, a in enumerate(self.ids_sorted)}
        self.idx_to_id = {i: a for i, a in enumerate(self.ids_sorted)}

    def __len__(self) -> int:
        return len(self.ids_sorted)

    def idx(self, agent_id: int) -> int:
        return self.id_to_idx[agent_id]

    def id(self, idx: int) -> int:
        return self.idx_to_id[idx]


@dataclass
class Params:
    # Adoption model parameters
    alpha: float = 0.0
    gamma: float = 0.0
    beta_f: float = 0.0
    beta_w: float = 0.0
    beta_c: float = 0.0
    beta_r: float = 0.0
    beta_i: float = 0.0
    age_effects: Dict[str, float] = field(default_factory=dict)   # exclude baseline
    occ_effects: Dict[str, float] = field(default_factory=dict)   # exclude baseline
    tau: float = 1.0
    reg_l2: float = 1.0

    # Info diffusion parameters
    phi_family: float = 0.1
    phi_work: float = 0.1
    phi_community: float = 0.05
    lambda_broadcast: float = 0.02
    rho_info_decay: float = 0.8

    # Derived layer weights from peer coefficients (normalized absolute values)
    @property
    def w_family(self) -> float:
        vals = np.array([abs(self.beta_f), abs(self.beta_w), abs(self.beta_c)]) + 1e-9
        return float(vals[0] / vals.sum())

    @property
    def w_work(self) -> float:
        vals = np.array([abs(self.beta_f), abs(self.beta_w), abs(self.beta_c)]) + 1e-9
        return float(vals[1] / vals.sum())

    @property
    def w_community(self) -> float:
        vals = np.array([abs(self.beta_f), abs(self.beta_w), abs(self.beta_c)]) + 1e-9
        return float(vals[2] / vals.sum())


# ------------------------- Data Loading and Validation -------------------------

REQUIRED_AGENT_COLUMNS = [
    "agent_id",
    "age",
    "age_group",
    "occupation",
    "risk_perception",
    "initial_mask_wearing",
    "family_connections",
    "work_school_connections",
    "community_connections",
    "total_connections",
]

REQUIRED_TRAIN_COLUMNS = [
    "day",
    "agent_id",
    "wearing_mask",
    "received_info",
]


def load_agent_attributes(path: str) -> pd.DataFrame:
    """Load and validate agent attributes CSV."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"Failed to read agent attributes from {path}: {e}")
        raise

    missing = [c for c in REQUIRED_AGENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"agent_attributes.csv missing columns: {missing}")

    # Basic type enforcement
    df = df.drop_duplicates(subset=["agent_id"]).copy()
    df["agent_id"] = df["agent_id"].astype(int)
    df["risk_perception"] = df["risk_perception"].astype(float).clip(0.0, 1.0)
    df["initial_mask_wearing"] = df["initial_mask_wearing"].astype(bool)
    # Ensure categorical consistency
    df["age_group"] = df["age_group"].astype(str)
    df["occupation"] = df["occupation"].astype(str)
    return df


def load_social_network(path: str) -> Dict[str, Dict[str, List[int]]]:
    """Load social network JSON."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read social network from {path}: {e}")
        raise
    # Expecting a dict keyed by agent_id as strings
    if not isinstance(data, dict):
        raise ValueError("social_network.json root must be a JSON object (dict).")
    return data


def load_train_data(path: str) -> pd.DataFrame:
    """Load and validate training time series CSV."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"Failed to read train data from {path}: {e}")
        raise
    missing = [c for c in REQUIRED_TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"train_data.csv missing columns: {missing}")
    df["day"] = df["day"].astype(int)
    df["agent_id"] = df["agent_id"].astype(int)
    df["wearing_mask"] = df["wearing_mask"].astype(bool)
    df["received_info"] = df["received_info"].astype(bool)
    return df


# ------------------------- Network Preprocessing -------------------------

def build_multiplex_network(
    net_raw: Dict[str, Dict[str, List[int]]],
    node_index: NodeIndex,
    layers: Tuple[str, str, str] = ("family", "work_school", "community"),
) -> Dict[str, List[np.ndarray]]:
    """
    Build symmetrized, deduplicated adjacency lists per layer aligned to NodeIndex.
    Returns a dict from layer -> list of numpy arrays of neighbor indices for each node idx.
    """
    valid_ids = set(node_index.ids_sorted)
    # Initialize adjacency sets for symmetry and deduplication
    adj_sets: Dict[str, List[set]] = {layer: [set() for _ in range(len(node_index))] for layer in layers}

    # Add edges from raw data
    for key_str, nbrs in net_raw.items():
        try:
            i = int(key_str)
        except ValueError:
            continue
        if i not in valid_ids:
            continue
        i_idx = node_index.idx(i)
        for layer in layers:
            neigh_list = nbrs.get(layer, []) or []
            for j in neigh_list:
                if j in valid_ids:
                    j_idx = node_index.idx(j)
                    adj_sets[layer][i_idx].add(j_idx)

    # Enforce symmetry
    for layer in layers:
        sets = adj_sets[layer]
        for i in range(len(sets)):
            for j in list(sets[i]):
                sets[j].add(i)

    # Convert to numpy arrays
    adj: Dict[str, List[np.ndarray]] = {}
    for layer in layers:
        adj[layer] = [np.fromiter(sorted(s), dtype=np.int64) if len(s) > 0 else np.empty(0, dtype=np.int64)
                      for s in adj_sets[layer]]

    return adj


def compute_neighbor_share(mask_vec: np.ndarray, neighbors: List[np.ndarray]) -> np.ndarray:
    """
    Compute fraction of neighbors wearing mask for each agent.
    mask_vec: shape (N,), neighbors: list length N with arrays of neighbor indices.
    """
    N = mask_vec.shape[0]
    shares = np.zeros(N, dtype=np.float64)
    for i in range(N):
        nbrs = neighbors[i]
        deg = nbrs.size
        if deg == 0:
            shares[i] = 0.0
        else:
            shares[i] = float(mask_vec[nbrs].sum()) / float(deg)
    return shares


# ------------------------- Feature Engineering -------------------------

def one_hot_per_agent(
    series: pd.Series,
    baseline: Optional[str] = None
) -> Tuple[np.ndarray, List[str], str]:
    """
    Create one-hot encoding per agent for a categorical series aligned to NodeIndex order.
    Returns (matrix NxKminus1), column names, and baseline category.
    """
    cats = series.astype(str)
    # Choose baseline
    if baseline is None:
        baseline = cats.value_counts().idxmax()
    uniq = [c for c in sorted(cats.unique()) if c != baseline]
    N = cats.shape[0]
    mat = np.zeros((N, len(uniq)), dtype=np.float64)
    colnames = []
    cat_to_idx = {c: i for i, c in enumerate(uniq)}
    for i, c in enumerate(cats.tolist()):
        if c == baseline:
            continue
        j = cat_to_idx[c]
        mat[i, j] = 1.0
    for c in uniq:
        colnames.append(f"{series.name}={c}")
    return mat, colnames, baseline


def build_design_matrices(
    agent_df: pd.DataFrame,
    node_index: NodeIndex,
    train_mask: np.ndarray,
    train_info: np.ndarray,
    shares_f: np.ndarray,
    shares_w: np.ndarray,
    shares_c: np.ndarray,
    rho_candidates: List[float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build adoption model design matrices for t=1..T-1 using different rho candidates for info memory.
    Returns:
        X_by_rho: dict of rho -> X matrix (M x P)
        y: vector of outcomes (M,)
        meta: dict with feature names and shapes (age/occ dims, etc.)
    """
    T, N = train_mask.shape
    # Prepare static features per agent
    # Align agent_df to NodeIndex order
    agent_df_sorted = agent_df.set_index("agent_id").loc[node_index.ids_sorted].reset_index(drop=False)

    risk = agent_df_sorted["risk_perception"].to_numpy(dtype=np.float64)
    age_mat, age_cols, age_baseline = one_hot_per_agent(agent_df_sorted["age_group"])
    occ_mat, occ_cols, occ_baseline = one_hot_per_agent(agent_df_sorted["occupation"])

    # Build dependent variable y and base features (without memory)
    rows = []
    # We'll pre-compute prev_wear and shares for each (t, i)
    for t in range(1, T):
        prev = train_mask[t - 1, :].astype(np.float64)
        y_t = train_mask[t, :].astype(np.float64)
        sf = shares_f[t, :]
        sw = shares_w[t, :]
        sc = shares_c[t, :]
        # For each i, we will append row data; build arrays for x components
        # We'll assemble matrices by stacking later for performance
        # For now, accumulate in lists then stack to arrays
        rows.append({
            "prev": prev,
            "sf": sf,
            "sw": sw,
            "sc": sc,
            "y": y_t,
            "day_index": np.full(N, t, dtype=np.int64)
        })
    # Concatenate
    prev_all = np.concatenate([r["prev"] for r in rows], axis=0)
    sf_all = np.concatenate([r["sf"] for r in rows], axis=0)
    sw_all = np.concatenate([r["sw"] for r in rows], axis=0)
    sc_all = np.concatenate([r["sc"] for r in rows], axis=0)
    y = np.concatenate([r["y"] for r in rows], axis=0)
    # Broadcast static features over days
    risk_all = np.tile(risk, T - 1)
    age_all = np.tile(age_mat, (T - 1, 1))
    occ_all = np.tile(occ_mat, (T - 1, 1))
    # For info memory, compute per rho candidate
    X_by_rho: Dict[float, np.ndarray] = {}
    # Build received_info matrix in same structure: for t=1..T-1, info[t]
    info_flat = np.concatenate([train_info[t, :].astype(np.float64) for t in range(1, T)], axis=0)
    # For memory, we need memory per (t, i): memory[t] depends on previous days
    # We'll compute full memory tensor for each rho, then flatten over t=1..T-1
    for rho in rho_candidates:
        memory = np.zeros_like(train_info, dtype=np.float64)
        memory[0, :] = train_info[0, :].astype(np.float64)
        for t in range(1, T):
            memory[t, :] = rho * memory[t - 1, :] + train_info[t, :].astype(np.float64)
            memory[t, :] = np.clip(memory[t, :], 0.0, 1.0)
        mem_all = np.concatenate([memory[t, :] for t in range(1, T)], axis=0)
        # Assemble X: [1, prev_wear, sf, sw, sc, risk, mem, age dummies..., occ dummies...]
        intercept = np.ones_like(prev_all, dtype=np.float64)
        X_parts = [
            intercept,
            prev_all,
            sf_all,
            sw_all,
            sc_all,
            risk_all,
            mem_all,
            age_all,
            occ_all
        ]
        X = np.column_stack([p if p.ndim == 1 else p for p in X_parts])
        X_by_rho[rho] = X

    meta = {
        "feature_names": ["intercept", "prev", "share_family", "share_work", "share_community", "risk", "info_memory"]
                         + age_cols + occ_cols,
        "age_cols": age_cols,
        "occ_cols": occ_cols,
        "age_baseline": age_baseline,
        "occ_baseline": occ_baseline,
        "T": T,
        "N": N,
    }
    return X_by_rho, y, meta


# ------------------------- Calibration: Info Diffusion -------------------------

def info_loss_and_grad(
    v: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Info model: p = 1 - exp(-X v), with v >= 0. Loss is NLL: -sum(y log p + (1 - y) log(1 - p)) + (l2/2)||v||^2
    Returns (loss, grad).
    """
    # Enforce positivity inside function (softplus or clipping) for stability in gradient
    # We'll compute with v_clipped but return gradient w.r.t v (using straight-through estimator for clipping).
    v_clipped = np.clip(v, 0.0, None)
    z = X.dot(v_clipped)
    z = np.clip(z, 1e-12, 100.0)  # ensure positivity for exp
    p = 1.0 - np.exp(-z)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    # NLL
    nll = - (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).sum()
    # Gradient: sum_i [(1 - y/p) x *? Wait earlier derivation -> (1 - y/p) * x?
    # Derived: dL/dv = sum_i [ (1 - y_i/p_i) * (1 - p_i) * x_i ]? Re-derive carefully...
    # We'll use the simplified exact form: dL/dv = X^T * ((p - y) / p)
    # Actually, correct form: dL/dv = X^T * (1 - y/p) * (1 - p)  -- This is inconsistent with prior.
    # Use the rigorous derivation: L' = (1 - y/p) * dp/dv, and dp/dv = (1 - p) X.
    # So gradient per sample: (1 - y/p) * (1 - p) * x
    # Vectorize:
    grad_factor = (1.0 - y / p) * (1.0 - p)
    grad = X.T.dot(grad_factor)
    # Add L2
    if l2 > 0.0:
        nll += 0.5 * l2 * float((v_clipped ** 2).sum())
        grad += l2 * v_clipped
    return float(nll), grad


def optimize_info_params(
    X: np.ndarray,
    y: np.ndarray,
    bounds: List[Tuple[float, float]],
    l2: float = 0.0,
    max_iter: int = 500,
    lr: float = 0.05,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Gradient descent with projection for info parameters.
    """
    rng = np.random.default_rng(42)
    v = np.array([rng.uniform(low=b[0], high=max(b[0], min(b[1], 0.1))) for b in bounds], dtype=np.float64)
    prev_loss = np.inf
    for it in range(max_iter):
        loss, grad = info_loss_and_grad(v, X, y, l2=l2)
        v_new = v - lr * grad
        # Project to bounds
        for j, (low, high) in enumerate(bounds):
            v_new[j] = float(np.clip(v_new[j], low, high))
        # Check improvement with simple backtracking
        for _ in range(5):
            loss_new, _ = info_loss_and_grad(v_new, X, y, l2=l2)
            if loss_new <= loss:
                break
            v_new = v - (lr * 0.5) * grad
            for j, (low, high) in enumerate(bounds):
                v_new[j] = float(np.clip(v_new[j], low, high))
        v = v_new
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return v


# ------------------------- Calibration: Logistic Regression -------------------------

def logistic_loss_and_grad(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 0.0,
    intercept_penalty: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    Logistic regression NLL with L2 regularization.
    """
    z = X.dot(w)
    p = sigmoid(z)
    eps = 1e-9
    nll = - (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)).sum()
    # Gradient: X^T (p - y)
    grad = X.T.dot(p - y)
    if l2 > 0.0:
        if intercept_penalty:
            nll += 0.5 * l2 * float((w ** 2).sum())
            grad += l2 * w
        else:
            # do not penalize intercept
            w_reg = w.copy()
            w_reg[0] = 0.0
            nll += 0.5 * l2 * float((w_reg ** 2).sum())
            grad += l2 * w_reg
    return float(nll), grad


def fit_logistic_adam(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    max_iter: int = 1000,
    lr: float = 0.05,
    tol: float = 1e-6,
    intercept_penalty: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Fit logistic regression with Adam optimizer.
    Returns (weights, final_loss).
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.01, size=X.shape[1]).astype(np.float64)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    prev_loss = np.inf
    for t in range(1, max_iter + 1):
        loss, grad = logistic_loss_and_grad(w, X, y, l2=l2, intercept_penalty=intercept_penalty)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return w, prev_loss


# ------------------------- Simulation and Evaluation -------------------------

def compute_info_prob(
    share_f: np.ndarray,
    share_w: np.ndarray,
    share_c: np.ndarray,
    day: int,
    phi_f: float,
    phi_w: float,
    phi_c: float,
    lambda_b: float,
    broadcast_start: int = 10,
) -> np.ndarray:
    """
    Compute probability of receiving info at given day using multiplex peer exposure and broadcast.
    """
    z = phi_f * share_f + phi_w * share_w + phi_c * share_c
    if day >= broadcast_start:
        z = z + lambda_b
    z = np.clip(z, 0.0, 100.0)
    p = 1.0 - np.exp(-z)
    return np.clip(p, 0.0, 1.0)


def run_simulation(
    params: Params,
    adj: Dict[str, List[np.ndarray]],
    node_index: NodeIndex,
    agent_df: pd.DataFrame,
    init_mask: np.ndarray,
    init_memory: np.ndarray,
    start_day: int,
    end_day: int,
    broadcast_start: int = 10,
    K: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run K stochastic simulations from start_day to end_day inclusive.
    Returns dict with:
      - "daily_rate_mean": array of length L with mean adoption rate across K runs
      - "daily_rate_std": std across K runs
      - "per_run_daily_rates": K x L array
      - "per_day_prob_mean": L array of mean predicted p across agents and runs
    """
    rng = np.random.default_rng(seed)
    N = len(node_index)
    # Prepare static features
    agent_df_sorted = agent_df.set_index("agent_id").loc[node_index.ids_sorted].reset_index(drop=False)
    risk = agent_df_sorted["risk_perception"].to_numpy(dtype=np.float64)
    age_mat, age_cols, age_baseline = one_hot_per_agent(agent_df_sorted["age_group"])
    occ_mat, occ_cols, occ_baseline = one_hot_per_agent(agent_df_sorted["occupation"])

    L = end_day - start_day + 1
    per_run_daily_rates = np.zeros((K, L), dtype=np.float64)
    per_day_prob_mean = np.zeros(L, dtype=np.float64)

    for k in range(K):
        mask = init_mask.copy().astype(np.float64)
        memory = init_memory.copy().astype(np.float64)
        daily_rates = []
        per_day_prob = []
        for di, day in enumerate(range(start_day, end_day + 1)):
            # Neighbor shares from previous day's mask
            sf = compute_neighbor_share(mask, adj["family"])
            sw = compute_neighbor_share(mask, adj["work_school"])
            sc = compute_neighbor_share(mask, adj["community"])

            # Info probability and sampling
            p_info = compute_info_prob(
                sf, sw, sc, day,
                params.phi_family, params.phi_work, params.phi_community,
                params.lambda_broadcast,
                broadcast_start=broadcast_start
            )
            info_realization = (rng.random(N) < p_info).astype(np.float64)
            memory = np.clip(params.rho_info_decay * memory + info_realization, 0.0, 1.0)

            # Build adoption probabilities
            intercept = np.ones(N, dtype=np.float64)
            prev = mask  # previous day's state
            # Linear predictor
            z = (
                params.alpha * intercept
                + params.gamma * prev
                + params.beta_f * sf
                + params.beta_w * sw
                + params.beta_c * sc
                + params.beta_r * risk
                + params.beta_i * memory
            )
            # Add demographic effects
            if len(params.age_effects) > 0:
                for j, col in enumerate(age_cols):
                    beta = params.age_effects.get(col, 0.0)
                    if beta != 0.0:
                        z += beta * age_mat[:, j]
            if len(params.occ_effects) > 0:
                for j, col in enumerate(occ_cols):
                    beta = params.occ_effects.get(col, 0.0)
                    if beta != 0.0:
                        z += beta * occ_mat[:, j]
            # Tau scaling
            if params.tau != 1.0 and params.tau > 0.0:
                z = z / params.tau

            p_wear = sigmoid(z)
            # Sample next state
            wear_realization = (rng.random(N) < p_wear).astype(np.float64)

            mask = wear_realization
            daily_rates.append(mask.mean())
            per_day_prob.append(float(p_wear.mean()))
        per_run_daily_rates[k, :] = np.array(daily_rates)
        per_day_prob_mean += np.array(per_day_prob)

    per_day_prob_mean /= float(K)
    return {
        "daily_rate_mean": per_run_daily_rates.mean(axis=0),
        "daily_rate_std": per_run_daily_rates.std(axis=0),
        "per_run_daily_rates": per_run_daily_rates,
        "per_day_prob_mean": per_day_prob_mean,
        "age_cols": age_cols,
        "occ_cols": occ_cols,
        "age_baseline": age_baseline,
        "occ_baseline": occ_baseline,
    }


def compute_aggregate_metrics(
    pred_rates: np.ndarray,
    obs_rates: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Compute aggregate metrics if obs_rates provided.
    """
    metrics = {}
    if obs_rates is None:
        return metrics
    if len(pred_rates) != len(obs_rates):
        logging.warning("Length mismatch: predicted vs observed daily rates. Skipping metrics.")
        return metrics
    diff = pred_rates - obs_rates
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    metrics["RMSE_aggregate"] = rmse
    metrics["MAE_aggregate"] = mae
    return metrics


# ------------------------- Main Pipeline -------------------------

def main():
    """
    End-to-end pipeline:
      - Load data
      - Preprocess network
      - Build training matrices
      - Calibrate info diffusion and adoption parameters
      - Simulate holdout days (30-39)
      - Evaluate if observations available
      - Print outputs
    """
    # Configuration
    data_folder = os.environ.get("DATA_FOLDER", "data_fitting/mask_adoption_data")
    agent_path = os.path.join(data_folder, "agent_attributes.csv")
    network_path = os.path.join(data_folder, "social_network.json")
    train_path = os.path.join(data_folder, "train_data.csv")
    broadcast_start_day = 10
    # Holdout days
    holdout_start = 30
    holdout_end = 39
    # Random seed and simulation replicates
    seed = int(os.environ.get("SEED", "42"))
    K_runs = int(os.environ.get("K_SIM", "10"))
    # Regularization strengths
    l2_info = float(os.environ.get("L2_INFO", "0.0"))
    l2_logit = float(os.environ.get("L2_LOGIT", "1.0"))

    try:
        agent_df = load_agent_attributes(agent_path)
        net_raw = load_social_network(network_path)
        train_df = load_train_data(train_path)
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return

    # Align IDs across files
    ids_agents = set(agent_df["agent_id"].astype(int).tolist())
    ids_net = set(int(k) for k in net_raw.keys() if k.isdigit())
    ids_train = set(train_df["agent_id"].astype(int).unique().tolist())
    common_ids = list(ids_agents & ids_net & ids_train)
    if not common_ids:
        logging.error("No common agent IDs across datasets.")
        return

    node_index = NodeIndex(common_ids)
    N = len(node_index)
    logging.info(f"Aligned population size: {N}")

    # Filter dataframes to aligned IDs
    agent_df = agent_df[agent_df["agent_id"].isin(node_index.ids_sorted)].copy()
    train_df = train_df[train_df["agent_id"].isin(node_index.ids_sorted)].copy()

    # Build network
    adj = build_multiplex_network(net_raw, node_index, layers=("family", "work_school", "community"))

    # Build train matrices (days x N)
    # Determine day range
    T_max = int(train_df["day"].max())
    T = T_max + 1
    # Pivot into arrays
    # Ensure full grid
    df_sorted = train_df.sort_values(["day", "agent_id"]).copy()
    # Build matrices
    mask_mat = np.zeros((T, N), dtype=np.float64)
    info_mat = np.zeros((T, N), dtype=np.float64)
    # Create mapping from id->idx
    id_to_idx = {a: node_index.idx(a) for a in node_index.ids_sorted}
    for _, row in df_sorted.iterrows():
        t = int(row["day"])
        if t < 0 or t >= T:
            continue
        i_idx = id_to_idx[int(row["agent_id"])]
        mask_mat[t, i_idx] = 1.0 if bool(row["wearing_mask"]) else 0.0
        info_mat[t, i_idx] = 1.0 if bool(row["received_info"]) else 0.0

    # Neighbor shares per day using previous day's mask
    shares_f = np.zeros_like(mask_mat)
    shares_w = np.zeros_like(mask_mat)
    shares_c = np.zeros_like(mask_mat)
    for t in range(1, T):
        prev = mask_mat[t - 1, :]
        shares_f[t, :] = compute_neighbor_share(prev, adj["family"])
        shares_w[t, :] = compute_neighbor_share(prev, adj["work_school"])
        shares_c[t, :] = compute_neighbor_share(prev, adj["community"])
    # t=0 shares left as zeros

    # Calibrate info diffusion parameters using observed received_info if available for t >= 1
    X_info_rows = []
    y_info_rows = []
    for t in range(1, T):
        # Features: [share_f, share_w, share_c, I_broadcast]
        I_b = 1.0 if t >= broadcast_start_day else 0.0
        X_block = np.column_stack([
            shares_f[t, :],
            shares_w[t, :],
            shares_c[t, :],
            np.full(N, I_b, dtype=np.float64),
        ])
        y_block = info_mat[t, :]
        X_info_rows.append(X_block)
        y_info_rows.append(y_block)
    X_info = np.vstack(X_info_rows) if X_info_rows else np.zeros((0, 4), dtype=np.float64)
    y_info = np.concatenate(y_info_rows) if y_info_rows else np.zeros((0,), dtype=np.float64)

    # Optimize phi and lambda_broadcast with bounds
    # bounds: phi in [0,2], lambda in [0,0.5]
    bounds = [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 0.5)]
    if X_info.shape[0] > 0:
        v_opt = optimize_info_params(X_info, y_info, bounds=bounds, l2=l2_info, max_iter=500, lr=0.1, tol=1e-7)
        phi_f, phi_w, phi_c, lambda_b = float(v_opt[0]), float(v_opt[1]), float(v_opt[2]), float(v_opt[3])
    else:
        # Fallback defaults
        phi_f, phi_w, phi_c, lambda_b = 0.1, 0.1, 0.05, 0.02

    logging.info(f"Calibrated info params: phi_f={phi_f:.4f}, phi_w={phi_w:.4f}, phi_c={phi_c:.4f}, lambda_b={lambda_b:.4f}")

    # Build adoption design matrices for a set of rho candidates
    rho_candidates = [0.0, 0.3, 0.5, 0.75, 0.9]
    X_by_rho, y_adopt, meta = build_design_matrices(agent_df, node_index, mask_mat, info_mat, shares_f, shares_w, shares_c, rho_candidates)

    # Fit logistic over rho candidates, choose best by NLL
    best_w = None
    best_loss = np.inf
    best_rho = rho_candidates[0]
    for rho, X in X_by_rho.items():
        w, final_loss = fit_logistic_adam(X, y_adopt, l2=l2_logit, max_iter=800, lr=0.05, tol=1e-7, intercept_penalty=False, seed=seed)
        if final_loss < best_loss:
            best_loss = final_loss
            best_w = w
            best_rho = rho

    logging.info(f"Selected rho_info_decay={best_rho:.3f} with training NLL={best_loss:.3f}")

    # Decode coefficients into Params
    # Feature order: intercept, prev, share_family, share_work, share_community, risk, info_memory, age_cols..., occ_cols...
    feature_names = meta["feature_names"]
    age_cols = meta["age_cols"]
    occ_cols = meta["occ_cols"]
    # Map names to indices
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    # Extract
    alpha = float(best_w[name_to_idx["intercept"]])
    gamma = float(best_w[name_to_idx["prev"]])
    beta_f = float(best_w[name_to_idx["share_family"]])
    beta_w = float(best_w[name_to_idx["share_work"]])
    beta_c = float(best_w[name_to_idx["share_community"]])
    beta_r = float(best_w[name_to_idx["risk"]])
    beta_i = float(best_w[name_to_idx["info_memory"]])
    # Demographic effects
    age_effects = {col: float(best_w[name_to_idx[col]]) for col in age_cols}
    occ_effects = {col: float(best_w[name_to_idx[col]]) for col in occ_cols}

    params = Params(
        alpha=alpha,
        gamma=gamma,
        beta_f=beta_f,
        beta_w=beta_w,
        beta_c=beta_c,
        beta_r=beta_r,
        beta_i=beta_i,
        age_effects=age_effects,
        occ_effects=occ_effects,
        tau=1.0,
        reg_l2=l2_logit,
        phi_family=phi_f,
        phi_work=phi_w,
        phi_community=phi_c,
        lambda_broadcast=lambda_b,
        rho_info_decay=best_rho
    )

    # Prepare initialization for simulation from last observed train day (T-1)
    init_mask = mask_mat[T - 1, :].copy()
    # Build memory aligned to training with best_rho
    memory = np.zeros_like(info_mat, dtype=np.float64)
    memory[0, :] = info_mat[0, :]
    for t in range(1, T):
        memory[t, :] = np.clip(params.rho_info_decay * memory[t - 1, :] + info_mat[t, :], 0.0, 1.0)
    init_memory = memory[T - 1, :].copy()

    # Run simulation for holdout window
    sim_res = run_simulation(
        params=params,
        adj=adj,
        node_index=node_index,
        agent_df=agent_df,
        init_mask=init_mask,
        init_memory=init_memory,
        start_day=holdout_start,
        end_day=holdout_end,
        broadcast_start=broadcast_start_day,
        K=K_runs,
        seed=seed
    )

    # Compute observed daily rates on holdout if available
    unique_days = sorted(train_df["day"].unique().tolist())
    if holdout_start in unique_days and holdout_end in unique_days:
        # Build obs rates for holdout
        L = holdout_end - holdout_start + 1
        obs_rates = np.zeros(L, dtype=np.float64)
        for di, day in enumerate(range(holdout_start, holdout_end + 1)):
            df_day = train_df[train_df["day"] == day]
            if df_day.empty:
                obs_rates[di] = np.nan
            else:
                obs_rates[di] = df_day["wearing_mask"].astype(bool).mean()
        if np.any(np.isnan(obs_rates)):
            obs_rates = None
    else:
        obs_rates = None

    metrics = compute_aggregate_metrics(sim_res["daily_rate_mean"], obs_rates)

    # Print summary
    print("Calibrated Parameters:")
    print(f"  alpha={params.alpha:.4f}, gamma={params.gamma:.4f}")
    print(f"  beta_f={params.beta_f:.4f}, beta_w={params.beta_w:.4f}, beta_c={params.beta_c:.4f}")
    print(f"  beta_r={params.beta_r:.4f}, beta_i={params.beta_i:.4f}")
    print(f"  w_family={params.w_family:.3f}, w_work={params.w_work:.3f}, w_community={params.w_community:.3f}")
    print(f"  age_effects: {params.age_effects}")
    print(f"  occ_effects: {params.occ_effects}")
    print(f"  Info params: phi_f={params.phi_family:.4f}, phi_w={params.phi_work:.4f}, phi_c={params.phi_community:.4f}, lambda_broadcast={params.lambda_broadcast:.4f}, rho_info_decay={params.rho_info_decay:.3f}")
    print()

    print(f"Forecasted aggregate mask-wearing rates (mean across {K_runs} runs) for days {holdout_start}-{holdout_end}:")
    for di, day in enumerate(range(holdout_start, holdout_end + 1)):
        rate = sim_res["daily_rate_mean"][di]
        std = sim_res["daily_rate_std"][di]
        print(f"  Day {day}: mean={rate:.4f}, std={std:.4f}")

    if metrics:
        print("\nValidation Metrics on holdout:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("\nNo holdout observations available for evaluation. Metrics not computed.")


# Execute main for both direct execution and sandbox wrapper invocation

# Execute main for both direct execution and sandbox wrapper invocation
main()