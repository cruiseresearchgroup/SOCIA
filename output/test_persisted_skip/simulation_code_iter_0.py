def main():
    pass

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd


@dataclass
class Agent:
    agent_id: int
    age: int
    age_group: str
    occupation: str
    risk_perception: float
    initial_mask_wearing: bool
    family_connections: int
    work_school_connections: int
    community_connections: int
    total_connections: int
    role: str = field(init=False)

    def __post_init__(self) -> None:
        self.role = derive_role(self.occupation, self.age_group)


def derive_role(occupation: str, age_group: str) -> str:
    occ = str(occupation).strip().lower()
    ag = str(age_group).strip().lower()
    if occ == "student":
        return "Student"
    if occ == "blue collar":
        return "Worker_BlueCollar"
    if occ == "white collar":
        return "Worker_WhiteCollar"
    # Youth or Elderly without occupation fall into Nonworker
    if ag in {"youth", "elderly"}:
        return "Nonworker"
    return "Nonworker"


class DataLoader:
    def __init__(self, data_folder: str) -> None:
        self.data_folder = data_folder

    def _path(self, name: str) -> str:
        return os.path.join(self.data_folder, name)

    def load_agent_attributes(self, filename: str = "agent_attributes.csv") -> pd.DataFrame:
        path = self._path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise IOError(f"Failed to read {path}: {e}")
        required_cols = [
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
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"agent_attributes.csv missing columns: {missing}")
        # Type enforcement and validation
        if df["agent_id"].duplicated().any():
            dupes = df[df["agent_id"].duplicated()]["agent_id"].tolist()
            raise ValueError(f"Duplicate agent_id in agent_attributes.csv: {dupes[:5]}...")
        if not ((df["risk_perception"] >= 0) & (df["risk_perception"] <= 1)).all():
            raise ValueError("risk_perception must be within [0,1]")
        if not df["initial_mask_wearing"].isin([True, False]).all():
            # Cast to bool if represented as 0/1 or strings
            df["initial_mask_wearing"] = df["initial_mask_wearing"].astype(bool)
        # Check degree consistency
        deg_sum = (
            df["family_connections"] + df["work_school_connections"] + df["community_connections"]
        )
        inconsist = (deg_sum != df["total_connections"])
        if inconsist.any():
            # Fix if obviously a rounding or entry issue by recomputing total
            df.loc[inconsist, "total_connections"] = deg_sum[inconsist]
        return df

    def load_social_network(self, filename: str = "social_network.json") -> Dict[int, Dict[str, List[int]]]:
        path = self._path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except Exception as e:
            raise IOError(f"Failed to read {path}: {e}")
        if not isinstance(raw, dict):
            raise ValueError("social_network.json should be a JSON object keyed by agent IDs")
        # Convert keys to ints
        net: Dict[int, Dict[str, List[int]]] = {}
        for k, v in raw.items():
            try:
                i = int(k)
            except Exception:
                raise ValueError(f"Invalid agent id key in social_network.json: {k}")
            # Ensure layer lists exist
            fam = list(map(int, v.get("family", [])))
            work = list(map(int, v.get("work_school", [])))
            comm = list(map(int, v.get("community", [])))
            all_layer = list(map(int, v.get("all", [])))
            net[i] = {"family": fam, "work_school": work, "community": comm, "all": all_layer}
        return net

    def load_train_panel(self, filename: str = "train_data.csv") -> pd.DataFrame:
        path = self._path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise IOError(f"Failed to read {path}: {e}")
        required_cols = ["day", "agent_id", "wearing_mask", "received_info"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"train_data.csv missing columns: {missing}")
        # Cast types
        df["day"] = df["day"].astype(int)
        df["agent_id"] = df["agent_id"].astype(int)
        df["wearing_mask"] = df["wearing_mask"].astype(bool)
        df["received_info"] = df["received_info"].astype(bool)
        # Validate uniqueness
        if df.duplicated(subset=["agent_id", "day"]).any():
            raise ValueError("Duplicate (agent_id, day) pairs in train_data.csv")
        return df


class NetworkBuilder:
    def __init__(self, N: int) -> None:
        self.N = N
        self.layers = ["family", "work_school", "community", "all"]
        self.adj: Dict[str, List[List[int]]] = {layer: [[] for _ in range(N)] for layer in self.layers}

    @staticmethod
    def _dedup_list(int_list: Iterable[int]) -> List[int]:
        seen = set()
        res = []
        for x in int_list:
            if x not in seen:
                seen.add(x)
                res.append(x)
        return res

    def from_json_adjacency(self, raw_net: Dict[int, Dict[str, List[int]]]) -> "NetworkBuilder":
        # Fill adjacency lists
        for i in range(self.N):
            entry = raw_net.get(i, {"family": [], "work_school": [], "community": [], "all": []})
            fam = self._dedup_list(entry.get("family", []))
            work = self._dedup_list(entry.get("work_school", []))
            comm = self._dedup_list(entry.get("community", []))
            # We'll recompute 'all' as union
            self.adj["family"][i] = [n for n in fam if 0 <= n < self.N and n != i]
            self.adj["work_school"][i] = [n for n in work if 0 <= n < self.N and n != i]
            self.adj["community"][i] = [n for n in comm if 0 <= n < self.N and n != i]
        # Enforce symmetry for undirected graph per layer
        for layer in ["family", "work_school", "community"]:
            for i in range(self.N):
                for j in list(self.adj[layer][i]):
                    if i not in self.adj[layer][j]:
                        self.adj[layer][j].append(i)
            # Dedup and remove self, enforce bounds
            for i in range(self.N):
                self.adj[layer][i] = self._dedup_list([n for n in self.adj[layer][i] if 0 <= n < self.N and n != i])
        # Compute 'all' layer as union
        for i in range(self.N):
            union = self.adj["family"][i] + self.adj["work_school"][i] + self.adj["community"][i]
            self.adj["all"][i] = self._dedup_list(union)
        return self

    def degrees(self) -> Dict[str, np.ndarray]:
        return {layer: np.array([len(self.adj[layer][i]) for i in range(self.N)], dtype=np.int32) for layer in self.layers}

    def neighbor_mask_shares(self, masks: np.ndarray) -> Dict[str, np.ndarray]:
        shares: Dict[str, np.ndarray] = {}
        mask_int = masks.astype(np.int8)
        for layer in ["family", "work_school", "community"]:
            out = np.zeros(self.N, dtype=np.float32)
            deg = np.zeros(self.N, dtype=np.int32)
            for i in range(self.N):
                neigh = self.adj[layer][i]
                d = len(neigh)
                deg[i] = d
                if d == 0:
                    out[i] = 0.0
                else:
                    s = 0
                    # Sum masks of neighbors
                    for j in neigh:
                        s += int(mask_int[j])
                    out[i] = s / d if d > 0 else 0.0
            shares[layer] = out
        return shares

    def diagnostics(self) -> Dict[str, Dict[str, float]]:
        degs = self.degrees()
        stats = {}
        for layer in ["family", "work_school", "community", "all"]:
            d = degs[layer]
            stats[layer] = {
                "min_degree": float(d.min() if len(d) else 0.0),
                "max_degree": float(d.max() if len(d) else 0.0),
                "mean_degree": float(d.mean() if len(d) else 0.0),
            }
        # Reciprocity check (undirected enforced; reciprocity ~ 1)
        reciprocity = {}
        for layer in ["family", "work_school", "community"]:
            total = 0
            reciprocated = 0
            for i in range(self.N):
                for j in self.adj[layer][i]:
                    total += 1
                    if i in self.adj[layer][j]:
                        reciprocated += 1
            reciprocity[layer] = float(reciprocated / total) if total > 0 else 1.0
        stats["reciprocity"] = reciprocity
        return stats


class InfoDiffusion:
    def __init__(
        self,
        network: NetworkBuilder,
        beta_family: float = 0.1,
        beta_work_school: float = 0.1,
        beta_community: float = 0.05,
        info_rate_broadcast: float = 0.05,
        info_decay: int = 3,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.network = network
        self.beta = {
            "family": float(np.clip(beta_family, 0.0, 1.0)),
            "work_school": float(np.clip(beta_work_school, 0.0, 1.0)),
            "community": float(np.clip(beta_community, 0.0, 1.0)),
        }
        self.info_rate_broadcast = float(np.clip(info_rate_broadcast, 0.0, 1.0))
        self.info_decay = max(1, int(info_decay))
        self.rng = rng if rng is not None else random.Random(42)
        self.N = self.network.N
        # Memory of last exposures for decay
        self.memory = np.zeros(self.N, dtype=np.int16)

    def reset(self) -> None:
        self.memory[:] = 0

    def step(
        self,
        informed_prev: np.ndarray,
        mandate_strength: float,
    ) -> np.ndarray:
        # Base broadcast
        broadcast = np.array([self.rng.random() < self.info_rate_broadcast for _ in range(self.N)], dtype=bool)
        # Mandate: everyone gets info with probability proportional to mandate_strength
        mandate_hits = np.array([self.rng.random() < float(np.clip(mandate_strength, 0.0, 1.0)) for _ in range(self.N)], dtype=bool)
        # Neighbor transmission
        received = np.zeros(self.N, dtype=bool)
        informed_set = np.where(informed_prev)[0].tolist()
        informed_mask = informed_prev.astype(bool)
        for layer in ["family", "work_school", "community"]:
            beta = self.beta[layer]
            if beta <= 0:
                continue
            adj = self.network.adj[layer]
            for i in range(self.N):
                neighbors = adj[i]
                if not neighbors:
                    continue
                # Probability of at least one transmission from informed neighbors
                p_no = 1.0
                any_informed = False
                for j in neighbors:
                    if informed_mask[j]:
                        any_informed = True
                        p_no *= (1.0 - beta)
                if any_informed:
                    p = 1.0 - p_no
                    if self.rng.random() < p:
                        received[i] = True
        # Combine signals
        out = broadcast | mandate_hits | received
        # Apply decay memory so previous exposure can persist info for a few days
        # If exposed today, set memory=info_decay; else decay by 1
        newly_informed = out.copy()
        self.memory[newly_informed] = self.info_decay
        self.memory[~newly_informed] = np.maximum(0, self.memory[~newly_informed] - 1)
        persistent = self.memory > 0
        return persistent


class LogisticRegressionSimple:
    def __init__(self, lr: float = 0.05, l2: float = 0.001, max_iter: int = 1000, tol: float = 1e-6, seed: int = 42):
        self.lr = lr
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "LogisticRegressionSimple":
        n, p = X.shape
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0, 0.1, size=p)
        b = 0.0
        y_float = y.astype(np.float64)
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64)
        sw = sample_weight
        last_loss = float("inf")
        for it in range(self.max_iter):
            z = X @ w + b
            p_hat = self._sigmoid(z)
            # Loss: weighted log loss + L2
            eps = 1e-9
            loss = -np.sum(sw * (y_float * np.log(p_hat + eps) + (1 - y_float) * np.log(1 - p_hat + eps))) / n
            loss += 0.5 * self.l2 * np.sum(w * w)
            # Gradients
            diff = (p_hat - y_float) * sw
            grad_w = (X.T @ diff) / n + self.l2 * w
            grad_b = np.sum(diff) / n
            # Update
            w -= self.lr * grad_w
            b -= self.lr * grad_b
            if abs(last_loss - loss) < self.tol:
                break
            last_loss = loss
        self.coef_ = w
        self.intercept_ = float(b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        z = X @ self.coef_ + self.intercept_
        return self._sigmoid(z)


@dataclass
class DecisionPolicy:
    # per-role logistic models
    models: Dict[str, LogisticRegressionSimple]
    feature_names: List[str]

    @staticmethod
    def default_feature_names(role: str) -> List[str]:
        if role == "Nonworker":
            return [
                "mask_prev",
                "neighbor_family",
                "neighbor_community",
                "risk_perception",
                "received_info",
                "mandate_strength",
                "mask_cost",
            ]
        else:
            return [
                "mask_prev",
                "neighbor_family",
                "neighbor_work_school",
                "neighbor_community",
                "risk_perception",
                "received_info",
                "mandate_strength",
                "mask_cost",
            ]

    def predict_proba(self, role: str, X_role: np.ndarray) -> np.ndarray:
        if role not in self.models:
            raise ValueError(f"No model for role {role}")
        return self.models[role].predict_proba(X_role)


def temporal_holdout_days(days: List[int], train_frac: float = 0.8) -> Tuple[List[int], List[int]]:
    unique_days = sorted(set(days))
    if not unique_days:
        return [], []
    cutoff_idx = int(math.floor(len(unique_days) * train_frac)) - 1
    cutoff_idx = max(0, min(cutoff_idx, len(unique_days) - 2))
    train_days = unique_days[: cutoff_idx + 1]
    val_days = unique_days[cutoff_idx + 1 :]
    return train_days, val_days


def build_agents(df_attr: pd.DataFrame) -> Dict[int, Agent]:
    agents: Dict[int, Agent] = {}
    for _, row in df_attr.iterrows():
        a = Agent(
            agent_id=int(row["agent_id"]),
            age=int(row["age"]),
            age_group=str(row["age_group"]),
            occupation=str(row["occupation"]),
            risk_perception=float(row["risk_perception"]),
            initial_mask_wearing=bool(row["initial_mask_wearing"]),
            family_connections=int(row["family_connections"]),
            work_school_connections=int(row["work_school_connections"]),
            community_connections=int(row["community_connections"]),
            total_connections=int(row["total_connections"]),
        )
        agents[a.agent_id] = a
    return agents


def prepare_training_matrices(
    df_train: pd.DataFrame,
    agents: Dict[int, Agent],
    network: NetworkBuilder,
    train_days: List[int],
    mandate_schedule: Dict[int, float],
    mask_cost_schedule: Dict[int, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[str]]]:
    N = len(agents)
    id_to_idx = {aid: idx for idx, aid in enumerate(sorted(agents.keys()))}
    # Build day->mask arrays and day->received_info arrays from observed training data
    unique_days = sorted(set(df_train["day"].unique().tolist()))
    min_day = min(unique_days)
    max_day = max(unique_days)
    # Ensure we have all agent-day records; fill missing with previous day values or defaults
    # Build panel frame index aligned
    df_pivot_mask = df_train.pivot(index="agent_id", columns="day", values="wearing_mask").reindex(index=sorted(agents.keys()))
    df_pivot_info = df_train.pivot(index="agent_id", columns="day", values="received_info").reindex(index=sorted(agents.keys()))
    # Fill missing values
    df_pivot_mask = df_pivot_mask.fillna(method="ffill", axis=1).fillna(False)
    df_pivot_info = df_pivot_info.fillna(method="ffill", axis=1).fillna(False)
    # Day arrays
    day_to_mask = {d: df_pivot_mask[d].astype(bool).reindex(sorted(agents.keys())).to_numpy() for d in unique_days if d in df_pivot_mask.columns}
    day_to_info = {d: df_pivot_info[d].astype(bool).reindex(sorted(agents.keys())).to_numpy() for d in unique_days if d in df_pivot_info.columns}
    # Build training samples per role
    X_by_role: Dict[str, List[List[float]]] = {"Student": [], "Worker_BlueCollar": [], "Worker_WhiteCollar": [], "Nonworker": []}
    y_by_role: Dict[str, List[int]] = {"Student": [], "Worker_BlueCollar": [], "Worker_WhiteCollar": [], "Nonworker": []}
    features_by_role: Dict[str, List[str]] = {}
    # Construct features for t in train_days excluding the minimum day because we need t-1
    train_days_sorted = sorted(train_days)
    train_days_effective = [t for t in train_days_sorted if t > min_day]
    for t in train_days_effective:
        masks_prev = day_to_mask[t - 1]
        neighbor_shares = network.neighbor_mask_shares(masks_prev)
        info_t = day_to_info[t]
        masks_t = day_to_mask[t]
        mandate = float(mandate_schedule.get(t, 0.0))
        mask_cost = float(mask_cost_schedule.get(t, 0.0))
        # For each agent, assemble feature vector
        for idx, aid in enumerate(sorted(agents.keys())):
            agent = agents[aid]
            role = agent.role
            if role == "Nonworker":
                features = [
                    float(masks_prev[idx]),
                    float(neighbor_shares["family"][idx]),
                    float(neighbor_shares["community"][idx]),
                    float(agent.risk_perception),
                    float(info_t[idx]),
                    mandate,
                    mask_cost,
                ]
                features_by_role[role] = DecisionPolicy.default_feature_names(role)
            else:
                features = [
                    float(masks_prev[idx]),
                    float(neighbor_shares["family"][idx]),
                    float(neighbor_shares["work_school"][idx]),
                    float(neighbor_shares["community"][idx]),
                    float(agent.risk_perception),
                    float(info_t[idx]),
                    mandate,
                    mask_cost,
                ]
                features_by_role[role] = DecisionPolicy.default_feature_names(role)
            X_by_role[role].append(features)
            y_by_role[role].append(int(masks_t[idx]))
    # Convert to numpy
    X_np = {role: np.array(X_by_role[role], dtype=np.float64) if len(X_by_role[role]) > 0 else np.zeros((0, 1)) for role in X_by_role}
    y_np = {role: np.array(y_by_role[role], dtype=np.int32) if len(y_by_role[role]) > 0 else np.zeros((0,), dtype=np.int32) for role in y_by_role}
    return X_np, y_np, features_by_role


def fit_decision_policy(
    X_by_role: Dict[str, np.ndarray],
    y_by_role: Dict[str, np.ndarray],
    seed: int = 42,
) -> DecisionPolicy:
    models: Dict[str, LogisticRegressionSimple] = {}
    for role in ["Student", "Worker_BlueCollar", "Worker_WhiteCollar", "Nonworker"]:
        X = X_by_role[role]
        y = y_by_role[role]
        if X.shape[0] == 0:
            # No data to fit; create a dummy model defaulting to low adoption
            model = LogisticRegressionSimple(seed=seed)
            model.coef_ = np.zeros(X.shape[1] if X.ndim == 2 else 1)
            model.intercept_ = -2.0
            models[role] = model
            continue
        model = LogisticRegressionSimple(lr=0.1, l2=0.001, max_iter=2000, tol=1e-7, seed=seed)
        model.fit(X, y)
        models[role] = model
    # Feature names resolved from non-empty roles
    # Since we may have different feature sets, we do not enforce a global feature list; store role-specific names outside
    return DecisionPolicy(models=models, feature_names=[])


def simulate_forward(
    start_day: int,
    end_day: int,
    agents: Dict[int, Agent],
    network: NetworkBuilder,
    policy: DecisionPolicy,
    init_masks: np.ndarray,
    observed_info_by_day: Optional[Dict[int, np.ndarray]],
    info_model: Optional[InfoDiffusion],
    mandate_schedule: Dict[int, float],
    mask_cost_schedule: Dict[int, float],
    rng: Optional[random.Random] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Simulate from start_day+1 to end_day inclusive.
    Returns:
      - masks_by_day: dict day-> mask bool array
      - probs_by_day: dict day-> predicted probability array
      - info_by_day: dict day-> received_info bool array
    """
    if rng is None:
        rng = random.Random(123)
    N = len(agents)
    sorted_ids = sorted(agents.keys())
    idx_map = {aid: idx for idx, aid in enumerate(sorted_ids)}
    role_indices = {
        "Student": np.array([idx_map[aid] for aid in sorted_ids if agents[aid].role == "Student"], dtype=np.int32),
        "Worker_BlueCollar": np.array([idx_map[aid] for aid in sorted_ids if agents[aid].role == "Worker_BlueCollar"], dtype=np.int32),
        "Worker_WhiteCollar": np.array([idx_map[aid] for aid in sorted_ids if agents[aid].role == "Worker_WhiteCollar"], dtype=np.int32),
        "Nonworker": np.array([idx_map[aid] for aid in sorted_ids if agents[aid].role == "Nonworker"], dtype=np.int32),
    }
    masks_prev = init_masks.copy().astype(bool)
    info_prev = observed_info_by_day[start_day] if (observed_info_by_day and start_day in observed_info_by_day) else np.zeros(N, dtype=bool)
    masks_by_day: Dict[int, np.ndarray] = {start_day: masks_prev.copy()}
    probs_by_day: Dict[int, np.ndarray] = {}
    info_by_day: Dict[int, np.ndarray] = {start_day: info_prev.copy()}
    # Prepare arrays for static agent features
    risk = np.array([agents[aid].risk_perception for aid in sorted_ids], dtype=np.float64)

    if info_model is not None:
        info_model.reset()

    for t in range(start_day + 1, end_day + 1):
        # Info for day t
        if observed_info_by_day is not None and t in observed_info_by_day:
            info_t = observed_info_by_day[t].copy()
        else:
            if info_model is None:
                # default: no info
                info_t = np.zeros(N, dtype=bool)
            else:
                info_t = info_model.step(info_prev, mandate_schedule.get(t, 0.0))
        info_by_day[t] = info_t.copy()
        # Neighbor shares based on masks_prev
        neighbor_shares = network.neighbor_mask_shares(masks_prev)
        # Build features per role batches
        probs_t = np.zeros(N, dtype=np.float64)
        # Iterate roles
        for role, idxs in role_indices.items():
            if idxs.size == 0:
                continue
            if role == "Nonworker":
                X = np.column_stack([
                    masks_prev[idxs].astype(np.float64),
                    neighbor_shares["family"][idxs].astype(np.float64),
                    neighbor_shares["community"][idxs].astype(np.float64),
                    risk[idxs],
                    info_t[idxs].astype(np.float64),
                    np.full_like(risk[idxs], fill_value=float(mandate_schedule.get(t, 0.0))),
                    np.full_like(risk[idxs], fill_value=float(mask_cost_schedule.get(t, 0.0))),
                ])
            else:
                X = np.column_stack([
                    masks_prev[idxs].astype(np.float64),
                    neighbor_shares["family"][idxs].astype(np.float64),
                    neighbor_shares["work_school"][idxs].astype(np.float64),
                    neighbor_shares["community"][idxs].astype(np.float64),
                    risk[idxs],
                    info_t[idxs].astype(np.float64),
                    np.full_like(risk[idxs], fill_value=float(mandate_schedule.get(t, 0.0))),
                    np.full_like(risk[idxs], fill_value=float(mask_cost_schedule.get(t, 0.0))),
                ])
            probs = policy.predict_proba(role, X)
            probs_t[idxs] = probs
        # Sample new masks from probabilities
        rnd = np.array([rng.random() for _ in range(N)], dtype=np.float64)
        masks_t = (rnd < probs_t).astype(bool)
        masks_by_day[t] = masks_t.copy()
        probs_by_day[t] = probs_t.copy()
        # Advance
        masks_prev = masks_t
        info_prev = info_t
    return masks_by_day, probs_by_day, info_by_day


def compute_metrics(
    observed_masks_by_day: Dict[int, np.ndarray],
    predicted_masks_by_day: Dict[int, np.ndarray],
    probs_by_day: Dict[int, np.ndarray],
    val_days: List[int],
) -> Dict[str, float]:
    days_eval = [d for d in val_days if d in observed_masks_by_day and d in predicted_masks_by_day and d in probs_by_day]
    if not days_eval:
        return {"RMSE_aggregate": float("nan"), "MAE_aggregate": float("nan"), "Brier": float("nan"), "TransitionFit": float("nan")}
    agg_obs = []
    agg_pred = []
    brier_list = []
    trans_err_list = []
    for d in days_eval:
        y = observed_masks_by_day[d].astype(np.float64)
        p = probs_by_day[d].astype(np.float64)
        y_hat = predicted_masks_by_day[d].astype(np.float64)
        agg_obs.append(y.mean())
        agg_pred.append(y_hat.mean())
        brier_list.append(np.mean((p - y) ** 2))
        # Transition errors computed between d-1 -> d
        if (d - 1) in observed_masks_by_day and (d - 1) in predicted_masks_by_day:
            y_prev = observed_masks_by_day[d - 1].astype(np.int8)
            y_curr = y.astype(np.int8)
            yhat_prev = predicted_masks_by_day[d - 1].astype(np.int8)
            yhat_curr = y_hat.astype(np.int8)
            def trans_rates(prev: np.ndarray, curr: np.ndarray) -> Tuple[float, float, float, float]:
                on_prev = prev == 1
                off_prev = prev == 0
                total_on = max(1, int(on_prev.sum()))
                total_off = max(1, int(off_prev.sum()))
                p11 = float(((prev == 1) & (curr == 1)).sum()) / total_on
                p10 = float(((prev == 1) & (curr == 0)).sum()) / total_on
                p01 = float(((prev == 0) & (curr == 1)).sum()) / total_off
                p00 = float(((prev == 0) & (curr == 0)).sum()) / total_off
                return p11, p10, p01, p00
            obs_rates = trans_rates(y_prev, y_curr)
            pred_rates = trans_rates(yhat_prev, yhat_curr)
            trans_err = np.mean(np.abs(np.array(obs_rates) - np.array(pred_rates)))
            trans_err_list.append(trans_err)
    rmse = math.sqrt(np.mean((np.array(agg_pred) - np.array(agg_obs)) ** 2))
    mae = float(np.mean(np.abs(np.array(agg_pred) - np.array(agg_obs))))
    brier = float(np.mean(brier_list)) if brier_list else float("nan")
    transition_fit = float(np.mean(trans_err_list)) if trans_err_list else float("nan")
    return {"RMSE_aggregate": float(rmse), "MAE_aggregate": mae, "Brier": brier, "TransitionFit": transition_fit}


def save_results(
    out_dir: str,
    agents: Dict[int, Agent],
    network: NetworkBuilder,
    train_days: List[int],
    val_days: List[int],
    observed_masks: Dict[int, np.ndarray],
    sim_masks_obsinfo: Dict[int, np.ndarray],
    probs_obsinfo: Dict[int, np.ndarray],
    sim_masks_endoinfo: Dict[int, np.ndarray],
    probs_endoinfo: Dict[int, np.ndarray],
    forecast_masks: Dict[int, np.ndarray],
    probs_forecast: Dict[int, np.ndarray],
    info_obsinfo: Dict[int, np.ndarray],
    info_endoinfo: Dict[int, np.ndarray],
    mandate_schedule: Dict[int, float],
    mask_cost_schedule: Dict[int, float],
    diagnostics: Dict[str, Dict[str, float]],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sorted_ids = sorted(agents.keys())
    def write_panel(filename: str, masks_by_day: Dict[int, np.ndarray], probs_by_day: Dict[int, np.ndarray], info_by_day: Optional[Dict[int, np.ndarray]] = None) -> None:
        rows = []
        for d, masks in sorted(masks_by_day.items()):
            probs = probs_by_day.get(d, np.zeros_like(masks, dtype=np.float64))
            info = info_by_day.get(d, np.zeros_like(masks, dtype=bool)) if info_by_day is not None else np.zeros_like(masks, dtype=bool)
            for idx, aid in enumerate(sorted_ids):
                rows.append({
                    "day": d,
                    "agent_id": aid,
                    "wearing_mask": bool(masks[idx]),
                    "prob": float(probs[idx]),
                    "received_info": bool(info[idx]),
                    "role": agents[aid].role,
                    "risk_perception": agents[aid].risk_perception,
                })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, filename), index=False)

    write_panel("validation_sim_forecast_info.csv", sim_masks_obsinfo, probs_obsinfo, info_obsinfo)
    write_panel("validation_sim_endogenous_info.csv", sim_masks_endoinfo, probs_endoinfo, info_endoinfo)
    write_panel("forecast_days_30_39.csv", forecast_masks, probs_forecast, None)

    # Aggregate adoption curves
    def aggregate_curve(masks_by_day: Dict[int, np.ndarray]) -> pd.DataFrame:
        rows = []
        for d, masks in sorted(masks_by_day.items()):
            rows.append({"day": d, "adoption_rate": float(np.mean(masks.astype(np.float64)))})
        return pd.DataFrame(rows)

    aggregate_curve(observed_masks).to_csv(os.path.join(out_dir, "observed_aggregate.csv"), index=False)
    aggregate_curve(sim_masks_obsinfo).to_csv(os.path.join(out_dir, "sim_aggregate_obsinfo.csv"), index=False)
    aggregate_curve(sim_masks_endoinfo).to_csv(os.path.join(out_dir, "sim_aggregate_endoinfo.csv"), index=False)
    aggregate_curve(forecast_masks).to_csv(os.path.join(out_dir, "forecast_aggregate.csv"), index=False)

    # Transition matrices per day
    def transition_matrix_series(masks_by_day: Dict[int, np.ndarray]) -> pd.DataFrame:
        rows = []
        for d, masks in sorted(masks_by_day.items()):
            if (d - 1) in masks_by_day:
                prev = masks_by_day[d - 1].astype(np.int8)
                curr = masks.astype(np.int8)
                on_prev = prev == 1
                off_prev = prev == 0
                total_on = max(1, int(on_prev.sum()))
                total_off = max(1, int(off_prev.sum()))
                P11 = float(((prev == 1) & (curr == 1)).sum()) / total_on
                P10 = float(((prev == 1) & (curr == 0)).sum()) / total_on
                P01 = float(((prev == 0) & (curr == 1)).sum()) / total_off
                P00 = float(((prev == 0) & (curr == 0)).sum()) / total_off
                rows.append({"day": d, "P11": P11, "P10": P10, "P01": P01, "P00": P00})
        return pd.DataFrame(rows)

    transition_matrix_series(observed_masks).to_csv(os.path.join(out_dir, "observed_transitions.csv"), index=False)
    transition_matrix_series(sim_masks_obsinfo).to_csv(os.path.join(out_dir, "sim_obsinfo_transitions.csv"), index=False)
    transition_matrix_series(sim_masks_endoinfo).to_csv(os.path.join(out_dir, "sim_endoinfo_transitions.csv"), index=False)
    transition_matrix_series(forecast_masks).to_csv(os.path.join(out_dir, "forecast_transitions.csv"), index=False)

    # Save schedules and diagnostics
    with open(os.path.join(out_dir, "network_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    pd.DataFrame(sorted(mandate_schedule.items()), columns=["day", "mandate_strength"]).to_csv(
        os.path.join(out_dir, "mandate_schedule.csv"), index=False
    )
    pd.DataFrame(sorted(mask_cost_schedule.items()), columns=["day", "mask_cost"]).to_csv(
        os.path.join(out_dir, "mask_cost_schedule.csv"), index=False
    )


# Execute main for both direct execution and sandbox wrapper invocation
main()