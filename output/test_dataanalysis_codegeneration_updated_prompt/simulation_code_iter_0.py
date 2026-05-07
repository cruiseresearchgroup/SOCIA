import os
import json
import traceback
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from json import JSONDecodeError


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    # numerically stable softplus
    return np.where(x > 30, x, np.log1p(np.exp(x)))


@dataclass
class Config:
    data_folder: str = "/Users/z3546829/PycharmProjects/SOCIA/data_fitting/mask_adoption_data"
    agents_csv: str = "agent_attributes.csv"
    network_json: str = "social_network.json"
    train_csv: str = "train_data.csv"

    random_seed: int = 42

    # Validation simulation window (external)
    validation_horizon_days: int = 10
    n_validation_runs: int = 10

    # Temporal holdout within the provided training window
    holdout_fraction: float = 0.2  # last 20% of days reserved for internal validation

    # Information memory decay
    rho_info_decay: float = 0.5

    # Adoption model (logistic) hyperparameters
    adopt_l2_reg: float = 1.0
    adopt_learning_rate: float = 0.1
    adopt_max_epochs: int = 800

    # Info diffusion model (exponential mapping) hyperparameters
    info_phi_l2_reg: float = 0.01
    info_learning_rate: float = 0.05
    info_max_epochs: int = 800
    info_grad_clip: float = 5.0

    # Government intervention
    broadcast_intervention_day: int = 10  # lambda changes at this day

    verbose: bool = False


class MultiplexNetwork:
    """
    MultiplexNetwork handles graph layers and efficient neighbor-share computations.

    Attributes:
      id_list: list of agent IDs aligned to data arrays
      id_to_index: mapping from agent ID to index in arrays
      N: number of agents
      layers: dict layer->list of np.ndarray of neighbor indices per node
      layer_edges: dict layer->(rows, cols) adjacency edge lists for vectorized computation
      layer_degrees: dict layer->np.ndarray of degrees per node
    """
    def __init__(self, id_list: List[int], id_to_index: Dict[int, int]):
        self.id_list: List[int] = id_list
        self.id_to_index: Dict[int, int] = id_to_index
        self.N: int = len(id_list)
        self.layers: Dict[str, List[np.ndarray]] = {
            "family": [np.array([], dtype=np.int32) for _ in range(self.N)],
            "work_school": [np.array([], dtype=np.int32) for _ in range(self.N)],
            "community": [np.array([], dtype=np.int32) for _ in range(self.N)],
        }
        self.layer_edges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.layer_degrees: Dict[str, np.ndarray] = {}

    @staticmethod
    def from_json(path: str, allowed_ids: Set[int], id_to_index: Dict[int, int]) -> "MultiplexNetwork":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Network file not found: {path}")
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON at {path}: {str(e)}")

        layer_names = ["family", "work_school", "community"]
        adj: Dict[str, Dict[int, Set[int]]] = {ln: {} for ln in layer_names}
        for ln in layer_names:
            for aid in allowed_ids:
                adj[ln][aid] = set()

        for k, v in data.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if i not in allowed_ids:
                continue
            if not isinstance(v, dict):
                continue
            for ln in layer_names:
                neigh_list = v.get(ln, [])
                if not isinstance(neigh_list, list):
                    neigh_list = []
                for j in neigh_list:
                    try:
                        jj = int(j)
                    except (TypeError, ValueError):
                        continue
                    if jj not in allowed_ids:
                        continue
                    adj[ln][i].add(jj)
                    adj[ln][jj].add(i)

        id_list = sorted(list(allowed_ids))
        net = MultiplexNetwork(id_list=id_list, id_to_index=id_to_index)

        for ln in layer_names:
            rows: List[int] = []
            cols: List[int] = []
            degrees = np.zeros(net.N, dtype=np.int32)
            for aid in id_list:
                idx = id_to_index[aid]
                neigh_ids = sorted(list(adj[ln][aid]))
                idxs = [id_to_index[nid] for nid in neigh_ids if nid in id_to_index and nid != aid]
                net.layers[ln][idx] = np.array(idxs, dtype=np.int32)
                degrees[idx] = len(idxs)
                if idxs:
                    rows.extend([idx] * len(idxs))
                    cols.extend(idxs)
            if rows and cols:
                net.layer_edges[ln] = (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))
            else:
                net.layer_edges[ln] = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
            net.layer_degrees[ln] = degrees
        return net

    def neighbor_indices(self, layer: str) -> List[np.ndarray]:
        return self.layers[layer]

    def degrees(self, layer: str) -> np.ndarray:
        return self.layer_degrees[layer].copy()

    def compute_neighbor_share(self, layer: str, wear_vec: np.ndarray) -> np.ndarray:
        """
        Compute neighbor wearing share per node for a given layer using vectorized accumulation.

        Args:
          layer: one of {'family', 'work_school', 'community'}
          wear_vec: shape (N,) boolean or float array indicating current wearing status
        Returns:
          shares: shape (N,) float array in [0,1]
        """
        N = self.N
        rows, cols = self.layer_edges[layer]
        deg = self.layer_degrees[layer]
        if rows.size == 0:
            return np.zeros(N, dtype=float)
        sums = np.zeros(N, dtype=float)
        wearf = wear_vec.astype(float)
        np.add.at(sums, rows, wearf[cols])
        shares = np.zeros(N, dtype=float)
        mask = deg > 0
        shares[mask] = sums[mask] / deg[mask]
        return shares


class LogisticRegressor:
    """
    Simple logistic regression with L2 regularization and Adam optimizer.
    Supports optional validation set for early stopping.
    """
    def __init__(self, l2_reg: float = 1.0, learning_rate: float = 0.1, max_epochs: int = 500, verbose: bool = False):
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.coef_: Optional[np.ndarray] = None

    def _loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        n = X.shape[0]
        z = X @ w
        p = sigmoid(z)
        w_reg = w.copy()
        w_reg[0] = 0.0
        loss = (-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) + self.l2_reg * np.sum(w_reg * w_reg) / n
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "LogisticRegressor":
        n, d = X.shape
        if n <= 0:
            raise ValueError("LogisticRegressor.fit: empty training set.")
        w = np.zeros(d, dtype=np.float64)
        m = np.zeros_like(w)
        v = np.zeros_like(w)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        best_loss = float("inf")
        patience = 50
        patience_counter = 0
        use_val = X_val is not None and y_val is not None and X_val.shape[0] > 0

        for epoch in range(1, self.max_epochs + 1):
            z = X @ w
            p = sigmoid(z)
            w_reg = w.copy()
            w_reg[0] = 0.0
            error = p - y
            grad = (X.T @ error) / n + (2.0 * self.l2_reg / n) * w_reg
            # Adam update
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / (1.0 - beta1 ** epoch)
            v_hat = v / (1.0 - beta2 ** epoch)
            w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # Monitor loss
            if use_val:
                curr_loss = self._loss(X_val, y_val, w)
            else:
                curr_loss = self._loss(X, y, w)

            if curr_loss + 1e-6 < best_loss:
                best_loss = curr_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (epoch % 50 == 0 or epoch == 1):
                tr_loss = self._loss(X, y, w)
                if use_val:
                    print(f"[LogReg] epoch={epoch} train_loss={tr_loss:.6f} val_loss={curr_loss:.6f}")
                else:
                    print(f"[LogReg] epoch={epoch} loss={tr_loss:.6f}")

            if patience_counter >= patience:
                if self.verbose:
                    print(f"[LogReg] Early stopping at epoch {epoch} with best_loss={best_loss:.6f}")
                break

        self.coef_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        return sigmoid(X @ self.coef_)

    def get_coefficients(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        return self.coef_.copy()


class InfoDiffusionModel:
    """
    Information diffusion model with exponential mapping:
      p_info = 1 - exp(-(phi_f*sf + phi_w*sw + phi_c*sc + lambda_t))
    where lambda_t is a broadcast rate that can change at an intervention day.
    Parameters are constrained to be non-negative via softplus reparameterization.
    """
    def __init__(self, intervention_day: int, l2_reg: float = 0.0, learning_rate: float = 0.05, max_epochs: int = 800,
                 grad_clip: float = 5.0, verbose: bool = False):
        self.intervention_day = intervention_day
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.verbose = verbose

        # unconstrained parameters (theta_*); actual params via softplus
        self.theta_phi_f: float = 0.0
        self.theta_phi_w: float = 0.0
        self.theta_phi_c: float = 0.0
        self.theta_lambda_pre: float = 0.0
        self.theta_lambda_post: float = 0.0

    def get_params(self) -> Dict[str, float]:
        phi_f = float(softplus(np.array(self.theta_phi_f)))
        phi_w = float(softplus(np.array(self.theta_phi_w)))
        phi_c = float(softplus(np.array(self.theta_phi_c)))
        lambda_pre = float(softplus(np.array(self.theta_lambda_pre)))
        lambda_post = float(softplus(np.array(self.theta_lambda_post)))
        return {
            "phi_family": phi_f,
            "phi_work": phi_w,
            "phi_community": phi_c,
            "lambda_pre": lambda_pre,
            "lambda_post": lambda_post
        }

    def _predict_p(self, sf: np.ndarray, sw: np.ndarray, sc: np.ndarray, days: np.ndarray) -> np.ndarray:
        params = self.get_params()
        lambda_t = np.where(days >= self.intervention_day, params["lambda_post"], params["lambda_pre"])
        x = params["phi_family"] * sf + params["phi_work"] * sw + params["phi_community"] * sc + lambda_t
        x = np.clip(x, 0.0, 50.0)
        p = 1.0 - np.exp(-x)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return p

    def _loss_and_grads(self, sf: np.ndarray, sw: np.ndarray, sc: np.ndarray, days: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
        # forward
        params = self.get_params()
        lambda_t = np.where(days >= self.intervention_day, params["lambda_post"], params["lambda_pre"])
        x = params["phi_family"] * sf + params["phi_work"] * sw + params["phi_community"] * sc + lambda_t
        x = np.clip(x, 0.0, 50.0)
        p = 1.0 - np.exp(-x)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        # negative log-likelihood
        n = y.shape[0]
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        # L2 on phi and lambdas
        l2 = self.l2_reg * (params["phi_family"] ** 2 + params["phi_work"] ** 2 + params["phi_community"] ** 2 +
                            params["lambda_pre"] ** 2 + params["lambda_post"] ** 2) / n
        loss = nll + l2
        # gradients
        # dL/dx = 1 - y/p
        dLdx = 1.0 - (y / p)
        # accumulate dL/dparam (for constrained params)
        grad_phi_f = float(np.mean(dLdx * sf)) + (2.0 * self.l2_reg / n) * params["phi_family"]
        grad_phi_w = float(np.mean(dLdx * sw)) + (2.0 * self.l2_reg / n) * params["phi_work"]
        grad_phi_c = float(np.mean(dLdx * sc)) + (2.0 * self.l2_reg / n) * params["phi_community"]
        mask_pre = (days < self.intervention_day).astype(float)
        mask_post = 1.0 - mask_pre
        grad_lambda_pre = float(np.mean(dLdx * mask_pre)) + (2.0 * self.l2_reg / n) * params["lambda_pre"]
        grad_lambda_post = float(np.mean(dLdx * mask_post)) + (2.0 * self.l2_reg / n) * params["lambda_post"]

        # chain rule to unconstrained theta via derivative of softplus = sigmoid
        def dsoftplus(theta: float) -> float:
            return float(1.0 / (1.0 + np.exp(-theta)))

        grads = {
            "theta_phi_f": grad_phi_f * dsoftplus(self.theta_phi_f),
            "theta_phi_w": grad_phi_w * dsoftplus(self.theta_phi_w),
            "theta_phi_c": grad_phi_c * dsoftplus(self.theta_phi_c),
            "theta_lambda_pre": grad_lambda_pre * dsoftplus(self.theta_lambda_pre),
            "theta_lambda_post": grad_lambda_post * dsoftplus(self.theta_lambda_post),
        }
        return float(loss), grads

    def fit(self,
            sf_train: np.ndarray, sw_train: np.ndarray, sc_train: np.ndarray, days_train: np.ndarray, y_train: np.ndarray,
            sf_val: Optional[np.ndarray] = None, sw_val: Optional[np.ndarray] = None, sc_val: Optional[np.ndarray] = None,
            days_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "InfoDiffusionModel":
        # initialize parameters (unconstrained)
        rng = np.random.default_rng(123)
        self.theta_phi_f = float(rng.normal(0.0, 0.1))
        self.theta_phi_w = float(rng.normal(0.0, 0.1))
        self.theta_phi_c = float(rng.normal(0.0, 0.1))
        self.theta_lambda_pre = float(rng.normal(0.0, 0.1))
        self.theta_lambda_post = float(rng.normal(0.0, 0.1))

        # degenerate handling
        y_sum = float(np.sum(y_train))
        if y_sum < 1e-6:
            # all zeros
            self.theta_phi_f = -20.0
            self.theta_phi_w = -20.0
            self.theta_phi_c = -20.0
            self.theta_lambda_pre = -20.0
            self.theta_lambda_post = -20.0
            if self.verbose:
                print("[InfoDiffusion] Degenerate y (all zeros); set near-zero parameters.")
            return self
        if y_sum > (y_train.shape[0] - 1e-6):
            # all ones
            self.theta_phi_f = 5.0
            self.theta_phi_w = 5.0
            self.theta_phi_c = 5.0
            self.theta_lambda_pre = 5.0
            self.theta_lambda_post = 5.0
            if self.verbose:
                print("[InfoDiffusion] Degenerate y (all ones); set large parameters.")
            return self

        # Adam optimizer on thetas
        thetas = np.array([self.theta_phi_f, self.theta_phi_w, self.theta_phi_c, self.theta_lambda_pre, self.theta_lambda_post], dtype=float)
        m = np.zeros_like(thetas)
        v = np.zeros_like(thetas)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        patience, patience_counter = 50, 0
        best_loss = float("inf")
        use_val = sf_val is not None and y_val is not None and sf_val.shape[0] > 0

        for epoch in range(1, self.max_epochs + 1):
            # sync current thetas to attributes
            self.theta_phi_f, self.theta_phi_w, self.theta_phi_c, self.theta_lambda_pre, self.theta_lambda_post = thetas.tolist()

            loss, grads = self._loss_and_grads(sf_train, sw_train, sc_train, days_train, y_train)
            gvec = np.array([grads["theta_phi_f"], grads["theta_phi_w"], grads["theta_phi_c"], grads["theta_lambda_pre"], grads["theta_lambda_post"]], dtype=float)
            # gradient clipping
            norm = float(np.linalg.norm(gvec))
            if norm > self.grad_clip:
                gvec = gvec * (self.grad_clip / (norm + 1e-12))
            # Adam update
            m = beta1 * m + (1.0 - beta1) * gvec
            v = beta2 * v + (1.0 - beta2) * (gvec * gvec)
            m_hat = m / (1.0 - beta1 ** epoch)
            v_hat = v / (1.0 - beta2 ** epoch)
            thetas = thetas - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # evaluate loss
            if use_val:
                # set attributes to current thetas for proper prediction
                self.theta_phi_f, self.theta_phi_w, self.theta_phi_c, self.theta_lambda_pre, self.theta_lambda_post = thetas.tolist()
                val_p = self._predict_p(sf_val, sw_val, sc_val, days_val)
                val_loss = -float(np.mean(y_val * np.log(val_p) + (1 - y_val) * np.log(1 - val_p)))
            else:
                val_loss = loss

            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (epoch % 50 == 0 or epoch == 1):
                if use_val:
                    print(f"[InfoDiffusion] epoch={epoch} train_loss={loss:.6f} val_nll={val_loss:.6f}")
                else:
                    print(f"[InfoDiffusion] epoch={epoch} nll={loss:.6f}")

            if patience_counter >= patience:
                if self.verbose:
                    print(f"[InfoDiffusion] Early stopping at epoch {epoch} with best_val={best_loss:.6f}")
                break

        # final sync
        self.theta_phi_f, self.theta_phi_w, self.theta_phi_c, self.theta_lambda_pre, self.theta_lambda_post = thetas.tolist()
        return self

    def predict_proba(self, sf: np.ndarray, sw: np.ndarray, sc: np.ndarray, days: np.ndarray) -> np.ndarray:
        return self._predict_p(sf, sw, sc, days)


class Evaluator:
    @staticmethod
    def rmse_aggregate(pred_rates: np.ndarray, obs_rates: np.ndarray) -> float:
        if pred_rates.shape != obs_rates.shape:
            raise ValueError("pred_rates and obs_rates must have same shape")
        return float(np.sqrt(np.mean((pred_rates - obs_rates) ** 2)))

    @staticmethod
    def mae_aggregate(pred_rates: np.ndarray, obs_rates: np.ndarray) -> float:
        if pred_rates.shape != obs_rates.shape:
            raise ValueError("pred_rates and obs_rates must have same shape")
        return float(np.mean(np.abs(pred_rates - obs_rates)))

    @staticmethod
    def brier_score(pred_probs: np.ndarray, obs: np.ndarray) -> float:
        if pred_probs.shape != obs.shape:
            raise ValueError("pred_probs and obs must have same shape")
        return float(np.mean((pred_probs - obs) ** 2))

    @staticmethod
    def transition_metrics(obs_prev: np.ndarray, obs_seq: np.ndarray) -> Dict[str, float]:
        """
        Compute observed transition probabilities aggregated over the validation window.
        obs_prev: shape (N,) bool for day before validation start
        obs_seq: shape (H, N) bool for validation days
        Returns dict with P01, P10, P11, P00
        """
        prev = obs_prev.astype(bool).copy()
        H, N = obs_seq.shape
        n01 = 0
        n10 = 0
        n11 = 0
        n00 = 0
        d0 = 0
        d1 = 0
        for t in range(H):
            curr = obs_seq[t, :].astype(bool)
            prev0 = ~prev
            prev1 = prev
            n01 += int(np.sum(prev0 & curr))
            n00 += int(np.sum(prev0 & (~curr)))
            n11 += int(np.sum(prev1 & curr))
            n10 += int(np.sum(prev1 & (~curr)))
            d0 += int(np.sum(prev0))
            d1 += int(np.sum(prev1))
            prev = curr
        def safe_div(a, b):
            return float(a) / float(b) if b > 0 else 0.0
        return {
            "P01": safe_div(n01, d0),
            "P00": safe_div(n00, d0),
            "P11": safe_div(n11, d1),
            "P10": safe_div(n10, d1)
        }

    @staticmethod
    def transition_fit(pred: Dict[str, float], obs: Dict[str, float], mode: str = "MAE") -> float:
        keys = ["P01", "P00", "P11", "P10"]
        diffs = [abs(pred.get(k, 0.0) - obs.get(k, 0.0)) for k in keys]
        if mode.upper() == "MSE":
            return float(np.mean([d * d for d in diffs]))
        return float(np.mean(diffs))


class SimulationEngine:
    """
    Simulation engine for mask adoption with information diffusion and government intervention.
    Handles data loading, calibration with temporal holdout, and validation simulation.
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.df_agents: Optional[pd.DataFrame] = None
        self.df_train: Optional[pd.DataFrame] = None

        self.id_list: List[int] = []
        self.id_to_index: Dict[int, int] = {}
        self.network: Optional[MultiplexNetwork] = None

        # Attributes aligned to id_list
        self.risk: Optional[np.ndarray] = None
        self.age_groups: List[str] = []
        self.occupations: List[str] = []
        self.age_baseline: str = ""
        self.occ_baseline: str = ""
        self.age_dummies: Dict[str, np.ndarray] = {}
        self.occ_dummies: Dict[str, np.ndarray] = {}

        # Training matrices
        self.days_train: List[int] = []
        self.T_train: int = 0
        self.wear_matrix: Optional[np.ndarray] = None  # shape (T, N) bool
        self.info_matrix: Optional[np.ndarray] = None  # shape (T, N) bool
        self.mem_matrix: Optional[np.ndarray] = None   # shape (T, N) float
        self.share_f: Optional[np.ndarray] = None      # shape (T, N) float
        self.share_w: Optional[np.ndarray] = None
        self.share_c: Optional[np.ndarray] = None

        # Split indices
        self.train_transition_steps: int = 0  # number of t used for training transitions (predict t+1)
        self.holdout_transition_steps: int = 0  # number for internal validation

        # Models
        self.adopt_model: Optional[LogisticRegressor] = None
        self.info_model: Optional[InfoDiffusionModel] = None

    def load_data(self) -> None:
        # Load agents
        agents_path = os.path.join(self.cfg.data_folder, self.cfg.agents_csv)
        if not os.path.exists(agents_path):
            raise FileNotFoundError(f"Agents CSV not found at {agents_path}")
        self.df_agents = pd.read_csv(agents_path)
        required_cols = {"agent_id", "risk_perception", "initial_mask_wearing", "age_group", "occupation"}
        missing = required_cols - set(self.df_agents.columns)
        if missing:
            raise ValueError(f"Missing required columns in agent_attributes.csv: {missing}")
        self.df_agents["agent_id"] = self.df_agents["agent_id"].astype(int)
        self.df_agents["risk_perception"] = self.df_agents["risk_perception"].astype(float)
        self.df_agents["initial_mask_wearing"] = self.df_agents["initial_mask_wearing"].astype(bool)
        self.df_agents["age_group"] = self.df_agents["age_group"].astype(str)
        self.df_agents["occupation"] = self.df_agents["occupation"].astype(str)

        # Load training panel
        train_path = os.path.join(self.cfg.data_folder, self.cfg.train_csv)
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training CSV not found at {train_path}")
        self.df_train = pd.read_csv(train_path)
        required_cols_train = {"day", "agent_id", "wearing_mask", "received_info"}
        missing_t = required_cols_train - set(self.df_train.columns)
        if missing_t:
            raise ValueError(f"Missing required columns in train_data.csv: {missing_t}")
        self.df_train["day"] = self.df_train["day"].astype(int)
        self.df_train["agent_id"] = self.df_train["agent_id"].astype(int)
        self.df_train["wearing_mask"] = self.df_train["wearing_mask"].astype(bool)
        self.df_train["received_info"] = self.df_train["received_info"].astype(bool)

        # Determine ID intersection with network
        network_path = os.path.join(self.cfg.data_folder, self.cfg.network_json)
        if not os.path.exists(network_path):
            raise FileNotFoundError(f"Network JSON not found at {network_path}")
        try:
            with open(network_path, "r") as f:
                raw_network = json.load(f)
        except JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON at {network_path}: {str(e)}")

        network_ids: Set[int] = set()
        for k in raw_network.keys():
            try:
                network_ids.add(int(k))
            except Exception:
                continue

        agent_ids = set(self.df_agents["agent_id"].unique().tolist())
        train_ids = set(self.df_train["agent_id"].unique().tolist())

        common_ids = agent_ids & train_ids & network_ids
        if len(common_ids) == 0:
            raise ValueError("No common agent IDs across agents, train, and network files.")
        self.id_list = sorted(list(common_ids))
        self.id_to_index = {aid: idx for idx, aid in enumerate(self.id_list)}

        # Build network for the aligned set
        self.network = MultiplexNetwork.from_json(network_path, common_ids, self.id_to_index)

        # Align and cache attributes
        dfA = self.df_agents.set_index("agent_id").loc[self.id_list]
        self.risk = dfA["risk_perception"].values.astype(float)

        # Demographics baselines from aligned subset
        self.age_groups = sorted(dfA["age_group"].astype(str).unique().tolist())
        self.occupations = sorted(dfA["occupation"].astype(str).unique().tolist())
        self.age_baseline = "Middle Age" if "Middle Age" in self.age_groups else self.age_groups[0]
        self.occ_baseline = "White Collar" if "White Collar" in self.occupations else self.occupations[0]

        # Build dummies aligned to id_list
        age_series = dfA["age_group"].astype(str)
        occ_series = dfA["occupation"].astype(str)
        self.age_dummies = {}
        for ag in self.age_groups:
            if ag == self.age_baseline:
                continue
            self.age_dummies[ag] = (age_series == ag).values.astype(float)
        self.occ_dummies = {}
        for oc in self.occupations:
            if oc == self.occ_baseline:
                continue
            self.occ_dummies[oc] = (occ_series == oc).values.astype(float)

        # Build training matrices
        dfT = self.df_train[self.df_train["agent_id"].isin(self.id_list)].copy()
        # Determine training day index set; ensure day 0 exists (insert if missing)
        days_present = sorted(dfT["day"].unique().tolist())
        if 0 not in days_present:
            days_present = [0] + days_present
        # Pivot
        wear_df = dfT.pivot(index="day", columns="agent_id", values="wearing_mask").reindex(index=days_present, columns=self.id_list)
        info_df = dfT.pivot(index="day", columns="agent_id", values="received_info").reindex(index=days_present, columns=self.id_list)

        # Seed day 0 wearing from initial_mask_wearing for any missing values
        initial_mask_map = dfA["initial_mask_wearing"].astype(bool).to_dict()
        if 0 in wear_df.index:
            # set entire row to initial when NaN; preserve provided values
            wear_df.loc[0] = wear_df.loc[0].fillna(pd.Series(initial_mask_map))
        else:
            wear_df.loc[0] = pd.Series(initial_mask_map)
        wear_df = wear_df.sort_index()
        info_df = info_df.sort_index()

        # Forward fill wearing status; any remaining NaN to False
        wear_df = wear_df.fillna(method="ffill")
        wear_df = wear_df.fillna(False)

        # Received_info: fill missing with False (including day 0)
        info_df = info_df.fillna(False)

        self.days_train = wear_df.index.tolist()
        self.T_train = len(self.days_train)
        wear_values = wear_df.values.astype(bool)
        info_values = info_df.values.astype(bool)

        if self.T_train < 2:
            raise ValueError(f"Training window too short (T={self.T_train}). Need at least 2 days.")

        self.wear_matrix = wear_values  # (T, N)
        self.info_matrix = info_values  # (T, N)

        # Memory of info with decay
        self.mem_matrix = np.zeros_like(self.info_matrix, dtype=float)
        self.mem_matrix[0, :] = self.info_matrix[0, :].astype(float)
        for t in range(1, self.T_train):
            self.mem_matrix[t, :] = np.clip(self.cfg.rho_info_decay * self.mem_matrix[t - 1, :] + self.info_matrix[t, :].astype(float), 0.0, 1.0)

        # Precompute neighbor shares per layer per day (vectorized)
        self.share_f = np.zeros_like(self.mem_matrix, dtype=float)
        self.share_w = np.zeros_like(self.mem_matrix, dtype=float)
        self.share_c = np.zeros_like(self.mem_matrix, dtype=float)
        for t in range(self.T_train):
            self.share_f[t, :] = self.network.compute_neighbor_share("family", self.wear_matrix[t, :])
            self.share_w[t, :] = self.network.compute_neighbor_share("work_school", self.wear_matrix[t, :])
            self.share_c[t, :] = self.network.compute_neighbor_share("community", self.wear_matrix[t, :])

        # Determine internal temporal holdout split for transitions: use first 80% of transitions (T-1 steps) for training
        total_transitions = self.T_train - 1
        self.train_transition_steps = int(np.floor((1.0 - self.cfg.holdout_fraction) * total_transitions))
        self.holdout_transition_steps = total_transitions - self.train_transition_steps
        if self.train_transition_steps <= 0:
            self.train_transition_steps = max(1, total_transitions - 1)
            self.holdout_transition_steps = total_transitions - self.train_transition_steps

    def _build_adoption_design_matrices(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """
        Build training and internal holdout design matrices for adoption model.
        Returns: X_train, y_train, X_val, y_val, columns
        """
        N = len(self.id_list)
        total_transitions = self.T_train - 1
        train_steps = self.train_transition_steps
        val_steps = self.holdout_transition_steps

        columns: List[str] = []
        columns.extend(["intercept", "inertia", "share_family", "share_work_school", "share_community", "risk", "info_memory"])
        age_keys = list(self.age_dummies.keys())
        occ_keys = list(self.occ_dummies.keys())
        columns.extend([f"age[{k}]" for k in age_keys])
        columns.extend([f"occ[{k}]" for k in occ_keys])
        d = len(columns)

        # Allocate
        X_train = np.zeros((train_steps * N, d), dtype=float)
        y_train = np.zeros(train_steps * N, dtype=float)
        X_val = np.zeros((val_steps * N, d), dtype=float) if val_steps > 0 else None
        y_val = np.zeros(val_steps * N, dtype=float) if val_steps > 0 else None

        # fill helper
        def build_block_for_t(t: int) -> Tuple[np.ndarray, np.ndarray]:
            inertia = self.wear_matrix[t, :].astype(float)
            sf = self.share_f[t, :]
            sw = self.share_w[t, :]
            sc = self.share_c[t, :]
            risk = self.risk
            mem = self.mem_matrix[t, :]
            base_block = np.column_stack([
                np.ones(N, dtype=float),
                inertia,
                sf,
                sw,
                sc,
                risk,
                mem
            ])
            parts = [base_block]
            if age_keys:
                parts.append(np.column_stack([self.age_dummies[k] for k in age_keys]))
            if occ_keys:
                parts.append(np.column_stack([self.occ_dummies[k] for k in occ_keys]))
            X_block = np.column_stack(parts)
            y_block = self.wear_matrix[t + 1, :].astype(float)
            return X_block, y_block

        # Training transitions
        row = 0
        for t in range(train_steps):
            Xb, yb = build_block_for_t(t)
            X_train[row:row + N, :] = Xb
            y_train[row:row + N] = yb
            row += N

        # Holdout transitions
        if val_steps > 0:
            row = 0
            for t in range(train_steps, total_transitions):
                Xb, yb = build_block_for_t(t)
                X_val[row:row + N, :] = Xb
                y_val[row:row + N] = yb
                row += N

        return X_train, y_train, X_val, y_val, columns

    def _build_info_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                 Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Assemble info diffusion training and internal holdout datasets.
        Returns:
          sf_tr, sw_tr, sc_tr, days_tr, y_tr,
          sf_val, sw_val, sc_val, days_val, y_val
        """
        N = len(self.id_list)
        T = self.T_train
        days_arr = np.array(self.days_train, dtype=int)
        # Flatten data
        sf = self.share_f.reshape(T * N)
        sw = self.share_w.reshape(T * N)
        sc = self.share_c.reshape(T * N)
        y = self.info_matrix.astype(float).reshape(T * N)
        days_flat = np.repeat(days_arr, repeats=N)

        # Split by days (not by transitions)
        train_T = int(np.floor((1.0 - self.cfg.holdout_fraction) * T))
        if train_T <= 0:
            train_T = max(1, T - 1)
        val_T = T - train_T

        mask_tr = np.isin(days_flat, days_arr[:train_T])
        mask_val = np.isin(days_flat, days_arr[train_T:]) if val_T > 0 else np.zeros_like(days_flat, dtype=bool)

        sf_tr = sf[mask_tr]
        sw_tr = sw[mask_tr]
        sc_tr = sc[mask_tr]
        days_tr = days_flat[mask_tr]
        y_tr = y[mask_tr]

        sf_val = sf[mask_val] if val_T > 0 else None
        sw_val = sw[mask_val] if val_T > 0 else None
        sc_val = sc[mask_val] if val_T > 0 else None
        days_val = days_flat[mask_val] if val_T > 0 else None
        y_val = y[mask_val] if val_T > 0 else None

        return sf_tr, sw_tr, sc_tr, days_tr, y_tr, sf_val, sw_val, sc_val, days_val, y_val

    def calibrate(self) -> None:
        # Adoption model
        X_tr, y_tr, X_val, y_val, cols_adopt = self._build_adoption_design_matrices()
        adopt_model = LogisticRegressor(
            l2_reg=self.cfg.adopt_l2_reg,
            learning_rate=self.cfg.adopt_learning_rate,
            max_epochs=self.cfg.adopt_max_epochs,
            verbose=self.cfg.verbose
        )
        if self.cfg.verbose:
            print(f"Fitting adoption model on {X_tr.shape[0]} samples, {X_tr.shape[1]} features (holdout {0 if X_val is None else X_val.shape[0]} samples)")
        adopt_model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        self.adopt_model = adopt_model
        if self.cfg.verbose:
            coefs = adopt_model.get_coefficients()
            print(f"Adoption coefficients: {dict(zip(cols_adopt, [float(c) for c in coefs]))}")

        # Information diffusion model (exponential mapping with broadcast intervention)
        sf_tr, sw_tr, sc_tr, days_tr, y_tr_info, sf_val, sw_val, sc_val, days_val, y_val_info = self._build_info_training_data()
        info_model = InfoDiffusionModel(
            intervention_day=self.cfg.broadcast_intervention_day,
            l2_reg=self.cfg.info_phi_l2_reg,
            learning_rate=self.cfg.info_learning_rate,
            max_epochs=self.cfg.info_max_epochs,
            grad_clip=self.cfg.info_grad_clip,
            verbose=self.cfg.verbose
        )
        if self.cfg.verbose:
            n_val = 0 if y_val_info is None else y_val_info.shape[0]
            print(f"Fitting info diffusion model on {y_tr_info.shape[0]} samples (holdout {n_val})")
        info_model.fit(sf_tr, sw_tr, sc_tr, days_tr, y_tr_info, sf_val, sw_val, sc_val, days_val, y_val_info)
        self.info_model = info_model
        if self.cfg.verbose and self.info_model is not None:
            print(f"Info parameters: {self.info_model.get_params()}")

    def simulate_validation(self) -> Dict[str, object]:
        if self.adopt_model is None or self.info_model is None:
            raise RuntimeError("Models not calibrated. Call calibrate() first.")

        # Start from last observed day in training
        T_last_idx = self.T_train - 1
        N = len(self.id_list)
        horizon = self.cfg.validation_horizon_days
        runs = self.cfg.n_validation_runs

        wear_last = self.wear_matrix[T_last_idx, :].copy()
        mem_last = self.mem_matrix[T_last_idx, :].copy()

        # Prepare to collect per-day aggregated rates and per-agent probabilities averaged over runs
        mean_prob_by_day = np.zeros(horizon, dtype=float)
        mean_wear_by_day = np.zeros(horizon, dtype=float)

        # For Brier scoring, accumulate predicted probabilities per day averaged over runs
        prob_accum: List[np.ndarray] = []  # each element shape (N,)
        # For transitions (predicted)
        pred_transitions_runs: List[Dict[str, float]] = []

        # Observed validation data if present (days >= T_train in df_train)
        dfT = self.df_train
        all_days = sorted(dfT["day"].unique().tolist())
        val_days_required = list(range(self.days_train[-1] + 1, self.days_train[-1] + 1 + horizon))
        observed_available = all([d in all_days for d in val_days_required])
        obs_matrix = None
        if observed_available:
            obs_df = dfT[(dfT["day"].isin(val_days_required)) & (dfT["agent_id"].isin(self.id_list))]
            obs_pivot = obs_df.pivot(index="day", columns="agent_id", values="wearing_mask").reindex(index=val_days_required, columns=self.id_list)
            if obs_pivot.isna().any().any():
                obs_pivot = obs_pivot.fillna(False)
            obs_matrix = obs_pivot.values.astype(bool)

        # Average over runs
        for run in range(runs):
            rng = np.random.default_rng(self.cfg.random_seed + run + 1000)
            wear_curr = wear_last.copy()
            mem_curr = mem_last.copy()
            run_mean_wear = np.zeros(horizon, dtype=float)
            run_prob_per_day: List[np.ndarray] = []

            # For transitions across the horizon
            n01 = 0
            n10 = 0
            n11 = 0
            n00 = 0
            d0 = 0
            d1 = 0

            for step in range(horizon):
                day_abs = self.days_train[-1] + 1 + step  # absolute day number
                # Compute neighbor shares from current wearing
                sf = self.network.compute_neighbor_share("family", wear_curr)
                sw = self.network.compute_neighbor_share("work_school", wear_curr)
                sc = self.network.compute_neighbor_share("community", wear_curr)
                # Info probability using diffusion model
                days_vec = np.full(N, day_abs, dtype=int)
                p_info = self.info_model.predict_proba(sf, sw, sc, days_vec)
                received_info = rng.uniform(size=N) < p_info
                # Update memory
                mem_curr = np.clip(self.cfg.rho_info_decay * mem_curr + received_info.astype(float), 0.0, 1.0)
                # Adoption probability
                inertia = wear_curr.astype(float)
                base_block = np.column_stack([
                    np.ones(N, dtype=float),
                    inertia,
                    sf,
                    sw,
                    sc,
                    self.risk,
                    mem_curr
                ])
                parts = [base_block]
                if self.age_dummies:
                    parts.append(np.column_stack([self.age_dummies[k] for k in self.age_dummies.keys()]))
                if self.occ_dummies:
                    parts.append(np.column_stack([self.occ_dummies[k] for k in self.occ_dummies.keys()]))
                X_adopt = np.column_stack(parts)
                p_wear = self.adopt_model.predict_proba(X_adopt)
                wear_next = rng.uniform(size=N) < p_wear

                # Aggregate
                run_mean_wear[step] = float(np.mean(wear_next.astype(float)))
                run_prob_per_day.append(p_wear)

                # Transition counts
                prev = wear_curr
                curr = wear_next
                prev0 = ~prev
                prev1 = prev
                n01 += int(np.sum(prev0 & curr))
                n00 += int(np.sum(prev0 & (~curr)))
                n11 += int(np.sum(prev1 & curr))
                n10 += int(np.sum(prev1 & (~curr)))
                d0 += int(np.sum(prev0))
                d1 += int(np.sum(prev1))

                # Advance
                wear_curr = wear_next

            mean_wear_by_day += run_mean_wear / runs
            for t in range(horizon):
                if run == 0:
                    prob_accum.append(run_prob_per_day[t] / runs)
                else:
                    prob_accum[t] += run_prob_per_day[t] / runs

            # Compute per-run transition probabilities
            def safe_div(a, b):
                return float(a) / float(b) if b > 0 else 0.0
            pred_transitions_runs.append({
                "P01": safe_div(n01, d0),
                "P00": safe_div(n00, d0),
                "P11": safe_div(n11, d1),
                "P10": safe_div(n10, d1)
            })

        # Compute mean probability aggregated per day
        for t in range(horizon):
            mean_prob_by_day[t] = float(np.mean(prob_accum[t]))

        # Average predicted transitions across runs
        if len(pred_transitions_runs) > 0:
            pred_transitions_avg = {
                "P01": float(np.mean([r["P01"] for r in pred_transitions_runs])),
                "P00": float(np.mean([r["P00"] for r in pred_transitions_runs])),
                "P11": float(np.mean([r["P11"] for r in pred_transitions_runs])),
                "P10": float(np.mean([r["P10"] for r in pred_transitions_runs]))
            }
        else:
            pred_transitions_avg = {"P01": 0.0, "P00": 0.0, "P11": 0.0, "P10": 0.0}

        results: Dict[str, object] = {
            "pred_mean_prob_by_day": mean_prob_by_day,
            "pred_mean_wear_by_day": mean_wear_by_day,
            "per_agent_prob_by_day": prob_accum,
            "has_observed_validation": observed_available,
            "pred_transitions": pred_transitions_avg
        }

        if observed_available and obs_matrix is not None:
            # Aggregate observed rates
            obs_rates = np.mean(obs_matrix.astype(float), axis=1)  # shape (horizon,)
            results["obs_rates"] = obs_rates
            results["RMSE_aggregate"] = Evaluator.rmse_aggregate(mean_prob_by_day, obs_rates)
            results["MAE_aggregate"] = Evaluator.mae_aggregate(mean_prob_by_day, obs_rates)
            # Brier: flatten probs and observed
            probs_flat = np.concatenate([prob_accum[t] for t in range(horizon)], axis=0)
            obs_flat = obs_matrix.astype(float).reshape(-1)
            results["Brier"] = Evaluator.brier_score(probs_flat, obs_flat)

            # Transition metrics: use wear_last as prev for first step
            obs_transitions = Evaluator.transition_metrics(wear_last, obs_matrix)
            results["obs_transitions"] = obs_transitions
            results["TransitionFit"] = Evaluator.transition_fit(pred_transitions_avg, obs_transitions, mode="MAE")
        else:
            results["message"] = "Validation ground truth not available for requested horizon; metrics skipped."

        return results


def main():
    cfg = Config()
    try:
        engine = SimulationEngine(cfg)
        print("Loading data...")
        engine.load_data()
        print("Calibrating models...")
        engine.calibrate()
        print("Simulating validation window...")
        results = engine.simulate_validation()

        # Reporting
        pred_rates = results.get("pred_mean_prob_by_day", None)
        if pred_rates is not None:
            start_day = engine.days_train[-1] + 1
            horizon = cfg.validation_horizon_days
            print("Predicted aggregate adoption rates (probabilities) by day:")
            for i in range(horizon):
                print(f"  Day {start_day + i}: {float(pred_rates[i]):.4f}")
        if results.get("has_observed_validation", False):
            print("Evaluation metrics on validation window:")
            print(f"  RMSE_aggregate: {results['RMSE_aggregate']:.6f}")
            print(f"  MAE_aggregate: {results['MAE_aggregate']:.6f}")
            print(f"  Brier: {results['Brier']:.6f}")
            if "TransitionFit" in results:
                print(f"  TransitionFit (MAE over P01,P00,P11,P10): {results['TransitionFit']:.6f}")
                pt = results.get("pred_transitions", {})
                ot = results.get("obs_transitions", {})
                print(f"  Pred transitions: {pt}")
                print(f"  Obs transitions:  {ot}")
        else:
            msg = results.get("message", "")
            if msg:
                print(msg)

    except Exception as e:
        print("An error occurred during simulation:")
        print(str(e))
        traceback.print_exc()


# Execute main
main()