import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


State = Dict[str, int]


def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def load_trajectories_csv(path: str) -> Dict[int, List[State]]:
    trajectories: Dict[int, List[State]] = {}
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"trajectory_id", "time_step", "S", "I", "R"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV schema mismatch for {path}. Expected columns {sorted(required)}; "
                    f"got {reader.fieldnames}"
                )

            rows = []
            for row in reader:
                tid = _safe_int(row.get("trajectory_id"))
                t = _safe_int(row.get("time_step"))
                S = _safe_int(row.get("S"))
                I = _safe_int(row.get("I"))
                R = _safe_int(row.get("R"))
                rows.append((tid, t, {"time_step": t, "S": S, "I": I, "R": R}))
    except OSError as e:
        raise RuntimeError(f"Failed to read trajectories from '{path}': {e}") from e

    rows.sort(key=lambda x: (x[0], x[1]))
    for tid, _, state in rows:
        trajectories.setdefault(tid, []).append(state)
    return trajectories


@dataclass
class SIRParams:
    beta: float
    gamma: float


class SimulatorStep:
    def __init__(self, params: SIRParams):
        self.params = params

    @staticmethod
    def _binomial(n: int, p: float, rng: random.Random) -> int:
        if n <= 0 or p <= 0.0:
            return 0
        if p >= 1.0:
            return n
        c = 0
        for _ in range(n):
            if rng.random() < p:
                c += 1
        return c

    def step(self, state_t: State, action, rng: random.Random) -> State:
        S = int(state_t["S"])
        I = int(state_t["I"])
        R = int(state_t["R"])
        N = S + I + R
        if N <= 0:
            return {"time_step": int(state_t.get("time_step", 0)) + 1, "S": 0, "I": 0, "R": 0}

        p_inf = 1.0 - math.exp(-max(0.0, self.params.beta) * (I / N))
        p_inf = min(1.0, max(0.0, p_inf))
        new_inf = self._binomial(S, p_inf, rng)

        p_rec = min(1.0, max(0.0, self.params.gamma))
        new_rec = self._binomial(I, p_rec, rng)

        new_inf = min(new_inf, S)
        new_rec = min(new_rec, I + new_inf)

        S2 = S - new_inf
        I2 = I + new_inf - new_rec
        R2 = R + new_rec

        S2 = max(0, min(N, S2))
        I2 = max(0, min(N, I2))
        R2 = max(0, min(N, R2))

        total2 = S2 + I2 + R2
        if total2 != N:
            delta = N - total2
            if delta != 0:
                s_adj = max(-S2, min(delta, N - S2))
                S2 += s_adj
                delta -= s_adj
            if delta != 0:
                i_adj = max(-I2, min(delta, N - I2))
                I2 += i_adj
                delta -= i_adj
            if delta != 0:
                r_adj = max(-R2, min(delta, N - R2))
                R2 += r_adj

        return {"time_step": int(state_t.get("time_step", 0)) + 1, "S": int(S2), "I": int(I2), "R": int(R2)}


def estimate_params_from_data(train_trajs: Dict[int, List[State]]) -> SIRParams:
    beta_vals: List[float] = []
    gamma_vals: List[float] = []

    for _, states in train_trajs.items():
        if len(states) < 2:
            continue
        for t in range(len(states) - 1):
            s0 = states[t]
            s1 = states[t + 1]
            S0, I0, R0 = int(s0["S"]), int(s0["I"]), int(s0["R"])
            S1, I1, R1 = int(s1["S"]), int(s1["I"]), int(s1["R"])
            N = S0 + I0 + R0
            if N <= 0:
                continue

            new_inf = max(0, min(S0, S0 - S1))
            new_rec = max(0, min(I0, R1 - R0))

            if I0 > 0 and S0 > 0 and 0 < new_inf < S0:
                frac = new_inf / S0
                frac = min(1.0 - 1e-12, max(1e-12, frac))
                hazard = -math.log(1.0 - frac)
                beta_est = hazard * (N / I0)
                if math.isfinite(beta_est) and beta_est >= 0.0:
                    beta_vals.append(beta_est)

            if I0 > 0:
                gamma_est = new_rec / I0
                if math.isfinite(gamma_est) and gamma_est >= 0.0:
                    gamma_vals.append(gamma_est)

    beta = sum(beta_vals) / len(beta_vals) if beta_vals else 0.2
    gamma = sum(gamma_vals) / len(gamma_vals) if gamma_vals else 0.1

    beta = min(5.0, max(0.0, beta))
    gamma = min(1.0, max(0.0, gamma))
    return SIRParams(beta=beta, gamma=gamma)


def predict_next_expected(state_t: State, params: SIRParams) -> State:
    S = int(state_t["S"])
    I = int(state_t["I"])
    R = int(state_t["R"])
    N = S + I + R
    if N <= 0:
        return {"time_step": int(state_t.get("time_step", 0)) + 1, "S": 0, "I": 0, "R": 0}

    p_inf = 1.0 - math.exp(-max(0.0, params.beta) * (I / N))
    p_inf = min(1.0, max(0.0, p_inf))
    inf_mean = S * p_inf

    rec_mean = min(1.0, max(0.0, params.gamma)) * I

    new_inf = int(round(inf_mean))
    new_rec = int(round(rec_mean))

    new_inf = max(0, min(S, new_inf))
    new_rec = max(0, min(I, new_rec))

    S2 = S - new_inf
    I2 = I + new_inf - new_rec
    R2 = R + new_rec

    S2 = max(0, min(N, S2))
    I2 = max(0, min(N, I2))
    R2 = max(0, min(N, R2))
    total2 = S2 + I2 + R2
    if total2 != N:
        delta = N - total2
        s_adj = max(-S2, min(delta, N - S2))
        S2 += s_adj
        delta -= s_adj
        if delta != 0:
            i_adj = max(-I2, min(delta, N - I2))
            I2 += i_adj
            delta -= i_adj
        if delta != 0:
            r_adj = max(-R2, min(delta, N - R2))
            R2 += r_adj

    return {"time_step": int(state_t.get("time_step", 0)) + 1, "S": int(S2), "I": int(I2), "R": int(R2)}


def mse_metrics(trajs: Dict[int, List[State]], params: SIRParams) -> Tuple[float, Dict[str, float]]:
    se_total = 0.0
    n_total = 0

    se_dim = {"S": 0.0, "I": 0.0, "R": 0.0}
    n_dim = {"S": 0, "I": 0, "R": 0}

    for _, states in trajs.items():
        if len(states) < 2:
            continue
        for t in range(len(states) - 1):
            pred = predict_next_expected(states[t], params)
            obs = states[t + 1]
            for k in ("S", "I", "R"):
                err = float(int(pred[k]) - int(obs[k]))
                se = err * err
                se_total += se
                n_total += 1
                se_dim[k] += se
                n_dim[k] += 1

    mse = se_total / n_total if n_total else float("nan")
    mse_per_dim = {k: (se_dim[k] / n_dim[k] if n_dim[k] else float("nan")) for k in ("S", "I", "R")}
    return mse, mse_per_dim


def simulate_trajectory(initial_state: State, T: int, params: SIRParams, seed: int = 0) -> List[State]:
    rng = random.Random(seed)
    stepper = SimulatorStep(params)
    traj = [dict(initial_state)]
    current = dict(initial_state)
    for _ in range(T):
        nxt = stepper.step(current, action=None, rng=rng)
        traj.append(nxt)
        current = nxt
    return traj


def main():
    base_dir = os.path.join("data_fitting", "calibrasim_sir")
    train_path = os.path.join(base_dir, "train_seed_10_n_100.csv")
    val_path = os.path.join(base_dir, "val_seed_10_n_100.csv")
    test_path = os.path.join(base_dir, "test_seed_10_n_100.csv")

    paths = {"train": train_path, "val": val_path, "test": test_path}
    available = {k: (p, os.path.exists(p)) for k, p in paths.items()}

    if not available["train"][1]:
        params = SIRParams(beta=0.6, gamma=0.15)
        init = {"time_step": 0, "S": 80, "I": 3, "R": 0}
        demo = simulate_trajectory(init, T=10, params=params, seed=1)
        for s in demo:
            print(f'{s["time_step"]},{s["S"]},{s["I"]},{s["R"]}')
        return

    train_trajs = load_trajectories_csv(train_path)
    params = estimate_params_from_data(train_trajs)

    train_mse, train_mse_dim = mse_metrics(train_trajs, params)

    print(f"estimated_params,beta={params.beta:.6f},gamma={params.gamma:.6f}")
    print(f"train_mse,{train_mse:.6f}")
    print(f"train_mse_S,{train_mse_dim['S']:.6f}")
    print(f"train_mse_I,{train_mse_dim['I']:.6f}")
    print(f"train_mse_R,{train_mse_dim['R']:.6f}")

    if available["val"][1]:
        val_trajs = load_trajectories_csv(val_path)
        val_mse, val_mse_dim = mse_metrics(val_trajs, params)
        print(f"val_mse,{val_mse:.6f}")
        print(f"val_mse_S,{val_mse_dim['S']:.6f}")
        print(f"val_mse_I,{val_mse_dim['I']:.6f}")
        print(f"val_mse_R,{val_mse_dim['R']:.6f}")

    if available["test"][1]:
        test_trajs = load_trajectories_csv(test_path)
        test_mse, test_mse_dim = mse_metrics(test_trajs, params)
        print(f"test_mse,{test_mse:.6f}")
        print(f"test_mse_S,{test_mse_dim['S']:.6f}")
        print(f"test_mse_I,{test_mse_dim['I']:.6f}")
        print(f"test_mse_R,{test_mse_dim['R']:.6f}")


main()