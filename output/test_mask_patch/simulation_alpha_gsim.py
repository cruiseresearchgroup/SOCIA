"""
G-Sim baseline 复现脚本 - 口罩数据集
使用 simulation_alpha_improved.py 的真实 API
"""
import argparse
import json
import os
import types
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import wasserstein_distance
except ImportError as exc:
    raise ImportError("scipy 未安装，无法计算 Wasserstein 距离。请先执行 `pip install scipy`.") from exc

# 动态加载 simulation_alpha_improved，避免执行 main()
def _load_simulation_module():
    module_path = os.path.join(os.path.dirname(__file__), "simulation_alpha_improved.py")
    with open(module_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == "main()":
        lines.pop()
    source = "\n".join(lines) + "\n"

    class SandboxDict(dict):
        def __init__(self):
            super().__init__()
            self["__builtins__"] = __builtins__
            self["__name__"] = "simulation_alpha_impl"

        def __setitem__(self, key, value):
            if key == "main" and callable(value):
                def _stub(*args, **kwargs):
                    return None
                super().__setitem__(key, _stub)
                super().__setitem__("__actual_main__", value)
            else:
                super().__setitem__(key, value)

    sandbox = SandboxDict()
    exec(compile(source, module_path, "exec"), sandbox)
    return types.SimpleNamespace(**sandbox)


_mod = _load_simulation_module()
Simulation = _mod.Simulation
SimulationConfig = _mod.SimulationConfig
ParameterRegistry = _mod.ParameterRegistry
ensure_dir = _mod.ensure_dir
set_global_seed = _mod.set_global_seed

SUMMARY_FEATURES_PER_DAY = 5


def compute_transition_stats(wearing_window: np.ndarray) -> np.ndarray:
    """逐日统计转移概率，返回形状 (T, 4) : p01,p11,p10,p00."""
    T = wearing_window.shape[0]
    stats = np.zeros((T, 4), dtype=np.float32)
    if T <= 1:
        return stats
    prev = wearing_window[:-1]
    curr = wearing_window[1:]
    prev_bin = (prev >= 0.5).astype(np.int8)
    curr_bin = (curr >= 0.5).astype(np.int8)
    p01 = ((prev_bin == 0) & (curr_bin == 1)).mean(axis=1)
    p11 = ((prev_bin == 1) & (curr_bin == 1)).mean(axis=1)
    p10 = ((prev_bin == 1) & (curr_bin == 0)).mean(axis=1)
    p00 = ((prev_bin == 0) & (curr_bin == 0)).mean(axis=1)
    stats[1:, 0] = p01
    stats[1:, 1] = p11
    stats[1:, 2] = p10
    stats[1:, 3] = p00
    return stats


def make_summary(wearing_window: np.ndarray, window: int) -> np.ndarray:
    """生成固定长度 summary（window * 5）。"""
    window = int(window)
    assert window > 0, "window 必须为正"
    T = wearing_window.shape[0]
    means = wearing_window.mean(axis=1)
    stds = wearing_window.std(axis=1)
    transitions = compute_transition_stats(wearing_window)
    features: List[float] = []
    for idx in range(window):
        ref = min(idx, T - 1)
        mean_val = float(means[ref]) if T > 0 else 0.0
        std_val = float(stds[ref]) if T > 0 else 0.0
        p01 = float(transitions[ref, 0]) if T > 0 else 0.0
        p11 = float(transitions[ref, 1]) if T > 0 else 0.0
        p10 = float(transitions[ref, 2]) if T > 0 else 0.0
        features.extend([mean_val, std_val, p01, p11, p10])
    return np.array(features, dtype=np.float32)


def vector_to_param_dict(theta: np.ndarray, names: List[str]) -> Dict[str, float]:
    return {names[i]: float(theta[i]) for i in range(len(names))}


def param_dict_to_registry_mapping(
    params: Dict[str, float],
) -> Dict[str, float]:
    """将 prior_info.json 中的参数名映射到 ParameterRegistry 的键名。"""
    mapping: Dict[str, float] = {}
    # Decision 参数
    if "alpha" in params:
        mapping["Decision.alpha"] = params["alpha"]
    if "gamma" in params:
        mapping["Decision.gamma"] = params["gamma"]
    if "theta_f" in params:
        mapping["Decision.theta_f"] = params["theta_f"]
    if "theta_w" in params:
        mapping["Decision.theta_w"] = params["theta_w"]
    if "theta_c" in params:
        mapping["Decision.theta_c"] = params["theta_c"]
    if "beta_r" in params:
        mapping["Decision.beta_r"] = params["beta_r"]
    if "beta_i" in params:
        mapping["Decision.beta_i"] = params["beta_i"]
    if "tau" in params:
        mapping["Decision.tau"] = params["tau"]
    # Layer weights
    if "family" in params:
        mapping["Layers.family_weight"] = params["family"]
    if "work_school" in params:
        mapping["Layers.work_weight"] = params["work_school"]
    if "community" in params:
        mapping["Layers.community_weight"] = params["community"]
    # Info 参数
    if "phi_family" in params:
        mapping["Info.phi_family"] = params["phi_family"]
    if "phi_work" in params:
        mapping["Info.phi_work"] = params["phi_work"]
    if "phi_community" in params:
        mapping["Info.phi_community"] = params["phi_community"]
    if "lambda_broadcast_base" in params:
        mapping["Info.lambda_broadcast_base"] = params["lambda_broadcast_base"]
    if "lambda_broadcast_factor_after_day10" in params:
        mapping["Info.lambda_broadcast_factor_after"] = params["lambda_broadcast_factor_after_day10"]
    if "rho_info_decay" in params:
        mapping["Info.rho_info_decay"] = params["rho_info_decay"]
    # Demographic effects (需要根据实际数据扩展)
    for k, v in params.items():
        if k.startswith("age_"):
            idx = int(k.split("_")[1])
            mapping[f"Decision.age_effects.{idx}"] = v
        elif k.startswith("occ_"):
            idx = int(k.split("_")[1])
            mapping[f"Decision.occ_effects.{idx}"] = v
    return mapping


def load_prior_info(path: str) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"prior_info.json 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    names = data["parameter_names"]
    bounds_raw: Dict[str, List[float]] = data["prior_bounds"]
    bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds_raw.items()}
    return names, bounds


def generate_prior_info_from_registry(sim: Simulation) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """从 ParameterRegistry 自动生成 prior_info。"""
    registry = sim.param_registry
    names: List[str] = []
    bounds: Dict[str, Tuple[float, float]] = {}
    for name, param_def in registry.definitions.items():
        if param_def.frozen:
            continue
        if name.startswith("Decision."):
            base_name = name.replace("Decision.", "")
            if base_name in ("alpha", "gamma", "theta_f", "theta_w", "theta_c", "beta_r", "beta_i", "tau"):
                names.append(base_name)
                bounds[base_name] = param_def.bounds
            elif base_name.startswith("age_effects."):
                idx = int(base_name.split(".")[1])
                names.append(f"age_{idx}")
                bounds[f"age_{idx}"] = param_def.bounds
            elif base_name.startswith("occ_effects."):
                idx = int(base_name.split(".")[1])
                names.append(f"occ_{idx}")
                bounds[f"occ_{idx}"] = param_def.bounds
        elif name.startswith("Layers."):
            base_name = name.replace("Layers.", "").replace("_weight", "")
            if base_name == "family":
                names.append("family")
                bounds["family"] = param_def.bounds
            elif base_name in ("work", "work_school"):
                names.append("work_school")
                bounds["work_school"] = param_def.bounds
            elif base_name == "community":
                names.append("community")
                bounds["community"] = param_def.bounds
        elif name.startswith("Info."):
            base_name = name.replace("Info.", "")
            if base_name == "phi_family":
                names.append("phi_family")
                bounds["phi_family"] = param_def.bounds
            elif base_name == "phi_work":
                names.append("phi_work")
                bounds["phi_work"] = param_def.bounds
            elif base_name == "phi_community":
                names.append("phi_community")
                bounds["phi_community"] = param_def.bounds
            elif base_name == "lambda_broadcast_base":
                names.append("lambda_broadcast_base")
                bounds["lambda_broadcast_base"] = param_def.bounds
            elif base_name == "lambda_broadcast_factor_after":
                names.append("lambda_broadcast_factor_after_day10")
                bounds["lambda_broadcast_factor_after_day10"] = param_def.bounds
            elif base_name == "rho_info_decay":
                names.append("rho_info_decay")
                bounds["rho_info_decay"] = param_def.bounds
    return names, bounds


def load_mask_environment(
    seed: int,
    artifacts_root: str,
    data_dir: str,
) -> Tuple[Simulation, np.ndarray, List[int], Dict[str, Any]]:
    ensure_dir(artifacts_root)
    cfg = SimulationConfig(seed=seed)
    sim = Simulation(data_dir=data_dir, cfg=cfg, artifacts_dir=artifacts_root)
    sim.load_data()
    wearing_train = sim.obs_wearing_train
    days_train = sim.obs_days_train
    metadata = {
        "sim": sim,
        "days": days_train,
    }
    return sim, wearing_train, days_train, metadata


def generate_simulation_bank(
    sim: Simulation,
    param_names: List[str],
    prior_bounds: Dict[str, Tuple[float, float]],
    bank_size: int,
    train_start: int,
    train_end: int,
    summary_window: int,
    seed: int,
    bank_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    ensure_dir(os.path.dirname(bank_path) or ".")
    rng = np.random.default_rng(seed)
    theta_samples = np.zeros((bank_size, len(param_names)), dtype=np.float32)
    summary_samples = np.zeros((bank_size, summary_window * SUMMARY_FEATURES_PER_DAY), dtype=np.float32)
    base_values = deepcopy(sim.param_registry.values)
    for idx in range(bank_size):
        theta_vec = np.zeros(len(param_names), dtype=np.float32)
        sample_dict: Dict[str, float] = {}
        for j, name in enumerate(param_names):
            low, high = prior_bounds[name]
            val = rng.uniform(low, high)
            theta_vec[j] = float(val)
            sample_dict[name] = float(val)
        mapping = param_dict_to_registry_mapping(sample_dict)
        sim.param_registry.set_values(mapping)
        sim.set_params(mapping)
        pred = sim.run(
            start_idx=train_start,
            end_idx=train_end,
            init_from_previous_day=True,
            k_runs=1,
            run_seed=seed + idx,
        )
        wearing_pred = np.clip(pred["wearing_pred"], 0.0, 1.0)
        summary_vec = make_summary(wearing_pred, summary_window)
        theta_samples[idx, :] = theta_vec
        summary_samples[idx, :] = summary_vec
        if (idx + 1) % max(1, bank_size // 10) == 0:
            print(f"[Bank] 生成 {idx + 1}/{bank_size}")
    sim.param_registry.set_values(base_values)
    np.savez_compressed(
        bank_path,
        theta=theta_samples,
        summary=summary_samples,
        param_names=np.array(param_names),
        train_window=np.array([train_start, train_end], dtype=np.int32),
        summary_window=np.array([summary_window], dtype=np.int32),
    )
    return theta_samples, summary_samples


def load_simulation_bank(bank_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], int, Tuple[int, int]]:
    data = np.load(bank_path, allow_pickle=True)
    theta = data["theta"]
    summary = data["summary"]
    param_names = data["param_names"].tolist()
    summary_window = int(data["summary_window"][0])
    train_start, train_end = data["train_window"].tolist()
    return theta, summary, param_names, summary_window, (int(train_start), int(train_end))


def train_snpe_posterior(
    theta_samples: np.ndarray,
    summary_samples: np.ndarray,
    param_names: List[str],
    prior_bounds: Dict[str, Tuple[float, float]],
    device: str,
):
    import torch
    from sbi import utils as sbi_utils
    from sbi.inference import SNPE

    theta = torch.tensor(theta_samples, dtype=torch.float32)
    x = torch.tensor(summary_samples, dtype=torch.float32)
    if device == "cuda":
        theta = theta.to(device)
        x = x.to(device)
    prior_low = torch.tensor([prior_bounds[name][0] for name in param_names], dtype=torch.float32)
    prior_high = torch.tensor([prior_bounds[name][1] for name in param_names], dtype=torch.float32)
    prior = sbi_utils.BoxUniform(low=prior_low, high=prior_high)
    inference = SNPE(prior=prior, device=device)
    density_estimator = inference.append_simulations(theta, x).train(show_train_summary=False)
    posterior = inference.build_posterior(density_estimator)
    return posterior


def posterior_predictive_curves(
    sim: Simulation,
    posterior,
    param_names: List[str],
    train_start: int,
    train_end: int,
    posterior_samples: int,
    mc_per_sample: int,
    seed: int,
    x_observed: np.ndarray,
    device: str,
) -> np.ndarray:
    import torch

    x_tensor = torch.tensor(x_observed, dtype=torch.float32)
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(0)
    if device == "cuda":
        x_tensor = x_tensor.to(device)
    samples = posterior.sample((posterior_samples,), x=x_tensor)
    samples_np = samples.detach().cpu().numpy()
    curves = []
    base_values = deepcopy(sim.param_registry.values)
    for idx, theta_vec in enumerate(samples_np):
        sample_dict = vector_to_param_dict(theta_vec, param_names)
        mapping = param_dict_to_registry_mapping(sample_dict)
        sim.param_registry.set_values(mapping)
        sim.set_params(mapping)
        pred = sim.run(
            start_idx=train_start,
            end_idx=train_end,
            init_from_previous_day=True,
            k_runs=max(1, mc_per_sample),
            run_seed=seed + 10_000 + idx * mc_per_sample,
        )
        wearing_pred = np.clip(pred["wearing_pred"], 0.0, 1.0)
        rates = wearing_pred.mean(axis=1)
        curves.append(rates)
    sim.param_registry.set_values(base_values)
    return np.stack(curves, axis=0)


def evaluate_curves(
    curves: np.ndarray,
    observed_rates: np.ndarray,
) -> Dict[str, Any]:
    mean_curve = curves.mean(axis=0)
    rmse_mean_curve = float(np.sqrt(np.mean((mean_curve - observed_rates) ** 2)))
    mae_mean_curve = float(np.mean(np.abs(mean_curve - observed_rates)))
    wdist_mean_curve = float(wasserstein_distance(observed_rates.flatten(), mean_curve.flatten()))

    per_sample_w = [
        float(wasserstein_distance(observed_rates.flatten(), curve.flatten()))
        for curve in curves
    ]
    wdist_mean = float(np.mean(per_sample_w))
    wdist_std = float(np.std(per_sample_w, ddof=1)) if len(per_sample_w) > 1 else 0.0

    lower = np.percentile(curves, 5, axis=0)
    upper = np.percentile(curves, 95, axis=0)
        return {
        "rmse_mean_curve": rmse_mean_curve,
        "mae_mean_curve": mae_mean_curve,
        "wasserstein_mean_curve": wdist_mean_curve,
        "wasserstein_mean": wdist_mean,
        "wasserstein_std": wdist_std,
        "wasserstein_per_sample": per_sample_w,
        "mean_curve": mean_curve.tolist(),
        "ci5_curve": lower.tolist(),
        "ci95_curve": upper.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask 数据集 G-Sim baseline 复现脚本")
    parser.add_argument("--bank-size", type=int, default=1000, help="simulation bank 样本数")
    parser.add_argument("--posterior-samples", type=int, default=200, help="posterior 采样数")
    parser.add_argument("--mc-per-sample", type=int, default=20, help="posterior 每个样本的 Monte Carlo 运行次数")
    parser.add_argument("--summary-window", type=int, default=11, help="summary 时间窗口长度")
    parser.add_argument("--train-start", type=int, default=0, help="训练窗口起始 day index")
    parser.add_argument("--train-end", type=int, default=11, help="训练窗口结束 day index (exclusive)")
    parser.add_argument("--ood-shift", type=int, default=5, help="OOD 窗口相对 train 窗口的偏移，若无法满足则跳过")
    parser.add_argument(
        "--prior-info",
        type=str,
        default=None,
        help="prior_info.json 路径（如果未提供，将从 ParameterRegistry 自动生成）",
    )
    parser.add_argument("--data-dir", type=str, default=os.path.join("data_fitting", "mask_adoption_data"), help="数据目录")
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join("data_fitting", "mask_adoption_data", "gsim_mask_baseline"), help="输出目录"
    )
    parser.add_argument("--bank-path", type=str, default=None, help="自定义 simulation bank 保存路径")
    parser.add_argument("--force-bank", action="store_true", help="强制重新生成 simulation bank")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="SNPE 训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)
    bank_path = args.bank_path or os.path.join(args.output_dir, "simulation_bank.npz")

    sim, wearing_train, days_train, metadata = load_mask_environment(
        seed=args.seed,
        artifacts_root=os.path.join(args.output_dir, "artifacts"),
        data_dir=args.data_dir,
    )
    T_total = wearing_train.shape[0]
    train_start = max(0, args.train_start)
    train_end = min(T_total, args.train_end)
    if train_end <= train_start:
        raise ValueError("训练窗口无效 (train_end <= train_start)")
    if (train_end - train_start) < args.summary_window:
        raise ValueError("训练窗口长度必须不少于 summary_window")

    observed_window = wearing_train[train_start:train_end]
    observed_summary = make_summary(observed_window, args.summary_window)
    observed_rates = observed_window.mean(axis=1)

    if args.prior_info and os.path.exists(args.prior_info):
        param_names, prior_bounds = load_prior_info(args.prior_info)
        print(f"[Prior] 从文件加载: {args.prior_info}")
            else:
        param_names, prior_bounds = generate_prior_info_from_registry(sim)
        print(f"[Prior] 从 ParameterRegistry 自动生成，共 {len(param_names)} 个参数")

    if os.path.exists(bank_path) and not args.force_bank:
        theta_samples, summary_samples, param_names_loaded, summary_window_loaded, window_loaded = load_simulation_bank(bank_path)
        if param_names_loaded != param_names:
            raise ValueError("加载的 bank 参数顺序与 prior_info 不一致，请重新生成 bank")
        if summary_window_loaded != args.summary_window:
            raise ValueError("加载的 bank summary_window 与当前设置不一致，请重新生成 bank")
        if tuple(window_loaded) != (train_start, train_end):
            raise ValueError("加载的 bank 训练窗口与当前设置不一致，请重新生成 bank")
        print(f"[Bank] 复用已有 simulation bank: {bank_path}")
            else:
        theta_samples, summary_samples = generate_simulation_bank(
            sim=sim,
            param_names=param_names,
            prior_bounds=prior_bounds,
            bank_size=args.bank_size,
            train_start=train_start,
            train_end=train_end,
            summary_window=args.summary_window,
            seed=args.seed,
            bank_path=bank_path,
        )

    posterior = train_snpe_posterior(
        theta_samples=theta_samples,
        summary_samples=summary_samples,
        param_names=param_names,
        prior_bounds=prior_bounds,
        device=args.device,
    )

    curves = posterior_predictive_curves(
        sim=sim,
        posterior=posterior,
        param_names=param_names,
        train_start=train_start,
        train_end=train_end,
        posterior_samples=args.posterior_samples,
        mc_per_sample=args.mc_per_sample,
        seed=args.seed,
        x_observed=observed_summary,
        device=args.device,
    )
    metrics_id = evaluate_curves(curves, observed_rates)

    results = {
        "train_window": [train_start, train_end],
        "summary_window": args.summary_window,
        "in_distribution": metrics_id,
    }

    ood_start = train_start + args.ood_shift
    ood_end = train_end + args.ood_shift
    if 0 <= ood_start < ood_end <= T_total:
        ood_window = wearing_train[ood_start:ood_end]
        ood_rates = ood_window.mean(axis=1)
        ood_summary = make_summary(ood_window, args.summary_window)
        ood_curves = posterior_predictive_curves(
            sim=sim,
            posterior=posterior,
            param_names=param_names,
            train_start=ood_start,
            train_end=ood_end,
            posterior_samples=args.posterior_samples,
            mc_per_sample=args.mc_per_sample,
            seed=args.seed + 123,
            x_observed=ood_summary,
            device=args.device,
        )
        metrics_ood = evaluate_curves(ood_curves, ood_rates)
        results["ood_window"] = [ood_start, ood_end]
        results["ood"] = metrics_ood
    else:
        results["ood"] = None
        results["ood_window"] = None

    ensure_dir(args.output_dir)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, "posterior_curves.npz"),
        curves=curves,
        observed_rates=observed_rates,
    )
    print(f"完成 G-Sim baseline 复现，指标已保存到 {metrics_path}")


if __name__ == "__main__":
    main()
