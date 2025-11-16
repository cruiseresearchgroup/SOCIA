# BoCalibrator循环和最终报告分析

## ✅ 确认：BoCalibrator确实实现了完整的循环和最终报告

经过详细的代码分析，我可以确认BoCalibrator完全按照您的要求实现了循环直到预算用完、选择最优参数，并在validation data上报告最终结果。

## 🔄 **1. 循环直到预算用完**

### ✅ **主优化循环**

**BoCalibrator._run_bayesian_optimization中的循环：**
```python
# 预算定义：n_trials次评估
for iteration in range(n_init, self.n_trials):
    print(f"\n--- Iteration {iteration + 1}/{self.n_trials} ---")
    
    # Step 2: 拟合GP模型
    gp_model = self._fit_gp_model(self.X_train, self.Y_train)
    
    # Step 3: 获取采集函数并优化
    acq_func = self._get_acquisition_function(gp_model, best_objective)
    candidates = self._optimize_acquisition(acq_func, n_candidates=1)
    
    # Step 4: 评估新候选参数
    new_params = candidates[0].numpy()
    new_objective = self._objective_function(new_params, bundle, evaluator, train_window, seed, iteration=iteration)
    
    # Step 5: 更新训练数据和最佳参数
    self.X_train = torch.cat([self.X_train, candidates], dim=0)
    self.Y_train = torch.cat([self.Y_train, torch.tensor([[new_objective]], dtype=torch.float64)], dim=0)
    
    if new_objective < best_objective:
        best_params = new_params.copy()
        best_objective = new_objective
        print(f"  ✓ New best found: objective = {best_objective:.4f}")
    
    # 进度报告
    if (iteration + 1) % 10 == 0 or iteration == self.n_trials - 1:
        print(f"  Progress: {iteration + 1}/{self.n_trials} iterations, Best objective: {best_objective:.4f}")

print(f"\nBoCalibrator: Optimization completed!")
print(f"  Final best objective: {best_objective:.4f}")
print(f"  Total evaluations: {len(self.optimization_history) + n_init}")
```

### ✅ **预算控制**

**BoCalibrator初始化时的预算设置：**
```python
def __init__(self, n_trials: int = 100, ...):
    self.n_trials = n_trials  # 总评估预算

# 使用示例
calibrator = get_calibrator("bo", n_trials=100, ...)  # 100次评估预算
```

**循环预算分配：**
```python
# 初始化阶段
n_init = min(10, self.n_trials // 3)  # 初始样本：10个或1/3预算

# 主优化阶段
print(f"Step 2-5: Running main optimization loop for {self.n_trials - n_init} iterations...")
# 剩余预算用于贝叶斯优化循环
```

## 🎯 **2. 最优参数 = 在train上误差最小的ω**

### ✅ **最优参数跟踪**

**在循环中持续跟踪最优参数：**
```python
# 初始化最优参数
best_params = X_init[best_idx].numpy()  # 初始最佳参数
best_objective = Y_init[best_idx].item()  # 初始最佳误差

# 循环中更新最优参数
for iteration in range(n_init, self.n_trials):
    # 评估新候选参数
    new_objective = self._objective_function(new_params, ...)
    
    # 🎯 如果新参数在train上误差更小，更新最优参数
    if new_objective < best_objective:
        best_params = new_params.copy()  # 更新最优参数ω*
        best_objective = new_objective   # 更新最优误差
        print(f"  ✓ New best found: objective = {best_objective:.4f}")
```

### ✅ **最优参数返回**

**BoCalibrator.fit方法返回最优参数：**
```python
def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int) -> FittedParams:
    # 运行贝叶斯优化找到最优参数
    optimal_params, optimal_objective = self._run_bayesian_optimization(
        bundle, evaluator, train_window, seed
    )
    
    # 🎯 转换最优参数为FittedParams格式
    fitted_params = self._sample_to_fitted_params(
        optimal_params, self.param_names, seed, train_window
    )
    
    # 记录最优参数信息
    fitted_params.meta.update({
        'optimal_objective_value': float(optimal_objective),  # 最优误差
        'n_optimization_trials': self.n_trials,              # 总评估次数
    })
    
    return fitted_params  # 返回最优参数ω*
```

### ✅ **训练数据上的误差评估**

**evaluate_params函数在训练窗口上评估：**
```python
def evaluate_params(simulator, params: FittedParams, window) -> Dict[str, Any]:
    # 提取训练窗口信息
    wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg = window
    train_start, train_end = train_window  # 训练窗口
    
    # 🎯 在训练窗口上运行仿真计算误差
    result = evaluate_on_validation(
        wearing, neighbors, risk, age_oh, occ_oh,
        age_cat_names, occ_cat_names, legacy_params, 
        train_start, train_end, cfg.k_runs  # 使用训练窗口
    )
    
    return result  # 返回训练误差
```

## 📊 **3. 最后报告：用最优参数ω* rollout模拟器 → 在validation data上算指标**

### ✅ **数据划分**

**训练/验证数据划分：**
```python
# 数据划分
train_end_idx, val_start_idx, val_end_idx = build_train_validation_splits(days, cfg.val_split_ratio)

# 训练窗口：用于参数优化
train_window = (1, train_end_idx)

# 验证窗口：用于最终报告
# val_start_idx 到 val_end_idx
```

### ✅ **最优参数在验证数据上的最终评估**

**main函数中的最终报告：**
```python
# 1. 使用训练数据找到最优参数ω*
fitted_params = calibrator.fit(
    bundle=bundle,
    simulator=None,
    evaluator=evaluate_params,  # 在train_window上评估
    train_window=train_window,   # 训练窗口
    seed=cfg.seed
)

# 2. 转换最优参数
params = fitted_params.to_parameters()  # 最优参数ω*

# 3. 🎯 用最优参数ω*在validation data上rollout模拟器
metrics = evaluate_on_validation(
    wearing=wearing,
    neighbors=neighbors,
    risk=risk_perception,
    age_oh=age_oh,
    occ_oh=occ_oh,
    age_cat_names=age_cat_names,
    occ_cat_names=occ_cat_names,
    params=params,              # 最优参数ω*
    val_start_idx=val_start_idx, # 验证数据开始
    val_end_idx=val_end_idx,     # 验证数据结束
    k_runs=cfg.k_runs,          # 多次运行统计
)
```

### ✅ **验证数据上的指标计算**

**evaluate_on_validation在验证窗口上的评估：**
```python
def evaluate_on_validation(..., val_start_idx: int, val_end_idx: int, k_runs: int):
    # 🎯 获取验证数据的观测值
    obs_rates = wearing[val_start_idx:val_end_idx, :].mean(axis=1)  # 验证期间观测率
    obs_trans = transition_probs(prev_obs, curr_obs)                # 验证期间转移概率
    
    # 🎯 用最优参数ω*运行k_runs次模拟
    for r in range(k_runs):
        sim_states, sim_info, sim_probs = simulate_window(
            start_states=init_states,
            neighbors=neighbors,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            params=params,                    # 最优参数ω*
            start_day_index=val_start_idx - 1, # 验证窗口开始
            end_day_index=val_end_idx - 1,     # 验证窗口结束
        )
        
        # 🎯 计算验证指标
        sim_rates = sim_states.mean(axis=1)
        
        # RMSE: 验证期间预测 vs 观测
        rmse = math.sqrt(float(np.mean((sim_rates - obs_rates) ** 2)))
        
        # MAE: 验证期间预测 vs 观测
        mae = float(np.mean(np.abs(sim_rates - obs_rates)))
        
        # Brier: 验证期间预测概率 vs 观测状态
        brier = float(np.mean((sim_probs - wearing[val_start_idx:val_end_idx, :]) ** 2))
        
        # TransitionFit: 验证期间转移概率误差
        sim_trans = transition_probs(prev_sim, sim_states)
        trans_err = float(np.mean([abs(sim_trans[k] - obs_trans[k]) for k in ["P01", "P11", "P10", "P00"]]))
    
    # 🎯 返回验证数据上的最终指标
    return {
        "RMSE_aggregate_mean": rmse_mean,
        "MAE_aggregate_mean": mae_mean,
        "Brier_mean": brier_mean,
        "TransitionFit_mean": trans_mean,
        "observed_daily_rates": obs_rates.tolist(),
        "predicted_daily_rates_mean": daily_mean.tolist(),
        "predicted_daily_rates_CI95": daily_ci.tolist(),
    }
```

### ✅ **最终结果保存**

**保存最终报告：**
```python
# 保存最终结果
save_json({"config": asdict(cfg)}, os.path.join(out_dir, "config.json"))
save_json(fitted_params.to_dict(), os.path.join(out_dir, "calibrated_parameters.json"))  # 最优参数ω*
save_json(metrics, os.path.join(out_dir, "validation_metrics.json"))                     # 验证指标
save_json(forecast, os.path.join(out_dir, "forecast.json"))                             # 预测结果
```

## 🎯 **完整流程总结**

### ✅ **BoCalibrator的完整执行流程**

1. **🔧 初始化**:
   - 设置评估预算 `n_trials`
   - 初始化 `n_init` 个随机参数样本

2. **🔄 循环直到预算用完**:
   ```python
   for iteration in range(n_init, self.n_trials):  # 预算控制
       # 拟合GP模型
       # 优化采集函数选择新参数
       # 评估新参数（在train_window上）
       # 更新最优参数（如果更好）
   ```

3. **🎯 选择最优参数**:
   - `best_params` = 在训练数据上误差最小的参数ω*
   - 返回 `FittedParams` 包含最优参数

4. **📊 最终报告**:
   - 用最优参数ω*在validation data上rollout模拟器
   - 计算验证指标：RMSE/MAE/Brier/TransitionFit
   - 保存最终结果到文件

### ✅ **数据流确认**

```
训练阶段：
参数ω → 在train_window上rollout → 计算训练误差 → GP建模 → 采集函数 → 新参数ω

最终报告：
最优参数ω* → 在val_start_idx到val_end_idx上rollout → 计算验证指标 → 最终报告
```

## 🎯 **总结确认**

BoCalibrator确实完全按照您的要求实现了：

1. **✅ 循环直到预算用完**: `for iteration in range(n_init, self.n_trials)`
2. **✅ 最优参数 = 在train上误差最小的ω**: 持续跟踪`best_params`和`best_objective`
3. **✅ 最后报告**: 用最优参数ω*在validation data上rollout模拟器计算最终指标

BoCalibrator实现了完整的贝叶斯优化流程，确保在预算内找到最优参数，并在验证数据上提供可靠的最终性能报告！
