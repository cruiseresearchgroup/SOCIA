# BoCalibrator Rollout采样分析报告

## ✅ 确认：BoCalibrator确实执行了完整的Rollout采样

经过详细的代码分析，我可以确认BoCalibrator完全按照您的要求实现了rollout采样，与RandomSearchCalibrator使用完全相同的方式。

## 🔄 **Rollout采样流程确认**

### 1. ✅ **每次评估参数ω时运行模拟器生成预测轨迹**

**BoCalibrator._objective_function中的调用链：**
```python
def _objective_function(self, params: np.ndarray, bundle, evaluator, 
                       train_window: Tuple[int, int], seed: int, iteration: int = 0) -> float:
    # 1. 转换参数格式
    fitted_params = self._sample_to_fitted_params(params, self.param_names, seed, train_window)
    
    # 2. 调用evaluator进行rollout采样
    result = evaluator(None, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
    # ↑ 这里会运行完整的模拟器生成预测轨迹
```

**evaluator函数链：**
```python
# evaluator = evaluate_params
def evaluate_params(simulator, params: FittedParams, window) -> Dict[str, Any]:
    # 调用evaluate_on_validation进行rollout采样
    result = evaluate_on_validation(
        wearing, neighbors, risk, age_oh, occ_oh,
        age_cat_names, occ_cat_names, legacy_params, 
        train_start, train_end, cfg.k_runs  # 运行k_runs次模拟
    )
```

### 2. ✅ **运行模拟器生成预测轨迹**

**evaluate_on_validation中的rollout采样：**
```python
def evaluate_on_validation(..., val_start_idx: int, val_end_idx: int, k_runs: int):
    # 获取观测数据
    obs_rates = wearing[val_start_idx:val_end_idx, :].mean(axis=1)
    obs_trans = transition_probs(prev_obs, curr_obs)
    
    # 运行k_runs次模拟生成预测轨迹
    for r in range(k_runs):
        # 🎯 关键：运行模拟器生成预测轨迹
        sim_states, sim_info, sim_probs = simulate_window(
            start_states=init_states,           # 初始状态
            neighbors=neighbors,                # 邻居网络
            risk=risk,                          # 风险感知
            age_oh=age_oh,                      # 年龄特征
            occ_oh=occ_oh,                      # 职业特征
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            params=params,                      # 待评估的参数ω
            start_day_index=val_start_idx - 1,  # 开始时间
            end_day_index=val_end_idx - 1,      # 结束时间
        )
        # sim_states: T_val x N - 生成的预测轨迹
        # sim_probs: T_val x N - 生成的预测概率
```

### 3. ✅ **simulate_window生成完整预测轨迹**

**simulate_window函数实现完整的轨迹生成：**
```python
def simulate_window(...):
    N = start_states.shape[0]
    days_count = end_day_index - start_day_index
    states = np.zeros((days_count, N), dtype=np.float64)    # 预测轨迹
    info = np.zeros((days_count, N), dtype=np.float64)      # 信息接收
    probs = np.zeros((days_count, N), dtype=np.float64)     # 预测概率
    
    prev_states = start_states.copy()
    for d in range(days_count):  # 逐天生成轨迹
        global_day = start_day_index + d + 1
        
        # 信息传播步骤
        rec = simulate_step_info(prev_states, neighbors, ...)
        
        # 决策步骤 - 使用参数ω计算概率
        logits = compute_logit(prev_states, share_f, share_w, share_c, risk, mem, age_oh, occ_oh, params, ...)
        p = sigmoid(logits)
        
        # 状态更新 - 随机采样生成轨迹
        new_states = (np.random.rand(N) < p).astype(np.float64)
        states[d, :] = new_states  # 存储预测轨迹
        probs[d, :] = p           # 存储预测概率
        
        prev_states = new_states
    
    return states, info, probs  # 返回完整的预测轨迹
```

### 4. ✅ **预测轨迹与训练数据对齐计算误差指标**

**轨迹对比和误差计算：**
```python
for r in range(k_runs):
    # 生成预测轨迹
    sim_states, sim_info, sim_probs = simulate_window(...)
    
    # 计算预测轨迹的聚合指标
    sim_rates = sim_states.mean(axis=1)  # 每日平均佩戴率
    
    # 🎯 与观测数据对齐计算误差
    # RMSE: 预测轨迹 vs 观测轨迹
    rmse = math.sqrt(float(np.mean((sim_rates - obs_rates) ** 2)))
    
    # MAE: 预测轨迹 vs 观测轨迹  
    mae = float(np.mean(np.abs(sim_rates - obs_rates)))
    
    # Brier: 预测概率 vs 观测状态
    brier = float(np.mean((sim_probs - wearing[val_start_idx:val_end_idx, :]) ** 2))
    
    # TransitionFit: 预测转移概率 vs 观测转移概率
    prev_sim = np.vstack([init_states.reshape(1, -1), sim_states[:-1, :]])
    sim_trans = transition_probs(prev_sim, sim_states)
    trans_err = float(np.mean([abs(sim_trans[k] - obs_trans[k]) for k in ["P01", "P11", "P10", "P00"]]))
```

## 🔍 **与RandomSearchCalibrator的对比**

### 相同的Rollout采样方式
```python
# RandomSearchCalibrator.fit中
for trial in range(self.n_trials):
    # 生成候选参数
    candidate_params = self._sample_candidate_params(...)
    
    # 🎯 使用相同的evaluator进行rollout采样
    result = evaluator(simulator, candidate_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
    score = result.get('RMSE_aggregate_mean', float('inf'))

# BoCalibrator._objective_function中  
def _objective_function(self, params: np.ndarray, ...):
    # 转换参数格式
    fitted_params = self._sample_to_fitted_params(...)
    
    # 🎯 使用相同的evaluator进行rollout采样
    result = evaluator(None, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
    objective_value = self._get_objective_value(result, iteration)
```

### 相同的误差计算方式
- **RMSE**: `math.sqrt(np.mean((sim_rates - obs_rates) ** 2))`
- **MAE**: `np.mean(np.abs(sim_rates - obs_rates))`
- **Brier**: `np.mean((sim_probs - observed_wearing) ** 2)`
- **TransitionFit**: 转移概率的绝对误差

## 📊 **Rollout采样的具体实现细节**

### 轨迹生成过程
1. **初始状态**: 使用`wearing[val_start_idx-1, :]`作为初始状态
2. **逐天模拟**: 从`val_start_idx`到`val_end_idx`的每一天
3. **状态转移**: 基于参数ω计算概率，随机采样生成新状态
4. **轨迹存储**: 保存完整的`states`和`probs`数组

### 多次运行统计
- **k_runs次运行**: 每次评估运行k_runs次独立模拟
- **统计聚合**: 计算均值和95%置信区间
- **轨迹对齐**: 每次运行都使用相同的初始条件和时间窗口

### 误差指标计算
- **时间对齐**: 预测轨迹与观测轨迹按天对齐
- **空间聚合**: 计算所有智能体的平均佩戴率
- **概率对比**: 预测概率与观测状态的逐点比较

## 🎯 **总结确认**

BoCalibrator确实完全按照您的要求实现了rollout采样：

1. **✅ 每次评估参数ω时运行模拟器**: 通过`evaluate_params` → `evaluate_on_validation` → `simulate_window`调用链
2. **✅ 生成预测轨迹**: `simulate_window`生成完整的`states`和`probs`轨迹
3. **✅ 与训练数据对齐**: 使用相同的`train_window`进行轨迹对比
4. **✅ 计算误差指标**: RMSE/MAE/Brier/TransitionFit与RandomSearchCalibrator完全相同

BoCalibrator与RandomSearchCalibrator使用完全相同的rollout采样和误差计算方式，唯一的区别是参数搜索策略（贝叶斯优化 vs 随机搜索）。
