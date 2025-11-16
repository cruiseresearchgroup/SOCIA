# BoCalibrator仿真次数和快速模式详细解释

## 🤔 为什么"一次评估需要多次仿真"？

### 📊 **1. 一次评估 = 多次仿真的原理**

**核心概念：**
- **一次评估** = 用一组参数ω运行模拟器，得到一个误差值
- **多次仿真** = 由于模拟器包含随机性，需要多次运行取平均值来获得稳定的误差估计

**具体流程：**
```python
def evaluate_on_validation(..., k_runs: int):
    """
    k_runs: 每次评估需要运行的仿真次数
    """
    run_rmse = []
    run_mae = []
    run_brier = []
    run_trans_err = []
    
    # 🎯 关键：用相同参数运行k_runs次仿真
    for r in range(k_runs):  # 例如：k_runs = 20
        # 每次仿真都使用相同的参数params
        sim_states, sim_info, sim_probs = simulate_window(
            params=params,  # 相同的参数ω
            start_day_index=val_start_idx - 1,
            end_day_index=val_end_idx - 1,
        )
        
        # 计算这次仿真的误差
        sim_rates = sim_states.mean(axis=1)
        rmse = math.sqrt(np.mean((sim_rates - obs_rates) ** 2))
        run_rmse.append(rmse)  # 收集每次仿真的RMSE
    
    # 🎯 计算k_runs次仿真的平均误差
    rmse_mean = np.mean(run_rmse)  # 20次仿真的平均RMSE
    rmse_ci = np.std(run_rmse) * 1.96 / np.sqrt(k_runs)  # 95%置信区间
    
    return {"RMSE_aggregate_mean": rmse_mean, "RMSE_aggregate_CI95": rmse_ci}
```

### 🎲 **2. 为什么需要多次仿真？**

**模拟器的随机性来源：**
```python
def simulate_window(..., params: Parameters):
    """
    模拟器包含多个随机过程：
    1. 信息传播的随机性
    2. 个体决策的随机性（基于概率）
    3. 噪声参数的随机性
    """
    for d in range(days_count):
        # 🎲 随机信息接收
        rec = simulate_step_info(prev_states, neighbors, ...)
        
        # 🎲 随机决策（基于概率）
        p = sigmoid(logits)  # 决策概率
        new_states = (np.random.rand(N) < p).astype(np.float64)  # 随机决策
```

**为什么需要多次运行：**
1. **随机性影响**: 每次运行结果都不同，需要多次运行获得稳定估计
2. **误差估计**: 需要计算置信区间，评估参数的真实性能
3. **鲁棒性**: 避免被单次运行的随机结果误导

### ⚡ **3. 快速模式 vs 完整模式**

**完整模式 (默认 k_runs=20):**
```python
# 配置文件中通常设置
cfg.k_runs = 20  # 每次评估运行20次仿真

# 一次完整评估的计算量：
# - 20次完整仿真
# - 每次仿真包含整个时间窗口的逐日计算
# - 计算量大，但结果稳定可靠
```

**快速模式 (k_runs=5):**
```python
# BoCalibrator快速模式
if fast_mode:  # iteration < fast_mode_iterations (默认30)
    cfg.k_runs = 5  # 只运行5次仿真

# 一次快速评估的计算量：
# - 5次完整仿真
# - 计算量减少75%，但结果可能不够稳定
```

### 🔄 **4. BoCalibrator的预算分配策略**

**预算分配详解：**
```python
# 总预算：100次评估
n_trials = 100

# 初始化阶段：10次评估
n_init = min(10, n_trials // 3)  # 10次

# 主优化阶段：90次评估
remaining_iterations = 90

# 快速模式：前30次迭代 (包括初始化10次 + 主循环前20次)
fast_mode_iterations = 30
# 快速模式评估次数：30次 × 5次仿真 = 150次仿真

# 完整模式：后70次迭代
full_mode_iterations = 70  
# 完整模式评估次数：70次 × 20次仿真 = 1400次仿真

# 总仿真次数：150 + 1400 = 1550次仿真
```

**为什么这样设计：**
1. **早期探索**: 前30次用快速模式快速探索参数空间
2. **后期精确**: 后70次用完整模式精确评估候选参数
3. **效率平衡**: 在计算效率和结果精度之间找到平衡

### 📈 **5. 计算量对比**

**BoCalibrator vs RandomSearchCalibrator:**

```python
# RandomSearchCalibrator: 简单随机搜索
RandomSearchCalibrator(n_trials=100)
# 每次评估：20次仿真
# 总仿真次数：100 × 20 = 2000次仿真
# 特点：均匀搜索，没有智能引导

# BoCalibrator: 智能贝叶斯优化
BoCalibrator(n_trials=100, fast_mode_iterations=30)
# 快速模式：30次评估 × 5次仿真 = 150次仿真
# 完整模式：70次评估 × 20次仿真 = 1400次仿真
# 总仿真次数：1550次仿真
# 特点：智能引导，更有可能找到最优参数
```

### 🎯 **6. 实际运行示例**

**一次完整评估的详细过程：**
```python
# 假设评估参数 ω = [alpha=0.5, gamma=1.2, ...]

# 完整模式 (k_runs=20):
for r in range(20):
    # 第1次仿真：可能得到 RMSE = 0.15
    # 第2次仿真：可能得到 RMSE = 0.18  
    # 第3次仿真：可能得到 RMSE = 0.16
    # ...
    # 第20次仿真：可能得到 RMSE = 0.17

# 计算平均误差
rmse_values = [0.15, 0.18, 0.16, ..., 0.17]  # 20个值
rmse_mean = np.mean(rmse_values)  # 例如：0.165
rmse_std = np.std(rmse_values)    # 例如：0.012

# 返回：RMSE = 0.165 ± 0.012 (95%置信区间)
```

**快速模式 (k_runs=5):**
```python
# 快速模式 (k_runs=5):
for r in range(5):
    # 第1次仿真：RMSE = 0.15
    # 第2次仿真：RMSE = 0.18
    # 第3次仿真：RMSE = 0.16
    # 第4次仿真：RMSE = 0.17
    # 第5次仿真：RMSE = 0.19

# 计算平均误差
rmse_values = [0.15, 0.18, 0.16, 0.17, 0.19]  # 5个值
rmse_mean = np.mean(rmse_values)  # 例如：0.17
rmse_std = np.std(rmse_values)    # 例如：0.016

# 返回：RMSE = 0.17 ± 0.016 (95%置信区间)
# 注意：置信区间更宽，结果不够稳定
```

### 🎯 **7. 总结**

**"一次评估需要多次仿真"的含义：**

1. **一次评估** = 用一组参数ω计算一个误差值
2. **多次仿真** = 由于模拟器随机性，需要多次运行取平均
3. **k_runs=20** = 每次评估运行20次仿真，结果稳定
4. **k_runs=5** = 每次评估运行5次仿真，结果快速但不稳定

**BoCalibrator的智能策略：**
- **早期探索**: 用快速模式(5次仿真)快速探索参数空间
- **后期精确**: 用完整模式(20次仿真)精确评估候选参数
- **总计算量**: 1550次仿真 vs RandomSearch的2000次仿真
- **预期效果**: 更有可能找到最优参数

这样设计既保证了计算效率，又确保了最终结果的可靠性！
