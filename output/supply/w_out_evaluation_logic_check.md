# W_out评估逻辑检查报告

## 一、当前代码逻辑（improved文件）

### 1.1 W_out评估流程（第2653-2666行）

```python
if len(test_ood_trajectories) > 0:
    # 步骤1: 创建临时simulator，设置lead_time_L=5
    ood_params_dict = fitted.to_dict()
    ood_params_dict["module_params"]["supply"]["lead_time_L"] = 5  # 强制设置lead_time=5
    ood_simulator = BeerGameSimulator(ood_params_dict["module_params"])
    
    # 步骤2: 使用这个simulator rollout OOD测试数据
    res_ood = ood_simulator.rollout(test_ood_trajectories)
    
    # 步骤3: 计算模拟结果和观测数据的Wasserstein距离
    w_out = evaluator.compute_joint_wasserstein_per_t(res_ood, n_samples=evaluator.n_samples)
    w_out_val = float(w_out)
```

**关键点：**
- ✅ 使用优化后的参数（`poisson_lambda=4.907`等）
- ✅ 但强制设置`lead_time_L=5`（OOD数据的lead_time）
- ✅ 使用这个simulator来模拟OOD测试数据
- ✅ 然后与真实的OOD观测数据比较

### 1.2 Wasserstein距离计算（compute_joint_wasserstein_per_t方法，第958-1045行）

```python
def compute_joint_wasserstein_per_t(self, results: SimulationResults, ...):
    # 1. 收集所有轨迹的所有时间步的inventory和backlog
    inv_sim_all = []
    inv_obs_all = []
    b_sim_all = []
    b_obs_all = []
    
    # 2. 使用轨迹ID的交集确保对齐
    tids_sim = set(inv_sim.keys()) & set(b_sim.keys())
    tids_obs = set(inv_obs.keys()) & set(b_obs.keys())
    tids = list(tids_sim & tids_obs)
    
    # 3. 对每个轨迹，对齐长度后收集数据
    for tid in tids:
        min_len = min(len(inv_sim_traj), len(b_sim_traj), len(inv_obs_traj), len(b_obs_traj))
        inv_sim_all.extend(inv_sim_traj[:min_len].tolist())
        # ... 其他数据
    
    # 4. 组合成2D状态向量（inventory, backlog）
    sim_states = np.column_stack([inv_sim_aligned, b_sim_aligned])
    obs_states = np.column_stack([inv_obs_aligned, b_obs_aligned])
    
    # 5. 使用ot.emd()计算真正的Wasserstein距离
    cost_matrix = ot.dist(sim_states, obs_states, metric="euclidean")
    transport_plan = ot.emd(a, b, cost_matrix)
    wass = float(np.sum(cost_matrix * transport_plan))
```

**关键点：**
- ✅ 使用所有数据（不采样）
- ✅ 使用轨迹ID交集确保对齐
- ✅ 组合成2D状态向量（inventory, backlog）
- ✅ 使用`ot.emd()`计算真正的Wasserstein距离（不是正则化的Sinkhorn）
- ✅ 与GSIM的`wasserstein_distance_nd()`对齐

## 二、与GSIM对齐检查

### 2.1 Wasserstein距离计算方式

**GSIM的wasserstein_distance_nd()函数（supply_new/simulation_code_iter_0.py，第161-192行）：**
```python
def wasserstein_distance_nd(X: np.ndarray, Y: np.ndarray) -> float:
    N, M = X.shape[0], Y.shape[0]
    X, Y = X.reshape(N, -1), Y.reshape(M, -1)
    cost_matrix = ot.dist(X, Y, metric="euclidean")
    a, b = np.ones(N) / N, np.ones(M) / M
    transport_plan = ot.emd(a, b, cost_matrix)
    return float(np.sum(cost_matrix * transport_plan))
```

**当前improved文件的实现：**
```python
cost_matrix = ot.dist(sim_states, obs_states, metric="euclidean")
a = np.ones(N, dtype=float) / float(N)
b = np.ones(M, dtype=float) / float(M)
transport_plan = ot.emd(a, b, cost_matrix)
wass = float(np.sum(cost_matrix * transport_plan))
```

**对比结果：**
- ✅ **完全一致**：都使用`ot.emd()`计算真正的Wasserstein距离
- ✅ **完全一致**：都使用欧几里得距离作为cost matrix
- ✅ **完全一致**：都使用均匀分布作为权重（`a = ones(N)/N`, `b = ones(M)/M`）

### 2.2 状态向量组合方式

**GSIM的compute_metrics方法（supply_new/simulation_code_iter_0.py，第2687-2702行）：**
```python
# Combine into 2D state vectors (inventory, backlog)
sim_states = np.column_stack([inv_sim_aligned, back_sim_aligned])
obs_states = np.column_stack([inv_obs_aligned, back_obs_aligned])
metrics["wass"] = wasserstein_distance_nd(sim_states, obs_states)
```

**当前improved文件的实现：**
```python
# Combine into 2D state vectors (inventory, backlog) - aligned with GSIM
sim_states = np.column_stack([inv_sim_aligned, b_sim_aligned])
obs_states = np.column_stack([inv_obs_aligned, b_obs_aligned])
# ... 然后使用ot.emd()计算
```

**对比结果：**
- ✅ **完全一致**：都使用2D状态向量（inventory, backlog）
- ✅ **完全一致**：都使用`np.column_stack()`组合

### 2.3 OOD评估逻辑

**关键问题：GSIM的env.py是否也使用优化后的参数来评估OOD？**

从逻辑上分析：
- **OOD评估的正确逻辑应该是：**
  - 使用优化后的参数（但lead_time=5）来模拟OOD数据
  - 然后与真实的OOD观测数据比较
  - 这样可以测试模型在分布外数据上的泛化能力

- **当前improved文件的实现：**
  - ✅ 使用优化后的参数（`poisson_lambda=4.907`等）
  - ✅ 但强制设置`lead_time_L=5`
  - ✅ 使用这个simulator来模拟OOD测试数据
  - ✅ 然后与真实的OOD观测数据比较

**结论：逻辑应该是正确的，与GSIM应该一致。**

## 三、潜在问题分析

### 3.1 W_out仍然很小的可能原因

**当前结果：**
- W_in: 1.044 (修复前: 0.373, 改善了2.80倍)
- W_out: 0.035 (修复前: 0.044, 变化不大)
- 对比GSIM: W_in ~3.720, W_out ~4.554

**可能的原因：**

1. **优化后的参数对OOD数据也拟合得很好（正常情况）**
   - 优化后的`poisson_lambda=4.907`可能对`lead_time=5`的数据也拟合得很好
   - 这说明模型泛化能力强，这是正常的

2. **数据对齐可能有问题**
   - 需要检查轨迹ID是否完全匹配
   - 需要检查时间步长度是否对齐

3. **参数使用可能有问题**
   - 需要确认OOD评估时使用的参数是否正确
   - 需要确认是否所有参数都被正确传递

### 3.2 需要进一步检查的点

1. **轨迹ID匹配：**
   - 检查`test_ood_trajectories`中的轨迹ID是否与模拟结果的轨迹ID匹配
   - 检查是否有轨迹被遗漏

2. **时间步对齐：**
   - 检查每个轨迹的时间步长度是否一致
   - 检查是否有时间步被截断

3. **参数传递：**
   - 检查`fitted.to_dict()`是否正确返回所有参数
   - 检查`BeerGameSimulator`是否正确使用这些参数

4. **数据统计：**
   - 检查模拟的OOD数据和真实的OOD观测数据的统计特性
   - 检查是否有异常值或缺失值

## 四、代码逻辑正确性总结

### ✅ 正确的部分：

1. **Wasserstein距离计算：**
   - ✅ 使用`ot.emd()`计算真正的Wasserstein距离
   - ✅ 与GSIM的`wasserstein_distance_nd()`完全一致
   - ✅ 使用2D状态向量（inventory, backlog）

2. **OOD评估逻辑：**
   - ✅ 使用优化后的参数（但lead_time=5）来模拟OOD数据
   - ✅ 然后与真实的OOD观测数据比较
   - ✅ 逻辑应该是正确的

3. **数据对齐：**
   - ✅ 使用轨迹ID交集确保对齐
   - ✅ 对每个轨迹对齐长度

### ⚠️ 需要进一步确认的部分：

1. **GSIM的env.py是否也使用优化后的参数来评估OOD？**
   - 如果是，那么W_out小可能是正常的（模型泛化好）
   - 如果不是，那么需要修改代码逻辑

2. **数据统计特性：**
   - 需要检查模拟的OOD数据和真实的OOD观测数据的统计特性
   - 确认是否有异常

3. **参数传递：**
   - 需要确认所有参数都被正确传递和使用

## 五、建议

1. **确认GSIM的env.py的评估逻辑：**
   - 查看GSIM的env.py文件，确认OOD评估时使用的参数
   - 确认是否也使用优化后的参数（但lead_time=5）

2. **检查数据统计特性：**
   - 打印模拟的OOD数据和真实的OOD观测数据的统计特性
   - 检查是否有异常

3. **添加调试信息：**
   - 在评估时打印使用的参数
   - 打印轨迹ID匹配情况
   - 打印数据统计特性

4. **如果W_out小是正常的：**
   - 说明模型泛化能力强，这是好事
   - 但需要确认是否与GSIM的评估方式一致

