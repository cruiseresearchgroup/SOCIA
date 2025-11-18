# 训练和评估阶段对比报告

## 文件对比
- **文件1**: `simulation_code_iter_0_alpha.py`
- **文件2**: `simulation_code_iter_0_alpha_poisson_improved.py`

---

## 一、训练阶段差异

### 1.1 校准器类型
- **文件1**: 使用 `SBICalibrator` 类（直接实现）
- **文件2**: 使用 `SNPECalibrator` 类（继承自 `Calibrator` 接口）

### 1.2 特征计算方式

#### 文件1 (`simulation_code_iter_0_alpha.py`):
- **特征函数**: `compute_summary_features()` (全局函数)
- **特征维度**: 13维特征
  - `inv_mu`, `inv_sd`, `b_mu`, `b_sd`, `p_mu`, `p_sd`
  - `inv_vol`, `b_vol`
  - `last_inv`
  - `inv_acf1`, `b_acf1`, `p_acf1`
- **计算方式**: 
  - 使用 `holdout_ranges` 来截取训练窗口
  - 对每个轨迹使用 `tr.t <= te` 来mask数据
  - 计算所有轨迹的统计特征（均值、标准差、波动率、ACF等）

#### 文件2 (`simulation_code_iter_0_alpha_poisson_improved.py`):
- **特征函数**: `features_from_observed()` (局部函数，在 `SNPECalibrator.fit()` 内部)
- **特征维度**: 10维特征
  - `inv_mean`, `inv_std`, `bk_mean`, `bk_std`, `pl_mean`, `pl_std`
  - `dinv_mean`, `dinv_std`, `db_mean`, `db_std`
- **计算方式**:
  - 使用 `train_window` (start, end) 来截取窗口
  - 直接使用数组切片 `inv[s:e+1]`
  - 计算均值、标准差，以及差分（diff）的均值和标准差

**关键差异**:
- 文件1使用13维特征（包含ACF、波动率、最后值）
- 文件2使用10维特征（只包含均值、标准差、差分统计）
- 文件1的特征更丰富，但可能更复杂

### 1.3 训练窗口处理

#### 文件1:
```python
for tr in self.train_trajectories:
    te, _ = self.holdout_ranges[tr.trajectory_id]
    horizon = int(np.sum(tr.t <= te))
    mask = tr.t <= te
    inv_series.append(tr.inventory_obs[mask])
```

#### 文件2:
```python
for tr in trajs:
    s = max(0, min(start, T - 1))
    e = max(0, min(end, T - 1))
    if e < s:
        s, e = 0, T - 1
    inv_w = inv[s:e+1]
```

**关键差异**:
- 文件1使用 `holdout_ranges` 字典，每个轨迹有独立的结束时间 `te`
- 文件2使用统一的 `train_window` (start, end)，所有轨迹使用相同的窗口

### 1.4 模拟包装器 (Simulation Wrapper)

#### 文件1:
```python
def _simulation_wrapper(self, theta: Any) -> Any:
    theta_np = theta.detach().cpu().numpy().astype(float)
    params = self._theta_to_params(theta_np)
    feats = self._simulate_with_params_for_sbi(params)
    return torch.tensor(feats.astype(np.float32), dtype=torch.float32)
```

#### 文件2:
```python
def sim_wrapper_fn(theta_t: torch.Tensor) -> torch.Tensor:
    # Handle batch input
    if theta_np.ndim > 1:
        # Process each sample separately and stack results
        results = []
        for theta_single in theta_np:
            # ... process single sample
            results.append(torch.tensor(feats, dtype=torch.float32))
        return torch.stack(results)
    # Single sample case
    return torch.tensor(feats, dtype=torch.float32)
```

**关键差异**:
- 文件2明确处理batch输入，文件1可能只处理单个样本
- 文件2的batch处理更健壮

### 1.5 参数空间定义

#### 文件1:
- 在 `SBICalibrator.__init__()` 中定义参数空间
- 使用 `_get_base_param_space()` 和 `_build_param_space_with_alpha()` 方法

#### 文件2:
- 在 `SNPECalibrator.fit()` 中动态定义参数空间
- 直接从 `adapter._bounds()` 获取边界
- 根据 `demand_family` 动态添加需求参数

**关键差异**:
- 文件1的参数空间在初始化时确定
- 文件2的参数空间在fit时动态确定，更灵活

---

## 二、评估阶段差异

### 2.1 OOD评估逻辑

#### 文件1 (`simulation_code_iter_0_alpha.py`):
```python
if len(test_trajectories_ood) > 0:
    # 使用真实的OOD测试数据（lead_time=5生成的真实数据）
    sim_results_test_ood = simulator.rollout(test_trajectories_ood)
    # ❌ 问题：simulator使用的是优化后的参数（lead_time_L=2）
    # 但OOD数据是用lead_time=5生成的！
```

**问题**: 使用优化后的 `simulator`（lead_time_L=2）来模拟OOD数据（lead_time=5）

#### 文件2 (`simulation_code_iter_0_alpha_poisson_improved.py`):
```python
if len(test_ood_trajectories) > 0:
    # For OOD evaluation, we need to use lead_time=5
    ood_params_dict = fitted.to_dict()
    ood_params_dict["module_params"]["supply"]["lead_time_L"] = 5
    ood_simulator = BeerGameSimulator(ood_params_dict["module_params"])
    res_ood = ood_simulator.rollout(test_ood_trajectories)
    # ✅ 修复：使用lead_time=5的simulator来模拟OOD数据
```

**修复**: 创建临时simulator，强制设置 `lead_time_L=5`

### 2.2 W_in和W_out的计算方式

#### 文件1:
```python
# 从metrics中提取
w_in = metrics_test.get("distributional", {}).get("Wasserstein_per_t", 0.0)
w_out = metrics_test_ood.get("distributional", {}).get("Wasserstein_per_t", 0.0)
```

#### 文件2:
```python
# 直接计算
res_test = simulator.rollout(test_trajectories)
w_in = evaluator.compute_joint_wasserstein_per_t(res_test, n_samples=evaluator.n_samples)

res_ood = ood_simulator.rollout(test_ood_trajectories)
w_out = evaluator.compute_joint_wasserstein_per_t(res_ood, n_samples=evaluator.n_samples)
```

**关键差异**:
- 文件1从已计算的metrics中提取
- 文件2直接调用evaluator计算，更直接

### 2.3 评估器初始化

#### 文件1:
```python
evaluator = Evaluator()
```

#### 文件2:
```python
evaluator = Evaluator(
    ot_method=args.ot_method, 
    ot_epsilon=args.ot_epsilon, 
    ot_max_iter=args.ot_max_iter, 
    n_samples=args.ot_samples
)
```

**关键差异**:
- 文件2允许通过CLI参数配置评估器
- 文件1使用默认参数

### 2.4 结果保存格式

#### 文件1:
- 保存 `wasserstein_summary.json`，包含 `W_in`, `W_out`, `W_total`
- 使用 `save_results()` 函数保存详细metrics

#### 文件2:
- 保存 `wasserstein_in_out.json`，包含 `W_in`, `W_out`, `W_total`
- 使用 `sim_wrapper.save_results()` 保存metrics

**关键差异**:
- 文件名不同（`wasserstein_summary.json` vs `wasserstein_in_out.json`）
- 保存方式不同（全局函数 vs 类方法）

---

## 三、潜在问题汇总

### 3.1 严重问题

#### 问题1: OOD评估使用错误的lead_time (文件1)
- **位置**: `simulation_code_iter_0_alpha.py` 第2594行
- **问题**: 使用优化后的参数（lead_time_L=2）来模拟OOD数据（lead_time=5）
- **影响**: W_out值会异常小，因为用错误的参数模拟了正确的数据
- **状态**: 文件2已修复

#### 问题2: 特征维度不一致
- **文件1**: 13维特征
- **文件2**: 10维特征
- **影响**: 两个文件训练出的模型不可直接比较
- **建议**: 统一特征维度

### 3.2 中等问题

#### 问题3: 训练窗口处理方式不同
- **文件1**: 每个轨迹有独立的结束时间（`holdout_ranges`）
- **文件2**: 所有轨迹使用统一的窗口（`train_window`）
- **影响**: 训练数据范围可能不同
- **建议**: 确认哪种方式更合理

#### 问题4: Batch处理能力
- **文件1**: 可能只处理单个样本
- **文件2**: 明确处理batch输入
- **影响**: 文件1在batch输入时可能出错
- **建议**: 文件1需要添加batch处理

### 3.3 轻微问题

#### 问题5: 评估器配置
- **文件1**: 使用默认参数
- **文件2**: 允许CLI配置
- **影响**: 文件1无法自定义评估器参数
- **建议**: 文件1应该支持CLI配置

#### 问题6: 结果保存格式
- **文件1**: `wasserstein_summary.json`
- **文件2**: `wasserstein_in_out.json`
- **影响**: 结果文件名不一致，可能造成混淆
- **建议**: 统一文件名

---

## 四、建议修复优先级

### 高优先级
1. ✅ **文件1的OOD评估问题** - 已修复（文件2）
2. ⚠️ **特征维度统一** - 需要决定使用哪种特征集
3. ⚠️ **训练窗口处理统一** - 需要确认哪种方式更合理

### 中优先级
4. ⚠️ **Batch处理能力** - 文件1需要添加batch处理
5. ⚠️ **评估器配置** - 文件1应该支持CLI配置

### 低优先级
6. ⚠️ **结果保存格式** - 统一文件名和保存方式

---

## 五、代码架构差异

### 文件1架构:
```
SBICalibrator (独立类)
  - __init__() 初始化参数空间
  - fit() 调用 _fit_sbi() 或 _fit_gradient_free()
  - _fit_sbi() 实现SBI训练
  - _simulation_wrapper() 包装模拟函数
  - _simulate_with_params_for_sbi() 执行模拟
```

### 文件2架构:
```
SNPECalibrator (继承Calibrator接口)
  - fit() 方法内部实现所有逻辑
  - 使用局部函数 features_from_observed()
  - 使用局部函数 sim_wrapper_fn()
  - 更符合Calibrator接口规范
```

**关键差异**:
- 文件1使用独立类，所有逻辑在类内部
- 文件2使用接口继承，更符合设计模式
- 文件2的架构更灵活，但代码更集中

---

## 六、总结

### 主要差异:
1. **校准器类型**: SBICalibrator vs SNPECalibrator
2. **特征计算**: 13维 vs 10维，计算方式不同
3. **训练窗口**: holdout_ranges vs train_window
4. **OOD评估**: 文件1有bug，文件2已修复
5. **Batch处理**: 文件2更健壮
6. **架构设计**: 文件2更符合接口规范

### 最关键的问题:
**文件1的OOD评估使用了错误的lead_time参数**，这会导致W_out值异常小。文件2已经修复了这个问题。

### 建议:
1. 统一特征维度（建议使用文件1的13维特征，更丰富）
2. 统一训练窗口处理方式（建议使用文件1的holdout_ranges，更灵活）
3. 文件1需要修复OOD评估问题（参考文件2的修复）
4. 文件1需要添加batch处理能力（参考文件2的实现）

