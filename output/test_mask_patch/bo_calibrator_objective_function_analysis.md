# BoCalibrator目标函数分析报告

## ✅ 确认：BoCalibrator完全符合要求

经过详细分析，BoCalibrator的实现完全符合您提出的要求：

### 1. ✅ 定义目标函数

**输入：参数向量 ω（d维）**
- BoCalibrator优化23维参数向量：`['alpha', 'gamma', 'theta_f', 'theta_w', 'theta_c', 'beta_r', 'beta_i', 'family', 'work_school', 'community', 'phi_family', 'phi_work', 'phi_community', 'lambda_broadcast_base', 'lambda_broadcast_factor_after_day10', 'rho_info_decay', 'tau', 'age_0', 'age_1', 'age_2', 'occ_0', 'occ_1', 'occ_2']`
- 参数边界与SBI校准器完全一致

**输出：误差指标（RMSE/MAE/...）**
- 支持多种误差指标：RMSE, MAE, Brier, TransitionFit
- 支持组合指标：`0.5×RMSE + 0.3×Brier + 0.2×TransitionFit`
- 所有指标都进行归一化处理，统一为"越小越好"的方向

### 2. ✅ 常见做法：只在train data上拟合/最小化误差

**训练数据使用：**
```python
# 在BoCalibrator._objective_function中
def _objective_function(self, params: np.ndarray, bundle, evaluator, 
                       train_window: Tuple[int, int], seed: int, iteration: int = 0) -> float:
    # ...
    # 使用train_window进行评估
    result = evaluator(None, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
```

**训练窗口定义：**
```python
# 在main函数中
train_window = (1, train_end_idx)  # 只使用训练数据
```

**evaluate_params函数确保在训练窗口上评估：**
```python
def evaluate_params(simulator, params: FittedParams, window) -> Dict[str, Any]:
    # ...
    train_start, train_end = train_window
    # 在训练窗口上运行仿真
    result = evaluate_on_validation(
        wearing, neighbors, risk, age_oh, occ_oh,
        age_cat_names, occ_cat_names, legacy_params, 
        train_start, train_end, cfg.k_runs  # 使用train_start, train_end
    )
```

### 3. ✅ 在validation data上报告泛化结果

**验证数据评估：**
```python
# 在main函数中，校准完成后
# 在validation窗口上评估泛化性能
metrics = evaluate_on_validation(
    wearing=wearing,
    neighbors=neighbors,
    risk=risk_perception,
    age_oh=age_oh,
    occ_oh=occ_oh,
    age_cat_names=age_cat_names,
    occ_cat_names=occ_cat_names,
    params=params,
    val_start_idx=val_start_idx,    # 验证数据开始
    val_end_idx=val_end_idx,        # 验证数据结束
    k_runs=cfg.k_runs,
)
```

**数据划分：**
```python
# 训练/验证数据划分
train_end_idx, val_start_idx, val_end_idx = build_train_validation_splits(days, cfg.val_split_ratio)
train_window = (1, train_end_idx)        # 训练窗口：1 到 train_end_idx
# 验证窗口：val_start_idx 到 val_end_idx
```

## 📊 BoCalibrator的目标函数特性

### 参数空间
- **维度**: 23维参数空间
- **边界**: 与SBI校准器完全一致
- **类型**: 连续参数空间，支持BoTorch优化

### 目标函数
```python
def _objective_function(self, params: np.ndarray, bundle, evaluator, 
                       train_window: Tuple[int, int], seed: int, iteration: int = 0) -> float:
    """
    输入: 23维参数向量 params
    输出: 标量目标值（越小越好）
    """
    # 1. 转换参数格式
    fitted_params = self._sample_to_fitted_params(params, self.param_names, seed, train_window)
    
    # 2. 在训练窗口上运行仿真
    result = evaluator(None, fitted_params, (wearing, neighbors, risk, age_oh, occ_oh, train_window, cfg))
    
    # 3. 计算目标值（支持多种指标）
    objective_value = self._get_objective_value(result, iteration)
    
    return objective_value
```

### 支持的指标类型
1. **单一指标**: RMSE, MAE, Brier, TransitionFit
2. **组合指标**: 加权组合多个指标
3. **自适应指标**: 根据优化阶段自动选择指标
4. **加权RMSE**: 考虑不确定性的RMSE

### 归一化处理
```python
def _normalize_metric(self, value: float, metric_name: str) -> float:
    """
    将所有指标归一化到[0,1]范围
    """
    metric_ranges = {
        'rmse': (0.0, 1.0),
        'mae': (0.0, 1.0), 
        'brier': (0.0, 0.25),
        'transition': (0.0, 1.0)
    }
    # 归一化处理...
```

## 🔄 与LogitHead/SBI的一致性

### 相同的接口设计
- 所有校准器都实现相同的`fit`接口
- 使用相同的`evaluator`函数
- 相同的`train_window`和验证评估流程

### 相同的数据划分
- 使用相同的`build_train_validation_splits`函数
- 相同的训练/验证窗口定义
- 相同的验证评估方法

### 相同的评估指标
- RMSE, MAE, Brier, TransitionFit
- 相同的统计计算方法
- 相同的置信区间计算

## 🎯 总结

BoCalibrator完全符合您的要求：

1. **✅ 定义目标函数**: 23维参数向量 → 误差指标
2. **✅ 训练数据拟合**: 只在train_window上最小化误差
3. **✅ 验证数据报告**: 在validation数据上报告泛化结果
4. **✅ 与现有校准器一致**: 与LogitHead/SBI使用相同的接口和流程

BoCalibrator提供了更强大的优化能力，同时保持了与现有框架的完全兼容性。
