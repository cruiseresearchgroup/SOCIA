# TuRBO 领域知识指导优化总结

## 🎯 优化目标

针对口罩采用模拟的贝叶斯优化，我们实现了基于领域知识和风险评估的TuRBO增强版本，主要包含以下两个核心改进：

1. **领域知识指导的参数初始化** - 使用专家知识定义合理的参数区间
2. **风险感知的信赖域管理** - 基于参数风险评估的个性化信赖域长度

## 📊 领域知识指导的参数区间

### 高影响决策参数
```python
# 核心行为参数 (HIGH IMPACT)
'alpha': (-1.5, 1.0)      # 风险厌恶：适度负偏见到轻微正偏见
'gamma': (1.0, 2.0)       # 概率权重：决策制定的现实范围
'theta_f': (0.8, 1.5)     # 家庭影响：适度到强烈
'theta_w': (0.7, 1.3)     # 工作影响：适度（低于家庭）
'theta_c': (0.6, 1.2)     # 社区影响：适度（低于家庭/工作）
'beta_r': (-1.0, 1.0)     # 风险感知系数：适度范围
'beta_i': (-0.8, 0.8)     # 信息系数：适度范围
```

### 社会网络影响参数
```python
# 社会层权重 (MEDIUM-HIGH IMPACT)
'family': (0.3, 1.2)      # 家庭层：强烈但不极端的影响
'work_school': (0.2, 1.0) # 工作/学校层：适度影响
'community': (0.15, 0.8)  # 社区层：较弱但存在的影响
```

### 信息传播参数（高风险）
```python
# 信息传播率 (HIGH RISK - 可能导致不稳定)
'phi_family': (0.02, 0.15)     # 家庭信息传播：低到适度
'phi_work': (0.015, 0.12)      # 工作信息传播：略低
'phi_community': (0.01, 0.08)  # 社区信息传播：最低
'lambda_broadcast_base': (0.02, 0.1)           # 基础广播率：低
'lambda_broadcast_factor_after_day10': (1.2, 2.2) # 广播增长：适度
'rho_info_decay': (0.3, 0.7)   # 信息衰减：适度持续性
```

### 人口统计学效应
```python
# 年龄组效应 (MEDIUM IMPACT)
'age_0': (-0.8, 0.8)      # 年轻年龄效应：适度范围
'age_1': (-0.6, 0.6)      # 中年年龄效应：较小范围（参考组）
'age_2': (-1.0, 1.0)      # 老年年龄效应：较大范围（更多变）

# 职业组效应 (MEDIUM IMPACT)
'occ_0': (-0.8, 0.8)      # 职业0效应：适度范围
'occ_1': (-0.6, 0.6)      # 职业1效应：较小范围
'occ_2': (-1.0, 1.0)      # 职业2效应：较大范围
```

## 🎲 风险评估和信赖域策略

### 参数风险分级

#### 🔴 高风险参数（信息传播）
- `phi_family`, `phi_work`, `phi_community`
- `lambda_broadcast_base`, `lambda_broadcast_factor_after_day10`
- **初始信赖域长度**: 0.15（非常保守）
- **扩展上限**: 0.3
- **收缩下限**: 0.05

#### 🟠 中高风险参数（核心决策）
- `alpha`, `gamma`, `beta_r`, `beta_i`
- **初始信赖域长度**: 0.25（保守）
- **扩展上限**: 0.4
- **收缩下限**: 0.08

#### 🟡 中等风险参数（社会影响）
- `theta_f`, `theta_w`, `theta_c`, `family`, `work_school`, `community`, `rho_info_decay`
- **初始信赖域长度**: 0.35（适度）
- **扩展上限**: 0.5
- **收缩下限**: 0.12

#### 🟢 低风险参数（噪声和人口统计学）
- `tau`, `age_0`, `age_1`, `age_2`, `occ_0`, `occ_1`, `occ_2`
- **初始信赖域长度**: 0.45（更多探索）
- **扩展上限**: 0.6
- **收缩下限**: 0.15

## ⚙️ 保守扩展与激进收缩策略

### 🐌 保守扩展策略
- **成功门槛**: 5次成功（vs 原来的3次）
- **扩展因子**: 1.5倍（vs 原来的2.0倍）
- **最大信赖域**: 0.8（vs 原来的1.0）
- **风险限制**: 高风险参数限制更严格的扩展

### ⚡ 激进收缩策略
- **失败门槛**: 7次失败（vs 原来的10次）
- **收缩因子**: 0.4倍（vs 原来的0.5倍）
- **最小信赖域**: 基于风险级别的不同下限
- **风险保护**: 高风险参数收缩到更小的最小值

## 🚀 初始化改进

### 智能中心点选择
- 使用领域知识指导的bounds进行初始采样
- 从每个参数的安全中间区域（60%范围）进行采样
- 避免参数空间的极端区域

### 参数特定信赖域
- 每个参数根据其风险级别有独立的信赖域长度
- 信赖域更新时考虑参数的稳定性要求
- 高风险参数始终保持更小的探索范围

## 📈 预期优势

1. **🎯 更稳定的收敛**: 避免在不稳定的参数区域过度探索
2. **⚡ 更快的收敛**: 在有希望的区域开始搜索
3. **🛡️ 更鲁棒的优化**: 高风险参数的保守处理
4. **🔍 更精确的局部搜索**: 参数特定的信赖域长度
5. **📚 领域知识整合**: 充分利用口罩采用行为的先验知识

## 💻 使用示例

```python
# 增强的TuRBO配置
calibrator = get_calibrator("bo", 
    n_trials=300, 
    acquisition_function='EI', 
    kernel_type='RBF', 
    random_state=42,
    metric_type='composite', 
    metric_weights={'rmse': 0.4, 'mae': 0.2, 'brier': 0.2, 'transition': 0.2},
    normalize_metrics=True, 
    fast_mode_iterations=50,
    use_turbo=True,
    turbo_config={
        'trust_region_size': 0.6,       # 更保守的初始大小
        'success_tolerance': 5,         # 需要更多成功次数（保守扩展）
        'failure_tolerance': 7,         # 需要更少失败次数（激进收缩）
        'expansion_factor': 1.5,        # 保守扩展因子
        'contraction_factor': 0.4,      # 激进收缩因子
        'min_trust_region': 1e-8,       # 最小信赖域大小
        'max_trust_region': 0.8,        # 更保守的最大值
        'domain_informed': True,        # 使用领域知识初始化
        'risk_aware_lengths': True      # 使用参数特定的信赖域长度
    }
)
```

## 🔬 技术实现亮点

1. **`_define_domain_informed_bounds()`**: 基于口罩采用文献的专家知识边界
2. **`_define_parameter_risk_levels()`**: 基于参数对模型稳定性影响的风险评估
3. **`_get_initial_trust_region_lengths()`**: 风险感知的初始信赖域长度
4. **`_generate_domain_informed_initial_center()`**: 智能初始中心点生成
5. **`_update_turbo_state()`**: 保守扩展和激进收缩的信赖域更新
6. **`_generate_turbo_bounds()`**: 参数特定的信赖域边界生成

这些改进使TuRBO能够在复杂的口罩采用模拟任务中实现更稳定、更高效的参数优化。
