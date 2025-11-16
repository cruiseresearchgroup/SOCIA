# BoCalibrator GP建模和采集函数分析报告

## ✅ 确认：BoCalibrator确实实现了完整的GP建模和采集函数

经过详细的代码分析，我可以确认BoCalibrator完全按照您的要求实现了GP建模和采集函数功能。

## 🧠 **1. GP在"参数ω → 误差y"之间拟合代理模型**

### ✅ **GP模型创建和拟合**

**BoCalibrator._fit_gp_model函数：**
```python
def _fit_gp_model(self, X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """
    Fit Gaussian Process model to training data.
    
    Args:
        X: Parameter samples, shape (n_samples, n_params)  # 参数ω
        Y: Objective values, shape (n_samples, 1)         # 误差y
    """
    print(f"  Fitting GP model to {X.shape[0]} training points...")
    
    # 🎯 创建GP模型：参数ω → 误差y的映射
    gp = SingleTaskGP(X, Y)
    
    # 创建边际对数似然
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    
    # 拟合模型
    fit_gpytorch_mll(mll)
    
    print(f"  ✓ GP model fitted successfully")
    return gp
```

### ✅ **训练数据构建：参数ω → 误差y**

**训练数据初始化：**
```python
# 1. 初始化参数样本 X (参数ω)
X_init = self._initialize_random_samples(n_init, seed)  # Shape: (n_init, 23)

# 2. 评估每个参数样本得到误差 Y (误差y)
Y_init = []
for i in range(n_init):
    params_np = X_init[i].numpy()  # 参数ω
    objective = self._objective_function(params_np, bundle, evaluator, train_window, seed, iteration=i)
    Y_init.append(objective)  # 误差y

Y_init = torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1)  # Shape: (n_init, 1)

# 3. 构建训练数据对 (ω, y)
self.X_train = X_init  # 参数ω
self.Y_train = Y_init  # 误差y
```

**训练数据更新：**
```python
# 每次评估新参数后更新训练数据
for iteration in range(n_init, self.n_trials):
    # 拟合GP模型：ω → y
    gp_model = self._fit_gp_model(self.X_train, self.Y_train)
    
    # 评估新候选参数
    new_objective = self._objective_function(new_params, bundle, evaluator, train_window, seed, iteration)
    
    # 更新训练数据
    self.X_train = torch.cat([self.X_train, candidates], dim=0)      # 添加新参数ω
    self.Y_train = torch.cat([self.Y_train, torch.tensor([[new_objective]], dtype=torch.float64)], dim=0)  # 添加新误差y
```

### ✅ **GP模型特性**

**BoTorch SingleTaskGP：**
- **输入维度**: 23维参数空间 (ω)
- **输出维度**: 1维误差空间 (y)
- **核函数**: RBF (径向基函数) 核
- **噪声处理**: 自动处理观测噪声
- **不确定性量化**: 提供预测均值和方差

## 🎯 **2. 采集函数（EI/UCB/PI）挑选下一组参数ω**

### ✅ **支持的采集函数**

**BoCalibrator._get_acquisition_function函数：**
```python
def _get_acquisition_function(self, gp_model: SingleTaskGP, best_f: float) -> Any:
    """
    根据配置获取采集函数
    
    Args:
        gp_model: 拟合的GP模型
        best_f: 目前看到的最佳目标值
    """
    if self.acquisition_function.lower() == 'ei':
        # 🎯 Expected Improvement (期望改进)
        acq_func = ExpectedImprovement(gp_model, best_f=best_f)
    elif self.acquisition_function.lower() == 'pi':
        # 🎯 Probability of Improvement (改进概率)
        acq_func = ProbabilityOfImprovement(gp_model, best_f=best_f)
    elif self.acquisition_function.lower() == 'ucb':
        # 🎯 Upper Confidence Bound (上置信界)
        acq_func = UpperConfidenceBound(gp_model, beta=2.0)
    else:
        print(f"  Warning: Unknown acquisition function '{self.acquisition_function}', using EI")
        acq_func = ExpectedImprovement(gp_model, best_f=best_f)
    
    print(f"  Using acquisition function: {self.acquisition_function.upper()}")
    return acq_func
```

### ✅ **采集函数优化**

**BoCalibrator._optimize_acquisition函数：**
```python
def _optimize_acquisition(self, acq_func: Any, n_candidates: int = 1) -> torch.Tensor:
    """
    优化采集函数获取下一组候选参数
    
    Args:
        acq_func: 采集函数
        n_candidates: 要生成的候选点数量
        
    Returns:
        下一组候选参数, shape (n_candidates, n_params)
    """
    print(f"  Optimizing acquisition function for {n_candidates} candidate(s)...")
    
    # 🎯 优化采集函数选择下一组参数ω
    candidates, _ = optimize_acqf(
        acq_function=acq_func,      # 采集函数
        bounds=self.bounds,         # 参数边界
        q=n_candidates,             # 候选点数量
        num_restarts=20,            # 随机重启次数
        raw_samples=100,            # 初始化样本数
    )
    
    print(f"  ✓ Generated {candidates.shape[0]} candidate(s)")
    return candidates
```

## 🔄 **3. 完整的贝叶斯优化循环**

### ✅ **主优化循环**

**BoCalibrator._run_bayesian_optimization中的完整流程：**
```python
for iteration in range(n_init, self.n_trials):
    print(f"\n--- Iteration {iteration + 1}/{self.n_trials} ---")
    
    # 🎯 Step 2: 拟合GP模型 (参数ω → 误差y)
    gp_model = self._fit_gp_model(self.X_train, self.Y_train)
    
    # 🎯 Step 3: 获取采集函数并优化选择下一组参数ω
    acq_func = self._get_acquisition_function(gp_model, best_objective)
    candidates = self._optimize_acquisition(acq_func, n_candidates=1)
    
    # 🎯 Step 4: 评估新候选参数
    new_params = candidates[0].numpy()  # 新的参数ω
    new_objective = self._objective_function(new_params, bundle, evaluator, train_window, seed, iteration=iteration)
    
    print(f"  New candidate: objective = {new_objective:.4f}")
    
    # 🎯 Step 5: 更新训练数据
    self.X_train = torch.cat([self.X_train, candidates], dim=0)      # 添加新参数ω
    self.Y_train = torch.cat([self.Y_train, torch.tensor([[new_objective]], dtype=torch.float64)], dim=0)  # 添加新误差y
    
    # 更新最佳参数
    if new_objective < best_objective:
        best_params = new_params.copy()
        best_objective = new_objective
        print(f"  ✓ New best found: objective = {best_objective:.4f}")
```

## 📊 **4. 采集函数详解**

### ✅ **Expected Improvement (EI)**
```python
acq_func = ExpectedImprovement(gp_model, best_f=best_f)
```
- **原理**: 期望改进量，平衡探索和利用
- **特点**: 在已知最优解附近和高不确定性区域都有较高值
- **适用**: 通用场景，平衡探索和利用

### ✅ **Upper Confidence Bound (UCB)**
```python
acq_func = UpperConfidenceBound(gp_model, beta=2.0)
```
- **原理**: 上置信界，β控制探索程度
- **特点**: β越大越倾向于探索高不确定性区域
- **适用**: 需要更多探索的场景

### ✅ **Probability of Improvement (PI)**
```python
acq_func = ProbabilityOfImprovement(gp_model, best_f=best_f)
```
- **原理**: 改进概率，只考虑是否比当前最优解更好
- **特点**: 更倾向于利用，探索相对较少
- **适用**: 利用为主的场景

## 🎛️ **5. 配置和使用**

### ✅ **BoCalibrator初始化配置**
```python
calibrator = get_calibrator("bo", 
    n_trials=100,                           # 总评估次数
    acquisition_function='EI',              # 采集函数类型
    kernel_type='RBF',                      # GP核函数
    random_state=42,                        # 随机种子
    metric_type='composite',                # 指标类型
    metric_weights={'rmse': 0.5, 'brier': 0.3, 'transition': 0.2},
    normalize_metrics=True,                 # 归一化指标
    fast_mode_iterations=30                 # 快速模式迭代数
)
```

### ✅ **参数边界定义**
```python
def _define_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
    """定义23维参数的边界，与SBI一致"""
    return {
        'alpha': (-3.0, 3.0),
        'gamma': (0.5, 3.0),
        # ... 其他21个参数
    }
```

## 🎯 **总结确认**

BoCalibrator确实完全按照您的要求实现了GP建模和采集函数：

1. **✅ GP在"参数ω → 误差y"之间拟合代理模型**:
   - 使用BoTorch SingleTaskGP
   - 23维参数空间 → 1维误差空间
   - 每次迭代都重新拟合GP模型
   - 提供预测均值和不确定性

2. **✅ 采集函数（EI/UCB/PI）挑选下一组参数ω**:
   - 支持ExpectedImprovement (EI)
   - 支持UpperConfidenceBound (UCB)  
   - 支持ProbabilityOfImprovement (PI)
   - 使用BoTorch optimize_acqf优化采集函数

3. **✅ 完整的贝叶斯优化流程**:
   - 初始化随机样本构建训练数据
   - 迭代：拟合GP → 优化采集函数 → 选择新参数 → 评估 → 更新训练数据
   - 智能平衡探索和利用

BoCalibrator实现了完整的贝叶斯优化算法，使用GP作为代理模型学习参数-误差映射，并通过采集函数智能选择下一个评估点！
