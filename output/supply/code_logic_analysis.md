# 代码逻辑全面梳理报告

## 文件：`output/supply/simulation_code_iter_0_alpha_poisson_improved.py`

---

## 一、代码总体功能

### 1.1 核心任务
**这是一个Supply Chain（供应链）仿真校准和评估系统**

- **任务类型**：Supply Chain Simulation（单阶段Beer Game系统）
- **主要功能**：
  1. 加载供应链轨迹数据（train/val/test/OOD）
  2. 使用SBI（Simulation-Based Inference）校准仿真器参数
  3. 在验证集和测试集上评估仿真器性能
  4. 计算Wasserstein距离（W_in和W_out）作为主要评估指标

### 1.2 核心组件

1. **BeerGameSimulator**：单阶段Beer Game仿真器
   - 管理inventory（库存）、backlog（积压订单）、pipeline（在途货物）
   - 支持Poisson需求模型
   - 可配置lead_time（提前期）

2. **SNPECalibrator**：SBI校准器
   - 使用SNPE（Sequential Neural Posterior Estimation）进行参数校准
   - 支持1000次模拟训练
   - 从后验分布中采样5000个样本

3. **Evaluator**：评估器
   - 计算Wasserstein距离（与GSIM对齐）
   - 使用2D状态向量（inventory, backlog）
   - 使用`ot.emd()`计算真正的Wasserstein距离

---

## 二、数据加载和对齐检查

### 2.1 数据加载逻辑（load_data函数，第248-274行）

```python
def load_data(
    data_dir: str,
    train_file: str,
    val_file: str,
    test_file: Optional[str],
    metadata_file: str,
    test_ood_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    # 加载train_data.csv, val_data.csv, test_data.csv, test_ood_data.csv
    # 加载metadata.json
```

**数据文件：**
- `train_data.csv`：训练数据（lead_time=2）
- `val_data.csv`：验证数据（lead_time=2）
- `test_data.csv`：测试数据（lead_time=2，In-Distribution）
- `test_ood_data.csv`：OOD测试数据（lead_time=5，Out-of-Distribution）
- `metadata.json`：元数据（包含数据文件路径、lead_times等信息）

**与修改前版本对比：**
- ✅ **完全一致**：`simulation_code_iter_0_alpha.py`也使用相同的`load_data`函数
- ✅ **数据文件相同**：都从`data_fitting/supply_data/`加载相同的数据文件
- ✅ **支持OOD数据**：两个版本都支持`test_ood_file`参数

### 2.2 轨迹构建逻辑（build_trajectories函数，第277-370行）

```python
def build_trajectories(df: pd.DataFrame, metadata: Dict[str, Any]) -> List[TrajectoryData]:
    # 从DataFrame构建TrajectoryData列表
    # 提取：trajectory_id, time_step, inventory, backlog, pipeline_len, actions
```

**关键字段：**
- `trajectory_id`：轨迹ID
- `t`：时间步数组
- `actions`：动作数组（固定为4）
- `inventory_obs`：观测的库存
- `backlog_obs`：观测的积压订单
- `pipeline_len_obs_counts`：观测的管道长度

**与修改前版本对比：**
- ✅ **完全一致**：两个版本使用相同的`build_trajectories`函数
- ✅ **数据结构相同**：都构建`TrajectoryData`对象

---

## 三、仿真器检查

### 3.1 BeerGameSimulator（第783-930行）

```python
class BeerGameSimulator:
    """
    Simulator orchestrator for single-stage Beer Game system.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = dict(params)
    
    def rollout(self, trajectories: List[TrajectoryData]) -> SimulationResults:
        # 使用当前参数在轨迹上运行仿真
        # 返回SimulationResults（包含inventory_sim, backlog_sim, pipeline_len_sim等）
```

**参数结构：**
```python
params = {
    "supply": {
        "lead_time_L": int,  # 提前期（1-8）
        "arrival_flag": int  # 到达约定（0或1）
    },
    "demand": {
        "demand_family": "Poisson",
        "poisson_lambda": float  # Poisson需求参数（0.1-20.0）
    }
}
```

**与修改前版本对比：**
- ✅ **完全一致**：两个版本使用相同的`BeerGameSimulator`类
- ✅ **参数结构相同**：都使用相同的参数结构

---

## 四、校准流程检查

### 4.1 校准器类型

**当前版本（improved）：**
- 使用`SNPECalibrator`（继承自`Calibrator`接口）
- 从配置文件读取参数（`snpe_config.json`）
- 支持1000次模拟训练

**修改前版本（alpha.py）：**
- 使用`SBICalibrator`（独立类）
- 从CLI参数读取配置
- 支持1000次模拟训练

**关键差异：**
- 架构不同（接口继承 vs 独立类）
- 但功能相同（都使用SBI进行参数校准）

### 4.2 训练窗口

**当前版本：**
```python
calib_window = (0, 48)  # 默认训练窗口：时间步0-48
```

**修改前版本：**
```python
train_trajs, holdout_ranges = holdout_split(train_trajectories, train_end_inclusive=48)
# 使用holdout_ranges，每个轨迹有独立的结束时间
```

**关键差异：**
- 当前版本：使用统一的训练窗口（0-48）
- 修改前版本：使用holdout_ranges（每个轨迹独立）
- ⚠️ **可能影响训练数据范围**

---

## 五、评估流程检查

### 5.1 W_in评估（In-Distribution）

**当前版本（第2640-2648行）：**
```python
if len(test_trajectories) > 0:
    metrics_test = sim_wrapper.evaluate(split="test")
    res_test = simulator.rollout(test_trajectories)  # 使用优化后的simulator
    w_in = evaluator.compute_joint_wasserstein_per_t(res_test, ...)
    w_in_val = float(w_in)
```

**修改前版本（第2575-2595行）：**
```python
if len(test_trajectories) > 0:
    sim_results_test = simulator.rollout(test_trajectories)  # 使用优化后的simulator
    metrics_test = evaluator.compute_metrics(sim_results_test, ...)
    w_in = metrics_test.get("distributional", {}).get("Wasserstein_per_t", 0.0)
```

**关键差异：**
- 当前版本：直接调用`compute_joint_wasserstein_per_t`
- 修改前版本：从`compute_metrics`的结果中提取
- ⚠️ **评估方法不同，但应该计算相同的指标**

### 5.2 W_out评估（Out-of-Distribution）

**当前版本（第2653-2666行）：**
```python
if len(test_ood_trajectories) > 0:
    # 创建临时simulator，设置lead_time_L=5
    ood_params_dict = fitted.to_dict()
    ood_params_dict["module_params"]["supply"]["lead_time_L"] = 5
    ood_simulator = BeerGameSimulator(ood_params_dict["module_params"])
    res_ood = ood_simulator.rollout(test_ood_trajectories)
    w_out = evaluator.compute_joint_wasserstein_per_t(res_ood, ...)
```

**修改前版本（第2591-2605行）：**
```python
if len(test_trajectories_ood) > 0:
    sim_results_test_ood = simulator.rollout(test_trajectories_ood)
    # ❌ 问题：使用优化后的simulator（lead_time_L=2）来模拟OOD数据（lead_time=5）
    metrics_test_ood = evaluator.compute_metrics(sim_results_test_ood, ...)
```

**关键差异：**
- ✅ **当前版本已修复**：使用`lead_time=5`的simulator来模拟OOD数据
- ❌ **修改前版本有bug**：使用`lead_time=2`的simulator来模拟OOD数据

### 5.3 Wasserstein距离计算

**当前版本（compute_joint_wasserstein_per_t，第958-1045行）：**
```python
def compute_joint_wasserstein_per_t(self, results: SimulationResults, ...):
    # 1. 收集所有轨迹的所有时间步的inventory和backlog
    # 2. 组合成2D状态向量（inventory, backlog）
    # 3. 使用ot.emd()计算真正的Wasserstein距离
    # 4. 与GSIM的wasserstein_distance_nd()对齐
```

**修改前版本（_compute_wass_mmd_per_t，第1814-1900行）：**
```python
def _compute_wass_mmd_per_t(self, results: SimulationResults, ...):
    # 1. 在每个时间步计算Wasserstein距离
    # 2. 对每个时间步采样200个样本
    # 3. 使用1D Wasserstein距离（分别计算inventory、backlog、pipeline_len）
    # 4. 取所有时间步的平均值
```

**关键差异：**
- ✅ **当前版本已修复**：
  - 使用所有数据（不采样）
  - 使用2D状态向量（inventory, backlog）
  - 使用`ot.emd()`计算真正的Wasserstein距离
  - 与GSIM对齐

- ❌ **修改前版本的问题**：
  - 在每个时间步采样200个样本
  - 使用1D Wasserstein距离
  - 逐时间步计算然后取平均值

---

## 六、数据集对齐检查

### 6.1 数据文件

**当前版本和修改前版本都使用：**
- `data_fitting/supply_data/train_data.csv`
- `data_fitting/supply_data/val_data.csv`
- `data_fitting/supply_data/test_data.csv`
- `data_fitting/supply_data/test_ood_data.csv`
- `data_fitting/supply_data/metadata.json`

**✅ 数据集完全对齐**

### 6.2 数据加载逻辑

**两个版本都使用：**
- 相同的`load_data`函数
- 相同的`build_trajectories`函数
- 相同的`TrajectoryData`数据结构

**✅ 数据加载逻辑完全对齐**

---

## 七、评估指标对齐检查

### 7.1 主要评估指标

**当前版本：**
- `W_in`：In-Distribution Wasserstein距离（test set with lead_time=2）
- `W_out`：Out-of-Distribution Wasserstein距离（test set with lead_time=5）
- `W_total`：W_in和W_out的平均值

**修改前版本：**
- `W_in`：从`metrics_test["distributional"]["Wasserstein_per_t"]`提取
- `W_out`：从`metrics_test_ood["distributional"]["Wasserstein_per_t"]`提取
- `W_total`：W_in和W_out的平均值

**关键差异：**
- ⚠️ **计算方法不同**：
  - 当前版本：直接调用`compute_joint_wasserstein_per_t`（与GSIM对齐）
  - 修改前版本：从`compute_metrics`的结果中提取（可能使用不同的计算方法）

### 7.2 Wasserstein距离计算方法

**当前版本（与GSIM对齐）：**
```python
# 1. 收集所有轨迹的所有时间步的inventory和backlog
# 2. 组合成2D状态向量（inventory, backlog）
sim_states = np.column_stack([inv_sim_aligned, b_sim_aligned])
obs_states = np.column_stack([inv_obs_aligned, b_obs_aligned])
# 3. 使用ot.emd()计算真正的Wasserstein距离
cost_matrix = ot.dist(sim_states, obs_states, metric="euclidean")
transport_plan = ot.emd(a, b, cost_matrix)
wass = float(np.sum(cost_matrix * transport_plan))
```

**修改前版本：**
```python
# 1. 在每个时间步计算Wasserstein距离
# 2. 对每个时间步采样200个样本
# 3. 使用1D Wasserstein距离（分别计算inventory、backlog、pipeline_len）
# 4. 取所有时间步的平均值
```

**关键差异：**
- ✅ **当前版本已修复**：与GSIM的`wasserstein_distance_nd()`对齐
- ❌ **修改前版本有问题**：使用不同的计算方法

---

## 八、评估数据对齐检查

### 8.1 W_in评估数据

**当前版本：**
- 使用`test_trajectories`（从`test_data.csv`加载，lead_time=2）
- 使用优化后的`simulator`（lead_time_L=2）来模拟
- 与真实的观测数据比较

**修改前版本：**
- 使用`test_trajectories`（从`test_data.csv`加载，lead_time=2）
- 使用优化后的`simulator`（lead_time_L=2）来模拟
- 与真实的观测数据比较

**✅ 评估数据对齐**

### 8.2 W_out评估数据

**当前版本：**
- 使用`test_ood_trajectories`（从`test_ood_data.csv`加载，lead_time=5）
- 使用临时`ood_simulator`（lead_time_L=5，其他参数优化后）来模拟
- 与真实的OOD观测数据比较

**修改前版本：**
- 使用`test_trajectories_ood`（从`test_ood_data.csv`加载，lead_time=5）
- 使用优化后的`simulator`（lead_time_L=2）来模拟 ❌ **BUG**
- 与真实的OOD观测数据比较

**关键差异：**
- ✅ **当前版本已修复**：使用`lead_time=5`的simulator
- ❌ **修改前版本有bug**：使用`lead_time=2`的simulator

---

## 九、总结

### 9.1 任务类型
✅ **确认：这是Supply Chain任务**
- 使用`BeerGameSimulator`（单阶段Beer Game系统）
- 管理inventory、backlog、pipeline
- 支持Poisson需求模型

### 9.2 数据集对齐
✅ **数据集完全对齐**
- 使用相同的数据文件（train_data.csv, val_data.csv, test_data.csv, test_ood_data.csv）
- 使用相同的数据加载逻辑（`load_data`和`build_trajectories`）
- 使用相同的数据结构（`TrajectoryData`）

### 9.3 评估指标对齐
⚠️ **评估指标计算方法不同，但当前版本已修复并与GSIM对齐**
- 当前版本：使用`compute_joint_wasserstein_per_t`（与GSIM对齐）
- 修改前版本：使用`_compute_wass_mmd_per_t`（不同的计算方法）

### 9.4 评估数据对齐
✅ **评估数据对齐，但当前版本修复了OOD评估的bug**
- W_in：两个版本都使用相同的test_trajectories
- W_out：当前版本修复了使用错误lead_time的问题

### 9.5 关键修复
1. ✅ **Wasserstein距离计算**：与GSIM对齐（使用ot.emd()，2D状态向量，所有数据）
2. ✅ **OOD评估**：修复了使用错误lead_time的问题
3. ✅ **数据对齐**：使用轨迹ID交集确保对齐

### 9.6 潜在问题
1. ⚠️ **训练窗口处理**：当前版本使用统一窗口，修改前版本使用holdout_ranges
2. ⚠️ **W_out仍然很小**：可能需要进一步检查参数设置和数据对齐

