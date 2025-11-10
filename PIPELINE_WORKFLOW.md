# SOCIA 项目 Pipeline Workflow 总览

## 📋 目录结构

```
SOCIA-1/
├── main.py                          # 主入口
├── orchestration/
│   ├── workflow_manager.py          # 主工作流管理器（支持 full/lite/blueprint/calibrasim）
│   ├── container.py                 # 依赖注入容器（管理所有 agents）
│   └── Calibrasim_workflow_manager.py  # CalibraSim 专用工作流（备用）
├── agents/                          # 所有 Agent 实现
│   ├── task_understanding/          # 标准任务理解
│   ├── task_understanding_odd/      # ODD 模式任务理解
│   ├── data_analysis/               # 标准数据分析
│   ├── data_analysis_odd/           # ODD 模式数据分析
│   ├── model_planning/              # 标准模型规划
│   ├── code_generation/             # 标准代码生成
│   ├── code_generation_odd/         # ODD 模式代码生成
│   ├── code_generation_calibrasim/  # CalibraSim 模式代码生成
│   ├── code_verification/           # 代码验证
│   ├── simulation_execution/        # 仿真执行
│   ├── result_evaluation/           # 结果评估
│   ├── feedback_generation/         # 反馈生成
│   └── iteration_control/           # 迭代控制
└── config.yaml                      # 全局配置
```

---

## 🚀 主流程（Main Pipeline）

### 1. 入口点：`main.py`

**命令行参数**：
```bash
python main.py \
  --task "任务描述" \
  --task-file "任务文件.json" \
  --output "./output" \
  --mode [lite|medium|full|blueprint|calibrasim] \
  --auto [True|False] \
  --iterations 3 \
  --selfloop 0
```

**执行流程**：
1. 解析命令行参数
2. 设置日志系统
3. 检查 API Key
4. 初始化依赖注入容器（`AgentContainer`）
5. 创建 `WorkflowManager` 并运行工作流

---

## 🔄 工作流模式（Workflow Modes）

### Mode 1: **full**（完整模式）

**流程步骤**：
```
1. Task Understanding（任务理解）
   └─> 解析任务描述，生成 task_spec.json

2. Data Analysis（数据分析）
   └─> 分析数据文件，生成 data_analysis.json

3. Model Planning（模型规划）
   └─> 基于任务和数据，生成 model_plan.json

4. Code Generation（代码生成）
   └─> 生成 simulation_code_iter_{N}.py

5. Code Verification（代码验证）
   └─> 验证代码语法和逻辑

6. Simulation Execution（仿真执行）
   └─> 执行仿真，生成结果

7. Result Evaluation（结果评估）
   └─> 评估仿真结果

8. Feedback Generation（反馈生成）
   └─> 生成改进建议

9. Iteration Control（迭代控制）
   └─> 决定是否继续迭代
```

**使用的 Agents**：
- `task_understanding`
- `data_analysis`
- `model_planning`
- `code_generation`
- `code_verification`
- `simulation_execution`
- `result_evaluation`
- `feedback_generation`
- `iteration_control`

---

### Mode 2: **lite**（精简模式）

**流程步骤**：
```
1. Task Spec（简化任务规范）
   └─> 直接使用任务描述作为 task_spec

2. Code Generation（代码生成）
   └─> 跳过数据分析和模型规划，直接生成代码

3. Code Verification（轻量验证）
   └─> 不使用 Docker sandbox，仅语法检查

4. Simulation Execution（子进程执行）
   └─> 使用 subprocess 执行仿真

5. Result Evaluation（轻量评估）
   └─> 不对比真实数据

6. Feedback Generation + Iteration Control
```

**使用的 Agents**：
- `code_generation`（使用 lite template）
- `code_verification`（轻量模式）
- `simulation_execution`（lite 模式）
- `result_evaluation`（轻量模式）
- `feedback_generation`
- `iteration_control`

---

### Mode 3: **blueprint**（蓝图模式）

**流程步骤**：
```
1. Task Understanding（ODD 协议）
   └─> 使用 ODD 协议解析任务，初始化 Blueprint

2. Code Generation（基于蓝图）
   └─> 使用 Blueprint 信息生成代码

3. Code Verification（轻量验证）

4. Simulation Execution（子进程执行）

5. Result Evaluation（轻量评估）

6. Blueprint Update（更新蓝图）
   └─> 根据结果更新 Blueprint 状态

7. Feedback Generation + Iteration Control
```

**特点**：
- 使用 **ODD（Overview, Design concepts, Details）协议**
- Blueprint 作为跨 Agent 共享数据存储
- 每次迭代更新 Blueprint，累积信息

**使用的 Agents**：
- `task_understanding`（blueprint 模式）
- `code_generation`（使用 lite template + blueprint）
- `code_verification`（blueprint 模式）
- `simulation_execution`（lite 模式）
- `result_evaluation`（轻量模式）
- `feedback_generation`
- `iteration_control`

---

### Mode 4: **calibrasim**（CalibraSim 模式）

**流程步骤**：
```
1. Task Understanding
   └─> 标准任务理解

2. Data Analysis
   └─> 数据分析

3. Model Planning（CalibraSim 模板）
   └─> 使用 Calibrasim_model_planning_prompt.txt
   └─> 生成包含参数定义的 model_plan.json

4. Parameters File Generation
   └─> 从 model_plan 生成 parameters.json
   └─> 生成 parameter_definitions.json

5. Code Generation（CalibraSim 模板）
   └─> 使用 Calibrasim_code_generation_prompt.txt
   └─> 生成参数化的仿真代码

6. Code Verification

7. Simulation Execution

8. Result Evaluation

9. Feedback Generation + Iteration Control
```

**特点**：
- 专门为 CalibraSim 框架设计
- 自动生成参数文件（`parameters.json`）
- 支持参数化仿真代码生成

**使用的 Agents**：
- `task_understanding`
- `data_analysis`
- `model_planning_calibrasim`（专用）
- `code_generation_calibrasim`（专用）
- `code_verification`
- `simulation_execution`
- `result_evaluation`
- `feedback_generation`
- `iteration_control`

---

## 🔧 ODD 模式（Overview, Design concepts, Details）

### ODD Agents 架构

**ODD 相关 Agents**：
- `task_understanding_odd`：使用 ODD 协议解析任务
- `data_analysis_odd`：ODD 模式数据分析
- `code_generation_odd`：基于 ODD 生成代码

**ODD 协议结构**：
```json
{
  "overview": {
    "purpose": "模拟器目的",
    "entities": ["实体列表"],
    "state_variables": ["状态变量"],
    "scale": "时间/空间尺度"
  },
  "design_concepts": {
    "basic_principles": "基本原理",
    "emergence": "涌现现象",
    "adaptation": "适应机制",
    "objectives": "目标"
  },
  "details": {
    "initialization": "初始化规则",
    "input_data": "输入数据",
    "submodels": ["子模型"]
  }
}
```

**当前状态**：
- ODD Agents 已实现（`agents/task_understanding_odd/`, `agents/data_analysis_odd/`, `agents/code_generation_odd/`）
- 已注册到 `AgentContainer`
- **但尚未集成到主工作流**（`workflow_manager.py` 中没有 ODD 模式分支）

**如何启用 ODD 模式**：
1. 在 `workflow_manager.py` 的 `_run_iteration()` 中添加 `elif self.mode == 'odd'` 分支
2. 使用 ODD Agents：`task_understanding_odd`, `data_analysis_odd`, `code_generation_odd`
3. 使用 ODD 模板：`templates/task_understanding_odd_prompt.txt`, `templates/code_generation_odd_prompt.txt`

---

## 📊 Agent 执行流程

### Agent 基类：`BaseAgent`

所有 Agents 继承自 `BaseAgent`，提供：
- LLM 调用接口
- 配置管理
- 日志记录
- 模板加载

### Agent 调用链

```
WorkflowManager._run_iteration()
  ├─> agents["task_understanding"].process(...)
  ├─> agents["data_analysis"].process(...)
  ├─> agents["model_planning"].process(...)
  ├─> agents["code_generation"].process(...)
  ├─> agents["code_verification"].process(...)
  ├─> agents["simulation_execution"].process(...)
  ├─> agents["result_evaluation"].process(...)
  ├─> agents["feedback_generation"].process(...)
  └─> agents["iteration_control"].process(...)
```

### Agent 输入/输出

**Task Understanding**：
- 输入：`task_description`, `task_data`（可选）
- 输出：`task_spec.json`

**Data Analysis**：
- 输入：`data_path`, `task_spec`
- 输出：`data_analysis.json`

**Model Planning**：
- 输入：`task_spec`, `data_analysis`
- 输出：`model_plan.json`（包含参数定义）

**Code Generation**：
- 输入：`task_spec`, `model_plan`, `data_analysis`, `feedback`（可选）, `previous_code`（可选）
- 输出：`simulation_code_iter_{N}.py`

**Code Verification**：
- 输入：`code`, `task_spec`, `data_path`
- 输出：`verification_results.json`（包含 `passed`, `critical_issues`, `warnings`）

**Simulation Execution**：
- 输入：`code_path`, `task_spec`, `data_path`
- 输出：`simulation_results.json`（包含 `execution_status`, `simulation_metrics`, `time_series_data`）

**Result Evaluation**：
- 输入：`simulation_results`, `task_spec`, `data_analysis`
- 输出：`evaluation_results.json`（包含 `overall_score`, `metrics`, `recommendations`）

**Feedback Generation**：
- 输入：`task_spec`, `model_plan`, `generated_code`, `verification_results`, `simulation_results`, `evaluation_results`, `current_code`, `previous_code`, `historical_fix_log`
- 输出：`feedback.json`（包含 `feedback_sections`, `summary`, `critical_issues`）

**Iteration Control**：
- 输入：`feedback`, `verification_results`, `evaluation_results`, `current_iteration`, `max_iterations`, `auto_mode`, `user_feedback`
- 输出：`iteration_decision.json`（包含 `continue`, `reason`）

---

## 🔄 迭代机制

### 迭代控制逻辑

1. **硬限制**（Hard Limit）：用户指定的最大迭代次数
2. **软限制**（Soft Limit）：初始为 3，可动态扩展
3. **停止条件**：
   - 达到硬限制
   - `iteration_decision["continue"] == False`
   - 手动模式：用户输入 `#STOP#`

### 状态管理

**WorkflowManager.state**：
```python
{
  "task_spec": None,
  "data_analysis": None,
  "model_plan": None,
  "generated_code": None,
  "verification_results": None,
  "simulation_results": None,
  "evaluation_results": None,
  "feedback": None,
  "iteration_decision": None,
  "code_memory": {}  # 存储每轮迭代的代码
}
```

### 历史修复日志（Historical Fix Log）

- 记录每轮迭代的 `critical_issues`
- 跟踪问题状态（`open`, `fixed`）
- 传递给代码生成 Agent，避免重复错误

---

## 📁 输出文件结构

```
output/
├── task_spec_iter_1.json
├── data_analysis_iter_1.json
├── model_plan_iter_1.json
├── generated_code_iter_1.json
├── simulation_code_iter_1.py
├── verification_results_iter_1.json
├── simulation_results_iter_1.json
├── evaluation_results_iter_1.json
├── feedback_iter_1.json
├── iteration_decision_iter_1.json
├── state_iter_1.json
├── historical_fix_log.json
├── parameters.json              # CalibraSim 模式
├── parameter_definitions.json   # CalibraSim 模式
└── blueprint_iter_1.json        # Blueprint 模式
```

---

## 🎯 CalibraSim 集成点

### Phase 1: 全参数 SBI（当前阶段）

**目标**：从 ODD 模式搬运 SBI 实现到 CalibraSim 流程

**当前状态**：
- ✅ CalibraSim 模式已实现（`--mode calibrasim`）
- ✅ 参数文件生成已实现（`parameters.json`, `parameter_definitions.json`）
- ✅ CalibraSim 专用 Agents 已实现（`model_planning_calibrasim`, `code_generation_calibrasim`）
- ⏳ SBI 校准框架待集成

**下一步**：
1. 在生成的代码中集成 SBI 校准器
2. 实现 Phase 1：全参数 SBI（无嵌入误差）
3. 实现 Phase 2：全局敏感性分析
4. 实现 Phase 3：嵌入误差 + 再次 SBI
5. 实现 Phase 4：TextGrad 回写机制

---

## 🔍 关键接口

### WorkflowManager.run()

主工作流入口，负责：
- 循环执行迭代
- 管理迭代限制
- 保存最终状态

### WorkflowManager._run_iteration()

单次迭代执行，根据 `mode` 选择不同流程。

### AgentContainer.agent_providers()

返回所有注册的 Agents 字典，供 WorkflowManager 使用。

---

## 📝 配置管理

### config.yaml

- 定义所有 Agents 的配置
- LLM 提供商配置（OpenAI, Gemini, Anthropic）
- 工作流参数（max_iterations, auto_mode）

### 依赖注入

使用 `dependency_injector` 库：
- `AgentContainer` 管理所有 Agents
- 配置通过 `config.yaml` 加载
- Agents 通过 `@inject` 装饰器注入

---

## 🚧 待完成功能

1. **ODD 模式集成**：
   - 在 `workflow_manager.py` 中添加 ODD 模式分支
   - 使用 ODD Agents 和模板

2. **SBI 校准集成**：
   - Phase 1: 全参数 SBI
   - Phase 2: 敏感性分析
   - Phase 3: 嵌入误差 SBI
   - Phase 4: TextGrad 回写

3. **Medium 模式**：
   - 当前为占位符，待实现

---

## 📚 参考文档

- `CalibraSim_TODO.md`：CalibraSim 框架详细计划
- `README.md`：项目使用说明
- `config.yaml`：配置文件说明

---

*最后更新：2025-01-XX*

