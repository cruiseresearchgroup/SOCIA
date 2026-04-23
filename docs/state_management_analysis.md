# 跨迭代状态管理分析

## 情况 1：正常流程（没有加载持久化代码）

### 数据结构及其内容

#### 1. `state` 字典
- **作用范围**: 当前迭代内
- **内容**:
  ```python
  {
      "task_spec": None,              # 任务规范（全局共享，只在data analysis时初始化一次）
      "data_analysis": None,          # 数据分析结果（只在第0次使用）
      "model_plan": None,             # 模型计划（odd/ace mode不使用）
      "generated_code": {...},        # 当前迭代生成的代码字典
      "verification_results": {...},  # 当前迭代的验证结果
      "simulation_results": {...},    # 当前迭代的模拟执行结果
      "evaluation_results": {...},    # 当前迭代的评估结果
      "feedback": {...},              # 上一迭代的feedback（传递给当前迭代）
      "iteration_decision": {...},    # 当前迭代的控制决策
      "code_memory": {...}            # 指向code_memory的引用
  }
  ```
- **特点**: 
  - 大部分字段在每次迭代都会被覆盖（verification_results, simulation_results等）
  - `feedback` 会从上一迭代传递到下一迭代，用于代码生成的输入

#### 2. `code_memory` 字典
- **作用范围**: 跨所有迭代，累积存储
- **内容**:
  ```python
  {
      0: {"simulation_code_iter_0.py": "...代码内容..."},
      1: {"simulation_code_iter_1.py": "...代码内容..."},
      2: {"simulation_code_iter_2.py": "...代码内容..."},
      ...
  }
  ```
- **用途**:
  - Code Generation: 读取 `code_memory[current_iteration - 1]` 作为 `previous_code`
  - Feedback Generation: 读取 `code_memory[current_iteration]` 和 `code_memory[current_iteration - 1]` 进行对比

#### 3. `historical_fix_log` 字典
- **作用范围**: 跨所有迭代，累积存储
- **内容**: 记录历史修复日志（具体格式取决于feedback_generation agent的实现）
- **用途**: 传递给 code_generation agent，避免重复犯同样的错误

#### 4. `task_spec` 字典
- **作用范围**: 整个workflow共享
- **内容**:
  ```python
  {
      "task_description": "...",
      "data_analysis_result": {
          # blueprint内容
          "overall_simulation_design": {...},
          "agent_archetypes": {...},
          ...
          # file_summaries
          "file_summaries": [...]
      },
      "file_summaries": [...],  # 数据文件摘要
      "schemas": {...}          # 数据schema
  }
  ```
- **特点**: 
  - 在data analysis后生成，只初始化一次
  - ACE mode可能在每次迭代通过blueprint feedback更新
  - 所有agent共享同一个task_spec

#### 5. `playbook` 对象 (ACE mode only)
- **作用范围**: 跨所有迭代，累积存储
- **内容**:
  ```python
  {
      "playbook_metadata": {...},
      "strategies": {
          "issue-001": {
              "meta_info": {...},
              "reflection": {...}
          },
          ...
      }
  }
  ```
- **用途**: 传递给 code_generation_ace agent，提供历史经验

### 工作流程中的数据传递

**Iteration N (N >= 1) 的 Code Generation 阶段**:
```python
# 需要访问：
- task_spec (全局)
- state["feedback"] (来自iteration N-1)
- code_memory[N-1] (previous_code)
- historical_fix_log (累积)
- playbook (ACE mode，累积)
- simulation_results_iter_{N-1}.json (ACE mode, patch prompt)
```

**Iteration N 的 Feedback Generation 阶段**:
```python
# 需要访问：
- task_spec (全局)
- state["generated_code"] (当前迭代)
- state["verification_results"] (当前迭代)
- state["simulation_results"] (当前迭代)
- state["evaluation_results"] (当前迭代)
- code_memory[N] (current_code)
- code_memory[N-1] (previous_code，用于对比)
- historical_fix_log (累积)
```

---

## 情况 2：加载持久化代码后的状态恢复

### 问题场景

假设用命令行加载 `simulation_code_iter_1.py`，期望：
- 跳过 iteration 1 的代码生成
- 继续执行 iteration 1 的 Simulation Execution → Feedback Generation
- 然后进入 iteration 2 的完整流程

### 需要恢复的数据

#### 必须恢复的数据：

1. **`task_spec`** ✅ 已恢复
   - 从 `task_spec_iter_1.json` 或 `task_spec_iter_0.json` 加载
   - 当前代码已实现

2. **`code_memory[1]`** ✅ 已恢复
   - 从 `simulation_code_iter_1.py` 加载
   - 当前代码已实现

3. **`code_memory[0]`** ❌ 需要添加
   - 从 `simulation_code_iter_0.py` 加载（如果存在）
   - **原因**: Feedback Generation需要对比current和previous code
   - **影响**: 如果缺失，Feedback Generation的previous_code将为None

4. **`playbook`** ✅ 已恢复（ACE mode）
   - 从 `playbook_storage/current/playbook.json` 加载
   - 当前代码已实现

#### 可选恢复的数据：

5. **`historical_fix_log`** ⚠️ 可能需要
   - 从 `historical_fix_log.json` 加载（如果存在）
   - **原因**: Code Generation (iteration 2+) 需要避免重复错误
   - **影响**: 如果缺失，可能会重复之前的错误

6. **之前迭代的results** ❌ 不需要恢复到state
   - `simulation_results_iter_0.json` 等文件已持久化
   - Code Generation会按需从文件加载（已实现，见line 1324-1331）
   - 不需要恢复到state中

### 当前代码的缺失

查看你删除的代码块，我发现你删除了：
- 加载previous iterations的code（code_memory[0]）
- 加载historical_fix_log
- 加载previous iterations的results（这个确实不需要）

**关键问题**：
- 如果不加载 `code_memory[0]`，Feedback Generation阶段的 `previous_code` 将为 `None`
- 如果不加载 `historical_fix_log`，后续迭代可能重复之前的错误

### 建议的修复方案

建议恢复**最小必要的加载逻辑**：

```python
# 在加载persisted code后添加：

# 1. Load previous iterations' code (for feedback generation comparison)
logger.info(f"Loading code from previous iterations (0 to {persisted_iteration-1})...")
for prev_iter in range(persisted_iteration):
    prev_code_file = os.path.join(args.output, f"simulation_code_iter_{prev_iter}.py")
    if os.path.exists(prev_code_file):
        try:
            with open(prev_code_file, 'r', encoding='utf-8') as f:
                prev_code = f.read()
            code_memory[prev_iter] = {f"simulation_code_iter_{prev_iter}.py": prev_code}
            logger.info(f"✓ Loaded code from iteration {prev_iter}")
        except Exception as e:
            logger.warning(f"Failed to load code from iteration {prev_iter}: {e}")

# 2. Load historical fix log (for future iterations' code generation)
historical_fix_log_file = os.path.join(args.output, "historical_fix_log.json")
if os.path.exists(historical_fix_log_file):
    try:
        with open(historical_fix_log_file, 'r', encoding='utf-8') as f:
            historical_fix_log = json.load(f)
        logger.info(f"✓ Loaded historical fix log: {len(historical_fix_log)} entries")
    except Exception as e:
        logger.warning(f"Failed to load historical fix log: {e}")
        historical_fix_log = {}
```

**不需要加载**：
- ❌ 之前迭代的 verification_results, simulation_results, evaluation_results
  - 原因：这些只是历史记录，当前迭代会重新生成
  - Code Generation的patch prompt会按需从文件加载simulation_results（已实现）
- ❌ 之前迭代的 feedback
  - 原因：只有最近一次的feedback会传递到下一迭代，历史feedback不需要

