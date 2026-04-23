# 状态恢复示例

## 修复前 vs 修复后的对比

### 场景：加载 `simulation_code_iter_1.py` 继续执行

---

## ❌ 修复前的问题

```python
# 加载persisted code后：
code_memory = {
    1: {"simulation_code_iter_1.py": "..."}  # 只有iter_1
}
historical_fix_log = {}  # 空字典
current_iteration = 1

# 进入Feedback Generation阶段：
previous_code = None
if current_iteration > 0 and current_iteration - 1 in code_memory:
    # 1 > 0 ✅ True
    # 0 in code_memory ❌ False -> 条件失败！
    ...

# 结果：
# - previous_code = None (无法对比代码变化)
# - historical_fix_log = {} (后续迭代可能重复错误)
```

**影响**：
1. **Feedback Generation**: 无法生成有效的代码差异分析
2. **Code Generation (iter 2+)**: 可能重复之前已修复的错误

---

## ✅ 修复后的逻辑

```python
# 加载persisted code后，执行状态恢复：

# Step 1: 加载之前迭代的代码
for prev_iter in range(1):  # range(persisted_iteration)
    # 加载 simulation_code_iter_0.py
    code_memory[0] = {"simulation_code_iter_0.py": "..."}
    
code_memory[1] = {"simulation_code_iter_1.py": "..."}  # 已加载

# Step 2: 加载历史修复日志
historical_fix_log = {
    "iter_0_issue_1": {...},
    "iter_0_issue_2": {...},
    ...
}

current_iteration = 1

# 进入Feedback Generation阶段：
previous_code = None
if current_iteration > 0 and current_iteration - 1 in code_memory:
    # 1 > 0 ✅ True
    # 0 in code_memory ✅ True -> 条件成功！
    prev_code_dict = code_memory[0]
    previous_code = prev_code_dict["simulation_code_iter_0.py"]  # ✅ 成功获取

# 结果：
# - previous_code = "...iter_0的代码..." (可以对比变化)
# - historical_fix_log = {...} (避免重复错误)
```

**效果**：
1. **Feedback Generation**: ✅ 可以对比 iter_1 vs iter_0 的代码差异
2. **Code Generation (iter 2+)**: ✅ 知道历史错误，避免重复

---

## 完整的数据恢复清单

当加载 `simulation_code_iter_N.py` 时，系统会恢复：

### ✅ 必须恢复（已实现）

| 数据 | 来源 | 用途 |
|-----|------|------|
| `task_spec` | `task_spec_iter_N.json` 或 `task_spec_iter_0.json` | 所有agent需要的任务规范 |
| `code_memory[N]` | `simulation_code_iter_N.py` (persisted) | 当前迭代的代码 |
| `code_memory[0..N-1]` | `simulation_code_iter_{0..N-1}.py` | Feedback Generation对比用 |
| `historical_fix_log` | `historical_fix_log.json` | Code Generation避免重复错误 |
| `playbook` | `playbook_storage/current/playbook.json` | ACE mode的策略知识库 |

### ❌ 不需要恢复

| 数据 | 原因 |
|-----|------|
| `verification_results_iter_{0..N-1}.json` | 历史记录，当前迭代会重新生成 |
| `simulation_results_iter_{0..N-1}.json` | Code Generation会按需从文件加载 |
| `evaluation_results_iter_{0..N-1}.json` | 历史记录，不影响当前迭代 |
| `feedback_iter_{0..N-1}.json` | 历史记录，feedback不跨多次迭代传递 |

---

## 测试验证建议

### 测试用例1：加载 iter_1 继续执行

```bash
cd /Users/z3546829/PycharmProjects/SOCIA
conda activate SOCIA

# 前提：已经执行过iter_0和iter_1，现在从iter_1继续
python test_data_analysis.py \
    --config config.yaml \
    --mode ace \
    --output output/test_ace_resume \
    --task "test task" \
    --data_path data/test \
    --iterations 3 \
    --persisted_code_file output/test_ace_resume/simulation_code_iter_1.py \
    --auto
```

**预期日志输出**：
```
INFO - Loading persisted code from: output/test_ace_resume/simulation_code_iter_1.py
INFO - Extracted iteration number: 1
INFO - Loading code from previous iterations (0 to 0) for state restoration...
INFO -   ✓ Loaded code from iteration 0
INFO -   ✓ Loaded historical fix log: 2 entries
INFO - State restoration complete. Ready to continue from iteration 1.
INFO - Persisted code loaded successfully as iter_1, will skip code generation...
...
INFO - STARTING ITERATION 1/3
INFO - Skipping code generation for iteration 1 (using persisted code)
INFO - Reset skip_initial_code_generation flag - next iteration (2) will generate code
INFO - SIMULATION EXECUTION
...
INFO - FEEDBACK GENERATION
INFO - Retrieved previous_code from iteration 0 for comparison  # ✅ 关键点
...
INFO - STARTING ITERATION 2/3
INFO - CODE GENERATION  # ✅ 应该生成新代码
```

### 测试用例2：加载 iter_0 继续执行

```bash
python test_data_analysis.py \
    --config config.yaml \
    --mode ace \
    --output output/test_ace_resume \
    --task "test task" \
    --data_path data/test \
    --iterations 3 \
    --persisted_code_file output/test_ace_resume/simulation_code_iter_0.py \
    --auto
```

**预期日志输出**：
```
INFO - Extracted iteration number: 0
INFO - State restoration complete. Ready to continue from iteration 0.
INFO - No previous iterations to load (starting from iter_0)
INFO - No historical fix log found, starting with empty log
...
INFO - STARTING ITERATION 0/3
INFO - Skipping code generation for iteration 0 (using persisted code)
INFO - SIMULATION EXECUTION
...
INFO - FEEDBACK GENERATION
INFO - No previous code available (first iteration)  # ✅ 正常，iter_0没有previous
```

---

## 关键改进点总结

1. **🔍 问题诊断**: 明确了 `previous_code = None` 的根本原因是 `code_memory` 缺少历史迭代数据

2. **🛠️ 最小修复**: 只恢复必需的数据（previous code + historical fix log），不恢复不必要的历史结果

3. **📝 清晰日志**: 添加详细的状态恢复日志，便于调试和验证

4. **✅ 向后兼容**: 
   - 如果 `persisted_iteration = 0`，不尝试加载previous iterations
   - 如果历史文件不存在，优雅降级（使用空字典）

5. **🚀 完整性**: 确保从任意迭代恢复后，后续workflow能正常进行

