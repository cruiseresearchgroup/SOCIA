# Simulation Execution Output File 路径设置流程

## 📍 完整调用链

### 1️⃣ 入口：`test_data_analysis.py` (Line 1449-1458)

```python
state["simulation_results"] = agents["simulation_execution"].process(
    code_path=code_file_path,                    # 例如: output/.../simulation_code_iter_0.py
    task_spec=task_spec,
    data_path=data_path,
    mode="ace",
    output_dir=args.output,                      # ✅ 关键参数1: 输出目录
    iteration=current_iteration,                 # ✅ 关键参数2: 迭代号
    project_root=project_root,
    openai_api_key=openai_api_key
)
```

**参数来源**：
- `output_dir = args.output` (命令行参数 `--output` 的值)
- `iteration = current_iteration` (当前迭代号，例如 0, 1, 2...)

---

### 2️⃣ 路径生成：`simulation_execution_ace/agent.py` (Line 212-217)

```python
# Generate output file path if output_dir and iteration are provided
output_file = None
if output_dir and iteration is not None:
    # Create output filename: simulation_results_iter_{N}.json
    output_file = os.path.join(output_dir, f"simulation_results_iter_{iteration}.json")
    self.logger.info(f"Output file for simulation: {output_file}")
```

**路径生成规则**：
```
output_file = os.path.join(output_dir, f"simulation_results_iter_{iteration}.json")
```

**示例**：
- `output_dir = "output/test_data_analysis_ace_modified_selfloop_llmob"`
- `iteration = 0`
- **结果**: `output/test_data_analysis_ace_modified_selfloop_llmob/simulation_results_iter_0.json`

---

### 3️⃣ 传递给执行函数：`simulation_execution_ace/agent.py` (Line 219-225)

```python
execution_result = self._execute_code_with_subprocess(
    code_path, 
    data_path,
    output_file=output_file,        # ✅ 传递生成的路径
    project_root=project_root,
    openai_api_key=openai_api_key
)
```

---

### 4️⃣ 构建命令：`simulation_execution_ace/agent.py` → `run_python_script()` (Line 64-66)

```python
# Build command with optional --output argument
cmd = ["python", script_file]
if output_file:
    cmd.extend(["--output", output_file])  # ✅ 添加到命令行参数
```

**最终执行的命令**：
```bash
python simulation_code_iter_0.py --output output/.../simulation_results_iter_0.json
```

---

### 5️⃣ 读取结果：`simulation_execution_ace/agent.py` (Line 383-415)

执行完成后，从 output_file 读取结果：

```python
if output_file and os.path.exists(output_file):
    try:
        self.logger.info(f"Reading simulation results from output file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            simulation_output = json.load(f)
        
        # Store full content
        execution_result["simulation_output"] = simulation_output
        
        # Extract key fields for convenience
        if "calibrated_parameters" in simulation_output:
            execution_result["calibrated_parameters"] = simulation_output["calibrated_parameters"]
        if "evaluation_results_on_validation" in simulation_output:
            execution_result["evaluation_results_on_validation"] = simulation_output["evaluation_results_on_validation"]
        # ... 更多字段提取
```

---

## 📊 完整流程图

```
test_data_analysis.py
    │
    ├─> args.output = "output/test_ace/..."
    ├─> current_iteration = 0
    │
    └─> agents["simulation_execution"].process(
            output_dir=args.output,        # "output/test_ace/..."
            iteration=0                    # 0
        )
            │
            ▼
simulation_execution_ace/agent.py::process()
    │
    ├─> output_file = os.path.join(
    │       "output/test_ace/...",
    │       "simulation_results_iter_0.json"
    │   )
    │   = "output/test_ace/.../simulation_results_iter_0.json"
    │
    └─> _execute_code_with_subprocess(
            output_file="output/test_ace/.../simulation_results_iter_0.json"
        )
            │
            ▼
run_python_script()
    │
    ├─> cmd = [
    │       "python",
    │       "simulation_code_iter_0.py",
    │       "--output",
    │       "output/test_ace/.../simulation_results_iter_0.json"
    │   ]
    │
    └─> subprocess.run(cmd, ...)
            │
            ▼
simulation_code_iter_0.py 执行
    │
    ├─> 解析 --output 参数
    ├─> 运行模拟
    └─> 将结果写入 JSON 文件
            │
            ▼
_execute_code_with_subprocess() 读取结果
    │
    ├─> 检查 output_file 是否存在
    ├─> 读取 JSON 内容
    └─> 解析并返回 execution_result
            │
            ▼
test_data_analysis.py 接收结果
    │
    └─> state["simulation_results"] = execution_result
```

---

## 🔍 关键代码位置

| 步骤 | 文件 | 行号 | 功能 |
|-----|------|------|------|
| **1. 调用** | `test_data_analysis.py` | 1449-1458 | 传递 `output_dir` 和 `iteration` |
| **2. 生成路径** | `agents/simulation_execution_ace/agent.py` | 214-216 | 生成 `simulation_results_iter_{N}.json` 路径 |
| **3. 传递路径** | `agents/simulation_execution_ace/agent.py` | 222 | 传递给 `_execute_code_with_subprocess` |
| **4. 构建命令** | `agents/simulation_execution_ace/agent.py` | 65-66 | 添加 `--output` 参数到命令 |
| **5. 读取结果** | `agents/simulation_execution_ace/agent.py` | 383-415 | 从 output_file 读取并解析 JSON |

---

## 📝 路径格式总结

### 路径组成

```
{output_dir}/simulation_results_iter_{iteration}.json
```

### 实际示例

| output_dir | iteration | 最终路径 |
|-----------|-----------|---------|
| `output/test_ace` | `0` | `output/test_ace/simulation_results_iter_0.json` |
| `output/test_ace` | `1` | `output/test_ace/simulation_results_iter_1.json` |
| `output/test_ace_modified` | `2` | `output/test_ace_modified/simulation_results_iter_2.json` |

---

## ⚠️ 注意事项

1. **路径生成条件**：
   - 必须同时提供 `output_dir` 和 `iteration` 参数
   - 如果任一缺失，`output_file` 将为 `None`，不会添加 `--output` 参数

2. **文件创建**：
   - 文件由被执行的 Python 脚本（`simulation_code_iter_N.py`）创建
   - 脚本需要解析 `--output` 参数并写入 JSON

3. **结果读取**：
   - 执行完成后，agent 会检查 output_file 是否存在
   - 如果存在，读取并解析 JSON 内容
   - 如果不存在或解析失败，会记录警告但不会导致执行失败

4. **目录创建**：
   - 目录由 Python 脚本负责创建（如果不存在）
   - Agent 不会预先创建目录

