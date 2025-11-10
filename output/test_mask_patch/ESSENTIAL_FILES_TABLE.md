# 必要文件清单表格

## 核心文件（7个）

| 序号 | 文件 | 描述 | 必要性 | 类别 | 大小 | 行数 | 用途 |
|------|------|------|--------|------|------|------|------|
| 1 | `simulation_alpha.py` | Alpha-SBI主文件（可以选择参数生成alpha） | ✅ 必要 | 核心代码 | 101,846 字节 | 2,397 行 | 运行Alpha-SBI校准，生成alpha参数 |
| 2 | `simulation_gsa.py` | 敏感性分析文件（GSA） | ✅ 必要 | 核心代码 | 90,681 字节 | 2,183 行 | 运行敏感性分析，选择高敏感性参数 |
| 3 | `run_codegen.sh` | 代码生成shell脚本 | ✅ 必要 | 代码生成 | 1,744 字节 | 61 行 | 根据alpha结果生成新代码 |
| 4 | `generate_code_from_feedback.py` | 代码生成Python脚本 | ✅ 必要 | 代码生成 | 14,485 字节 | 352 行 | 代码生成流程（被run_codegen.sh调用） |
| 5 | `task_spec_iter_0.json` | 任务说明文件 | ✅ 必要 | 配置 | 44,791 字节 | 1,070 行 | 定义任务需求、数据路径、模型要求 |
| 6 | `feedback_for_codegen.json` | 反馈文件 | ⚠️ 可选 | 配置 | 20,370 字节 | 204 行 | 代码改进建议（可重新生成） |
| 7 | `alpha_results.json` | Alpha结果文件 | ⚠️ 可选 | 配置 | 3,438 字节 | 146 行 | Alpha运行结果（可重新运行生成） |

## 文件详细说明

### 1. simulation_alpha.py
- **用途**: Alpha-SBI主文件，可以选择参数生成alpha
- **运行**: `python simulation_alpha.py`
- **功能**: 
  - 实现Alpha-SBI校准
  - 支持Double Monte Carlo评估
  - 生成alpha参数和统计信息
- **输出**: `alpha_results.json`

### 2. simulation_gsa.py
- **用途**: 敏感性分析文件（Global Sensitivity Analysis）
- **运行**: `python simulation_gsa.py --gsa`
- **功能**:
  - 实现GSACalibrator类（Morris方法）
  - 分析参数敏感性
  - 选择高敏感性参数
- **输出**: `gsa_results.json`

### 3. run_codegen.sh
- **用途**: 代码生成shell脚本
- **运行**: `bash run_codegen.sh`
- **功能**:
  - 自动检查依赖
  - 运行代码生成流程
  - 检查生成结果
- **输出**: `simulation_alpha_improved.py`

### 4. generate_code_from_feedback.py
- **用途**: 代码生成Python脚本
- **运行**: `python generate_code_from_feedback.py`（或通过run_codegen.sh）
- **功能**:
  - 加载alpha结果和任务说明
  - 生成反馈
  - 根据反馈生成新代码

### 5. task_spec_iter_0.json
- **用途**: 任务说明文件
- **内容**: 定义任务需求、数据路径、模型要求等
- **必需**: 是

### 6. feedback_for_codegen.json
- **用途**: 反馈文件
- **内容**: 代码改进建议、关键问题、模型改进建议等
- **生成**: 由FeedbackGenerationAgent生成
- **必需**: 否（可重新生成）

### 7. alpha_results.json
- **用途**: Alpha运行结果
- **内容**: Alpha参数、统计信息、验证指标等
- **生成**: 由simulation_alpha.py生成
- **必需**: 否（可重新运行生成）

## 文件依赖关系

```
simulation_alpha.py
  └─> 生成 alpha_results.json
       └─> 用于 generate_code_from_feedback.py
            └─> 生成 feedback_for_codegen.json
                 └─> 用于代码生成
                      └─> 生成 simulation_alpha_improved.py

simulation_gsa.py
  └─> 独立的敏感性分析工具
       └─> 可通过 --gsa 参数运行
            └─> 生成 gsa_results.json
```

## 运行流程

### 流程1: 运行Alpha-SBI
```bash
python simulation_alpha.py
# 生成: alpha_results.json
```

### 流程2: 运行敏感性分析
```bash
python simulation_gsa.py --gsa
# 生成: gsa_results.json
```

### 流程3: 根据Alpha结果生成新代码
```bash
bash run_codegen.sh
# 或
python generate_code_from_feedback.py
# 生成: simulation_alpha_improved.py
```

## 可删除的文件

以下文件可以删除（临时文件、日志文件、备份文件等）：

### 临时文件
- `simulation_alpha_improved.py` - 生成的代码（可重新生成）
- `codegen_result.json` - 代码生成结果（可重新生成）
- `llm_raw_response.txt` - LLM原始响应（调试用）
- `actual_prompt.txt` - 实际prompt（调试用）

### 日志文件
- `*.log` - 所有日志文件
- `codegen_run_*.log` - 代码生成运行日志

### 备份文件
- `simulation_code_using_calibration_template_*.py` - 备份文件
- `*_backup_*.py` - 备份文件
- `simulation_code_iter_0 copy.py` - 复制文件

### 测试文件
- `test_*.py` - 测试文件
- `debug_*.py` - 调试文件

### 文档文件
- `*.md` - 文档文件（除了这个文件）
- `FINAL_FLOW_ANALYSIS.md`
- `FIX_CODE_GENERATION_ISSUE.md`
- `PROGRESS_IMPROVEMENTS.md`
- `QUICK_START.md`
- `RUN_CODE_GENERATION.md`
- 等

## 总结

**必要文件（7个）**:
1. ✅ `simulation_alpha.py` - Alpha-SBI主文件
2. ✅ `simulation_gsa.py` - 敏感性分析文件
3. ✅ `run_codegen.sh` - 代码生成shell脚本
4. ✅ `generate_code_from_feedback.py` - 代码生成Python脚本
5. ✅ `task_spec_iter_0.json` - 任务说明文件
6. ⚠️ `feedback_for_codegen.json` - 反馈文件（可选，可重新生成）
7. ⚠️ `alpha_results.json` - Alpha结果文件（可选，可重新运行生成）

**核心必要文件（5个）**: 1, 2, 3, 4, 5
**可选文件（2个）**: 6, 7（可重新生成）

