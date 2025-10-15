# SOCIA SBI Agent

轻量级SBI Agent，专门针对SOCIA项目设计，支持模块化SBI校准。

## 功能特性

- **SOCIA集成**: 深度集成SOCIA工作流，支持模块依赖关系分析
- **模块化校准**: 支持分阶段模块校准，避免参数对消
- **智能摘要统计**: 基于SOCIA数据分析结果设计摘要统计
- **自动参数管理**: 自动加载SOCIA参数定义和当前参数
- **结果格式兼容**: 生成符合SOCIA格式的结果文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
from simple_sbi_agent import SimpleSBIAgent

# 初始化SBI Agent
agent = SimpleSBIAgent(
    output_dir="output/mask_adoption_calibrasim_debug_run3",
    data_dir="data_fitting/mask_adoption_data"
)

# 执行SBI校准
results = agent.calibrate()

# 查看结果
print(results.summary)
```

### SOCIA集成示例

```python
# 加载SOCIA配置
agent.load_socia_configs()

# 分析模块依赖关系
dependencies = agent.analyze_module_dependencies()

# 执行分阶段校准
for module in dependencies:
    agent.calibrate_module(module)
```

## 项目结构

```
sbi_agent/
├── simple_sbi_agent.py    # 主文件
├── requirements.txt       # 依赖文件
├── README.md             # 使用说明
├── file_utils.py         # 文件操作工具
├── param_utils.py        # 参数管理工具
├── log_utils.py          # 日志工具
├── data_utils.py         # 数据处理工具
├── socia_utils.py        # SOCIA集成工具
└── examples/             # 示例目录
    └── example_usage.py
```

## 支持的SOCIA任务

- mask_adoption: 口罩采纳仿真
- agent_society: 智能体社会仿真
- llmob: 大规模语言模型仿真

## 技术栈

- **SBI**: 基于神经网络的贝叶斯推理
- **SOCIA**: 社会仿真框架集成
- **NetworkX**: 模块依赖关系分析
- **Pandas**: 数据处理和分析
- **Matplotlib**: 结果可视化





