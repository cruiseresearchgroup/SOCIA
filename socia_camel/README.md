# SOCIA-CAMEL

## 基于CAMEL架构的社会模拟智能体系统

SOCIA-CAMEL是SOCIA（Simulation Orchestration for City Intelligence and Agents）的一个扩展实现，使用CAMEL（Communicative Agents for "Mind" Exploration of LLM Society）作为Agent基础架构，以实现更自然、更人性化的社会模拟。

### 核心架构

本项目基于CAMEL架构，为每个Agent实现了以下核心模块：

1. **Memory Module（记忆模块）**
   - 存储Agent接触过的信息
   - 记录Agent自身的行为历史
   - 保存推理过程和决策依据

2. **Action Module（行为模块）**
   - 提供丰富的交互行为
   - 支持模拟相关的专门行为
   - 包含数据分析、代码生成等能力

3. **Chain-of-Thought（思维链）**
   - 为每个行为提供推理支持
   - 增强模拟的可解释性
   - 通过思维链实现更自然的决策过程

### 系统组件

SOCIA-CAMEL包含以下主要的Agent角色：

1. **任务理解Agent**：解析用户需求
2. **数据分析Agent**：分析真实世界数据
3. **模型规划Agent**：设计模拟方法和结构
4. **代码生成Agent**：转换计划为Python代码
5. **模拟执行Agent**：运行生成的模拟代码

### 使用方法

```bash
python main.py --task "Create a simple epidemic simulation model that models the spread of a virus in a population of 1000 people." --output "./output/my_sim_output"
```

### 特点与优势

- 基于大语言模型驱动的多智能体系统
- 智能体具有记忆、行为和推理能力
- 可生成完整的模拟代码并执行
- 自然语言接口，无需编程经验
- 可解释的决策过程 