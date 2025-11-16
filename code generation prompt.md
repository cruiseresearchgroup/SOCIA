# Code Generation Prompt 设计文档

## 项目背景

### 目标
创建一个增强版的代码生成prompt，能够生成支持模块化架构和SBI校准的仿真代码，为CalibraSim框架提供强大的代码生成基础。

### 核心需求
1. **兼容Calibrasim model plan**：支持modules、signals、parameters、observables等字段
2. **支持模块化代码生成**：生成清晰的模块化架构
3. **支持参数文件驱动**：CLI接口和参数管理
4. **支持SBI校准**：模块级和系统级SBI校准能力
5. **保持向后兼容**：支持传统entities/behaviors/interactions结构

## 架构设计决策

### 一体化设计 vs 分离式设计

#### 问题分析
- **分离式设计**：代码生成Agent生成基础代码，SBI Agent后续添加接口
  - ❌ 代码解析复杂，容易出错
  - ❌ 修改现有代码容易引入bug
  - ❌ 接口兼容性问题
  - ❌ 依赖关系处理复杂

#### 最终选择：一体化设计
- ✅ **避免代码解析**：直接生成完整的代码
- ✅ **保证一致性**：所有接口在生成时就设计好
- ✅ **减少错误**：不需要修改现有代码
- ✅ **简化调试**：问题定位在一个文件中

## 功能设计

### 1. 基础功能（继承自原版）
- ✅ **可执行性**：生成的代码必须能直接运行
- ✅ **路径处理**：正确处理数据文件路径
- ✅ **OpenAI API集成**：支持LLM agent调用
- ✅ **反馈应用**：能应用之前的反馈修改
- ✅ **结果保存**：保存仿真结果和可视化

### 2. 模块化架构支持
- ✅ **模块定义**：每个模块实现`forward(state, buffers, params, t)`方法
- ✅ **信号注册表**：从signals列表构建，验证模块I/O一致性
- ✅ **参数注册表**：从parameters列表构建，支持运行时参数管理
- ✅ **DAG调度器**：基于依赖关系和tick_rate的执行顺序控制
- ✅ **可观测值绑定**：observables绑定到真实数据字段
- ✅ **计划验证**：validate_plan()函数，全面的验证检查

### 3. 参数文件驱动
- ✅ **CLI接口**：`--param-file parameters.json`和`--set key=value`覆盖
- ✅ **冻结参数处理**：frozen参数不可覆盖，记录警告
- ✅ **参数持久化**：执行后保存`parameters_used.json`
- ✅ **参数验证**：边界检查和类型验证

### 4. SBI校准支持（条件性）
- ✅ **模块级SBI接口**：每个模块都能独立进行SBI校准
- ✅ **系统级SBI接口**：支持整个系统的SBI校准
- ✅ **参数采样**：支持从先验分布采样参数
- ✅ **批量仿真**：支持批量运行仿真收集数据
- ✅ **条件性生成**：根据model plan中的sbi_calibration字段决定是否生成SBI功能

## 实现策略

### 条件性SBI支持
```text
# SBI Calibration Support (Conditional)
If the model plan contains modules with sbi_calibration: true:
- Add sbi_ready: boolean flag to each SBI-enabled module
- Add simulate_for_sbi(parameter_samples, n_simulations) method to SBI-enabled modules
- Add run_module_sbi(module_name, parameter_samples) method to Simulation class
- Add SBI-specific imports (sbi, torch, numpy) when SBI modules are present
- Implement parameter sampling and batch simulation for SBI-enabled modules
- Add SBI result export functionality

If no modules have sbi_calibration: true:
- Generate standard simulation code without SBI interfaces
- Keep code simple and focused on basic simulation functionality
```

### 生成的代码结构
```python
# 模块定义（条件性SBI支持）
class InformationDiffusion:
    def __init__(self, params):
        self.params = params
        self.sbi_ready = True  # 如果sbi_calibration: true
    
    def forward(self, state, buffers, params, t):
        # 基础仿真逻辑
        pass
    
    def simulate_for_sbi(self, parameter_samples, n_simulations):
        # SBI接口（如果sbi_calibration: true）
        if self.sbi_ready:
            # SBI逻辑
            pass
        else:
            raise NotImplementedError("SBI not enabled for this module")

# 主仿真类
class Simulation:
    def __init__(self, model_plan):
        self.modules = self._build_modules(model_plan)
        self.scheduler = DAGScheduler(self.modules)
    
    def run(self, start_day, end_day):
        # 基础仿真
        pass
    
    def run_module_sbi(self, module_name, parameter_samples):
        # SBI接口（如果模块支持SBI）
        module = self.modules[module_name]
        if hasattr(module, 'simulate_for_sbi'):
            return module.simulate_for_sbi(parameter_samples)
        else:
            raise ValueError(f"Module {module_name} does not support SBI")
```

## 内容来源

### 从现有prompt学习的内容

#### 1. 从Calibrasim prompt学习
- **模块化架构实现**：modules、signals、parameters、observables支持
- **DAG调度器**：基于依赖关系和tick_rate的调度
- **参数管理**：CLI接口、参数覆盖、持久化
- **完整实现要求**：详细的实现清单

#### 2. 从ODD prompt学习
- **校准算法设计理念**：拟合参数、稳定接口、模块化
- **数据分割和评估流程**：训练/验证数据分割、前向仿真、误差计算
- **模块化校准接口**：每个模块独立校准的能力

#### 3. 从Full prompt学习
- **基础结构**：反馈处理、路径处理、API集成
- **向后兼容性**：支持传统entities/behaviors/interactions
- **代码质量**：PEP 8合规、完整文档、错误处理

## 最终目标

### 生成的代码特点
- **模块化**：清晰的模块边界和接口
- **可扩展**：新模块可以轻松添加
- **可校准**：每个模块都能独立进行SBI
- **可配置**：通过参数文件灵活控制
- **可验证**：完整的验证和错误检查
- **条件性**：根据配置决定是否包含SBI功能

### 支持的使用场景
1. **传统仿真**：使用entities/behaviors/interactions的传统模式
2. **模块化仿真**：使用modules/signals/parameters的模块化模式
3. **SBI校准**：支持模块级和系统级的SBI校准
4. **混合模式**：同时支持传统和模块化字段

## 总结

这个增强版的代码生成prompt是一个**功能完整、高度兼容、面向未来的代码生成解决方案**，能够：

1. **完全兼容**Calibrasim和传统model plan格式
2. **生成模块化代码**，支持清晰的架构设计
3. **支持参数文件驱动**，包含完整的参数管理系统
4. **支持SBI校准**，包括模块级和系统级校准
5. **保持向后兼容**，支持传统仿真功能
6. **提供完整验证**，确保代码质量和正确性

通过一体化设计，避免了分离式设计的复杂性，为CalibraSim框架提供了强大而可靠的代码生成基础。
