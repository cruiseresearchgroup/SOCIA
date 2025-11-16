# 🔬 CalibraSim: 嵌入式模型误差驱动的 CPS 模拟器校准框架

## 📋 项目概述

### 研究背景
现有的 CPS 模拟器生成方法（如 SOCIA、G-Sim）虽然能够自动生成模拟器并支持一定程度的校准，但存在以下局限性：
- **过于工程化**：依赖大模型直接修改代码，缺乏系统化的优化框架
- **创新度有限**：缺乏理论指导，难以处理复杂的模块化系统
- **校准效率低**：没有充分利用参数敏感性和结构诊断能力

### 核心创新
我们提出 **CalibraSim** 框架，首次将 **嵌入式模型误差（Embedded Model Error）+ SBI + TextGrad** 引入 CPS 模拟器校准：

1. **嵌入式模型误差**：在参数层注入随机修正，用 α 后验诊断结构问题
2. **双阶段 SBI**：无误差（基线）→ 含误差（诊断），统一在系统级做推断
3. **结构化回写**：α 触发 TextGrad，对可疑模块代码微改并复验
4. **统一评估**：参数不确定性、结构不充分（α）、预测误差

### 技术路线
```
任务理解 → 代码生成 → 参数化 → 全参数SBI → 敏感性分析 → 嵌入式误差(α,ξ) → 再次SBI → TextGrad回写 → 系统集成
```

### 总体思路
- **阶段1**：全参数 SBI（无嵌入误差）- 建立稳定基线、粗定参数范围
- **阶段2**：敏感性分析 - 找出最敏感的 2-5 个参数，作为嵌入误差的候选
- **阶段3**：嵌入误差 + 再次 SBI - 在 Top-K 参数上加入 α，训练 q(λ, α | y)
- **阶段4**：TextGrad 回写与复验 - 根据 α 后验显著大的参数，回写模块代码并重验
- 不再依赖"有无中间真实值 z"的前提，统一在系统级用 SBI 做推断

### 预期贡献
- **方法论贡献**：首次将嵌入式模型误差引入 CPS 模拟器校准
- **工程贡献**：可运行的自动化 pipeline，支持诊断-回写闭环
- **实验贡献**：在多个 CPS 数据集上验证方法有效性
- **论文目标**：NeurIPS 2025 / ICLR 2026

---

## 🎯 阶段里程碑

### Phase 1｜全参数 SBI（无嵌入误差）- 1-2周
- [ ] 从 ODD 模式搬运现有 SBI 实现到 CalibraSim 流程
- [ ] 适配 CalibraSim 的参数 schema 和模块结构
- [ ] 确保参数文件生成与 SBI 训练数据的兼容性
- [ ] 验证在口罩数据集上的基线校准效果
- **验收标准**：成功复现 ODD 模式的 SBI 校准能力，得到稳定后验

### Phase 2｜全局敏感性 - 1-2周
- [ ] Morris 筛选 → Sobol' 深挖（目标=RMSE/峰值/达峰日）
- [ ] 生成 Top-K 参数列表（K=2-5）
- **验收标准**：Top-K 在多 seed 下排名稳定

### Phase 3｜嵌入误差 (α, ξ) 与再次 SBI - 2-3周
- [ ] 在 Top-K 参数的 schema 里打开 embedded_error.enabled
- [ ] 选加法/乘法形式与 α 先验（HalfNormal 优先）
- [ ] 训练 q(λ, α | y)，保持输入一致（真实端=单条轨迹）
- [ ] 诊断报告：α 的后验统计、模块嫌疑度排序
- **验收标准**：至少一个 α 显著偏大或显著收缩（对比阶段1/3 PPC）

### Phase 4｜TextGrad 回写与复验 - 1-2周
- [ ] 触发条件：P(α_j>τ) 阈值策略（默认 τ 取 α 先验上界的 30-50%）
- [ ] 生成回写补丁建议（结构项/阈值/滞后/饱和/异质性）
- [ ] 回写后重复 Phase 3（含 α）
- **验收标准**：目标模块 α 分布明显向 0 收缩、RMSE 改善

### Phase 5｜Benchmark & 打包 - 2-3周
- [ ] 在 G-Sim 复现 Baseline-0/1/2/3
- [ ] 打包可复现实验脚本与结果（repo/figs）
- **验收标准**：至少 2 个数据集上形成端到端结果

---

## 1. 基础代码与架构完善

### 1.1 任务理解与模型规划
- [x] 完善 task_understanding agent，解析数据与目标
- [x] 生成 model_plan.json，包含模块定义、接口规范、参数范围
- [x] 支持 SBI 校准条件的识别（基于 observables 的 target_data_field）

### 1.2 代码生成与参数化
- [x] 更新 code_generation prompt，确保生成参数显式化、模块化的模拟器代码
- [x] 自动生成 parameters.json，包含统一 schema（dtype、bounds、default、scope、owner_module、frozen）
- [ ] 实现模块间接口标准化（输入/输出格式、信号传递机制）

### 1.3 参数文件管理系统
- [x] CLI 支持 --param-file & --set key=value，冻结参数不可覆盖
- [x] 参数验证与边界检查
- [ ] 参数版本管理与回滚机制

### 1.4 嵌入式误差参数 Schema 扩展
- [ ] 扩展参数 schema，支持 embedded_error 配置：
```json
{
  "key": "beta_family",
  "dtype": "float",
  "default": 1.2,
  "bounds": [0.0, 3.0],
  "owner_module": "SocialInfluenceAdoption",
  "sbi_include": true,
  "embedded_error": {
    "enabled": false,
    "form": "additive",
    "alpha_key": "alpha_beta_family",
    "alpha_prior": "HalfNormal(0, 0.5)",
    "xi_dist": "Normal(0,1)"
  }
}
```

### 1.5 Simulation 核心功能
- [ ] Module 定义：信息扩散、口罩采纳、聚合器（可扩展架构）
- [ ] Scheduler（tick 调度 + buffers→state 提交）
- [ ] **可插拔的嵌入层**：在敏感参数处挂钩，支持 per-parameter 的 α
- [ ] 结果保存：sim_results.csv, metrics.json, parameters_used.json, 可视化图表

### 1.6 模拟接口契约
- [ ] 实现 simulate(theta, alpha=None, seed=None) -> trajectory
- [ ] 支持两种嵌入形式：
  - 加法：Λ = λ + α·ξ
  - 乘法：Λ = λ·(1 + α·ξ)（适合正参数）
- [ ] 随机源管理：主 seed + per-sample seed（落盘），保证可复现

---

## 2. SBI 校准框架实现

### 2.1 Phase 1: 全参数 SBI（无嵌入误差）
- [ ] 从 ODD 模式搬运 SBI 核心实现
- [ ] 适配 CalibraSim 的参数文件格式和模块接口
- [ ] 确保与现有模拟器代码的兼容性
- [ ] 验证基线校准效果与 ODD 模式一致

### 2.2 Phase 2: 全局敏感性分析
- [ ] Morris 筛选器实现
- [ ] Sobol' 指数计算
- [ ] Top-K 参数识别与排序
- [ ] 敏感性报告生成

### 2.3 Phase 3: 嵌入误差 + 再次 SBI
- [ ] 在选定参数上注入 α 误差
- [ ] 训练 q(λ, α | y) 联合后验
- [ ] α 后验统计与诊断报告
- [ ] 模块嫌疑度排序

### 2.4 Phase 4: TextGrad 回写机制
- [ ] 触发条件判断：P(α_j > τ) > 0.8
- [ ] 生成回写补丁建议（结构项/阈值/滞后/饱和/异质性）
- [ ] 代码修订建议结构化输出（JSON）
- [ ] 回写后重跑 Phase 3 验证

### 2.5 自动化闭环
- [ ] 任务描述 → 模拟器生成 → 校准 → 代码修订 → 再训练
- [ ] 失败检测与自动重试机制
- [ ] 校准进度可视化与监控

---

## 3. 数据集准备与实验

### 3.1 口罩佩戴数据集（主要实验）
- [x] 数据特征分析：agent profile + 社交网络 + 前30天行为
- [x] 目标定义：后10天 mask adoption 预测
- [ ] 完成全流程 CalibraSim 实验（包含 4 个 Phase）

### 3.2 Baseline 定义
- [ ] **Baseline-0**：当前固定默认参数（无校准）
- [ ] **Baseline-1**：Phase 1 的全参数 SBI
- [ ] **Baseline-2**：Phase 3 的嵌入误差 SBI（λ+α）
- [ ] **Baseline-3**：回写后的再次嵌入误差 SBI（λ+α）

### 3.3 新 CPS 数据集要求
- [ ] 个体/区域级 agent（带行为/状态）
- [ ] 时间序列轨迹（30+ 天）
- [ ] 最终系统输出（目标指标）
- [ ] 系统级 loss 定义能力

### 3.4 候选数据集评估
- [ ] G-Sim benchmark 数据集
- [ ] NYC 出租车数据：区域级流量 + 收入预测
- [ ] 电力负荷数据：区域级用电 + 峰值预测  
- [ ] SEIR 疫情数据：病例曲线 + 干预对比

---

## 4. 实验设计与评估

### 4.1 评估指标
- [ ] **误差指标**：RMSE/MAE、峰值误差、达峰时间误差
- [ ] **解释性指标**：α 的后验均值/中位数/95%区间、模块级"嫌疑度"排序
- [ ] **稳健性指标**：不同 seed、不同规模 N 的一致性
- [ ] **效率指标**：采样数、训练时长、推理时长

### 4.2 消融实验
- [ ] 无 α vs 有 α 的校准效果对比
- [ ] 选不同参数加 α 的效果
- [ ] 回写前/后 α 的变化
- [ ] Baseline-1/2/3 对比

### 4.3 实验环境
- [ ] 多数据集验证（口罩 + 1-2个新数据集）
- [ ] 不同复杂度任务（简单 vs 复杂模块交互）
- [ ] 不同数据规模（小规模 vs 大规模 agent 网络）

---

## 5. 关键接口与落盘规范

### 5.1 目录结构建议
```
calibrasim/
  calibration/
    sbi_baseline.py        # Phase 1
    sensitivity.py         # Phase 2
    embedded_error.py      # Phase 3（注入）
    sbi_with_error.py      # Phase 3（训练）
    diagnosis_report.py    # α 可视化与模块排序
    textgrad_trigger.py    # Phase 4
  configs/
    priors.yaml
    embedded_error.yaml
  outputs/
    runs/<run_id>/{params.json, posterior.pt, ppc.csv, alpha_report.json, plots/}
```

### 5.2 落盘约定
- [ ] `posterior_summary.json`：λ 与 α 的均值/中位/95%区间
- [ ] `alpha_report.json`：每个嵌入参数的 α 后验 + 嫌疑度
- [ ] `ppc.csv/ppc.png`：预测检查
- [ ] `seed_log.json`：全流程 seeds

---

## 6. 论文写作与发表

### 6.1 引言部分
- [ ] 研究动机：现有方法过于工程化，缺乏系统化优化框架
- [ ] 问题定义：模块化 CPS 模拟器的协同优化与校准挑战
- [ ] 核心贡献：首次将嵌入式模型误差引入 CPS 模拟器校准
- [ ] 论文结构预览

### 6.2 相关工作
- [ ] CPS 模拟器生成方法综述
- [ ] 嵌入式模型误差表示及其在贝叶斯校准里的位置
- [ ] 基于仿真的推理（SBI）方法
- [ ] 大模型在代码生成与优化中的应用

### 6.3 方法论
- [ ] CalibraSim 总体架构与流程
- [ ] 嵌入式模型误差在模拟器校准中的适配
- [ ] 四阶段校准机制（全参数SBI → 敏感性 → 嵌入误差 → 回写）
- [ ] TextGrad 反馈机制设计

### 6.4 实验部分
- [ ] 实验设置：数据集、评估指标、实验环境
- [ ] 主要结果：与 baseline 的对比
- [ ] 消融实验：各组件贡献分析
- [ ] 案例分析：具体任务的校准过程展示

---

## 7. 顶会投递计划

### 7.1 目标会议
- [ ] **NeurIPS 2025**（主要目标，6月截止）
- [ ] **ICLR 2026**（备选，10月截止）
- [ ] **AAAI 2026**（备选，8月截止）

### 7.2 关键时间节点
- [ ] 🔴 **2个月内**：完成 Phase 1-2（搬运ODD的SBI + 敏感性分析）
- [ ] 🔴 **3个月内**：完成 Phase 3-4（嵌入误差 + 回写验证）
- [ ] 🔴 **4个月内**：完成所有实验 + 初步结果分析
- [ ] 🔴 **5个月内**：论文初稿完成
- [ ] 🔴 **截止前1个月**：最终定稿 + 提交

---

## 🚀 最紧急的 3 步（优先级排序）

1. [ ] **完成 Phase 1：搬运 ODD 的 SBI 实现** → 建立稳定可运行 baseline
2. [ ] **完成 Phase 2：敏感性分析** → 识别 Top-K 参数
3. [ ] **完成 Phase 3：嵌入误差 SBI** → 验证核心方法

---

## 📊 进度统计

- **总任务数**: 45
- **已完成**: 8
- **进行中**: 0
- **待开始**: 37

---

## ✅ 已额外完成（未在原 TODO 中明确列出）

- [x] 新增 `calibrasim` 模式至 `orchestration/workflow_manager.py`，集成参数文件生成流程
- [x] 依赖注入与配置扩展：`config.yaml` 与 `orchestration/container.py` 注册 Calibrasim 专用 agents
- [x] 主入口 `main.py` 增加 `--mode calibrasim`
- [x] 模型规划模板（Calibrasim）格式安全加固：布尔与占位示例统一为字符串表达，降低 JSON 解析风险
- [x] 解析失败可观测性增强：落盘原始 LLM 响应至 `debug_output/`

---

## 🎯 成功标准

- [ ] 在口罩数据集上显著优于 SOCIA/G-Sim baseline
- [ ] 在至少 1 个新数据集上验证方法通用性
- [ ] 消融实验证明各组件有效性
- [ ] 论文被 NeurIPS/ICLR/AAAI 接收

---

## ⚠️ 风险管理

- **维度爆炸**：一次只给 1-2 个参数加 α；不做"给所有参数都加"
- **训练不稳定**：先用轨迹输入，summary-SBI 暂时备选；若启用 summary，真实端方差=0/ε
- **α 过拟合**：HalfNormal 先验 + α 上界试探；若 α 总被推大，检查模型/数据一致性与噪声设计
- **PCE**：列为"未来加速器"，不在当前主路径；除非有确证能稳定产出等价特征

---

*最后更新: 2024年12月*