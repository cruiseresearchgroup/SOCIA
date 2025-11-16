#!/usr/bin/env python3
"""
独立测试 Model Planning Agent 的脚本
用于快速验证 model plan agent 的功能是否正常，无需运行整个流程
"""

import os
import sys
import json
import logging
import yaml
from typing import Dict, Any

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.model_planning.agent import ModelPlanningAgent

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_model_plan.log')
        ]
    )

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"错误：无法加载配置文件: {e}")
        return {}

def load_existing_task_spec() -> Dict[str, Any]:
    """加载现有的任务规范文件"""
    task_spec_file = "output/mask_adoption_calibrasim_test/task_spec_iter_1.json"
    try:
        with open(task_spec_file, 'r', encoding='utf-8') as f:
            task_spec = json.load(f)
        # 移除错误字段
        if "error" in task_spec:
            del task_spec["error"]
        return task_spec
    except Exception as e:
        print(f"警告：无法加载现有任务规范文件 {task_spec_file}: {e}")
        return create_fallback_task_spec()

def load_existing_data_analysis() -> Dict[str, Any]:
    """加载现有的数据分析文件"""
    data_analysis_file = "output/mask_adoption_calibrasim_test/data_analysis_iter_1.json"
    try:
        with open(data_analysis_file, 'r', encoding='utf-8') as f:
            data_analysis = json.load(f)
        # 移除错误字段
        if "error" in data_analysis:
            del data_analysis["error"]
        return data_analysis
    except Exception as e:
        print(f"警告：无法加载现有数据分析文件 {data_analysis_file}: {e}")
        return create_fallback_data_analysis()

def create_fallback_task_spec() -> Dict[str, Any]:
    """创建备用任务规范"""
    return {
        "title": "口罩采用行为模拟",
        "description": "开发一个多智能体模拟系统，模拟口罩佩戴行为通过社交网络的传播",
        "simulation_focus": [
            "预测智能体在第30-39天的口罩佩戴行为",
            "模拟社交影响和政府干预导致的行为变化"
        ],
        "data_folder": "data_fitting/mask_adoption_data/",
        "data_files": {
            "agent_attributes.csv": "包含每个智能体的人口统计和行为属性",
            "social_network.json": "包含结构化社交网络数据",
            "train_data.csv": "前30天的时间序列数据，用于训练模型"
        },
        "evaluation_metrics": {
            "RMSE": {
                "description": "均方根误差，衡量口罩采用率预测的整体准确性",
                "interpretation": "由于0-1范围，可直接解释为百分比偏差",
                "formula": "RMSE = sqrt(sum((predicted_rate - actual_rate)^2) / n)"
            }
        }
    }

def create_fallback_data_analysis() -> Dict[str, Any]:
    """创建备用数据分析结果"""
    return {
        "data_summary": {
            "key_patterns": [
                {
                    "name": "异质性风险感知",
                    "description": "智能体具有连续的风险感知值，在人群中变化很大",
                    "relevance": "驱动基线口罩采用和对社交/信息信号的敏感性"
                }
            ]
        },
        "file_references": {
            "processed_agent_attributes": "data_fitting/mask_adoption_data/agent_attributes.csv",
            "processed_social_network": "data_fitting/mask_adoption_data/social_network.json",
            "processed_train_data": "data_fitting/mask_adoption_data/train_data.csv"
        }
    }

def test_model_plan_agent():
    """测试 Model Planning Agent"""
    print("=" * 60)
    print("开始测试 Model Planning Agent")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger("TestModelPlan")
    
    # 加载配置
    config = load_config()
    if not config:
        print("错误：无法加载配置，退出测试")
        return False
    
    # 创建 agent 配置 - 使用 Calibrasim 配置
    agent_config = config.get("agents", {}).get("model_planning_calibrasim", {})
    if not agent_config:
        print("警告：未找到 model_planning_calibrasim 配置，使用默认配置")
        agent_config = {
            "prompt_template": "templates/Calibrasim_model_planning_prompt.txt",
            "output_format": "json"
        }
    else:
        # 确保使用正确的 prompt 模板
        agent_config["prompt_template"] = "templates/Calibrasim_model_planning_prompt.txt"
        print(f"使用 Calibrasim prompt 模板: {agent_config['prompt_template']}")
    
    try:
        # 初始化 Model Planning Agent
        print("正在初始化 Model Planning Agent...")
        agent = ModelPlanningAgent(agent_config)
        print("✓ Agent 初始化成功")
        
        # 准备测试数据 - 使用现有文件
        print("正在加载现有数据文件...")
        task_spec = load_existing_task_spec()
        data_analysis = load_existing_data_analysis()
        print("✓ 数据文件加载完成")
        print(f"  - 任务规范: {len(json.dumps(task_spec))} 字符")
        print(f"  - 数据分析: {len(json.dumps(data_analysis))} 字符")
        
        # 执行模型规划
        print("正在执行模型规划...")
        print("-" * 40)
        model_plan = agent.process(task_spec=task_spec, data_analysis=data_analysis)
        print("-" * 40)
        print("✓ 模型规划完成")
        
        # 验证输出
        print("正在验证输出...")
        if isinstance(model_plan, dict):
            print("✓ 输出格式正确 (字典)")
            
            # 检查关键字段
            required_fields = ["model_type", "description", "entities", "behaviors", "modules", "parameters"]
            missing_fields = [field for field in required_fields if field not in model_plan]
            
            if missing_fields:
                print(f"⚠ 缺少关键字段: {missing_fields}")
            else:
                print("✓ 包含所有关键字段")
            
            # 显示模型计划摘要
            print("\n模型计划摘要:")
            print(f"  模型类型: {model_plan.get('model_type', 'N/A')}")
            print(f"  描述: {model_plan.get('description', 'N/A')[:100]}...")
            print(f"  实体数量: {len(model_plan.get('entities', []))}")
            print(f"  行为数量: {len(model_plan.get('behaviors', []))}")
            print(f"  模块数量: {len(model_plan.get('modules', []))}")
            print(f"  参数数量: {len(model_plan.get('parameters', []))}")
            
        else:
            print(f"✗ 输出格式错误: {type(model_plan)}")
            return False
        
        # 保存结果
        output_file = "test_model_plan_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_plan, f, indent=2, ensure_ascii=False)
        print(f"✓ 结果已保存到: {output_file}")
        
        print("\n" + "=" * 60)
        print("Model Planning Agent 测试完成！")
        print("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("Model Planning Agent 独立测试工具")
    print("此工具用于快速验证 model plan agent 的功能")
    print()
    
    success = test_model_plan_agent()
    
    if success:
        print("\n🎉 测试成功！Model Planning Agent 功能正常")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
