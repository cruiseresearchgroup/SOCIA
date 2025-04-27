#!/usr/bin/env python3
# SOCIA-CAMEL: 基于CAMEL架构的社会模拟智能体系统

import argparse
import logging
import os
import sys
import yaml
import json
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 使用相对导入
from orchestration.workflow_manager import WorkflowManager
from agents.task_understanding_agent import TaskUnderstandingAgent
from agents.model_planning_agent import ModelPlanningAgent
from agents.code_generation_agent import CodeGenerationAgent
from agents.simulation_execution_agent import SimulationExecutionAgent
from utils.llm_utils import load_api_key

def setup_logging(output_path=None):
    """配置日志系统"""
    # 尝试从配置文件读取日志级别
    log_level = logging.INFO  # 默认级别为INFO
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            log_level_str = config.get("logging", {}).get("level", "INFO")
            log_level = getattr(logging, log_level_str)
    except Exception as e:
        print(f"警告：无法从配置文件读取日志级别: {e}")
    
    # 创建处理器列表
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # 如果提供了输出路径，添加文件处理器
    if output_path:
        try:
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            # 创建日志文件路径
            log_file_path = os.path.join(output_path, "socia_camel.log")
            
            # 添加文件处理器到处理器列表
            handlers.append(logging.FileHandler(log_file_path))
            print(f"日志记录到文件: {log_file_path}")
        except Exception as e:
            print(f"警告：无法设置日志文件记录: {e}")
    
    # 配置日志系统
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('SOCIA-CAMEL')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SOCIA-CAMEL：基于CAMEL架构的社会模拟智能体系统')
    parser.add_argument('--task', type=str, help='模拟任务描述')
    parser.add_argument('--data', type=str, help='输入数据目录路径')
    parser.add_argument('--output', type=str, default='./output', help='输出目录路径')
    parser.add_argument('--config', type=str, default='./config.yaml', help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--run-example', action='store_true', help='运行示例流行病模拟')
    parser.add_argument('--setup-api-key', action='store_true', help='设置API密钥')
    
    args = parser.parse_args()
    
    # 验证在不运行示例和不设置API密钥时必须提供--task参数
    if not args.run_example and not args.setup_api_key and not args.task:
        parser.error("除非指定了--run-example或--setup-api-key，否则必须提供--task参数")
    
    return args

def setup_api_key():
    """设置API密钥"""
    print("设置API密钥")
    
    providers = ["OPENAI", "GEMINI", "ANTHROPIC"]
    keys = {}
    
    for provider in providers:
        key = input(f"请输入{provider} API密钥（留空则跳过）: ").strip()
        if key:
            keys[f"{provider}_API_KEY"] = key
    
    if not keys:
        print("未提供任何API密钥，操作取消")
        return False
    
    # 写入keys.py文件
    try:
        # 获取当前脚本所在目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建可能的keys.py路径
        potential_paths = [
            os.path.join(os.getcwd(), "keys.py"),  # 当前工作目录
            os.path.join(current_dir, "keys.py"),  # socia_camel目录
            os.path.join(current_dir, "..", "keys.py"),  # 项目根目录
        ]
        
        existing_file = None
        for path in potential_paths:
            if os.path.exists(path):
                existing_file = path
                print(f"发现现有的API密钥文件: {path}")
                break
        
        # 如果没有找到现有文件，则优先选择项目根目录
        if not existing_file:
            root_dir = os.path.join(current_dir, "..")
            file_path = os.path.join(root_dir, "keys.py")
        else:
            file_path = existing_file
        
        with open(file_path, 'w') as f:
            f.write("# API Keys for SOCIA-CAMEL\n\n")
            for key_name, key_value in keys.items():
                f.write(f'{key_name} = "{key_value}"\n')
        
        print(f"API密钥已保存到 {os.path.abspath(file_path)}")
        return True
    except Exception as e:
        print(f"保存API密钥失败: {e}")
        return False

def run_example_simulation(output_dir: str, logger: logging.Logger):
    """运行示例流行病模拟"""
    logger.info("运行示例流行病模拟")
    
    # 示例任务描述
    task_description = "Create a simple epidemic simulation model that models the spread of a virus in a population of 1000 people."
    
    # 创建工作流管理器
    workflow_manager = WorkflowManager("./config.yaml")
    workflow_manager.set_output_dir(output_dir)
    
    # 运行工作流
    result = workflow_manager.run_workflow(task_description)
    
    # 返回结果
    return result

def run_workflow(task_description: str, output_dir: str, config_path: str, logger: logging.Logger):
    """运行自定义工作流"""
    logger.info(f"运行自定义工作流: {task_description}")
    
    # 创建工作流管理器
    workflow_manager = WorkflowManager(config_path)
    workflow_manager.set_output_dir(output_dir)
    
    # 运行工作流
    result = workflow_manager.run_workflow(task_description)
    
    # 返回结果
    return result

def print_result_summary(result: Dict[str, Any], logger: logging.Logger):
    """打印结果摘要"""
    if not result.get("success", False):
        logger.error(f"模拟执行失败: {result.get('error', '未知错误')}")
        return
    
    task_analysis = result.get("task_analysis", {})
    model_plan = result.get("model_plan", {})
    simulation_result = result.get("simulation_result", {})
    
    print("\n" + "="*50)
    print("模拟执行成功!")
    print("="*50)
    
    print("\n任务类型:", task_analysis.get("task_analysis", {}).get("任务类型", "未知"))
    print("模拟目标:", task_analysis.get("task_analysis", {}).get("模拟目标", "未知"))
    
    print("\n选择的模型:", model_plan.get("model_selection", {}).get("推荐模型", "未知"))
    
    analysis = simulation_result.get("analysis", {})
    print("\n执行状态:", analysis.get("执行状态", "未知"))
    
    if "关键指标" in analysis and analysis["关键指标"]:
        print("\n关键指标:")
        for idx, indicator in enumerate(analysis["关键指标"], 1):
            print(f"  {idx}. {indicator.get('指标名称', '未知')}: {indicator.get('指标值', '未知')}")
    
    print("\n总体分析:", analysis.get("总体分析", "未提供"))
    print("\n结论:", analysis.get("结论", "未提供"))
    
    print("\n"+"="*50)
    print(f"详细结果已保存到：{os.path.abspath(result.get('output_dir', './output'))}")
    print("="*50)

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志系统
    logger = setup_logging(args.output if not args.setup_api_key else None)
    
    # 如果是设置API密钥
    if args.setup_api_key:
        if setup_api_key():
            print("API密钥设置成功!")
        else:
            print("API密钥设置失败!")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # 运行示例或自定义任务
        if args.run_example:
            result = run_example_simulation(args.output, logger)
        else:
            result = run_workflow(args.task, args.output, args.config, logger)
        
        # 打印结果摘要
        print_result_summary(result, logger)
        
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=args.debug)
        print(f"执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 