#!/usr/bin/env python3
# 运行SOCIA-CAMEL示例

import os
import sys
import logging
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('SOCIA-CAMEL')

# 导入工作流管理器和模拟执行代理
from socia_camel.orchestration.workflow_manager import WorkflowManager
from socia_camel.agents.simulation_execution_agent import SimulationExecutionAgent

def run_full_workflow():
    """运行完整的模拟工作流"""
    # 示例任务描述
    task_description = "Create a simple epidemic simulation model that models the spread of a virus in a population of 1000 people."
    
    # 创建输出目录
    output_dir = "./socia_camel/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建工作流管理器
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    workflow_manager = WorkflowManager(config_path)
    workflow_manager.set_output_dir(output_dir)
    
    # 运行工作流
    logger.info(f"开始运行示例流行病模拟任务")
    result = workflow_manager.run_workflow(task_description)
    
    # 检查结果
    if result.get("success", False):
        logger.info("模拟执行成功!")
        logger.info(f"结果保存在: {output_dir}")
    else:
        logger.error(f"模拟执行失败: {result.get('error', '未知错误')}")

def run_simulation_only():
    """只运行模拟执行阶段"""
    # 创建输出目录
    output_dir = "./socia_camel/output/simulation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟执行代理
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    execution_agent = SimulationExecutionAgent(config_path)
    
    # 读取生成的代码
    code_file = os.path.join(os.path.dirname(__file__), "output", "simulation.py")
    if not os.path.exists(code_file):
        logger.error(f"找不到生成的代码文件: {code_file}")
        return
    
    with open(code_file, "r") as f:
        code = f.read()
    
    # 准备环境并执行模拟
    logger.info("开始执行模拟...")
    result = execution_agent.execute_simulation(code, output_dir)
    
    # 检查结果
    if result.get("success", False):
        logger.info("模拟执行成功!")
        logger.info(f"结果保存在: {output_dir}")
    else:
        logger.error(f"模拟执行失败: {result.get('error', '未知错误')}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行SOCIA-CAMEL示例')
    parser.add_argument('--simulation-only', action='store_true', help='只运行模拟执行阶段')
    args = parser.parse_args()
    
    if args.simulation_only:
        run_simulation_only()
    else:
        run_full_workflow()

if __name__ == "__main__":
    main() 