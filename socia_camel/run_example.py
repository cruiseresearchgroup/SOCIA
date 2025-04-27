#!/usr/bin/env python3
# 运行SOCIA-CAMEL示例

import os
import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('SOCIA-CAMEL')

# 导入工作流管理器
from orchestration.workflow_manager import WorkflowManager

def main():
    """运行示例流行病模拟"""
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

if __name__ == "__main__":
    main() 