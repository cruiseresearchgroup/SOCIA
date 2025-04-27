"""
Templates module for SOCIA-CAMEL system.
Contains prompt templates for LLM interactions.
"""

import os

# 函数用于加载模板文件
def load_template(template_name):
    """
    加载指定的模板文件内容
    
    Args:
        template_name: 模板文件名，不包含路径
        
    Returns:
        模板文件内容字符串
    """
    templates_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(templates_dir, template_name)
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

# 导出的函数和变量
__all__ = ['load_template'] 