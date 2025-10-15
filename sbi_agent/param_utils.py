"""
参数管理工具模块
提供参数验证、边界检查、类型转换、SOCIA参数格式支持等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """参数类型枚举"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    INTEGER = "integer"

class ParameterConstraint:
    """参数约束类"""
    
    def __init__(self, param_name: str, param_type: ParameterType, 
                 bounds: Tuple[float, float] = None, 
                 categories: List[Any] = None,
                 default_value: Any = None):
        self.param_name = param_name
        self.param_type = param_type
        self.bounds = bounds
        self.categories = categories
        self.default_value = default_value
    
    def validate(self, value: Any) -> bool:
        """验证参数值是否符合约束"""
        try:
            if self.param_type == ParameterType.CONTINUOUS:
                if self.bounds is None:
                    return True
                return self.bounds[0] <= value <= self.bounds[1]
            
            elif self.param_type == ParameterType.DISCRETE:
                if self.bounds is None:
                    return True
                return self.bounds[0] <= value <= self.bounds[1] and isinstance(value, (int, float))
            
            elif self.param_type == ParameterType.CATEGORICAL:
                if self.categories is None:
                    return True
                return value in self.categories
            
            elif self.param_type == ParameterType.BOOLEAN:
                return isinstance(value, bool)
            
            elif self.param_type == ParameterType.INTEGER:
                return isinstance(value, int)
            
            return True
        except Exception as e:
            logger.warning(f"参数验证失败: {self.param_name}={value}, 错误: {e}")
            return False
    
    def transform(self, value: Any) -> Any:
        """转换参数值"""
        try:
            if self.param_type == ParameterType.CONTINUOUS:
                return float(value)
            elif self.param_type == ParameterType.DISCRETE:
                return float(value)
            elif self.param_type == ParameterType.INTEGER:
                return int(value)
            elif self.param_type == ParameterType.BOOLEAN:
                return bool(value)
            else:
                return value
        except Exception as e:
            logger.warning(f"参数转换失败: {self.param_name}={value}, 错误: {e}")
            return self.default_value

class ParameterManager:
    """参数管理器类"""
    
    def __init__(self):
        self.constraints: Dict[str, ParameterConstraint] = {}
        self.current_params: Dict[str, Any] = {}
        self.param_definitions: Dict[str, Any] = {}
    
    def load_parameter_definitions(self, definitions: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """加载参数定义"""
        self.param_definitions = definitions
        
        # 处理列表格式的参数定义
        if isinstance(definitions, list):
            for param_info in definitions:
                param_name = param_info.get('key', '')
                param_type = self._parse_parameter_type(param_info.get('dtype', 'continuous'))
                bounds = param_info.get('bounds')
                categories = param_info.get('categories')
                default_value = param_info.get('default')
                
                constraint = ParameterConstraint(
                    param_name=param_name,
                    param_type=param_type,
                    bounds=bounds,
                    categories=categories,
                    default_value=default_value
                )
                
                self.constraints[param_name] = constraint
                logger.info(f"参数定义加载: {param_name} ({param_type.value})")
        # 处理字典格式的参数定义
        elif isinstance(definitions, dict):
            for param_name, param_info in definitions.items():
                param_type = self._parse_parameter_type(param_info.get('type', 'continuous'))
                bounds = param_info.get('bounds')
                categories = param_info.get('categories')
                default_value = param_info.get('default')
                
                constraint = ParameterConstraint(
                    param_name=param_name,
                    param_type=param_type,
                    bounds=bounds,
                    categories=categories,
                    default_value=default_value
                )
                
                self.constraints[param_name] = constraint
                logger.info(f"参数定义加载: {param_name} ({param_type.value})")
    
    def load_current_parameters(self, params: Dict[str, Any]) -> None:
        """加载当前参数"""
        self.current_params = params.copy()
        logger.info(f"当前参数加载: {len(params)} 个参数")
    
    def _parse_parameter_type(self, type_str: str) -> ParameterType:
        """解析参数类型"""
        type_mapping = {
            'continuous': ParameterType.CONTINUOUS,
            'discrete': ParameterType.DISCRETE,
            'categorical': ParameterType.CATEGORICAL,
            'boolean': ParameterType.BOOLEAN,
            'integer': ParameterType.INTEGER
        }
        return type_mapping.get(type_str.lower(), ParameterType.CONTINUOUS)
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证参数"""
        is_valid = True
        errors = []
        
        for param_name, value in params.items():
            if param_name not in self.constraints:
                logger.warning(f"未知参数: {param_name}")
                continue
            
            constraint = self.constraints[param_name]
            if not constraint.validate(value):
                is_valid = False
                error_msg = f"参数 {param_name}={value} 不符合约束"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return is_valid, errors
    
    def transform_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """转换参数"""
        transformed = {}
        
        for param_name, value in params.items():
            if param_name in self.constraints:
                constraint = self.constraints[param_name]
                transformed[param_name] = constraint.transform(value)
            else:
                transformed[param_name] = value
                logger.warning(f"参数 {param_name} 没有约束定义，保持原值")
        
        return transformed
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """获取参数边界"""
        bounds = {}
        
        for param_name, constraint in self.constraints.items():
            if constraint.bounds is not None:
                bounds[param_name] = constraint.bounds
        
        return bounds
    
    def get_parameter_types(self) -> Dict[str, ParameterType]:
        """获取参数类型"""
        return {name: constraint.param_type for name, constraint in self.constraints.items()}
    
    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """更新参数"""
        # 验证新参数
        is_valid, errors = self.validate_parameters(new_params)
        if not is_valid:
            raise ValueError(f"参数验证失败: {errors}")
        
        # 转换参数
        transformed_params = self.transform_parameters(new_params)
        
        # 更新当前参数
        self.current_params.update(transformed_params)
        logger.info(f"参数更新成功: {len(transformed_params)} 个参数")
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """获取参数摘要"""
        summary = {
            'total_parameters': len(self.constraints),
            'parameter_types': self.get_parameter_types(),
            'parameter_bounds': self.get_parameter_bounds(),
            'current_values': self.current_params.copy()
        }
        return summary
    
    def export_parameters(self, file_path: str) -> None:
        """导出参数到文件"""
        import json
        from pathlib import Path
        
        export_data = {
            'parameters': self.current_params,
            'definitions': self.param_definitions,
            'summary': self.get_parameter_summary()
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"参数导出成功: {file_path}")
    
    def reset_to_defaults(self) -> None:
        """重置为默认值"""
        for param_name, constraint in self.constraints.items():
            if constraint.default_value is not None:
                self.current_params[param_name] = constraint.default_value
        
        logger.info("参数已重置为默认值")
