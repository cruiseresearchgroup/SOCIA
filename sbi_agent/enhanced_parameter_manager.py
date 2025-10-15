"""
增强的参数管理器
专门针对SOCIA项目设计，支持模块化参数空间构建和智能约束处理
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import networkx as nx

logger = logging.getLogger(__name__)

class SOCIAParameterManager:
    """SOCIA参数管理器"""
    
    def __init__(self):
        self.parameter_definitions = {}
        self.current_parameters = {}
        self.parameter_constraints = {}
        self.parameter_sensitivity = {}
        self.parameter_correlations = {}
        self.module_parameters = {}
        self.calibration_strategy = {}
    
    def load_parameter_configuration(self, 
                                   param_defs_path: Union[str, Path],
                                   current_params_path: Union[str, Path],
                                   model_plan_path: Union[str, Path]) -> None:
        """加载参数配置"""
        try:
            # 加载参数定义
            with open(param_defs_path, 'r', encoding='utf-8') as f:
                self.parameter_definitions = json.load(f)
            
            # 加载当前参数
            with open(current_params_path, 'r', encoding='utf-8') as f:
                self.current_parameters = json.load(f)
            
            # 加载模型计划
            with open(model_plan_path, 'r', encoding='utf-8') as f:
                model_plan = json.load(f)
            
            # 分析参数配置
            self._analyze_parameter_structure()
            self._build_module_parameter_mapping(model_plan)
            self._compute_parameter_constraints()
            self._analyze_parameter_sensitivity()
            self._analyze_parameter_correlations()
            
            logger.info(f"参数配置加载完成: {len(self.parameter_definitions)} 个参数")
        except Exception as e:
            logger.error(f"参数配置加载失败: {e}")
            raise
    
    def _analyze_parameter_structure(self) -> None:
        """分析参数结构"""
        for param_name, param_info in self.parameter_definitions.items():
            # 分析参数类型
            param_type = param_info.get('type', 'continuous')
            
            # 分析参数边界
            bounds = param_info.get('bounds', [0, 1])
            if isinstance(bounds, list) and len(bounds) == 2:
                bounds = tuple(bounds)
            
            # 分析参数先验
            prior = param_info.get('prior', 'uniform')
            
            # 分析参数变换
            transform = param_info.get('transform', 'none')
            
            self.parameter_constraints[param_name] = {
                'type': param_type,
                'bounds': bounds,
                'prior': prior,
                'transform': transform,
                'module': param_info.get('module', 'unknown'),
                'description': param_info.get('description', ''),
                'sensitivity': param_info.get('sensitivity', 'medium')
            }
    
    def _build_module_parameter_mapping(self, model_plan: Dict[str, Any]) -> None:
        """构建模块参数映射"""
        if 'modules' not in model_plan:
            logger.warning("模型计划中没有找到模块定义")
            return
        
        for module_name, module_info in model_plan['modules'].items():
            # 获取模块的可校准参数
            calibration_params = module_info.get('calibration_parameters', [])
            
            # 构建模块参数映射
            self.module_parameters[module_name] = {
                'calibration_parameters': calibration_params,
                'sbi_calibration': module_info.get('sbi_calibration', False),
                'target_signals': module_info.get('sbi_target_signals', []),
                'dependencies': module_info.get('dependencies', [])
            }
            
            logger.debug(f"模块参数映射: {module_name} -> {len(calibration_params)} 个参数")
    
    def _compute_parameter_constraints(self) -> None:
        """计算参数约束"""
        for param_name, constraints in self.parameter_constraints.items():
            # 基于参数类型设置约束
            if constraints['type'] == 'continuous':
                constraints['constraint_type'] = 'bounds'
                constraints['constraint_value'] = constraints['bounds']
            elif constraints['type'] == 'discrete':
                constraints['constraint_type'] = 'bounds'
                constraints['constraint_value'] = constraints['bounds']
            elif constraints['type'] == 'categorical':
                constraints['constraint_type'] = 'categories'
                constraints['constraint_value'] = constraints.get('categories', [])
            elif constraints['type'] == 'boolean':
                constraints['constraint_type'] = 'categories'
                constraints['constraint_value'] = [True, False]
            
            # 基于先验分布设置约束
            if constraints['prior'] == 'uniform':
                constraints['prior_params'] = constraints['bounds']
            elif constraints['prior'] == 'normal':
                mean = np.mean(constraints['bounds'])
                std = (constraints['bounds'][1] - constraints['bounds'][0]) / 4
                constraints['prior_params'] = {'mean': mean, 'std': std}
            elif constraints['prior'] == 'lognormal':
                constraints['prior_params'] = {'mean': 0, 'std': 1}
    
    def _analyze_parameter_sensitivity(self) -> None:
        """分析参数敏感性"""
        for param_name, constraints in self.parameter_constraints.items():
            # 基于参数描述和类型推断敏感性
            sensitivity_score = self._infer_parameter_sensitivity(param_name, constraints)
            
            self.parameter_sensitivity[param_name] = {
                'sensitivity_score': sensitivity_score,
                'importance_rank': 0,  # 将在后续计算
                'interaction_strength': 0.0,
                'stability': 'medium'
            }
        
        # 计算重要性排名
        self._compute_importance_ranking()
    
    def _infer_parameter_sensitivity(self, param_name: str, constraints: Dict[str, Any]) -> float:
        """推断参数敏感性"""
        # 基于参数名称推断敏感性
        high_sensitivity_keywords = ['rate', 'probability', 'effect', 'influence', 'threshold']
        medium_sensitivity_keywords = ['weight', 'factor', 'multiplier', 'strength']
        low_sensitivity_keywords = ['offset', 'bias', 'noise', 'random']
        
        param_lower = param_name.lower()
        
        if any(keyword in param_lower for keyword in high_sensitivity_keywords):
            return 0.8
        elif any(keyword in param_lower for keyword in medium_sensitivity_keywords):
            return 0.5
        elif any(keyword in param_lower for keyword in low_sensitivity_keywords):
            return 0.2
        else:
            return 0.5  # 默认中等敏感性
    
    def _compute_importance_ranking(self) -> None:
        """计算重要性排名"""
        # 按敏感性分数排序
        sorted_params = sorted(
            self.parameter_sensitivity.items(),
            key=lambda x: x[1]['sensitivity_score'],
            reverse=True
        )
        
        for rank, (param_name, _) in enumerate(sorted_params):
            self.parameter_sensitivity[param_name]['importance_rank'] = rank + 1
    
    def _analyze_parameter_correlations(self) -> None:
        """分析参数相关性"""
        # 基于参数名称和模块归属推断相关性
        for param_name, constraints in self.parameter_constraints.items():
            correlations = {}
            
            for other_param_name, other_constraints in self.parameter_constraints.items():
                if param_name != other_param_name:
                    # 基于模块归属计算相关性
                    if constraints['module'] == other_constraints['module']:
                        correlations[other_param_name] = 0.3  # 同模块参数有中等相关性
                    elif self._are_related_modules(constraints['module'], other_constraints['module']):
                        correlations[other_param_name] = 0.1  # 相关模块参数有低相关性
                    else:
                        correlations[other_param_name] = 0.0  # 无关模块参数无相关性
            
            self.parameter_correlations[param_name] = correlations
    
    def _are_related_modules(self, module1: str, module2: str) -> bool:
        """判断模块是否相关"""
        # 基于模块名称判断相关性
        related_module_pairs = [
            ('InformationDiffusion', 'PolicyAndMessaging'),
            ('PolicyAndMessaging', 'SocialInfluenceAdoption'),
            ('InformationDiffusion', 'SocialInfluenceAdoption')
        ]
        
        return (module1, module2) in related_module_pairs or (module2, module1) in related_module_pairs
    
    def get_module_parameters(self, module_name: str) -> Dict[str, Any]:
        """获取模块参数"""
        if module_name not in self.module_parameters:
            return {}
        
        module_info = self.module_parameters[module_name]
        calibration_params = module_info['calibration_parameters']
        
        module_params = {}
        for param_name in calibration_params:
            if param_name in self.parameter_definitions:
                module_params[param_name] = {
                    'definition': self.parameter_definitions[param_name],
                    'constraints': self.parameter_constraints[param_name],
                    'sensitivity': self.parameter_sensitivity[param_name],
                    'correlations': self.parameter_correlations[param_name]
                }
        
        return module_params
    
    def build_parameter_space_for_module(self, module_name: str) -> Dict[str, Any]:
        """为模块构建参数空间"""
        module_params = self.get_module_parameters(module_name)
        
        if not module_params:
            logger.warning(f"模块 {module_name} 没有可校准参数")
            return {}
        
        parameter_space = {
            'module_name': module_name,
            'parameters': module_params,
            'parameter_count': len(module_params),
            'parameter_names': list(module_params.keys()),
            'constraints': {name: info['constraints'] for name, info in module_params.items()},
            'sensitivity': {name: info['sensitivity'] for name, info in module_params.items()},
            'correlations': {name: info['correlations'] for name, info in module_params.items()}
        }
        
        return parameter_space
    
    def get_parameter_transforms(self) -> Dict[str, Callable]:
        """获取参数变换函数"""
        transforms = {
            'log': lambda x: np.log(np.maximum(x, 1e-10)),
            'logit': lambda x: np.log(np.maximum(x, 1e-10) / np.maximum(1 - x, 1e-10)),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
            'square': lambda x: x ** 2,
            'none': lambda x: x
        }
        return transforms
    
    def apply_parameter_transforms(self, params: Dict[str, Any], 
                                 transform_direction: str = 'forward') -> Dict[str, Any]:
        """应用参数变换"""
        transformed_params = {}
        transforms = self.get_parameter_transforms()
        
        for param_name, value in params.items():
            if param_name in self.parameter_constraints:
                transform_type = self.parameter_constraints[param_name]['transform']
                
                if transform_type in transforms:
                    if transform_direction == 'forward':
                        transformed_params[param_name] = transforms[transform_type](value)
                    elif transform_direction == 'inverse':
                        # 实现逆变换
                        if transform_type == 'log':
                            transformed_params[param_name] = np.exp(value)
                        elif transform_type == 'logit':
                            transformed_params[param_name] = 1 / (1 + np.exp(-value))
                        elif transform_type == 'sigmoid':
                            transformed_params[param_name] = -np.log(1 / value - 1)
                        elif transform_type == 'sqrt':
                            transformed_params[param_name] = value ** 2
                        elif transform_type == 'square':
                            transformed_params[param_name] = np.sqrt(value)
                        else:
                            transformed_params[param_name] = value
                    else:
                        transformed_params[param_name] = value
                else:
                    transformed_params[param_name] = value
            else:
                transformed_params[param_name] = value
        
        return transformed_params
    
    def reduce_parameter_dimensions(self, module_name: str, 
                                   target_dimensions: int = 10) -> Dict[str, Any]:
        """降维参数空间"""
        module_params = self.get_module_parameters(module_name)
        
        if len(module_params) <= target_dimensions:
            return module_params
        
        # 基于敏感性排序选择最重要的参数
        param_importance = []
        for param_name, param_info in module_params.items():
            sensitivity_score = param_info['sensitivity']['sensitivity_score']
            importance_rank = param_info['sensitivity']['importance_rank']
            
            # 综合评分
            combined_score = sensitivity_score * (1 / importance_rank)
            param_importance.append((param_name, combined_score))
        
        # 按综合评分排序
        param_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最重要的参数
        selected_params = dict(param_importance[:target_dimensions])
        
        logger.info(f"参数空间降维: {module_name} {len(module_params)} -> {len(selected_params)}")
        return selected_params
    
    def get_parameter_bounds(self, module_name: str) -> Dict[str, Tuple[float, float]]:
        """获取模块参数边界"""
        module_params = self.get_module_parameters(module_name)
        bounds = {}
        
        for param_name, param_info in module_params.items():
            constraints = param_info['constraints']
            if constraints['constraint_type'] == 'bounds':
                bounds[param_name] = constraints['constraint_value']
        
        return bounds
    
    def validate_parameters(self, params: Dict[str, Any], module_name: str) -> Tuple[bool, List[str]]:
        """验证参数"""
        is_valid = True
        errors = []
        
        module_params = self.get_module_parameters(module_name)
        
        for param_name, value in params.items():
            if param_name not in module_params:
                continue
            
            param_info = module_params[param_name]
            constraints = param_info['constraints']
            
            # 检查参数类型
            if constraints['type'] == 'continuous' and not isinstance(value, (int, float)):
                is_valid = False
                errors.append(f"参数 {param_name} 应该是数值类型")
                continue
            
            # 检查参数边界
            if constraints['constraint_type'] == 'bounds':
                bounds = constraints['constraint_value']
                if not (bounds[0] <= value <= bounds[1]):
                    is_valid = False
                    errors.append(f"参数 {param_name} 超出边界 [{bounds[0]}, {bounds[1]}]")
            
            # 检查分类参数
            elif constraints['constraint_type'] == 'categories':
                categories = constraints['constraint_value']
                if value not in categories:
                    is_valid = False
                    errors.append(f"参数 {param_name} 不在允许的类别中: {categories}")
        
        return is_valid, errors
    
    def get_calibration_strategy(self) -> Dict[str, Any]:
        """获取校准策略"""
        strategy = {
            'calibration_modules': list(self.module_parameters.keys()),
            'module_parameters': self.module_parameters,
            'parameter_constraints': self.parameter_constraints,
            'parameter_sensitivity': self.parameter_sensitivity,
            'parameter_correlations': self.parameter_correlations
        }
        
        return strategy
    
    def export_parameters(self, params: Dict[str, Any], 
                         output_path: Union[str, Path]) -> None:
        """导出参数"""
        export_data = {
            'parameters': params,
            'parameter_info': {
                name: {
                    'constraints': self.parameter_constraints.get(name, {}),
                    'sensitivity': self.parameter_sensitivity.get(name, {}),
                    'correlations': self.parameter_correlations.get(name, {})
                }
                for name in params.keys()
            },
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"参数导出完成: {output_path}")
    
    def get_parameter_summary(self) -> Dict[str, Any]:
        """获取参数摘要"""
        summary = {
            'total_parameters': len(self.parameter_definitions),
            'calibration_modules': len(self.module_parameters),
            'parameter_types': {
                'continuous': sum(1 for c in self.parameter_constraints.values() if c['type'] == 'continuous'),
                'discrete': sum(1 for c in self.parameter_constraints.values() if c['type'] == 'discrete'),
                'categorical': sum(1 for c in self.parameter_constraints.values() if c['type'] == 'categorical'),
                'boolean': sum(1 for c in self.parameter_constraints.values() if c['type'] == 'boolean')
            },
            'high_sensitivity_parameters': [
                name for name, info in self.parameter_sensitivity.items()
                if info['sensitivity_score'] > 0.7
            ],
            'module_parameter_counts': {
                module: len(info['calibration_parameters'])
                for module, info in self.module_parameters.items()
            }
        }
        
        return summary





