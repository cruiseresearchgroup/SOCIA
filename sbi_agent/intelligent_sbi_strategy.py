"""
智能SBI策略模块
根据模块类型选择SBI方法，实现自适应参数空间调整，支持多模块联合校准
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from abc import ABC, abstractmethod
from enum import Enum
import time
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class SBIMethod(Enum):
    """SBI方法枚举"""
    SNPE = "snpe"
    SNLE = "snle"
    SNRE = "snre"
    ABC = "abc"
    MCMC = "mcmc"

class ModuleType(Enum):
    """模块类型枚举"""
    INFORMATION_DIFFUSION = "information_diffusion"
    POLICY_MESSAGING = "policy_messaging"
    SOCIAL_INFLUENCE = "social_influence"
    NETWORK_ENGINE = "network_engine"
    UNKNOWN = "unknown"

class SBIStrategySelector:
    """SBI策略选择器"""
    
    def __init__(self):
        self.module_sbi_mapping = {
            ModuleType.INFORMATION_DIFFUSION: SBIMethod.SNPE,
            ModuleType.POLICY_MESSAGING: SBIMethod.SNPE,
            ModuleType.SOCIAL_INFLUENCE: SBIMethod.SNLE,
            ModuleType.NETWORK_ENGINE: SBIMethod.ABC
        }
        
        self.parameter_complexity_thresholds = {
            'low': 5,
            'medium': 15,
            'high': 30
        }
    
    def select_sbi_method(self, module_name: str, parameter_count: int, 
                         data_complexity: str = 'medium') -> SBIMethod:
        """选择SBI方法"""
        module_type = self._classify_module_type(module_name)
        
        # 基于模块类型选择基础方法
        base_method = self.module_sbi_mapping.get(module_type, SBIMethod.SNPE)
        
        # 基于参数复杂度调整
        if parameter_count > self.parameter_complexity_thresholds['high']:
            # 高维参数空间，使用更稳定的方法
            if base_method == SBIMethod.SNPE:
                return SBIMethod.SNLE
            elif base_method == SBIMethod.SNLE:
                return SBIMethod.ABC
        
        # 基于数据复杂度调整
        if data_complexity == 'high':
            if base_method == SBIMethod.SNPE:
                return SBIMethod.SNRE
        
        return base_method
    
    def _classify_module_type(self, module_name: str) -> ModuleType:
        """分类模块类型"""
        module_lower = module_name.lower()
        
        if 'information' in module_lower and 'diffusion' in module_lower:
            return ModuleType.INFORMATION_DIFFUSION
        elif 'policy' in module_lower and 'messaging' in module_lower:
            return ModuleType.POLICY_MESSAGING
        elif 'social' in module_lower and 'influence' in module_lower:
            return ModuleType.SOCIAL_INFLUENCE
        elif 'network' in module_lower and 'engine' in module_lower:
            return ModuleType.NETWORK_ENGINE
        else:
            return ModuleType.UNKNOWN
    
    def get_sbi_parameters(self, method: SBIMethod, module_name: str, 
                          parameter_count: int) -> Dict[str, Any]:
        """获取SBI参数"""
        base_params = {
            'num_simulations': max(1000, parameter_count * 100),
            'num_rounds': 3,
            'num_atoms': 10,
            'training_batch_size': 50,
            'validation_fraction': 0.1
        }
        
        if method == SBIMethod.SNPE:
            return {
                **base_params,
                'density_estimator': 'maf',
                'num_transforms': 5,
                'learning_rate': 0.001
            }
        elif method == SBIMethod.SNLE:
            return {
                **base_params,
                'density_estimator': 'maf',
                'num_transforms': 3,
                'learning_rate': 0.0005
            }
        elif method == SBIMethod.SNRE:
            return {
                **base_params,
                'classifier': 'resnet',
                'learning_rate': 0.001,
                'num_epochs': 100
            }
        elif method == SBIMethod.ABC:
            return {
                **base_params,
                'distance_threshold': 0.1,
                'num_samples': 10000,
                'epsilon': 0.05
            }
        else:
            return base_params

class AdaptiveParameterSpaceAdjuster:
    """自适应参数空间调整器"""
    
    def __init__(self):
        self.adjustment_history = []
        self.parameter_sensitivity_cache = {}
    
    def adjust_parameter_space(self, module_name: str, 
                             current_params: Dict[str, Any],
                             calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """调整参数空间"""
        adjustment = {
            'module_name': module_name,
            'adjustments': {},
            'reason': '',
            'timestamp': time.time()
        }
        
        # 基于校准结果调整参数空间
        if not calibration_results.get('convergence_achieved', False):
            adjustment = self._adjust_for_non_convergence(module_name, current_params, calibration_results)
        else:
            final_metrics = calibration_results.get('final_metrics', {})
            if final_metrics.get('rmse', 1.0) > 0.3:
                adjustment = self._adjust_for_poor_performance(module_name, current_params, final_metrics)
            else:
                adjustment = self._adjust_for_good_performance(module_name, current_params, final_metrics)
        
        self.adjustment_history.append(adjustment)
        return adjustment
    
    def _adjust_for_non_convergence(self, module_name: str, 
                                  current_params: Dict[str, Any],
                                  calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """为不收敛情况调整参数空间"""
        adjustments = {}
        
        # 扩大参数边界
        for param_name, param_value in current_params.items():
            if isinstance(param_value, (int, float)):
                # 扩大边界到原来的1.5倍
                adjustments[param_name] = {
                    'action': 'expand_bounds',
                    'factor': 1.5,
                    'reason': 'non_convergence'
                }
        
        return {
            'module_name': module_name,
            'adjustments': adjustments,
            'reason': 'non_convergence',
            'timestamp': time.time()
        }
    
    def _adjust_for_poor_performance(self, module_name: str,
                                   current_params: Dict[str, Any],
                                   final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """为性能差的情况调整参数空间"""
        adjustments = {}
        
        # 基于敏感性调整参数
        for param_name, param_value in current_params.items():
            if isinstance(param_value, (int, float)):
                sensitivity = self._get_parameter_sensitivity(param_name)
                
                if sensitivity > 0.7:  # 高敏感性参数
                    adjustments[param_name] = {
                        'action': 'refine_bounds',
                        'factor': 0.8,
                        'reason': 'high_sensitivity_poor_performance'
                    }
                else:  # 低敏感性参数
                    adjustments[param_name] = {
                        'action': 'expand_bounds',
                        'factor': 1.2,
                        'reason': 'low_sensitivity_poor_performance'
                    }
        
        return {
            'module_name': module_name,
            'adjustments': adjustments,
            'reason': 'poor_performance',
            'timestamp': time.time()
        }
    
    def _adjust_for_good_performance(self, module_name: str,
                                   current_params: Dict[str, Any],
                                   final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """为性能好的情况调整参数空间"""
        adjustments = {}
        
        # 微调参数空间
        for param_name, param_value in current_params.items():
            if isinstance(param_value, (int, float)):
                adjustments[param_name] = {
                    'action': 'refine_bounds',
                    'factor': 0.9,
                    'reason': 'good_performance_refinement'
                }
        
        return {
            'module_name': module_name,
            'adjustments': adjustments,
            'reason': 'good_performance',
            'timestamp': time.time()
        }
    
    def _get_parameter_sensitivity(self, param_name: str) -> float:
        """获取参数敏感性"""
        if param_name in self.parameter_sensitivity_cache:
            return self.parameter_sensitivity_cache[param_name]
        
        # 基于参数名称推断敏感性
        high_sensitivity_keywords = ['rate', 'probability', 'effect', 'influence']
        medium_sensitivity_keywords = ['weight', 'factor', 'multiplier']
        low_sensitivity_keywords = ['offset', 'bias', 'noise']
        
        param_lower = param_name.lower()
        
        if any(keyword in param_lower for keyword in high_sensitivity_keywords):
            sensitivity = 0.8
        elif any(keyword in param_lower for keyword in medium_sensitivity_keywords):
            sensitivity = 0.5
        elif any(keyword in param_lower for keyword in low_sensitivity_keywords):
            sensitivity = 0.2
        else:
            sensitivity = 0.5
        
        self.parameter_sensitivity_cache[param_name] = sensitivity
        return sensitivity

class MultiModuleJointCalibrator:
    """多模块联合校准器"""
    
    def __init__(self):
        self.joint_calibration_history = []
        self.module_interactions = {}
        self.parameter_correlations = {}
    
    def analyze_module_interactions(self, modules: List[str], 
                                  parameter_spaces: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """分析模块交互"""
        interactions = {
            'direct_interactions': {},
            'parameter_correlations': {},
            'data_flow_dependencies': {}
        }
        
        # 分析直接交互
        for i, module1 in enumerate(modules):
            for j, module2 in enumerate(modules):
                if i != j:
                    interaction_strength = self._calculate_interaction_strength(
                        module1, module2, parameter_spaces
                    )
                    if interaction_strength > 0.3:
                        interactions['direct_interactions'][f"{module1}->{module2}"] = interaction_strength
        
        # 分析参数相关性
        for module_name, param_space in parameter_spaces.items():
            param_names = param_space.get('parameter_names', [])
            if len(param_names) > 1:
                correlations = self._calculate_parameter_correlations(param_names)
                interactions['parameter_correlations'][module_name] = correlations
        
        return interactions
    
    def _calculate_interaction_strength(self, module1: str, module2: str, 
                                     parameter_spaces: Dict[str, Dict[str, Any]]) -> float:
        """计算模块交互强度"""
        # 基于模块名称和参数空间计算交互强度
        interaction_keywords = [
            ('information', 'policy'),
            ('policy', 'social'),
            ('information', 'social'),
            ('network', 'information'),
            ('network', 'social')
        ]
        
        module1_lower = module1.lower()
        module2_lower = module2.lower()
        
        for keyword1, keyword2 in interaction_keywords:
            if keyword1 in module1_lower and keyword2 in module2_lower:
                return 0.7
            elif keyword2 in module1_lower and keyword1 in module2_lower:
                return 0.7
        
        return 0.1  # 默认低交互强度
    
    def _calculate_parameter_correlations(self, param_names: List[str]) -> Dict[str, float]:
        """计算参数相关性"""
        correlations = {}
        
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                if i != j:
                    # 基于参数名称计算相关性
                    correlation = self._estimate_parameter_correlation(param1, param2)
                    correlations[f"{param1}->{param2}"] = correlation
        
        return correlations
    
    def _estimate_parameter_correlation(self, param1: str, param2: str) -> float:
        """估计参数相关性"""
        # 基于参数名称的相似性估计相关性
        param1_lower = param1.lower()
        param2_lower = param2.lower()
        
        # 检查是否有共同的关键词
        common_keywords = ['rate', 'probability', 'effect', 'influence', 'weight', 'factor']
        
        for keyword in common_keywords:
            if keyword in param1_lower and keyword in param2_lower:
                return 0.6
        
        # 检查是否有相反的关键词
        opposite_keywords = [('adoption', 'abandon'), ('increase', 'decrease'), ('positive', 'negative')]
        
        for pos, neg in opposite_keywords:
            if (pos in param1_lower and neg in param2_lower) or (neg in param1_lower and pos in param2_lower):
                return -0.4
        
        return 0.1  # 默认低相关性
    
    def design_joint_calibration_strategy(self, modules: List[str],
                                        interactions: Dict[str, Any]) -> Dict[str, Any]:
        """设计联合校准策略"""
        strategy = {
            'calibration_phases': [],
            'parameter_constraints': {},
            'interaction_handling': {}
        }
        
        # 基于交互强度设计校准阶段
        high_interaction_pairs = [
            pair for pair, strength in interactions['direct_interactions'].items()
            if strength > 0.5
        ]
        
        if high_interaction_pairs:
            # 高交互模块需要联合校准
            strategy['calibration_phases'].append({
                'phase_type': 'joint',
                'modules': [pair.split('->')[0] for pair in high_interaction_pairs],
                'description': '联合校准高交互模块'
            })
        
        # 独立校准低交互模块
        low_interaction_modules = [
            module for module in modules
            if not any(module in pair for pair in high_interaction_pairs)
        ]
        
        for module in low_interaction_modules:
            strategy['calibration_phases'].append({
                'phase_type': 'independent',
                'modules': [module],
                'description': f'独立校准模块 {module}'
            })
        
        return strategy
    
    def handle_parameter_cancellation(self, module_name: str,
                                    parameter_values: Dict[str, Any],
                                    calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """处理参数对消问题"""
        cancellation_analysis = {
            'module_name': module_name,
            'cancellation_detected': False,
            'problematic_parameters': [],
            'suggestions': []
        }
        
        # 检测参数对消
        if self._detect_parameter_cancellation(parameter_values, calibration_results):
            cancellation_analysis['cancellation_detected'] = True
            cancellation_analysis['problematic_parameters'] = self._identify_problematic_parameters(parameter_values)
            cancellation_analysis['suggestions'] = self._generate_cancellation_suggestions(module_name)
        
        return cancellation_analysis
    
    def _detect_parameter_cancellation(self, parameter_values: Dict[str, Any],
                                     calibration_results: Dict[str, Any]) -> bool:
        """检测参数对消"""
        # 基于参数值的分布检测对消
        param_values = list(parameter_values.values())
        
        if len(param_values) < 2:
            return False
        
        # 检查参数值是否集中在边界
        boundary_ratio = sum(1 for v in param_values if v <= 0.1 or v >= 0.9) / len(param_values)
        
        # 检查参数值是否高度相关
        if len(param_values) >= 3:
            correlation_matrix = np.corrcoef(param_values)
            high_correlation_count = np.sum(np.abs(correlation_matrix) > 0.8) - len(param_values)
            
            return boundary_ratio > 0.5 or high_correlation_count > len(param_values) / 2
        
        return boundary_ratio > 0.7
    
    def _identify_problematic_parameters(self, parameter_values: Dict[str, Any]) -> List[str]:
        """识别有问题的参数"""
        problematic = []
        
        for param_name, param_value in parameter_values.items():
            if isinstance(param_value, (int, float)):
                if param_value <= 0.1 or param_value >= 0.9:
                    problematic.append(param_name)
        
        return problematic
    
    def _generate_cancellation_suggestions(self, module_name: str) -> List[str]:
        """生成对消建议"""
        suggestions = [
            f"模块 {module_name} 检测到参数对消，建议调整参数约束",
            "考虑使用参数变换（如log变换）来改善参数分布",
            "增加参数独立性约束，避免参数间强相关",
            "考虑分阶段校准，先固定部分参数再校准其他参数"
        ]
        
        return suggestions

class IntelligentSBIStrategy:
    """智能SBI策略管理器"""
    
    def __init__(self):
        self.strategy_selector = SBIStrategySelector()
        self.parameter_adjuster = AdaptiveParameterSpaceAdjuster()
        self.joint_calibrator = MultiModuleJointCalibrator()
        self.calibration_history = []
    
    def design_calibration_strategy(self, modules: List[str],
                                  parameter_spaces: Dict[str, Dict[str, Any]],
                                  data_complexity: str = 'medium') -> Dict[str, Any]:
        """设计校准策略"""
        strategy = {
            'modules': modules,
            'sbi_methods': {},
            'parameter_adjustments': {},
            'joint_calibration': {},
            'calibration_phases': []
        }
        
        # 为每个模块选择SBI方法
        for module_name in modules:
            param_count = parameter_spaces.get(module_name, {}).get('parameter_count', 0)
            sbi_method = self.strategy_selector.select_sbi_method(module_name, param_count, data_complexity)
            sbi_params = self.strategy_selector.get_sbi_parameters(sbi_method, module_name, param_count)
            
            strategy['sbi_methods'][module_name] = {
                'method': sbi_method.value,
                'parameters': sbi_params
            }
        
        # 分析模块交互
        interactions = self.joint_calibrator.analyze_module_interactions(modules, parameter_spaces)
        strategy['joint_calibration'] = interactions
        
        # 设计联合校准策略
        joint_strategy = self.joint_calibrator.design_joint_calibration_strategy(modules, interactions)
        strategy['calibration_phases'] = joint_strategy['calibration_phases']
        
        return strategy
    
    def adjust_parameter_space(self, module_name: str,
                             current_params: Dict[str, Any],
                             calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """调整参数空间"""
        return self.parameter_adjuster.adjust_parameter_space(
            module_name, current_params, calibration_results
        )
    
    def handle_parameter_cancellation(self, module_name: str,
                                    parameter_values: Dict[str, Any],
                                    calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """处理参数对消"""
        return self.joint_calibrator.handle_parameter_cancellation(
            module_name, parameter_values, calibration_results
        )
    
    def get_calibration_recommendations(self, module_name: str,
                                      calibration_results: Dict[str, Any]) -> List[str]:
        """获取校准建议"""
        recommendations = []
        
        if not calibration_results.get('convergence_achieved', False):
            recommendations.append(f"模块 {module_name} 未收敛，建议增加迭代次数或调整参数空间")
        
        final_metrics = calibration_results.get('final_metrics', {})
        if final_metrics.get('rmse', 1.0) > 0.5:
            recommendations.append(f"模块 {module_name} RMSE较高，建议检查数据质量或参数约束")
        
        if final_metrics.get('r2', 0.0) < 0.6:
            recommendations.append(f"模块 {module_name} R²较低，建议优化摘要统计设计")
        
        return recommendations
    
    def update_calibration_history(self, module_name: str,
                                 calibration_results: Dict[str, Any]) -> None:
        """更新校准历史"""
        self.calibration_history.append({
            'module_name': module_name,
            'timestamp': time.time(),
            'results': calibration_results
        })
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        return {
            'total_calibrations': len(self.calibration_history),
            'successful_calibrations': sum(
                1 for record in self.calibration_history
                if record['results'].get('convergence_achieved', False)
            ),
            'average_rmse': np.mean([
                record['results'].get('final_metrics', {}).get('rmse', 0)
                for record in self.calibration_history
            ]) if self.calibration_history else 0,
            'parameter_adjustments': len(self.parameter_adjuster.adjustment_history)
        }





