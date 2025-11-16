"""
增强的数据加载器
专门针对SOCIA项目设计，支持模块依赖关系分析和智能摘要统计设计
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class SOCIAModuleDependencyAnalyzer:
    """SOCIA模块依赖关系分析器"""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.calibration_modules = []
        self.module_info = {}
        self.calibration_order = []
    
    def load_model_plan(self, model_plan_path: Union[str, Path]) -> None:
        """加载模型计划并分析依赖关系"""
        try:
            with open(model_plan_path, 'r', encoding='utf-8') as f:
                model_plan = json.load(f)
            
            self._parse_modules(model_plan)
            self._build_dependency_graph()
            self._identify_calibration_modules()
            self._compute_calibration_order()
            
            logger.info(f"模块依赖关系分析完成: {len(self.calibration_modules)} 个可校准模块")
        except Exception as e:
            logger.error(f"模型计划加载失败: {e}")
            raise
    
    def _parse_modules(self, model_plan: Dict[str, Any]) -> None:
        """解析模块信息"""
        if 'modules' not in model_plan:
            logger.warning("模型计划中没有找到模块定义")
            return
        
        for module_name, module_info in model_plan['modules'].items():
            self.module_info[module_name] = module_info
            logger.debug(f"解析模块: {module_name}")
    
    def _build_dependency_graph(self) -> None:
        """构建依赖关系图"""
        self.dependency_graph.clear()
        
        for module_name, module_info in self.module_info.items():
            # 添加模块节点
            self.dependency_graph.add_node(module_name, **module_info)
            
            # 添加依赖边
            if 'dependencies' in module_info:
                for dep in module_info['dependencies']:
                    self.dependency_graph.add_edge(dep, module_name)
                    logger.debug(f"添加依赖: {dep} -> {module_name}")
            
            # 添加输入输出关系
            if 'inputs' in module_info:
                for input_signal in module_info['inputs']:
                    # 查找提供该信号的模块
                    for other_module, other_info in self.module_info.items():
                        if 'outputs' in other_info and input_signal in other_info['outputs']:
                            self.dependency_graph.add_edge(other_module, module_name)
                            logger.debug(f"添加信号依赖: {other_module} -> {module_name} (信号: {input_signal})")
    
    def _identify_calibration_modules(self) -> None:
        """识别可校准模块"""
        self.calibration_modules = []
        
        for module_name, module_info in self.module_info.items():
            if module_info.get('sbi_calibration', False):
                self.calibration_modules.append(module_name)
                logger.info(f"识别可校准模块: {module_name}")
    
    def _compute_calibration_order(self) -> None:
        """计算校准顺序（拓扑排序）"""
        try:
            # 只考虑可校准模块的子图
            calibration_subgraph = self.dependency_graph.subgraph(self.calibration_modules)
            self.calibration_order = list(nx.topological_sort(calibration_subgraph))
            
            logger.info(f"校准顺序: {self.calibration_order}")
        except nx.NetworkXError as e:
            logger.error(f"拓扑排序失败: {e}")
            self.calibration_order = self.calibration_modules.copy()
    
    def get_calibration_order(self) -> List[str]:
        """获取校准顺序"""
        return self.calibration_order.copy()
    
    def get_module_dependencies(self, module_name: str) -> List[str]:
        """获取模块的直接依赖"""
        if module_name not in self.dependency_graph:
            return []
        return list(self.dependency_graph.predecessors(module_name))
    
    def get_module_dependents(self, module_name: str) -> List[str]:
        """获取模块的直接依赖者"""
        if module_name not in self.dependency_graph:
            return []
        return list(self.dependency_graph.successors(module_name))
    
    def get_all_dependencies(self, module_name: str) -> List[str]:
        """获取模块的所有依赖（递归）"""
        all_deps = set()
        to_process = [module_name]
        
        while to_process:
            current = to_process.pop(0)
            direct_deps = self.get_module_dependencies(current)
            for dep in direct_deps:
                if dep not in all_deps:
                    all_deps.add(dep)
                    to_process.append(dep)
        
        return list(all_deps)
    
    def get_calibration_strategy(self) -> Dict[str, Any]:
        """获取校准策略"""
        strategy = {
            'calibration_order': self.calibration_order,
            'calibration_modules': self.calibration_modules,
            'module_dependencies': {},
            'module_dependents': {},
            'calibration_phases': []
        }
        
        # 为每个可校准模块分析依赖关系
        for module_name in self.calibration_modules:
            strategy['module_dependencies'][module_name] = self.get_module_dependencies(module_name)
            strategy['module_dependents'][module_name] = self.get_module_dependents(module_name)
        
        # 生成校准阶段
        self._generate_calibration_phases(strategy)
        
        return strategy
    
    def _generate_calibration_phases(self, strategy: Dict[str, Any]) -> None:
        """生成校准阶段"""
        phases = []
        processed_modules = set()
        
        for module_name in self.calibration_order:
            if module_name in processed_modules:
                continue
            
            # 获取该模块的所有依赖
            all_deps = self.get_all_dependencies(module_name)
            
            # 创建阶段
            phase = {
                'phase_id': len(phases),
                'modules': [module_name],
                'dependencies': all_deps,
                'description': f"校准模块 {module_name} 及其依赖"
            }
            
            phases.append(phase)
            processed_modules.add(module_name)
        
        strategy['calibration_phases'] = phases
        logger.info(f"生成校准阶段: {len(phases)} 个阶段")

class IntelligentSummaryStatisticsDesigner:
    """智能摘要统计设计器"""
    
    def __init__(self):
        self.data_patterns = {}
        self.summary_statistics = {}
        self.morphological_features = {}
    
    def load_data_analysis(self, analysis_path: Union[str, Path]) -> None:
        """加载数据分析结果"""
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                data_analysis = json.load(f)
            
            self._extract_data_patterns(data_analysis)
            self._design_summary_statistics()
            
            logger.info("数据分析结果加载完成")
        except Exception as e:
            logger.error(f"数据分析结果加载失败: {e}")
            raise
    
    def _extract_data_patterns(self, data_analysis: Dict[str, Any]) -> None:
        """提取数据模式"""
        if 'data_summary' in data_analysis:
            self.data_patterns = data_analysis['data_summary']
        else:
            logger.warning("数据分析结果中没有找到数据模式")
    
    def _design_summary_statistics(self) -> None:
        """设计摘要统计"""
        # 基础统计指标
        self.summary_statistics = {
            'basic_metrics': ['rmse', 'mae', 'r2', 'correlation'],
            'temporal_metrics': ['peak_time', 'peak_value', 'slope', 'trend'],
            'morphological_metrics': ['inflection_point', 'plateau_level', 'sigmoid_fit'],
            'group_metrics': ['group_variance', 'group_correlation', 'subgroup_effects']
        }
        
        # 基于数据模式设计形态特征
        self._design_morphological_features()
        
        logger.info("摘要统计设计完成")
    
    def _design_morphological_features(self) -> None:
        """设计形态特征"""
        self.morphological_features = {
            'sigmoid_features': {
                'inflection_point': self._find_inflection_point,
                'sigmoid_parameters': self._fit_sigmoid,
                'sigmoid_goodness_of_fit': self._sigmoid_r2
            },
            'temporal_features': {
                'time_to_50_percent': self._time_to_threshold,
                'time_to_peak': self._time_to_peak,
                'early_slope': self._early_slope,
                'late_slope': self._late_slope
            },
            'shape_features': {
                'peak_value': self._peak_value,
                'plateau_level': self._plateau_level,
                'asymmetry': self._asymmetry_measure
            }
        }
    
    def get_summary_statistics_for_module(self, module_name: str) -> Dict[str, Any]:
        """获取模块的摘要统计设计"""
        module_stats = {
            'basic_metrics': self.summary_statistics['basic_metrics'],
            'temporal_metrics': self.summary_statistics['temporal_metrics'],
            'morphological_metrics': self.summary_statistics['morphological_metrics'],
            'group_metrics': self.summary_statistics['group_metrics'],
            'custom_metrics': self._get_custom_metrics_for_module(module_name)
        }
        
        return module_stats
    
    def _get_custom_metrics_for_module(self, module_name: str) -> List[str]:
        """获取模块的自定义指标"""
        custom_metrics = []
        
        # 基于模块类型添加特定指标
        if 'InformationDiffusion' in module_name:
            custom_metrics.extend(['info_cascade_speed', 'info_reach_rate', 'info_persistence'])
        elif 'PolicyAndMessaging' in module_name:
            custom_metrics.extend(['policy_effectiveness', 'messaging_penetration', 'compliance_rate'])
        elif 'SocialInfluenceAdoption' in module_name:
            custom_metrics.extend(['adoption_cascade_speed', 'peer_effect_strength', 'threshold_sensitivity'])
        
        return custom_metrics
    
    # 形态特征计算方法
    def _find_inflection_point(self, data: np.ndarray) -> float:
        """找到拐点"""
        if len(data) < 3:
            return 0.0
        
        # 计算二阶导数
        second_derivative = np.diff(data, n=2)
        inflection_idx = np.argmax(np.abs(second_derivative))
        return float(inflection_idx)
    
    def _fit_sigmoid(self, data: np.ndarray) -> Dict[str, float]:
        """拟合S型曲线"""
        try:
            from scipy.optimize import curve_fit
            
            def sigmoid(x, a, b, c, d):
                return a / (1 + np.exp(-b * (x - c))) + d
            
            x = np.arange(len(data))
            popt, _ = curve_fit(sigmoid, x, data, maxfev=1000)
            
            return {
                'a': float(popt[0]),
                'b': float(popt[1]),
                'c': float(popt[2]),
                'd': float(popt[3])
            }
        except Exception:
            return {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}
    
    def _sigmoid_r2(self, data: np.ndarray) -> float:
        """计算S型曲线拟合的R²"""
        try:
            sigmoid_params = self._fit_sigmoid(data)
            x = np.arange(len(data))
            
            def sigmoid(x, a, b, c, d):
                return a / (1 + np.exp(-b * (x - c))) + d
            
            predicted = sigmoid(x, **sigmoid_params)
            ss_res = np.sum((data - predicted) ** 2)
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        except Exception:
            return 0.0
    
    def _time_to_threshold(self, data: np.ndarray, threshold: float = 0.5) -> float:
        """计算达到阈值的时间"""
        for i, value in enumerate(data):
            if value >= threshold:
                return float(i)
        return float(len(data) - 1)
    
    def _time_to_peak(self, data: np.ndarray) -> float:
        """计算达到峰值的时间"""
        return float(np.argmax(data))
    
    def _early_slope(self, data: np.ndarray, early_ratio: float = 0.3) -> float:
        """计算早期斜率"""
        early_end = int(len(data) * early_ratio)
        if early_end < 2:
            return 0.0
        return float((data[early_end] - data[0]) / early_end)
    
    def _late_slope(self, data: np.ndarray, late_ratio: float = 0.3) -> float:
        """计算后期斜率"""
        late_start = int(len(data) * (1 - late_ratio))
        if late_start >= len(data) - 1:
            return 0.0
        return float((data[-1] - data[late_start]) / (len(data) - 1 - late_start))
    
    def _peak_value(self, data: np.ndarray) -> float:
        """计算峰值"""
        return float(np.max(data))
    
    def _plateau_level(self, data: np.ndarray, plateau_ratio: float = 0.2) -> float:
        """计算平台水平"""
        plateau_start = int(len(data) * (1 - plateau_ratio))
        return float(np.mean(data[plateau_start:]))
    
    def _asymmetry_measure(self, data: np.ndarray) -> float:
        """计算不对称性"""
        peak_idx = np.argmax(data)
        if peak_idx == 0 or peak_idx == len(data) - 1:
            return 0.0
        
        left_side = data[:peak_idx]
        right_side = data[peak_idx:]
        
        if len(left_side) == 0 or len(right_side) == 0:
            return 0.0
        
        left_slope = np.mean(np.diff(left_side)) if len(left_side) > 1 else 0.0
        right_slope = np.mean(np.diff(right_side)) if len(right_side) > 1 else 0.0
        
        return float(left_slope - right_slope)

class AdvancedParameterSpaceBuilder:
    """高级参数空间构建器"""
    
    def __init__(self):
        self.parameter_definitions = {}
        self.parameter_constraints = {}
        self.parameter_sensitivity = {}
        self.parameter_correlations = {}
    
    def load_parameter_definitions(self, param_defs_path: Union[str, Path]) -> None:
        """加载参数定义"""
        try:
            with open(param_defs_path, 'r', encoding='utf-8') as f:
                self.parameter_definitions = json.load(f)
            
            self._analyze_parameter_constraints()
            self._compute_parameter_sensitivity()
            self._analyze_parameter_correlations()
            
            logger.info(f"参数定义加载完成: {len(self.parameter_definitions)} 个参数")
        except Exception as e:
            logger.error(f"参数定义加载失败: {e}")
            raise
    
    def _analyze_parameter_constraints(self) -> None:
        """分析参数约束"""
        for param_name, param_info in self.parameter_definitions.items():
            constraints = {
                'bounds': param_info.get('bounds'),
                'type': param_info.get('type', 'continuous'),
                'prior': param_info.get('prior', 'uniform'),
                'transform': param_info.get('transform', 'none')
            }
            
            # 添加模块归属
            if 'module' in param_info:
                constraints['module'] = param_info['module']
            
            self.parameter_constraints[param_name] = constraints
    
    def _compute_parameter_sensitivity(self) -> None:
        """计算参数敏感性"""
        # 这里可以实现基于历史数据的参数敏感性分析
        # 暂时使用默认值
        for param_name in self.parameter_definitions.keys():
            self.parameter_sensitivity[param_name] = {
                'sensitivity_score': 1.0,
                'importance_rank': 1,
                'interaction_strength': 0.0
            }
    
    def _analyze_parameter_correlations(self) -> None:
        """分析参数相关性"""
        # 这里可以实现基于历史数据的参数相关性分析
        # 暂时使用默认值
        for param_name in self.parameter_definitions.keys():
            self.parameter_correlations[param_name] = {}
    
    def build_parameter_space(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """构建参数空间"""
        if module_name:
            # 构建特定模块的参数空间
            module_params = {
                name: info for name, info in self.parameter_definitions.items()
                if info.get('module') == module_name
            }
        else:
            module_params = self.parameter_definitions
        
        parameter_space = {
            'parameters': module_params,
            'constraints': {name: self.parameter_constraints[name] for name in module_params.keys()},
            'sensitivity': {name: self.parameter_sensitivity[name] for name in module_params.keys()},
            'correlations': {name: self.parameter_correlations[name] for name in module_params.keys()}
        }
        
        return parameter_space
    
    def get_parameter_transforms(self) -> Dict[str, callable]:
        """获取参数变换函数"""
        transforms = {
            'log': lambda x: np.log(x),
            'logit': lambda x: np.log(x / (1 - x)),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'none': lambda x: x
        }
        return transforms
    
    def apply_parameter_transforms(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """应用参数变换"""
        transformed_params = {}
        transforms = self.get_parameter_transforms()
        
        for param_name, value in params.items():
            if param_name in self.parameter_constraints:
                transform_type = self.parameter_constraints[param_name].get('transform', 'none')
                if transform_type in transforms:
                    transformed_params[param_name] = transforms[transform_type](value)
                else:
                    transformed_params[param_name] = value
            else:
                transformed_params[param_name] = value
        
        return transformed_params
    
    def reduce_parameter_dimensions(self, params: Dict[str, Any], 
                                   target_dimensions: int = 10) -> Dict[str, Any]:
        """降维参数空间"""
        if len(params) <= target_dimensions:
            return params
        
        # 基于敏感性排序选择最重要的参数
        param_importance = []
        for param_name in params.keys():
            if param_name in self.parameter_sensitivity:
                importance = self.parameter_sensitivity[param_name]['sensitivity_score']
                param_importance.append((param_name, importance))
        
        # 按重要性排序
        param_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最重要的参数
        selected_params = dict(param_importance[:target_dimensions])
        
        logger.info(f"参数空间降维: {len(params)} -> {len(selected_params)}")
        return selected_params

class SOCIADataProcessor:
    """SOCIA数据处理器"""
    
    def __init__(self):
        self.target_data = None
        self.daily_rates = None
        self.data_quality_report = {}
    
    def load_and_preprocess_data(self, data_path: Union[str, Path]) -> None:
        """加载和预处理数据"""
        try:
            # 加载原始数据
            raw_data = pd.read_csv(data_path)
            logger.info(f"原始数据加载: {raw_data.shape}")
            
            # 数据预处理
            self.target_data = self._preprocess_data(raw_data)
            
            # 计算日率
            self.daily_rates = self._calculate_daily_rates()
            
            # 数据质量验证
            self.data_quality_report = self._validate_data_quality()
            
            logger.info(f"数据预处理完成: {self.target_data.shape}")
        except Exception as e:
            logger.error(f"数据加载和预处理失败: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        processed_data = data.copy()
        
        # 确保时间列存在
        if 'day' not in processed_data.columns:
            processed_data['day'] = processed_data.index
        
        # 转换数据类型
        processed_data['day'] = pd.to_numeric(processed_data['day'], errors='coerce')
        processed_data['wearing_mask'] = pd.to_numeric(processed_data['wearing_mask'], errors='coerce')
        processed_data['received_info'] = pd.to_numeric(processed_data['received_info'], errors='coerce')
        
        # 筛选训练窗口 (day 0-29)
        processed_data = processed_data[
            (processed_data['day'] >= 0) & 
            (processed_data['day'] <= 29)
        ].copy()
        
        # 处理缺失值
        processed_data = processed_data.dropna(subset=['day', 'wearing_mask', 'received_info'])
        
        return processed_data
    
    def _calculate_daily_rates(self) -> pd.DataFrame:
        """计算日率"""
        if self.target_data is None:
            return None
        
        daily_rates = self.target_data.groupby('day').agg({
            'wearing_mask': ['mean', 'count'],
            'received_info': ['mean', 'count']
        }).reset_index()
        
        # 展平列名
        daily_rates.columns = [
            'day', 'adoption_rate', 'adoption_count',
            'info_rate', 'info_count'
        ]
        
        return daily_rates
    
    def _validate_data_quality(self) -> Dict[str, Any]:
        """验证数据质量"""
        if self.target_data is None:
            return {}
        
        quality_report = {
            'total_rows': len(self.target_data),
            'missing_values': self.target_data.isnull().sum().to_dict(),
            'data_types': self.target_data.dtypes.to_dict(),
            'value_ranges': {},
            'quality_score': 0.0
        }
        
        # 计算值范围
        for col in self.target_data.select_dtypes(include=[np.number]).columns:
            quality_report['value_ranges'][col] = {
                'min': self.target_data[col].min(),
                'max': self.target_data[col].max(),
                'mean': self.target_data[col].mean(),
                'std': self.target_data[col].std()
            }
        
        # 计算质量分数
        missing_ratio = self.target_data.isnull().sum().sum() / (len(self.target_data) * len(self.target_data.columns))
        quality_score = max(0, 1 - missing_ratio)
        quality_report['quality_score'] = quality_score
        
        return quality_report
    
    def get_target_signals(self) -> Dict[str, np.ndarray]:
        """获取目标信号"""
        if self.daily_rates is None:
            return {}
        
        return {
            'adoption_rate': self.daily_rates['adoption_rate'].values,
            'info_rate': self.daily_rates['info_rate'].values
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        return {
            'target_data_shape': self.target_data.shape if self.target_data is not None else None,
            'daily_rates_shape': self.daily_rates.shape if self.daily_rates is not None else None,
            'data_quality': self.data_quality_report
        }
