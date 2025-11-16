"""
SOCIA集成工具模块
提供SOCIA模块解析、依赖关系分析、配置加载等功能
"""

import json
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class SOCIAModuleParser:
    """SOCIA模块解析器"""
    
    def __init__(self):
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.dependencies: nx.DiGraph = nx.DiGraph()
        self.calibration_modules: List[str] = []
    
    def load_model_plan(self, model_plan_path: Union[str, Path]) -> None:
        """加载模型计划"""
        try:
            with open(model_plan_path, 'r', encoding='utf-8') as f:
                model_plan = json.load(f)
            
            self._parse_modules(model_plan)
            self._build_dependency_graph()
            self._identify_calibration_modules()
            
            logger.info(f"模型计划加载成功: {len(self.modules)} 个模块")
        except Exception as e:
            logger.error(f"模型计划加载失败: {e}")
            raise
    
    def _parse_modules(self, model_plan: Dict[str, Any]) -> None:
        """解析模块"""
        # 检查是否有modules字段
        if 'modules' in model_plan:
            modules_data = model_plan['modules']
            if isinstance(modules_data, list):
                # modules是列表格式
                for module_info in modules_data:
                    if isinstance(module_info, dict) and 'name' in module_info:
                        module_name = module_info['name']
                        self.modules[module_name] = module_info
                        logger.debug(f"解析模块: {module_name}")
            elif isinstance(modules_data, dict):
                # modules是字典格式
                for module_name, module_info in modules_data.items():
                    self.modules[module_name] = module_info
                    logger.debug(f"解析模块: {module_name}")
        # 检查是否有processes字段（SOCIA格式）
        elif 'processes' in model_plan:
            for process in model_plan['processes']:
                if isinstance(process, dict) and 'name' in process:
                    module_name = process['name']
                    self.modules[module_name] = process
                    logger.debug(f"解析进程模块: {module_name}")
        else:
            logger.warning("模型计划中没有找到模块定义")
            return
    
    def _build_dependency_graph(self) -> None:
        """构建依赖关系图"""
        self.dependencies.clear()
        
        for module_name, module_info in self.modules.items():
            # 添加模块节点
            if isinstance(module_info, dict):
                self.dependencies.add_node(module_name, **module_info)
                
                # 添加依赖边
                if 'dependencies' in module_info:
                    for dep in module_info['dependencies']:
                        self.dependencies.add_edge(dep, module_name)
                        logger.debug(f"添加依赖: {dep} -> {module_name}")
            else:
                # 如果不是字典，只添加节点名
                self.dependencies.add_node(module_name)
                logger.debug(f"添加模块节点: {module_name}")
        
        logger.info(f"依赖关系图构建完成: {len(self.dependencies.nodes)} 个节点, {len(self.dependencies.edges)} 条边")
    
    def _identify_calibration_modules(self) -> None:
        """识别可校准模块"""
        self.calibration_modules = []
        
        for module_name, module_info in self.modules.items():
            if module_info.get('sbi_calibration', False):
                self.calibration_modules.append(module_name)
                logger.info(f"识别可校准模块: {module_name}")
    
    def get_calibration_order(self) -> List[str]:
        """获取校准顺序（拓扑排序）"""
        try:
            # 只考虑可校准模块的子图
            calibration_subgraph = self.dependencies.subgraph(self.calibration_modules)
            calibration_order = list(nx.topological_sort(calibration_subgraph))
            
            logger.info(f"校准顺序: {calibration_order}")
            return calibration_order
        except nx.NetworkXError as e:
            logger.error(f"拓扑排序失败: {e}")
            return self.calibration_modules.copy()
    
    def get_module_dependencies(self, module_name: str) -> List[str]:
        """获取模块依赖"""
        if module_name not in self.dependencies:
            return []
        
        return list(self.dependencies.predecessors(module_name))
    
    def get_module_dependents(self, module_name: str) -> List[str]:
        """获取模块依赖者"""
        if module_name not in self.dependencies:
            return []
        
        return list(self.dependencies.successors(module_name))
    
    def is_calibration_module(self, module_name: str) -> bool:
        """检查是否为可校准模块"""
        return module_name in self.calibration_modules
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """获取模块信息"""
        return self.modules.get(module_name, {})
    
    def get_calibration_parameters(self, module_name: str) -> List[str]:
        """获取模块的可校准参数"""
        module_info = self.get_module_info(module_name)
        return module_info.get('calibration_parameters', [])
    
    def get_calibration_order(self) -> List[str]:
        """获取校准顺序（拓扑排序）"""
        try:
            # 只考虑可校准模块的子图
            calibration_subgraph = self.dependencies.subgraph(self.calibration_modules)
            calibration_order = list(nx.topological_sort(calibration_subgraph))
            logger.info(f"校准顺序: {calibration_order}")
            return calibration_order
        except nx.NetworkXError as e:
            logger.error(f"拓扑排序失败: {e}")
            return self.calibration_modules.copy()
    
    def get_target_signals(self, module_name: str) -> List[str]:
        """获取模块的目标信号"""
        module_info = self.get_module_info(module_name)
        return module_info.get('sbi_target_signals', [])

class SOCIADataAnalyzer:
    """SOCIA数据分析器"""
    
    def __init__(self):
        self.data_analysis: Dict[str, Any] = {}
        self.data_patterns: Dict[str, Any] = {}
    
    def load_data_analysis(self, analysis_path: Union[str, Path]) -> None:
        """加载数据分析结果"""
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                self.data_analysis = json.load(f)
            
            self._extract_data_patterns()
            logger.info("数据分析结果加载成功")
        except Exception as e:
            logger.error(f"数据分析结果加载失败: {e}")
            raise
    
    def _extract_data_patterns(self) -> None:
        """提取数据模式"""
        if 'data_patterns' in self.data_analysis:
            self.data_patterns = self.data_analysis['data_patterns']
        else:
            logger.warning("数据分析结果中没有找到数据模式")
    
    def get_summary_statistics_design(self, module_name: str) -> Dict[str, Any]:
        """获取摘要统计设计"""
        module_patterns = self.data_patterns.get(module_name, {})
        
        design = {
            'basic_metrics': ['rmse', 'mae', 'r2'],
            'shape_features': [],
            'temporal_features': [],
            'group_features': []
        }
        
        # 基于数据模式设计摘要统计
        if 'temporal_patterns' in module_patterns:
            design['temporal_features'] = ['peak_time', 'slope', 'trend']
        
        if 'shape_patterns' in module_patterns:
            design['shape_features'] = ['peak_value', 'inflection_point', 'plateau']
        
        if 'group_patterns' in module_patterns:
            design['group_features'] = ['group_variance', 'group_correlation']
        
        logger.info(f"摘要统计设计: {module_name} -> {design}")
        return design
    
    def get_parameter_constraints(self, module_name: str) -> Dict[str, Any]:
        """获取参数约束"""
        module_patterns = self.data_patterns.get(module_name, {})
        
        constraints = {
            'bounds': {},
            'dependencies': {},
            'regularization': {}
        }
        
        # 基于数据模式设置参数约束
        if 'parameter_sensitivity' in module_patterns:
            constraints['bounds'] = module_patterns['parameter_sensitivity']
        
        if 'parameter_correlations' in module_patterns:
            constraints['dependencies'] = module_patterns['parameter_correlations']
        
        logger.info(f"参数约束: {module_name} -> {constraints}")
        return constraints

class SOCIAVerificationAnalyzer:
    """SOCIA验证分析器"""
    
    def __init__(self):
        self.verification_results: Dict[str, Any] = {}
        self.issues: List[Dict[str, Any]] = []
        self.constraints: Dict[str, Any] = {}
    
    def load_verification_results(self, verification_path: Union[str, Path]) -> None:
        """加载验证结果"""
        try:
            with open(verification_path, 'r', encoding='utf-8') as f:
                self.verification_results = json.load(f)
            
            self._extract_issues()
            self._generate_constraints()
            logger.info("验证结果加载成功")
        except Exception as e:
            logger.error(f"验证结果加载失败: {e}")
            raise
    
    def _extract_issues(self) -> None:
        """提取问题"""
        if 'issues' in self.verification_results:
            self.issues = self.verification_results['issues']
        else:
            logger.warning("验证结果中没有找到问题列表")
    
    def _generate_constraints(self) -> None:
        """生成约束"""
        self.constraints = {
            'parameter_bounds': {},
            'convergence_requirements': {},
            'quality_thresholds': {}
        }
        
        # 基于问题生成约束
        for issue in self.issues:
            if issue.get('type') == 'parameter_bound':
                param_name = issue.get('parameter')
                bounds = issue.get('bounds')
                if param_name and bounds:
                    self.constraints['parameter_bounds'][param_name] = bounds
            
            elif issue.get('type') == 'convergence':
                module_name = issue.get('module')
                requirements = issue.get('requirements')
                if module_name and requirements:
                    self.constraints['convergence_requirements'][module_name] = requirements
            
            elif issue.get('type') == 'quality':
                threshold = issue.get('threshold')
                metric = issue.get('metric')
                if threshold and metric:
                    self.constraints['quality_thresholds'][metric] = threshold
        
        logger.info(f"约束生成完成: {len(self.constraints['parameter_bounds'])} 个参数边界")
    
    def get_parameter_constraints(self, module_name: str) -> Dict[str, Any]:
        """获取模块的参数约束"""
        module_constraints = {}
        
        for param_name, bounds in self.constraints['parameter_bounds'].items():
            if param_name.startswith(module_name):
                module_constraints[param_name] = bounds
        
        return module_constraints
    
    def get_convergence_requirements(self, module_name: str) -> Dict[str, Any]:
        """获取模块的收敛要求"""
        return self.constraints['convergence_requirements'].get(module_name, {})
    
    def get_quality_thresholds(self) -> Dict[str, float]:
        """获取质量阈值"""
        return self.constraints['quality_thresholds']

class SOCIAConfigManager:
    """SOCIA配置管理器"""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.module_parser = SOCIAModuleParser()
        self.data_analyzer = SOCIADataAnalyzer()
        self.verification_analyzer = SOCIAVerificationAnalyzer()
    
    def load_all_configs(self) -> Dict[str, Any]:
        """加载所有SOCIA配置"""
        configs = {}
        
        try:
            # 加载模型计划
            model_plan_path = self.output_dir / "model_plan_iter_1.json"
            if model_plan_path.exists():
                self.module_parser.load_model_plan(model_plan_path)
                configs['model_plan'] = self.module_parser
            else:
                logger.warning(f"模型计划文件不存在: {model_plan_path}")
            
            # 加载数据分析
            data_analysis_path = self.output_dir / "data_analysis_iter_1.json"
            if data_analysis_path.exists():
                self.data_analyzer.load_data_analysis(data_analysis_path)
                configs['data_analysis'] = self.data_analyzer
            else:
                logger.warning(f"数据分析文件不存在: {data_analysis_path}")
            
            # 加载验证结果
            verification_path = self.output_dir / "verification_results_iter_1.json"
            if verification_path.exists():
                self.verification_analyzer.load_verification_results(verification_path)
                configs['verification'] = self.verification_analyzer
            else:
                logger.warning(f"验证结果文件不存在: {verification_path}")
            
            logger.info("SOCIA配置加载完成")
            return configs
            
        except Exception as e:
            logger.error(f"SOCIA配置加载失败: {e}")
            raise
    
    def get_calibration_strategy(self) -> Dict[str, Any]:
        """获取校准策略"""
        strategy = {
            'calibration_order': self.module_parser.get_calibration_order(),
            'calibration_modules': self.module_parser.calibration_modules,
            'summary_statistics': {},
            'parameter_constraints': {},
            'convergence_requirements': {}
        }
        
        # 为每个可校准模块设计策略
        for module_name in strategy['calibration_order']:
            # 摘要统计设计
            strategy['summary_statistics'][module_name] = \
                self.data_analyzer.get_summary_statistics_design(module_name)
            
            # 参数约束
            strategy['parameter_constraints'][module_name] = \
                self.verification_analyzer.get_parameter_constraints(module_name)
            
            # 收敛要求
            strategy['convergence_requirements'][module_name] = \
                self.verification_analyzer.get_convergence_requirements(module_name)
        
        logger.info(f"校准策略生成完成: {len(strategy['calibration_order'])} 个模块")
        return strategy
