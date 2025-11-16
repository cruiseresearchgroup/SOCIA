"""
数据处理工具模块
提供数据清洗、格式转换、缺失值处理、SOCIA数据格式支持等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.cleaning_rules: Dict[str, callable] = {}
        self.missing_strategies: Dict[str, str] = {}
    
    def add_cleaning_rule(self, column: str, rule: callable) -> None:
        """添加清洗规则"""
        self.cleaning_rules[column] = rule
        logger.info(f"添加清洗规则: {column}")
    
    def set_missing_strategy(self, column: str, strategy: str) -> None:
        """设置缺失值处理策略"""
        valid_strategies = ['drop', 'fill_mean', 'fill_median', 'fill_mode', 'forward_fill', 'backward_fill']
        if strategy not in valid_strategies:
            raise ValueError(f"无效的缺失值处理策略: {strategy}")
        
        self.missing_strategies[column] = strategy
        logger.info(f"设置缺失值处理策略: {column} -> {strategy}")
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        cleaned_data = data.copy()
        
        # 应用清洗规则
        for column, rule in self.cleaning_rules.items():
            if column in cleaned_data.columns:
                try:
                    cleaned_data[column] = cleaned_data[column].apply(rule)
                    logger.info(f"应用清洗规则: {column}")
                except Exception as e:
                    logger.warning(f"清洗规则应用失败: {column}, 错误: {e}")
        
        # 处理缺失值
        for column, strategy in self.missing_strategies.items():
            if column in cleaned_data.columns:
                cleaned_data = self._handle_missing_values(cleaned_data, column, strategy)
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
        """处理缺失值"""
        if strategy == 'drop':
            data = data.dropna(subset=[column])
        elif strategy == 'fill_mean':
            data[column] = data[column].fillna(data[column].mean())
        elif strategy == 'fill_median':
            data[column] = data[column].fillna(data[column].median())
        elif strategy == 'fill_mode':
            mode_value = data[column].mode()
            if not mode_value.empty:
                data[column] = data[column].fillna(mode_value[0])
        elif strategy == 'forward_fill':
            data[column] = data[column].fillna(method='ffill')
        elif strategy == 'backward_fill':
            data[column] = data[column].fillna(method='bfill')
        
        logger.info(f"处理缺失值: {column} -> {strategy}")
        return data

class SOCIADataProcessor:
    """SOCIA数据处理器"""
    
    def __init__(self):
        self.data_cleaner = DataCleaner()
        self.target_columns = ['wearing_mask', 'received_info']
        self.time_column = 'day'
    
    def load_socia_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """加载SOCIA数据"""
        try:
            data = pd.read_csv(data_path)
            logger.info(f"SOCIA数据加载成功: {len(data)} 行, {len(data.columns)} 列")
            return data
        except Exception as e:
            logger.error(f"SOCIA数据加载失败: {e}")
            raise
    
    def preprocess_socia_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理SOCIA数据"""
        processed_data = data.copy()
        
        # 确保时间列存在
        if self.time_column not in processed_data.columns:
            logger.warning(f"时间列 {self.time_column} 不存在，使用索引作为时间")
            processed_data[self.time_column] = processed_data.index
        
        # 转换数据类型
        processed_data[self.time_column] = pd.to_numeric(processed_data[self.time_column], errors='coerce')
        
        # 处理目标列
        for col in self.target_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # 应用数据清洗
        processed_data = self.data_cleaner.clean_data(processed_data)
        
        logger.info(f"SOCIA数据预处理完成: {len(processed_data)} 行")
        return processed_data
    
    def calculate_daily_rates(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算日率"""
        if self.time_column not in data.columns:
            raise ValueError(f"时间列 {self.time_column} 不存在")
        
        daily_rates = data.groupby(self.time_column).agg({
            'wearing_mask': ['mean', 'count'],
            'received_info': ['mean', 'count']
        }).reset_index()
        
        # 展平列名
        daily_rates.columns = [
            'day', 'adoption_rate', 'adoption_count',
            'info_rate', 'info_count'
        ]
        
        logger.info(f"日率计算完成: {len(daily_rates)} 天")
        return daily_rates
    
    def filter_training_window(self, data: pd.DataFrame, 
                              start_day: int = 0, end_day: int = 29) -> pd.DataFrame:
        """筛选训练窗口"""
        filtered_data = data[
            (data[self.time_column] >= start_day) & 
            (data[self.time_column] <= end_day)
        ].copy()
        
        logger.info(f"训练窗口筛选: {start_day}-{end_day} 天, {len(filtered_data)} 行")
        return filtered_data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """验证数据质量"""
        quality_report = {
            'total_rows': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'value_ranges': {},
            'quality_score': 0.0
        }
        
        # 计算值范围
        for col in data.select_dtypes(include=[np.number]).columns:
            quality_report['value_ranges'][col] = {
                'min': data[col].min(),
                'max': data[col].max(),
                'mean': data[col].mean(),
                'std': data[col].std()
            }
        
        # 计算质量分数
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        quality_score = max(0, 1 - missing_ratio)
        quality_report['quality_score'] = quality_score
        
        logger.info(f"数据质量验证完成: 质量分数 {quality_score:.3f}")
        return quality_report

class SummaryStatisticsCalculator:
    """摘要统计计算器"""
    
    def __init__(self):
        self.statistics_functions = {
            'rmse': self._calculate_rmse,
            'mae': self._calculate_mae,
            'r2': self._calculate_r2,
            'correlation': self._calculate_correlation,
            'peak_value': self._calculate_peak_value,
            'peak_time': self._calculate_peak_time,
            'slope': self._calculate_slope
        }
    
    def calculate_summary_statistics(self, 
                                   observed: np.ndarray, 
                                   simulated: np.ndarray,
                                   statistics: List[str] = None) -> Dict[str, float]:
        """计算摘要统计"""
        if statistics is None:
            statistics = list(self.statistics_functions.keys())
        
        results = {}
        
        for stat_name in statistics:
            if stat_name in self.statistics_functions:
                try:
                    value = self.statistics_functions[stat_name](observed, simulated)
                    results[stat_name] = value
                except Exception as e:
                    logger.warning(f"统计计算失败: {stat_name}, 错误: {e}")
                    results[stat_name] = np.nan
        
        logger.info(f"摘要统计计算完成: {len(results)} 个指标")
        return results
    
    def _calculate_rmse(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """计算RMSE"""
        return np.sqrt(np.mean((observed - simulated) ** 2))
    
    def _calculate_mae(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """计算MAE"""
        return np.mean(np.abs(observed - simulated))
    
    def _calculate_r2(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """计算R²"""
        ss_res = np.sum((observed - simulated) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _calculate_correlation(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """计算相关系数"""
        return np.corrcoef(observed, simulated)[0, 1]
    
    def _calculate_peak_value(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """计算峰值"""
        return np.max(simulated)
    
    def _calculate_peak_time(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """计算峰时"""
        return np.argmax(simulated)
    
    def _calculate_slope(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """计算斜率"""
        if len(simulated) < 2:
            return 0
        return (simulated[-1] - simulated[0]) / (len(simulated) - 1)

class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.validation_rules: Dict[str, callable] = {}
    
    def add_validation_rule(self, name: str, rule: callable) -> None:
        """添加验证规则"""
        self.validation_rules[name] = rule
        logger.info(f"添加验证规则: {name}")
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, bool]:
        """验证数据"""
        results = {}
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                is_valid = rule_func(data)
                results[rule_name] = is_valid
                logger.info(f"验证规则 {rule_name}: {'通过' if is_valid else '失败'}")
            except Exception as e:
                logger.error(f"验证规则执行失败: {rule_name}, 错误: {e}")
                results[rule_name] = False
        
        return results
    
    def validate_socia_data(self, data: pd.DataFrame) -> Dict[str, bool]:
        """验证SOCIA数据"""
        validation_results = {}
        
        # 检查必需列
        required_columns = ['day', 'wearing_mask', 'received_info']
        missing_columns = [col for col in required_columns if col not in data.columns]
        validation_results['required_columns'] = len(missing_columns) == 0
        
        if missing_columns:
            logger.warning(f"缺少必需列: {missing_columns}")
        
        # 检查数据类型
        if 'day' in data.columns:
            validation_results['day_numeric'] = pd.api.types.is_numeric_dtype(data['day'])
        
        # 检查值范围
        if 'wearing_mask' in data.columns:
            mask_values = data['wearing_mask'].dropna()
            validation_results['mask_binary'] = mask_values.isin([0, 1]).all()
        
        if 'received_info' in data.columns:
            info_values = data['received_info'].dropna()
            validation_results['info_binary'] = info_values.isin([0, 1]).all()
        
        logger.info(f"SOCIA数据验证完成: {validation_results}")
        return validation_results





