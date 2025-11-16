"""
文件操作工具模块
提供文件读写、路径处理、JSON/CSV操作等功能
"""

import os
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

class FileUtils:
    """文件操作工具类"""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """确保目录存在，如果不存在则创建"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {file_path}, 错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载JSON文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
                  indent: int = 2, ensure_ascii: bool = False) -> None:
        """保存JSON文件"""
        try:
            file_path = Path(file_path)
            FileUtils.ensure_dir(file_path.parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logger.info(f"JSON文件保存成功: {file_path}")
        except Exception as e:
            logger.error(f"保存JSON文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """加载CSV文件"""
        try:
            return pd.read_csv(file_path, **kwargs)
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            raise
        except Exception as e:
            logger.error(f"加载CSV文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def save_csv(data: pd.DataFrame, file_path: Union[str, Path], 
                 index: bool = False, **kwargs) -> None:
        """保存CSV文件"""
        try:
            file_path = Path(file_path)
            FileUtils.ensure_dir(file_path.parent)
            data.to_csv(file_path, index=index, **kwargs)
            logger.info(f"CSV文件保存成功: {file_path}")
        except Exception as e:
            logger.error(f"保存CSV文件失败: {file_path}, 错误: {e}")
            raise
    
    @staticmethod
    def get_file_list(directory: Union[str, Path], 
                      pattern: str = "*", 
                      recursive: bool = False) -> List[Path]:
        """获取文件列表"""
        directory = Path(directory)
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> bool:
        """验证文件是否存在"""
        return Path(file_path).exists()
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """获取文件大小（字节）"""
        return Path(file_path).stat().st_size
    
    @staticmethod
    def backup_file(file_path: Union[str, Path], 
                   backup_suffix: str = ".backup") -> Path:
        """备份文件"""
        file_path = Path(file_path)
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        
        if file_path.exists():
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"文件备份成功: {file_path} -> {backup_path}")
        
        return backup_path
    
    @staticmethod
    def clean_temp_files(directory: Union[str, Path], 
                        pattern: str = "*.tmp") -> None:
        """清理临时文件"""
        directory = Path(directory)
        temp_files = directory.glob(pattern)
        
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                logger.info(f"临时文件已删除: {temp_file}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {temp_file}, 错误: {e}")
    
    @staticmethod
    def get_relative_path(base_path: Union[str, Path], 
                         target_path: Union[str, Path]) -> Path:
        """获取相对路径"""
        return Path(target_path).relative_to(Path(base_path))
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """标准化路径"""
        return Path(path).resolve()





