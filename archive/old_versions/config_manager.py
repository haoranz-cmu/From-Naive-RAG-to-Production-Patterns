"""
LLM Configuration Manager

这个模块管理LLM的所有配置，支持环境变量和配置文件。
"""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMConfigManager:
    """LLM配置管理器"""
    
    def __init__(self, config_file: str = "config_llm.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """仅从配置文件加载配置（必要时再应用环境变量覆盖）"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"未找到配置文件: {self.config_file}. 请创建该文件（例如 config_llm.yaml）。"
            )

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败 {self.config_file}: {e}")

        
        return cfg
    
    def get_default_model_config(self) -> Dict[str, Any]:
        """获取默认模型配置"""
        return self.config["models"]["default"]

    def get_fallback_model_config(self) -> Dict[str, Any]:
        """获取备用模型配置"""
        return self.config["models"]["fallback"]
    

    def get_generation_config(self, strategy: str = None) -> Dict[str, Any]:
        """获取生成配置"""
        base_config = self.config["generation"].copy()
        
        # 如果指定了策略，尝试获取策略特定配置
        if strategy and "strategies" in self.config["prompting"]:
            strategy_config = self.config["prompting"]["strategies"].get(strategy, {})
            base_config.update(strategy_config)
        
        return base_config
    
    def get_prompting_config(self) -> Dict[str, Any]:
        """获取提示配置"""
        return self.config["prompting"]
    
    def get_api_config(self) -> Dict[str, Any]:
        """获取API配置"""
        return self.config["api"]
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config["logging"]
    
    
