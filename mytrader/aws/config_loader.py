"""
AWS Config Loader

Loads deployed resource configuration from yaml file and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from loguru import logger


class AWSConfigLoader:
    """Loader for AWS deployed resource configuration."""
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "aws" / "config" / "deployed_resources.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader.
        
        Args:
            config_path: Path to deployed_resources.yaml, defaults to standard location
        """
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._config: Optional[Dict[str, Any]] = None
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from yaml file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If yaml module not available
        """
        if not YAML_AVAILABLE:
            raise ValueError("PyYAML is required. Install with: pip install pyyaml")
            
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"AWS config not found at {self.config_path}. "
                "Deploy infrastructure first or specify correct path."
            )
        
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)
            
        logger.info(f"Loaded AWS config from {self.config_path}")
        return self._config
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get loaded configuration, loading if needed."""
        if self._config is None:
            self.load()
        return self._config
    
    def get_agent_ids(self) -> Dict[str, str]:
        """Get agent IDs mapped to standard names.
        
        Returns:
            Dictionary mapping agent names to IDs:
            {'data': '...', 'decision': '...', 'risk': '...', 'learning': '...'}
        """
        agents = self.config.get('agents', {})
        return {
            'data': agents.get('data_ingestion', {}).get('id', ''),
            'decision': agents.get('decision_engine', {}).get('id', ''),
            'risk': agents.get('risk_control', {}).get('id', ''),
            'learning': agents.get('learning', {}).get('id', ''),
        }
    
    def get_agent_alias_ids(self) -> Dict[str, str]:
        """Get agent alias IDs mapped to standard names.
        
        Returns:
            Dictionary mapping agent names to alias IDs
        """
        agents = self.config.get('agents', {})
        return {
            'data': agents.get('data_ingestion', {}).get('alias_id', ''),
            'decision': agents.get('decision_engine', {}).get('alias_id', ''),
            'risk': agents.get('risk_control', {}).get('alias_id', ''),
            'learning': agents.get('learning', {}).get('alias_id', ''),
        }
    
    def get_knowledge_base_id(self) -> str:
        """Get Knowledge Base ID."""
        return self.config.get('knowledge_base', {}).get('id', '')
    
    def get_s3_bucket(self) -> str:
        """Get S3 data bucket name."""
        return self.config.get('s3', {}).get('data_bucket', '')
    
    def get_step_function_arn(self, flow_name: str) -> str:
        """Get Step Function ARN by name.
        
        Args:
            flow_name: Either 'signal_flow' or 'nightly_flow'
            
        Returns:
            Step Function ARN
        """
        return self.config.get('step_functions', {}).get(flow_name, '')
    
    def get_region(self) -> str:
        """Get AWS region."""
        return self.config.get('region', 'us-east-1')
    
    def to_invoker_config(self) -> Dict[str, Any]:
        """Convert to AgentInvoker configuration format.
        
        Returns:
            Dictionary ready for AgentInvoker initialization
        """
        return {
            'region_name': self.get_region(),
            'agent_ids': self.get_agent_ids(),
            'agent_alias_ids': self.get_agent_alias_ids(),
            's3_bucket': self.get_s3_bucket(),
            'knowledge_base_id': self.get_knowledge_base_id(),
            'signal_flow_arn': self.get_step_function_arn('signal_flow'),
            'nightly_flow_arn': self.get_step_function_arn('nightly_flow'),
        }


def load_aws_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load AWS configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    loader = AWSConfigLoader(config_path)
    return loader.to_invoker_config()


# Module-level singleton for easy access
_loader: Optional[AWSConfigLoader] = None


def get_aws_config() -> AWSConfigLoader:
    """Get module-level config loader singleton.
    
    Returns:
        AWSConfigLoader instance
    """
    global _loader
    if _loader is None:
        _loader = AWSConfigLoader()
    return _loader
