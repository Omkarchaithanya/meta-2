#!/usr/bin/env python3
"""
Configuration management for OpenEnv SME environment.
Allows flexible configuration of environment parameters, server settings, etc.
"""
from typing import Dict, Optional, Any
import yaml
import json
from pathlib import Path


class Config:
    """Configuration container."""
    
    # Default values
    DEFAULTS = {
        # Environment settings
        "env": {
            "seed": None,
            "tasks": ["easy", "medium", "hard"],
            "max_episode_length": 12,
        },
        # Server settings
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "log_level": "INFO",
        },
        # Client settings
        "client": {
            "server_url": "ws://localhost:8000/ws/openenv-sme",
            "timeout": 30.0,
            "max_retries": 3,
        },
        # Evaluation settings
        "evaluation": {
            "num_episodes": 100,
            "strategies": ["fallback", "random", "heuristic"],
            "metrics_output_dir": "./eval_results",
        },
        # Training settings
        "training": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "num_generations": 8,
            "max_completion_length": 1024,
        },
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize config with defaults and optional overrides."""
        self.config = {}
        self._load_defaults()
        if config_dict:
            self._merge_config(config_dict)
    
    def _load_defaults(self):
        """Load default configuration."""
        self.config = self._deep_copy(self.DEFAULTS)
    
    def _merge_config(self, custom_config: Dict[str, Any]):
        """Merge custom config with defaults (deep merge)."""
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in self.config:
                if isinstance(self.config[key], dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                self.config[key] = value
    
    @staticmethod
    def _deep_copy(d: Dict) -> Dict:
        """Create a deep copy of a dictionary."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = Config._deep_copy(value)
            else:
                result[key] = value
        return result
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get a config value by dot-notation path.
        
        Example:
            config.get("server.port")  # Returns 8000
            config.get("env.seed")     # Returns None
        """
        keys = path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any):
        """Set a config value by dot-notation path."""
        keys = path.split(".")
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict:
        """Export config as dictionary."""
        return self._deep_copy(self.config)
    
    def to_yaml(self, output_path: str):
        """Save config to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Configuration saved to: {output_path}")
    
    def to_json(self, output_path: str):
        """Save config to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {output_path}")
    
    def __repr__(self):
        """String representation."""
        return f"Config({self.config})"


# Global config instance
_global_config = None


def get_config() -> Config:
    """Get global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set global config instance."""
    global _global_config
    _global_config = config


def load_config(path: str) -> Config:
    """Load configuration from file and set as global."""
    if path.endswith('.yaml') or path.endswith('.yml'):
        config = Config.from_yaml(path)
    elif path.endswith('.json'):
        config = Config.from_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path}")
    
    set_config(config)
    return config


if __name__ == "__main__":
    # Example usage
    config = Config()
    
    print("Default config:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Access values
    print(f"\nServer port: {config.get('server.port')}")
    print(f"Learning rate: {config.get('training.learning_rate')}")
    
    # Modify values
    config.set('server.port', 9000)
    print(f"Modified server port: {config.get('server.port')}")
