"""Configuration manager for applying and tracking strategy adjustments."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..utils.logger import logger
from .adaptive_engine import StrategyAdjustment


class ConfigurationManager:
    """Manages strategy configuration updates with versioning and rollback."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        updates_log_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None
    ):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml
            updates_log_path: Path to strategy_updates.json
            backup_dir: Directory for config backups
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        if updates_log_path is None:
            project_root = Path(__file__).parent.parent.parent
            updates_log_path = project_root / "data" / "strategy_updates.json"
        
        if backup_dir is None:
            project_root = Path(__file__).parent.parent.parent
            backup_dir = project_root / "data" / "config_backups"
        
        self.config_path = Path(config_path)
        self.updates_log_path = Path(updates_log_path)
        self.backup_dir = Path(backup_dir)
        
        # Ensure directories exist
        self.updates_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize updates log if it doesn't exist
        if not self.updates_log_path.exists():
            self._init_updates_log()
        
        logger.info(f"ConfigurationManager initialized: {self.config_path}")
    
    def _init_updates_log(self) -> None:
        """Initialize the updates log file."""
        initial_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updates": []
        }
        
        with open(self.updates_log_path, "w") as f:
            json.dump(initial_data, f, indent=2)
        
        logger.info(f"Initialized updates log: {self.updates_log_path}")
    
    def load_config(self) -> dict:
        """Load current configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def backup_config(self, reason: str = "auto") -> Path:
        """Create a timestamped backup of current config.
        
        Args:
            reason: Reason for backup
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"config_backup_{timestamp}_{reason}.yaml"
        backup_path = self.backup_dir / backup_filename
        
        shutil.copy2(self.config_path, backup_path)
        logger.info(f"Config backed up to: {backup_path}")
        
        return backup_path
    
    def apply_adjustments(
        self,
        adjustments: List[StrategyAdjustment],
        performance_before: Optional[dict] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Apply approved parameter adjustments to configuration.
        
        Args:
            adjustments: List of adjustments to apply
            performance_before: Performance metrics before changes
            dry_run: If True, simulate changes without applying
            
        Returns:
            Dictionary with application results
        """
        # Filter to only approved adjustments
        approved = [adj for adj in adjustments if adj.approved]
        
        if not approved:
            return {
                "success": False,
                "message": "No approved adjustments to apply",
                "applied_count": 0
            }
        
        logger.info(f"Applying {len(approved)} approved adjustments (dry_run={dry_run})")
        
        # Load current config
        config = self.load_config()
        
        if not config:
            return {
                "success": False,
                "message": "Failed to load configuration",
                "applied_count": 0
            }
        
        # Backup current config
        if not dry_run:
            backup_path = self.backup_config(reason="pre_adjustment")
        
        # Apply changes
        changes_made = {}
        errors = []
        
        for adjustment in approved:
            try:
                # Navigate to the correct section
                param_path = self._get_parameter_path(adjustment.parameter)
                
                if param_path is None:
                    errors.append(f"Unknown parameter: {adjustment.parameter}")
                    continue
                
                # Update the value
                old_value = self._get_nested_value(config, param_path)
                
                if old_value is None:
                    logger.warning(f"Parameter {adjustment.parameter} not found in config")
                    errors.append(f"Parameter not found: {adjustment.parameter}")
                    continue
                
                # Verify old value matches
                if old_value != adjustment.old_value:
                    logger.warning(
                        f"Old value mismatch for {adjustment.parameter}: "
                        f"expected {adjustment.old_value}, got {old_value}"
                    )
                
                if not dry_run:
                    self._set_nested_value(config, param_path, adjustment.new_value)
                
                changes_made[adjustment.parameter] = {
                    "old_value": old_value,
                    "new_value": adjustment.new_value,
                    "reasoning": adjustment.reasoning,
                    "confidence": adjustment.confidence,
                    "risk_level": adjustment.risk_level
                }
                
                adjustment.applied = True
                
                logger.info(
                    f"{'[DRY RUN] ' if dry_run else ''}Applied: {adjustment.parameter} "
                    f"{old_value} → {adjustment.new_value}"
                )
                
            except Exception as e:
                error_msg = f"Failed to apply {adjustment.parameter}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Save updated config
        if not dry_run and changes_made:
            try:
                with open(self.config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                logger.info(f"Configuration updated: {len(changes_made)} changes applied")
            except Exception as e:
                logger.error(f"Failed to save config: {e}")
                return {
                    "success": False,
                    "message": f"Failed to save configuration: {e}",
                    "applied_count": 0
                }
        
        # Log the update
        if not dry_run and changes_made:
            self._log_update(changes_made, adjustments, performance_before)
        
        return {
            "success": True,
            "message": f"Applied {len(changes_made)} parameter adjustments",
            "applied_count": len(changes_made),
            "changes": changes_made,
            "errors": errors,
            "dry_run": dry_run,
            "backup_path": str(backup_path) if not dry_run else None
        }
    
    def _get_parameter_path(self, parameter: str) -> Optional[List[str]]:
        """Get the path to a parameter in the config structure.
        
        Args:
            parameter: Parameter name
            
        Returns:
            List of keys to navigate to parameter, or None
        """
        # Map parameter names to config paths
        param_map = {
            # Strategy parameters
            "rsi_buy": ["strategies", 0, "params", "rsi_buy"],
            "rsi_sell": ["strategies", 0, "params", "rsi_sell"],
            "sentiment_buy": ["strategies", 0, "params", "sentiment_buy"],
            "sentiment_sell": ["strategies", 0, "params", "sentiment_sell"],
            "sentiment_weight": ["strategies", 0, "params", "sentiment_weight"],
            
            # Trading parameters
            "stop_loss_ticks": ["trading", "stop_loss_ticks"],
            "take_profit_ticks": ["trading", "take_profit_ticks"],
            "max_position_size": ["trading", "max_position_size"],
            "max_daily_loss": ["trading", "max_daily_loss"],
            "max_daily_trades": ["trading", "max_daily_trades"],
            
            # LLM parameters
            "min_confidence_threshold": ["llm", "min_confidence_threshold"],
            "temperature": ["llm", "temperature"],
            
            # Risk parameters
            "atr_multiplier": ["risk", "atr_multiplier"],
            "max_portfolio_heat": ["risk", "max_portfolio_heat"],
        }
        
        return param_map.get(parameter)
    
    def _get_nested_value(self, config: dict, path: List[str]) -> Any:
        """Get a value from nested dictionary.
        
        Args:
            config: Configuration dictionary
            path: List of keys to navigate
            
        Returns:
            Value at path, or None if not found
        """
        current = config
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
            elif isinstance(current, list) and isinstance(key, int):
                if 0 <= key < len(current):
                    current = current[key]
                else:
                    return None
            else:
                return None
            
            if current is None:
                return None
        
        return current
    
    def _set_nested_value(self, config: dict, path: List[str], value: Any) -> None:
        """Set a value in nested dictionary.
        
        Args:
            config: Configuration dictionary
            path: List of keys to navigate
            value: Value to set
        """
        current = config
        for key in path[:-1]:
            if isinstance(current, dict):
                if key not in current:
                    current[key] = {}
                current = current[key]
            elif isinstance(current, list) and isinstance(key, int):
                current = current[key]
        
        final_key = path[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list) and isinstance(final_key, int):
            current[final_key] = value
    
    def _log_update(
        self,
        changes: Dict[str, dict],
        adjustments: List[StrategyAdjustment],
        performance_before: Optional[dict]
    ) -> None:
        """Log parameter update to updates log.
        
        Args:
            changes: Dictionary of changes made
            adjustments: List of adjustments
            performance_before: Performance before changes
        """
        # Load existing log
        with open(self.updates_log_path, "r") as f:
            log_data = json.load(f)
        
        # Create update entry
        update_entry = {
            "timestamp": datetime.now().isoformat(),
            "changes": changes,
            "reasoning": adjustments[0].reasoning if adjustments else "",
            "confidence": adjustments[0].confidence if adjustments else 0.0,
            "risk_level": adjustments[0].risk_level if adjustments else "unknown",
            "performance_before": performance_before,
            "performance_after": None,  # To be filled in later
            "rollback_path": None,  # For potential rollback
            "applied_by": "autonomous_system"
        }
        
        # Add to updates list
        log_data["updates"].append(update_entry)
        
        # Save updated log
        with open(self.updates_log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Logged {len(changes)} parameter changes to updates log")
    
    def get_update_history(self, limit: int = 10) -> List[dict]:
        """Get recent update history.
        
        Args:
            limit: Maximum number of updates to return
            
        Returns:
            List of recent updates
        """
        try:
            with open(self.updates_log_path, "r") as f:
                log_data = json.load(f)
            
            updates = log_data.get("updates", [])
            return updates[-limit:] if len(updates) > limit else updates
            
        except Exception as e:
            logger.error(f"Failed to get update history: {e}")
            return []
    
    def rollback_to_backup(self, backup_path: Path) -> bool:
        """Rollback configuration to a previous backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        try:
            # Backup current config before rollback
            self.backup_config(reason="pre_rollback")
            
            # Restore from backup
            shutil.copy2(backup_path, self.config_path)
            
            logger.info(f"Configuration rolled back to: {backup_path}")
            
            # Log the rollback
            with open(self.updates_log_path, "r") as f:
                log_data = json.load(f)
            
            rollback_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": "rollback",
                "restored_from": str(backup_path),
                "reason": "manual_rollback"
            }
            
            log_data["updates"].append(rollback_entry)
            
            with open(self.updates_log_path, "w") as f:
                json.dump(log_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback config: {e}")
            return False
    
    def rollback_last_update(self) -> bool:
        """Rollback the most recent parameter update.
        
        Returns:
            True if successful
        """
        # Get most recent backup
        backups = sorted(self.backup_dir.glob("config_backup_*_pre_adjustment.yaml"))
        
        if not backups:
            logger.error("No backup found for rollback")
            return False
        
        latest_backup = backups[-1]
        logger.info(f"Rolling back to: {latest_backup}")
        
        return self.rollback_to_backup(latest_backup)
    
    def update_performance_after(
        self,
        update_index: int,
        performance_after: dict
    ) -> bool:
        """Update the performance_after field for a specific update.
        
        Args:
            update_index: Index of update in log (0-based from end)
            performance_after: Performance metrics after change
            
        Returns:
            True if successful
        """
        try:
            with open(self.updates_log_path, "r") as f:
                log_data = json.load(f)
            
            if 0 <= update_index < len(log_data["updates"]):
                log_data["updates"][-(update_index + 1)]["performance_after"] = performance_after
                
                with open(self.updates_log_path, "w") as f:
                    json.dump(log_data, f, indent=2)
                
                logger.info(f"Updated performance_after for update index {update_index}")
                return True
            else:
                logger.error(f"Invalid update index: {update_index}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update performance_after: {e}")
            return False
    
    def get_current_strategy_params(self) -> dict:
        """Get current strategy parameters.
        
        Returns:
            Dictionary of current strategy parameters
        """
        config = self.load_config()
        
        # Extract key parameters
        strategy_params = {}
        
        # From strategies section
        if "strategies" in config and len(config["strategies"]) > 0:
            params = config["strategies"][0].get("params", {})
            strategy_params.update(params)
        
        # From trading section
        if "trading" in config:
            strategy_params.update({
                "stop_loss_ticks": config["trading"].get("stop_loss_ticks"),
                "take_profit_ticks": config["trading"].get("take_profit_ticks"),
                "max_position_size": config["trading"].get("max_position_size"),
                "max_daily_loss": config["trading"].get("max_daily_loss"),
            })
        
        # From LLM section
        if "llm" in config:
            strategy_params.update({
                "min_confidence_threshold": config["llm"].get("min_confidence_threshold"),
                "temperature": config["llm"].get("temperature"),
            })
        
        # From risk section
        if "risk" in config:
            strategy_params.update({
                "atr_multiplier": config["risk"].get("atr_multiplier"),
                "max_portfolio_heat": config["risk"].get("max_portfolio_heat"),
            })
        
        return strategy_params
