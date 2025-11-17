"""
Configurable Risk Management Parameters
Allows dynamic configuration of stop-loss, take-profit, and other risk settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
from pathlib import Path
import json

from ..utils.logger import logger


@dataclass
class StopLossConfig:
    """Configuration for stop-loss parameters."""
    
    # Stop-loss type
    type: Literal["fixed_ticks", "atr_based", "percentage"] = "atr_based"
    
    # Fixed ticks stop-loss
    fixed_ticks: float = 10.0
    
    # ATR-based stop-loss
    atr_multiplier: float = 2.0  # Stop at 2x ATR from entry
    
    # Percentage-based stop-loss
    percentage: float = 1.0  # 1% stop-loss
    
    # Trailing stop configuration
    use_trailing: bool = True
    trailing_type: Literal["fixed", "atr_based"] = "atr_based"
    trailing_distance_ticks: float = 5.0
    trailing_atr_multiplier: float = 1.5
    
    def get_stop_distance(self, atr: float = 0.0, tick_size: float = 0.25) -> float:
        """Calculate stop-loss distance based on configuration.
        
        Args:
            atr: Current ATR value
            tick_size: Minimum price tick
            
        Returns:
            Stop-loss distance in price points
        """
        if self.type == "fixed_ticks":
            return self.fixed_ticks * tick_size
        elif self.type == "atr_based":
            if atr <= 0:
                logger.warning("ATR is 0, falling back to fixed ticks")
                return self.fixed_ticks * tick_size
            return self.atr_multiplier * atr
        elif self.type == "percentage":
            # Percentage needs current price, return None to signal caller needs to compute
            return 0.0
        else:
            return self.fixed_ticks * tick_size


@dataclass
class TakeProfitConfig:
    """Configuration for take-profit parameters."""
    
    # Take-profit type
    type: Literal["fixed_ticks", "risk_reward_ratio", "atr_based", "percentage"] = "risk_reward_ratio"
    
    # Fixed ticks take-profit
    fixed_ticks: float = 20.0
    
    # Risk:reward ratio (e.g., 2.0 means 2:1 reward:risk)
    risk_reward_ratio: float = 2.0
    
    # ATR-based take-profit
    atr_multiplier: float = 3.0
    
    # Percentage-based take-profit
    percentage: float = 2.0  # 2% take-profit
    
    # Partial profit taking
    use_partial_exits: bool = False
    partial_exit_levels: list = field(default_factory=lambda: [0.5, 0.75])  # Take profit at 50%, 75% of target
    partial_exit_sizes: list = field(default_factory=lambda: [0.3, 0.3])  # Exit 30%, 30% at each level
    
    def get_take_profit_distance(
        self,
        stop_distance: float,
        atr: float = 0.0,
        tick_size: float = 0.25
    ) -> float:
        """Calculate take-profit distance based on configuration.
        
        Args:
            stop_distance: Stop-loss distance (for risk:reward calculation)
            atr: Current ATR value
            tick_size: Minimum price tick
            
        Returns:
            Take-profit distance in price points
        """
        if self.type == "fixed_ticks":
            return self.fixed_ticks * tick_size
        elif self.type == "risk_reward_ratio":
            return stop_distance * self.risk_reward_ratio
        elif self.type == "atr_based":
            if atr <= 0:
                logger.warning("ATR is 0, falling back to fixed ticks")
                return self.fixed_ticks * tick_size
            return self.atr_multiplier * atr
        elif self.type == "percentage":
            return 0.0  # Needs current price
        else:
            return self.fixed_ticks * tick_size


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    
    method: Literal["fixed", "kelly", "risk_parity", "volatility_scaled"] = "kelly"
    
    # Fixed sizing
    fixed_size: int = 1  # Number of contracts
    
    # Kelly Criterion
    kelly_fraction: float = 0.5  # Use half-Kelly for safety (0.0-1.0)
    
    # Risk per trade (as % of capital)
    risk_per_trade_pct: float = 1.0  # Risk 1% per trade
    
    # Maximum position size
    max_contracts: int = 4
    
    # Scaling based on confidence
    scale_by_confidence: bool = True
    min_confidence_for_max_size: float = 0.85


@dataclass
class RiskLimitsConfig:
    """Configuration for overall risk limits."""
    
    # Daily limits
    max_daily_loss: float = 2000.0  # Maximum daily loss in dollars
    max_daily_trades: int = 20  # Maximum number of trades per day
    max_consecutive_losses: int = 5  # Stop after N consecutive losses
    
    # Position limits
    max_position_size: int = 4  # Maximum contracts per position
    max_total_exposure: float = 0.5  # Maximum % of capital at risk
    
    # Time-based restrictions
    no_trade_hours: list = field(default_factory=list)  # Hours to avoid trading (24h format)
    no_trade_days: list = field(default_factory=list)  # Days to avoid (e.g., ["Monday"])
    
    # Circuit breakers
    enable_circuit_breaker: bool = True
    circuit_breaker_loss_pct: float = 5.0  # Stop all trading if down 5% for the day


@dataclass
class RiskManagementConfig:
    """Complete risk management configuration."""
    
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RiskManagementConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            RiskManagementConfig instance
        """
        stop_loss = StopLossConfig(**config_dict.get("stop_loss", {}))
        take_profit = TakeProfitConfig(**config_dict.get("take_profit", {}))
        position_sizing = PositionSizingConfig(**config_dict.get("position_sizing", {}))
        limits = RiskLimitsConfig(**config_dict.get("limits", {}))
        
        return cls(
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_sizing=position_sizing,
            limits=limits
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'RiskManagementConfig':
        """Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            RiskManagementConfig instance
        """
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            
            logger.info(f"Loaded risk management config from {json_path}")
            return cls.from_dict(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load risk config from {json_path}: {e}")
            logger.info("Using default risk management configuration")
            return cls()
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "stop_loss": {
                "type": self.stop_loss.type,
                "fixed_ticks": self.stop_loss.fixed_ticks,
                "atr_multiplier": self.stop_loss.atr_multiplier,
                "percentage": self.stop_loss.percentage,
                "use_trailing": self.stop_loss.use_trailing,
                "trailing_type": self.stop_loss.trailing_type,
                "trailing_distance_ticks": self.stop_loss.trailing_distance_ticks,
                "trailing_atr_multiplier": self.stop_loss.trailing_atr_multiplier,
            },
            "take_profit": {
                "type": self.take_profit.type,
                "fixed_ticks": self.take_profit.fixed_ticks,
                "risk_reward_ratio": self.take_profit.risk_reward_ratio,
                "atr_multiplier": self.take_profit.atr_multiplier,
                "percentage": self.take_profit.percentage,
                "use_partial_exits": self.take_profit.use_partial_exits,
                "partial_exit_levels": self.take_profit.partial_exit_levels,
                "partial_exit_sizes": self.take_profit.partial_exit_sizes,
            },
            "position_sizing": {
                "method": self.position_sizing.method,
                "fixed_size": self.position_sizing.fixed_size,
                "kelly_fraction": self.position_sizing.kelly_fraction,
                "risk_per_trade_pct": self.position_sizing.risk_per_trade_pct,
                "max_contracts": self.position_sizing.max_contracts,
                "scale_by_confidence": self.position_sizing.scale_by_confidence,
                "min_confidence_for_max_size": self.position_sizing.min_confidence_for_max_size,
            },
            "limits": {
                "max_daily_loss": self.limits.max_daily_loss,
                "max_daily_trades": self.limits.max_daily_trades,
                "max_consecutive_losses": self.limits.max_consecutive_losses,
                "max_position_size": self.limits.max_position_size,
                "max_total_exposure": self.limits.max_total_exposure,
                "no_trade_hours": self.limits.no_trade_hours,
                "no_trade_days": self.limits.no_trade_days,
                "enable_circuit_breaker": self.limits.enable_circuit_breaker,
                "circuit_breaker_loss_pct": self.limits.circuit_breaker_loss_pct,
            }
        }
    
    def save_to_json(self, json_path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            json_path: Path to save JSON file
        """
        try:
            config_dict = self.to_dict()
            
            # Create parent directory if needed
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved risk management config to {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save risk config to {json_path}: {e}")
    
    def validate(self) -> bool:
        """Validate configuration parameters.
        
        Returns:
            True if valid, False otherwise
        """
        errors = []
        
        # Validate stop-loss
        if self.stop_loss.atr_multiplier <= 0:
            errors.append("Stop-loss ATR multiplier must be positive")
        
        if self.stop_loss.percentage <= 0 or self.stop_loss.percentage > 100:
            errors.append("Stop-loss percentage must be between 0 and 100")
        
        # Validate take-profit
        if self.take_profit.risk_reward_ratio <= 0:
            errors.append("Risk:reward ratio must be positive")
        
        # Validate position sizing
        if self.position_sizing.kelly_fraction < 0 or self.position_sizing.kelly_fraction > 1:
            errors.append("Kelly fraction must be between 0 and 1")
        
        if self.position_sizing.risk_per_trade_pct <= 0 or self.position_sizing.risk_per_trade_pct > 10:
            errors.append("Risk per trade must be between 0 and 10%")
        
        # Validate limits
        if self.limits.max_daily_loss <= 0:
            errors.append("Max daily loss must be positive")
        
        if self.limits.max_daily_trades <= 0:
            errors.append("Max daily trades must be positive")
        
        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return False
        
        return True


# Create a default configuration instance
DEFAULT_RISK_CONFIG = RiskManagementConfig()
