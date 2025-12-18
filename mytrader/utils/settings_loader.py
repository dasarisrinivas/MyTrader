"""Helper for loading application settings from YAML overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ..config import Settings, StrategyConfig


def load_settings(path: str | Path | None = None) -> Settings:
    settings = Settings()
    if path is None:
        return settings

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    # Allow environment overrides for critical runtime thresholds
    env_overrides = [
        ("rag", "min_similar_trades", "MIN_SIMILAR_TRADES", int),
        ("rag", "min_weighted_win_rate", "MIN_WEIGHTED_WIN_RATE", float),
        ("rag", "min_weighted_win_rate_soft_floor", "MIN_WEIGHTED_WIN_RATE_SOFT_FLOOR", float),
        ("rag", "min_similar_trades_for_full_threshold", "MIN_SIMILAR_TRADES_FOR_FULL_THRESHOLD", int),
        ("trading", "confidence_threshold", "CONFIDENCE_THRESHOLD", float),
    ]
    for section, key, env_name, caster in env_overrides:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            cast_value = caster(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} value '{raw_value}' is invalid") from exc
        section_data = data.setdefault(section, {})
        section_data[key] = cast_value

    # Handle strategies separately since they need special parsing
    if "strategies" in data:
        strategies_data = data.pop("strategies")
        settings.strategies = [
            StrategyConfig(
                name=s.get("name", "unknown"),
                enabled=s.get("enabled", True),
                params=s.get("params", {})
            )
            for s in strategies_data
        ]
    
    update_dataclass(settings, data)
    settings.validate()
    return settings


def update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            continue
        attr = getattr(instance, key)
        if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
            update_dataclass(attr, value)
        else:
            setattr(instance, key, value)
