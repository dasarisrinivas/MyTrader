"""Strategy abstractions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class Signal:
    action: str
    confidence: float
    metadata: Dict[str, float]


class BaseStrategy(ABC):
    name: str

    @abstractmethod
    def generate(self, features: pd.DataFrame) -> Signal:
        """Return trading signal based on latest row."""
