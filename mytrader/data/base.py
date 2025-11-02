"""Data collection interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DataCollector(ABC):
    """Abstract base for data collectors."""

    @abstractmethod
    async def collect(self) -> pd.DataFrame:
        """Fetch latest data snapshot."""

    @abstractmethod
    async def stream(self) -> Any:
        """Asynchronous generator yielding streaming data rows."""
