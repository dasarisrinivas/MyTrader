"""Data orchestration pipeline."""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from typing import AsyncIterator, Deque, Dict, List

import pandas as pd

from ..utils.logger import logger
from .base import DataCollector


class MarketDataPipeline:
    """Combines multiple collectors into a merged DataFrame stream."""

    def __init__(self, collectors: List[DataCollector], window: int = 1000) -> None:
        self.collectors = collectors
        self.window = window
        self.buffers: Dict[str, Deque[dict]] = {}

    async def snapshot(self) -> pd.DataFrame:
        frames = []
        for collector in self.collectors:
            data = await collector.collect()
            frames.append(data)
        return self._merge_frames(frames)

    async def stream(self) -> AsyncIterator[pd.DataFrame]:
        tasks = [asyncio.create_task(self._fan_in(collector)) for collector in self.collectors]
        try:
            while True:
                await asyncio.sleep(1)
                combined = self._combine_buffers()
                if not combined.empty:
                    yield combined
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fan_in(self, collector: DataCollector) -> None:
        name = collector.__class__.__name__
        buffer: Deque[dict] = self.buffers.setdefault(name, deque(maxlen=self.window))
        try:
            async for item in collector.stream():
                item.setdefault("timestamp", datetime.utcnow())
                buffer.append(item)
        except asyncio.CancelledError:  # noqa: TRY301
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Collector %s failed: %s", name, exc)

    def _combine_buffers(self) -> pd.DataFrame:
        frames = []
        for records in self.buffers.values():
            if not records:
                continue
            df = pd.DataFrame(list(records))
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return self._merge_frames(frames)

    @staticmethod
    def _merge_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(frames, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        return df.sort_index()
