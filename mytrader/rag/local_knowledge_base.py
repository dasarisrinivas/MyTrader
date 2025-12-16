"""Local knowledge base built on LocalVectorStore."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from ..learning.trade_learning import TradeLearningPayload
from .local_vector_store import LocalVectorStore


class LocalKnowledgeBase:
    """Provides similarity search + summary stats without OpenSearch."""

    def __init__(
        self,
        store_path: str = "rag_data/local_kb/local_kb.sqlite",
        embedding_dim: int = 256,
    ) -> None:
        self.store = LocalVectorStore(store_path, dimension=embedding_dim)

    # ---------------------------------------------------------------- ingestion
    def bootstrap_from_outcomes(
        self,
        outcomes_dir: str,
        limit: Optional[int] = 750,
    ) -> int:
        """Ingest existing trade outcomes from disk."""
        root = Path(outcomes_dir)
        if not root.exists():
            return 0
        ingested = 0
        files: Iterable[Path] = sorted(root.rglob("*.json"))
        for path in files:
            if limit and ingested >= limit:
                break
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if self.store.document_exists(data["trade_cycle_id"]):
                    continue
                self.record_trade(data)
                ingested += 1
            except Exception as exc:
                logger.debug("Skipping %s: %s", path, exc)
        if ingested:
            logger.info("Seeded %d trades into local knowledge base", ingested)
        return ingested

    def record_trade(self, payload: TradeLearningPayload | Dict[str, Any]) -> None:
        """Persist a trade context into the local vector store."""
        if isinstance(payload, TradeLearningPayload):
            data = payload.to_dict()
        else:
            data = payload
        doc_id = data["trade_cycle_id"]
        text = self._format_trade_text(data)
        metadata = {
            "side": data.get("side"),
            "regime": data.get("regime"),
            "volatility": data.get("volatility"),
            "outcome": data.get("outcome"),
            "pnl": data.get("pnl", 0.0),
            "confidence": data.get("signal_confidence"),
        }
        self.store.upsert_document(doc_id, text, metadata)

    # ---------------------------------------------------------------- retrieval
    def query(self, context: Dict[str, Any], top_k: int = 6) -> Dict[str, Any]:
        """Return AWS-like KB signal derived from local history."""
        text = self._format_query_text(context)
        matches = self.store.similarity_search(
            text,
            top_k=top_k,
            metadata_filter={
                "side": context.get("action"),
                "regime": context.get("trend"),
            },
        )
        if not matches:
            return {}
        wins = 0
        total_pnl = 0.0
        for _, _, _, metadata in matches:
            if metadata.get("outcome") == "WIN":
                wins += 1
            total_pnl += float(metadata.get("pnl", 0.0))
        win_rate = wins / len(matches)
        avg_pnl = total_pnl / len(matches)
        confidence_adjustment = self._calc_adjustment(win_rate, len(matches))
        reasoning = (
            f"{len(matches)} similar trades ({wins} wins). "
            f"Win rate {win_rate:.0%}, avg pnl {avg_pnl:.0f}."
        )
        return {
            "confidence_adjustment": confidence_adjustment,
            "similar_patterns": len(matches),
            "historical_win_rate": win_rate,
            "reasoning": reasoning,
        }

    # ---------------------------------------------------------------- helpers
    def _format_trade_text(self, data: Dict[str, Any]) -> str:
        parts = [
            f"side={data.get('side')}",
            f"regime={data.get('regime')}",
            f"volatility={data.get('volatility')}",
            f"signal={data.get('signal_type')}",
            f"confidence={data.get('signal_confidence', 0)}",
        ]
        features = data.get("features") or {}
        if features:
            for key in ("rsi", "macd_hist", "atr", "ema_9", "ema_20"):
                if key in features:
                    parts.append(f"{key}={features[key]}")
        return " ".join(str(p) for p in parts if p)

    def _format_query_text(self, context: Dict[str, Any]) -> str:
        parts = [
            f"side={context.get('action')}",
            f"regime={context.get('trend')}",
            f"volatility={context.get('volatility')}",
            f"confidence={context.get('confidence', 0)}",
        ]
        return " ".join(str(p) for p in parts if p)

    def _calc_adjustment(self, win_rate: float, count: int) -> float:
        if count < 3:
            return 0.0
        if win_rate >= 0.65:
            return 0.12
        if win_rate >= 0.55:
            return 0.05
        if win_rate <= 0.35:
            return -0.15
        if win_rate <= 0.45:
            return -0.05
        return 0.0
