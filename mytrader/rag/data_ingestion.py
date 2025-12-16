"""External data ingestion helpers for RAG knowledge base."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .rag_storage_manager import RAGStorageManager, get_rag_storage


@dataclass
class IngestionResult:
    """Summary of ingestion run."""

    news_docs: int = 0
    macro_docs: int = 0
    sentiment_docs: int = 0

    @property
    def total(self) -> int:
        return self.news_docs + self.macro_docs + self.sentiment_docs

    def to_dict(self) -> Dict[str, int]:
        return {
            "news_docs": self.news_docs,
            "macro_docs": self.macro_docs,
            "sentiment_docs": self.sentiment_docs,
            "total": self.total,
        }


class RAGDataIngestionPipeline:
    """Loads curated news, macro, and sentiment data into RAG storage."""

    def __init__(
        self,
        storage: Optional[RAGStorageManager] = None,
        data_dir: str = "data",
    ):
        self.storage = storage or get_rag_storage()
        self.data_dir = Path(data_dir)
        self.news_dir = self.data_dir / "news"
        self.macro_dir = self.data_dir / "macro"
        self.sentiment_dir = self.data_dir / "sentiment"
        logger.info(f"RAGDataIngestionPipeline watching {self.data_dir}")

    def ingest(self, date: Optional[str] = None) -> IngestionResult:
        """Run ingestion for all supported sources."""
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = IngestionResult()
        result.news_docs = self._ingest_directory(self.news_dir, "news", date)
        result.macro_docs = self._ingest_directory(self.macro_dir, "macro", date)
        result.sentiment_docs = self._ingest_directory(self.sentiment_dir, "sentiment", date)
        logger.info(f"RAG ingestion complete: {result.to_dict()}")
        return result

    def _ingest_directory(self, path: Path, category: str, date: str) -> int:
        if not path.exists():
            return 0
        count = 0
        for file in sorted(path.glob("*.json")):
            try:
                payload = json.loads(file.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed {category} file: {file}")
                continue
            doc_id = f"{date}_{file.stem}.json"
            content = json.dumps(
                {
                    "source_file": str(file),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "payload": payload,
                }
            )
            self.storage.save_dynamic_doc(category, doc_id, content)
            count += 1
        return count


__all__ = ["RAGDataIngestionPipeline", "IngestionResult"]
