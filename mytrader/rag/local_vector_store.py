"""Lightweight SQLite-backed vector store for local knowledge base."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger


class LocalVectorStore:
    """Stores embeddings + metadata in a SQLite database."""

    def __init__(
        self,
        db_path: str = "rag_data/local_kb/local_kb.sqlite",
        dimension: int = 256,
    ) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.conn = sqlite3.connect(
            str(self.path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at)"
        )
        self.conn.commit()
        logger.info(f"Local vector store ready at {self.path}")

    # ------------------------------------------------------------------ utils
    def _embed_text(self, text: str) -> np.ndarray:
        """Create a deterministic hashing-based embedding."""
        vector = np.zeros(self.dimension, dtype=np.float32)
        if not text:
            return vector
        tokens = text.lower().split()
        for token in tokens:
            token_hash = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(token_hash[:4], "big") % self.dimension
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector

    def _serialize_vec(self, vec: np.ndarray) -> bytes:
        vec = vec.astype(np.float32)
        return vec.tobytes()

    def _deserialize_vec(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    # ----------------------------------------------------------------- storage
    def document_exists(self, doc_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM documents WHERE doc_id=? LIMIT 1", (doc_id,))
        return cur.fetchone() is not None

    def upsert_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        embedding = self._embed_text(text)
        metadata_json = json.dumps(metadata or {})
        payload = (
            doc_id,
            text,
            metadata_json,
            self._serialize_vec(embedding),
            time.time(),
        )
        self.conn.execute(
            """
            INSERT INTO documents (doc_id, text, metadata, embedding, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                text=excluded.text,
                metadata=excluded.metadata,
                embedding=excluded.embedding,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def bulk_upsert(self, documents: Iterable[Tuple[str, str, Dict[str, Any]]]) -> None:
        for doc_id, text, metadata in documents:
            self.upsert_document(doc_id, text, metadata)

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM documents")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    # ---------------------------------------------------------------- retrieval
    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Return (doc_id, text, score, metadata) tuples sorted by similarity."""
        if not query_text:
            return []
        query_vec = self._embed_text(query_text)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        results: List[Tuple[str, str, float, Dict[str, Any]]] = []
        cur = self.conn.execute("SELECT doc_id, text, metadata, embedding FROM documents")
        for doc_id, text, metadata_json, blob in cur.fetchall():
            metadata = json.loads(metadata_json) if metadata_json else {}
            if metadata_filter:
                mismatch = False
                for key, expected in metadata_filter.items():
                    if expected is None:
                        continue
                    if metadata.get(key) != expected:
                        mismatch = True
                        break
                if mismatch:
                    continue
            vec = self._deserialize_vec(blob)
            doc_norm = np.linalg.norm(vec)
            if doc_norm == 0:
                continue
            cosine = float(np.dot(query_vec, vec) / (query_norm * doc_norm))
            score = (cosine + 1.0) / 2.0  # normalize to 0-1
            results.append((doc_id, text, score, metadata))

        results.sort(key=lambda item: item[2], reverse=True)
        return results[:top_k]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
