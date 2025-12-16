"""FAISS Embedding Builder - Builds and manages vector embeddings for RAG.

This module provides embedding generation and FAISS index management:
- Generates embeddings using sentence-transformers or AWS Titan
- Builds and saves FAISS indexes to S3
- Provides similarity search for RAG retrieval

All vectors and metadata are stored in AWS S3:
- Bucket: rag-bot-storage
- Prefix: spy-futures-bot/vectors/
"""
import hashlib
import io
import json
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# Import S3 storage
from .s3_storage import S3Storage, get_s3_storage, S3StorageError

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available - install with: pip install sentence-transformers")


class EmbeddingBuilder:
    """Builds and manages FAISS vector embeddings for RAG retrieval.
    
    All index data is stored in S3:
    - spy-futures-bot/vectors/index.faiss
    - spy-futures-bot/vectors/metadata.pkl
    
    Supports two embedding backends:
    1. sentence-transformers (local, fast, free)
    2. AWS Titan embeddings (via Bedrock, higher quality)
    """
    
    # Default embedding model - all-MiniLM-L6-v2 is fast and effective
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        bucket_name: str = "rag-bot-storage-897729113303",
        prefix: str = "spy-futures-bot/",
        model_name: str = DEFAULT_MODEL,
        use_bedrock: bool = False,
        bedrock_client: Optional[Any] = None,
    ):
        """Initialize the embedding builder with S3 backend.
        
        Args:
            bucket_name: S3 bucket name
            prefix: S3 key prefix
            model_name: Sentence transformer model name
            use_bedrock: Use AWS Titan embeddings instead of local model
            bedrock_client: Optional boto3 Bedrock client
        """
        # Initialize S3 storage
        try:
            self.s3 = S3Storage(bucket_name=bucket_name, prefix=prefix)
        except S3StorageError as e:
            logger.warning(f"S3 storage unavailable ({e}) - operating in local-only mode")
            self.s3 = None
        
        self.model_name = model_name
        self.use_bedrock = use_bedrock
        self.bedrock_client = bedrock_client
        
        # Initialize embedding model
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        self._fallback_mode = False
        
        if not use_bedrock:
            self._init_sentence_transformer()
        else:
            self.embedding_dim = 1536  # Titan embedding dimension
        
        # FAISS index and metadata
        self.index: Optional[Any] = None
        self.doc_metadata: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self._local_embeddings: Optional[np.ndarray] = None
        self._logged_empty_index = False
        
        # Try to load existing index from S3
        self._load_index()
        
        logger.info(f"EmbeddingBuilder initialized with S3 (dim={self.embedding_dim}, backend={'bedrock' if use_bedrock else 'local'})")
    
    def _init_sentence_transformer(self) -> None:
        """Initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self._enable_fallback("sentence-transformers not available")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence transformer: {self.model_name} (dim={self.embedding_dim})")
        except Exception as e:
            self._enable_fallback(f"Failed to load sentence transformer: {e}")

    def _enable_fallback(self, reason: str) -> None:
        """Enable lightweight hash-based embeddings when transformers are unavailable."""
        self._fallback_mode = True
        if self.embedding_dim <= 0 or not self.embedding_dim:
            self.embedding_dim = 256
        logger.warning(f"Using fallback embedding mode ({self.embedding_dim} dims) - {reason}")
        self.model = None

    def has_index(self) -> bool:
        """Return True if FAISS index has vectors."""
        return self.index is not None and getattr(self.index, "ntotal", 0) > 0

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _store_local_embeddings(
        self,
        embeddings: np.ndarray,
        replace: bool = False,
    ) -> None:
        """Keep in-memory embeddings for CPU fallback search."""
        normalized = self._normalize_vectors(embeddings.astype(np.float32))
        if replace or self._local_embeddings is None:
            self._local_embeddings = normalized
        else:
            self._local_embeddings = np.vstack([self._local_embeddings, normalized])

    def _ensure_local_embeddings(self) -> bool:
        """Ensure local embeddings exist for fallback search."""
        if self._local_embeddings is not None:
            expected = len(self.doc_texts)
            if expected == self._local_embeddings.shape[0]:
                return True
        if not self.doc_texts:
            return False
        batch = self.embed_batch(self.doc_texts)
        if batch is None:
            return False
        self._store_local_embeddings(batch, replace=True)
        return True

    def _log_empty_index_if_needed(self) -> None:
        if not self._logged_empty_index:
            logger.error("FAISS index is empty - falling back to CPU similarity search")
            self._logged_empty_index = True

    def _fallback_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        filter_type: Optional[str],
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """CPU-based similarity search used when FAISS is unavailable."""
        if not self.doc_texts:
            return []
        if not self._ensure_local_embeddings():
            return []
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []
        query_vec = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm
        sims = np.dot(self._local_embeddings, query_vec)
        if sims.size == 0:
            return []
        ranked_idx = np.argsort(sims)[::-1]
        results: List[Tuple[str, str, float, Dict[str, Any]]] = []
        for idx in ranked_idx:
            score = float((sims[idx] + 1.0) / 2.0)  # map cosine (-1,1) -> (0,1)
            if score < min_score:
                continue
            metadata = self.doc_metadata[idx]
            if filter_type and metadata.get("type") != filter_type:
                continue
            results.append((self.doc_ids[idx], self.doc_texts[idx], score, metadata))
            if len(results) >= top_k:
                break
        return results
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding or None on error
        """
        if self.use_bedrock:
            return self._embed_with_bedrock(text)
        if self._fallback_mode or self.model is None:
            return self._embed_with_fallback(text)
        return self._embed_with_transformer(text)
    
    def _embed_with_transformer(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using sentence transformer.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding
        """
        if self.model is None:
            logger.error("Sentence transformer not initialized")
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _embed_with_bedrock(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using AWS Titan.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding
        """
        if self.bedrock_client is None:
            logger.error("Bedrock client not initialized")
            return None
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            result = json.loads(response["body"].read())
            embedding = np.array(result["embedding"], dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate Bedrock embedding: {e}")
            return None

    def _embed_with_fallback(self, text: str) -> Optional[np.ndarray]:
        """Generate deterministic hashing-based embedding when transformers are unavailable."""
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        if not text:
            return vector
        tokens = text.lower().split()
        for token in tokens:
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = token_hash % self.embedding_dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector
    
    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings (N x dim)
        """
        if not texts:
            return None
        
        if self.use_bedrock:
            # Bedrock doesn't support batch, so embed one by one
            embeddings = []
            for text in texts:
                emb = self._embed_with_bedrock(text)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    # Use zero vector as fallback
                    embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            return np.vstack(embeddings)
        if self._fallback_mode or self.model is None:
            vectors = [self._embed_with_fallback(text) for text in texts]
            return np.vstack(vectors)
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return None
    
    def build_index(
        self,
        documents: List[Tuple[str, str, Dict[str, Any]]],
        save: bool = True,
    ) -> bool:
        """Build FAISS index from documents.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            save: Whether to save the index to disk
            
        Returns:
            True if successful
        """
        if not documents:
            logger.warning("No documents to index")
            return False
        
        logger.info(f"Building FAISS index with {len(documents)} documents")
        
        # Extract texts and metadata
        self.doc_ids = [doc[0] for doc in documents]
        self.doc_texts = [doc[1] for doc in documents]
        self.doc_metadata = [doc[2] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embed_batch(self.doc_texts)
        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return False
        embeddings = embeddings.astype(np.float32)
        self._store_local_embeddings(embeddings, replace=True)
        self._logged_empty_index = False
        
        if FAISS_AVAILABLE:
            # Create FAISS index - using L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add vectors to index
            self.index.add(embeddings)
            
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            
            if save:
                self._save_index()
        else:
            self.index = None
            logger.warning("FAISS not available - using CPU fallback search only")
        
        return True
    
    def add_documents(
        self,
        documents: List[Tuple[str, str, Dict[str, Any]]],
        save: bool = True,
    ) -> int:
        """Add new documents to existing index.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            save: Whether to save the index after adding
            
        Returns:
            Number of documents added
        """
        if self.index is None and not self.doc_ids:
            # No existing data, build from scratch
            if self.build_index(documents, save=save):
                return len(documents)
            return 0
        
        # Filter out existing doc_ids
        existing_ids = set(self.doc_ids)
        new_docs = [(did, txt, meta) for did, txt, meta in documents if did not in existing_ids]
        
        if not new_docs:
            logger.info("No new documents to add")
            return 0
        
        # Generate embeddings for new docs
        new_ids = [doc[0] for doc in new_docs]
        new_texts = [doc[1] for doc in new_docs]
        new_metadata = [doc[2] for doc in new_docs]
        
        embeddings = self.embed_batch(new_texts)
        if embeddings is None:
            return 0
        embeddings = embeddings.astype(np.float32)
        
        # Add to index if available, always update local cache
        if self.index is not None:
            self.index.add(embeddings)
        self._store_local_embeddings(embeddings)
        
        # Update metadata
        self.doc_ids.extend(new_ids)
        self.doc_texts.extend(new_texts)
        self.doc_metadata.extend(new_metadata)
        
        logger.info(f"Added {len(new_docs)} documents to index (total: {self.index.ntotal})")
        
        if save:
            self._save_index()
        
        return len(new_docs)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_type: Optional[str] = None,
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Search for similar documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            filter_type: Optional filter by document type
            
        Returns:
            List of (doc_id, content, score, metadata) tuples
        """
        if not self.has_index():
            results = self._fallback_search(query, top_k, min_score, filter_type)
            if not results:
                self._log_empty_index_if_needed()
            return results
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        if query_embedding is None:
            return []
        
        # Search
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
        
        # Convert L2 distances to similarity scores (inverse)
        # Lower distance = higher similarity
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            
            # Convert distance to similarity score (0-1 range)
            # Using exponential decay: score = exp(-distance)
            score = float(np.exp(-dist / 10))  # Normalize by 10 for reasonable range
            
            if score < min_score:
                continue
            
            metadata = self.doc_metadata[idx]
            
            # Apply type filter
            if filter_type and metadata.get("type") != filter_type:
                continue
            
            results.append((
                self.doc_ids[idx],
                self.doc_texts[idx],
                score,
                metadata,
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_with_context(
        self,
        query: str,
        market_context: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """Search with additional market context for better retrieval.
        
        Args:
            query: Base query text
            market_context: Additional context (trend, volatility, etc.)
            top_k: Number of results
            
        Returns:
            List of search results
        """
        # Enhance query with context
        enhanced_query = f"{query}\n"
        enhanced_query += f"Market trend: {market_context.get('trend', 'unknown')}\n"
        enhanced_query += f"Volatility: {market_context.get('volatility_regime', 'unknown')}\n"
        enhanced_query += f"Time of day: {market_context.get('time_of_day', 'unknown')}\n"
        
        if market_context.get("near_pdh"):
            enhanced_query += "Price is near previous day high\n"
        if market_context.get("near_pdl"):
            enhanced_query += "Price is near previous day low\n"
        
        return self.search(enhanced_query, top_k=top_k)
    
    def _save_index(self) -> bool:
        """Save FAISS index and metadata to S3.
        
        Returns:
            True if successful
        """
        if self.index is None:
            return False
        
        if self.s3 is None:
            return True
        
        try:
            # Serialize FAISS index to bytes
            index_buffer = io.BytesIO()
            faiss.write_index(self.index, faiss.PyCallbackIOWriter(index_buffer.write))
            index_bytes = index_buffer.getvalue()
            
            # Upload index to S3
            self.s3.save_binary_to_s3("vectors/index.faiss", index_bytes)
            
            # Serialize and save metadata
            metadata = {
                "doc_ids": self.doc_ids,
                "doc_texts": self.doc_texts,
                "doc_metadata": self.doc_metadata,
                "embedding_dim": self.embedding_dim,
                "model_name": self.model_name,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            metadata_bytes = pickle.dumps(metadata)
            self.s3.save_binary_to_s3("vectors/metadata.pkl", metadata_bytes)
            
            logger.info(f"Saved FAISS index to S3 ({self.index.ntotal} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index to S3: {e}")
            return False
    
    def _load_index(self) -> bool:
        """Load FAISS index and metadata from S3.
        
        Returns:
            True if successful
        """
        if not FAISS_AVAILABLE or self.s3 is None:
            return False
        
        try:
            # Download index from S3
            index_bytes = self.s3.read_binary_from_s3("vectors/index.faiss")
            if index_bytes is None:
                logger.info("No existing index found in S3")
                return False
            
            # Load FAISS index from bytes
            index_buffer = io.BytesIO(index_bytes)
            self.index = faiss.read_index(faiss.PyCallbackIOReader(index_buffer.read, len(index_bytes)))
            
            # Download and load metadata
            metadata_bytes = self.s3.read_binary_from_s3("vectors/metadata.pkl")
            if metadata_bytes is None:
                logger.warning("Index found but metadata missing in S3")
                return False
            
            metadata = pickle.loads(metadata_bytes)
            
            self.doc_ids = metadata["doc_ids"]
            self.doc_texts = metadata["doc_texts"]
            self.doc_metadata = metadata["doc_metadata"]
            self._local_embeddings = None
            self._logged_empty_index = False
            
            logger.info(f"Loaded FAISS index from S3 with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index from S3: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.doc_texts:
            return {"status": "not_initialized", "documents": 0}
        
        # Count by type
        type_counts = {}
        for meta in self.doc_metadata:
            doc_type = meta.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "status": "ready",
            "engine": "faiss" if self.has_index() else "cpu",
            "documents": self.index.ntotal if self.has_index() else len(self.doc_texts),
            "embedding_dim": self.embedding_dim,
            "model": self.model_name if not self.use_bedrock else "bedrock-titan",
            "type_counts": type_counts,
        }


def create_embedding_builder(
    bucket_name: str = "rag-bot-storage-897729113303",
    prefix: str = "spy-futures-bot/",
    use_bedrock: bool = False,
) -> EmbeddingBuilder:
    """Factory function to create an EmbeddingBuilder with S3 backend.
    
    Args:
        bucket_name: S3 bucket name
        prefix: S3 key prefix
        use_bedrock: Use AWS Titan embeddings
        
    Returns:
        EmbeddingBuilder instance
    """
    return EmbeddingBuilder(bucket_name=bucket_name, prefix=prefix, use_bedrock=use_bedrock)
