"""FAISS Embedding Builder - Builds and manages vector embeddings for RAG.

This module provides embedding generation and FAISS index management:
- Generates embeddings using sentence-transformers or AWS Titan
- Builds and saves FAISS indexes
- Provides similarity search for RAG retrieval
"""
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

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
    
    Supports two embedding backends:
    1. sentence-transformers (local, fast, free)
    2. AWS Titan embeddings (via Bedrock, higher quality)
    """
    
    # Default embedding model - all-MiniLM-L6-v2 is fast and effective
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        vectors_path: str = "rag_data/vectors",
        model_name: str = DEFAULT_MODEL,
        use_bedrock: bool = False,
        bedrock_client: Optional[Any] = None,
    ):
        """Initialize the embedding builder.
        
        Args:
            vectors_path: Path to save/load FAISS indexes
            model_name: Sentence transformer model name
            use_bedrock: Use AWS Titan embeddings instead of local model
            bedrock_client: Optional boto3 Bedrock client
        """
        self.vectors_path = Path(vectors_path)
        self.vectors_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.use_bedrock = use_bedrock
        self.bedrock_client = bedrock_client
        
        # Initialize embedding model
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        
        if not use_bedrock:
            self._init_sentence_transformer()
        else:
            self.embedding_dim = 1536  # Titan embedding dimension
        
        # FAISS index and metadata
        self.index: Optional[Any] = None
        self.doc_metadata: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        
        # Try to load existing index
        self._load_index()
        
        logger.info(f"EmbeddingBuilder initialized (dim={self.embedding_dim}, backend={'bedrock' if use_bedrock else 'local'})")
    
    def _init_sentence_transformer(self) -> None:
        """Initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence transformer: {self.model_name} (dim={self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding or None on error
        """
        if self.use_bedrock:
            return self._embed_with_bedrock(text)
        else:
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
        else:
            if self.model is None:
                logger.error("Model not initialized")
                return None
            
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
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return False
        
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
        
        # Create FAISS index - using L2 distance
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
        if save:
            self._save_index()
        
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
        if self.index is None:
            # No existing index, build from scratch
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
        
        # Add to index
        self.index.add(embeddings)
        
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
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty or not initialized")
            return []
        
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
        """Save FAISS index and metadata to disk.
        
        Returns:
            True if successful
        """
        if self.index is None:
            return False
        
        try:
            # Save FAISS index
            index_path = self.vectors_path / "index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.vectors_path / "metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump({
                    "doc_ids": self.doc_ids,
                    "doc_texts": self.doc_texts,
                    "doc_metadata": self.doc_metadata,
                    "embedding_dim": self.embedding_dim,
                    "model_name": self.model_name,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }, f)
            
            logger.info(f"Saved FAISS index to {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def _load_index(self) -> bool:
        """Load FAISS index and metadata from disk.
        
        Returns:
            True if successful
        """
        if not FAISS_AVAILABLE:
            return False
        
        index_path = self.vectors_path / "index.faiss"
        metadata_path = self.vectors_path / "metadata.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.info("No existing index found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            self.doc_ids = metadata["doc_ids"]
            self.doc_texts = metadata["doc_texts"]
            self.doc_metadata = metadata["doc_metadata"]
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.
        
        Returns:
            Statistics dictionary
        """
        if self.index is None:
            return {"status": "not_initialized", "documents": 0}
        
        # Count by type
        type_counts = {}
        for meta in self.doc_metadata:
            doc_type = meta.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "status": "ready",
            "documents": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "model": self.model_name if not self.use_bedrock else "bedrock-titan",
            "type_counts": type_counts,
        }


def create_embedding_builder(
    vectors_path: str = "rag_data/vectors",
    use_bedrock: bool = False,
) -> EmbeddingBuilder:
    """Factory function to create an EmbeddingBuilder.
    
    Args:
        vectors_path: Path to vectors folder
        use_bedrock: Use AWS Titan embeddings
        
    Returns:
        EmbeddingBuilder instance
    """
    return EmbeddingBuilder(vectors_path=vectors_path, use_bedrock=use_bedrock)
