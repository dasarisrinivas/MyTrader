"""Retrieval-Augmented Generation (RAG) Engine for enhanced LLM responses."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ..utils.logger import logger
from .bedrock_client import BedrockClient


class RAGEngine:
    """Retrieval-Augmented Generation engine using AWS Titan Embeddings and FAISS."""
    
    def __init__(
        self,
        bedrock_client: BedrockClient,
        embedding_model_id: str = "amazon.titan-embed-text-v1",
        region_name: str = "us-east-1",
        vector_store_path: Optional[str] = None,
        dimension: int = 1536,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        """Initialize RAG engine.
        
        Args:
            bedrock_client: BedrockClient instance for LLM generation
            embedding_model_id: Bedrock embedding model ID
            region_name: AWS region
            vector_store_path: Path to persist FAISS index (optional)
            dimension: Embedding dimension (1536 for Titan v1)
            cache_enabled: Enable query result caching
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for RAG. Install with: pip install boto3"
            )
        
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for RAG. Install with: pip install faiss-cpu"
            )
        
        self.bedrock_client = bedrock_client
        self.embedding_model_id = embedding_model_id
        self.region_name = region_name
        self.dimension = dimension
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Initialize Bedrock runtime client for embeddings
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )
        
        # Initialize FAISS index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Document storage (maps index ID to document text)
        self.documents: List[str] = []
        
        # Query cache: {query_hash: (results, timestamp)}
        self.query_cache: Dict[str, Tuple[List[Tuple[str, float]], float]] = {}
        
        # Vector store persistence
        self.vector_store_path = Path(vector_store_path) if vector_store_path else None
        
        # Load existing index if available
        if self.vector_store_path:
            index_file = self.vector_store_path.with_suffix(".faiss")
            if index_file.exists():
                self._load_index()
                logger.info(f"Loaded RAG index with {len(self.documents)} documents")
            else:
                logger.info("Initialized new RAG index")
        else:
            logger.info("Initialized new RAG index")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using AWS Titan Embeddings.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Prepare request body
            body = json.dumps({"inputText": text})
            
            # Invoke Titan Embeddings model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.embedding_model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read())
            embedding = np.array(response_body.get("embedding"), dtype=np.float32)
            
            # Normalize for cosine similarity (required for IndexFlatIP)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except ClientError as e:
            logger.error(f"AWS Bedrock embedding error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _batch_embeddings(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            Array of normalized embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                try:
                    embedding = self._get_embedding(text)
                    embeddings.append(embedding)
                    
                    # Rate limiting: small delay between requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to embed text (length {len(text)}): {e}")
                    # Use zero vector as fallback
                    embeddings.append(np.zeros(self.dimension, dtype=np.float32))
            
            # Progress logging
            logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} documents")
        
        return np.array(embeddings, dtype=np.float32)
    
    def ingest_documents(
        self,
        documents: List[str],
        clear_existing: bool = False,
        batch_size: int = 10
    ) -> None:
        """Ingest documents into the vector store.
        
        Args:
            documents: List of document texts to ingest
            clear_existing: Clear existing documents before ingestion
            batch_size: Number of documents to embed at once
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return
        
        logger.info(f"Ingesting {len(documents)} documents into RAG index...")
        
        if clear_existing:
            self._clear_index()
            logger.info("Cleared existing index")
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self._batch_embeddings(documents, batch_size=batch_size)
        elapsed = time.time() - start_time
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents
        self.documents.extend(documents)
        
        # Clear cache after ingestion
        self.query_cache.clear()
        
        logger.info(
            f"Successfully ingested {len(documents)} documents "
            f"(total: {len(self.documents)}) in {elapsed:.2f}s"
        )
        
        # Persist index
        if self.vector_store_path:
            self._save_index()
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if len(self.documents) == 0:
            logger.warning("No documents in index for retrieval")
            return []
        
        # Check cache
        if self.cache_enabled:
            query_hash = self._hash_query(query)
            cached_result = self._get_cached_query(query_hash)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result[:top_k]
        
        # Generate query embedding
        try:
            query_embedding = self._get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        
        # Search FAISS index
        try:
            # Search for top_k * 2 to allow filtering by score
            search_k = min(top_k * 2, len(self.documents))
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Build results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score >= score_threshold:
                    results.append((self.documents[idx], float(score)))
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k
            results = results[:top_k]
            
            # Cache results
            if self.cache_enabled and results:
                self._cache_query(query_hash, results)
            
            logger.info(f"Retrieved {len(results)} documents for query (top_k={top_k})")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def generate_with_rag(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.5,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_scores: bool = False
    ) -> Dict:
        """Generate response using RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of context documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            max_tokens: Override default max tokens
            temperature: Override default temperature
            include_scores: Include retrieval scores in response
            
        Returns:
            Dictionary with generated response and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve context
        logger.info(f"RAG Query: {query[:100]}...")
        retrieved_docs = self.retrieve_context(query, top_k, score_threshold)
        
        if not retrieved_docs:
            logger.warning("No relevant documents retrieved, generating without context")
            retrieved_context = "No relevant context found."
        else:
            # Format retrieved context
            context_parts = []
            for i, (doc, score) in enumerate(retrieved_docs, 1):
                if include_scores:
                    context_parts.append(f"[Document {i}] (relevance: {score:.3f})\n{doc}")
                else:
                    context_parts.append(f"[Document {i}]\n{doc}")
            
            retrieved_context = "\n\n".join(context_parts)
        
        # Step 2: Build augmented prompt
        augmented_prompt = self._build_augmented_prompt(retrieved_context, query)
        
        # Step 3: Generate response using LLM
        try:
            response_text = self.bedrock_client.generate_text(
                prompt=augmented_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generation_time = time.time() - start_time
            
            # Build result
            result = {
                "query": query,
                "response": response_text,
                "retrieved_documents": [doc for doc, _ in retrieved_docs],
                "retrieval_scores": [score for _, score in retrieved_docs] if include_scores else None,
                "num_documents_retrieved": len(retrieved_docs),
                "generation_time_seconds": generation_time,
                "model_id": self.bedrock_client.model_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"RAG generation completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error during RAG generation: {e}")
            return {
                "query": query,
                "response": "",
                "error": str(e),
                "retrieved_documents": [doc for doc, _ in retrieved_docs],
                "num_documents_retrieved": len(retrieved_docs)
            }
    
    def _build_augmented_prompt(self, context: str, query: str) -> str:
        """Build augmented prompt with retrieved context.
        
        Args:
            context: Retrieved context documents
            query: Original user query
            
        Returns:
            Formatted prompt with context
        """
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so and provide your best answer based on your knowledge.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
        
        return prompt
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cached_query(self, query_hash: str) -> Optional[List[Tuple[str, float]]]:
        """Get cached query results if not expired.
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            Cached results or None if not found/expired
        """
        if query_hash not in self.query_cache:
            return None
        
        results, timestamp = self.query_cache[query_hash]
        
        # Check if expired
        if time.time() - timestamp > self.cache_ttl_seconds:
            del self.query_cache[query_hash]
            return None
        
        return results
    
    def _cache_query(self, query_hash: str, results: List[Tuple[str, float]]) -> None:
        """Cache query results.
        
        Args:
            query_hash: Hash of the query
            results: Results to cache
        """
        self.query_cache[query_hash] = (results, time.time())
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        self.query_cache.clear()
        logger.info("Cleared query cache")
    
    def _clear_index(self) -> None:
        """Clear the FAISS index and documents."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.query_cache.clear()
    
    def _save_index(self) -> None:
        """Save FAISS index and documents to disk."""
        if not self.vector_store_path:
            return
        
        try:
            # Create directory if needed
            self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = self.vector_store_path.with_suffix(".faiss")
            faiss.write_index(self.index, str(index_path))
            
            # Save documents
            docs_path = self.vector_store_path.with_suffix(".pkl")
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved RAG index to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _load_index(self) -> None:
        """Load FAISS index and documents from disk."""
        try:
            # Load FAISS index
            index_path = self.vector_store_path.with_suffix(".faiss")
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            
            # Load documents
            docs_path = self.vector_store_path.with_suffix(".pkl")
            if docs_path.exists():
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
            
            logger.info(f"Loaded RAG index from {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._clear_index()
    
    def get_stats(self) -> Dict:
        """Get RAG engine statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "num_documents": len(self.documents),
            "embedding_dimension": self.dimension,
            "cache_size": len(self.query_cache),
            "cache_enabled": self.cache_enabled,
            "vector_store_path": str(self.vector_store_path) if self.vector_store_path else None,
            "embedding_model": self.embedding_model_id,
            "llm_model": self.bedrock_client.model_id
        }
