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
    from botocore.exceptions import ClientError, EndpointConnectionError, BotoCoreError
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


class RAGEngineError(Exception):
    """Base exception for RAG engine errors."""
    pass


class EmbeddingError(RAGEngineError):
    """Exception raised when embedding generation fails."""
    pass


class RetrievalError(RAGEngineError):
    """Exception raised when document retrieval fails."""
    pass


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
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial delay in seconds
        self.retry_backoff = 2.0  # Exponential backoff multiplier
        
        # Latency tracking
        self.embedding_latencies: List[float] = []
        self.max_latency_samples = 100
        
        # Error tracking
        self.error_count = 0
        self.last_error_time: Optional[float] = None
        
        # Initialize Bedrock runtime client for embeddings
        try:
            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name
            )
            logger.info(f"Initialized Bedrock runtime client for region {region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock runtime client: {e}")
            raise RAGEngineError(f"Bedrock client initialization failed: {e}")
        
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
                try:
                    self._load_index()
                    logger.info(f"✅ Loaded RAG index with {len(self.documents)} documents")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}")
                    logger.warning("Initializing new empty index")
                    self._clear_index()
            else:
                logger.info("Initialized new RAG index")
        else:
            logger.info("Initialized new RAG index (no persistence)")
    
    def _get_embedding_with_retry(self, text: str, retry_count: int = 0) -> np.ndarray:
        """Get embedding with retry logic and latency tracking.
        
        Args:
            text: Input text to embed
            retry_count: Current retry attempt
            
        Returns:
            Normalized embedding vector
            
        Raises:
            EmbeddingError: If embedding fails after all retries
        """
        start_time = time.time()
        
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
            
            # Track latency
            latency = time.time() - start_time
            self._record_latency(latency)
            
            if latency > 2.0:  # Warn if embedding takes more than 2 seconds
                logger.warning(f"Slow embedding generation: {latency:.2f}s for {len(text)} chars")
            
            return embedding
            
        except (ClientError, EndpointConnectionError, BotoCoreError) as e:
            self.error_count += 1
            self.last_error_time = time.time()
            
            error_msg = str(e)
            logger.error(f"AWS Bedrock embedding error (attempt {retry_count + 1}/{self.max_retries}): {error_msg}")
            
            # Retry with exponential backoff
            if retry_count < self.max_retries:
                delay = self.retry_delay * (self.retry_backoff ** retry_count)
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
                return self._get_embedding_with_retry(text, retry_count + 1)
            else:
                raise EmbeddingError(f"Failed to generate embedding after {self.max_retries} attempts: {error_msg}")
                
        except Exception as e:
            self.error_count += 1
            self.last_error_time = time.time()
            logger.error(f"Unexpected error generating embedding: {e}")
            raise EmbeddingError(f"Unexpected embedding error: {e}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using AWS Titan Embeddings.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        return self._get_embedding_with_retry(text, retry_count=0)
    
    def _record_latency(self, latency: float) -> None:
        """Record embedding latency for monitoring.
        
        Args:
            latency: Latency in seconds
        """
        self.embedding_latencies.append(latency)
        
        # Keep only recent samples
        if len(self.embedding_latencies) > self.max_latency_samples:
            self.embedding_latencies = self.embedding_latencies[-self.max_latency_samples:]
    
    def get_avg_latency(self) -> float:
        """Get average embedding latency.
        
        Returns:
            Average latency in seconds, or 0 if no samples
        """
        if not self.embedding_latencies:
            return 0.0
        return sum(self.embedding_latencies) / len(self.embedding_latencies)
    
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
    ) -> Dict[str, any]:
        """Ingest documents into the vector store with enhanced error handling.
        
        Args:
            documents: List of document texts to ingest
            clear_existing: Clear existing documents before ingestion
            batch_size: Number of documents to embed at once
            
        Returns:
            Dictionary with ingestion statistics and errors
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return {
                "success": False,
                "num_documents": 0,
                "num_errors": 0,
                "message": "No documents provided"
            }
        
        logger.info(f"🔄 Ingesting {len(documents)} documents into RAG index...")
        
        if clear_existing:
            self._clear_index()
            logger.info("✅ Cleared existing index")
        
        # Validate documents
        valid_documents = []
        invalid_count = 0
        
        for i, doc in enumerate(documents):
            if not doc or not isinstance(doc, str):
                logger.warning(f"Skipping invalid document at index {i}: {type(doc)}")
                invalid_count += 1
                continue
            
            # Trim excessive whitespace
            doc = doc.strip()
            if len(doc) < 10:  # Skip very short documents
                logger.warning(f"Skipping very short document at index {i}: {len(doc)} chars")
                invalid_count += 1
                continue
            
            valid_documents.append(doc)
        
        if invalid_count > 0:
            logger.warning(f"⚠️  Skipped {invalid_count} invalid documents")
        
        if not valid_documents:
            logger.error("No valid documents to ingest")
            return {
                "success": False,
                "num_documents": 0,
                "num_errors": invalid_count,
                "message": "No valid documents"
            }
        
        # Generate embeddings with error tracking
        start_time = time.time()
        embeddings = []
        embedding_errors = 0
        
        try:
            for i in range(0, len(valid_documents), batch_size):
                batch = valid_documents[i:i + batch_size]
                
                for doc in batch:
                    try:
                        embedding = self._get_embedding(doc)
                        embeddings.append(embedding)
                        
                        # Rate limiting: small delay between requests
                        time.sleep(0.1)
                        
                    except EmbeddingError as e:
                        logger.error(f"Failed to embed document (length {len(doc)}): {e}")
                        embedding_errors += 1
                        # Use zero vector as fallback
                        embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                
                # Progress logging
                progress = min(i + batch_size, len(valid_documents))
                logger.info(f"📊 Progress: {progress}/{len(valid_documents)} documents embedded")
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            elapsed = time.time() - start_time
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store documents
            self.documents.extend(valid_documents)
            
            # Clear cache after ingestion
            self.query_cache.clear()
            
            logger.info(
                f"✅ Successfully ingested {len(valid_documents)} documents "
                f"(total: {len(self.documents)}) in {elapsed:.2f}s"
            )
            
            if embedding_errors > 0:
                logger.warning(f"⚠️  Encountered {embedding_errors} embedding errors (using fallback vectors)")
            
            # Persist index
            if self.vector_store_path:
                try:
                    self._save_index()
                except Exception as e:
                    logger.error(f"Failed to persist index: {e}")
                    # Don't fail the entire operation
            
            return {
                "success": True,
                "num_documents": len(valid_documents),
                "num_errors": invalid_count + embedding_errors,
                "total_documents": len(self.documents),
                "elapsed_seconds": elapsed,
                "message": f"Ingested {len(valid_documents)} documents successfully"
            }
            
        except Exception as e:
            logger.error(f"Fatal error during document ingestion: {e}")
            return {
                "success": False,
                "num_documents": len(embeddings),
                "num_errors": invalid_count + embedding_errors + 1,
                "message": f"Ingestion failed: {str(e)}"
            }
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Retrieve relevant documents for a query with enhanced error handling.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (document, score) tuples, sorted by relevance
            
        Raises:
            RetrievalError: If retrieval fails critically
        """
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query type: {type(query)}")
            return []
        
        query = query.strip()
        if not query:
            logger.warning("Empty query provided")
            return []
        
        if len(self.documents) == 0:
            logger.warning("No documents in index for retrieval")
            return []
        
        # Check cache
        if self.cache_enabled:
            query_hash = self._hash_query(query)
            cached_result = self._get_cached_query(query_hash)
            
            if cached_result is not None:
                logger.debug(f"✅ Cache hit for query: {query[:50]}...")
                return cached_result[:top_k]
        
        # Generate query embedding with error handling
        try:
            query_embedding = self._get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)
            
        except EmbeddingError as e:
            logger.error(f"Failed to generate query embedding: {e}")
            # Return empty results instead of crashing
            return []
        except Exception as e:
            logger.error(f"Unexpected error generating query embedding: {e}")
            return []
        
        # Search FAISS index
        try:
            # Search for top_k * 2 to allow filtering by score
            search_k = min(top_k * 2, len(self.documents))
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Build results with validation
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Validate index
                if idx < 0 or idx >= len(self.documents):
                    logger.warning(f"Invalid document index returned: {idx}")
                    continue
                
                # Filter by score threshold
                if score >= score_threshold:
                    results.append((self.documents[idx], float(score)))
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k
            results = results[:top_k]
            
            # Cache results
            if self.cache_enabled and results:
                self._cache_query(query_hash, results)
            
            logger.info(f"📚 Retrieved {len(results)}/{top_k} documents (threshold={score_threshold})")
            
            if results:
                avg_score = sum(s for _, s in results) / len(results)
                logger.debug(f"   Average relevance score: {avg_score:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during FAISS retrieval: {e}")
            raise RetrievalError(f"Document retrieval failed: {e}")
    
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
        """Get RAG engine statistics with enhanced monitoring metrics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        avg_latency = self.get_avg_latency()
        
        return {
            "num_documents": len(self.documents),
            "embedding_dimension": self.dimension,
            "cache_size": len(self.query_cache),
            "cache_enabled": self.cache_enabled,
            "vector_store_path": str(self.vector_store_path) if self.vector_store_path else None,
            "embedding_model": self.embedding_model_id,
            "llm_model": self.bedrock_client.model_id,
            "avg_embedding_latency_ms": round(avg_latency * 1000, 2),
            "error_count": self.error_count,
            "last_error_time": self.last_error_time,
            "health_status": self._get_health_status()
        }
    
    def _get_health_status(self) -> str:
        """Determine health status of RAG engine.
        
        Returns:
            Health status string: "healthy", "degraded", or "unhealthy"
        """
        # Check if we have documents
        if len(self.documents) == 0:
            return "unhealthy"  # No knowledge base
        
        # Check recent error rate
        if self.error_count > 0 and self.last_error_time:
            time_since_error = time.time() - self.last_error_time
            if time_since_error < 60:  # Error within last minute
                return "degraded"
        
        # Check latency
        avg_latency = self.get_avg_latency()
        if avg_latency > 3.0:  # Slow embeddings
            return "degraded"
        
        return "healthy"
