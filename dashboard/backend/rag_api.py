"""
RAG API Endpoints for MyTrader Dashboard
Provides Retrieval-Augmented Generation functionality via REST API
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mytrader.config import RAGConfig
from mytrader.llm.bedrock_client import BedrockClient
from mytrader.llm.rag_engine import RAGEngine
from mytrader.utils.logger import logger
from mytrader.utils.settings_loader import load_settings


# Global RAG engine instance
_rag_engine: Optional[RAGEngine] = None


# Pydantic models for API
class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion."""
    documents: List[str] = Field(..., description="List of document texts to ingest")
    clear_existing: bool = Field(False, description="Clear existing documents before ingestion")
    batch_size: int = Field(10, description="Number of documents to process at once", ge=1, le=50)


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    num_documents_ingested: int
    total_documents: int
    message: str


class RAGQueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="User query for RAG generation")
    top_k: int = Field(3, description="Number of documents to retrieve", ge=1, le=10)
    score_threshold: float = Field(0.5, description="Minimum similarity score", ge=0.0, le=1.0)
    include_scores: bool = Field(True, description="Include retrieval scores in response")
    max_tokens: Optional[int] = Field(None, description="Override default max tokens")
    temperature: Optional[float] = Field(None, description="Override default temperature", ge=0.0, le=1.0)


class RAGQueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str
    response: str
    retrieved_documents: List[str]
    retrieval_scores: Optional[List[float]]
    num_documents_retrieved: int
    generation_time_seconds: float
    model_id: str
    timestamp: str
    error: Optional[str] = None


class RetrievalRequest(BaseModel):
    """Request model for document retrieval only."""
    query: str = Field(..., description="Query for document retrieval")
    top_k: int = Field(3, description="Number of documents to retrieve", ge=1, le=10)
    score_threshold: float = Field(0.5, description="Minimum similarity score", ge=0.0, le=1.0)


class RetrievalResponse(BaseModel):
    """Response model for document retrieval."""
    query: str
    documents: List[Dict[str, Any]]  # List of {text: str, score: float}
    num_results: int


class RAGStatsResponse(BaseModel):
    """Response model for RAG statistics."""
    num_documents: int
    embedding_dimension: int
    cache_size: int
    cache_enabled: bool
    vector_store_path: Optional[str]
    embedding_model: str
    llm_model: str
    status: str


class ClearCacheResponse(BaseModel):
    """Response model for cache clearing."""
    success: bool
    message: str


# Create API router
router = APIRouter(prefix="/rag", tags=["RAG"])


def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine instance.
    
    Returns:
        RAG engine instance
        
    Raises:
        HTTPException: If RAG is not configured or initialization fails
    """
    global _rag_engine
    
    if _rag_engine is None:
        try:
            # Load settings
            settings = load_settings()
            
            backend_mode = getattr(settings.rag, "backend", "off")
            if not settings.rag.enabled or backend_mode == "off":
                raise HTTPException(
                    status_code=503,
                    detail="RAG is not enabled. Set rag.enabled=true and choose a backend in config.yaml"
                )
            
            # Create Bedrock client
            bedrock_client = BedrockClient(
                model_id=settings.llm.model_id,
                region_name=settings.llm.region_name,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature
            )
            
            # Create RAG engine
            _rag_engine = RAGEngine(
                bedrock_client=bedrock_client,
                embedding_model_id=settings.rag.embedding_model_id,
                region_name=settings.rag.region_name,
                vector_store_path=settings.rag.vector_store_path,
                dimension=settings.rag.embedding_dimension,
                cache_enabled=settings.rag.cache_enabled,
                cache_ttl_seconds=settings.rag.cache_ttl_seconds
            )
            
            logger.info("RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize RAG engine: {str(e)}"
            )
    
    return _rag_engine


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_documents(request: DocumentIngestionRequest):
    """Ingest documents into the RAG knowledge base.
    
    Args:
        request: Document ingestion request
        
    Returns:
        Ingestion result with statistics
    """
    try:
        rag_engine = get_rag_engine()
        
        # Ingest documents (now returns dict with stats)
        result = rag_engine.ingest_documents(
            documents=request.documents,
            clear_existing=request.clear_existing,
            batch_size=request.batch_size
        )
        
        # Handle the dict response
        if not result.get("success", False):
            logger.warning(f"Document ingestion had issues: {result.get('message')}")
        
        stats = rag_engine.get_stats()
        
        return DocumentIngestionResponse(
            success=result.get("success", False),
            num_documents_ingested=result.get("num_documents", 0),
            total_documents=stats["num_documents"],
            message=result.get("message", "Ingestion completed")
        )
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_model=RAGQueryResponse)
async def ask_with_rag(request: RAGQueryRequest):
    """Ask a question using RAG (retrieval + generation).
    
    Args:
        request: RAG query request
        
    Returns:
        Generated response with retrieved context
    """
    try:
        rag_engine = get_rag_engine()
        
        # Generate with RAG
        result = rag_engine.generate_with_rag(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_scores=request.include_scores
        )
        
        return RAGQueryResponse(**result)
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return RAGQueryResponse(
            query=request.query,
            response="",
            retrieved_documents=[],
            retrieval_scores=None,
            num_documents_retrieved=0,
            generation_time_seconds=0.0,
            model_id="",
            timestamp="",
            error=str(e)
        )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest):
    """Retrieve relevant documents without generation.
    
    Args:
        request: Retrieval request
        
    Returns:
        Retrieved documents with scores
    """
    try:
        rag_engine = get_rag_engine()
        
        # Retrieve documents
        results = rag_engine.retrieve_context(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )
        
        # Format results
        documents = [
            {"text": doc, "score": score}
            for doc, score in results
        ]
        
        return RetrievalResponse(
            query=request.query,
            documents=documents,
            num_results=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Get RAG engine statistics.
    
    Returns:
        RAG statistics including document count, cache size, etc.
    """
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_stats()
        
        return RAGStatsResponse(
            status="active",
            **stats
        )
        
    except HTTPException as e:
        # RAG not enabled
        return RAGStatsResponse(
            num_documents=0,
            embedding_dimension=0,
            cache_size=0,
            cache_enabled=False,
            vector_store_path=None,
            embedding_model="",
            llm_model="",
            status="disabled"
        )
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache", response_model=ClearCacheResponse)
async def clear_query_cache():
    """Clear the RAG query cache.
    
    Returns:
        Success message
    """
    try:
        rag_engine = get_rag_engine()
        rag_engine.clear_cache()
        
        return ClearCacheResponse(
            success=True,
            message="Query cache cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for RAG service.
    
    Returns:
        Health status
    """
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_stats()
        
        return {
            "status": "healthy",
            "rag_enabled": True,
            "documents_loaded": stats["num_documents"] > 0,
            "num_documents": stats["num_documents"]
        }
        
    except HTTPException:
        return {
            "status": "disabled",
            "rag_enabled": False,
            "message": "RAG is not enabled in configuration"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "rag_enabled": True,
            "error": str(e)
        }


# Convenience function to include router in main app
def include_rag_router(app):
    """Include RAG router in FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(router)
    logger.info("RAG API endpoints registered")
