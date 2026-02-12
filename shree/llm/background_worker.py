"""
Background LLM Worker Thread
Runs LLM/RAG calls asynchronously without blocking the main trading loop.
"""
import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from queue import Queue, Empty

import pandas as pd
from loguru import logger


@dataclass
class LLMRequest:
    """Request for LLM analysis."""
    request_id: str
    features: pd.DataFrame
    signal: Any  # Trading signal
    context: Dict[str, Any]
    timestamp: datetime


@dataclass
class LLMResponse:
    """Response from LLM analysis."""
    request_id: str
    recommendation: Optional[Dict[str, Any]]
    commentary: str
    timestamp: datetime
    processing_time: float
    error: Optional[str] = None


class BackgroundLLMWorker:
    """
    Background worker thread for LLM/RAG calls.
    
    The main trading loop submits requests and reads cached results.
    The worker processes requests asynchronously without blocking.
    """
    
    def __init__(
        self,
        trade_advisor,
        cache_timeout_seconds: int = 300,
        max_queue_size: int = 10,
    ):
        """
        Initialize background LLM worker.
        
        Args:
            trade_advisor: TradeAdvisor or RAGEnhancedTradeAdvisor instance
            cache_timeout_seconds: How long to cache LLM responses
            max_queue_size: Maximum requests in queue
        """
        self.trade_advisor = trade_advisor
        self.cache_timeout_seconds = cache_timeout_seconds
        self.max_queue_size = max_queue_size
        
        # Request queue and response cache
        self.request_queue: Queue = Queue(maxsize=max_queue_size)
        self.response_cache: Dict[str, LLMResponse] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Worker thread
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Statistics
        self.total_requests = 0
        self.total_responses = 0
        self.total_errors = 0
        self.avg_processing_time = 0.0
        
        logger.info("BackgroundLLMWorker initialized")
    
    def start(self):
        """Start the background worker thread."""
        if self.running:
            logger.warning("Worker already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="LLMWorkerThread"
        )
        self.worker_thread.start()
        logger.info("✅ Background LLM worker started")
    
    def stop(self):
        """Stop the background worker thread."""
        if not self.running:
            return
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Background LLM worker stopped")
    
    def submit_request(
        self,
        features: pd.DataFrame,
        signal: Any,
        context: Dict[str, Any]
    ) -> str:
        """
        Submit a request for LLM analysis.
        
        Non-blocking - returns immediately with request ID.
        
        Args:
            features: Market data and indicators
            signal: Trading signal to analyze
            context: Additional context
            
        Returns:
            Request ID for retrieving results
        """
        request_id = f"req_{int(time.time() * 1000)}"
        request = LLMRequest(
            request_id=request_id,
            features=features.copy(),
            signal=signal,
            context=context.copy() if context else {},
            timestamp=datetime.now()
        )
        
        try:
            # Try to add to queue (non-blocking)
            self.request_queue.put_nowait(request)
            self.total_requests += 1
            logger.debug(f"Submitted LLM request: {request_id}")
            return request_id
        except:
            # Queue full - skip this request
            logger.warning("LLM request queue full - skipping request")
            return ""
    
    def get_cached_response(self, request_id: str) -> Optional[LLMResponse]:
        """
        Get cached LLM response if available.
        
        Non-blocking - returns None if not ready or expired.
        
        Args:
            request_id: Request ID from submit_request()
            
        Returns:
            LLMResponse or None
        """
        if not request_id:
            return None
        
        # Check if response exists and is not expired
        if request_id in self.response_cache:
            cache_time = self.cache_timestamps.get(request_id)
            if cache_time:
                age = (datetime.now() - cache_time).total_seconds()
                if age < self.cache_timeout_seconds:
                    return self.response_cache[request_id]
                else:
                    # Expired - remove from cache
                    del self.response_cache[request_id]
                    del self.cache_timestamps[request_id]
        
        return None
    
    def get_latest_commentary(self) -> Optional[str]:
        """
        Get the most recent LLM commentary.
        
        Returns:
            Commentary string or None
        """
        if not self.response_cache:
            return None
        
        # Find most recent response
        latest_response = None
        latest_time = None
        
        for request_id, response in self.response_cache.items():
            if latest_time is None or response.timestamp > latest_time:
                latest_response = response
                latest_time = response.timestamp
        
        if latest_response and latest_response.commentary:
            return latest_response.commentary
        
        return None
    
    def _worker_loop(self):
        """Main worker loop (runs in background thread)."""
        logger.info("LLM worker loop started")
        
        while self.running:
            try:
                # Get request from queue (block with timeout)
                try:
                    request = self.request_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process request
                start_time = time.time()
                response = self._process_request(request)
                processing_time = time.time() - start_time
                
                response.processing_time = processing_time
                
                # Update statistics
                self.total_responses += 1
                self.avg_processing_time = (
                    (self.avg_processing_time * (self.total_responses - 1) + processing_time) /
                    self.total_responses
                )
                
                # Cache response
                self.response_cache[request.request_id] = response
                self.cache_timestamps[request.request_id] = datetime.now()
                
                # Clean up old cache entries (keep last 100)
                if len(self.response_cache) > 100:
                    oldest_key = min(self.cache_timestamps.keys(), 
                                   key=lambda k: self.cache_timestamps[k])
                    del self.response_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]
                
                logger.debug(f"Processed LLM request {request.request_id} in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in LLM worker loop: {e}", exc_info=True)
                self.total_errors += 1
                time.sleep(1.0)  # Back off on error
    
    def _process_request(self, request: LLMRequest) -> LLMResponse:
        """
        Process an LLM request.
        
        Args:
            request: LLM request to process
            
        Returns:
            LLM response
        """
        try:
            # Build trading context
            from ..llm.data_schema import TradingContext
            
            latest = request.features.iloc[-1]
            context = TradingContext(
                symbol=request.context.get('symbol', 'ES'),
                current_price=float(latest.get('close', 0.0)),
                timestamp=latest.name if hasattr(latest, 'name') else pd.Timestamp.now(),
                rsi=float(latest.get('RSI_14', 50.0)),
                macd=float(latest.get('MACD_12_26_9', 0.0)),
                macd_signal=float(latest.get('MACDsignal_12_26_9', 0.0)),
                macd_hist=float(latest.get('MACDhist_12_26_9', 0.0)),
                atr=float(latest.get('ATR_14', 0.0)),
                adx=float(latest.get('ADX_14', 0.0)) if 'ADX_14' in latest else None,
                bb_percent=float(latest.get('BB_percent', 0.5)) if 'BB_percent' in latest else None,
                sentiment_score=float(latest.get('sentiment_score', 0.0)),
                sentiment_sources=None,
                current_position=int(request.context.get('position', 0)),
                unrealized_pnl=float(request.context.get('unrealized_pnl', 0.0)),
                portfolio_heat=float(request.context.get('portfolio_heat', 0.0)),
                daily_pnl=float(request.context.get('daily_pnl', 0.0)),
                win_rate=float(request.context.get('win_rate', 0.0)),
                market_regime=request.context.get('market_regime'),
                volatility_regime=request.context.get('volatility_regime'),
            )
            
            # Call trade advisor (this may call LLM/RAG)
            # NOTE: Trade advisor should return commentary only, not override signals
            enhanced_signal, llm_rec = self.trade_advisor.enhance_signal(
                request.signal,
                context
            )
            
            # Extract commentary
            commentary = ""
            recommendation = None
            
            if llm_rec:
                recommendation = llm_rec.to_dict() if hasattr(llm_rec, 'to_dict') else {}
                commentary = recommendation.get('reasoning', '') or recommendation.get('commentary', '')
            
            return LLMResponse(
                request_id=request.request_id,
                recommendation=recommendation,
                commentary=commentary,
                timestamp=datetime.now(),
                processing_time=0.0  # Will be set by caller
            )
        
        except Exception as e:
            logger.error(f"Error processing LLM request: {e}", exc_info=True)
            return LLMResponse(
                request_id=request.request_id,
                recommendation=None,
                commentary="",
                timestamp=datetime.now(),
                processing_time=0.0,
                error=str(e)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "running": self.running,
            "total_requests": self.total_requests,
            "total_responses": self.total_responses,
            "total_errors": self.total_errors,
            "avg_processing_time": self.avg_processing_time,
            "queue_size": self.request_queue.qsize(),
            "cache_size": len(self.response_cache),
        }
