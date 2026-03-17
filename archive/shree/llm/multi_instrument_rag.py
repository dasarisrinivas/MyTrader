"""
Multi-Instrument RAG Support
Enables RAG knowledge base to support multiple trading instruments beyond SPY.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..utils.logger import logger
from .rag_engine import RAGEngine


class MultiInstrumentRAG:
    """
    Manages separate RAG contexts for multiple trading instruments.
    Allows for instrument-specific knowledge retrieval and trading strategies.
    """
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        instruments_config_path: Optional[str] = None
    ):
        """Initialize multi-instrument RAG manager.
        
        Args:
            rag_engine: Base RAG engine instance
            instruments_config_path: Path to JSON config with instrument metadata
        """
        self.rag_engine = rag_engine
        self.instruments: Dict[str, Dict] = {}
        
        # Load instrument configuration if provided
        if instruments_config_path:
            self._load_instruments_config(instruments_config_path)
        
        # Default instruments
        self._initialize_default_instruments()
        
        logger.info(f"Initialized MultiInstrumentRAG with {len(self.instruments)} instruments")
    
    def _initialize_default_instruments(self) -> None:
        """Initialize default instrument configurations."""
        if "SPY" not in self.instruments:
            self.instruments["SPY"] = {
                "name": "SPDR S&P 500 ETF",
                "type": "ETF",
                "market": "US",
                "trading_hours": "09:30-16:00 ET",
                "features": ["high_liquidity", "sp500_tracking"]
            }
        
        if "ES" not in self.instruments:
            self.instruments["ES"] = {
                "name": "E-mini S&P 500 Futures",
                "type": "Futures",
                "market": "CME",
                "trading_hours": "23:00-22:00 ET (Sun-Fri)",
                "contract_size": 50,
                "features": ["leveraged", "nearly_24h_trading"]
            }
    
    def _load_instruments_config(self, config_path: str) -> None:
        """Load instrument configurations from JSON file.
        
        Args:
            config_path: Path to instruments config JSON
        """
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r') as f:
                    config = json.load(f)
                    self.instruments.update(config.get("instruments", {}))
                logger.info(f"Loaded {len(config.get('instruments', {}))} instrument configs")
            else:
                logger.warning(f"Instruments config not found: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load instruments config: {e}")
    
    def add_instrument(
        self,
        symbol: str,
        name: str,
        instrument_type: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a new instrument to the system.
        
        Args:
            symbol: Trading symbol (e.g., "NQ", "QQQ")
            name: Full instrument name
            instrument_type: Type (ETF, Futures, Stock, etc.)
            metadata: Additional metadata dict
        """
        instrument_data = {
            "name": name,
            "type": instrument_type,
            **(metadata or {})
        }
        
        self.instruments[symbol] = instrument_data
        logger.info(f"Added instrument: {symbol} ({name})")
    
    def get_instrument_context(self, symbol: str) -> str:
        """Get formatted context string for an instrument.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Formatted context string for the instrument
        """
        if symbol not in self.instruments:
            return f"Unknown instrument: {symbol}"
        
        info = self.instruments[symbol]
        context_parts = [
            f"Instrument: {symbol}",
            f"Name: {info.get('name', 'N/A')}",
            f"Type: {info.get('type', 'N/A')}",
        ]
        
        if "market" in info:
            context_parts.append(f"Market: {info['market']}")
        
        if "trading_hours" in info:
            context_parts.append(f"Trading Hours: {info['trading_hours']}")
        
        if "contract_size" in info:
            context_parts.append(f"Contract Size: {info['contract_size']}")
        
        return "\n".join(context_parts)
    
    def retrieve_with_instrument_context(
        self,
        query: str,
        symbol: str,
        top_k: int = 3,
        score_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Retrieve documents with instrument-specific context.
        
        Args:
            query: User query
            symbol: Trading symbol to add context for
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        # Augment query with instrument context
        instrument_context = self.get_instrument_context(symbol)
        augmented_query = f"{instrument_context}\n\nQuery: {query}"
        
        logger.info(f"Retrieving for {symbol}: {query[:50]}...")
        
        # Use base RAG engine for retrieval
        results = self.rag_engine.retrieve_context(
            query=augmented_query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        return results
    
    def ingest_instrument_documents(
        self,
        symbol: str,
        documents: List[str],
        tag_documents: bool = True
    ) -> Dict:
        """Ingest documents specific to an instrument.
        
        Args:
            symbol: Trading symbol
            documents: List of document texts
            tag_documents: Whether to add instrument tags to documents
            
        Returns:
            Ingestion result dictionary
        """
        if symbol not in self.instruments:
            logger.warning(f"Instrument {symbol} not registered, adding as unknown type")
            self.add_instrument(symbol, f"Unknown ({symbol})", "Unknown")
        
        # Optionally tag documents with instrument identifier
        if tag_documents:
            tagged_docs = [
                f"[Instrument: {symbol}] {doc}" 
                for doc in documents
            ]
        else:
            tagged_docs = documents
        
        logger.info(f"Ingesting {len(documents)} documents for {symbol}")
        
        # Use base RAG engine for ingestion
        result = self.rag_engine.ingest_documents(
            documents=tagged_docs,
            clear_existing=False  # Keep existing docs
        )
        
        return result
    
    def list_instruments(self) -> List[str]:
        """Get list of registered instruments.
        
        Returns:
            List of instrument symbols
        """
        return list(self.instruments.keys())
    
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about an instrument.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Instrument information dict or None
        """
        return self.instruments.get(symbol)
    
    def generate_trading_context(
        self,
        symbol: str,
        market_data: Dict,
        include_historical: bool = True
    ) -> str:
        """Generate comprehensive trading context for RAG queries.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data (price, volume, etc.)
            include_historical: Include historical context
            
        Returns:
            Formatted trading context string
        """
        context_parts = [self.get_instrument_context(symbol)]
        
        # Add current market data
        if market_data:
            context_parts.append("\nCurrent Market Data:")
            for key, value in market_data.items():
                context_parts.append(f"  {key}: {value}")
        
        # Could add historical context retrieval here
        if include_historical:
            context_parts.append("\n[Historical context would be retrieved here]")
        
        return "\n".join(context_parts)
