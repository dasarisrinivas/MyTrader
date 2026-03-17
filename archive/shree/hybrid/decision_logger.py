"""Decision Logger - Audit trail for all hybrid decisions.

Logs every D-engine signal, H-engine advisory, and final decision
for full auditability and analysis.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logger import logger
from .d_engine import DEngineSignal
from .h_engine import HEngineAdvisory
from .confidence import ConfidenceResult
from .safety import SafetyCheck


class DecisionLogger:
    """Logger for hybrid decision audit trail.
    
    Persists:
    - Full decision records to decision_log.json
    - Summary rows to decisions.csv
    - LLM prompts and responses to llm_calls.json
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        json_file: str = "decision_log.json",
        csv_file: str = "decisions.csv",
        llm_file: str = "llm_calls.json",
        max_json_entries: int = 1000,
    ):
        """Initialize decision logger.
        
        Args:
            log_dir: Directory for log files
            json_file: JSON log file name
            csv_file: CSV summary file name
            llm_file: LLM calls log file name
            max_json_entries: Maximum entries in JSON before rotation
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_path = self.log_dir / json_file
        self.csv_path = self.log_dir / csv_file
        self.llm_path = self.log_dir / llm_file
        
        self.max_json_entries = max_json_entries
        self._decision_count = 0
        
        # Initialize CSV with headers if new
        if not self.csv_path.exists():
            self._init_csv()
        
        logger.info(f"DecisionLogger initialized: {self.log_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            "timestamp",
            "d_action",
            "d_score",
            "d_candidate",
            "h_recommendation",
            "h_confidence",
            "h_cached",
            "final_confidence",
            "final_action",
            "should_trade",
            "safety_passed",
            "position_size_pct",
            "entry_price",
            "stop_loss",
            "take_profit",
            "order_id",
            "execution_status",
        ]
        
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_decision(
        self,
        d_signal: DEngineSignal,
        h_advisory: Optional[HEngineAdvisory],
        confidence_result: ConfidenceResult,
        safety_check: SafetyCheck,
        order_id: Optional[str] = None,
        execution_status: str = "pending",
    ):
        """Log a complete decision record.
        
        Args:
            d_signal: D-engine signal
            h_advisory: H-engine advisory (may be None)
            confidence_result: Final confidence calculation
            safety_check: Safety check result
            order_id: Order ID if placed
            execution_status: Order execution status
        """
        timestamp = datetime.now(timezone.utc)
        self._decision_count += 1
        
        # Build full record
        record = {
            "decision_id": self._decision_count,
            "timestamp": timestamp.isoformat(),
            "d_engine": d_signal.to_dict(),
            "h_engine": h_advisory.to_dict() if h_advisory else None,
            "confidence": confidence_result.to_dict(),
            "safety": safety_check.to_dict(),
            "order_id": order_id,
            "execution_status": execution_status,
        }
        
        # Append to JSON log
        self._append_json(record)
        
        # Append to CSV summary
        self._append_csv(
            timestamp=timestamp,
            d_signal=d_signal,
            h_advisory=h_advisory,
            confidence_result=confidence_result,
            safety_check=safety_check,
            order_id=order_id,
            execution_status=execution_status,
        )
        
        logger.debug(f"Decision logged: #{self._decision_count}")
    
    def _append_json(self, record: Dict[str, Any]):
        """Append record to JSON log file."""
        try:
            # Load existing data
            if self.json_path.exists():
                with open(self.json_path, "r") as f:
                    data = json.load(f)
            else:
                data = {"decisions": []}
            
            # Append new record
            data["decisions"].append(record)
            
            # Rotate if needed
            if len(data["decisions"]) > self.max_json_entries:
                # Keep last N entries
                data["decisions"] = data["decisions"][-self.max_json_entries:]
            
            # Write back
            with open(self.json_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to write JSON log: {e}")
    
    def _append_csv(
        self,
        timestamp: datetime,
        d_signal: DEngineSignal,
        h_advisory: Optional[HEngineAdvisory],
        confidence_result: ConfidenceResult,
        safety_check: SafetyCheck,
        order_id: Optional[str],
        execution_status: str,
    ):
        """Append summary row to CSV."""
        try:
            row = [
                timestamp.isoformat(),
                d_signal.action,
                f"{d_signal.technical_score:.3f}",
                d_signal.is_candidate,
                h_advisory.recommendation if h_advisory else "",
                f"{h_advisory.model_confidence:.3f}" if h_advisory else "",
                h_advisory.cached if h_advisory else "",
                f"{confidence_result.final_confidence:.3f}",
                confidence_result.action,
                confidence_result.should_trade,
                safety_check.is_safe,
                f"{confidence_result.position_size_pct:.2f}",
                f"{d_signal.entry_price:.2f}",
                f"{d_signal.stop_loss:.2f}",
                f"{d_signal.take_profit:.2f}",
                order_id or "",
                execution_status,
            ]
            
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Failed to write CSV log: {e}")
    
    def log_llm_call(
        self,
        prompt: str,
        response: str,
        model: str,
        latency_ms: float,
        context_hash: str,
        trigger: str,
    ):
        """Log an LLM call for audit.
        
        Args:
            prompt: Full prompt sent to LLM
            response: LLM response
            model: Model ID
            latency_ms: Call latency in milliseconds
            context_hash: Context hash for caching
            trigger: Trigger type
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "trigger": trigger,
            "context_hash": context_hash,
            "latency_ms": latency_ms,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "prompt": prompt[:1000] + "..." if len(prompt) > 1000 else prompt,
            "response": response,
        }
        
        try:
            # Load existing data
            if self.llm_path.exists():
                with open(self.llm_path, "r") as f:
                    data = json.load(f)
            else:
                data = {"llm_calls": []}
            
            data["llm_calls"].append(record)
            
            # Keep last 100 calls
            if len(data["llm_calls"]) > 100:
                data["llm_calls"] = data["llm_calls"][-100:]
            
            with open(self.llm_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to write LLM log: {e}")
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent decision records.
        
        Args:
            count: Number of records to return
            
        Returns:
            List of recent decision records
        """
        try:
            if self.json_path.exists():
                with open(self.json_path, "r") as f:
                    data = json.load(f)
                return data.get("decisions", [])[-count:]
        except Exception as e:
            logger.error(f"Failed to read decisions: {e}")
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.
        
        Returns:
            Dictionary of logging stats
        """
        try:
            decisions = []
            if self.json_path.exists():
                with open(self.json_path, "r") as f:
                    data = json.load(f)
                decisions = data.get("decisions", [])
            
            # Calculate stats
            total = len(decisions)
            candidates = sum(1 for d in decisions if d.get("d_engine", {}).get("is_candidate"))
            traded = sum(1 for d in decisions if d.get("confidence", {}).get("should_trade"))
            
            return {
                "total_decisions": total,
                "candidates": candidates,
                "trades_executed": traded,
                "candidate_rate": candidates / total if total > 0 else 0,
                "trade_rate": traded / total if total > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Failed to calculate stats: {e}")
            return {}
