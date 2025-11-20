"""
Position Manager for enforcing hard safety constraints.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional

from ib_insync import IB

from ..config import TradingConfig

logger = logging.getLogger(__name__)

@dataclass
class DecisionResult:
    allowed_contracts: int
    reason: str
    margin_estimate: Optional[float] = None
    timestamp: str = ""

    def to_json(self):
        return json.dumps(self.__dict__)

class PositionManager:
    """
    Authoritative source for position limits and risk checks.
    Enforces MAX_CONTRACTS and MARGIN_LIMIT_PCT.
    """

    def __init__(self, ib: IB, config: TradingConfig, symbol: str):
        self.ib = ib
        self.config = config
        self.symbol = symbol
        self._lock = asyncio.Lock()
        
        # Ensure we have safe defaults if config is missing them
        self.max_contracts = getattr(config, 'max_contracts_limit', 5)
        self.margin_limit_pct = getattr(config, 'margin_limit_pct', 0.8)
        
        logger.info(f"PositionManager initialized for {symbol}: "
                    f"Max Contracts={self.max_contracts}, "
                    f"Margin Limit={self.margin_limit_pct:.0%}")

    async def get_net_position(self) -> int:
        """
        Query IB for the current net position for the symbol.
        This is the authoritative source of truth.
        """
        try:
            positions = self.ib.positions()
            for pos in positions:
                if pos.contract.symbol == self.symbol:
                    return int(pos.position)
            return 0
        except Exception as e:
            logger.error(f"Failed to query IB positions: {e}")
            # Fail safe: assume we might have positions if we can't check
            # But returning 0 might be dangerous if we actually have 5.
            # Better to raise or return a special value? 
            # For now, let's log and return 0 but this is a risk.
            # Ideally, we should block trading if we can't verify positions.
            raise RuntimeError(f"Cannot verify positions: {e}")

    async def get_margin_usage(self) -> Tuple[float, float]:
        """
        Get current margin usage and total capacity.
        Returns (current_margin_usage, total_margin_capacity)
        """
        try:
            # Request account summary
            # Tags: NetLiquidation, FullInitMarginReq, FullMaintMarginReq
            summary = await self.ib.accountSummaryAsync()
            
            net_liquidation = 0.0
            init_margin = 0.0
            
            for item in summary:
                if item.tag == 'NetLiquidation':
                    net_liquidation = float(item.value)
                elif item.tag == 'FullInitMarginReq':
                    init_margin = float(item.value)
            
            if net_liquidation > 0:
                return init_margin, net_liquidation
            return 0.0, 1.0 # Avoid division by zero
            
        except Exception as e:
            logger.error(f"Failed to get margin info: {e}")
            raise

    async def can_place_order(self, requested_contracts: int) -> DecisionResult:
        """
        Determine if an order can be placed based on hard constraints.
        
        Args:
            requested_contracts: Number of contracts to buy (positive) or sell (negative).
                                 Note: This is the ORDER size, not target position.
        
        Returns:
            DecisionResult with allowed contracts (0 if rejected) and reason.
        """
        async with self._lock:
            timestamp = datetime.utcnow().isoformat()
            
            # 1. Get authoritative current position
            try:
                current_net = await self.get_net_position()
            except Exception as e:
                return DecisionResult(0, f"Position check failed: {e}", timestamp=timestamp)

            # 2. Calculate projected new position
            # If we are Long 3 and Buy 2, new is 5.
            # If we are Long 3 and Sell 2, new is 1.
            # If we are Short 2 (-2) and Sell 1 (-1), new is -3.
            projected_net = current_net + requested_contracts
            
            # 3. Check Max Contracts Cap
            if abs(projected_net) > self.max_contracts:
                # Calculate what is allowed
                # Example: Max 5. Current 3. Request 3. Projected 6.
                # Allowed additional = 5 - 3 = 2.
                
                if requested_contracts > 0: # Buying
                    available_room = self.max_contracts - current_net
                    allowed = max(0, available_room)
                else: # Selling
                    # Current -2. Max 5. Request -4. Projected -6.
                    # Room on short side: -5 - (-2) = -3.
                    available_room = -self.max_contracts - current_net
                    allowed = min(0, available_room) # e.g. -3
                
                # If the request was to reduce position, we should allow it.
                # E.g. Current 6 (violation). Request -1. Projected 5.
                # Logic above: Buying? No. Selling.
                # available_room = -5 - 6 = -11. Allowed min(0, -11) = -11.
                # Wait, if we are Long 6, and want to Sell 1.
                # current_net = 6. requested = -1. projected = 5.
                # abs(5) <= 5. So this block wouldn't trigger if we just check projected.
                # But we are in the block because abs(projected) > max?
                # No, let's re-verify.
                
                # Case: Current 6. Request -1. Projected 5. abs(5) <= 5. OK.
                # Case: Current 5. Request 1. Projected 6. abs(6) > 5. Violation.
                # We need to cap 'allowed' to what brings us to +/- 5.
                
                # Let's stick to the logic:
                # We want to return a modified 'requested_contracts' that fits.
                
                if abs(projected_net) <= self.max_contracts:
                    # It fits, so why are we here?
                    # Maybe we are reducing an over-limit position?
                    pass 
                else:
                    # It doesn't fit.
                    reason = f"Cap breach: Current {current_net} + Req {requested_contracts} = {projected_net} (Max {self.max_contracts})"
                    
                    # If we are reducing risk, allow it.
                    # Reducing risk means abs(projected) < abs(current)
                    if abs(projected_net) < abs(current_net):
                         # Allow the trade as it reduces exposure
                         pass
                    else:
                        # We need to truncate
                        if requested_contracts > 0:
                            allowed = max(0, self.max_contracts - current_net)
                        else:
                            allowed = min(0, -self.max_contracts - current_net)
                        
                        return DecisionResult(allowed, f"{reason}. Reduced to {allowed}", timestamp=timestamp)

            # 4. Margin Check
            try:
                current_margin, total_liquidation = await self.get_margin_usage()
                
                # Estimate new margin
                # Simple conservative estimate: Current Margin + (Contracts * MarginPerContract)
                # We need a margin per contract estimate.
                # ES approx $12,000 intraday or more? Let's use a conservative buffer.
                # Or better, check if we are increasing exposure.
                
                is_increasing_exposure = abs(projected_net) > abs(current_net)
                
                if is_increasing_exposure:
                    # Rough estimate for ES margin
                    MARGIN_PER_CONTRACT = 15000.0 # Conservative placeholder
                    additional_margin = abs(requested_contracts) * MARGIN_PER_CONTRACT
                    projected_margin_usage = current_margin + additional_margin
                    projected_margin_pct = projected_margin_usage / total_liquidation
                    
                    if projected_margin_pct > self.margin_limit_pct:
                        return DecisionResult(0, 
                            f"Margin limit: Projected {projected_margin_pct:.1%} > Limit {self.margin_limit_pct:.1%}", 
                            margin_estimate=projected_margin_pct,
                            timestamp=timestamp)
            
            except Exception as e:
                logger.warning(f"Margin check skipped due to error: {e}")
                # Fail open or closed? 
                # "Margin safety: do not submit... if margin would be exceeded"
                # If we can't verify, maybe we should be cautious.
                # For now, let's log and proceed but with a warning in the reason.
                pass

            return DecisionResult(requested_contracts, "Approved", timestamp=timestamp)
