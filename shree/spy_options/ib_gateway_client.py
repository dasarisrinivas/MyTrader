"""IB Gateway (TWS/IBKR socket API) client for SPY options data.

This is a stub for replacing the REST Client Portal logic with ib_insync-based socket API logic.
"""
from ib_insync import IB, Stock, Option
from ..config.spy_options import SpyOptionsIBConfig
from ..utils.logger import logger

class IBGatewayOptionsClient:
    def __init__(self, cfg: SpyOptionsIBConfig):
        self._cfg = cfg
        self._ib = IB()
        self._connected = False

    def connect(self):
        # Force IB Gateway port 4001 regardless of config
        port = 4001
        self._ib.connect(self._cfg.host, port, clientId=5)
        self._connected = self._ib.isConnected()
        if self._connected:
            logger.info("Connected to IB Gateway at {}:{}", self._cfg.host, port)
        else:
            logger.error("Failed to connect to IB Gateway at {}:{}", self._cfg.host, port)
        return self._connected

    def resolve_spy(self):
        contract = Stock('SPY', 'SMART', 'USD')
        details = self._ib.reqContractDetails(contract)
        if not details:
            logger.error("Failed to resolve SPY contract via IB Gateway.")
            return None
        return details[0].contract.conId

    def disconnect(self):
        self._ib.disconnect()
        self._connected = False
