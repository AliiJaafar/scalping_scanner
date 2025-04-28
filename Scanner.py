import ccxt.async_support as ccxt
import pandas as pd
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Configuration for scanner
@dataclass
class ScannerConfig:
    exchange_name: str = "binance"
    timeframe: str = "15m"
    limit: int = 200
    quote_currency: str = "USDT"
    ema_short: int = 20
    ema_long: int = 200

class Scanner:
    def __init__(self, config: ScannerConfig = ScannerConfig()):
        """Initialize the scanner with given configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })

    async def fetch_symbols(self) -> List[str]:
        """Fetch active trading pairs ending with the configured quote currency."""
        try:
            markets = await self.exchange.fetch_markets()
            symbols = [
                market['symbol'] for market in markets
                if market['symbol'].endswith(f"/{self.config.quote_currency}")
                and market['active']
            ]
            self.logger.info(f"Fetched {len(symbols)} active {self.config.quote_currency} trading pairs")
            return symbols
        except Exception as e:
            self.logger.error(f"Error fetching symbols: {str(e)}")
            return []

    async def fetch_ohlcv(self, symbol: str) -> Optional[List[List[float]]]:
        """Fetch OHLCV data for a given symbol."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=self.config.timeframe,
                limit=self.config.limit
            )
            return ohlcv if len(ohlcv) >= self.config.limit else None
        except ccxt.NetworkError as e:
            self.logger.warning(f"Network error fetching {symbol}: {str(e)}")
        except ccxt.ExchangeError as e:
            self.logger.warning(f"Exchange error fetching {symbol}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {symbol}: {str(e)}")
        return None

    def calculate_ema(self, prices: List[float], period: int) -> Optional[List[float]]:
        """Calculate EMA using pandas for accuracy."""
        if len(prices) < period:
            self.logger.warning(f"Insufficient data for EMA calculation: {len(prices)} < {period}")
            return None
        try:
            return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            return None

    async def scan_market(self) -> List[Tuple[str, float]]:
        """Scan market for trading signals based on EMA crossover and price dip."""
        symbols = await self.fetch_symbols()
        signals = []

        for symbol in symbols:
            self.logger.debug(f"Scanning {symbol}")
            ohlcv = await self.fetch_ohlcv(symbol)

            if not ohlcv:
                continue

            close_prices = [candle[4] for candle in ohlcv]
            ema_short = self.calculate_ema(close_prices, self.config.ema_short)
            ema_long = self.calculate_ema(close_prices, self.config.ema_long)

            if ema_short and ema_long and len(ema_short) > 0 and len(ema_long) > 0:
                if (ema_short[-1] > ema_long[-1] and
                    close_prices[-1] < ema_short[-1]):
                    signals.append((symbol, close_prices[-1]))
                    self.logger.info(f"Signal detected for {symbol} at price {close_prices[-1]:.4f}")

        return signals

    async def close(self):
        """Close exchange connection."""
        try:
            await self.exchange.close()
            self.logger.info("Exchange connection closed")
        except Exception as e:
            self.logger.error(f"Error closing exchange connection: {str(e)}")