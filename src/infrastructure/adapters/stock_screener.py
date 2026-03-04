"""
Yahoo Finance Stock Screener Adapter

Architectural Intent:
- Implements StockScreenerPort using yfinance batch downloads
- Caches results for 5 minutes to avoid excessive API calls
- Covers ~100 popular US tickers as the screening universe
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import yfinance as yf

from src.domain.entities.screening import ScreenResult
from src.domain.services.screening import StockScreenerPort

logger = logging.getLogger(__name__)

POPULAR_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "JPM", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT",
    "DHR", "NEE", "TXN", "PM", "LIN", "CMCSA", "VZ", "BMY", "ADBE", "RTX",
    "HON", "INTC", "NFLX", "AMD", "QCOM", "LOW", "UPS", "COP", "BA", "CAT",
    "GS", "SBUX", "BLK", "ISRG", "GILD", "MDLZ", "ADP", "SYK", "PLD", "ADI",
    "NOW", "CB", "VRTX", "MMC", "CI", "SO", "DUK", "CL", "REGN", "MO",
    "ZTS", "PANW", "SNPS", "CDNS", "CME", "BSX", "FI", "ICE", "ANET", "SHW",
    "PYPL", "CRWD", "MRVL", "ABNB", "DDOG", "SNOW", "SQ", "COIN", "RIVN", "LCID",
    "NIO", "PLTR", "SOFI", "RBLX", "HOOD", "ROKU", "U", "PINS", "SNAP", "LYFT",
]

CACHE_TTL_SECONDS = 300  # 5 minutes


class YahooFinanceScreenerAdapter(StockScreenerPort):
    """Fetches batch data from yfinance for ~100 popular tickers with caching."""

    def __init__(self):
        self._cache: Optional[List[ScreenResult]] = None
        self._cache_time: float = 0.0

    def fetch_screen_data(self) -> List[ScreenResult]:
        now = time.time()
        if self._cache is not None and (now - self._cache_time) < CACHE_TTL_SECONDS:
            return self._cache

        results: List[ScreenResult] = []
        try:
            tickers_str = " ".join(POPULAR_TICKERS)
            data = yf.download(tickers_str, period="2d", group_by="ticker", threads=True, progress=False)

            for ticker in POPULAR_TICKERS:
                try:
                    if len(POPULAR_TICKERS) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    if ticker_data.empty or len(ticker_data) < 2:
                        continue

                    prev_close = float(ticker_data["Close"].iloc[-2])
                    curr_close = float(ticker_data["Close"].iloc[-1])
                    volume = int(ticker_data["Volume"].iloc[-1])

                    if prev_close == 0:
                        continue

                    change_pct = round((curr_close - prev_close) / prev_close * 100, 2)

                    # Get extra info (name, sector, market_cap)
                    info = self._get_ticker_info(ticker)

                    results.append(ScreenResult(
                        symbol=ticker,
                        name=info.get("name", ticker),
                        price=round(curr_close, 2),
                        change_pct=change_pct,
                        volume=volume,
                        market_cap=info.get("market_cap"),
                        sector=info.get("sector"),
                    ))
                except Exception:
                    continue

        except Exception as exc:
            logger.error("Failed to fetch screening data: %s", exc)

        self._cache = results
        self._cache_time = now
        return results

    def _get_ticker_info(self, ticker: str) -> dict:
        """Quick lookup for name/sector. Uses a static map to avoid slow API calls."""
        # Static map for the most common tickers (avoids yf.Ticker().info per ticker)
        _INFO = {
            "AAPL": ("Apple Inc.", "Technology"),
            "MSFT": ("Microsoft Corp.", "Technology"),
            "AMZN": ("Amazon.com Inc.", "Consumer Cyclical"),
            "NVDA": ("NVIDIA Corp.", "Technology"),
            "GOOGL": ("Alphabet Inc.", "Communication Services"),
            "META": ("Meta Platforms Inc.", "Communication Services"),
            "TSLA": ("Tesla Inc.", "Consumer Cyclical"),
            "BRK.B": ("Berkshire Hathaway", "Financial Services"),
            "UNH": ("UnitedHealth Group", "Healthcare"),
            "JNJ": ("Johnson & Johnson", "Healthcare"),
            "V": ("Visa Inc.", "Financial Services"),
            "XOM": ("Exxon Mobil Corp.", "Energy"),
            "JPM": ("JPMorgan Chase", "Financial Services"),
            "WMT": ("Walmart Inc.", "Consumer Defensive"),
            "MA": ("Mastercard Inc.", "Financial Services"),
            "PG": ("Procter & Gamble", "Consumer Defensive"),
            "HD": ("Home Depot Inc.", "Consumer Cyclical"),
            "CVX": ("Chevron Corp.", "Energy"),
            "MRK": ("Merck & Co.", "Healthcare"),
            "ABBV": ("AbbVie Inc.", "Healthcare"),
            "LLY": ("Eli Lilly", "Healthcare"),
            "PEP": ("PepsiCo Inc.", "Consumer Defensive"),
            "KO": ("Coca-Cola Co.", "Consumer Defensive"),
            "COST": ("Costco Wholesale", "Consumer Defensive"),
            "AVGO": ("Broadcom Inc.", "Technology"),
            "NFLX": ("Netflix Inc.", "Communication Services"),
            "AMD": ("AMD Inc.", "Technology"),
            "INTC": ("Intel Corp.", "Technology"),
            "BA": ("Boeing Co.", "Industrials"),
            "CAT": ("Caterpillar Inc.", "Industrials"),
            "GS": ("Goldman Sachs", "Financial Services"),
            "SBUX": ("Starbucks Corp.", "Consumer Cyclical"),
            "PYPL": ("PayPal Holdings", "Financial Services"),
            "COIN": ("Coinbase Global", "Financial Services"),
            "PLTR": ("Palantir Technologies", "Technology"),
        }
        name, sector = _INFO.get(ticker, (ticker, None))
        return {"name": name, "sector": sector, "market_cap": None}
