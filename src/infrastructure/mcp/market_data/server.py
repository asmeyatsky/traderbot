"""
Market Data MCP server — bounded context for reads only.

Layer: infrastructure
Ports used: MarketDataPort, AIModelPort, NewsAnalysisPort, TechnicalAnalysisPort
MCP integration: 7 tools (all reads), no writes
Stack choice: in-process — see src/infrastructure/mcp/__init__.py

Zero business logic here — this is a thin wrapper that validates input via the
base class and delegates to domain ports. Any branching (e.g. ensemble
predictor preferred over single model) lives in the domain service, not here.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Optional

from src.domain.ports import AIModelPort, MarketDataPort, NewsAnalysisPort
from src.domain.services.technical_analysis import TechnicalAnalysisPort
from src.domain.value_objects import Symbol
from src.infrastructure.mcp.base import McpServer, McpTool


_SYMBOL_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {
            "type": "string",
            "description": "Stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
        }
    },
    "required": ["symbol"],
}


class MarketDataMcpServer(McpServer):
    """All read-only market-data tools live here."""

    context = "market_data"

    def __init__(
        self,
        market_data_service: MarketDataPort,
        ai_model_service: AIModelPort,
        news_analysis_service: NewsAnalysisPort,
        technical_analysis_port: Optional[TechnicalAnalysisPort] = None,
        stock_screener: Optional[Any] = None,
        exchange_registry: Optional[Any] = None,
        ensemble_predictor: Optional[Any] = None,
    ) -> None:
        self._market_data = market_data_service
        self._ai_model = ai_model_service
        self._news = news_analysis_service
        self._technical_analysis = technical_analysis_port
        self._stock_screener = stock_screener
        self._exchange_registry = exchange_registry
        self._ensemble_predictor = ensemble_predictor
        super().__init__()

    def _register(self) -> None:
        self._add_tool(
            McpTool(
                name="get_stock_price",
                description="Get the current real-time price for a stock symbol.",
                input_schema=_SYMBOL_SCHEMA,
            )
        )
        self._add_tool(
            McpTool(
                name="get_ml_prediction",
                description=(
                    "Get an AI/ML price movement prediction for a stock. "
                    "Returns predicted direction and confidence."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "days": {
                            "type": "integer",
                            "description": "Number of days to predict ahead (default: 1)",
                            "default": 1,
                        },
                    },
                    "required": ["symbol"],
                },
            )
        )
        self._add_tool(
            McpTool(
                name="get_trading_signal",
                description=(
                    "Get a BUY/SELL/HOLD trading signal for a stock based on "
                    "ensemble ML models."
                ),
                input_schema=_SYMBOL_SCHEMA,
            )
        )
        self._add_tool(
            McpTool(
                name="get_news_sentiment",
                description=(
                    "Get news sentiment analysis for a stock. Returns sentiment "
                    "score and recent headlines."
                ),
                input_schema=_SYMBOL_SCHEMA,
            )
        )
        self._add_tool(
            McpTool(
                name="get_technical_analysis",
                description=(
                    "Get detailed technical analysis indicators: RSI, MACD, SMA, "
                    "EMA, Bollinger Bands, ATR, Stochastic, ADX."
                ),
                input_schema=_SYMBOL_SCHEMA,
            )
        )
        self._add_tool(
            McpTool(
                name="screen_stocks",
                description=(
                    "Screen stocks by criteria or use prebuilt screens "
                    "(top_gainers, top_losers, most_active, high_momentum, oversold_rsi)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "prebuilt_screen": {
                            "type": "string",
                            "enum": [
                                "top_gainers",
                                "top_losers",
                                "most_active",
                                "high_momentum",
                                "oversold_rsi",
                            ],
                        },
                        "min_change_pct": {"type": "number"},
                        "max_change_pct": {"type": "number"},
                        "min_volume": {"type": "integer"},
                        "sectors": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer"},
                    },
                },
            )
        )
        self._add_tool(
            McpTool(
                name="get_market_status",
                description=(
                    "Check which global stock exchanges are currently open or "
                    "closed, with next open/close times."
                ),
                input_schema={"type": "object", "properties": {}},
            )
        )

    async def _dispatch(
        self, tool_name: str, args: Mapping[str, Any], actor_user_id: str
    ) -> Mapping[str, Any]:
        if tool_name == "get_stock_price":
            return self._get_stock_price(args)
        if tool_name == "get_ml_prediction":
            return self._get_ml_prediction(args)
        if tool_name == "get_trading_signal":
            return self._get_trading_signal(args)
        if tool_name == "get_news_sentiment":
            return self._get_news_sentiment(args)
        if tool_name == "get_technical_analysis":
            return self._get_technical_analysis(args)
        if tool_name == "screen_stocks":
            return self._screen_stocks(args)
        if tool_name == "get_market_status":
            return self._get_market_status()
        raise AssertionError(f"unreachable: tool {tool_name!r} registered but not dispatched")

    # ------------------------------------------------------------------
    # Tool bodies — each one delegates to a domain port. Zero branching
    # that isn't strictly about port availability.
    # ------------------------------------------------------------------

    def _get_stock_price(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        symbol = Symbol(args["symbol"].upper())
        price = self._market_data.get_current_price(symbol)
        if price is None:
            raise RuntimeError(f"Could not fetch price for {symbol}")
        return {
            "symbol": str(symbol),
            "price": float(price.amount),
            "currency": price.currency,
        }

    def _get_ml_prediction(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        symbol = Symbol(args["symbol"].upper())
        days = args.get("days", 1)

        if self._ensemble_predictor is not None:
            try:
                pred = self._ensemble_predictor.predict(symbol)
                return {
                    "symbol": pred.symbol,
                    "direction": pred.direction,
                    "confidence": pred.confidence,
                    "predicted_change_pct": pred.predicted_change_pct,
                    "model_votes": pred.model_votes,
                    "top_features": pred.top_features,
                    "days_ahead": days,
                }
            except Exception:
                pass  # fall through to the single-model prediction

        prediction = self._ai_model.predict_price_movement(symbol, days)
        return {
            "symbol": str(symbol),
            "predicted_movement": prediction,
            "days_ahead": days,
            "direction": (
                "UP" if prediction > 0 else "DOWN" if prediction < 0 else "FLAT"
            ),
        }

    def _get_trading_signal(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        symbol = Symbol(args["symbol"].upper())

        if self._ensemble_predictor is not None:
            try:
                pred = self._ensemble_predictor.predict(symbol)
                if pred.direction == "UP" and pred.confidence >= 0.55:
                    signal = "BUY"
                elif pred.direction == "DOWN" and pred.confidence >= 0.55:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                return {
                    "symbol": pred.symbol,
                    "signal": signal,
                    "confidence": pred.confidence,
                    "direction": pred.direction,
                    "model_votes": pred.model_votes,
                    "top_features": pred.top_features,
                }
            except Exception:
                pass

        signal = self._ai_model.get_trading_signal(symbol)
        return {"symbol": str(symbol), "signal": signal}

    def _get_news_sentiment(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        symbol = Symbol(args["symbol"].upper())
        news = self._market_data.get_market_news(symbol)
        if not news:
            return {
                "symbol": str(symbol),
                "average_sentiment": 0,
                "sentiment_label": "no data",
                "headlines": [],
                "num_articles": 0,
            }
        sentiments = self._news.batch_analyze_sentiment(news[:5])
        avg_score = (
            sum(s.score for s in sentiments) / len(sentiments)
            if sentiments
            else Decimal("0")
        )
        return {
            "symbol": str(symbol),
            "average_sentiment": float(avg_score),
            "sentiment_label": (
                "positive" if avg_score > 5 else "negative" if avg_score < -5 else "neutral"
            ),
            "headlines": news[:5],
            "num_articles": len(news),
        }

    def _get_technical_analysis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._technical_analysis is None:
            raise RuntimeError("Technical analysis service not available")
        from src.domain.services.technical_analysis import generate_signal_summary

        symbol = Symbol(args["symbol"].upper())
        indicators = self._technical_analysis.compute_indicators(symbol)
        signals = generate_signal_summary(indicators)
        return {
            "symbol": indicators.symbol,
            "current_price": indicators.current_price,
            "indicators": {
                "RSI_14": indicators.rsi_14,
                "MACD_line": indicators.macd_line,
                "MACD_signal": indicators.macd_signal,
                "MACD_histogram": indicators.macd_histogram,
                "SMA_20": indicators.sma_20,
                "SMA_50": indicators.sma_50,
                "SMA_200": indicators.sma_200,
                "EMA_12": indicators.ema_12,
                "EMA_26": indicators.ema_26,
                "BB_upper": indicators.bb_upper,
                "BB_middle": indicators.bb_middle,
                "BB_lower": indicators.bb_lower,
                "ATR_14": indicators.atr_14,
                "Stochastic_K": indicators.stoch_k,
                "Stochastic_D": indicators.stoch_d,
                "ADX": indicators.adx,
            },
            "signals": signals,
        }

    def _screen_stocks(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._stock_screener is None:
            raise RuntimeError("Stock screener not available")
        return {"results": self._stock_screener.screen(dict(args))}

    def _get_market_status(self) -> Mapping[str, Any]:
        if self._exchange_registry is None:
            raise RuntimeError("Exchange registry not available")
        return {"exchanges": self._exchange_registry.get_all_statuses()}
