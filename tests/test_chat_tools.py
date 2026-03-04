"""
Tests for Chat Tool Handlers (Phases 2-6)

Tests cover:
- get_technical_analysis tool
- get_orders, cancel_order, get_position_details tools
- screen_stocks tool
- get_market_status tool
- run_backtest tool
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from decimal import Decimal

from src.application.use_cases.chat import ChatUseCase
from src.domain.entities.trading import Order, OrderStatus, OrderType, PositionType, Position, Portfolio
from src.domain.value_objects import Symbol, Money
from src.domain.services.technical_analysis import TechnicalIndicators


def _make_chat_use_case(**overrides):
    """Helper to build a ChatUseCase with mocked dependencies."""
    defaults = {
        "ai_chat_port": MagicMock(),
        "conversation_repository": MagicMock(),
        "market_data_service": MagicMock(),
        "ai_model_service": MagicMock(),
        "news_analysis_service": MagicMock(),
        "portfolio_repository": MagicMock(),
        "user_repository": MagicMock(),
    }
    defaults.update(overrides)
    return ChatUseCase(**defaults)


# ── Phase 2: Technical Analysis ──────────────────────────────────────────

class TestTechnicalAnalysisTool:
    @pytest.mark.asyncio
    async def test_returns_indicators(self):
        mock_ta = MagicMock()
        mock_ta.compute_indicators.return_value = TechnicalIndicators(
            symbol="AAPL", rsi_14=45.0, macd_histogram=0.5, current_price=150.0,
            sma_50=145.0,
        )
        uc = _make_chat_use_case(technical_analysis_port=mock_ta)
        result = await uc._tool_get_technical_analysis({"symbol": "AAPL"})
        data = json.loads(result.content)
        assert data["symbol"] == "AAPL"
        assert data["indicators"]["RSI_14"] == 45.0
        assert "signals" in data

    @pytest.mark.asyncio
    async def test_missing_symbol(self):
        uc = _make_chat_use_case(technical_analysis_port=MagicMock())
        result = await uc._tool_get_technical_analysis({})
        assert result.is_error

    @pytest.mark.asyncio
    async def test_no_adapter(self):
        uc = _make_chat_use_case()
        result = await uc._tool_get_technical_analysis({"symbol": "AAPL"})
        assert result.is_error


# ── Phase 3: Order Management ────────────────────────────────────────────

def _make_order(status=OrderStatus.PENDING, user_id="user1"):
    return Order(
        id="order-1",
        user_id=user_id,
        symbol=Symbol("AAPL"),
        order_type=OrderType.MARKET,
        position_type=PositionType.LONG,
        quantity=10,
        status=status,
        placed_at=datetime.utcnow(),
        price=Money(Decimal("150.00"), "USD"),
    )


class TestGetOrdersTool:
    @pytest.mark.asyncio
    async def test_returns_all_orders(self):
        mock_repo = MagicMock()
        mock_repo.get_by_user_id.return_value = [_make_order()]
        uc = _make_chat_use_case(order_repository=mock_repo)
        result = await uc._tool_get_orders({}, "user1")
        data = json.loads(result.content)
        assert data["total"] == 1
        assert data["orders"][0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_filter_by_status(self):
        mock_repo = MagicMock()
        mock_repo.get_by_user_id.return_value = [
            _make_order(status=OrderStatus.PENDING),
            _make_order(status=OrderStatus.EXECUTED),
        ]
        uc = _make_chat_use_case(order_repository=mock_repo)
        result = await uc._tool_get_orders({"status": "PENDING"}, "user1")
        data = json.loads(result.content)
        assert all(o["status"] == "PENDING" for o in data["orders"])

    @pytest.mark.asyncio
    async def test_no_order_repo(self):
        uc = _make_chat_use_case()
        result = await uc._tool_get_orders({}, "user1")
        assert result.is_error


class TestCancelOrderTool:
    @pytest.mark.asyncio
    async def test_cancel_pending_order(self):
        mock_repo = MagicMock()
        order = _make_order()
        mock_repo.get_by_id.return_value = order
        cancelled = _make_order(status=OrderStatus.CANCELLED)
        mock_repo.update_status.return_value = cancelled
        uc = _make_chat_use_case(order_repository=mock_repo)
        result = await uc._tool_cancel_order({"order_id": "order-1"}, "user1")
        data = json.loads(result.content)
        assert data["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cannot_cancel_executed(self):
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = _make_order(status=OrderStatus.EXECUTED)
        uc = _make_chat_use_case(order_repository=mock_repo)
        result = await uc._tool_cancel_order({"order_id": "order-1"}, "user1")
        assert result.is_error

    @pytest.mark.asyncio
    async def test_unauthorized(self):
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = _make_order(user_id="other")
        uc = _make_chat_use_case(order_repository=mock_repo)
        result = await uc._tool_cancel_order({"order_id": "order-1"}, "user1")
        assert result.is_error


class TestGetPositionDetailsTool:
    @pytest.mark.asyncio
    async def test_returns_position(self):
        pos = Position(
            id="pos-1", user_id="user1", symbol=Symbol("AAPL"),
            position_type=PositionType.LONG, quantity=10,
            average_buy_price=Money(Decimal("140.00"), "USD"),
            current_price=Money(Decimal("150.00"), "USD"),
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        )
        portfolio = Portfolio(id="pf-1", user_id="user1", positions=[pos])
        mock_pf = MagicMock()
        mock_pf.get_by_user_id.return_value = portfolio
        uc = _make_chat_use_case(portfolio_repository=mock_pf)
        result = await uc._tool_get_position_details({"symbol": "AAPL"}, "user1")
        data = json.loads(result.content)
        assert data["symbol"] == "AAPL"
        assert data["quantity"] == 10

    @pytest.mark.asyncio
    async def test_no_position(self):
        portfolio = Portfolio(id="pf-1", user_id="user1", positions=[])
        mock_pf = MagicMock()
        mock_pf.get_by_user_id.return_value = portfolio
        uc = _make_chat_use_case(portfolio_repository=mock_pf)
        result = await uc._tool_get_position_details({"symbol": "AAPL"}, "user1")
        assert result.is_error


# ── Phase 1: Stock Screening ─────────────────────────────────────────────

class TestScreenStocksTool:
    @pytest.mark.asyncio
    async def test_delegates_to_screener(self):
        mock_screener = MagicMock()
        mock_screener.screen.return_value = {"screen": "top_gainers", "count": 1, "results": []}
        uc = _make_chat_use_case(stock_screener=mock_screener)
        result = await uc._tool_screen_stocks({"prebuilt_screen": "top_gainers"})
        data = json.loads(result.content)
        assert data["screen"] == "top_gainers"

    @pytest.mark.asyncio
    async def test_no_screener(self):
        uc = _make_chat_use_case()
        result = await uc._tool_screen_stocks({})
        assert result.is_error


# ── Phase 4: Market Status ───────────────────────────────────────────────

class TestGetMarketStatusTool:
    @pytest.mark.asyncio
    async def test_returns_statuses(self):
        mock_registry = MagicMock()
        mock_registry.get_all_statuses.return_value = {
            "exchanges": [{"code": "NYSE", "is_open": True}]
        }
        uc = _make_chat_use_case(exchange_registry=mock_registry)
        result = await uc._tool_get_market_status()
        data = json.loads(result.content)
        assert len(data["exchanges"]) == 1

    @pytest.mark.asyncio
    async def test_no_registry(self):
        uc = _make_chat_use_case()
        result = await uc._tool_get_market_status()
        assert result.is_error


# ── Phase 6: Backtesting ─────────────────────────────────────────────────

class TestRunBacktestTool:
    @pytest.mark.asyncio
    async def test_delegates_to_use_case(self):
        mock_bt = MagicMock()
        mock_bt.run.return_value = {"strategy": "SMA", "total_return_pct": 5.0}
        uc = _make_chat_use_case(backtest_use_case=mock_bt)
        result = await uc._tool_run_backtest({"strategy": "sma_crossover", "symbol": "AAPL"})
        data = json.loads(result.content)
        assert data["total_return_pct"] == 5.0

    @pytest.mark.asyncio
    async def test_missing_args(self):
        uc = _make_chat_use_case(backtest_use_case=MagicMock())
        result = await uc._tool_run_backtest({"strategy": "sma_crossover"})
        assert result.is_error

    @pytest.mark.asyncio
    async def test_no_use_case(self):
        uc = _make_chat_use_case()
        result = await uc._tool_run_backtest({"strategy": "sma_crossover", "symbol": "AAPL"})
        assert result.is_error
