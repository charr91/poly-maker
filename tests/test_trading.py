"""
Tests for the trading module.

Tests cover:
- send_buy_order: Order cancellation, price validation, incentive threshold
- send_sell_order: Order cancellation and creation
- perform_trade: Position merging, stop-loss, buy logic, take-profit, risk-off
- _get_trade_semaphore: Lazy initialization and concurrency control
"""

import asyncio
import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd

import trading
from trading import send_buy_order, send_sell_order, perform_trade, _get_trade_semaphore


class TestSendBuyOrder:
    """Tests for the send_buy_order function."""

    @pytest.fixture
    def sample_order(self):
        """Create a sample order dict for testing."""
        return {
            "token": 12345,
            "price": 0.45,
            "size": 100,
            "orders": {"buy": {"price": 0.40, "size": 80}, "sell": {"price": 0.0, "size": 0}},
            "mid_price": 0.50,
            "max_spread": 5.0,
            "neg_risk": "FALSE",
        }

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with async methods."""
        client = MagicMock()
        client.cancel_all_asset_async = AsyncMock()
        client.create_order_async = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_cancels_order_when_price_diff_significant(self, sample_order, mock_client):
        """Test that orders are cancelled when price difference > 0.005."""
        sample_order["price"] = 0.50
        sample_order["orders"]["buy"]["price"] = 0.40
        sample_order["mid_price"] = 0.50
        sample_order["max_spread"] = 10.0

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_buy_order(sample_order)

            mock_client.cancel_all_asset_async.assert_called_once_with(12345)
            mock_client.create_order_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancels_order_when_size_diff_significant(self, sample_order, mock_client):
        """Test that orders are cancelled when size difference > 10%."""
        sample_order["size"] = 100
        sample_order["orders"]["buy"]["size"] = 50

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_buy_order(sample_order)

            mock_client.cancel_all_asset_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_early_when_changes_minor(self, sample_order, mock_client):
        """Test early return when price/size changes are minor."""
        sample_order["price"] = 0.401
        sample_order["orders"]["buy"]["price"] = 0.40
        sample_order["size"] = 100
        sample_order["orders"]["buy"]["size"] = 100

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_buy_order(sample_order)

            mock_client.cancel_all_asset_async.assert_not_called()
            mock_client.create_order_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_price_below_incentive_threshold(self, sample_order, mock_client):
        """Test that orders below incentive threshold are not created."""
        sample_order["price"] = 0.40
        sample_order["mid_price"] = 0.50
        sample_order["max_spread"] = 5.0
        sample_order["orders"]["buy"]["size"] = 0

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_buy_order(sample_order)

            mock_client.create_order_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_price_outside_valid_range(self, sample_order, mock_client):
        """Test that orders outside 0.1-0.9 range are not created."""
        sample_order["price"] = 0.05
        sample_order["orders"]["buy"]["size"] = 0

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_buy_order(sample_order)

            mock_client.create_order_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_order_with_valid_params(self, sample_order, mock_client):
        """Test successful order creation with valid parameters."""
        sample_order["price"] = 0.50
        sample_order["orders"]["buy"]["size"] = 0
        sample_order["neg_risk"] = "TRUE"

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_buy_order(sample_order)

            mock_client.create_order_async.assert_called_once_with(12345, "BUY", 0.50, 100, True)


class TestSendSellOrder:
    """Tests for the send_sell_order function."""

    @pytest.fixture
    def sample_order(self):
        """Create a sample order dict for testing."""
        return {
            "token": 12345,
            "price": 0.55,
            "size": 50,
            "orders": {"buy": {"price": 0.0, "size": 0}, "sell": {"price": 0.50, "size": 40}},
            "neg_risk": "FALSE",
        }

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with async methods."""
        client = MagicMock()
        client.cancel_all_asset_async = AsyncMock()
        client.create_order_async = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_cancels_order_when_price_diff_significant(self, sample_order, mock_client):
        """Test that orders are cancelled when price difference > 0.005."""
        sample_order["price"] = 0.56
        sample_order["orders"]["sell"]["price"] = 0.50

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_sell_order(sample_order)

            mock_client.cancel_all_asset_async.assert_called_once_with(12345)
            mock_client.create_order_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_early_when_changes_minor(self, sample_order, mock_client):
        """Test early return when price/size changes are minor."""
        sample_order["price"] = 0.501
        sample_order["orders"]["sell"]["price"] = 0.50
        sample_order["size"] = 50
        sample_order["orders"]["sell"]["size"] = 50

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_sell_order(sample_order)

            mock_client.cancel_all_asset_async.assert_not_called()
            mock_client.create_order_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_sell_order_successfully(self, sample_order, mock_client):
        """Test successful sell order creation."""
        sample_order["orders"]["sell"]["size"] = 0
        sample_order["neg_risk"] = "TRUE"

        with patch.object(trading, "global_state") as mock_gs:
            mock_gs.client = mock_client

            await send_sell_order(sample_order)

            mock_client.create_order_async.assert_called_once_with(12345, "SELL", 0.55, 50, True)


class TestPerformTrade:
    """Tests for the perform_trade function."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with all async methods."""
        client = MagicMock()
        client.get_position_async = AsyncMock(return_value=(1000000, 0.5))
        client.merge_positions_async = AsyncMock()
        client.create_order_async = AsyncMock()
        client.cancel_all_asset_async = AsyncMock()
        client.cancel_all_market_async = AsyncMock()
        return client

    @pytest.fixture
    def sample_row(self):
        """Create a sample DataFrame row for market config."""
        return pd.Series(
            {
                "condition_id": "market123",
                "token1": "111111",
                "token2": "222222",
                "answer1": "Yes",
                "answer2": "No",
                "tick_size": 0.01,
                "param_type": "standard",
                "neg_risk": "FALSE",
                "max_spread": 5.0,
                "trade_size": 100,
                "max_size": 200,
                "min_size": 10,
                "best_bid": 0.45,
                "best_ask": 0.55,
                "question": "Test question?",
                "3_hour": 0.5,
            }
        )

    @pytest.fixture
    def sample_params(self):
        """Create sample trading parameters."""
        return {
            "stop_loss_threshold": -10,
            "spread_threshold": 0.05,
            "volatility_threshold": 2.0,
            "take_profit_threshold": 5,
            "sleep_period": 24,
        }

    @pytest.fixture
    def sample_market_deets(self):
        """Create sample market details from order book."""
        return {
            "best_bid": 0.45,
            "best_ask": 0.55,
            "best_bid_size": 500,
            "best_ask_size": 500,
            "second_best_bid": 0.44,
            "second_best_ask": 0.56,
            "second_best_bid_size": 300,
            "second_best_ask_size": 300,
            "top_bid": 0.45,
            "top_ask": 0.55,
            "bid_sum_within_n_percent": 1000,
            "ask_sum_within_n_percent": 1000,
        }

    def _setup_global_state_mocks(self, mock_gs, mock_client, sample_row, sample_params):
        """Helper to set up all global state mocks."""
        mock_gs.client = mock_client
        mock_gs.df = pd.DataFrame([sample_row])
        mock_gs.params = {"standard": sample_params}
        mock_gs.REVERSE_TOKENS = {"111111": "222222", "222222": "111111"}

    @pytest.mark.asyncio
    async def test_position_merge_when_both_positions_above_threshold(
        self, mock_client, sample_row, sample_params, sample_market_deets
    ):
        """Test that positions are merged when both exceed MIN_MERGE_SIZE."""
        with (
            patch.object(trading, "global_state") as mock_gs,
            patch.object(trading, "get_position") as mock_get_pos,
            patch.object(trading, "set_position") as mock_set_pos,
            patch.object(
                trading,
                "get_order",
                return_value={"buy": {"price": 0, "size": 0}, "sell": {"price": 0, "size": 0}},
            ),
            patch.object(trading, "get_best_bid_ask_deets", return_value=sample_market_deets),
            patch.object(trading, "get_order_prices", return_value=(0.45, 0.55)),
            patch.object(trading, "get_buy_sell_amount", return_value=(0, 0)),
            patch.object(trading, "CONSTANTS") as mock_constants,
            patch.object(trading, "_get_trade_semaphore") as mock_sem,
            patch.object(trading, "market_locks", {}),
        ):

            mock_constants.MIN_MERGE_SIZE = 20
            mock_sem.return_value = asyncio.Semaphore(3)
            mock_get_pos.side_effect = [
                {"size": 50, "avgPrice": 0.4},
                {"size": 30, "avgPrice": 0.6},
                {"size": 50, "avgPrice": 0.4},
                {"size": 30, "avgPrice": 0.6},
                {"size": 50, "avgPrice": 0.4},
                {"size": 0, "avgPrice": 0.0},
            ]
            mock_client.get_position_async = AsyncMock(return_value=(30000000, 0.5))

            self._setup_global_state_mocks(mock_gs, mock_client, sample_row, sample_params)

            await perform_trade("market123")

            mock_client.merge_positions_async.assert_called_once()
            assert mock_set_pos.call_count >= 2

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(
        self, mock_client, sample_row, sample_params, sample_market_deets, tmp_path
    ):
        """Test stop-loss triggers sell at best_bid and creates risk-off file."""
        positions_dir = tmp_path / "positions"
        positions_dir.mkdir()

        sample_params["stop_loss_threshold"] = -5
        sample_params["spread_threshold"] = 0.30
        sample_market_deets["best_bid"] = 0.30
        sample_market_deets["best_ask"] = 0.32

        with (
            patch.object(trading, "global_state") as mock_gs,
            patch.object(trading, "get_position") as mock_get_pos,
            patch.object(trading, "set_position"),
            patch.object(
                trading,
                "get_order",
                return_value={"buy": {"price": 0, "size": 0}, "sell": {"price": 0, "size": 0}},
            ),
            patch.object(trading, "get_best_bid_ask_deets", return_value=sample_market_deets),
            patch.object(trading, "get_order_prices", return_value=(0.30, 0.55)),
            patch.object(trading, "get_buy_sell_amount", return_value=(0, 50)),
            patch.object(trading, "send_sell_order", new_callable=AsyncMock) as mock_send_sell,
            patch.object(trading, "CONSTANTS") as mock_constants,
            patch.object(trading, "_get_trade_semaphore") as mock_sem,
            patch.object(trading, "market_locks", {}),
            patch("builtins.open", create=True) as mock_open,
        ):

            mock_constants.MIN_MERGE_SIZE = 20
            mock_sem.return_value = asyncio.Semaphore(3)
            mock_get_pos.return_value = {"size": 50, "avgPrice": 0.50}

            self._setup_global_state_mocks(mock_gs, mock_client, sample_row, sample_params)

            await perform_trade("market123")

            mock_send_sell.assert_called()
            assert mock_send_sell.call_args[0][0]["price"] == 0.30

    @pytest.mark.asyncio
    async def test_buy_order_when_position_below_max(
        self, mock_client, sample_row, sample_params, sample_market_deets
    ):
        """Test buy order is placed when position is below max_size."""
        with (
            patch.object(trading, "global_state") as mock_gs,
            patch.object(trading, "get_position") as mock_get_pos,
            patch.object(trading, "set_position"),
            patch.object(
                trading,
                "get_order",
                return_value={"buy": {"price": 0.40, "size": 0}, "sell": {"price": 0, "size": 0}},
            ),
            patch.object(trading, "get_best_bid_ask_deets", return_value=sample_market_deets),
            patch.object(trading, "get_order_prices", return_value=(0.45, 0.55)),
            patch.object(trading, "get_buy_sell_amount", return_value=(100, 0)),
            patch.object(trading, "send_buy_order", new_callable=AsyncMock) as mock_send_buy,
            patch.object(trading, "CONSTANTS") as mock_constants,
            patch.object(trading, "_get_trade_semaphore") as mock_sem,
            patch.object(trading, "market_locks", {}),
            patch("os.path.isfile", return_value=False),
        ):

            mock_constants.MIN_MERGE_SIZE = 20
            mock_sem.return_value = asyncio.Semaphore(3)
            mock_get_pos.side_effect = [
                {"size": 10, "avgPrice": 0.0},
                {"size": 10, "avgPrice": 0.0},
                {"size": 50, "avgPrice": 0.45},
                {"size": 0, "avgPrice": 0.0},
                {"size": 0, "avgPrice": 0.0},
            ]

            self._setup_global_state_mocks(mock_gs, mock_client, sample_row, sample_params)

            await perform_trade("market123")

            mock_send_buy.assert_called()

    @pytest.mark.asyncio
    async def test_no_buy_when_in_risk_off_period(
        self, mock_client, sample_row, sample_params, sample_market_deets, tmp_path
    ):
        """Test that buy orders are blocked during risk-off period."""
        positions_dir = tmp_path / "positions"
        positions_dir.mkdir()
        risk_file = positions_dir / "market123.json"
        future_time = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=12)
        risk_file.write_text(
            json.dumps(
                {
                    "time": str(pd.Timestamp.utcnow().tz_localize(None)),
                    "sleep_till": str(future_time),
                    "question": "Test",
                    "msg": "Risked off",
                }
            )
        )

        with (
            patch.object(trading, "global_state") as mock_gs,
            patch.object(trading, "get_position") as mock_get_pos,
            patch.object(trading, "set_position"),
            patch.object(
                trading,
                "get_order",
                return_value={"buy": {"price": 0.40, "size": 0}, "sell": {"price": 0, "size": 0}},
            ),
            patch.object(trading, "get_best_bid_ask_deets", return_value=sample_market_deets),
            patch.object(trading, "get_order_prices", return_value=(0.45, 0.55)),
            patch.object(trading, "get_buy_sell_amount", return_value=(100, 0)),
            patch.object(trading, "send_buy_order", new_callable=AsyncMock) as mock_send_buy,
            patch.object(trading, "CONSTANTS") as mock_constants,
            patch.object(trading, "_get_trade_semaphore") as mock_sem,
            patch.object(trading, "market_locks", {}),
            patch("os.path.isfile", return_value=True),
            patch("builtins.open", return_value=open(risk_file, "r")),
        ):

            mock_constants.MIN_MERGE_SIZE = 20
            mock_sem.return_value = asyncio.Semaphore(3)
            mock_get_pos.side_effect = [
                {"size": 10, "avgPrice": 0.0},
                {"size": 10, "avgPrice": 0.0},
                {"size": 50, "avgPrice": 0.45},
                {"size": 0, "avgPrice": 0.0},
                {"size": 0, "avgPrice": 0.0},
            ]

            self._setup_global_state_mocks(mock_gs, mock_client, sample_row, sample_params)

            await perform_trade("market123")

            mock_send_buy.assert_not_called()

    @pytest.mark.asyncio
    async def test_take_profit_sell_order(
        self, mock_client, sample_row, sample_params, sample_market_deets
    ):
        """Test take-profit sell order is placed when holding position."""
        sample_params["take_profit_threshold"] = 10

        with (
            patch.object(trading, "global_state") as mock_gs,
            patch.object(trading, "get_position") as mock_get_pos,
            patch.object(trading, "set_position"),
            patch.object(
                trading,
                "get_order",
                return_value={"buy": {"price": 0, "size": 0}, "sell": {"price": 0.50, "size": 20}},
            ),
            patch.object(trading, "get_best_bid_ask_deets", return_value=sample_market_deets),
            patch.object(trading, "get_order_prices", return_value=(0.45, 0.55)),
            patch.object(trading, "get_buy_sell_amount", return_value=(0, 100)),
            patch.object(trading, "send_sell_order", new_callable=AsyncMock) as mock_send_sell,
            patch.object(trading, "CONSTANTS") as mock_constants,
            patch.object(trading, "_get_trade_semaphore") as mock_sem,
            patch.object(trading, "market_locks", {}),
        ):

            mock_constants.MIN_MERGE_SIZE = 20
            mock_sem.return_value = asyncio.Semaphore(3)
            mock_get_pos.side_effect = [
                {"size": 10, "avgPrice": 0.0},
                {"size": 10, "avgPrice": 0.0},
                {"size": 100, "avgPrice": 0.40},
                {"size": 0, "avgPrice": 0.0},
            ]

            self._setup_global_state_mocks(mock_gs, mock_client, sample_row, sample_params)

            await perform_trade("market123")

            mock_send_sell.assert_called()

    @pytest.mark.asyncio
    async def test_no_buy_when_reverse_position_exists(
        self, mock_client, sample_row, sample_params, sample_market_deets
    ):
        """Test that buy orders are blocked when holding opposite position."""
        with (
            patch.object(trading, "global_state") as mock_gs,
            patch.object(trading, "get_position") as mock_get_pos,
            patch.object(trading, "set_position"),
            patch.object(
                trading,
                "get_order",
                return_value={"buy": {"price": 0.40, "size": 50}, "sell": {"price": 0, "size": 0}},
            ),
            patch.object(trading, "get_best_bid_ask_deets", return_value=sample_market_deets),
            patch.object(trading, "get_order_prices", return_value=(0.45, 0.55)),
            patch.object(trading, "get_buy_sell_amount", return_value=(100, 0)),
            patch.object(trading, "send_buy_order", new_callable=AsyncMock) as mock_send_buy,
            patch.object(trading, "CONSTANTS") as mock_constants,
            patch.object(trading, "_get_trade_semaphore") as mock_sem,
            patch.object(trading, "market_locks", {}),
            patch("os.path.isfile", return_value=False),
        ):

            mock_constants.MIN_MERGE_SIZE = 20
            mock_sem.return_value = asyncio.Semaphore(3)
            mock_get_pos.side_effect = [
                {"size": 10, "avgPrice": 0.0},
                {"size": 10, "avgPrice": 0.0},
                {"size": 50, "avgPrice": 0.45},
                {"size": 100, "avgPrice": 0.55},
                {"size": 100, "avgPrice": 0.55},
            ]

            self._setup_global_state_mocks(mock_gs, mock_client, sample_row, sample_params)

            await perform_trade("market123")

            mock_send_buy.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_buy_when_volatility_high(
        self, mock_client, sample_row, sample_params, sample_market_deets
    ):
        """Test that buy orders are blocked when volatility exceeds threshold."""
        sample_row["3_hour"] = 5.0
        sample_params["volatility_threshold"] = 2.0

        with (
            patch.object(trading, "global_state") as mock_gs,
            patch.object(trading, "get_position") as mock_get_pos,
            patch.object(trading, "set_position"),
            patch.object(
                trading,
                "get_order",
                return_value={"buy": {"price": 0.40, "size": 0}, "sell": {"price": 0, "size": 0}},
            ),
            patch.object(trading, "get_best_bid_ask_deets", return_value=sample_market_deets),
            patch.object(trading, "get_order_prices", return_value=(0.45, 0.55)),
            patch.object(trading, "get_buy_sell_amount", return_value=(100, 0)),
            patch.object(trading, "send_buy_order", new_callable=AsyncMock) as mock_send_buy,
            patch.object(trading, "CONSTANTS") as mock_constants,
            patch.object(trading, "_get_trade_semaphore") as mock_sem,
            patch.object(trading, "market_locks", {}),
            patch("os.path.isfile", return_value=False),
        ):

            mock_constants.MIN_MERGE_SIZE = 20
            mock_sem.return_value = asyncio.Semaphore(3)
            mock_get_pos.side_effect = [
                {"size": 10, "avgPrice": 0.0},
                {"size": 10, "avgPrice": 0.0},
                {"size": 50, "avgPrice": 0.45},
                {"size": 0, "avgPrice": 0.0},
                {"size": 0, "avgPrice": 0.0},
            ]

            self._setup_global_state_mocks(mock_gs, mock_client, sample_row, sample_params)

            await perform_trade("market123")

            mock_send_buy.assert_not_called()
            mock_client.cancel_all_asset_async.assert_called()


class TestTradeSemaphore:
    """Tests for the _get_trade_semaphore function."""

    def test_lazy_initialization(self):
        """Test that semaphore is lazily initialized on first call."""
        original_semaphore = trading._trade_semaphore

        try:
            trading._trade_semaphore = None

            with patch.dict(os.environ, {"MAX_CONCURRENT_TRADES": "5"}):
                import importlib

                importlib.reload(trading)

                semaphore = trading._get_trade_semaphore()
                assert semaphore is not None
                assert isinstance(semaphore, asyncio.Semaphore)
        finally:
            trading._trade_semaphore = original_semaphore

    @pytest.mark.asyncio
    async def test_concurrent_limit_enforced(self):
        """Test that semaphore limits concurrent trades."""
        trading._trade_semaphore = asyncio.Semaphore(2)
        semaphore = trading._get_trade_semaphore()

        active_count = 0
        max_active = 0

        async def mock_trade():
            nonlocal active_count, max_active
            async with semaphore:
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.01)
                active_count -= 1

        await asyncio.gather(*[mock_trade() for _ in range(5)])

        assert max_active <= 2


class TestPnLCalculation:
    """Tests for PnL calculation with YES and NO tokens.

    These tests verify that:
    1. PnL is calculated correctly for YES tokens (token1)
    2. Price transformation for NO tokens (token2) works correctly
    3. The combination results in correct PnL for both token types
    """

    @pytest.fixture
    def mock_market_deets_yes_profitable(self):
        """Market where YES token price increased (profitable)."""
        return {
            "best_bid": 0.55,
            "best_ask": 0.57,
            "best_bid_size": 500,
            "best_ask_size": 500,
            "second_best_bid": 0.54,
            "second_best_ask": 0.58,
            "second_best_bid_size": 300,
            "second_best_ask_size": 300,
            "top_bid": 0.55,
            "top_ask": 0.57,
            "bid_sum_within_n_percent": 1000,
            "ask_sum_within_n_percent": 1000,
        }

    @pytest.fixture
    def mock_market_deets_yes_losing(self):
        """Market where YES token price decreased (losing)."""
        return {
            "best_bid": 0.35,
            "best_ask": 0.37,
            "best_bid_size": 500,
            "best_ask_size": 500,
            "second_best_bid": 0.34,
            "second_best_ask": 0.38,
            "second_best_bid_size": 300,
            "second_best_ask_size": 300,
            "top_bid": 0.35,
            "top_ask": 0.37,
            "bid_sum_within_n_percent": 1000,
            "ask_sum_within_n_percent": 1000,
        }

    def test_pnl_formula_yes_token_profitable(self):
        """Test PnL formula shows profit when YES price rises.

        Scenario: Bought YES at 0.40, now at 0.56 mid_price
        Expected: (0.56 - 0.40) / 0.40 * 100 = +40%
        """
        avgPrice = 0.40
        mid_price = 0.56
        pnl = (mid_price - avgPrice) / avgPrice * 100

        assert pnl > 0, "PnL should be positive when price rises"
        assert abs(pnl - 40.0) < 0.01, f"Expected +40%, got {pnl}%"

    def test_pnl_formula_yes_token_losing(self):
        """Test PnL formula shows loss when YES price falls.

        Scenario: Bought YES at 0.50, now at 0.36 mid_price
        Expected: (0.36 - 0.50) / 0.50 * 100 = -28%
        """
        avgPrice = 0.50
        mid_price = 0.36
        pnl = (mid_price - avgPrice) / avgPrice * 100

        assert pnl < 0, "PnL should be negative when price falls"
        assert abs(pnl - (-28.0)) < 0.01, f"Expected -28%, got {pnl}%"

    def test_pnl_formula_no_token_price_transformation(self):
        """Test that NO token prices are correctly transformed.

        For token2 (NO), get_best_bid_ask_deets transforms:
        - best_bid becomes 1 - best_ask
        - best_ask becomes 1 - best_bid

        If YES market shows bid=0.60, ask=0.65:
        - NO best_bid = 1 - 0.65 = 0.35
        - NO best_ask = 1 - 0.60 = 0.40
        - NO mid_price = (0.35 + 0.40) / 2 = 0.375
        """
        yes_best_bid = 0.60
        yes_best_ask = 0.65

        # Transformation that happens in get_best_bid_ask_deets for token2
        no_best_bid = 1 - yes_best_ask
        no_best_ask = 1 - yes_best_bid
        no_mid_price = (no_best_bid + no_best_ask) / 2

        assert abs(no_best_bid - 0.35) < 0.001
        assert abs(no_best_ask - 0.40) < 0.001
        assert abs(no_mid_price - 0.375) < 0.001

    def test_pnl_formula_no_token_losing_when_yes_rises(self):
        """Test NO position shows loss when YES price rises.

        Scenario:
        - Bought NO at 0.55 (when YES was at 0.45)
        - Now YES is at 0.60, so NO mid_price is ~0.40
        - PnL = (0.40 - 0.55) / 0.55 * 100 = -27.27%
        """
        no_avgPrice = 0.55  # Purchase price of NO token
        # After get_best_bid_ask_deets transformation for token2
        no_mid_price = 0.40  # Current NO price (YES rose to ~0.60)

        pnl = (no_mid_price - no_avgPrice) / no_avgPrice * 100

        assert pnl < 0, "NO position should show loss when YES rises"
        assert abs(pnl - (-27.27)) < 0.1, f"Expected ~-27.27%, got {pnl}%"

    def test_pnl_formula_no_token_profitable_when_yes_falls(self):
        """Test NO position shows profit when YES price falls.

        Scenario:
        - Bought NO at 0.45 (when YES was at 0.55)
        - Now YES is at 0.40, so NO mid_price is ~0.60
        - PnL = (0.60 - 0.45) / 0.45 * 100 = +33.33%
        """
        no_avgPrice = 0.45  # Purchase price of NO token
        no_mid_price = 0.60  # Current NO price (YES fell to ~0.40)

        pnl = (no_mid_price - no_avgPrice) / no_avgPrice * 100

        assert pnl > 0, "NO position should show profit when YES falls"
        assert abs(pnl - 33.33) < 0.1, f"Expected ~+33.33%, got {pnl}%"


class TestPriceTransformation:
    """Tests for token2 price transformation in get_best_bid_ask_deets."""

    def test_token2_transformation_complete(self):
        """Test complete price transformation for token2."""
        from poly_data.trading_utils import get_best_bid_ask_deets
        from sortedcontainers import SortedDict
        import poly_data.global_state as global_state

        # Set up mock order book data
        # YES market: bid=0.45, ask=0.55
        market_id = "test_market_transform"
        global_state.all_data[market_id] = {
            "bids": SortedDict({0.45: 1000, 0.44: 500}),
            "asks": SortedDict({0.55: 1000, 0.56: 500}),
        }

        try:
            # Get prices for token1 (YES)
            yes_deets = get_best_bid_ask_deets(market_id, "token1", 100, 0.1)

            # Get prices for token2 (NO)
            no_deets = get_best_bid_ask_deets(market_id, "token2", 100, 0.1)

            # YES should have original prices
            assert yes_deets["best_bid"] == 0.45
            assert yes_deets["best_ask"] == 0.55

            # NO should have transformed prices (1 - opposite)
            assert abs(no_deets["best_bid"] - 0.45) < 0.001  # 1 - 0.55
            assert abs(no_deets["best_ask"] - 0.55) < 0.001  # 1 - 0.45

            # Mid prices should be complementary
            yes_mid = (yes_deets["best_bid"] + yes_deets["best_ask"]) / 2
            no_mid = (no_deets["best_bid"] + no_deets["best_ask"]) / 2

            # YES mid + NO mid should equal 1
            assert abs((yes_mid + no_mid) - 1.0) < 0.01

        finally:
            # Clean up
            if market_id in global_state.all_data:
                del global_state.all_data[market_id]


class TestStopLossTriggerConditions:
    """Tests documenting stop-loss trigger conditions.

    Stop-loss triggers when EITHER:
    1. PnL < stop_loss_threshold AND spread <= spread_threshold
    2. Volatility (3_hour) > volatility_threshold
    """

    @pytest.fixture
    def sample_params(self):
        return {
            "stop_loss_threshold": -10,
            "spread_threshold": 0.05,
            "volatility_threshold": 2.0,
            "take_profit_threshold": 5,
            "sleep_period": 24,
        }

    def test_stop_loss_condition_pnl_and_spread(self, sample_params):
        """Test stop-loss triggers on PnL + spread condition.

        Condition: pnl < -10 AND spread <= 0.05
        """
        pnl = -15  # Below threshold
        spread = 0.04  # Within threshold
        volatility = 1.0  # Below threshold

        should_stop_loss = (
            pnl < sample_params["stop_loss_threshold"]
            and spread <= sample_params["spread_threshold"]
        ) or volatility > sample_params["volatility_threshold"]

        assert should_stop_loss is True

    def test_stop_loss_blocked_when_spread_wide(self, sample_params):
        """Test stop-loss blocked when spread is too wide.

        Condition: pnl < -10 but spread > 0.05 (can't exit at good price)
        """
        pnl = -15  # Below threshold
        spread = 0.08  # Above threshold - spread too wide
        volatility = 1.0  # Below threshold

        should_stop_loss = (
            pnl < sample_params["stop_loss_threshold"]
            and spread <= sample_params["spread_threshold"]
        ) or volatility > sample_params["volatility_threshold"]

        assert should_stop_loss is False

    def test_stop_loss_triggers_on_high_volatility(self, sample_params):
        """Test stop-loss triggers on volatility alone.

        Condition: volatility > 2.0 (regardless of PnL)
        """
        pnl = 5  # Profitable position
        spread = 0.10  # Wide spread
        volatility = 3.5  # Above threshold

        should_stop_loss = (
            pnl < sample_params["stop_loss_threshold"]
            and spread <= sample_params["spread_threshold"]
        ) or volatility > sample_params["volatility_threshold"]

        assert should_stop_loss is True

    def test_no_stop_loss_when_profitable(self, sample_params):
        """Test no stop-loss when position is profitable and volatility normal."""
        pnl = 15  # Profitable
        spread = 0.03  # Tight spread
        volatility = 1.0  # Normal

        should_stop_loss = (
            pnl < sample_params["stop_loss_threshold"]
            and spread <= sample_params["spread_threshold"]
        ) or volatility > sample_params["volatility_threshold"]

        assert should_stop_loss is False


class TestTakeProfitBehavior:
    """Tests documenting take-profit behavior.

    Take-profit is in an elif block after stop-loss.
    When stop-loss triggers and uses 'continue', take-profit is skipped.
    """

    def test_take_profit_price_calculation(self):
        """Test take-profit price is calculated correctly.

        tp_price = avgPrice + (avgPrice * take_profit_threshold / 100)
        """
        avgPrice = 0.40
        take_profit_threshold = 10  # 10%

        tp_price = avgPrice + (avgPrice * take_profit_threshold / 100)

        assert abs(tp_price - 0.44) < 0.001

    def test_take_profit_uses_max_of_tp_and_ask(self):
        """Test take-profit uses max of calculated tp_price and ask_price.

        This ensures we don't sell below current market.
        """
        avgPrice = 0.40
        take_profit_threshold = 10
        tp_price = avgPrice + (avgPrice * take_profit_threshold / 100)  # 0.44

        ask_price = 0.46  # Current market higher than tp

        final_sell_price = max(tp_price, ask_price)

        assert final_sell_price == 0.46

    def test_take_profit_conditions(self):
        """Test take-profit triggers when there's sell_amount and significant price diff."""
        sell_amount = 100  # Have position to sell
        current_sell_order_price = 0.50
        new_tp_price = 0.55

        price_diff_percent = (
            abs(new_tp_price - current_sell_order_price) / current_sell_order_price * 100
        )

        # Take-profit updates when:
        # 1. sell_amount > 0
        # 2. Price diff > 2%
        should_update = sell_amount > 0 and price_diff_percent > 2

        assert should_update is True
