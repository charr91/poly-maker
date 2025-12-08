"""
Tests for market cleanup functionality when markets are removed from Google Sheets.

Tests cover:
- detect_removed_markets: Comparison logic
- add_to_pending_removal: Grace period queueing
- process_pending_removals: Grace period enforcement
- cleanup_market: Order cancellation, position selling, state cleanup
- close_positions: P&L-based closing logic
- Configuration via environment variables
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pandas as pd

import poly_data.data_utils as data_utils
import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS
from poly_data.merge_circuit_breaker import MergeCircuitBreaker


def _create_sample_df(markets):
    """Create a sample DataFrame with market data."""
    rows = []
    for m in markets:
        rows.append(
            {
                "condition_id": m["condition_id"],
                "token1": m.get("token1", f"{m['condition_id']}_token1"),
                "token2": m.get("token2", f"{m['condition_id']}_token2"),
                "question": m.get("question", f"Question for {m['condition_id']}"),
                "neg_risk": m.get("neg_risk", "FALSE"),
            }
        )
    return pd.DataFrame(rows)


class TestDetectRemovedMarkets:
    """Tests for detect_removed_markets function."""

    def test_detects_single_removed_market(self):
        """Test detection of one removed market."""
        with patch.object(data_utils, "global_state") as mock_gs:
            # Current state has 2 markets
            mock_gs.df = _create_sample_df(
                [{"condition_id": "market1"}, {"condition_id": "market2"}]
            )

            # New state has only 1 market
            new_df = _create_sample_df([{"condition_id": "market1"}])

            removed = data_utils.detect_removed_markets(new_df)

            assert removed == {"market2"}

    def test_detects_multiple_removed_markets(self):
        """Test detection of multiple removed markets."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.df = _create_sample_df(
                [
                    {"condition_id": "market1"},
                    {"condition_id": "market2"},
                    {"condition_id": "market3"},
                ]
            )

            new_df = _create_sample_df([{"condition_id": "market2"}])

            removed = data_utils.detect_removed_markets(new_df)

            assert removed == {"market1", "market3"}

    def test_no_change_returns_empty_set(self):
        """Test no changes detected when markets unchanged."""
        with patch.object(data_utils, "global_state") as mock_gs:
            markets = [{"condition_id": "market1"}, {"condition_id": "market2"}]
            mock_gs.df = _create_sample_df(markets)

            new_df = _create_sample_df(markets)

            removed = data_utils.detect_removed_markets(new_df)

            assert removed == set()

    def test_handles_empty_current_df(self):
        """Test graceful handling when current df is empty."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.df = pd.DataFrame()

            new_df = _create_sample_df([{"condition_id": "market1"}])

            removed = data_utils.detect_removed_markets(new_df)

            assert removed == set()

    def test_handles_none_current_df(self):
        """Test graceful handling when current df is None."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.df = None

            new_df = _create_sample_df([{"condition_id": "market1"}])

            removed = data_utils.detect_removed_markets(new_df)

            assert removed == set()


class TestGracePeriod:
    """Tests for grace period functionality."""

    def test_market_queued_with_timestamp(self):
        """Test that removed markets are queued with timestamp."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.pending_removal = {}

            condition_id = "test_market"
            tokens = ["token1", "token2"]

            data_utils.add_to_pending_removal(condition_id, tokens, "Test Question", False)

            assert condition_id in mock_gs.pending_removal
            removal_info = mock_gs.pending_removal[condition_id]
            assert "timestamp" in removal_info
            assert removal_info["tokens"] == tokens
            assert removal_info["question"] == "Test Question"
            assert removal_info["neg_risk"] is False

    def test_duplicate_add_ignored(self):
        """Test that adding same market twice is ignored."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.pending_removal = {}

            condition_id = "test_market"
            tokens = ["token1", "token2"]

            # First add
            data_utils.add_to_pending_removal(condition_id, tokens, "First", False)
            first_timestamp = mock_gs.pending_removal[condition_id]["timestamp"]

            # Wait a bit
            time.sleep(0.01)

            # Second add - should be ignored
            data_utils.add_to_pending_removal(condition_id, tokens, "Second", True)

            # Should still have first values
            assert mock_gs.pending_removal[condition_id]["question"] == "First"
            assert mock_gs.pending_removal[condition_id]["timestamp"] == first_timestamp

    @pytest.mark.asyncio
    async def test_cleanup_not_triggered_before_grace_period(self):
        """Test cleanup doesn't execute immediately when market removed."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_constants.CLEANUP_GRACE_PERIOD = 30

            condition_id = "test_market"
            mock_gs.pending_removal = {
                condition_id: {
                    "timestamp": time.time(),  # Just added
                    "tokens": ["token1", "token2"],
                    "question": "Test",
                    "neg_risk": False,
                }
            }

            with patch.object(data_utils, "cleanup_market", new_callable=AsyncMock) as mock_cleanup:
                await data_utils.process_pending_removals()

                # Should not have been called (grace period not elapsed)
                mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_executes_after_grace_period(self):
        """Test cleanup executes after grace period expires."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_constants.CLEANUP_GRACE_PERIOD = 0.01  # 10ms

            condition_id = "test_market"
            mock_gs.pending_removal = {
                condition_id: {
                    "timestamp": time.time() - 1,  # 1 second ago
                    "tokens": ["token1", "token2"],
                    "question": "Test",
                    "neg_risk": False,
                }
            }

            with patch.object(data_utils, "cleanup_market", new_callable=AsyncMock) as mock_cleanup:
                await data_utils.process_pending_removals()

                # Should have been called
                mock_cleanup.assert_called_once()

                # Should have been removed from pending
                assert condition_id not in mock_gs.pending_removal


class TestPositionClosing:
    """Tests for position closing based on P&L."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with async methods."""
        client = MagicMock()
        client.create_order_async = AsyncMock()
        client.merge_positions_async = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_in_profit_position_sells_at_market(self, mock_client):
        """Test that in-profit positions sell at best bid (market)."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_gs.client = mock_client
            mock_constants.CLEANUP_FORCE_MARKET_SELL = False
            mock_constants.MIN_MERGE_SIZE = 20

            token = "token1"
            condition_id = "market1"

            # Position bought at 0.40, current best bid is 0.50 (in profit)
            mock_gs.positions = {token: {"size": 100, "avgPrice": 0.40}}
            mock_gs.all_data = {condition_id: {"bids": {0.50: 1000, 0.49: 500}}}

            await data_utils.close_positions([token], condition_id, False)

            # Should sell at best bid (0.50)
            mock_client.create_order_async.assert_called_once_with(token, "SELL", 0.50, 100, False)

    @pytest.mark.asyncio
    async def test_underwater_position_places_limit_at_breakeven(self, mock_client):
        """Test that underwater positions place limit order at break-even."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_gs.client = mock_client
            mock_constants.CLEANUP_FORCE_MARKET_SELL = False
            mock_constants.MIN_MERGE_SIZE = 20

            token = "token1"
            condition_id = "market1"

            # Position bought at 0.60, current best bid is 0.40 (underwater)
            mock_gs.positions = {token: {"size": 100, "avgPrice": 0.60}}
            mock_gs.all_data = {condition_id: {"bids": {0.40: 1000}}}

            await data_utils.close_positions([token], condition_id, False)

            # Should place limit at break-even (0.60)
            mock_client.create_order_async.assert_called_once_with(token, "SELL", 0.60, 100, False)

    @pytest.mark.asyncio
    async def test_force_market_sell_overrides_breakeven(self, mock_client):
        """Test that CLEANUP_FORCE_MARKET_SELL overrides break-even logic."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_gs.client = mock_client
            mock_constants.CLEANUP_FORCE_MARKET_SELL = True
            mock_constants.MIN_MERGE_SIZE = 20

            token = "token1"
            condition_id = "market1"

            # Position underwater
            mock_gs.positions = {token: {"size": 100, "avgPrice": 0.60}}
            mock_gs.all_data = {condition_id: {"bids": {0.40: 1000}}}

            await data_utils.close_positions([token], condition_id, False)

            # Should sell at market (0.40) despite being underwater
            mock_client.create_order_async.assert_called_once_with(token, "SELL", 0.40, 100, False)

    @pytest.mark.asyncio
    async def test_no_bids_places_limit_at_breakeven(self, mock_client):
        """Test that when no bids available, places limit at break-even."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_gs.client = mock_client
            mock_constants.CLEANUP_FORCE_MARKET_SELL = False
            mock_constants.MIN_MERGE_SIZE = 20

            token = "token1"
            condition_id = "market1"

            mock_gs.positions = {token: {"size": 100, "avgPrice": 0.50}}
            mock_gs.all_data = {condition_id: {"bids": {}}}  # No bids

            await data_utils.close_positions([token], condition_id, False)

            # Should place limit at break-even
            mock_client.create_order_async.assert_called_once_with(token, "SELL", 0.50, 100, False)

    @pytest.mark.asyncio
    async def test_no_position_skips_sell(self, mock_client):
        """Test that zero position skips sell."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_gs.client = mock_client
            mock_constants.CLEANUP_FORCE_MARKET_SELL = False
            mock_constants.MIN_MERGE_SIZE = 20

            token = "token1"
            condition_id = "market1"

            mock_gs.positions = {token: {"size": 0, "avgPrice": 0}}
            mock_gs.all_data = {condition_id: {"bids": {0.50: 1000}}}

            await data_utils.close_positions([token], condition_id, False)

            # Should not create any orders
            mock_client.create_order_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_opposing_positions_merge_first(self, mock_client):
        """Test that opposing positions are merged before selling."""
        with (
            patch.object(data_utils, "global_state") as mock_gs,
            patch.object(data_utils, "CONSTANTS") as mock_constants,
        ):
            mock_gs.client = mock_client
            mock_constants.CLEANUP_FORCE_MARKET_SELL = False
            mock_constants.MIN_MERGE_SIZE = 20

            token1 = "token1"
            token2 = "token2"
            condition_id = "market1"

            # Both tokens have positions above merge threshold
            mock_gs.positions = {
                token1: {"size": 50, "avgPrice": 0.40},
                token2: {"size": 30, "avgPrice": 0.60},
            }
            mock_gs.all_data = {condition_id: {"bids": {0.50: 1000}}}

            await data_utils.close_positions([token1, token2], condition_id, False)

            # Should have called merge first (30 is min of 50, 30)
            mock_client.merge_positions_async.assert_called_once()
            merge_call = mock_client.merge_positions_async.call_args
            assert merge_call[0][0] == 30 * 1e6  # Amount in micro units


class TestStateCleanup:
    """Tests for cleaning up all state locations."""

    def test_cleanup_removes_from_all_tokens(self):
        """Test that tokens are removed from all_tokens list."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.all_tokens = ["token1", "token2", "token3"]
            mock_gs.REVERSE_TOKENS = {}
            mock_gs.orders = {}
            mock_gs.all_data = {}
            mock_gs.performing = {}
            mock_gs.performing_timestamps = {}
            mock_gs.last_trade_update = {}

            with patch("os.path.isfile", return_value=False):
                data_utils.cleanup_market_state("market1", ["token1", "token2"])

            assert "token1" not in mock_gs.all_tokens
            assert "token2" not in mock_gs.all_tokens
            assert "token3" in mock_gs.all_tokens

    def test_cleanup_removes_from_reverse_tokens(self):
        """Test that tokens are removed from REVERSE_TOKENS mapping."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.all_tokens = []
            mock_gs.REVERSE_TOKENS = {"token1": "token2", "token2": "token1"}
            mock_gs.orders = {}
            mock_gs.all_data = {}
            mock_gs.performing = {}
            mock_gs.performing_timestamps = {}
            mock_gs.last_trade_update = {}

            with patch("os.path.isfile", return_value=False):
                data_utils.cleanup_market_state("market1", ["token1", "token2"])

            assert "token1" not in mock_gs.REVERSE_TOKENS
            assert "token2" not in mock_gs.REVERSE_TOKENS

    def test_cleanup_removes_from_orders(self):
        """Test that orders are removed."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.all_tokens = []
            mock_gs.REVERSE_TOKENS = {}
            mock_gs.orders = {"token1": {}, "token2": {}}
            mock_gs.all_data = {}
            mock_gs.performing = {}
            mock_gs.performing_timestamps = {}
            mock_gs.last_trade_update = {}

            with patch("os.path.isfile", return_value=False):
                data_utils.cleanup_market_state("market1", ["token1", "token2"])

            assert "token1" not in mock_gs.orders
            assert "token2" not in mock_gs.orders

    def test_cleanup_removes_order_book_data(self):
        """Test that order book data is removed."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.all_tokens = []
            mock_gs.REVERSE_TOKENS = {}
            mock_gs.orders = {}
            mock_gs.all_data = {"market1": {}}
            mock_gs.performing = {}
            mock_gs.performing_timestamps = {}
            mock_gs.last_trade_update = {}

            with patch("os.path.isfile", return_value=False):
                data_utils.cleanup_market_state("market1", ["token1", "token2"])

            assert "market1" not in mock_gs.all_data

    def test_cleanup_removes_performing_sets(self):
        """Test that performing sets are cleared."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.all_tokens = []
            mock_gs.REVERSE_TOKENS = {}
            mock_gs.orders = {}
            mock_gs.all_data = {}
            mock_gs.performing = {"token1_buy": set(), "token1_sell": set()}
            mock_gs.performing_timestamps = {"token1_buy": {}, "token1_sell": {}}
            mock_gs.last_trade_update = {}

            with patch("os.path.isfile", return_value=False):
                data_utils.cleanup_market_state("market1", ["token1"])

            assert "token1_buy" not in mock_gs.performing
            assert "token1_sell" not in mock_gs.performing

    def test_cleanup_removes_risk_off_file(self, tmp_path):
        """Test that risk-off JSON file is deleted."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.all_tokens = []
            mock_gs.REVERSE_TOKENS = {}
            mock_gs.orders = {}
            mock_gs.all_data = {}
            mock_gs.performing = {}
            mock_gs.performing_timestamps = {}
            mock_gs.last_trade_update = {}

            # Create a temp positions directory and file
            positions_dir = tmp_path / "positions"
            positions_dir.mkdir()
            risk_file = positions_dir / "market1.json"
            risk_file.write_text('{"sleep_till": "2025-01-01"}')

            with patch("os.path.isfile", return_value=True), patch("os.remove") as mock_remove:
                data_utils.cleanup_market_state("market1", ["token1"])

                mock_remove.assert_called_once_with("positions/market1.json")


class TestConfiguration:
    """Tests for environment variable configuration."""

    def test_cleanup_enabled_by_default(self):
        """Test that cleanup is enabled when env vars not set."""
        # Import fresh to get actual values
        assert CONSTANTS.CLEANUP_CANCEL_ORDERS is True
        assert CONSTANTS.CLEANUP_SELL_POSITIONS is True
        # CLEANUP_FORCE_MARKET_SELL defaults to false
        assert CONSTANTS.CLEANUP_FORCE_MARKET_SELL is False

    def test_grace_period_default(self):
        """Test that CLEANUP_GRACE_PERIOD has correct default."""
        assert CONSTANTS.CLEANUP_GRACE_PERIOD == 30


class TestSkipRemovingMarkets:
    """Tests for skipping trades on markets being removed."""

    @pytest.mark.asyncio
    async def test_schedule_trade_skips_removing_markets(self):
        """Test that schedule_trade skips markets being removed."""
        from poly_data import data_processing

        market = "test_market"

        with patch.object(data_processing, "global_state") as mock_gs:
            mock_gs.removing_markets = {market}

            # Clear any existing tasks
            data_processing._pending_trades.clear()
            data_processing._pending_trade_tasks.clear()

            # Try to schedule trade - should return early without creating task
            data_processing.schedule_trade(market)

            # Should not have created a pending task
            assert market not in data_processing._pending_trade_tasks

    @pytest.mark.asyncio
    async def test_schedule_trade_works_for_normal_markets(self):
        """Test that schedule_trade works for markets not being removed."""
        from poly_data import data_processing

        market = "test_market"

        with patch.object(data_processing, "global_state") as mock_gs:
            mock_gs.removing_markets = set()  # Empty - market not being removed

            # Clear any existing tasks
            data_processing._pending_trades.clear()
            data_processing._pending_trade_tasks.clear()

            # Schedule trade
            data_processing.schedule_trade(market)

            # Should have created a pending task
            assert market in data_processing._pending_trade_tasks

            # Clean up the task
            task = data_processing._pending_trade_tasks[market]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestCircuitBreakerCleanup:
    """Tests for circuit breaker cleanup method."""

    def test_clear_market_removes_state(self):
        """Test that clear_market removes state for a market."""
        cb = MergeCircuitBreaker(initial_cooldown=60)
        condition_id = "test_market"

        # Record a failure to create state
        cb.record_failure(condition_id, "Test error")

        # Verify state exists
        assert condition_id in cb._states

        # Clear market
        cb.clear_market(condition_id)

        # Verify state removed
        assert condition_id not in cb._states

    def test_clear_market_handles_nonexistent(self):
        """Test that clear_market handles non-existent markets gracefully."""
        cb = MergeCircuitBreaker(initial_cooldown=60)

        # Should not raise
        cb.clear_market("nonexistent_market")


class TestWaitForPerformingTrades:
    """Tests for waiting for performing trades to complete."""

    @pytest.mark.asyncio
    async def test_returns_immediately_when_no_trades(self):
        """Test that it returns immediately when no trades are performing."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.performing = {}
            mock_gs.performing_timestamps = {}

            start = time.time()
            result = await data_utils._wait_for_performing_trades(["token1"], timeout=5.0)
            elapsed = time.time() - start

            assert result is True
            assert elapsed < 0.5  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_times_out_when_trades_stuck(self):
        """Test that it times out when trades are stuck."""
        with patch.object(data_utils, "global_state") as mock_gs:
            mock_gs.performing = {"token1_buy": {"trade1"}}  # Has pending trades
            mock_gs.performing_timestamps = {}

            start = time.time()
            result = await data_utils._wait_for_performing_trades(["token1"], timeout=0.2)
            elapsed = time.time() - start

            assert result is False
            assert elapsed >= 0.2
            # Should have been force-cleaned
            assert "token1_buy" not in mock_gs.performing
