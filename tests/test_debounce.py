"""
Tests for the debounce logic in data_processing module.

Tests verify that:
- Rapid updates extend the quiet period rather than dropping trades
- A trade is always executed after the debounce period ends
- Cancelling previous tasks works correctly
"""

import asyncio
import sys
import time
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

# We need to mock the trading module before importing data_processing
# to avoid import errors from missing dependencies


def _clear_module_cache():
    """Clear data_processing module from cache to ensure fresh import."""
    modules_to_clear = [key for key in list(sys.modules.keys()) if "data_processing" in key]
    for module in modules_to_clear:
        del sys.modules[module]


class TestDebounceLogic:
    """Tests for the debounce scheduling logic."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        # Mock the perform_trade function
        self.mock_perform_trade = AsyncMock()

        # Mock global_state to avoid import issues
        self.mock_global_state = MagicMock()

        # Clear any previously cached version of data_processing
        _clear_module_cache()

        with patch.dict(
            "sys.modules",
            {
                "trading": MagicMock(perform_trade=self.mock_perform_trade),
                "poly_data.global_state": self.mock_global_state,
                "poly_data.CONSTANTS": MagicMock(),
                "poly_data.data_utils": MagicMock(),
            },
        ):
            # Clear again inside the patch context to ensure fresh import
            _clear_module_cache()

            # Now import the module with mocks in place
            from poly_data import data_processing

            self.data_processing = data_processing

            # Clear any existing state
            data_processing._pending_trades.clear()
            data_processing._pending_trade_tasks.clear()

            # Replace perform_trade with our mock
            data_processing.perform_trade = self.mock_perform_trade

            yield

        # Clear module cache after test
        _clear_module_cache()

    @pytest.mark.asyncio
    async def test_single_trade_executes_after_debounce(self):
        """Test that a single scheduled trade executes after the debounce period."""
        # Set a short debounce for testing
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.05  # 50ms

        try:
            market = "test_market_1"

            # Schedule a trade
            self.data_processing.schedule_trade(market)

            # Trade should not have executed yet
            assert self.mock_perform_trade.call_count == 0

            # Wait for debounce period + buffer
            await asyncio.sleep(0.1)

            # Trade should have executed
            assert self.mock_perform_trade.call_count == 1
            self.mock_perform_trade.assert_called_with(market)

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce

    @pytest.mark.asyncio
    async def test_rapid_updates_extend_debounce(self):
        """Test that rapid updates extend the debounce period and only execute once."""
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.1  # 100ms

        try:
            market = "test_market_2"

            # Schedule initial trade
            self.data_processing.schedule_trade(market)

            # Wait 50ms (half of debounce)
            await asyncio.sleep(0.05)

            # Trade should not have executed yet
            assert self.mock_perform_trade.call_count == 0

            # Schedule another trade (should cancel first and restart timer)
            self.data_processing.schedule_trade(market)

            # Wait another 50ms (still within new debounce period)
            await asyncio.sleep(0.05)

            # Trade should still not have executed
            assert self.mock_perform_trade.call_count == 0

            # Wait for the full debounce period to complete
            await asyncio.sleep(0.1)

            # Now trade should have executed exactly once
            assert self.mock_perform_trade.call_count == 1
            self.mock_perform_trade.assert_called_with(market)

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce

    @pytest.mark.asyncio
    async def test_trade_not_dropped_on_reschedule(self):
        """Test that rescheduling doesn't drop trades - they always execute."""
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.05  # 50ms

        try:
            market = "test_market_3"

            # Schedule multiple trades rapidly
            for _ in range(5):
                self.data_processing.schedule_trade(market)
                await asyncio.sleep(0.01)  # 10ms between each

            # Wait for debounce to complete
            await asyncio.sleep(0.1)

            # Trade should have executed exactly once (not dropped, not multiple)
            assert self.mock_perform_trade.call_count == 1
            self.mock_perform_trade.assert_called_with(market)

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce

    @pytest.mark.asyncio
    async def test_different_markets_debounce_independently(self):
        """Test that different markets have independent debounce timers."""
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.05  # 50ms

        try:
            market_a = "market_a"
            market_b = "market_b"

            # Schedule trades for both markets
            self.data_processing.schedule_trade(market_a)
            self.data_processing.schedule_trade(market_b)

            # Wait for debounce
            await asyncio.sleep(0.1)

            # Both trades should have executed
            assert self.mock_perform_trade.call_count == 2

            # Check both markets were called
            calls = [call[0][0] for call in self.mock_perform_trade.call_args_list]
            assert market_a in calls
            assert market_b in calls

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce

    @pytest.mark.asyncio
    async def test_cancelled_task_does_not_execute(self):
        """Test that when a task is cancelled, a new one runs instead."""
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.1  # 100ms

        try:
            market = "test_market_cancel"

            # Schedule first trade
            self.data_processing.schedule_trade(market)
            first_task = self.data_processing._pending_trade_tasks.get(market)

            # Wait a bit
            await asyncio.sleep(0.02)

            # Schedule second trade (should cancel first)
            self.data_processing.schedule_trade(market)
            second_task = self.data_processing._pending_trade_tasks.get(market)

            # Second task should be different
            assert second_task is not first_task

            # Give time for cancellation to process
            await asyncio.sleep(0.01)

            # First task should be cancelled or done (cancelled raises CancelledError which marks it done)
            assert first_task.cancelled() or first_task.done()

            # Wait for second task to complete
            await asyncio.sleep(0.15)

            # Only one trade should have executed
            assert self.mock_perform_trade.call_count == 1

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce

    @pytest.mark.asyncio
    async def test_pending_state_cleared_after_execution(self):
        """Test that pending state is properly cleared after trade execution."""
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.05  # 50ms

        try:
            market = "test_market_cleanup"

            # Schedule a trade
            self.data_processing.schedule_trade(market)

            # Verify pending state exists
            assert market in self.data_processing._pending_trades
            assert market in self.data_processing._pending_trade_tasks

            # Wait for execution
            await asyncio.sleep(0.1)

            # Pending state should be cleared
            assert market not in self.data_processing._pending_trades
            assert market not in self.data_processing._pending_trade_tasks

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce


class TestDebounceEdgeCases:
    """Edge case tests for debounce logic."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_perform_trade = AsyncMock()
        self.mock_global_state = MagicMock()

        # Clear any previously cached version of data_processing
        _clear_module_cache()

        with patch.dict(
            "sys.modules",
            {
                "trading": MagicMock(perform_trade=self.mock_perform_trade),
                "poly_data.global_state": self.mock_global_state,
                "poly_data.CONSTANTS": MagicMock(),
                "poly_data.data_utils": MagicMock(),
            },
        ):
            # Clear again inside the patch context to ensure fresh import
            _clear_module_cache()

            from poly_data import data_processing

            self.data_processing = data_processing
            data_processing._pending_trades.clear()
            data_processing._pending_trade_tasks.clear()
            data_processing.perform_trade = self.mock_perform_trade
            yield

        # Clear module cache after test
        _clear_module_cache()

    @pytest.mark.asyncio
    async def test_schedule_after_completion(self):
        """Test that scheduling after a trade completes works correctly."""
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.03  # 30ms

        try:
            market = "test_market_after"

            # Schedule first trade
            self.data_processing.schedule_trade(market)

            # Wait for it to complete
            await asyncio.sleep(0.05)

            assert self.mock_perform_trade.call_count == 1

            # Schedule second trade
            self.data_processing.schedule_trade(market)

            # Wait for it to complete
            await asyncio.sleep(0.05)

            # Both trades should have executed
            assert self.mock_perform_trade.call_count == 2

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce

    @pytest.mark.asyncio
    async def test_high_frequency_updates(self):
        """Test behavior under high-frequency updates."""
        original_debounce = self.data_processing.TRADE_DEBOUNCE_SEC
        self.data_processing.TRADE_DEBOUNCE_SEC = 0.05  # 50ms

        try:
            market = "test_high_freq"

            # Send many rapid updates
            for _ in range(20):
                self.data_processing.schedule_trade(market)
                await asyncio.sleep(0.005)  # 5ms between each

            # Wait for debounce to complete
            await asyncio.sleep(0.1)

            # Should have executed exactly once
            assert self.mock_perform_trade.call_count == 1

        finally:
            self.data_processing.TRADE_DEBOUNCE_SEC = original_debounce
