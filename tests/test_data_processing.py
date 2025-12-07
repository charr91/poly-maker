"""
Tests for data processing input handling and validation.

Tests verify that:
- process_data handles both single dict and list inputs
- process_user_data handles both single dict and list inputs
- Invalid messages are logged and skipped gracefully
- Unknown tokens are handled correctly
"""

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, patch, MagicMock
import pytest


def _clear_module_cache():
    """Clear data_processing module from cache to ensure fresh import."""
    modules_to_clear = [key for key in sys.modules if "data_processing" in key]
    for module in modules_to_clear:
        del sys.modules[module]


class TestProcessDataInputHandling:
    """Tests for process_data input normalization."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        # Clear module cache before importing
        _clear_module_cache()

        self.mock_perform_trade = AsyncMock()
        self.mock_global_state = MagicMock()
        self.mock_global_state.all_data = {}

        with patch.dict(
            "sys.modules",
            {
                "trading": MagicMock(perform_trade=self.mock_perform_trade),
                "poly_data.global_state": self.mock_global_state,
                "poly_data.CONSTANTS": MagicMock(),
                "poly_data.data_utils": MagicMock(),
            },
        ):
            from poly_data import data_processing

            self.data_processing = data_processing
            data_processing.perform_trade = self.mock_perform_trade
            data_processing.global_state = self.mock_global_state
            yield

        # Clear module cache after test to avoid polluting other tests
        _clear_module_cache()

    def test_process_data_with_single_dict(self):
        """Verify single dict input is handled correctly."""
        # Set up mock data
        self.mock_global_state.all_data = {}

        # Single dict message (how websocket passes it)
        message = {
            "event_type": "book",
            "market": "test_market_123",
            "asset_id": "token_456",
            "bids": [{"price": "0.5", "size": "100"}],
            "asks": [{"price": "0.55", "size": "50"}],
        }

        # Should not raise TypeError
        self.data_processing.process_data(message, trade=False)

        # Verify data was processed
        assert "test_market_123" in self.mock_global_state.all_data

    def test_process_data_with_list(self):
        """Verify list input is handled correctly."""
        self.mock_global_state.all_data = {}

        # List of messages
        messages = [
            {
                "event_type": "book",
                "market": "market_1",
                "asset_id": "token_1",
                "bids": [],
                "asks": [],
            },
            {
                "event_type": "book",
                "market": "market_2",
                "asset_id": "token_2",
                "bids": [],
                "asks": [],
            },
        ]

        self.data_processing.process_data(messages, trade=False)

        # Both markets should be processed
        assert "market_1" in self.mock_global_state.all_data
        assert "market_2" in self.mock_global_state.all_data

    def test_process_data_with_empty_list(self):
        """Verify empty list is handled gracefully."""
        self.mock_global_state.all_data = {}

        # Should not raise
        self.data_processing.process_data([], trade=False)

    def test_process_data_skips_invalid_message(self):
        """Verify invalid messages are skipped without crashing."""
        self.mock_global_state.all_data = {}

        # Message missing required fields
        invalid_message = {"foo": "bar"}

        # Should not raise
        self.data_processing.process_data(invalid_message, trade=False)

        # No data should be added
        assert len(self.mock_global_state.all_data) == 0

    def test_process_data_handles_string_iteration_bug(self):
        """Verify the original string iteration bug is fixed.

        Previously, passing a dict would iterate over keys (strings),
        causing TypeError: string indices must be integers when accessing
        json_data['event_type'].
        """
        self.mock_global_state.all_data = {}

        # This is exactly what the websocket handler passes
        single_dict = {
            "event_type": "book",
            "market": "some_market",
            "asset_id": "some_token",
            "bids": [],
            "asks": [],
        }

        # This should NOT raise TypeError
        try:
            self.data_processing.process_data(single_dict, trade=False)
        except TypeError as e:
            pytest.fail(f"String iteration bug not fixed: {e}")


class TestProcessUserDataInputHandling:
    """Tests for process_user_data input normalization."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        # Clear module cache before importing
        _clear_module_cache()

        self.mock_perform_trade = AsyncMock()
        self.mock_global_state = MagicMock()
        self.mock_global_state.REVERSE_TOKENS = {"token_123": "token_456", "token_456": "token_123"}
        self.mock_global_state.performing = {}
        self.mock_global_state.performing_timestamps = {}
        self.mock_global_state.positions = {}
        self.mock_global_state.client = MagicMock()
        self.mock_global_state.client.browser_wallet = "0xtest_wallet"

        self.mock_set_position = MagicMock()
        self.mock_set_order = MagicMock()
        self.mock_create_task = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "trading": MagicMock(perform_trade=self.mock_perform_trade),
                "poly_data.global_state": self.mock_global_state,
                "poly_data.CONSTANTS": MagicMock(),
                "poly_data.data_utils": MagicMock(
                    set_position=self.mock_set_position,
                    set_order=self.mock_set_order,
                    update_positions=MagicMock(),
                ),
            },
        ):
            from poly_data import data_processing

            self.data_processing = data_processing
            data_processing.perform_trade = self.mock_perform_trade
            data_processing.global_state = self.mock_global_state
            data_processing.set_position = self.mock_set_position
            data_processing.set_order = self.mock_set_order

            # Use patch context for asyncio.create_task to avoid polluting global state
            with patch.object(data_processing.asyncio, "create_task", self.mock_create_task):
                yield

        # Clear module cache after test to avoid polluting other tests
        _clear_module_cache()

    def test_process_user_data_with_single_dict(self):
        """Verify single dict input is handled correctly."""
        # Single dict message (how websocket passes it)
        message = {
            "market": "test_market",
            "side": "BUY",
            "asset_id": "token_123",
            "event_type": "order",
            "status": "OPEN",
            "type": "LIMIT",
            "original_size": "100",
            "size_matched": "0",
            "price": "0.5",
        }

        # Should not raise TypeError
        self.data_processing.process_user_data(message)

        # Verify set_order was called
        self.mock_set_order.assert_called()

    def test_process_user_data_with_list(self):
        """Verify list input is handled correctly."""
        messages = [
            {
                "market": "market_1",
                "side": "BUY",
                "asset_id": "token_123",
                "event_type": "order",
                "status": "OPEN",
                "type": "LIMIT",
                "original_size": "50",
                "size_matched": "0",
                "price": "0.4",
            },
            {
                "market": "market_2",
                "side": "SELL",
                "asset_id": "token_456",
                "event_type": "order",
                "status": "OPEN",
                "type": "LIMIT",
                "original_size": "25",
                "size_matched": "0",
                "price": "0.6",
            },
        ]

        self.data_processing.process_user_data(messages)

        # Both messages should trigger set_order
        assert self.mock_set_order.call_count == 2

    def test_process_user_data_skips_unknown_token(self):
        """Verify unknown tokens are skipped gracefully."""
        message = {
            "market": "test_market",
            "side": "BUY",
            "asset_id": "unknown_token_xyz",  # Not in REVERSE_TOKENS
            "event_type": "order",
            "status": "OPEN",
            "type": "LIMIT",
            "original_size": "100",
            "size_matched": "0",
            "price": "0.5",
        }

        # Should not raise, should skip silently
        self.data_processing.process_user_data(message)

        # set_order should NOT be called
        self.mock_set_order.assert_not_called()

    def test_process_user_data_skips_invalid_message(self):
        """Verify invalid messages are skipped without crashing."""
        # Message missing required fields
        invalid_message = {"foo": "bar"}

        # Should not raise
        self.data_processing.process_user_data(invalid_message)

    def test_process_user_data_handles_string_iteration_bug(self):
        """Verify the original string iteration bug is fixed."""
        single_dict = {
            "market": "test_market",
            "side": "BUY",
            "asset_id": "token_123",
            "event_type": "order",
            "status": "OPEN",
            "type": "LIMIT",
            "original_size": "100",
            "size_matched": "0",
            "price": "0.5",
        }

        # This should NOT raise TypeError
        try:
            self.data_processing.process_user_data(single_dict)
        except TypeError as e:
            pytest.fail(f"String iteration bug not fixed: {e}")


class TestValidationFunctions:
    """Tests for message validation helper functions."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        # Clear module cache before importing
        _clear_module_cache()

        with patch.dict(
            "sys.modules",
            {
                "trading": MagicMock(),
                "poly_data.global_state": MagicMock(),
                "poly_data.CONSTANTS": MagicMock(),
                "poly_data.data_utils": MagicMock(),
            },
        ):
            from poly_data import data_processing

            self.data_processing = data_processing
            yield

        # Clear module cache after test to avoid polluting other tests
        _clear_module_cache()

    def test_validate_market_message_valid(self):
        """Test validation passes for valid market message."""
        valid = {"event_type": "book", "market": "test_market"}
        assert self.data_processing._validate_market_message(valid) is True

    def test_validate_market_message_missing_event_type(self):
        """Test validation fails when event_type is missing."""
        invalid = {"market": "test_market"}
        assert self.data_processing._validate_market_message(invalid) is False

    def test_validate_market_message_missing_market(self):
        """Test validation fails when market is missing."""
        invalid = {"event_type": "book"}
        assert self.data_processing._validate_market_message(invalid) is False

    def test_validate_market_message_not_dict(self):
        """Test validation fails for non-dict input."""
        assert self.data_processing._validate_market_message("string") is False
        assert self.data_processing._validate_market_message(123) is False
        assert self.data_processing._validate_market_message(None) is False

    def test_validate_user_message_valid(self):
        """Test validation passes for valid user message."""
        valid = {
            "market": "test_market",
            "side": "BUY",
            "asset_id": "token_123",
            "event_type": "order",
        }
        assert self.data_processing._validate_user_message(valid) is True

    def test_validate_user_message_missing_fields(self):
        """Test validation fails when required fields are missing."""
        # Missing market
        assert (
            self.data_processing._validate_user_message(
                {"side": "BUY", "asset_id": "token", "event_type": "order"}
            )
            is False
        )

        # Missing side
        assert (
            self.data_processing._validate_user_message(
                {"market": "mkt", "asset_id": "token", "event_type": "order"}
            )
            is False
        )

        # Missing asset_id
        assert (
            self.data_processing._validate_user_message(
                {"market": "mkt", "side": "BUY", "event_type": "order"}
            )
            is False
        )

        # Missing event_type
        assert (
            self.data_processing._validate_user_message(
                {"market": "mkt", "side": "BUY", "asset_id": "token"}
            )
            is False
        )
