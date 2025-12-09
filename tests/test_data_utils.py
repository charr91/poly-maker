"""
Tests for the data_utils module.

Tests cover:
- set_order: Order tracking that preserves both sides
- get_order: Order retrieval
"""

import pytest
from unittest.mock import patch, MagicMock

import poly_data.global_state as global_state
from poly_data.data_utils import set_order, get_order


class TestSetOrder:
    """Tests for the set_order function."""

    def setup_method(self):
        """Reset global state before each test."""
        global_state.orders = {}

    def teardown_method(self):
        """Clean up global state after each test."""
        global_state.orders = {}

    def test_set_order_preserves_other_side(self):
        """Test that updating one side preserves the other side's data."""
        token = "test_token_123"

        # Set initial buy order
        set_order(token, "buy", 100, 0.45)

        # Verify buy order is set
        assert global_state.orders[token]["buy"]["size"] == 100
        assert global_state.orders[token]["buy"]["price"] == 0.45

        # Set sell order - this should NOT overwrite buy
        set_order(token, "sell", 50, 0.55)

        # Verify BOTH sides are preserved
        assert global_state.orders[token]["buy"]["size"] == 100
        assert global_state.orders[token]["buy"]["price"] == 0.45
        assert global_state.orders[token]["sell"]["size"] == 50
        assert global_state.orders[token]["sell"]["price"] == 0.55

    def test_set_order_updates_same_side(self):
        """Test that updating same side replaces values correctly."""
        token = "test_token_123"

        set_order(token, "buy", 100, 0.45)
        set_order(token, "buy", 150, 0.50)  # Update buy

        assert global_state.orders[token]["buy"]["size"] == 150
        assert global_state.orders[token]["buy"]["price"] == 0.50

    def test_set_order_initializes_both_sides(self):
        """Test that new token gets both sides initialized."""
        token = "new_token"

        set_order(token, "sell", 100, 0.55)

        # Verify both sides exist
        assert "buy" in global_state.orders[token]
        assert "sell" in global_state.orders[token]
        assert global_state.orders[token]["buy"]["size"] == 0
        assert global_state.orders[token]["buy"]["price"] == 0

    def test_set_order_handles_string_token_id(self):
        """Test that token ID is converted to string."""
        set_order(12345, "buy", 100, 0.45)

        assert "12345" in global_state.orders

    def test_set_order_handles_int_token_id(self):
        """Test that integer token ID works correctly."""
        token_int = 98765432109876543210
        set_order(token_int, "sell", 200, 0.60)

        assert str(token_int) in global_state.orders
        assert global_state.orders[str(token_int)]["sell"]["size"] == 200

    def test_set_order_handles_corrupted_state_missing_buy(self):
        """Test defensive handling when buy side is missing from existing state."""
        token = "corrupted_token"
        # Simulate corrupted state (only sell side exists)
        global_state.orders[token] = {"sell": {"price": 0.50, "size": 50}}

        # Should not crash, should add buy side
        set_order(token, "buy", 100, 0.45)

        assert global_state.orders[token]["buy"]["size"] == 100
        assert global_state.orders[token]["buy"]["price"] == 0.45
        # Sell side should still exist
        assert global_state.orders[token]["sell"]["size"] == 50

    def test_set_order_handles_corrupted_state_missing_sell(self):
        """Test defensive handling when sell side is missing from existing state."""
        token = "corrupted_token"
        # Simulate corrupted state (only buy side exists)
        global_state.orders[token] = {"buy": {"price": 0.45, "size": 100}}

        # Should not crash, should add sell side
        set_order(token, "sell", 50, 0.55)

        assert global_state.orders[token]["sell"]["size"] == 50
        assert global_state.orders[token]["sell"]["price"] == 0.55
        # Buy side should still exist
        assert global_state.orders[token]["buy"]["size"] == 100

    def test_set_order_converts_size_to_float(self):
        """Test that size is converted to float."""
        token = "test_token"
        set_order(token, "buy", "100", 0.45)  # String size

        assert global_state.orders[token]["buy"]["size"] == 100.0
        assert isinstance(global_state.orders[token]["buy"]["size"], float)

    def test_set_order_converts_price_to_float(self):
        """Test that price is converted to float."""
        token = "test_token"
        set_order(token, "buy", 100, "0.45")  # String price

        assert global_state.orders[token]["buy"]["price"] == 0.45
        assert isinstance(global_state.orders[token]["buy"]["price"], float)

    def test_set_order_multiple_tokens(self):
        """Test that multiple tokens can be tracked independently."""
        token1 = "token_1"
        token2 = "token_2"

        set_order(token1, "buy", 100, 0.45)
        set_order(token2, "sell", 200, 0.60)

        assert global_state.orders[token1]["buy"]["size"] == 100
        assert global_state.orders[token2]["sell"]["size"] == 200

    def test_set_order_does_not_mutate_original(self):
        """Test that the function doesn't mutate the original dict unexpectedly."""
        token = "test_token"
        set_order(token, "buy", 100, 0.45)

        # Get a reference to the orders dict
        orders_ref = global_state.orders[token]

        # Update via set_order
        set_order(token, "sell", 50, 0.55)

        # The reference should still be valid (not a completely new dict)
        # but the buy side should be preserved
        assert orders_ref["buy"]["size"] == 100


class TestGetOrder:
    """Tests for the get_order function."""

    def setup_method(self):
        """Reset global state before each test."""
        global_state.orders = {}

    def teardown_method(self):
        """Clean up global state after each test."""
        global_state.orders = {}

    def test_get_order_returns_existing_order(self):
        """Test that get_order returns existing order data."""
        token = "test_token"
        global_state.orders[token] = {
            "buy": {"price": 0.45, "size": 100},
            "sell": {"price": 0.55, "size": 50},
        }

        result = get_order(token)

        assert result["buy"]["price"] == 0.45
        assert result["buy"]["size"] == 100
        assert result["sell"]["price"] == 0.55
        assert result["sell"]["size"] == 50

    def test_get_order_returns_default_for_missing_token(self):
        """Test that get_order returns default struct for missing token."""
        result = get_order("nonexistent_token")

        assert result == {"buy": {"price": 0, "size": 0}, "sell": {"price": 0, "size": 0}}

    def test_get_order_handles_string_token(self):
        """Test that get_order handles string token ID."""
        token = "string_token"
        global_state.orders[token] = {
            "buy": {"price": 0.45, "size": 100},
            "sell": {"price": 0, "size": 0},
        }

        result = get_order(token)

        assert result["buy"]["size"] == 100


class TestSetOrderIntegration:
    """Integration tests for set_order with get_order."""

    def setup_method(self):
        """Reset global state before each test."""
        global_state.orders = {}

    def teardown_method(self):
        """Clean up global state after each test."""
        global_state.orders = {}

    def test_set_then_get_order(self):
        """Test set_order followed by get_order returns correct data."""
        token = "integration_token"

        set_order(token, "buy", 100, 0.45)
        set_order(token, "sell", 50, 0.55)

        result = get_order(token)

        assert result["buy"]["size"] == 100
        assert result["buy"]["price"] == 0.45
        assert result["sell"]["size"] == 50
        assert result["sell"]["price"] == 0.55

    def test_realistic_trading_scenario(self):
        """Test a realistic sequence of order updates."""
        token = "realistic_token"

        # Initial buy order placed
        set_order(token, "buy", 300, 0.14)
        orders = get_order(token)
        assert orders["buy"]["size"] == 300
        assert orders["sell"]["size"] == 0

        # Trade filled, sell order placed for take-profit
        set_order(token, "sell", 300, 0.15)
        orders = get_order(token)
        assert orders["buy"]["size"] == 300  # Buy order should still be tracked
        assert orders["sell"]["size"] == 300
        assert orders["sell"]["price"] == 0.15

        # Buy order cancelled (updated to 0)
        set_order(token, "buy", 0, 0)
        orders = get_order(token)
        assert orders["buy"]["size"] == 0
        assert orders["sell"]["size"] == 300  # Sell should still be there
