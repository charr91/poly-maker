"""
Tests for the polymarket_client module.

Tests cover:
- create_order: Balance validation for SELL orders
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestCreateOrderBalanceValidation:
    """Tests for balance validation in create_order() for SELL orders."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PolymarketClient with necessary attributes."""
        with patch("poly_data.polymarket_client.ClobClient"):
            with patch("poly_data.polymarket_client.Web3"):
                with patch("poly_data.polymarket_client.Account"):
                    with patch.dict(
                        "os.environ",
                        {
                            "PK": "0x" + "1" * 64,
                            "BROWSER_ADDRESS": "0x" + "2" * 40,
                        },
                    ):
                        from poly_data.polymarket_client import PolymarketClient

                        # Create instance with mocked dependencies
                        client = PolymarketClient.__new__(PolymarketClient)
                        client.client = MagicMock()
                        client.browser_wallet = "0x" + "2" * 40
                        client.conditional_tokens = MagicMock()

                        # Mock rate limiter
                        with patch(
                            "poly_data.polymarket_client.get_rate_limit_manager"
                        ) as mock_rate_limit:
                            mock_manager = MagicMock()
                            mock_rate_limit.return_value = mock_manager

                            yield client

    def test_sell_order_skipped_when_balance_zero(self, mock_client):
        """Test that sell order returns {} when balance is zero."""
        # Mock zero balance
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = 0

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            # Use numeric token ID (tokenIds are large integers in Polymarket)
            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="SELL",
                price=0.55,
                size=100,
                neg_risk=False,
            )

        assert result == {}
        # post_order should not have been called
        mock_client.client.post_order.assert_not_called()

    def test_sell_order_skipped_when_balance_below_dust(self, mock_client):
        """Test that sell order returns {} when balance is below dust threshold (1)."""
        # Mock balance of 0.5 (500000 raw) - below dust threshold
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = 500000

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="SELL",
                price=0.55,
                size=100,
                neg_risk=False,
            )

        assert result == {}
        mock_client.client.post_order.assert_not_called()

    def test_sell_order_adjusts_size_to_actual_balance(self, mock_client):
        """Test that sell size is adjusted when balance < requested but >= 1."""
        # Mock balance of 75 (75_000_000 raw)
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            75_000_000
        )

        # Mock the order creation flow
        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="SELL",
                price=0.55,
                size=100,  # Requesting 100 but only have 75
                neg_risk=False,
            )

        # Verify create_order was called with adjusted size
        call_args = mock_client.client.create_order.call_args
        order_args = call_args[0][0]  # First positional argument is OrderArgs
        assert order_args.size == 75.0

    def test_sell_order_proceeds_when_balance_sufficient(self, mock_client):
        """Test that sell order proceeds normally when balance is sufficient."""
        # Mock balance of 200 (200_000_000 raw)
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            200_000_000
        )

        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="SELL",
                price=0.55,
                size=100,  # Have 200, requesting 100
                neg_risk=False,
            )

        # Verify order was created with original size
        call_args = mock_client.client.create_order.call_args
        order_args = call_args[0][0]
        assert order_args.size == 100

        # Verify post_order was called
        mock_client.client.post_order.assert_called_once()
        assert result == {"orderID": "test123"}

    def test_buy_order_bypasses_balance_check(self, mock_client):
        """Test that BUY orders do not check token balance."""
        # Don't set up balance mock - it shouldn't be called for BUY

        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="BUY",
                price=0.45,
                size=100,
                neg_risk=False,
            )

        # balanceOf should not have been called for BUY
        mock_client.conditional_tokens.functions.balanceOf.assert_not_called()

        # Order should proceed
        mock_client.client.post_order.assert_called_once()
        assert result == {"orderID": "test123"}

    def test_balance_check_failure_continues_with_order(self, mock_client):
        """Test that order proceeds if balance check raises exception."""
        # Make balance check raise an exception
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.side_effect = (
            Exception("RPC error")
        )

        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            # Should not raise, should continue with order
            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="SELL",
                price=0.55,
                size=100,
                neg_risk=False,
            )

        # Order should still proceed
        mock_client.client.post_order.assert_called_once()
        assert result == {"orderID": "test123"}

    def test_sell_order_with_neg_risk_true(self, mock_client):
        """Test balance validation works with neg_risk=True."""
        # Mock balance of 50 (50_000_000 raw)
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            50_000_000
        )

        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="SELL",
                price=0.55,
                size=100,  # Have 50, requesting 100
                neg_risk=True,
            )

        # Verify size was adjusted
        call_args = mock_client.client.create_order.call_args
        order_args = call_args[0][0]
        assert order_args.size == 50.0

        # Verify neg_risk option was passed
        assert call_args[1]["options"].neg_risk is True

    def test_sell_order_exact_balance_proceeds(self, mock_client):
        """Test that sell order proceeds when balance exactly matches size."""
        # Mock balance of exactly 100 (100_000_000 raw)
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            100_000_000
        )

        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId="12345678901234567890",
                action="SELL",
                price=0.55,
                size=100,  # Exactly matches balance
                neg_risk=False,
            )

        # Should proceed with original size (no adjustment needed)
        call_args = mock_client.client.create_order.call_args
        order_args = call_args[0][0]
        assert order_args.size == 100

        mock_client.client.post_order.assert_called_once()
