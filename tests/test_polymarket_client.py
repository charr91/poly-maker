"""
Tests for the polymarket_client module.

Tests cover:
- create_order: Balance validation for SELL orders
- BalanceCache: Thread-safe balance caching
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestCreateOrderBalanceValidation:
    """Tests for balance validation in create_order() for SELL orders."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PolymarketClient with necessary attributes."""
        # Clear the global balance cache before each test to ensure fresh state
        from poly_data.polymarket_client import _balance_cache

        _balance_cache.clear()

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
                        client.neg_risk_ctf_exchange = MagicMock()

                        # Mock rate limiter
                        with patch(
                            "poly_data.polymarket_client.get_rate_limit_manager"
                        ) as mock_rate_limit:
                            mock_manager = MagicMock()
                            mock_rate_limit.return_value = mock_manager

                            yield client

                        # Clear cache after test as well
                        _balance_cache.clear()

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
        """Test balance validation works with neg_risk=True.

        For neg_risk markets, balance is checked from neg_risk_ctf_exchange contract,
        not the regular conditional_tokens contract.
        """
        # Mock balance of 50 from neg_risk_ctf_exchange (50_000_000 raw)
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.return_value.call.return_value = (
            50_000_000
        )
        # Ensure conditional_tokens returns different value to verify correct contract is used
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
                size=100,  # Have 50 in neg_risk, requesting 100
                neg_risk=True,
            )

        # Verify size was adjusted to 50 (from neg_risk_ctf_exchange, not 200 from conditional_tokens)
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


class TestBalanceCache:
    """Tests for the BalanceCache class."""

    def test_set_and_get_returns_cached_value(self):
        """Test that set followed by get returns the cached value."""
        from poly_data.polymarket_client import BalanceCache

        cache = BalanceCache(ttl_seconds=10.0)
        cache.set("token123", 100.5)

        result = cache.get("token123")
        assert result == 100.5

    def test_get_nonexistent_returns_none(self):
        """Test that get for non-existent key returns None."""
        from poly_data.polymarket_client import BalanceCache

        cache = BalanceCache(ttl_seconds=10.0)

        result = cache.get("nonexistent")
        assert result is None

    def test_get_expired_returns_none(self):
        """Test that get returns None after TTL expires."""
        from poly_data.polymarket_client import BalanceCache
        import time

        cache = BalanceCache(ttl_seconds=0.05)  # 50ms TTL
        cache.set("token123", 100.5)

        # Wait for expiration
        time.sleep(0.1)

        result = cache.get("token123")
        assert result is None

    def test_get_before_expiry_returns_value(self):
        """Test that get returns value before TTL expires."""
        from poly_data.polymarket_client import BalanceCache
        import time

        cache = BalanceCache(ttl_seconds=1.0)  # 1 second TTL
        cache.set("token123", 100.5)

        # Don't wait long
        time.sleep(0.01)

        result = cache.get("token123")
        assert result == 100.5

    def test_invalidate_removes_entry(self):
        """Test that invalidate removes the cached entry."""
        from poly_data.polymarket_client import BalanceCache

        cache = BalanceCache(ttl_seconds=10.0)
        cache.set("token123", 100.5)

        cache.invalidate("token123")

        result = cache.get("token123")
        assert result is None

    def test_invalidate_nonexistent_no_error(self):
        """Test that invalidating a non-existent key doesn't raise."""
        from poly_data.polymarket_client import BalanceCache

        cache = BalanceCache(ttl_seconds=10.0)

        # Should not raise
        cache.invalidate("nonexistent")

    def test_clear_removes_all_entries(self):
        """Test that clear removes all cached entries."""
        from poly_data.polymarket_client import BalanceCache

        cache = BalanceCache(ttl_seconds=10.0)
        cache.set("token1", 100.0)
        cache.set("token2", 200.0)
        cache.set("token3", 300.0)

        cache.clear()

        assert cache.get("token1") is None
        assert cache.get("token2") is None
        assert cache.get("token3") is None

    def test_overwrite_updates_value_and_timestamp(self):
        """Test that setting same key updates value and resets TTL."""
        from poly_data.polymarket_client import BalanceCache
        import time

        cache = BalanceCache(ttl_seconds=0.1)  # 100ms TTL
        cache.set("token123", 100.0)

        # Wait 60ms
        time.sleep(0.06)

        # Update the value (resets TTL)
        cache.set("token123", 200.0)

        # Wait another 60ms (would be expired if TTL wasn't reset)
        time.sleep(0.06)

        result = cache.get("token123")
        assert result == 200.0

    def test_multiple_tokens_independent(self):
        """Test that different tokens have independent cache entries."""
        from poly_data.polymarket_client import BalanceCache

        cache = BalanceCache(ttl_seconds=10.0)
        cache.set("token1", 100.0)
        cache.set("token2", 200.0)

        assert cache.get("token1") == 100.0
        assert cache.get("token2") == 200.0

        cache.invalidate("token1")

        assert cache.get("token1") is None
        assert cache.get("token2") == 200.0

    def test_thread_safety(self):
        """Test that cache is thread-safe with concurrent access."""
        from poly_data.polymarket_client import BalanceCache
        import threading
        import time

        cache = BalanceCache(ttl_seconds=10.0)
        errors = []

        def writer(token_id, value, iterations):
            try:
                for _ in range(iterations):
                    cache.set(token_id, value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader(token_id, iterations):
            try:
                for _ in range(iterations):
                    cache.get(token_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("token1", 100.0, 50)),
            threading.Thread(target=writer, args=("token1", 200.0, 50)),
            threading.Thread(target=reader, args=("token1", 100)),
            threading.Thread(target=writer, args=("token2", 300.0, 50)),
            threading.Thread(target=reader, args=("token2", 100)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestNegRiskContractSelection:
    """Regression tests for neg_risk contract selection (fix b909fca).

    This tests the fix for querying the correct contract based on market type:
    - Regular markets: ConditionalTokens contract
    - Negative-risk markets: NegRiskCtfExchange contract
    """

    @pytest.fixture
    def mock_client(self):
        """Create a mock PolymarketClient with necessary attributes."""
        from poly_data.polymarket_client import _balance_cache

        _balance_cache.clear()

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

                        client = PolymarketClient.__new__(PolymarketClient)
                        client.client = MagicMock()
                        client.browser_wallet = "0x" + "2" * 40
                        client.conditional_tokens = MagicMock()
                        client.neg_risk_ctf_exchange = MagicMock()

                        with patch(
                            "poly_data.polymarket_client.get_rate_limit_manager"
                        ) as mock_rate_limit:
                            mock_manager = MagicMock()
                            mock_rate_limit.return_value = mock_manager

                            yield client

                        _balance_cache.clear()

    def test_get_raw_position_uses_conditional_tokens_for_regular_market(self, mock_client):
        """Test that regular markets query ConditionalTokens contract."""
        # Set up mock return value
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            100_000_000
        )

        result = mock_client.get_raw_position("12345678901234567890", neg_risk=False)

        # Verify ConditionalTokens was called
        mock_client.conditional_tokens.functions.balanceOf.assert_called_once()
        # Verify NegRiskCtfExchange was NOT called
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.assert_not_called()

        assert result == 100_000_000

    def test_get_raw_position_uses_neg_risk_exchange_for_neg_risk_market(self, mock_client):
        """Test that neg_risk markets query NegRiskCtfExchange contract."""
        # Set up mock return value
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.return_value.call.return_value = (
            75_000_000
        )

        result = mock_client.get_raw_position("12345678901234567890", neg_risk=True)

        # Verify NegRiskCtfExchange was called
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.assert_called_once()
        # Verify ConditionalTokens was NOT called
        mock_client.conditional_tokens.functions.balanceOf.assert_not_called()

        assert result == 75_000_000

    def test_balance_cache_separates_neg_risk_keys(self, mock_client):
        """Test that balance cache uses separate keys for neg_risk flag.

        This prevents cross-contamination between regular and neg_risk markets
        with the same token ID.
        """
        from poly_data.polymarket_client import _balance_cache

        token_id = "12345678901234567890"

        # Set different balances for regular vs neg_risk
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            100_000_000
        )
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.return_value.call.return_value = (
            50_000_000
        )

        # Simulate what happens in create_order for SELL
        # First call for regular market
        regular_cache_key = f"{token_id}_False"
        regular_balance = mock_client.get_raw_position(token_id, neg_risk=False) / 1e6
        _balance_cache.set(regular_cache_key, regular_balance)

        # Second call for neg_risk market (same token ID)
        neg_risk_cache_key = f"{token_id}_True"
        neg_risk_balance = mock_client.get_raw_position(token_id, neg_risk=True) / 1e6
        _balance_cache.set(neg_risk_cache_key, neg_risk_balance)

        # Verify cache has separate entries
        assert _balance_cache.get(regular_cache_key) == 100.0
        assert _balance_cache.get(neg_risk_cache_key) == 50.0

    def test_sell_order_queries_correct_contract_for_neg_risk(self, mock_client):
        """Integration test: SELL order for neg_risk market uses correct contract.

        This is a regression test for the bug where SELL orders on neg_risk markets
        would query the wrong contract, get balance=0, and fail to execute stop-loss.
        """
        # Set up: neg_risk market has balance of 50
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.return_value.call.return_value = (
            50_000_000
        )
        # Regular contract has different balance (should NOT be used)
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
                size=100,  # Requesting 100, but have 50 in neg_risk
                neg_risk=True,
            )

        # Verify NegRiskCtfExchange was queried (not ConditionalTokens)
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.assert_called()

        # Verify size was adjusted to 50 (from neg_risk contract, not 200)
        call_args = mock_client.client.create_order.call_args
        order_args = call_args[0][0]
        assert (
            order_args.size == 50.0
        ), f"Size should be 50 from neg_risk contract, got {order_args.size}"


class TestBalanceCacheInvalidation:
    """Tests for balance cache invalidation after successful SELL orders."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock PolymarketClient with necessary attributes."""
        from poly_data.polymarket_client import _balance_cache

        _balance_cache.clear()

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

                        client = PolymarketClient.__new__(PolymarketClient)
                        client.client = MagicMock()
                        client.browser_wallet = "0x" + "2" * 40
                        client.conditional_tokens = MagicMock()
                        client.neg_risk_ctf_exchange = MagicMock()

                        with patch(
                            "poly_data.polymarket_client.get_rate_limit_manager"
                        ) as mock_rate_limit:
                            mock_manager = MagicMock()
                            mock_rate_limit.return_value = mock_manager

                            yield client

                        _balance_cache.clear()

    def test_cache_invalidated_after_successful_sell(self, mock_client):
        """Test balance cache is invalidated after successful SELL order."""
        from poly_data.polymarket_client import _balance_cache

        token_id = "12345678901234567890"

        # Pre-populate cache
        cache_key = f"{token_id}_False"
        _balance_cache.set(cache_key, 100.0)

        # Verify cache is populated
        assert _balance_cache.get(cache_key) == 100.0

        # Set up successful order
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            100_000_000
        )
        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId=token_id,
                action="SELL",
                price=0.55,
                size=50,
                neg_risk=False,
            )

        # Verify order was successful
        assert result == {"orderID": "test123"}

        # Verify cache was invalidated
        assert (
            _balance_cache.get(cache_key) is None
        ), "Cache should be invalidated after successful SELL"

    def test_cache_not_invalidated_after_buy(self, mock_client):
        """Test balance cache not affected by BUY orders."""
        from poly_data.polymarket_client import _balance_cache

        token_id = "12345678901234567890"

        # Pre-populate cache
        cache_key = f"{token_id}_False"
        _balance_cache.set(cache_key, 100.0)

        # Set up successful BUY order
        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId=token_id,
                action="BUY",
                price=0.45,
                size=50,
                neg_risk=False,
            )

        # Verify order was successful
        assert result == {"orderID": "test123"}

        # Verify cache was NOT invalidated (BUY doesn't affect token balance)
        assert _balance_cache.get(cache_key) == 100.0, "Cache should be preserved after BUY order"

    def test_cache_not_invalidated_on_failed_sell(self, mock_client):
        """Test balance cache preserved when SELL order fails."""
        from poly_data.polymarket_client import _balance_cache
        import requests

        token_id = "12345678901234567890"

        # Pre-populate cache
        cache_key = f"{token_id}_False"
        _balance_cache.set(cache_key, 100.0)

        # Set up failed order (post_order raises exception)
        mock_client.conditional_tokens.functions.balanceOf.return_value.call.return_value = (
            100_000_000
        )
        mock_client.client.create_order.return_value = MagicMock()

        # Create a mock response for the HTTPError
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_client.client.post_order.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            result = mock_client.create_order(
                marketId=token_id,
                action="SELL",
                price=0.55,
                size=50,
                neg_risk=False,
            )

        # Verify order failed
        assert result == {}

        # Verify cache was NOT invalidated (order failed)
        assert (
            _balance_cache.get(cache_key) == 100.0
        ), "Cache should be preserved when SELL order fails"

    def test_cache_invalidation_uses_correct_key_with_neg_risk(self, mock_client):
        """Test cache invalidation uses correct key including neg_risk flag."""
        from poly_data.polymarket_client import _balance_cache

        token_id = "12345678901234567890"

        # Pre-populate cache for both regular and neg_risk
        regular_cache_key = f"{token_id}_False"
        neg_risk_cache_key = f"{token_id}_True"
        _balance_cache.set(regular_cache_key, 100.0)
        _balance_cache.set(neg_risk_cache_key, 50.0)

        # Set up successful neg_risk SELL order
        mock_client.neg_risk_ctf_exchange.functions.balanceOf.return_value.call.return_value = (
            50_000_000
        )
        mock_client.client.create_order.return_value = MagicMock()
        mock_client.client.post_order.return_value = {"orderID": "test123"}

        with patch("poly_data.polymarket_client.get_rate_limit_manager") as mock_rl:
            mock_rl.return_value = MagicMock()

            mock_client.create_order(
                marketId=token_id,
                action="SELL",
                price=0.55,
                size=25,
                neg_risk=True,  # neg_risk order
            )

        # Verify neg_risk cache was invalidated
        assert (
            _balance_cache.get(neg_risk_cache_key) is None
        ), "neg_risk cache should be invalidated"

        # Verify regular cache was NOT invalidated (different key)
        assert _balance_cache.get(regular_cache_key) == 100.0, "Regular cache should be preserved"
