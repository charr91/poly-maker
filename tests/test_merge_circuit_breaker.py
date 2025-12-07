"""
Tests for the merge circuit breaker module.

Tests cover:
- MergeCircuitBreaker: Per-market cooldowns with exponential backoff
- Node.js availability check
- Cooldown calculation and expiration
- Environment variable configuration
"""

import os
import time
import threading
from unittest.mock import patch

import pytest

from poly_data.merge_circuit_breaker import (
    MergeCircuitBreaker,
    MarketMergeState,
    get_merge_circuit_breaker,
    _get_env_float,
)


class TestMergeCircuitBreaker:
    """Tests for the MergeCircuitBreaker class."""

    def test_init_defaults(self):
        """Test MergeCircuitBreaker initializes with correct defaults."""
        cb = MergeCircuitBreaker()
        assert cb.initial_cooldown == 60.0
        assert cb.max_cooldown == 3600.0
        assert cb.multiplier == 2.0

    def test_init_custom_values(self):
        """Test MergeCircuitBreaker initializes with custom values."""
        cb = MergeCircuitBreaker(initial_cooldown=30.0, max_cooldown=1800.0, multiplier=1.5)
        assert cb.initial_cooldown == 30.0
        assert cb.max_cooldown == 1800.0
        assert cb.multiplier == 1.5

    def test_init_from_env_vars(self):
        """Test MergeCircuitBreaker reads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "MERGE_COOLDOWN_INITIAL": "120",
                "MERGE_COOLDOWN_MAX": "7200",
                "MERGE_COOLDOWN_MULTIPLIER": "3.0",
            },
        ):
            cb = MergeCircuitBreaker()
            assert cb.initial_cooldown == 120.0
            assert cb.max_cooldown == 7200.0
            assert cb.multiplier == 3.0

    def test_can_merge_returns_true_initially(self):
        """Test can_merge returns True for new markets."""
        cb = MergeCircuitBreaker()
        with patch.object(cb, "check_node_available", return_value=True):
            assert cb.can_merge("market123") is True

    def test_can_merge_returns_false_when_node_unavailable(self):
        """Test can_merge returns False when Node.js is not available."""
        cb = MergeCircuitBreaker()
        cb._node_available = False
        assert cb.can_merge("market123") is False

    def test_can_merge_returns_false_during_cooldown(self):
        """Test can_merge returns False during cooldown period."""
        cb = MergeCircuitBreaker(initial_cooldown=1.0)
        cb._node_available = True

        cb.record_failure("market123", "Test error")
        assert cb.can_merge("market123") is False

    def test_can_merge_returns_true_after_cooldown_expires(self):
        """Test can_merge returns True after cooldown expires."""
        cb = MergeCircuitBreaker(initial_cooldown=0.1)
        cb._node_available = True

        cb.record_failure("market123", "Test error")
        assert cb.can_merge("market123") is False

        time.sleep(0.15)
        assert cb.can_merge("market123") is True

    def test_record_failure_increments_counter(self):
        """Test record_failure increments consecutive failure count."""
        cb = MergeCircuitBreaker()

        cb.record_failure("market123", "Error 1")
        state = cb.get_status("market123")
        assert state.consecutive_failures == 1

        cb.record_failure("market123", "Error 2")
        state = cb.get_status("market123")
        assert state.consecutive_failures == 2

    def test_record_failure_exponential_backoff(self):
        """Test cooldown increases exponentially with failures."""
        cb = MergeCircuitBreaker(initial_cooldown=10.0, multiplier=2.0)

        cb.record_failure("market123", "Error 1")
        state1 = cb.get_status("market123")
        cooldown1 = state1.cooldown_until - state1.last_failure_time

        cb.record_failure("market123", "Error 2")
        state2 = cb.get_status("market123")
        cooldown2 = state2.cooldown_until - state2.last_failure_time

        # Second cooldown should be ~2x first (with tolerance for timing)
        assert 18 <= cooldown2 <= 22

    def test_record_failure_respects_max_cooldown(self):
        """Test cooldown doesn't exceed max_cooldown."""
        cb = MergeCircuitBreaker(initial_cooldown=100.0, max_cooldown=200.0, multiplier=10.0)

        # After many failures, cooldown should cap at max
        for i in range(10):
            cb.record_failure("market123", f"Error {i}")

        state = cb.get_status("market123")
        cooldown = state.cooldown_until - state.last_failure_time
        assert cooldown <= 200.0

    def test_record_success_resets_state(self):
        """Test record_success clears failure state."""
        cb = MergeCircuitBreaker()

        cb.record_failure("market123", "Error")
        assert cb.get_status("market123") is not None

        cb.record_success("market123")
        assert cb.get_status("market123") is None

    def test_check_node_available_caches_result(self):
        """Test Node.js check is cached after first call."""
        cb = MergeCircuitBreaker()

        with patch("shutil.which", return_value="/usr/bin/node") as mock_which:
            result1 = cb.check_node_available()
            result2 = cb.check_node_available()

            assert result1 is True
            assert result2 is True
            assert mock_which.call_count == 1

    def test_check_node_available_returns_false_when_not_found(self):
        """Test check_node_available returns False when node not found."""
        cb = MergeCircuitBreaker()

        with patch("shutil.which", return_value=None):
            result = cb.check_node_available()
            assert result is False

    def test_different_markets_independent(self):
        """Test that different markets have independent cooldowns."""
        cb = MergeCircuitBreaker(initial_cooldown=1.0)
        cb._node_available = True

        cb.record_failure("market1", "Error")

        assert cb.can_merge("market1") is False
        assert cb.can_merge("market2") is True

    def test_thread_safety(self):
        """Test circuit breaker is thread-safe."""
        cb = MergeCircuitBreaker()
        cb._node_available = True
        errors = []

        def worker(market_id):
            try:
                for _ in range(50):
                    cb.can_merge(market_id)
                    cb.record_failure(market_id, "Error")
                    cb.record_success(market_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"market{i}",)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_error_message_truncation(self):
        """Test that long error messages are truncated."""
        cb = MergeCircuitBreaker()
        long_error = "x" * 500

        cb.record_failure("market123", long_error)
        state = cb.get_status("market123")

        assert len(state.last_error) == 200


class TestMarketMergeState:
    """Tests for the MarketMergeState dataclass."""

    def test_default_values(self):
        """Test MarketMergeState has correct defaults."""
        state = MarketMergeState()
        assert state.last_failure_time == 0.0
        assert state.consecutive_failures == 0
        assert state.cooldown_until == 0.0
        assert state.last_error == ""


class TestGetMergeCircuitBreaker:
    """Tests for the singleton getter function."""

    def test_returns_instance(self):
        """Test get_merge_circuit_breaker returns an instance."""
        cb = get_merge_circuit_breaker()
        assert isinstance(cb, MergeCircuitBreaker)

    def test_returns_same_instance(self):
        """Test get_merge_circuit_breaker returns singleton."""
        cb1 = get_merge_circuit_breaker()
        cb2 = get_merge_circuit_breaker()
        assert cb1 is cb2


class TestEnvHelpers:
    """Tests for environment variable helper functions."""

    def test_get_env_float_returns_default(self):
        """Test _get_env_float returns default when env var not set."""
        result = _get_env_float("NONEXISTENT_VAR_12345", 42.0)
        assert result == 42.0

    def test_get_env_float_parses_env_var(self):
        """Test _get_env_float parses environment variable."""
        with patch.dict(os.environ, {"TEST_FLOAT": "123.5"}):
            result = _get_env_float("TEST_FLOAT", 0.0)
            assert result == 123.5

    def test_get_env_float_returns_default_on_invalid(self):
        """Test _get_env_float returns default for invalid values."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_number"}):
            result = _get_env_float("TEST_FLOAT", 99.0)
            assert result == 99.0
