"""
Tests for the rate limiter module.
"""

import time
import asyncio
import threading
from unittest.mock import Mock, patch
import pytest

from poly_data.rate_limiter import (
    RateLimiter,
    with_retry,
    with_retry_async,
    _calculate_delay,
    is_rate_limited,
    is_server_error,
    configure_rate_limiter,
    get_rate_limiter,
)


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_init_defaults(self):
        """Test RateLimiter initializes with correct defaults."""
        limiter = RateLimiter()
        assert limiter.requests_per_second == 5.0
        assert limiter.burst_size == 10
        assert limiter.tokens == 10

    def test_init_custom_values(self):
        """Test RateLimiter initializes with custom values."""
        limiter = RateLimiter(requests_per_second=10.0, burst_size=20)
        assert limiter.requests_per_second == 10.0
        assert limiter.burst_size == 20
        assert limiter.tokens == 20

    def test_acquire_sync_consumes_token(self):
        """Test that acquire_sync consumes a token."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)
        initial_tokens = limiter.tokens

        limiter.acquire_sync()

        # Tokens should decrease (accounting for small refill during execution)
        assert limiter.tokens < initial_tokens

    def test_acquire_sync_blocks_when_no_tokens(self):
        """Test that acquire_sync blocks when tokens are exhausted."""
        limiter = RateLimiter(requests_per_second=10, burst_size=1)

        # Consume the only token
        limiter.acquire_sync()

        # Record start time
        start = time.monotonic()

        # This should block until a token is available
        limiter.acquire_sync()

        elapsed = time.monotonic() - start

        # Should have waited approximately 0.1 seconds (1/10 rps)
        assert elapsed >= 0.05  # Allow some tolerance

    def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)

        # Consume all tokens
        for _ in range(10):
            limiter.acquire_sync()

        # Wait for refill
        time.sleep(0.1)  # Should refill ~10 tokens at 100 rps

        # Force refill calculation
        limiter._refill()

        # Should have some tokens now
        assert limiter.tokens > 0

    def test_tokens_dont_exceed_burst_size(self):
        """Test that tokens don't exceed burst_size."""
        limiter = RateLimiter(requests_per_second=100, burst_size=5)

        # Wait a long time
        time.sleep(0.2)

        # Force refill
        limiter._refill()

        # Tokens should not exceed burst size
        assert limiter.tokens <= 5

    @pytest.mark.asyncio
    async def test_acquire_async(self):
        """Test async acquire works correctly."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)
        initial_tokens = limiter.tokens

        await limiter.acquire()

        assert limiter.tokens < initial_tokens

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        limiter = RateLimiter(requests_per_second=1000, burst_size=100)
        call_count = [0]
        lock = threading.Lock()

        def worker():
            for _ in range(10):
                limiter.acquire_sync()
                with lock:
                    call_count[0] += 1

        threads = [threading.Thread(target=worker) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All calls should complete
        assert call_count[0] == 50


class TestWithRetry:
    """Tests for the with_retry decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        call_count = [0]

        @with_retry(max_retries=3)
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_connection_error(self):
        """Test retry on connection errors."""
        import requests

        call_count = [0]

        @with_retry(max_retries=2, base_delay=0.01)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise requests.exceptions.ConnectionError("Connection failed")
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count[0] == 3

    def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries."""
        import requests

        call_count = [0]

        @with_retry(max_retries=2, base_delay=0.01)
        def always_failing():
            call_count[0] += 1
            raise requests.exceptions.ConnectionError("Always fails")

        with pytest.raises(requests.exceptions.ConnectionError):
            always_failing()

        assert call_count[0] == 3  # Initial + 2 retries

    def test_rate_limit_response_triggers_retry(self):
        """Test that 429 responses trigger retry."""
        import requests

        call_count = [0]
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 429
        mock_response.headers = {}

        success_response = Mock(spec=requests.Response)
        success_response.status_code = 200

        @with_retry(max_retries=3, base_delay=0.01)
        def rate_limited_func():
            call_count[0] += 1
            if call_count[0] < 2:
                return mock_response
            return success_response

        result = rate_limited_func()

        assert result.status_code == 200
        assert call_count[0] == 2

    def test_server_error_triggers_retry(self):
        """Test that 5xx errors trigger retry."""
        import requests

        call_count = [0]
        error_response = Mock(spec=requests.Response)
        error_response.status_code = 503

        success_response = Mock(spec=requests.Response)
        success_response.status_code = 200

        @with_retry(max_retries=3, base_delay=0.01)
        def server_error_func():
            call_count[0] += 1
            if call_count[0] < 2:
                return error_response
            return success_response

        result = server_error_func()

        assert result.status_code == 200
        assert call_count[0] == 2

    def test_non_retryable_exception_not_retried(self):
        """Test that non-retryable exceptions are raised immediately."""
        call_count = [0]

        @with_retry(max_retries=3, base_delay=0.01)
        def value_error_func():
            call_count[0] += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            value_error_func()

        assert call_count[0] == 1

    def test_with_rate_limiter(self):
        """Test that rate limiter is applied before each request."""
        limiter = RateLimiter(requests_per_second=100, burst_size=10)
        call_count = [0]

        @with_retry(max_retries=0, rate_limiter=limiter)
        def func_with_limiter():
            call_count[0] += 1
            return "success"

        # Make multiple calls
        for _ in range(3):
            func_with_limiter()

        assert call_count[0] == 3


class TestWithRetryAsync:
    """Tests for the async retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_async_call(self):
        """Test successful async function call."""
        call_count = [0]

        @with_retry_async(max_retries=3)
        async def async_func():
            call_count[0] += 1
            return "async_success"

        result = await async_func()

        assert result == "async_success"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_error(self):
        """Test retry on async errors."""
        call_count = [0]

        @with_retry_async(max_retries=2, base_delay=0.01)
        async def failing_async():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Async connection failed")
            return "recovered"

        result = await failing_async()

        assert result == "recovered"
        assert call_count[0] == 2


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_rate_limited(self):
        """Test rate limit detection."""
        import requests

        response_429 = Mock(spec=requests.Response)
        response_429.status_code = 429

        response_200 = Mock(spec=requests.Response)
        response_200.status_code = 200

        assert is_rate_limited(response_429) is True
        assert is_rate_limited(response_200) is False

    def test_is_server_error(self):
        """Test server error detection."""
        import requests

        for status_code in [500, 502, 503, 504]:
            response = Mock(spec=requests.Response)
            response.status_code = status_code
            assert is_server_error(response) is True

        response_200 = Mock(spec=requests.Response)
        response_200.status_code = 200
        assert is_server_error(response_200) is False

        response_404 = Mock(spec=requests.Response)
        response_404.status_code = 404
        assert is_server_error(response_404) is False

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        base = 1.0
        max_delay = 60.0

        # Without jitter for predictable testing
        delay_0 = _calculate_delay(0, base, max_delay, jitter=False)
        delay_1 = _calculate_delay(1, base, max_delay, jitter=False)
        delay_2 = _calculate_delay(2, base, max_delay, jitter=False)

        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0

    def test_calculate_delay_respects_max(self):
        """Test that delay doesn't exceed max_delay."""
        delay = _calculate_delay(10, base_delay=1.0, max_delay=30.0, jitter=False)
        assert delay == 30.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness."""
        delays = [_calculate_delay(0, 1.0, 60.0, jitter=True) for _ in range(10)]

        # With jitter, not all delays should be identical
        # (very small chance of failure due to randomness)
        assert len(set(delays)) > 1

    def test_configure_rate_limiter(self):
        """Test global rate limiter configuration."""
        configure_rate_limiter(requests_per_second=20.0, burst_size=30)

        limiter = get_rate_limiter()
        assert limiter.requests_per_second == 20.0
        assert limiter.burst_size == 30

        # Reset to defaults for other tests
        configure_rate_limiter(requests_per_second=5.0, burst_size=10)
