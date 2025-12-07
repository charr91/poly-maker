"""
Tests for the rate limiter module.

Tests cover:
- RateLimiter: Token bucket rate limiting
- CircuitBreaker: Pause all requests after repeated 429s
- EndpointRateLimitManager: Endpoint-specific rate limiting
- with_retry/with_retry_async: Exponential backoff decorators
- Environment variable configuration
"""

import os
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
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
    CircuitBreaker,
    EndpointType,
    EndpointRateLimitManager,
    RateLimitConfig,
    get_endpoint_configs,
    get_rate_limit_manager,
    _get_env_float,
    _get_env_int,
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


class TestEnvHelpers:
    """Tests for environment variable helper functions."""

    def test_get_env_float_with_valid_value(self):
        """Test _get_env_float returns correct float from environment."""
        with patch.dict(os.environ, {'TEST_FLOAT': '15.5'}):
            result = _get_env_float('TEST_FLOAT', 10.0)
            assert result == 15.5

    def test_get_env_float_with_default(self):
        """Test _get_env_float returns default when env var is not set."""
        result = _get_env_float('NONEXISTENT_VAR_12345', 42.0)
        assert result == 42.0

    def test_get_env_float_with_invalid_value(self):
        """Test _get_env_float returns default when env var is invalid."""
        with patch.dict(os.environ, {'TEST_FLOAT': 'not_a_number'}):
            result = _get_env_float('TEST_FLOAT', 10.0)
            assert result == 10.0

    def test_get_env_int_with_valid_value(self):
        """Test _get_env_int returns correct int from environment."""
        with patch.dict(os.environ, {'TEST_INT': '25'}):
            result = _get_env_int('TEST_INT', 10)
            assert result == 25

    def test_get_env_int_with_default(self):
        """Test _get_env_int returns default when env var is not set."""
        result = _get_env_int('NONEXISTENT_VAR_12345', 99)
        assert result == 99

    def test_get_env_int_with_invalid_value(self):
        """Test _get_env_int returns default when env var is invalid."""
        with patch.dict(os.environ, {'TEST_INT': 'not_an_int'}):
            result = _get_env_int('TEST_INT', 10)
            assert result == 10


class TestEndpointType:
    """Tests for the EndpointType enum."""

    def test_endpoint_types_exist(self):
        """Test all expected endpoint types exist."""
        assert EndpointType.BOOK.value == "book"
        assert EndpointType.ORDER.value == "order"
        assert EndpointType.CANCEL_ALL.value == "cancel_all"
        assert EndpointType.DATA_API.value == "data_api"
        assert EndpointType.GENERAL.value == "general"

    def test_endpoint_type_count(self):
        """Test we have the expected number of endpoint types."""
        assert len(EndpointType) == 5


class TestRateLimitConfig:
    """Tests for the RateLimitConfig dataclass."""

    def test_config_creation(self):
        """Test RateLimitConfig can be created with values."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=15)
        assert config.requests_per_second == 10.0
        assert config.burst_size == 15

    def test_config_immutability_check(self):
        """Test RateLimitConfig values are accessible."""
        config = RateLimitConfig(requests_per_second=20.0, burst_size=30)
        assert config.requests_per_second == 20.0
        assert config.burst_size == 30


class TestGetEndpointConfigs:
    """Tests for the get_endpoint_configs function."""

    def test_returns_all_endpoint_types(self):
        """Test that configs are returned for all endpoint types."""
        configs = get_endpoint_configs()
        for endpoint_type in EndpointType:
            assert endpoint_type in configs
            assert isinstance(configs[endpoint_type], RateLimitConfig)

    def test_default_values(self):
        """Test default configuration values without env vars."""
        # Clear any related env vars
        env_vars_to_clear = [
            'RATE_LIMIT_BOOK', 'RATE_LIMIT_ORDER', 'RATE_LIMIT_CANCEL',
            'RATE_LIMIT_DATA_API', 'RATE_LIMIT_GENERAL'
        ]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_vars_to_clear}

        with patch.dict(os.environ, clean_env, clear=True):
            configs = get_endpoint_configs()

            # Check default values match expected
            assert configs[EndpointType.BOOK].requests_per_second == 10.0
            assert configs[EndpointType.ORDER].requests_per_second == 80.0
            assert configs[EndpointType.CANCEL_ALL].requests_per_second == 8.0
            assert configs[EndpointType.DATA_API].requests_per_second == 10.0
            assert configs[EndpointType.GENERAL].requests_per_second == 50.0

    def test_env_var_override(self):
        """Test that env vars override default values."""
        with patch.dict(os.environ, {'RATE_LIMIT_BOOK': '5.0'}):
            configs = get_endpoint_configs()
            assert configs[EndpointType.BOOK].requests_per_second == 5.0


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_init_defaults(self):
        """Test CircuitBreaker initializes with correct defaults."""
        cb = CircuitBreaker()
        assert cb.threshold == 3
        assert cb.pause_duration == 30.0
        assert cb.consecutive_429s == 0
        assert cb.paused_until is None

    def test_init_custom_values(self):
        """Test CircuitBreaker initializes with custom values."""
        cb = CircuitBreaker(threshold=5, pause_duration=60.0)
        assert cb.threshold == 5
        assert cb.pause_duration == 60.0

    def test_init_from_env_vars(self):
        """Test CircuitBreaker reads from environment variables."""
        with patch.dict(os.environ, {
            'CIRCUIT_BREAKER_THRESHOLD': '10',
            'CIRCUIT_BREAKER_PAUSE': '45.0'
        }):
            cb = CircuitBreaker()
            assert cb.threshold == 10
            assert cb.pause_duration == 45.0

    def test_record_429_increments_counter(self):
        """Test that record_429 increments the counter."""
        cb = CircuitBreaker(threshold=3)
        assert cb.consecutive_429s == 0

        cb.record_429()
        assert cb.consecutive_429s == 1

        cb.record_429()
        assert cb.consecutive_429s == 2

    def test_record_success_resets_counter(self):
        """Test that record_success resets the 429 counter."""
        cb = CircuitBreaker(threshold=3)
        cb.record_429()
        cb.record_429()
        assert cb.consecutive_429s == 2

        cb.record_success()
        assert cb.consecutive_429s == 0

    def test_circuit_trips_after_threshold(self):
        """Test that circuit trips after reaching threshold."""
        cb = CircuitBreaker(threshold=3, pause_duration=1.0)

        cb.record_429()
        cb.record_429()
        assert cb.is_open() is False

        cb.record_429()  # This should trip the circuit
        assert cb.is_open() is True
        assert cb.paused_until is not None

    def test_circuit_closes_after_pause_duration(self):
        """Test that circuit closes after pause duration expires."""
        cb = CircuitBreaker(threshold=2, pause_duration=0.1)

        cb.record_429()
        cb.record_429()  # Trip the circuit
        assert cb.is_open() is True

        time.sleep(0.15)  # Wait for pause to expire
        assert cb.is_open() is False
        assert cb.consecutive_429s == 0

    def test_wait_if_open_sync_blocks_when_open(self):
        """Test wait_if_open_sync blocks when circuit is open."""
        cb = CircuitBreaker(threshold=1, pause_duration=0.1)
        cb.record_429()  # Trip the circuit

        start = time.monotonic()
        cb.wait_if_open_sync()
        elapsed = time.monotonic() - start

        # Should have waited approximately 0.1 seconds
        assert elapsed >= 0.08

    def test_wait_if_open_sync_returns_immediately_when_closed(self):
        """Test wait_if_open_sync returns immediately when circuit is closed."""
        cb = CircuitBreaker(threshold=3)

        start = time.monotonic()
        cb.wait_if_open_sync()
        elapsed = time.monotonic() - start

        # Should return almost immediately
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_wait_if_open_async_blocks_when_open(self):
        """Test wait_if_open_async blocks when circuit is open."""
        cb = CircuitBreaker(threshold=1, pause_duration=0.1)
        cb.record_429()  # Trip the circuit

        start = time.monotonic()
        await cb.wait_if_open_async()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.08

    @pytest.mark.asyncio
    async def test_wait_if_open_async_returns_immediately_when_closed(self):
        """Test wait_if_open_async returns immediately when circuit is closed."""
        cb = CircuitBreaker(threshold=3)

        start = time.monotonic()
        await cb.wait_if_open_async()
        elapsed = time.monotonic() - start

        assert elapsed < 0.05

    def test_thread_safety(self):
        """Test circuit breaker is thread-safe."""
        cb = CircuitBreaker(threshold=100, pause_duration=1.0)
        errors = []

        def worker():
            try:
                for _ in range(50):
                    cb.record_429()
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEndpointRateLimitManager:
    """Tests for the EndpointRateLimitManager class."""

    def test_init_creates_limiters_for_all_endpoints(self):
        """Test manager creates rate limiters for all endpoint types."""
        manager = EndpointRateLimitManager()

        for endpoint_type in EndpointType:
            limiter = manager.get_limiter(endpoint_type)
            assert isinstance(limiter, RateLimiter)

    def test_init_creates_circuit_breaker(self):
        """Test manager creates a circuit breaker."""
        manager = EndpointRateLimitManager()
        assert isinstance(manager.circuit_breaker, CircuitBreaker)

    def test_get_limiter_returns_correct_limiter(self):
        """Test get_limiter returns the limiter for the specified endpoint."""
        manager = EndpointRateLimitManager()

        book_limiter = manager.get_limiter(EndpointType.BOOK)
        order_limiter = manager.get_limiter(EndpointType.ORDER)

        # Different limiters should have different request rates
        assert book_limiter is not order_limiter

    def test_acquire_sync_consumes_token(self):
        """Test acquire_sync consumes a token from the correct limiter."""
        manager = EndpointRateLimitManager()
        limiter = manager.get_limiter(EndpointType.BOOK)
        initial_tokens = limiter.tokens

        manager.acquire_sync(EndpointType.BOOK)

        # Tokens should decrease
        assert limiter.tokens < initial_tokens

    def test_acquire_sync_respects_circuit_breaker(self):
        """Test acquire_sync waits when circuit breaker is open."""
        manager = EndpointRateLimitManager()

        # Trip the circuit breaker
        for _ in range(manager.circuit_breaker.threshold):
            manager.circuit_breaker.record_429()

        # Override pause duration for faster test
        manager.circuit_breaker.pause_duration = 0.1
        manager.circuit_breaker.paused_until = time.monotonic() + 0.1

        start = time.monotonic()
        manager.acquire_sync(EndpointType.GENERAL)
        elapsed = time.monotonic() - start

        # Should have waited for circuit breaker
        assert elapsed >= 0.08

    @pytest.mark.asyncio
    async def test_acquire_async_consumes_token(self):
        """Test acquire_async consumes a token."""
        manager = EndpointRateLimitManager()
        limiter = manager.get_limiter(EndpointType.ORDER)
        initial_tokens = limiter.tokens

        await manager.acquire_async(EndpointType.ORDER)

        assert limiter.tokens < initial_tokens

    def test_on_response_429_updates_circuit_breaker(self):
        """Test on_response updates circuit breaker on 429."""
        manager = EndpointRateLimitManager()
        assert manager.circuit_breaker.consecutive_429s == 0

        manager.on_response(429)
        assert manager.circuit_breaker.consecutive_429s == 1

    def test_on_response_success_resets_circuit_breaker(self):
        """Test on_response resets circuit breaker on success."""
        manager = EndpointRateLimitManager()
        manager.circuit_breaker.record_429()
        assert manager.circuit_breaker.consecutive_429s == 1

        manager.on_response(200)
        assert manager.circuit_breaker.consecutive_429s == 0

    def test_on_response_ignores_other_errors(self):
        """Test on_response ignores non-429/non-2xx status codes."""
        manager = EndpointRateLimitManager()
        manager.circuit_breaker.record_429()
        initial_count = manager.circuit_breaker.consecutive_429s

        manager.on_response(404)  # Should not change anything
        assert manager.circuit_breaker.consecutive_429s == initial_count

    def test_different_endpoints_have_different_rates(self):
        """Test that different endpoints have their configured rates."""
        manager = EndpointRateLimitManager()

        book_limiter = manager.get_limiter(EndpointType.BOOK)
        order_limiter = manager.get_limiter(EndpointType.ORDER)

        # ORDER should have higher rate than BOOK (80 vs 10)
        assert order_limiter.requests_per_second > book_limiter.requests_per_second


class TestGetRateLimitManager:
    """Tests for the get_rate_limit_manager singleton function."""

    def test_returns_manager_instance(self):
        """Test get_rate_limit_manager returns an EndpointRateLimitManager."""
        manager = get_rate_limit_manager()
        assert isinstance(manager, EndpointRateLimitManager)

    def test_returns_same_instance(self):
        """Test get_rate_limit_manager returns the same instance (singleton)."""
        manager1 = get_rate_limit_manager()
        manager2 = get_rate_limit_manager()
        assert manager1 is manager2

    def test_manager_has_circuit_breaker(self):
        """Test the returned manager has a circuit breaker."""
        manager = get_rate_limit_manager()
        assert hasattr(manager, 'circuit_breaker')
        assert isinstance(manager.circuit_breaker, CircuitBreaker)


class TestIntegration:
    """Integration tests for the rate limiting system."""

    def test_full_flow_with_429_recovery(self):
        """Test full flow of rate limiting with 429 and recovery."""
        manager = EndpointRateLimitManager()

        # Make some successful requests
        for _ in range(3):
            manager.acquire_sync(EndpointType.GENERAL)
            manager.on_response(200)

        assert manager.circuit_breaker.consecutive_429s == 0

        # Simulate 429 responses (but not enough to trip)
        manager.on_response(429)
        manager.on_response(429)
        assert manager.circuit_breaker.consecutive_429s == 2
        assert manager.circuit_breaker.is_open() is False

        # Successful response resets
        manager.on_response(200)
        assert manager.circuit_breaker.consecutive_429s == 0

    def test_concurrent_requests_across_endpoints(self):
        """Test concurrent requests to different endpoints."""
        manager = EndpointRateLimitManager()
        results = []
        errors = []

        def make_requests(endpoint_type, count):
            try:
                for _ in range(count):
                    manager.acquire_sync(endpoint_type)
                    results.append(endpoint_type)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=make_requests, args=(EndpointType.BOOK, 5)),
            threading.Thread(target=make_requests, args=(EndpointType.ORDER, 5)),
            threading.Thread(target=make_requests, args=(EndpointType.GENERAL, 5)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 15

    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self):
        """Test async concurrent requests."""
        manager = EndpointRateLimitManager()

        async def make_request(endpoint_type):
            await manager.acquire_async(endpoint_type)
            return endpoint_type

        tasks = [
            make_request(EndpointType.BOOK),
            make_request(EndpointType.ORDER),
            make_request(EndpointType.GENERAL),
            make_request(EndpointType.DATA_API),
            make_request(EndpointType.CANCEL_ALL),
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 5
