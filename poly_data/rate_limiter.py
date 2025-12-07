"""
Rate limiting utilities for Polymarket API calls.

Provides token bucket rate limiting and exponential backoff retry logic
to prevent API bans when operating on many markets.

Features:
- Endpoint-specific rate limits (different limits for /book, /order, /cancel-all, etc.)
- Circuit breaker to pause all requests after repeated 429s
- Environment variable configuration for all rate limits
- Thread-safe implementation for both sync and async code
"""

import os
import time
import asyncio
import functools
import threading
import random
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar, Any, Dict, Optional

import requests


# Configure logging
logger = logging.getLogger('poly_maker.rate_limiter')

# Type variable for generic decorator
T = TypeVar('T')


class EndpointType(Enum):
    """API endpoint categories with different rate limits."""
    BOOK = "book"              # /book, /price, /midpoint - 20 req/sec official
    ORDER = "order"            # POST/DELETE /order - 240 req/sec burst official
    CANCEL_ALL = "cancel_all"  # /cancel-all - 20 req/sec burst official
    DATA_API = "data_api"      # data-api.polymarket.com - 20 req/sec official
    GENERAL = "general"        # catch-all - 500 req/sec official


@dataclass
class RateLimitConfig:
    """Configuration for an endpoint's rate limit."""
    requests_per_second: float
    burst_size: int


def _get_env_float(name: str, default: float) -> float:
    """Get a float from environment variable with default."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    """Get an int from environment variable with default."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_endpoint_configs() -> Dict[EndpointType, RateLimitConfig]:
    """
    Get rate limit configurations for each endpoint type.

    Configurable via environment variables:
    - RATE_LIMIT_BOOK: /book, /price endpoints (default: 10 req/sec, 50% of 20 limit)
    - RATE_LIMIT_ORDER: POST/DELETE /order (default: 80 req/sec, 33% of 240 limit)
    - RATE_LIMIT_CANCEL: /cancel-all (default: 8 req/sec, 40% of 20 limit)
    - RATE_LIMIT_DATA_API: data-api.polymarket.com (default: 10 req/sec, 50% of 20 limit)
    - RATE_LIMIT_GENERAL: catch-all (default: 50 req/sec, 10% of 500 limit)
    """
    return {
        EndpointType.BOOK: RateLimitConfig(
            requests_per_second=_get_env_float('RATE_LIMIT_BOOK', 10.0),
            burst_size=_get_env_int('RATE_LIMIT_BOOK_BURST', 15)
        ),
        EndpointType.ORDER: RateLimitConfig(
            requests_per_second=_get_env_float('RATE_LIMIT_ORDER', 80.0),
            burst_size=_get_env_int('RATE_LIMIT_ORDER_BURST', 120)
        ),
        EndpointType.CANCEL_ALL: RateLimitConfig(
            requests_per_second=_get_env_float('RATE_LIMIT_CANCEL', 8.0),
            burst_size=_get_env_int('RATE_LIMIT_CANCEL_BURST', 12)
        ),
        EndpointType.DATA_API: RateLimitConfig(
            requests_per_second=_get_env_float('RATE_LIMIT_DATA_API', 10.0),
            burst_size=_get_env_int('RATE_LIMIT_DATA_API_BURST', 15)
        ),
        EndpointType.GENERAL: RateLimitConfig(
            requests_per_second=_get_env_float('RATE_LIMIT_GENERAL', 50.0),
            burst_size=_get_env_int('RATE_LIMIT_GENERAL_BURST', 75)
        ),
    }


class RateLimiter:
    """
    Token bucket rate limiter for controlling API request rates.

    Thread-safe implementation that works with both sync and async code.
    """

    def __init__(self, requests_per_second: float = 5.0, burst_size: int = 10):
        """
        Initialize the rate limiter.

        Args:
            requests_per_second: Maximum sustained request rate
            burst_size: Maximum number of requests allowed in a burst
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
        self._async_lock = None  # Lazy initialization for async lock

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.requests_per_second
        )
        self.last_update = now

    def acquire_sync(self) -> None:
        """
        Acquire a token synchronously, blocking if necessary.

        Use this in regular (non-async) code.
        """
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                # Calculate wait time
                wait_time = (1 - self.tokens) / self.requests_per_second

            time.sleep(wait_time)

    async def acquire(self) -> None:
        """
        Acquire a token asynchronously, yielding if necessary.

        Use this in async code.
        """
        # Lazy initialize async lock
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        while True:
            async with self._async_lock:
                with self._lock:
                    self._refill()
                    if self.tokens >= 1:
                        self.tokens -= 1
                        return
                    wait_time = (1 - self.tokens) / self.requests_per_second

            await asyncio.sleep(wait_time)


class RetryableError(Exception):
    """Exception indicating the request should be retried."""
    pass


def is_rate_limited(response: requests.Response) -> bool:
    """Check if a response indicates rate limiting."""
    return response.status_code == 429


def is_server_error(response: requests.Response) -> bool:
    """Check if a response indicates a server error (potentially retryable)."""
    return 500 <= response.status_code < 600


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    rate_limiter: RateLimiter = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for exponential backoff retry on failures and rate limits.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delays
        rate_limiter: Optional RateLimiter instance to use before each request

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Apply rate limiting if provided
                    if rate_limiter is not None:
                        rate_limiter.acquire_sync()

                    result = func(*args, **kwargs)

                    # Check if result is a Response object with rate limit status
                    if isinstance(result, requests.Response):
                        if is_rate_limited(result):
                            # Get retry-after header if available
                            retry_after = result.headers.get('Retry-After')
                            if retry_after:
                                delay = float(retry_after)
                            else:
                                delay = _calculate_delay(attempt, base_delay, max_delay, jitter)

                            print(f"Rate limited (429). Waiting {delay:.2f}s before retry {attempt + 1}/{max_retries}")
                            time.sleep(delay)
                            continue

                        if is_server_error(result):
                            delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
                            print(f"Server error ({result.status_code}). Waiting {delay:.2f}s before retry {attempt + 1}/{max_retries}")
                            time.sleep(delay)
                            continue

                    return result

                except (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        RetryableError) as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
                        print(f"Request failed ({type(e).__name__}). Waiting {delay:.2f}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(delay)
                    else:
                        print(f"Request failed after {max_retries} retries: {e}")
                        raise

                except Exception as e:
                    # Non-retryable exceptions are raised immediately
                    raise

            # If we've exhausted retries due to rate limiting/server errors
            if last_exception:
                raise last_exception
            return result

        return wrapper
    return decorator


def with_retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    rate_limiter: RateLimiter = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async version of with_retry decorator.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delays
        rate_limiter: Optional RateLimiter instance to use before each request

    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    # Apply rate limiting if provided
                    if rate_limiter is not None:
                        await rate_limiter.acquire()

                    result = await func(*args, **kwargs)
                    return result

                except (asyncio.TimeoutError, ConnectionError, RetryableError) as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = _calculate_delay(attempt, base_delay, max_delay, jitter)
                        print(f"Async request failed ({type(e).__name__}). Waiting {delay:.2f}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(delay)
                    else:
                        print(f"Async request failed after {max_retries} retries: {e}")
                        raise

                except Exception as e:
                    raise

            if last_exception:
                raise last_exception
            return result

        return wrapper
    return decorator


def _calculate_delay(attempt: int, base_delay: float, max_delay: float, jitter: bool) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)

    if jitter:
        # Add random jitter between 0% and 25% of the delay
        delay = delay * (1 + random.random() * 0.25)

    return delay


class CircuitBreaker:
    """
    Circuit breaker to pause all requests after repeated 429 responses.

    When the circuit is "open" (tripped), all requests are paused for a
    configurable duration to allow the API rate limits to reset.
    """

    def __init__(
        self,
        threshold: int = None,
        pause_duration: float = None
    ):
        """
        Initialize the circuit breaker.

        Args:
            threshold: Number of consecutive 429s before tripping (default from env: 3)
            pause_duration: Seconds to pause when tripped (default from env: 30)
        """
        self.threshold = threshold or _get_env_int('CIRCUIT_BREAKER_THRESHOLD', 3)
        self.pause_duration = pause_duration or _get_env_float('CIRCUIT_BREAKER_PAUSE', 30.0)
        self.consecutive_429s = 0
        self.paused_until: Optional[float] = None
        self._lock = threading.Lock()

    def record_429(self) -> None:
        """Record a 429 response. May trip the circuit breaker."""
        with self._lock:
            self.consecutive_429s += 1
            logger.warning(f"Rate limit 429 received ({self.consecutive_429s}/{self.threshold})")
            if self.consecutive_429s >= self.threshold:
                self.paused_until = time.monotonic() + self.pause_duration
                logger.error(
                    f"Circuit breaker TRIPPED - pausing ALL requests for {self.pause_duration}s"
                )

    def record_success(self) -> None:
        """Record a successful response. Resets the 429 counter."""
        with self._lock:
            if self.consecutive_429s > 0:
                logger.debug(f"Circuit breaker reset after {self.consecutive_429s} consecutive 429s")
            self.consecutive_429s = 0

    def is_open(self) -> bool:
        """Check if the circuit breaker is currently open (tripped)."""
        with self._lock:
            if self.paused_until and time.monotonic() < self.paused_until:
                return True
            if self.paused_until:
                logger.info("Circuit breaker CLOSED - resuming requests")
                self.paused_until = None
                self.consecutive_429s = 0
            return False

    def wait_if_open_sync(self) -> None:
        """Block until the circuit breaker closes (sync version)."""
        while self.is_open():
            with self._lock:
                if self.paused_until:
                    remaining = self.paused_until - time.monotonic()
                else:
                    remaining = 0
            if remaining > 0:
                logger.debug(f"Circuit breaker open, waiting {remaining:.1f}s")
                time.sleep(min(remaining, 1.0))

    async def wait_if_open_async(self) -> None:
        """Wait until the circuit breaker closes (async version)."""
        while self.is_open():
            with self._lock:
                if self.paused_until:
                    remaining = self.paused_until - time.monotonic()
                else:
                    remaining = 0
            if remaining > 0:
                logger.debug(f"Circuit breaker open, waiting {remaining:.1f}s")
                await asyncio.sleep(min(remaining, 0.5))


class EndpointRateLimitManager:
    """
    Manages rate limiters for different API endpoint types.

    Provides endpoint-specific rate limiting with a shared circuit breaker.
    """

    def __init__(self):
        """Initialize the manager with endpoint-specific rate limiters."""
        self._limiters: Dict[EndpointType, RateLimiter] = {}
        self._circuit_breaker = CircuitBreaker()
        self._lock = threading.Lock()

        # Initialize limiters for each endpoint type
        configs = get_endpoint_configs()
        for endpoint_type, config in configs.items():
            self._limiters[endpoint_type] = RateLimiter(
                requests_per_second=config.requests_per_second,
                burst_size=config.burst_size
            )
            logger.info(
                f"Rate limiter initialized: {endpoint_type.value} = "
                f"{config.requests_per_second} req/sec, burst={config.burst_size}"
            )

    def get_limiter(self, endpoint_type: EndpointType) -> RateLimiter:
        """Get the rate limiter for a specific endpoint type."""
        return self._limiters[endpoint_type]

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the shared circuit breaker."""
        return self._circuit_breaker

    def acquire_sync(self, endpoint_type: EndpointType = EndpointType.GENERAL) -> None:
        """
        Acquire a rate limit token synchronously.

        Waits if circuit breaker is open, then acquires from endpoint-specific limiter.

        Args:
            endpoint_type: The type of endpoint being called
        """
        self._circuit_breaker.wait_if_open_sync()
        self._limiters[endpoint_type].acquire_sync()
        logger.debug(f"Acquired rate limit token for {endpoint_type.value}")

    async def acquire_async(self, endpoint_type: EndpointType = EndpointType.GENERAL) -> None:
        """
        Acquire a rate limit token asynchronously.

        Waits if circuit breaker is open, then acquires from endpoint-specific limiter.

        Args:
            endpoint_type: The type of endpoint being called
        """
        await self._circuit_breaker.wait_if_open_async()
        await self._limiters[endpoint_type].acquire()
        logger.debug(f"Acquired rate limit token for {endpoint_type.value}")

    def on_response(self, status_code: int) -> None:
        """
        Record an API response status code.

        Updates the circuit breaker based on whether the request was rate limited.

        Args:
            status_code: HTTP status code from the API response
        """
        if status_code == 429:
            self._circuit_breaker.record_429()
        elif 200 <= status_code < 300:
            self._circuit_breaker.record_success()


# Global rate limit manager instance (singleton)
_rate_limit_manager: Optional[EndpointRateLimitManager] = None


def get_rate_limit_manager() -> EndpointRateLimitManager:
    """
    Get the global rate limit manager instance.

    Creates the manager on first access (lazy initialization).
    """
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = EndpointRateLimitManager()
    return _rate_limit_manager


# Legacy global rate limiter instance for backward compatibility
# New code should use get_rate_limit_manager() instead
api_rate_limiter = RateLimiter(requests_per_second=5.0, burst_size=10)


def get_rate_limiter() -> RateLimiter:
    """
    Get the legacy global rate limiter instance.

    DEPRECATED: Use get_rate_limit_manager() for endpoint-specific rate limiting.
    Kept for backward compatibility with existing code.
    """
    return api_rate_limiter


def configure_rate_limiter(requests_per_second: float = 5.0, burst_size: int = 10) -> None:
    """
    Configure the legacy global rate limiter.

    DEPRECATED: Use environment variables to configure rate limits instead.

    Args:
        requests_per_second: Maximum sustained request rate
        burst_size: Maximum number of requests allowed in a burst
    """
    global api_rate_limiter
    api_rate_limiter = RateLimiter(requests_per_second, burst_size)
