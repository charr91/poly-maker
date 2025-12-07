"""
Rate limiting utilities for Polymarket API calls.

Provides token bucket rate limiting and exponential backoff retry logic
to prevent API bans when operating on many markets.
"""

import time
import asyncio
import functools
import threading
import random
from typing import Callable, TypeVar, Any

import requests


# Type variable for generic decorator
T = TypeVar('T')


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


# Global rate limiter instance for the application
# Can be configured at startup
api_rate_limiter = RateLimiter(requests_per_second=5.0, burst_size=10)


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    return api_rate_limiter


def configure_rate_limiter(requests_per_second: float = 5.0, burst_size: int = 10) -> None:
    """
    Configure the global rate limiter.

    Args:
        requests_per_second: Maximum sustained request rate
        burst_size: Maximum number of requests allowed in a burst
    """
    global api_rate_limiter
    api_rate_limiter = RateLimiter(requests_per_second, burst_size)
