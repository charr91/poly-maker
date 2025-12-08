"""
Circuit breaker for position merge operations.

Prevents continuous merge failure spam by implementing per-market cooldowns
with exponential backoff. When a merge fails, that specific market enters
a cooldown period before retries are allowed.

Configuration via environment variables:
- MERGE_COOLDOWN_INITIAL: Initial cooldown in seconds (default: 60)
- MERGE_COOLDOWN_MAX: Maximum cooldown in seconds (default: 3600)
- MERGE_COOLDOWN_MULTIPLIER: Backoff multiplier (default: 2.0)
"""

import os
import time
import shutil
import logging
import threading
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("poly_maker.merge_circuit_breaker")


def _get_env_float(name: str, default: float) -> float:
    """Get a float from environment variable with default."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class MarketMergeState:
    """Tracks merge failure state for a single market."""

    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error: str = ""


class MergeCircuitBreaker:
    """
    Per-market circuit breaker for merge operations.

    Implements exponential backoff cooldowns when merge operations fail,
    preventing continuous retry spam while allowing eventual recovery.
    """

    def __init__(
        self,
        initial_cooldown: float = None,
        max_cooldown: float = None,
        multiplier: float = None,
    ):
        """
        Initialize the merge circuit breaker.

        Args:
            initial_cooldown: Initial cooldown duration in seconds
            max_cooldown: Maximum cooldown duration in seconds
            multiplier: Backoff multiplier for consecutive failures
        """
        self.initial_cooldown = initial_cooldown or _get_env_float("MERGE_COOLDOWN_INITIAL", 60.0)
        self.max_cooldown = max_cooldown or _get_env_float("MERGE_COOLDOWN_MAX", 3600.0)
        self.multiplier = multiplier or _get_env_float("MERGE_COOLDOWN_MULTIPLIER", 2.0)

        self._states: Dict[str, MarketMergeState] = {}
        self._lock = threading.Lock()
        self._node_available: Optional[bool] = None

    def check_node_available(self) -> bool:
        """
        Check if Node.js is available in the environment.

        Caches the result after first check.

        Returns:
            bool: True if Node.js is available, False otherwise
        """
        if self._node_available is not None:
            return self._node_available

        self._node_available = shutil.which("node") is not None

        if not self._node_available:
            logger.error(
                "Node.js not found in PATH. Position merging is disabled. "
                "Install Node.js or ensure poly_merger dependencies are installed."
            )
        else:
            logger.info("Node.js found, position merging is enabled")

        return self._node_available

    def can_merge(self, condition_id: str) -> bool:
        """
        Check if a merge operation is allowed for a market.

        Args:
            condition_id: Market condition ID

        Returns:
            bool: True if merge is allowed, False if in cooldown
        """
        if not self.check_node_available():
            return False

        with self._lock:
            state = self._states.get(condition_id)
            if state is None:
                return True

            now = time.monotonic()
            if now >= state.cooldown_until:
                return True

            remaining = state.cooldown_until - now
            logger.debug(
                f"Merge blocked for {condition_id[:16]}... "
                f"({remaining:.0f}s remaining, failures={state.consecutive_failures})"
            )
            return False

    def record_failure(self, condition_id: str, error: str) -> None:
        """
        Record a merge failure and enter/extend cooldown.

        Args:
            condition_id: Market condition ID
            error: Error message from the failure
        """
        with self._lock:
            now = time.monotonic()
            state = self._states.get(condition_id)

            if state is None:
                state = MarketMergeState()
                self._states[condition_id] = state

            state.consecutive_failures += 1
            state.last_failure_time = now
            state.last_error = error[:200]

            cooldown = min(
                self.initial_cooldown * (self.multiplier ** (state.consecutive_failures - 1)),
                self.max_cooldown,
            )
            state.cooldown_until = now + cooldown

            logger.warning(
                f"Merge failed for {condition_id[:16]}... "
                f"(attempt {state.consecutive_failures}, cooldown {cooldown:.0f}s): "
                f"{error[:100]}"
            )

    def record_success(self, condition_id: str) -> None:
        """
        Record a successful merge and reset cooldown state.

        Args:
            condition_id: Market condition ID
        """
        with self._lock:
            if condition_id in self._states:
                logger.info(
                    f"Merge succeeded for {condition_id[:16]}... "
                    f"(resetting after {self._states[condition_id].consecutive_failures} failures)"
                )
                del self._states[condition_id]

    def get_status(self, condition_id: str) -> Optional[MarketMergeState]:
        """
        Get current merge state for a market.

        Args:
            condition_id: Market condition ID

        Returns:
            MarketMergeState if in cooldown, None otherwise
        """
        with self._lock:
            return self._states.get(condition_id)

    def clear_market(self, condition_id: str) -> None:
        """
        Remove all state for a market (used during cleanup).

        Args:
            condition_id: Market condition ID to clear
        """
        with self._lock:
            if condition_id in self._states:
                del self._states[condition_id]
                logger.debug(f"Cleared merge circuit breaker state for {condition_id[:16]}...")


_merge_circuit_breaker: Optional[MergeCircuitBreaker] = None


def get_merge_circuit_breaker() -> MergeCircuitBreaker:
    """
    Get the global merge circuit breaker instance.

    Creates the instance on first access (lazy initialization).
    """
    global _merge_circuit_breaker
    if _merge_circuit_breaker is None:
        _merge_circuit_breaker = MergeCircuitBreaker()
    return _merge_circuit_breaker
