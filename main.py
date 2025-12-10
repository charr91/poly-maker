import gc  # Garbage collection
import os  # Environment variables
import time  # Time functions
import asyncio  # Asynchronous I/O
import logging  # Logging
import threading  # Thread management
import random  # For jitter in retry logic
import zoneinfo  # Timezone support
from datetime import datetime  # Datetime handling
from typing import Optional  # Type hints

import requests.exceptions  # For type-based error detection

from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import (
    update_markets,
    update_positions,
    update_orders,
    cleanup_orphaned_positions,
)
from poly_data.websocket_handlers import connect_market_websocket, connect_user_websocket
import poly_data.global_state as global_state
from poly_data.data_processing import remove_from_performing
from dotenv import load_dotenv

load_dotenv()

# Retry configuration for transient API errors
UPDATE_MAX_RETRIES = int(os.getenv("UPDATE_MAX_RETRIES", "3"))
UPDATE_BASE_DELAY = float(os.getenv("UPDATE_BASE_DELAY", "5.0"))
UPDATE_MAX_DELAY = float(os.getenv("UPDATE_MAX_DELAY", "60.0"))


# Custom formatter for timezone-aware logs
class TimezoneFormatter(logging.Formatter):
    """Formatter that converts log timestamps to configured timezone."""

    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


# Configure logging - level and timezone controllable via env vars
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TIMEZONE = os.getenv("LOG_TIMEZONE", "America/Los_Angeles")

handler = logging.StreamHandler()
handler.setFormatter(
    TimezoneFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        tz=zoneinfo.ZoneInfo(LOG_TIMEZONE),
    )
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    handlers=[handler],
)
logger = logging.getLogger("poly_maker")


def update_once():
    """
    Initialize the application state by fetching market data, positions, and orders.
    """
    update_markets()  # Get market information from Google Sheets
    update_positions()  # Get current positions from Polymarket
    update_orders()  # Get current orders from Polymarket


def remove_from_pending():
    """
    Clean up stale trades that have been pending for too long (>15 seconds).
    This prevents the system from getting stuck on trades that may have failed.
    """
    try:
        current_time = time.time()

        # Iterate through all performing trades
        for col in list(global_state.performing.keys()):
            for trade_id in list(global_state.performing[col]):

                try:
                    # If trade has been pending for more than 15 seconds, remove it
                    if (
                        current_time
                        - global_state.performing_timestamps[col].get(trade_id, current_time)
                        > 15
                    ):
                        logger.warning(
                            "Removing stale trade %s from %s after 15s timeout", trade_id, col[:30]
                        )
                        remove_from_performing(col, trade_id)
                except Exception as e:
                    logger.error("Error removing stale trade: %s", e)
                    logger.debug("Full traceback:", exc_info=True)
    except Exception as e:
        logger.error("Error in remove_from_pending: %s", e)
        logger.debug("Full traceback:", exc_info=True)


def _is_transient_error(exception: Exception) -> bool:
    """
    Check if an exception is a transient error that should be retried.

    Uses both type-based checking (more reliable) and string-based fallback
    for comprehensive error detection.

    Transient errors include:
    - HTTP 502 Bad Gateway
    - HTTP 503 Service Unavailable
    - HTTP 504 Gateway Timeout
    - Connection errors (ConnectionError, ConnectionRefusedError, etc.)
    - Timeout errors (TimeoutError, ReadTimeout, ConnectTimeout)

    Args:
        exception: The exception to check

    Returns:
        True if the error is transient and should be retried
    """
    # Type-based checks (most reliable)
    transient_exception_types = (
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
    )

    if isinstance(exception, transient_exception_types):
        return True

    # Check HTTP status codes for requests.HTTPError
    if isinstance(exception, requests.exceptions.HTTPError):
        response = getattr(exception, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code in (502, 503, 504):
                return True

    # String-based fallback for errors that don't match known types
    # (e.g., wrapped exceptions, custom error messages)
    error_str = str(exception).lower()
    transient_indicators = [
        "503",
        "502",
        "504",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "temporarily unavailable",
    ]
    return any(indicator in error_str for indicator in transient_indicators)


def _execute_with_retry(func, func_name: str) -> bool:
    """
    Execute a function with retry logic for transient errors.

    Uses exponential backoff with jitter for retries.

    Args:
        func: Function to execute (no arguments)
        func_name: Name for logging purposes

    Returns:
        True if successful, False if failed after all retries
    """
    last_exception: Optional[Exception] = None

    for attempt in range(UPDATE_MAX_RETRIES + 1):
        try:
            func()
            return True
        except Exception as e:
            last_exception = e
            is_last_attempt = attempt >= UPDATE_MAX_RETRIES

            if _is_transient_error(e) and not is_last_attempt:
                # Calculate delay with exponential backoff and jitter
                delay = min(UPDATE_BASE_DELAY * (2**attempt), UPDATE_MAX_DELAY)
                jitter = delay * random.uniform(0, 0.25)
                total_delay = delay + jitter

                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    func_name,
                    attempt + 1,
                    UPDATE_MAX_RETRIES + 1,
                    e,
                    total_delay,
                )
                time.sleep(total_delay)
            else:
                # Non-transient error or max retries exceeded - fail immediately
                logger.error("%s failed: %s", func_name, e)
                logger.debug("Full traceback:", exc_info=True)
                return False

    # This should never be reached due to the return statements above,
    # but provides a safety net and satisfies static analysis
    logger.error(
        "%s failed after %d attempts: %s", func_name, UPDATE_MAX_RETRIES + 1, last_exception
    )
    return False


def update_periodically():
    """
    Background thread function that periodically updates market data, positions and orders.

    Features:
    - Positions and orders are updated every 5 seconds
    - Market data is updated every 30 seconds (every 6 cycles)
    - Stale pending trades are removed each cycle
    - Transient API errors (503, etc.) are retried with exponential backoff
    """
    i = 1
    while True:
        time.sleep(5)  # Update every 5 seconds

        # Clean up stale trades (low risk, no retry needed)
        try:
            remove_from_pending()
        except Exception as e:
            logger.error("Error in remove_from_pending: %s", e)
            logger.debug("Full traceback:", exc_info=True)

        # Update positions with retry for transient errors
        _execute_with_retry(lambda: update_positions(avgOnly=True), "update_positions")

        # Update orders with retry for transient errors
        _execute_with_retry(update_orders, "update_orders")

        # Update market data every 6th cycle (30 seconds)
        if i % 6 == 0:
            _execute_with_retry(update_markets, "update_markets")
            i = 1

        gc.collect()  # Force garbage collection to free memory
        i += 1


async def main():
    """
    Main application entry point. Initializes client, data, and manages websocket connections.
    """
    # Store event loop reference FIRST - before any code that might need it
    # This allows background threads to schedule async tasks on the main event loop
    global_state.event_loop = asyncio.get_running_loop()

    # Initialize client
    global_state.client = PolymarketClient()

    # Initialize state and fetch initial data
    global_state.all_tokens = []
    update_once()

    # Cleanup orphaned positions from previous sessions
    # (positions whose markets were removed from sheet while bot was stopped)
    await cleanup_orphaned_positions()

    logger.debug("Initial orders: %s", global_state.orders)
    logger.debug("Initial positions: %s", global_state.positions)

    logger.info(
        "Started with %d markets, %d positions, %d orders",
        len(global_state.df),
        len(global_state.positions),
        len(global_state.orders),
    )

    # Start background update thread
    update_thread = threading.Thread(target=update_periodically, daemon=True)
    update_thread.start()

    # Main loop - maintain websocket connections
    while True:
        try:
            # Connect to market and user websockets simultaneously
            await asyncio.gather(
                connect_market_websocket(global_state.all_tokens), connect_user_websocket()
            )
            logger.info("Reconnecting to websocket...")
        except Exception as e:
            logger.error("Error in main loop: %s", e)
            logger.debug("Full traceback:", exc_info=True)

        await asyncio.sleep(1)
        gc.collect()  # Clean up memory


if __name__ == "__main__":
    asyncio.run(main())
