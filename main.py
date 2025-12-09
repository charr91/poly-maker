import gc  # Garbage collection
import os  # Environment variables
import time  # Time functions
import asyncio  # Asynchronous I/O
import logging  # Logging
import threading  # Thread management

from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import update_markets, update_positions, update_orders
from poly_data.websocket_handlers import connect_market_websocket, connect_user_websocket
import poly_data.global_state as global_state
from poly_data.data_processing import remove_from_performing
from dotenv import load_dotenv

load_dotenv()

# Configure logging - level controllable via LOG_LEVEL env var (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
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


def update_periodically():
    """
    Background thread function that periodically updates market data, positions and orders.
    - Positions and orders are updated every 5 seconds
    - Market data is updated every 30 seconds (every 6 cycles)
    - Stale pending trades are removed each cycle
    """
    i = 1
    while True:
        time.sleep(5)  # Update every 5 seconds

        try:
            # Clean up stale trades
            remove_from_pending()

            # Update positions and orders every cycle
            update_positions(avgOnly=True)  # Only update average price, not position size
            update_orders()

            # Update market data every 6th cycle (30 seconds)
            if i % 6 == 0:
                update_markets()
                i = 1

            gc.collect()  # Force garbage collection to free memory
            i += 1
        except Exception as e:
            logger.error("Error in update_periodically: %s", e)
            logger.debug("Full traceback:", exc_info=True)


async def main():
    """
    Main application entry point. Initializes client, data, and manages websocket connections.
    """
    # Initialize client
    global_state.client = PolymarketClient()

    # Initialize state and fetch initial data
    global_state.all_tokens = []
    update_once()

    logger.debug("Initial orders: %s", global_state.orders)
    logger.debug("Initial positions: %s", global_state.positions)

    logger.info(
        "Started with %d markets, %d positions, %d orders",
        len(global_state.df),
        len(global_state.positions),
        len(global_state.orders),
    )

    # Store event loop reference for thread-safe async scheduling
    # This allows the background thread to schedule async tasks on the main event loop
    global_state.event_loop = asyncio.get_running_loop()

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
