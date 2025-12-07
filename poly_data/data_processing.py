import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Union, Any
from sortedcontainers import SortedDict

import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS
from trading import perform_trade
from poly_data.data_utils import set_position, set_order, update_positions


# Configure logging
logger = logging.getLogger("poly_maker.data_processing")


def _validate_market_message(json_data: dict) -> bool:
    """Validate that a market message has required fields."""
    required = ["event_type", "market"]
    return isinstance(json_data, dict) and all(key in json_data for key in required)


def _validate_user_message(row: dict) -> bool:
    """Validate that a user message has required fields."""
    required = ["market", "side", "asset_id", "event_type"]
    return isinstance(row, dict) and all(key in row for key in required)


# Debouncing configuration
# Configurable via TRADE_DEBOUNCE_MS env var (default: 300ms)
TRADE_DEBOUNCE_MS = int(os.getenv("TRADE_DEBOUNCE_MS", "300"))
TRADE_DEBOUNCE_SEC = TRADE_DEBOUNCE_MS / 1000.0

# Track pending trades per market for debouncing
_pending_trades: Dict[str, float] = {}
_pending_trade_tasks: Dict[str, asyncio.Task] = {}


async def _debounced_perform_trade(market: str) -> None:
    """
    Execute perform_trade with debouncing to prevent flooding.

    Waits for TRADE_DEBOUNCE_SEC before executing. If this task is cancelled
    (because a newer trade was scheduled), it exits gracefully.
    """
    try:
        # Wait for debounce period
        await asyncio.sleep(TRADE_DEBOUNCE_SEC)

        # Clear pending state and execute trade
        _pending_trades.pop(market, None)
        _pending_trade_tasks.pop(market, None)

        logger.debug(f"Executing debounced trade for {market}")
        await perform_trade(market)

    except asyncio.CancelledError:
        # Task was cancelled because a newer trade was scheduled
        logger.debug(f"Debounced trade for {market} cancelled (superseded by newer request)")
        raise  # Re-raise to properly handle cancellation


def schedule_trade(market: str) -> None:
    """
    Schedule a debounced trade for a market.

    If a trade is already pending for this market, cancels the pending task
    and schedules a new one. This ensures:
    1. Rapid updates extend the quiet period (debounce behavior)
    2. A trade is always executed after the quiet period ends
    """
    now = time.time()

    # Cancel any existing pending task for this market
    existing_task = _pending_trade_tasks.get(market)
    if existing_task and not existing_task.done():
        existing_task.cancel()
        logger.debug(f"Cancelled pending trade for {market}, scheduling new one")

    # Schedule new debounced trade
    _pending_trades[market] = now
    task = asyncio.create_task(_debounced_perform_trade(market))
    _pending_trade_tasks[market] = task
    logger.debug(f"Scheduled debounced trade for {market}")


def process_book_data(asset, json_data):
    global_state.all_data[asset] = {
        "asset_id": json_data["asset_id"],  # token_id for the Yes token
        "bids": SortedDict(),
        "asks": SortedDict(),
    }

    global_state.all_data[asset]["bids"].update(
        {float(entry["price"]): float(entry["size"]) for entry in json_data["bids"]}
    )
    global_state.all_data[asset]["asks"].update(
        {float(entry["price"]): float(entry["size"]) for entry in json_data["asks"]}
    )


def process_price_change(
    asset: str, asset_id: str, side: str, price_level: float, new_size: float
) -> None:
    """
    Process a price change event for a market.

    Args:
        asset: Market identifier
        asset_id: Token ID from the price change event (used to filter duplicates)
        side: 'bids' or 'asks'
        price_level: Price level that changed
        new_size: New size at this price level (0 means remove)
    """
    if asset not in global_state.all_data:
        logger.debug("Price change for unknown asset: %s", asset)
        return
    if asset_id != global_state.all_data[asset]["asset_id"]:
        return  # skip updates for the No token to prevent duplicated updates
    if side == "bids":
        book = global_state.all_data[asset]["bids"]
    else:
        book = global_state.all_data[asset]["asks"]

    if new_size == 0:
        if price_level in book:
            del book[price_level]
    else:
        book[price_level] = new_size


def process_data(json_datas: Union[dict, List[dict]], trade: bool = True) -> None:
    """
    Process incoming WebSocket market data.

    Uses debouncing to prevent flooding perform_trade calls when receiving
    rapid price updates. Each market gets at most one trade execution per
    TRADE_DEBOUNCE_MS period.

    Args:
        json_datas: Single dict or list of dicts from WebSocket message
        trade: Whether to trigger trade execution (default True)
    """
    # Normalize to list for consistent processing
    if isinstance(json_datas, dict):
        json_datas = [json_datas]

    for json_data in json_datas:
        # Validate message before processing
        if not _validate_market_message(json_data):
            logger.warning(
                "Invalid market message (missing required fields): %s", str(json_data)[:200]
            )
            continue

        event_type = json_data["event_type"]
        asset = json_data["market"]

        if event_type == "book":
            process_book_data(asset, json_data)

            if trade:
                # Use debounced scheduling to prevent flooding
                schedule_trade(asset)

        elif event_type == "price_change":
            asset_id = json_data.get("asset_id")
            for data in json_data["price_changes"]:
                side = "bids" if data["side"] == "BUY" else "asks"
                price_level = float(data["price"])
                new_size = float(data["size"])
                process_price_change(asset, asset_id, side, price_level, new_size)

            # Only schedule one trade per price_change event (not per price level)
            if trade:
                schedule_trade(asset)


def add_to_performing(col, id):
    if col not in global_state.performing:
        global_state.performing[col] = set()

    if col not in global_state.performing_timestamps:
        global_state.performing_timestamps[col] = {}

    # Add the trade ID and track its timestamp
    global_state.performing[col].add(id)
    global_state.performing_timestamps[col][id] = time.time()


def remove_from_performing(col, id):
    if col in global_state.performing:
        global_state.performing[col].discard(id)

    if col in global_state.performing_timestamps:
        global_state.performing_timestamps[col].pop(id, None)


def process_user_data(rows: Union[dict, List[dict]]) -> None:
    """
    Process incoming WebSocket user data (trades and orders).

    Args:
        rows: Single dict or list of dicts from WebSocket message
    """
    # Normalize to list for consistent processing
    if isinstance(rows, dict):
        rows = [rows]

    for row in rows:
        # Validate message before processing
        if not _validate_user_message(row):
            logger.warning("Invalid user message (missing required fields): %s", str(row)[:200])
            continue

        market = row["market"]
        side = row["side"].lower()
        token = row["asset_id"]

        # Skip tokens we're not tracking
        if token not in global_state.REVERSE_TOKENS:
            logger.debug("User data for market %s: token %s not tracked", market, token[:20])
            continue

        col = token + "_" + side

        if row["event_type"] == "trade":
            size = 0
            price = 0
            maker_outcome = ""
            taker_outcome = row["outcome"]

            is_user_maker = False
            for maker_order in row["maker_orders"]:
                if (
                    maker_order["maker_address"].lower()
                    == global_state.client.browser_wallet.lower()
                ):
                    logger.debug("User is maker for trade %s", row["id"])
                    size = float(maker_order["matched_amount"])
                    price = float(maker_order["price"])

                    is_user_maker = True
                    maker_outcome = maker_order["outcome"]

                    if maker_outcome == taker_outcome:
                        side = "buy" if side == "sell" else "sell"
                    else:
                        token = global_state.REVERSE_TOKENS[token]

            if not is_user_maker:
                size = float(row["size"])
                price = float(row["price"])
                logger.debug("User is taker for trade %s", row["id"])

            logger.info(
                "Trade: market=%s id=%s status=%s side=%s size=%.2f",
                market[:20],
                row["id"],
                row["status"],
                side,
                size,
            )

            if row["status"] == "CONFIRMED" or row["status"] == "FAILED":
                if row["status"] == "FAILED":
                    logger.warning("Trade failed for token %s, updating positions", token[:20])
                    asyncio.create_task(asyncio.sleep(2))
                    update_positions()
                else:
                    remove_from_performing(col, row["id"])
                    logger.debug(
                        "Trade confirmed: performing_count=%d",
                        len(global_state.performing.get(col, set())),
                    )
                    asyncio.create_task(perform_trade(market))

            elif row["status"] == "MATCHED":
                add_to_performing(col, row["id"])
                logger.debug(
                    "Trade matched: performing_count=%d",
                    len(global_state.performing.get(col, set())),
                )
                set_position(token, side, size, price)
                logger.debug(
                    "Position after matching: %s", global_state.positions.get(str(token), {})
                )
                asyncio.create_task(perform_trade(market))
            elif row["status"] == "MINED":
                remove_from_performing(col, row["id"])

        elif row["event_type"] == "order":
            logger.info(
                "Order: market=%s status=%s type=%s side=%s size=%s",
                market[:20],
                row["status"],
                row.get("type", "N/A"),
                side,
                row["original_size"],
            )

            set_order(
                token, side, float(row["original_size"]) - float(row["size_matched"]), row["price"]
            )
            asyncio.create_task(perform_trade(market))
