import asyncio
import logging
import os
import time
from typing import Set

import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS
from poly_data.utils import get_sheet_df

logger = logging.getLogger("poly_maker.data_utils")


def update_positions(avgOnly=False):
    """Update positions from the API. Rate limiting handled in polymarket_client."""
    pos_df = global_state.client.get_all_positions()

    for idx, row in pos_df.iterrows():
        asset = str(row["asset"])

        if asset in global_state.positions:
            position = global_state.positions[asset].copy()
        else:
            position = {"size": 0, "avgPrice": 0}

        position["avgPrice"] = row["avgPrice"]

        if not avgOnly:
            position["size"] = row["size"]
        else:

            for col in [f"{asset}_sell", f"{asset}_buy"]:
                # need to review this
                if (
                    col not in global_state.performing
                    or not isinstance(global_state.performing[col], set)
                    or len(global_state.performing[col]) == 0
                ):
                    try:
                        old_size = position["size"]
                    except:
                        old_size = 0

                    if asset in global_state.last_trade_update:
                        if time.time() - global_state.last_trade_update[asset] < 5:
                            logger.debug(
                                "Skipping position update for %s: recent trade update", asset[:16]
                            )
                            continue

                    if old_size != row["size"]:
                        logger.debug(
                            "Position update from API: %s -> %s (avgPrice: %s)",
                            old_size,
                            row["size"],
                            row["avgPrice"],
                        )

                    position["size"] = row["size"]
                else:
                    logger.debug(
                        "Skipping position update for %s: trades pending (%s)",
                        asset[:16],
                        global_state.performing[col],
                    )

        global_state.positions[asset] = position


def get_position(token):
    token = str(token)
    if token in global_state.positions:
        return global_state.positions[token]
    else:
        return {"size": 0, "avgPrice": 0}


def set_position(token, side, size, price, source="websocket"):
    token = str(token)
    size = float(size)
    price = float(price)

    global_state.last_trade_update[token] = time.time()

    if side.lower() == "sell":
        size *= -1

    if token in global_state.positions:

        prev_price = global_state.positions[token]["avgPrice"]
        prev_size = global_state.positions[token]["size"]

        if size > 0:
            if prev_size == 0:
                # Starting a new position
                avgPrice_new = price
            else:
                # Buying more; update average price
                avgPrice_new = (prev_price * prev_size + price * size) / (prev_size + size)
        elif size < 0:
            # Selling; average price remains the same
            avgPrice_new = prev_price
        else:
            # No change in position
            avgPrice_new = prev_price

        global_state.positions[token]["size"] += size
        global_state.positions[token]["avgPrice"] = avgPrice_new
    else:
        global_state.positions[token] = {"size": size, "avgPrice": price}

    logger.debug("Position updated from %s: %s", source, global_state.positions[token])


def update_orders():
    """Update orders from the API. Rate limiting handled in polymarket_client."""
    all_orders = global_state.client.get_all_orders()

    orders = {}

    if len(all_orders) > 0:
        for token in all_orders["asset_id"].unique():

            if token not in orders:
                orders[str(token)] = {
                    "buy": {"price": 0, "size": 0},
                    "sell": {"price": 0, "size": 0},
                }

            curr_orders = all_orders[all_orders["asset_id"] == str(token)]

            if len(curr_orders) > 0:
                sel_orders = {}
                sel_orders["buy"] = curr_orders[curr_orders["side"] == "BUY"]
                sel_orders["sell"] = curr_orders[curr_orders["side"] == "SELL"]

                for order_type in ["buy", "sell"]:
                    curr = sel_orders[order_type]

                    if len(curr) > 1:
                        logger.warning(
                            "Multiple %s orders found for %s, cancelling", order_type, token[:16]
                        )
                        global_state.client.cancel_all_asset(
                            token
                        )  # Rate limiting in polymarket_client
                        orders[str(token)] = {
                            "buy": {"price": 0, "size": 0},
                            "sell": {"price": 0, "size": 0},
                        }
                    elif len(curr) == 1:
                        orders[str(token)][order_type]["price"] = float(curr.iloc[0]["price"])
                        orders[str(token)][order_type]["size"] = float(
                            curr.iloc[0]["original_size"] - curr.iloc[0]["size_matched"]
                        )

    global_state.orders = orders


def get_order(token):
    token = str(token)
    if token in global_state.orders:

        if "buy" not in global_state.orders[token]:
            global_state.orders[token]["buy"] = {"price": 0, "size": 0}

        if "sell" not in global_state.orders[token]:
            global_state.orders[token]["sell"] = {"price": 0, "size": 0}

        return global_state.orders[token]
    else:
        return {"buy": {"price": 0, "size": 0}, "sell": {"price": 0, "size": 0}}


def set_order(token, side, size, price):
    curr = {}
    curr = {side: {"price": 0, "size": 0}}

    curr[side]["size"] = float(size)
    curr[side]["price"] = float(price)

    global_state.orders[str(token)] = curr
    logger.debug("Order updated: %s", curr)


# ============ Market Cleanup Functions ============


def detect_removed_markets(new_df) -> Set[str]:
    """
    Compare new market DataFrame with current state to detect removed markets.

    Args:
        new_df: New DataFrame from Google Sheets

    Returns:
        Set of condition_ids that have been removed
    """
    if global_state.df is None or len(global_state.df) == 0:
        return set()

    current_condition_ids = set(global_state.df["condition_id"].astype(str))
    new_condition_ids = set(new_df["condition_id"].astype(str))

    return current_condition_ids - new_condition_ids


def detect_orphaned_tokens() -> Set[str]:
    """
    Detect tokens with positions/orders that aren't in the current sheet.

    This function is called at startup to find positions from previous sessions
    that are no longer in the Selected Markets sheet.

    Returns:
        Set of orphaned token IDs
    """
    if global_state.df is None or len(global_state.df) == 0:
        return set()

    # Get all tokens currently in sheet
    known_tokens = set()
    for _, row in global_state.df.iterrows():
        known_tokens.add(str(row["token1"]))
        known_tokens.add(str(row["token2"]))

    # Find orphans in positions and orders
    orphaned = set()
    for token in global_state.positions.keys():
        if token not in known_tokens:
            orphaned.add(token)
    for token in global_state.orders.keys():
        if token not in known_tokens:
            orphaned.add(token)

    return orphaned


async def cleanup_orphaned_positions() -> None:
    """
    Cleanup positions/orders for markets not in the sheet.

    Called at startup to handle positions from previous sessions where the market
    was removed from the sheet while the bot was not running.
    """
    orphaned_tokens = detect_orphaned_tokens()
    if not orphaned_tokens:
        logger.debug("No orphaned tokens found at startup")
        return

    logger.info("Found %d orphaned tokens, fetching market info...", len(orphaned_tokens))

    # Group tokens by market to avoid duplicate API calls and cleanups
    processed_conditions = set()

    for token in orphaned_tokens:
        market_info = await global_state.client.get_market_by_token_async(token)
        if not market_info:
            logger.warning("Could not fetch market info for orphaned token %s", token[:20])
            continue

        # Gamma API returns camelCase field names
        condition_id = market_info.get("conditionId")
        if not condition_id:
            logger.warning(
                "Market info missing conditionId for token %s, keys: %s",
                token[:20],
                list(market_info.keys()),
            )
            continue
        if condition_id in processed_conditions:
            continue
        processed_conditions.add(condition_id)

        # clobTokenIds is a list of token ID strings from Gamma API
        tokens = [str(t) for t in market_info.get("clobTokenIds", [])]

        neg_risk = market_info.get("negRisk", False)
        question = market_info.get("question", "Unknown")

        removal_info = {
            "tokens": tokens,
            "question": question,
            "neg_risk": neg_risk,
        }

        logger.info(
            "Cleaning up orphaned market: %s - %s",
            condition_id[:16],
            question[:50] if question else "Unknown",
        )

        try:
            await cleanup_market(condition_id, removal_info)
        except Exception as e:
            logger.error("Error cleaning up orphaned market %s: %s", condition_id[:16], e)


def add_to_pending_removal(condition_id: str, tokens: list, question: str, neg_risk: bool) -> None:
    """
    Add a market to the pending removal queue with grace period.

    Args:
        condition_id: Market condition ID
        tokens: List of token IDs [token1, token2]
        question: Market question for logging
        neg_risk: Whether market uses negative risk
    """
    if condition_id not in global_state.pending_removal:
        global_state.pending_removal[condition_id] = {
            "timestamp": time.time(),
            "tokens": tokens,
            "question": question,
            "neg_risk": neg_risk,
        }
        logger.info(
            "Market queued for removal (grace period): %s - %s",
            condition_id[:16],
            question[:50] if question else "Unknown",
        )


async def process_pending_removals() -> None:
    """
    Process markets that have passed their grace period.

    Checks each pending market and triggers cleanup if grace period has elapsed.
    """
    current_time = time.time()
    markets_to_cleanup = []

    for condition_id, removal_info in list(global_state.pending_removal.items()):
        elapsed = current_time - removal_info["timestamp"]

        if elapsed >= CONSTANTS.CLEANUP_GRACE_PERIOD:
            markets_to_cleanup.append((condition_id, removal_info))

    for condition_id, removal_info in markets_to_cleanup:
        logger.info(
            "Cleanup triggered for market: %s - %s",
            condition_id[:16],
            removal_info["question"][:50] if removal_info.get("question") else "Unknown",
        )

        try:
            await cleanup_market(condition_id, removal_info)
            global_state.pending_removal.pop(condition_id, None)
        except Exception as e:
            logger.error("Error cleaning up market %s: %s", condition_id[:16], e)


async def cleanup_market(condition_id: str, removal_info: dict) -> None:
    """
    Perform cleanup operations for a removed market.

    Args:
        condition_id: Market condition ID
        removal_info: Dict with tokens, question, neg_risk
    """
    tokens = removal_info["tokens"]
    neg_risk = removal_info.get("neg_risk", False)
    client = global_state.client

    # Mark market as being removed to skip WebSocket updates
    global_state.removing_markets.add(condition_id)

    try:
        # 1. Cancel pending trade tasks for this market
        await _cancel_pending_trade_tasks(condition_id)

        # 2. Wait for performing trades to complete (with timeout)
        await _wait_for_performing_trades(tokens, timeout=5.0)

        # 3. Cancel orders (if enabled)
        if CONSTANTS.CLEANUP_CANCEL_ORDERS:
            try:
                await client.cancel_all_market_async(condition_id)
                logger.info("Cancelled all orders for market %s", condition_id[:16])
            except Exception as e:
                logger.error("Failed to cancel orders for %s: %s", condition_id[:16], e)

        # 4. Close positions (if enabled)
        if CONSTANTS.CLEANUP_SELL_POSITIONS:
            await close_positions(tokens, condition_id, neg_risk)

        # 5. Clean up state
        cleanup_market_state(condition_id, tokens)

    finally:
        # Remove from removing_markets set
        global_state.removing_markets.discard(condition_id)


async def _cancel_pending_trade_tasks(condition_id: str) -> None:
    """Cancel any pending debounced trade tasks for a market."""
    # Import here to avoid circular import
    from poly_data.data_processing import _pending_trades, _pending_trade_tasks

    task = _pending_trade_tasks.get(condition_id)
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        logger.info("Cancelled pending trade task for market %s", condition_id[:16])

    _pending_trades.pop(condition_id, None)
    _pending_trade_tasks.pop(condition_id, None)


async def _wait_for_performing_trades(tokens: list, timeout: float = 5.0) -> bool:
    """
    Wait for performing trades to complete with timeout.

    Args:
        tokens: List of token IDs
        timeout: Maximum time to wait in seconds

    Returns:
        True if all trades completed, False if timed out
    """
    keys_to_check = []
    for token in tokens:
        token_str = str(token)
        keys_to_check.extend([f"{token_str}_buy", f"{token_str}_sell"])

    start = time.time()
    while time.time() - start < timeout:
        active = any(global_state.performing.get(k) for k in keys_to_check)
        if not active:
            return True
        await asyncio.sleep(0.1)

    # Force cleanup after timeout
    for key in keys_to_check:
        global_state.performing.pop(key, None)
        global_state.performing_timestamps.pop(key, None)
    logger.warning("Forced cleanup of performing trades for tokens after timeout")
    return False


async def close_positions(tokens: list, condition_id: str, neg_risk: bool) -> None:
    """
    Close positions for removed market based on P&L.

    - In-profit positions: Sell at market (best bid) for immediate close
    - Underwater positions: Place limit sell at break-even (avgPrice)

    Args:
        tokens: List of token IDs [token1, token2]
        condition_id: Market condition ID
        neg_risk: Whether market uses negative risk
    """
    client = global_state.client

    # Get position sizes for both tokens
    positions = []
    for token in tokens:
        token_str = str(token)
        pos = global_state.positions.get(token_str, {})
        size = pos.get("size", 0)
        avg_price = pos.get("avgPrice", 0)
        if size > 0:
            positions.append((token_str, size, avg_price))

    if not positions:
        logger.debug("No positions to close for market %s", condition_id[:16])
        return

    # Merge opposing positions first if both exceed threshold
    if len(positions) == 2:
        size1, size2 = positions[0][1], positions[1][1]
        amount_to_merge = min(size1, size2)
        if amount_to_merge > CONSTANTS.MIN_MERGE_SIZE:
            try:
                await client.merge_positions_async(
                    int(amount_to_merge * 1e6), condition_id, neg_risk
                )
                logger.info(
                    "Merged %s positions before closing for market %s",
                    amount_to_merge,
                    condition_id[:16],
                )
                # Update position sizes after merge
                positions = [(t, max(0, s - amount_to_merge), p) for t, s, p in positions]
            except Exception as e:
                logger.warning("Merge failed during cleanup for %s: %s", condition_id[:16], e)

    # Get order book data once (contains YES token bids/asks)
    order_book = global_state.all_data.get(condition_id, {})
    yes_bids = order_book.get("bids", {})
    yes_asks = order_book.get("asks", {})

    # Close remaining positions
    for token_str, size, avg_price in positions:
        if size <= 0:
            continue

        # Determine if this is token1 (YES) or token2 (NO)
        # tokens list is always [token1, token2] where token1=YES, token2=NO
        is_token2 = len(tokens) > 1 and token_str == str(tokens[1])

        # Get best bid based on token type
        # YES: use YES bids directly
        # NO: best_bid = 1 - YES_best_ask (price transformation for complementary outcome)
        if is_token2:
            # For NO token: best_bid = 1 - YES_best_ask
            best_bid = (1 - min(yes_asks.keys())) if yes_asks else None
        else:
            # For YES token: use YES best_bid directly
            best_bid = max(yes_bids.keys()) if yes_bids else None

        # Determine sell strategy based on P&L
        if CONSTANTS.CLEANUP_FORCE_MARKET_SELL:
            # Force sell at best bid regardless of P&L
            if best_bid:
                sell_price = best_bid
                order_type = "limit (forced)"
            else:
                sell_price = avg_price
                order_type = "limit (no bids)"
        elif best_bid is not None and best_bid >= avg_price:
            # In profit - sell at best bid
            sell_price = best_bid
            order_type = "limit (in profit)"
        else:
            # Underwater or no bids - place limit at break-even
            sell_price = avg_price
            order_type = "limit (break-even)"

        try:
            logger.info(
                "Closing position: %s of token %s at %.4f (%s)",
                size,
                token_str[:20],
                sell_price,
                order_type,
            )
            await client.create_order_async(token_str, "SELL", sell_price, size, neg_risk)
        except Exception as e:
            logger.error("Failed to place cleanup sell for token %s: %s", token_str[:20], e)


def cleanup_market_state(condition_id: str, tokens: list) -> None:
    """
    Clean up all in-memory state for a removed market.

    Args:
        condition_id: Market condition ID
        tokens: List of token IDs
    """
    for token in tokens:
        token_str = str(token)

        # Remove from all_tokens (WebSocket subscriptions)
        if token_str in global_state.all_tokens:
            global_state.all_tokens.remove(token_str)

        # Remove from REVERSE_TOKENS
        global_state.REVERSE_TOKENS.pop(token_str, None)

        # Remove from orders
        global_state.orders.pop(token_str, None)

        # Note: Don't remove positions - they still exist on blockchain
        # Just clear from active tracking if needed

        # Remove from performing sets
        for suffix in ["_buy", "_sell"]:
            col = token_str + suffix
            global_state.performing.pop(col, None)
            global_state.performing_timestamps.pop(col, None)

        # Remove from last_trade_update
        global_state.last_trade_update.pop(token_str, None)

    # Remove order book data
    global_state.all_data.pop(condition_id, None)

    # Remove risk-off file if exists
    risk_file = f"positions/{condition_id}.json"
    if os.path.isfile(risk_file):
        try:
            os.remove(risk_file)
            logger.info("Removed risk-off file for market %s", condition_id[:16])
        except Exception as e:
            logger.warning("Failed to remove risk-off file: %s", e)

    # Clear circuit breaker state
    try:
        from poly_data.merge_circuit_breaker import get_merge_circuit_breaker

        get_merge_circuit_breaker().clear_market(condition_id)
    except Exception as e:
        logger.debug("Could not clear circuit breaker state: %s", e)

    # Clear market lock from trading module
    try:
        from trading import market_locks

        market_locks.pop(condition_id, None)
    except Exception as e:
        logger.debug("Could not clear market lock: %s", e)

    logger.info("Cleaned up state for market %s", condition_id[:16])


def _schedule_pending_removals() -> None:
    """
    Schedule async processing of pending removals (thread-safe).

    This function is called from a background thread (update_periodically),
    so we use loop.call_soon_threadsafe() to schedule the async task on
    the main event loop.
    """
    loop = getattr(global_state, "event_loop", None)

    if loop is None:
        logger.warning("No event loop available, skipping pending removal processing")
        return

    if loop.is_closed():
        logger.warning("Event loop is closed, skipping pending removal processing")
        return

    try:
        # Thread-safe way to schedule async task from background thread
        loop.call_soon_threadsafe(lambda: asyncio.create_task(process_pending_removals()))
    except RuntimeError as e:
        logger.warning("Failed to schedule pending removals: %s", e)


def update_markets():
    """
    Update market data from Google Sheets with cleanup detection.

    This function:
    1. Fetches new market data from Google Sheets
    2. Detects removed markets and queues them for cleanup
    3. Processes pending removals that have passed grace period
    4. Updates global state with new market data
    """
    received_df, received_params = get_sheet_df()

    if len(received_df) == 0:
        logger.warning("Received empty DataFrame from sheet, skipping update")
        return

    # 1. Detect removed markets (before overwriting df)
    removed_markets = detect_removed_markets(received_df)

    # 2. Queue removed markets for cleanup (with grace period)
    for condition_id in removed_markets:
        if condition_id not in global_state.pending_removal:
            try:
                row = global_state.df[global_state.df["condition_id"] == condition_id].iloc[0]
                add_to_pending_removal(
                    condition_id,
                    [str(row["token1"]), str(row["token2"])],
                    row.get("question", "Unknown"),
                    row.get("neg_risk", "FALSE") == "TRUE",
                )
            except (IndexError, KeyError) as e:
                logger.warning("Could not queue market %s for removal: %s", condition_id[:16], e)

    # 3. Check if any markets returned (cancel their pending removal)
    new_condition_ids = set(received_df["condition_id"].astype(str))
    for condition_id in list(global_state.pending_removal.keys()):
        if condition_id in new_condition_ids:
            logger.info("Market %s returned to sheet, cancelling removal", condition_id[:16])
            global_state.pending_removal.pop(condition_id, None)

    # 4. Process expired pending removals (async)
    _schedule_pending_removals()

    # 5. Update global state with new market data
    global_state.df, global_state.params = received_df.copy(), received_params

    # 6. Register new tokens (existing logic)
    for _, row in global_state.df.iterrows():
        for col in ["token1", "token2"]:
            row[col] = str(row[col])

        if row["token1"] not in global_state.all_tokens:
            global_state.all_tokens.append(row["token1"])

        if row["token1"] not in global_state.REVERSE_TOKENS:
            global_state.REVERSE_TOKENS[row["token1"]] = row["token2"]

        if row["token2"] not in global_state.REVERSE_TOKENS:
            global_state.REVERSE_TOKENS[row["token2"]] = row["token1"]

        for col2 in [
            f"{row['token1']}_buy",
            f"{row['token1']}_sell",
            f"{row['token2']}_buy",
            f"{row['token2']}_sell",
        ]:
            if col2 not in global_state.performing:
                global_state.performing[col2] = set()
