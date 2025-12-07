import asyncio  # Asynchronous I/O
import json  # JSON handling
import logging  # Logging
import websockets  # WebSocket client

from poly_data.data_processing import process_data, process_user_data
import poly_data.global_state as global_state

logger = logging.getLogger("poly_maker.websocket")


async def connect_market_websocket(chunk):
    """
    Connect to Polymarket's market WebSocket API and process market updates.

    This function:
    1. Establishes a WebSocket connection to the Polymarket API
    2. Subscribes to updates for a specified list of market tokens
    3. Processes incoming order book and price updates

    Args:
        chunk (list): List of token IDs to subscribe to

    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
        # Prepare and send subscription message
        message = {"assets_ids": chunk}
        await websocket.send(json.dumps(message))

        logger.info("Market websocket connected, subscribed to %d assets", len(chunk))
        logger.debug("Subscription assets: %s", chunk)

        try:
            # Process incoming market data indefinitely
            while True:
                message = await websocket.recv()
                json_data = json.loads(message)
                # Process order book updates and trigger trading as needed
                process_data(json_data)
        except websockets.ConnectionClosed:
            logger.warning("Market websocket connection closed")
        except Exception as e:
            logger.error("Market websocket error: %s", e)
            logger.debug("Full traceback:", exc_info=True)
        finally:
            # Brief delay before attempting to reconnect
            await asyncio.sleep(5)


async def connect_user_websocket():
    """
    Connect to Polymarket's user WebSocket API and process order/trade updates.

    This function:
    1. Establishes a WebSocket connection to the Polymarket user API
    2. Authenticates using API credentials
    3. Processes incoming order and trade updates for the user

    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
        # Prepare authentication message with API credentials (never log secrets)
        message = {
            "type": "user",
            "auth": {
                "apiKey": global_state.client.client.creds.api_key,
                "secret": global_state.client.client.creds.api_secret,
                "passphrase": global_state.client.client.creds.api_passphrase,
            },
        }

        # Send authentication message
        await websocket.send(json.dumps(message))

        logger.info("User websocket connected and authenticated")

        try:
            # Process incoming user data indefinitely
            while True:
                message = await websocket.recv()
                json_data = json.loads(message)
                # Process trade and order updates
                process_user_data(json_data)
        except websockets.ConnectionClosed:
            logger.warning("User websocket connection closed")
        except Exception as e:
            logger.error("User websocket error: %s", e)
            logger.debug("Full traceback:", exc_info=True)
        finally:
            # Brief delay before attempting to reconnect
            await asyncio.sleep(5)
