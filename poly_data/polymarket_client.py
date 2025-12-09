from dotenv import load_dotenv  # Environment variable management
import os  # Operating system interface
import asyncio  # Async support
import atexit  # Cleanup handlers
from concurrent.futures import ThreadPoolExecutor
import functools  # For partial function binding
from typing import Optional

# Polymarket API client libraries
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    BalanceAllowanceParams,
    AssetType,
    PartialCreateOrderOptions,
)
from py_clob_client.constants import POLYGON

# Web3 libraries for blockchain interaction
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account

import requests  # HTTP requests
import pandas as pd  # Data analysis
import json  # JSON processing
import subprocess  # For calling external processes

from py_clob_client.clob_types import OpenOrderParams

# Rate limiting utilities
from poly_data.rate_limiter import (
    with_retry,
    get_rate_limiter,
    get_rate_limit_manager,
    EndpointType,
)

# Smart contract ABIs
from poly_data.abis import NegRiskAdapterABI, ConditionalTokenABI, erc20_abi

# Load environment variables
load_dotenv()

# Thread pool for blocking API calls - prevents event loop starvation
# Configurable via API_THREAD_POOL_SIZE env var (default: 10)
_api_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("API_THREAD_POOL_SIZE", "10")), thread_name_prefix="polymarket_api"
)


@atexit.register
def _shutdown_api_executor():
    """Ensure clean shutdown of the thread pool."""
    _api_executor.shutdown(wait=True)


class PolymarketClient:
    """
    Client for interacting with Polymarket's API and smart contracts.

    This class provides methods for:
    - Creating and managing orders
    - Querying order book data
    - Checking balances and positions
    - Merging positions

    The client connects to both the Polymarket API and the Polygon blockchain.
    """

    def __init__(self, pk="default") -> None:
        """
        Initialize the Polymarket client with API and blockchain connections.

        Args:
            pk (str, optional): Private key identifier, defaults to 'default'
        """
        host = "https://clob.polymarket.com"

        # Get credentials from environment variables
        key = os.getenv("PK")
        browser_address = os.getenv("BROWSER_ADDRESS")

        # Don't print sensitive wallet information
        print("Initializing Polymarket client...")
        chain_id = POLYGON
        self.browser_wallet = Web3.to_checksum_address(browser_address)

        # Initialize the Polymarket API client
        self.client = ClobClient(
            host=host, key=key, chain_id=chain_id, funder=self.browser_wallet, signature_type=2
        )

        # Set up API credentials
        self.creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(creds=self.creds)

        # Initialize Web3 connection to Polygon
        web3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
        web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        # Set up USDC contract for balance checks
        self.usdc_contract = web3.eth.contract(
            address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", abi=erc20_abi
        )

        # Store key contract addresses
        self.addresses = {
            "neg_risk_adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
            "collateral": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
            "conditional_tokens": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
        }

        # Initialize contract interfaces
        self.neg_risk_adapter = web3.eth.contract(
            address=self.addresses["neg_risk_adapter"], abi=NegRiskAdapterABI
        )

        self.conditional_tokens = web3.eth.contract(
            address=self.addresses["conditional_tokens"], abi=ConditionalTokenABI
        )

        self.web3 = web3

    def create_order(self, marketId, action, price, size, neg_risk=False):
        """
        Create and submit a new order to the Polymarket order book.

        Args:
            marketId (str): ID of the market token to trade
            action (str): "BUY" or "SELL"
            price (float): Order price (0-1 range for prediction markets)
            size (float): Order size in USDC
            neg_risk (bool, optional): Whether this is a negative risk market. Defaults to False.

        Returns:
            dict: Response from the API containing order details, or empty dict on error
        """
        # Rate limit before making API call
        get_rate_limit_manager().acquire_sync(EndpointType.ORDER)

        # Create order parameters
        order_args = OrderArgs(token_id=str(marketId), price=price, size=size, side=action)

        signed_order = None

        # Handle regular vs negative risk markets differently
        if neg_risk == False:
            signed_order = self.client.create_order(order_args)
        else:
            signed_order = self.client.create_order(
                order_args, options=PartialCreateOrderOptions(neg_risk=True)
            )

        try:
            # Submit the signed order to the API
            resp = self.client.post_order(signed_order)
            get_rate_limit_manager().on_response(200)
            return resp
        except requests.exceptions.HTTPError as ex:
            # Extract status code and report to circuit breaker
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            print(f"HTTP error {status_code}: {ex}")
            return {}
        except Exception as ex:
            # For non-HTTP exceptions, assume server error for circuit breaker
            get_rate_limit_manager().on_response(500)
            print(f"Error creating order: {ex}")
            return {}

    def get_order_book(self, market):
        """
        Get the current order book for a specific market.

        Args:
            market (str): Market ID to query

        Returns:
            tuple: (bids_df, asks_df) - DataFrames containing bid and ask orders

        Raises:
            Exception: If the API call fails after rate limit handling
        """
        # Rate limit before making API call
        get_rate_limit_manager().acquire_sync(EndpointType.BOOK)

        try:
            orderBook = self.client.get_order_book(market)
            get_rate_limit_manager().on_response(200)
            return pd.DataFrame(orderBook.bids).astype(float), pd.DataFrame(orderBook.asks).astype(
                float
            )
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            raise
        except Exception as ex:
            get_rate_limit_manager().on_response(500)
            raise

    def get_usdc_balance(self):
        """
        Get the USDC balance of the connected wallet.

        Returns:
            float: USDC balance in decimal format
        """
        return self.usdc_contract.functions.balanceOf(self.browser_wallet).call() / 10**6

    def _fetch_pos_balance(self, url):
        """Internal method to fetch position balance with retry logic."""
        get_rate_limit_manager().acquire_sync(EndpointType.DATA_API)
        try:
            response = requests.get(url)
            response.raise_for_status()
            get_rate_limit_manager().on_response(response.status_code)
            return response
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            raise
        except Exception as ex:
            get_rate_limit_manager().on_response(500)
            raise

    def get_pos_balance(self):
        """
        Get the total value of all positions for the connected wallet.

        Returns:
            float: Total position value in USDC
        """
        res = self._fetch_pos_balance(
            f"https://data-api.polymarket.com/value?user={self.browser_wallet}"
        )
        return float(res.json()["value"])

    def get_total_balance(self):
        """
        Get the combined value of USDC balance and all positions.

        Returns:
            float: Total account value in USDC
        """
        return self.get_usdc_balance() + self.get_pos_balance()

    def _fetch_all_positions(self, url):
        """Internal method to fetch all positions with retry logic."""
        get_rate_limit_manager().acquire_sync(EndpointType.DATA_API)
        try:
            response = requests.get(url)
            response.raise_for_status()
            get_rate_limit_manager().on_response(response.status_code)
            return response
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            raise
        except Exception as ex:
            get_rate_limit_manager().on_response(500)
            raise

    def get_all_positions(self):
        """
        Get all positions for the connected wallet across all markets.

        Returns:
            DataFrame: All positions with details like market, size, avgPrice
        """
        res = self._fetch_all_positions(
            f"https://data-api.polymarket.com/positions?user={self.browser_wallet}"
        )
        return pd.DataFrame(res.json())

    def get_raw_position(self, tokenId):
        """
        Get the raw token balance for a specific market outcome token.

        Args:
            tokenId (int): Token ID to query

        Returns:
            int: Raw token amount (before decimal conversion)
        """
        return int(
            self.conditional_tokens.functions.balanceOf(self.browser_wallet, int(tokenId)).call()
        )

    def get_position(self, tokenId):
        """
        Get both raw and formatted position size for a token.

        Args:
            tokenId (int): Token ID to query

        Returns:
            tuple: (raw_position, shares) - Raw token amount and decimal shares
                   Shares less than 1 are treated as 0 to avoid dust amounts
        """
        raw_position = self.get_raw_position(tokenId)
        shares = float(raw_position / 1e6)

        # Ignore very small positions (dust)
        if shares < 1:
            shares = 0

        return raw_position, shares

    def get_all_orders(self):
        """
        Get all open orders for the connected wallet.

        Returns:
            DataFrame: All open orders with their details
        """
        # Rate limit before making API call
        get_rate_limit_manager().acquire_sync(EndpointType.GENERAL)

        try:
            orders_df = pd.DataFrame(self.client.get_orders())
            get_rate_limit_manager().on_response(200)

            # Convert numeric columns to float
            for col in ["original_size", "size_matched", "price"]:
                if col in orders_df.columns:
                    orders_df[col] = orders_df[col].astype(float)

            return orders_df
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            raise
        except Exception as ex:
            get_rate_limit_manager().on_response(500)
            raise

    def get_market_orders(self, market):
        """
        Get all open orders for a specific market.

        Args:
            market (str): Market ID to query

        Returns:
            DataFrame: Open orders for the specified market
        """
        # Rate limit before making API call
        get_rate_limit_manager().acquire_sync(EndpointType.GENERAL)

        try:
            orders_df = pd.DataFrame(
                self.client.get_orders(
                    OpenOrderParams(
                        market=market,
                    )
                )
            )
            get_rate_limit_manager().on_response(200)

            # Convert numeric columns to float
            for col in ["original_size", "size_matched", "price"]:
                if col in orders_df.columns:
                    orders_df[col] = orders_df[col].astype(float)

            return orders_df
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            raise
        except Exception as ex:
            get_rate_limit_manager().on_response(500)
            raise

    def get_market_by_token(self, token_id: str) -> Optional[dict]:
        """
        Get market info by token ID via Polymarket Gamma API.

        This is useful for looking up market details (condition_id, question, neg_risk)
        when you only have a token ID (e.g., from a position or order).

        Args:
            token_id (str): The CLOB token ID to look up

        Returns:
            dict: Market info including condition_id, question, neg_risk, tokens, etc.
                  Returns None if the token is not found or API call fails.
        """
        get_rate_limit_manager().acquire_sync(EndpointType.DATA_API)

        try:
            url = f"https://gamma-api.polymarket.com/markets?clob_token_ids={token_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            get_rate_limit_manager().on_response(response.status_code)

            markets = response.json()
            return markets[0] if markets else None
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            return None
        except Exception:
            get_rate_limit_manager().on_response(500)
            return None

    def cancel_all_asset(self, asset_id):
        """
        Cancel all orders for a specific asset token.

        Args:
            asset_id (str): Asset token ID
        """
        # Rate limit before making API call
        get_rate_limit_manager().acquire_sync(EndpointType.CANCEL_ALL)

        try:
            self.client.cancel_market_orders(asset_id=str(asset_id))
            get_rate_limit_manager().on_response(200)
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            raise
        except Exception as ex:
            get_rate_limit_manager().on_response(500)
            raise

    def cancel_all_market(self, marketId):
        """
        Cancel all orders in a specific market.

        Args:
            marketId (str): Market ID
        """
        # Rate limit before making API call
        get_rate_limit_manager().acquire_sync(EndpointType.CANCEL_ALL)

        try:
            self.client.cancel_market_orders(market=marketId)
            get_rate_limit_manager().on_response(200)
        except requests.exceptions.HTTPError as ex:
            status_code = ex.response.status_code if ex.response is not None else 500
            get_rate_limit_manager().on_response(status_code)
            raise
        except Exception as ex:
            get_rate_limit_manager().on_response(500)
            raise

    def merge_positions(self, amount_to_merge, condition_id, is_neg_risk_market):
        """
        Merge positions in a market to recover collateral.

        This function calls the external poly_merger Node.js script to execute
        the merge operation on-chain. When you hold both YES and NO positions
        in the same market, merging them recovers your USDC.

        Uses circuit breaker pattern to prevent retry spam on failures.

        Args:
            amount_to_merge (int): Raw token amount to merge (before decimal conversion)
            condition_id (str): Market condition ID
            is_neg_risk_market (bool): Whether this is a negative risk market

        Returns:
            str: Transaction hash or output from the merge script, or None if skipped

        Raises:
            Exception: If the merge operation fails (after recording in circuit breaker)
        """
        from poly_data.merge_circuit_breaker import get_merge_circuit_breaker

        circuit_breaker = get_merge_circuit_breaker()

        if not circuit_breaker.can_merge(condition_id):
            return None

        amount_to_merge_str = str(amount_to_merge)
        node_command = (
            f"node poly_merger/merge.js {amount_to_merge_str} {condition_id} "
            f'{"true" if is_neg_risk_market else "false"}'
        )
        print(f"Executing merge: {node_command}")

        try:
            result = subprocess.run(
                node_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
                circuit_breaker.record_failure(condition_id, error_msg)
                raise Exception(f"Error in merging positions: {error_msg}")

            circuit_breaker.record_success(condition_id)
            print("Done merging")
            return result.stdout

        except subprocess.TimeoutExpired:
            circuit_breaker.record_failure(condition_id, "Merge operation timed out (120s)")
            raise Exception("Merge operation timed out")
        except Exception as e:
            if "Error in merging positions" not in str(e):
                circuit_breaker.record_failure(condition_id, str(e))
            raise

    # =========================================================================
    # Async Wrappers - Run blocking sync methods in thread pool
    # =========================================================================
    # These methods prevent event loop starvation when scaling to 100+ markets.
    # Use these in async code instead of the sync versions above.

    async def create_order_async(self, marketId, action, price, size, neg_risk=False):
        """
        Async wrapper for create_order - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor,
            functools.partial(self.create_order, marketId, action, price, size, neg_risk),
        )

    async def get_order_book_async(self, market):
        """
        Async wrapper for get_order_book - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor, functools.partial(self.get_order_book, market)
        )

    async def get_all_orders_async(self):
        """
        Async wrapper for get_all_orders - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_api_executor, self.get_all_orders)

    async def get_all_positions_async(self):
        """
        Async wrapper for get_all_positions - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_api_executor, self.get_all_positions)

    async def get_position_async(self, tokenId):
        """
        Async wrapper for get_position - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor, functools.partial(self.get_position, tokenId)
        )

    async def get_raw_position_async(self, tokenId):
        """
        Async wrapper for get_raw_position - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor, functools.partial(self.get_raw_position, tokenId)
        )

    async def cancel_all_asset_async(self, asset_id):
        """
        Async wrapper for cancel_all_asset - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor, functools.partial(self.cancel_all_asset, asset_id)
        )

    async def cancel_all_market_async(self, marketId):
        """
        Async wrapper for cancel_all_market - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor, functools.partial(self.cancel_all_market, marketId)
        )

    async def merge_positions_async(self, amount_to_merge, condition_id, is_neg_risk_market):
        """
        Async wrapper for merge_positions - runs in thread pool.

        Critical: merge_positions blocks for 1-30 seconds (subprocess call).
        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor,
            functools.partial(
                self.merge_positions, amount_to_merge, condition_id, is_neg_risk_market
            ),
        )

    async def get_market_by_token_async(self, token_id: str) -> Optional[dict]:
        """
        Async wrapper for get_market_by_token - runs in thread pool.

        Use this in async code to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _api_executor, functools.partial(self.get_market_by_token, token_id)
        )
