# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poly-Maker is a market making bot for Polymarket prediction markets. It automates liquidity provision by maintaining orders on both sides of the book with sophisticated risk management and position merging capabilities.

## Development Setup

### Package Management
This project uses UV for dependency management (faster than pip/poetry):

```bash
# Install dependencies
uv sync

# Install with dev dependencies (includes black, pytest)
uv sync --extra dev

# Run scripts
uv run python main.py
uv run python update_markets.py
uv run python update_stats.py
```

### Initial Setup Requirements

1. **Python Environment**: Python 3.9.10+ required (specified in `.python-version`)

2. **Node.js Dependencies** (for position merger):
```bash
cd poly_merger && npm install && cd ..
```

3. **Environment Configuration**:
   - Copy `.env.example` to `.env`
   - Set `PK` (Polymarket private key) and `BROWSER_ADDRESS` (wallet address)
   - **Important**: Wallet must have completed at least one trade through the UI for proper permissions
   - Add `credentials.json` (Google Service Account) to project root
   - Configure `SPREADSHEET_URL` pointing to your Google Sheets configuration

4. **Google Sheets Setup**:
   - Copy the sample sheet: https://docs.google.com/spreadsheets/d/1Kt6yGY7CZpB75cLJJAdWo7LSp9Oz7pjqfuVWwgtn7Ns/edit
   - Share with your Google service account email (with edit permissions)
   - The sheet contains three worksheets:
     - **Selected Markets**: Markets you actively trade
     - **All Markets**: Database of available markets (populated by `update_markets.py`)
     - **Hyperparameters**: Trading configuration parameters (spread, size, risk thresholds)

### Docker Deployment

The bot supports three Docker deployment modes via the `bot.sh` helper script:

```bash
# Standard mode with VPN (PIA Dedicated IP via gluetun)
./bot.sh start vpn

# PIA mode with standard regions
./bot.sh start pia

# No VPN mode (local development)
./bot.sh start novpn

# View logs
./bot.sh logs

# Stop bot
./bot.sh stop
```

The VPN setup uses gluetun for network isolation and requires PIA credentials in `.env`.

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_rate_limiter.py
```

### Code Formatting

Uses Black with 100-character line length (configured in `pyproject.toml`):

```bash
uv run black .
```

## Architecture

### Core Components

**Main Entry Point** (`main.py`):
- Initializes the PolymarketClient and global state
- Starts background thread for periodic updates (positions every 5s, markets every 30s)
- Maintains WebSocket connections to Polymarket (market data + user trades)
- Auto-reconnects on WebSocket failures

**Trading Engine** (`trading.py`):
- `perform_trade(market)`: Main market-making logic per market
  - Position merging when holding both YES and NO (frees capital)
  - Bid/ask price calculation based on order book depth and spreads
  - Risk management: stop-loss (sells on negative PnL), take-profit levels
  - Order management: only updates orders when prices/sizes change significantly
  - Prevents trading when volatility is high or during "risk-off" periods

**Data Layer** (`poly_data/`):
- `polymarket_client.py`: Wrapper around py-clob-client for Polymarket API
  - Handles order creation, cancellation, position queries
  - Manages blockchain interactions (Web3, smart contracts)
  - Uses rate limiting to avoid API bans
- `websocket_handlers.py`: WebSocket management
  - `connect_market_websocket()`: Order book updates
  - `connect_user_websocket()`: User order/trade events
- `data_processing.py`: Processes WebSocket events
  - Maintains in-memory order book (SortedDict for bids/asks)
  - Triggers `perform_trade()` on price changes
  - Tracks pending trades to prevent race conditions
- `global_state.py`: Shared state (positions, orders, markets, parameters)
- `rate_limiter.py`: Token bucket rate limiter with exponential backoff
  - Default: 5 requests/second with burst capacity of 10
  - Async and sync support

**Position Merger** (`poly_merger/`):
- Node.js utility based on open-source Polymarket code
- Merges opposing positions (YES + NO → USDC) to reduce gas and free capital
- Called automatically by trading engine when opposing positions exceed `MIN_MERGE_SIZE`
- Can be run standalone: `node merge.js [amount] [condition_id] [is_neg_risk]`

**Market Data Updater** (`update_markets.py`):
- Fetches all available markets from Polymarket API
- Should run continuously on a separate IP (rate limit concerns)
- Updates the "All Markets" sheet in Google Sheets

### Key Trading Logic Flow

1. **WebSocket receives order book update** → triggers `perform_trade(market)`
2. **Position Merging**: If holding both YES and NO above threshold, merge positions
3. **For each outcome** (YES/NO):
   - Get order book depth, calculate optimal bid/ask prices
   - **Stop-Loss**: If PnL < threshold and spread is tight, sell at best bid and enter "risk-off" period
   - **Buy Logic**: Place buy orders if:
     - Position < max_size and position < 250
     - Not in risk-off period
     - Volatility is acceptable
     - No significant opposing position
     - Market has healthy buy/sell ratio
   - **Sell Logic**: Place sell orders at take-profit price when holding position
4. **Order Optimization**: Only cancel/replace orders if price/size changes are significant (>0.5 cents or >10%)

### Risk Management Features

- **Stop-Loss**: Exits positions when PnL drops below threshold (configurable per market type)
- **Volatility Filter**: Blocks new trades when 3-hour volatility exceeds threshold
- **Position Limits**: Configurable `max_size` per market (defaults to `trade_size`)
- **Risk-Off Periods**: After stop-loss trigger, pauses trading for configurable hours
- **Price Sanity Checks**: Only places orders between 0.1-0.9 price range
- **Opposing Position Detection**: Won't buy more if holding significant opposite side
- **Market Cleanup**: Auto-cancels orders and closes positions when markets are removed from Selected Markets sheet (see below)

### Market Cleanup

When a market is removed from the "Selected Markets" sheet, the bot automatically cleans up:
1. **Grace period** (30s default): Allows market to return if accidentally removed
2. **Order cancellation**: Cancels all open orders for the market
3. **Position closing**:
   - In-profit positions → sell at market (immediate)
   - Underwater positions → limit order at break-even price
4. **State cleanup**: Removes from all in-memory structures, deletes risk-off files

Configuration via `.env`:
- `CLEANUP_CANCEL_ORDERS`: Cancel orders on removal (default: true)
- `CLEANUP_SELL_POSITIONS`: Close positions on removal (default: true)
- `CLEANUP_GRACE_PERIOD`: Seconds before cleanup triggers (default: 30)
- `CLEANUP_FORCE_MARKET_SELL`: Force market sell even if underwater (default: false)

### Global State Management

`poly_data/global_state.py` stores:
- `df`: DataFrame of markets from Google Sheets
- `positions`: Dict[token_id, {size, avgPrice}]
- `orders`: Dict[token_id, {buy: {price, size}, sell: {price, size}}]
- `all_data`: Real-time order book data from WebSocket
- `performing`: Set of trades currently in flight (prevents duplicates)
- `params`: Hyperparameters per market type (from Google Sheets)
- `client`: PolymarketClient instance
- `pending_removal`: Markets queued for cleanup (grace period)
- `removing_markets`: Markets currently being cleaned up (skips trades)

### Rate Limiting

All API calls use the global rate limiter from `poly_data/rate_limiter.py`:
- Sync calls: `get_rate_limiter().acquire_sync()`
- Async calls: `await get_rate_limiter().acquire()`

This prevents API bans when operating on many markets simultaneously.

### Important Constants

Defined in `poly_data/CONSTANTS.py`:
- `MIN_MERGE_SIZE`: Minimum position size to trigger merging
- `CLEANUP_*`: Market cleanup configuration (see Market Cleanup section)

## Common Workflows

### Running the Bot Locally
```bash
# First time: Update market data (run in background)
uv run python update_markets.py

# Start the market maker
uv run python main.py
```

### Running with Docker
```bash
# Build and start with VPN
./bot.sh start vpn

# Check logs
./bot.sh logs -f

# Restart
./bot.sh restart
```

### Adding a New Market
1. Run `update_markets.py` to refresh market list
2. Add market to "Selected Markets" sheet in Google Sheets
3. Configure parameters in "Hyperparameters" sheet for the market's `param_type`
4. Bot will automatically start trading when it detects the new market

### Debugging Position Issues
- Check `positions/` directory for risk-off state files (JSON with market condition_id as filename)
- Review logs for "Risking off" messages indicating stop-loss triggers
- Verify position sizes: `global_state.positions` vs blockchain truth via `client.get_position(token)`

## Security Notes

- Never commit `.env`, `credentials.json`, or files in `positions/` directory
- Private keys are loaded from environment variables only
- The bot trades with real money on live markets - test thoroughly with small amounts first
- VPN is recommended to prevent IP-based rate limiting and for privacy