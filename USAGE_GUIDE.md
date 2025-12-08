# Poly-Maker Usage Guide

Run the bot with optional VPN routing through PIA.

## First-Time Setup

### 1. Configure `.env`:
```bash
PIA_USER=p1234567          # PIA username (p-number)
PIA_PASS=your_password     # PIA password
PIA_DIP_TOKEN=DIP...       # Dedicated IP token (optional)
```

### 2. Ensure `credentials.json` exists
(Google Sheets service account credentials)

### 3. Build the image:
```bash
./bot.sh build
```

### 4. Generate VPN config (for dedicated IP mode):
```bash
# Requires jq: sudo apt install jq
./bot.sh vpn setup
```

## Quick Start

```bash
./bot.sh start          # Start with dedicated IP VPN (default)
./bot.sh start novpn    # Start without VPN
./bot.sh start pia      # Start with regular PIA region
./bot.sh stop           # Stop bot
```

## All Commands

| Command | Description |
|---------|-------------|
| `./bot.sh start` | Start with dedicated IP VPN |
| `./bot.sh start novpn` | Start without VPN |
| `./bot.sh start pia` | Start with regular PIA region |
| `./bot.sh stop` | Stop bot |
| `./bot.sh restart` | Restart in current mode |
| `./bot.sh status` | Show status and public IP |
| `./bot.sh logs` | Follow all logs |
| `./bot.sh logs polybot` | Follow bot logs only |
| `./bot.sh logs vpn` | Follow VPN logs only |
| `./bot.sh ip` | Show current public IP |
| `./bot.sh vpn setup` | Regenerate dedicated IP config |
| `./bot.sh vpn region "US Texas"` | Set PIA region for 'pia' mode |
| `./bot.sh build` | Rebuild bot image |

## VPN Modes

| Mode | Command | Description |
|------|---------|-------------|
| Dedicated IP | `./bot.sh start` | Uses your PIA dedicated IP (default) |
| No VPN | `./bot.sh start novpn` | Direct connection, no VPN |
| Regular PIA | `./bot.sh start pia` | Uses regular PIA server by region |

### Change PIA Region (for 'pia' mode)
```bash
./bot.sh vpn region "US Texas"
./bot.sh vpn region "CA Vancouver"
./bot.sh vpn region "UK London"
./bot.sh start pia
```

## Update Bot Code

```bash
git pull
./bot.sh build
./bot.sh start
```

## Troubleshooting

```bash
./bot.sh status         # Check what's running
./bot.sh logs vpn       # VPN connection issues
./bot.sh logs polybot   # Bot errors
./bot.sh stop && ./bot.sh start   # Full restart
```

## Architecture

```
┌─────────────────────────────────────┐
│  gluetun (VPN)                      │
│  - PIA Dedicated IP / Region        │
│  - Kill switch enabled              │
└──────────────┬──────────────────────┘
               │ network_mode: service
┌──────────────▼──────────────────────┐
│  polybot (main.py)                  │
│  - All traffic routes through VPN   │
│  - Rate limiting enabled            │
└─────────────────────────────────────┘
```

## Configuration

Trading behavior is configured via `.env`. See `.env.example` for all options.

### Market Cleanup

When you remove a market from the "Selected Markets" sheet, the bot automatically cleans up:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLEANUP_CANCEL_ORDERS` | `true` | Cancel open orders when market removed |
| `CLEANUP_SELL_POSITIONS` | `true` | Close positions when market removed |
| `CLEANUP_GRACE_PERIOD` | `30` | Seconds before cleanup triggers |
| `CLEANUP_FORCE_MARKET_SELL` | `false` | Force market sell even if underwater |

Position closing behavior (when `CLEANUP_FORCE_MARKET_SELL=false`):
- **In-profit positions**: Sell at market price (immediate)
- **Underwater positions**: Place limit order at break-even price

## Files

| File | Purpose |
|------|---------|
| `bot.sh` | Helper script for all operations |
| `docker-compose.yml` | Dedicated IP VPN mode |
| `docker-compose.novpn.yml` | No VPN mode |
| `docker-compose.pia.yml` | Regular PIA region mode |
| `vpn/setup_pia_dip.sh` | Generates dedicated IP config |
| `.env` | All configuration variables |
