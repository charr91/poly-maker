import os

# Minimum position size to trigger position merging
# Positions smaller than this will be ignored to save on gas costs
MIN_MERGE_SIZE = 20

# ============ Market Cleanup Configuration ============
# These settings control behavior when markets are removed from the Selected Markets sheet

# Whether to cancel open orders when a market is removed (default: true)
CLEANUP_CANCEL_ORDERS = os.getenv("CLEANUP_CANCEL_ORDERS", "true").lower() == "true"

# Whether to close positions when a market is removed (default: true)
CLEANUP_SELL_POSITIONS = os.getenv("CLEANUP_SELL_POSITIONS", "true").lower() == "true"

# Grace period in seconds before cleanup triggers (default: 30 = 1 update cycle)
# Allows market to "return" if accidentally removed
CLEANUP_GRACE_PERIOD = int(os.getenv("CLEANUP_GRACE_PERIOD", "30"))

# Position closing behavior:
# - False (default): In-profit positions sell at market; underwater positions get limit at break-even
# - True: Always sell at market regardless of P&L (use for immediate full exit)
CLEANUP_FORCE_MARKET_SELL = os.getenv("CLEANUP_FORCE_MARKET_SELL", "false").lower() == "true"
