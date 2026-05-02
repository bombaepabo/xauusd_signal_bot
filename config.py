"""
config.py — All settings for XAUUSD Signal Bot

HOW TO SET UP:
  Option A (recommended): Set environment variables in your shell or .env file
  Option B: Edit the DEFAULT values directly below
"""

import os

# ─────────────────────────────────────────────
#  DISCORD SETTINGS
# ─────────────────────────────────────────────

# Get from: Discord Developer Portal → Your App → Bot → Token
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "YOUR_BOT_TOKEN_HERE")

# Right-click your channel → Copy Channel ID  (enable Developer Mode in Discord settings)
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "YOUR_CHANNEL_ID_HERE")

# Optional: Channel Settings → Integrations → Webhooks → Copy URL
# If set, bot also sends via webhook (works even without a bot token)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1498208909200982037/NRvYIFw6j1x6MdOlL1esjw_1JZExkstyLXkn717KBueZH5_6n8tWNAa-YN6tPl3gOzVy")


# ─────────────────────────────────────────────
#  SIGNAL SETTINGS
# ─────────────────────────────────────────────

# How many trades per signal (1, 2, or 3)
TP_COUNT = int(os.getenv("TP_COUNT", "3"))

# SL style: "tight" | "normal" | "wide"
# tight  → SL at 0.35x / 0.55x / 0.80x ATR per trade
# normal → SL at 0.55x / 0.75x / 1.00x ATR per trade
# wide   → SL at 0.75x / 1.00x / 1.30x ATR per trade
SL_STYLE = os.getenv("SL_STYLE", "normal")

# Lot size per trade (e.g., 0.10 = 1 mini lot)
LOT_SIZE = float(os.getenv("LOT_SIZE", "0.03"))

# ATR period for volatility calculation
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# Trend score must exceed this to fire a BUY signal (0-100)
# Score below (100 - TREND_THRESHOLD) fires SELL
# Scores between threshold and (100-threshold) = no signal (ranging market)
TREND_THRESHOLD = float(os.getenv("TREND_THRESHOLD", "60"))

# How often to scan for signals (in seconds)
# 300 = every 5 minutes, 900 = every 15 minutes
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
