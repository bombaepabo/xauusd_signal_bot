"""
config.py — All settings, read from environment variables
Set values in .env file (copy from .env.example)
"""
import os

# Discord
DISCORD_TOKEN       = os.getenv("DISCORD_TOKEN", "")
DISCORD_CHANNEL_ID  = os.getenv("DISCORD_CHANNEL_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1498208909200982037/NRvYIFw6j1x6MdOlL1esjw_1JZExkstyLXkn717KBueZH5_6n8tWNAa-YN6tPl3gOzVy")

# Signal
TP_COUNT              = int(os.getenv("TP_COUNT", "3"))
SL_STYLE              = os.getenv("SL_STYLE", "normal")
LOT_SIZE              = float(os.getenv("LOT_SIZE", "0.01"))
ATR_PERIOD            = int(os.getenv("ATR_PERIOD", "14"))
TREND_THRESHOLD       = float(os.getenv("TREND_THRESHOLD", "65.0"))
ADX_MIN               = float(os.getenv("ADX_MIN", "40.0"))   # updated: ADX 60 sweet spot
ADX_MAX               = float(os.getenv("ADX_MAX", "60.0"))   # updated: tighter window
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))

# Support & Resistance
USE_SR           = os.getenv("USE_SR", "true").lower() == "true"
SR_SWING_WINDOW  = int(os.getenv("SR_SWING_WINDOW", "10"))
SR_SNAP_TPS      = os.getenv("SR_SNAP_TPS", "true").lower() == "true"
SR_FILTER_WALLS  = os.getenv("SR_FILTER_WALLS", "true").lower() == "true"
