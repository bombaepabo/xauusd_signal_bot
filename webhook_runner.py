"""
webhook_runner.py
─────────────────
Run this if you ONLY want webhook signals (no Discord bot token needed).
Just paste your webhook URL and it scans on a loop.

Usage:
    python webhook_runner.py
"""

import time
import logging
from datetime import datetime, timezone

from config import (
    DISCORD_WEBHOOK_URL,
    ATR_PERIOD,
    TP_COUNT,
    SL_STYLE,
    SCAN_INTERVAL_SECONDS,
    TREND_THRESHOLD,
    LOT_SIZE,

    ADX_MIN,
    ADX_MAX,

    USE_SR,
    SR_SWING_WINDOW,
    SR_SNAP_TPS,
    SR_FILTER_WALLS,
)
from signal_engine import SignalEngine
from discord_sender import send_webhook_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)
 

def main():
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == "":
        log.error("Set DISCORD_WEBHOOK_URL in config.py or as env var.")
        return

    engine = SignalEngine(
    atr_period=ATR_PERIOD,
    tp_count=TP_COUNT,
    sl_style=SL_STYLE,
    lot_size=LOT_SIZE,
    trend_threshold=TREND_THRESHOLD,

    adx_min=ADX_MIN,
    adx_max=ADX_MAX,

    use_sr=USE_SR,
    sr_swing_window=SR_SWING_WINDOW,
    sr_snap_tps=SR_SNAP_TPS,
    sr_filter_walls=SR_FILTER_WALLS,
)

    last_dir = None
    last_time = None

    log.info(f"Webhook runner started. Scanning every {SCAN_INTERVAL_SECONDS}s.")
    log.info(f"Webhook URL: {DISCORD_WEBHOOK_URL[:40]}...")

    while True:
        try:
            signal = engine.get_signal()

            if signal:
                now = datetime.now(timezone.utc)
                hours_since = ((now - last_time).total_seconds() / 3600) if last_time else 99

                if signal["direction"] != last_dir or hours_since >= 4:
                    send_webhook_signal(signal, DISCORD_WEBHOOK_URL)
                    last_dir = signal["direction"]
                    last_time = now
                    log.info(f"Signal sent: {signal['direction']} @ {signal['entry']}")
                else:
                    log.info(f"Same direction ({last_dir}), skipping.")
            else:
                log.info("No signal — market ranging.")

        except Exception as e:
            log.error(f"Error: {e}")

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
