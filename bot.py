"""
XAUUSD Signal Bot — Real-Time with Discord Alerts
Data: yfinance (free, no API key) + Twelve Data (free tier fallback)
Signal: Discord webhook + optional bot command
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import discord
from discord.ext import tasks
import yfinance as yf
import pandas as pd
import numpy as np

from config import (
    DISCORD_TOKEN, DISCORD_CHANNEL_ID, DISCORD_WEBHOOK_URL,
    ATR_PERIOD, TP_COUNT, SL_STYLE, SCAN_INTERVAL_SECONDS,
    TREND_THRESHOLD, LOT_SIZE
)
from signal_engine import SignalEngine
from discord_sender import send_signal_embed, send_webhook_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


class XAUUSDBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.engine = SignalEngine(
            atr_period=ATR_PERIOD,
            tp_count=TP_COUNT,
            sl_style=SL_STYLE,
            lot_size=LOT_SIZE,
            trend_threshold=TREND_THRESHOLD
        )
        self.last_signal_dir = None   # avoid spam — only send on direction change
        self.last_signal_time = None

    async def on_ready(self):
        log.info(f"Bot online as {self.user}")
        self.scan_loop.start()

    async def on_message(self, message):
        if message.author == self.user:
            return

        # Manual command: !signal
        if message.content.strip().lower() == "!signal":
            await message.channel.send("Fetching live XAUUSD signal...")
            signal = await asyncio.to_thread(self.engine.get_signal)
            if signal:
                await send_signal_embed(message.channel, signal)
            else:
                await message.channel.send("No clear signal right now. Market is ranging.")

        # Manual command: !price
        if message.content.strip().lower() == "!price":
            data = await asyncio.to_thread(fetch_price)
            if data:
                await message.channel.send(
                    f"**XAUUSD** — `{data['price']:.2f}` | ATR: `{data['atr']:.2f}` | "
                    f"RSI: `{data['rsi']:.1f}` | Trend: `{data['trend_score']}/100`"
                )

        # Manual command: !status
        if message.content.strip().lower() == "!status":
            status = (
                f"Bot running — scanning every {SCAN_INTERVAL_SECONDS}s\n"
                f"Last signal direction: **{self.last_signal_dir or 'None'}**\n"
                f"Last signal time: **{self.last_signal_time or 'N/A'}**\n"
                f"TP count: **{TP_COUNT}** | SL style: **{SL_STYLE}** | Lot: **{LOT_SIZE}**"
            )
            await message.channel.send(status)

    @tasks.loop(seconds=SCAN_INTERVAL_SECONDS)
    async def scan_loop(self):
        try:
            log.info("Scanning XAUUSD...")
            signal = await asyncio.to_thread(self.engine.get_signal)

            if not signal:
                log.info("No signal generated.")
                return

            # Only fire if direction changed OR it's been > 4 hours
            now = datetime.now(timezone.utc)
            hours_since = 99
            if self.last_signal_time:
                hours_since = (now - self.last_signal_time).total_seconds() / 3600

            if signal["direction"] == self.last_signal_dir and hours_since < 4:
                log.info(f"Same direction ({signal['direction']}), skipping duplicate.")
                return

            self.last_signal_dir = signal["direction"]
            self.last_signal_time = now

            # Send via Discord bot channel
            if DISCORD_CHANNEL_ID:
                channel = self.get_channel(int(DISCORD_CHANNEL_ID))
                if channel:
                    await send_signal_embed(channel, signal)

            # Send via webhook (optional backup)
            if DISCORD_WEBHOOK_URL:
                await asyncio.to_thread(send_webhook_signal, signal, DISCORD_WEBHOOK_URL)

            log.info(f"Signal sent: {signal['direction']} | Entry: {signal['entry']:.2f}")

        except Exception as e:
            log.error(f"Scan error: {e}")

    @scan_loop.before_loop
    async def before_scan(self):
        await self.wait_until_ready()


def fetch_price():
    """Fetch latest XAUUSD price data via yfinance."""
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period="5d", interval="1h")
        if df.empty:
            return None
        close = df["Close"].dropna()
        atr = compute_atr(df, 14)
        rsi = compute_rsi(close, 14)
        return {
            "price": float(close.iloc[-1]),
            "atr": float(atr.iloc[-1]),
            "rsi": float(rsi.iloc[-1]),
            "trend_score": rsi_to_trend(float(rsi.iloc[-1]))
        }
    except Exception as e:
        log.error(f"Price fetch error: {e}")
        return None


def compute_atr(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_to_trend(rsi):
    """Convert RSI to 0-100 trend score (50=neutral, 100=strong bull)."""
    return int(max(0, min(100, rsi)))


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        log.error("DISCORD_TOKEN not set in config.py or .env")
        exit(1)

    bot = XAUUSDBot()
    bot.run(DISCORD_TOKEN)
