"""
Discord Sender
Sends rich embed signals via discord.py channel OR webhook (no bot token needed).
"""

import logging
import requests
from datetime import datetime, timezone

import discord

log = logging.getLogger(__name__)

# Colors
COLOR_BUY  = 0x1D9E75   # green
COLOR_SELL = 0xD85A30   # orange-red
COLOR_INFO = 0xC9A84C   # gold

TP_EMOJIS  = ["1️⃣", "2️⃣", "3️⃣"]
SL_COLORS  = ["🟢", "🟡", "🔴"]   # T1=tightest, T3=widest
DIR_EMOJI  = {"BUY": "📈", "SELL": "📉"}


def build_embed(signal: dict) -> discord.Embed:
    """Build a rich Discord embed for the signal."""
    is_buy    = signal["direction"] == "BUY"
    color     = COLOR_BUY if is_buy else COLOR_SELL
    dir_arrow = "▲" if is_buy else "▼"
    dir_emoji = DIR_EMOJI[signal["direction"]]

    embed = discord.Embed(
        title=f"{dir_emoji}  XAUUSD {signal['direction']} SIGNAL  {dir_arrow}",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )

    embed.add_field(
        name="Entry",
        value=f"```{signal['entry']:.2f}```",
        inline=True
    )
    embed.add_field(
        name="ATR (14)",
        value=f"```{signal['atr']:.2f}```",
        inline=True
    )
    embed.add_field(
        name="Lot Size",
        value=f"```{signal['lot_size']:.2f}```",
        inline=True
    )

    # Trend strength bar
    filled = int(signal["strength"] / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    embed.add_field(
        name=f"Trend Bias — {signal['strength_label']} ({signal['trend_score']}/100)",
        value=f"`{bar}` {'🐂 Bullish' if is_buy else '🐻 Bearish'}",
        inline=False
    )

    # Per-trade TP/SL fields
    for i, t in enumerate(signal["trades"]):
        be_note = ""
        if i > 0:
            be_note = f"\n↳ Move SL → BE ({t['be']:.2f}) after TP{i} hits"

        embed.add_field(
            name=f"{TP_EMOJIS[i]}  Trade {t['index']}",
            value=(
                f"🟢 TP{t['index']}: `{t['tp']:.2f}`\n"
                f"🔴 SL{t['index']}: `{t['sl']:.2f}`\n"
                f"📏 R:R `1:{t['rr']:.1f}`   Risk `${t['pnl_sl']:.0f}`"
                f"{be_note}"
            ),
            inline=True
        )

    # Summary footer field
    embed.add_field(
        name="Summary",
        value=(
            f"Trades: **{signal['tp_count']}**  |  "
            f"SL Style: **{signal['sl_style'].capitalize()}**  |  "
            f"Total risk: **${signal['total_risk']:.0f}**  |  "
            f"Max reward: **${signal['total_reward']:.0f}**"
        ),
        inline=False
    )

    embed.add_field(
        name="Breakeven Rule",
        value=(
            "After each TP is hit, move the remaining trades' SL to **entry (breakeven)**.\n"
            "This locks in partial profit and eliminates risk on open trades."
        ),
        inline=False
    )

    embed.set_footer(text=f"XAUUSD Bot  •  {signal['timestamp']}")
    embed.set_thumbnail(url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Gold_bullion_bars.jpg/320px-Gold_bullion_bars.jpg")

    return embed


async def send_signal_embed(channel: discord.TextChannel, signal: dict):
    """Send signal as rich embed to a Discord channel."""
    try:
        embed = build_embed(signal)
        await channel.send(embed=embed)
        log.info(f"Signal embed sent to #{channel.name}")
    except Exception as e:
        log.error(f"Failed to send embed: {e}")


def send_webhook_signal(signal: dict, webhook_url: str):
    """Send signal via Discord webhook (no bot token needed)."""
    try:
        is_buy = signal["direction"] == "BUY"
        color  = COLOR_BUY if is_buy else COLOR_SELL

        fields = [
            {"name": "Entry",    "value": f"`{signal['entry']:.2f}`",   "inline": True},
            {"name": "ATR (14)", "value": f"`{signal['atr']:.2f}`",     "inline": True},
            {"name": "Lot Size", "value": f"`{signal['lot_size']:.2f}`","inline": True},
        ]

        filled = int(signal["strength"] / 10)
        bar    = "█" * filled + "░" * (10 - filled)
        fields.append({
            "name": f"Trend — {signal['strength_label']} ({signal['trend_score']}/100)",
            "value": f"`{bar}` {'🐂 Bullish' if is_buy else '🐻 Bearish'}",
            "inline": False
        })

        for i, t in enumerate(signal["trades"]):
            be_note = f"\n↳ BE after TP{i} hits" if i > 0 else ""
            fields.append({
                "name": f"{TP_EMOJIS[i]} Trade {t['index']}",
                "value": (
                    f"🟢 TP: `{t['tp']:.2f}`\n"
                    f"🔴 SL: `{t['sl']:.2f}`\n"
                    f"R:R `1:{t['rr']:.1f}`  Risk `${t['pnl_sl']:.0f}`"
                    f"{be_note}"
                ),
                "inline": True
            })

        fields.append({
            "name": "Summary",
            "value": (
                f"Trades: **{signal['tp_count']}** | SL: **{signal['sl_style'].capitalize()}** | "
                f"Risk: **${signal['total_risk']:.0f}** | Reward: **${signal['total_reward']:.0f}**"
            ),
            "inline": False
        })

        payload = {
            "username": "XAUUSD Signal Bot",
            "avatar_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Gold_bullion_bars.jpg/320px-Gold_bullion_bars.jpg",
            "embeds": [{
                "title": f"{'📈' if is_buy else '📉'}  XAUUSD {signal['direction']} SIGNAL  {'▲' if is_buy else '▼'}",
                "color": color,
                "fields": fields,
                "footer": {"text": f"XAUUSD Bot  •  {signal['timestamp']}"},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        }

        r = requests.post(webhook_url, json=payload, timeout=10)
        r.raise_for_status()
        log.info("Webhook signal sent.")
    except Exception as e:
        log.error(f"Webhook send failed: {e}")


# Re-use emoji list for this module
TP_EMOJIS = ["1️⃣", "2️⃣", "3️⃣"]
