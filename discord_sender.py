"""
discord_sender.py — XAUUSD Signal Bot
Sends rich embeds with full momentum context:
  - ADX value + label + slope direction
  - RSI value + interpretation
  - Stochastic %K/%D
  - +DI / -DI spread
  - Entry guidance (limit vs stop)
  - HTF bias
"""

import logging
import requests
from datetime import datetime, timezone

import discord

log = logging.getLogger(__name__)

COLOR_BUY  = 0x1D9E75
COLOR_SELL = 0xD85A30
COLOR_GOLD = 0xC9A84C

TP_EMOJIS = ["1️⃣", "2️⃣", "3️⃣"]
DIR_EMOJI  = {"BUY": "📈", "SELL": "📉"}


def _adx_bar(adx: float) -> str:
    """Visual bar showing where ADX sits in the 0-70 range."""
    filled = min(int(adx / 7), 10)
    return "█" * filled + "░" * (10 - filled)


def _rsi_bar(rsi: float) -> str:
    filled = min(int(rsi / 10), 10)
    return "█" * filled + "░" * (10 - filled)


def _slope_arrow(adx_rising) -> str:
    if adx_rising is True:  return "↑"
    if adx_rising is False: return "↓"
    return "→"


def build_embed(signal: dict) -> discord.Embed:
    is_buy    = signal["direction"] == "BUY"
    color     = COLOR_BUY if is_buy else COLOR_SELL
    dir_arrow = "▲" if is_buy else "▼"
    dir_emoji = DIR_EMOJI[signal["direction"]]

    embed = discord.Embed(
        title=f"{dir_emoji}  XAUUSD {signal['direction']} SIGNAL  {dir_arrow}",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )

    # ── Row 1: Entry / ATR / Lot ───────────────────────────────────────────
    embed.add_field(name="Entry",    value=f"```{signal['entry']:.2f}```",    inline=True)
    embed.add_field(name="ATR (14)", value=f"```{signal['atr']:.2f}```",      inline=True)
    embed.add_field(name="Lot Size", value=f"```{signal['lot_size']:.2f}```", inline=True)

    # ── Trend score bar ────────────────────────────────────────────────────
    filled = int(signal["strength"] / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    embed.add_field(
        name=f"Trend Bias — {signal['strength_label']} ({signal['trend_score']}/100)",
        value=f"`{bar}` {'🐂 Bullish' if is_buy else '🐻 Bearish'}",
        inline=False
    )

    # ── Momentum & Strength section (the new section) ─────────────────────
    adx        = signal.get("adx", 0)
    adx_label  = signal.get("adx_label", "")
    adx_advice = signal.get("adx_advice", "")
    adx_slope  = signal.get("adx_slope_label", "")
    adx_rising = signal.get("adx_rising", None)
    rsi        = signal.get("rsi", 50.0)
    rsi_label  = signal.get("rsi_label", "")
    stoch_k    = signal.get("stoch_k", 50.0)
    stoch_d    = signal.get("stoch_d", 50.0)
    stoch_lbl  = signal.get("stoch_label", "")
    plus_di    = signal.get("plus_di", 0.0)
    minus_di   = signal.get("minus_di", 0.0)
    di_label   = signal.get("di_label", "")

    adx_bar_str = _adx_bar(adx)
    rsi_bar_str = _rsi_bar(rsi)
    slope_arrow = _slope_arrow(adx_rising)

    embed.add_field(
        name="📊 Momentum & Strength",
        value=(
            f"**ADX: `{adx:.1f}`** {slope_arrow}  —  {adx_label}\n"
            f"`{adx_bar_str}` *(0 ░░░ 25 ░░░ 50 ░░░ 70)*\n"
            f"Slope: {adx_slope}\n"
            f"⚙️ *{adx_advice}*"
        ),
        inline=False
    )

    embed.add_field(
        name="📉 RSI & Stochastic",
        value=(
            f"RSI: `{rsi:.1f}` — {rsi_label}\n"
            f"`{rsi_bar_str}`\n"
            f"Stoch %K: `{stoch_k:.1f}`  %D: `{stoch_d:.1f}` — {stoch_lbl}"
        ),
        inline=True
    )

    embed.add_field(
        name="🧭 Directional Index",
        value=(
            f"+DI: `{plus_di:.1f}`\n"
            f"−DI: `{minus_di:.1f}`\n"
            f"Spread: `{abs(plus_di-minus_di):.1f}` — {di_label}"
        ),
        inline=True
    )

    # HTF bias
    htf = signal.get("htf_bias", "NEUTRAL")
    htf_emoji = "🟢" if htf == "BUY" else "🔴" if htf == "SELL" else "⚪"
    embed.add_field(
        name="🕐 4H Trend Bias",
        value=f"{htf_emoji} `{htf}` — Higher timeframe alignment",
        inline=True
    )

    # ── Entry guidance ─────────────────────────────────────────────────────
    entry_action = signal.get("entry_action", "")
    embed.add_field(
        name="🎯 Entry Guidance",
        value=f"{entry_action}",
        inline=False
    )

    # ── Per-trade TP/SL ────────────────────────────────────────────────────
    for i, t in enumerate(signal["trades"]):
        be_note = f"\n↳ BE after TP{i} hits" if i > 0 else ""
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

    # ── Summary ────────────────────────────────────────────────────────────
    embed.add_field(
        name="Summary",
        value=(
            f"Trades: **{signal['tp_count']}**  |  "
            f"SL: **{signal['sl_style'].capitalize()}**  |  "
            f"Risk: **${signal['total_risk']:.0f}**  |  "
            f"Reward: **${signal['total_reward']:.0f}**"
        ),
        inline=False
    )

    embed.set_footer(text=f"XAUUSD Bot  •  {signal['timestamp']}")
    embed.set_thumbnail(
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Gold_bullion_bars.jpg/320px-Gold_bullion_bars.jpg"
    )
    return embed


async def send_signal_embed(channel: discord.TextChannel, signal: dict):
    try:
        embed = build_embed(signal)
        await channel.send(embed=embed)
        log.info(f"Embed sent to #{channel.name}")
    except Exception as e:
        log.error(f"Failed to send embed: {e}")


def send_webhook_signal(signal: dict, webhook_url: str):
    try:
        is_buy = signal["direction"] == "BUY"
        color  = COLOR_BUY if is_buy else COLOR_SELL

        adx        = signal.get("adx", 0)
        adx_label  = signal.get("adx_label", "")
        adx_advice = signal.get("adx_advice", "")
        adx_slope  = signal.get("adx_slope_label", "")
        adx_rising = signal.get("adx_rising", None)
        rsi        = signal.get("rsi", 50.0)
        rsi_label  = signal.get("rsi_label", "")
        stoch_k    = signal.get("stoch_k", 50.0)
        stoch_d    = signal.get("stoch_d", 50.0)
        stoch_lbl  = signal.get("stoch_label", "")
        plus_di    = signal.get("plus_di", 0.0)
        minus_di   = signal.get("minus_di", 0.0)
        di_label   = signal.get("di_label", "")
        slope_arrow = _slope_arrow(adx_rising)
        adx_bar_str = _adx_bar(adx)
        rsi_bar_str = _rsi_bar(rsi)
        htf         = signal.get("htf_bias", "NEUTRAL")
        htf_emoji   = "🟢" if htf == "BUY" else "🔴" if htf == "SELL" else "⚪"

        # Trend bar
        filled = int(signal["strength"] / 10)
        bar    = "█" * filled + "░" * (10 - filled)

        fields = [
            # Row 1
            {"name": "Entry",    "value": f"`{signal['entry']:.2f}`",    "inline": True},
            {"name": "ATR (14)", "value": f"`{signal['atr']:.2f}`",      "inline": True},
            {"name": "Lot Size", "value": f"`{signal['lot_size']:.2f}`", "inline": True},
            # Trend
            {
                "name": f"Trend Bias — {signal['strength_label']} ({signal['trend_score']}/100)",
                "value": f"`{bar}` {'🐂 Bullish' if is_buy else '🐻 Bearish'}",
                "inline": False
            },
            # ADX momentum
            {
                "name": "📊 Momentum & Strength",
                "value": (
                    f"**ADX: `{adx:.1f}`** {slope_arrow}  —  {adx_label}\n"
                    f"`{adx_bar_str}` *(0 ░░ 25 ░░ 50 ░░ 70)*\n"
                    f"Slope: {adx_slope}\n"
                    f"⚙️ *{adx_advice}*"
                ),
                "inline": False
            },
            # RSI + Stoch
            {
                "name": "📉 RSI & Stochastic",
                "value": (
                    f"RSI: `{rsi:.1f}` — {rsi_label}\n"
                    f"`{rsi_bar_str}`\n"
                    f"Stoch %K: `{stoch_k:.1f}`  %D: `{stoch_d:.1f}` — {stoch_lbl}"
                ),
                "inline": True
            },
            # DI
            {
                "name": "🧭 Directional Index",
                "value": (
                    f"+DI: `{plus_di:.1f}`\n"
                    f"−DI: `{minus_di:.1f}`\n"
                    f"Spread: `{abs(plus_di-minus_di):.1f}` — {di_label}"
                ),
                "inline": True
            },
            # HTF
            {
                "name": "🕐 4H Trend Bias",
                "value": f"{htf_emoji} `{htf}` — Higher timeframe",
                "inline": True
            },
            # Entry guidance
            {
                "name": "🎯 Entry Guidance",
                "value": signal.get("entry_action", ""),
                "inline": False
            },
        ]

        # Trades
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

        # Summary
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
