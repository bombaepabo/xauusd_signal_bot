"""
signal_engine.py — XAUUSD Signal Engine v2
───────────────────────────────────────────
Fixes applied vs v1:
  1. ADX filter  — only trade when ADX > 22 (trending market, not ranging)
  2. Wider SL    — 1.2x / 1.5x / 2.0x ATR (v1 was 0.55x too tight, instant stop-outs)
  3. Better TP   — 1.5x / 2.5x / 3.5x ATR (proper R:R > 1.2 on every trade)
  4. EMA-200     — macro trend filter, price must be on right side of 200 EMA
  5. ADX +DI/-DI — directional component in scoring replaces stochastic overweight
  6. Higher threshold — 65 (was 60), fewer but higher-quality signals
  7. Doji filter — skip low-body candles (noise, not impulse)
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# v2 SL multipliers — wider to survive XAUUSD noise ($10-20 swings common)
SL_MULTIPLIERS = {
    "tight":  [1.0, 1.3, 1.7],
    "normal": [1.2, 1.5, 2.0],
    "wide":   [1.5, 2.0, 2.5],
}

# v2 TP multipliers — genuine R:R > 1.2 on every trade
TP_MULTIPLIERS = {
    1: [2.0],
    2: [1.5, 2.8],
    3: [1.5, 2.5, 3.5],
}


class SignalEngine:
    def __init__(
        self,
        atr_period: int        = 14,
        tp_count: int          = 3,
        sl_style: str          = "normal",
        lot_size: float        = 0.10,
        trend_threshold: float = 65.0,
        adx_threshold: float   = 22.0,
        cooldown_bars: int     = 8,
    ):
        self.atr_period      = atr_period
        self.tp_count        = tp_count
        self.sl_style        = sl_style
        self.lot_size        = lot_size
        self.trend_threshold = trend_threshold
        self.adx_threshold   = adx_threshold
        self.cooldown_bars   = cooldown_bars

    def fetch_data(self, period="10d", interval="1h"):
        try:
            ticker = yf.Ticker("GC=F")
            df = ticker.history(period=period, interval=interval)
            if df.empty or len(df) < 60:
                log.warning("Not enough data.")
                return None
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            log.error(f"Fetch failed: {e}")
            return None

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]

        # ATR
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # EMAs
        df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
        df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
        df["ema200"] = c.ewm(span=200, adjust=False).mean()

        # Stochastic
        lo14 = l.rolling(14).min()
        hi14 = h.rolling(14).max()
        df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ADX + Directional Indicators
        plus_dm  = (h - h.shift(1)).clip(lower=0)
        minus_dm = (l.shift(1) - l).clip(lower=0)
        plus_dm  = plus_dm.where(plus_dm > minus_dm, 0.0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
        atr14    = tr.ewm(span=14, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["adx"]      = dx.ewm(span=14, adjust=False).mean()
        df["plus_di"]  = plus_di
        df["minus_di"] = minus_di

        # Candle body
        df["body"]    = (c - o).abs()
        df["vol_sma"] = df["Volume"].rolling(20).mean()

        return df

    def score_trend(self, row: pd.Series) -> float:
        needed = ["rsi", "ema20", "ema50", "ema200", "adx", "plus_di", "minus_di", "stoch_k", "stoch_d"]
        if any(pd.isna(row.get(k, np.nan)) for k in needed):
            return 50.0

        score = 50.0
        rsi   = row["rsi"]
        ema20, ema50, ema200 = row["ema20"], row["ema50"], row["ema200"]
        sk, sd               = row["stoch_k"], row["stoch_d"]
        plus_di, minus_di    = row["plus_di"], row["minus_di"]
        close                = row["Close"]

        # RSI (±18 pts)
        if rsi > 55:   score += min((rsi - 55) * 0.72, 18)
        elif rsi < 45: score -= min((45 - rsi) * 0.72, 18)

        # EMA 20/50 (±14 pts)
        ema_gap = (ema20 - ema50) / max(abs(ema50), 1) * 100
        if ema20 > ema50:  score += min(ema_gap * 4, 14)
        elif ema20 < ema50: score -= min(-ema_gap * 4, 14)

        # EMA 200 macro bias (±8 pts)
        if close > ema200:  score += 8
        elif close < ema200: score -= 8

        # Stochastic (±10 pts)
        if sk > 55 and sk > sd:   score += 7
        elif sk < 45 and sk < sd: score -= 7
        if sk > 70:  score += 3
        elif sk < 30: score -= 3

        # ADX directional (±10 pts)
        if plus_di > minus_di:  score += 10
        elif minus_di > plus_di: score -= 10

        return float(max(0, min(100, score)))

    def build_signal(self, entry, atr, direction, trend_score, adx):
        is_buy   = direction == "BUY"
        sl_mults = SL_MULTIPLIERS[self.sl_style]
        tp_mults = TP_MULTIPLIERS[self.tp_count]
        pip_val  = self.lot_size * 1000 * 0.1

        trades = []
        for i in range(self.tp_count):
            tp    = entry + atr * tp_mults[i] if is_buy else entry - atr * tp_mults[i]
            sl    = entry - atr * sl_mults[i] if is_buy else entry + atr * sl_mults[i]
            tp_d  = abs(tp - entry)
            sl_d  = abs(sl - entry)
            trades.append({
                "index":   i + 1,
                "tp":      round(tp, 2),
                "sl":      round(sl, 2),
                "be":      round(entry, 2),
                "tp_dist": round(tp_d, 2),
                "sl_dist": round(sl_d, 2),
                "rr":      round(tp_d / max(sl_d, 0.01), 2),
                "pnl_tp":  round((tp_d / 0.1) * pip_val, 2),
                "pnl_sl":  round((sl_d / 0.1) * pip_val, 2),
            })

        strength = abs(trend_score - 50) * 2
        s_label  = ("Weak" if strength < 30 else
                    "Moderate" if strength < 60 else
                    "Strong" if strength < 85 else "Very Strong")

        return {
            "direction":      direction,
            "entry":          round(entry, 2),
            "atr":            round(atr, 2),
            "adx":            round(adx, 1),
            "lot_size":       self.lot_size,
            "trend_score":    round(trend_score, 1),
            "strength":       round(strength, 1),
            "strength_label": s_label,
            "tp_count":       self.tp_count,
            "sl_style":       self.sl_style,
            "trades":         trades,
            "total_risk":     round(sum(t["pnl_sl"] for t in trades), 2),
            "total_reward":   round(sum(t["pnl_tp"] for t in trades), 2),
            "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

    def get_signal(self):
        df = self.fetch_data()
        if df is None:
            return None
        df  = self.compute_indicators(df)
        row = df.iloc[-1]

        adx   = float(row.get("adx", 0) or 0)
        entry = float(row["Close"])
        atr   = float(row["atr"])

        if adx < self.adx_threshold:
            log.info(f"ADX {adx:.1f} < {self.adx_threshold} — ranging, skip.")
            return None

        body_ratio = float(row["body"]) / max(float(row["atr"]), 0.01)
        if body_ratio < 0.15:
            log.info("Doji candle — skip.")
            return None

        score = self.score_trend(row)
        log.info(f"Price:{entry:.2f} ATR:{atr:.2f} ADX:{adx:.1f} Score:{score:.1f}")

        if score >= self.trend_threshold:
            direction = "BUY"
        elif score <= (100 - self.trend_threshold):
            direction = "SELL"
        else:
            log.info("Neutral zone — no signal.")
            return None

        return self.build_signal(entry, atr, direction, score, adx)