"""
signal_engine.py — XAUUSD Signal Engine v4
═══════════════════════════════════════════
New in v4:
  - Support & Resistance detection (swing highs/lows + round numbers + ATR clustering)
  - S&R used to: filter bad entries, snap TPs to levels, warn about walls
  - ADX window updated to 55-68 based on user's ADX 60 = best PF finding
  - ADX label updated: 55-68 = "Sweet spot ★"
  - S&R section in signal dict for Discord embed
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

log = logging.getLogger(__name__)

SL_MULTIPLIERS = {
    "tight":  [1.0, 1.3, 1.7],
    "normal": [1.2, 1.5, 2.0],
    "wide":   [1.5, 2.0, 2.5],
}
TP_MULTIPLIERS = {
    1: [2.0],
    2: [1.5, 2.8],
    3: [1.5, 2.5, 3.5],
}


# ══════════════════════════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SREngine:
    """
    Detects support and resistance levels from price data.
    Uses three methods combined:
      1. Swing highs / lows (rolling window peaks and troughs)
      2. Round numbers ($50 and $100 increments for XAUUSD)
      3. ATR clustering (merge levels within 0.5xATR to avoid noise)
    """

    def __init__(self, swing_window: int = 10, round_number_step: float = 50.0):
        self.swing_window      = swing_window
        self.round_number_step = round_number_step

    def detect_levels(self, df: pd.DataFrame, atr: float) -> dict:
        """
        Returns a dict with:
          levels        — list of {price, type, strength, touches}
          supports      — levels below current price, sorted nearest first
          resistances   — levels above current price, sorted nearest first
        """
        close   = float(df["Close"].iloc[-1])
        cluster = max(atr * 0.5, 2.0)   # levels within this range get merged

        raw = []

        # 1. Swing highs and lows
        w = self.swing_window
        highs = df["High"]
        lows  = df["Low"]

        for i in range(w, len(df) - w):
            # Swing high: highest in window
            if highs.iloc[i] == highs.iloc[i-w:i+w+1].max():
                raw.append({"price": float(highs.iloc[i]), "type": "resistance", "source": "swing"})
            # Swing low: lowest in window
            if lows.iloc[i] == lows.iloc[i-w:i+w+1].min():
                raw.append({"price": float(lows.iloc[i]), "type": "support", "source": "swing"})

        # 2. Round numbers — XAUUSD respects $50 and $100 levels strongly
        price_range_low  = close - atr * 8
        price_range_high = close + atr * 8
        step = self.round_number_step
        rn = price_range_low - (price_range_low % step)
        while rn <= price_range_high:
            # $100 levels are stronger than $50 levels
            strength = "major" if rn % 100 == 0 else "minor"
            raw.append({"price": round(rn, 2), "type": "both", "source": f"round_{strength}"})
            rn += step

        # 3. Cluster and score levels
        levels = self._cluster(raw, cluster, df, close)

        supports    = sorted([l for l in levels if l["price"] < close],
                              key=lambda x: x["price"], reverse=True)
        resistances = sorted([l for l in levels if l["price"] > close],
                              key=lambda x: x["price"])

        return {
            "levels":      levels,
            "supports":    supports,
            "resistances": resistances,
            "current":     close,
        }

    def _cluster(self, raw: list, cluster_dist: float, df: pd.DataFrame, close: float) -> list:
        """Merge levels within cluster_dist, count touches, assign strength."""
        if not raw:
            return []

        # Sort by price
        raw_sorted = sorted(raw, key=lambda x: x["price"])

        # Merge nearby levels
        merged = []
        group  = [raw_sorted[0]]

        for level in raw_sorted[1:]:
            if level["price"] - group[-1]["price"] <= cluster_dist:
                group.append(level)
            else:
                merged.append(group)
                group = [level]
        merged.append(group)

        results = []
        for group in merged:
            prices   = [l["price"] for l in group]
            avg_price = round(float(np.mean(prices)), 2)
            sources  = [l["source"] for l in group]
            touches  = self._count_touches(df, avg_price, cluster_dist)

            # Strength: major round + many touches = strongest
            has_major_round = any("round_major" in s for s in sources)
            has_minor_round = any("round_minor" in s for s in sources)
            has_swing       = any(s == "swing" for s in sources)

            if has_major_round and touches >= 3:
                strength_score = 5
                strength_label = "Very Strong"
            elif has_major_round or (has_swing and touches >= 3):
                strength_score = 4
                strength_label = "Strong"
            elif has_minor_round and touches >= 2:
                strength_score = 3
                strength_label = "Moderate"
            elif has_swing and touches >= 2:
                strength_score = 3
                strength_label = "Moderate"
            else:
                strength_score = 2
                strength_label = "Weak"

            # Determine type: below = support, above = resistance, both = zone
            if avg_price < close:
                ltype = "support"
            elif avg_price > close:
                ltype = "resistance"
            else:
                ltype = "zone"

            results.append({
                "price":   avg_price,
                "type":    ltype,
                "strength": strength_label,
                "score":   strength_score,
                "touches": touches,
                "sources": list(set(sources)),
            })

        return results

    def _count_touches(self, df: pd.DataFrame, level: float, tolerance: float) -> int:
        """Count how many bars came within tolerance of this level."""
        touches = 0
        for _, row in df.iterrows():
            if (abs(float(row["High"]) - level) <= tolerance or
                abs(float(row["Low"])  - level) <= tolerance or
                abs(float(row["Close"])- level) <= tolerance):
                touches += 1
        return min(touches, 10)   # cap at 10


def analyse_sr_context(
    entry: float,
    atr: float,
    direction: str,
    sr: dict,
    trades: list,
) -> dict:
    """
    Given the S&R data and the proposed trades, return:
      - sr_zone        : AT_SUPPORT | AT_RESISTANCE | IN_RANGE | BREAKOUT
      - sr_context     : human-readable context string
      - wall_warning   : True if a S&R level sits between entry and TP1
      - snapped_tps    : TPs adjusted to sit just before S&R levels
      - adjusted_sl    : SL adjusted to sit just outside nearest S&R
    """
    is_buy       = direction == "BUY"
    supports     = sr["supports"]
    resistances  = sr["resistances"]
    tp1          = trades[0]["tp"] if trades else entry + atr
    tolerance    = atr * 0.3

    # ── Zone classification ───────────────────────────────────────────────────
    nearest_support    = supports[0]    if supports    else None
    nearest_resistance = resistances[0] if resistances else None

    at_support    = nearest_support    and abs(entry - nearest_support["price"])    <= tolerance
    at_resistance = nearest_resistance and abs(entry - nearest_resistance["price"]) <= tolerance

    if at_support and not at_resistance:
        sr_zone = "AT_SUPPORT"
    elif at_resistance and not at_support:
        sr_zone = "AT_RESISTANCE"
    elif at_support and at_resistance:
        sr_zone = "AT_ZONE"        # inside a congestion zone
    else:
        # Check if entry is breaking above resistance or below support
        if (is_buy and nearest_resistance and
            entry > nearest_resistance["price"] - tolerance):
            sr_zone = "BREAKOUT_UP"
        elif (not is_buy and nearest_support and
              entry < nearest_support["price"] + tolerance):
            sr_zone = "BREAKOUT_DOWN"
        else:
            sr_zone = "IN_RANGE"

    # ── Context string ────────────────────────────────────────────────────────
    parts = []
    if nearest_support:
        dist = round(entry - nearest_support["price"], 2)
        parts.append(f"{nearest_support['strength']} support at {nearest_support['price']:.2f} (${dist:.0f} below)")
    if nearest_resistance:
        dist = round(nearest_resistance["price"] - entry, 2)
        parts.append(f"{nearest_resistance['strength']} resistance at {nearest_resistance['price']:.2f} (${dist:.0f} above)")
    sr_context = " | ".join(parts) if parts else "No major S&R levels nearby"

    # ── Wall warning — S&R between entry and TP1 ─────────────────────────────
    wall_warning  = False
    wall_level    = None
    wall_strength = None

    if is_buy:
        # Resistance between entry and TP1 is a wall
        walls = [r for r in resistances
                 if entry < r["price"] < tp1 and r["score"] >= 3]
    else:
        # Support between entry and TP1 is a wall
        walls = [s for s in supports
                 if tp1 < s["price"] < entry and s["score"] >= 3]

    if walls:
        wall_warning  = True
        strongest     = max(walls, key=lambda x: x["score"])
        wall_level    = strongest["price"]
        wall_strength = strongest["strength"]

    # ── Snap TPs to just before S&R levels ───────────────────────────────────
    snapped_tps  = []
    tp_adjusted  = False
    snap_margin  = atr * 0.15   # stop 0.15x ATR before the level

    for t in trades:
        tp_orig = t["tp"]
        tp_new  = tp_orig

        if is_buy:
            # Find any resistance between entry and original TP
            blocking = [r for r in resistances
                        if entry < r["price"] <= tp_orig + snap_margin and r["score"] >= 3]
            if blocking:
                nearest_block = min(blocking, key=lambda x: x["price"])
                tp_new = round(nearest_block["price"] - snap_margin, 2)
                if tp_new > entry:   # only snap if still profitable
                    tp_adjusted = True
        else:
            # Find any support between entry and original TP
            blocking = [s for s in supports
                        if tp_orig - snap_margin <= s["price"] < entry and s["score"] >= 3]
            if blocking:
                nearest_block = max(blocking, key=lambda x: x["price"])
                tp_new = round(nearest_block["price"] + snap_margin, 2)
                if tp_new < entry:
                    tp_adjusted = True

        snapped_tps.append(round(tp_new, 2))

    # ── Adjust SL to sit just outside nearest S&R ────────────────────────────
    sl_margin    = atr * 0.2
    adjusted_sl  = None

    if is_buy and nearest_support:
        # SL goes just below nearest support
        candidate = round(nearest_support["price"] - sl_margin, 2)
        sl_orig   = trades[0]["sl"] if trades else None
        if sl_orig and candidate > entry - atr * 2.5:   # don't widen too much
            adjusted_sl = candidate
    elif not is_buy and nearest_resistance:
        candidate = round(nearest_resistance["price"] + sl_margin, 2)
        sl_orig   = trades[0]["sl"] if trades else None
        if sl_orig and candidate < entry + atr * 2.5:
            adjusted_sl = candidate

    return {
        "sr_zone":         sr_zone,
        "sr_context":      sr_context,
        "wall_warning":    wall_warning,
        "wall_level":      wall_level,
        "wall_strength":   wall_strength,
        "snapped_tps":     snapped_tps,
        "tp_adjusted":     tp_adjusted,
        "adjusted_sl":     adjusted_sl,
        "nearest_support":    nearest_support,
        "nearest_resistance": nearest_resistance,
    }


def sr_zone_advice(sr_zone: str, direction: str) -> str:
    """Human-readable advice for each zone/direction combo."""
    is_buy = direction == "BUY"
    advice = {
        "AT_SUPPORT":    "Buying at support — ideal entry zone ✅" if is_buy
                         else "Selling at support — risky, wait for break ⚠️",
        "AT_RESISTANCE": "Buying at resistance — risky, wait for breakout ⚠️" if is_buy
                         else "Selling at resistance — ideal entry zone ✅",
        "AT_ZONE":       "Price in congestion zone — expect choppy action ⚠️",
        "BREAKOUT_UP":   "Breaking above resistance — strong BUY momentum ✅" if is_buy
                         else "Counter-trend sell into breakout — high risk ❌",
        "BREAKOUT_DOWN": "Breaking below support — strong SELL momentum ✅" if not is_buy
                         else "Counter-trend buy into breakdown — high risk ❌",
        "IN_RANGE":      "Clear space between levels — room to reach TP ✅",
    }
    return advice.get(sr_zone, "")


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SignalEngine:
    def __init__(
        self,
        atr_period: int        = 14,
        tp_count: int          = 3,
        sl_style: str          = "normal",
        lot_size: float        = 0.10,
        trend_threshold: float = 65.0,
        adx_min: float         = 55.0,   # updated: ADX 60 = best PF per user data
        adx_max: float         = 68.0,   # updated: tighter sweet spot
        cooldown_bars: int     = 8,
        use_htf_filter: bool   = True,
        use_sr: bool           = True,   # NEW: enable S&R engine
        sr_swing_window: int   = 10,     # bars to look back for swing points
        sr_snap_tps: bool      = True,   # snap TPs to S&R levels
        sr_filter_walls: bool  = True,   # skip if strong wall before TP1
    ):
        self.atr_period      = atr_period
        self.tp_count        = tp_count
        self.sl_style        = sl_style
        self.lot_size        = lot_size
        self.trend_threshold = trend_threshold
        self.adx_min         = adx_min
        self.adx_max         = adx_max
        self.cooldown_bars   = cooldown_bars
        self.use_htf_filter  = use_htf_filter
        self.use_sr          = use_sr
        self.sr_snap_tps     = sr_snap_tps
        self.sr_filter_walls = sr_filter_walls
        self.sr_engine       = SREngine(swing_window=sr_swing_window)

    # ── fetch ─────────────────────────────────────────────────────────────────
    def fetch_data(self, period="15d", interval="1h"):
        try:
            ticker = yf.Ticker("GC=F")
            df = ticker.history(period=period, interval=interval)
            if df.empty or len(df) < 60:
                log.warning("Not enough 1h data.")
                return None
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception as e:
            log.error(f"1h fetch: {e}")
            return None

    def fetch_htf(self):
        try:
            df = yf.Ticker("GC=F").history(period="60d", interval="4h")
            if df.empty or len(df) < 50: return None
            df = df[["Open", "High", "Low", "Close"]].dropna()
            df["ema50"]  = df["Close"].ewm(span=50,  adjust=False).mean()
            df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
            return df
        except Exception as e:
            log.error(f"4h fetch: {e}")
            return None

    def get_htf_bias(self):
        df = self.fetch_htf()
        if df is None or len(df) < 5: return "NEUTRAL"
        last  = df.iloc[-1]
        slope = df["ema50"].iloc[-1] - df["ema50"].iloc[-4]
        if last["Close"] > last["ema50"] > last["ema200"] and slope > 0: return "BUY"
        if last["Close"] < last["ema50"] < last["ema200"] and slope < 0: return "SELL"
        return "NEUTRAL"

    # ── indicators ────────────────────────────────────────────────────────────
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]

        prev_c = c.shift(1)
        tr = pd.concat([(h-l),(h-prev_c).abs(),(l-prev_c).abs()], axis=1).max(axis=1)
        df["atr"]   = tr.ewm(span=self.atr_period, adjust=False).mean()

        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
        df["rsi"]   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
        df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
        df["ema200"] = c.ewm(span=200, adjust=False).mean()
        df["ema20_slope"] = df["ema20"].diff(3)
        df["ema50_slope"] = df["ema50"].diff(3)

        lo14 = l.rolling(14).min()
        hi14 = h.rolling(14).max()
        df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

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
        df["body"]     = (c - o).abs()
        return df

    # ── scoring ───────────────────────────────────────────────────────────────
    def score_trend(self, row: pd.Series) -> float:
        needed = ["rsi","ema20","ema50","ema200","adx","plus_di","minus_di",
                  "stoch_k","stoch_d","ema20_slope","ema50_slope"]
        if any(pd.isna(row.get(k, np.nan)) for k in needed): return 50.0

        score = 50.0
        rsi   = row["rsi"]
        ema20, ema50, ema200 = row["ema20"], row["ema50"], row["ema200"]
        sk, sd = row["stoch_k"], row["stoch_d"]
        plus_di, minus_di = row["plus_di"], row["minus_di"]
        close = row["Close"]
        s20, s50 = row["ema20_slope"], row["ema50_slope"]

        if rsi > 55:   score += min((rsi-55)*0.64, 16)
        elif rsi < 45: score -= min((45-rsi)*0.64, 16)

        gap = (ema20-ema50) / max(abs(ema50), 1) * 100
        if ema20>ema50:  score += min(gap*3.5, 12)
        elif ema20<ema50: score -= min(-gap*3.5, 12)

        if s20>0 and s50>0:   score += 10
        elif s20<0 and s50<0: score -= 10
        elif s20>0:  score += 5
        elif s20<0: score -= 5

        if close>ema200:  score += 8
        elif close<ema200: score -= 8

        if sk>55 and sk>sd:   score += 6
        elif sk<45 and sk<sd: score -= 6
        if sk>70:  score += 2
        elif sk<30: score -= 2

        if plus_di>minus_di:  score += 6
        elif minus_di>plus_di: score -= 6

        return float(max(0, min(100, score)))

    # ── signal builder ────────────────────────────────────────────────────────
    def build_signal(self, entry, atr, direction, trend_score, adx,
                     htf_bias="NEUTRAL", indicators=None, sr_data=None):
        is_buy   = direction == "BUY"
        sl_mults = SL_MULTIPLIERS[self.sl_style]
        tp_mults = TP_MULTIPLIERS[self.tp_count]
        pip_val  = self.lot_size * 1000 * 0.1
        ind      = indicators or {}

        # ── Build base trades ────────────────────────────────────────────────
        trades = []
        for i in range(self.tp_count):
            sl_m = sl_mults[min(i, len(sl_mults)-1)]
            tp_m = tp_mults[i]
            tp   = entry + atr*tp_m if is_buy else entry - atr*tp_m
            sl   = entry - atr*sl_m if is_buy else entry + atr*sl_m
            tp_d = abs(tp-entry)
            sl_d = abs(sl-entry)
            trades.append({
                "index":   i+1,
                "tp":      round(tp, 2),
                "sl":      round(sl, 2),
                "be":      round(entry, 2),
                "tp_dist": round(tp_d, 2),
                "sl_dist": round(sl_d, 2),
                "rr":      round(tp_d/max(sl_d,0.01), 2),
                "pnl_tp":  round((tp_d/0.1)*pip_val, 2),
                "pnl_sl":  round((sl_d/0.1)*pip_val, 2),
            })

        # ── S&R analysis and TP/SL adjustment ───────────────────────────────
        sr_context_data = {}
        if sr_data and self.use_sr:
            ctx = analyse_sr_context(entry, atr, direction, sr_data, trades)
            sr_context_data = ctx

            # Snap TPs to S&R levels
            if self.sr_snap_tps and ctx["snapped_tps"]:
                for i, t in enumerate(trades):
                    new_tp = ctx["snapped_tps"][i]
                    if abs(new_tp - t["tp"]) > 0.5:   # only apply meaningful snaps
                        tp_d = abs(new_tp - entry)
                        sl_d = abs(t["sl"] - entry)
                        t["tp"]      = new_tp
                        t["tp_dist"] = round(tp_d, 2)
                        t["rr"]      = round(tp_d/max(sl_d,0.01), 2)
                        t["pnl_tp"]  = round((tp_d/0.1)*pip_val, 2)

            # Adjust SL to just outside nearest S&R
            if ctx["adjusted_sl"]:
                new_sl = ctx["adjusted_sl"]
                for t in trades:
                    sl_d = abs(new_sl - entry)
                    t["sl"]      = new_sl
                    t["sl_dist"] = round(sl_d, 2)
                    t["rr"]      = round(t["tp_dist"]/max(sl_d,0.01), 2)
                    t["pnl_sl"]  = round((sl_d/0.1)*pip_val, 2)

        # ── ADX labels — updated for ADX 60 sweet spot ──────────────────────
        adx_val = round(adx, 1)
        if adx_val < 25:
            adx_label  = "Emerging trend ⚠️"
            adx_advice = "Very weak — skip or use limit order only"
        elif adx_val < 40:
            adx_label  = "Moderate trend"
            adx_advice = "Decent — TP1 likely, TP2/3 uncertain"
        elif adx_val < 55:
            adx_label  = "Strong trend 💪"
            adx_advice = "Good — TP2 likely to hit"
        elif adx_val <= 68:
            adx_label  = "Sweet spot ★"         # user's ADX 60 finding
            adx_advice = "Optimal zone — best PF, trust all 3 TPs"
        else:
            adx_label  = "Overextended ⚡"
            adx_advice = "Parabolic move — protect TP1, may reverse fast"

        # ADX slope
        adx_slope = ind.get("adx_slope", 0.0)
        if adx_slope > 0.5:
            adx_slope_label = "Rising ↑ — hold for TP3"
            adx_rising = True
        elif adx_slope < -0.5:
            adx_slope_label = "Falling ↓ — take profit at TP1"
            adx_rising = False
        else:
            adx_slope_label = "Flat → — neutral momentum"
            adx_rising = None

        # RSI
        rsi_val = round(ind.get("rsi", 50.0), 1)
        if rsi_val >= 70:   rsi_label = "Overbought 🔴"
        elif rsi_val >= 55: rsi_label = "Bullish 🟢"
        elif rsi_val <= 30: rsi_label = "Oversold 🔴"
        elif rsi_val <= 45: rsi_label = "Bearish 🔴"
        else:               rsi_label = "Neutral ⚪"

        # Stochastic
        stoch_k = round(ind.get("stoch_k", 50.0), 1)
        stoch_d = round(ind.get("stoch_d", 50.0), 1)
        if stoch_k >= 80:     stoch_label = "Overbought"
        elif stoch_k <= 20:   stoch_label = "Oversold"
        elif stoch_k > stoch_d: stoch_label = "Bullish cross"
        else:                 stoch_label = "Bearish cross"

        # DI
        plus_di  = round(ind.get("plus_di",  0.0), 1)
        minus_di = round(ind.get("minus_di", 0.0), 1)
        di_spread = abs(plus_di - minus_di)
        if di_spread > 15:   di_label = "Strong directional"
        elif di_spread > 8:  di_label = "Clear directional"
        else:                di_label = "Weak directional"

        # Entry guidance — now also uses S&R zone
        sr_zone = sr_context_data.get("sr_zone", "IN_RANGE")
        if adx_val >= 55:
            entry_action = "Buy/Sell Stop at entry — sweet spot, high conviction"
        elif adx_val >= 40:
            entry_action = "Buy/Sell Stop at entry — momentum confirmed"
        else:
            entry_action = "Limit order — wait for entry price to be reached"

        strength = abs(trend_score-50)*2
        s_label  = ("Weak" if strength<30 else "Moderate" if strength<60
                    else "Strong" if strength<85 else "Very Strong")

        # ── S&R summary fields for embed ─────────────────────────────────────
        ns  = sr_context_data.get("nearest_support",    None)
        nr  = sr_context_data.get("nearest_resistance", None)
        sr_zone_adv = sr_zone_advice(sr_zone, direction) if sr_zone else ""

        return {
            # Core
            "direction":       direction,
            "entry":           round(entry, 2),
            "atr":             round(atr, 2),
            "lot_size":        self.lot_size,
            "trades":          trades,
            "total_risk":      round(sum(t["pnl_sl"] for t in trades), 2),
            "total_reward":    round(sum(t["pnl_tp"] for t in trades), 2),
            "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),

            # Trend
            "trend_score":     round(trend_score, 1),
            "strength":        round(strength, 1),
            "strength_label":  s_label,
            "tp_count":        self.tp_count,
            "sl_style":        self.sl_style,

            # ADX
            "adx":             adx_val,
            "adx_label":       adx_label,
            "adx_advice":      adx_advice,
            "adx_slope":       round(adx_slope, 2),
            "adx_slope_label": adx_slope_label,
            "adx_rising":      adx_rising,

            # Indicators
            "rsi":             rsi_val,
            "rsi_label":       rsi_label,
            "stoch_k":         stoch_k,
            "stoch_d":         stoch_d,
            "stoch_label":     stoch_label,
            "plus_di":         plus_di,
            "minus_di":        minus_di,
            "di_label":        di_label,

            # Entry
            "entry_action":    entry_action,
            "htf_bias":        htf_bias,

            # S&R
            "sr_zone":           sr_zone,
            "sr_zone_advice":    sr_zone_adv,
            "sr_context":        sr_context_data.get("sr_context", ""),
            "wall_warning":      sr_context_data.get("wall_warning",  False),
            "wall_level":        sr_context_data.get("wall_level",    None),
            "wall_strength":     sr_context_data.get("wall_strength", None),
            "tp_adjusted":       sr_context_data.get("tp_adjusted",   False),
            "nearest_support":   {"price": ns["price"], "strength": ns["strength"], "touches": ns["touches"]} if ns else None,
            "nearest_resistance":{"price": nr["price"], "strength": nr["strength"], "touches": nr["touches"]} if nr else None,
            "sr_levels":         sr_context_data.get("levels", []) if "levels" in sr_context_data else [],
        }

    # ── main entry ────────────────────────────────────────────────────────────
    def get_signal(self):
        df = self.fetch_data()
        if df is None: return None
        df  = self.compute_indicators(df)
        df["adx_slope"] = df["adx"].diff(3)

        row   = df.iloc[-1]
        adx   = float(row.get("adx", 0) or 0)
        entry = float(row["Close"])
        atr   = float(row["atr"])

        # ADX window gate — updated to sweet spot 55-68
        if not (self.adx_min <= adx <= self.adx_max):
            log.info(f"ADX {adx:.1f} outside [{self.adx_min}-{self.adx_max}] — skip.")
            return None

        # Doji filter
        if float(row["body"]) / max(float(row["atr"]), 0.01) < 0.15:
            log.info("Doji — skip.")
            return None

        score = self.score_trend(row)
        log.info(f"Price:{entry:.2f} ATR:{atr:.2f} ADX:{adx:.1f} Score:{score:.1f}")

        if score >= self.trend_threshold:
            direction = "BUY"
        elif score <= (100 - self.trend_threshold):
            direction = "SELL"
        else:
            log.info("Neutral — skip.")
            return None

        # HTF alignment
        htf_bias = "NEUTRAL"
        if self.use_htf_filter:
            htf_bias = self.get_htf_bias()
            if htf_bias != "NEUTRAL" and htf_bias != direction:
                log.info(f"HTF {htf_bias} conflicts {direction} — skip.")
                return None

        # S&R detection
        sr_data = None
        if self.use_sr:
            try:
                sr_data = self.sr_engine.detect_levels(df, atr)
                sr_data["levels"] = sr_data.get("levels", [])

                # Filter: skip if entry is AT RESISTANCE on a BUY (or AT SUPPORT on SELL)
                ctx_check = analyse_sr_context(entry, atr, direction, sr_data, [])
                if direction == "BUY" and ctx_check["sr_zone"] == "AT_RESISTANCE":
                    log.info("BUY rejected — entry is AT resistance level.")
                    return None
                if direction == "SELL" and ctx_check["sr_zone"] == "AT_SUPPORT":
                    log.info("SELL rejected — entry is AT support level.")
                    return None

                # Filter: skip if strong wall before TP1
                if self.sr_filter_walls and ctx_check["wall_warning"]:
                    if ctx_check["wall_strength"] in ("Very Strong", "Strong"):
                        log.info(f"Signal filtered — strong S&R wall at {ctx_check['wall_level']} before TP1.")
                        return None

                log.info(f"S&R zone: {ctx_check['sr_zone']} | {ctx_check['sr_context'][:60]}")
            except Exception as e:
                log.warning(f"S&R detection failed: {e}")
                sr_data = None

        indicators = {
            "rsi":       float(row.get("rsi",       50.0) or 50.0),
            "stoch_k":   float(row.get("stoch_k",   50.0) or 50.0),
            "stoch_d":   float(row.get("stoch_d",   50.0) or 50.0),
            "plus_di":   float(row.get("plus_di",    0.0) or 0.0),
            "minus_di":  float(row.get("minus_di",   0.0) or 0.0),
            "adx_slope": float(row.get("adx_slope",  0.0) or 0.0),
            "ema20":     float(row.get("ema20",       0.0) or 0.0),
            "ema50":     float(row.get("ema50",       0.0) or 0.0),
        }

        return self.build_signal(entry, atr, direction, score, adx,
                                 htf_bias, indicators, sr_data)
