"""
backtest.py — XAUUSD Signal Bot Backtester v2
──────────────────────────────────────────────
What changed from v1:
  - ADX filter: only enter when market is trending (ADX > adx_threshold)
  - Wider SL:   1.2x–2.0x ATR instead of 0.55x–1.0x
  - Better TP:  1.5x–3.5x ATR for genuine R:R > 1
  - Cooldown:   minimum bars between signals
  - EMA-200:    macro trend gate
  - Doji filter: skip indecision candles
  - v1 vs v2 comparison chart included

Usage:
    python backtest.py
    python backtest.py --period 6mo --sl_style normal --tp_count 3 --threshold 65 --adx 22
    python backtest.py --compare        # side-by-side v1 vs v2
"""

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── v2 constants ──────────────────────────────────────────────────────────────
SL_MULTIPLIERS_V2 = {
    "tight":  [1.0, 1.3, 1.7],
    "normal": [1.2, 1.5, 2.0],
    "wide":   [1.5, 2.0, 2.5],
}
TP_MULTIPLIERS_V2 = {
    1: [2.0],
    2: [1.5, 2.8],
    3: [1.5, 2.5, 3.5],
}

# ── v1 constants (for comparison) ────────────────────────────────────────────
SL_MULTIPLIERS_V1 = {
    "tight":  [0.35, 0.55, 0.80],
    "normal": [0.55, 0.75, 1.00],
    "wide":   [0.75, 1.00, 1.30],
}
TP_MULTIPLIERS_V1 = {
    1: [1.00],
    2: [0.80, 1.80],
    3: [0.70, 1.40, 2.20],
}



# ── v3 constants (data-driven from 303-trade analysis) ───────────────────────
SL_MULTIPLIERS_V3 = {
    "tight":  [1.0, 1.4],
    "normal": [1.2, 1.7],
    "wide":   [1.5, 2.1],
}
TP_MULTIPLIERS_V3 = {
    1: [2.0],
    2: [1.5, 3.2],
    3: [1.5, 2.5, 3.8],
}

# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    signal_index: int
    direction: str
    entry: float
    entry_time: str
    exit_time: str
    trade_num: int
    tp: float
    sl: float
    exit_price: float
    outcome: str       # TP | SL | BE_SL | TIMEOUT
    pnl_pts: float
    pnl_usd: float
    rr_achieved: float
    hold_bars: int
    trend_score: float
    adx: float


@dataclass
class BacktestStats:
    total_signals: int   = 0
    total_trades: int    = 0
    winning_trades: int  = 0
    losing_trades: int   = 0
    be_trades: int       = 0
    timeout_trades: int  = 0
    gross_profit: float  = 0.0
    gross_loss: float    = 0.0
    net_pnl: float       = 0.0
    win_rate: float      = 0.0
    profit_factor: float = 0.0
    avg_win: float       = 0.0
    avg_loss: float      = 0.0
    max_drawdown: float  = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float  = 0.0
    best_trade: float    = 0.0
    worst_trade: float   = 0.0
    avg_hold_bars: float = 0.0
    buy_signals: int     = 0
    sell_signals: int    = 0
    avg_rr: float        = 0.0


# ── indicators ────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]

    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    df["atr"]   = tr.ewm(span=14, adjust=False).mean()

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

    df["body"] = (c - o).abs()
    return df


def score_trend_v2(row: pd.Series) -> float:
    needed = ["rsi", "ema20", "ema50", "ema200", "adx", "plus_di", "minus_di", "stoch_k", "stoch_d"]
    if any(pd.isna(row.get(k, np.nan)) for k in needed):
        return 50.0
    score = 50.0
    rsi, ema20, ema50, ema200 = row["rsi"], row["ema20"], row["ema50"], row["ema200"]
    sk, sd = row["stoch_k"], row["stoch_d"]
    plus_di, minus_di = row["plus_di"], row["minus_di"]
    close = row["Close"]

    if rsi > 55:   score += min((rsi - 55) * 0.72, 18)
    elif rsi < 45: score -= min((45 - rsi) * 0.72, 18)

    ema_gap = (ema20 - ema50) / max(abs(ema50), 1) * 100
    if ema20 > ema50:  score += min(ema_gap * 4, 14)
    elif ema20 < ema50: score -= min(-ema_gap * 4, 14)

    if close > ema200:  score += 8
    elif close < ema200: score -= 8

    if sk > 55 and sk > sd:   score += 7
    elif sk < 45 and sk < sd: score -= 7
    if sk > 70:  score += 3
    elif sk < 30: score -= 3

    if plus_di > minus_di:  score += 10
    elif minus_di > plus_di: score -= 10

    return float(max(0, min(100, score)))


def score_trend_v1(row: pd.Series) -> float:
    needed = ["rsi", "ema20", "ema50", "stoch_k", "stoch_d"]
    if any(pd.isna(row.get(k, np.nan)) for k in needed):
        return 50.0
    score = 50.0
    rsi, ema20, ema50 = row["rsi"], row["ema20"], row["ema50"]
    sk, sd = row["stoch_k"], row["stoch_d"]

    if rsi > 55:   score += min((rsi - 55) * 0.8, 20)
    elif rsi < 45: score -= min((45 - rsi) * 0.8, 20)

    ema_gap = (ema20 - ema50) / ema50 * 100
    if ema20 > ema50:  score += min(ema_gap * 5, 15)
    elif ema20 < ema50: score -= min(-ema_gap * 5, 15)

    if sk > 55 and sk > sd:   score += 10
    elif sk < 45 and sk < sd: score -= 10
    if sk > 70:  score += 5
    elif sk < 30: score -= 5

    return float(max(0, min(100, score)))



def score_trend_v3(row: pd.Series) -> float:
    needed = ["rsi","ema20","ema50","ema200","adx","plus_di","minus_di",
              "stoch_k","stoch_d","ema20_slope","ema50_slope"]
    if any(pd.isna(row.get(k, np.nan)) for k in needed):
        return 50.0
    score = 50.0
    rsi = row["rsi"]
    ema20, ema50, ema200 = row["ema20"], row["ema50"], row["ema200"]
    sk, sd = row["stoch_k"], row["stoch_d"]
    plus_di, minus_di = row["plus_di"], row["minus_di"]
    close = row["Close"]
    ema20_slope = row.get("ema20_slope", 0) or 0
    ema50_slope = row.get("ema50_slope", 0) or 0
    if rsi > 55:   score += min((rsi-55)*0.64, 16)
    elif rsi < 45: score -= min((45-rsi)*0.64, 16)
    ema_gap = (ema20-ema50)/max(abs(ema50),1)*100
    if ema20>ema50:  score += min(ema_gap*3.5, 12)
    elif ema20<ema50: score -= min(-ema_gap*3.5, 12)
    if ema20_slope>0 and ema50_slope>0:   score += 10
    elif ema20_slope<0 and ema50_slope<0: score -= 10
    elif ema20_slope>0:  score += 5
    elif ema20_slope<0: score -= 5
    if close>ema200:  score += 8
    elif close<ema200: score -= 8
    if sk>55 and sk>sd:   score += 6
    elif sk<45 and sk<sd: score -= 6
    if sk>70:  score += 2
    elif sk<30: score -= 2
    if plus_di>minus_di:  score += 6
    elif minus_di>plus_di: score -= 6
    return float(max(0, min(100, score)))

# ── simulation ────────────────────────────────────────────────────────────────

def simulate_trade(
    df: pd.DataFrame,
    signal_bar: int,
    entry: float,
    tp: float,
    sl: float,
    be_trigger: Optional[float],
    is_buy: bool,
    lot_size: float,
    max_bars: int = 300,
) -> tuple:
    pip_val    = lot_size * 1000 * 0.1
    current_sl = sl
    be_moved   = False

    for i in range(signal_bar + 1, min(signal_bar + max_bars, len(df))):
        bar  = df.iloc[i]
        h, l = float(bar["High"]), float(bar["Low"])
        ts   = str(bar.name)[:16]

        # Move SL to breakeven when prior TP triggers
        if be_trigger and not be_moved:
            if (is_buy and h >= be_trigger) or (not is_buy and l <= be_trigger):
                current_sl = entry
                be_moved   = True

        # TP check
        if (is_buy and h >= tp) or (not is_buy and l <= tp):
            pts = abs(tp - entry)
            return "TP", tp, (pts / 0.1) * pip_val, ts, i - signal_bar

        # SL check
        if (is_buy and l <= current_sl) or (not is_buy and h >= current_sl):
            pts = abs(current_sl - entry)
            pnl = -(pts / 0.1) * pip_val if current_sl != entry else 0.0
            return ("BE_SL" if be_moved else "SL"), current_sl, pnl, ts, i - signal_bar

    last   = df.iloc[min(signal_bar + max_bars, len(df) - 1)]
    exit_p = float(last["Close"])
    pts    = (exit_p - entry) if is_buy else (entry - exit_p)
    return "TIMEOUT", exit_p, (pts / 0.1) * pip_val, str(last.name)[:16], max_bars


# ── backtester ────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(
        self,
        version: str           = "v2",     # "v1" | "v2"
        period: str            = "6mo",
        interval: str          = "1h",
        tp_count: int          = 3,
        sl_style: str          = "normal",
        lot_size: float        = 0.10,
        trend_threshold: float = 65.0,
        adx_threshold: float   = 22.0,
        adx_max: float         = 9999.0,
        cooldown_bars: int     = 8,
        initial_balance: float = 10_000.0,
        max_hold_bars: int     = 300,
    ):
        self.version          = version
        self.period           = period
        self.interval         = interval
        self.tp_count         = tp_count
        self.sl_style         = sl_style
        self.lot_size         = lot_size
        self.trend_threshold  = trend_threshold
        self.adx_threshold    = adx_threshold
        self.adx_max          = adx_max
        self.cooldown_bars    = cooldown_bars
        self.initial_balance  = initial_balance
        self.max_hold_bars    = max_hold_bars

        self.trades: list      = []
        self.equity_curve: list= []
        self.signal_log: list  = []

        if version == "v3":
            self.SL = SL_MULTIPLIERS_V3
            self.TP = TP_MULTIPLIERS_V3
            self.score_fn = score_trend_v3
        elif version == "v2":
            self.SL = SL_MULTIPLIERS_V2
            self.TP = TP_MULTIPLIERS_V2
            self.score_fn = score_trend_v2
        else:
            self.SL = SL_MULTIPLIERS_V1
            self.TP = TP_MULTIPLIERS_V1
            self.score_fn = score_trend_v1

    def fetch(self, df_override=None) -> pd.DataFrame:
        if df_override is not None:
            return df_override
        log.info(f"Fetching GC=F | {self.period} | {self.interval}")
        ticker = yf.Ticker("GC=F")
        df = ticker.history(period=self.period, interval=self.interval)
        if df.empty:
            raise ValueError("No data.")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        log.info(f"Got {len(df)} candles.")
        return df

    def run(self, df_override=None) -> BacktestStats:
        df_raw = self.fetch(df_override)
        df     = compute_indicators(df_raw)

        warmup     = 210   # need 200 bars for EMA-200
        balance    = self.initial_balance
        skip_until = 0
        last_signal_bar = -999
        sig_idx    = 0

        sl_mults = self.SL[self.sl_style]
        tp_mults = self.TP[self.tp_count]

        self.equity_curve = [balance] * warmup

        for i in range(warmup, len(df)):
            if i < skip_until:
                self.equity_curve.append(balance)
                continue

            # Cooldown gate
            if (i - last_signal_bar) < self.cooldown_bars:
                self.equity_curve.append(balance)
                continue

            row   = df.iloc[i]
            score = self.score_fn(row)
            adx   = float(row.get("adx", 0) or 0)

            # v2/v3 gates
            if self.version in ("v2", "v3"):
                if adx < self.adx_threshold:
                    self.equity_curve.append(balance)
                    continue
                if self.version == "v3" and hasattr(self, "adx_max") and adx > self.adx_max:
                    self.equity_curve.append(balance)
                    continue
                body_ratio = float(row["body"]) / max(float(row["atr"]), 0.01)
                if body_ratio < 0.15:
                    self.equity_curve.append(balance)
                    continue

            if score >= self.trend_threshold:
                direction = "BUY"
            elif score <= (100 - self.trend_threshold):
                direction = "SELL"
            else:
                self.equity_curve.append(balance)
                continue

            is_buy = direction == "BUY"
            entry  = float(row["Close"])
            atr    = float(row["atr"])
            if pd.isna(atr) or atr <= 0:
                self.equity_curve.append(balance)
                continue

            sig_idx += 1
            last_signal_bar = i
            self.signal_log.append({
                "signal": sig_idx, "dir": direction,
                "bar": i, "entry": entry, "score": score, "adx": adx
            })

            max_exit_bar = i
            signal_pnl   = 0.0

            for t_idx in range(self.tp_count):
                tp = entry + atr * tp_mults[t_idx] if is_buy else entry - atr * tp_mults[t_idx]
                sl = entry - atr * sl_mults[t_idx] if is_buy else entry + atr * sl_mults[t_idx]

                be_trigger = None
                if t_idx > 0:
                    prev_tp = (entry + atr * tp_mults[t_idx-1] if is_buy
                               else entry - atr * tp_mults[t_idx-1])
                    be_trigger = prev_tp

                outcome, exit_p, pnl_usd, exit_ts, hold = simulate_trade(
                    df, i, entry, tp, sl, be_trigger, is_buy,
                    self.lot_size, self.max_hold_bars
                )

                pnl_pts = (exit_p - entry) if is_buy else (entry - exit_p)
                sl_dist = abs(sl - entry)
                rr      = round(abs(pnl_pts) / max(sl_dist, 0.01), 2) if pnl_pts >= 0 else 0.0

                self.trades.append(TradeResult(
                    signal_index = sig_idx,
                    direction    = direction,
                    entry        = round(entry, 2),
                    entry_time   = str(row.name)[:16],
                    exit_time    = exit_ts,
                    trade_num    = t_idx + 1,
                    tp           = round(tp, 2),
                    sl           = round(sl, 2),
                    exit_price   = round(exit_p, 2),
                    outcome      = outcome,
                    pnl_pts      = round(pnl_pts, 2),
                    pnl_usd      = round(pnl_usd, 2),
                    rr_achieved  = rr,
                    hold_bars    = hold,
                    trend_score  = round(score, 1),
                    adx          = round(adx, 1),
                ))

                signal_pnl  += pnl_usd
                max_exit_bar = max(max_exit_bar, i + hold)

            balance    += signal_pnl
            skip_until  = max_exit_bar + 1

            for _ in range(i, min(max_exit_bar + 1, len(df))):
                self.equity_curve.append(balance)

        while len(self.equity_curve) < len(df):
            self.equity_curve.append(self.equity_curve[-1] if self.equity_curve else self.initial_balance)
        self.equity_curve = self.equity_curve[:len(df)]

        return self._stats()

    def _stats(self) -> BacktestStats:
        s = BacktestStats()
        s.total_signals = len(self.signal_log)
        s.total_trades  = len(self.trades)
        s.buy_signals   = sum(1 for x in self.signal_log if x["dir"] == "BUY")
        s.sell_signals  = sum(1 for x in self.signal_log if x["dir"] == "SELL")

        for t in self.trades:
            if t.outcome == "TP" or (t.outcome == "TIMEOUT" and t.pnl_usd > 0):
                s.winning_trades += 1
                s.gross_profit   += t.pnl_usd
            elif t.outcome == "SL" or (t.outcome == "TIMEOUT" and t.pnl_usd <= 0):
                s.losing_trades += 1
                s.gross_loss    += abs(t.pnl_usd)
            elif t.outcome == "BE_SL":
                s.be_trades += 1
            if t.outcome == "TIMEOUT":
                s.timeout_trades += 1

        s.net_pnl        = s.gross_profit - s.gross_loss
        denom            = s.winning_trades + s.losing_trades
        s.win_rate       = s.winning_trades / denom * 100 if denom else 0
        s.profit_factor  = s.gross_profit / s.gross_loss if s.gross_loss else float("inf")
        s.avg_win        = float(np.mean([t.pnl_usd for t in self.trades if t.pnl_usd > 0])) if any(t.pnl_usd > 0 for t in self.trades) else 0
        s.avg_loss       = float(np.mean([t.pnl_usd for t in self.trades if t.pnl_usd < 0])) if any(t.pnl_usd < 0 for t in self.trades) else 0
        s.best_trade     = max((t.pnl_usd for t in self.trades), default=0)
        s.worst_trade    = min((t.pnl_usd for t in self.trades), default=0)
        s.avg_hold_bars  = float(np.mean([t.hold_bars for t in self.trades])) if self.trades else 0
        s.avg_rr         = float(np.mean([t.rr_achieved for t in self.trades if t.rr_achieved > 0])) if self.trades else 0

        eq   = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd   = peak - eq
        s.max_drawdown     = float(dd.max())
        s.max_drawdown_pct = float((dd / np.maximum(peak, 1)).max() * 100)

        pnls = [t.pnl_usd for t in self.trades]
        if len(pnls) > 1:
            mean_r, std_r  = np.mean(pnls), np.std(pnls, ddof=1)
            s.sharpe_ratio = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0

        return s


# ── reporting ─────────────────────────────────────────────────────────────────

def print_report(stats: BacktestStats, cfg: dict, label=""):
    tag = f" [{label}]" if label else ""
    sep = "─" * 54
    print(f"\n{'═'*54}")
    print(f"  XAUUSD BACKTEST REPORT{tag}")
    print(f"  {cfg['period']} | {cfg['interval']} | SL:{cfg['sl_style']} | TP:{cfg['tp_count']}")
    print(f"{'═'*54}")
    print(f"  Signals          : {stats.total_signals:>6}   (BUY {stats.buy_signals} / SELL {stats.sell_signals})")
    print(f"  Total trades     : {stats.total_trades:>6}")
    print(sep)
    print(f"  Win (TP)         : {stats.winning_trades:>6}")
    print(f"  Loss (SL)        : {stats.losing_trades:>6}")
    print(f"  Breakeven BE-SL  : {stats.be_trades:>6}")
    print(sep)
    print(f"  Win rate         : {stats.win_rate:>6.1f}%")
    print(f"  Profit factor    : {stats.profit_factor:>7.2f}")
    print(f"  Sharpe ratio     : {stats.sharpe_ratio:>7.2f}")
    print(f"  Avg R:R achieved : {stats.avg_rr:>7.2f}")
    print(sep)
    print(f"  Net P&L          : ${stats.net_pnl:>10.2f}")
    print(f"  Gross profit     : ${stats.gross_profit:>10.2f}")
    print(f"  Gross loss       : ${stats.gross_loss:>10.2f}")
    print(sep)
    print(f"  Avg win          : ${stats.avg_win:>10.2f}")
    print(f"  Avg loss         : ${stats.avg_loss:>10.2f}")
    print(f"  Best trade       : ${stats.best_trade:>10.2f}")
    print(f"  Worst trade      : ${stats.worst_trade:>10.2f}")
    print(sep)
    print(f"  Max drawdown     : ${stats.max_drawdown:>10.2f}  ({stats.max_drawdown_pct:.1f}%)")
    print(f"  Avg hold (bars)  : {stats.avg_hold_bars:>6.1f}")
    print(f"{'═'*54}\n")


def save_trade_log(trades, path="backtest_trades.csv"):
    pd.DataFrame([asdict(t) for t in trades]).to_csv(path, index=False)
    log.info(f"Trade log → {path}")


def save_charts(bt: Backtester, stats: BacktestStats, cfg: dict, path="backtest_report.png"):
    trades = bt.trades
    equity = bt.equity_curve

    BG, BG2 = "#0F0F0F", "#161616"
    GOLD, GREEN, RED, AMBER, MUTED = "#C9A84C", "#1D9E75", "#D85A30", "#EF9F27", "#444444"
    TEXT = "#CCCCCC"

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle(
        f"XAUUSD Backtest v2 — {cfg['period']} | {cfg['interval']} | "
        f"SL: {cfg['sl_style']} | TP: {cfg['tp_count']} | "
        f"Threshold: {cfg.get('trend_threshold',65)} | ADX: {cfg.get('adx_threshold',22)}",
        color=GOLD, fontsize=13, fontweight="bold", y=0.99
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    ax_eq   = fig.add_subplot(gs[0, :])    # equity — full width
    ax_pie  = fig.add_subplot(gs[1, 0])    # pie
    ax_pnl  = fig.add_subplot(gs[1, 1:])   # P&L per signal
    ax_slot = fig.add_subplot(gs[2, 0])    # outcomes by slot
    ax_hold = fig.add_subplot(gs[2, 1])    # hold hist
    ax_kpi  = fig.add_subplot(gs[2, 2])    # KPI scorecard

    for ax in [ax_eq, ax_pie, ax_pnl, ax_slot, ax_hold, ax_kpi]:
        ax.set_facecolor(BG2)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(MUTED)

    # ── equity curve ──────────────────────────────────────────────────────────
    eq_arr = np.array(equity)
    peak   = np.maximum.accumulate(eq_arr)
    ax_eq.plot(eq_arr, color=GOLD, lw=1.5, label="Equity")
    ax_eq.fill_between(range(len(eq_arr)), eq_arr, peak, alpha=0.3, color=RED, label="Drawdown")
    ax_eq.axhline(bt.initial_balance, color=MUTED, lw=0.8, ls="--", label="Start")

    # Mark each signal on equity curve
    for sig in bt.signal_log:
        bar = sig["bar"]
        if bar < len(eq_arr):
            col = GREEN if sig["dir"] == "BUY" else RED
            ax_eq.axvline(bar, color=col, alpha=0.12, lw=0.7)

    ax_eq.set_title("Equity curve  (green/red lines = signal entries)", color=TEXT, fontsize=10)
    ax_eq.set_ylabel("Balance ($)", color=TEXT, fontsize=8)
    ax_eq.set_xlabel("Bars", color=TEXT, fontsize=8)
    ax_eq.legend(fontsize=8, facecolor=BG2, labelcolor=TEXT, framealpha=0.5)
    pnl_c = GREEN if stats.net_pnl >= 0 else RED
    ax_eq.annotate(
        f"Net P&L: ${stats.net_pnl:+,.2f}  |  Win rate: {stats.win_rate:.1f}%  |  "
        f"PF: {stats.profit_factor:.2f}  |  Sharpe: {stats.sharpe_ratio:.2f}  |  "
        f"Signals: {stats.total_signals}",
        xy=(0.01, 0.04), xycoords="axes fraction", color=pnl_c, fontsize=9, fontweight="bold"
    )

    # ── pie ───────────────────────────────────────────────────────────────────
    labels  = ["Win (TP)", "Loss (SL)", "Breakeven", "Timeout"]
    sizes   = [stats.winning_trades, stats.losing_trades, stats.be_trades, stats.timeout_trades]
    colors  = [GREEN, RED, AMBER, MUTED]
    nz      = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if nz:
        s2, l2, c2 = zip(*nz)
        ax_pie.pie(s2, labels=l2, colors=c2, autopct="%1.0f%%",
                   textprops={"color": TEXT, "fontsize": 8},
                   pctdistance=0.78, startangle=90)
    ax_pie.set_title("Trade outcomes", color=TEXT, fontsize=10)

    # ── P&L per signal bar ────────────────────────────────────────────────────
    sig_pnl: dict = {}
    for t in trades:
        sig_pnl[t.signal_index] = sig_pnl.get(t.signal_index, 0) + t.pnl_usd
    ids, vals = list(sig_pnl.keys()), list(sig_pnl.values())
    bcols = [GREEN if v >= 0 else RED for v in vals]
    ax_pnl.bar(ids, vals, color=bcols, width=0.7, alpha=0.85)
    ax_pnl.axhline(0, color=MUTED, lw=0.8)
    ax_pnl.set_title("P&L per signal", color=TEXT, fontsize=10)
    ax_pnl.set_xlabel("Signal #", color=TEXT, fontsize=8)
    ax_pnl.set_ylabel("USD", color=TEXT, fontsize=8)

    # ── outcomes by trade slot ────────────────────────────────────────────────
    tp_count = cfg.get("tp_count", 3)
    slot_colors = {"TP": GREEN, "SL": RED, "BE_SL": AMBER, "TIMEOUT": MUTED}
    for t_num in range(1, tp_count + 1):
        sub    = [t for t in trades if t.trade_num == t_num]
        counts = {k: sum(1 for t in sub if t.outcome == k) for k in slot_colors}
        bottom = 0
        for k, col in slot_colors.items():
            v = counts[k]
            if v:
                ax_slot.bar(t_num - 1, v, bottom=bottom, color=col, width=0.5, alpha=0.85)
                bottom += v
    ax_slot.set_xticks(range(tp_count))
    ax_slot.set_xticklabels([f"Trade {i+1}" for i in range(tp_count)], color=TEXT, fontsize=8)
    ax_slot.set_title("Outcomes by trade slot", color=TEXT, fontsize=10)
    ax_slot.set_ylabel("Count", color=TEXT, fontsize=8)
    patches = [Patch(color=c, label=l) for l, c in slot_colors.items()]
    ax_slot.legend(handles=patches, fontsize=7, facecolor=BG2, labelcolor=TEXT, framealpha=0.5)

    # ── hold histogram ────────────────────────────────────────────────────────
    holds = [t.hold_bars for t in trades if t.hold_bars < 300]
    if holds:
        ax_hold.hist(holds, bins=25, color=GOLD, alpha=0.85, edgecolor=BG)
    ax_hold.set_title("Hold duration (bars)", color=TEXT, fontsize=10)
    ax_hold.set_xlabel("Bars", color=TEXT, fontsize=8)
    ax_hold.set_ylabel("Count", color=TEXT, fontsize=8)

    # ── KPI scorecard ─────────────────────────────────────────────────────────
    ax_kpi.axis("off")
    kpis = [
        ("Signals",        f"{stats.total_signals}"),
        ("Win rate",       f"{stats.win_rate:.1f}%"),
        ("Profit factor",  f"{stats.profit_factor:.2f}"),
        ("Sharpe",         f"{stats.sharpe_ratio:.2f}"),
        ("Avg R:R",        f"1:{stats.avg_rr:.2f}"),
        ("Net P&L",        f"${stats.net_pnl:+,.0f}"),
        ("Max DD",         f"${stats.max_drawdown:,.0f}"),
        ("Avg hold",       f"{stats.avg_hold_bars:.1f} bars"),
        ("Best trade",     f"${stats.best_trade:,.0f}"),
        ("Worst trade",    f"${stats.worst_trade:,.0f}"),
    ]
    ax_kpi.set_title("Key metrics", color=TEXT, fontsize=10)
    for idx, (label, val) in enumerate(kpis):
        y = 0.92 - idx * 0.092
        ax_kpi.text(0.02, y, label, transform=ax_kpi.transAxes,
                    color=MUTED, fontsize=9, va="top")
        color = TEXT
        if "P&L" in label:   color = GREEN if stats.net_pnl >= 0 else RED
        if "Win" in label:   color = GREEN if stats.win_rate >= 45 else (AMBER if stats.win_rate >= 38 else RED)
        if "Profit" in label: color = GREEN if stats.profit_factor >= 1 else RED
        ax_kpi.text(0.98, y, val, transform=ax_kpi.transAxes,
                    color=color, fontsize=9, va="top", ha="right", fontweight="bold")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"Chart → {path}")


def save_comparison_chart(s1: BacktestStats, s2: BacktestStats, eq1, eq2, path="backtest_comparison.png"):
    """Side-by-side equity curve + KPI table: v1 vs v2."""
    BG, BG2 = "#0F0F0F", "#161616"
    GOLD, GREEN, RED, AMBER, MUTED, TEXT = "#C9A84C","#1D9E75","#D85A30","#EF9F27","#444","#CCCCCC"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle("v1 (original) vs v2 (fixed) — Equity Curve Comparison", color=GOLD, fontsize=13, fontweight="bold")

    for ax, eq, stats, label, col in [
        (axes[0], eq1, s1, "v1 — Original", RED),
        (axes[1], eq2, s2, "v2 — Fixed",    GREEN),
    ]:
        ax.set_facecolor(BG2)
        eq_arr = np.array(eq)
        peak   = np.maximum.accumulate(eq_arr)
        ax.plot(eq_arr, color=col, lw=1.5)
        ax.fill_between(range(len(eq_arr)), eq_arr, peak, alpha=0.25, color=RED)
        ax.axhline(10000, color=MUTED, lw=0.8, ls="--")
        ax.set_title(label, color=TEXT, fontsize=11)
        ax.set_ylabel("Balance ($)", color=TEXT, fontsize=9)
        ax.set_xlabel("Bars", color=TEXT, fontsize=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(MUTED)

        summary = (
            f"Signals: {stats.total_signals}  |  Win: {stats.win_rate:.1f}%  |  "
            f"PF: {stats.profit_factor:.2f}\n"
            f"Net P&L: ${stats.net_pnl:+,.0f}  |  Max DD: ${stats.max_drawdown:,.0f}"
        )
        c = GREEN if stats.net_pnl >= 0 else RED
        ax.annotate(summary, xy=(0.02, 0.05), xycoords="axes fraction",
                    color=c, fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG2, alpha=0.7))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"Comparison chart → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--period",    default="6mo")
    p.add_argument("--interval",  default="1h")
    p.add_argument("--tp_count",  default=3,    type=int, choices=[1,2,3])
    p.add_argument("--sl_style",  default="normal", choices=["tight","normal","wide"])
    p.add_argument("--lot_size",  default=0.10, type=float)
    p.add_argument("--threshold", default=65.0, type=float)
    p.add_argument("--adx",       default=22.0, type=float)
    p.add_argument("--cooldown",  default=8,    type=int)
    p.add_argument("--balance",   default=10000.0, type=float)
    p.add_argument("--max_hold",  default=300,  type=int)
    p.add_argument("--out_dir",   default=".")
    p.add_argument("--compare",   action="store_true", help="Run v1 vs v2 comparison")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = dict(
        period=args.period, interval=args.interval, tp_count=args.tp_count,
        sl_style=args.sl_style, lot_size=args.lot_size, trend_threshold=args.threshold,
        adx_threshold=args.adx, initial_balance=args.balance,
    )

    if args.compare:
        log.info("Running v1 vs v2 comparison...")
        # Fetch once, share data
        ticker = yf.Ticker("GC=F")
        raw    = ticker.history(period=args.period, interval=args.interval)
        raw    = raw[["Open","High","Low","Close","Volume"]].dropna()

        bt1 = Backtester(version="v1", trend_threshold=60.0, adx_threshold=0, cooldown_bars=0,
                         tp_count=args.tp_count, sl_style=args.sl_style, lot_size=args.lot_size,
                         initial_balance=args.balance, max_hold_bars=args.max_hold)
        s1  = bt1.run(raw.copy())

        bt2 = Backtester(version="v2", trend_threshold=args.threshold, adx_threshold=args.adx,
                         cooldown_bars=args.cooldown, tp_count=args.tp_count, sl_style=args.sl_style,
                         lot_size=args.lot_size, initial_balance=args.balance, max_hold_bars=args.max_hold)
        s2  = bt2.run(raw.copy())

        print_report(s1, cfg, "v1 ORIGINAL")
        print_report(s2, cfg, "v2 FIXED")
        save_comparison_chart(s1, s2, bt1.equity_curve, bt2.equity_curve,
                              os.path.join(args.out_dir, "backtest_comparison.png"))
        save_charts(bt2, s2, cfg, os.path.join(args.out_dir, "backtest_report.png"))
        save_trade_log(bt2.trades, os.path.join(args.out_dir, "backtest_trades.csv"))

    else:
        bt = Backtester(version="v2", trend_threshold=args.threshold, adx_threshold=args.adx,
                        cooldown_bars=args.cooldown, tp_count=args.tp_count, sl_style=args.sl_style,
                        lot_size=args.lot_size, initial_balance=args.balance, max_hold_bars=args.max_hold)
        stats = bt.run()
        print_report(stats, cfg)
        save_charts(bt, stats, cfg, os.path.join(args.out_dir, "backtest_report.png"))
        save_trade_log(bt.trades, os.path.join(args.out_dir, "backtest_trades.csv"))

        with open(os.path.join(args.out_dir, "backtest_stats.json"), "w") as f:
            json.dump({**asdict(stats), "config": cfg,
                       "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)

    log.info("Done.")


if __name__ == "__main__":
    main()
