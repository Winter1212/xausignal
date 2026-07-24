"""
Gold Signal Terminal — Free polling bot (no TradingView subscription needed)

This is a 1:1 re-implementation of the Pine Script v6 indicator
"Gold Signal Terminal — Entry/SL/TP1-3 + Winrate + Supertrend":

  - 9/21 EMA crossover entries
  - RSI(14) filter (blocks buys above 70, sells below 30)
  - Supertrend(10, 3.0) trend filter — a signal only fires WITH the
    Supertrend direction (this replaces any multi-timeframe filter; the
    indicator only ever looks at the ONE chart timeframe, so this bot only
    ever polls ONE timeframe too)
  - ATR(14) x 1.5 stop distance, clamped into a fixed [10, 12] point band
  - TP1 = 1R, TP2 = 2R, TP3 = 3R
  - Stop loss NEVER moves to breakeven. It stays fixed at its original
    level for the life of the trade. The ONLY thing that can move it is the
    optional "trail runner with Supertrend after TP2" behavior, which only
    ever ratchets the stop in the trade's favor.
  - Three selectable Position/P&L modes, exactly matching the indicator:
      "partial"   -> Partial Closes (Cascade TP1 -> TP2 -> TP3)  [default]
      "tp1_only"  -> Full Size @ TP1 Only
      "first_hit" -> Full Size @ Whichever TP Hit First

Unlike the indicator (which only draws on a chart), this bot tracks an OPEN
POSITION across polls (in state.json) and sends a Telegram alert for every
event: entry, TP1 partial, TP2 partial, runner trail stop, and final close.

Data source: Twelve Data free API (https://twelvedata.com — free signup, no card required)
Alerts: sent directly to Telegram via the Bot API

Deploy as a Render free Web Service. Since Render free web services sleep
when idle, use a free external pinger (e.g. https://cron-job.org) to hit the
/check endpoint every 5 minutes (or however often TIMEFRAME closes) to keep
it awake and checked on schedule.
"""

import os
import json
import requests
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# ---------------------- CONFIG (env vars, set these in Render) ----------------------
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")
TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID", "")
SYMBOL               = os.environ.get("SYMBOL", "XAU/USD")
TIMEFRAME            = os.environ.get("TIMEFRAME", "5min")  # the ONE chart timeframe, matches the indicator

# ---------------------- SIGNAL ENGINE PARAMETERS (exact indicator defaults) ----------------------
FAST_LEN = 12
SLOW_LEN = 35
USE_RSI = True
RSI_LEN = 15
RSI_OB = 70   # block buys above this
RSI_OS = 30   # block sells below this

# ---------------------- SUPERTREND TREND FILTER (exact indicator defaults) ----------------------
ST_ATR_PERIOD = 15
ST_FACTOR = 5.0

# ---------------------- RISK MANAGEMENT (exact indicator defaults) ----------------------
ATR_LEN = 12
SL_MULT = 1
SL_MIN_PTS = 10
SL_MAX_PTS = 12
RR1, RR2, RR3 = 1.8, 2.5, 3.0

TP1_CLOSE_PCT = 50          # % of ORIGINAL position closed at TP1 (Partial mode only)
TP2_CUMULATIVE_PCT = 75     # cumulative % of ORIGINAL position closed by TP2 (Partial mode only)

# "partial" | "tp1_only" | "first_hit"  (matches the indicator's pnlMode dropdown)
PNL_MODE = os.environ.get("PNL_MODE", "partial")

# ---------------------- RUNNER MANAGEMENT (exact indicator default) ----------------------
USE_TRAILING_RUNNER = True  # trail SL to Supertrend after TP1+TP2 booked (Partial mode only)

# ---------------------- POSITION SIZING (exact indicator defaults) ----------------------
LOT_SIZE = float(os.environ.get("LOT_SIZE", 0.01))
UNITS_PER_LOT = float(os.environ.get("UNITS_PER_LOT", 100))

STATE_FILE = "state.json"


# ---------------------- DATA FETCH ----------------------
def fetch_candles(interval, outputsize=200):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "order": "ASC",
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error for {interval}: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ---------------------- INDICATORS ----------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def true_range(df):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    return pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df, length):
    return true_range(df).ewm(alpha=1 / length, adjust=False).mean()


def supertrend(df, period, factor):
    """
    Classic Supertrend, mirrors Pine's ta.supertrend(factor, period).
    Returns (supertrend_series, direction_series) where direction == 1
    means bullish/uptrend (Pine's stDirection < 0) and direction == -1
    means bearish/downtrend (Pine's stDirection > 0).
    """
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    upperband = hl2 + factor * atr_val
    lowerband = hl2 - factor * atr_val

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    direction = pd.Series(index=df.index, dtype="int64")
    st = pd.Series(index=df.index, dtype="float64")

    for i in range(len(df)):
        if i == 0:
            final_upper.iloc[i] = upperband.iloc[i]
            final_lower.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
            st.iloc[i] = final_lower.iloc[i]
            continue

        final_upper.iloc[i] = (
            upperband.iloc[i]
            if (upperband.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1])
            else final_upper.iloc[i - 1]
        )
        final_lower.iloc[i] = (
            lowerband.iloc[i]
            if (lowerband.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1])
            else final_lower.iloc[i - 1]
        )

        if df["close"].iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return st, direction


def trend_arrow(d):
    return "▲" if d == 1 else "▼" if d == -1 else "→"


# ---------------------- STATE ----------------------
DEFAULT_STATE = {
    "last_signal_bar": None,
    "last_exit_bar": None,
    "position": None,
    "history": [],
    "stats": {
        "total_trades": 0, "wins": 0, "losses": 0, "sum_pnl": 0.0,
        "best_trade": None, "worst_trade": None,
    },
}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        for k, v in DEFAULT_STATE.items():
            state.setdefault(k, v)
        return state
    return json.loads(json.dumps(DEFAULT_STATE))


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------- TELEGRAM ----------------------
def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    return resp.json()


# ---------------------- TRADE LOG ----------------------
def log_trade(state, side, entry, sl, tp1, tp2, tp3, exit_price, result, points, pnl):
    state["history"].insert(0, {
        "side": side, "entry": entry, "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "exit": exit_price,
        "result": result, "points": round(points, 2), "pnl": round(pnl, 2),
    })
    state["history"] = state["history"][:50]
    state["stats"]["sum_pnl"] += pnl


def settle_trade(state, pos, exit_price, result_label):
    """Fully closes whatever size remains and settles win/loss stats.
    A trade counts as a win overall if this final leg was positive, OR if an
    earlier TP1/TP2 partial already locked in profit — same rule as the
    indicator's closeTrade()."""
    points = (exit_price - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - exit_price)
    pnl = points * LOT_SIZE * pos["remaining_size"] * UNITS_PER_LOT

    log_trade(state, "BUY" if pos["dir"] == 1 else "SELL",
              pos["entry"], pos["sl"], pos["tp1"], pos["tp2"], pos["tp3"],
              exit_price, result_label, points, pnl)

    stats = state["stats"]
    stats["total_trades"] += 1
    combined_positive = pnl > 0 or pos["tp1_hit"] or pos["tp2_hit"]
    if combined_positive:
        stats["wins"] += 1
    else:
        stats["losses"] += 1
    stats["best_trade"] = pnl if stats["best_trade"] is None else max(stats["best_trade"], pnl)
    stats["worst_trade"] = pnl if stats["worst_trade"] is None else min(stats["worst_trade"], pnl)

    state["position"] = None
    return pnl


def partial_close_tp1(state, pos):
    """Books TP1_CLOSE_PCT% of the ORIGINAL size at TP1.
    IMPORTANT: unlike a breakeven bot, the SL is left untouched here — the
    indicator explicitly removed the breakeven cascade."""
    close_frac = TP1_CLOSE_PCT / 100.0
    points = (pos["tp1"] - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - pos["tp1"])
    pnl = points * LOT_SIZE * close_frac * UNITS_PER_LOT

    log_trade(state, "BUY" if pos["dir"] == 1 else "SELL",
              pos["entry"], pos["sl"], pos["tp1"], pos["tp2"], pos["tp3"],
              pos["tp1"], f"TP1 Hit ({TP1_CLOSE_PCT}% Partial)", points, pnl)

    pos["tp1_hit"] = True
    pos["remaining_size"] = round(1.0 - close_frac, 6)
    return pnl


def partial_close_tp2(state, pos):
    """Books enough size at TP2 so TP2_CUMULATIVE_PCT% of the original
    position is closed in total. SL again left untouched."""
    target_remaining = 1.0 - (TP2_CUMULATIVE_PCT / 100.0)
    close_frac = max(pos["remaining_size"] - target_remaining, 0.0)
    points = (pos["tp2"] - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - pos["tp2"])
    pnl = points * LOT_SIZE * close_frac * UNITS_PER_LOT

    log_trade(state, "BUY" if pos["dir"] == 1 else "SELL",
              pos["entry"], pos["sl"], pos["tp1"], pos["tp2"], pos["tp3"],
              pos["tp2"], f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", points, pnl)

    pos["tp2_hit"] = True
    pos["remaining_size"] = round(target_remaining, 6)
    return pnl


def trail_runner_sl(pos, st_value):
    """After TP1+TP2 are booked, ratchet the SL to the current Supertrend
    value — but only in the trade's favor, exactly like trailRunnerSL()."""
    if st_value is None or pd.isna(st_value):
        return
    if pos["dir"] == 1 and st_value > pos["sl"]:
        pos["sl"] = st_value
    elif pos["dir"] == -1 and st_value < pos["sl"]:
        pos["sl"] = st_value


# ---------------------- POSITION MANAGEMENT (mirrors the Pine if-cascade) ----------------------
def manage_position(state, last_candle, st_value):
    pos = state["position"]
    if pos is None:
        return False

    high, low = last_candle["high"], last_candle["low"]
    events = []

    def buy_side():
        if PNL_MODE == "tp1_only":
            if low <= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                events.append(("SL Hit", pos["sl"], pnl))
            elif high >= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        elif PNL_MODE == "first_hit":
            if low <= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                events.append(("SL Hit", pos["sl"], pnl))
            elif high >= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Full Close)")
                events.append(("TP3 Hit (Full Close)", pos["tp3"], pnl))
            elif high >= pos["tp2"]:
                pnl = settle_trade(state, pos, pos["tp2"], "TP2 Hit (Full Close)")
                events.append(("TP2 Hit (Full Close)", pos["tp2"], pnl))
            elif high >= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        else:  # partial
            if not pos["tp1_hit"]:
                if low <= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                    events.append(("SL Hit", pos["sl"], pnl))
                elif high >= pos["tp3"]:
                    pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Gap)")
                    events.append(("TP3 Hit (Gap)", pos["tp3"], pnl))
                elif high >= pos["tp2"]:
                    p1 = partial_close_tp1(state, pos)
                    events.append((f"TP1 Hit ({TP1_CLOSE_PCT}% Partial)", pos["tp1"], p1))
                    p2 = partial_close_tp2(state, pos)
                    events.append((f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", pos["tp2"], p2))
                elif high >= pos["tp1"]:
                    p1 = partial_close_tp1(state, pos)
                    events.append((f"TP1 Hit ({TP1_CLOSE_PCT}% Partial)", pos["tp1"], p1))
            elif pos["tp1_hit"] and not pos["tp2_hit"]:
                if low <= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit (After TP1 Partial)")
                    events.append(("SL Hit (After TP1 Partial)", pos["sl"], pnl))
                elif high >= pos["tp3"]:
                    pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Gap)")
                    events.append(("TP3 Hit (Gap)", pos["tp3"], pnl))
                elif high >= pos["tp2"]:
                    p2 = partial_close_tp2(state, pos)
                    events.append((f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", pos["tp2"], p2))
            else:
                if USE_TRAILING_RUNNER:
                    trail_runner_sl(pos, st_value)
                    if low <= pos["sl"]:
                        pnl = settle_trade(state, pos, pos["sl"], "Runner Stopped (Supertrend Trail)")
                        events.append(("Runner Stopped (Supertrend Trail)", pos["sl"], pnl))
                else:
                    if low <= pos["sl"]:
                        pnl = settle_trade(state, pos, pos["sl"], "SL Hit (After TP1+TP2 Partials)")
                        events.append(("SL Hit (After TP1+TP2 Partials)", pos["sl"], pnl))
                    elif high >= pos["tp3"]:
                        pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Runner)")
                        events.append(("TP3 Hit (Runner)", pos["tp3"], pnl))

    def sell_side():
        if PNL_MODE == "tp1_only":
            if high >= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                events.append(("SL Hit", pos["sl"], pnl))
            elif low <= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        elif PNL_MODE == "first_hit":
            if high >= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                events.append(("SL Hit", pos["sl"], pnl))
            elif low <= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Full Close)")
                events.append(("TP3 Hit (Full Close)", pos["tp3"], pnl))
            elif low <= pos["tp2"]:
                pnl = settle_trade(state, pos, pos["tp2"], "TP2 Hit (Full Close)")
                events.append(("TP2 Hit (Full Close)", pos["tp2"], pnl))
            elif low <= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        else:  # partial
            if not pos["tp1_hit"]:
                if high >= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                    events.append(("SL Hit", pos["sl"], pnl))
                elif low <= pos["tp3"]:
                    pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Gap)")
                    events.append(("TP3 Hit (Gap)", pos["tp3"], pnl))
                elif low <= pos["tp2"]:
                    p1 = partial_close_tp1(state, pos)
                    events.append((f"TP1 Hit ({TP1_CLOSE_PCT}% Partial)", pos["tp1"], p1))
                    p2 = partial_close_tp2(state, pos)
                    events.append((f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", pos["tp2"], p2))
                elif low <= pos["tp1"]:
                    p1 = partial_close_tp1(state, pos)
                    events.append((f"TP1 Hit ({TP1_CLOSE_PCT}% Partial)", pos["tp1"], p1))
            elif pos["tp1_hit"] and not pos["tp2_hit"]:
                if high >= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit (After TP1 Partial)")
                    events.append(("SL Hit (After TP1 Partial)", pos["sl"], pnl))
                elif low <= pos["tp3"]:
                    pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Gap)")
                    events.append(("TP3 Hit (Gap)", pos["tp3"], pnl))
                elif low <= pos["tp2"]:
                    p2 = partial_close_tp2(state, pos)
                    events.append((f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", pos["tp2"], p2))
            else:
                if USE_TRAILING_RUNNER:
                    trail_runner_sl(pos, st_value)
                    if high >= pos["sl"]:
                        pnl = settle_trade(state, pos, pos["sl"], "Runner Stopped (Supertrend Trail)")
                        events.append(("Runner Stopped (Supertrend Trail)", pos["sl"], pnl))
                else:
                    if high >= pos["sl"]:
                        pnl = settle_trade(state, pos, pos["sl"], "SL Hit (After TP1+TP2 Partials)")
                        events.append(("SL Hit (After TP1+TP2 Partials)", pos["sl"], pnl))
                    elif low <= pos["tp3"]:
                        pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Runner)")
                        events.append(("TP3 Hit (Runner)", pos["tp3"], pnl))

    if pos["dir"] == 1:
        buy_side()
    else:
        sell_side()

    side = "BUY" if pos["dir"] == 1 else "SELL"
    for label, price, pnl in events:
        msg = (
            f"XAUUSD {side} — {label}\n"
            f"Price: {price:.2f}\n"
            f"P&L (this leg): {'+' if pnl >= 0 else ''}${pnl:.2f}"
        )
        send_telegram(msg)

    return len(events) > 0


def open_position(side, entry, sl, tp1, tp2, tp3, bar_time):
    return {
        "dir": 1 if side == "BUY" else -1,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "entry_time": bar_time,
        "tp1_hit": False, "tp2_hit": False,
        "remaining_size": 1.0,
    }


# ---------------------- CORE CHECK ----------------------
@app.route("/check", methods=["GET"])
def check():
    if not TWELVE_DATA_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    df = fetch_candles(TIMEFRAME, outputsize=200)
    df["emaFast"] = ema(df["close"], FAST_LEN)
    df["emaSlow"] = ema(df["close"], SLOW_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)
    st_series, dir_series = supertrend(df, ST_ATR_PERIOD, ST_FACTOR)
    df["st"] = st_series
    df["st_dir"] = dir_series

    last = df.iloc[-1]
    bar_time = str(last["datetime"])

    state = load_state()
    result = {"bar_time": bar_time, "event": None}

    # 1) Manage an already-open position against this bar's high/low
    if state["position"] is not None and state.get("last_exit_bar") != bar_time:
        acted = manage_position(state, last, last["st"])
        state["last_exit_bar"] = bar_time
        if acted:
            result["event"] = "position_update"
        save_state(state)

    # 2) Only look for a NEW entry if we're currently flat
    if state["position"] is None and state.get("last_signal_bar") != bar_time:
        prev = df.iloc[-2]

        ema_cross_up = prev["emaFast"] <= prev["emaSlow"] and last["emaFast"] > last["emaSlow"]
        ema_cross_down = prev["emaFast"] >= prev["emaSlow"] and last["emaFast"] < last["emaSlow"]

        rsi_ok_long = (not USE_RSI) or last["rsi"] < RSI_OB
        rsi_ok_short = (not USE_RSI) or last["rsi"] > RSI_OS

        st_bullish = last["st_dir"] == 1
        st_bearish = last["st_dir"] == -1

        long_cond = ema_cross_up and rsi_ok_long and st_bullish
        short_cond = ema_cross_down and rsi_ok_short and st_bearish

        state["last_signal_bar"] = bar_time

        if long_cond or short_cond:
            entry = last["close"]
            sl_dist = min(max(last["atr"] * SL_MULT, SL_MIN_PTS), SL_MAX_PTS)

            if long_cond:
                sl = entry - sl_dist
                risk = entry - sl
                tp1, tp2, tp3 = entry + risk * RR1, entry + risk * RR2, entry + risk * RR3
                side = "BUY"
            else:
                sl = entry + sl_dist
                risk = sl - entry
                tp1, tp2, tp3 = entry - risk * RR1, entry - risk * RR2, entry - risk * RR3
                side = "SELL"

            state["position"] = open_position(side, entry, sl, tp1, tp2, tp3, bar_time)

            msg = (
                f"XAUUSD {side} signal (Supertrend)\n"
                f"Entry: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP1: {tp1:.2f}\n"
                f"TP2: {tp2:.2f}\n"
                f"TP3: {tp3:.2f}\n"
                f"Supertrend: {trend_arrow(last['st_dir'])}\n"
                f"Bar: {bar_time}"
            )
            send_telegram(msg)
            result["event"] = "entry"
            result["side"] = side
            result["message"] = msg

        save_state(state)

    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    state = load_state()
    s = state["stats"]
    win_rate = (s["wins"] / s["total_trades"] * 100) if s["total_trades"] else 0
    return jsonify({
        "position": state["position"],
        "win_rate": round(win_rate, 1),
        "total_trades": s["total_trades"],
        "wins": s["wins"], "losses": s["losses"],
        "net_pnl": round(s["sum_pnl"], 2),
        "best_trade": s["best_trade"], "worst_trade": s["worst_trade"],
        "recent_log": state["history"][:10],
    })


@app.route("/test", methods=["GET"])
def test_signal():
    """Sends a forced test message through the real Telegram path using live
    price data. Does not touch state.json or open a real position."""
    if not TWELVE_DATA_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    df = fetch_candles(TIMEFRAME, outputsize=100)
    df["atr"] = atr(df, ATR_LEN)
    last = df.iloc[-1]

    entry = last["close"]
    sl_dist = min(max(last["atr"] * SL_MULT, SL_MIN_PTS), SL_MAX_PTS)
    sl = entry - sl_dist
    risk = entry - sl
    tp1, tp2, tp3 = entry + risk * RR1, entry + risk * RR2, entry + risk * RR3

    msg = (
        f"[TEST] XAUUSD BUY signal (Supertrend)\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f}\n"
        f"TP2: {tp2:.2f}\n"
        f"TP3: {tp3:.2f}\n"
        f"Bar: {last['datetime']}\n"
        f"(This is a forced test message, not a real signal)"
    )
    send_telegram(msg)
    return jsonify({"status": "test message sent", "message": msg})


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
