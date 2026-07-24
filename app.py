"""
Gold Signal Terminal — Free polling bot (no TradingView subscription needed)

Full re-implementation of the Pine Script "Gold Signal Terminal" indicator:
  - 9/21 EMA crossover entries
  - RSI(14) filter (blocks buys above 70, sells below 30)
  - ATR(14) x1.5 Stop Loss, TP1 = 1R, TP2 = 2R, TP3 = 3R
  - 1H + 4H trend agreement filter (both must agree by default)
  - Counter-trend setups allowed only when RSI is extreme (<=20 / >=80)
    AND reward:risk (TP2 multiple) >= 2.5
  - Cascading breakeven, exactly like the indicator:
        TP1 hit -> close 50% of position, SL -> Entry
        TP2 hit -> close to 75% cumulative closed, SL -> TP1
        TP3 hit -> runner closes, trade fully settled
  - Win-rate / trend vs counter-trend stats, matching the indicator's stats panel

Unlike the indicator (which only draws on a chart), this bot actually tracks
an OPEN POSITION across polls (in state.json) and sends a Telegram alert for
every event: entry, TP1 partial, TP2 partial, and final close (TP3 / SL /
breakeven stop).

Data source: Twelve Data free API (https://twelvedata.com — free signup, no card required)
Alerts: sent directly to Telegram via the Bot API (no relay/Pipedream needed)

Deploy as a Render free Web Service. Since Render free web services sleep when idle,
use a free external pinger (e.g. https://cron-job.org) to hit the /check endpoint
every 5 minutes — that keeps it awake and checks for new signals/exits on schedule.
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

# ---------------------- STRATEGY PARAMETERS (mirrors the Pine Script inputs, same defaults) ----
FAST_LEN = 9
SLOW_LEN = 15
USE_RSI = True
RSI_LEN = 15
RSI_OB = 70
RSI_OS = 30

ATR_LEN = 8
SL_MULT = 3
RR1, RR2, RR3 = 1.45, 1.77, 2

REQUIRE_BOTH_ALIGN = True   # 1H AND 4H must agree for a trend trade
ALLOW_COUNTER_TREND = True
COUNTER_RSI_OS = 20         # counter-trend LONG needs RSI <= this
COUNTER_RSI_OB = 80         # counter-trend SHORT needs RSI >= this
COUNTER_MIN_RR = 2.5        # counter-trend min reward:risk (TP2 multiple)

# Position sizing (mirrors the indicator's "Position Sizing" group)
LOT_SIZE = float(os.environ.get("LOT_SIZE", 0.01))
UNITS_PER_LOT = float(os.environ.get("UNITS_PER_LOT", 100))

# Partial-close behavior (mirrors the indicator's "Risk Management" group)
TP1_CLOSE_PCT = 50          # % of ORIGINAL position closed at TP1
TP2_CUMULATIVE_PCT = 75     # cumulative % of ORIGINAL position closed by TP2
CLOSE_FULL_AT_TP1 = False   # if True, TP1 closes 100% and TP2/TP3 are skipped

ENTRY_INTERVAL = "5min"
HTF1_INTERVAL = "1h"
HTF2_INTERVAL = "4h"

STATE_FILE = "state.json"


# ---------------------- DATA FETCH ----------------------
def fetch_candles(interval, outputsize=100):
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


def atr(df, length):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def htf_trend(df):
    """Returns 1 (bullish), -1 (bearish), or 0 (flat) based on the last closed candle."""
    fast = ema(df["close"], FAST_LEN)
    slow = ema(df["close"], SLOW_LEN)
    f, s = fast.iloc[-1], slow.iloc[-1]
    if f > s:
        return 1
    elif f < s:
        return -1
    return 0


def trend_arrow(t):
    return "▲" if t == 1 else "▼" if t == -1 else "→"


# ---------------------- STATE ----------------------
DEFAULT_STATE = {
    "last_signal_bar": None,   # last 5min bar_time we evaluated for a NEW entry
    "last_exit_bar": None,     # last 5min bar_time we evaluated for exits on an open position
    "position": None,          # dict or None, see open_position()
    "history": [],             # closed/partial trade log, newest first
    "stats": {
        "total_trades": 0, "wins": 0, "losses": 0, "sum_pnl": 0.0,
        "best_trade": None, "worst_trade": None,
        "trend_trades": 0, "trend_wins": 0,
        "counter_trades": 0, "counter_wins": 0,
    },
}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        # backfill any missing keys if the schema grows later
        for k, v in DEFAULT_STATE.items():
            state.setdefault(k, v)
        return state
    return json.loads(json.dumps(DEFAULT_STATE))  # deep copy


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
def log_trade(state, side, mode, entry, sl, tp1, tp2, tp3, exit_price, result, points, pnl):
    state["history"].insert(0, {
        "side": side, "mode": mode, "entry": entry, "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "exit": exit_price,
        "result": result, "points": round(points, 2), "pnl": round(pnl, 2),
    })
    state["history"] = state["history"][:50]
    state["stats"]["sum_pnl"] += pnl


def settle_trade(state, pos, exit_price, result_label):
    """Fully closes whatever size remains and updates win/loss + trend/counter stats."""
    points = (exit_price - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - exit_price)
    pnl = points * LOT_SIZE * pos["remaining_size"] * UNITS_PER_LOT

    log_trade(state, "BUY" if pos["dir"] == 1 else "SELL",
              "Counter" if pos["is_counter"] else "Trend",
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

    if pos["is_counter"]:
        stats["counter_trades"] += 1
        if combined_positive:
            stats["counter_wins"] += 1
    else:
        stats["trend_trades"] += 1
        if combined_positive:
            stats["trend_wins"] += 1

    state["position"] = None
    return pnl


def partial_close_tp1(state, pos):
    close_frac = TP1_CLOSE_PCT / 100.0
    points = (pos["tp1"] - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - pos["tp1"])
    pnl = points * LOT_SIZE * close_frac * UNITS_PER_LOT

    log_trade(state, "BUY" if pos["dir"] == 1 else "SELL",
              "Counter" if pos["is_counter"] else "Trend",
              pos["entry"], pos["sl"], pos["tp1"], pos["tp2"], pos["tp3"],
              pos["tp1"], f"TP1 Hit ({TP1_CLOSE_PCT}% Partial)", points, pnl)

    pos["tp1_hit"] = True
    pos["remaining_size"] = round(1.0 - close_frac, 6)
    pos["sl"] = pos["entry"]  # SL -> breakeven, always
    return pnl


def partial_close_tp2(state, pos):
    target_remaining = 1.0 - (TP2_CUMULATIVE_PCT / 100.0)
    close_frac = max(pos["remaining_size"] - target_remaining, 0.0)
    points = (pos["tp2"] - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - pos["tp2"])
    pnl = points * LOT_SIZE * close_frac * UNITS_PER_LOT

    log_trade(state, "BUY" if pos["dir"] == 1 else "SELL",
              "Counter" if pos["is_counter"] else "Trend",
              pos["entry"], pos["sl"], pos["tp1"], pos["tp2"], pos["tp3"],
              pos["tp2"], f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", points, pnl)

    pos["tp2_hit"] = True
    pos["remaining_size"] = round(target_remaining, 6)
    pos["sl"] = pos["tp1"]  # SL -> TP1, always
    return pnl


# ---------------------- POSITION MANAGEMENT (mirrors the Pine if-cascade) ----------------------
def manage_position(state, last_candle):
    """Checks the most recently CLOSED candle's high/low against SL/TP1/TP2/TP3,
    exactly like the indicator's long/short management blocks. Sends a Telegram
    message for every partial close or final close. Returns True if anything happened.
    """
    pos = state["position"]
    if pos is None:
        return False

    high, low = last_candle["high"], last_candle["low"]
    events = []

    if pos["dir"] == 1:
        if CLOSE_FULL_AT_TP1:
            if low <= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                events.append(("SL Hit", pos["sl"], pnl))
            elif high >= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        elif not pos["tp1_hit"]:
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
                pnl = settle_trade(state, pos, pos["sl"], "Breakeven Stop")
                events.append(("Breakeven Stop", pos["sl"], pnl))
            elif high >= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Gap)")
                events.append(("TP3 Hit (Gap)", pos["tp3"], pnl))
            elif high >= pos["tp2"]:
                p2 = partial_close_tp2(state, pos)
                events.append((f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", pos["tp2"], p2))
        else:
            if low <= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "TP1 Stop (Locked Profit)")
                events.append(("TP1 Stop (Locked Profit)", pos["sl"], pnl))
            elif high >= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Runner)")
                events.append(("TP3 Hit (Runner)", pos["tp3"], pnl))

    else:  # short, mirror image
        if CLOSE_FULL_AT_TP1:
            if high >= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                events.append(("SL Hit", pos["sl"], pnl))
            elif low <= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        elif not pos["tp1_hit"]:
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
                pnl = settle_trade(state, pos, pos["sl"], "Breakeven Stop")
                events.append(("Breakeven Stop", pos["sl"], pnl))
            elif low <= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Gap)")
                events.append(("TP3 Hit (Gap)", pos["tp3"], pnl))
            elif low <= pos["tp2"]:
                p2 = partial_close_tp2(state, pos)
                events.append((f"TP2 Hit ({TP2_CUMULATIVE_PCT}% Cumulative)", pos["tp2"], p2))
        else:
            if high >= pos["sl"]:
                pnl = settle_trade(state, pos, pos["sl"], "TP1 Stop (Locked Profit)")
                events.append(("TP1 Stop (Locked Profit)", pos["sl"], pnl))
            elif low <= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Runner)")
                events.append(("TP3 Hit (Runner)", pos["tp3"], pnl))

    side = "BUY" if pos["dir"] == 1 else "SELL"
    for label, price, pnl in events:
        msg = (
            f"XAUUSD {side} — {label}\n"
            f"Price: {price:.2f}\n"
            f"P&L (this leg): {'+' if pnl >= 0 else ''}${pnl:.2f}"
        )
        send_telegram(msg)

    return len(events) > 0


def open_position(state, side, entry, sl, tp1, tp2, tp3, is_counter, bar_time):
    state["position"] = {
        "dir": 1 if side == "BUY" else -1,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "entry_time": bar_time,
        "is_counter": is_counter,
        "tp1_hit": False, "tp2_hit": False,
        "remaining_size": 1.0,
    }


# ---------------------- CORE CHECK ----------------------
@app.route("/check", methods=["GET"])
def check():
    if not TWELVE_DATA_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    entry_df = fetch_candles(ENTRY_INTERVAL, outputsize=100)
    entry_df["emaFast"] = ema(entry_df["close"], FAST_LEN)
    entry_df["emaSlow"] = ema(entry_df["close"], SLOW_LEN)
    entry_df["rsi"] = rsi(entry_df["close"], RSI_LEN)
    entry_df["atr"] = atr(entry_df, ATR_LEN)

    last = entry_df.iloc[-1]
    bar_time = str(last["datetime"])

    state = load_state()
    result = {"bar_time": bar_time, "event": None}

    # 1) Manage an already-open position against this bar's high/low
    if state["position"] is not None and state.get("last_exit_bar") != bar_time:
        acted = manage_position(state, last)
        state["last_exit_bar"] = bar_time
        if acted:
            result["event"] = "position_update"
        save_state(state)

    # 2) Only look for a NEW entry if we're currently flat
    if state["position"] is None and state.get("last_signal_bar") != bar_time:
        prev = entry_df.iloc[-2]

        ema_cross_up = prev["emaFast"] <= prev["emaSlow"] and last["emaFast"] > last["emaSlow"]
        ema_cross_down = prev["emaFast"] >= prev["emaSlow"] and last["emaFast"] < last["emaSlow"]

        rsi_ok_long = (not USE_RSI) or last["rsi"] < RSI_OB
        rsi_ok_short = (not USE_RSI) or last["rsi"] > RSI_OS

        htf1_df = fetch_candles(HTF1_INTERVAL, outputsize=60)
        htf2_df = fetch_candles(HTF2_INTERVAL, outputsize=60)
        trend1h = htf_trend(htf1_df)
        trend4h = htf_trend(htf2_df)

        if REQUIRE_BOTH_ALIGN:
            htf_bullish = trend1h == 1 and trend4h == 1
            htf_bearish = trend1h == -1 and trend4h == -1
        else:
            htf_bullish = trend1h == 1 or trend4h == 1
            htf_bearish = trend1h == -1 or trend4h == -1

        counter_long_ok = ALLOW_COUNTER_TREND and last["rsi"] <= COUNTER_RSI_OS and RR2 >= COUNTER_MIN_RR
        counter_short_ok = ALLOW_COUNTER_TREND and last["rsi"] >= COUNTER_RSI_OB and RR2 >= COUNTER_MIN_RR

        long_trend_ok = htf_bullish or (not htf_bullish and not htf_bearish)
        short_trend_ok = htf_bearish or (not htf_bullish and not htf_bearish)

        long_is_counter = (not long_trend_ok) and counter_long_ok
        short_is_counter = (not short_trend_ok) and counter_short_ok

        long_cond = ema_cross_up and rsi_ok_long and (long_trend_ok or long_is_counter)
        short_cond = ema_cross_down and rsi_ok_short and (short_trend_ok or short_is_counter)

        state["last_signal_bar"] = bar_time

        if long_cond or short_cond:
            entry = last["close"]
            risk = last["atr"] * SL_MULT

            if long_cond:
                sl = entry - risk
                tp1, tp2, tp3 = entry + risk * RR1, entry + risk * RR2, entry + risk * RR3
                is_counter, side = long_is_counter, "BUY"
            else:
                sl = entry + risk
                tp1, tp2, tp3 = entry - risk * RR1, entry - risk * RR2, entry - risk * RR3
                is_counter, side = short_is_counter, "SELL"

            open_position(state, side, entry, sl, tp1, tp2, tp3, is_counter, bar_time)

            mode = "Counter" if is_counter else "Trend"
            msg = (
                f"XAUUSD {side} signal ({mode})\n"
                f"Entry: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP1: {tp1:.2f}\n"
                f"TP2: {tp2:.2f}\n"
                f"TP3: {tp3:.2f}\n"
                f"1H trend: {trend_arrow(trend1h)}  4H trend: {trend_arrow(trend4h)}\n"
                f"Bar: {bar_time}"
            )
            send_telegram(msg)
            result["event"] = "entry"
            result["side"] = side
            result["mode"] = mode
            result["message"] = msg

        save_state(state)

    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    """Mirrors the indicator's stats panel (win rate, trend vs counter, net P&L)."""
    state = load_state()
    s = state["stats"]
    win_rate = (s["wins"] / s["total_trades"] * 100) if s["total_trades"] else 0
    trend_wr = (s["trend_wins"] / s["trend_trades"] * 100) if s["trend_trades"] else 0
    counter_wr = (s["counter_wins"] / s["counter_trades"] * 100) if s["counter_trades"] else 0
    return jsonify({
        "position": state["position"],
        "win_rate_all": round(win_rate, 1),
        "win_rate_trend": round(trend_wr, 1),
        "win_rate_counter": round(counter_wr, 1),
        "total_trades": s["total_trades"],
        "wins": s["wins"], "losses": s["losses"],
        "net_pnl": round(s["sum_pnl"], 2),
        "best_trade": s["best_trade"], "worst_trade": s["worst_trade"],
        "recent_log": state["history"][:10],
    })


@app.route("/test", methods=["GET"])
def test_signal():
    """
    Forces a fake BUY signal through the exact same message-building and
    Telegram-sending code as a real signal, using live price data for
    realistic numbers. Does NOT touch state.json or open a real position,
    so it can be called as many times as you want without affecting real trading.
    """
    if not TWELVE_DATA_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    entry_df = fetch_candles(ENTRY_INTERVAL, outputsize=100)
    entry_df["atr"] = atr(entry_df, ATR_LEN)
    last = entry_df.iloc[-1]

    entry = last["close"]
    risk = last["atr"] * SL_MULT
    sl = entry - risk
    tp1, tp2, tp3 = entry + risk * RR1, entry + risk * RR2, entry + risk * RR3

    msg = (
        f"[TEST] XAUUSD BUY signal (Trend)\n"
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
