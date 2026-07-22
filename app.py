"""
Gold Signal Terminal — Free polling version (Finnhub data source)

Re-implements the entry-signal logic from the original Pine Script indicator:
  - 9/21 EMA crossover
  - RSI filter (blocks buys above 70, sells below 30)
  - ATR-based Stop Loss, TP1 (1R), TP2 (2R), TP3 (3R)
  - 1H + 4H trend agreement filter
  - Counter-trend setups allowed only when RSI is extreme AND reward:risk >= 2.5

Data source: Finnhub (https://finnhub.io — free signup, no card required)
  Uses the /forex/candle endpoint. Finnhub's supported candle resolutions are
  1, 5, 15, 30, 60, D, W, M — there is no native "4h" resolution, so 4H bars
  are built here by fetching 60-min candles and resampling them.

  IMPORTANT: Finnhub has tightened free-tier access to candle data over time —
  some free accounts get {"error": "You don't have access to this resource."}
  on /forex/candle even with a valid key. Test the /test endpoint (or curl the
  URL printed by fetch_candles' underlying request) right after deploying to
  confirm your key actually has candle access before relying on this.

Alerts: sent directly to Telegram via the Bot API (no relay/Pipedream needed)

Deploy as a Render free Web Service. Since Render free web services sleep when idle,
use a free external pinger (e.g. https://cron-job.org) to hit the /check endpoint
every 5 minutes — that keeps it awake and checks for new signals on schedule.
"""

import os
import json
import time
import requests
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# ---------------------- CONFIG (env vars, set these in Render) ----------------------
FINNHUB_API_KEY    = os.environ.get("FINNHUB_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
# Finnhub forex symbols are "EXCHANGE:BASE_QUOTE". OANDA:XAU_USD is spot gold vs USD.
SYMBOL             = os.environ.get("SYMBOL", "OANDA:XAU_USD")

# ---------------------- STRATEGY PARAMETERS (mirrors the Pine Script inputs) --------
FAST_LEN = 9
SLOW_LEN = 20
RSI_LEN = 20
RSI_OB = 70
RSI_OS = 30

ATR_LEN = 12
SL_MULT = 3
RR1, RR2, RR3 = 1.45, 1.9, 2.5

REQUIRE_BOTH_ALIGN = True
ALLOW_COUNTER_TREND = True
COUNTER_RSI_OS = 20
COUNTER_RSI_OB = 80
COUNTER_MIN_RR = 2.5

ENTRY_INTERVAL = "5min"
HTF1_INTERVAL = "1h"
HTF2_INTERVAL = "4h"

STATE_FILE = "state.json"

# Maps our interval names to (Finnhub resolution, seconds-per-base-candle).
# "4h" has no native Finnhub resolution, so we fetch 60-min candles and resample.
_INTERVAL_MAP = {
    "5min": ("5", 300),
    "1h": ("60", 3600),
    "4h": ("60", 3600),
}


# ---------------------- DATA FETCH ----------------------
def fetch_candles(interval, outputsize=100):
    resolution, base_seconds = _INTERVAL_MAP[interval]

    # How many BASE-resolution candles do we need before any resampling?
    base_count = outputsize * 4 if interval == "4h" else outputsize

    # Forex is closed roughly 2 days out of 7, so pad the requested time span
    # generously to make sure we still end up with `outputsize` closed candles.
    buffer_factor = 1.6
    span_seconds = int(base_count * base_seconds * buffer_factor)

    to_ts = int(time.time())
    from_ts = to_ts - span_seconds

    url = "https://finnhub.io/api/v1/forex/candle"
    params = {
        "symbol": SYMBOL,
        "resolution": resolution,
        "from": from_ts,
        "to": to_ts,
        "token": FINNHUB_API_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"Finnhub error for {interval}: {data}")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["t"], unit="s", utc=True),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
    })
    df = df.sort_values("datetime").reset_index(drop=True)

    if interval == "4h":
        df = (
            df.set_index("datetime")
              .resample("4h")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
              .dropna()
              .reset_index()
        )

    df = df.tail(outputsize).reset_index(drop=True)
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


# ---------------------- STATE (avoid duplicate alerts for the same candle) ----------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_alert_time": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ---------------------- TELEGRAM ----------------------
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
    return resp.json()


# ---------------------- CORE CHECK ----------------------
@app.route("/check", methods=["GET"])
def check():
    if not FINNHUB_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    entry_df = fetch_candles(ENTRY_INTERVAL, outputsize=100)
    htf1_df = fetch_candles(HTF1_INTERVAL, outputsize=60)
    htf2_df = fetch_candles(HTF2_INTERVAL, outputsize=60)

    entry_df["emaFast"] = ema(entry_df["close"], FAST_LEN)
    entry_df["emaSlow"] = ema(entry_df["close"], SLOW_LEN)
    entry_df["rsi"] = rsi(entry_df["close"], RSI_LEN)
    entry_df["atr"] = atr(entry_df, ATR_LEN)

    last = entry_df.iloc[-1]
    prev = entry_df.iloc[-2]

    ema_cross_up = prev["emaFast"] <= prev["emaSlow"] and last["emaFast"] > last["emaSlow"]
    ema_cross_down = prev["emaFast"] >= prev["emaSlow"] and last["emaFast"] < last["emaSlow"]

    rsi_ok_long = last["rsi"] < RSI_OB
    rsi_ok_short = last["rsi"] > RSI_OS

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

    long_is_counter = not long_trend_ok and counter_long_ok
    short_is_counter = not short_trend_ok and counter_short_ok

    long_cond = ema_cross_up and rsi_ok_long and (long_trend_ok or long_is_counter)
    short_cond = ema_cross_down and rsi_ok_short and (short_trend_ok or short_is_counter)

    state = load_state()
    bar_time = str(last["datetime"])
    result = {"bar_time": bar_time, "signal": None}

    if (long_cond or short_cond) and state.get("last_alert_time") != bar_time:
        entry = last["close"]
        risk = last["atr"] * SL_MULT

        if long_cond:
            sl = entry - risk
            tp1, tp2, tp3 = entry + risk * RR1, entry + risk * RR2, entry + risk * RR3
            mode = "Counter" if long_is_counter else "Trend"
            side = "BUY"
        else:
            sl = entry + risk
            tp1, tp2, tp3 = entry - risk * RR1, entry - risk * RR2, entry - risk * RR3
            mode = "Counter" if short_is_counter else "Trend"
            side = "SELL"

        msg = (
            f"XAUUSD {side} signal ({mode})\n"
            f"Entry: {entry:.2f}\n"
            f"SL: {sl:.2f}\n"
            f"TP1: {tp1:.2f}\n"
            f"TP2: {tp2:.2f}\n"
            f"TP3: {tp3:.2f}\n"
            f"1H trend: {'▲' if trend1h==1 else '▼' if trend1h==-1 else '→'}  "
            f"4H trend: {'▲' if trend4h==1 else '▼' if trend4h==-1 else '→'}\n"
            f"Bar: {bar_time}"
        )
        send_telegram(msg)

        state["last_alert_time"] = bar_time
        save_state(state)
        result["signal"] = side
        result["message"] = msg

    return jsonify(result)


@app.route("/test", methods=["GET"])
def test_signal():
    """
    Forces a fake BUY signal through the exact same message-building and
    Telegram-sending code as a real signal, using live price data for
    realistic numbers. Use this to confirm the full pipeline works without
    waiting for a real EMA crossover. Does NOT check state.json, so you can
    call this as many times as you want without it blocking real alerts.
    """
    if not FINNHUB_API_KEY or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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
