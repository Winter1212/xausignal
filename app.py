"""
Gold Signal Terminal — Free polling bot (no TradingView subscription needed)

This is a 1:1 re-implementation of the Pine Script v6 indicator
"Gold Signal Terminal — Entry/SL/TP1-3 + Winrate + Supertrend", INCLUDING:

  - 9/21 EMA crossover entries
  - RSI(14) filter (blocks buys above 70, sells below 30)
  - Supertrend(10, 3.0) trend filter on the entry timeframe — a signal only
    fires WITH the Supertrend direction
  - Supertrend "flip confirmation" — a fresh flip must hold for
    ST_CONFIRM_BARS bars before it's allowed to trigger a trade. This is
    what stops the bot from entering on the exact bar of a flip that then
    immediately reverses back (classic whipsaw).
  - Higher-timeframe Supertrend confirmation — BUY signals additionally
    require the HTF Supertrend to be bullish, SELL signals require it to be
    bearish. This is the fix for "overall trend is up, a small pullback
    flips the entry-TF Supertrend, we short, and get stopped as price
    resumes up." Costs one extra API call per poll (see CONFIG below).
  - Daily Trade Guarantee — mirrors the indicator's "Daily Trade Guarantee"
    group. If no organic BUY/SELL has opened a trade yet today by the
    configured Force-Entry Hour/Minute, the bot forces one entry in the
    direction of the prevailing trend (entry-TF Supertrend, falling back to
    HTF Supertrend, falling back to EMA position) so every day gets at
    least one trade. Forced entries are tagged "(Forced Daily)" everywhere
    (Telegram message + trade log) so you can evaluate/disable them
    separately from organic "(Supertrend)" signals. Exactly like the
    indicator, this does NOT loosen the organic filters — it's a
    separately-tagged fallback that only fires when the day would
    otherwise end with zero trades.
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

NOTE ON API CREDIT USAGE: enabling USE_HTF (on by default, matching the
indicator's default) means every /check call fetches TWO timeframes instead
of one — your entry timeframe AND the higher timeframe. This roughly
doubles Twelve Data credit consumption per poll. If you're tight on free-tier
credits, either set USE_HTF=false or add more keys via
TWELVE_DATA_API_KEY_2 / _3.

NOTE ON THE DAILY TRADE GUARANTEE AND TIME ZONES: every call to Twelve
Data's time_series endpoint below passes a `timezone` parameter
(FORCE_TIMEZONE, default "Asia/Bangkok" — same UTC+7 offset as Cambodia,
and more consistently supported by Twelve Data's timezone list than
"Asia/Phnom_Penh"). That makes Twelve Data return each bar's "datetime"
value ALREADY LOCALIZED to that zone, rather than in the exchange's raw
timezone. FORCE_HOUR / FORCE_MINUTE are then compared directly against
that localized timestamp, so "9:00" means 9:00am in FORCE_TIMEZONE
regardless of what timezone the underlying exchange feed uses. If you want
a different session's local time instead of Cambodia, just change
FORCE_TIMEZONE to another IANA zone (e.g. "America/New_York",
"Europe/London") — every downstream calculation (day-of-week/day boundary,
force-entry check) follows automatically since it's all derived from the
same localized "datetime" column. Also note (same caveat as the indicator):
if TIMEFRAME is a daily-or-higher bar, every bar's hour/minute will be the
same value (typically 00:00), so set FORCE_HOUR/FORCE_MINUTE to 0/0 in that
case or just use this bot on an intraday timeframe.
"""

import os
import json
import requests
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# ---------------------- CONFIG (env vars, set these in Render) ----------------------
# Multi-key rotation: set up to 3 separate Twelve Data API keys
# (TWELVE_DATA_API_KEY_1/2/3) to spread requests across accounts and get up
# to 3x the daily credit budget of a single free-tier key. You can also set
# just TWELVE_DATA_API_KEY_1 alone if you only have one key. Falls back to
# the legacy single-var TWELVE_DATA_API_KEY if none of the numbered vars
# are set.
_raw_keys = [
    os.environ.get("TWELVE_DATA_API_KEY_1", ""),
    os.environ.get("TWELVE_DATA_API_KEY_2", ""),
    os.environ.get("TWELVE_DATA_API_KEY_3", ""),
]
TWELVE_DATA_API_KEYS = [k for k in _raw_keys if k]
if not TWELVE_DATA_API_KEYS:
    legacy = os.environ.get("TWELVE_DATA_API_KEY", "")
    if legacy:
        TWELVE_DATA_API_KEYS = [legacy]

TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID", "")
SYMBOL               = os.environ.get("SYMBOL", "XAU/USD")
TIMEFRAME            = os.environ.get("TIMEFRAME", "5min")  # the entry chart timeframe, matches the indicator

# IANA timezone Twelve Data localizes every bar's "datetime" value into
# (see the big NOTE at the top of this file). Default is Cambodia's
# UTC+7 offset via "Asia/Bangkok". This drives the Daily Trade Guarantee's
# day-boundary AND its FORCE_HOUR/FORCE_MINUTE check below.
FORCE_TIMEZONE = os.environ.get("FORCE_TIMEZONE", "Asia/Bangkok")

KEY_STATE_FILE = "api_key_state.json"  # persists rotation index + per-key daily usage across polls

# ---------------------- SIGNAL ENGINE PARAMETERS (exact indicator defaults) ----------------------
# Pine inputs: fastLen=9, slowLen=21, useRSI=true, rsiLen=14, rsiOB=70, rsiOS=30
FAST_LEN = 9
SLOW_LEN = 21
USE_RSI = True
RSI_LEN = 14
RSI_OB = 70   # block buys above this
RSI_OS = 30   # block sells below this

# ---------------------- SUPERTREND TREND FILTER (exact indicator defaults) ----------------------
# Pine inputs: stAtrPeriod=10, stFactor=3.0
ST_ATR_PERIOD = 10
ST_FACTOR = 3.0
# Require a Supertrend flip to hold this many bars before it's tradeable
# (matches indicator's "stConfirmBars" input, default 2).
ST_CONFIRM_BARS = int(os.environ.get("ST_CONFIRM_BARS", 2))

# ---------------------- HIGHER TIMEFRAME CONFIRMATION (exact indicator defaults) ----------------------
# Matches the indicator's "Higher Timeframe Confirmation" group. Only takes
# BUY signals when the HTF Supertrend is bullish, only takes SELL signals
# when it's bearish.
USE_HTF = os.environ.get("USE_HTF", "true").lower() == "true"
# Twelve Data interval strings, NOT Pine's "60" minute-count style.
# Indicator default is "60" minutes -> Twelve Data equivalent is "1h".
HTF_TIMEFRAME = os.environ.get("HTF_TIMEFRAME", "1h")
HTF_ATR_PERIOD = int(os.environ.get("HTF_ATR_PERIOD", 10))
HTF_FACTOR = float(os.environ.get("HTF_FACTOR", 3.0))

# ---------------------- DAILY TRADE GUARANTEE (exact indicator defaults) ----------------------
# Matches the indicator's "Daily Trade Guarantee" group. If nothing organic
# has opened a trade yet today by FORCE_HOUR:FORCE_MINUTE (evaluated in
# FORCE_TIMEZONE, default Cambodia/UTC+7 — see NOTE at top of file), force
# one entry in the direction of the prevailing trend so every day gets >= 1
# trade. This is a SEPARATE, clearly-tagged fallback — it never loosens the
# organic EMA/RSI/Supertrend/HTF stack above.
GUARANTEE_DAILY_TRADE = os.environ.get("GUARANTEE_DAILY_TRADE", "true").lower() == "true"
FORCE_HOUR   = int(os.environ.get("FORCE_HOUR", 9))    # 0-23, in FORCE_TIMEZONE (see NOTE above)
FORCE_MINUTE = int(os.environ.get("FORCE_MINUTE", 0))  # 0-59

# ---------------------- RISK MANAGEMENT (exact indicator defaults) ----------------------
# Pine inputs: atrLen=14, slMult=1.5, slMinPts=10, slMaxPts=12, rr1=1.0, rr2=2.0, rr3=3.0
ATR_LEN = 14
SL_MULT = 1.5
SL_MIN_PTS = 10.0
SL_MAX_PTS = 12.0
RR1, RR2, RR3 = 1.0, 2.0, 3.0

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


# ---------------------- API KEY ROTATION ----------------------
def _load_key_state():
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(KEY_STATE_FILE):
        with open(KEY_STATE_FILE) as f:
            ks = json.load(f)
    else:
        ks = {}
    if ks.get("date") != today:
        # new UTC day -> reset per-key usage counters
        ks = {"date": today, "next_index": 0, "usage": {}}
    ks.setdefault("next_index", 0)
    ks.setdefault("usage", {})
    return ks


def _save_key_state(ks):
    with open(KEY_STATE_FILE, "w") as f:
        json.dump(ks, f, indent=2)


def _key_label(i):
    return f"key_{i + 1}"


def fetch_candles(interval, outputsize=200, credits_per_call=3):
    """
    Fetches candles from Twelve Data, rotating across up to 3 configured API
    keys. Each call advances the rotation by one key (round-robin), and
    tracks approximate credit usage per key per UTC day in KEY_STATE_FILE
    purely for visibility via /keys. If a key comes back rate-limited (HTTP
    429 or a Twelve Data error payload mentioning the limit), it
    automatically retries the SAME request on the next key instead of
    failing the whole /check call.

    Passes `timezone=FORCE_TIMEZONE` on every request so the returned
    "datetime" values are pre-localized to that zone (see the NOTE at the
    top of this file) — this is what makes the Daily Trade Guarantee's
    FORCE_HOUR/FORCE_MINUTE and day-boundary checks correct for Cambodia
    (or whatever FORCE_TIMEZONE is set to) regardless of the underlying
    exchange feed's own timezone.
    """
    if not TWELVE_DATA_API_KEYS:
        raise RuntimeError("No Twelve Data API key configured. Set TWELVE_DATA_API_KEY_1 (and optionally _2 / _3).")

    ks = _load_key_state()
    url = "https://api.twelvedata.com/time_series"
    n = len(TWELVE_DATA_API_KEYS)
    last_error = None

    for attempt in range(n):
        idx = (ks["next_index"] + attempt) % n
        key = TWELVE_DATA_API_KEYS[idx]
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": key,
            "order": "ASC",
            "timezone": FORCE_TIMEZONE,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
        except Exception as e:
            last_error = str(e)
            continue

        rate_limited = (
            r.status_code == 429
            or (isinstance(data, dict) and data.get("code") in (429, 8, 400) and "limit" in str(data.get("message", "")).lower())
        )

        if "values" in data:
            # success on this key -> record usage, advance rotation past it,
            # persist, and return
            label = _key_label(idx)
            ks["usage"][label] = ks["usage"].get(label, 0) + credits_per_call
            ks["next_index"] = (idx + 1) % n
            _save_key_state(ks)

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("datetime").reset_index(drop=True)
            return df

        last_error = data
        if not rate_limited:
            # A non-rate-limit error (bad symbol, bad interval, bad
            # timezone name, etc.) will fail the same way on every key, so
            # don't bother rotating.
            break
        # else: rate-limited -> loop continues and tries the next key

    raise RuntimeError(f"Twelve Data error for {interval} (tried {min(attempt + 1, n)} key(s)): {last_error}")


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


def bars_since_supertrend_flip(dir_series):
    """
    Mirrors the indicator's stFlipBar / barsSinceFlip logic: walks backward
    from the last bar counting how many consecutive bars have held the
    CURRENT direction. 0 means the current bar IS the flip bar (direction
    just changed this bar); 1 means it held for one bar after the flip, etc.
    """
    n = len(dir_series)
    if n == 0:
        return 9999
    last_dir = dir_series.iloc[-1]
    count = 0
    for i in range(n - 1, -1, -1):
        if dir_series.iloc[i] == last_dir:
            count += 1
        else:
            break
    return count - 1  # bars since flip (matches Pine's bar_index - stFlipBar)


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
    # --- Daily Trade Guarantee tracking (mirrors tradedToday / forceAttemptedTdy) ---
    "current_day": None,       # "YYYY-MM-DD" of the last bar we processed, in FORCE_TIMEZONE
    "traded_today": False,     # flips true the instant ANY trade (organic or forced) opens
    "force_attempted_today": False,  # true once we've made our one forced-entry attempt today
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


# ---------------------- DAILY TRADE GUARANTEE HELPERS ----------------------
def roll_daily_guarantee_state(state, bar_dt):
    """Mirrors the indicator's `newDay` reset block: whenever the bar's
    calendar day (in FORCE_TIMEZONE, since bar_dt comes pre-localized from
    fetch_candles) changes vs. the last bar we processed, reset
    traded_today / force_attempted_today for the new day."""
    day_str = bar_dt.strftime("%Y-%m-%d")
    if state.get("current_day") != day_str:
        state["current_day"] = day_str
        state["traded_today"] = False
        state["force_attempted_today"] = False


def compute_force_entry(state, bar_dt, st_dir, htf_dir, ema_fast_last, ema_slow_last):
    """
    Mirrors the indicator's forceEntryNow / forceDirection logic exactly:

      isPastForceTime = (hour*60+minute) >= (forceHour*60+forceMinute)
      forceEntryNow    = guaranteeDailyTrade and isPastForceTime
                          and not forceAttemptedTdy and not tradedToday
                          and flat
      forceDirection   = entry-TF Supertrend, else HTF Supertrend,
                          else EMA position (always resolves to +-1)

    bar_dt is already localized to FORCE_TIMEZONE (Twelve Data's `timezone`
    param does this at fetch time), so bar_dt.hour/bar_dt.minute here mean
    "9:00" = 9:00am in FORCE_TIMEZONE, not exchange time.

    Returns (force_entry_now: bool, force_direction: int [1 or -1]).
    """
    if not GUARANTEE_DAILY_TRADE:
        return False, 0

    is_past_force_time = (bar_dt.hour * 60 + bar_dt.minute) >= (FORCE_HOUR * 60 + FORCE_MINUTE)
    force_entry_now = (
        is_past_force_time
        and not state["force_attempted_today"]
        and not state["traded_today"]
        and state["position"] is None
    )

    if st_dir == 1 or st_dir == -1:
        force_direction = st_dir
    elif htf_dir == 1 or htf_dir == -1:
        force_direction = htf_dir
    else:
        force_direction = 1 if ema_fast_last >= ema_slow_last else -1

    return force_entry_now, force_direction


# ---------------------- CORE CHECK ----------------------
@app.route("/check", methods=["GET"])
def check():
    if not TWELVE_DATA_API_KEYS or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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

    # 0) Roll the Daily Trade Guarantee's day-tracking forward if this bar
    #    is a new calendar day IN FORCE_TIMEZONE (mirrors the indicator's
    #    `newDay` reset, now anchored to Cambodia/FORCE_TIMEZONE rather than
    #    the raw exchange feed timezone).
    roll_daily_guarantee_state(state, last["datetime"])

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

        # --- Supertrend flip-confirmation (matches indicator's stFlipConfirmed) ---
        bars_since_flip = bars_since_supertrend_flip(dir_series)
        st_flip_confirmed = bars_since_flip >= ST_CONFIRM_BARS

        # --- Higher-timeframe Supertrend confirmation (matches useHTF) ---
        htf_dir = 0
        if USE_HTF:
            htf_df = fetch_candles(HTF_TIMEFRAME, outputsize=100)
            _, htf_dir_series = supertrend(htf_df, HTF_ATR_PERIOD, HTF_FACTOR)
            htf_dir = int(htf_dir_series.iloc[-1])
        htf_bullish = (not USE_HTF) or htf_dir == 1
        htf_bearish = (not USE_HTF) or htf_dir == -1

        long_cond = ema_cross_up and rsi_ok_long and st_bullish and st_flip_confirmed and htf_bullish
        short_cond = ema_cross_down and rsi_ok_short and st_bearish and st_flip_confirmed and htf_bearish

        # --- Daily Trade Guarantee: forced fallback entry (matches forceEntryNow/forceDirection) ---
        # NOTE: if USE_HTF is False, htf_dir stays 0 above; compute_force_entry
        # still falls back to HTF Supertrend direction when the entry-TF
        # Supertrend is flat, so it fetches HTF Supertrend direction only
        # when USE_HTF already fetched it, otherwise falls through to EMA.
        force_entry_now, force_direction = compute_force_entry(
            state, last["datetime"], int(last["st_dir"]), htf_dir,
            last["emaFast"], last["emaSlow"],
        )
        force_long_cond = force_entry_now and force_direction == 1
        force_short_cond = force_entry_now and force_direction == -1

        is_long_entry = long_cond or force_long_cond
        is_short_entry = short_cond or force_short_cond

        state["last_signal_bar"] = bar_time

        if is_long_entry or is_short_entry:
            entry = last["close"]
            sl_dist = min(max(last["atr"] * SL_MULT, SL_MIN_PTS), SL_MAX_PTS)

            if is_long_entry:
                sl = entry - sl_dist
                risk = entry - sl
                tp1, tp2, tp3 = entry + risk * RR1, entry + risk * RR2, entry + risk * RR3
                side = "BUY"
                is_forced = force_long_cond and not long_cond
            else:
                sl = entry + sl_dist
                risk = sl - entry
                tp1, tp2, tp3 = entry - risk * RR1, entry - risk * RR2, entry - risk * RR3
                side = "SELL"
                is_forced = force_short_cond and not short_cond

            state["position"] = open_position(side, entry, sl, tp1, tp2, tp3, bar_time)

            # Mark the Daily Trade Guarantee as satisfied for today, exactly
            # like the indicator setting tradedToday/forceAttemptedTdy on
            # ANY entry (organic or forced).
            state["traded_today"] = True
            state["force_attempted_today"] = True

            tag = "(Forced Daily)" if is_forced else "(Supertrend)"
            htf_line = f"HTF Supertrend ({HTF_TIMEFRAME}): {trend_arrow(htf_dir)}\n" if USE_HTF else ""
            msg = (
                f"XAUUSD {side} signal {tag}\n"
                f"Entry: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP1: {tp1:.2f}\n"
                f"TP2: {tp2:.2f}\n"
                f"TP3: {tp3:.2f}\n"
                f"Supertrend ({TIMEFRAME}): {trend_arrow(last['st_dir'])}\n"
                f"{htf_line}"
                f"Bar: {bar_time} ({FORCE_TIMEZONE})"
            )
            send_telegram(msg)
            result["event"] = "entry"
            result["side"] = side
            result["forced"] = is_forced
            result["message"] = msg
        elif force_entry_now:
            # Force-entry time passed, but direction resolution somehow
            # produced neither (shouldn't happen since forceDirection always
            # resolves to +-1) — mark the attempt so we don't retry forever
            # on subsequent bars today.
            state["force_attempted_today"] = True

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
        "daily_guarantee": {
            "enabled": GUARANTEE_DAILY_TRADE,
            "force_hour": FORCE_HOUR,
            "force_minute": FORCE_MINUTE,
            "force_timezone": FORCE_TIMEZONE,
            "current_day": state.get("current_day"),
            "traded_today": state.get("traded_today"),
            "force_attempted_today": state.get("force_attempted_today"),
        },
    })


@app.route("/test", methods=["GET"])
def test_signal():
    """Sends a forced test message through the real Telegram path using live
    price data. Does not touch state.json or open a real position."""
    if not TWELVE_DATA_API_KEYS or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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
        f"Bar: {last['datetime']} ({FORCE_TIMEZONE})\n"
        f"(This is a forced test message, not a real signal)"
    )
    send_telegram(msg)
    return jsonify({"status": "test message sent", "message": msg})


@app.route("/keys", methods=["GET"])
def keys_status():
    """Shows how many keys are configured, rotation position, and today's
    approximate credit usage per key (resets at UTC midnight)."""
    ks = _load_key_state()
    return jsonify({
        "configured_keys": len(TWELVE_DATA_API_KEYS),
        "date_utc": ks["date"],
        "next_key_index": ks["next_index"] + 1,  # 1-based for readability
        "usage_today": ks["usage"],
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "keys_configured": len(TWELVE_DATA_API_KEYS),
        "use_htf": USE_HTF,
        "htf_timeframe": HTF_TIMEFRAME if USE_HTF else None,
        "st_confirm_bars": ST_CONFIRM_BARS,
        "guarantee_daily_trade": GUARANTEE_DAILY_TRADE,
        "force_hour": FORCE_HOUR,
        "force_minute": FORCE_MINUTE,
        "force_timezone": FORCE_TIMEZONE,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
