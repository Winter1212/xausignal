import os
import json
import requests
import pandas as pd
from flask import Flask, jsonify, request, Response

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


# (see the big NOTE at the top of this file). Default matches the
# indicator's "Force-Entry Timezone" input default ("Asia/Phnom_Penh",
# Cambodia, UTC+7). This drives the Daily Trade Guarantee's day-boundary
# AND its FORCE_HOUR/FORCE_MINUTE check below, AND (now) the weekend
# no-new-signals guard below.
FORCE_TIMEZONE = os.environ.get("FORCE_TIMEZONE", "Asia/Phnom_Penh")

# ---------------------- WEEKEND GUARD ----------------------
# The market (XAU/USD spot) is closed over the weekend. When enabled (the
# default), the bot will NOT open any NEW signal -- organic or forced by
# the Daily Trade Guarantee -- while the current bar's calendar day (in
# FORCE_TIMEZONE) is a Saturday or Sunday. It still manages/closes any
# already-open position against incoming bars, in case your feed still
# ticks over the weekend. Set DISABLE_WEEKEND_SIGNALS=false to turn this
# off again.
DISABLE_WEEKEND_SIGNALS = os.environ.get("DISABLE_WEEKEND_SIGNALS", "true").lower() == "true"

KEY_STATE_FILE = "api_key_state.json"  # persists rotation index + per-key daily usage across polls

# ---------------------- SIGNAL ENGINE PARAMETERS (exact indicator defaults) ----------------------
# Pine inputs: fastLen=12, slowLen=35, useRSI=true, rsiLen=15, rsiOB=70, rsiOS=30
FAST_LEN = 27
SLOW_LEN = 32
USE_RSI = True
RSI_LEN = 15
RSI_OB = 70   # block buys above this
RSI_OS = 30   # block sells below this

# ---------------------- SUPERTREND TREND FILTER (exact indicator defaults) ----------------------
# Pine inputs: stAtrPeriod=15, stFactor=5.0
ST_ATR_PERIOD = 8
ST_FACTOR = 2.0
# Require a Supertrend flip to hold this many bars before it's tradeable
# (matches indicator's "stConfirmBars" input, default 2).
ST_CONFIRM_BARS = int(os.environ.get("ST_CONFIRM_BARS", 1))

# ---------------------- HIGHER TIMEFRAME CONFIRMATION (exact indicator defaults) ----------------------
# Matches the indicator's "Higher Timeframe Confirmation" group. Only takes
# BUY signals when the HTF Supertrend is bullish, only takes SELL signals
# when it's bearish.
USE_HTF = os.environ.get("USE_HTF", "true").lower() == "true"
# Twelve Data interval strings, NOT Pine's "240" minute-count style.
# Indicator default is "240" minutes -> Twelve Data equivalent is "4h".
HTF_TIMEFRAME = os.environ.get("HTF_TIMEFRAME", "4h")
HTF_ATR_PERIOD = int(os.environ.get("HTF_ATR_PERIOD", 4))
HTF_FACTOR = float(os.environ.get("HTF_FACTOR", 8))

# ---------------------- ENTRY TIMING: PULLBACK CONFIRMATION (exact indicator defaults) ----------------------
# Mirrors the indicator's "Entry Timing (Pullback Confirmation)" group.
# Instead of entering the instant the EMA cross + RSI + Supertrend + HTF
# stack all agree ("the base condition"), that agreement only ARMS a
# pending signal. The trade is only actually opened once price retraces
# back within PULLBACK_MAX_ATR of the Fast EMA (while the trend filters
# still agree), or the pending signal is dropped if it hasn't pulled back
# within PULLBACK_TIMEOUT_BARS bars.
#
# Because this bot polls periodically (it doesn't walk bar-by-bar like the
# Pine script does on a chart), the "armed" state has to be persisted in
# state.json across polls, and re-anchored to the current bar's position in
# the freshly-fetched candle window every time /check runs. See
# roll_pullback_state() below.
USE_PULLBACK_ENTRY = os.environ.get("USE_PULLBACK_ENTRY", "true").lower() == "true"
PULLBACK_MAX_ATR = float(os.environ.get("PULLBACK_MAX_ATR", 1.2))
PULLBACK_TIMEOUT_BARS = int(os.environ.get("PULLBACK_TIMEOUT_BARS", 4))

# Overextension filter: blocks ANY organic entry (pullback or immediate
# mode) if price is currently too far from the Fast EMA in ATR terms —
# the "already extended, about to mean-revert" state the indicator's
# changelog calls out as the cause of its losing SELLs.
USE_EXTENSION_FILTER = os.environ.get("USE_EXTENSION_FILTER", "true").lower() == "true"
MAX_EXTENSION_ATR = float(os.environ.get("MAX_EXTENSION_ATR", 2.0))

# ---------------------- DAILY TRADE GUARANTEE (exact indicator defaults) ----------------------
# Matches the indicator's "Daily Trade Guarantee" group. If nothing organic
# has opened a trade yet today by FORCE_HOUR:FORCE_MINUTE (evaluated in
# FORCE_TIMEZONE, default Cambodia/UTC+7 — see NOTE at top of file), force
# one entry in the direction of the prevailing trend so every day gets >= 1
# trade. This is a SEPARATE, clearly-tagged fallback — it never loosens the
# organic EMA/RSI/Supertrend/HTF/pullback/extension stack above. It is also
# subject to the weekend guard above, so it will not fire on Sat/Sun.
GUARANTEE_DAILY_TRADE = os.environ.get("GUARANTEE_DAILY_TRADE", "true").lower() == "true"
FORCE_HOUR   = int(os.environ.get("FORCE_HOUR", 9))    # 0-23, in FORCE_TIMEZONE (see NOTE above)
FORCE_MINUTE = int(os.environ.get("FORCE_MINUTE", 0))  # 0-59

# ---------------------- RISK MANAGEMENT (exact indicator defaults) ----------------------
# Pine inputs: atrLen=12, slMult=1.0, slMinPts=10, slMaxPts=10, rr1=1.8, rr2=2.8, rr3=3.5
ATR_LEN = 12
SL_MULT = 1
SL_MIN_PTS = 10.0
SL_MAX_PTS = 10.0
RR1, RR2, RR3 = 1.9, 2.8, 3.5

TP1_CLOSE_PCT = 50          # % of ORIGINAL position closed at TP1 (Partial mode only)
TP2_CUMULATIVE_PCT = 75     # cumulative % of ORIGINAL position closed by TP2 (Partial mode only)

# "partial" | "tp1_only" | "first_hit"  (matches the indicator's pnlMode dropdown)
# Pine's default is "Partial Closes (Cascade TP1->TP2->TP3)" -> "partial".
# (Previously this defaulted to "tp1_only" here, which did NOT match the
# indicator's default behavior — fixed.)
PNL_MODE = os.environ.get("PNL_MODE", "tp1_only")

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


def is_weekend_bar(bar_dt):
    """
    True when bar_dt (already localized to FORCE_TIMEZONE by fetch_candles'
    timezone= param) falls on a Saturday or Sunday. pandas/Python weekday()
    is 0=Monday ... 5=Saturday, 6=Sunday.
    """
    return bar_dt.weekday() >= 5


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
    # --- Pullback Confirmation tracking (mirrors pendingLong / pendingShort / pendingBar) ---
    "pending_dir": None,        # 1 (armed long), -1 (armed short), or None
    "pending_bar_time": None,   # datetime string of the bar that armed the pending signal
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


def _notify_requested():
    """True when the caller asked for the result to also be pushed to
    Telegram, e.g. GET /check?notify=1 or GET /stats?notify=true."""
    val = request.args.get("notify", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _fmt_price(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def _fmt_money(v):
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else ''}{v:.2f}"


def build_check_telegram_message(result, state):
    """Builds a '[Manual Check]' summary for the dashboard's 'send to
    Telegram' toggle on /check. This is DELIBERATELY separate from the
    automatic entry/exit alerts fired inside manage_position()/the entry
    block above, so a manual dashboard check never gets confused with a
    real automated signal — it's always clearly tagged."""
    lines = ["📊 [Manual Check] XAUUSD", f"Bar: {result.get('bar_time')} ({FORCE_TIMEZONE})"]

    event = result.get("event")
    if event == "entry":
        lines.append(f"Result: NEW {result.get('side')} signal opened this check {'(Forced Daily)' if result.get('forced') else '(Supertrend)'}.")
    elif event == "position_update":
        lines.append("Result: open position was updated this check (TP/SL/trail — see the alert above).")
    elif result.get("weekend_skipped"):
        lines.append("Result: weekend — new-signal checks skipped (existing position, if any, still managed).")
    else:
        pending = state.get("pending_dir")
        if pending:
            lines.append(f"Result: no new signal on this bar. Pending {'LONG' if pending == 1 else 'SHORT'} signal armed, waiting for pullback.")
        else:
            lines.append("Result: no new signal on this bar.")

    pos = state.get("position")
    if pos:
        side = "BUY" if pos["dir"] == 1 else "SELL"
        lines.append(
            f"Open position: {side} @ {_fmt_price(pos['entry'])} | "
            f"SL {_fmt_price(pos['sl'])} | TP1 {_fmt_price(pos['tp1'])} "
            f"TP2 {_fmt_price(pos['tp2'])} TP3 {_fmt_price(pos['tp3'])} | "
            f"remaining {round(pos['remaining_size'] * 100)}%"
        )
    else:
        lines.append("No open position.")

    return "\n".join(lines)


def build_stats_telegram_message(payload, state):
    """Builds a '[Manual Stats]' summary for the dashboard's 'send to
    Telegram' toggle on /stats."""
    lines = [
        "📈 [Manual Stats] XAUUSD Bot",
        f"Win rate: {payload['win_rate']}% ({payload['wins']}W / {payload['losses']}L / {payload['total_trades']} total)",
        f"Net P&L: {_fmt_money(payload['net_pnl'])}",
        f"Best trade: {_fmt_money(payload['best_trade'])} | Worst trade: {_fmt_money(payload['worst_trade'])}",
    ]

    pos = payload.get("position")
    if pos:
        side = "BUY" if pos["dir"] == 1 else "SELL"
        lines.append(
            f"Open position: {side} @ {_fmt_price(pos['entry'])} | SL {_fmt_price(pos['sl'])}"
        )
    else:
        lines.append("No open position.")

    dg = payload["daily_guarantee"]
    lines.append(
        f"Daily guarantee: {'ON' if dg['enabled'] else 'OFF'} @ "
        f"{dg['force_hour']:02d}:{dg['force_minute']:02d} {dg['force_timezone']} | "
        f"traded today: {'yes' if dg['traded_today'] else 'no'}"
    )

    return "\n".join(lines)


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
                          and flat and not weekend
      forceDirection   = entry-TF Supertrend, else HTF Supertrend,
                          else EMA position (always resolves to +-1)

    bar_dt is already localized to FORCE_TIMEZONE (Twelve Data's `timezone`
    param does this at fetch time), so bar_dt.hour/bar_dt.minute here mean
    "9:00" = 9:00am in FORCE_TIMEZONE, not exchange time.

    Returns (force_entry_now: bool, force_direction: int [1 or -1]).
    """
    if not GUARANTEE_DAILY_TRADE:
        return False, 0

    if DISABLE_WEEKEND_SIGNALS and is_weekend_bar(bar_dt):
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


# ---------------------- ENTRY TIMING HELPERS (Pullback Confirmation) ----------------------
def bars_since_pending_armed(df, pending_bar_time):
    """
    Finds the row in the freshly-fetched df matching the bar that armed the
    pending signal, and returns how many bars have elapsed since then
    (0 = armed on the current/last bar). If the armed bar has fallen out of
    the fetched window (too old), returns a large number so the caller
    treats it as timed out — mirrors what would happen on a real chart if
    price never pulled back for that long.
    """
    if pending_bar_time is None:
        return None
    target = pd.to_datetime(pending_bar_time)
    matches = df.index[df["datetime"] == target]
    if len(matches) == 0:
        return PULLBACK_TIMEOUT_BARS + 1
    return (len(df) - 1) - matches[0]


def roll_pullback_state(state, df, bar_time, st_bullish, st_bearish, htf_bullish, htf_bearish,
                         base_long_cond, base_short_cond):
    """
    Mirrors the indicator's pending-signal block:

      if baseLongCond:  pendingLong := true,  pendingShort := false
      if baseShortCond: pendingShort := true, pendingLong := false
      if pendingLong  and (not stBullish or not htfBullish or timeout): pendingLong := false
      if pendingShort and (not stBearish or not htfBearish or timeout): pendingShort := false

    Mutates state["pending_dir"] / state["pending_bar_time"] in place.
    """
    if base_long_cond:
        state["pending_dir"] = 1
        state["pending_bar_time"] = bar_time
    if base_short_cond:
        state["pending_dir"] = -1
        state["pending_bar_time"] = bar_time

    bars_since = bars_since_pending_armed(df, state["pending_bar_time"])

    if state["pending_dir"] == 1:
        timed_out = bars_since is not None and bars_since > PULLBACK_TIMEOUT_BARS
        if not st_bullish or not htf_bullish or timed_out:
            state["pending_dir"] = None
            state["pending_bar_time"] = None
    elif state["pending_dir"] == -1:
        timed_out = bars_since is not None and bars_since > PULLBACK_TIMEOUT_BARS
        if not st_bearish or not htf_bearish or timed_out:
            state["pending_dir"] = None
            state["pending_bar_time"] = None


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

    # 1) Manage an already-open position against this bar's high/low.
    #    This still runs on weekends -- if a position is open going into
    #    the weekend we keep tracking SL/TP against whatever the feed
    #    reports, we just won't OPEN anything new below.
    if state["position"] is not None and state.get("last_exit_bar") != bar_time:
        acted = manage_position(state, last, last["st"])
        state["last_exit_bar"] = bar_time
        if acted:
            result["event"] = "position_update"
        save_state(state)

    # 1b) Weekend guard: don't evaluate/open any NEW signal (organic or
    #     forced) while the current bar is a Saturday/Sunday in
    #     FORCE_TIMEZONE. Still update last_signal_bar so we don't just
    #     spin re-checking the same closed bar all weekend. Also
    #     explicitly clears any pending pullback signal that was armed
    #     before the weekend started -- this mirrors the indicator's
    #     "weekendBlocked" cancellation of pendingLong/pendingShort, so a
    #     signal armed Friday afternoon can never silently fire on the
    #     Sunday-night/Monday-open bar. Without this it would simply have
    #     been *frozen* (never re-evaluated) and left to expire on its own
    #     via the pullback timeout once Monday's bar was processed, which
    #     usually resolves the same way but wasn't a guaranteed match.
    weekend_now = DISABLE_WEEKEND_SIGNALS and is_weekend_bar(last["datetime"])
    if weekend_now:
        state["last_signal_bar"] = bar_time
        if state.get("pending_dir") is not None:
            state["pending_dir"] = None
            state["pending_bar_time"] = None
        result["weekend_skipped"] = True
        save_state(state)

    # 2) Only look for a NEW entry if we're currently flat, it's not a bar
    #    we've already processed, and it's not the weekend.
    if (not weekend_now) and state["position"] is None and state.get("last_signal_bar") != bar_time:
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

        # --- Overextension filter (matches useExtensionFilter/extensionOk) ---
        extension_atr = (abs(last["close"] - last["emaFast"]) / last["atr"]) if last["atr"] > 0 else 0.0
        extension_ok = (not USE_EXTENSION_FILTER) or extension_atr <= MAX_EXTENSION_ATR

        # --- Pullback distance (matches pullbackDistATR/pullbackOk) ---
        pullback_dist_atr = (abs(last["close"] - last["emaFast"]) / last["atr"]) if last["atr"] > 0 else 0.0
        pullback_ok = pullback_dist_atr <= PULLBACK_MAX_ATR

        # --- Base condition: the whole filter stack agrees (matches
        #     baseLongCond / baseShortCond). This is what ARMS a pending
        #     signal, and (in immediate mode) is also the entry condition. ---
        base_long_cond = ema_cross_up and rsi_ok_long and st_bullish and st_flip_confirmed and htf_bullish
        base_short_cond = ema_cross_down and rsi_ok_short and st_bearish and st_flip_confirmed and htf_bearish

        # --- Arm / cancel the pending pullback signal (matches the
        #     indicator's pendingLong/pendingShort block) ---
        roll_pullback_state(
            state, df, bar_time, st_bullish, st_bearish, htf_bullish, htf_bearish,
            base_long_cond, base_short_cond,
        )

        # --- Organic entry conditions (matches longCond/shortCond) ---
        if USE_PULLBACK_ENTRY:
            long_cond = (
                extension_ok and state["pending_dir"] == 1 and pullback_ok
                and st_bullish and htf_bullish and rsi_ok_long
            )
            short_cond = (
                extension_ok and state["pending_dir"] == -1 and pullback_ok
                and st_bearish and htf_bearish and rsi_ok_short
            )
        else:
            long_cond = extension_ok and base_long_cond
            short_cond = extension_ok and base_short_cond

        # --- Daily Trade Guarantee: forced fallback entry (matches forceEntryNow/forceDirection) ---
        force_entry_now, force_direction = compute_force_entry(
            state, last["datetime"], int(last["st_dir"]), htf_dir,
            last["emaFast"], last["emaSlow"],
        )
        force_long_cond = force_entry_now and force_direction == 1
        force_short_cond = force_entry_now and force_direction == -1

        is_long_entry = long_cond or force_long_cond
        is_short_entry = short_cond or force_short_cond

        state["last_signal_bar"] = bar_time
        result["pending"] = {
            "dir": state["pending_dir"],
            "extension_atr": round(extension_atr, 3),
            "extension_ok": extension_ok,
        }

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

            # Clear the pending signal on entry (matches the indicator
            # setting pendingLong/pendingShort := false in both entry blocks).
            state["pending_dir"] = None
            state["pending_bar_time"] = None

            is_pullback = USE_PULLBACK_ENTRY and not is_forced
            tag = "(Forced Daily)" if is_forced else "(Supertrend, Pullback)" if is_pullback else "(Supertrend)"
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
            result["pullback"] = is_pullback
            result["message"] = msg
            result["entry"] = entry
            result["sl"] = sl
            result["tp1"] = tp1
            result["tp2"] = tp2
            result["tp3"] = tp3
        elif force_entry_now:
            # Force-entry time passed, but direction resolution somehow
            # produced neither (shouldn't happen since forceDirection always
            # resolves to +-1) — mark the attempt so we don't retry forever
            # on subsequent bars today.
            state["force_attempted_today"] = True

        save_state(state)

    # Reload state fresh so the "position" snapshot in the Telegram summary
    # (and the JSON response, for the dashboard) reflects everything that
    # just happened above.
    state = load_state()
    result["position"] = state.get("position")

    if _notify_requested():
        send_telegram(build_check_telegram_message(result, state))
        result["telegram_notified"] = True

    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    state = load_state()
    s = state["stats"]
    win_rate = (s["wins"] / s["total_trades"] * 100) if s["total_trades"] else 0
    payload = {
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
        "pending_signal": {
            "dir": state.get("pending_dir"),
            "armed_bar_time": state.get("pending_bar_time"),
        },
        "weekend_guard": {
            "enabled": DISABLE_WEEKEND_SIGNALS,
        },
    }

    if _notify_requested():
        send_telegram(build_stats_telegram_message(payload, state))
        payload["telegram_notified"] = True

    return jsonify(payload)


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
        "use_pullback_entry": USE_PULLBACK_ENTRY,
        "pullback_max_atr": PULLBACK_MAX_ATR,
        "pullback_timeout_bars": PULLBACK_TIMEOUT_BARS,
        "use_extension_filter": USE_EXTENSION_FILTER,
        "max_extension_atr": MAX_EXTENSION_ATR,
        "pnl_mode": PNL_MODE,
        "guarantee_daily_trade": GUARANTEE_DAILY_TRADE,
        "force_hour": FORCE_HOUR,
        "force_minute": FORCE_MINUTE,
        "force_timezone": FORCE_TIMEZONE,
        "disable_weekend_signals": DISABLE_WEEKEND_SIGNALS,
        "dashboard": "/dashboard",
    })


# ---------------------- DASHBOARD UI ----------------------
@app.route("/dashboard", methods=["GET"])
def dashboard():
    return Response(DASHBOARD_HTML, mimetype="text/html")


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>XAUUSD Signal Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b0e11;
    --surface:#12161c;
    --surface-2:#181e26;
    --border:#242b34;
    --gold:#d4af6a;
    --gold-dim:#8a7346;
    --gold-glow:rgba(212,175,106,.16);
    --buy:#3fbf7f;
    --buy-glow:rgba(63,191,127,.14);
    --sell:#e0574c;
    --sell-glow:rgba(224,87,76,.14);
    --text:#eef1f5;
    --text-dim:#8b93a1;
    --text-faint:#5b6472;
    --mono:'JetBrains Mono',monospace;
    --sans:'Inter',sans-serif;
    --display:'Space Grotesk',sans-serif;
  }
  *{box-sizing:border-box;}
  body{
    margin:0;
    background:
      radial-gradient(circle at 12% -10%, rgba(212,175,106,.10), transparent 45%),
      radial-gradient(circle at 100% 0%, rgba(63,191,127,.06), transparent 40%),
      var(--bg);
    color:var(--text);
    font-family:var(--sans);
    min-height:100vh;
    padding:28px 18px 60px;
  }
  .wrap{max-width:920px;margin:0 auto;}

  /* ---- Header / ticker ---- */
  header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;}
  .brand{display:flex;align-items:center;gap:12px;}
  .brand .mark{
    width:38px;height:38px;border-radius:10px;
    background:linear-gradient(155deg, var(--gold), #a7823f);
    display:flex;align-items:center;justify-content:center;
    font-family:var(--display);font-weight:700;color:#0b0e11;font-size:15px;
    box-shadow:0 0 0 1px rgba(212,175,106,.35), 0 8px 20px -6px rgba(212,175,106,.5);
  }
  .brand h1{font-family:var(--display);font-size:19px;margin:0;font-weight:600;letter-spacing:.2px;}
  .brand .sub{font-family:var(--mono);font-size:11px;color:var(--text-faint);letter-spacing:.4px;text-transform:uppercase;}
  .live-pill{
    display:flex;align-items:center;gap:7px;
    font-family:var(--mono);font-size:11.5px;color:var(--text-dim);
    background:var(--surface);border:1px solid var(--border);
    padding:7px 12px;border-radius:100px;
  }
  .dot{width:7px;height:7px;border-radius:50%;background:var(--buy);box-shadow:0 0 0 3px var(--buy-glow);animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}

  /* ---- Candlestick divider (signature element) ---- */
  .divider{display:flex;align-items:flex-end;gap:3px;height:34px;margin:22px 0 26px;opacity:.9;}
  .divider .bar{flex:1;max-width:9px;border-radius:2px 2px 0 0;background:var(--border);}
  .divider .bar.up{background:linear-gradient(180deg, var(--buy), transparent);}
  .divider .bar.down{background:linear-gradient(180deg, var(--sell), transparent);}
  .divider .bar.gold{background:linear-gradient(180deg, var(--gold), transparent);}

  /* ---- Action cards ---- */
  .actions{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  @media (max-width:600px){.actions{grid-template-columns:1fr;}}
  .card{
    background:linear-gradient(180deg, var(--surface), var(--surface-2));
    border:1px solid var(--border);border-radius:14px;padding:20px;
    display:flex;flex-direction:column;gap:14px;
  }
  .card h2{font-family:var(--display);font-size:15px;margin:0;font-weight:600;}
  .card p.desc{margin:0;font-size:12.5px;color:var(--text-dim);line-height:1.5;}
  .toggle-row{display:flex;align-items:center;justify-content:space-between;gap:10px;}
  .toggle-label{font-size:12.5px;color:var(--text-dim);display:flex;align-items:center;gap:7px;}
  .toggle-label svg{width:14px;height:14px;opacity:.8;}
  .switch{position:relative;width:38px;height:22px;flex:none;}
  .switch input{opacity:0;width:0;height:0;}
  .slider{position:absolute;inset:0;background:#252b34;border-radius:20px;cursor:pointer;transition:.2s;border:1px solid var(--border);}
  .slider::before{content:"";position:absolute;width:16px;height:16px;left:2px;top:2px;background:var(--text-faint);border-radius:50%;transition:.2s;}
  .switch input:checked + .slider{background:var(--gold-glow);border-color:var(--gold-dim);}
  .switch input:checked + .slider::before{transform:translateX(16px);background:var(--gold);}
  button.run{
    margin-top:auto;
    background:var(--gold);color:#141008;border:none;border-radius:9px;
    font-family:var(--sans);font-weight:600;font-size:13.5px;
    padding:11px 16px;cursor:pointer;transition:.15s;
    display:flex;align-items:center;justify-content:center;gap:8px;
  }
  button.run:hover{filter:brightness(1.08);transform:translateY(-1px);}
  button.run:active{transform:translateY(0);}
  button.run:disabled{opacity:.55;cursor:progress;transform:none;}
  button.run.secondary{background:transparent;border:1px solid var(--gold-dim);color:var(--gold);}
  .spinner{width:13px;height:13px;border-radius:50%;border:2px solid rgba(20,16,8,.35);border-top-color:#141008;animation:spin .7s linear infinite;display:none;}
  button.run.loading .spinner{display:inline-block;}
  @keyframes spin{to{transform:rotate(360deg);}}

  /* ---- Results panel ---- */
  .results{margin-top:20px;}
  .empty{
    border:1px dashed var(--border);border-radius:14px;padding:32px 20px;
    text-align:center;color:var(--text-faint);font-size:13px;font-family:var(--mono);
  }
  .panel{
    background:var(--surface);border:1px solid var(--border);border-radius:14px;
    overflow:hidden;
  }
  .panel-head{
    display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;
    padding:14px 18px;border-bottom:1px solid var(--border);background:var(--surface-2);
  }
  .panel-head .title{font-family:var(--display);font-size:13.5px;font-weight:600;display:flex;align-items:center;gap:9px;}
  .badge{
    font-family:var(--mono);font-size:10.5px;font-weight:600;letter-spacing:.4px;
    padding:3px 9px;border-radius:100px;text-transform:uppercase;
  }
  .badge.buy{background:var(--buy-glow);color:var(--buy);}
  .badge.sell{background:var(--sell-glow);color:var(--sell);}
  .badge.neutral{background:var(--gold-glow);color:var(--gold);}
  .badge.muted{background:#20262e;color:var(--text-faint);}
  .timestamp{font-family:var(--mono);font-size:11px;color:var(--text-faint);}

  .panel-body{padding:18px;}
  .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
  @media (max-width:560px){.grid4{grid-template-columns:repeat(2,1fr);}}
  .stat-box{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:12px;}
  .stat-box .k{font-size:10.5px;color:var(--text-faint);text-transform:uppercase;letter-spacing:.4px;margin-bottom:5px;}
  .stat-box .v{font-family:var(--mono);font-size:16px;font-weight:600;}
  .v.pos{color:var(--buy);} .v.neg{color:var(--sell);} .v.gold{color:var(--gold);}

  .kv{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);font-size:12.5px;}
  .kv:last-child{border-bottom:none;}
  .kv .k{color:var(--text-dim);}
  .kv .v{font-family:var(--mono);font-weight:600;}

  table{width:100%;border-collapse:collapse;margin-top:6px;font-size:12px;}
  thead th{
    text-align:left;color:var(--text-faint);font-weight:500;font-size:10.5px;
    text-transform:uppercase;letter-spacing:.3px;padding:6px 8px;border-bottom:1px solid var(--border);
  }
  tbody td{padding:8px;border-bottom:1px solid var(--border);font-family:var(--mono);}
  tbody tr:last-child td{border-bottom:none;}
  .side-tag{font-weight:700;font-size:11px;}
  .side-tag.buy{color:var(--buy);} .side-tag.sell{color:var(--sell);}
  .scroll-x{overflow-x:auto;}

  .section-label{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--text-faint);margin:18px 0 8px;font-family:var(--mono);}
  .error-box{background:var(--sell-glow);border:1px solid rgba(224,87,76,.3);color:#ff9d95;padding:12px 16px;border-radius:10px;font-size:12.5px;font-family:var(--mono);}

  footer{text-align:center;margin-top:34px;font-size:11px;color:var(--text-faint);font-family:var(--mono);}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="brand">
      <div class="mark">Au</div>
      <div>
        <h1>XAUUSD Signal Desk</h1>
        <div class="sub">Supertrend + EMA/RSI engine</div>
      </div>
    </div>
    <div class="live-pill"><span class="dot"></span> engine online</div>
  </header>

  <div class="divider" id="divider"></div>

  <div class="actions">
    <div class="card">
      <h2>Run Check</h2>
      <p class="desc">Pulls the latest bar, evaluates entry/exit conditions and updates any open position.</p>
      <div class="toggle-row">
        <span class="toggle-label">Send result to Telegram</span>
        <label class="switch"><input type="checkbox" id="notifyCheck"><span class="slider"></span></label>
      </div>
      <button class="run" id="btnCheck" onclick="runCheck()">
        <span class="spinner"></span><span class="label">Run Check</span>
      </button>
    </div>

    <div class="card">
      <h2>Get Stats</h2>
      <p class="desc">Win rate, net P&amp;L, open position and the daily-trade-guarantee status.</p>
      <div class="toggle-row">
        <span class="toggle-label">Send result to Telegram</span>
        <label class="switch"><input type="checkbox" id="notifyStats"><span class="slider"></span></label>
      </div>
      <button class="run secondary" id="btnStats" onclick="runStats()">
        <span class="spinner"></span><span class="label">Get Stats</span>
      </button>
    </div>
  </div>

  <div class="results" id="results">
    <div class="empty">Run a check or pull stats to see live output here.</div>
  </div>

  <footer>state persists server-side · times shown in bot's FORCE_TIMEZONE</footer>
</div>

<script>
function buildDivider(){
  const el = document.getElementById('divider');
  const pattern = ['gold','up','up','down','up','down','down','up','gold','up','down','up','up','gold','down','up','down','up','gold','up'];
  el.innerHTML = pattern.map((cls,i)=>{
    const h = 8 + Math.round(Math.sin(i*1.3)*8 + 14);
    return `<div class="bar ${cls}" style="height:${h}px"></div>`;
  }).join('');
}
buildDivider();

function setLoading(btn, loading){
  btn.disabled = loading;
  btn.classList.toggle('loading', loading);
}

function fmtMoney(v){
  if(v === null || v === undefined) return '—';
  const s = v >= 0 ? '+' : '';
  return `${s}$${Number(v).toFixed(2)}`;
}
function fmtPrice(v){
  if(v === null || v === undefined) return '—';
  return Number(v).toFixed(2);
}
function esc(s){
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

async function runCheck(){
  const btn = document.getElementById('btnCheck');
  const notify = document.getElementById('notifyCheck').checked;
  setLoading(btn, true);
  try{
    const res = await fetch(`/check?notify=${notify ? 1 : 0}`);
    const data = await res.json();
    if(!res.ok){ renderError(data.error || 'Request failed'); return; }
    renderCheck(data);
  }catch(e){
    renderError('Could not reach the server: ' + e.message);
  }finally{
    setLoading(btn, false);
  }
}

async function runStats(){
  const btn = document.getElementById('btnStats');
  const notify = document.getElementById('notifyStats').checked;
  setLoading(btn, true);
  try{
    const res = await fetch(`/stats?notify=${notify ? 1 : 0}`);
    const data = await res.json();
    if(!res.ok){ renderError(data.error || 'Request failed'); return; }
    renderStats(data);
  }catch(e){
    renderError('Could not reach the server: ' + e.message);
  }finally{
    setLoading(btn, false);
  }
}

function renderError(msg){
  document.getElementById('results').innerHTML = `<div class="error-box">⚠ ${esc(msg)}</div>`;
}

function eventBadge(event, weekendSkipped){
  if(event === 'entry') return `<span class="badge neutral">New Entry</span>`;
  if(event === 'position_update') return `<span class="badge muted">Position Updated</span>`;
  if(weekendSkipped) return `<span class="badge muted">Weekend — Skipped</span>`;
  return `<span class="badge muted">No Signal</span>`;
}

function positionKv(pos){
  if(!pos) return `<div class="kv"><span class="k">Open position</span><span class="v">None</span></div>`;
  const side = pos.dir === 1 ? 'BUY' : 'SELL';
  const sideClass = pos.dir === 1 ? 'buy' : 'sell';
  return `
    <div class="kv"><span class="k">Side</span><span class="v side-tag ${sideClass}">${side}</span></div>
    <div class="kv"><span class="k">Entry</span><span class="v">${fmtPrice(pos.entry)}</span></div>
    <div class="kv"><span class="k">Stop Loss</span><span class="v">${fmtPrice(pos.sl)}</span></div>
    <div class="kv"><span class="k">TP1 / TP2 / TP3</span><span class="v">${fmtPrice(pos.tp1)} / ${fmtPrice(pos.tp2)} / ${fmtPrice(pos.tp3)}</span></div>
    <div class="kv"><span class="k">Remaining size</span><span class="v">${Math.round(pos.remaining_size*100)}%</span></div>
  `;
}

function renderCheck(data){
  const notifiedBadge = data.telegram_notified ? `<span class="badge neutral">Sent to Telegram</span>` : '';
  let entryBlock = '';
  if(data.event === 'entry'){
    const sideClass = data.side === 'BUY' ? 'buy' : 'sell';
    entryBlock = `
      <div class="section-label">New Signal</div>
      <div class="grid4">
        <div class="stat-box"><div class="k">Side</div><div class="v side-tag ${sideClass}">${data.side}${data.forced ? ' ⚡' : ''}</div></div>
        <div class="stat-box"><div class="k">Entry</div><div class="v gold">${fmtPrice(data.entry)}</div></div>
        <div class="stat-box"><div class="k">Stop Loss</div><div class="v neg">${fmtPrice(data.sl)}</div></div>
        <div class="stat-box"><div class="k">Take Profits</div><div class="v pos" style="font-size:12.5px">${fmtPrice(data.tp1)} · ${fmtPrice(data.tp2)} · ${fmtPrice(data.tp3)}</div></div>
      </div>
      ${data.forced ? `<p style="font-size:11.5px;color:var(--text-faint);margin-top:8px;">⚡ Opened by the Daily Trade Guarantee fallback, not the organic signal stack.</p>` : ''}
      ${data.pullback ? `<p style="font-size:11.5px;color:var(--text-faint);margin-top:8px;">↩ Opened after a pullback confirmation, not on the initial cross bar.</p>` : ''}
    `;
  }

  let pendingBlock = '';
  if(data.event !== 'entry' && data.pending && data.pending.dir){
    const dirLabel = data.pending.dir === 1 ? 'LONG' : 'SHORT';
    pendingBlock = `<div class="kv"><span class="k">Pending signal</span><span class="v">${dirLabel} (waiting for pullback)</span></div>`;
  }

  let weekendBlock = '';
  if(data.weekend_skipped){
    weekendBlock = `<p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">🌙 Weekend — new-signal checks are paused. Any open position is still tracked.</p>`;
  }

  document.getElementById('results').innerHTML = `
    <div class="panel">
      <div class="panel-head">
        <div class="title">Check Result ${eventBadge(data.event, data.weekend_skipped)} ${notifiedBadge}</div>
        <div class="timestamp">${esc(data.bar_time || '—')}</div>
      </div>
      <div class="panel-body">
        ${entryBlock}
        ${weekendBlock}
        ${pendingBlock}
        <div class="section-label">Current Position</div>
        ${positionKv(data.position)}
      </div>
    </div>
  `;
}

function renderStats(data){
  const notifiedBadge = data.telegram_notified ? `<span class="badge neutral">Sent to Telegram</span>` : '';
  const pnlClass = data.net_pnl > 0 ? 'pos' : (data.net_pnl < 0 ? 'neg' : '');
  const rows = (data.recent_log || []).map(t => {
    const sideClass = t.side === 'BUY' ? 'buy' : 'sell';
    const pnlC = t.pnl > 0 ? 'pos' : (t.pnl < 0 ? 'neg' : '');
    return `
      <tr>
        <td class="side-tag ${sideClass}">${esc(t.side)}</td>
        <td>${fmtPrice(t.entry)}</td>
        <td>${fmtPrice(t.exit)}</td>
        <td>${esc(t.result)}</td>
        <td class="${pnlC}">${fmtMoney(t.pnl)}</td>
      </tr>
    `;
  }).join('') || `<tr><td colspan="5" style="color:var(--text-faint);text-align:center;">No trades logged yet</td></tr>`;

  const dg = data.daily_guarantee || {};
  const ps = data.pending_signal || {};

  document.getElementById('results').innerHTML = `
    <div class="panel">
      <div class="panel-head">
        <div class="title">Performance ${notifiedBadge}</div>
        <div class="timestamp">${dg.current_day || ''} · ${esc(dg.force_timezone || '')}</div>
      </div>
      <div class="panel-body">
        <div class="grid4">
          <div class="stat-box"><div class="k">Win Rate</div><div class="v gold">${data.win_rate}%</div></div>
          <div class="stat-box"><div class="k">Record</div><div class="v">${data.wins}W – ${data.losses}L</div></div>
          <div class="stat-box"><div class="k">Net P&amp;L</div><div class="v ${pnlClass}">${fmtMoney(data.net_pnl)}</div></div>
          <div class="stat-box"><div class="k">Total Trades</div><div class="v">${data.total_trades}</div></div>
        </div>

        <div class="section-label">Current Position</div>
        ${positionKv(data.position)}

        <div class="section-label">Pending Signal</div>
        <div class="kv"><span class="k">Armed</span><span class="v">${ps.dir ? (ps.dir === 1 ? 'LONG (waiting for pullback)' : 'SHORT (waiting for pullback)') : 'None'}</span></div>

        <div class="section-label">Daily Trade Guarantee</div>
        <div class="kv"><span class="k">Status</span><span class="v">${dg.enabled ? 'Enabled' : 'Disabled'} @ ${String(dg.force_hour).padStart(2,'0')}:${String(dg.force_minute).padStart(2,'0')}</span></div>
        <div class="kv"><span class="k">Traded today</span><span class="v">${dg.traded_today ? 'Yes' : 'No'}</span></div>

        <div class="section-label">Recent Trades</div>
        <div class="scroll-x">
          <table>
            <thead><tr><th>Side</th><th>Entry</th><th>Exit</th><th>Result</th><th>P&amp;L</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
