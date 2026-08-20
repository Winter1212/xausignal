import os
import json
import requests
import pandas as pd
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# ---------------------- CONFIG (env vars, set these in Render) ----------------------
# Multi-key rotation: set up to 3 separate Twelve Data API key
# (TWELVE_DATA_API_KEY_1/2/3) to spread requests across accounts and get up
# to 3x the daily credit budget of a single free-tier key. You can also se
# just TWELVE_DATA_API_KEY_1 alone if you only have one key. Falls back to
# the legacy single-var TWELVE_DATA_API_KEY if none of the numbered vars

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
# no-new-signals guard below, AND the trading-hours guard below.
FORCE_TIMEZONE = os.environ.get("FORCE_TIMEZONE", "Asia/Phnom_Penh")

# ---------------------- WEEKEND GUARD ----------------------
DISABLE_WEEKEND_SIGNALS = os.environ.get("DISABLE_WEEKEND_SIGNALS", "true").lower() == "true"

# ---------------------- TRADING HOURS GUARD ----------------------
USE_TRADING_HOURS   = os.environ.get("USE_TRADING_HOURS", "true").lower() == "true"
TRADING_START_HOUR   = int(os.environ.get("TRADING_START_HOUR", 6))
TRADING_START_MINUTE = int(os.environ.get("TRADING_START_MINUTE", 30))
TRADING_END_HOUR     = int(os.environ.get("TRADING_END_HOUR", 23))
TRADING_END_MINUTE   = int(os.environ.get("TRADING_END_MINUTE", 59))

KEY_STATE_FILE = "api_key_state.json"

# ---------------------- SIGNAL ENGINE PARAMETERS ----------------------
FAST_LEN = 27
SLOW_LEN = 32
USE_RSI = True
RSI_LEN = 16
RSI_OB = 70
RSI_OS = 35

# ---------------------- SUPERTREND TREND FILTER ----------------------
ST_ATR_PERIOD = 8
ST_FACTOR = 2.0
ST_CONFIRM_BARS = int(os.environ.get("ST_CONFIRM_BARS", 1))

# ---------------------- HIGHER TIMEFRAME CONFIRMATION ----------------------
USE_HTF = os.environ.get("USE_HTF", "true").lower() == "true"
HTF_TIMEFRAME = os.environ.get("HTF_TIMEFRAME", "4h")
HTF_ATR_PERIOD = int(os.environ.get("HTF_ATR_PERIOD", 4))
HTF_FACTOR = float(os.environ.get("HTF_FACTOR", 8))

# ---------------------- ENTRY TIMING: PULLBACK CONFIRMATION ----------------------
USE_PULLBACK_ENTRY = os.environ.get("USE_PULLBACK_ENTRY", "true").lower() == "true"
PULLBACK_MAX_ATR = float(os.environ.get("PULLBACK_MAX_ATR", 1.4))
PULLBACK_TIMEOUT_BARS = int(os.environ.get("PULLBACK_TIMEOUT_BARS", 8))

USE_EXTENSION_FILTER = os.environ.get("USE_EXTENSION_FILTER", "true").lower() == "true"
MAX_EXTENSION_ATR = float(os.environ.get("MAX_EXTENSION_ATR", 2.5))

# ---------------------- DAILY TRADE GUARANTEE ----------------------
GUARANTEE_DAILY_TRADE = os.environ.get("GUARANTEE_DAILY_TRADE", "true").lower() == "true"
FORCE_HOUR   = int(os.environ.get("FORCE_HOUR", 11))
FORCE_MINUTE = int(os.environ.get("FORCE_MINUTE", 0))

FORCE_REQUIRE_QUALITY_FILTERS = os.environ.get("FORCE_REQUIRE_QUALITY_FILTERS", "true").lower() == "true"

FORCE_HARD_HOUR   = int(os.environ.get("FORCE_HARD_HOUR", 23))
FORCE_HARD_MINUTE = int(os.environ.get("FORCE_HARD_MINUTE", 45))

FORCE_SKIP_IF_NEVER_VALID = os.environ.get("FORCE_SKIP_IF_NEVER_VALID", "false").lower() == "true"

# ---------------------- RISK MANAGEMENT ----------------------
ATR_LEN = 12
SL_MULT = 1
SL_MIN_PTS = 10.0
SL_MAX_PTS = 10.0
RR1, RR2, RR3, RR4 = 1.9, 2.5, 2.5, 3.5

PNL_MODE = os.environ.get("PNL_MODE", "partial")

USE_TRAILING_RUNNER = True

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
        ks = {"date": today, "next_index": 0, "usage": {}}
    ks.setdefault("next_index", 0)
    ks.setdefault("usage", {})
    return ks


def _save_key_state(ks):
    with open(KEY_STATE_FILE, "w") as f:
        json.dump(ks, f, indent=2)


def _key_label(i):
    return f"key_{i + 1}"


def fetch_candles(interval, outputsize=5000, credits_per_call=3):
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
            break

    raise RuntimeError(f"Twelve Data error for {interval} (tried {min(attempt + 1, n)} key(s)): {last_error}")


def fetch_candles_range(interval, start_date, end_date, credits_per_call=3):
    """
    Same key-rotation logic as fetch_candles(), but pulls an explicit
    date range instead of a fixed outputsize -- used by the backtester
    to request a specific historical window (e.g. "last 30 days") rather
    than "most recent N candles from right now".

    NOTE: Twelve Data generally caps a single response at 5000 candles
    regardless of the requested range. For 5-minute XAU/USD bars, 30
    days of ~24/5 trading can exceed that on some plans. If the returned
    range is shorter than requested, this function returns what it got
    rather than fabricating missing candles -- run_backtest() reports
    the actual covered window in its response so you can see if that
    happened.
    """
    if not TWELVE_DATA_API_KEYS:
        raise RuntimeError("No Twelve Data API key configured.")

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
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": 5000,
            "apikey": key,
            "order": "ASC",
            "timezone": FORCE_TIMEZONE,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
        except Exception as e:
            last_error = str(e)
            continue

        rate_limited = (
            r.status_code == 429
            or (isinstance(data, dict) and data.get("code") in (429, 8, 400) and "limit" in str(data.get("message", "")).lower())
        )

        if "values" in data:
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
            break

    raise RuntimeError(f"Twelve Data error for {interval} range backtest: {last_error}")


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
    return count - 1


def trend_arrow(d):
    return "▲" if d == 1 else "▼" if d == -1 else "→"


def is_weekend_bar(bar_dt):
    return bar_dt.weekday() >= 5


def is_outside_trading_hours(bar_dt):
    if not USE_TRADING_HOURS:
        return False
    minutes_now = bar_dt.hour * 60 + bar_dt.minute
    start_min = TRADING_START_HOUR * 60 + TRADING_START_MINUTE
    end_min = TRADING_END_HOUR * 60 + TRADING_END_MINUTE
    return not (start_min <= minutes_now <= end_min)


def is_new_entries_blocked(bar_dt):
    weekend = DISABLE_WEEKEND_SIGNALS and is_weekend_bar(bar_dt)
    outside_hours = is_outside_trading_hours(bar_dt)
    return weekend, outside_hours, (weekend or outside_hours)


# ---------------------- STATE ----------------------
DEFAULT_STATE = {
    "last_signal_bar": None,
    "last_closed_bar_time": None,
    "position": None,
    "history": [],
    "stats": {
        "total_trades": 0, "wins": 0, "losses": 0, "sum_pnl": 0.0,
        "best_trade": None, "worst_trade": None,
    },
    "current_day": None,
    "traded_today": False,
    "force_attempted_today": False,
    "force_skipped_today": False,
    "pending_dir": None,
    "pending_bar_time": None,
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
    val = request.args.get("notify", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _fmt_price(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def _fmt_money(v):
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else ''}{v:.2f}"


def build_check_telegram_message(result, state):
    lines = ["📊 [Manual Check] XAUUSD", f"Bar: {result.get('bar_time')} ({FORCE_TIMEZONE})"]

    event = result.get("event")
    if event == "entry":
        lines.append(f"Result: NEW {result.get('side')} signal opened this check {'(Forced Daily)' if result.get('forced') else '(Supertrend)'}.")
    elif event == "position_update":
        lines.append("Result: open position was updated this check (TP/SL/trail — see the alert above).")
    elif result.get("weekend_skipped"):
        lines.append("Result: weekend — new-signal checks skipped (existing position, if any, still managed).")
    elif result.get("trading_hours_skipped"):
        lines.append("Result: outside trading hours — new-signal checks skipped (existing position, if any, still managed).")
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
            f"TP2 {_fmt_price(pos['tp2'])} TP3 {_fmt_price(pos['tp3'])} "
            f"TP4 {_fmt_price(pos['tp4'])} | "
            f"remaining {round(pos['remaining_size'] * 100)}%"
        )
    else:
        lines.append("No open position.")

    return "\n".join(lines)


def build_stats_telegram_message(payload, state):
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

    th = payload.get("trading_hours", {})
    if th.get("enabled"):
        lines.append(
            f"Trading hours: {th['start_hour']:02d}:{th['start_minute']:02d}"
            f"-{th['end_hour']:02d}:{th['end_minute']:02d} {th['timezone']}"
        )

    return "\n".join(lines)


# ---------------------- TRADE LOG ----------------------
def log_trade(state, side, entry, sl, tp1, tp2, tp3, tp4, exit_price, result, points, pnl):
    state["history"].insert(0, {
        "side": side, "entry": entry, "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp3": tp3, "tp4": tp4, "exit": exit_price,
        "result": result, "points": round(points, 2), "pnl": round(pnl, 2),
    })
    state["history"] = state["history"][:50]
    state["stats"]["sum_pnl"] += pnl


def settle_trade(state, pos, exit_price, result_label, pnl_override=None):
    points = (exit_price - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - exit_price)
    pnl = pnl_override if pnl_override is not None else points * LOT_SIZE * UNITS_PER_LOT

    log_trade(state, "BUY" if pos["dir"] == 1 else "SELL",
              pos["entry"], pos["sl"], pos["tp1"], pos["tp2"], pos["tp3"], pos["tp4"],
              exit_price, result_label, points, pnl)

    stats = state["stats"]
    stats["total_trades"] += 1
    combined_positive = pnl >= 0
    if combined_positive:
        stats["wins"] += 1
    else:
        stats["losses"] += 1
    stats["best_trade"] = pnl if stats["best_trade"] is None else max(stats["best_trade"], pnl)
    stats["worst_trade"] = pnl if stats["worst_trade"] is None else min(stats["worst_trade"], pnl)

    state["position"] = None
    return pnl


# ---------------------- RATCHET HELPERS ----------------------
def ratchet_to_breakeven(pos):
    pos["tp1_hit"] = True
    pos["remaining_size"] = 0.75
    pos["sl"] = pos["entry"]


def ratchet_to_tp1(pos):
    pos["tp2_hit"] = True
    pos["remaining_size"] = 0.5
    pos["sl"] = pos["tp1"]


def ratchet_to_tp2(pos):
    pos["tp3_hit"] = True
    pos["remaining_size"] = 0.25
    pos["sl"] = pos["tp2"]


def trail_runner_sl(pos, st_value):
    if st_value is None or pd.isna(st_value):
        return
    if pos["dir"] == 1 and st_value > pos["sl"]:
        pos["sl"] = st_value
    elif pos["dir"] == -1 and st_value < pos["sl"]:
        pos["sl"] = st_value


# ---------------------- POSITION MANAGEMENT ----------------------
def manage_position(state, last_candle, st_value, silent=False):
    """
    silent=True is used by the backtester (run_backtest()) so replaying
    history doesn't fire real Telegram alerts. Live /check calls this
    with no third arg, so silent defaults to False and behavior there is
    unchanged.
    """
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
            elif high >= pos["tp4"]:
                pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Full Close)")
                events.append(("TP4 Hit (Full Close)", pos["tp4"], pnl))
            elif high >= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Full Close)")
                events.append(("TP3 Hit (Full Close)", pos["tp3"], pnl))
            elif high >= pos["tp2"]:
                pnl = settle_trade(state, pos, pos["tp2"], "TP2 Hit (Full Close)")
                events.append(("TP2 Hit (Full Close)", pos["tp2"], pnl))
            elif high >= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        else:
            if not pos["tp1_hit"]:
                if low <= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                    events.append(("SL Hit", pos["sl"], pnl))
                elif high >= pos["tp4"]:
                    pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Gap, Full Size)")
                    events.append(("TP4 Hit (Gap, Full Size)", pos["tp4"], pnl))
                elif high >= pos["tp3"]:
                    ratchet_to_breakeven(pos)
                    events.append(("TP1 Hit (SL->Entry)", pos["tp1"], None))
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
                    ratchet_to_tp2(pos)
                    events.append(("TP3 Hit (SL->TP2)", pos["tp3"], None))
                elif high >= pos["tp2"]:
                    ratchet_to_breakeven(pos)
                    events.append(("TP1 Hit (SL->Entry)", pos["tp1"], None))
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
                elif high >= pos["tp1"]:
                    ratchet_to_breakeven(pos)
                    events.append(("TP1 Hit (SL->Entry)", pos["tp1"], None))
            elif pos["tp1_hit"] and not pos["tp2_hit"]:
                if low <= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit (After TP1 — Breakeven, $0)")
                    events.append(("SL Hit (After TP1 — Breakeven, $0)", pos["sl"], pnl))
                elif high >= pos["tp4"]:
                    pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Gap)")
                    events.append(("TP4 Hit (Gap)", pos["tp4"], pnl))
                elif high >= pos["tp3"]:
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
                    ratchet_to_tp2(pos)
                    events.append(("TP3 Hit (SL->TP2)", pos["tp3"], None))
                elif high >= pos["tp2"]:
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
            elif pos["tp2_hit"] and not pos["tp3_hit"]:
                if low <= pos["sl"]:
                    points = pos["tp1"] - pos["entry"]
                    pnl = points * LOT_SIZE * UNITS_PER_LOT
                    settle_trade(state, pos, pos["sl"], "SL Hit (After TP1+TP2 — Locked at TP1)", pnl_override=pnl)
                    events.append(("SL Hit (After TP1+TP2 — Locked at TP1)", pos["sl"], pnl))
                elif high >= pos["tp4"]:
                    pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Gap)")
                    events.append(("TP4 Hit (Gap)", pos["tp4"], pnl))
                elif high >= pos["tp3"]:
                    ratchet_to_tp2(pos)
                    events.append(("TP3 Hit (SL->TP2)", pos["tp3"], None))
            else:
                if low <= pos["sl"]:
                    points = pos["tp2"] - pos["entry"]
                    pnl = points * LOT_SIZE * UNITS_PER_LOT
                    settle_trade(state, pos, pos["sl"], "SL Hit (After TP1+TP2+TP3 — Locked at TP2)", pnl_override=pnl)
                    events.append(("SL Hit (After TP1+TP2+TP3 — Locked at TP2)", pos["sl"], pnl))
                elif high >= pos["tp4"]:
                    points = pos["tp4"] - pos["entry"]
                    pnl = points * LOT_SIZE * UNITS_PER_LOT
                    settle_trade(state, pos, pos["tp4"], "TP4 Hit (Final Leg)", pnl_override=pnl)
                    events.append(("TP4 Hit (Final Leg)", pos["tp4"], pnl))

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
            elif low <= pos["tp4"]:
                pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Full Close)")
                events.append(("TP4 Hit (Full Close)", pos["tp4"], pnl))
            elif low <= pos["tp3"]:
                pnl = settle_trade(state, pos, pos["tp3"], "TP3 Hit (Full Close)")
                events.append(("TP3 Hit (Full Close)", pos["tp3"], pnl))
            elif low <= pos["tp2"]:
                pnl = settle_trade(state, pos, pos["tp2"], "TP2 Hit (Full Close)")
                events.append(("TP2 Hit (Full Close)", pos["tp2"], pnl))
            elif low <= pos["tp1"]:
                pnl = settle_trade(state, pos, pos["tp1"], "TP1 Hit (Full Close)")
                events.append(("TP1 Hit (Full Close)", pos["tp1"], pnl))
        else:
            if not pos["tp1_hit"]:
                if high >= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit")
                    events.append(("SL Hit", pos["sl"], pnl))
                elif low <= pos["tp4"]:
                    pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Gap, Full Size)")
                    events.append(("TP4 Hit (Gap, Full Size)", pos["tp4"], pnl))
                elif low <= pos["tp3"]:
                    ratchet_to_breakeven(pos)
                    events.append(("TP1 Hit (SL->Entry)", pos["tp1"], None))
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
                    ratchet_to_tp2(pos)
                    events.append(("TP3 Hit (SL->TP2)", pos["tp3"], None))
                elif low <= pos["tp2"]:
                    ratchet_to_breakeven(pos)
                    events.append(("TP1 Hit (SL->Entry)", pos["tp1"], None))
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
                elif low <= pos["tp1"]:
                    ratchet_to_breakeven(pos)
                    events.append(("TP1 Hit (SL->Entry)", pos["tp1"], None))
            elif pos["tp1_hit"] and not pos["tp2_hit"]:
                if high >= pos["sl"]:
                    pnl = settle_trade(state, pos, pos["sl"], "SL Hit (After TP1 — Breakeven, $0)")
                    events.append(("SL Hit (After TP1 — Breakeven, $0)", pos["sl"], pnl))
                elif low <= pos["tp4"]:
                    pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Gap)")
                    events.append(("TP4 Hit (Gap)", pos["tp4"], pnl))
                elif low <= pos["tp3"]:
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
                    ratchet_to_tp2(pos)
                    events.append(("TP3 Hit (SL->TP2)", pos["tp3"], None))
                elif low <= pos["tp2"]:
                    ratchet_to_tp1(pos)
                    events.append(("TP2 Hit (SL->TP1)", pos["tp2"], None))
            elif pos["tp2_hit"] and not pos["tp3_hit"]:
                if high >= pos["sl"]:
                    points = pos["entry"] - pos["tp1"]
                    pnl = points * LOT_SIZE * UNITS_PER_LOT
                    settle_trade(state, pos, pos["sl"], "SL Hit (After TP1+TP2 — Locked at TP1)", pnl_override=pnl)
                    events.append(("SL Hit (After TP1+TP2 — Locked at TP1)", pos["sl"], pnl))
                elif low <= pos["tp4"]:
                    pnl = settle_trade(state, pos, pos["tp4"], "TP4 Hit (Gap)")
                    events.append(("TP4 Hit (Gap)", pos["tp4"], pnl))
                elif low <= pos["tp3"]:
                    ratchet_to_tp2(pos)
                    events.append(("TP3 Hit (SL->TP2)", pos["tp3"], None))
            else:
                if high >= pos["sl"]:
                    points = pos["entry"] - pos["tp2"]
                    pnl = points * LOT_SIZE * UNITS_PER_LOT
                    settle_trade(state, pos, pos["sl"], "SL Hit (After TP1+TP2+TP3 — Locked at TP2)", pnl_override=pnl)
                    events.append(("SL Hit (After TP1+TP2+TP3 — Locked at TP2)", pos["sl"], pnl))
                elif low <= pos["tp4"]:
                    points = pos["entry"] - pos["tp4"]
                    pnl = points * LOT_SIZE * UNITS_PER_LOT
                    settle_trade(state, pos, pos["tp4"], "TP4 Hit (Final Leg)", pnl_override=pnl)
                    events.append(("TP4 Hit (Final Leg)", pos["tp4"], pnl))

    if pos["dir"] == 1:
        buy_side()
    else:
        sell_side()

    if not silent:
        side = "BUY" if pos["dir"] == 1 else "SELL"
        for label, price, pnl in events:
            if pnl is None:
                msg = (
                    f"XAUUSD {side} — {label}\n"
                    f"Price: {price:.2f}\n"
                    f"(Stop loss moved — no P&L booked yet, position still open)"
                )
            else:
                msg = (
                    f"XAUUSD {side} — {label}\n"
                    f"Price: {price:.2f}\n"
                    f"P&L (trade closed): {'+' if pnl >= 0 else ''}${pnl:.2f}"
                )
            send_telegram(msg)

    return len(events) > 0


def open_position(side, entry, sl, tp1, tp2, tp3, tp4, bar_time):
    return {
        "dir": 1 if side == "BUY" else -1,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "tp4": tp4,
        "entry_time": bar_time,
        "tp1_hit": False, "tp2_hit": False, "tp3_hit": False,
        "remaining_size": 1.0,
    }


# ---------------------- DAILY TRADE GUARANTEE HELPERS ----------------------
def roll_daily_guarantee_state(state, bar_dt):
    day_str = bar_dt.strftime("%Y-%m-%d")
    if state.get("current_day") != day_str:
        state["current_day"] = day_str
        state["traded_today"] = False
        state["force_attempted_today"] = False
        state["force_skipped_today"] = False


def compute_force_entry(state, bar_dt, st_dir, htf_dir, ema_fast_last, ema_slow_last,
                         rsi_val, extension_atr):
    if not GUARANTEE_DAILY_TRADE:
        return False, 0

    _, _, blocked = is_new_entries_blocked(bar_dt)
    if blocked:
        return False, 0

    if st_dir == 1 or st_dir == -1:
        force_direction = st_dir
    elif htf_dir == 1 or htf_dir == -1:
        force_direction = htf_dir
    else:
        force_direction = 1 if ema_fast_last >= ema_slow_last else -1

    is_past_force_time = (bar_dt.hour * 60 + bar_dt.minute) >= (FORCE_HOUR * 60 + FORCE_MINUTE)
    is_past_hard_time = (bar_dt.hour * 60 + bar_dt.minute) >= (FORCE_HARD_HOUR * 60 + FORCE_HARD_MINUTE)

    force_rsi_ok = (not USE_RSI) or (rsi_val < RSI_OB if force_direction == 1 else rsi_val > RSI_OS)
    force_extension_ok = (not USE_EXTENSION_FILTER) or extension_atr <= MAX_EXTENSION_ATR
    force_quality_ok = (not FORCE_REQUIRE_QUALITY_FILTERS) or (force_rsi_ok and force_extension_ok)

    if (is_past_hard_time and not force_quality_ok and FORCE_REQUIRE_QUALITY_FILTERS
            and FORCE_SKIP_IF_NEVER_VALID and not state["traded_today"]):
        state["force_skipped_today"] = True

    force_can_fire_now = force_quality_ok or (is_past_hard_time and not FORCE_SKIP_IF_NEVER_VALID)

    force_entry_now = (
        is_past_force_time
        and not state["force_attempted_today"]
        and not state["traded_today"]
        and state["position"] is None
        and force_can_fire_now
        and not state.get("force_skipped_today", False)
    )

    return force_entry_now, force_direction


# ---------------------- ENTRY TIMING HELPERS ----------------------
def bars_since_pending_armed(df, pending_bar_time):
    if pending_bar_time is None:
        return None
    target = pd.to_datetime(pending_bar_time)
    matches = df.index[df["datetime"] == target]
    if len(matches) == 0:
        return PULLBACK_TIMEOUT_BARS + 1
    return (len(df) - 1) - matches[0]


def roll_pullback_state(state, df, bar_time, st_bullish, st_bearish, htf_bullish, htf_bearish,
                         base_long_cond, base_short_cond, entries_blocked=False):
    if base_long_cond:
        state["pending_dir"] = 1
        state["pending_bar_time"] = bar_time
    if base_short_cond:
        state["pending_dir"] = -1
        state["pending_bar_time"] = bar_time

    bars_since = bars_since_pending_armed(df, state["pending_bar_time"])

    if state["pending_dir"] == 1:
        timed_out = bars_since is not None and bars_since > PULLBACK_TIMEOUT_BARS
        if not st_bullish or not htf_bullish or timed_out or entries_blocked:
            state["pending_dir"] = None
            state["pending_bar_time"] = None
    elif state["pending_dir"] == -1:
        timed_out = bars_since is not None and bars_since > PULLBACK_TIMEOUT_BARS
        if not st_bearish or not htf_bearish or timed_out or entries_blocked:
            state["pending_dir"] = None
            state["pending_bar_time"] = None


# ---------------------- BACKTEST ----------------------
def run_backtest(days=30):
    """
    Replays the bot's real entry/exit/session/risk logic over the last
    `days` days of history, using a fresh in-memory state (never touches
    state.json / your live position), and returns a winrate report --
    the equivalent of what the Pine Script shows on the TradingView
    chart, but computed from your bot's actual Twelve Data feed and
    actual live logic instead of FXCM chart data.
    """
    end_date = pd.Timestamp.utcnow()
    # pad the fetch window so warm-up-hungry indicators (EMA32, ATR12,
    # Supertrend, RSI16) are already stable by the time we reach the
    # actual `days`-ago cutoff -- otherwise the first ~50-100 bars of
    # the requested window would have garbage indicator values.
    fetch_start = end_date - pd.Timedelta(days=days + 6)

    df = fetch_candles_range(TIMEFRAME, fetch_start, end_date)
    if len(df) < 50:
        return {"error": f"Not enough candles returned ({len(df)}) to backtest. Try a shorter 'days' window."}

    df["emaFast"] = ema(df["close"], FAST_LEN)
    df["emaSlow"] = ema(df["close"], SLOW_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)
    st_series, dir_series = supertrend(df, ST_ATR_PERIOD, ST_FACTOR)
    df["st"] = st_series
    df["st_dir"] = dir_series

    if USE_HTF:
        htf_df = fetch_candles_range(HTF_TIMEFRAME, fetch_start, end_date)
        _, htf_dir_series = supertrend(htf_df, HTF_ATR_PERIOD, HTF_FACTOR)
        htf_df = htf_df[["datetime"]].copy()
        htf_df["htf_st_dir"] = htf_dir_series.values
        # align each entry-TF bar to the most recent CLOSED higher-TF bar
        df = pd.merge_asof(df.sort_values("datetime"), htf_df.sort_values("datetime"),
                            on="datetime", direction="backward")
        df["htf_st_dir"] = df["htf_st_dir"].fillna(0).astype(int)
    else:
        df["htf_st_dir"] = 0

    cutoff = end_date - pd.Timedelta(days=days)
    cutoff_matches = df.index[df["datetime"] >= cutoff]
    start_idx = int(cutoff_matches[0]) if len(cutoff_matches) else max(0, len(df) - 1)
    start_idx = max(start_idx, 1)  # need a previous bar for cross detection

    bt_state = {
        "position": None,
        "history": [],
        "pending_dir": None,
        "pending_bar_time": None,
        "current_day": None,
        "traded_today": False,
        "force_attempted_today": False,
        "force_skipped_today": False,
        "stats": {"total_trades": 0, "wins": 0, "losses": 0, "sum_pnl": 0.0,
                   "best_trade": None, "worst_trade": None},
    }

    for i in range(start_idx, len(df)):
        bar = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_dt = bar["datetime"]

        roll_daily_guarantee_state(bt_state, bar_dt)

        if bt_state["position"] is not None:
            manage_position(bt_state, bar, bar["st"], silent=True)

        weekend_now, outside_hours_now, blocked_now = is_new_entries_blocked(bar_dt)

        if blocked_now:
            if bt_state["pending_dir"] is not None:
                bt_state["pending_dir"] = None
                bt_state["pending_bar_time"] = None
            continue

        if bt_state["position"] is not None:
            continue  # already in a trade this bar, no new entry evaluation

        ema_cross_up = prev["emaFast"] <= prev["emaSlow"] and bar["emaFast"] > bar["emaSlow"]
        ema_cross_down = prev["emaFast"] >= prev["emaSlow"] and bar["emaFast"] < bar["emaSlow"]
        rsi_ok_long = (not USE_RSI) or bar["rsi"] < RSI_OB
        rsi_ok_short = (not USE_RSI) or bar["rsi"] > RSI_OS
        st_bullish = bar["st_dir"] == 1
        st_bearish = bar["st_dir"] == -1

        bars_since_flip = bars_since_supertrend_flip(dir_series.iloc[:i + 1])
        st_flip_confirmed = bars_since_flip >= ST_CONFIRM_BARS

        h_dir = int(bar["htf_st_dir"])
        htf_bullish = (not USE_HTF) or h_dir == 1
        htf_bearish = (not USE_HTF) or h_dir == -1

        extension_atr = (abs(bar["close"] - bar["emaFast"]) / bar["atr"]) if bar["atr"] > 0 else 0.0
        extension_ok = (not USE_EXTENSION_FILTER) or extension_atr <= MAX_EXTENSION_ATR
        pullback_ok = extension_atr <= PULLBACK_MAX_ATR

        base_long_cond = ema_cross_up and rsi_ok_long and st_bullish and st_flip_confirmed and htf_bullish
        base_short_cond = ema_cross_down and rsi_ok_short and st_bearish and st_flip_confirmed and htf_bearish

        roll_pullback_state(
            bt_state, df.iloc[:i + 1], str(bar_dt), st_bullish, st_bearish, htf_bullish, htf_bearish,
            base_long_cond, base_short_cond, entries_blocked=blocked_now,
        )

        if USE_PULLBACK_ENTRY:
            long_cond = (extension_ok and bt_state["pending_dir"] == 1 and pullback_ok
                         and st_bullish and htf_bullish and rsi_ok_long)
            short_cond = (extension_ok and bt_state["pending_dir"] == -1 and pullback_ok
                          and st_bearish and htf_bearish and rsi_ok_short)
        else:
            long_cond = extension_ok and base_long_cond
            short_cond = extension_ok and base_short_cond

        force_entry_now, force_direction = compute_force_entry(
            bt_state, bar_dt, int(bar["st_dir"]), h_dir,
            bar["emaFast"], bar["emaSlow"], bar["rsi"], extension_atr,
        )
        force_long_cond = force_entry_now and force_direction == 1
        force_short_cond = force_entry_now and force_direction == -1

        is_long_entry = long_cond or force_long_cond
        is_short_entry = short_cond or force_short_cond

        if is_long_entry or is_short_entry:
            entry = bar["close"]
            sl_dist = min(max(bar["atr"] * SL_MULT, SL_MIN_PTS), SL_MAX_PTS)
            if is_long_entry:
                sl = entry - sl_dist
                risk = entry - sl
                tp1, tp2, tp3, tp4 = entry + risk * RR1, entry + risk * RR2, entry + risk * RR3, entry + risk * RR4
                side = "BUY"
            else:
                sl = entry + sl_dist
                risk = sl - entry
                tp1, tp2, tp3, tp4 = entry - risk * RR1, entry - risk * RR2, entry - risk * RR3, entry - risk * RR4
                side = "SELL"

            bt_state["position"] = open_position(side, entry, sl, tp1, tp2, tp3, tp4, str(bar_dt))
            bt_state["traded_today"] = True
            bt_state["force_attempted_today"] = True
            bt_state["pending_dir"] = None
            bt_state["pending_bar_time"] = None
        elif force_entry_now:
            bt_state["force_attempted_today"] = True

    s = bt_state["stats"]
    win_rate = (s["wins"] / s["total_trades"] * 100) if s["total_trades"] else 0

    return {
        "requested_days": days,
        "data_covers_from": str(df.iloc[start_idx]["datetime"]),
        "data_covers_to": str(df.iloc[-1]["datetime"]),
        "total_candles_used": len(df) - start_idx,
        "total_trades": s["total_trades"],
        "wins": s["wins"],
        "losses": s["losses"],
        "win_rate": round(win_rate, 1),
        "net_pnl": round(s["sum_pnl"], 2),
        "best_trade": s["best_trade"],
        "worst_trade": s["worst_trade"],
        "trade_log": bt_state["history"],
        "position_still_open_at_end": bt_state["position"],
    }


# ---------------------- CORE CHECK ----------------------
@app.route("/check", methods=["GET"])
def check():
    if not TWELVE_DATA_API_KEYS or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    df = fetch_candles(TIMEFRAME, outputsize=500)
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

    # ---------------------------------------------------------------
    # LAG FIX (v2): the previous version of this endpoint (and the first
    # patch of this fix) deduped position-management by bar timestamp --
    # "if we've already processed this bar_time, skip." That's wrong for
    # the CURRENT bar: Twelve Data returns the still-forming candle with a
    # live-updating high/low as price moves inside it, so a 5min timeframe
    # polled every 2min shows the SAME bar timestamp 2-3 times before that
    # candle finally closes. Deduping by timestamp meant only the FIRST
    # poll of that candle ever checked its high/low against SL/TP -- every
    # later poll (where the candle's low had actually dropped further, or
    # high risen further) was skipped entirely. That's exactly how an SL
    # can visibly trade through on the live chart while the bot still
    # shows the position open.
    #
    # Fix: split management into two passes.
    #   (a) Backfill fully CLOSED bars we haven't processed yet (covers a
    #       poller outage/slow cron skipping whole bars) -- deduped by
    #       state["last_closed_bar_time"] since closed bars never change.
    #   (b) ALWAYS re-check the current (possibly still-forming) last bar
    #       on every single poll, no dedup -- this is what actually catches
    #       an SL/TP touched mid-candle. manage_position() is safe to call
    #       repeatedly: it only acts on tp1_hit/tp2_hit/tp3_hit flags and a
    #       monotonically growing high/low, so re-checking the same
    #       (bigger) high/low again can't double-fire or un-trigger
    #       anything.
    # ---------------------------------------------------------------
    last_closed_str = state.get("last_closed_bar_time")
    if last_closed_str and len(df) > 1:
        last_closed_ts = pd.to_datetime(last_closed_str)
        closed_backlog = df.iloc[:-1]
        closed_backlog = closed_backlog[closed_backlog["datetime"] > last_closed_ts]
    else:
        closed_backlog = df.iloc[0:0]

    any_managed_event = False

    # (a) backfill any closed bars we haven't processed yet
    for _, bar in closed_backlog.iterrows():
        roll_daily_guarantee_state(state, bar["datetime"])
        if state["position"] is not None:
            any_managed_event = manage_position(state, bar, bar["st"]) or any_managed_event
        state["last_closed_bar_time"] = str(bar["datetime"])

    # (b) always re-check the current/latest bar, dedup or not
    roll_daily_guarantee_state(state, last["datetime"])
    if state["position"] is not None:
        any_managed_event = manage_position(state, last, last["st"]) or any_managed_event

    if any_managed_event:
        result["event"] = "position_update"
    save_state(state)

    # 1b) Session guard: don't evaluate/open any NEW signal (organic or
    #     forced) while the current bar is a Saturday/Sunday in
    #     FORCE_TIMEZONE, OR while it's outside the configured trading-hours
    #     window. Still update last_signal_bar so we don't just spin
    #     re-checking the same closed bar. Also explicitly clears any
    #     pending pullback signal that was armed before the block started.
    weekend_now, outside_hours_now, blocked_now = is_new_entries_blocked(last["datetime"])
    if blocked_now:
        state["last_signal_bar"] = bar_time
        if state.get("pending_dir") is not None:
            state["pending_dir"] = None
            state["pending_bar_time"] = None
        result["weekend_skipped"] = weekend_now
        result["trading_hours_skipped"] = outside_hours_now
        save_state(state)

    # 2) Only look for a NEW entry if we're currently flat, it's not a bar
    #    we've already processed, and it's not weekend/outside trading hours.
    if (not blocked_now) and state["position"] is None and state.get("last_signal_bar") != bar_time:
        prev = df.iloc[-2]

        ema_cross_up = prev["emaFast"] <= prev["emaSlow"] and last["emaFast"] > last["emaSlow"]
        ema_cross_down = prev["emaFast"] >= prev["emaSlow"] and last["emaFast"] < last["emaSlow"]

        rsi_ok_long = (not USE_RSI) or last["rsi"] < RSI_OB
        rsi_ok_short = (not USE_RSI) or last["rsi"] > RSI_OS

        st_bullish = last["st_dir"] == 1
        st_bearish = last["st_dir"] == -1

        bars_since_flip = bars_since_supertrend_flip(dir_series)
        st_flip_confirmed = bars_since_flip >= ST_CONFIRM_BARS

        htf_dir = 0
        if USE_HTF:
            htf_df = fetch_candles(HTF_TIMEFRAME, outputsize=500)
            _, htf_dir_series = supertrend(htf_df, HTF_ATR_PERIOD, HTF_FACTOR)
            htf_dir = int(htf_dir_series.iloc[-1])
        htf_bullish = (not USE_HTF) or htf_dir == 1
        htf_bearish = (not USE_HTF) or htf_dir == -1

        extension_atr = (abs(last["close"] - last["emaFast"]) / last["atr"]) if last["atr"] > 0 else 0.0
        extension_ok = (not USE_EXTENSION_FILTER) or extension_atr <= MAX_EXTENSION_ATR

        pullback_dist_atr = (abs(last["close"] - last["emaFast"]) / last["atr"]) if last["atr"] > 0 else 0.0
        pullback_ok = pullback_dist_atr <= PULLBACK_MAX_ATR

        base_long_cond = ema_cross_up and rsi_ok_long and st_bullish and st_flip_confirmed and htf_bullish
        base_short_cond = ema_cross_down and rsi_ok_short and st_bearish and st_flip_confirmed and htf_bearish

        roll_pullback_state(
            state, df, bar_time, st_bullish, st_bearish, htf_bullish, htf_bearish,
            base_long_cond, base_short_cond, entries_blocked=blocked_now,
        )

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

        force_entry_now, force_direction = compute_force_entry(
            state, last["datetime"], int(last["st_dir"]), htf_dir,
            last["emaFast"], last["emaSlow"],
            last["rsi"], extension_atr,
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
                tp1 = entry + risk * RR1
                tp2 = entry + risk * RR2
                tp3 = entry + risk * RR3
                tp4 = entry + risk * RR4
                side = "BUY"
                is_forced = force_long_cond and not long_cond
            else:
                sl = entry + sl_dist
                risk = sl - entry
                tp1 = entry - risk * RR1
                tp2 = entry - risk * RR2
                tp3 = entry - risk * RR3
                tp4 = entry - risk * RR4
                side = "SELL"
                is_forced = force_short_cond and not short_cond

            state["position"] = open_position(side, entry, sl, tp1, tp2, tp3, tp4, bar_time)
            state["traded_today"] = True
            state["force_attempted_today"] = True
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
                f"TP4: {tp4:.2f}\n"
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
            result["tp4"] = tp4
        elif force_entry_now:
            state["force_attempted_today"] = True

        save_state(state)

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
            "force_skipped_today": state.get("force_skipped_today", False),
            "require_quality_filters": FORCE_REQUIRE_QUALITY_FILTERS,
            "hard_cutoff_hour": FORCE_HARD_HOUR,
            "hard_cutoff_minute": FORCE_HARD_MINUTE,
            "skip_if_never_valid": FORCE_SKIP_IF_NEVER_VALID,
        },
        "pending_signal": {
            "dir": state.get("pending_dir"),
            "armed_bar_time": state.get("pending_bar_time"),
        },
        "weekend_guard": {
            "enabled": DISABLE_WEEKEND_SIGNALS,
        },
        "trading_hours": {
            "enabled": USE_TRADING_HOURS,
            "start_hour": TRADING_START_HOUR,
            "start_minute": TRADING_START_MINUTE,
            "end_hour": TRADING_END_HOUR,
            "end_minute": TRADING_END_MINUTE,
            "timezone": FORCE_TIMEZONE,
        },
    }

    if _notify_requested():
        send_telegram(build_stats_telegram_message(payload, state))
        payload["telegram_notified"] = True

    return jsonify(payload)


@app.route("/backtest", methods=["GET"])
def backtest():
    """
    GET /backtest              -> last 30 days (default)
    GET /backtest?days=14      -> last 14 days
    GET /backtest?notify=true  -> also posts a summary to Telegram

    Replays the bot's real signal/risk logic over historical Twelve Data
    candles and reports the winrate, trade count, net P&L, and full
    trade log -- the equivalent of what the Pine Script shows on the
    TradingView chart, but from the bot's own data feed and own logic.
    """
    if not TWELVE_DATA_API_KEYS:
        return jsonify({"error": "Missing Twelve Data API key"}), 500

    days = int(request.args.get("days", 30))
    try:
        result = run_backtest(days=days)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if "error" in result:
        return jsonify(result), 400

    if _notify_requested():
        msg = (
            f"📊 [Backtest] XAUUSD — last {result['requested_days']}d\n"
            f"Window: {result['data_covers_from']} -> {result['data_covers_to']}\n"
            f"Win rate: {result['win_rate']}% "
            f"({result['wins']}W / {result['losses']}L / {result['total_trades']} total)\n"
            f"Net P&L: {'+' if result['net_pnl'] >= 0 else ''}${result['net_pnl']:.2f}"
        )
        send_telegram(msg)
        result["telegram_notified"] = True

    return jsonify(result)


@app.route("/test", methods=["GET"])
def test_signal():
    if not TWELVE_DATA_API_KEYS or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    df = fetch_candles(TIMEFRAME, outputsize=100)
    df["atr"] = atr(df, ATR_LEN)
    last = df.iloc[-1]

    entry = last["close"]
    sl_dist = min(max(last["atr"] * SL_MULT, SL_MIN_PTS), SL_MAX_PTS)
    sl = entry - sl_dist
    risk = entry - sl
    tp1 = entry + risk * RR1
    tp2 = entry + risk * RR2
    tp3 = entry + risk * RR3
    tp4 = entry + risk * RR4

    msg = (
        f"[TEST] XAUUSD BUY signal (Supertrend)\n"
        f"Entry: {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f}\n"
        f"TP2: {tp2:.2f}\n"
        f"TP3: {tp3:.2f}\n"
        f"TP4: {tp4:.2f}\n"
        f"Bar: {last['datetime']} ({FORCE_TIMEZONE})\n"
        f"(This is a forced test message, not a real signal)"
    )
    send_telegram(msg)
    return jsonify({"status": "test message sent", "message": msg})


@app.route("/keys", methods=["GET"])
def keys_status():
    ks = _load_key_state()
    return jsonify({
        "configured_keys": len(TWELVE_DATA_API_KEYS),
        "date_utc": ks["date"],
        "next_key_index": ks["next_index"] + 1,
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
        "force_require_quality_filters": FORCE_REQUIRE_QUALITY_FILTERS,
        "force_hard_hour": FORCE_HARD_HOUR if FORCE_REQUIRE_QUALITY_FILTERS else None,
        "force_hard_minute": FORCE_HARD_MINUTE if FORCE_REQUIRE_QUALITY_FILTERS else None,
        "force_skip_if_never_valid": FORCE_SKIP_IF_NEVER_VALID,
        "disable_weekend_signals": DISABLE_WEEKEND_SIGNALS,
        "use_trading_hours": USE_TRADING_HOURS,
        "trading_start": f"{TRADING_START_HOUR:02d}:{TRADING_START_MINUTE:02d}" if USE_TRADING_HOURS else None,
        "trading_end": f"{TRADING_END_HOUR:02d}:{TRADING_END_MINUTE:02d}" if USE_TRADING_HOURS else None,
        "dashboard": "/dashboard",
    })


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
  .grid5{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;}
  @media (max-width:560px){.grid4{grid-template-columns:repeat(2,1fr);} .grid5{grid-template-columns:repeat(2,1fr);}}
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

function eventBadge(event, weekendSkipped, hoursSkipped){
  if(event === 'entry') return `<span class="badge neutral">New Entry</span>`;
  if(event === 'position_update') return `<span class="badge muted">Position Updated</span>`;
  if(weekendSkipped) return `<span class="badge muted">Weekend — Skipped</span>`;
  if(hoursSkipped) return `<span class="badge muted">Outside Trading Hours</span>`;
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
    <div class="kv"><span class="k">TP1 / TP2 / TP3 / TP4</span><span class="v">${fmtPrice(pos.tp1)} / ${fmtPrice(pos.tp2)} / ${fmtPrice(pos.tp3)} / ${fmtPrice(pos.tp4)}</span></div>
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
      <div class="grid5">
        <div class="stat-box"><div class="k">Side</div><div class="v side-tag ${sideClass}">${data.side}${data.forced ? ' ⚡' : ''}</div></div>
        <div class="stat-box"><div class="k">Entry</div><div class="v gold">${fmtPrice(data.entry)}</div></div>
        <div class="stat-box"><div class="k">Stop Loss</div><div class="v neg">${fmtPrice(data.sl)}</div></div>
        <div class="stat-box"><div class="k">TP1 / TP2</div><div class="v pos" style="font-size:12.5px">${fmtPrice(data.tp1)} · ${fmtPrice(data.tp2)}</div></div>
        <div class="stat-box"><div class="k">TP3 / TP4</div><div class="v pos" style="font-size:12.5px">${fmtPrice(data.tp3)} · ${fmtPrice(data.tp4)}</div></div>
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

  let sessionBlock = '';
  if(data.weekend_skipped){
    sessionBlock = `<p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">🌙 Weekend — new-signal checks are paused. Any open position is still tracked.</p>`;
  }else if(data.trading_hours_skipped){
    sessionBlock = `<p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">🌙 Outside trading hours — new-signal checks are paused. Any open position is still tracked.</p>`;
  }

  document.getElementById('results').innerHTML = `
    <div class="panel">
      <div class="panel-head">
        <div class="title">Check Result ${eventBadge(data.event, data.weekend_skipped, data.trading_hours_skipped)} ${notifiedBadge}</div>
        <div class="timestamp">${esc(data.bar_time || '—')}</div>
      </div>
      <div class="panel-body">
        ${entryBlock}
        ${sessionBlock}
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
  const th = data.trading_hours || {};

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

        <div class="section-label">Trading Hours</div>
        <div class="kv"><span class="k">Window</span><span class="v">${th.enabled ? `${String(th.start_hour).padStart(2,'0')}:${String(th.start_minute).padStart(2,'0')}-${String(th.end_hour).padStart(2,'0')}:${String(th.end_minute).padStart(2,'0')}` : 'Off (24h)'}</span></div>

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
