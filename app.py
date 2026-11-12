import os
import json
import math
import calendar
import numpy as np
import requests
import pandas as pd
from flask import Flask, jsonify, request, Response

app = Flask(__name__)
from ea_control import register_ea_control
inject_ea_ui = register_ea_control(app)

# =====================================================================
#  SYNCED WITH "Gold Signal Terminal v2 — Trend + SMC + Flow Proxy"
#  What changed vs the previous bot:
#   - NEW: per-method ON/OFF switches ENABLE_TREND / ENABLE_SMC /
#     ENABLE_FORCE (mirror the indicator's "Trade Methods" group).
#     ENABLE_SMC falls back to the old USE_SMC_TRIGGER env var and
#     ENABLE_FORCE to the old GUARANTEE_DAILY_TRADE env var, so existing
#     Render settings keep working.
#   - FIXED: PNL_MODE read os.environ.get("tp1_only") (an env var literally
#     named "tp1_only"), so it was always None -> silently ratcheted.
#     It now reads the PNL_MODE env var, default "ratchet".
#   - FIXED: defaults that had drifted from the indicator are back in sync
#     (trading hours 06:30-23:59, month-end pause 7d, sweep lookback 8,
#     max extension 2.0, SL clamp 8-20 [was 10-10 so ATR did nothing],
#     spread 0.3, lot 0.05). Override any of them with env vars.
#   - Defaults equal the v2 indicator inputs (EMA 30/34, RSI 20,
#     Supertrend 8/2.0, HTF 2h ATR 4 / factor 8, SL = ATR x1.5, RR 2.0 / 4.5,
#     lock +5 ...).
#   - Only TP1 / TP2. Ratchet mode = lock SL at entry +/- LOCK_PTS after
#     TP1, close the whole trade at TP2.
#   - Full SMC engine ported: swing pivots, BOS, liquidity sweeps, FVG,
#     order blocks, premium/discount, tick-volume delta proxy -> 0-5 score.
#   - Trend entries are gated by the SMC score; standalone SMC sweep
#     entries; SMC trades use structure SL (beyond sweep wick).
#   - Month-end pause, risk guard (max losses/day + cooldown bars).
#   - Spread/cost deducted from every trade's points.
#   - No exit on the entry bar; wins = pnl > 0.
#   - HTF direction uses the PREVIOUS completed HTF candle ([1] in Pine).
# =====================================================================


def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() == "true"


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


# ---------------------- CONFIG (env vars, set these in Render) ----------------------
CAPITAL_API_KEY    = os.environ.get("CAPITAL_API_KEY", "")
CAPITAL_IDENTIFIER = os.environ.get("CAPITAL_IDENTIFIER", "")
CAPITAL_PASSWORD   = os.environ.get("CAPITAL_PASSWORD", "")
CAPITAL_BASE_URL   = os.environ.get("CAPITAL_BASE_URL", "https://demo-api-capital.backend-capital.com")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
SYMBOL             = os.environ.get("SYMBOL", "GOLD")
TIMEFRAME          = os.environ.get("TIMEFRAME", "5min")

# Used for the session filters, day filter, month-end pause AND the daily
# guarantee (the indicator's "Session Timezone" and "Force-Entry Timezone"
# are both Asia/Phnom_Penh by default).
FORCE_TIMEZONE = os.environ.get("FORCE_TIMEZONE", "Asia/Phnom_Penh")

# ---------------------- TRADE METHODS (ON/OFF) ----------------------
ENABLE_TREND = _env_bool("ENABLE_TREND", False)
ENABLE_SMC   = _env_bool("ENABLE_SMC",   _env_bool("USE_SMC_TRIGGER", True))
ENABLE_FORCE = _env_bool("ENABLE_FORCE", _env_bool("GUARANTEE_DAILY_TRADE", True))

# ---------------------- SESSION FILTERS ----------------------
DISABLE_WEEKEND_SIGNALS = _env_bool("DISABLE_WEEKEND_SIGNALS", True)

USE_TRADING_HOURS    = _env_bool("USE_TRADING_HOURS", True)
TRADING_START_HOUR   = _env_int("TRADING_START_HOUR", 7)
TRADING_START_MINUTE = _env_int("TRADING_START_MINUTE", 00)
TRADING_END_HOUR     = _env_int("TRADING_END_HOUR", 23)
TRADING_END_MINUTE   = _env_int("TRADING_END_MINUTE", 00)

USE_DAY_FILTER = _env_bool("USE_DAY_FILTER", True)
TRADE_MON = _env_bool("TRADE_MON", False)
TRADE_TUE = _env_bool("TRADE_TUE", True)
TRADE_WED = _env_bool("TRADE_WED", True)
TRADE_THU = _env_bool("TRADE_THU", True)
TRADE_FRI = _env_bool("TRADE_FRI", True)
TRADE_SAT = _env_bool("TRADE_SAT", False)
TRADE_SUN = _env_bool("TRADE_SUN", False)

_DAY_FLAGS_BY_WEEKDAY = {
    0: TRADE_MON, 1: TRADE_TUE, 2: TRADE_WED, 3: TRADE_THU,
    4: TRADE_FRI, 5: TRADE_SAT, 6: TRADE_SUN,
}
_DAY_LABEL_FLAGS = [
    ("Monday", TRADE_MON), ("Tuesday", TRADE_TUE), ("Wednesday", TRADE_WED),
    ("Thursday", TRADE_THU), ("Friday", TRADE_FRI), ("Saturday", TRADE_SAT),
    ("Sunday", TRADE_SUN),
]

USE_MONTH_END_PAUSE   = _env_bool("USE_MONTH_END_PAUSE", True)
MONTH_END_PAUSE_DAYS  = _env_int("MONTH_END_PAUSE_DAYS", 1)

# ---------------------- SIGNAL ENGINE ----------------------
FAST_LEN = _env_int("FAST_LEN", 30)
SLOW_LEN = _env_int("SLOW_LEN", 34)
USE_RSI  = _env_bool("USE_RSI", True)
RSI_LEN  = _env_int("RSI_LEN", 20)
RSI_OB   = _env_int("RSI_OB", 80)
RSI_OS   = _env_int("RSI_OS", 40)

ST_ATR_PERIOD   = _env_int("ST_ATR_PERIOD", 8)
ST_FACTOR       = _env_float("ST_FACTOR", 2.0)
ST_CONFIRM_BARS = _env_int("ST_CONFIRM_BARS", 1)

USE_HTF        = _env_bool("USE_HTF", True)
HTF_TIMEFRAME  = os.environ.get("HTF_TIMEFRAME", "2h")
HTF_ATR_PERIOD = _env_int("HTF_ATR_PERIOD", 4)
HTF_FACTOR     = _env_float("HTF_FACTOR", 8)

# ---------------------- SMC ----------------------
USE_SMC_GATE      = _env_bool("USE_SMC_GATE", True)
GATE_MIN_SCORE    = _env_int("GATE_MIN_SCORE", 3)
TRIG_MIN_SCORE    = _env_int("TRIG_MIN_SCORE", 3)
SWING_LEN         = _env_int("SWING_LEN", 7)
SWEEP_LOOKBACK    = _env_int("SWEEP_LOOKBACK", 6)
FVG_MIN_ATR       = _env_float("FVG_MIN_ATR", 0.3)
ZONE_MAX_AGE      = _env_int("ZONE_MAX_AGE", 60)
PD_LEN            = _env_int("PD_LEN", 60)
USE_STRUCTURE_SL  = _env_bool("USE_STRUCTURE_SL", True)
SL_BUFFER_PTS     = _env_float("SL_BUFFER_PTS", 1.0)

USE_FLOW          = _env_bool("USE_FLOW", True)
FLOW_LEN          = _env_int("FLOW_LEN", 1.5)
REQUIRE_VOL_SPIKE = _env_bool("REQUIRE_VOL_SPIKE", True)
VOL_MULT          = _env_float("VOL_MULT", 1.5)

# ---------------------- ENTRY TIMING ----------------------
USE_PULLBACK_ENTRY    = _env_bool("USE_PULLBACK_ENTRY", True)
PULLBACK_MAX_ATR      = _env_float("PULLBACK_MAX_ATR", 1.2)
PULLBACK_TIMEOUT_BARS = _env_int("PULLBACK_TIMEOUT_BARS", 4)
USE_EXTENSION_FILTER  = _env_bool("USE_EXTENSION_FILTER", True)
MAX_EXTENSION_ATR     = _env_float("MAX_EXTENSION_ATR", 2.0)

# ---------------------- DAILY TRADE GUARANTEE (FORCED method settings) ----------------------
FORCE_HOUR   = _env_int("FORCE_HOUR", 9)
FORCE_MINUTE = _env_int("FORCE_MINUTE", 0)
FORCE_REQUIRE_QUALITY_FILTERS = _env_bool("FORCE_REQUIRE_QUALITY_FILTERS", True)
FORCE_HARD_HOUR   = _env_int("FORCE_HARD_HOUR", 23)
FORCE_HARD_MINUTE = _env_int("FORCE_HARD_MINUTE", 45)
FORCE_SKIP_IF_NEVER_VALID = _env_bool("FORCE_SKIP_IF_NEVER_VALID", False)

# ---------------------- RISK MANAGEMENT ----------------------
ATR_LEN    = _env_int("ATR_LEN", 20)
SL_MULT    = _env_float("SL_MULT", 1.5)
SL_MIN_PTS = _env_float("SL_MIN_PTS", 10)
SL_MAX_PTS = _env_float("SL_MAX_PTS", 10)
RR1 = _env_float("RR1", 2.0)
RR2 = _env_float("RR2", 3.0)
LOCK_PTS   = _env_float("LOCK_PTS", 10)
SPREAD_PTS = _env_float("SPREAD_PTS", 0)

# "ratchet"   = lock SL after TP1, close whole trade at TP2   (indicator default)
# "tp1_only"  = full size closes at TP1
# "first_hit" = full size closes at whichever TP is hit first
PNL_MODE = os.environ.get("PNL_MODE", "ratchet").strip().lower()
if PNL_MODE not in ("ratchet", "tp1_only", "first_hit"):
    PNL_MODE = "tp1_only"

MAX_LOSSES_PER_DAY = _env_int("MAX_LOSSES_PER_DAY", 2)   # 0 = unlimited
COOLDOWN_BARS      = _env_int("COOLDOWN_BARS", 6)

LOT_SIZE      = _env_float("LOT_SIZE", 0.1)
UNITS_PER_LOT = _env_float("UNITS_PER_LOT", 100)

STATE_FILE = "state.json"

# ---------------------- CAPITAL.COM DATA FETCHING ----------------------
_CAPITAL_RESOLUTION_MAP = {
    "1min": "MINUTE", "5min": "MINUTE_5", "15min": "MINUTE_15", "30min": "MINUTE_30",
    "45min": "MINUTE_30",
    "1h": "HOUR", "4h": "HOUR_4",
    "1day": "DAY", "1week": "WEEK",
}
_RESAMPLE_FROM = {"2h": ("1h", 2)}
_INTERVAL_MINUTES = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30, "45min": 30,
    "1h": 60, "2h": 120, "4h": 240, "1day": 1440, "1week": 10080,
}


def _capital_resolution(interval):
    return _CAPITAL_RESOLUTION_MAP.get(interval, interval)


def _capital_session_headers():
    if not (CAPITAL_API_KEY and CAPITAL_IDENTIFIER and CAPITAL_PASSWORD):
        raise RuntimeError(
            "Missing Capital.com credentials. Set CAPITAL_API_KEY, "
            "CAPITAL_IDENTIFIER (your login/email), and CAPITAL_PASSWORD."
        )
    url = f"{CAPITAL_BASE_URL}/api/v1/session"
    headers = {"X-CAP-API-KEY": CAPITAL_API_KEY, "Content-Type": "application/json"}
    body = {"identifier": CAPITAL_IDENTIFIER, "password": CAPITAL_PASSWORD, "encryptedPassword": False}
    r = requests.post(url, headers=headers, json=body, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Capital.com session error ({r.status_code}): {r.text}")
    cst = r.headers.get("CST")
    security_token = r.headers.get("X-SECURITY-TOKEN")
    if not cst or not security_token:
        raise RuntimeError("Capital.com session response was missing CST/X-SECURITY-TOKEN headers.")
    return {"CST": cst, "X-SECURITY-TOKEN": security_token}


def _local_naive_to_utc_naive(dt_local):
    ts = pd.Timestamp(dt_local)
    if ts.tzinfo is None:
        ts = ts.tz_localize(FORCE_TIMEZONE)
    return ts.tz_convert("UTC").tz_localize(None).to_pydatetime()


def _capital_prices_to_df(payload):
    prices = payload.get("prices", [])
    if not prices:
        raise RuntimeError("Capital.com returned no candles for this request.")
    rows = []
    for p in prices:
        o, c, h, l = p["openPrice"], p["closePrice"], p["highPrice"], p["lowPrice"]
        rows.append({
            "datetime": p["snapshotTimeUTC"],
            "open": o["bid"], "high": h["bid"], "low": l["bid"], "close": c["bid"],
            "volume": p.get("lastTradedVolume", 0) or 0,
        })
    df = pd.DataFrame(rows)
    df["datetime"] = (
        pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
        .dt.tz_convert(FORCE_TIMEZONE).dt.tz_localize(None)
    )
    return df.sort_values("datetime").reset_index(drop=True)


def _resample_candles(df, hours):
    tmp = df.copy()
    tmp["utc"] = (
        tmp["datetime"]
        .dt.tz_localize(FORCE_TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    tmp = tmp.dropna(subset=["utc"]).set_index("utc")
    rule = f"{hours}h"
    agg = tmp.resample(rule, origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = tmp["close"].resample(rule, origin="start_day").count()
    agg = agg[counts > 0].dropna()
    counts = counts[counts > 0]
    if len(agg) > 1 and counts.iloc[0] < hours:
        agg = agg.iloc[1:]
    agg.index = agg.index.tz_convert(FORCE_TIMEZONE).tz_localize(None)
    agg.index.name = "datetime"
    return agg.reset_index()[["datetime", "open", "high", "low", "close", "volume"]]


def fetch_candles(interval, outputsize=500):
    if interval in _RESAMPLE_FROM:
        base_interval, factor = _RESAMPLE_FROM[interval]
        base_df = fetch_candles(base_interval, outputsize=min(outputsize * factor, 1000))
        return _resample_candles(base_df, factor)
    headers = _capital_session_headers()
    params = {"resolution": _capital_resolution(interval), "max": min(outputsize, 1000)}
    r = requests.get(f"{CAPITAL_BASE_URL}/api/v1/prices/{SYMBOL}", headers=headers, params=params, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Capital.com prices error ({r.status_code}): {r.text}")
    return _capital_prices_to_df(r.json())


def fetch_candles_range(interval, start_date, end_date, session_headers=None):
    """Paginated range fetch (see earlier revisions): halves the window when
    Capital.com answers error.invalid.max.daterange."""
    if interval in _RESAMPLE_FROM:
        base_interval, factor = _RESAMPLE_FROM[interval]
        base_df = fetch_candles_range(
            base_interval, pd.Timestamp(start_date) - pd.Timedelta(hours=factor), end_date,
            session_headers=session_headers,
        )
        return _resample_candles(base_df, factor)

    own_session = session_headers is None
    headers = session_headers or _capital_session_headers()
    resolution = _capital_resolution(interval)
    interval_minutes = _INTERVAL_MINUTES.get(interval, 5)
    max_chunk_span = pd.Timedelta(minutes=interval_minutes * 1000)

    start_utc = _local_naive_to_utc_naive(start_date)
    end_utc = _local_naive_to_utc_naive(end_date)

    frames = []
    cur_start = start_utc
    chunk_span = max_chunk_span

    while cur_start < end_utc:
        cur_end = min(cur_start + chunk_span, end_utc)
        while True:
            params = {
                "resolution": resolution, "max": 1000,
                "from": cur_start.strftime("%Y-%m-%dT%H:%M:%S"),
                "to": cur_end.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            r = requests.get(f"{CAPITAL_BASE_URL}/api/v1/prices/{SYMBOL}", headers=headers, params=params, timeout=30)
            if r.status_code in (401, 403) and own_session:
                headers = _capital_session_headers()
                r = requests.get(f"{CAPITAL_BASE_URL}/api/v1/prices/{SYMBOL}", headers=headers, params=params, timeout=30)
            if r.status_code == 400 and "error.invalid.max.daterange" in r.text:
                new_span = (cur_end - cur_start) / 2
                if new_span < pd.Timedelta(minutes=interval_minutes):
                    raise RuntimeError(
                        f"Capital.com prices range error ({r.status_code}): {r.text} "
                        f"(still rejected after shrinking the window down to {new_span})"
                    )
                cur_end = cur_start + new_span
                chunk_span = new_span
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Capital.com prices range error ({r.status_code}): {r.text}")
            break

        payload = r.json()
        if payload.get("prices"):
            frames.append(_capital_prices_to_df(payload))
        cur_start = cur_end
        chunk_span = min(chunk_span, max_chunk_span)

    if not frames:
        raise RuntimeError("Capital.com returned no candles for this request.")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    )


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
        (high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df, length):
    return true_range(df).ewm(alpha=1 / length, adjust=False).mean()


def supertrend(df, period, factor):
    """Pine ta.supertrend. direction here: 1 = bullish, -1 = bearish
    (Pine's own sign is the opposite: <0 bullish)."""
    hl2 = ((df["high"] + df["low"]) / 2).tolist()
    close = df["close"].tolist()
    atr_val = true_range(df).ewm(alpha=1 / period, adjust=False).mean().tolist()

    n = len(df)
    final_upper = [0.0] * n
    final_lower = [0.0] * n
    direction = [0] * n
    st = [0.0] * n

    for i in range(n):
        upper = hl2[i] + factor * atr_val[i]
        lower = hl2[i] - factor * atr_val[i]
        if i == 0:
            final_upper[i] = upper
            final_lower[i] = lower
            direction[i] = -1
        else:
            prev_upper = final_upper[i - 1]
            prev_lower = final_lower[i - 1]
            final_lower[i] = lower if (lower > prev_lower or close[i - 1] < prev_lower) else prev_lower
            final_upper[i] = upper if (upper < prev_upper or close[i - 1] > prev_upper) else prev_upper
            if direction[i - 1] == -1:
                direction[i] = 1 if close[i] > final_upper[i] else -1
            else:
                direction[i] = -1 if close[i] < final_lower[i] else 1
        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return (
        pd.Series(st, index=df.index, dtype="float64"),
        pd.Series(direction, index=df.index, dtype="int64"),
    )


def add_indicators(df):
    df["emaFast"] = ema(df["close"], FAST_LEN)
    df["emaSlow"] = ema(df["close"], SLOW_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)
    st_series, dir_series = supertrend(df, ST_ATR_PERIOD, ST_FACTOR)
    df["st"] = st_series
    df["st_dir"] = dir_series

    # bars since the last Supertrend flip (Pine: bar_index - stFlipBar)
    d = df["st_dir"].tolist()
    out, last_flip = [], None
    for i, x in enumerate(d):
        if i > 0 and x != d[i - 1]:
            last_flip = i
        out.append(9999 if last_flip is None else i - last_flip)
    df["bars_since_flip"] = out
    return df


def attach_htf_direction(df, htf_df):
    """
    Pine: request.security(htf, supertrend()[1], lookahead_on) -> while a chart
    bar sits inside HTF candle H it sees the supertrend of candle H-1 (the
    previous COMPLETED HTF candle). Shifting each HTF timestamp (candle open)
    by one full HTF candle and doing a backward as-of merge reproduces that.
    Used by BOTH the live check and the backtest.
    """
    _, d = supertrend(htf_df, HTF_ATR_PERIOD, HTF_FACTOR)
    h = htf_df[["datetime"]].copy()
    h["htf_st_dir"] = d.values
    h["datetime"] = h["datetime"] + pd.Timedelta(minutes=_INTERVAL_MINUTES.get(HTF_TIMEFRAME, 120))
    out = pd.merge_asof(df.sort_values("datetime"), h.sort_values("datetime"),
                        on="datetime", direction="backward")
    out["htf_st_dir"] = out["htf_st_dir"].fillna(0).astype(int)
    return out.reset_index(drop=True)


def trend_arrow(d):
    return "▲" if d == 1 else "▼" if d == -1 else "→"


# ---------------------- SESSION / DAY / MONTH FILTERS ----------------------
def is_weekend_bar(bar_dt):
    return bar_dt.weekday() >= 5


def is_outside_trading_hours(bar_dt):
    if not USE_TRADING_HOURS:
        return False
    minutes_now = bar_dt.hour * 60 + bar_dt.minute
    start_min = TRADING_START_HOUR * 60 + TRADING_START_MINUTE
    end_min = TRADING_END_HOUR * 60 + TRADING_END_MINUTE
    return not (start_min <= minutes_now <= end_min)


def is_day_blocked(bar_dt):
    if not USE_DAY_FILTER:
        return False
    return not _DAY_FLAGS_BY_WEEKDAY.get(bar_dt.weekday(), False)


def is_month_end_blocked(bar_dt):
    if not USE_MONTH_END_PAUSE:
        return False
    dim = calendar.monthrange(bar_dt.year, bar_dt.month)[1]
    return bar_dt.day > (dim - MONTH_END_PAUSE_DAYS)


def is_new_entries_blocked(bar_dt):
    weekend = DISABLE_WEEKEND_SIGNALS and is_weekend_bar(bar_dt)
    outside_hours = is_outside_trading_hours(bar_dt)
    day_blocked = is_day_blocked(bar_dt)
    month_blocked = is_month_end_blocked(bar_dt)
    return weekend, outside_hours, day_blocked, month_blocked, (weekend or outside_hours or day_blocked or month_blocked)


# ---------------------- SMC ENGINE (port of the Pine SMC block) ----------------------
def _ok(x):
    return x == x  # not NaN


class SMCEngine:
    """
    Bar-by-bar port of the indicator's SMC section. Call step(i) for every bar
    in order. After an entry fires call reset_sweeps() (Pine sets
    lastSweepLowBar/lastSweepHighBar := na when a trade opens).
    """

    def __init__(self, df):
        n = len(df)
        self.n = n
        self.o = df["open"].to_numpy(dtype=float)
        self.h = df["high"].to_numpy(dtype=float)
        self.l = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)
        self.atr = df["atr"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.zeros(n)

        rng = self.h - self.l
        with np.errstate(divide="ignore", invalid="ignore"):
            buy = np.where(rng > 0, vol * (self.c - self.l) / rng, vol * 0.5)
        delta = buy - (vol - buy)
        self.flow_avg = pd.Series(delta).rolling(FLOW_LEN).mean().to_numpy()
        self.vol_spike = vol > (pd.Series(vol).rolling(20).mean().to_numpy() * VOL_MULT)
        self.range_hi = pd.Series(self.h).rolling(PD_LEN).max().to_numpy()
        self.range_lo = pd.Series(self.l).rolling(PD_LEN).min().to_numpy()

        nan = math.nan
        self.sw_h = nan; self.sw_l = nan
        self.sw_h_used = False; self.sw_l_used = False
        self.struct_dir = 0
        self.last_sweep_low_bar = None; self.last_sweep_high_bar = None
        self.sweep_low_px = nan; self.sweep_high_px = nan
        self.b_top = nan; self.b_bot = nan; self.b_bar = -1
        self.s_top = nan; self.s_bot = nan; self.s_bar = -1
        self.last_bear_hi = nan; self.last_bear_lo = nan
        self.last_bull_hi = nan; self.last_bull_lo = nan
        self.bull_top = nan; self.bull_bot = nan; self.bull_bar = -1
        self.bear_top = nan; self.bear_bot = nan; self.bear_bar = -1

    def reset_sweeps(self):
        self.last_sweep_low_bar = None
        self.last_sweep_high_bar = None

    def step(self, i):
        o, h, l, c = self.o, self.h, self.l, self.c
        L = SWING_LEN
        nan = math.nan

        # swing pivots (confirmed L bars later)
        if i >= 2 * L:
            ci = i - L
            if h[ci] == h[i - 2 * L:i + 1].max():
                self.sw_h = h[ci]; self.sw_h_used = False
            if l[ci] == l[i - 2 * L:i + 1].min():
                self.sw_l = l[ci]; self.sw_l_used = False

        # BOS
        bos_up = _ok(self.sw_h) and (not self.sw_h_used) and c[i] > self.sw_h
        bos_dn = _ok(self.sw_l) and (not self.sw_l_used) and c[i] < self.sw_l
        if bos_up:
            self.struct_dir = 1; self.sw_h_used = True
        if bos_dn:
            self.struct_dir = -1; self.sw_l_used = True

        # liquidity sweeps
        sweep_low = _ok(self.sw_l) and l[i] < self.sw_l and c[i] > self.sw_l
        sweep_high = _ok(self.sw_h) and h[i] > self.sw_h and c[i] < self.sw_h
        if sweep_low:
            self.last_sweep_low_bar = i; self.sweep_low_px = l[i]
        if sweep_high:
            self.last_sweep_high_bar = i; self.sweep_high_px = h[i]
        recent_low = self.last_sweep_low_bar is not None and (i - self.last_sweep_low_bar) <= SWEEP_LOOKBACK
        recent_high = self.last_sweep_high_bar is not None and (i - self.last_sweep_high_bar) <= SWEEP_LOOKBACK

        # fair value gaps
        a = self.atr[i]
        if i >= 2:
            if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= FVG_MIN_ATR * a and c[i - 1] > o[i - 1]:
                self.b_top, self.b_bot, self.b_bar = l[i], h[i - 2], i
            if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= FVG_MIN_ATR * a and c[i - 1] < o[i - 1]:
                self.s_top, self.s_bot, self.s_bar = l[i - 2], h[i], i
        if _ok(self.b_bot) and (c[i] < self.b_bot or i - self.b_bar > ZONE_MAX_AGE):
            self.b_top = nan; self.b_bot = nan
        if _ok(self.s_top) and (c[i] > self.s_top or i - self.s_bar > ZONE_MAX_AGE):
            self.s_top = nan; self.s_bot = nan

        # order blocks (last opposite candle before a BOS)
        if bos_up and _ok(self.last_bear_lo):
            self.bull_top, self.bull_bot, self.bull_bar = self.last_bear_hi, self.last_bear_lo, i
        if bos_dn and _ok(self.last_bull_lo):
            self.bear_top, self.bear_bot, self.bear_bar = self.last_bull_hi, self.last_bull_lo, i
        if c[i] < o[i]:
            self.last_bear_hi, self.last_bear_lo = h[i], l[i]
        if c[i] > o[i]:
            self.last_bull_hi, self.last_bull_lo = h[i], l[i]
        if _ok(self.bull_bot) and (c[i] < self.bull_bot or i - self.bull_bar > ZONE_MAX_AGE):
            self.bull_top = nan; self.bull_bot = nan
        if _ok(self.bear_top) and (c[i] > self.bear_top or i - self.bear_bar > ZONE_MAX_AGE):
            self.bear_top = nan; self.bear_bot = nan

        in_bull_fvg = _ok(self.b_top) and i > self.b_bar and l[i] <= self.b_top and c[i] >= self.b_bot
        in_bear_fvg = _ok(self.s_bot) and i > self.s_bar and h[i] >= self.s_bot and c[i] <= self.s_top
        in_bull_ob = _ok(self.bull_top) and i > self.bull_bar and l[i] <= self.bull_top and c[i] >= self.bull_bot
        in_bear_ob = _ok(self.bear_bot) and i > self.bear_bar and h[i] >= self.bear_bot and c[i] <= self.bear_top
        in_bull_zone = in_bull_fvg or in_bull_ob
        in_bear_zone = in_bear_fvg or in_bear_ob

        # premium / discount
        eq = (self.range_hi[i] + self.range_lo[i]) / 2
        in_discount = bool(c[i] < eq)
        in_premium = bool(c[i] > eq)

        # delta proxy
        fa = self.flow_avg[i]
        flow_bull = USE_FLOW and bool(fa > 0)
        flow_bear = USE_FLOW and bool(fa < 0)

        score_long = int(self.struct_dir == 1) + int(recent_low) + int(in_bull_zone) + int(in_discount) + int(flow_bull)
        score_short = int(self.struct_dir == -1) + int(recent_high) + int(in_bear_zone) + int(in_premium) + int(flow_bear)

        return {
            "score_long": score_long, "score_short": score_short,
            "recent_sweep_low": recent_low, "recent_sweep_high": recent_high,
            "sweep_low_px": self.sweep_low_px, "sweep_high_px": self.sweep_high_px,
            "vol_spike": bool(self.vol_spike[i]), "struct_dir": self.struct_dir,
        }


# ---------------------- STATE ----------------------
DEFAULT_STATS = {
    "total_trades": 0, "wins": 0, "losses": 0, "sum_pnl": 0.0,
    "best_trade": None, "worst_trade": None,
    "trend_n": 0, "trend_w": 0, "smc_n": 0, "smc_w": 0, "forced_n": 0, "forced_w": 0,
}

DEFAULT_STATE = {
    "last_signal_bar": None,
    "last_closed_bar_time": None,
    "position": None,
    "history": [],
    "stats": dict(DEFAULT_STATS),
    "current_day": None,
    "traded_today": False,
    "force_attempted_today": False,
    "force_skipped_today": False,
    "losses_today": 0,
    "last_exit_bar_time": None,
    "smc_reset_times": [],
    "pending_dir": None,
    "pending_bar_time": None,
}


def fresh_state():
    return json.loads(json.dumps(DEFAULT_STATE))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        for k, v in DEFAULT_STATE.items():
            state.setdefault(k, json.loads(json.dumps(v)))
        for k, v in DEFAULT_STATS.items():
            state["stats"].setdefault(k, v)
        return state
    return fresh_state()


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
    return request.args.get("notify", "").strip().lower() in ("1", "true", "yes", "on")


def _fmt_price(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def _fmt_money(v):
    if v is None:
        return "—"
    return f"{'+' if v >= 0 else ''}{v:.2f}"


def _methods_text():
    return (f"Trend {'ON' if ENABLE_TREND else 'OFF'} | SMC {'ON' if ENABLE_SMC else 'OFF'} | "
            f"Forced {'ON' if ENABLE_FORCE else 'OFF'}")


def build_check_telegram_message(result, state):
    lines = ["📊 [Manual Check] XAUUSD", f"Bar: {result.get('bar_time')} ({FORCE_TIMEZONE})"]
    event = result.get("event")
    if event == "entry":
        lines.append(f"Result: NEW {result.get('side')} signal opened this check ({result.get('setup')}).")
    elif event == "position_update":
        lines.append("Result: open position was updated this check (TP/SL — see the alert above).")
    elif result.get("weekend_skipped"):
        lines.append("Result: weekend — new-signal checks skipped (existing position, if any, still managed).")
    elif result.get("trading_hours_skipped"):
        lines.append("Result: outside trading hours — new-signal checks skipped (existing position, if any, still managed).")
    elif result.get("day_skipped"):
        lines.append("Result: today isn't a selected trading day — new-signal checks skipped.")
    elif result.get("month_skipped"):
        lines.append("Result: month-end pause active — new-signal checks skipped.")
    else:
        pending = state.get("pending_dir")
        if pending:
            lines.append(f"Result: no new signal on this bar. Pending {'LONG' if pending == 1 else 'SHORT'} signal armed, waiting for pullback.")
        else:
            lines.append("Result: no new signal on this bar.")
    lines.append(f"Methods: {_methods_text()}")

    pos = state.get("position")
    if pos:
        side = "BUY" if pos["dir"] == 1 else "SELL"
        lines.append(
            f"Open position: {side} ({pos.get('setup', '')}) @ {_fmt_price(pos['entry'])} | "
            f"SL {_fmt_price(pos['sl'])} | TP1 {_fmt_price(pos['tp1'])} TP2 {_fmt_price(pos['tp2'])}"
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
    ss = payload.get("setup_stats", {})
    lines.append(
        f"Trend {ss.get('trend_w', 0)}/{ss.get('trend_n', 0)} | SMC {ss.get('smc_w', 0)}/{ss.get('smc_n', 0)} | "
        f"Forced {ss.get('forced_w', 0)}/{ss.get('forced_n', 0)}"
    )
    lines.append(f"Methods: {_methods_text()}")
    pos = payload.get("position")
    if pos:
        side = "BUY" if pos["dir"] == 1 else "SELL"
        lines.append(f"Open position: {side} @ {_fmt_price(pos['entry'])} | SL {_fmt_price(pos['sl'])}")
    else:
        lines.append("No open position.")
    dg = payload["daily_guarantee"]
    lines.append(
        f"Daily guarantee: {'ON' if dg['enabled'] else 'OFF'} @ "
        f"{dg['force_hour']:02d}:{dg['force_minute']:02d} {dg['force_timezone']} | "
        f"traded today: {'yes' if dg['traded_today'] else 'no'}"
    )
    rg = payload.get("risk_guard", {})
    lines.append(f"Losses today: {rg.get('losses_today', 0)} / {rg.get('max_losses_per_day') or '∞'}")
    th = payload.get("trading_hours", {})
    if th.get("enabled"):
        lines.append(
            f"Trading hours: {th['start_hour']:02d}:{th['start_minute']:02d}"
            f"-{th['end_hour']:02d}:{th['end_minute']:02d} {th['timezone']}"
        )
    dayf = payload.get("day_filter", {})
    if dayf.get("enabled"):
        lines.append(f"Day filter: {', '.join(dayf.get('selected_days', [])) or 'none'}")
    return "\n".join(lines)


# ---------------------- TRADE LOG / SETTLEMENT ----------------------
def settle_trade(state, pos, exit_price, result_label, bar_time):
    """Mirror of Pine closeTrade(): spread is deducted from the points."""
    raw = (exit_price - pos["entry"]) if pos["dir"] == 1 else (pos["entry"] - exit_price)
    points = raw - SPREAD_PTS
    pnl = points * LOT_SIZE * UNITS_PER_LOT
    setup = pos.get("setup", "TREND")

    state["history"].insert(0, {
        "side": "BUY" if pos["dir"] == 1 else "SELL", "setup": setup,
        "entry": pos["entry"], "sl": pos["sl"], "tp1": pos["tp1"], "tp2": pos["tp2"],
        "exit": exit_price, "result": result_label,
        "points": round(points, 2), "pnl": round(pnl, 2),
    })
    state["history"] = state["history"][:50]

    s = state["stats"]
    s["sum_pnl"] += pnl
    s["total_trades"] += 1
    is_win = pnl > 0
    if is_win:
        s["wins"] += 1
    else:
        s["losses"] += 1
        state["losses_today"] = state.get("losses_today", 0) + 1
    key = {"SMC": "smc", "FORCED": "forced"}.get(setup, "trend")
    s[f"{key}_n"] = s.get(f"{key}_n", 0) + 1
    if is_win:
        s[f"{key}_w"] = s.get(f"{key}_w", 0) + 1
    s["best_trade"] = pnl if s["best_trade"] is None else max(s["best_trade"], pnl)
    s["worst_trade"] = pnl if s["worst_trade"] is None else min(s["worst_trade"], pnl)

    state["last_exit_bar_time"] = bar_time
    state["position"] = None
    return pnl


def open_position(e, bar_time):
    return {
        "dir": 1 if e["side"] == "BUY" else -1,
        "entry": e["entry"], "sl": e["sl"], "tp1": e["tp1"], "tp2": e["tp2"],
        "tp3": None, "tp4": None,   # kept so the dashboard still renders
        "setup": e["setup"], "entry_time": bar_time,
        "tp1_hit": False, "lock_bar": None, "remaining_size": 1.0,
    }


def commit_entry(state, e, bar_time):
    state["position"] = open_position(e, bar_time)
    state["traded_today"] = True
    state["force_attempted_today"] = True
    state["pending_dir"] = None
    state["pending_bar_time"] = None
    resets = state.setdefault("smc_reset_times", [])
    resets.append(bar_time)
    state["smc_reset_times"] = resets[-200:]


# ---------------------- POSITION MANAGEMENT ----------------------
def manage_position(state, candle, silent=False):
    """
    Exits are only evaluated on bars AFTER the entry bar (like the indicator).
    Ratchet mode: TP1 -> SL moves to entry +/- LOCK_PTS (the bar that locks is
    not checked again for SL/TP2, same as Pine's if/else chain), TP2 closes
    the whole trade. silent=True is used by the backtest.
    """
    pos = state["position"]
    if pos is None:
        return False

    bar_t = str(candle["datetime"])
    if bar_t <= str(pos["entry_time"]):
        return False
    if pos.get("tp1_hit") and pos.get("lock_bar") and bar_t <= str(pos["lock_bar"]):
        return False  # candle that already locked TP1 (re-polled forming candle)

    high, low = candle["high"], candle["low"]
    d = pos["dir"]
    events = []

    def close_(price, label):
        pnl = settle_trade(state, pos, price, label, bar_t)
        events.append((label, price, pnl))

    hit_sl = (low <= pos["sl"]) if d == 1 else (high >= pos["sl"])
    hit_tp1 = (high >= pos["tp1"]) if d == 1 else (low <= pos["tp1"])
    hit_tp2 = (high >= pos["tp2"]) if d == 1 else (low <= pos["tp2"])

    if PNL_MODE == "tp1_only":
        if hit_sl:
            close_(pos["sl"], "SL Hit")
        elif hit_tp1:
            close_(pos["tp1"], "TP1 Hit (Full Close)")
    elif PNL_MODE == "first_hit":
        if hit_sl:
            close_(pos["sl"], "SL Hit")
        elif hit_tp2:
            close_(pos["tp2"], "TP2 Hit (Full Close)")
        elif hit_tp1:
            close_(pos["tp1"], "TP1 Hit (Full Close)")
    else:  # ratchet
        if not pos["tp1_hit"]:
            if hit_sl:
                close_(pos["sl"], "SL Hit")
            elif hit_tp2:
                close_(pos["tp2"], "TP2 Hit (Gap, Full Size)")
            elif hit_tp1:
                pos["tp1_hit"] = True
                pos["lock_bar"] = bar_t
                pos["sl"] = pos["entry"] + LOCK_PTS if d == 1 else pos["entry"] - LOCK_PTS
                events.append(("TP1 Hit (SL->Locked Profit)", pos["tp1"], None))
        else:
            if hit_sl:
                close_(pos["sl"], f"SL After TP1 (Locked +{LOCK_PTS})")
            elif hit_tp2:
                close_(pos["tp2"], "TP2 Hit (Final)")

    if not silent:
        side = "BUY" if d == 1 else "SELL"
        for label, price, pnl in events:
            if pnl is None:
                msg = (f"XAUUSD {side} — {label}\nPrice: {price:.2f}\n"
                       f"(Stop loss moved — no P&L booked yet, position still open)")
            else:
                msg = (f"XAUUSD {side} — {label}\nPrice: {price:.2f}\n"
                       f"P&L (trade closed): {'+' if pnl >= 0 else '-'}${abs(pnl):.2f}")
            send_telegram(msg)

    return len(events) > 0


# ---------------------- DAILY / FORCE HELPERS ----------------------
def roll_daily_guarantee_state(state, bar_dt):
    day_str = bar_dt.strftime("%Y-%m-%d")
    if state.get("current_day") != day_str:
        state["current_day"] = day_str
        state["traded_today"] = False
        state["force_attempted_today"] = False
        state["force_skipped_today"] = False
        state["losses_today"] = 0


def compute_force_entry(state, bar_dt, st_dir, htf_dir, ema_fast_last, ema_slow_last,
                        rsi_val, extension_atr, risk_ok):
    if not ENABLE_FORCE:
        return False, 0

    if st_dir in (1, -1):
        force_direction = st_dir
    elif htf_dir in (1, -1):
        force_direction = htf_dir
    else:
        force_direction = 1 if ema_fast_last >= ema_slow_last else -1

    mins = bar_dt.hour * 60 + bar_dt.minute
    is_past_force_time = mins >= (FORCE_HOUR * 60 + FORCE_MINUTE)
    is_past_hard_time = mins >= (FORCE_HARD_HOUR * 60 + FORCE_HARD_MINUTE)

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
        and risk_ok                       # Pine: forced entries also need entryOk (cooldown / loss cap)
        and force_can_fire_now
        and not state.get("force_skipped_today", False)
    )
    return force_entry_now, force_direction


# ---------------------- PULLBACK STATE ----------------------
def bars_since_pending_armed(df, i, pending_bar_time):
    if pending_bar_time is None:
        return None
    arr = df["datetime"].values
    target = np.datetime64(pd.to_datetime(pending_bar_time))
    pos = arr.searchsorted(target)
    if pos >= len(arr) or arr[pos] != target:
        return PULLBACK_TIMEOUT_BARS + 1
    return i - pos


def roll_pullback_state(state, df, i, bar_time, st_bullish, st_bearish, htf_bullish, htf_bearish,
                        base_long_cond, base_short_cond, entries_blocked=False):
    if base_long_cond:
        state["pending_dir"] = 1
        state["pending_bar_time"] = bar_time
    if base_short_cond:
        state["pending_dir"] = -1
        state["pending_bar_time"] = bar_time

    bars_since = bars_since_pending_armed(df, i, state["pending_bar_time"])
    timed_out = bars_since is not None and bars_since > PULLBACK_TIMEOUT_BARS

    if state["pending_dir"] == 1:
        if not st_bullish or not htf_bullish or timed_out or entries_blocked:
            state["pending_dir"] = None
            state["pending_bar_time"] = None
    elif state["pending_dir"] == -1:
        if not st_bearish or not htf_bearish or timed_out or entries_blocked:
            state["pending_dir"] = None
            state["pending_bar_time"] = None


# ---------------------- ENTRY EVALUATION (shared by /check and the backtest) ----------------------
def evaluate_entry(state, df, i, smc, bars_since_exit):
    """
    Evaluate bar i (a CLOSED bar) the way the indicator does. The caller must
    only call this when flat. Returns (entry_dict_or_None, info_dict).
    df needs: emaFast, emaSlow, rsi, atr, st_dir, bars_since_flip, htf_st_dir.
    Each method (TREND / SMC / FORCED) is gated by its ENABLE_* switch.
    """
    bar = df.iloc[i]
    prev = df.iloc[i - 1]
    bar_dt = bar["datetime"]
    bar_time = str(bar_dt)

    weekend, outside, day_blk, month_blk, blocked = is_new_entries_blocked(bar_dt)
    info = {"weekend": weekend, "outside_hours": outside, "day_blocked": day_blk,
            "month_blocked": month_blk, "blocked": blocked}
    if blocked:
        state["pending_dir"] = None
        state["pending_bar_time"] = None
        return None, info

    cooldown_ok = bars_since_exit is None or bars_since_exit >= COOLDOWN_BARS
    loss_ok = MAX_LOSSES_PER_DAY == 0 or state.get("losses_today", 0) < MAX_LOSSES_PER_DAY
    risk_ok = cooldown_ok and loss_ok
    info["risk_ok"] = risk_ok

    close_, open_ = bar["close"], bar["open"]
    ema_cross_up = prev["emaFast"] <= prev["emaSlow"] and bar["emaFast"] > bar["emaSlow"]
    ema_cross_down = prev["emaFast"] >= prev["emaSlow"] and bar["emaFast"] < bar["emaSlow"]
    rsi_ok_long = (not USE_RSI) or bar["rsi"] < RSI_OB
    rsi_ok_short = (not USE_RSI) or bar["rsi"] > RSI_OS

    st_bullish = bar["st_dir"] == 1
    st_bearish = bar["st_dir"] == -1
    st_flip_confirmed = bar["bars_since_flip"] >= ST_CONFIRM_BARS

    h_dir = int(bar["htf_st_dir"])
    htf_bullish = (not USE_HTF) or h_dir == 1
    htf_bearish = (not USE_HTF) or h_dir == -1

    extension_atr = (abs(close_ - bar["emaFast"]) / bar["atr"]) if bar["atr"] > 0 else 0.0
    extension_ok = (not USE_EXTENSION_FILTER) or extension_atr <= MAX_EXTENSION_ATR
    pullback_ok = extension_atr <= PULLBACK_MAX_ATR

    base_long_cond = ema_cross_up and rsi_ok_long and st_bullish and st_flip_confirmed and htf_bullish
    base_short_cond = ema_cross_down and rsi_ok_short and st_bearish and st_flip_confirmed and htf_bearish

    roll_pullback_state(state, df, i, bar_time, st_bullish, st_bearish, htf_bullish, htf_bearish,
                        base_long_cond, base_short_cond, entries_blocked=False)

    gate_long = (not USE_SMC_GATE) or smc["score_long"] >= GATE_MIN_SCORE
    gate_short = (not USE_SMC_GATE) or smc["score_short"] >= GATE_MIN_SCORE

    if USE_PULLBACK_ENTRY:
        trend_long_core = state["pending_dir"] == 1 and pullback_ok and st_bullish and htf_bullish and rsi_ok_long
        trend_short_core = state["pending_dir"] == -1 and pullback_ok and st_bearish and htf_bearish and rsi_ok_short
    else:
        trend_long_core, trend_short_core = base_long_cond, base_short_cond

    # TREND method (EMA-cross), gated by the SMC score
    long_cond = ENABLE_TREND and extension_ok and risk_ok and gate_long and trend_long_core
    short_cond = ENABLE_TREND and extension_ok and risk_ok and gate_short and trend_short_core

    # SMC method: standalone sweep -> confirmation candle
    bull_confirm = close_ > open_ and close_ > prev["close"]
    bear_confirm = close_ < open_ and close_ < prev["close"]
    vol_ok = (not REQUIRE_VOL_SPIKE) or smc["vol_spike"]
    smc_long = (ENABLE_SMC and risk_ok and smc["recent_sweep_low"] and bull_confirm and htf_bullish
                and rsi_ok_long and vol_ok and smc["score_long"] >= TRIG_MIN_SCORE)
    smc_short = (ENABLE_SMC and risk_ok and smc["recent_sweep_high"] and bear_confirm and htf_bearish
                 and rsi_ok_short and vol_ok and smc["score_short"] >= TRIG_MIN_SCORE)

    # FORCED method (returns False when ENABLE_FORCE is off)
    force_entry_now, force_direction = compute_force_entry(
        state, bar_dt, int(bar["st_dir"]), h_dir, bar["emaFast"], bar["emaSlow"],
        bar["rsi"], extension_atr, risk_ok,
    )
    force_long = force_entry_now and force_direction == 1
    force_short = force_entry_now and force_direction == -1

    info["pending_dir"] = state["pending_dir"]
    info["extension_atr"] = round(float(extension_atr), 3)
    info["extension_ok"] = bool(extension_ok)
    info["htf_dir"] = h_dir
    info["st_dir"] = int(bar["st_dir"])

    is_long = long_cond or smc_long or force_long
    is_short = short_cond or smc_short or force_short
    if not (is_long or is_short):
        return None, info

    sl_dist = min(max(bar["atr"] * SL_MULT, SL_MIN_PTS), SL_MAX_PTS)
    entry = close_

    if is_long:
        is_smc = smc_long and not long_cond
        is_forced = force_long and not long_cond and not smc_long
        if is_smc and USE_STRUCTURE_SL and _ok(smc["sweep_low_px"]):
            sl_dist = min(max(close_ - smc["sweep_low_px"] + SL_BUFFER_PTS, SL_MIN_PTS), SL_MAX_PTS)
        sl = entry - sl_dist
        risk = entry - sl
        tp1, tp2 = entry + risk * RR1, entry + risk * RR2
        side = "BUY"
    else:
        is_smc = smc_short and not short_cond
        is_forced = force_short and not short_cond and not smc_short
        if is_smc and USE_STRUCTURE_SL and _ok(smc["sweep_high_px"]):
            sl_dist = min(max(smc["sweep_high_px"] - close_ + SL_BUFFER_PTS, SL_MIN_PTS), SL_MAX_PTS)
        sl = entry + sl_dist
        risk = sl - entry
        tp1, tp2 = entry - risk * RR1, entry - risk * RR2
        side = "SELL"

    setup = "FORCED" if is_forced else "SMC" if is_smc else "TREND"
    entry_dict = {
        "side": side, "setup": setup, "entry": float(entry), "sl": float(sl),
        "tp1": float(tp1), "tp2": float(tp2),
        "forced": setup == "FORCED", "smc": setup == "SMC",
        "pullback": USE_PULLBACK_ENTRY and setup == "TREND",
        "htf_dir": h_dir, "st_dir": int(bar["st_dir"]),
    }
    return entry_dict, info


# ---------------------- SIGNAL-COUNT BACKTEST (last N signals) ----------------------
def run_signal_backtest(target_count=120, initial_days=30, max_days=730):
    """
    Replays the bot's own logic (Trend / SMC / Forced, honouring the ENABLE_*
    switches) on closed bars only, with the same HTF handling, spread,
    cooldown, loss cap and SMC engine as live, until `target_count` signals
    are found.
    """
    headers = _capital_session_headers()
    days = initial_days
    df = None
    start_idx = 0
    all_signals = []

    while True:
        end_date = pd.Timestamp.now(tz=FORCE_TIMEZONE).tz_localize(None)
        fetch_start = end_date - pd.Timedelta(days=days + 6)

        df = fetch_candles_range(TIMEFRAME, fetch_start, end_date, session_headers=headers)
        if len(df) > 1:
            df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming candle
        if len(df) < 50:
            if days >= max_days:
                return {"error": f"Not enough candles returned ({len(df)}) even at the {max_days}-day cap."}
            days = min(days * 2, max_days)
            continue

        add_indicators(df)
        if USE_HTF:
            htf_df = fetch_candles_range(HTF_TIMEFRAME, fetch_start, end_date, session_headers=headers)
            df = attach_htf_direction(df, htf_df)
        else:
            df["htf_st_dir"] = 0

        cutoff = end_date - pd.Timedelta(days=days)
        cutoff_matches = df.index[df["datetime"] >= cutoff]
        start_idx = int(cutoff_matches[0]) if len(cutoff_matches) else max(0, len(df) - 1)
        start_idx = max(start_idx, 1)

        bt = fresh_state()
        engine = SMCEngine(df)
        all_signals = []
        open_signal_idx = None
        last_exit_idx = None

        for i in range(len(df)):
            smc = engine.step(i)
            if i < start_idx:
                continue

            bar = df.iloc[i]
            bar_dt = bar["datetime"]
            roll_daily_guarantee_state(bt, bar_dt)

            if bt["position"] is not None:
                manage_position(bt, bar, silent=True)
                if bt["position"] is None:
                    last_exit_idx = i
                    if open_signal_idx is not None:
                        closed = bt["history"][0]
                        all_signals[open_signal_idx].update({
                            "status": "closed", "exit": closed["exit"], "result": closed["result"],
                            "points": closed["points"], "pnl": closed["pnl"],
                        })
                        open_signal_idx = None
                continue  # no new entry on a bar where a position was open

            bars_since_exit = None if last_exit_idx is None else i - last_exit_idx
            entry, _info = evaluate_entry(bt, df, i, smc, bars_since_exit)
            if entry:
                commit_entry(bt, entry, str(bar_dt))
                engine.reset_sweeps()
                all_signals.append({
                    "entry_time": str(bar_dt), "side": entry["side"], "setup": entry["setup"],
                    "entry": round(entry["entry"], 2), "sl": round(entry["sl"], 2),
                    "tp1": round(entry["tp1"], 2), "tp2": round(entry["tp2"], 2),
                    "forced": entry["forced"], "smc": entry["smc"], "pullback": entry["pullback"],
                    "status": "open", "exit": None, "result": None, "points": None, "pnl": None,
                })
                open_signal_idx = len(all_signals) - 1

        if len(all_signals) >= target_count or days >= max_days:
            break
        days = min(days * 2, max_days)

    last_signals = all_signals[-target_count:] if len(all_signals) > target_count else all_signals
    closed = [s for s in last_signals if s["status"] == "closed"]
    wins = sum(1 for s in closed if s["pnl"] is not None and s["pnl"] > 0)
    losses = len(closed) - wins
    net_pnl = sum(s["pnl"] for s in closed) if closed else 0.0
    win_rate = round((wins / len(closed) * 100), 1) if closed else 0
    pnl_values = [s["pnl"] for s in closed]

    by_setup = {}
    for name in ("TREND", "SMC", "FORCED"):
        grp = [s for s in closed if s["setup"] == name]
        by_setup[name] = {"n": len(grp), "wins": sum(1 for s in grp if s["pnl"] > 0),
                          "net_pnl": round(sum(s["pnl"] for s in grp), 2)}

    return {
        "requested_signal_count": target_count,
        "signals_found_in_window": len(all_signals),
        "signals_returned": len(last_signals),
        "days_scanned": days,
        "data_covers_from": str(df.iloc[start_idx]["datetime"]),
        "data_covers_to": str(df.iloc[-1]["datetime"]),
        "closed_trades": len(closed),
        "still_open": len(last_signals) - len(closed),
        "wins": wins, "losses": losses, "win_rate": win_rate,
        "net_pnl": round(net_pnl, 2),
        "best_trade": max(pnl_values) if pnl_values else None,
        "worst_trade": min(pnl_values) if pnl_values else None,
        "by_setup": by_setup,
        "signals": last_signals,
    }


# ---------------------- CORE CHECK ----------------------
@app.route("/check", methods=["GET"])
def check():
    if not (CAPITAL_API_KEY and CAPITAL_IDENTIFIER and CAPITAL_PASSWORD) or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    df = fetch_candles(TIMEFRAME, outputsize=1000)
    add_indicators(df)

    # Newest candle is still forming: NEW SIGNALS use the last CLOSED bar only;
    # the forming candle is only used to catch SL/TP touches.
    live = df.iloc[-1]
    sig_df = df.iloc[:-1].reset_index(drop=True)
    if len(sig_df) < 3:
        return jsonify({"error": "Not enough closed candles returned."}), 500
    last = sig_df.iloc[-1]
    bar_time = str(last["datetime"])

    state = load_state()
    result = {"bar_time": bar_time, "live_bar_time": str(live["datetime"]), "event": None}

    last_closed_str = state.get("last_closed_bar_time")
    if last_closed_str:
        closed_backlog = sig_df[sig_df["datetime"] > pd.to_datetime(last_closed_str)]
    else:
        closed_backlog = sig_df.iloc[0:0]

    any_managed_event = False
    flat_before_signal_bar = None

    # (a) backfill closed bars not processed yet
    for _, bar in closed_backlog.iterrows():
        if str(bar["datetime"]) == bar_time:
            flat_before_signal_bar = state["position"] is None
        roll_daily_guarantee_state(state, bar["datetime"])
        if state["position"] is not None:
            any_managed_event = manage_position(state, bar) or any_managed_event
        state["last_closed_bar_time"] = str(bar["datetime"])

    if flat_before_signal_bar is None:
        flat_before_signal_bar = state["position"] is None
    if state.get("last_closed_bar_time") is None:
        state["last_closed_bar_time"] = bar_time
    roll_daily_guarantee_state(state, last["datetime"])

    # (b) always re-check the forming candle for intrabar SL/TP touches
    if state["position"] is not None:
        any_managed_event = manage_position(state, live) or any_managed_event

    if any_managed_event:
        result["event"] = "position_update"
    save_state(state)

    weekend_now, outside_now, day_now, month_now, blocked_now = is_new_entries_blocked(last["datetime"])
    if blocked_now:
        state["last_signal_bar"] = bar_time
        state["pending_dir"] = None
        state["pending_bar_time"] = None
        result["weekend_skipped"] = weekend_now
        result["trading_hours_skipped"] = outside_now
        result["day_skipped"] = day_now
        result["month_skipped"] = month_now
        save_state(state)

    elif state.get("last_signal_bar") != bar_time:
        state["last_signal_bar"] = bar_time

        if flat_before_signal_bar and state["position"] is None:
            if USE_HTF:
                htf_df = fetch_candles(HTF_TIMEFRAME, outputsize=500)
                sig_df = attach_htf_direction(sig_df, htf_df)
            else:
                sig_df["htf_st_dir"] = 0

            # Replay the SMC engine over the fetched window. Entries wipe the
            # sweep memory, so re-apply those wipes at the stored entry times.
            engine = SMCEngine(sig_df)
            reset_times = set(state.get("smc_reset_times", []))
            smc = None
            for j in range(len(sig_df)):
                smc = engine.step(j)
                if str(sig_df.at[j, "datetime"]) in reset_times and j < len(sig_df) - 1:
                    engine.reset_sweeps()

            bars_since_exit = None
            if state.get("last_exit_bar_time"):
                exit_ts = pd.to_datetime(state["last_exit_bar_time"])
                bars_since_exit = int((sig_df["datetime"] > exit_ts).sum())

            i = len(sig_df) - 1
            entry, info = evaluate_entry(state, sig_df, i, smc, bars_since_exit)
            result["pending"] = {
                "dir": state["pending_dir"],
                "extension_atr": info.get("extension_atr"),
                "extension_ok": info.get("extension_ok"),
            }
            result["smc"] = {"score_long": smc["score_long"], "score_short": smc["score_short"],
                             "structure": smc["struct_dir"]}

            if entry:
                commit_entry(state, entry, bar_time)
                tag = {"FORCED": "(Forced Daily)", "SMC": "(SMC Sweep)"}.get(
                    entry["setup"], "(Trend, Pullback)" if entry["pullback"] else "(Trend)")
                htf_line = f"HTF Supertrend ({HTF_TIMEFRAME}): {trend_arrow(entry['htf_dir'])}\n" if USE_HTF else ""
                msg = (
                    f"XAUUSD {entry['side']} signal {tag}\n"
                    f"Entry: {entry['entry']:.2f}\n"
                    f"SL: {entry['sl']:.2f}\n"
                    f"TP1: {entry['tp1']:.2f}\n"
                    f"TP2: {entry['tp2']:.2f}\n"
                    f"Supertrend ({TIMEFRAME}): {trend_arrow(entry['st_dir'])}\n"
                    f"{htf_line}"
                    f"SMC score L/S: {smc['score_long']} / {smc['score_short']}\n"
                    f"Bar: {bar_time} ({FORCE_TIMEZONE})"
                )
                send_telegram(msg)
                result.update({
                    "event": "entry", "side": entry["side"], "setup": entry["setup"],
                    "forced": entry["forced"], "smc_entry": entry["smc"], "pullback": entry["pullback"],
                    "message": msg, "entry": entry["entry"], "sl": entry["sl"],
                    "tp1": entry["tp1"], "tp2": entry["tp2"],
                })

        save_state(state)

    state = load_state()
    result["position"] = state.get("position")
    result["trade_methods"] = {"trend": ENABLE_TREND, "smc": ENABLE_SMC, "forced": ENABLE_FORCE}

    if _notify_requested():
        send_telegram(build_check_telegram_message(result, state))
        result["telegram_notified"] = True

    return jsonify(result)


@app.route("/stats", methods=["GET"])
def stats():
    state = load_state()
    s = state["stats"]
    win_rate = (s["wins"] / s["total_trades"] * 100) if s["total_trades"] else 0
    selected_days = [name for name, flag in _DAY_LABEL_FLAGS if flag]
    payload = {
        "position": state["position"],
        "win_rate": round(win_rate, 1),
        "total_trades": s["total_trades"],
        "wins": s["wins"], "losses": s["losses"],
        "net_pnl": round(s["sum_pnl"], 2),
        "best_trade": s["best_trade"], "worst_trade": s["worst_trade"],
        "setup_stats": {k: s.get(k, 0) for k in
                        ("trend_n", "trend_w", "smc_n", "smc_w", "forced_n", "forced_w")},
        "trade_methods": {"trend": ENABLE_TREND, "smc": ENABLE_SMC, "forced": ENABLE_FORCE},
        "recent_log": state["history"][:10],
        "daily_guarantee": {
            "enabled": ENABLE_FORCE,
            "force_hour": FORCE_HOUR, "force_minute": FORCE_MINUTE,
            "force_timezone": FORCE_TIMEZONE,
            "current_day": state.get("current_day"),
            "traded_today": state.get("traded_today"),
            "force_attempted_today": state.get("force_attempted_today"),
            "force_skipped_today": state.get("force_skipped_today", False),
            "require_quality_filters": FORCE_REQUIRE_QUALITY_FILTERS,
            "hard_cutoff_hour": FORCE_HARD_HOUR, "hard_cutoff_minute": FORCE_HARD_MINUTE,
            "skip_if_never_valid": FORCE_SKIP_IF_NEVER_VALID,
        },
        "risk_guard": {
            "losses_today": state.get("losses_today", 0),
            "max_losses_per_day": MAX_LOSSES_PER_DAY,
            "cooldown_bars": COOLDOWN_BARS,
        },
        "pending_signal": {"dir": state.get("pending_dir"), "armed_bar_time": state.get("pending_bar_time")},
        "weekend_guard": {"enabled": DISABLE_WEEKEND_SIGNALS},
        "month_end_pause": {"enabled": USE_MONTH_END_PAUSE, "days": MONTH_END_PAUSE_DAYS},
        "trading_hours": {
            "enabled": USE_TRADING_HOURS,
            "start_hour": TRADING_START_HOUR, "start_minute": TRADING_START_MINUTE,
            "end_hour": TRADING_END_HOUR, "end_minute": TRADING_END_MINUTE,
            "timezone": FORCE_TIMEZONE,
        },
        "day_filter": {"enabled": USE_DAY_FILTER, "selected_days": selected_days, "timezone": FORCE_TIMEZONE},
    }
    if _notify_requested():
        send_telegram(build_stats_telegram_message(payload, state))
        payload["telegram_notified"] = True
    return jsonify(payload)


@app.route("/signals", methods=["GET"])
def signals_history():
    """GET /signals?count=120[&notify=true] -- replay the bot's logic on history."""
    if not (CAPITAL_API_KEY and CAPITAL_IDENTIFIER and CAPITAL_PASSWORD):
        return jsonify({"error": "Missing Capital.com credentials (CAPITAL_API_KEY / CAPITAL_IDENTIFIER / CAPITAL_PASSWORD)"}), 500

    count = int(request.args.get("count", 120))
    try:
        result = run_signal_backtest(target_count=count)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if "error" in result:
        return jsonify(result), 400

    if _notify_requested():
        bs = result["by_setup"]
        msg = (
            f"📊 [Last {result['signals_returned']} Signals] XAUUSD\n"
            f"Scanned back {result['days_scanned']}d "
            f"({result['closed_trades']} closed, {result['still_open']} still open)\n"
            f"Win rate: {result['win_rate']}% ({result['wins']}W / {result['losses']}L)\n"
            f"Net P&L: {'+' if result['net_pnl'] >= 0 else '-'}${abs(result['net_pnl']):.2f}\n"
            f"Trend {bs['TREND']['wins']}/{bs['TREND']['n']} | SMC {bs['SMC']['wins']}/{bs['SMC']['n']} | "
            f"Forced {bs['FORCED']['wins']}/{bs['FORCED']['n']}\n"
            f"Methods: {_methods_text()}"
        )
        send_telegram(msg)
        result["telegram_notified"] = True
    return jsonify(result)


@app.route("/test", methods=["GET"])
def test_signal():
    if not (CAPITAL_API_KEY and CAPITAL_IDENTIFIER and CAPITAL_PASSWORD) or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "Missing required environment variables"}), 500

    df = fetch_candles(TIMEFRAME, outputsize=100)
    df["atr"] = atr(df, ATR_LEN)
    last = df.iloc[-1]
    entry = last["close"]
    sl_dist = min(max(last["atr"] * SL_MULT, SL_MIN_PTS), SL_MAX_PTS)
    sl = entry - sl_dist
    risk = entry - sl
    msg = (
        f"[TEST] XAUUSD BUY signal (Trend)\n"
        f"Entry: {entry:.2f}\nSL: {sl:.2f}\n"
        f"TP1: {entry + risk * RR1:.2f}\nTP2: {entry + risk * RR2:.2f}\n"
        f"Bar: {last['datetime']} ({FORCE_TIMEZONE})\n"
        f"(This is a forced test message, not a real signal)"
    )
    send_telegram(msg)
    return jsonify({"status": "test message sent", "message": msg})


@app.route("/capital-status", methods=["GET"])
def capital_status():
    if not (CAPITAL_API_KEY and CAPITAL_IDENTIFIER and CAPITAL_PASSWORD):
        return jsonify({"configured": False, "error": "CAPITAL_API_KEY / CAPITAL_IDENTIFIER / CAPITAL_PASSWORD not fully set"}), 500
    try:
        headers = _capital_session_headers()
        r = requests.get(
            f"{CAPITAL_BASE_URL}/api/v1/prices/{SYMBOL}",
            headers=headers, params={"resolution": "MINUTE", "max": 1}, timeout=15,
        )
        symbol_available = r.status_code == 200 and bool(r.json().get("prices"))
    except Exception as e:
        return jsonify({"configured": True, "connected": False, "error": str(e)}), 500
    return jsonify({
        "configured": True, "connected": True, "base_url": CAPITAL_BASE_URL,
        "symbol": SYMBOL, "symbol_available": symbol_available,
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "synced_with": "Gold Signal Terminal v2 (Trend + SMC + Flow Proxy)",
        "data_provider": "Capital.com",
        "capital_configured": bool(CAPITAL_API_KEY and CAPITAL_IDENTIFIER and CAPITAL_PASSWORD),
        "capital_base_url": CAPITAL_BASE_URL,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "signals_evaluated_on": "last closed bar",
        "trade_methods": {"trend": ENABLE_TREND, "smc": ENABLE_SMC, "forced": ENABLE_FORCE},
        "supertrend": f"ATR {ST_ATR_PERIOD} / factor {ST_FACTOR}",
        "rsi": f"len {RSI_LEN}, block buys > {RSI_OB}, block sells < {RSI_OS}",
        "ema": f"{FAST_LEN}/{SLOW_LEN}",
        "use_htf": USE_HTF,
        "htf": f"{HTF_TIMEFRAME} ATR {HTF_ATR_PERIOD} / factor {HTF_FACTOR}" if USE_HTF else None,
        "st_confirm_bars": ST_CONFIRM_BARS,
        "smc": {
            "gate": USE_SMC_GATE, "gate_min_score": GATE_MIN_SCORE,
            "trigger": ENABLE_SMC, "trigger_min_score": TRIG_MIN_SCORE,
            "swing_len": SWING_LEN, "sweep_lookback": SWEEP_LOOKBACK,
            "structure_sl": USE_STRUCTURE_SL, "flow_proxy": USE_FLOW,
        },
        "use_pullback_entry": USE_PULLBACK_ENTRY,
        "pullback_max_atr": PULLBACK_MAX_ATR,
        "pullback_timeout_bars": PULLBACK_TIMEOUT_BARS,
        "use_extension_filter": USE_EXTENSION_FILTER,
        "max_extension_atr": MAX_EXTENSION_ATR,
        "risk": {"sl_mult": SL_MULT, "sl_min": SL_MIN_PTS, "sl_max": SL_MAX_PTS,
                 "rr1": RR1, "rr2": RR2, "lock_pts": LOCK_PTS, "spread_pts": SPREAD_PTS,
                 "lot": LOT_SIZE, "pnl_mode": PNL_MODE},
        "risk_guard": {"max_losses_per_day": MAX_LOSSES_PER_DAY, "cooldown_bars": COOLDOWN_BARS},
        "guarantee_daily_trade": ENABLE_FORCE,
        "force": f"{FORCE_HOUR:02d}:{FORCE_MINUTE:02d} {FORCE_TIMEZONE}, hard {FORCE_HARD_HOUR:02d}:{FORCE_HARD_MINUTE:02d}",
        "force_require_quality_filters": FORCE_REQUIRE_QUALITY_FILTERS,
        "force_skip_if_never_valid": FORCE_SKIP_IF_NEVER_VALID,
        "disable_weekend_signals": DISABLE_WEEKEND_SIGNALS,
        "trading_hours": f"{TRADING_START_HOUR:02d}:{TRADING_START_MINUTE:02d}-{TRADING_END_HOUR:02d}:{TRADING_END_MINUTE:02d}" if USE_TRADING_HOURS else None,
        "selected_days": [name for name, flag in _DAY_LABEL_FLAGS if flag] if USE_DAY_FILTER else None,
        "month_end_pause_days": MONTH_END_PAUSE_DAYS if USE_MONTH_END_PAUSE else None,
        "dashboard": "/dashboard",
        "signals_endpoint": "/signals?count=120",
    })


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return Response(inject_ea_ui(DASHBOARD_HTML), mimetype="text/html")


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

  .divider{display:flex;align-items:flex-end;gap:3px;height:34px;margin:22px 0 26px;opacity:.9;}
  .divider .bar{flex:1;max-width:9px;border-radius:2px 2px 0 0;background:var(--border);}
  .divider .bar.up{background:linear-gradient(180deg, var(--buy), transparent);}
  .divider .bar.down{background:linear-gradient(180deg, var(--sell), transparent);}
  .divider .bar.gold{background:linear-gradient(180deg, var(--gold), transparent);}

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
  .field-row{display:flex;align-items:center;justify-content:space-between;gap:10px;}
  .field-row label{font-size:12.5px;color:var(--text-dim);}
  .field-row input[type="number"]{
    width:70px;background:var(--surface-2);border:1px solid var(--border);color:var(--text);
    font-family:var(--mono);font-size:13px;border-radius:7px;padding:6px 8px;text-align:right;
  }

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
        <div class="sub">Trend + SMC + Flow proxy engine</div>
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

    <div class="card">
      <h2>Last N Signals</h2>
      <p class="desc">Walks back through history (past the 1000-candle-per-request cap, as far as needed) and replays the bot's own logic until it has this many trade signals, then reports the result.</p>
      <div class="field-row">
        <label for="signalCount">How many signals</label>
        <input type="number" id="signalCount" value="120" min="1" max="1000" step="1">
      </div>
      <div class="toggle-row">
        <span class="toggle-label">Send result to Telegram</span>
        <label class="switch"><input type="checkbox" id="notifySignals"><span class="slider"></span></label>
      </div>
      <button class="run secondary" id="btnSignals" onclick="runSignals()">
        <span class="spinner"></span><span class="label">Get Last N Signals</span>
      </button>
    </div>
  </div>

  <div class="results" id="results">
    <div class="empty">Run a check, pull stats, or fetch signal history to see live output here.</div>
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

async function runSignals(){
  const btn = document.getElementById('btnSignals');
  const notify = document.getElementById('notifySignals').checked;
  const count = document.getElementById('signalCount').value || 120;
  setLoading(btn, true);
  try{
    const res = await fetch(`/signals?count=${encodeURIComponent(count)}&notify=${notify ? 1 : 0}`);
    const data = await res.json();
    if(!res.ok){ renderError(data.error || 'Request failed'); return; }
    renderSignals(data);
  }catch(e){
    renderError('Could not reach the server: ' + e.message);
  }finally{
    setLoading(btn, false);
  }
}

function renderError(msg){
  document.getElementById('results').innerHTML = `<div class="error-box">⚠ ${esc(msg)}</div>`;
}

function eventBadge(event, weekendSkipped, hoursSkipped, daySkipped, monthSkipped){
  if(event === 'entry') return `<span class="badge neutral">New Entry</span>`;
  if(event === 'position_update') return `<span class="badge muted">Position Updated</span>`;
  if(weekendSkipped) return `<span class="badge muted">Weekend — Skipped</span>`;
  if(hoursSkipped) return `<span class="badge muted">Outside Trading Hours</span>`;
  if(daySkipped) return `<span class="badge muted">Day Not Selected</span>`;
  if(monthSkipped) return `<span class="badge muted">Month-End Pause</span>`;
  return `<span class="badge muted">No Signal</span>`;
}

function methodsKv(m){
  if(!m) return '';
  const on = v => v ? 'ON' : 'OFF';
  return `
    <div class="section-label">Trade Methods</div>
    <div class="kv"><span class="k">Trend (EMA-cross)</span><span class="v">${on(m.trend)}</span></div>
    <div class="kv"><span class="k">SMC (Sweep)</span><span class="v">${on(m.smc)}</span></div>
    <div class="kv"><span class="k">Forced daily</span><span class="v">${on(m.forced)}</span></div>
  `;
}

function positionKv(pos){
  if(!pos) return `<div class="kv"><span class="k">Open position</span><span class="v">None</span></div>`;
  const side = pos.dir === 1 ? 'BUY' : 'SELL';
  const sideClass = pos.dir === 1 ? 'buy' : 'sell';
  return `
    <div class="kv"><span class="k">Side</span><span class="v side-tag ${sideClass}">${side}${pos.setup ? ' (' + esc(pos.setup) + ')' : ''}</span></div>
    <div class="kv"><span class="k">Entry</span><span class="v">${fmtPrice(pos.entry)}</span></div>
    <div class="kv"><span class="k">Stop Loss</span><span class="v">${fmtPrice(pos.sl)}${pos.tp1_hit ? ' (locked)' : ''}</span></div>
    <div class="kv"><span class="k">TP1 / TP2</span><span class="v">${fmtPrice(pos.tp1)} / ${fmtPrice(pos.tp2)}</span></div>
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
        <div class="stat-box"><div class="k">Side</div><div class="v side-tag ${sideClass}">${data.side}${data.forced ? ' ⚡' : (data.smc_entry ? ' ◆' : '')}</div></div>
        <div class="stat-box"><div class="k">Setup</div><div class="v gold" style="font-size:13px">${esc(data.setup || '—')}</div></div>
        <div class="stat-box"><div class="k">Entry</div><div class="v gold">${fmtPrice(data.entry)}</div></div>
        <div class="stat-box"><div class="k">Stop Loss</div><div class="v neg">${fmtPrice(data.sl)}</div></div>
        <div class="stat-box"><div class="k">TP1 / TP2</div><div class="v pos" style="font-size:12.5px">${fmtPrice(data.tp1)} · ${fmtPrice(data.tp2)}</div></div>
      </div>
      ${data.forced ? `<p style="font-size:11.5px;color:var(--text-faint);margin-top:8px;">⚡ Opened by the Daily Trade Guarantee fallback, not the organic signal stack.</p>` : ''}
      ${data.smc_entry ? `<p style="font-size:11.5px;color:var(--text-faint);margin-top:8px;">◆ Standalone SMC setup: liquidity sweep + confirmation candle.</p>` : ''}
      ${data.pullback ? `<p style="font-size:11.5px;color:var(--text-faint);margin-top:8px;">↩ Opened after a pullback confirmation, not on the initial cross bar.</p>` : ''}
    `;
  }

  let pendingBlock = '';
  if(data.event !== 'entry' && data.pending && data.pending.dir){
    const dirLabel = data.pending.dir === 1 ? 'LONG' : 'SHORT';
    pendingBlock = `<div class="kv"><span class="k">Pending signal</span><span class="v">${dirLabel} (waiting for pullback)</span></div>`;
  }
  let smcBlock = '';
  if(data.smc){
    smcBlock = `<div class="kv"><span class="k">SMC score L / S</span><span class="v">${data.smc.score_long} / ${data.smc.score_short}</span></div>`;
  }

  let sessionBlock = '';
  if(data.weekend_skipped){
    sessionBlock = `<p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">🌙 Weekend — new-signal checks are paused. Any open position is still tracked.</p>`;
  }else if(data.trading_hours_skipped){
    sessionBlock = `<p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">🌙 Outside trading hours — new-signal checks are paused. Any open position is still tracked.</p>`;
  }else if(data.day_skipped){
    sessionBlock = `<p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">📅 Today isn't a selected trading day — new-signal checks are paused. Any open position is still tracked.</p>`;
  }else if(data.month_skipped){
    sessionBlock = `<p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">📅 Month-end pause — new-signal checks are paused. Any open position is still tracked.</p>`;
  }

  document.getElementById('results').innerHTML = `
    <div class="panel">
      <div class="panel-head">
        <div class="title">Check Result ${eventBadge(data.event, data.weekend_skipped, data.trading_hours_skipped, data.day_skipped, data.month_skipped)} ${notifiedBadge}</div>
        <div class="timestamp">${esc(data.bar_time || '—')}</div>
      </div>
      <div class="panel-body">
        ${entryBlock}
        ${sessionBlock}
        ${pendingBlock}
        ${smcBlock}
        ${methodsKv(data.trade_methods)}
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
        <td>${esc(t.setup || '')}</td>
        <td>${fmtPrice(t.entry)}</td>
        <td>${fmtPrice(t.exit)}</td>
        <td>${esc(t.result)}</td>
        <td class="${pnlC}">${fmtMoney(t.pnl)}</td>
      </tr>
    `;
  }).join('') || `<tr><td colspan="6" style="color:var(--text-faint);text-align:center;">No trades logged yet</td></tr>`;

  const dg = data.daily_guarantee || {};
  const ps = data.pending_signal || {};
  const th = data.trading_hours || {};
  const df = data.day_filter || {};
  const ss = data.setup_stats || {};
  const rg = data.risk_guard || {};

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

        ${methodsKv(data.trade_methods)}

        <div class="section-label">By Setup (wins / trades)</div>
        <div class="kv"><span class="k">Trend</span><span class="v">${ss.trend_w || 0} / ${ss.trend_n || 0}</span></div>
        <div class="kv"><span class="k">SMC</span><span class="v">${ss.smc_w || 0} / ${ss.smc_n || 0}</span></div>
        <div class="kv"><span class="k">Forced</span><span class="v">${ss.forced_w || 0} / ${ss.forced_n || 0}</span></div>

        <div class="section-label">Current Position</div>
        ${positionKv(data.position)}

        <div class="section-label">Pending Signal</div>
        <div class="kv"><span class="k">Armed</span><span class="v">${ps.dir ? (ps.dir === 1 ? 'LONG (waiting for pullback)' : 'SHORT (waiting for pullback)') : 'None'}</span></div>

        <div class="section-label">Risk Guard</div>
        <div class="kv"><span class="k">Losses today / cap</span><span class="v">${rg.losses_today || 0} / ${rg.max_losses_per_day ? rg.max_losses_per_day : '∞'}</span></div>

        <div class="section-label">Daily Trade Guarantee</div>
        <div class="kv"><span class="k">Status</span><span class="v">${dg.enabled ? 'Enabled' : 'Disabled'} @ ${String(dg.force_hour).padStart(2,'0')}:${String(dg.force_minute).padStart(2,'0')}</span></div>
        <div class="kv"><span class="k">Traded today</span><span class="v">${dg.traded_today ? 'Yes' : 'No'}</span></div>

        <div class="section-label">Trading Hours</div>
        <div class="kv"><span class="k">Window</span><span class="v">${th.enabled ? `${String(th.start_hour).padStart(2,'0')}:${String(th.start_minute).padStart(2,'0')}-${String(th.end_hour).padStart(2,'0')}:${String(th.end_minute).padStart(2,'0')}` : 'Off (24h)'}</span></div>

        <div class="section-label">Day Filter</div>
        <div class="kv"><span class="k">Selected days</span><span class="v">${df.enabled ? (df.selected_days && df.selected_days.length ? df.selected_days.join(', ') : 'None selected') : 'Off (all days)'}</span></div>

        <div class="section-label">Recent Trades</div>
        <div class="scroll-x">
          <table>
            <thead><tr><th>Side</th><th>Setup</th><th>Entry</th><th>Exit</th><th>Result</th><th>P&amp;L</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

function renderSignals(data){
  const notifiedBadge = data.telegram_notified ? `<span class="badge neutral">Sent to Telegram</span>` : '';
  const pnlClass = data.net_pnl > 0 ? 'pos' : (data.net_pnl < 0 ? 'neg' : '');
  const bs = data.by_setup || {};
  const rows = (data.signals || []).map(s => {
    const sideClass = s.side === 'BUY' ? 'buy' : 'sell';
    const pnlC = s.pnl > 0 ? 'pos' : (s.pnl < 0 ? 'neg' : '');
    const tag = s.forced ? ' ⚡' : (s.smc ? ' ◆' : (s.pullback ? ' ↩' : ''));
    const statusLabel = s.status === 'open' ? '<span class="badge neutral">Open</span>' : esc(s.result || '—');
    return `
      <tr>
        <td>${esc(s.entry_time)}</td>
        <td class="side-tag ${sideClass}">${esc(s.side)}${tag}</td>
        <td>${fmtPrice(s.entry)}</td>
        <td>${s.exit !== null ? fmtPrice(s.exit) : '—'}</td>
        <td>${statusLabel}</td>
        <td class="${pnlC}">${s.pnl !== null ? fmtMoney(s.pnl) : '—'}</td>
      </tr>
    `;
  }).join('') || `<tr><td colspan="6" style="color:var(--text-faint);text-align:center;">No signals found in the scanned window</td></tr>`;

  const setupLine = ['TREND','SMC','FORCED'].map(k => {
    const b = bs[k] || {n:0,wins:0,net_pnl:0};
    return `${k} ${b.wins}/${b.n} (${fmtMoney(b.net_pnl)})`;
  }).join(' · ');

  document.getElementById('results').innerHTML = `
    <div class="panel">
      <div class="panel-head">
        <div class="title">Last ${data.signals_returned} Signals ${notifiedBadge}</div>
        <div class="timestamp">${esc(data.data_covers_from || '')} → ${esc(data.data_covers_to || '')}</div>
      </div>
      <div class="panel-body">
        <div class="grid4">
          <div class="stat-box"><div class="k">Win Rate</div><div class="v gold">${data.win_rate}%</div></div>
          <div class="stat-box"><div class="k">Record</div><div class="v">${data.wins}W – ${data.losses}L</div></div>
          <div class="stat-box"><div class="k">Net P&amp;L</div><div class="v ${pnlClass}">${fmtMoney(data.net_pnl)}</div></div>
          <div class="stat-box"><div class="k">Days Scanned</div><div class="v">${data.days_scanned}</div></div>
        </div>
        <p style="font-size:11.5px;color:var(--text-faint);margin-top:10px;">
          ${data.closed_trades} closed${data.still_open ? `, ${data.still_open} still open` : ''} ·
          best ${fmtMoney(data.best_trade)} · worst ${fmtMoney(data.worst_trade)}
        </p>
        <p style="font-size:11.5px;color:var(--text-faint);margin-top:4px;">${setupLine}</p>
        <div class="section-label">Signal Log</div>
        <div class="scroll-x">
          <table>
            <thead><tr><th>Bar Time</th><th>Side</th><th>Entry</th><th>Exit</th><th>Result</th><th>P&amp;L</th></tr></thead>
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
