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
#  SYNCED WITH "Gold Signal Terminal v2.1 — Trend + SMC + Flow Proxy"
#  (all default VALUES are unchanged from the previous bot)
#
#  v2.1 sync:
#   - NEW: SMC_AFTER_FORCE (default true). SMC trades are locked until the
#     FORCED daily trade has fired AND closed (SL/TP) today, and are locked
#     before the force time. Only sweeps formed AFTER the unlock can trigger
#     an SMC entry (older sweeps are ignored). The SMC score used by the
#     Trend gate is not affected.
#   - Auto-unlock (after the force time) when: forced disabled / forced
#     skipped as "never valid" / a Trend trade pre-empted the forced trade
#     and has closed.
#   - A forced trade that closes on a later calendar day than it was entered
#     does NOT unlock the new day's SMC (same as the indicator).
#   - FIXED: PNL_MODE validation was `not in ("tp1_only")` (a substring test),
#     which forced every mode to tp1_only. Now valid modes are
#     ratchet / tp1_only / first_hit; anything else falls back to ratchet.
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
# SMC only after the forced trade of the day has closed (indicator v2.1)
SMC_AFTER_FORCE = _env_bool("SMC_AFTER_FORCE", True)

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
    PNL_MODE = "ratchet"

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
    if not cst or
