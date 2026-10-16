"""
ea_control.py -- remote enable/disable switch + schedule for MT5 Expert Advisors.

Each MT5 terminal runs an EA that polls GET /ea/poll every few seconds,
identifying itself by account number. The dashboard flips a per-account
"enabled" flag. Accounts register themselves on their first poll, so two
MT5 windows with different accounts appear as two separate rows.

Per account you can set:
  * a daily auto schedule (Cambodia time, 24h): "enable at HH:MM, disable at HH:MM"
    (overnight windows such as 22:00 -> 06:00 work too)
  * what happens to open trades when the EA gets disabled (manual OR scheduled):
        manage     keep managing them (SL/TP ladder continues)        [default]
        remove_tp  keep them open, REMOVE the TP, KEEP the SL, hands off
        close      close all trades / pending orders (one-shot)

Env vars:
  EA_API_TOKEN          required. Shared secret used by the EAs and the dashboard.
  EA_TIMEZONE           schedule timezone (default Asia/Phnom_Penh, UTC+7, no DST).
  EA_DEFAULT_ENABLED    "true"/"false" (default false): state of a newly seen account.
  EA_DEFAULT_ON_DISABLE manage | remove_tp | close (default manage) for new accounts.
  EA_OFFLINE_AFTER_SEC  seconds without a poll before an account shows offline (default 30).
  EA_STATE_FILE         where switch state is stored (default ea_state.json).
"""
import os
import re
import json
import time
import hmac
import threading
from datetime import datetime, timedelta, timezone

from flask import request, jsonify, Response

EA_STATE_FILE = os.environ.get("EA_STATE_FILE", "ea_state.json")
EA_API_TOKEN = os.environ.get("EA_API_TOKEN", "")
EA_DEFAULT_ENABLED = os.environ.get("EA_DEFAULT_ENABLED", "false").lower() == "true"
EA_OFFLINE_AFTER_SEC = int(os.environ.get("EA_OFFLINE_AFTER_SEC", 30))
EA_TIMEZONE = os.environ.get("EA_TIMEZONE", "Asia/Phnom_Penh")

ON_DISABLE_MODES = ("manage", "remove_tp", "close")
EA_DEFAULT_ON_DISABLE = os.environ.get("EA_DEFAULT_ON_DISABLE", "manage")
if EA_DEFAULT_ON_DISABLE not in ON_DISABLE_MODES:
    EA_DEFAULT_ON_DISABLE = "manage"

# Cambodia has no DST, so if tz data is missing we fall back to a fixed UTC+7.
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(EA_TIMEZONE)
except Exception:
    _TZ = timezone(timedelta(hours=7))

_HM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

# Separate file + lock so EA polls never collide with the bot's state.json writes.
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# storage / helpers
# ---------------------------------------------------------------------------
def _load():
    if os.path.exists(EA_STATE_FILE):
        try:
            with open(EA_STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save(data):
    tmp = EA_STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, EA_STATE_FILE)


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _now_local():
    return datetime.now(_TZ)


def _parse_hm(s):
    """'HH:MM' (24h) -> minutes since midnight, or None if invalid."""
    m = _HM_RE.match(str(s or "").strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _in_window(minutes, start, end):
    if start < end:
        return start <= minutes < end
    return minutes >= start or minutes < end  # overnight window


def _norm(a):
    """Make sure an account record has every field (older files won't)."""
    a.setdefault("enabled", EA_DEFAULT_ENABLED)
    a.setdefault("close_requested", False)
    a.setdefault("on_disable", EA_DEFAULT_ON_DISABLE)
    sch = a.setdefault("schedule", {})
    sch.setdefault("on", False)
    sch.setdefault("start", "06:30")
    sch.setdefault("end", "23:30")
    sch.setdefault("last_in", None)
    return a


def _new_account():
    return _norm({"first_seen": time.time()})


def _set_enabled(a, enabled, source):
    """Single place where the switch changes, so the disable policy is applied
    the same way for manual clicks and for the schedule."""
    a["enabled"] = bool(enabled)
    a["changed_at"] = time.time()
    a["changed_by"] = source
    # 'close' is a one-shot command delivered on the next poll.
    # 'remove_tp' needs no flag: the server reports release=1 for as long as
    # the EA is disabled, and the EA strips TP from the trades it manages.
    a["close_requested"] = (not enabled) and a.get("on_disable") == "close"


def _schedule_inside(a, now_local):
    s = a["schedule"]
    start, end = _parse_hm(s.get("start")), _parse_hm(s.get("end"))
    if start is None or end is None or start == end:
        return None
    return _in_window(now_local.hour * 60 + now_local.minute, start, end)


def _apply_schedule(a, now_local):
    """Fire enable/disable when the clock crosses a window edge.
    Returns True if the record changed."""
    s = a["schedule"]
    if not s.get("on"):
        return False
    inside = _schedule_inside(a, now_local)
    if inside is None:
        return False
    if s.get("last_in") is None:
        s["last_in"] = inside
        return True
    if inside != s["last_in"]:
        s["last_in"] = inside
        _set_enabled(a, inside, "schedule")
        return True
    return False


def _auth_error():
    if not EA_API_TOKEN:
        return jsonify({"error": "EA_API_TOKEN is not set on the server"}), 503
    supplied = request.headers.get("X-EA-Token") or request.args.get("token", "")
    if not hmac.compare_digest(supplied.encode(), EA_API_TOKEN.encode()):
        return jsonify({"error": "unauthorized"}), 401
    return None


# ---------------------------------------------------------------------------
# Dashboard pieces (injected into DASHBOARD_HTML)
# ---------------------------------------------------------------------------
EA_CSS = r"""
  .ea-row{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;
    padding:14px 0;border-bottom:1px solid var(--border);}
  .ea-row:last-child{border-bottom:none;}
  .ea-id{display:flex;flex-direction:column;gap:3px;min-width:180px;}
  .ea-id .num{font-family:var(--mono);font-weight:600;font-size:14px;display:flex;align-items:center;gap:8px;}
  .ea-id .meta{font-size:11px;color:var(--text-faint);font-family:var(--mono);}
  .ea-nums{font-family:var(--mono);font-size:12px;color:var(--text-dim);display:flex;gap:14px;flex-wrap:wrap;}
  .ea-ctl{display:flex;align-items:center;gap:10px;}
  .ea-state{font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:.4px;min-width:30px;text-align:right;}
  .ea-state.on{color:var(--buy);} .ea-state.off{color:var(--text-faint);}
  .dot.off{animation:none;background:var(--text-faint);box-shadow:none;}
  .ea-input{background:var(--surface-2);border:1px solid var(--border);color:var(--text);
    font-family:var(--mono);font-size:13px;border-radius:7px;padding:6px 10px;width:200px;}
  .ea-input.sm{width:auto;padding:5px 8px;font-size:12.5px;}
  .ea-btn{background:transparent;border:1px solid var(--gold-dim);color:var(--gold);border-radius:8px;
    font-family:var(--sans);font-weight:600;font-size:12px;padding:7px 12px;cursor:pointer;}
  .ea-btn:hover{background:var(--gold-glow);}
  .ea-btn.danger{border-color:rgba(224,87,76,.5);color:var(--sell);}
  .ea-btn.danger:hover{background:var(--sell-glow);}
  .ea-settings{flex-basis:100%;display:flex;flex-direction:column;gap:9px;
    background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:11px 13px;}
  .ea-set-line{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--text-dim);}
  .ea-set-line .lbl{min-width:0;}
  .ea-set-line .sep{color:var(--text-faint);}
  .ea-set-line label.chk{display:flex;align-items:center;gap:5px;cursor:pointer;}
  .ea-pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:100px;background:#20262e;color:var(--text-faint);}
  .ea-pill.in{background:var(--buy-glow);color:var(--buy);}
  .ea-clock{font-family:var(--mono);font-size:11.5px;color:var(--gold);}
"""

EA_CARD_HTML = r"""
  <div class="card" id="eaCard" style="margin-top:16px">
    <div class="panel-head" style="padding:0;border:none;background:none;">
      <h2>MT5 EA Control</h2>
      <div class="field-row" style="gap:8px">
        <label for="eaKey">Access key</label>
        <input class="ea-input" type="password" id="eaKey" placeholder="EA_API_TOKEN" onchange="eaSaveKey()">
      </div>
    </div>
    <p class="desc">Switch each MT5 account's EA on or off, now or on a daily schedule. Choose per account what happens to open trades when it is disabled. Accounts appear automatically once their EA connects.</p>
    <div class="ea-clock" id="eaClock"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button class="ea-btn" onclick="eaToggle('all', true)">Enable all</button>
      <button class="ea-btn danger" onclick="eaToggle('all', false)">Disable all</button>
    </div>
    <div id="eaList"></div>
  </div>
"""

EA_JS = r"""
/* ---------- MT5 EA control ---------- */
(function(){ try{ const k = localStorage.getItem('eaKey'); if(k) document.getElementById('eaKey').value = k; }catch(e){} })();
let eaDirtyUntil = 0;                       // pause auto-refresh while the user is editing a schedule
function eaDirty(){ eaDirtyUntil = Date.now() + 30000; }
function eaKey(){ return (document.getElementById('eaKey').value || '').trim(); }
function eaHeaders(){ return {'X-EA-Token': eaKey(), 'Content-Type': 'application/json'}; }
function eaSaveKey(){ try{ localStorage.setItem('eaKey', eaKey()); }catch(e){} eaLoad(true); }
function eaPad(n){ return String(n).padStart(2, '0'); }
function eaOpts(count, sel){
  let h = '';
  for(let i = 0; i < count; i++) h += `<option value="${i}" ${i === sel ? 'selected' : ''}>${eaPad(i)}</option>`;
  return h;
}
function eaAgo(s){
  if(s === null || s === undefined) return 'never';
  if(s < 60) return Math.round(s) + 's ago';
  if(s < 3600) return Math.round(s/60) + 'm ago';
  return Math.round(s/3600) + 'h ago';
}
function eaMoney(v){ return (v === null || v === undefined) ? '—' : '$' + Number(v).toFixed(2); }

async function eaPost(path, body){
  try{
    const res = await fetch(path, {method:'POST', headers: eaHeaders(), body: JSON.stringify(body)});
    const d = await res.json().catch(() => ({}));
    if(!res.ok){ alert(d.error || 'Request failed'); return false; }
    return true;
  }catch(e){ alert('Could not reach the server: ' + e.message); return false; }
}

async function eaLoad(force){
  const box = document.getElementById('eaList');
  if(!eaKey()){ box.innerHTML = '<div class="empty">Enter your access key to manage MT5 accounts.</div>'; return; }
  if(!force && Date.now() < eaDirtyUntil) return;
  try{
    const res = await fetch('/ea/accounts', {headers: eaHeaders()});
    const data = await res.json();
    if(!res.ok){ box.innerHTML = `<div class="error-box">⚠ ${esc(data.error || 'Request failed')}</div>`; return; }
    document.getElementById('eaClock').textContent = `Schedule clock: ${data.now_local} (${data.timezone}, 24h)`;
    eaRender(data.accounts || []);
  }catch(e){
    box.innerHTML = `<div class="error-box">⚠ Could not reach the server: ${esc(e.message)}</div>`;
  }
}

function eaRow(a){
  const s = a.schedule || {on:false, start:'06:30', end:'23:30', inside:null};
  const [sh, sm] = String(s.start || '06:30').split(':').map(Number);
  const [eh, em] = String(s.end || '23:30').split(':').map(Number);
  const pill = !s.on ? '<span class="ea-pill">schedule off</span>'
             : (s.inside ? '<span class="ea-pill in">inside window</span>' : '<span class="ea-pill">outside window</span>');
  const id = a.account;
  return `
    <div class="ea-row">
      <div class="ea-id">
        <span class="num"><span class="dot ${a.online ? '' : 'off'}"></span>${esc(id)}</span>
        <span class="meta">${esc(a.name || '')} · ${esc(a.server || '')} · ${a.online ? 'online' : 'offline'} (${eaAgo(a.age_sec)})</span>
      </div>
      <div class="ea-nums">
        <span>Bal ${eaMoney(a.balance)}</span>
        <span>Eq ${eaMoney(a.equity)}</span>
        <span>Pos ${a.positions ?? 0}</span>
      </div>
      <div class="ea-ctl">
        <span class="ea-state ${a.enabled ? 'on' : 'off'}">${a.enabled ? 'ON' : 'OFF'}</span>
        <label class="switch"><input type="checkbox" ${a.enabled ? 'checked' : ''} onchange="eaToggle('${id}', this.checked)"><span class="slider"></span></label>
        <button class="ea-btn danger" title="Forget this account" onclick="eaRemove('${id}')">✕</button>
      </div>
      <div class="ea-settings">
        <div class="ea-set-line">
          <span class="lbl">When disabled, open trades:</span>
          <select class="ea-input sm" onchange="eaSaveOnDisable('${id}', this.value)">
            <option value="manage" ${a.on_disable === 'manage' ? 'selected' : ''}>Keep managing (SL/TP ladder continues)</option>
            <option value="remove_tp" ${a.on_disable === 'remove_tp' ? 'selected' : ''}>Keep open — remove TP, keep SL</option>
            <option value="close" ${a.on_disable === 'close' ? 'selected' : ''}>Close all trades</option>
          </select>
        </div>
        <div class="ea-set-line">
          <label class="switch"><input type="checkbox" id="sch_on_${id}" ${s.on ? 'checked' : ''} onchange="eaDirty()"><span class="slider"></span></label>
          <span class="lbl">Auto schedule</span>
          <span class="sep">·</span> Enable at
          <select class="ea-input sm" id="sch_sh_${id}" onchange="eaDirty()">${eaOpts(24, sh)}</select>:<select class="ea-input sm" id="sch_sm_${id}" onchange="eaDirty()">${eaOpts(60, sm)}</select>
          <span class="sep">·</span> Disable at
          <select class="ea-input sm" id="sch_eh_${id}" onchange="eaDirty()">${eaOpts(24, eh)}</select>:<select class="ea-input sm" id="sch_em_${id}" onchange="eaDirty()">${eaOpts(60, em)}</select>
          <button class="ea-btn" onclick="eaSaveSchedule('${id}')">Save</button>
          <label class="chk"><input type="checkbox" id="sch_all_${id}"> all accounts</label>
          ${pill}
        </div>
      </div>
    </div>`;
}

function eaRender(list){
  const box = document.getElementById('eaList');
  if(!list.length){
    box.innerHTML = '<div class="empty">No MT5 account has connected yet. Attach the EA to a chart and allow WebRequest for this site URL.</div>';
    return;
  }
  box.innerHTML = list.map(eaRow).join('');
}

async function eaToggle(account, enabled){
  await eaPost('/ea/toggle', {account: account, enabled: enabled});
  eaLoad(true);
}

async function eaSaveOnDisable(account, mode){
  await eaPost('/ea/settings', {account: account, on_disable: mode});
  eaLoad(true);
}

async function eaSaveSchedule(account){
  const v = id => document.getElementById(id + account);
  const body = {
    account: v('sch_all_').checked ? 'all' : account,
    schedule: {
      on: v('sch_on_').checked,
      start: eaPad(v('sch_sh_').value) + ':' + eaPad(v('sch_sm_').value),
      end:   eaPad(v('sch_eh_').value) + ':' + eaPad(v('sch_em_').value)
    }
  };
  if(await eaPost('/ea/settings', body)) eaDirtyUntil = 0;
  eaLoad(true);
}

async function eaRemove(account){
  if(!confirm('Forget account ' + account + '? It will reappear if its EA is still running.')) return;
  await eaPost('/ea/remove', {account: account});
  eaLoad(true);
}

eaLoad(true);
setInterval(() => eaLoad(false), 5000);
"""


def inject_ea_ui(html):
    """Insert the EA card, CSS and JS into the existing dashboard HTML."""
    html = html.replace("</style>", EA_CSS + "</style>", 1)
    html = html.replace('<div class="results" id="results">', EA_CARD_HTML + '\n  <div class="results" id="results">', 1)
    idx = html.rfind("</script>")
    if idx != -1:
        html = html[:idx] + EA_JS + html[idx:]
    return html


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def register_ea_control(app):

    @app.route("/ea/poll", methods=["GET"])
    def ea_poll():
        """Called by the EA. Plain-text reply so MQL5 can parse it trivially:
              enabled=1
              close=0
              release=0
        close=1    delivered exactly once after a disable with on_disable=close.
        release=1  sent on every poll while the EA is disabled and on_disable=remove_tp
                   (the EA then strips the TP from trades it manages, keeping the SL)."""
        err = _auth_error()
        if err:
            return err
        acct = request.args.get("account", "").strip()
        if not acct.isdigit():
            return Response("error=bad_account\n", status=400, mimetype="text/plain")

        now_local = _now_local()
        with _lock:
            data = _load()
            a = _norm(data.setdefault(acct, _new_account()))
            a["name"] = request.args.get("name", a.get("name", ""))[:60]
            a["server"] = request.args.get("server", a.get("server", ""))[:60]
            a["balance"] = _num(request.args.get("balance"), a.get("balance"))
            a["equity"] = _num(request.args.get("equity"), a.get("equity"))
            a["positions"] = int(_num(request.args.get("positions"), a.get("positions", 0)))
            a["last_seen"] = time.time()

            _apply_schedule(a, now_local)

            enabled = bool(a["enabled"])
            close = bool(a.get("close_requested"))
            a["close_requested"] = False
            release = (not enabled) and a.get("on_disable") == "remove_tp"
            _save(data)

        return Response(
            f"enabled={1 if enabled else 0}\nclose={1 if close else 0}\nrelease={1 if release else 0}\n",
            mimetype="text/plain",
        )

    @app.route("/ea/accounts", methods=["GET"])
    def ea_accounts():
        err = _auth_error()
        if err:
            return err
        now = time.time()
        now_local = _now_local()
        with _lock:
            data = _load()
            changed = False
            out = []
            for acct, a in sorted(data.items()):
                _norm(a)
                if _apply_schedule(a, now_local):
                    changed = True
                last = a.get("last_seen")
                age = (now - last) if last else None
                s = a["schedule"]
                out.append({
                    "account": acct,
                    "name": a.get("name", ""),
                    "server": a.get("server", ""),
                    "enabled": bool(a.get("enabled")),
                    "on_disable": a.get("on_disable"),
                    "schedule": {
                        "on": bool(s.get("on")),
                        "start": s.get("start"),
                        "end": s.get("end"),
                        "inside": _schedule_inside(a, now_local),
                    },
                    "balance": a.get("balance"),
                    "equity": a.get("equity"),
                    "positions": a.get("positions", 0),
                    "age_sec": age,
                    "online": age is not None and age <= EA_OFFLINE_AFTER_SEC,
                })
            if changed:
                _save(data)
        return jsonify({
            "accounts": out,
            "now_local": now_local.strftime("%H:%M:%S"),
            "timezone": EA_TIMEZONE,
        })

    @app.route("/ea/toggle", methods=["POST"])
    def ea_toggle():
        err = _auth_error()
        if err:
            return err
        body = request.get_json(silent=True) or {}
        acct = str(body.get("account", "")).strip()
        enabled = bool(body.get("enabled"))

        with _lock:
            data = _load()
            if acct == "all":
                targets = list(data.keys())
            elif acct in data:
                targets = [acct]
            else:
                return jsonify({"error": f"unknown account {acct}"}), 404
            for t in targets:
                _set_enabled(_norm(data[t]), enabled, "manual")
            _save(data)
        return jsonify({"ok": True, "accounts": targets, "enabled": enabled})

    @app.route("/ea/settings", methods=["POST"])
    def ea_settings():
        """Body: {account | "all", on_disable?: manage|remove_tp|close,
                  schedule?: {on: bool, start: "HH:MM", end: "HH:MM"}}
        Saving a schedule with on=true applies the window to the switch immediately."""
        err = _auth_error()
        if err:
            return err
        body = request.get_json(silent=True) or {}
        acct = str(body.get("account", "")).strip()

        on_disable = body.get("on_disable")
        if on_disable is not None and on_disable not in ON_DISABLE_MODES:
            return jsonify({"error": "on_disable must be one of: " + ", ".join(ON_DISABLE_MODES)}), 400

        sch = body.get("schedule")
        sch_on, start, end = False, None, None
        if sch is not None:
            sch_on = bool(sch.get("on"))
            start, end = str(sch.get("start", "")).strip(), str(sch.get("end", "")).strip()
            s_min, e_min = _parse_hm(start), _parse_hm(end)
            if sch_on:
                if s_min is None or e_min is None:
                    return jsonify({"error": "Times must be HH:MM in 24-hour format"}), 400
                if s_min == e_min:
                    return jsonify({"error": "Enable time and disable time must be different"}), 400

        now_local = _now_local()
        with _lock:
            data = _load()
            if acct == "all":
                targets = list(data.keys())
            elif acct in data:
                targets = [acct]
            else:
                return jsonify({"error": f"unknown account {acct}"}), 404

            for t in targets:
                a = _norm(data[t])
                if on_disable is not None:
                    a["on_disable"] = on_disable
                if sch is not None:
                    cfg = a["schedule"]
                    cfg["on"] = sch_on
                    if _parse_hm(start) is not None and _parse_hm(end) is not None:
                        cfg["start"], cfg["end"] = start, end
                    if sch_on:
                        inside = _schedule_inside(a, now_local)
                        cfg["last_in"] = inside
                        if inside is not None and bool(a["enabled"]) != inside:
                            _set_enabled(a, inside, "schedule")
                    else:
                        cfg["last_in"] = None
            _save(data)
        return jsonify({"ok": True, "accounts": targets})

    @app.route("/ea/remove", methods=["POST"])
    def ea_remove():
        err = _auth_error()
        if err:
            return err
        acct = str((request.get_json(silent=True) or {}).get("account", "")).strip()
        with _lock:
            data = _load()
            data.pop(acct, None)
            _save(data)
        return jsonify({"ok": True})

    return inject_ea_ui
