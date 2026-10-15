"""
ea_control.py -- remote enable/disable switch for MT5 Expert Advisors.

Each MT5 terminal runs an EA that polls GET /ea/poll every few seconds,
identifying itself by account number. The dashboard flips a per-account
"enabled" flag. Accounts register themselves on their first poll, so two
MT5 windows with different accounts appear as two separate switches.

Env vars:
  EA_API_TOKEN          required. Shared secret used by the EAs and the dashboard.
  EA_DEFAULT_ENABLED    "true"/"false" (default false): state of a newly seen account.
  EA_OFFLINE_AFTER_SEC  seconds without a poll before an account shows offline (default 30).
  EA_STATE_FILE         where switch state is stored (default ea_state.json).
"""
import os
import json
import time
import hmac
import threading

from flask import request, jsonify, Response

EA_STATE_FILE = os.environ.get("EA_STATE_FILE", "ea_state.json")
EA_API_TOKEN = os.environ.get("EA_API_TOKEN", "")
EA_DEFAULT_ENABLED = os.environ.get("EA_DEFAULT_ENABLED", "false").lower() == "true"
EA_OFFLINE_AFTER_SEC = int(os.environ.get("EA_OFFLINE_AFTER_SEC", 30))

# Separate file + lock so EA polls never collide with the bot's state.json writes.
_lock = threading.Lock()


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
    padding:12px 0;border-bottom:1px solid var(--border);}
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
  .ea-btn{background:transparent;border:1px solid var(--gold-dim);color:var(--gold);border-radius:8px;
    font-family:var(--sans);font-weight:600;font-size:12px;padding:7px 12px;cursor:pointer;}
  .ea-btn:hover{background:var(--gold-glow);}
  .ea-btn.danger{border-color:rgba(224,87,76,.5);color:var(--sell);}
  .ea-btn.danger:hover{background:var(--sell-glow);}
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
    <p class="desc">Switch each MT5 account's EA on or off. Disabled EAs stop opening new trades within a few seconds. Accounts appear here automatically once their EA connects.</p>
    <div class="toggle-row">
      <span class="toggle-label">Also close open positions &amp; pending orders when disabling</span>
      <label class="switch"><input type="checkbox" id="eaCloseOnDisable"><span class="slider"></span></label>
    </div>
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
function eaKey(){ return (document.getElementById('eaKey').value || '').trim(); }
function eaHeaders(){ return {'X-EA-Token': eaKey(), 'Content-Type': 'application/json'}; }
function eaSaveKey(){ try{ localStorage.setItem('eaKey', eaKey()); }catch(e){} eaLoad(); }
function eaAgo(s){
  if(s === null || s === undefined) return 'never';
  if(s < 60) return Math.round(s) + 's ago';
  if(s < 3600) return Math.round(s/60) + 'm ago';
  return Math.round(s/3600) + 'h ago';
}
function eaMoney(v){ return (v === null || v === undefined) ? '—' : '$' + Number(v).toFixed(2); }

async function eaLoad(){
  const box = document.getElementById('eaList');
  if(!eaKey()){ box.innerHTML = '<div class="empty">Enter your access key to manage MT5 accounts.</div>'; return; }
  try{
    const res = await fetch('/ea/accounts', {headers: eaHeaders()});
    const data = await res.json();
    if(!res.ok){ box.innerHTML = `<div class="error-box">⚠ ${esc(data.error || 'Request failed')}</div>`; return; }
    eaRender(data.accounts || []);
  }catch(e){
    box.innerHTML = `<div class="error-box">⚠ Could not reach the server: ${esc(e.message)}</div>`;
  }
}

function eaRender(list){
  const box = document.getElementById('eaList');
  if(!list.length){
    box.innerHTML = '<div class="empty">No MT5 account has connected yet. Attach the EA to a chart and allow WebRequest for this site URL.</div>';
    return;
  }
  box.innerHTML = list.map(a => `
    <div class="ea-row">
      <div class="ea-id">
        <span class="num"><span class="dot ${a.online ? '' : 'off'}"></span>${esc(a.account)}</span>
        <span class="meta">${esc(a.name || '')} · ${esc(a.server || '')} · ${a.online ? 'online' : 'offline'} (${eaAgo(a.age_sec)})</span>
      </div>
      <div class="ea-nums">
        <span>Bal ${eaMoney(a.balance)}</span>
        <span>Eq ${eaMoney(a.equity)}</span>
        <span>Pos ${a.positions ?? 0}</span>
      </div>
      <div class="ea-ctl">
        <span class="ea-state ${a.enabled ? 'on' : 'off'}">${a.enabled ? 'ON' : 'OFF'}</span>
        <label class="switch"><input type="checkbox" ${a.enabled ? 'checked' : ''} onchange="eaToggle('${a.account}', this.checked)"><span class="slider"></span></label>
        <button class="ea-btn danger" title="Forget this account" onclick="eaRemove('${a.account}')">✕</button>
      </div>
    </div>`).join('');
}

async function eaToggle(account, enabled){
  const closeNow = document.getElementById('eaCloseOnDisable').checked;
  try{
    const res = await fetch('/ea/toggle', {method:'POST', headers: eaHeaders(),
      body: JSON.stringify({account: account, enabled: enabled, close_positions: closeNow})});
    if(!res.ok){ const d = await res.json(); alert(d.error || 'Toggle failed'); }
  }catch(e){ alert('Could not reach the server: ' + e.message); }
  eaLoad();
}

async function eaRemove(account){
  if(!confirm('Forget account ' + account + '? It will reappear if its EA is still running.')) return;
  await fetch('/ea/remove', {method:'POST', headers: eaHeaders(), body: JSON.stringify({account: account})});
  eaLoad();
}

eaLoad();
setInterval(eaLoad, 5000);
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
        'close=1' is delivered exactly once after a 'disable + close' command."""
        err = _auth_error()
        if err:
            return err
        acct = request.args.get("account", "").strip()
        if not acct.isdigit():
            return Response("error=bad_account\n", status=400, mimetype="text/plain")

        with _lock:
            data = _load()
            a = data.setdefault(acct, {
                "enabled": EA_DEFAULT_ENABLED,
                "close_requested": False,
                "first_seen": time.time(),
            })
            a["name"] = request.args.get("name", a.get("name", ""))[:60]
            a["server"] = request.args.get("server", a.get("server", ""))[:60]
            a["balance"] = _num(request.args.get("balance"), a.get("balance"))
            a["equity"] = _num(request.args.get("equity"), a.get("equity"))
            a["positions"] = int(_num(request.args.get("positions"), a.get("positions", 0)))
            a["last_seen"] = time.time()
            enabled = bool(a.get("enabled"))
            close = bool(a.get("close_requested"))
            a["close_requested"] = False
            _save(data)

        return Response(f"enabled={1 if enabled else 0}\nclose={1 if close else 0}\n", mimetype="text/plain")

    @app.route("/ea/accounts", methods=["GET"])
    def ea_accounts():
        err = _auth_error()
        if err:
            return err
        now = time.time()
        with _lock:
            data = _load()
        out = []
        for acct, a in sorted(data.items()):
            last = a.get("last_seen")
            age = (now - last) if last else None
            out.append({
                "account": acct,
                "name": a.get("name", ""),
                "server": a.get("server", ""),
                "enabled": bool(a.get("enabled")),
                "balance": a.get("balance"),
                "equity": a.get("equity"),
                "positions": a.get("positions", 0),
                "age_sec": age,
                "online": age is not None and age <= EA_OFFLINE_AFTER_SEC,
            })
        return jsonify({"accounts": out})

    @app.route("/ea/toggle", methods=["POST"])
    def ea_toggle():
        err = _auth_error()
        if err:
            return err
        body = request.get_json(silent=True) or {}
        acct = str(body.get("account", "")).strip()
        enabled = bool(body.get("enabled"))
        close = bool(body.get("close_positions"))

        with _lock:
            data = _load()
            if acct == "all":
                targets = list(data.keys())
            elif acct in data:
                targets = [acct]
            else:
                return jsonify({"error": f"unknown account {acct}"}), 404
            for t in targets:
                data[t]["enabled"] = enabled
                data[t]["close_requested"] = (not enabled) and close
                data[t]["changed_at"] = time.time()
            _save(data)
        return jsonify({"ok": True, "accounts": targets, "enabled": enabled})

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
