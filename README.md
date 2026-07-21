# XAU Gold Signal → Telegram (free, no TradingView subscription)

Polls free gold price data every 5 minutes, re-runs the same EMA/RSI/ATR/trend
logic from your original Pine Script indicator, and sends a Telegram message
whenever a BUY or SELL signal fires. No TradingView webhook, no paid plan.

## 1. Get a free Twelve Data API key
1. Go to https://twelvedata.com and sign up (free, no card required).
2. Copy your API key from the dashboard.
3. Free tier covers 800 requests/day, 8/minute — this app uses ~3 requests
   every 5 minutes (~864/day), which fits comfortably if you space checks
   every 5 minutes as planned.

## 2. Get your Telegram bot token + chat ID
(Skip if you already have these from earlier.)
1. Message @BotFather on Telegram → `/newbot` → copy the token.
2. Message your bot once, then visit:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find `"chat":{"id": ...}` — that's your chat ID.

## 3. Deploy to Render (free)
1. Push this folder to a GitHub repo.
2. On Render.com → New → Web Service → connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add these Environment Variables in Render's dashboard:
   - `TWELVE_DATA_API_KEY` = your Twelve Data key
   - `TELEGRAM_BOT_TOKEN` = your bot token
   - `TELEGRAM_CHAT_ID` = your chat ID
   - `SYMBOL` = `XAU/USD` (default, change if trading something else)
6. Deploy. Render will give you a URL like `https://your-app.onrender.com`.

## 4. Keep it alive + checking every 5 minutes
Render's free tier sleeps after inactivity, so use a free external pinger
to hit your `/check` endpoint every 5 minutes — this both keeps the app
awake and triggers the actual signal check.

1. Go to https://cron-job.org (free) and sign up.
2. Create a new cron job:
   - URL: `https://your-app.onrender.com/check`
   - Schedule: every 5 minutes
3. Save. That's it — you'll now get a Telegram message whenever a
   BUY/SELL signal fires.

## Notes
- `state.json` tracks the last alerted candle to avoid duplicate messages.
  On Render's free tier this file may reset if the service redeploys or
  restarts — in that case you might get one repeat alert after a restart,
  which is a minor cosmetic issue, not a functional problem.
- This checks entry signals only (matches the original `alertcondition()`
  calls in your Pine Script) — it does not track TP1/TP2/TP3 partial exits
  or moving stop-losses, since those weren't wired to alerts originally either.
- To test manually anytime, just visit `https://your-app.onrender.com/check`
  in a browser — it returns JSON showing whether a signal fired.
