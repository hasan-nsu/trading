# Swing Trader Dashboard

A news triage + AI sentiment dashboard for swing trading. **Not a buy/sell signal generator.**

What it does:
- Pulls news for 18 curated stocks from Yahoo Finance + Google News
- Runs AI sentiment analysis on each stock's news (Groq Llama-3.3-70B)
- Shows macro brief (Fed, war, tariffs, inflation news)
- Flags stocks with upcoming earnings (don't hold through earnings!)
- Detects high news velocity (something is happening)
- Warns about correlated open positions
- "Calm Mode" hides noise, shows only what needs attention
- Free-form Q&A with AI grounded in your watchlist data

What it does NOT do:
- Tell you what to buy
- Predict prices
- Replace your stop losses or journaling

---

## Setup — Run on Streamlit Cloud (always-on, free)

### Step 1: Get a free Groq API key (2 minutes)
1. Go to https://console.groq.com
2. Sign up (free, no credit card)
3. Click "API Keys" → "Create API Key"
4. Copy it. Keep it safe.

### Step 2: Push code to GitHub (5 minutes)
1. Create a new GitHub repo (private if you want)
2. Add these files:
   - `dashboard.py`
   - `requirements.txt`
   - `.gitignore`
3. **Do NOT push secrets.toml** — that's why .gitignore is there.

### Step 3: Deploy on Streamlit Cloud (5 minutes)
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Pick your repo, branch `main`, main file `dashboard.py`
5. Click "Advanced settings" → "Secrets"
6. Paste this:
   ```
   GROQ_API_KEY = "your_groq_key_here"
   ```
7. Click Deploy. Done.

You get a URL like `https://your-app.streamlit.app` — open it on phone or laptop.

---

## Setup — Run locally (simpler, no cloud needed)

```bash
pip install -r requirements.txt
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml
streamlit run dashboard.py
```

Browser opens at http://localhost:8501

---

## Customizing the watchlist

Edit `WATCHLIST` dict at top of `dashboard.py`. Each entry:
```python
"TICKER": {"name": "Company", "sector": "Sector", "type": "swing|stable", "priority": "high|med|low"},
```

`priority` affects display order. `type=swing` shown before `stable`.

---

## Cost

- Streamlit Cloud: **free**
- Groq API free tier: 14,400 requests/day. Dashboard uses ~30/refresh. **You'll never hit the limit.**
- yfinance, Google News RSS: **free, no key needed**

Total: **€0/month forever** (unless Groq changes their free tier).

---

## If Groq's free tier changes

Swap to Google Gemini Flash (free 1500/day):
1. `pip install google-generativeai` and add to requirements.txt
2. In `get_groq_client()` and the AI calls, swap to Gemini's API
3. Re-deploy

Or just run without AI — the dashboard falls back to keyword sentiment automatically.

---

## Honest disclaimer

This is a tool for filtering news, not generating trades. AI sentiment is wrong sometimes. Headlines can be misleading. Markets price in news before retail sees it. Always:
- Use stop loss orders
- Risk no more than 1-2% per trade
- Don't hold through earnings without reason
- Journal every trade

Not financial advice. Risk only what you can afford to lose.
