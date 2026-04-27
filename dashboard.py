"""
Swing Trader Dashboard
======================
News + AI sentiment analysis for a curated watchlist.
Designed for swing trading (days to weeks holding period).

NOT a buy/sell signal generator. A news triage tool.

Author: Built for Shakil
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser
import plotly.graph_objects as go
from groq import Groq

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Swing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Watchlist with metadata
WATCHLIST = {
    # YOUR CORE THREE
    "NVDA":  {"name": "NVIDIA",          "sector": "Tech/AI",        "type": "swing",  "priority": "high"},
    "AMD":   {"name": "AMD",             "sector": "Tech/AI",        "type": "swing",  "priority": "high"},
    "TSLA":  {"name": "Tesla",           "sector": "Auto/Tech",      "type": "swing",  "priority": "high"},
    # TECH ADDS
    "PLTR":  {"name": "Palantir",        "sector": "Tech/AI",        "type": "swing",  "priority": "med"},
    "SHOP":  {"name": "Shopify",         "sector": "Tech/Consumer",  "type": "swing",  "priority": "med"},
    "NET":   {"name": "Cloudflare",      "sector": "Tech/Cyber",     "type": "swing",  "priority": "med"},
    # FINANCIAL
    "JPM":   {"name": "JPMorgan",        "sector": "Financials",     "type": "swing",  "priority": "med"},
    "BAC":   {"name": "Bank of America", "sector": "Financials",     "type": "swing",  "priority": "med"},
    # CYCLICAL
    "F":     {"name": "Ford",            "sector": "Auto/Cyclical",  "type": "swing",  "priority": "med"},
    # CONSUMER
    "DIS":   {"name": "Disney",          "sector": "Consumer/Media", "type": "swing",  "priority": "med"},
    "NKE":   {"name": "Nike",            "sector": "Consumer",       "type": "swing",  "priority": "med"},
    # ENERGY (war/oil news driver)
    "XOM":   {"name": "ExxonMobil",      "sector": "Energy",         "type": "swing",  "priority": "med"},
    "OXY":   {"name": "Occidental",      "sector": "Energy",         "type": "swing",  "priority": "med"},
    # HEALTHCARE
    "PFE":   {"name": "Pfizer",          "sector": "Healthcare",     "type": "swing",  "priority": "med"},
    # CRYPTO/VOL
    "COIN":  {"name": "Coinbase",        "sector": "Crypto/Fin",     "type": "swing",  "priority": "med"},
    # STABLE LONG-TERM
    "MSFT":  {"name": "Microsoft",       "sector": "Tech",           "type": "stable", "priority": "low"},
    "JNJ":   {"name": "Johnson&Johnson", "sector": "Healthcare",     "type": "stable", "priority": "low"},
    "BRK-B": {"name": "Berkshire B",     "sector": "Diversified",    "type": "stable", "priority": "low"},
}

# Correlated pairs - warn if user opens both at same time
CORRELATED_PAIRS = [
    ("NVDA", "AMD"),    # both AI chips
    ("AMD", "PLTR"),    # AI narrative
    ("JPM", "BAC"),     # both big banks
    ("XOM", "OXY"),     # both oil
]

# Macro events to flag
MACRO_KEYWORDS = {
    "fed":      ["fed", "fomc", "powell", "interest rate", "rate cut", "rate hike", "federal reserve"],
    "war":      ["war", "ukraine", "russia", "israel", "gaza", "iran", "missile", "invasion", "ceasefire"],
    "tariffs":  ["tariff", "trade war", "import duty", "sanctions", "export ban", "china trade"],
    "macro":    ["inflation", "cpi", "ppi", "unemployment", "gdp", "recession", "yield curve"],
}

# Sentiment keywords (fallback when AI unavailable)
POSITIVE_WORDS = {
    "beat", "beats", "surge", "soar", "rally", "rallies", "jump", "gain", "rises", "rose",
    "strong", "record", "growth", "profit", "upgrade", "upgrades", "positive", "outperform",
    "buy", "bullish", "exceed", "exceeds", "boost", "boosts", "approval", "win", "wins",
    "raised", "raises", "tops", "topped", "breakthrough", "expansion",
}

NEGATIVE_WORDS = {
    "miss", "misses", "fall", "falls", "fell", "drop", "drops", "plunge", "crash",
    "weak", "loss", "losses", "downgrade", "downgrades", "negative", "underperform",
    "sell", "bearish", "concern", "concerns", "worry", "fear", "fears", "warning",
    "lawsuit", "investigation", "probe", "fraud", "decline", "cuts", "cut", "layoff",
    "layoffs", "bankruptcy", "missed", "slump",
}


# ============================================================================
# AI BRAIN (GROQ)
# ============================================================================

def get_groq_client() -> Optional[Groq]:
    """Initialize Groq client from secrets or env. Returns None if not configured."""
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)  # 30 min cache
def ai_summarize_stock(ticker: str, headlines: list[str]) -> dict:
    """Send headlines to Groq, get back sentiment + summary.

    Returns dict with: sentiment (str), score (-1 to +1), summary (str), action (str)
    """
    client = get_groq_client()
    if not client or not headlines:
        return _fallback_sentiment(headlines)

    headlines_text = "\n".join(f"- {h}" for h in headlines[:10])
    prompt = f"""Analyze these recent news headlines about {ticker} ({WATCHLIST.get(ticker, {}).get('name', ticker)}).

Headlines:
{headlines_text}

Respond with EXACTLY this format (no extra text):
SENTIMENT: [POSITIVE/NEGATIVE/MIXED/NEUTRAL]
SCORE: [number from -1.0 to 1.0]
SUMMARY: [one sentence, max 25 words]
ACTION: [WATCH/INVESTIGATE/IGNORE]

Be honest. If headlines are routine/noise, say NEUTRAL and IGNORE. Only say INVESTIGATE if something material is happening."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        text = response.choices[0].message.content
        return _parse_ai_response(text, headlines)
    except Exception as e:
        return _fallback_sentiment(headlines, error=str(e))


def _parse_ai_response(text: str, headlines: list[str]) -> dict:
    """Parse the structured AI response."""
    result = {"sentiment": "NEUTRAL", "score": 0.0, "summary": "", "action": "IGNORE", "source": "ai"}
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("SENTIMENT:"):
            result["sentiment"] = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("SCORE:"):
            try:
                result["score"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                result["score"] = 0.0
        elif line.upper().startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ACTION:"):
            result["action"] = line.split(":", 1)[1].strip().upper()
    if not result["summary"] and headlines:
        result["summary"] = headlines[0][:100]
    return result


def _fallback_sentiment(headlines: list[str], error: str = "") -> dict:
    """Keyword-based sentiment when AI is unavailable."""
    if not headlines:
        return {"sentiment": "QUIET", "score": 0.0, "summary": "No recent news.",
                "action": "IGNORE", "source": "fallback"}
    pos = neg = 0
    for h in headlines:
        words = set(h.lower().split())
        pos += len(words & POSITIVE_WORDS)
        neg += len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        sent, score, action = "NEUTRAL", 0.0, "IGNORE"
    else:
        score = (pos - neg) / max(total, 1)
        if score > 0.3:
            sent, action = "POSITIVE", "WATCH"
        elif score < -0.3:
            sent, action = "NEGATIVE", "INVESTIGATE"
        else:
            sent, action = "MIXED", "WATCH"
    summary = (headlines[0] or "")[:100]
    return {"sentiment": sent, "score": score, "summary": summary,
            "action": action, "source": "fallback" + (f" ({error})" if error else "")}


@st.cache_data(ttl=1800, show_spinner=False)
def ai_macro_brief(macro_news: list[str]) -> str:
    """Generate a daily macro brief from big news."""
    client = get_groq_client()
    if not client or not macro_news:
        return ""
    news_text = "\n".join(f"- {n}" for n in macro_news[:8])
    prompt = f"""Summarize today's important macro/geopolitical news for a swing trader.

News:
{news_text}

In 3 sentences max:
1. What's the most important development?
2. Which sectors are affected (positive/negative)?
3. Should a swing trader be cautious today? Why or why not?

Be direct. No fluff."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)  # 5 min cache
def fetch_price_data(ticker: str) -> dict:
    """Get current price, day change, week change, basic indicators."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo", interval="1d")
        if hist.empty or len(hist) < 2:
            return {"error": "no data"}

        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2]
        day_pct = (last - prev) / prev * 100

        week_ago_idx = max(0, len(hist) - 6)
        week_ago = hist["Close"].iloc[week_ago_idx]
        week_pct = (last - week_ago) / week_ago * 100

        # Simple 50-day MA position
        ma50 = hist["Close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else last
        above_ma50 = last > ma50

        # RSI calculation (14-day)
        delta = hist["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = (100 - 100 / (1 + rs)).iloc[-1]

        # Volatility (ATR-ish)
        avg_range = ((hist["High"] - hist["Low"]) / hist["Close"]).rolling(20).mean().iloc[-1] * 100

        # 52-week high/low position
        year_hist = t.history(period="1y", interval="1d")
        hi_52 = year_hist["High"].max() if not year_hist.empty else last
        lo_52 = year_hist["Low"].min() if not year_hist.empty else last
        pct_from_hi = (last - hi_52) / hi_52 * 100

        return {
            "price": float(last),
            "day_pct": float(day_pct),
            "week_pct": float(week_pct),
            "rsi": float(rsi) if not pd.isna(rsi) else 50.0,
            "above_ma50": bool(above_ma50),
            "avg_daily_range": float(avg_range) if not pd.isna(avg_range) else 0.0,
            "hi_52w": float(hi_52),
            "lo_52w": float(lo_52),
            "pct_from_hi": float(pct_from_hi),
            "history": hist,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=900, show_spinner=False)  # 15 min cache
def fetch_news_yahoo(ticker: str) -> list[dict]:
    """Yahoo Finance news via yfinance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        items = []
        for n in news[:15]:
            content = n.get("content", n)  # newer yfinance nests under 'content'
            title = content.get("title") or n.get("title", "")
            pub_date = content.get("pubDate") or content.get("displayTime") or ""
            link = ""
            click_through = content.get("clickThroughUrl") or {}
            canonical = content.get("canonicalUrl") or {}
            link = click_through.get("url") or canonical.get("url") or n.get("link", "")
            publisher = content.get("provider", {}).get("displayName") or n.get("publisher", "")
            if title:
                items.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "time": pub_date,
                })
        return items
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_google(ticker: str, name: str) -> list[dict]:
    """Google News RSS as backup news source."""
    try:
        query = f"{name}+stock"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:10]:
            items.append({
                "title": entry.get("title", ""),
                "publisher": entry.get("source", {}).get("title", "Google News"),
                "link": entry.get("link", ""),
                "time": entry.get("published", ""),
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_news() -> list[dict]:
    """Pull macro/geopolitical news from Google News."""
    queries = [
        "federal+reserve+interest+rate",
        "ukraine+russia+war",
        "china+tariffs+trade",
        "inflation+cpi+economy",
    ]
    all_items = []
    for q in queries:
        try:
            url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                all_items.append({
                    "title": entry.get("title", ""),
                    "publisher": entry.get("source", {}).get("title", ""),
                    "link": entry.get("link", ""),
                    "category": q.split("+")[0],
                })
        except Exception:
            continue
    return all_items


@st.cache_data(ttl=3600, show_spinner=False)  # 1h cache
def fetch_market_overview() -> dict:
    """SPY, QQQ, VIX, DAX overview."""
    out = {}
    for sym, label in [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^VIX", "VIX"), ("^GDAXI", "DAX")]:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d", interval="1d")
            if not hist.empty and len(hist) >= 2:
                last = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                pct = (last - prev) / prev * 100
                out[label] = {"price": float(last), "pct": float(pct)}
        except Exception:
            continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_date(ticker: str) -> Optional[str]:
    """Get next earnings date if within 14 days."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                if isinstance(ed, list) and ed:
                    ed = ed[0]
                if hasattr(ed, "strftime"):
                    days_away = (ed - datetime.now().date()).days if hasattr(ed, "year") else 999
                    if 0 <= days_away <= 14:
                        return ed.strftime("%b %d") + f" ({days_away}d)"
        return None
    except Exception:
        return None


# ============================================================================
# UI HELPERS
# ============================================================================

def sentiment_color(sent: str) -> str:
    return {
        "POSITIVE": "#10b981",
        "NEGATIVE": "#ef4444",
        "MIXED": "#f59e0b",
        "NEUTRAL": "#6b7280",
        "QUIET": "#9ca3af",
    }.get(sent.upper(), "#6b7280")


def sentiment_icon(sent: str) -> str:
    return {
        "POSITIVE": "🟢",
        "NEGATIVE": "🔴",
        "MIXED": "🟡",
        "NEUTRAL": "⚪",
        "QUIET": "⚪",
    }.get(sent.upper(), "⚪")


def action_badge(action: str) -> str:
    return {
        "INVESTIGATE": "🔍 INVESTIGATE",
        "WATCH":       "👁  WATCH",
        "IGNORE":      "💤 IGNORE",
    }.get(action.upper(), action)


def detect_macro_in_headlines(headlines: list[str]) -> list[str]:
    """Detect which macro categories are showing up."""
    found = set()
    text = " ".join(headlines).lower()
    for category, keywords in MACRO_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.add(category)
    return sorted(found)


# ============================================================================
# UI: HEADER & STYLING
# ============================================================================

st.markdown("""
<style>
.main > div { padding-top: 1rem; }
.stock-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.metric-row { display: flex; justify-content: space-between; align-items: center; }
.ticker-name { font-size: 1.5rem; font-weight: 700; }
.ticker-sector { font-size: 0.8rem; color: #94a3b8; }
h1 { font-family: 'Georgia', serif; }
.macro-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #818cf8;
}
.warning-card {
    background: #422006;
    border: 1px solid #ca8a04;
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    color: #fde68a;
}
.calm-mode-empty { text-align: center; padding: 3rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Swing Trader Dashboard")
st.caption(f"News triage + AI sentiment · {datetime.now().strftime('%a %b %d, %Y · %H:%M')}")

# Top controls
col_a, col_b, col_c, col_d = st.columns([1.5, 1.5, 2, 2])
with col_a:
    calm_mode = st.toggle("🧘 Calm Mode", value=False, help="Hide noise. Show only items needing attention.")
with col_b:
    refresh = st.button("🔄 Refresh", use_container_width=True)
with col_c:
    open_positions = st.text_input("📌 Open positions (comma-separated tickers)", "",
                                   placeholder="e.g. NVDA, F").upper()
with col_d:
    ai_status = "✅ AI: Groq Llama-3.3" if get_groq_client() else "⚠️ AI: Off (using keywords)"
    st.markdown(f"<div style='padding-top:0.5rem;'>{ai_status}</div>", unsafe_allow_html=True)

if refresh:
    st.cache_data.clear()
    st.rerun()

open_pos_set = {t.strip() for t in open_positions.split(",") if t.strip()}

# ============================================================================
# SECTION 1: MARKET OVERVIEW
# ============================================================================

st.markdown("### 🌍 Market Overview")
overview = fetch_market_overview()
if overview:
    cols = st.columns(len(overview))
    for col, (label, data) in zip(cols, overview.items()):
        col.metric(label, f"{data['price']:.0f}", f"{data['pct']:+.2f}%")
else:
    st.info("Market data loading...")

# VIX warning
if "VIX" in overview and overview["VIX"]["price"] > 25:
    st.markdown(
        f"<div class='warning-card'>⚠️ <b>VIX is elevated at {overview['VIX']['price']:.1f}</b> "
        f"— market is in fear mode. Reduce position sizes today, widen stops.</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# SECTION 2: MACRO BRIEF
# ============================================================================

st.markdown("### 📰 Macro Brief")
macro_news = fetch_macro_news()
macro_headlines = [n["title"] for n in macro_news[:8]]
brief = ai_macro_brief(macro_headlines)

with st.container():
    if brief:
        st.markdown(f"<div class='macro-card'>{brief}</div>", unsafe_allow_html=True)
    elif macro_headlines:
        st.markdown("<div class='macro-card'>" +
                    "<br>".join(f"• {h}" for h in macro_headlines[:5]) +
                    "</div>", unsafe_allow_html=True)
    else:
        st.info("No macro news loaded.")

with st.expander("📃 See all macro headlines"):
    for n in macro_news:
        st.markdown(f"- **[{n.get('category','?').upper()}]** [{n['title']}]({n['link']})")

# ============================================================================
# SECTION 3: WATCHLIST ANALYSIS
# ============================================================================

st.markdown("### 📈 Watchlist")

# Determine display order: open positions first, then high priority swing, then others
def sort_key(t: str) -> tuple:
    in_pos = t in open_pos_set
    priority_rank = {"high": 0, "med": 1, "low": 2}.get(WATCHLIST[t]["priority"], 3)
    type_rank = 0 if WATCHLIST[t]["type"] == "swing" else 1
    return (not in_pos, type_rank, priority_rank, t)

ordered_tickers = sorted(WATCHLIST.keys(), key=sort_key)

# Build analyses
@st.cache_data(ttl=900, show_spinner=False)
def build_stock_card_data(ticker: str) -> dict:
    meta = WATCHLIST[ticker]
    price = fetch_price_data(ticker)

    yh_news = fetch_news_yahoo(ticker)
    if len(yh_news) < 3:
        gn_news = fetch_news_google(ticker, meta["name"])
        # de-dupe by title prefix
        seen = {n["title"][:50] for n in yh_news}
        for n in gn_news:
            if n["title"][:50] not in seen:
                yh_news.append(n)
                seen.add(n["title"][:50])

    headlines = [n["title"] for n in yh_news[:10]]
    ai = ai_summarize_stock(ticker, headlines)
    macros = detect_macro_in_headlines(headlines)
    earnings = fetch_earnings_date(ticker)

    return {
        "meta": meta,
        "price": price,
        "news": yh_news[:8],
        "ai": ai,
        "macros": macros,
        "earnings": earnings,
        "news_count": len(yh_news),
    }


# Pre-load with progress (so UI feels responsive)
with st.spinner("Analyzing watchlist..."):
    cards = {}
    progress = st.progress(0)
    for idx, ticker in enumerate(ordered_tickers):
        cards[ticker] = build_stock_card_data(ticker)
        progress.progress((idx + 1) / len(ordered_tickers))
    progress.empty()


# Filter for calm mode
def show_in_calm(card: dict, ticker: str) -> bool:
    """In calm mode, only show stocks with action != IGNORE, or with earnings soon, or open positions."""
    if ticker in open_pos_set:
        return True
    if card["ai"]["action"] in ("INVESTIGATE", "WATCH") and card["ai"]["sentiment"] != "NEUTRAL":
        return True
    if card["earnings"]:
        return True
    if card["news_count"] >= 8:  # high news velocity
        return True
    return False


visible_tickers = [t for t in ordered_tickers if (not calm_mode or show_in_calm(cards[t], t))]

if calm_mode and not visible_tickers:
    st.markdown(
        "<div class='calm-mode-empty'>"
        "<h3>🧘 Nothing urgent today</h3>"
        "<p>No watchlist stock has news that needs your attention right now.<br>"
        "<i>The best trade is often no trade.</i></p>"
        "</div>",
        unsafe_allow_html=True,
    )

# Correlation warning
if len(open_pos_set) >= 2:
    triggered = []
    for a, b in CORRELATED_PAIRS:
        if a in open_pos_set and b in open_pos_set:
            triggered.append((a, b))
    if triggered:
        for a, b in triggered:
            st.markdown(
                f"<div class='warning-card'>⚠️ <b>Correlated positions:</b> "
                f"{a} and {b} move together. You're doubling your bet — consider closing one.</div>",
                unsafe_allow_html=True,
            )

# Render stock cards
for ticker in visible_tickers:
    card = cards[ticker]
    meta = card["meta"]
    price = card["price"]
    ai = card["ai"]

    if price.get("error"):
        st.warning(f"{ticker}: {price['error']}")
        continue

    is_open = ticker in open_pos_set
    border_color = "#818cf8" if is_open else "#1e293b"

    with st.container(border=True):
        # HEADER ROW
        c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.5, 2])

        with c1:
            pin = "📌 " if is_open else ""
            st.markdown(f"**{pin}{ticker}** · {meta['name']}")
            st.caption(f"{meta['sector']} · {meta['type']}")

        with c2:
            st.metric("Price", f"${price['price']:.2f}", f"{price['day_pct']:+.2f}%",
                      label_visibility="collapsed")

        with c3:
            st.metric("Week", f"{price['week_pct']:+.1f}%", label_visibility="collapsed")
            st.caption(f"RSI: {price['rsi']:.0f}")

        with c4:
            st.markdown(f"{sentiment_icon(ai['sentiment'])} **{ai['sentiment']}**")
            st.caption(f"Score: {ai['score']:+.2f}")

        with c5:
            st.markdown(f"**{action_badge(ai['action'])}**")
            badges = []
            if card["earnings"]:
                badges.append(f"🚫 Earnings {card['earnings']}")
            if card["news_count"] >= 8:
                badges.append("⚡ High news volume")
            for m in card["macros"]:
                badges.append(f"🌐 {m}")
            if badges:
                st.caption(" · ".join(badges))

        # AI SUMMARY
        if ai["summary"]:
            st.markdown(f"💬 _{ai['summary']}_")

        # EXPANDED VIEW
        with st.expander(f"📃 News & details for {ticker}"):
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.markdown(f"**Headlines ({card['news_count']}):**")
                for n in card["news"]:
                    if n["link"]:
                        st.markdown(f"- [{n['title']}]({n['link']}) · _{n['publisher']}_")
                    else:
                        st.markdown(f"- {n['title']} · _{n['publisher']}_")
            with cc2:
                st.markdown("**Technicals:**")
                st.markdown(f"- 50d trend: {'📈 Above' if price['above_ma50'] else '📉 Below'}")
                st.markdown(f"- RSI: {price['rsi']:.0f} "
                            f"{'(oversold)' if price['rsi']<30 else '(overbought)' if price['rsi']>70 else ''}")
                st.markdown(f"- 52w high: ${price['hi_52w']:.2f} ({price['pct_from_hi']:+.1f}%)")
                st.markdown(f"- Avg daily range: {price['avg_daily_range']:.1f}%")
                st.markdown(f"- AI source: `{ai['source']}`")

                # Mini chart
                if "history" in price and not price["history"].empty:
                    fig = go.Figure(data=[go.Candlestick(
                        x=price["history"].index[-30:],
                        open=price["history"]["Open"].iloc[-30:],
                        high=price["history"]["High"].iloc[-30:],
                        low=price["history"]["Low"].iloc[-30:],
                        close=price["history"]["Close"].iloc[-30:],
                    )])
                    fig.update_layout(
                        height=200, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_rangeslider_visible=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8"),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}")

# ============================================================================
# SECTION 4: ASK THE AI
# ============================================================================

st.markdown("---")
st.markdown("### 🤖 Ask the AI")
st.caption("Ask anything about today's news, your watchlist, or a specific stock.")

user_q = st.text_input(
    "Your question",
    placeholder="e.g. Should I be worried about my NVDA position given today's news?",
    label_visibility="collapsed",
)

if user_q:
    client = get_groq_client()
    if not client:
        st.warning("AI is not configured. Set GROQ_API_KEY in Streamlit secrets.")
    else:
        # Build context from current cards
        context_lines = ["Today's watchlist analysis:"]
        for t, card in cards.items():
            ai = card["ai"]
            context_lines.append(
                f"- {t} ({card['meta']['name']}, {card['meta']['sector']}): "
                f"{ai['sentiment']} ({ai['score']:+.2f}). {ai['summary']}"
            )
        if open_pos_set:
            context_lines.append(f"\nUser holds open positions in: {', '.join(open_pos_set)}")
        if brief:
            context_lines.append(f"\nMacro brief: {brief}")
        context = "\n".join(context_lines)

        full_prompt = f"""{context}

User question: {user_q}

Answer in 4 sentences max. Be direct, honest, and specific. If you don't know, say so. Don't give generic financial advice — give specific reasoning based on the watchlist data above. End with one concrete suggestion if appropriate."""
        try:
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=300,
                    temperature=0.4,
                )
                st.markdown(f"**AI:** {response.choices[0].message.content.strip()}")
        except Exception as e:
            st.error(f"AI error: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(
    "⚠️ **Not financial advice.** This dashboard is a news triage tool. "
    "AI sentiment is not a buy signal. Always do your own research. "
    "Use stop losses. Risk only what you can afford to lose."
)
