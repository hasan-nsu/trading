"""
Swing Trader Dashboard v2
==========================
News + AI sentiment analysis for a curated watchlist of 20 stocks.
Designed for swing trading (days to weeks holding period).

KEY UPGRADES FROM v1:
- Per-headline AI verdict (each news item: GOOD/BAD/NEUTRAL + WHY for that stock)
- Real summaries with reasoning, not 3-word labels
- JSON output (more reliable parsing)
- Cross-stock impact: macro news -> which of YOUR stocks are affected
- Better error visibility
- Added AAPL + AMZN

NOT a buy/sell signal generator. A news triage tool.
"""

import os
import json
import re
from datetime import datetime
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

# Watchlist - 20 STOCKS (added AAPL + AMZN)
WATCHLIST = {
    "NVDA":  {"name": "NVIDIA",          "sector": "Tech/AI",        "type": "swing",  "priority": "high"},
    "AMD":   {"name": "AMD",             "sector": "Tech/AI",        "type": "swing",  "priority": "high"},
    "TSLA":  {"name": "Tesla",           "sector": "Auto/Tech",      "type": "swing",  "priority": "high"},
    "AAPL":  {"name": "Apple",           "sector": "Tech/Consumer",  "type": "swing",  "priority": "high"},
    "AMZN":  {"name": "Amazon",          "sector": "Tech/Retail",    "type": "swing",  "priority": "high"},
    "PLTR":  {"name": "Palantir",        "sector": "Tech/AI",        "type": "swing",  "priority": "med"},
    "SHOP":  {"name": "Shopify",         "sector": "Tech/Consumer",  "type": "swing",  "priority": "med"},
    "NET":   {"name": "Cloudflare",      "sector": "Tech/Cyber",     "type": "swing",  "priority": "med"},
    "JPM":   {"name": "JPMorgan",        "sector": "Financials",     "type": "swing",  "priority": "med"},
    "BAC":   {"name": "Bank of America", "sector": "Financials",     "type": "swing",  "priority": "med"},
    "F":     {"name": "Ford",            "sector": "Auto/Cyclical",  "type": "swing",  "priority": "med"},
    "DIS":   {"name": "Disney",          "sector": "Consumer/Media", "type": "swing",  "priority": "med"},
    "NKE":   {"name": "Nike",            "sector": "Consumer",       "type": "swing",  "priority": "med"},
    "XOM":   {"name": "ExxonMobil",      "sector": "Energy",         "type": "swing",  "priority": "med"},
    "OXY":   {"name": "Occidental",      "sector": "Energy",         "type": "swing",  "priority": "med"},
    "PFE":   {"name": "Pfizer",          "sector": "Healthcare",     "type": "swing",  "priority": "med"},
    "COIN":  {"name": "Coinbase",        "sector": "Crypto/Fin",     "type": "swing",  "priority": "med"},
    "MSFT":  {"name": "Microsoft",       "sector": "Tech",           "type": "stable", "priority": "low"},
    "JNJ":   {"name": "Johnson&Johnson", "sector": "Healthcare",     "type": "stable", "priority": "low"},
    "BRK-B": {"name": "Berkshire B",     "sector": "Diversified",    "type": "stable", "priority": "low"},
}

CORRELATED_PAIRS = [
    ("NVDA", "AMD"), ("AMD", "PLTR"), ("NVDA", "PLTR"),
    ("JPM", "BAC"), ("XOM", "OXY"),
    ("AAPL", "MSFT"), ("AMZN", "SHOP"),
]

MACRO_KEYWORDS = {
    "fed":      ["fed", "fomc", "powell", "interest rate", "rate cut", "rate hike", "federal reserve"],
    "war":      ["war", "ukraine", "russia", "israel", "gaza", "iran", "missile", "invasion", "ceasefire"],
    "tariffs":  ["tariff", "trade war", "import duty", "sanctions", "export ban", "china trade"],
    "macro":    ["inflation", "cpi", "ppi", "unemployment", "gdp", "recession", "yield curve"],
}

POSITIVE_WORDS = {
    "beat", "beats", "surge", "soar", "rally", "rallies", "jump", "gain", "rises", "rose",
    "strong", "record", "growth", "profit", "upgrade", "upgrades", "positive", "outperform",
    "buy", "bullish", "exceed", "exceeds", "boost", "boosts", "approval", "win", "wins",
    "raised", "raises", "tops", "topped", "breakthrough", "expansion", "explosive",
}

NEGATIVE_WORDS = {
    "miss", "misses", "fall", "falls", "fell", "drop", "drops", "plunge", "crash",
    "weak", "loss", "losses", "downgrade", "downgrades", "negative", "underperform",
    "sell", "bearish", "concern", "concerns", "worry", "fear", "fears", "warning",
    "lawsuit", "investigation", "probe", "fraud", "decline", "cuts", "cut", "layoff",
    "layoffs", "bankruptcy", "missed", "slump", "shortage", "overvalued",
}


# ============================================================================
# AI BRAIN - REWRITTEN FOR PER-HEADLINE ANALYSIS
# ============================================================================

def get_groq_client() -> Optional[Groq]:
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


def _extract_json(text: str) -> Optional[dict]:
    """Robust JSON extraction - handles markdown wrapping or preamble."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


@st.cache_data(ttl=900, show_spinner=False)
def ai_analyze_stock(ticker: str, headlines: list[str]) -> dict:
    """
    Per-headline impact analysis + overall verdict for a stock.

    Returns:
      sentiment, score, summary (with reasoning), action,
      headline_analysis (list of {headline, impact, why}),
      source, error
    """
    client = get_groq_client()
    if not client or not headlines:
        return _fallback_sentiment(headlines)

    company = WATCHLIST.get(ticker, {}).get("name", ticker)
    sector = WATCHLIST.get(ticker, {}).get("sector", "")
    headlines_numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:8]))

    prompt = f"""You are a STRICT, skeptical equity research analyst. Most financial headlines are NOISE, not SIGNAL. Your default rating is NEUTRAL. Only rate GOOD or BAD when there is a clear, concrete, material catalyst.

Analyze recent news for {ticker} ({company}, sector: {sector}).

Headlines:
{headlines_numbered}

Respond ONLY with valid JSON (no markdown, no preamble):
{{
  "headlines": [
    {{"n": 1, "impact": "GOOD"|"BAD"|"NEUTRAL", "why": "short specific reason, max 15 words"}}
  ],
  "news_briefing": "PLAIN-LANGUAGE PARAGRAPH of 4-6 sentences summarizing ALL the news as a story. Tell the user what is happening with this stock right now. Mention concrete events (earnings, products, deals, lawsuits, sector moves). Connect related headlines. Mention specific competitors or events by name when relevant. End with a 1-sentence bottom line: what does this collectively mean for the stock? Write naturally like you're briefing a colleague over coffee. NO bullet points. NO repeating headlines verbatim.",
  "overall_sentiment": "POSITIVE"|"NEGATIVE"|"MIXED"|"NEUTRAL",
  "overall_score": -1.0 to 1.0,
  "summary": "1 SHORT sentence with the bottom-line take, max 20 words.",
  "action": "WATCH"|"INVESTIGATE"|"IGNORE"
}}

STRICT RULES — these override everything else:

ALWAYS NEUTRAL (no exceptions):
- "Will X happen?", "Could X end?", "Is X over?", "What to expect" → speculation, NEUTRAL
- "Stock rose X%", "Stock fell X%" → after-the-fact reporting, no new info, NEUTRAL
- "Best stocks to buy", "Top picks", "X stocks to watch" → listicle clickbait, NEUTRAL
- "Ways to play", "Options strategy", "How to trade" → educational/strategy, NEUTRAL
- "Compared to peers", "X vs Y", "Which is better" → comparison filler, NEUTRAL
- Headlines from 7+ days ago → stale, NEUTRAL
- General market/sector commentary that doesn't name {ticker} specifically → NEUTRAL

NEVER spin BAD news as GOOD:
- "Tech stocks slide" / "Sector sells off" → BAD for {ticker} if it's in that sector. Do NOT call this a "buying opportunity."
- Cathie Wood / institutional selling → BAD, period. Do not soften.
- Insider selling → BAD.
- Industry-wide negative news → BAD if {ticker} is in that industry.

Real GOOD catalysts (use sparingly):
- Earnings BEAT with raised guidance
- Major product launch or successful rollout
- Big new contract / customer win (specific, named)
- Analyst UPGRADE from a major bank
- Buyback announcement, dividend raise
- Direct competitor stumbles in a way that benefits {ticker}
- Insider BUYING

Real BAD catalysts (use sparingly):
- Earnings MISS or cut guidance
- Product recall, safety issue, FDA rejection
- Lawsuit / fraud probe / regulatory action
- Analyst DOWNGRADE from a major bank
- Insider/institutional SELLING
- CEO/CFO departure under cloud
- Major customer/contract loss

If most headlines are speculation, listicles, or noise → overall_sentiment is NEUTRAL and action is IGNORE. That is the CORRECT answer most days. Do not invent signal.

Be honest. Be skeptical. Default to NEUTRAL."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise financial analyst. You output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        parsed = _extract_json(text)
        if not parsed:
            return _fallback_sentiment(headlines, error="JSON parse failed")

        per_headline = []
        for item in parsed.get("headlines", []):
            n = item.get("n", 0)
            if 1 <= n <= len(headlines):
                per_headline.append({
                    "headline": headlines[n-1],
                    "impact": str(item.get("impact", "NEUTRAL")).upper(),
                    "why": str(item.get("why", "")),
                })

        return {
            "sentiment": str(parsed.get("overall_sentiment", "NEUTRAL")).upper(),
            "score": float(parsed.get("overall_score", 0.0)),
            "summary": str(parsed.get("summary", "")),
            "news_briefing": str(parsed.get("news_briefing", "")),
            "action": str(parsed.get("action", "IGNORE")).upper(),
            "headline_analysis": per_headline,
            "source": "ai",
        }
    except Exception as e:
        return _fallback_sentiment(headlines, error=str(e))


def _fallback_sentiment(headlines: list[str], error: str = "") -> dict:
    """Keyword fallback - now also produces per-headline output."""
    if not headlines:
        return {
            "sentiment": "QUIET", "score": 0.0,
            "summary": "No recent news for this ticker.",
            "news_briefing": "",
            "action": "IGNORE", "headline_analysis": [],
            "source": f"fallback{(' (' + error + ')') if error else ''}",
        }

    per_headline = []
    pos_total = neg_total = 0
    for h in headlines[:8]:
        words = set(h.lower().split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        if pos > neg:
            impact, why = "GOOD", "Contains positive keywords"
        elif neg > pos:
            impact, why = "BAD", "Contains negative keywords"
        else:
            impact, why = "NEUTRAL", "No clear sentiment"
        per_headline.append({"headline": h, "impact": impact, "why": why})
        pos_total += pos
        neg_total += neg

    total = pos_total + neg_total
    if total == 0:
        sent, score, action = "NEUTRAL", 0.0, "IGNORE"
        summary = "No clear signal in headlines. Routine market noise."
    else:
        score = (pos_total - neg_total) / max(total, 1)
        if score > 0.3:
            sent, action = "POSITIVE", "WATCH"
            summary = "Headlines lean positive based on keyword analysis. AI unavailable for deeper analysis."
        elif score < -0.3:
            sent, action = "NEGATIVE", "INVESTIGATE"
            summary = "Headlines lean negative based on keyword analysis. AI unavailable for deeper analysis."
        else:
            sent, action = "MIXED", "WATCH"
            summary = "Mixed signals in headlines. AI unavailable for deeper analysis."

    # Build a simple briefing from headlines when AI is unavailable
    briefing = ("AI is currently unavailable, so this is a keyword-based view: " +
                " ".join(headlines[:5])[:600] +
                " — verify by reading the full headlines below.")

    return {
        "sentiment": sent, "score": score, "summary": summary,
        "news_briefing": briefing,
        "action": action, "headline_analysis": per_headline,
        "source": f"fallback{(' (' + error + ')') if error else ''}",
    }


@st.cache_data(ttl=1800, show_spinner=False)
def ai_macro_brief(macro_news: list[str], watchlist_tickers: list[str]) -> dict:
    """Macro brief + identifies which watchlist stocks are affected."""
    client = get_groq_client()
    if not client or not macro_news:
        return {"brief": "", "winners": [], "losers": [], "source": "none"}

    news_text = "\n".join(f"- {n}" for n in macro_news[:10])
    tickers_text = ", ".join(watchlist_tickers)

    prompt = f"""You are a macro analyst. Read today's macro/geopolitical news and identify impact on a watchlist.

Today's macro news:
{news_text}

Watchlist tickers: {tickers_text}

Respond ONLY with valid JSON:
{{
  "brief": "3 sentences max. (1) Most important macro development. (2) Which sectors helped/hurt. (3) Should a swing trader be cautious today and why?",
  "winners": ["TICKER1", "TICKER2"],
  "losers": ["TICKER3", "TICKER4"],
  "winner_reason": "1 sentence why these benefit",
  "loser_reason": "1 sentence why these suffer"
}}

Only include tickers from the watchlist. Be specific. If macro news has no clear stock impact, leave winners/losers empty."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        parsed = _extract_json(response.choices[0].message.content)
        if not parsed:
            return {"brief": "", "winners": [], "losers": [], "source": "parse_failed"}

        winners = [t.upper() for t in parsed.get("winners", []) if t.upper() in watchlist_tickers]
        losers = [t.upper() for t in parsed.get("losers", []) if t.upper() in watchlist_tickers]

        return {
            "brief": str(parsed.get("brief", "")),
            "winners": winners,
            "losers": losers,
            "winner_reason": str(parsed.get("winner_reason", "")),
            "loser_reason": str(parsed.get("loser_reason", "")),
            "source": "ai",
        }
    except Exception as e:
        return {"brief": "", "winners": [], "losers": [], "source": f"error: {e}"}


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(ticker: str) -> dict:
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
        ma50 = hist["Close"].rolling(50).mean().iloc[-1] if len(hist) >= 50 else last
        delta = hist["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = (100 - 100 / (1 + rs)).iloc[-1]
        avg_range = ((hist["High"] - hist["Low"]) / hist["Close"]).rolling(20).mean().iloc[-1] * 100
        year_hist = t.history(period="1y", interval="1d")
        hi_52 = year_hist["High"].max() if not year_hist.empty else last
        lo_52 = year_hist["Low"].min() if not year_hist.empty else last
        pct_from_hi = (last - hi_52) / hi_52 * 100
        return {
            "price": float(last), "day_pct": float(day_pct), "week_pct": float(week_pct),
            "rsi": float(rsi) if not pd.isna(rsi) else 50.0,
            "above_ma50": bool(last > ma50),
            "avg_daily_range": float(avg_range) if not pd.isna(avg_range) else 0.0,
            "hi_52w": float(hi_52), "lo_52w": float(lo_52),
            "pct_from_hi": float(pct_from_hi),
            "history": hist, "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_yahoo(ticker: str) -> list[dict]:
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        items = []
        for n in news[:15]:
            content = n.get("content", n)
            title = content.get("title") or n.get("title", "")
            pub_date = content.get("pubDate") or content.get("displayTime") or ""
            click_through = content.get("clickThroughUrl") or {}
            canonical = content.get("canonicalUrl") or {}
            link = click_through.get("url") or canonical.get("url") or n.get("link", "")
            publisher = content.get("provider", {}).get("displayName") or n.get("publisher", "")
            if title:
                items.append({"title": title, "publisher": publisher, "link": link, "time": pub_date})
        return items
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_google(ticker: str, name: str) -> list[dict]:
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_overview() -> dict:
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

def sentiment_icon(sent: str) -> str:
    return {"POSITIVE": "🟢", "NEGATIVE": "🔴", "MIXED": "🟡",
            "NEUTRAL": "⚪", "QUIET": "⚪"}.get(sent.upper(), "⚪")


def impact_badge(impact: str) -> str:
    """Per-headline impact label - the new feature."""
    if impact == "GOOD":
        return "<span style='background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.8em'>✓ GOOD</span>"
    elif impact == "BAD":
        return "<span style='background:#fee2e2;color:#b91c1c;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.8em'>✗ BAD</span>"
    else:
        return "<span style='background:#f1f5f9;color:#475569;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.8em'>— NEUTRAL</span>"


def action_badge(action: str) -> str:
    return {"INVESTIGATE": "🔍 INVESTIGATE", "WATCH": "👁  WATCH",
            "IGNORE": "💤 IGNORE"}.get(action.upper(), action)


def detect_macro_in_headlines(headlines: list[str]) -> list[str]:
    found = set()
    text = " ".join(headlines).lower()
    for category, keywords in MACRO_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.add(category)
    return sorted(found)


# ============================================================================
# UI: STYLING
# ============================================================================

st.markdown("""
<style>
.main > div { padding-top: 1rem; }
h1 { font-family: 'Georgia', serif; }
.macro-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 0.5rem;
    border-left: 4px solid #818cf8; color: #e0e7ff;
}
.warning-card {
    background: #422006; border: 1px solid #ca8a04; border-radius: 8px;
    padding: 0.75rem; margin: 0.5rem 0; color: #fde68a;
}
.calm-mode-empty { text-align: center; padding: 3rem; color: #94a3b8; }
.headline-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid #f1f5f9;
}
.headline-row:last-child { border-bottom: none; }
.impact-cell { flex-shrink: 0; min-width: 90px; padding-top: 2px; }
.headline-cell { flex: 1; }
.headline-title { font-weight: 500; line-height: 1.4; }
.headline-why {
    font-size: 0.85em; color: #64748b;
    font-style: italic; margin-top: 2px;
}
.macro-affected {
    background: #f0fdf4; border-radius: 6px; padding: 8px 12px;
    margin: 4px 0; font-size: 0.9em; color: #166534;
}
.macro-affected.bad { background: #fef2f2; color: #991b1b; }
.news-briefing {
    background: #f8fafc;
    border-left: 3px solid #0891b2;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 8px 0 6px 0;
}
.briefing-label {
    font-size: 0.75em;
    font-weight: 700;
    color: #0891b2;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.briefing-body {
    font-size: 0.92em;
    line-height: 1.55;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Swing Trader Dashboard")
st.caption(f"News triage + AI sentiment · {datetime.now().strftime('%a %b %d, %Y · %H:%M')}")

# Top controls
col_a, col_b, col_c, col_d = st.columns([1.5, 1.5, 2, 2])
with col_a:
    calm_mode = st.toggle("🧘 Calm Mode", value=False)
with col_b:
    refresh = st.button("🔄 Refresh", use_container_width=True)
with col_c:
    open_positions = st.text_input("📌 Open positions", "", placeholder="e.g. NVDA, F").upper()
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

if "VIX" in overview and overview["VIX"]["price"] > 25:
    st.markdown(
        f"<div class='warning-card'>⚠️ <b>VIX is elevated at {overview['VIX']['price']:.1f}</b> "
        f"— market is in fear mode. Reduce position sizes today, widen stops.</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# SECTION 2: MACRO BRIEF + WATCHLIST IMPACT
# ============================================================================

st.markdown("### 📰 Macro Brief")
macro_news = fetch_macro_news()
macro_headlines = [n["title"] for n in macro_news[:10]]
macro_result = ai_macro_brief(macro_headlines, list(WATCHLIST.keys()))

if macro_result.get("brief"):
    st.markdown(f"<div class='macro-card'>{macro_result['brief']}</div>", unsafe_allow_html=True)
    if macro_result.get("winners"):
        st.markdown(
            f"<div class='macro-affected'>🟢 <b>Likely winners on your watchlist:</b> "
            f"{', '.join(macro_result['winners'])} — <i>{macro_result.get('winner_reason','')}</i></div>",
            unsafe_allow_html=True
        )
    if macro_result.get("losers"):
        st.markdown(
            f"<div class='macro-affected bad'>🔴 <b>Likely losers on your watchlist:</b> "
            f"{', '.join(macro_result['losers'])} — <i>{macro_result.get('loser_reason','')}</i></div>",
            unsafe_allow_html=True
        )
elif macro_headlines:
    st.markdown("<div class='macro-card'>" +
                "<br>".join(f"• {h}" for h in macro_headlines[:5]) + "</div>",
                unsafe_allow_html=True)

with st.expander("📃 See all macro headlines"):
    for n in macro_news:
        st.markdown(f"- **[{n.get('category','?').upper()}]** [{n['title']}]({n['link']})")

# ============================================================================
# SECTION 3: WATCHLIST
# ============================================================================

st.markdown("### 📈 Watchlist")
st.caption("Each headline tagged GOOD / BAD / NEUTRAL with reasoning. Click any stock to expand.")

def sort_key(t: str) -> tuple:
    in_pos = t in open_pos_set
    is_winner = t in (macro_result.get("winners") or [])
    is_loser = t in (macro_result.get("losers") or [])
    macro_relevant = is_winner or is_loser
    priority_rank = {"high": 0, "med": 1, "low": 2}.get(WATCHLIST[t]["priority"], 3)
    type_rank = 0 if WATCHLIST[t]["type"] == "swing" else 1
    return (not in_pos, not macro_relevant, type_rank, priority_rank, t)

ordered_tickers = sorted(WATCHLIST.keys(), key=sort_key)


@st.cache_data(ttl=900, show_spinner=False)
def build_stock_card_data(ticker: str) -> dict:
    meta = WATCHLIST[ticker]
    price = fetch_price_data(ticker)
    yh_news = fetch_news_yahoo(ticker)
    if len(yh_news) < 3:
        gn_news = fetch_news_google(ticker, meta["name"])
        seen = {n["title"][:50] for n in yh_news}
        for n in gn_news:
            if n["title"][:50] not in seen:
                yh_news.append(n)
                seen.add(n["title"][:50])
    headlines = [n["title"] for n in yh_news[:8]]
    ai = ai_analyze_stock(ticker, headlines)
    macros = detect_macro_in_headlines(headlines)
    earnings = fetch_earnings_date(ticker)
    return {
        "meta": meta, "price": price, "news": yh_news[:8],
        "ai": ai, "macros": macros, "earnings": earnings,
        "news_count": len(yh_news),
    }


with st.spinner("Analyzing watchlist (this takes ~45s on first load — AI is reading every headline)..."):
    cards = {}
    progress = st.progress(0)
    for idx, ticker in enumerate(ordered_tickers):
        cards[ticker] = build_stock_card_data(ticker)
        progress.progress((idx + 1) / len(ordered_tickers))
    progress.empty()


def show_in_calm(card: dict, ticker: str) -> bool:
    if ticker in open_pos_set:
        return True
    if card["ai"]["action"] in ("INVESTIGATE", "WATCH") and card["ai"]["sentiment"] != "NEUTRAL":
        return True
    if card["earnings"]:
        return True
    if card["news_count"] >= 8:
        return True
    if ticker in (macro_result.get("winners") or []) or ticker in (macro_result.get("losers") or []):
        return True
    return False


visible_tickers = [t for t in ordered_tickers if (not calm_mode or show_in_calm(cards[t], t))]

if calm_mode and not visible_tickers:
    st.markdown(
        "<div class='calm-mode-empty'>"
        "<h3>🧘 Nothing urgent today</h3>"
        "<p>No watchlist stock has news that needs your attention.<br>"
        "<i>The best trade is often no trade.</i></p></div>",
        unsafe_allow_html=True,
    )

# Correlation warnings
if len(open_pos_set) >= 2:
    triggered = [(a, b) for a, b in CORRELATED_PAIRS if a in open_pos_set and b in open_pos_set]
    for a, b in triggered:
        st.markdown(
            f"<div class='warning-card'>⚠️ <b>Correlated positions:</b> "
            f"{a} and {b} move together. You're doubling your bet — consider closing one.</div>",
            unsafe_allow_html=True,
        )

# Render cards
for ticker in visible_tickers:
    card = cards[ticker]
    meta = card["meta"]
    price = card["price"]
    ai = card["ai"]

    if price.get("error"):
        st.warning(f"{ticker}: {price['error']}")
        continue

    is_open = ticker in open_pos_set
    is_macro_winner = ticker in (macro_result.get("winners") or [])
    is_macro_loser = ticker in (macro_result.get("losers") or [])

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.5, 2])

        with c1:
            pin = "📌 " if is_open else ""
            macro_tag = ""
            if is_macro_winner:
                macro_tag = " <span style='background:#dcfce7;color:#15803d;padding:1px 6px;border-radius:4px;font-size:0.75em'>MACRO WIN</span>"
            elif is_macro_loser:
                macro_tag = " <span style='background:#fee2e2;color:#b91c1c;padding:1px 6px;border-radius:4px;font-size:0.75em'>MACRO LOSS</span>"
            st.markdown(f"**{pin}{ticker}** · {meta['name']}{macro_tag}", unsafe_allow_html=True)
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
                badges.append("⚡ High news vol")
            for m in card["macros"]:
                badges.append(f"🌐 {m}")
            if badges:
                st.caption(" · ".join(badges))

        # Quick bottom-line summary (1 sentence)
        if ai.get("summary"):
            st.markdown(f"💬 _{ai['summary']}_")

        # NEWS BRIEFING - the new prose summary of all the news for this stock
        if ai.get("news_briefing"):
            st.markdown(
                f"<div class='news-briefing'>"
                f"<div class='briefing-label'>📋 News briefing</div>"
                f"<div class='briefing-body'>{ai['news_briefing']}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # PER-HEADLINE BREAKDOWN - the new feature you asked for
        with st.expander(f"📃 Headlines with AI verdict for {ticker} ({card['news_count']} items)"):
            cc1, cc2 = st.columns([3, 1.2])
            with cc1:
                if ai.get("headline_analysis"):
                    url_map = {n["title"]: n.get("link", "") for n in card["news"]}
                    pub_map = {n["title"]: n.get("publisher", "") for n in card["news"]}
                    for item in ai["headline_analysis"]:
                        h = item["headline"]
                        url = url_map.get(h, "")
                        pub = pub_map.get(h, "")
                        title_html = f"<a href='{url}' target='_blank'>{h}</a>" if url else h
                        why = item.get("why", "")
                        st.markdown(
                            f"<div class='headline-row'>"
                            f"<div class='impact-cell'>{impact_badge(item['impact'])}</div>"
                            f"<div class='headline-cell'>"
                            f"<div class='headline-title'>{title_html}</div>"
                            f"<div class='headline-why'>→ {why} <span style='color:#94a3b8'>· {pub}</span></div>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )
                else:
                    for n in card["news"]:
                        if n["link"]:
                            st.markdown(f"- [{n['title']}]({n['link']}) · _{n['publisher']}_")
                        else:
                            st.markdown(f"- {n['title']} · _{n['publisher']}_")

            with cc2:
                st.markdown("**Technicals:**")
                st.markdown(f"- 50d trend: {'📈 Above' if price['above_ma50'] else '📉 Below'}")
                rsi_label = ('(oversold)' if price['rsi'] < 30
                             else '(overbought)' if price['rsi'] > 70 else '')
                st.markdown(f"- RSI: {price['rsi']:.0f} {rsi_label}")
                st.markdown(f"- 52w high: ${price['hi_52w']:.2f} ({price['pct_from_hi']:+.1f}%)")
                st.markdown(f"- Avg daily range: {price['avg_daily_range']:.1f}%")
                st.caption(f"AI source: `{ai['source']}`")

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
st.caption("Ask about today's news, your watchlist, or any specific stock.")

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
        context_lines = ["Today's watchlist analysis:"]
        for t, card in cards.items():
            ai = card["ai"]
            context_lines.append(
                f"- {t} ({card['meta']['name']}, {card['meta']['sector']}): "
                f"{ai['sentiment']} ({ai['score']:+.2f}). {ai['summary']}"
            )
        if open_pos_set:
            context_lines.append(f"\nUser holds open positions in: {', '.join(open_pos_set)}")
        if macro_result.get("brief"):
            context_lines.append(f"\nMacro brief: {macro_result['brief']}")
        context = "\n".join(context_lines)

        full_prompt = f"""{context}

User question: {user_q}

Answer in 4 sentences max. Be direct, honest, specific. Reference the watchlist data above. End with one concrete suggestion if relevant. No generic financial advice fluff."""
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
