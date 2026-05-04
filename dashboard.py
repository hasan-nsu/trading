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
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser
import plotly.graph_objects as go
from groq import Groq

# Optional dependencies — graceful fallback if not installed
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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

Headlines (each prefixed with age — RECENT news (last 6h) matters most, OLD news (3+ days) is mostly stale):
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
- Headlines older than 3 days [3d ago+] → mostly stale, lean NEUTRAL unless story still developing
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


@st.cache_data(ttl=900, show_spinner=False)  # 15 min cache
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


def _parse_news_timestamp(time_str: str) -> Optional[datetime]:
    """Parse various timestamp formats into datetime. Returns None if unparseable."""
    if not time_str:
        return None
    # Try ISO format (Yahoo new format: "2026-04-27T14:30:00Z")
    try:
        # Handle Z suffix
        cleaned = time_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        pass
    # Try Unix timestamp (Yahoo old format)
    try:
        ts = int(time_str)
        return datetime.fromtimestamp(ts)
    except (ValueError, TypeError):
        pass
    # Try RFC822 (Google News RSS format: "Mon, 27 Apr 2026 14:30:00 GMT")
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(time_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        pass
    return None


def _format_age(pub_dt: Optional[datetime]) -> str:
    """Format datetime as relative age: '2h ago', '3d ago'."""
    if pub_dt is None:
        return "?"
    now = datetime.now()
    delta = now - pub_dt
    seconds = delta.total_seconds()
    if seconds < 0:
        return "just now"  # future timestamp = clock skew, treat as fresh
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 604800:
        return f"{int(seconds // 86400)}d ago"
    return pub_dt.strftime("%b %d")


def _relevance_score(title: str, ticker: str, company: str) -> str:
    """
    Tag headline relevance.
    HIGH: mentions ticker or company name directly
    MED:  about the sector/competitors
    LOW:  generic market commentary, listicles
    """
    title_lower = title.lower()
    company_short = company.lower().split()[0]  # "NVIDIA" from "NVIDIA Corp"

    # HIGH: explicit mention
    if ticker.lower() in title_lower or company.lower() in title_lower or company_short in title_lower:
        return "HIGH"

    # LOW: clear listicle/generic patterns
    low_patterns = [
        "best stocks", "top stocks", "stocks to buy", "stocks to watch",
        "stocks to avoid", "things to know", "what to know",
        "market wrap", "market roundup", "stocks moving", "movers and shakers",
        "biggest gainers", "biggest losers", "premarket", "after hours",
        "analyst picks", "wall street", "today's top",
    ]
    if any(p in title_lower for p in low_patterns):
        return "LOW"

    # MEDIUM: probably sector/competitor relevant
    return "MEDIUM"


# ════════════════════════════════════════════════════════════════════════════
# ARTICLE CONTENT EXTRACTION (full-text fetching)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)  # cache for 24 hours
def extract_article_text(url: str, max_chars: int = 2000) -> str:
    """
    Fetch and extract clean article text from a URL using trafilatura.
    Returns truncated plain text (max_chars) suitable for AI summarization.
    Returns empty string if unavailable or extraction fails.
    """
    if not TRAFILATURA_AVAILABLE or not url:
        return ""

    try:
        # trafilatura.fetch_url has built-in retry, timeout, and proper UA
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return ""
        # Extract main article text only (no nav, ads, comments)
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        if not text:
            return ""
        # Clean and truncate
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(' ', 1)[0] + "..."
        return text
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════════════
# MARKETAUX API (financial news with built-in sentiment)
# ════════════════════════════════════════════════════════════════════════════

def get_marketaux_key() -> Optional[str]:
    """Return Marketaux API key from secrets or env, if configured."""
    key = None
    try:
        key = st.secrets.get("MARKETAUX_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.environ.get("MARKETAUX_API_KEY")
    return key.strip() if key and key.strip() else None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_marketaux(ticker: str) -> list[dict]:
    """
    Fetch news from Marketaux API. Returns up to 10 items with sentiment scores.
    Free tier: 100 requests/day. Returns empty list if no API key or if errored.
    """
    key = get_marketaux_key()
    if not key or not REQUESTS_AVAILABLE:
        return []

    try:
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "symbols": ticker,
            "filter_entities": "true",
            "language": "en",
            "limit": 10,
            "api_token": key,
            "published_after": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M"),
        }
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return []
        data = r.json()
        items = []
        for art in data.get("data", []):
            pub_dt = _parse_news_timestamp(art.get("published_at", ""))
            # Marketaux sentiment per-entity
            entity_sent = None
            for ent in art.get("entities", []):
                if ent.get("symbol") == ticker:
                    entity_sent = ent.get("sentiment_score")
                    break
            items.append({
                "title": art.get("title", ""),
                "publisher": art.get("source", "Marketaux"),
                "link": art.get("url", ""),
                "pub_dt": pub_dt,
                "age_str": _format_age(pub_dt),
                "relevance": "HIGH",  # Marketaux already filters by ticker
                "snippet": art.get("description", "")[:500],
                "marketaux_sentiment": entity_sent,  # -1.0 to +1.0 if available
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_macro_marketaux() -> list[dict]:
    """Fetch macro/market news from Marketaux trending endpoint."""
    key = get_marketaux_key()
    if not key or not REQUESTS_AVAILABLE:
        return []

    try:
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "language": "en",
            "limit": 10,
            "filter_entities": "false",
            "api_token": key,
            "search": "fed OR inflation OR tariff OR economy OR rate",
            "published_after": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M"),
        }
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return []
        data = r.json()
        items = []
        for art in data.get("data", []):
            pub_dt = _parse_news_timestamp(art.get("published_at", ""))
            title = art.get("title", "")
            tl = title.lower()
            if any(k in tl for k in ["fed", "powell", "fomc", "rate"]):
                cat = "fed"
            elif any(k in tl for k in ["ukraine", "russia", "israel", "war", "iran"]):
                cat = "war"
            elif any(k in tl for k in ["tariff", "china", "trade war"]):
                cat = "tariffs"
            elif any(k in tl for k in ["inflation", "cpi", "jobs", "gdp", "recession"]):
                cat = "macro"
            else:
                cat = "market"
            items.append({
                "title": title,
                "publisher": art.get("source", "Marketaux"),
                "link": art.get("url", ""),
                "category": cat,
                "pub_dt": pub_dt,
                "age_str": _format_age(pub_dt),
                "source": "marketaux",
                "snippet": art.get("description", "")[:500],
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def get_article_summary_for_ai(item: dict, ticker: str) -> str:
    """
    Returns the best available text for AI to analyze for a given news item.
    Priority: Marketaux snippet (already there) > extracted article text > title only.
    """
    # If Marketaux already gave us a snippet, use it (no need to fetch)
    if item.get("snippet"):
        return item["snippet"]
    # Otherwise try to extract full article text
    url = item.get("link", "")
    if url and TRAFILATURA_AVAILABLE:
        text = extract_article_text(url, max_chars=1500)
        if text:
            return text
    # Fallback: just the title
    return item.get("title", "")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_yahoo(ticker: str) -> list[dict]:
    """Yahoo news with timestamp parsing + relevance tagging."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        company = WATCHLIST.get(ticker, {}).get("name", ticker)
        items = []
        for n in news[:20]:
            content = n.get("content", n)
            title = content.get("title") or n.get("title", "")
            pub_raw = (content.get("pubDate") or content.get("displayTime") or
                       n.get("providerPublishTime") or "")
            click_through = content.get("clickThroughUrl") or {}
            canonical = content.get("canonicalUrl") or {}
            link = click_through.get("url") or canonical.get("url") or n.get("link", "")
            publisher = content.get("provider", {}).get("displayName") or n.get("publisher", "")
            if not title:
                continue
            pub_dt = _parse_news_timestamp(str(pub_raw))
            items.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "pub_dt": pub_dt,
                "age_str": _format_age(pub_dt),
                "relevance": _relevance_score(title, ticker, company),
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_google(ticker: str, name: str) -> list[dict]:
    """Google News RSS — using ticker symbol for higher relevance."""
    try:
        # Search for ticker symbol first (more specific than company name)
        query = f"%22{ticker}%22+stock"  # %22 = quotes for exact match
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:15]:
            title = entry.get("title", "")
            if not title:
                continue
            pub_raw = entry.get("published", "")
            pub_dt = _parse_news_timestamp(pub_raw)
            items.append({
                "title": title,
                "publisher": entry.get("source", {}).get("title", "Google News"),
                "link": entry.get("link", ""),
                "pub_dt": pub_dt,
                "age_str": _format_age(pub_dt),
                "relevance": _relevance_score(title, ticker, name),
            })
        return items
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)  # 15 min cache (was 30)
def fetch_macro_news() -> list[dict]:
    """
    Pull macro/geopolitical news from multiple sources.
    - Google News with 'when:1d' filter (last 24 hours only)
    - Reuters business RSS
    - MarketWatch Top Stories RSS
    - CNBC top news RSS
    All items get timestamped, sorted newest-first, filtered to last 48h.
    """
    all_items = []
    now = datetime.now()

    # ━━━ Google News searches with when:1d (last 24h) ━━━
    google_queries = [
        ("fed",     "federal+reserve+interest+rate+when:1d"),
        ("war",     "ukraine+russia+israel+war+when:1d"),
        ("tariffs", "china+tariffs+trade+sanctions+when:1d"),
        ("macro",   "inflation+cpi+jobs+economy+when:1d"),
        ("market",  "stock+market+sp500+nasdaq+when:1d"),
    ]
    for category, q in google_queries:
        try:
            url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                pub_raw = entry.get("published", "")
                pub_dt = _parse_news_timestamp(pub_raw)
                # Skip if older than 48h or unparseable+from old source
                if pub_dt and (now - pub_dt).total_seconds() > 172800:  # 48h
                    continue
                all_items.append({
                    "title": entry.get("title", ""),
                    "publisher": entry.get("source", {}).get("title", "Google News"),
                    "link": entry.get("link", ""),
                    "category": category,
                    "pub_dt": pub_dt,
                    "age_str": _format_age(pub_dt),
                    "source": "google",
                })
        except Exception:
            continue

    # ━━━ Yahoo Finance Top Stories RSS (reliable, official) ━━━
    try:
        feed = feedparser.parse("https://finance.yahoo.com/news/rssindex")
        for entry in feed.entries[:10]:
            pub_dt = _parse_news_timestamp(entry.get("published", ""))
            if pub_dt and (now - pub_dt).total_seconds() > 172800:
                continue
            title = entry.get("title", "")
            tl = title.lower()
            if any(k in tl for k in ["fed", "powell", "fomc", "rate"]):
                cat = "fed"
            elif any(k in tl for k in ["ukraine", "russia", "israel", "war", "iran"]):
                cat = "war"
            elif any(k in tl for k in ["tariff", "china", "trade war"]):
                cat = "tariffs"
            elif any(k in tl for k in ["inflation", "cpi", "jobs", "gdp", "recession"]):
                cat = "macro"
            else:
                cat = "market"
            all_items.append({
                "title": title,
                "publisher": "Yahoo Finance",
                "link": entry.get("link", ""),
                "category": cat,
                "pub_dt": pub_dt,
                "age_str": _format_age(pub_dt),
                "source": "yahoo",
            })
    except Exception:
        pass

    # ━━━ MarketWatch Top Stories ━━━
    try:
        feed = feedparser.parse("https://feeds.content.dowjones.io/public/rss/mw_topstories")
        for entry in feed.entries[:8]:
            pub_dt = _parse_news_timestamp(entry.get("published", ""))
            if pub_dt and (now - pub_dt).total_seconds() > 172800:
                continue
            title = entry.get("title", "")
            tl = title.lower()
            if any(k in tl for k in ["fed", "powell", "rate"]):
                cat = "fed"
            elif any(k in tl for k in ["ukraine", "russia", "israel", "war"]):
                cat = "war"
            elif any(k in tl for k in ["tariff", "china"]):
                cat = "tariffs"
            elif any(k in tl for k in ["inflation", "cpi", "jobs"]):
                cat = "macro"
            else:
                cat = "market"
            all_items.append({
                "title": title,
                "publisher": "MarketWatch",
                "link": entry.get("link", ""),
                "category": cat,
                "pub_dt": pub_dt,
                "age_str": _format_age(pub_dt),
                "source": "marketwatch",
            })
    except Exception:
        pass

    # ━━━ CNBC Top News ━━━
    try:
        feed = feedparser.parse("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114")
        for entry in feed.entries[:6]:
            pub_dt = _parse_news_timestamp(entry.get("published", ""))
            if pub_dt and (now - pub_dt).total_seconds() > 172800:
                continue
            title = entry.get("title", "")
            tl = title.lower()
            if any(k in tl for k in ["fed", "powell", "rate"]):
                cat = "fed"
            elif any(k in tl for k in ["war", "ukraine", "israel"]):
                cat = "war"
            elif any(k in tl for k in ["tariff", "china"]):
                cat = "tariffs"
            elif any(k in tl for k in ["inflation", "cpi", "jobs"]):
                cat = "macro"
            else:
                cat = "market"
            all_items.append({
                "title": title,
                "publisher": "CNBC",
                "link": entry.get("link", ""),
                "category": cat,
                "pub_dt": pub_dt,
                "age_str": _format_age(pub_dt),
                "source": "cnbc",
            })
    except Exception:
        pass

    # ━━━ Marketaux Macro News (if API key configured) ━━━
    try:
        mx_macro = fetch_macro_marketaux()
        all_items.extend(mx_macro)
    except Exception:
        pass

    # Deduplicate by title (case-insensitive first 60 chars)
    seen = set()
    deduped = []
    for item in all_items:
        key = item["title"][:60].lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    # Sort newest first (items with no timestamp go to bottom)
    deduped.sort(key=lambda n: -(n["pub_dt"].timestamp() if n["pub_dt"] else 0))

    return deduped[:25]  # cap at 25 macro headlines


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


def sentiment_bar_html(score: float) -> str:
    """Visual sentiment score bar from -1 to +1."""
    # Clamp
    score = max(-1.0, min(1.0, score))
    # Position of marker in % (0% = far left = -1, 100% = far right = +1)
    pos_pct = (score + 1) / 2 * 100

    # Color based on score
    if score > 0.3:
        color = "#10b981"
        label = "BULLISH"
    elif score < -0.3:
        color = "#ef4444"
        label = "BEARISH"
    else:
        color = "#f59e0b"
        label = "NEUTRAL"

    return f"""
    <div class='sentiment-bar-wrap'>
      <div class='sentiment-bar-track'>
        <div class='sentiment-bar-zero'></div>
        <div class='sentiment-bar-marker' style='left: {pos_pct}%; background: {color};'></div>
      </div>
      <div class='sentiment-bar-labels'>
        <span style='color:#ef4444'>−1</span>
        <span style='color:{color}; font-weight:700'>{score:+.2f} · {label}</span>
        <span style='color:#10b981'>+1</span>
      </div>
    </div>
    """


def dip_buy_score(ticker: str, price: dict, ai: dict, card: dict) -> dict:
    """
    Score a stock as a dip-buy candidate based on Shakil's strategy:
    - Buying small dips on big stable companies
    - 5-7% recent drop = good dip
    - RSI 25-40 = oversold sweet spot
    - Not at 52-week lows (no falling knives)
    - News sentiment NEGATIVE/MIXED but not catastrophic
    - No earnings within 14 days (gap risk)

    Returns: {score: 0-100, status: str, color: str, reasons: list[str]}
    """
    if price.get("error"):
        return {"score": 0, "status": "—", "color": "#94a3b8", "reasons": ["No price data"]}

    score = 0
    reasons_pos = []
    reasons_neg = []

    rsi = price.get("rsi", 50)
    week_pct = price.get("week_pct", 0)
    day_pct = price.get("day_pct", 0)
    pct_from_hi = price.get("pct_from_hi", 0)
    above_ma50 = price.get("above_ma50", False)
    sentiment = ai.get("sentiment", "NEUTRAL")
    earnings = card.get("earnings")

    # ━━ POSITIVE SIGNALS ━━

    # RSI in dip-buy zone (25-40 = best, 40-50 = ok)
    if 25 <= rsi <= 40:
        score += 30
        reasons_pos.append(f"RSI {rsi:.0f} = oversold (sweet spot 25-40)")
    elif 40 < rsi <= 50:
        score += 15
        reasons_pos.append(f"RSI {rsi:.0f} = mildly oversold")
    elif rsi < 25:
        score += 5
        reasons_neg.append(f"RSI {rsi:.0f} < 25 = panic, falling knife risk")
    elif rsi > 60:
        reasons_neg.append(f"RSI {rsi:.0f} > 60 = not a dip yet")

    # Week change (-3% to -8% = ideal dip, beyond -10% = warning)
    if -8 <= week_pct <= -3:
        score += 25
        reasons_pos.append(f"Week {week_pct:+.1f}% = real dip (target zone)")
    elif -3 < week_pct < 0:
        score += 10
        reasons_pos.append(f"Week {week_pct:+.1f}% = mild pullback")
    elif week_pct < -10:
        score += 5
        reasons_neg.append(f"Week {week_pct:+.1f}% = severe drop, may keep falling")
    elif week_pct >= 2:
        reasons_neg.append(f"Week {week_pct:+.1f}% = already up, no dip")

    # Distance from 52-week high (good dips happen 10-25% below)
    if -25 <= pct_from_hi <= -10:
        score += 20
        reasons_pos.append(f"{pct_from_hi:+.0f}% from 52w high = healthy pullback zone")
    elif pct_from_hi > -10:
        score += 5
        reasons_pos.append(f"{pct_from_hi:+.0f}% from 52w high = still strong")
    elif pct_from_hi < -40:
        reasons_neg.append(f"{pct_from_hi:+.0f}% from high = deep damage, risk zone")

    # Above 50-day MA = healthy uptrend (good)
    if above_ma50:
        score += 10
        reasons_pos.append("Above 50d MA = uptrend intact")
    else:
        reasons_neg.append("Below 50d MA = trend broken")

    # News sentiment - for dip-buying, MIXED/slightly NEGATIVE is actually best
    # (POSITIVE = priced in, very NEGATIVE = real problem)
    if sentiment == "MIXED":
        score += 10
        reasons_pos.append("News mixed = noise, often buyable")
    elif sentiment == "NEUTRAL":
        score += 8
        reasons_pos.append("News quiet = no catalyst against you")
    elif sentiment == "NEGATIVE" and ai.get("score", 0) > -0.6:
        score += 5
        reasons_pos.append("News mildly negative = sentiment dip")
    elif sentiment == "NEGATIVE" and ai.get("score", 0) <= -0.6:
        reasons_neg.append("News heavily negative = real problem, not just dip")
    elif sentiment == "POSITIVE":
        reasons_neg.append("News positive = no dip, momentum already up")

    # ━━ HARD KILLS (subtract big) ━━
    if earnings:
        score -= 60  # earnings within 14d = always AVOID, gap risk too high
        reasons_neg.append(f"🚫 Earnings in {earnings} = DON'T BUY (gap risk)")
    if pct_from_hi < -50:
        score -= 40  # Deep damage = strong AVOID
        reasons_neg.append("Down 50%+ from high = falling knife / structural problem")
    if rsi < 20:
        score -= 20  # Extreme panic adds to falling knife signal
        reasons_neg.append("RSI < 20 = extreme panic, wait for reversal")

    # Clamp 0-100
    score = max(0, min(100, score))

    # Status label
    if score >= 65:
        status, color = "🟢 STRONG DIP", "#10b981"
    elif score >= 45:
        status, color = "🟡 MILD DIP", "#f59e0b"
    elif score >= 25:
        status, color = "⚪ NOT DIPPING", "#94a3b8"
    else:
        status, color = "🔴 AVOID", "#ef4444"

    return {
        "score": int(score),
        "status": status,
        "color": color,
        "reasons_pos": reasons_pos,
        "reasons_neg": reasons_neg,
    }


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

/* Sentiment bar */
.sentiment-bar-wrap { padding: 4px 0; }
.sentiment-bar-track {
    position: relative;
    height: 8px;
    background: linear-gradient(90deg, #fee2e2 0%, #fef3c7 50%, #dcfce7 100%);
    border-radius: 4px;
    margin-bottom: 4px;
}
.sentiment-bar-zero {
    position: absolute;
    left: 50%;
    top: -2px;
    height: 12px;
    width: 1px;
    background: #94a3b8;
}
.sentiment-bar-marker {
    position: absolute;
    top: -3px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    transform: translateX(-50%);
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.sentiment-bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75em;
}

/* Dip-buy badge */
.dip-buy-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    margin: 6px 0;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.dip-buy-score {
    font-size: 1.4em;
    font-weight: 800;
    line-height: 1;
}
.dip-buy-status {
    font-size: 0.85em;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.dip-buy-tag {
    font-size: 0.7em;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

/* Briefing - now small dropdown style */
.news-briefing-mini {
    font-size: 0.88em;
    line-height: 1.55;
    color: #475569;
    padding: 8px 4px 4px 4px;
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
    ai_on = bool(get_groq_client())
    mx_on = bool(get_marketaux_key())
    ext_on = TRAFILATURA_AVAILABLE
    ai_icon = "✅" if ai_on else "⚠️"
    ai_label = "Groq" if ai_on else "OFF"
    parts = [f"{ai_icon} AI: {ai_label}"]
    if mx_on:
        parts.append("📰 Marketaux")
    if ext_on:
        parts.append("📄 Articles")
    status_html = " · ".join(parts)
    st.markdown(
        f"<div style='padding-top:0.5rem; font-size:0.85em; color:#64748b;'>{status_html}</div>",
        unsafe_allow_html=True
    )

# ━━━ Custom ticker input ━━━
if "custom_tickers" not in st.session_state:
    st.session_state.custom_tickers = {}

ct_col1, ct_col2, ct_col3 = st.columns([2, 1, 3])
with ct_col1:
    new_ticker = st.text_input(
        "➕ Add custom stock to watchlist (this session only)",
        "",
        placeholder="e.g. GOOGL, META, ORCL",
        help="Add any US ticker. Persists for this session. Clears on browser refresh."
    ).upper().strip()
with ct_col2:
    add_clicked = st.button("➕ Add", use_container_width=True)
with ct_col3:
    if st.session_state.custom_tickers:
        custom_list = ", ".join(st.session_state.custom_tickers.keys())
        st.markdown(
            f"<div style='padding-top:0.5rem; font-size:0.85em; color:#64748b;'>"
            f"Custom: <span style='color:#0891b2; font-weight:600;'>{custom_list}</span> "
            f"<a href='?clear_custom=1' style='color:#dc2626; font-size:0.85em;'>(clear all)</a>"
            f"</div>",
            unsafe_allow_html=True
        )

# Handle clear custom tickers via URL param
if st.query_params.get("clear_custom"):
    st.session_state.custom_tickers = {}
    st.query_params.clear()
    st.rerun()

# Handle adding new ticker
if add_clicked and new_ticker:
    # Validate by trying to fetch price (lightweight check)
    if new_ticker in WATCHLIST or new_ticker in st.session_state.custom_tickers:
        st.warning(f"{new_ticker} is already in your watchlist.")
    elif not new_ticker.replace("-", "").replace(".", "").isalnum():
        st.error(f"Invalid ticker format: {new_ticker}")
    else:
        # Quick validation: try to fetch one day of data
        try:
            test = yf.Ticker(new_ticker)
            test_hist = test.history(period="5d")
            if test_hist.empty:
                st.error(f"Could not find ticker {new_ticker}. Check spelling.")
            else:
                # Get company name from yfinance info
                info = test.info if hasattr(test, "info") else {}
                name = info.get("shortName", new_ticker) or info.get("longName", new_ticker) or new_ticker
                sector = info.get("sector", "Custom")
                st.session_state.custom_tickers[new_ticker] = {
                    "name": str(name)[:30],
                    "sector": str(sector)[:20],
                    "type": "swing",
                    "priority": "high",  # custom adds get high priority so they show up
                }
                st.success(f"✅ Added {new_ticker} ({name})")
                st.rerun()
        except Exception as e:
            st.error(f"Error adding {new_ticker}: {str(e)[:100]}")

if refresh:
    st.cache_data.clear()
    st.rerun()

open_pos_set = {t.strip() for t in open_positions.split(",") if t.strip()}

# Merge custom tickers into the active watchlist for this session
ACTIVE_WATCHLIST = {**WATCHLIST, **st.session_state.custom_tickers}

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

# Compute freshness indicator: how recent is the freshest item?
fresh_count_24h = sum(1 for n in macro_news
                     if n.get("pub_dt") and (datetime.now() - n["pub_dt"]).total_seconds() < 86400)
fresh_count_6h = sum(1 for n in macro_news
                    if n.get("pub_dt") and (datetime.now() - n["pub_dt"]).total_seconds() < 21600)

# Show freshness indicator at top
freshness_color = "#10b981" if fresh_count_6h >= 5 else "#f59e0b" if fresh_count_24h >= 5 else "#ef4444"
freshness_label = ("✅ Fresh" if fresh_count_6h >= 5
                   else "⚠️ Mixed" if fresh_count_24h >= 5
                   else "🔴 Stale")
st.markdown(
    f"<div style='display:flex; gap:12px; flex-wrap:wrap; font-size:0.85em; color:#64748b; margin-bottom:8px;'>"
    f"<span style='color:{freshness_color}; font-weight:600;'>{freshness_label}</span>"
    f"<span>· {fresh_count_6h} fresh (last 6h)</span>"
    f"<span>· {fresh_count_24h} recent (last 24h)</span>"
    f"<span>· {len(macro_news)} total items</span>"
    f"<span>· Updated: {datetime.now().strftime('%H:%M')}</span>"
    f"</div>",
    unsafe_allow_html=True
)

# Pass headlines + content with age tags to AI for richer macro analysis
macro_headlines_with_age = []
for i, n in enumerate(macro_news[:12]):
    age = n.get("age_str", "?")
    title = n["title"]
    # Include rich content for top 5 most-recent items
    if i < 5:
        content = ""
        # Marketaux snippet first (free, already there)
        if n.get("snippet"):
            content = n["snippet"]
        # Otherwise try to extract for top 3 only (slower)
        elif i < 3 and n.get("link") and TRAFILATURA_AVAILABLE:
            content = extract_article_text(n["link"], max_chars=600)

        if content and len(content) > len(title) + 50:
            content_trimmed = content[:500].rsplit(' ', 1)[0]
            macro_headlines_with_age.append(
                f"[{age}] {title}\n   ARTICLE: {content_trimmed}..."
            )
        else:
            macro_headlines_with_age.append(f"[{age}] {title}")
    else:
        macro_headlines_with_age.append(f"[{age}] {title}")

macro_result = ai_macro_brief(macro_headlines_with_age, list(ACTIVE_WATCHLIST.keys()))

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
elif macro_news:
    st.markdown("<div class='macro-card'>" +
                "<br>".join(f"• {n['title']}" for n in macro_news[:5]) + "</div>",
                unsafe_allow_html=True)

with st.expander(f"📃 See all macro headlines ({len(macro_news)} items, sorted newest first)"):
    if not macro_news:
        st.info("No macro news available right now. Try refreshing.")
    for n in macro_news:
        age = n.get("age_str", "?")
        cat = n.get("category", "?").upper()
        publisher = n.get("publisher", "")
        # Color-code age
        age_color = "#15803d" if "h ago" in age and "h ago" in age else "#64748b"
        if "m ago" in age:
            age_color = "#15803d"
        elif "h ago" in age:
            try:
                hours = int(age.split("h")[0])
                age_color = "#15803d" if hours < 6 else "#0891b2" if hours < 24 else "#94a3b8"
            except (ValueError, IndexError):
                age_color = "#64748b"
        else:
            age_color = "#94a3b8"

        st.markdown(
            f"<div style='padding:6px 0; border-bottom:1px solid #f1f5f9;'>"
            f"<span style='background:#e0e7ff; color:#3730a3; padding:1px 6px; border-radius:3px; "
            f"font-size:0.7em; font-weight:600; margin-right:6px;'>{cat}</span>"
            f"<a href='{n['link']}' target='_blank' style='color:#1e40af;'>{n['title']}</a>"
            f"<div style='font-size:0.78em; color:#94a3b8; margin-top:2px;'>"
            f"{publisher} · <span style='color:{age_color}; font-weight:600;'>🕐 {age}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

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
    priority_rank = {"high": 0, "med": 1, "low": 2}.get(ACTIVE_WATCHLIST[t]["priority"], 3)
    type_rank = 0 if ACTIVE_WATCHLIST[t]["type"] == "swing" else 1
    return (not in_pos, not macro_relevant, type_rank, priority_rank, t)

ordered_tickers = sorted(ACTIVE_WATCHLIST.keys(), key=sort_key)


@st.cache_data(ttl=900, show_spinner=False)
def build_stock_card_data(ticker: str, meta_name: str = "", meta_sector: str = "",
                           meta_type: str = "swing", meta_priority: str = "med") -> dict:
    """Build all card data for a ticker. Meta passed explicitly to support custom tickers."""
    meta = {"name": meta_name or ticker, "sector": meta_sector, "type": meta_type, "priority": meta_priority}
    price = fetch_price_data(ticker)

    # Pull all available sources and merge (Marketaux first if available — best quality)
    mx_news = fetch_news_marketaux(ticker)  # empty list if no API key
    yh_news = fetch_news_yahoo(ticker)
    gn_news = fetch_news_google(ticker, meta["name"])
    seen = set()
    combined = []
    for n in mx_news + yh_news + gn_news:  # Marketaux first = priority
        key = n["title"][:60].lower()
        if key in seen:
            continue
        seen.add(key)
        combined.append(n)

    # Drop LOW-relevance items if we have enough HIGH/MEDIUM ones
    high_med = [n for n in combined if n["relevance"] in ("HIGH", "MEDIUM")]
    if len(high_med) >= 4:
        combined = high_med

    # Drop items older than 7 days (stale)
    fresh_cutoff = datetime.now() - timedelta(days=7)
    fresh = [n for n in combined if (n["pub_dt"] is None or n["pub_dt"] >= fresh_cutoff)]
    if len(fresh) >= 4:
        combined = fresh

    # Sort: HIGH relevance first, then by recency
    relevance_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    def sort_key(n):
        rel = relevance_rank.get(n["relevance"], 3)
        # Newer first → negate timestamp for ascending sort
        ts = n["pub_dt"].timestamp() if n["pub_dt"] else 0
        return (rel, -ts)
    combined.sort(key=sort_key)

    # Take top 8 for display & AI analysis
    final_news = combined[:8]

    # ━━ Enhance AI input: include article snippets/content for top 3 items ━━
    # We don't extract for all 8 because fetching takes 2-3s per article.
    # Top 3 most relevant + recent get full content; rest get title+age only.
    headlines_for_ai = []
    for i, n in enumerate(final_news):
        age = n.get("age_str", "?")
        title = n["title"]
        if i < 3:
            # Try to get rich content for top 3
            content = get_article_summary_for_ai(n, ticker)
            if content and len(content) > len(title) + 50:
                # Trim content to ~600 chars to keep prompt size reasonable
                content_trimmed = content[:600].rsplit(' ', 1)[0]
                headlines_for_ai.append(
                    f"[{age}] {title}\n   ARTICLE: {content_trimmed}..."
                )
            else:
                headlines_for_ai.append(f"[{age}] {title}")
        else:
            # For items 4-8, just title + age
            headlines_for_ai.append(f"[{age}] {title}")

    ai = ai_analyze_stock(ticker, headlines_for_ai)
    macros = detect_macro_in_headlines([n["title"] for n in final_news])
    earnings = fetch_earnings_date(ticker)

    # Count how many are recent (last 24h) — key signal of "something happening"
    recent_24h = sum(1 for n in final_news
                     if n["pub_dt"] and (datetime.now() - n["pub_dt"]).total_seconds() < 86400)

    return {
        "meta": meta, "price": price, "news": final_news,
        "ai": ai, "macros": macros, "earnings": earnings,
        "news_count": len(combined),
        "recent_24h": recent_24h,
    }


with st.spinner("Analyzing watchlist (this takes ~45s on first load — AI is reading every headline)..."):
    cards = {}
    progress = st.progress(0)
    for idx, ticker in enumerate(ordered_tickers):
        m = ACTIVE_WATCHLIST[ticker]
        cards[ticker] = build_stock_card_data(
            ticker,
            meta_name=m.get("name", ticker),
            meta_sector=m.get("sector", ""),
            meta_type=m.get("type", "swing"),
            meta_priority=m.get("priority", "med"),
        )
        progress.progress((idx + 1) / len(ordered_tickers))
    progress.empty()


# Pre-compute dip-buy scores for all stocks (needed for Top Picks + Calm Mode)
dip_scores = {}
for t in ordered_tickers:
    c = cards[t]
    if not c["price"].get("error"):
        dip_scores[t] = dip_buy_score(t, c["price"], c["ai"], c)


# ━━━━━ TOP DIP-BUY CANDIDATES BANNER ━━━━━
top_dips = sorted(
    [(t, d) for t, d in dip_scores.items() if d["score"] >= 45],
    key=lambda x: -x[1]["score"]
)[:5]

if top_dips:
    st.markdown("#### 🎯 Top Dip-Buy Candidates Right Now")
    cols = st.columns(min(len(top_dips), 5))
    for col, (ticker, dip) in zip(cols, top_dips):
        meta = ACTIVE_WATCHLIST[ticker]
        price = cards[ticker]["price"]
        d_color = dip["color"]
        d_score = dip["score"]
        d_status = dip["status"]
        d_price = price.get("price", 0)
        col.markdown(
            f"<div style='background:white; border-left:4px solid {d_color}; "
            f"border-radius:8px; padding:10px 12px; box-shadow:0 1px 2px rgba(0,0,0,0.05);'>"
            f"<div style='font-size:0.7em; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; font-weight:600;'>"
            f"{meta['sector']}</div>"
            f"<div style='font-size:1.3em; font-weight:700; color:#0f172a;'>{ticker}</div>"
            f"<div style='font-size:0.85em; color:#64748b;'>${d_price:.2f} · "
            f"<span style='color:{d_color}; font-weight:700'>{d_score}/100</span></div>"
            f"<div style='font-size:0.75em; color:{d_color}; margin-top:4px; font-weight:600'>{d_status}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    st.caption("⚠️ A high score is NOT a buy signal. Run the 5-question pre-trade checklist (in your PDF playbook) before any trade.")
    st.markdown("---")


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
    # Show stocks with strong dip-buy setup
    if dip_scores.get(ticker, {}).get("score", 0) >= 60:
        return True
    return False


visible_tickers = [t for t in ordered_tickers if (not calm_mode or show_in_calm(cards[t], t))]

# ━━━ STOCK NAVIGATION SLIDER ━━━
# Color-coded ticker badges that anchor-link to each stock's card below
if visible_tickers:
    nav_html = "<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:8px 10px; margin-bottom:14px;'>"
    nav_html += "<div style='font-size:0.7em; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; margin-bottom:6px;'>"
    nav_html += "🎯 Quick jump (click any ticker to jump to its card)"
    nav_html += "</div>"
    nav_html += "<div style='display:flex; flex-wrap:wrap; gap:6px;'>"
    for t in visible_tickers:
        card = cards[t]
        ai = card["ai"]
        dip = dip_scores.get(t, {"score": 0, "color": "#94a3b8"})
        # Determine badge color
        is_open = t in open_pos_set
        is_winner = t in (macro_result.get("winners") or [])
        is_loser = t in (macro_result.get("losers") or [])
        if is_open:
            border = "#3b82f6"; bg = "#dbeafe"; fg = "#1e40af"
        elif dip["score"] >= 65:
            border = "#10b981"; bg = "#dcfce7"; fg = "#166534"
        elif is_winner:
            border = "#10b981"; bg = "#ecfdf5"; fg = "#047857"
        elif is_loser:
            border = "#ef4444"; bg = "#fee2e2"; fg = "#991b1b"
        elif dip["score"] >= 45:
            border = "#f59e0b"; bg = "#fef3c7"; fg = "#92400e"
        elif ai["action"] == "INVESTIGATE":
            border = "#a855f7"; bg = "#f3e8ff"; fg = "#6b21a8"
        else:
            border = "#cbd5e1"; bg = "white"; fg = "#475569"

        # Tiny indicator
        indicator = ""
        if is_open:
            indicator = " 📌"
        elif card["earnings"]:
            indicator = " 🚫"
        elif card.get("recent_24h", 0) >= 3:
            indicator = " 🔥"
        elif dip["score"] >= 65:
            indicator = " 🟢"

        nav_html += (
            f"<a href='#stock-{t.replace('-','_')}' "
            f"style='display:inline-block; padding:4px 10px; border-radius:6px; "
            f"background:{bg}; color:{fg}; border:1px solid {border}; "
            f"font-size:0.85em; font-weight:700; text-decoration:none; "
            f"letter-spacing:0.3px; transition:all 0.15s;'>"
            f"{t}{indicator}"
            f"</a>"
        )
    nav_html += "</div>"

    # Legend
    nav_html += (
        "<div style='font-size:0.7em; color:#94a3b8; margin-top:8px; display:flex; gap:14px; flex-wrap:wrap;'>"
        "<span>📌 your open position</span>"
        "<span>🟢 strong dip-buy</span>"
        "<span>🔥 fresh news</span>"
        "<span>🚫 earnings soon</span>"
        "</div>"
    )
    nav_html += "</div>"
    st.markdown(nav_html, unsafe_allow_html=True)

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

    # Anchor target for the navigation slider above
    anchor_id = f"stock-{ticker.replace('-', '_')}"
    st.markdown(f"<div id='{anchor_id}' style='scroll-margin-top:80px;'></div>",
                unsafe_allow_html=True)

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
            # Sentiment bar replaces plain "Score: +0.20"
            st.markdown(sentiment_bar_html(ai["score"]), unsafe_allow_html=True)

        with c5:
            st.markdown(f"**{action_badge(ai['action'])}**")
            badges = []
            if card["earnings"]:
                badges.append(f"🚫 Earnings {card['earnings']}")
            recent_24h = card.get("recent_24h", 0)
            if recent_24h >= 3:
                badges.append(f"🔥 {recent_24h} fresh today")
            elif recent_24h == 0 and card["news_count"] > 0:
                badges.append("💤 No fresh news")
            if card["news_count"] >= 8:
                badges.append("⚡ High news vol")
            for m in card["macros"]:
                badges.append(f"🌐 {m}")
            if badges:
                st.caption(" · ".join(badges))

        # Quick bottom-line summary (1 sentence)
        if ai.get("summary"):
            st.markdown(f"💬 _{ai['summary']}_")

        # DIP-BUY ANALYSIS — strategy-specific scoring
        dip = dip_buy_score(ticker, price, ai, card)
        st.markdown(
            f"<div class='dip-buy-card' style='border-left: 4px solid {dip['color']};'>"
            f"<div class='dip-buy-score' style='color: {dip['color']};'>{dip['score']}<span style='font-size:0.5em; color:#94a3b8;'>/100</span></div>"
            f"<div style='flex:1;'>"
            f"<div class='dip-buy-tag'>Dip-Buy Score</div>"
            f"<div class='dip-buy-status' style='color: {dip['color']};'>{dip['status']}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # NEWS BRIEFING — now small dropdown as you asked
        if ai.get("news_briefing"):
            with st.expander("📋 Read full news briefing"):
                st.markdown(
                    f"<div class='news-briefing-mini'>{ai['news_briefing']}</div>",
                    unsafe_allow_html=True
                )

        # DIP-BUY REASONING — small dropdown
        if dip["reasons_pos"] or dip["reasons_neg"]:
            with st.expander(f"🎯 Why dip-buy score is {dip['score']}/100"):
                if dip["reasons_pos"]:
                    st.markdown("**✓ Working in favor of a dip-buy:**")
                    for r in dip["reasons_pos"]:
                        st.markdown(f"- {r}")
                if dip["reasons_neg"]:
                    st.markdown("**✗ Working against a dip-buy:**")
                    for r in dip["reasons_neg"]:
                        st.markdown(f"- {r}")
                # Action prompt based on score
                if dip["score"] >= 65:
                    st.markdown(
                        "<div style='margin-top:10px; padding:10px; background:#ecfdf5; border-radius:6px; "
                        "color:#047857; font-weight:600;'>"
                        "💡 Strong dip-buy candidate. Run pre-trade checklist (PDF playbook), set 5-7% stop, max €30 risk.</div>",
                        unsafe_allow_html=True
                    )
                elif dip["score"] >= 45:
                    st.markdown(
                        "<div style='margin-top:10px; padding:10px; background:#fffbeb; border-radius:6px; "
                        "color:#92400e; font-weight:600;'>"
                        "💡 Mild setup. Watch for confirmation (RSI turning, volume on bounce) before entering.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='margin-top:10px; padding:10px; background:#f1f5f9; border-radius:6px; "
                        "color:#475569;'>"
                        "💡 Not a buyable dip right now. Check back later or look at another stock.</div>",
                        unsafe_allow_html=True
                    )

        # PER-HEADLINE BREAKDOWN
        with st.expander(f"📃 Headlines with AI verdict for {ticker} ({card['news_count']} items)"):
            cc1, cc2 = st.columns([3, 1.2])
            with cc1:
                if ai.get("headline_analysis"):
                    # Build lookup that handles "[Xh ago] Title" format from AI
                    def strip_prefix(s: str) -> str:
                        # Remove "[2h ago] " or "[3d ago] " etc.
                        return re.sub(r"^\[[^\]]+\]\s*", "", s)
                    news_lookup = {n["title"]: n for n in card["news"]}
                    for item in ai["headline_analysis"]:
                        h_with_prefix = item["headline"]
                        h_clean = strip_prefix(h_with_prefix)
                        n = news_lookup.get(h_clean) or news_lookup.get(h_with_prefix, {})
                        url = n.get("link", "")
                        pub = n.get("publisher", "")
                        age = n.get("age_str", "?")
                        relevance = n.get("relevance", "MEDIUM")
                        title_html = f"<a href='{url}' target='_blank'>{h_clean}</a>" if url else h_clean
                        why = item.get("why", "")

                        # Relevance badge
                        if relevance == "HIGH":
                            rel_badge = "<span style='background:#dbeafe;color:#1e40af;padding:1px 5px;border-radius:3px;font-size:0.7em;font-weight:600;'>DIRECT</span>"
                        elif relevance == "LOW":
                            rel_badge = "<span style='background:#f1f5f9;color:#94a3b8;padding:1px 5px;border-radius:3px;font-size:0.7em;'>GENERIC</span>"
                        else:
                            rel_badge = ""

                        # Age coloring: fresh news = green, stale = gray
                        if "m ago" in age or "just now" in age:
                            age_color = "#15803d"  # very fresh
                        elif "h ago" in age:
                            hours = int(age.split("h")[0]) if age.split("h")[0].isdigit() else 99
                            age_color = "#15803d" if hours < 6 else "#0891b2" if hours < 24 else "#94a3b8"
                        else:
                            age_color = "#94a3b8"

                        age_html = f"<span style='color:{age_color}; font-weight:600; font-size:0.78em;'>🕐 {age}</span>"

                        st.markdown(
                            f"<div class='headline-row'>"
                            f"<div class='impact-cell'>{impact_badge(item['impact'])}</div>"
                            f"<div class='headline-cell'>"
                            f"<div class='headline-title'>{title_html} {rel_badge}</div>"
                            f"<div class='headline-why'>→ {why} "
                            f"<span style='color:#94a3b8'>· {pub} · </span>{age_html}</div>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )
                else:
                    for n in card["news"]:
                        age = n.get("age_str", "?")
                        if n["link"]:
                            st.markdown(f"- [{n['title']}]({n['link']}) · _{n['publisher']}_ · 🕐 {age}")
                        else:
                            st.markdown(f"- {n['title']} · _{n['publisher']}_ · 🕐 {age}")

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
