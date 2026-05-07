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
import html
from datetime import datetime, timedelta, date
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser
import plotly.graph_objects as go
from groq import Groq

import storage

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

# ═══════════════════════════════════════════════════════════════════════════
# CATALYST CALENDAR — known macro events that create gap risk for swing trades
# Update this list yearly. Dates are in ISO format (YYYY-MM-DD).
# Sources: Federal Reserve schedule (fomc.gov), BLS release calendar (bls.gov)
# ═══════════════════════════════════════════════════════════════════════════
KNOWN_CATALYSTS_2026 = [
    # FOMC meetings (rate decisions) — typically 2-day, decision on day 2
    {"date": "2026-01-28", "event": "FOMC rate decision", "category": "fed", "severity": "high"},
    {"date": "2026-03-18", "event": "FOMC rate decision + SEP",  "category": "fed", "severity": "high"},
    {"date": "2026-04-29", "event": "FOMC rate decision", "category": "fed", "severity": "high"},
    {"date": "2026-06-17", "event": "FOMC rate decision + SEP",  "category": "fed", "severity": "high"},
    {"date": "2026-07-29", "event": "FOMC rate decision", "category": "fed", "severity": "high"},
    {"date": "2026-09-16", "event": "FOMC rate decision + SEP",  "category": "fed", "severity": "high"},
    {"date": "2026-11-04", "event": "FOMC rate decision", "category": "fed", "severity": "high"},
    {"date": "2026-12-16", "event": "FOMC rate decision + SEP",  "category": "fed", "severity": "high"},
    # CPI releases (typically mid-month, 8:30am ET)
    {"date": "2026-05-12", "event": "CPI (Apr)", "category": "macro", "severity": "high"},
    {"date": "2026-06-10", "event": "CPI (May)", "category": "macro", "severity": "high"},
    {"date": "2026-07-15", "event": "CPI (Jun)", "category": "macro", "severity": "high"},
    {"date": "2026-08-12", "event": "CPI (Jul)", "category": "macro", "severity": "high"},
    {"date": "2026-09-10", "event": "CPI (Aug)", "category": "macro", "severity": "high"},
    {"date": "2026-10-14", "event": "CPI (Sep)", "category": "macro", "severity": "high"},
    {"date": "2026-11-12", "event": "CPI (Oct)", "category": "macro", "severity": "high"},
    {"date": "2026-12-10", "event": "CPI (Nov)", "category": "macro", "severity": "high"},
    # NFP / Jobs reports (first Friday of each month)
    {"date": "2026-05-01", "event": "Jobs report (Apr)", "category": "macro", "severity": "med"},
    {"date": "2026-06-05", "event": "Jobs report (May)", "category": "macro", "severity": "med"},
    {"date": "2026-07-02", "event": "Jobs report (Jun)", "category": "macro", "severity": "med"},
    {"date": "2026-08-07", "event": "Jobs report (Jul)", "category": "macro", "severity": "med"},
    {"date": "2026-09-04", "event": "Jobs report (Aug)", "category": "macro", "severity": "med"},
    {"date": "2026-10-02", "event": "Jobs report (Sep)", "category": "macro", "severity": "med"},
    {"date": "2026-11-06", "event": "Jobs report (Oct)", "category": "macro", "severity": "med"},
    {"date": "2026-12-04", "event": "Jobs report (Nov)", "category": "macro", "severity": "med"},
]


def upcoming_catalysts(days_ahead: int = 30) -> list[dict]:
    """Return calendar events from today through `days_ahead` days, oldest first."""
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    out = []
    for c in KNOWN_CATALYSTS_2026:
        try:
            d = date.fromisoformat(c["date"])
            if today <= d <= cutoff:
                days_away = (d - today).days
                out.append({**c, "days_away": days_away, "date_obj": d})
        except (ValueError, TypeError):
            continue
    out.sort(key=lambda x: x["date_obj"])
    return out


# ============================================================================
# AI BRAIN - REWRITTEN FOR PER-HEADLINE ANALYSIS
# ============================================================================

def _load_groq_keys() -> list[str]:
    """
    Load all Groq API keys in priority order.
    Reads from Streamlit secrets first, then environment variables.

    Conventions supported:
      - GROQ_API_KEY        (primary)
      - GROQ_API_KEY_2      (first backup)
      - GROQ_API_KEY_3      (second backup)
      - GROQ_API_KEY_4, _5  (further backups, up to 5 total)

    Backups can be from a different Groq account, OR an OpenRouter/Cerebras
    key as long as the SDK call format matches (Groq SDK is OpenAI-compatible).
    """
    keys: list[str] = []
    seen: set[str] = set()

    def _try_add(name: str) -> None:
        val = None
        try:
            val = st.secrets.get(name)
        except Exception:
            pass
        if not val:
            val = os.environ.get(name)
        if val and val.strip() and val.strip() not in seen:
            seen.add(val.strip())
            keys.append(val.strip())

    _try_add("GROQ_API_KEY")
    for i in range(2, 6):
        _try_add(f"GROQ_API_KEY_{i}")
    return keys


class GroqKeyPool:
    """
    Failover pool for multiple Groq keys.
    On rate-limit / auth errors, the failing key is parked for a cooldown period
    and subsequent calls move to the next key.

    Cooldowns:
      - rate_limit_exceeded (per-minute):  60 seconds
      - rate_limit_exceeded (per-day):     6 hours
      - authentication errors:             until restart (key is bad)
    """
    def __init__(self) -> None:
        self.keys = _load_groq_keys()
        self._cooldown_until: dict[str, float] = {}
        self._lock = __import__("threading").Lock()
        # Cache a Groq client per key — avoids re-creating SDK objects every call
        self._clients: dict[str, Groq] = {}

    def has_any_key(self) -> bool:
        return len(self.keys) > 0

    def total_keys(self) -> int:
        return len(self.keys)

    def available_count(self) -> int:
        import time
        now = time.time()
        with self._lock:
            return sum(1 for k in self.keys if self._cooldown_until.get(k, 0) <= now)

    def _client_for(self, key: str) -> Optional[Groq]:
        if key not in self._clients:
            try:
                self._clients[key] = Groq(api_key=key)
            except Exception:
                return None
        return self._clients[key]

    def _cool_down(self, key: str, seconds: float) -> None:
        import time
        with self._lock:
            self._cooldown_until[key] = time.time() + seconds

    def chat_completion(self, **kwargs):
        """
        Call client.chat.completions.create with automatic failover across keys.
        Raises the last exception if every key fails.
        Returns None if no keys are configured.
        """
        import time
        if not self.keys:
            return None

        last_exc: Optional[Exception] = None
        now = time.time()

        # Try each key in order, skipping any that are still cooling down
        for key in self.keys:
            if self._cooldown_until.get(key, 0) > now:
                continue
            client = self._client_for(key)
            if client is None:
                continue
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                # Categorize the error and pick a cooldown window
                if "rate_limit" in msg or "429" in msg or "too many requests" in msg:
                    # Per-day quota errors mention "tokens per day" or "requests per day"
                    if "per day" in msg or "tpd" in msg or "rpd" in msg:
                        self._cool_down(key, 6 * 3600)  # 6 hours
                    else:
                        self._cool_down(key, 60)  # 1 minute
                    continue
                elif "auth" in msg or "401" in msg or "invalid api" in msg:
                    self._cool_down(key, 86400)  # disable for ~a day
                    continue
                else:
                    # Network / server / unknown errors: short cooldown, try next
                    self._cool_down(key, 30)
                    continue

        # All keys exhausted
        if last_exc:
            raise last_exc
        return None


@st.cache_resource(show_spinner=False)
def get_groq_pool() -> GroqKeyPool:
    """Singleton key pool — survives Streamlit reruns within the same process."""
    return GroqKeyPool()


def get_groq_client() -> Optional[Groq]:
    """
    Backwards-compat shim. Returns a working client if any key is available,
    None otherwise. New code should use get_groq_pool().chat_completion(...).
    """
    pool = get_groq_pool()
    if not pool.has_any_key():
        return None
    # Return the first non-cooled-down client for direct-use callers
    import time
    now = time.time()
    for key in pool.keys:
        if pool._cooldown_until.get(key, 0) <= now:
            return pool._client_for(key)
    # All keys cooled down — return the first client anyway so callers can still try
    return pool._client_for(pool.keys[0]) if pool.keys else None


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
def ai_analyze_stock(ticker: str, headlines: list[str],
                     news_items: Optional[list[dict]] = None) -> dict:
    """
    Per-headline impact analysis + overall verdict for a stock.

    Returns:
      sentiment, score, summary (with reasoning), action,
      headline_analysis (list of {headline, impact, why}),
      source, error

    news_items (optional): full news dicts that may contain 'marketaux_sentiment'
      scores. Used by the fallback when AI is unavailable, to give better
      results than keyword matching.
    """
    pool = get_groq_pool()
    if not pool.has_any_key() or not headlines:
        return _fallback_sentiment(headlines, news_items=news_items)

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
        response = pool.chat_completion(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise financial analyst. You output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        if response is None:
            return _fallback_sentiment(headlines, error="all keys cooled down",
                                       news_items=news_items)
        text = response.choices[0].message.content
        parsed = _extract_json(text)
        if not parsed:
            return _fallback_sentiment(headlines, error="JSON parse failed",
                                       news_items=news_items)

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
        return _fallback_sentiment(headlines, error=str(e), news_items=news_items)


def _fallback_sentiment(headlines: list[str], error: str = "",
                        news_items: Optional[list[dict]] = None) -> dict:
    """
    Fallback when AI is unavailable. Priority order:
      1. If news_items have Marketaux sentiment scores, use those (good)
      2. Otherwise, use keyword matching (basic)

    The first path is significantly better than keywords because Marketaux
    already ran a financial-NLP model on the headline. We just trust their
    score instead of running our own AI on top.
    """
    if not headlines:
        return {
            "sentiment": "QUIET", "score": 0.0,
            "summary": "No recent news for this ticker.",
            "news_briefing": "",
            "action": "IGNORE", "headline_analysis": [],
            "source": f"fallback{(' (' + error + ')') if error else ''}",
        }

    # Build a lookup: headline → marketaux_sentiment if we have it
    # Headlines passed to AI are prefixed with "[2h ago] ", we need to match against
    # raw titles. So index the news_items by their cleaned title.
    mx_sentiment_lookup: dict[str, float] = {}
    if news_items:
        for n in news_items:
            mx_score = n.get("marketaux_sentiment")
            if mx_score is not None and isinstance(mx_score, (int, float)):
                title = n.get("title", "")
                if title:
                    mx_sentiment_lookup[title] = float(mx_score)

    per_headline = []
    pos_total = neg_total = 0
    used_marketaux_count = 0
    used_keyword_count = 0

    # Strip "[Xh ago] " or "[Xm ago] " prefix that ai_analyze adds before passing
    prefix_re = re.compile(r"^\[[^\]]+\]\s*")

    for h_with_prefix in headlines[:8]:
        h_clean = prefix_re.sub("", h_with_prefix).split("\n")[0].strip()

        # Try Marketaux first
        mx_score = mx_sentiment_lookup.get(h_clean)
        if mx_score is None:
            # Try matching against just the title portion (no ARTICLE: chunk)
            mx_score = mx_sentiment_lookup.get(h_with_prefix.split("\n")[0])

        if mx_score is not None:
            # Use Marketaux's score directly
            used_marketaux_count += 1
            if mx_score > 0.15:
                impact = "GOOD"
                why = f"Marketaux sentiment: {mx_score:+.2f} (positive)"
                pos_total += abs(mx_score)
            elif mx_score < -0.15:
                impact = "BAD"
                why = f"Marketaux sentiment: {mx_score:+.2f} (negative)"
                neg_total += abs(mx_score)
            else:
                impact = "NEUTRAL"
                why = f"Marketaux sentiment: {mx_score:+.2f} (neutral)"
            per_headline.append({"headline": h_with_prefix, "impact": impact, "why": why})
        else:
            # Keyword fallback for this headline
            used_keyword_count += 1
            words = set(h_clean.lower().split())
            pos = len(words & POSITIVE_WORDS)
            neg = len(words & NEGATIVE_WORDS)
            if pos > neg:
                impact, why = "GOOD", "Contains positive keywords"
                pos_total += pos
            elif neg > pos:
                impact, why = "BAD", "Contains negative keywords"
                neg_total += neg
            else:
                impact, why = "NEUTRAL", "No clear sentiment"
            per_headline.append({"headline": h_with_prefix, "impact": impact, "why": why})

    # Aggregate score
    total = pos_total + neg_total
    if total == 0:
        sent, score, action = "NEUTRAL", 0.0, "IGNORE"
        summary = "No clear signal in headlines. Routine market noise."
    else:
        score = (pos_total - neg_total) / max(total, 1)
        # Cap score at +/- 1
        score = max(-1.0, min(1.0, score))
        if score > 0.3:
            sent, action = "POSITIVE", "WATCH"
        elif score < -0.3:
            sent, action = "NEGATIVE", "INVESTIGATE"
        else:
            sent, action = "MIXED", "WATCH"

        # Tailor summary based on which method was used
        if used_marketaux_count >= used_keyword_count and used_marketaux_count > 0:
            method_note = f"based on Marketaux sentiment scores ({used_marketaux_count} headlines)"
        elif used_marketaux_count > 0:
            method_note = (f"based on Marketaux ({used_marketaux_count}) and "
                           f"keyword analysis ({used_keyword_count})")
        else:
            method_note = "based on keyword analysis (Marketaux data unavailable)"

        if sent == "POSITIVE":
            summary = f"Headlines lean positive {method_note}. AI unavailable for deeper reasoning."
        elif sent == "NEGATIVE":
            summary = f"Headlines lean negative {method_note}. AI unavailable for deeper reasoning."
        else:
            summary = f"Mixed signals in headlines {method_note}. AI unavailable for deeper reasoning."

    # Build a simple briefing
    brief_method = ("Using Marketaux sentiment as backup" if used_marketaux_count > used_keyword_count
                    else "AI is currently unavailable, so this is a keyword-based view")
    briefing = (f"{brief_method}: " +
                " ".join([prefix_re.sub("", h).split(chr(10))[0] for h in headlines[:5]])[:600] +
                " — verify by reading the full headlines below.")

    # Tag the source so the user can see in the cards what method was used
    if used_marketaux_count > 0 and used_marketaux_count >= used_keyword_count:
        source_tag = f"marketaux ({used_marketaux_count}/{len(per_headline)} hl)"
    else:
        source_tag = f"fallback{(' (' + error + ')') if error else ''}"

    return {
        "sentiment": sent, "score": score, "summary": summary,
        "news_briefing": briefing,
        "action": action, "headline_analysis": per_headline,
        "source": source_tag,
    }


@st.cache_data(ttl=900, show_spinner=False)  # 15 min cache
def ai_macro_brief(macro_news: list[str], watchlist_tickers: list[str]) -> dict:
    """Macro brief + identifies which watchlist stocks are affected."""
    pool = get_groq_pool()
    if not pool.has_any_key() or not macro_news:
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
        response = pool.chat_completion(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        if response is None:
            return {"brief": "", "winners": [], "losers": [], "source": "all_keys_cooled_down"}
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

        # ━━ Volume analysis: distinguish panic dips from exhaustion dips ━━
        # vol_ratio = today's volume / 20-day average. >1.5 on a down day = panic.
        last_vol = hist["Volume"].iloc[-1] if "Volume" in hist.columns else 0
        avg_vol_20 = hist["Volume"].rolling(20).mean().iloc[-1] if len(hist) >= 20 else last_vol
        vol_ratio = float(last_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0
        last_was_down = bool(day_pct < 0)

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
            "vol_ratio": vol_ratio,
            "last_was_down": last_was_down,
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
    symbols = [
        ("^GSPC", "S&P 500"),
        ("^IXIC", "Nasdaq"),
        ("^VIX", "VIX"),
        ("^GDAXI", "DAX"),
        ("EURUSD=X", "EUR/USD"),  # FX exposure for EU-based USD-stock investor
    ]
    for sym, label in symbols:
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


@st.cache_data(ttl=86400, show_spinner=False)  # 24h cache (analysts move slowly)
def fetch_analyst_data(ticker: str) -> dict:
    """
    Fetch analyst recommendations + price targets from Yahoo Finance.

    Returns a dict with:
      - rec_label: human-readable rating ("Strong Buy" / "Buy" / "Hold" / "Sell" / "Strong Sell")
      - rec_mean: numeric (1.0 = Strong Buy, 5.0 = Strong Sell)
      - strong_buy / buy / hold / sell / strong_sell: counts (if available)
      - total_analysts: count of analysts giving recommendations
      - target_mean / target_high / target_low: price targets in USD
      - target_count: number of analysts with targets
      - upside_pct: (target_mean - current_price) / current_price * 100
      - error: present only if fetch failed

    All fields are optional — Yahoo may not have data for every ticker.
    """
    out: dict = {}
    try:
        t = yf.Ticker(ticker)

        # ── info dict has price targets and recommendation summary ──
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Price target fields
        if info.get("targetMeanPrice"):
            try:
                out["target_mean"] = float(info["targetMeanPrice"])
            except (TypeError, ValueError):
                pass
        if info.get("targetHighPrice"):
            try:
                out["target_high"] = float(info["targetHighPrice"])
            except (TypeError, ValueError):
                pass
        if info.get("targetLowPrice"):
            try:
                out["target_low"] = float(info["targetLowPrice"])
            except (TypeError, ValueError):
                pass
        if info.get("numberOfAnalystOpinions"):
            try:
                out["target_count"] = int(info["numberOfAnalystOpinions"])
            except (TypeError, ValueError):
                pass

        # Current price for upside calculation
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if current and out.get("target_mean"):
            try:
                cur = float(current)
                if cur > 0:
                    out["upside_pct"] = (out["target_mean"] - cur) / cur * 100
            except (TypeError, ValueError):
                pass

        # Recommendation fields
        rec_key = info.get("recommendationKey")
        if rec_key:
            label_map = {
                "strong_buy": "Strong Buy",
                "buy": "Buy",
                "hold": "Hold",
                "sell": "Sell",
                "strong_sell": "Strong Sell",
                "underperform": "Sell",  # Yahoo sometimes uses these
                "outperform": "Buy",
            }
            out["rec_label"] = label_map.get(rec_key.lower(), rec_key.replace("_", " ").title())
            out["rec_key"] = rec_key.lower()
        if info.get("recommendationMean"):
            try:
                out["rec_mean"] = float(info["recommendationMean"])
            except (TypeError, ValueError):
                pass

        # ── recommendations DataFrame: detailed buy/hold/sell counts ──
        try:
            recs_df = t.recommendations
            if recs_df is not None and not recs_df.empty:
                latest = recs_df.iloc[0]  # most recent row
                # Yahoo uses different column names — try common variants
                for src_col, dest_key in [
                    ("strongBuy", "strong_buy"),
                    ("buy", "buy"),
                    ("hold", "hold"),
                    ("sell", "sell"),
                    ("strongSell", "strong_sell"),
                ]:
                    if src_col in latest.index:
                        try:
                            out[dest_key] = int(latest[src_col])
                        except (TypeError, ValueError):
                            pass

                if any(k in out for k in ("strong_buy", "buy", "hold", "sell", "strong_sell")):
                    out["total_analysts"] = sum(
                        out.get(k, 0)
                        for k in ("strong_buy", "buy", "hold", "sell", "strong_sell")
                    )
        except Exception:
            pass  # no recommendations DataFrame available

    except Exception as e:
        out["error"] = str(e)
    return out


def analyst_rating_html(analyst: dict) -> str:
    """Render the analyst recommendation as a small colored badge."""
    if not analyst or analyst.get("error"):
        return ""
    rec_key = analyst.get("rec_key", "")
    rec_label = analyst.get("rec_label", "")
    if not rec_label:
        return ""

    # Color scheme: bullish green → bearish red
    color_map = {
        "strong_buy": ("#dcfce7", "#15803d"),
        "buy":        ("#d1fae5", "#047857"),
        "outperform": ("#d1fae5", "#047857"),
        "hold":       ("#fef3c7", "#92400e"),
        "sell":       ("#fee2e2", "#b91c1c"),
        "underperform": ("#fee2e2", "#b91c1c"),
        "strong_sell": ("#fecaca", "#991b1b"),
    }
    bg, fg = color_map.get(rec_key, ("#f1f5f9", "#475569"))

    # Detailed counts if we have them
    detail = ""
    total = analyst.get("total_analysts", 0)
    if total:
        bullish = analyst.get("strong_buy", 0) + analyst.get("buy", 0)
        detail = f" {bullish}/{total}"

    return (
        f"<span style='background:{bg}; color:{fg}; padding:2px 8px; border-radius:4px; "
        f"font-size:0.75em; font-weight:600;'>🎯 {html.escape(rec_label)}{detail}</span>"
    )


def price_target_html(analyst: dict, current_price: float) -> str:
    """Render the price-target info as a compact line."""
    if not analyst or not analyst.get("target_mean"):
        return ""
    mean = analyst["target_mean"]
    upside = analyst.get("upside_pct")
    if upside is None and current_price > 0:
        upside = (mean - current_price) / current_price * 100

    if upside is None:
        return f"<span style='color:#64748b; font-size:0.8em;'>💰 Target: ${mean:.2f}</span>"

    # Color upside vs downside
    if upside >= 10:
        color = "#15803d"
    elif upside >= 0:
        color = "#0891b2"
    elif upside >= -10:
        color = "#92400e"
    else:
        color = "#b91c1c"

    high = analyst.get("target_high")
    low = analyst.get("target_low")
    range_str = ""
    if high and low and high != low:
        range_str = f" <span style='color:#94a3b8;'>(${low:.0f}-${high:.0f})</span>"

    return (
        f"<span style='font-size:0.8em;'>💰 Target: <b>${mean:.2f}</b>{range_str} · "
        f"<b style='color:{color};'>{upside:+.1f}%</b></span>"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_date_obj(ticker: str) -> Optional[date]:
    """Return raw earnings date object for the catalyst calendar. Cached 1h."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if not isinstance(cal, dict):
            return None
        ed = cal.get("Earnings Date")
        if not ed:
            return None
        if isinstance(ed, list) and ed:
            ed = ed[0]
        if hasattr(ed, "year"):
            return ed
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


def sentiment_sparkline_svg(history: list[dict], width: int = 80, height: int = 22) -> str:
    """Tiny inline SVG sparkline of sentiment score over time (last N days)."""
    if not history or len(history) < 2:
        return ""
    scores = [max(-1.0, min(1.0, h.get("score", 0))) for h in history]
    n = len(scores)
    points = []
    for i, s in enumerate(scores):
        x = i * (width / max(1, n - 1))
        y = height - ((s + 1) / 2) * height
        points.append(f"{x:.1f},{y:.1f}")
    pts_str = " ".join(points)
    last = scores[-1]
    color = "#10b981" if last > 0.2 else "#ef4444" if last < -0.2 else "#94a3b8"
    return (
        f"<svg width='{width}' height='{height}' "
        f"style='vertical-align:middle; margin-left:4px;' aria-hidden='true'>"
        f"<line x1='0' y1='{height/2}' x2='{width}' y2='{height/2}' "
        f"stroke='#cbd5e1' stroke-width='0.5' stroke-dasharray='2,2'/>"
        f"<polyline fill='none' stroke='{color}' stroke-width='1.5' points='{pts_str}'/>"
        f"</svg>"
    )


def trend_badge(trend: str) -> str:
    """Small badge for sentiment trend: IMPROVING / DETERIORATING / STABLE."""
    if trend == "IMPROVING":
        return ("<span style='background:#dcfce7;color:#15803d;padding:1px 5px;"
                "border-radius:3px;font-size:0.7em;font-weight:600;'>↗ IMPROVING</span>")
    if trend == "DETERIORATING":
        return ("<span style='background:#fee2e2;color:#b91c1c;padding:1px 5px;"
                "border-radius:3px;font-size:0.7em;font-weight:600;'>↘ DETERIORATING</span>")
    if trend == "STABLE":
        return ("<span style='background:#f1f5f9;color:#64748b;padding:1px 5px;"
                "border-radius:3px;font-size:0.7em;'>→ STABLE</span>")
    return ""


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

    # ━━ VOLUME CONFIRMATION ━━
    # On a DOWN day: high volume = panic (falling knife), low volume = exhaustion (good dip)
    # On an UP day: not informative for dip-buying
    vol_ratio = price.get("vol_ratio", 1.0)
    last_was_down = price.get("last_was_down", False)
    if last_was_down:
        if vol_ratio >= 2.0:
            score -= 15
            reasons_neg.append(f"Volume {vol_ratio:.1f}× avg on down day = panic, falling knife risk")
        elif vol_ratio >= 1.5:
            score -= 5
            reasons_neg.append(f"Volume {vol_ratio:.1f}× avg on down day = elevated selling")
        elif vol_ratio < 0.8:
            score += 10
            reasons_pos.append(f"Volume {vol_ratio:.1f}× avg on down day = exhaustion, sellers running out")
        elif vol_ratio < 1.0:
            score += 5
            reasons_pos.append(f"Volume {vol_ratio:.1f}× avg on down day = mild selling, not panic")

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
    pool = get_groq_pool()
    n_keys = pool.total_keys()
    n_avail = pool.available_count() if n_keys else 0
    mx_on = bool(get_marketaux_key())
    ext_on = TRAFILATURA_AVAILABLE
    if n_keys == 0:
        ai_icon, ai_label = "⚠️", "AI: OFF"
    elif n_avail == 0:
        ai_icon, ai_label = "🟡", f"AI: {n_keys} keys (all cooled)"
    elif n_keys == 1:
        ai_icon, ai_label = "✅", "AI: 1 key"
    else:
        ai_icon, ai_label = "✅", f"AI: {n_avail}/{n_keys} keys ready"
    parts = [f"{ai_icon} {ai_label}"]
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
        # FX pairs need 4 decimals; indices look better as integers
        if "/" in label:
            price_str = f"{data['price']:.4f}"
        else:
            price_str = f"{data['price']:,.0f}"
        col.metric(label, price_str, f"{data['pct']:+.2f}%")

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
    st.markdown(
        f"<div class='macro-card'>{html.escape(macro_result['brief'])}</div>",
        unsafe_allow_html=True
    )
    if macro_result.get("winners"):
        winners_str = ", ".join(html.escape(w) for w in macro_result["winners"])
        st.markdown(
            f"<div class='macro-affected'>🟢 <b>Likely winners on your watchlist:</b> "
            f"{winners_str} — <i>{html.escape(macro_result.get('winner_reason',''))}</i></div>",
            unsafe_allow_html=True
        )
    if macro_result.get("losers"):
        losers_str = ", ".join(html.escape(l) for l in macro_result["losers"])
        st.markdown(
            f"<div class='macro-affected bad'>🔴 <b>Likely losers on your watchlist:</b> "
            f"{losers_str} — <i>{html.escape(macro_result.get('loser_reason',''))}</i></div>",
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
        # Color-code age (greener = fresher)
        if "m ago" in age or "just now" in age:
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
            f"font-size:0.7em; font-weight:600; margin-right:6px;'>{html.escape(cat)}</span>"
            f"<a href='{html.escape(n['link'], quote=True)}' target='_blank' rel='noopener noreferrer' "
            f"style='color:#1e40af;'>{html.escape(n['title'])}</a>"
            f"<div style='font-size:0.78em; color:#94a3b8; margin-top:2px;'>"
            f"{html.escape(publisher)} · <span style='color:{age_color}; font-weight:600;'>🕐 {age}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

# ============================================================================
# CATALYST CALENDAR — gap-risk events for the next 30 days
# ============================================================================

st.markdown("### 📅 Catalyst Calendar (next 30 days)")
st.caption("Macro events + your watchlist earnings. Avoid opening new swing positions ≤ 2 days before a high-severity event.")

upcoming = upcoming_catalysts(days_ahead=30)

# Pull earnings within 30 days for the active watchlist (cached helper = fast)
watchlist_earnings = []
for t in ACTIVE_WATCHLIST:
    ed = fetch_earnings_date_obj(t)
    if ed is None:
        continue
    days_away = (ed - date.today()).days
    if 0 <= days_away <= 30:
        watchlist_earnings.append({
            "date_obj": ed,
            "date": ed.isoformat(),
            "event": f"{t} earnings",
            "category": "earnings",
            "severity": "high" if days_away <= 14 else "med",
            "days_away": days_away,
            "ticker": t,
        })

all_events = upcoming + watchlist_earnings
all_events.sort(key=lambda e: e["date_obj"])

if not all_events:
    st.info("No major events scheduled in the next 30 days.")
else:
    # Render as a compact two-column grid
    n_cols = 3
    rows = [all_events[i:i+n_cols] for i in range(0, len(all_events), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, ev in zip(cols, row):
            sev = ev.get("severity", "med")
            cat = ev.get("category", "")
            days_away = ev["days_away"]
            # Color by urgency
            if days_away <= 2:
                bg, fg, border = "#fee2e2", "#991b1b", "#ef4444"
                urgency = "🔴 IMMINENT"
            elif days_away <= 7:
                bg, fg, border = "#fef3c7", "#92400e", "#f59e0b"
                urgency = "🟡 THIS WEEK"
            else:
                bg, fg, border = "#eff6ff", "#1e40af", "#3b82f6"
                urgency = "🔵 UPCOMING"
            cat_emoji = {"fed": "🏦", "macro": "📊", "earnings": "💰"}.get(cat, "📌")
            col.markdown(
                f"<div style='background:{bg}; border-left:3px solid {border}; "
                f"border-radius:6px; padding:8px 10px; margin-bottom:6px;'>"
                f"<div style='font-size:0.7em; color:{fg}; font-weight:700; letter-spacing:0.4px;'>{urgency}</div>"
                f"<div style='font-weight:600; color:{fg}; margin-top:2px;'>{cat_emoji} {html.escape(ev['event'])}</div>"
                f"<div style='font-size:0.8em; color:#64748b; margin-top:2px;'>"
                f"{ev['date_obj'].strftime('%a %b %d')} · in {days_away}d"
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

    ai = ai_analyze_stock(ticker, headlines_for_ai, news_items=final_news)

    # Persist today's sentiment so we can draw sparklines and detect trends
    try:
        if ai.get("source") == "ai" and ai.get("sentiment") not in (None, "QUIET"):
            storage.record_sentiment(ticker, ai["sentiment"], ai.get("score", 0.0))
    except Exception:
        pass  # Storage failure shouldn't break the dashboard

    macros = detect_macro_in_headlines([n["title"] for n in final_news])
    earnings = fetch_earnings_date(ticker)

    # Analyst recommendations + price targets (from Yahoo, no AI cost)
    analyst = fetch_analyst_data(ticker)

    # Count how many are recent (last 24h) — key signal of "something happening"
    recent_24h = sum(1 for n in final_news
                     if n["pub_dt"] and (datetime.now() - n["pub_dt"]).total_seconds() < 86400)

    return {
        "meta": meta, "price": price, "news": final_news,
        "ai": ai, "macros": macros, "earnings": earnings,
        "analyst": analyst,
        "news_count": len(combined),
        "recent_24h": recent_24h,
    }


with st.spinner("Analyzing watchlist (parallel fetch — should be ~15-20s on first load)..."):
    cards = {}
    progress = st.progress(0)
    total = len(ordered_tickers)

    def _build_one(ticker: str) -> tuple[str, dict]:
        m = ACTIVE_WATCHLIST[ticker]
        return ticker, build_stock_card_data(
            ticker,
            meta_name=m.get("name", ticker),
            meta_sector=m.get("sector", ""),
            meta_type=m.get("type", "swing"),
            meta_priority=m.get("priority", "med"),
        )

    # 4 workers: fast enough to feel snappy, conservative enough to stay under
    # Groq rate limits (concurrent AI calls are the limiting factor here).
    completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_build_one, t): t for t in ordered_tickers}
        for future in as_completed(futures):
            try:
                ticker, card = future.result()
                cards[ticker] = card
            except Exception as e:
                t = futures[future]
                cards[t] = {
                    "meta": ACTIVE_WATCHLIST[t], "price": {"error": str(e)},
                    "news": [], "ai": {"sentiment": "NEUTRAL", "score": 0, "summary": "",
                                       "news_briefing": "", "action": "IGNORE",
                                       "headline_analysis": [], "source": f"error: {e}"},
                    "macros": [], "earnings": None, "analyst": {},
                    "news_count": 0, "recent_24h": 0,
                }
            completed += 1
            progress.progress(completed / total)
    progress.empty()

    # Re-order cards dict to match the originally-ordered tickers (futures complete out of order)
    cards = {t: cards[t] for t in ordered_tickers if t in cards}


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
            st.markdown(f"**{pin}{html.escape(ticker)}** · {html.escape(meta['name'])}{macro_tag}",
                        unsafe_allow_html=True)
            st.caption(f"{meta['sector']} · {meta['type']}")

        with c2:
            st.metric("Price", f"${price['price']:.2f}", f"{price['day_pct']:+.2f}%",
                      label_visibility="collapsed")

        with c3:
            st.metric("Week", f"{price['week_pct']:+.1f}%", label_visibility="collapsed")
            st.caption(f"RSI: {price['rsi']:.0f}")

        with c4:
            # Get trend + sparkline data
            sent_history = storage.get_sentiment_history(ticker, days=14)
            trend = storage.get_sentiment_trend(ticker, days=5)
            sparkline = sentiment_sparkline_svg(sent_history) if sent_history else ""
            trend_html = trend_badge(trend) if trend != "NEW" else ""

            st.markdown(
                f"{sentiment_icon(ai['sentiment'])} **{html.escape(ai['sentiment'])}** "
                f"{trend_html} {sparkline}",
                unsafe_allow_html=True
            )
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
            # Strip backticks/asterisks that would break the markdown italics wrapping
            safe_summary = ai["summary"].replace("_", " ").replace("*", "").replace("`", "")
            st.markdown(f"💬 _{safe_summary}_")

        # ━━ ANALYST RATING + PRICE TARGET (Yahoo, free, no AI) ━━
        # Shows long-term Wall Street view as a complement to short-term news sentiment.
        analyst = card.get("analyst", {})
        rating_html = analyst_rating_html(analyst)
        target_html = price_target_html(analyst, price.get("price", 0))
        if rating_html or target_html:
            parts = [p for p in [rating_html, target_html] if p]
            st.markdown(
                "<div style='display:flex; gap:14px; align-items:center; flex-wrap:wrap; "
                "margin: 6px 0 4px 0; padding: 6px 10px; background: #f8fafc; "
                "border-radius: 6px; border-left: 3px solid #0891b2;'>"
                + " · ".join(parts)
                + "</div>",
                unsafe_allow_html=True
            )

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

        # ━━ Quick link to the checklist page (no extra code, just navigation) ━━
        try:
            st.page_link(
                "pages/02_Notes.py",
                label=f"📋 Open this in checklist (Notes page)",
                icon=None,
                use_container_width=False,
            )
        except Exception:
            # Older Streamlit versions don't have page_link — fail silently
            pass

        # NEWS BRIEFING — now small dropdown as you asked
        if ai.get("news_briefing"):
            with st.expander("📋 Read full news briefing"):
                st.markdown(
                    f"<div class='news-briefing-mini'>{html.escape(ai['news_briefing'])}</div>",
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

        # ════════════════════════════════════════════════════════════════
        # TRADE SETUP: pre-trade checklist + position sizing
        # Only appears for actionable setups (score >= 45) or open positions.
        # ════════════════════════════════════════════════════════════════
        show_setup = (dip["score"] >= 45) or is_open
        if show_setup:
            with st.expander("🛒 Trade Setup — checklist + position sizing"):
                # Default 5-question checklist; user can adapt to their PDF playbook
                checklist_questions = [
                    "Stock is down 3-8% over the past week (real dip, not free-fall)",
                    "RSI is between 25 and 50 (oversold, not panic)",
                    "No earnings within next 14 days",
                    "News is noise/sector — not a stock-specific catastrophe (fraud/CEO/recall)",
                    "I have a stop-loss price written down before I click Buy",
                ]

                checklist_key = f"checklist_{ticker}"
                if checklist_key not in st.session_state:
                    st.session_state[checklist_key] = [False] * len(checklist_questions)

                st.markdown("**Pre-trade checklist** — all 5 must be ✓ before sizing appears:")
                checks = []
                for i, q in enumerate(checklist_questions):
                    val = st.checkbox(
                        q,
                        value=st.session_state[checklist_key][i],
                        key=f"chk_{ticker}_{i}",
                    )
                    st.session_state[checklist_key][i] = val
                    checks.append(val)

                all_checked = all(checks)

                if not all_checked:
                    n_done = sum(checks)
                    st.markdown(
                        f"<div style='padding:8px; background:#f1f5f9; border-radius:6px; "
                        f"color:#64748b; font-size:0.85em;'>"
                        f"☑ {n_done}/{len(checklist_questions)} checked. "
                        f"Sizing calculator unlocks at 5/5.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div style='padding:8px; background:#dcfce7; border-radius:6px; "
                        "color:#166534; font-size:0.9em; font-weight:600; margin-bottom:8px;'>"
                        "✅ Checklist complete. Below: position sizing.</div>",
                        unsafe_allow_html=True
                    )

                    # ── Position sizing calculator ──
                    sz1, sz2, sz3 = st.columns(3)
                    with sz1:
                        max_risk_eur = st.number_input(
                            "Max risk (€)",
                            min_value=5.0, max_value=500.0, value=30.0, step=5.0,
                            key=f"risk_{ticker}",
                            help="Total euros you're willing to lose if the stop hits.",
                        )
                    with sz2:
                        stop_pct = st.slider(
                            "Stop loss (%)",
                            min_value=3.0, max_value=12.0, value=6.0, step=0.5,
                            key=f"stop_{ticker}",
                            help="How far below your entry the stop sits.",
                        )
                    with sz3:
                        eur_usd = overview.get("EUR/USD", {}).get("price", 1.08)
                        st.metric("EUR/USD", f"{eur_usd:.4f}",
                                  label_visibility="visible")

                    entry_usd = price["price"]
                    stop_price_usd = entry_usd * (1 - stop_pct / 100)
                    risk_per_share_usd = entry_usd - stop_price_usd
                    max_risk_usd = max_risk_eur * eur_usd
                    if risk_per_share_usd > 0:
                        n_shares = int(max_risk_usd / risk_per_share_usd)
                    else:
                        n_shares = 0
                    position_usd = n_shares * entry_usd
                    position_eur = position_usd / eur_usd if eur_usd > 0 else 0

                    if n_shares > 0:
                        st.markdown(
                            f"<div style='background:#f0fdf4; border-left:4px solid #10b981; "
                            f"border-radius:6px; padding:12px 14px; margin-top:8px;'>"
                            f"<div style='font-size:0.75em; color:#15803d; font-weight:600; "
                            f"text-transform:uppercase; letter-spacing:0.5px;'>Suggested position</div>"
                            f"<div style='font-size:1.4em; font-weight:700; color:#0f172a; margin:4px 0;'>"
                            f"{n_shares} shares @ ${entry_usd:.2f}</div>"
                            f"<div style='font-size:0.85em; color:#475569; line-height:1.6;'>"
                            f"Position value: <b>${position_usd:,.2f}</b> (€{position_eur:,.2f})<br>"
                            f"Stop-loss at: <b>${stop_price_usd:.2f}</b> ({stop_pct:.1f}% below entry)<br>"
                            f"Risk if stop hits: <b>€{max_risk_eur:.2f}</b> "
                            f"(${max_risk_usd:.2f})"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning("Position size rounds to 0 shares — share price too high "
                                   "for your risk budget. Increase max risk or pick a lower-priced stock.")

        # ════════════════════════════════════════════════════════════════
        # JOURNAL: log this decision (always available, even for skips)
        # ════════════════════════════════════════════════════════════════
        with st.expander("📝 Log this decision to journal"):
            jl1, jl2 = st.columns([1, 2])
            with jl1:
                action_choice = st.selectbox(
                    "Decision",
                    options=["WATCH", "BUY", "SKIP", "SELL"],
                    key=f"journal_action_{ticker}",
                    help="What did you decide right now?",
                )
            with jl2:
                reason_text = st.text_input(
                    "Reason (1 line)",
                    placeholder="e.g. RSI 32, news mixed, dip score 68 — entering",
                    key=f"journal_reason_{ticker}",
                )

            log_btn = st.button(
                f"📝 Log {action_choice} for {ticker}",
                key=f"journal_btn_{ticker}",
                use_container_width=False,
            )
            if log_btn:
                if not reason_text.strip():
                    st.error("Add a reason — future-you will thank you.")
                else:
                    try:
                        entry = storage.add_journal_entry(
                            ticker=ticker,
                            action=action_choice,
                            price_at_decision=price["price"],
                            reason=reason_text.strip(),
                            dip_score=dip["score"],
                            sentiment=ai.get("sentiment"),
                            sentiment_score=ai.get("score"),
                            rsi=price.get("rsi"),
                            week_pct=price.get("week_pct"),
                        )
                        st.success(f"✅ Logged. See the 📓 Journal page in the sidebar. "
                                   f"Entry id: {entry['id'][-6:]}")
                    except Exception as e:
                        st.error(f"Could not save: {e}")

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
                        # Escape user/web/AI-supplied text before injecting into HTML
                        h_clean_esc = html.escape(h_clean)
                        url_esc = html.escape(url, quote=True)
                        pub_esc = html.escape(pub)
                        why_esc = html.escape(item.get("why", ""))
                        title_html = (f"<a href='{url_esc}' target='_blank' rel='noopener noreferrer'>{h_clean_esc}</a>"
                                      if url else h_clean_esc)

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
                            f"<div class='headline-why'>→ {why_esc} "
                            f"<span style='color:#94a3b8'>· {pub_esc} · </span>{age_html}</div>"
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
    pool = get_groq_pool()
    if not pool.has_any_key():
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
                response = pool.chat_completion(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=300,
                    temperature=0.4,
                )
                if response is None:
                    st.error("All AI keys are rate-limited right now. Try again in a minute.")
                else:
                    answer = response.choices[0].message.content.strip()
                    st.markdown(f"**AI:** {html.escape(answer)}")
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
