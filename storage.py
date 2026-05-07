"""
storage.py
==========
Lightweight JSON persistence for:
  - Trade journal entries (decisions you logged)
  - Daily sentiment history per ticker (for sparklines + trend detection)

Files live in ./data/ — they auto-create on first write.
If you run on Streamlit Cloud, the disk is ephemeral (wiped on restart).
For persistent cloud storage, swap these functions to write to S3/GCS/Postgres.
"""
import json
import os
import threading
from datetime import datetime, date, timedelta
from typing import Optional

# ─── File locations ────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JOURNAL_FILE = os.path.join(_DATA_DIR, "trade_journal.json")
SENTIMENT_HISTORY_FILE = os.path.join(_DATA_DIR, "sentiment_history.json")

# Single mutex for both files — concurrent writes from threaded ticker fetches
_lock = threading.Lock()


def _ensure_data_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _safe_load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _safe_save(path: str, data) -> bool:
    """Atomic save: write to temp file, then rename."""
    try:
        _ensure_data_dir()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# JOURNAL
# ═══════════════════════════════════════════════════════════════════════════

def load_journal() -> list[dict]:
    """Return all journal entries, newest first."""
    with _lock:
        entries = _safe_load(JOURNAL_FILE, [])
    if not isinstance(entries, list):
        return []
    # Sort newest-first
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


def add_journal_entry(
    ticker: str,
    action: str,            # "BUY", "SKIP", "WATCH", "SELL"
    price_at_decision: float,
    reason: str,
    dip_score: Optional[int] = None,
    sentiment: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    rsi: Optional[float] = None,
    week_pct: Optional[float] = None,
    notes: str = "",
) -> dict:
    """Append a decision to the journal. Returns the saved entry."""
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker.upper(),
        "action": action.upper(),
        "price_at_decision": float(price_at_decision),
        "reason": reason,
        "dip_score": int(dip_score) if dip_score is not None else None,
        "sentiment": sentiment,
        "sentiment_score": float(sentiment_score) if sentiment_score is not None else None,
        "rsi": float(rsi) if rsi is not None else None,
        "week_pct": float(week_pct) if week_pct is not None else None,
        "notes": notes,
        # Outcome fields are filled later by update_outcome()
        "outcome_price": None,
        "outcome_pct": None,
        "outcome_review": None,
    }
    with _lock:
        entries = _safe_load(JOURNAL_FILE, [])
        if not isinstance(entries, list):
            entries = []
        entries.append(entry)
        _safe_save(JOURNAL_FILE, entries)
    return entry


def delete_journal_entry(entry_id: str) -> bool:
    with _lock:
        entries = _safe_load(JOURNAL_FILE, [])
        if not isinstance(entries, list):
            return False
        new_entries = [e for e in entries if e.get("id") != entry_id]
        if len(new_entries) == len(entries):
            return False
        return _safe_save(JOURNAL_FILE, new_entries)


def update_outcome(entry_id: str, outcome_price: float, review: str = "") -> bool:
    """Stamp an entry with its current price + a review note."""
    with _lock:
        entries = _safe_load(JOURNAL_FILE, [])
        if not isinstance(entries, list):
            return False
        for e in entries:
            if e.get("id") == entry_id:
                start = e.get("price_at_decision") or 0
                pct = ((outcome_price - start) / start * 100) if start else 0
                e["outcome_price"] = float(outcome_price)
                e["outcome_pct"] = round(pct, 2)
                e["outcome_review"] = review
                e["outcome_logged_at"] = datetime.now().isoformat(timespec="seconds")
                return _safe_save(JOURNAL_FILE, entries)
    return False


# ═══════════════════════════════════════════════════════════════════════════
# SENTIMENT HISTORY  (one daily snapshot per ticker per day)
# ═══════════════════════════════════════════════════════════════════════════

def record_sentiment(ticker: str, sentiment: str, score: float) -> None:
    """Save today's sentiment for this ticker. Idempotent within the same day."""
    today = date.today().isoformat()
    with _lock:
        history = _safe_load(SENTIMENT_HISTORY_FILE, {})
        if not isinstance(history, dict):
            history = {}
        ticker_hist = history.get(ticker.upper(), [])
        # Replace today's entry if it exists, else append
        ticker_hist = [h for h in ticker_hist if h.get("date") != today]
        ticker_hist.append({
            "date": today,
            "sentiment": sentiment,
            "score": round(float(score), 3),
        })
        # Keep only last 90 days to stop the file growing forever
        cutoff = (date.today() - timedelta(days=90)).isoformat()
        ticker_hist = [h for h in ticker_hist if h.get("date", "") >= cutoff]
        ticker_hist.sort(key=lambda h: h["date"])
        history[ticker.upper()] = ticker_hist
        _safe_save(SENTIMENT_HISTORY_FILE, history)


def get_sentiment_history(ticker: str, days: int = 14) -> list[dict]:
    """Return last `days` sentiment snapshots for ticker, oldest first."""
    with _lock:
        history = _safe_load(SENTIMENT_HISTORY_FILE, {})
    if not isinstance(history, dict):
        return []
    items = history.get(ticker.upper(), [])
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    return [h for h in items if h.get("date", "") >= cutoff]


def get_sentiment_trend(ticker: str, days: int = 5) -> str:
    """Return 'IMPROVING' | 'DETERIORATING' | 'STABLE' | 'NEW' based on recent scores."""
    hist = get_sentiment_history(ticker, days=days)
    if len(hist) < 2:
        return "NEW"
    recent_avg = sum(h["score"] for h in hist[-2:]) / max(1, len(hist[-2:]))
    earlier_avg = sum(h["score"] for h in hist[:2]) / max(1, len(hist[:2]))
    delta = recent_avg - earlier_avg
    if delta > 0.2:
        return "IMPROVING"
    if delta < -0.2:
        return "DETERIORATING"
    return "STABLE"
