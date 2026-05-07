"""
Trade Journal page (auto-discovered by Streamlit because it lives in pages/).
View, filter, and review every logged decision. Outcomes update from yfinance.
"""
import os
import sys
import io
import csv
from datetime import datetime, date

import streamlit as st
import pandas as pd
import yfinance as yf

# Make storage.py importable from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import storage  # noqa: E402

st.set_page_config(page_title="Journal · Swing Dashboard", page_icon="📓", layout="wide")

st.title("📓 Trade Journal")
st.caption("Every decision you logged. Review outcomes to find your behavioral patterns.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD ENTRIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

entries = storage.load_journal()

if not entries:
    st.info(
        "No entries yet. Go to the main dashboard, find a stock, and use "
        "**📝 Log this decision to journal** under any card. Future-you will thank present-you."
    )
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY STATS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

n_total = len(entries)
n_buys = sum(1 for e in entries if e.get("action") == "BUY")
n_skips = sum(1 for e in entries if e.get("action") == "SKIP")
n_watches = sum(1 for e in entries if e.get("action") == "WATCH")
n_sells = sum(1 for e in entries if e.get("action") == "SELL")

# Win/loss for BUY entries that have outcome data
buys_with_outcome = [e for e in entries if e.get("action") == "BUY" and e.get("outcome_pct") is not None]
n_buy_wins = sum(1 for e in buys_with_outcome if e.get("outcome_pct", 0) > 0)
n_buy_losses = sum(1 for e in buys_with_outcome if e.get("outcome_pct", 0) <= 0)
buy_win_rate = (n_buy_wins / len(buys_with_outcome) * 100) if buys_with_outcome else None

# Skip review: how often was skipping the right call?
skips_with_outcome = [e for e in entries if e.get("action") == "SKIP" and e.get("outcome_pct") is not None]
# A "good skip" = you skipped and the stock dropped or went sideways
n_good_skips = sum(1 for e in skips_with_outcome if e.get("outcome_pct", 0) <= 2)
skip_accuracy = (n_good_skips / len(skips_with_outcome) * 100) if skips_with_outcome else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total entries", n_total)
c2.metric("Buys logged", n_buys, f"{n_buy_wins}W / {n_buy_losses}L" if buys_with_outcome else "no outcomes yet")
c3.metric("Buy win rate", f"{buy_win_rate:.0f}%" if buy_win_rate is not None else "—",
          help="Of buys with logged outcomes, % that were profitable at outcome time.")
c4.metric("Skip accuracy", f"{skip_accuracy:.0f}%" if skip_accuracy is not None else "—",
          help="Of skips with outcomes, % where the stock didn't run away (≤+2%).")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILTERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

f1, f2, f3, f4 = st.columns([1.5, 1.5, 1.5, 1])
all_tickers = sorted({e.get("ticker", "") for e in entries})
all_actions = sorted({e.get("action", "") for e in entries})

with f1:
    sel_tickers = st.multiselect("Ticker", options=all_tickers, default=[])
with f2:
    sel_actions = st.multiselect("Action", options=all_actions, default=[])
with f3:
    show_only = st.selectbox("Show", ["All", "Without outcome (need review)",
                                       "With outcome (reviewed)"])
with f4:
    refresh_all = st.button("🔄 Refresh outcomes", use_container_width=True,
                             help="Pulls current price for every entry without an outcome.")

# Filter entries
def matches(e: dict) -> bool:
    if sel_tickers and e.get("ticker") not in sel_tickers:
        return False
    if sel_actions and e.get("action") not in sel_actions:
        return False
    has_outcome = e.get("outcome_pct") is not None
    if show_only == "Without outcome (need review)" and has_outcome:
        return False
    if show_only == "With outcome (reviewed)" and not has_outcome:
        return False
    return True

filtered = [e for e in entries if matches(e)]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REFRESH OUTCOMES (bulk update)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if refresh_all:
    pending = [e for e in filtered if e.get("outcome_pct") is None]
    if not pending:
        st.info("Nothing to refresh — every visible entry already has an outcome logged.")
    else:
        progress = st.progress(0)
        n_updated = 0
        for i, e in enumerate(pending):
            try:
                tk = yf.Ticker(e["ticker"])
                hist = tk.history(period="2d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])
                    storage.update_outcome(e["id"], current_price, review="Auto-refreshed")
                    n_updated += 1
            except Exception:
                pass
            progress.progress((i + 1) / len(pending))
        progress.empty()
        st.success(f"Updated {n_updated} of {len(pending)} entries.")
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.expander("⬇️ Export to CSV"):
    if filtered:
        buf = io.StringIO()
        fieldnames = ["timestamp", "ticker", "action", "price_at_decision", "outcome_price",
                      "outcome_pct", "dip_score", "sentiment", "sentiment_score", "rsi",
                      "week_pct", "reason", "outcome_review", "id"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in filtered:
            writer.writerow(e)
        st.download_button(
            "Download journal.csv",
            data=buf.getvalue(),
            file_name=f"journal_{date.today().isoformat()}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No entries match the current filter.")

st.caption(f"Showing {len(filtered)} of {n_total} entries.")
st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDER ENTRIES (newest first)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTION_STYLE = {
    "BUY":   {"bg": "#dcfce7", "fg": "#166534", "border": "#10b981"},
    "SELL":  {"bg": "#fef3c7", "fg": "#92400e", "border": "#f59e0b"},
    "SKIP":  {"bg": "#f1f5f9", "fg": "#475569", "border": "#94a3b8"},
    "WATCH": {"bg": "#dbeafe", "fg": "#1e40af", "border": "#3b82f6"},
}

for e in filtered:
    style = ACTION_STYLE.get(e.get("action", ""), ACTION_STYLE["WATCH"])
    ts = e.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        date_str = dt.strftime("%a %b %d, %Y · %H:%M")
        days_since = (datetime.now() - dt).days
    except (ValueError, TypeError):
        date_str = ts
        days_since = None

    # Compute outcome display
    pct = e.get("outcome_pct")
    if pct is not None:
        outcome_color = "#15803d" if pct > 0 else "#b91c1c" if pct < 0 else "#64748b"
        outcome_html = (
            f"<span style='color:{outcome_color}; font-weight:700;'>"
            f"{pct:+.2f}% since</span>"
        )
        outcome_label = f"${e.get('outcome_price', 0):.2f}"
    else:
        outcome_html = "<span style='color:#94a3b8;'>not yet reviewed</span>"
        outcome_label = "—"

    # Was it a "good call"?
    review_tag = ""
    if pct is not None:
        action = e.get("action")
        if action == "BUY" and pct > 1:
            review_tag = "<span style='background:#dcfce7;color:#15803d;padding:2px 6px;border-radius:4px;font-size:0.7em;font-weight:700;'>✓ Good buy</span>"
        elif action == "BUY" and pct < -1:
            review_tag = "<span style='background:#fee2e2;color:#b91c1c;padding:2px 6px;border-radius:4px;font-size:0.7em;font-weight:700;'>✗ Bad buy</span>"
        elif action == "SKIP" and pct > 3:
            review_tag = "<span style='background:#fef3c7;color:#92400e;padding:2px 6px;border-radius:4px;font-size:0.7em;font-weight:700;'>😬 Missed it</span>"
        elif action == "SKIP" and pct < 1:
            review_tag = "<span style='background:#dcfce7;color:#15803d;padding:2px 6px;border-radius:4px;font-size:0.7em;font-weight:700;'>✓ Good skip</span>"

    with st.container(border=True):
        h1, h2, h3, h4 = st.columns([1.3, 1.3, 2, 1])

        with h1:
            st.markdown(
                f"<div style='display:inline-block; background:{style['bg']}; color:{style['fg']}; "
                f"border:1px solid {style['border']}; border-radius:6px; padding:4px 12px; "
                f"font-weight:700; letter-spacing:0.5px;'>{e.get('action','—')}</div>"
                f"<div style='margin-top:6px; font-size:1.1em; font-weight:700;'>{e.get('ticker','—')}</div>",
                unsafe_allow_html=True
            )
        with h2:
            st.markdown(
                f"<div style='font-size:0.75em; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;'>"
                f"Decision price</div>"
                f"<div style='font-size:1.1em; font-weight:600;'>"
                f"${e.get('price_at_decision', 0):.2f}</div>"
                f"<div style='font-size:0.75em; color:#94a3b8; margin-top:2px;'>"
                f"{date_str}</div>",
                unsafe_allow_html=True
            )
        with h3:
            st.markdown(
                f"<div style='font-size:0.75em; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;'>"
                f"Outcome {('· ' + str(days_since) + 'd later') if days_since else ''}</div>"
                f"<div style='font-size:1.1em; font-weight:600;'>"
                f"{outcome_label}</div>"
                f"<div style='font-size:0.85em; margin-top:2px;'>{outcome_html} {review_tag}</div>",
                unsafe_allow_html=True
            )
        with h4:
            # Manual outcome refresh for this entry
            if st.button("🔄 Update", key=f"upd_{e['id']}", use_container_width=True):
                try:
                    tk = yf.Ticker(e["ticker"])
                    hist = tk.history(period="2d")
                    if not hist.empty:
                        current = float(hist["Close"].iloc[-1])
                        storage.update_outcome(e["id"], current, review="Manual refresh")
                        st.rerun()
                    else:
                        st.error("No price data")
                except Exception as ex:
                    st.error(f"Error: {ex}")
            if st.button("🗑️ Delete", key=f"del_{e['id']}", use_container_width=True):
                if storage.delete_journal_entry(e["id"]):
                    st.rerun()

        # Reason and context
        reason = e.get("reason", "")
        st.markdown(f"📝 _{reason}_" if reason else "_(no reason logged)_")

        # Context details (collapsible)
        ctx_parts = []
        if e.get("dip_score") is not None:
            ctx_parts.append(f"Dip score: **{e['dip_score']}/100**")
        if e.get("sentiment"):
            sent_score = e.get("sentiment_score")
            if sent_score is not None:
                ctx_parts.append(f"Sentiment: **{e['sentiment']}** ({sent_score:+.2f})")
            else:
                ctx_parts.append(f"Sentiment: **{e['sentiment']}**")
        if e.get("rsi") is not None:
            ctx_parts.append(f"RSI: **{e['rsi']:.0f}**")
        if e.get("week_pct") is not None:
            ctx_parts.append(f"Week: **{e['week_pct']:+.1f}%**")
        if ctx_parts:
            st.caption(" · ".join(ctx_parts))

        # Outcome review note (if any)
        if e.get("outcome_review") and e["outcome_review"] != "Auto-refreshed" \
                and e["outcome_review"] != "Manual refresh":
            st.markdown(f"💭 _Review: {e['outcome_review']}_")
