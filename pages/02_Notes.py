"""
Static HTML viewer page.

Auto-lists every .html file inside the ./static/ folder (next to dashboard.py)
and renders the selected one as a sandboxed iframe.

To add content: drop any .html file into ./static/  — it shows up here automatically.
"""
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Notes · Swing Dashboard", page_icon="📄", layout="wide")

st.title("📄 Notes & Playbooks")
st.caption("Static HTML files — drop any .html file into the `static/` folder to see it here.")

# Path: ../static/  relative to this pages/ folder
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)  # auto-create on first run

html_files = sorted(STATIC_DIR.glob("*.html"))

if not html_files:
    st.info(
        f"No HTML files yet.\n\n"
        f"**To add one:** put any `.html` file inside this folder:\n\n"
        f"`{STATIC_DIR}`\n\n"
        f"Then refresh this page. The file will appear in a dropdown above. "
        f"You can add as many as you want — playbook, checklists, notes, "
        f"market overview reports, anything HTML."
    )
    st.stop()

# Two columns: file picker on left, height control on right
c1, c2 = st.columns([3, 1])

with c1:
    file_names = [f.name for f in html_files]
    selected_name = st.selectbox(
        "Pick a file",
        options=file_names,
        index=0,
        help="Lists every .html file inside your static/ folder."
    )

with c2:
    height = st.slider(
        "Display height",
        min_value=600, max_value=3600, value=2200, step=100,
        help="Adjust if your content is cut off or has empty space below. "
             "Long checklists/playbooks often need 2200-2800px.",
    )

selected_file = STATIC_DIR / selected_name

# File metadata
size_kb = selected_file.stat().st_size / 1024
modified = selected_file.stat().st_mtime
import datetime as _dt
modified_str = _dt.datetime.fromtimestamp(modified).strftime("%b %d, %Y · %H:%M")

st.caption(f"📄 `{selected_name}` · {size_kb:.1f} KB · last modified {modified_str}")

# For long HTML files (checklists, playbooks, full-page dashboards),
# the default height may not be enough. Show a hint above the embed.
if size_kb > 30 and height < 2000:
    st.info(
        f"💡 This file is {size_kb:.0f} KB — large HTML pages need more height. "
        f"Try sliding **Display height** up to 2200-2800px if content looks cut off."
    )

st.markdown("---")

# Read + render
try:
    html_content = selected_file.read_text(encoding="utf-8")
except UnicodeDecodeError:
    # Fallback for non-UTF8 encoded files
    html_content = selected_file.read_text(encoding="latin-1")
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# Sandboxed iframe — safer than st.markdown(unsafe_allow_html=True) for arbitrary HTML
# because it isolates styles and scripts from the rest of the dashboard.
components.html(html_content, height=height, scrolling=True)

# ─── Optional: download / export buttons ──────────────────────────────────
with st.expander("⬇️ Download or copy raw HTML"):
    st.download_button(
        f"Download {selected_name}",
        data=html_content,
        file_name=selected_name,
        mime="text/html",
    )
    with st.expander("View raw source"):
        st.code(html_content, language="html")
