import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import os, sys

BUILD = "SANITY-CHECK-v1"

st.set_page_config(page_title=f"Sanity • {BUILD}", layout="wide")
st.title(f"🧪 Sanity Check • {BUILD}")

with st.sidebar:
    st.subheader("Debug")
    st.write({"__file__": __file__, "cwd": os.getcwd(), "python": sys.version})
    if st.button("Clear cache (once)"):
        st.cache_data.clear()
        st.experimental_rerun()

st.write("If you see this title with SANITY-CHECK-v1, **this exact file is running**.")

# --- Pure marker heat grid: 2 Assets x 3 TFs, markers ONLY (no text) ---
assets = ["BTC/USDT", "ETH/USDT"]
tfs = ["5m", "1h", "4h"]
biases = {
    ("BTC/USDT","5m"): "🟢 LONG",
    ("BTC/USDT","1h"): "⚪ NEUTRAL",
    ("BTC/USDT","4h"): "🔴 SHORT",
    ("ETH/USDT","5m"): "⚪ NEUTRAL",
    ("ETH/USDT","1h"): "🔴 SHORT",
    ("ETH/USDT","4h"): "🟢 LONG",
}

color_map = {"🟢 LONG": "#16a34a", "🔴 SHORT": "#b91c1c", "⚪ NEUTRAL": "#6b7280"}

xs, ys, colors, hovers = [], [], [], []
for i, a in enumerate(assets):
    for j, tf in enumerate(tfs):
        b = biases[(a, tf)]
        xs.append(j); ys.append(i)
        colors.append(color_map[b])
        hovers.append(f"{a} | {tf} | {b}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=xs, y=ys,
    mode="markers",  # <- markers only, NO text
    marker=dict(size=42, color=colors, line=dict(color="#000000", width=2), symbol="circle"),
    hovertemplate="%{text}<extra></extra>",
    text=hovers
))
fig.update_xaxes(tickvals=list(range(len(tfs))), ticktext=tfs, side="top", color="white", showgrid=False, zeroline=False)
fig.update_yaxes(tickvals=list(range(len(assets))), ticktext=assets, autorange="reversed", color="white", showgrid=False, zeroline=False)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=220, plot_bgcolor="#000000", paper_bgcolor="#000000")

st.subheader("Pure marker grid (no text, no emoji in cells)")
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Last update: {datetime.utcnow().isoformat()}Z • Build: {BUILD}")
