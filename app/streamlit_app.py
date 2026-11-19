"""Streamlit dashboard: forest area and loss over time.

The notebook (notebooks/deforestation.ipynb) writes the results to outputs/ first.
Then run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SOURCE_LABEL = "umeshuu/cambodia-deforestation"
REVISION = "44ae7414"

st.set_page_config(page_title="Deforestation Tracking", layout="wide")


def load_csv(name: str) -> pd.DataFrame | None:
    path = OUT / name
    return pd.read_csv(path) if path.exists() else None


st.title("Deforestation Tracking - forest loss over time")
st.caption(f"Source: Hugging Face dataset `{SOURCE_LABEL}` @ `{REVISION}` "
           "- model-predicted forest mask - 10 m - UTM 48N")

area = load_csv("forest_area.csv")
loss = load_csv("forest_loss.csv")
timeline = OUT / "forest_timeline.png"

if area is None:
    st.error("No results found. Run the analysis first:")
    st.code("jupyter notebook notebooks/deforestation.ipynb", language="powershell")
    st.stop()

years = area["year"].tolist()
first, last = area.iloc[0], area.iloc[-1]
net = first["forest_ha"] - last["forest_ha"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Forest {first['year']}", f"{first['forest_ha']:,.0f} ha")
c2.metric(f"Forest {last['year']}", f"{last['forest_ha']:,.0f} ha",
          delta=f"{-net:,.0f} ha")
c3.metric("Total lost", f"{loss['loss_ha'].sum():,.0f} ha" if loss is not None else "-")
c4.metric("Snapshots", len(years))

if timeline.exists():
    st.image(str(timeline), caption="Forest area (top) and loss per interval (bottom)",
             width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Forest area per year")
    st.dataframe(area, width="stretch", hide_index=True)
    st.download_button("Download forest_area.csv", area.to_csv(index=False),
                       "forest_area.csv", "text/csv")
with right:
    st.subheader("Loss per interval")
    if loss is not None:
        st.dataframe(loss, width="stretch", hide_index=True)
        st.download_button("Download forest_loss.csv", loss.to_csv(index=False),
                           "forest_loss.csv", "text/csv")
    else:
        st.info("No interval data (need at least two years).")

overlays = sorted(OUT.glob("overlay_*.png"))
if overlays:
    st.subheader("Where the forest was lost")
    labels = {p.stem.replace("overlay_", "").replace("_", "->"): p for p in overlays}
    choice = st.selectbox("Interval", list(labels))
    st.image(str(labels[choice]), caption=f"Forest loss {choice} (red)",
             width="stretch")
else:
    st.info("No loss-overlay maps found.")
