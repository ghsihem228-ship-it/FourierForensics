"""Run with: streamlit run app.py."""

import json
from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from fourierforensics.detectors.jpeg import ela
from fourierforensics.io import load_image
from fourierforensics.scoring import analyze
from fourierforensics.synthetic import variants

st.set_page_config(page_title="FourierForensics", page_icon="◈", layout="wide")
st.title("FourierForensics")
st.caption("Explore the mathematical traces of image processing")
st.info(
    "Scores describe processing artifacts, not the probability of forgery. Benign edits can raise scores; low scores do not prove authenticity."
)
with st.sidebar:
    st.header("Image laboratory")
    upload = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"]
    )
    sample = st.selectbox("Or explore a generated example", list(variants().keys()), index=4)
    st.caption(
        "Generated examples use seed 42. Analyze your original file to preserve JPEG evidence."
    )
    quality = st.slider("ELA recompression quality", 50, 100, 90)
    st.caption("ELA is a diagnostic visualization and does not contribute to the score.")


@st.cache_data(show_spinner=False, max_entries=4)
def run(data: bytes):
    loaded = load_image(data)
    return loaded, analyze(loaded.pixels, loaded.source_bytes)


if upload is not None:
    data = upload.getvalue()
    name = upload.name
else:
    pixels, data = variants(42)[sample]
    if data is None:
        stream = BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        data = stream.getvalue()
    name = f"Generated {sample}"
try:
    with st.spinner("Computing frequency and noise clues…"):
        loaded, result = run(data)
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()
for warning in loaded.warnings:
    st.warning(warning)
st.subheader(name)
a, b, c = st.columns(3)
a.metric("Artifact suspicion", f"{result.score:.2f} / 1.00")
b.metric("Threshold", "0.50 · uncalibrated")
c.metric("Analyzed pixels", f"{loaded.pixels.shape[1]} × {loaded.pixels.shape[0]}")
views, diagnostics = st.tabs(["Image & evidence", "Diagnostics & mathematics"])
with views:
    cols = st.columns(3)
    cols[0].image(loaded.pixels, caption="Original decoded pixels", width="stretch")
    for column, values, title, cmap, limits in [
        (cols[1], result.spectrum, "Centered log-magnitude spectrum", "magma", {}),
        (cols[2], result.heatmap, "Block noise anomaly map", "inferno", {"vmin": 0, "vmax": 1}),
    ]:
        fig, ax = plt.subplots(figsize=(4, 4))
        chart = ax.imshow(values, cmap=cmap, **limits)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(chart, ax=ax, shrink=0.7)
        column.pyplot(fig)
        plt.close(fig)
    for detector in result.detectors:
        st.write(
            f"**{detector.name.title()}** — "
            + (f"{detector.score:.2f}" if detector.available else "Unavailable")
        )
        st.caption(detector.explanation)
with diagnostics:
    left, right = st.columns(2)
    fig, ax = plt.subplots()
    ax.semilogy(result.radial_frequency, result.radial_power)
    ax.set(
        xlabel="Radial frequency (cycles/pixel)",
        ylabel="Mean power",
        title="Hann-windowed radial power",
    )
    left.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots()
    chart = ax.imshow(ela(loaded.pixels, quality), cmap="magma", vmin=0, vmax=0.08)
    ax.set_title(f"ELA at quality {quality} · fixed error scale")
    ax.axis("off")
    fig.colorbar(chart, ax=ax)
    right.pyplot(fig)
    plt.close(fig)
    st.write("Peak frequency (vertical, horizontal):", result.detectors[0].diagnostics["frequency"])
    st.latex(r"S=0.75\max_i s_i+0.25\frac{\sum_i w_i s_i}{\sum_i w_i}")
    st.caption(
        "Available detectors only. Weights: resampling 0.40, noise 0.35, JPEG 0.25. Heatmap resolution is one block, not one pixel."
    )
summary = result.summary() | {"filename": name, "warnings": loaded.warnings}
st.download_button(
    "Download analysis JSON",
    json.dumps(summary, indent=2),
    file_name="analysis.json",
    mime="application/json",
)
