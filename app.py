import streamlit as st
from PIL import Image
import numpy as np
import colorsys
import io
import cv2
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")


# --- Template 定義函數 ---
def get_harmonic_templates(rotation: float = 0) -> dict:
    return {
        "I": [(rotation % 360, 18), ((rotation + 180) % 360, 18)],
        "T": [(rotation % 360, 180)],
        "V": [(rotation % 360, 93.6)],
        "i": [(rotation % 360, 18)],
        "Y": [(rotation % 360, 93.6), ((rotation + 180) % 360, 18)],
        "X": [(rotation % 360, 93.6), ((rotation + 180) % 360, 93.6)],
        "L": [(rotation % 360, 18), ((rotation + 90) % 360, 79.2)],
        "mirror-L": [((rotation + 90) % 360, 18), (rotation % 360, 79.2)],
    }


# --- hue 環狀距離 ---
def hue_distance_array(hues, centers):
    diff = np.abs(hues[..., None] - centers)
    return np.minimum(diff, 360 - diff)


# --- Hue Histogram 萃取 ---
def extract_hue_histogram_from_image(image: Image.Image, num_bins: int = 180):
    arr = np.array(image.convert("RGB"))
    h_vals = []
    for r, g, b in arr.reshape(-1, 3):
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h_vals.append(h * 360)
    h_vals = np.array(h_vals)
    hist, bin_edges = np.histogram(h_vals, bins=num_bins, range=(0, 360))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return hist, bin_centers, hist.max()


# --- Plotly hue ring ---
def draw_hue_histogram_plotly(
    image, template_sectors, title="Hue Histogram", num_bins=180
):
    hist, bin_centers, max_count = extract_hue_histogram_from_image(image, num_bins)
    theta = bin_centers
    base_radius = 1.0
    r = base_radius - (hist / max_count * 0.6)
    r_end = np.full_like(r, base_radius)

    fig = go.Figure()
    for t, r_start, r_top, color in zip(theta, r, r_end, bin_centers):
        fig.add_trace(
            go.Barpolar(
                r=[r_top - r_start],
                base=r_start,
                theta=[t],
                width=[360 / num_bins],
                marker_color=f"hsl({int(color)}, 100%, 50%)",
                marker_line_color="white",
                marker_line_width=0,
                opacity=0.9,
            )
        )

    for center_deg, span_deg in template_sectors:
        start = center_deg - span_deg / 2
        end = center_deg + span_deg / 2
        arc = np.linspace(start, end, 100)
        r_arc = np.full_like(arc, base_radius)
        fig.add_trace(
            go.Scatterpolar(
                r=np.concatenate(([0], r_arc, [0])),
                theta=np.concatenate(([start], arc, [end])),
                mode="lines",
                fill="toself",
                fillcolor="rgba(200, 200, 200, 0.3)",
                line=dict(color="lightgray", width=1),
                hoverinfo="none",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        width=700,
        height=700,
        polar=dict(
            bgcolor="black",
            angularaxis=dict(showticklabels=False, ticks=""),
            radialaxis=dict(visible=False, range=[0, base_radius + 0.1]),
        ),
        showlegend=False,
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# --- Harmonization 向量化版 ---
def naive_hue_shifting_vectorized(image, sector, sigma_scale=0.5):
    rgb = np.array(image.convert("RGB")) / 255.0
    hsv = np.zeros_like(rgb)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            hsv[i, j] = colorsys.rgb_to_hsv(*rgb[i, j])

    h = hsv[..., 0] * 360
    s = hsv[..., 1]
    v = hsv[..., 2]

    sector_centers = np.array([c for c, _ in sector])
    sector_spans = np.array([s for _, s in sector])

    dists = hue_distance_array(h, sector_centers)
    min_indices = np.argmin(dists, axis=-1)
    nearest_centers = sector_centers[min_indices]
    nearest_spans = sector_spans[min_indices]

    diff = (h - nearest_centers + 540) % 360 - 180
    sigma = (nearest_spans / 2) * sigma_scale
    gaussian = np.exp(-0.5 * (diff / sigma) ** 2)
    adjusted_h = (nearest_centers + diff * (1 - gaussian)) % 360

    hsv_cv = np.stack(
        [
            (adjusted_h / 2).astype(np.uint8),
            (s * 255).astype(np.uint8),
            (v * 255).astype(np.uint8),
        ],
        axis=-1,
    )
    bgr = cv2.cvtColor(hsv_cv, cv2.COLOR_HSV2BGR)
    rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


# --- Hue + Saturation 萃取 ---
def extract_hue_saturation(image: Image.Image):
    rgb = np.array(image.convert("RGB")) / 255.0
    hsv = np.zeros_like(rgb)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            hsv[i, j] = colorsys.rgb_to_hsv(*rgb[i, j])
    h = hsv[..., 0].flatten() * 360
    s = hsv[..., 1].flatten()
    return h, s


# --- F 值計算 ---
def compute_harmony_score_with_saturation_fixed(hues, saturations, template_sectors):
    def hue_distance(h1, h2):
        d = np.abs(h1 - h2)
        return np.minimum(d, 360 - d)

    N = len(hues)
    sector_centers = np.array([c for c, _ in template_sectors])
    sector_spans = np.array([s for _, s in template_sectors])

    h_matrix = np.tile(hues[:, None], (1, len(sector_centers)))
    c_matrix = np.tile(sector_centers[None, :], (N, 1))
    s_matrix = np.tile(sector_spans[None, :], (N, 1))

    d_matrix = hue_distance(h_matrix, c_matrix)
    inside = d_matrix <= (s_matrix / 2)
    outside_dist = np.where(
        inside.any(axis=1), 0, (d_matrix - s_matrix / 2).clip(min=0).min(axis=1)
    )
    weighted_dist = outside_dist * saturations
    return weighted_dist.mean()


# --- 最佳 α 搜尋 ---
def F_continuous_alpha(alpha_deg, hues, saturations, template_type):
    sectors = get_harmonic_templates(rotation=alpha_deg % 360)[template_type]
    return compute_harmony_score_with_saturation_fixed(hues, saturations, sectors)


def find_optimal_alpha_minimize_scalar(hues, saturations, template_type):
    result = minimize_scalar(
        lambda alpha: F_continuous_alpha(alpha, hues, saturations, template_type),
        bounds=(0, 360),
        method="bounded",
    )
    return result.x, result.fun


# --- Streamlit 介面 ---
st.title("🎨 Color Harmonization Preview Tool (Plotly Hue Ring)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    h_flat, s_flat = extract_hue_saturation(image)

    with st.sidebar:
        st.markdown("## Harmonization Settings")
        template = st.selectbox(
            "Select Harmonic Template", list(get_harmonic_templates().keys()), index=5
        )
        best_alpha, best_score = find_optimal_alpha_minimize_scalar(
            h_flat, s_flat, template
        )
        st.markdown(
            f"**Recommended α:** {best_alpha:.2f}°\n\n**F-score:** {best_score:.4f}"
        )
        alpha = st.slider(
            "Template Rotation α (degrees)",
            min_value=0,
            max_value=360,
            value=int(best_alpha),
            step=1,
        )
        sigma_scale = st.slider(
            "Shifting Smoothness (σ scale)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Harmonized ({template} @ {alpha}°)")
        sector = get_harmonic_templates(rotation=alpha)[template]
        harmonized = naive_hue_shifting_vectorized(
            image, sector, sigma_scale=sigma_scale
        )
        st.image(harmonized, use_container_width=True)

        buf = io.BytesIO()
        harmonized.save(buf, format="PNG")
        st.download_button(
            "Download Harmonized Image", buf.getvalue(), file_name="harmonized.png"
        )

    with st.expander("🔍 Hue Histogram Preview"):
        st.plotly_chart(
            draw_hue_histogram_plotly(
                image, sector, title=f"Original Hue Histogram + {template} @ {alpha}°"
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            draw_hue_histogram_plotly(
                harmonized,
                sector,
                title=f"Harmonized Hue Histogram + {template} @ {alpha}°",
            ),
            use_container_width=True,
        )
