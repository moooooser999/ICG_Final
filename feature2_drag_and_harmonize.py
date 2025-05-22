import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import colorsys
import io
from scipy.optimize import minimize_scalar

st.set_page_config(layout="wide")
st.title("🖼 Feature 2: Drag-and-Drop Foreground Harmonization")


# --- Function: compute harmony score ---
def get_harmonic_templates(rotation: float = 0) -> dict:
    return {
        "X": [(rotation % 360, 93.6), ((rotation + 180) % 360, 93.6)],
        "I": [(rotation % 360, 18), ((rotation + 180) % 360, 18)],
        "Y": [(rotation % 360, 93.6), ((rotation + 180) % 360, 18)],
        "T": [(rotation % 360, 180)],
        "V": [(rotation % 360, 93.6)],
        "i": [(rotation % 360, 18)],
    }


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


def F_continuous_alpha(alpha_deg, hues, saturations, template_type="X"):
    sectors = get_harmonic_templates(rotation=alpha_deg)[template_type]
    return compute_harmony_score_with_saturation_fixed(hues, saturations, sectors)


def find_optimal_alpha(hues, saturations, template_type="X"):
    result = minimize_scalar(
        lambda alpha: F_continuous_alpha(alpha, hues, saturations, template_type),
        bounds=(0, 360),
        method="bounded",
    )
    return result.x


def extract_hue_sat(image: Image.Image):
    arr = np.array(image.convert("RGB"))
    hues, sats = [], []
    for r, g, b in arr.reshape(-1, 3):
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        hues.append(h * 360)
        sats.append(s)
    return np.array(hues), np.array(sats)


def harmonize_foreground(fg_img_np, alpha, template_type="X", sigma_ratio=0.5):
    h, w = fg_img_np.shape[:2]
    rgb = fg_img_np[:, :, :3] / 255.0
    mask = fg_img_np[:, :, 3] > 0
    hsv = np.zeros_like(rgb)
    for i in range(h):
        for j in range(w):
            hsv[i, j] = colorsys.rgb_to_hsv(*rgb[i, j])

    H = hsv[..., 0] * 360
    S = hsv[..., 1]
    V = hsv[..., 2]

    sectors = get_harmonic_templates(rotation=alpha)[template_type]
    C = np.array([c for c, _ in sectors])
    S_span = np.array([s for _, s in sectors])

    adjusted_H = H.copy()
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue
            h_ij = H[i, j]
            s_ij = S[i, j]
            min_d, best_c, best_w = 999, None, None
            for c, w_ in zip(C, S_span):
                d = min(abs(h_ij - c), 360 - abs(h_ij - c))
                if d < min_d:
                    min_d, best_c, best_w = d, c, w_
            sigma = best_w / 2 * sigma_ratio
            diff = (h_ij - best_c + 540) % 360 - 180
            g = np.exp(-0.5 * (diff / sigma) ** 2)
            adjusted_H[i, j] = (best_c + diff * (1 - g)) % 360

    new_rgb = np.zeros_like(rgb)
    for i in range(h):
        for j in range(w):
            new_rgb[i, j] = colorsys.hsv_to_rgb(
                adjusted_H[i, j] / 360, S[i, j], V[i, j]
            )

    out = (new_rgb * 255).astype(np.uint8)
    out_rgba = np.concatenate([out, fg_img_np[:, :, 3:4]], axis=-1)
    return Image.fromarray(out_rgba)


# Sidebar
st.sidebar.header("1️⃣ Upload")
bg_img_file = st.sidebar.file_uploader(
    "Upload Background Image", type=["jpg", "jpeg", "png"]
)
fg_img_file = st.sidebar.file_uploader(
    "Upload Foreground Image (with transparency)", type=["png"]
)

st.sidebar.header("2️⃣ Canvas Options")
canvas_width = st.sidebar.slider("Canvas width", 400, 1024, 800)
stroke_width = st.sidebar.slider("Bounding box thickness", 1, 5, 2)

st.sidebar.header("3️⃣ Harmonization Settings")
template_choice = st.sidebar.selectbox(
    "Template", list(get_harmonic_templates().keys()), index=0
)
alpha_source = st.sidebar.selectbox(
    "Compute α from", ["Background", "Foreground", "Composite"]
)
harmonize_with = st.sidebar.selectbox(
    "Harmonize based on", ["Background", "Foreground", "Composite"]
)
sigma_ratio = st.sidebar.slider("σ smoothing strength", 0.1, 1.0, 0.5)

# Load images
if bg_img_file:
    bg_image = Image.open(bg_img_file).convert("RGBA")
    bg_image = bg_image.resize(
        (canvas_width, int(bg_image.height * canvas_width / bg_image.width))
    )
else:
    st.warning("Please upload a background image to continue.")
    st.stop()

if fg_img_file:
    fg_image = Image.open(fg_img_file).convert("RGBA")
    fg_np = np.array(fg_image)
    init_left = int((bg_image.width - fg_image.width) / 2)
    init_top = int((bg_image.height - fg_image.height) / 2)
    init_objects = {
        "objects": [
            {
                "type": "rect",
                "left": init_left,
                "top": init_top,
                "width": fg_image.width,
                "height": fg_image.height,
                "fill": "rgba(255,255,255,0.3)",
                "stroke": "#888",
            }
        ]
    }
else:
    fg_image = None
    init_objects = {"objects": []}

st.header("4️⃣ Drag your object to position")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    background_image=bg_image,
    update_streamlit=True,
    initial_drawing=init_objects,
    height=bg_image.height,
    width=bg_image.width,
    drawing_mode="transform",
    key="canvas_transform",
)

if canvas_result.json_data is not None and fg_image is not None:
    objects = canvas_result.json_data["objects"]
    if objects:
        st.markdown("### 5️⃣ Harmonized vs Original Preview")
        preview = bg_image.copy()

        obj = objects[0]
        left = int(obj["left"])
        top = int(obj["top"])
        scale = obj.get("scaleX", 1.0)
        new_size = (int(fg_image.width * scale), int(fg_image.height * scale))
        fg_resized = fg_image.resize(new_size)

        # Compose preview without harmonization
        original_composite = preview.copy()
        original_composite.paste(fg_resized, (left, top), fg_resized)

        # Choose image for alpha computation
        if alpha_source == "Background":
            hues, sats = extract_hue_sat(bg_image)
        elif alpha_source == "Foreground":
            hues, sats = extract_hue_sat(fg_image)
        else:  # Composite
            composite = bg_image.copy()
            composite.paste(fg_resized, (left, top), fg_resized)
            hues, sats = extract_hue_sat(composite)

        alpha_value = find_optimal_alpha(hues, sats, template_type=template_choice)

        # Choose image for harmonization base
        if harmonize_with == "Background":
            base = bg_image
        elif harmonize_with == "Foreground":
            base = fg_image
        else:
            base = composite

        fg_harmonized = harmonize_foreground(
            fg_np, alpha_value, template_type=template_choice, sigma_ratio=sigma_ratio
        )
        fg_final = fg_harmonized.resize(new_size)
        harmonized_composite = bg_image.copy()
        harmonized_composite.paste(fg_final, (left, top), fg_final)

        # Show preview
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                original_composite.resize(
                    (canvas_width // 2, int(bg_image.height * 0.5))
                ),
                caption="Original Composite",
            )
        with col2:
            st.image(
                harmonized_composite.resize(
                    (canvas_width // 2, int(bg_image.height * 0.5))
                ),
                caption=f"Harmonized (α={alpha_value:.1f}°, σ={sigma_ratio})",
            )
    else:
        st.info("Please drag the object on canvas to position it.")
else:
    st.info("Upload a foreground image to start dragging.")
