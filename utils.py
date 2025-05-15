from typing import List, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from scipy.optimize import minimize_scalar


def convert2HSV(image):
    h_list = []
    s_list = []
    v_list = []
    arr = np.array(image.convert("RGB"))
    for r, g, b in arr.reshape(-1, 3):
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h_list.append(h * 360)
        s_list.append(s)
        v_list.append(v)

    return h_list, s_list, v_list


def extract_hue_histogram_from_image(
    image: Image.Image, num_bins: int = 180
) -> Tuple[np.ndarray, np.ndarray, float]:
    arr = np.array(image.convert("RGB"))
    h_vals = []
    for r, g, b in arr.reshape(-1, 3):
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h_vals.append(h * 360)
    h_vals = np.array(h_vals)
    hist, bin_edges = np.histogram(h_vals, bins=num_bins, range=(0, 360))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_count = hist.max()
    return hist, bin_centers, max_count


def draw_hue_histogram_with_template(
    image: Image.Image,
    template_sectors: List[Tuple[float, float]],
    title="Hue Histogram + Harmonic Template",
    num_bins=180,
):
    hist, bin_centers, max_count = extract_hue_histogram_from_image(image, num_bins)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    width = 2 * np.pi / num_bins
    inner_radius = 1.0
    ring_thickness = 0.06
    outer_scale = 0.9

    # Color ring
    for center_deg in bin_centers:
        theta = np.deg2rad(center_deg)
        rgb = colorsys.hsv_to_rgb(center_deg / 360.0, 1.0, 1.0)
        ax.bar(
            theta,
            ring_thickness,
            width=width,
            bottom=inner_radius - ring_thickness / 2,
            color=rgb,
            edgecolor=None,
        )

    # Histogram bars
    for count, center_deg in zip(hist, bin_centers):
        theta = np.deg2rad(center_deg)
        height = (count / max_count) * outer_scale * inner_radius
        bottom = inner_radius - height
        rgb = colorsys.hsv_to_rgb(center_deg / 360.0, 1.0, 1.0)
        ax.bar(theta, height, width=width, bottom=bottom, color=rgb, edgecolor=None)

    # Draw radial sectors
    for center_deg, span_deg in template_sectors:
        start_deg = center_deg - span_deg / 2
        end_deg = center_deg + span_deg / 2
        arc = np.deg2rad(np.linspace(start_deg, end_deg, 120))
        for a in arc:
            ax.plot(
                [a, a],
                [0, inner_radius + ring_thickness / 2],
                color="gray",
                alpha=0.3,
                linewidth=1.0,
            )

    ax.set_ylim(0, inner_radius + ring_thickness / 2)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.show()


def get_harmonic_template_by_type(rotation: float = 0, type_: str = "X") -> dict:
    if type_ == "X":
        return "X", [(rotation % 360, 93.6), ((rotation + 180) % 360, 93.6)]
    elif type_ == "Y":
        return "Y", [(rotation % 360, 93.6), ((rotation + 180) % 360, 18)]
    elif type_ == "I":
        return "I", [(rotation % 360, 18), ((rotation + 180) % 360, 18)]
    elif type_ == "T":
        return "T", [(rotation % 360, 180)]
    elif type_ == "V":
        return "V", [(rotation % 360, 93.6)]
    elif type_ == "i":
        return "i", [(rotation % 360, 18)]
    elif type_ == "L":
        return "L", [(rotation % 360, 18), ((rotation + 90) % 360, 79.2)]
    elif type_ == "mirror-L":
        return "mirror-L", [((rotation + 90) % 360, 18), (rotation % 360, 79.2)]
    else:
        raise ValueError(f"Invalid template type: {type_}")


# 所有 harmonic templates 的定義（角度分布）
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


# 批次畫出所有模板對應圖
def plot_all_templates_for_image(image: Image.Image, rotation: float = 30):
    templates = get_harmonic_templates(rotation)
    for name, sectors in templates.items():
        draw_hue_histogram_with_template(
            image, sectors, title=f"Hue Histogram + {name}-type Template"
        )


def compute_harmony_score_with_saturation_fixed(
    hues: np.ndarray,
    saturations: np.ndarray,
    template_sectors: List[Tuple[float, float]],
) -> float:
    """
    修正後版本：使用正確 broadcasting 計算 hue 與 sector 的最短距離，乘上 saturation 權重
    """

    def hue_distance(h1, h2):
        d = np.abs(h1 - h2)
        return np.minimum(d, 360 - d)

    N = len(hues)
    sector_centers = np.array([c for c, _ in template_sectors])
    sector_spans = np.array([s for _, s in template_sectors])

    # shape (N, M)
    h_matrix = np.tile(hues[:, None], (1, len(sector_centers)))
    c_matrix = np.tile(sector_centers[None, :], (N, 1))
    s_matrix = np.tile(sector_spans[None, :], (N, 1))

    d_matrix = hue_distance(h_matrix, c_matrix)  # 弧長距離
    inside = d_matrix <= (s_matrix / 2)
    outside_dist = np.where(
        inside.any(axis=1), 0, (d_matrix - s_matrix / 2).clip(min=0).min(axis=1)
    )

    weighted_dist = outside_dist * saturations
    return weighted_dist.mean()


def F_continuous_alpha(
    alpha_deg: float, hues: np.ndarray, saturations: np.ndarray, template_type: str
) -> float:
    """
    包裝版本的 F(X, (m, α))，接受連續 alpha（度數），用於 minimize_scalar
    """
    sectors = get_harmonic_templates(rotation=alpha_deg % 360)[template_type]
    return compute_harmony_score_with_saturation_fixed(hues, saturations, sectors)


def find_optimal_alpha_minimize_scalar(
    hues: np.ndarray, saturations: np.ndarray, template_type: str
):
    """
    用 scipy.optimize.minimize_scalar 找出使 F 最小的 α（連續版本）
    """
    result = minimize_scalar(
        lambda alpha: F_continuous_alpha(alpha, hues, saturations, template_type),
        bounds=(0, 360),
        method="bounded",
    )
    return result.x, result.fun  # 最佳 α, 對應的 F 值


def find_optimal_alpha_minimize_scalar_grid(
    hues: np.ndarray, saturations: np.ndarray, template_type: str
):
    """
    用 scipy.optimize.minimize_scalar 找出使 F 最小的 α（連續版本）
    """
    min_f = 10000
    for alpha in range(0, 360):
        f = F_continuous_alpha(alpha, hues, saturations, template_type)
        if f < min_f:
            min_f = f
            best_alpha = alpha
    return best_alpha, min_f


def hue_distance(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 360 - d)


def naive_hue_shifting(image, sector):
    h, s, v = convert2HSV(image)
    adjusted_hues = np.zeros(len(h))
    for i in range(len(h)):
        min_dis = 360
        nearest_center = None
        nearest_span = None
        for se in sector:
            center, span = se
            dis = hue_distance(h[i], center)
            if dis < min_dis:
                min_dis = dis
                nearest_center = center
                nearest_span = span

        diff = (h[i] - nearest_center + 540) % 360 - 180
        sigma = nearest_span / 2
        gaussian_factor = np.exp(-0.5 * (diff / sigma) ** 2)
        adjusted_hues[i] = nearest_center + sigma * (1 - gaussian_factor)

    # return back to image
    new_rgb = []
    for h, s, v in zip(adjusted_hues, s, v):
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
        new_rgb.append((int(r * 255), int(g * 255), int(b * 255)))
    new_rgb = np.array(new_rgb).reshape(image.size[1], image.size[0], 3)
    output_image = Image.fromarray(new_rgb.astype(np.uint8))
    return output_image
