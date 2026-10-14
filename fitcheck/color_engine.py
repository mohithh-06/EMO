"""
Color Extraction, Harmony Analysis, and Color Theory Engine for emo.
Extracts dominant palettes from outfit images, calculates color wheel relationships,
lightness contrast, and evaluates harmony scores dynamically.
"""

import math
from typing import List, Dict, Tuple, Any
from PIL import Image
import numpy as np


# Curated high-fashion color dictionary with reference RGBs
FASHION_COLOR_NAMES = {
    "Onyx Black": (18, 18, 20),
    "Jet Black": (28, 28, 30),
    "Charcoal Grey": (58, 60, 66),
    "Slate Grey": (108, 114, 125),
    "Heather Grey": (175, 178, 185),
    "Pure White": (250, 250, 252),
    "Warm Ivory": (245, 241, 230),
    "Oatmeal / Ecru": (222, 215, 198),
    "Camel / Tan": (193, 154, 107),
    "Espresso Brown": (60, 42, 33),
    "Cognac Leather": (154, 70, 32),
    "Terracotta / Rust": (190, 84, 52),
    "Crimson Red": (186, 24, 27),
    "Burgundy / Wine": (114, 24, 46),
    "Blush / Powder Pink": (235, 190, 198),
    "Dusty Rose": (196, 134, 146),
    "Cobalt / Royal Blue": (28, 70, 190),
    "Navy Blue": (18, 30, 68),
    "Midnight Blue": (12, 19, 44),
    "Sky / Baby Blue": (142, 190, 230),
    "Teal / Dark Cyan": (22, 115, 125),
    "Emerald Green": (24, 118, 74),
    "Forest Green": (26, 68, 44),
    "Sage Green": (138, 160, 140),
    "Olive Drab": (92, 98, 48),
    "Mustard / Ochre": (212, 160, 23),
    "Champagne Gold": (228, 205, 142),
    "Lilac / Lavender": (185, 165, 215),
    "Plum / Deep Violet": (88, 32, 94),
}


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to uppercase hex code."""
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex string to RGB tuple."""
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (H: 0-360, S: 0-1, V: 0-1)."""
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    c_max = max(r_n, g_n, b_n)
    c_min = min(r_n, g_n, b_n)
    delta = c_max - c_min

    if delta == 0:
        h = 0.0
    elif c_max == r_n:
        h = (60 * ((g_n - b_n) / delta) + 360) % 360
    elif c_max == g_n:
        h = (60 * ((b_n - r_n) / delta) + 120) % 360
    else:
        h = (60 * ((r_n - g_n) / delta) + 240) % 360

    s = 0.0 if c_max == 0 else (delta / c_max)
    v = c_max
    return h, s, v


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """Perceptually-weighted Euclidean distance in RGB space."""
    r_mean = (c1[0] + c2[0]) / 2.0
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return math.sqrt((2 + r_mean / 256.0) * (dr ** 2) + 4.0 * (dg ** 2) + (2 + (255 - r_mean) / 256.0) * (db ** 2))


def get_closest_fashion_color_name(rgb: Tuple[int, int, int]) -> str:
    """Map RGB values to the nearest high-fashion shade name."""
    closest_name = "Onyx Black"
    min_dist = float("inf")
    for name, ref_rgb in FASHION_COLOR_NAMES.items():
        dist = color_distance(rgb, ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """Calculate WCAG standard relative luminance (0.0 to 1.0)."""
    def adjust(c):
        c_n = c / 255.0
        return c_n / 12.92 if c_n <= 0.03928 else ((c_n + 0.055) / 1.055) ** 2.4
    return 0.2126 * adjust(rgb[0]) + 0.7152 * adjust(rgb[1]) + 0.0722 * adjust(rgb[2])


def extract_dominant_colors(image: Image.Image, n_colors: int = 5) -> List[Dict[str, Any]]:
    """
    Extract the top dominant colors from an outfit photo using adaptive spatial sampling
    and clustering. Prioritizes the clothing subject while avoiding repeated identical shade names.
    """
    img = image.convert("RGB")
    img_thumb = img.resize((150, 150), Image.Resampling.LANCZOS)
    arr = np.array(img_thumb)
    
    h, w, _ = arr.shape
    
    # 1. Estimate background color from upper corners
    top_corners = np.concatenate([
        arr[0:15, 0:20],
        arr[0:15, w-20:w]
    ], axis=0).reshape(-1, 3)
    bg_median = np.median(top_corners, axis=0)
    
    # 2. Focus on outfit region: central and lower vertical corridor
    torso_region = arr[int(h * 0.15):int(h * 0.95), int(w * 0.10):int(w * 0.90)]
    pixels_all = torso_region.reshape(-1, 3)
    
    # 3. Calculate distance from background wall color
    bg_diffs = np.linalg.norm(pixels_all - bg_median, axis=1)
    foreground_pixels = pixels_all[bg_diffs > 28]
    
    # Fallback to entire outfit region if background filtering leaves too few pixels
    if len(foreground_pixels) < 800:
        pixels_to_cluster = pixels_all
    else:
        pixels_to_cluster = foreground_pixels

    try:
        from sklearn.cluster import KMeans
        # Cluster with extra centers to allow deduplicating identical shade names
        cluster_k = min(n_colors + 3, len(pixels_to_cluster))
        kmeans = KMeans(n_clusters=cluster_k, n_init=3, max_iter=80, random_state=42)
        kmeans.fit(pixels_to_cluster)
        centers = kmeans.cluster_centers_
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
    except Exception:
        quantized = img_thumb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()[:n_colors * 3]
        centers = [palette[i:i+3] for i in range(0, len(palette), 3)]
        hist = quantized.histogram()[:n_colors]
        counts = np.array(hist)

    total_pixels = float(np.sum(counts))
    order = np.argsort(counts)[::-1]

    raw_palette = []
    seen_names = set()

    for idx in order:
        rgb_raw = centers[idx]
        rgb = (int(round(rgb_raw[0])), int(round(rgb_raw[1])), int(round(rgb_raw[2])))
        pct = round(float(counts[idx]) / total_pixels * 100.0, 1)
        hex_code = rgb_to_hex(*rgb)
        name = get_closest_fashion_color_name(rgb)
        h, s, v = rgb_to_hsv(*rgb)
        lum = round(relative_luminance(rgb), 3)

        # Merge or skip duplicate color names so list is diverse
        if name in seen_names and len(seen_names) < n_colors:
            # Accumulate percentage onto existing entry
            for existing in raw_palette:
                if existing["name"] == name:
                    existing["coverage_pct"] = round(existing["coverage_pct"] + pct, 1)
                    break
            continue

        seen_names.add(name)
        raw_palette.append({
            "rgb": rgb,
            "hex": hex_code,
            "name": name,
            "coverage_pct": pct,
            "hue": round(h, 1),
            "saturation": round(s, 2),
            "value": round(v, 2),
            "luminance": lum
        })

        if len(raw_palette) >= n_colors:
            break

    # Normalize coverage sum
    tot_cov = sum(c["coverage_pct"] for c in raw_palette) or 100.0
    for c in raw_palette:
        c["coverage_pct"] = round((c["coverage_pct"] / tot_cov) * 100.0, 1)

    return raw_palette


def evaluate_palette_harmony(palette: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze geometric relationships on the 360-degree color wheel, calculate
    contrast ratios, temperature balance, and output a dynamic Color Harmony Score (0-100).
    """
    if not palette:
        return {
            "harmony_type": "Neutral Monochrome",
            "harmony_score": 75,
            "contrast_ratio": 50,
            "contrast_grade": "Balanced",
            "temperature": "Neutral",
            "summary": "Clean, balanced neutral execution."
        }

    # Filter out near-zero saturation colors for hue wheel geometry
    saturated = [c for c in palette if c["saturation"] > 0.18 and 0.10 < c["value"] < 0.92]
    neutrals = [c for c in palette if c["saturation"] <= 0.18 or c["value"] <= 0.10 or c["value"] >= 0.92]
    
    # Calculate Lightness Contrast Ratio
    luminances = [c["luminance"] for c in palette]
    min_lum = min(luminances)
    max_lum = max(luminances)
    contrast_ratio = round((max_lum + 0.05) / (min_lum + 0.05), 2)
    contrast_score = min(100, int((contrast_ratio - 1.0) / 10.0 * 100.0))
    
    if contrast_ratio > 7.0:
        contrast_grade = "High Contrast (Crisp & Structured)"
    elif contrast_ratio >= 3.0:
        contrast_grade = "Medium Contrast (Soft & Harmonious)"
    else:
        contrast_grade = "Low Contrast (Tonal & Muted)"

    # Determine Temperature based on saturated and neutral tones
    warm_count = sum(1 for c in palette if (c["hue"] < 90 or c["hue"] > 300) and c["saturation"] > 0.15)
    cool_count = sum(1 for c in palette if (90 <= c["hue"] <= 300) and c["saturation"] > 0.15)
    
    if not saturated:
        temperature = "Neutral Equilibrium (Balanced Tonal Spectrum)"
    elif warm_count > cool_count:
        temperature = "Warm Paced (Golden, Terracotta, Earth Tones)"
    elif cool_count > warm_count:
        temperature = "Cool Paced (Navy, Slate, Icy Tones)"
    else:
        temperature = "Neutral Equilibrium (Balanced Tonal Spectrum)"

    # Classify Color Wheel Harmony Archetype
    if not saturated:
        harmony_type = "Monochromatic Neutrals (Onyx, Slate & Ivory)"
        harmony_score = 86
        summary = "Refined monochromatic neutral discipline with high tonal versatility."
    elif len(saturated) == 1:
        if neutrals:
            harmony_type = "Neutral Anchor with Saturated Accent"
            harmony_score = 88
            summary = f"Commanding neutral base featuring a deliberate {saturated[0]['name']} focal pop."
        else:
            harmony_type = "Saturated Monotone"
            harmony_score = 78
            summary = "Direct single-hue saturation without contrasting anchor."
    else:
        # Check angle differences between dominant saturated colors
        hues = sorted([c["hue"] for c in saturated])
        diffs = [(hues[i+1] - hues[i]) for i in range(len(hues)-1)]
        diffs.append(360 - (hues[-1] - hues[0]))
        max_diff = max(diffs)

        if all(d < 45 for d in diffs[:-1]):
            harmony_type = "Analogous Pacing (Adjacent Hue Spectrum)"
            harmony_score = 84
            summary = "Harmonious fluid gradient across neighboring color wheel segments."
        elif any(150 <= d <= 210 for d in diffs):
            harmony_type = "Complementary Tension (Direct Color Wheel Polar)"
            harmony_score = 89
            summary = "High-energy dynamic tension between opposite poles on the color wheel."
        elif any(100 <= d <= 140 for d in diffs):
            harmony_type = "Triadic Energy (Three-Point Geometry)"
            harmony_score = 82
            summary = "Rich triad balance providing multi-dimensional visual interest."
        elif max_diff > 250:
            harmony_type = "Split Tonal Accent"
            harmony_score = 80
            summary = "Accented tonal divergence with controlled chromatic spread."
        else:
            harmony_type = "Complex Chromatic Layering"
            harmony_score = 74
            summary = "Eclectic color layering; requires strict texture control to avoid visual clutter."

    # Bonus points for healthy contrast
    if 3.0 <= contrast_ratio <= 12.0:
        harmony_score = min(98, harmony_score + 4)
    elif contrast_ratio < 1.8:
        harmony_score = max(52, harmony_score - 10)

    return {
        "harmony_type": harmony_type,
        "harmony_score": int(harmony_score),
        "contrast_ratio": contrast_ratio,
        "contrast_grade": contrast_grade,
        "temperature": temperature,
        "summary": summary
    }
