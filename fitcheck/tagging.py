"""
Wardrobe Tagging & Attribute Extraction for emo.
Uses CLIP zero-shot classification combined with K-Means color analysis
to generate comprehensive, accurate metadata for digitized wardrobe items.
"""

import os
import json
from typing import Dict, Any, List
from PIL import Image
import imagehash

try:
    from color_engine import extract_dominant_colors
    from vision_engine import zero_shot_classify, GENDER_PROMPTS, FORMALITY_PROMPTS, TEXTURE_PROMPTS
except ImportError:
    from fitcheck.color_engine import extract_dominant_colors
    from fitcheck.vision_engine import zero_shot_classify, GENDER_PROMPTS, FORMALITY_PROMPTS, TEXTURE_PROMPTS


CLOSET_ITEM_PROMPTS = [
    ("Shirts", "a collared button-down shirt, oxford shirt, or formal dress shirt"),
    ("T-Shirts", "a casual cotton t-shirt or basic tee"),
    ("Jackets", "a jacket, blazer, coat, or outwear layer"),
    ("Pants", "a pair of pants, trousers, or slacks"),
    ("Shorts", "a pair of shorts"),
    ("Skirts", "a fashionable skirt"),
    ("Dresses", "a dress, gown, or one-piece outfit"),
    ("Shoes", "a pair of shoes, sneakers, boots, or heels"),
    ("Bags", "a handbag, purse, backpack, or tote bag"),
    ("Sunglasses", "a pair of sunglasses or eyewear"),
    ("Hats", "a hat, cap, or headwear")
]

LOCATION_PROMPTS = [
    ("Indoor", "indoor wardrobe clothing, home wear, or office interior wear"),
    ("Outdoor", "outdoor outerwear, walking shoes, sun protection, or street wear")
]


def classify(image: Image.Image, labels: List[str]) -> str:
    """
    Backwards-compatible classification helper.
    """
    candidates = [(lbl, f"a photo of {lbl}") for lbl in labels]
    res = zero_shot_classify(image, candidates, top_k=1)
    return res[0]["label"] if res else labels[0]


def tag_closet_item(image_path: str) -> Dict[str, Any]:
    """
    Tag a clothing item image with:
    - Item taxonomy
    - Precise color name and hex code
    - Dominant color palette (top 3)
    - Indoor / Outdoor suitability
    - Formality level
    - Gender style (Men's / Women's / Unisex)
    - Fabric texture
    - Perceptual hash for deduplication
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        phash = str(imagehash.phash(image))

        # 1. Color Extraction
        palette = extract_dominant_colors(image, n_colors=3)
        primary_color_name = palette[0]["name"] if palette else "Unknown"
        primary_color_hex = palette[0]["hex"] if palette else "#000000"

        # 2. Attribute Extraction via Zero-shot Prompts
        item_res = zero_shot_classify(image, CLOSET_ITEM_PROMPTS, top_k=1)
        item_type = item_res[0]["label"] if item_res else "Shirts"

        gender_res = zero_shot_classify(image, GENDER_PROMPTS, top_k=1)
        gender_raw = gender_res[0]["label"] if gender_res else "Unisex"
        if "Womens" in gender_raw or "Women" in gender_raw:
            gender = "Women's"
        elif "Mens" in gender_raw or "Men" in gender_raw:
            gender = "Men's"
        else:
            gender = "Unisex"

        formality_res = zero_shot_classify(image, FORMALITY_PROMPTS, top_k=1)
        formality_raw = formality_res[0]["label"] if formality_res else "Casual"
        if "Formal" in formality_raw:
            formality = "Formal"
        elif "Smart Casual" in formality_raw:
            formality = "Smart Casual"
        else:
            formality = "Casual"

        texture_res = zero_shot_classify(image, TEXTURE_PROMPTS, top_k=1)
        texture = texture_res[0]["label"] if texture_res else "Crisp Woven Cotton"

        # Footwear and sunglasses are typically outdoor
        if item_type in ["Shoes", "Sunglasses", "Hats"]:
            indoor_outdoor = "Outdoor"
        else:
            loc_res = zero_shot_classify(image, LOCATION_PROMPTS, top_k=1)
            indoor_outdoor = loc_res[0]["label"] if loc_res else "Indoor"

        # Relative clean path
        rel_path = os.path.relpath(image_path, start=os.getcwd()) if os.path.isabs(image_path) else image_path

        return {
            "image_id": phash,
            "name": f"{primary_color_name} {item_type}",
            "item_type": item_type,
            "color": primary_color_name,
            "color_hex": primary_color_hex,
            "palette": [{"hex": c["hex"], "name": c["name"], "coverage": c["coverage_pct"]} for c in palette],
            "indoor_outdoor": indoor_outdoor,
            "formality": formality,
            "gender": gender,
            "texture": texture,
            "path": rel_path,
            "folder": "Closet"
        }

    except Exception as e:
        return {"error": str(e)}
