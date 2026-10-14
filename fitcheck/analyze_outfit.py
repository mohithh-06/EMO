"""
Orchestration Engine for Outfit Analysis in emo.
Integrates Color Theory, Deep Vision (CLIP), Spatial Segmentation, and High-Fashion Editorial Critique.
Provides both structured dict output and formatted critique markdown for seamless UI and API consumption.
"""

import os
from typing import Dict, Any, Optional
from PIL import Image

try:
    from color_engine import extract_dominant_colors, evaluate_palette_harmony
    from vision_engine import analyze_outfit_deep_vision
    from critique_generator import generate_full_critique_package
except ImportError:
    from fitcheck.color_engine import extract_dominant_colors, evaluate_palette_harmony
    from fitcheck.vision_engine import analyze_outfit_deep_vision
    from fitcheck.critique_generator import generate_full_critique_package


def analyze_outfit_pipeline(image_input) -> Dict[str, Any]:
    """
    Run end-to-end dynamic analysis on an image (filepath or PIL.Image).
    Returns complete structured payload: palette, harmony, vision detections, scores, critique, and pivots.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Outfit image not found: {image_input}")
        image = Image.open(image_input).convert("RGB")
        image_path = image_input
    else:
        image = image_input.convert("RGB")
        image_path = getattr(image_input, "filename", "uploaded_outfit.jpg")

    # Step 1: Color Quantization & Color Wheel Harmony
    palette = extract_dominant_colors(image, n_colors=5)
    harmony = evaluate_palette_harmony(palette)

    # Step 2: Deep Vision Classification (Gender, Garment, Silhouette, Formality, Texture, Scorecard)
    vision_results = analyze_outfit_deep_vision(image, harmony)

    # Step 3: High-Fashion Editorial Critique & Dynamic Styling Pivots
    critique_pkg = generate_full_critique_package(vision_results, palette, harmony)

    # Step 4: Build Formatted Text Critique (matching standard format)
    formatted_markdown = (
        f"### {critique_pkg['headline']}\n\n"
        f"**Style:**\n{critique_pkg['breakdown']}\n\n"
        f"**Rating:** {critique_pkg['overall_score']}/100 — *{critique_pkg['tier_title']}*\n\n"
        f"**Comment:**\n\"{critique_pkg['savage_verdict']}\"\n\n"
    )

    return {
        "image_path": image_path,
        "palette": palette,
        "harmony": harmony,
        "vision": vision_results,
        "critique": critique_pkg,
        "formatted_text": formatted_markdown,
        "rating_score": critique_pkg["overall_score"] / 100.0,
        "overall_score": critique_pkg["overall_score"],
        "tier_title": critique_pkg["tier_title"],
        "tier_desc": critique_pkg["tier_desc"],
        "badge_color": critique_pkg["badge_color"],
        "metrics": critique_pkg["metrics"],
        "pivots": critique_pkg["pivots"]
    }


def analyze_outfit(image_input) -> str:
    """Returns the formatted markdown string analysis for backwards compatibility."""
    res = analyze_outfit_pipeline(image_input)
    return res["formatted_text"]


def extract_rating(analysis_payload) -> float:
    """Extract rating as float 0.0-1.0 from dict or text."""
    if isinstance(analysis_payload, dict):
        return analysis_payload.get("rating_score", 0.75)
    import re
    match = re.search(r'Rating:\s*(\d+)/100', str(analysis_payload))
    if match:
        return int(match.group(1)) / 100.0
    return 0.75


def extract_style_paragraph(analysis_payload) -> str:
    """Extract style breakdown from dict or text."""
    if isinstance(analysis_payload, dict):
        return analysis_payload.get("critique", {}).get("breakdown", "")
    import re
    match = re.search(r'\*\*Style:\*\*\s*(.*?)(?=\s*(\*\*Rating:|\*\*Comment:|$))', str(analysis_payload), re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(analysis_payload)


def extract_comment(analysis_payload) -> str:
    """Extract critique verdict from dict or text."""
    if isinstance(analysis_payload, dict):
        return analysis_payload.get("critique", {}).get("savage_verdict", "")
    import re
    match = re.search(r'\*\*Comment:\*\*\s*(.*)', str(analysis_payload), re.DOTALL)
    if match:
        return match.group(1).strip().strip('"')
    return ""


class OutfitAnalyzerTool:
    """LangChain / Tool compatible wrapper."""
    def invoke(self, image_path: str):
        return analyze_outfit_pipeline(image_path)


analyze_outfit_tool = OutfitAnalyzerTool()
