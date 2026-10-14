"""
Dynamic High-Fashion Editorial Critique & Styling Pivot Engine for emo.
Synthesizes deep vision detections (gender presentation, garment taxonomy, silhouette drape)
and color wheel harmonics to produce authentic, non-static styling critique and actionable upgrades.
"""

from typing import Dict, List, Any


def generate_editorial_headline(gender: str, garment: str, harmony_type: str, overall_score: int) -> str:
    """Generate dynamic high-fashion magazine headline tailored to detections."""
    is_casual = any(k in garment for k in ["T-Shirt", "Top", "Hoodie", "Sweatshirt", "Shorts", "Jeans"])
    if overall_score >= 88:
        if "Ethnic" in gender or "Saree" in garment:
            return "Ceremonial Radiance: Sovereign Heritage Drape with Flawless Chromatic Discipline"
        elif is_casual:
            return "Minimalist Precision: Elevated Everyday Casual with Impeccable Tonal Equilibrium"
        elif "Womenswear" in gender:
            return "Architectural Panache: Commanding Feminine Contours with Impeccable Pacing"
        elif "Menswear" in gender:
            return "Bespoke Precision: High-Authority Masculine Form with Razor-Sharp Lines"
        else:
            return "Sartorial Mastery: Flawlessly Balanced Proportions and Curated Minimalist Tension"
    elif overall_score >= 76:
        if "Ethnic" in gender or "Saree" in garment:
            return "Poised Heritage Silhouette: Expressive Texture Compromised by Pacing Gaps"
        elif is_casual:
            return f"Clean Casual Foundation: Relaxed {garment} Seeking Elevated Focal Accent"
        elif "Womenswear" in gender:
            return "Modern Feminine Poise: Elevated Visual Flow Seeking Sharper Accent Polish"
        elif "Menswear" in gender:
            return "Sharp Contemporary Form: Sound Structural Base Begging for Elevated Hardware"
        else:
            return "Calculated Urban Balance: Cohesive Baseline Ready for Intentional Layering"
    elif overall_score >= 60:
        return f"Casual Foundation with Loose Cohesion: {garment} Undermined by Flat Contrast"
    else:
        return f"Sartorial Discord: Clashing Value Weight and Uncalibrated Proportions in {garment}"


def generate_savage_verdict(gender: str, garment: str, overall_score: int, contrast_grade: str) -> str:
    """Generate a sharp, witty, authoritative stylist verdict matching the outfit score."""
    if overall_score >= 88:
        return (
            "I came ready to dismantle this, but the execution refuses to buckle. "
            f"The way this {garment.lower()} commands the frame while respecting proportional equilibrium "
            "is undeniable. Reluctant applause: this is how personal style commands a room without screaming for validation."
        )
    elif overall_score >= 76:
        return (
            f"There is genuine intelligence behind this fit—the {garment.lower()} has good structural presence. "
            f"However, settling for {contrast_grade.lower()} prevents it from achieving elite editorial status. "
            "It looks styled for a good afternoon; make it look designed for posterity."
        )
    elif overall_score >= 60:
        return (
            f"You put on clothes, but you didn't execute an outfit. A {garment.lower()} without intentional contrast "
            "reads as autopilot dressing. It’s not an aesthetic offense, but it is dangerously forgettable."
        )
    else:
        return (
            "This look is having an identity crisis in public. The proportions are wrestling each other for supremacy, "
            f"and the {garment.lower()} is bearing the brunt of the chaos. Strip back the clutter and rebuild from one clean anchor."
        )


def generate_styling_pivots(
    gender: str,
    garment: str,
    silhouette: str,
    harmony: Dict[str, Any],
    dominant_palette: List[Dict[str, Any]],
    overall_score: int
) -> List[Dict[str, Any]]:
    """
    Generate 3 distinct, evidence-based styling pivots:
    1. Chromatic Anchor / Contrast Pivot
    2. Proportional Architecture / Cut Pivot
    3. Hardware, Footwear & Layering Pivot
    """
    lead_color = dominant_palette[0]["name"] if dominant_palette else "Onyx Black"
    lead_hex = dominant_palette[0]["hex"] if dominant_palette else "#121214"
    second_hex = dominant_palette[1]["hex"] if len(dominant_palette) > 1 else "#E8E4D9"
    contrast_ratio = harmony.get("contrast_ratio", 4.0)

    pivots = []

    # Pivot 1: Color Shift / Accent Pacing
    if contrast_ratio < 2.5:
        # Low contrast - suggest high contrast neutral or metallic anchor
        pivot_1 = {
            "title": "Introduce High-Value Contrast Anchor",
            "category": "Chromatic Architecture",
            "flaw": f"The tonal spread between {lead_color} and surrounding pieces is overly narrow ({contrast_ratio}:1), causing visual flatness.",
            "action": f"Introduce a crisp Warm Ivory (#F5F1E6) or Jet Black (#1C1C1E) anchor layer to create an intentional focal contrast break.",
            "suggested_hex": "#F5F1E6",
            "suggested_color_name": "Warm Ivory",
            "tactical_rule": "The 60-30-10 Color Rule (60% dominant base, 30% structural secondary, 10% high-contrast accent)."
        }
    else:
        # High contrast - suggest rich grounding accent
        pivot_1 = {
            "title": "Harmonize High-Contrast Boundary with Mid-Tone Bridge",
            "category": "Chromatic Architecture",
            "flaw": f"The sharp contrast between {lead_color} and secondary tones risks severing vertical visual continuity at the waistline.",
            "action": "Introduce a muted intermediary shade (like Camel / Tan #C19A6B or Slate Grey #6C727D) via outerwear or knit layering to soften transition.",
            "suggested_hex": "#C19A6B",
            "suggested_color_name": "Camel / Tan",
            "tactical_rule": "The Tonal Bridge Principle (graduating value steps preserve vertical elongation)."
        }
    pivots.append(pivot_1)

    # Pivot 2: Silhouette & Cut Calibration (Gender & Garment aware)
    if "Saree" in garment or "Ethnic" in gender:
        pivot_2 = {
            "title": "Sculpt Pallu Pleat Density & Blouse Hemline",
            "category": "Silhouette Calibration",
            "flaw": "The traditional drape flow risks swallowing torso definition if pleat width lacks crisp pin-tucked architecture.",
            "action": "Narrow the shoulder drape pleats to 2.5 inches to expose clavicle architecture, allowing the embroidered border to frame the neckline as a choker.",
            "suggested_hex": "#72182E",
            "suggested_color_name": "Burgundy / Wine",
            "tactical_rule": "The Golden Ratio Drape (38% torso exposure to 62% continuous columnar skirt length)."
        }
    elif "Dress" in garment or "Womenswear" in gender:
        pivot_2 = {
            "title": "Calibrate Vertical Break & Waistline Index",
            "category": "Silhouette Calibration",
            "flaw": f"The current {silhouette.lower()} lacks a clear tension point between volume and body contours.",
            "action": "Pair with a structured high-waisted belt or tailored trench coat left unbuttoned to slice the silhouette into dynamic vertical thirds.",
            "suggested_hex": "#121214",
            "suggested_color_name": "Onyx Black",
            "tactical_rule": "The Rule of Thirds (1/3 upper bodice proportion counterbalanced by 2/3 sweeping lower volume)."
        }
    elif "Menswear" in gender:
        pivot_2 = {
            "title": "Structure Trouser Break & Shoulder Droop",
            "category": "Silhouette Calibration",
            "flaw": "Lower-body hemline pooling against footwear dampens the architectural sharpness of the top tailoring.",
            "action": "Tailor trousers to a clean no-break or subtle half-break hemline to showcase ankle line and footwear geometry.",
            "suggested_hex": "#3A3C42",
            "suggested_color_name": "Charcoal Grey",
            "tactical_rule": "The Clean Break Rule (zero fabric stacking preserves unhindered vertical stature)."
        }
    else:
        pivot_2 = {
            "title": "Balance Volume Distribution (Tight vs Loose)",
            "category": "Silhouette Calibration",
            "flaw": "Symmetrical looseness throughout the top and bottom dilutes silhouette definition.",
            "action": "Anchor the oversized top with tapered bottoms, or counterbalance wide trousers with a fitted inner layer.",
            "suggested_hex": "#585A62",
            "suggested_color_name": "Slate Grey",
            "tactical_rule": "The Volume Counterbalance Rule (pair volumetric shapes with rigid structural lines)."
        }
    pivots.append(pivot_2)

    # Pivot 3: Hardware, Footwear & Texture Upgrade
    if "Ethnic" in gender or "Saree" in garment:
        pivot_3 = {
            "title": "Anchor with Polished Temple or Antique Gold Hardware",
            "category": "Hardware & Footwear",
            "flaw": "Minimalist accessories fail to match the rich visual weight of ceremonial textile weaves.",
            "action": "Swap modern silver jewelry for raw brass, antique matte gold jhumkas, or sculpted metallic cuffs to resonate with fabric weave.",
            "suggested_hex": "#E4CD8E",
            "suggested_color_name": "Champagne Gold",
            "tactical_rule": "Textural Weight Parity (heavy brocade requires high-substance jewelry to prevent visual wash-out)."
        }
    elif "Womenswear" in gender:
        pivot_3 = {
            "title": "Upgrade Footwear Architecture & Metallic Accent",
            "category": "Hardware & Footwear",
            "flaw": "Footwear transition feels passive rather than serving as the definitive grounding anchor of the look.",
            "action": "Ground the look with pointed-toe slingbacks or structured leather ankle boots with brass/gold hardware buckles.",
            "suggested_hex": "#1C1C1E",
            "suggested_color_name": "Jet Black Leather",
            "tactical_rule": "The Pointed Toe Stature Principle (pointed toes extend the visual leg line by up to 2 inches)."
        }
    elif any(k in garment for k in ["T-Shirt", "Top", "Hoodie", "Shorts", "Jeans"]):
        pivot_3 = {
            "title": "Elevate Footwear & Minimalist Metal Accents",
            "category": "Hardware & Footwear",
            "flaw": "Basic casual footwear without structural form risks making the silhouette look uncurated or domestic.",
            "action": "Ground relaxed jersey basics with crisp retro leather trainers (e.g. minimalist low-tops, German Army Trainers) and anchor with a brushed stainless steel watch or signet ring.",
            "suggested_hex": "#E2E8F0",
            "suggested_color_name": "Brushed Steel / Chalk",
            "tactical_rule": "The Intentional Casual Rule (ground relaxed cotton pieces with crisp leather footwear and polished metal hardware)."
        }
    else:
        pivot_3 = {
            "title": "Anchor Footwear Volume & Leather Hardware",
            "category": "Hardware & Footwear",
            "flaw": "Lightweight sneakers or rounded shoe silhouettes fail to counterbalance the garment volume.",
            "action": "Swap soft canvas footwear for Goodyear-welted leather derbies, minimalist leather loafers, or structured Chelsea boots.",
            "suggested_hex": "#3C2A21",
            "suggested_color_name": "Espresso Leather",
            "tactical_rule": "The Footwear Visual Weight Law (shoes must support the heaviest fabric weight in the outfit)."
        }
    pivots.append(pivot_3)

    return pivots


def generate_full_critique_package(
    vision_data: Dict[str, Any],
    palette_data: List[Dict[str, Any]],
    harmony_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compile the full, dynamic editorial critique package for emo.
    """
    gender_label = vision_data["gender"]["label"]
    garment_label = vision_data["garments"]["primary"]["label"]
    sec_garment = vision_data["garments"]["secondary"]["label"] if vision_data["garments"].get("secondary") else None
    silhouette_label = vision_data["silhouette"]["label"]
    formality_label = vision_data["formality"]["label"]
    aesthetic_label = vision_data["aesthetic"]["label"]
    texture_label = vision_data["texture"]["label"]
    scorecard = vision_data["scorecard"]
    overall_score = scorecard["overall_score"]

    lead_colors = [c["name"] for c in palette_data[:3]]
    color_str = ", ".join(lead_colors) if lead_colors else "Neutrals"

    headline = generate_editorial_headline(gender_label, garment_label, harmony_data["harmony_type"], overall_score)
    savage_verdict = generate_savage_verdict(gender_label, garment_label, overall_score, harmony_data["contrast_grade"])

    # Detailed style breakdown paragraphs
    breakdown_text = (
        f"**Aesthetic & Silhouette:** Identified as **{gender_label}** framed within an **{aesthetic_label}** archetype. "
        f"The primary garment detected is a **{garment_label}**"
        + (f" paired alongside **{sec_garment}**" if sec_garment else "")
        + f", configured in **{silhouette_label}**. Proportions exhibit a {vision_data['spatial']['proportion_flow'].lower()}.\n\n"
        f"**Chromatic Chemistry:** Anchored across {color_str}. Evaluated as **{harmony_data['harmony_type']}** "
        f"with a **{harmony_data['contrast_ratio']}:1** contrast ratio ({harmony_data['contrast_grade']}) "
        f"and {harmony_data['temperature'].lower()}. {harmony_data['summary']}\n\n"
        f"**Fabric & Texture Dynamics:** Fabric surfaces predominantly register as **{texture_label}**, "
        f"calibrated for a **{formality_label}** context."
    )

    pivots = generate_styling_pivots(
        gender_label,
        garment_label,
        silhouette_label,
        harmony_data,
        palette_data,
        overall_score
    )

    return {
        "headline": headline,
        "overall_score": overall_score,
        "tier_title": scorecard["tier_title"],
        "tier_desc": scorecard["tier_desc"],
        "badge_color": scorecard["badge_color"],
        "breakdown": breakdown_text,
        "savage_verdict": savage_verdict,
        "metrics": scorecard["metrics"],
        "pivots": pivots
    }
