"""
emo — AI Stylist & Virtual Fitting Deck.
Generates dynamically coordinated outfits from digitized wardrobe inventory
using Color Wheel geometry, silhouette proportion laws, and gender presentation matching.
"""

import streamlit as st
import os
import json
import random
from typing import List, Dict, Optional, Any
from pathlib import Path

# Add fitcheck directory to sys.path
import sys
current_dir = Path(__file__).resolve().parent
fitcheck_dir = current_dir.parent
if str(fitcheck_dir) not in sys.path:
    sys.path.insert(0, str(fitcheck_dir))

from color_engine import FASHION_COLOR_NAMES
from design_system import DESIGN_TOKENS_CSS, render_privacy_notice

st.set_page_config(page_title="emo — AI Stylist Deck", page_icon="👔", layout="wide")

# Apply Shared Design Tokens
st.markdown(DESIGN_TOKENS_CSS, unsafe_allow_html=True)

# High-Fashion Styling
st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;800;900&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: #ffffff !important;
        color: #0f172a;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4 { font-family: 'Cinzel', serif !important; }

    .stylist-header {
        position: relative;
        text-align: center;
        padding: 2.8rem 2rem 2rem 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 50%, #f0f9ff 100%); border: 1px solid #e0e7ff;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        margin-bottom: 2.5rem;
        backdrop-filter: blur(25px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    .stylist-header::before {
        content: '';
        position: absolute;
        top: 0; left: 25%; right: 25%; height: 2px;
        background: linear-gradient(90deg, transparent, #818cf8, #f472b6, transparent);
    }

    .emo-micro-badge {
        display: inline-block;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.28em;
        font-weight: 800;
        color: #d4af37;
        background: rgba(212, 175, 55, 0.12);
        border: 1px solid rgba(212, 175, 55, 0.3);
        padding: 0.35rem 1.1rem;
        border-radius: 30px;
        margin-bottom: 0.8rem;
    }

    .emo-title {
        font-family: 'Cinzel', serif;
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 15%, #e2e8f0 45%, #c7d2fe 75%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.15em;
        margin: 0;
        line-height: 1.1;
    }

    .outfit-card {
        background: rgba(18, 22, 36, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 1.4rem;
        height: 100%;
        text-align: center;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .outfit-card:hover {
        transform: translateY(-6px);
        border-color: rgba(129, 140, 248, 0.5);
    }

    .outfit-item-badge {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 800;
        color: #818cf8;
        margin-bottom: 0.5rem;
    }

    .synthesis-card {
        background: linear-gradient(135deg, rgba(30, 36, 60, 0.85) 0%, rgba(16, 20, 34, 0.95) 100%);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-left: 5px solid #818cf8;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stylist-header">
    <div class="emo-micro-badge">Algorithmic Capsule Engine</div>
    <div class="emo-title">AI Stylist Deck</div>
    <div style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.24em; margin-top: 0.6rem; font-weight: 600;">
        Automated Wardrobe Coordination, Color Wheel Harmonics & Silhouette Pacing
    </div>
</div>
""", unsafe_allow_html=True)

# Curated high-precision color compatibility map
EXPANDED_COLOR_COMPATIBILITY = {
    "black": ["white", "grey", "gray", "red", "gold", "silver", "pink", "navy", "camel / tan", "warm ivory"],
    "white": ["black", "navy", "red", "green", "blue", "pink", "purple", "camel / tan", "charcoal grey"],
    "grey": ["pink", "red", "white", "black", "navy", "burgundy / wine", "cobalt / royal blue"],
    "gray": ["pink", "red", "white", "black", "navy", "burgundy / wine", "cobalt / royal blue"],
    "navy": ["white", "pink", "gold", "gray", "camel / tan", "burgundy / wine", "warm ivory"],
    "blue": ["white", "grey", "beige", "brown", "orange", "warm ivory", "camel / tan"],
    "red": ["black", "white", "grey", "blue", "gold", "camel / tan", "navy blue"],
    "green": ["white", "black", "brown", "beige", "gold", "warm ivory"],
    "camel / tan": ["navy blue", "pure white", "onyx black", "forest green", "burgundy / wine", "warm ivory"],
    "warm ivory": ["onyx black", "navy blue", "espresso brown", "terracotta / rust", "emerald green", "camel / tan"],
    "burgundy / wine": ["charcoal grey", "camel / tan", "pure white", "warm ivory", "navy blue"],
    "charcoal grey": ["pure white", "crimson red", "blush / powder pink", "camel / tan", "cobalt / royal blue"]
}

class WardrobeItem:
    def __init__(self, data: Dict[str, Any], project_root: Path):
        self.raw_data = data
        self.name = data.get("name", "Wardrobe Piece")
        self.item_type = data.get("item_type", "Garment").lower()
        self.color = data.get("color", "Neutral").lower()
        self.color_hex = data.get("color_hex", "#888888")
        self.formality = data.get("formality", "Casual")
        self.gender = data.get("gender", "Unisex")
        self.texture = data.get("texture", "Fabric")
        
        raw_path = data.get("path", "")
        clean_name = Path(raw_path).name
        
        cand1 = project_root / raw_path
        cand2 = project_root / "Images" / clean_name
        cand3 = project_root / "Closet" / clean_name
        
        if cand1.exists():
            self.image_path = str(cand1)
        elif cand2.exists():
            self.image_path = str(cand2)
        elif cand3.exists():
            self.image_path = str(cand3)
        elif os.path.exists(raw_path):
            self.image_path = raw_path
        else:
            self.image_path = ""

    def __repr__(self):
        return f"{self.name} ({self.color} {self.item_type})"


class EmoWardrobeCoordinator:
    def __init__(self, closet_dir: Path, project_root: Path):
        self.closet_dir = closet_dir
        self.project_root = project_root
        self.items = self.load_inventory()

    def load_inventory(self) -> List[WardrobeItem]:
        items = []
        if not self.closet_dir.exists():
            return items
        for p in self.closet_dir.glob("*.json"):
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    items.append(WardrobeItem(data, self.project_root))
            except Exception:
                continue
        return items

    def filter_candidates(self, category_keywords: List[str], gender: str = "Any", formality: str = "Any") -> List[WardrobeItem]:
        cands = []
        for it in self.items:
            type_match = any(kw in it.item_type for kw in category_keywords)
            if not type_match:
                continue
            
            if gender != "Any":
                if it.gender != "Unisex" and it.gender.lower() != gender.lower():
                    continue

            if formality != "Any":
                if it.formality.lower() != formality.lower():
                    continue

            cands.append(it)
        return cands

    def generate_outfit(self, gender: str, formality: str, preferred_color: Optional[str] = None) -> Dict[str, WardrobeItem]:
        outfit = {}

        if "women" in gender.lower() and formality == "Formal":
            archetypes = [
                ("Statement Piece", ["dress", "dresses", "gown", "skirt"]),
                ("Outerwear / Layer", ["jacket", "jackets", "blazer"]),
                ("Footwear", ["shoe", "shoes", "heels"]),
                ("Accessory", ["bag", "bags", "purse", "sunglasses"])
            ]
        elif "women" in gender.lower():
            archetypes = [
                ("Top Layer", ["shirt", "shirts", "t-shirt", "t-shirts", "sweater"]),
                ("Bottom Layer", ["pants", "shorts", "skirt", "skirts", "jeans"]),
                ("Footwear", ["shoe", "shoes"]),
                ("Accessory", ["bag", "bags", "sunglasses"])
            ]
        elif "men" in gender.lower() and formality == "Formal":
            archetypes = [
                ("Formal Shirt", ["shirt", "shirts"]),
                ("Tailored Jacket", ["jacket", "jackets", "blazer"]),
                ("Trousers", ["pants", "trousers"]),
                ("Footwear", ["shoe", "shoes"])
            ]
        else:
            archetypes = [
                ("Top / Layer", ["shirt", "shirts", "jacket", "jackets", "t-shirt"]),
                ("Bottom", ["pants", "shorts", "jeans"]),
                ("Footwear", ["shoe", "shoes"]),
                ("Accent", ["sunglasses", "hats", "bags", "purse"])
            ]

        lead_color = preferred_color.lower() if preferred_color and preferred_color != "Any" else None

        for role_name, kws in archetypes:
            cands = self.filter_candidates(kws, gender=gender, formality=formality)
            if not cands:
                cands = self.filter_candidates(kws, gender=gender, formality="Any")
            if not cands:
                cands = self.filter_candidates(kws, gender="Any", formality="Any")

            if not cands:
                continue

            if lead_color:
                compat = EXPANDED_COLOR_COMPATIBILITY.get(lead_color, [])
                matched = [c for c in cands if any(comp in c.color for comp in compat) or lead_color in c.color]
                chosen = random.choice(matched) if matched else random.choice(cands)
            else:
                chosen = random.choice(cands)
                lead_color = chosen.color

            outfit[role_name] = chosen

        return outfit


from hf_datasets_loader import query_formido_recommendations

project_root = fitcheck_dir.parent
closet_dir = project_root / "Closet"
coordinator = EmoWardrobeCoordinator(closet_dir, project_root)

tab_closet, tab_hf = st.tabs([
    "👔 Closet Capsule Coordinator",
    "🤗 Hugging Face AI Stylist (formido/outfit_recomendation)"
])

# -------------------------------------------------------------
# TAB 1: CLOSET CAPSULE COORDINATOR
# -------------------------------------------------------------
with tab_closet:
    st.markdown("### 🎛️ Digital Wardrobe Synthesis Parameters")
    st.caption("Coordinate outfits from your digitized closet inventory based on color theory and silhouette rules.")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_gender = st.selectbox("Presentation / Cut", ["Any", "Women's", "Men's", "Unisex"], key="closet_gen")
    with col2:
        selected_formality = st.selectbox("Occasion / Formality", ["Any", "Formal", "Smart Casual", "Casual"], key="closet_form")
    with col3:
        all_colors = ["Any", "Black", "White", "Navy", "Camel / Tan", "Burgundy / Wine", "Charcoal Grey", "Blue", "Red", "Green"]
        selected_color = st.selectbox("Lead Color Palette Anchor", all_colors, key="closet_col")

    generate_btn = st.button("✨ Synthesize Capsule Look", use_container_width=True, key="closet_gen_btn")

    if generate_btn or 'current_outfit' in st.session_state:
        if generate_btn:
            st.session_state.current_outfit = coordinator.generate_outfit(
                gender=selected_gender,
                formality=selected_formality,
                preferred_color=None if selected_color == "Any" else selected_color
            )

        outfit = st.session_state.get('current_outfit', {})

        if outfit:
            st.markdown("---")
            st.markdown("### 🌟 Curated Capsule Combination")
            
            cols = st.columns(len(outfit))
            for idx, (role_name, item) in enumerate(outfit.items()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="outfit-card">
                        <div class="outfit-item-badge">{role_name}</div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 0.6rem;">{item.name}</div>
                    """, unsafe_allow_html=True)

                    if item.image_path and os.path.exists(item.image_path):
                        st.image(item.image_path, use_container_width=True)
                    else:
                        st.markdown("""
                        <div style="height: 180px; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.3); border-radius: 12px; color: #64748b;">
                            👔 Visual Asset
                        </div>
                        """, unsafe_allow_html=True)

                    st.caption(f"{item.gender} • {item.color.title()} • {item.formality}")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="synthesis-card">
                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.18em; color: #818cf8; font-weight: 800;">Stylist Synthesis & Execution Directive</div>
                <p style="margin: 0.6rem 0 0 0; color: #cbd5e1; font-size: 0.98rem; line-height: 1.6;">
                    This combination balances structural anchors with deliberate chromatic pacing. 
                    Ensure footwear visual weight supports the hemline drop, and calibrate metallic hardware finishes to match the lead anchor piece.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No coordinating pieces found in closet matching those criteria. Try relaxing the filters or adding more garments.")

# -------------------------------------------------------------
# TAB 2: HUGGING FACE AI STYLIST (formido/outfit_recomendation)
# -------------------------------------------------------------
with tab_hf:
    st.markdown("### 🤗 Hugging Face Editorial Outfit Recommendation Engine")
    st.caption("Powered by `hf://datasets/formido/outfit_recomendation` (3,193 verified instruction-tuned styling profiles).")

    col_h1, col_h2 = st.columns([7, 3])
    with col_h1:
        user_query = st.text_input(
            "Describe your physique, vibe, or occasion",
            value="Athletic man looking for smart casual summer office outfit",
            placeholder="e.g. Petite woman attending evening cocktail party, tall athletic man casual weekend"
        )
    with col_h2:
        hf_gender = st.selectbox("Presentation", ["Any", "Menswear", "Womenswear"], key="hf_gen_select")

    # Quick prompt pills
    st.markdown("**Quick Preset Inquiries:**")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("👔 Athletic Smart Casual", use_container_width=True):
            user_query = "Athletic man smart casual blazer and chinos"
    with preset_cols[1]:
        if st.button("👗 Elegant Evening Gown", use_container_width=True):
            user_query = "Woman elegant evening dinner cocktail dress"
    with preset_cols[2]:
        if st.button("🛹 Relaxed Urban Streetwear", use_container_width=True):
            user_query = "Relaxed streetwear hoodie and sneakers"
    with preset_cols[3]:
        if st.button("☀️ Warm Weather Minimalist", use_container_width=True):
            user_query = "Summer linen shirt and shorts warm neutral"

    fetch_hf_btn = st.button("🚀 Retrieve Expert Outfits from Hugging Face", use_container_width=True, type="primary")

    if fetch_hf_btn or "hf_recommendations" in st.session_state:
        if fetch_hf_btn:
            with st.spinner("Connecting to Hugging Face Hub (formido/outfit_recomendation)..."):
                recs = query_formido_recommendations(
                    query_text=user_query,
                    gender=hf_gender,
                    max_results=3
                )
                st.session_state["hf_recommendations"] = recs

        recs = st.session_state.get("hf_recommendations", [])
        if recs:
            st.markdown("---")
            st.markdown(f"#### 📖 Curated Editorial Guidance ({len(recs)} Stylist Blueprints Found)")
            for r_idx, rec in enumerate(recs):
                with st.expander(f"✨ Blueprint #{r_idx+1}: {rec['instruction'][:90]}...", expanded=(r_idx == 0)):
                    st.markdown(f"""
                    <div style="background: rgba(18, 22, 38, 0.9); border: 1px solid rgba(129, 140, 248, 0.25); border-radius: 14px; padding: 1.4rem; margin-bottom: 1rem;">
                        <div style="color: #818cf8; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.4rem;">Client Profile & Context</div>
                        <p style="color: #0f172a; font-size: 0.98rem; font-style: italic; margin-bottom: 1rem;">"{rec['instruction']}"</p>
                        <div style="color: #f472b6; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.4rem;">Curated Outfit Execution</div>
                        <div style="color: #cbd5e1; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; white-space: pre-wrap; line-height: 1.6; background: rgba(0,0,0,0.35); padding: 1rem; border-radius: 10px;">{rec['output']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No matching styling profiles found. Try broadening your keywords.")