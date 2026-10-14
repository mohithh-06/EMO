"""
emo — Editorial Multimodal Outfit Critic & Deep Styling Advisor.
A state-of-the-art vision-language critique dashboard powered by CLIP zero-shot classification,
K-Means dominant color extraction, and color-wheel geometry algorithms.
"""

import streamlit as st
from PIL import Image
import io
import os
import json
import time
from datetime import datetime, timezone
import imagehash
from pathlib import Path

# Add fitcheck directory to sys.path cleanly
import sys
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from analyze_outfit import analyze_outfit_pipeline, extract_rating, extract_comment, extract_style_paragraph
from design_system import DESIGN_TOKENS_CSS, render_privacy_notice, render_brand_header

# -------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="emo — Haute Editorial Outfit Critic",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Design Tokens & Shared Base Styling
# -------------------------------------------------------
st.markdown(DESIGN_TOKENS_CSS, unsafe_allow_html=True)


# -------------------------------------------------------
# High-Fashion Editorial Dark CSS Injection
# -------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;800;900&family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@500;700;800&display=swap');

    /* Global Theme & Deep Obsidian Backdrop */
    .stApp {
        background: radial-gradient(circle at 50% -15%, #181c2e 0%, #0a0c14 45%, #040508 100%);
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 0.05em;
        font-weight: 700;
    }

    /* Brand Header Banner */
    .emo-brand-banner {
        position: relative;
        text-align: center;
        padding: 3rem 2rem 2.2rem 2rem;
        background: linear-gradient(180deg, rgba(28, 33, 54, 0.75) 0%, rgba(12, 14, 24, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        margin-bottom: 2rem;
        backdrop-filter: blur(25px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        overflow: hidden;
    }
    .emo-brand-banner::before {
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
        margin-bottom: 0.9rem;
    }
    .emo-title {
        font-family: 'Cinzel', serif;
        font-size: 4.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 15%, #e2e8f0 45%, #c7d2fe 75%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.22em;
        margin: 0;
        line-height: 1;
        text-transform: uppercase;
        filter: drop-shadow(0 4px 20px rgba(129, 140, 248, 0.25));
    }
    .emo-subtitle {
        font-size: 0.96rem;
        color: #94a3b8;
        letter-spacing: 0.32em;
        text-transform: uppercase;
        margin-top: 0.8rem;
        font-weight: 600;
    }
    .emo-ticker-bar {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-top: 1.6rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .emo-ticker-item {
        font-size: 0.75rem;
        color: #cbd5e1;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Glass Cards */
    .emo-card {
        background: rgba(18, 22, 36, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 1.6rem;
        backdrop-filter: blur(18px);
        margin-bottom: 1.3rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
    }

    /* Visual Dossier Frame */
    .dossier-frame {
        position: relative;
        background: rgba(15, 18, 30, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5);
    }
    .dossier-tag {
        position: absolute;
        top: 1.8rem;
        right: 1.8rem;
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #f8fafc;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.2em;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        backdrop-filter: blur(8px);
        z-index: 10;
    }

    /* Hero Score Showcase */
    .score-hero-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, rgba(26, 32, 54, 0.96) 0%, rgba(13, 16, 28, 0.96) 100%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 22px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    }
    .score-number {
        font-family: 'Cinzel', serif;
        font-size: 4.6rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.03em;
        text-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
    }
    .score-tier-badge {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 800;
        padding: 0.48rem 1.2rem;
        border-radius: 28px;
        display: inline-block;
        margin-bottom: 0.6rem;
    }

    /* Haute Tag Chips */
    .emo-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 0.95rem;
        border-radius: 12px;
        font-size: 0.84rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.14);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #c7d2fe;
        margin: 0.3rem 0.4rem 0.3rem 0;
        transition: transform 0.15s ease;
    }
    .emo-chip:hover {
        transform: translateY(-2px);
    }
    .emo-chip-pink {
        background: rgba(236, 72, 153, 0.14);
        border: 1px solid rgba(236, 72, 153, 0.35);
        color: #fbcfe8;
    }
    .emo-chip-emerald {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #a7f3d0;
    }
    .emo-chip-amber {
        background: rgba(245, 158, 11, 0.14);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fde68a;
    }
    .emo-chip-cyan {
        background: rgba(6, 182, 212, 0.14);
        border: 1px solid rgba(6, 182, 212, 0.35);
        color: #a5f3fc;
    }

    /* Palette Paint Swatches */
    .palette-container {
        display: flex;
        gap: 0.95rem;
        margin: 1.4rem 0;
        flex-wrap: wrap;
    }
    .palette-item {
        flex: 1;
        min-width: 110px;
        background: rgba(14, 18, 28, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 0.85rem;
        text-align: center;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease;
    }
    .palette-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        border-color: rgba(129, 140, 248, 0.5);
    }
    .palette-swatch {
        width: 100%;
        height: 60px;
        border-radius: 12px;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(255, 255, 255, 0.22);
        box-shadow: inset 0 2px 4px rgba(255, 255, 255, 0.2), 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    .palette-name {
        font-size: 0.84rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .palette-hex {
        font-size: 0.78rem;
        color: #94a3b8;
        font-family: monospace;
    }
    .palette-cov {
        font-size: 0.76rem;
        color: #818cf8;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    /* Savage Critic Verdict Card */
    .savage-card {
        background: linear-gradient(135deg, rgba(38, 14, 24, 0.9) 0%, rgba(20, 15, 30, 0.92) 100%);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-left: 6px solid #f43f5e;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin: 1.4rem 0;
        font-style: italic;
        color: #ffe4e6;
        line-height: 1.7;
        font-size: 1.05rem;
        box-shadow: 0 10px 30px rgba(244, 63, 94, 0.15);
        position: relative;
    }
    .savage-quote-mark {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        line-height: 0;
        position: absolute;
        top: 1.8rem;
        right: 1.4rem;
        color: rgba(244, 63, 94, 0.25);
    }

    /* Styling Pivot Deck */
    .pivot-card {
        background: rgba(18, 22, 36, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 18px;
        padding: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }
    .pivot-card-top-1 { border-top: 4px solid #6366f1; }
    .pivot-card-top-2 { border-top: 4px solid #f59e0b; }
    .pivot-card-top-3 { border-top: 4px solid #ec4899; }

    .pivot-title {
        font-family: 'Cinzel', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
        line-height: 1.35;
    }
    .pivot-category {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #818cf8;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .pivot-flaw {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.6;
        margin-bottom: 0.95rem;
        padding: 0.6rem 0.8rem;
        background: rgba(239, 68, 68, 0.08);
        border-radius: 8px;
        border-left: 3px solid #ef4444;
    }
    .pivot-action {
        font-size: 0.92rem;
        color: #ffffff;
        line-height: 1.6;
        background: rgba(99, 102, 241, 0.12);
        padding: 0.85rem;
        border-radius: 12px;
        border-left: 3px solid #818cf8;
        margin-bottom: 0.95rem;
    }
    .pivot-rule {
        font-size: 0.78rem;
        color: #94a3b8;
        font-style: italic;
        margin-top: 0.35rem;
    }

    /* Editorial Review Box */
    .editorial-review-card {
        background: rgba(16, 19, 32, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 1.8rem;
        line-height: 1.8;
        font-size: 0.98rem;
        color: #e2e8f0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .editorial-review-card strong {
        color: #f8fafc;
        font-weight: 700;
    }

    /* Custom Streamlit Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #ec4899 100%) !important;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Local-First Resilient Storage Setup (history/)
# -------------------------------------------------------
HISTORY_DIR = Path(current_dir.parent / "history")
HISTORY_FILE = HISTORY_DIR / "outfits.json"
HISTORY_DIR.mkdir(exist_ok=True)

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_to_history(record):
    records = load_history()
    records = [r for r in records if r.get("image_hash") != record.get("image_hash")]
    records.insert(0, record)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records[:30], f, indent=2)
    except Exception as e:
        print(f"[emo] Warning: Could not save history: {e}")

# -------------------------------------------------------
# Haute Hero Header
# -------------------------------------------------------
st.markdown("""
<div class="emo-brand-banner">
    <div class="emo-micro-badge">✨ Couture Intelligence Suite • Issue 2026</div>
    <div class="emo-title">emo</div>
    <div class="emo-subtitle">Editorial Multimodal Outfit Critic & AI Stylist</div>
    <div class="emo-ticker-bar">
        <div class="emo-ticker-item"><span>👁️</span> Zero-Shot CLIP ViT-B/32</div>
        <div class="emo-ticker-item"><span>🎨</span> CIE-LAB K-Means Harmony</div>
        <div class="emo-ticker-item"><span>📐</span> Golden-Ratio Spatial Proportions</div>
        <div class="emo-ticker-item"><span>🛡️</span> Calibrated Garment Taxonomy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------
if "current_image_path" not in st.session_state:
    st.session_state.current_image_path = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None

sample_images_dir = current_dir.parent / "Images"

# -------------------------------------------------------
# Sidebar Controls & Quick-Test Runway Archive
# -------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ Runway Look Archive")
    st.caption("Select any look to trigger instant multimodal deep learning analysis:")

    col_s1, col_s2 = st.columns(2)
    if (sample_images_dir / "dress.jpg").exists():
        if col_s1.button("👗 Evening Gown", use_container_width=True):
            st.session_state.current_image_path = str(sample_images_dir / "dress.jpg")
            st.session_state.current_image_name = "dress.jpg"
            st.session_state.current_results = None
            st.rerun()

    if (sample_images_dir / "fit_look_02.jpg").exists():
        if col_s2.button("🤵 Tailored Fit", use_container_width=True):
            st.session_state.current_image_path = str(sample_images_dir / "fit_look_02.jpg")
            st.session_state.current_image_name = "fit_look_02.jpg"
            st.session_state.current_results = None
            st.rerun()

    if (sample_images_dir / "jacket.jpg").exists():
        if col_s1.button("🧥 Moto Jacket", use_container_width=True):
            st.session_state.current_image_path = str(sample_images_dir / "jacket.jpg")
            st.session_state.current_image_name = "jacket.jpg"
            st.session_state.current_results = None
            st.rerun()

    if (sample_images_dir / "shoe.jpg").exists():
        if col_s2.button("👟 Nike Footwear", use_container_width=True):
            st.session_state.current_image_path = str(sample_images_dir / "shoe.jpg")
            st.session_state.current_image_name = "shoe.jpg"
            st.session_state.current_results = None
            st.rerun()

    if (sample_images_dir / "fit_look_01.jpg").exists():
        if col_s1.button("✨ Editorial Pair", use_container_width=True):
            st.session_state.current_image_path = str(sample_images_dir / "fit_look_01.jpg")
            st.session_state.current_image_name = "fit_look_01.jpg"
            st.session_state.current_results = None
            st.rerun()

    if (sample_images_dir / "sunglasses.jpg").exists():
        if col_s2.button("🕶️ Eyewear Spec", use_container_width=True):
            st.session_state.current_image_path = str(sample_images_dir / "sunglasses.jpg")
            st.session_state.current_image_name = "sunglasses.jpg"
            st.session_state.current_results = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 📜 Editorial Audit History")
    history_records = load_history()
    if history_records:
        for idx, rec in enumerate(history_records[:5]):
            date_str = rec.get("date", "Recent")[:10]
            score = rec.get("overall_score", 75)
            tier = rec.get("tier_title", "Evaluated")
            st.markdown(f"**#{idx+1} {rec.get('headline', 'Outfit')[:24]}...**")
            st.caption(f"Score: **{score}/100** ({tier}) • {date_str}")
            st.markdown("---")
    else:
        st.caption("No outfits audited yet. Snap or upload an outfit to initialize.")

    st.markdown("---")
    st.markdown("### 🧠 Deep Learning Studio")
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(129, 140, 248, 0.3); border-radius: 12px; padding: 0.9rem; font-size: 0.8rem; color: #cbd5e1; line-height: 1.55;">
        <strong style="color: #818cf8;">PyTorch Multi-Task Head</strong> active.<br>
        Open <strong>3 Model Training & Evaluation</strong> in the sidebar page menu to view loss curves, train epochs, or test accuracy.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# Truthful Privacy Architecture Disclosure
# -------------------------------------------------------
st.markdown(render_privacy_notice(
    storage_type="In-Memory Ephemeral Buffer",
    retention_policy="Images are processed purely in local RAM via PyTorch and discarded upon session reset. No external cloud upload."
), unsafe_allow_html=True)

# -------------------------------------------------------
# Input Selection: File Upload or Live Camera
# -------------------------------------------------------
input_tab1, input_tab2 = st.tabs(["📁 High-Resolution Photo Upload", "📸 Live Studio Camera"])


with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload a full-body or styled outfit photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="High-contrast, well-lit photos provide maximum classification precision."
    )
    if uploaded_file is not None:
        temp_dir = current_dir.parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        uploaded_save_path = temp_dir / uploaded_file.name
        with open(uploaded_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.session_state.current_image_path != str(uploaded_save_path):
            st.session_state.current_image_path = str(uploaded_save_path)
            st.session_state.current_image_name = uploaded_file.name
            st.session_state.current_results = None

with input_tab2:
    camera_photo = st.camera_input("Capture current outfit in real-time")
    if camera_photo is not None:
        temp_dir = current_dir.parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        cam_save_path = temp_dir / "webcam_snap.jpg"
        with open(cam_save_path, "wb") as f:
            f.write(camera_photo.getbuffer())
        
        if st.session_state.current_image_path != str(cam_save_path):
            st.session_state.current_image_path = str(cam_save_path)
            st.session_state.current_image_name = "webcam_snap.jpg"
            st.session_state.current_results = None

# -------------------------------------------------------
# Execution & Rendering
# -------------------------------------------------------
active_path = st.session_state.current_image_path

if active_path and os.path.exists(active_path):
    active_image = Image.open(active_path).convert("RGB")
    image_display_name = st.session_state.current_image_name or os.path.basename(active_path)

    # Compute or retrieve cached analysis
    if st.session_state.current_results is None:
        phash = str(imagehash.phash(active_image))
        with st.spinner("✨ emo is dissecting proportions, colors, and cut..."):
            results = analyze_outfit_pipeline(active_image)
            st.session_state.current_results = results

            # Log to local history
            history_entry = {
                "image_hash": phash,
                "filename": image_display_name,
                "date": datetime.now(timezone.utc).isoformat(),
                "overall_score": results["overall_score"],
                "tier_title": results["tier_title"],
                "headline": results["critique"]["headline"],
                "gender": results["vision"]["gender"]["label"],
                "garment": results["vision"]["garments"]["primary"]["label"]
            }
            save_to_history(history_entry)
    else:
        results = st.session_state.current_results

    # 1. Headline Banner
    st.markdown(f"""
    <div style="margin: 2.2rem 0 1.6rem 0; padding-bottom: 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.08);">
        <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.25em; color: #818cf8; font-weight: 800; margin-bottom: 0.5rem;">
            Runway Verdict & Editorial Review
        </div>
        <h2 style="color: #ffffff; margin: 0; font-size: 2.2rem; line-height: 1.28; font-family: 'Cinzel', serif;">
            {results['critique']['headline']}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # 2. Main Two-Column Layout
    col_left, col_right = st.columns([4, 6], gap="large")

    with col_left:
        # Display Image inside Dossier Frame
        st.markdown("""
        <div class="dossier-frame">
            <div class="dossier-tag">VERIFIED CAPTURE</div>
        """, unsafe_allow_html=True)
        st.image(active_image, use_container_width=True, caption=f"Audit Asset: {image_display_name}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Detected Attribute Tags
        v = results["vision"]
        st.markdown("""
        <div class="emo-card" style="margin-top: 1.2rem;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.2em; color: #94a3b8; font-weight: 800; margin-bottom: 0.85rem;">
                🔬 Deep Vision Multimodal Taxonomy
            </div>
        """, unsafe_allow_html=True)

        g_lbl = v["gender"]["label"]
        g_conf = v["gender"]["confidence"]
        gar_lbl = v["garments"]["primary"]["label"]
        gar_conf = v["garments"]["primary"]["confidence"]
        sil_lbl = v["silhouette"]["label"]
        form_lbl = v["formality"]["label"]
        aes_lbl = v["aesthetic"]["label"]
        tex_lbl = v["texture"]["label"]
        prop_lbl = v["spatial"]["proportion_flow"]

        st.markdown(f"""
            <span class="emo-chip emo-chip-pink">👤 {g_lbl} ({g_conf}%)</span>
            <span class="emo-chip emo-chip-emerald">👗 {gar_lbl} ({gar_conf}%)</span>
            <span class="emo-chip emo-chip-amber">📐 {sil_lbl}</span>
            <span class="emo-chip">👔 {form_lbl}</span>
            <span class="emo-chip emo-chip-cyan">🏛️ {aes_lbl}</span>
            <span class="emo-chip">🧵 {tex_lbl}</span>
            <span class="emo-chip">⚖️ {prop_lbl}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Hero Score Box
        badge_col = results["badge_color"]
        overall = results["overall_score"]
        tier = results["tier_title"]
        tier_desc = results["tier_desc"]

        st.markdown(f"""
        <div class="score-hero-container">
            <div>
                <div class="score-tier-badge" style="background: {badge_col}22; border: 1px solid {badge_col}66; color: {badge_col};">
                    ★ {tier}
                </div>
                <div style="font-size: 1.02rem; color: #cbd5e1; max-width: 360px; line-height: 1.5; font-family: 'Plus Jakarta Sans', sans-serif;">
                    {tier_desc}
                </div>
            </div>
            <div style="text-align: right;">
                <div class="score-number" style="color: {badge_col};">
                    {overall}<span style="font-size: 1.8rem; color: #64748b; font-weight: 500;">/100</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5 Dimension Sartorial Breakdown
        st.markdown("##### 📊 Sartorial Breakdown Dimensions")
        for metric_name, score_val in results["metrics"].items():
            col_m1, col_m2 = st.columns([8, 2])
            with col_m1:
                st.markdown(f"<span style='font-size: 0.88rem; font-weight: 600; color: #cbd5e1;'>{metric_name}</span>", unsafe_allow_html=True)
                st.progress(score_val / 100.0)
            with col_m2:
                st.markdown(f"<span style='font-size: 0.95rem; font-weight: 800; color: #f1f5f9;'>{score_val}%</span>", unsafe_allow_html=True)

        # Savage Editorial Verdict Card
        savage_text = results["critique"]["savage_verdict"]
        st.markdown(f"""
        <div class="savage-card">
            <div class="savage-quote-mark">“</div>
            <strong style="text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.78rem; color: #fb7185;">The Critic's Front-Row Verdict:</strong><br>
            <span style="font-family: 'Playfair Display', serif; font-size: 1.08rem;">"{savage_text}"</span>
        </div>
        """, unsafe_allow_html=True)

    # 3. Dynamic Color Theory & Palette Ribbon
    st.markdown("---")
    st.markdown("### 🎨 Chromatic Chemistry & Dominant Color Palette")
    h_data = results["harmony"]
    
    st.caption(
        f"**Harmonic Spectrum:** {h_data['harmony_type']} • **Lightness Contrast:** {h_data['contrast_ratio']}:1 ({h_data['contrast_grade']}) • **Color Temperature:** {h_data['temperature']}"
    )

    palette_html = '<div class="palette-container">'
    for col_info in results["palette"]:
        palette_html += f"""
        <div class="palette-item">
            <div class="palette-swatch" style="background-color: {col_info['hex']};"></div>
            <div class="palette-name">{col_info['name']}</div>
            <div class="palette-hex">{col_info['hex']}</div>
            <div class="palette-cov">{col_info['coverage_pct']}% Area</div>
        </div>
        """
    palette_html += '</div>'
    st.markdown(palette_html, unsafe_allow_html=True)

    # 4. Detailed Editorial In-Depth Review
    st.markdown("### 📖 In-Depth Editorial Dossier")
    st.markdown(f"""
    <div class="editorial-review-card">
        {results["critique"]["breakdown"]}
    </div>
    """, unsafe_allow_html=True)

    # 5. Three Actionable emo Styling Pivots
    st.markdown("---")
    st.markdown("### 🛠️ The 3 emo Styling Pivots")
    st.caption("Architectural, chromatic, and hardware adjustments engineered to elevate this fit:")

    col_p1, col_p2, col_p3 = st.columns(3)
    pivots = results["pivots"]
    pivot_classes = ["pivot-card-top-1", "pivot-card-top-2", "pivot-card-top-3"]

    for idx, (col, pivot, top_class) in enumerate(zip([col_p1, col_p2, col_p3], pivots, pivot_classes)):
        with col:
            s_hex = pivot.get("suggested_hex", "#ffffff")
            s_color = pivot.get("suggested_color_name", "Contrast Accent")
            st.markdown(f"""
            <div class="pivot-card {top_class}">
                <div>
                    <div class="pivot-category">{pivot['category']}</div>
                    <div class="pivot-title">{pivot['title']}</div>
                    <div class="pivot-flaw"><strong>Identified Flaw:</strong> {pivot['flaw']}</div>
                    <div class="pivot-action"><strong>Stylist Action:</strong> {pivot['action']}</div>
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin: 0.9rem 0; padding: 0.6rem 0.85rem; background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;">
                        <div style="width: 26px; height: 26px; border-radius: 8px; background-color: {s_hex}; border: 1px solid rgba(255,255,255,0.4); box-shadow: 0 2px 8px rgba(0,0,0,0.5);"></div>
                        <div>
                            <div style="font-size: 0.84rem; font-weight: 700; color: #f8fafc;">{s_color}</div>
                            <div style="font-size: 0.74rem; color: #94a3b8; font-family: monospace;">{s_hex}</div>
                        </div>
                    </div>
                    <div class="pivot-rule">💡 {pivot['tactical_rule']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 6. Hugging Face Curated Capsule Recommendation
    st.markdown("---")
    st.markdown("### 🤗 Hugging Face Editorial Capsule Blueprint")
    st.caption("Contextually retrieved from `hf://datasets/formido/outfit_recomendation` (3,193 verified styling profiles):")
    
    try:
        from hf_datasets_loader import query_formido_recommendations
        p_garment = results.get("garment", "Casual T-Shirt")
        p_gender = results.get("gender", "Any")
        p_form = results.get("formality", "Casual")
        hf_recs = query_formido_recommendations(query_text=p_garment, gender=p_gender, formality=p_form, max_results=1)
        if hf_recs:
            rec = hf_recs[0]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 36, 60, 0.85) 0%, rgba(16, 20, 34, 0.95) 100%); border: 1px solid rgba(129, 140, 248, 0.35); border-left: 5px solid #ec4899; border-radius: 18px; padding: 1.5rem; margin-top: 1rem;">
                <div style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.18em; color: #ec4899; font-weight: 800; margin-bottom: 0.4rem;">
                    Hugging Face Stylist Blueprint • Client Archetype
                </div>
                <p style="color: #f1f5f9; font-size: 0.95rem; font-style: italic; margin-bottom: 0.9rem;">"{rec['instruction']}"</p>
                <div style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.18em; color: #818cf8; font-weight: 800; margin-bottom: 0.4rem;">
                    Recommended Capsule Ensemble
                </div>
                <div style="color: #cbd5e1; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; white-space: pre-wrap; line-height: 1.6; background: rgba(0,0,0,0.35); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);">
{rec['output']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        print(f"[HF Blueprint note]: {e}")

else:
    # Empty State Guidance
    st.markdown("""
    <div style="text-align: center; padding: 4.5rem 1.5rem; color: #94a3b8; background: rgba(18, 22, 36, 0.5); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.12); margin-top: 2rem;">
        <div style="font-size: 3.5rem; margin-bottom: 0.8rem;">✨</div>
        <h3 style="color: #f8fafc; margin-bottom: 0.6rem; font-family: 'Cinzel', serif;">Awaiting Sartorial Canvas</h3>
        <p style="max-width: 520px; margin: 0 auto; font-size: 1.02rem; line-height: 1.7; color: #94a3b8;">
            Select any look from the <strong>Runway Look Archive</strong> in the left sidebar to audit immediately,
            or upload a high-resolution photo to run multimodal deep learning critique.
        </p>
    </div>
    """, unsafe_allow_html=True)
