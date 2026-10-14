"""
emo — Wardrobe Digitizer & Inventory Engine.
Digitizes personal wardrobe items with deep-vision tagging (gender, category, texture, formality)
and exact K-Means color quantization.
"""

import streamlit as st
import os
import json
from pathlib import Path
from PIL import Image

import sys
current_dir = Path(__file__).resolve().parent
fitcheck_dir = current_dir.parent
if str(fitcheck_dir) not in sys.path:
    sys.path.insert(0, str(fitcheck_dir))

from tagging import tag_closet_item
from design_system import DESIGN_TOKENS_CSS, render_privacy_notice

st.set_page_config(page_title="emo — Wardrobe Digitizer", page_icon="🧷", layout="wide")

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

    .digitizer-header {
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
    .digitizer-header::before {
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

    .emo-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.8rem;
        backdrop-filter: blur(18px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
    }

    .wardrobe-item-card {
        background: rgba(18, 22, 36, 0.92);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .wardrobe-item-card:hover {
        transform: translateY(-6px);
        border-color: rgba(129, 140, 248, 0.5);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
    }

    .swatch-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.45rem 0.85rem;
        border-radius: 10px;
        background: rgba(12, 15, 25, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.14);
        margin: 0.3rem 0.4rem 0.3rem 0;
        font-size: 0.82rem;
    }

    .meta-tag {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.15);
        color: #c7d2fe;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="digitizer-header">
    <div class="emo-micro-badge">Wardrobe Digitization Suite</div>
    <div class="emo-title">Garment Cataloger</div>
    <div style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.24em; margin-top: 0.6rem; font-weight: 600;">
        Zero-Shot Multimodal Attribute Tagging & Chromatic Extraction
    </div>
</div>
""", unsafe_allow_html=True)

closet_dir = fitcheck_dir.parent / "Closet"
images_dir = fitcheck_dir.parent / "Images"
closet_dir.mkdir(exist_ok=True)
images_dir.mkdir(exist_ok=True)

# Truthful Privacy Notice
st.markdown(render_privacy_notice(
    storage_type="Local File System (./Closet/)",
    retention_policy="Images are saved solely to your local device workspace. No third-party or cloud storage."
), unsafe_allow_html=True)

col_upload, col_preview = st.columns([5, 5], gap="large")

with col_upload:
    st.markdown("### 📸 Ingest Garment Asset")

    uploaded_file = st.file_uploader(
        "Upload a clear photo of an individual clothing item",
        type=["jpg", "jpeg", "png", "webp"],
        help="Solo clothing items laid flat or on hangers provide the cleanest feature extractions."
    )

    item_name_input = st.text_input("Custom Item Name (Optional)", placeholder="e.g. Navy Italian Wool Blazer")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        clean_filename = f"closet_{int(os.path.getmtime(closet_dir))}_{uploaded_file.name.replace(' ', '_')}"
        saved_img_path = images_dir / clean_filename
        image.save(saved_img_path)

        with st.spinner("✨ Dissecting fabric texture, garment taxonomy, and color spectrum..."):
            result = tag_closet_item(str(saved_img_path))

        if "error" in result:
            st.error(f"Analysis failed: {result['error']}")
        else:
            if item_name_input.strip():
                result["name"] = item_name_input.strip()

            base_name = Path(clean_filename).stem
            out_json_path = closet_dir / f"{base_name}.json"
            result["path"] = str(Path("Images") / clean_filename)
            
            with open(out_json_path, "w") as f:
                json.dump(result, f, indent=2)

            st.success(f"✨ Garment successfully digitized into closet as **{result['name']}**!")

with col_preview:
    if uploaded_file is not None and 'result' in locals() and "error" not in result:
        st.markdown("### 🔬 Extracted Sartorial Metadata")
        st.image(image, use_container_width=True, caption=result["name"])

        st.markdown(f"""
        <div class="emo-card" style="margin-top: 1rem;">
            <div style="margin-bottom: 0.6rem;">
                <span class="meta-tag">🏷️ {result['item_type']}</span>
                <span class="meta-tag">👤 {result['gender']}</span>
                <span class="meta-tag">👔 {result['formality']}</span>
                <span class="meta-tag">🧵 {result['texture']}</span>
                <span class="meta-tag">📍 {result['indoor_outdoor']}</span>
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 0.8rem;">
                <strong>Primary Tone:</strong> {result['color'].title()} ({result.get('color_hex', '#888888')})
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 🎨 Quantized Palette")
        chips_html = ""
        for c in result.get("palette", []):
            chips_html += f"""
            <div class="swatch-chip">
                <div style="width: 20px; height: 20px; border-radius: 6px; background: {c['hex']}; border: 1px solid rgba(255,255,255,0.25);"></div>
                <span style="font-size: 0.82rem; font-weight: 700; color: #0f172a;">{c['name']}</span>
                <span style="font-size: 0.74rem; color: #94a3b8; font-family: monospace;">{c['hex']}</span>
            </div>
            """
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.info("Upload an image on the left to inspect real-time zero-shot attribute extraction.")

st.markdown("---")
st.markdown("### 🧥 Current Digital Wardrobe Archive")

all_json_files = list(closet_dir.glob("*.json"))
if all_json_files:
    cols = st.columns(4)
    for idx, j_path in enumerate(all_json_files):
        try:
            with open(j_path, "r") as f:
                item_data = json.load(f)
            
            col_target = cols[idx % 4]
            with col_target:
                img_path = item_data.get("path", "")
                full_img_path = fitcheck_dir.parent / img_path
                if not full_img_path.exists():
                    full_img_path = images_dir / Path(img_path).name

                st.markdown('<div class="wardrobe-item-card">', unsafe_allow_html=True)
                if full_img_path.exists():
                    st.image(str(full_img_path), use_container_width=True)
                
                c_name = item_data.get("name", item_data.get("item_type", "Item"))
                c_gender = item_data.get("gender", "Unisex")
                c_color = item_data.get("color", "Neutral")
                c_form = item_data.get("formality", "Casual")
                c_hex = item_data.get("color_hex", "#888888")

                st.markdown(f"<div style='font-weight: 700; font-size: 0.95rem; color: #ffffff; margin-top: 0.6rem;'>{c_name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 0.76rem; color: #94a3b8; margin: 0.2rem 0;'><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background-color:{c_hex}; margin-right:4px;'></span>{c_color.title()} • {c_gender} • {c_form}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            continue
else:
    st.caption("Your digital closet is currently empty. Upload items above to start building your wardrobe.")
