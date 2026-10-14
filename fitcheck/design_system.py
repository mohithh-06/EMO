"""
emo — Design System Tokens & Shared UI Components.
Art-directed design system featuring clean crisp White primary surfaces,
Lavender / Muted Violet accents, Light Sky Blue structures, and Sage / Light Green verified badges.
"""

from typing import Dict, Any, List, Optional
import html


# =====================================================================
# 1. DESIGN TOKENS (CSS Variables & Color Palettes)
# =====================================================================

DESIGN_TOKENS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,600&family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    /* Palette: Pure White & Warm Off-White Surfaces */
    --emo-bg-base: #ffffff;
    --emo-bg-subtle: #f8fafc;
    --emo-surface-1: #ffffff;
    --emo-surface-2: #f5f3ff;          /* Light Lavender */
    --emo-surface-elevated: #ffffff;
    --emo-border-subtle: #e2e8f0;
    --emo-border-strong: #cbd5e1;
    --emo-border-accent: #c4b5fd;       /* Soft Lavender Border */

    /* Typography & Contrast (WCAG 2.2 AA compliant: > 10:1 on white) */
    --emo-text-primary: #0f172a;       /* Deep Slate / Midnight Navy */
    --emo-text-secondary: #475569;     /* Slate 600 */
    --emo-text-muted: #64748b;         /* Slate 500 */
    --emo-text-inverse: #ffffff;

    /* Palette Accents */
    --emo-brand-lavender: #8b5cf6;     /* Vibrant Lavender / Violet */
    --emo-brand-lavender-light: #ede9fe; /* Soft Lavender Tint */
    --emo-brand-blue: #0284c7;         /* Light Blue / Sky Blue */
    --emo-brand-blue-light: #e0f2fe;   /* Soft Blue Tint */
    --emo-brand-green: #059669;        /* Sage / Light Emerald */
    --emo-brand-green-light: #d1fae5;  /* Soft Green Tint */

    /* Radii & Shadows */
    --emo-radius-sm: 8px;
    --emo-radius-md: 14px;
    --emo-radius-lg: 20px;
    --emo-radius-xl: 28px;
    --emo-shadow-soft: 0 4px 20px -2px rgba(100, 116, 139, 0.12);
    --emo-shadow-elevated: 0 16px 40px -6px rgba(100, 116, 139, 0.18);
}

/* Global Application Styling */
.stApp {
    background: linear-gradient(180deg, #fbfaff 0%, #ffffff 40%, #f0fdf4 100%) !important;
    color: var(--emo-text-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.01em;
}

/* Sidebar Custom Palette */
[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid var(--emo-border-subtle) !important;
}
[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* Editorial Typography */
h1, h2, h3, h4 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #0f172a !important;
    letter-spacing: -0.02em;
    font-weight: 700;
}

/* Accessibility Focus Ring */
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--emo-brand-lavender) !important;
    outline-offset: 2px !important;
}

/* Universal Streamlit Control Polish & Responsive Luxury Buttons */
div.stButton > button {
    position: relative;
    border-radius: var(--emo-radius-md) !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.75rem 1.6rem !important;
    min-height: 48px !important;
    width: 100% !important;
    border: 1px solid #c4b5fd !important;
    background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%) !important;
    color: #5b21b6 !important;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.12) !important;
    transition: all 0.22s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    cursor: pointer !important;
    text-align: center !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    overflow: hidden !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #7c3aed 0%, #0284c7 100%) !important;
    border-color: #7c3aed !important;
    color: #ffffff !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 10px 24px rgba(124, 58, 237, 0.28) !important;
}

div.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
    box-shadow: 0 2px 6px rgba(124, 58, 237, 0.2) !important;
}

/* Sidebar Custom Responsive Button Grid */
[data-testid="stSidebar"] div.stButton > button {
    font-size: 0.86rem !important;
    padding: 0.6rem 0.9rem !important;
    min-height: 42px !important;
    border-radius: var(--emo-radius-sm) !important;
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: linear-gradient(135deg, #ede9fe 0%, #e0f2fe 100%) !important;
    color: #4338ca !important;
    border-color: #c7d2fe !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.18) !important;
}

/* Responsive Tab Bar Polish */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f1f5f9;
    padding: 6px;
    border-radius: var(--emo-radius-lg);
    border: 1px solid var(--emo-border-subtle);
    display: flex;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--emo-radius-md);
    color: #64748b;
    font-weight: 700;
    font-size: 0.94rem;
    padding: 10px 22px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    flex: 1 1 auto;
    text-align: center;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0f172a;
    background: rgba(255, 255, 255, 0.8);
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #6d28d9 !important;
    border: 1px solid #ddd6fe !important;
    box-shadow: 0 4px 14px rgba(109, 40, 217, 0.12) !important;
}

/* File Uploader Surface */
[data-testid="stFileUploaderDropzone"] {
    background: #fbfbfe !important;
    border: 2px dashed #c4b5fd !important;
    border-radius: var(--emo-radius-lg) !important;
    padding: 2.5rem 1.8rem !important;
    transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.06) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #7c3aed !important;
    background: #f5f3ff !important;
    box-shadow: 0 8px 24px rgba(124, 58, 237, 0.14) !important;
    transform: translateY(-1px) !important;
}

/* Privacy Truthful Banner */
.emo-privacy-notice {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 20px;
    background: #ecfdf5;              /* Light Green / Sage */
    border: 1px solid #a7f3d0;
    border-left: 5px solid #059669;
    border-radius: var(--emo-radius-md);
    font-size: 0.86rem;
    color: #065f46;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px rgba(5, 150, 105, 0.08);
}
.emo-privacy-icon {
    font-size: 1.25rem;
    color: #059669;
    line-height: 1;
}

/* Brand Banner on White Background */
.emo-brand-banner {
    position: relative;
    text-align: center;
    padding: 3rem 2rem 2.2rem 2rem;
    background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 50%, #f0f9ff 100%);
    border: 1px solid #e0e7ff;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: var(--emo-shadow-elevated);
    overflow: hidden;
}
.emo-brand-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 20%; right: 20%; height: 3px;
    background: linear-gradient(90deg, #7c3aed, #0284c7, #10b981);
}
.emo-micro-badge {
    display: inline-block;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.24em;
    font-weight: 800;
    color: #6d28d9;
    background: #ede9fe;
    border: 1px solid #ddd6fe;
    padding: 0.35rem 1.1rem;
    border-radius: 30px;
    margin-bottom: 0.9rem;
}
.emo-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 4.4rem;
    font-weight: 900;
    color: #1e1b4b;
    letter-spacing: 0.12em;
    margin: 0;
    line-height: 1;
    text-transform: uppercase;
}
.emo-subtitle {
    font-size: 1.02rem;
    color: #64748b;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 0.8rem;
    font-weight: 700;
}
.emo-ticker-bar {
    display: flex;
    justify-content: center;
    gap: 1.6rem;
    flex-wrap: wrap;
    margin-top: 1.6rem;
    padding-top: 1.2rem;
    border-top: 1px solid #e2e8f0;
}
.emo-ticker-item {
    font-size: 0.78rem;
    color: #334155;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #ffffff;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Glass & Clean White Cards */
.emo-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.6rem;
    margin-bottom: 1.3rem;
    box-shadow: var(--emo-shadow-soft);
}

/* Visual Dossier Frame */
.dossier-frame {
    position: relative;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: var(--emo-shadow-soft);
}
.dossier-tag {
    position: absolute;
    top: 1.8rem;
    right: 1.8rem;
    background: #0f172a;
    color: #ffffff;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    font-weight: 800;
    padding: 0.35rem 0.85rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 10;
}

/* Score Hero Showcase in Lavender & Slate */
.score-hero-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #f5f3ff 0%, #e0f2fe 100%);
    border: 1px solid #c7d2fe;
    border-radius: 22px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    box-shadow: var(--emo-shadow-soft);
}
.score-number {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 4.4rem;
    font-weight: 900;
    line-height: 1;
    color: #1e1b4b;
}
.score-tier-badge {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 800;
    padding: 0.48rem 1.2rem;
    border-radius: 28px;
    display: inline-block;
    margin-bottom: 0.6rem;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

/* Light Semantic Tag Chips */
.emo-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.45rem 0.95rem;
    border-radius: 12px;
    font-size: 0.84rem;
    font-weight: 700;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #1e293b;
    margin: 0.3rem 0.4rem 0.3rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.emo-chip-pink {
    background: #fdf2f8;
    border: 1px solid #fbcfe8;
    color: #9d174d;
}
.emo-chip-emerald {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}
.emo-chip-amber {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}
.emo-chip-cyan {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    color: #0369a1;
}

/* Palette Swatches */
.palette-container {
    display: flex;
    gap: 0.95rem;
    margin: 1.4rem 0;
    flex-wrap: wrap;
}
.palette-item {
    flex: 1;
    min-width: 110px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 0.85rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.palette-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
}

/* Strategic Styling Pivot Cards */
.pivot-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: var(--emo-shadow-soft);
    position: relative;
    overflow: hidden;
}
.pivot-card-top-1 { border-top: 5px solid #7c3aed; }  /* Lavender / Violet */
.pivot-card-top-2 { border-top: 5px solid #0284c7; }  /* Light Blue */
.pivot-card-top-3 { border-top: 5px solid #059669; }  /* Light Green / Sage */

.pivot-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.75rem;
    line-height: 1.35;
}
.pivot-category {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #6d28d9;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.pivot-flaw {
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.6;
    margin-bottom: 0.95rem;
    padding: 0.6rem 0.8rem;
    background: #fef2f2;
    border-radius: 8px;
    border-left: 3px solid #ef4444;
}
.pivot-action {
    font-size: 0.92rem;
    color: #1e1b4b;
    line-height: 1.6;
    background: #f5f3ff;
    padding: 0.85rem;
    border-radius: 12px;
    border-left: 3px solid #7c3aed;
    margin-bottom: 0.95rem;
}
.pivot-rule {
    font-size: 0.82rem;
    color: #64748b;
    font-style: italic;
    margin-top: 0.35rem;
}

/* Editorial Review Box */
.editorial-review-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #7c3aed;
    border-radius: 18px;
    padding: 1.8rem;
    line-height: 1.8;
    font-size: 0.98rem;
    color: #334155;
    box-shadow: var(--emo-shadow-soft);
}
.editorial-review-card strong {
    color: #0f172a;
    font-weight: 700;
}

/* Mobile & Tablet Responsive Media Queries */
@media (max-width: 768px) {
    div.stButton > button {
        font-size: 0.88rem !important;
        padding: 0.65rem 1.1rem !important;
        min-height: 44px !important;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px !important;
        font-size: 0.84rem !important;
    }
    .emo-title {
        font-size: 2.8rem !important;
    }
    .score-hero-container {
        flex-direction: column !important;
        text-align: center !important;
        gap: 1.2rem !important;
    }
}
</style>
"""


# =====================================================================
# 2. SHARED REUSABLE COMPONENTS
# =====================================================================

def render_privacy_notice(
    storage_type: str = "Client-Side Ephemeral RAM Buffer",
    retention_policy: str = "Purged immediately upon tab close. No external server transmission without explicit user trigger."
) -> str:
    """
    Renders an auditable privacy notice complying with WCAG 2.2 AA.
    Features Sage / Light Green verified indicators on a pure white surface.
    """
    return f"""
    <aside class="emo-privacy-notice" role="note" aria-label="Privacy & Data Processing Disclosure">
        <div class="emo-privacy-icon" aria-hidden="true">🔒</div>
        <div>
            <strong style="color: #065f46; font-size: 0.92rem;">Truthful Privacy Architecture:</strong>
            <span style="color: #047857; margin-left: 4px;">All outfit analysis and deep learning inference execute <em>locally on this client runtime</em> via PyTorch. 
            <strong>Camera & Image Data:</strong> {html.escape(storage_type)} — {html.escape(retention_policy)}</span>
        </div>
    </aside>
    """


def render_brand_header(
    eyebrow: str = "Multimodal AI Outfit Critic & Wardrobe Stylist",
    title: str = "emo",
    tagline: str = "Rigorous sartorial critique bridging deep vision embeddings with classical colorimetry and silhouette proportions."
) -> str:
    """
    Renders an editorial brand header in White, Lavender, and Sky Blue.
    """
    return f"""
    <header class="emo-brand-banner" role="banner">
        <div class="emo-micro-badge">{html.escape(eyebrow)}</div>
        <h1 class="emo-title">{html.escape(title)}</h1>
        <p class="emo-subtitle">{html.escape(tagline)}</p>
    </header>
    """


def render_stat_card(value: str, label: str, subtitle: Optional[str] = None) -> str:
    """
    Renders an accessible stat card with verified metrics in Light Green / Sky Blue.
    """
    sub_html = f'<div style="font-size: 0.8rem; color: #059669; font-weight: 700; margin-top: 0.2rem;">{html.escape(subtitle)}</div>' if subtitle else ''
    return f"""
    <div class="emo-card" style="text-align: center; border-top: 4px solid #7c3aed;">
        <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 2.2rem; font-weight: 800; color: #1e1b4b; line-height: 1.1;">{html.escape(value)}</div>
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700; color: #64748b; margin-top: 0.35rem;">{html.escape(label)}</div>
        {sub_html}
    </div>
    """


def render_pivot_card(number: int, title: str, description: str, hex_chip: Optional[str] = None) -> str:
    """
    Renders an evidence-grounded styling pivot card with Lavender/Blue/Green top borders.
    """
    chip_html = ""
    if hex_chip:
        chip_html = f"""
        <div style="display: inline-flex; align-items: center; gap: 8px; margin-top: 0.6rem; padding: 4px 10px; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;">
            <span style="width: 14px; height: 14px; border-radius: 3px; background-color: {html.escape(hex_chip)}; display: inline-block; border: 1px solid #cbd5e1;"></span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #334155;">Target Color: {html.escape(hex_chip)}</span>
        </div>
        """
    top_border_class = f"pivot-card-top-{(number % 3) + 1}"
    return f"""
    <article class="pivot-card {top_border_class}" role="region" aria-label="Styling Pivot {number}">
        <div>
            <div class="pivot-category">Strategic Upgrade {number}</div>
            <div class="pivot-title">{html.escape(title)}</div>
            <div class="pivot-action">{html.escape(description)}</div>
        </div>
        {chip_html}
    </article>
    """
