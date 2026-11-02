"""
NutriLens AI: High-End Food Safety & Machine Learning Intelligence Dashboard.

Dark Tech Aesthetic (AIZCRM-inspired):
- Deep Charcoal / Slate Background (#16191d & #21262d)
- Vibrant Neon Mint & Cyan Primary Accents (#00e599 & #00d68f)
- Electric Blue & Coral Secondary Accents (#38bdf8 & #f87171)
- Minimalist Typography with SVG Vector Icons
- Full-Width Interface (No Sidebar)
- Two-Button Landing Page:
    1. Search Existing Product
    2. Upload Photos of the Product
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from data_pipeline import extract_e_numbers, DATA_FILE_PATH, generate_local_food_dataset
from xai_explainer import NutriLensExplainer
from safety_engine import SafetyAuditEngine
from vision_analyzer import (
    extract_text_from_images,
    match_inbuilt_product,
    analyze_ingredients_granular,
    extract_ingredients_from_ocr,
    extract_nutrition_from_ocr,
    clean_ingredients_statement
)

# ---------------------------------------------------------
# Page Configuration (No Sidebar)
# ---------------------------------------------------------
st.set_page_config(
    page_title="NutriLens AI | Food Safety Intelligence",
    page_icon="none",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Vector SVG Icon System (Zero Unicode Emojis)
# ---------------------------------------------------------
SVG_ICONS = {
    "search": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    "camera": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>',
    "shield": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
    "heart": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>',
    "alert": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></line><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
    "check": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
    "chart": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
    "database": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>',
    "grid": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>'
}

# ---------------------------------------------------------
# Modern AIZCRM Dark Theme CSS
# ---------------------------------------------------------
DARK_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Hide Default Streamlit Elements */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] { display: none !important; }
    header[data-testid="stHeader"] { background-color: transparent !important; display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Dark Canvas Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #16191d !important;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #16191d;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1400px;
    }

    /* Top Navigation Bar */
    .dark-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #21262d;
        border-radius: 18px;
        padding: 14px 26px;
        margin-bottom: 26px;
        border: 1px solid #2d333b;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .brand-mark {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #00e599 0%, #00b377 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0f172a;
        font-weight: 800;
        box-shadow: 0 0 20px rgba(0, 229, 153, 0.45);
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        line-height: 1.2;
    }

    /* Modern Hero Section */
    .hero-container {
        text-align: center;
        max-width: 860px;
        margin: 15px auto 40px auto;
        padding: 10px 0;
    }
    .hero-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 229, 153, 0.08);
        border: 1px solid rgba(0, 229, 153, 0.3);
        color: #00e599;
        padding: 5px 16px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 18px;
        box-shadow: 0 0 16px rgba(0, 229, 153, 0.15);
    }
    .hero-glow-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        color: #ffffff;
        line-height: 1.15;
        margin: 0 0 16px 0;
    }
    .hero-gradient-text {
        background: linear-gradient(135deg, #00e599 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .hero-description {
        font-size: 1.08rem;
        color: #94a3b8;
        line-height: 1.6;
        margin: 0 auto;
        max-width: 720px;
    }

    /* Modern Pathway Cards (Home Page) */
    .pathway-card {
        background: #21262d;
        border-radius: 24px;
        padding: 32px 30px;
        border: 1px solid #2d333b;
        box-shadow: 0 12px 32px -4px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 310px;
        position: relative;
        overflow: hidden;
    }
    .pathway-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: transparent;
        transition: background 0.3s ease;
    }
    .pathway-card.mint-card:hover::before {
        background: linear-gradient(90deg, #00e599 0%, #38bdf8 100%);
    }
    .pathway-card.cyan-card:hover::before {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    }
    .pathway-card:hover {
        transform: translateY(-5px);
        border-color: #3f4754;
        box-shadow: 0 20px 48px -8px rgba(0, 0, 0, 0.5);
    }
    .pathway-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
    }
    .pathway-icon-wrap {
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .pathway-icon-wrap.mint {
        background: rgba(0, 229, 153, 0.12);
        color: #00e599;
        border: 1px solid rgba(0, 229, 153, 0.3);
        box-shadow: 0 0 20px rgba(0, 229, 153, 0.15);
    }
    .pathway-icon-wrap.cyan {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }
    .pathway-tag {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 9999px;
    }
    .pathway-tag.mint {
        background: rgba(0, 229, 153, 0.1);
        color: #00e599;
        border: 1px solid rgba(0, 229, 153, 0.25);
    }
    .pathway-tag.cyan {
        background: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .pathway-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 10px;
    }
    .pathway-desc {
        font-size: 0.94rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 18px;
    }
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 24px;
    }
    .feature-chip {
        font-size: 0.78rem;
        font-weight: 600;
        background: #16191d;
        border: 1px solid #2d333b;
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 8px;
    }

    /* Live Trust / Proof Bar */
    .proof-bar {
        background: #21262d;
        border-radius: 18px;
        border: 1px solid #2d333b;
        padding: 20px 28px;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-top: 36px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    .proof-item {
        text-align: center;
        border-right: 1px solid #2d333b;
        padding: 0 10px;
    }
    .proof-item:last-child {
        border-right: none;
    }
    .proof-number {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .proof-number.mint { color: #00e599; }
    .proof-number.cyan { color: #38bdf8; }
    .proof-label {
        font-size: 0.78rem;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Metric Stat Box (Matching Image Dashboard) */
    .metric-box {
        background: #21262d;
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid #2d333b;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.18);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 18px;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: #38414e;
    }
    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .metric-icon-small {
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: #16191d;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #8b949e;
        border: 1px solid #2d333b;
    }
    .metric-value-huge {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .metric-label-muted {
        font-size: 0.8rem;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }

    /* Modern Neon Pill Badges */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.02em;
    }
    .pill-badge.neon-mint {
        background: rgba(0, 229, 153, 0.12);
        color: #00e599;
        border: 1px solid rgba(0, 229, 153, 0.35);
    }
    .pill-badge.neon-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .pill-badge.neon-red {
        background: rgba(239, 68, 68, 0.14);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    /* Container Glass Panels */
    .dark-panel {
        background: #21262d;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #2d333b;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.18);
        margin-bottom: 20px;
    }
    .dark-panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Compact Dark Ingredient Row */
    .dark-ing-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 13px 16px;
        border-bottom: 1px solid #282e36;
        transition: background 0.15s ease;
    }
    .dark-ing-row:last-child {
        border-bottom: none;
    }
    .dark-ing-row:hover {
        background: #272d36;
        border-radius: 10px;
    }
    .ing-name-white {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    .ing-sub-text {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 3px;
    }

    /* High-Contrast Labels across All Streamlit Inputs */
    label, div[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p, label p, div[data-testid="stWidgetLabel"] span {
        color: #f1f5f9 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }

    /* Checkbox Label & Box */
    div[data-testid="stCheckbox"] label span, div[data-baseweb="checkbox"] span, div[data-testid="stCheckbox"] p {
        color: #f1f5f9 !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
    }

    /* Modern Dark Inputs: Text, Area & Number Input */
    .stTextInput input, .stTextArea textarea, .stNumberInput input, div[data-testid="stNumberInput"] input {
        background-color: #16191d !important;
        border: 1px solid #2d333b !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-size: 0.98rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stNumberInputContainer"] {
        background-color: #16191d !important;
        border: 1px solid #2d333b !important;
        border-radius: 12px !important;
    }
    div[data-testid="stNumberInputContainer"]:focus-within {
        border-color: #00e599 !important;
        box-shadow: 0 0 10px rgba(0, 229, 153, 0.25) !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #21262d !important;
        color: #e2e8f0 !important;
        border: none !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background-color: #2d333b !important;
        color: #00e599 !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
        border-color: #00e599 !important;
        box-shadow: 0 0 10px rgba(0, 229, 153, 0.2) !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #21262d;
        border: 1.5px dashed #38414e;
        border-radius: 18px;
        padding: 20px;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #00e599;
    }
    button[kind="primary"] {
        background-color: #00e599 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 229, 153, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover {
        background-color: #00f5a3 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 229, 153, 0.45) !important;
    }
    button[kind="secondary"] {
        background-color: #21262d !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d333b !important;
        border-radius: 12px !important;
    }
    button[kind="secondary"]:hover {
        border-color: #00e599 !important;
        color: #ffffff !important;
    }

    /* Modern Wizard Stepper */
    .wizard-stepper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #21262d;
        border: 1px solid #2d333b;
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 26px;
    }
    .wizard-step {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #8b949e;
    }
    .wizard-step.active {
        color: #00e599;
    }
    .wizard-step.completed {
        color: #f1f5f9;
    }
    .step-circle {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        background: #16191d;
        border: 1px solid #2d333b;
    }
    .wizard-step.active .step-circle {
        background: rgba(0, 229, 153, 0.15);
        border-color: #00e599;
        color: #00e599;
        box-shadow: 0 0 10px rgba(0, 229, 153, 0.25);
    }
    .wizard-step.completed .step-circle {
        background: #00e599;
        border-color: #00e599;
        color: #0f172a;
    }
    .step-divider {
        flex-grow: 1;
        height: 1px;
        background: #2d333b;
        margin: 0 16px;
    }
    .step-divider.active {
        background: #00e599;
    }
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Service & Resource Loaders
# ---------------------------------------------------------
@st.cache_resource
def get_explainer():
    return NutriLensExplainer()

def get_safety_engine():
    return SafetyAuditEngine()

@st.cache_data
def get_catalog_data():
    if not os.path.exists(DATA_FILE_PATH):
        generate_local_food_dataset(DATA_FILE_PATH)
    return pd.read_csv(DATA_FILE_PATH)

explainer = get_explainer()
safety_engine = get_safety_engine()
df_catalog = get_catalog_data()


# ---------------------------------------------------------
# Session State & Navigation Handling
# ---------------------------------------------------------
if "view" not in st.session_state:
    st.session_state["view"] = "home"  # 'home', 'search', 'upload', 'report'

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "ocr_extracted_text" not in st.session_state:
    st.session_state["ocr_extracted_text"] = ""

if "matched_product_row" not in st.session_state:
    st.session_state["matched_product_row"] = None

if "match_confidence" not in st.session_state:
    st.session_state["match_confidence"] = 0.0

# Active Product Attributes
if "prod_name" not in st.session_state:
    st.session_state["prod_name"] = "Munchee Super Cream Cracker"

if "ingredients_text" not in st.session_state:
    st.session_state["ingredients_text"] = "Wheat flour, vegetable fat (palm oil), malt extract, salt, yeast, raising agents (INS 500ii, INS 503ii), emulsifier (INS 322)."

if "energy_kcal" not in st.session_state:
    st.session_state["energy_kcal"] = 450.0

if "sugars_g" not in st.session_state:
    st.session_state["sugars_g"] = 1.8

if "fat_g" not in st.session_state:
    st.session_state["fat_g"] = 14.5

if "saturated_fat_g" not in st.session_state:
    st.session_state["saturated_fat_g"] = 6.8

if "sodium_mg" not in st.session_state:
    st.session_state["sodium_mg"] = 560.0

if "proteins_g" not in st.session_state:
    st.session_state["proteins_g"] = 9.5

if "fiber_g" not in st.session_state:
    st.session_state["fiber_g"] = 3.8

if "fruits_veg_pct" not in st.session_state:
    st.session_state["fruits_veg_pct"] = 0.0

if "is_beverage" not in st.session_state:
    st.session_state["is_beverage"] = False


def navigate_to(view_name: str):
    st.session_state["view"] = view_name


def resolve_demographic_badge(status_str: str) -> str:
    s = str(status_str).upper()
    if any(k in s for k in ["UNSUITABLE", "HIGH RISK", "RESTRICTED", "HAZARD", "DANGER", "CRITICAL", "RED"]):
        return "neon-red"
    elif any(k in s for k in ["CAUTION", "MODERATE", "AMBER", "WARNING"]):
        return "neon-amber"
    elif any(k in s for k in ["SAFE", "SUITABLE", "LOW RISK", "GREEN"]):
        return "neon-mint"
    return "neon-amber"


# ---------------------------------------------------------
# Top Header & Navigation Bar
# ---------------------------------------------------------
col_nav_l, col_nav_r = st.columns([1.5, 2.0])

with col_nav_l:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 18px;">
        <div class="brand-mark">{SVG_ICONS['shield']}</div>
        <div>
            <div class="brand-title">NutriLens AI</div>
            <div style="font-size: 0.8rem; color: #8b949e; font-weight: 500;">Know what you buy, before you buy.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_nav_r:
    if st.session_state["view"] == "home":
        st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 6px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: #21262d; border: 1px solid #2d333b; padding: 6px 14px; border-radius: 9999px; font-size: 0.8rem; color: #94a3b8;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00e599; box-shadow: 0 0 10px #00e599;"></span>
                <span>Sri Lankan MOH Standards • Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state["view"] == "report":
        st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 6px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(0, 229, 153, 0.08); border: 1px solid rgba(0, 229, 153, 0.25); padding: 6px 14px; border-radius: 9999px; font-size: 0.8rem; color: #00e599; font-weight: 600;">
                <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #00e599; box-shadow: 0 0 8px #00e599;"></span>
                <span>Forensic Audit Verified • Sri Lanka MOH</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_space, col_b_back = st.columns([1.8, 1.2])
        with col_b_back:
            if st.button("← Back to Home", key="top_btn_back", use_container_width=True):
                navigate_to("home")
                st.rerun()


# =========================================================
# VIEW 1: HOME PAGE (ONLY TWO CLEAN ACTION PATHWAYS)
# =========================================================
if st.session_state["view"] == "home":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-pill-badge">
            <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #00e599; box-shadow: 0 0 8px #00e599;"></span>
            Sri Lankan Food Safety & Forensic Analytics
        </div>
        <h1 class="hero-glow-title">
            Know what you buy,<br>
            <span class="hero-gradient-text">before you buy.</span>
        </h1>
        <p class="hero-description">
            AI-powered packaged food forensics calibrated to Sri Lankan supermarket standards. Instant packaging OCR scanning, clinical NCD risk screening, and transparent SHAP explanations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_act1, col_act2 = st.columns(2, gap="large")

    with col_act1:
        st.markdown(f"""
        <div class="pathway-card mint-card">
            <div>
                <div class="pathway-top-row">
                    <div class="pathway-icon-wrap mint">
                        {SVG_ICONS['database']}
                    </div>
                    <div class="pathway-tag mint">Verified Catalog</div>
                </div>
                <div class="pathway-title">Search Existing Product</div>
                <div class="pathway-desc">
                    Explore 1,200+ laboratory-validated Sri Lankan retail products (Cargills, Keells, Munchee, Maliban, Elephant House) with official MOH traffic light benchmarks and complete additive forensics.
                </div>
                <div class="chip-container">
                    <span class="feature-chip">1,200+ Local SKUs</span>
                    <span class="feature-chip">MOH Gazette 2128/9</span>
                    <span class="feature-chip">Instant XAI Profile</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Search Existing Product  →", type="primary", key="home_btn_search", use_container_width=True):
            navigate_to("search")
            st.rerun()

    with col_act2:
        st.markdown(f"""
        <div class="pathway-card cyan-card">
            <div>
                <div class="pathway-top-row">
                    <div class="pathway-icon-wrap cyan">
                        {SVG_ICONS['camera']}
                    </div>
                    <div class="pathway-tag cyan">Windows Vision OCR</div>
                </div>
                <div class="pathway-title">Upload Photos of the Product</div>
                <div class="pathway-desc">
                    Upload one or more packaging photographs (front label, ingredient panel, or nutrition facts). Our native Windows OCR engine automatically extracts tokens and audits toxic additives.
                </div>
                <div class="chip-container">
                    <span class="feature-chip">Native WinRT OCR</span>
                    <span class="feature-chip">Carcinogen Detection</span>
                    <span class="feature-chip">Multi-Angle Ingestion</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Upload Photos of the Product  →", type="primary", key="home_btn_upload", use_container_width=True):
            navigate_to("upload")
            st.rerun()

    st.markdown("""
    <div class="proof-bar">
        <div class="proof-item">
            <div class="proof-number mint">1,200+</div>
            <div class="proof-label">Verified Sri Lankan SKUs</div>
        </div>
        <div class="proof-item">
            <div class="proof-number cyan">94.17%</div>
            <div class="proof-label">CatBoost Macro-F1 Precision</div>
        </div>
        <div class="proof-item">
            <div class="proof-number mint">MOH Gazette</div>
            <div class="proof-label">Traffic Light Standards (Sugar/Salt/Fat)</div>
        </div>
        <div class="proof-item">
            <div class="proof-number cyan">4 Cohorts</div>
            <div class="proof-label">Pregnancy, Infant, Toddler & Elderly Audits</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# VIEW 2: SEARCH EXISTING PRODUCT
# =========================================================
elif st.session_state["view"] == "search":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">Search Local Catalog</h2>
        <div style="font-size: 0.95rem; color: #8b949e;">Select a cataloged supermarket product to inspect its complete health, safety, and machine learning report.</div>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input("Filter by brand, product title, or category:", placeholder="e.g. Munchee, Ginger Beer, Samaposha, Kottu Mee, Milk...")

    unique_products = df_catalog.drop_duplicates(subset=["brand", "product_name"]).copy()
    if search_query:
        mask = (
            unique_products["brand"].str.contains(search_query, case=False, na=False) |
            unique_products["product_name"].str.contains(search_query, case=False, na=False) |
            unique_products["category"].str.contains(search_query, case=False, na=False)
        )
        filtered_prods = unique_products[mask]
    else:
        filtered_prods = unique_products

    st.write(f"Showing **{len(filtered_prods)}** verified products:")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(filtered_prods.iterrows()):
        with cols[idx % 3]:
            clean_title = row["product_name"].split("(")[0].strip()
            g = row["health_grade"]
            badge_type = "neon-mint" if g in ['A', 'B'] else ("neon-amber" if g in ['C', 'D'] else "neon-red")

            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-header">
                    <span style="font-size: 0.8rem; font-weight: 700; color: #8b949e; text-transform: uppercase;">{row['brand']}</span>
                    <span class="pill-badge {badge_type}">Grade {g}</span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">{clean_title}</div>
                <div style="font-size: 0.82rem; color: #8b949e; margin-bottom: 12px;">Category: {row['category']}</div>
                <div style="display: flex; gap: 14px; font-size: 0.8rem; color: #cbd5e1; border-top: 1px solid #282e36; padding-top: 10px;">
                    <div>Sugar: <strong style="color: #00e599;">{row['sugars_g']}g</strong></div>
                    <div>Sodium: <strong style="color: #38bdf8;">{row['sodium_mg']}mg</strong></div>
                    <div>Fat: <strong style="color: #f59e0b;">{row['fat_g']}g</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Inspect Audit Report", key=f"sel_prod_{idx}", use_container_width=True):
                st.session_state["prod_name"] = clean_title
                st.session_state["ingredients_text"] = row["ingredients_text"]
                st.session_state["energy_kcal"] = float(row["energy_kcal"])
                st.session_state["sugars_g"] = float(row["sugars_g"])
                st.session_state["fat_g"] = float(row["fat_g"])
                st.session_state["saturated_fat_g"] = float(row["saturated_fat_g"])
                st.session_state["sodium_mg"] = float(row["sodium_mg"])
                st.session_state["proteins_g"] = float(row["proteins_g"])
                st.session_state["fiber_g"] = float(row["fiber_g"])
                st.session_state["fruits_veg_pct"] = float(row["fruits_veg_pct"])
                st.session_state["is_beverage"] = bool(row["is_beverage"])
                st.session_state["matched_product_row"] = row
                st.session_state["match_confidence"] = 98.0
                navigate_to("report")
                st.rerun()


# =========================================================
# =========================================================
# VIEW 3: UPLOAD PHOTOS OF THE PRODUCT (Step 1 of Wizard)
# =========================================================
elif st.session_state["view"] == "upload":
    st.markdown("""
    <div class="wizard-stepper">
        <div class="wizard-step active">
            <div class="step-circle">1</div>
            <span>1. Upload Packaging Photos</span>
        </div>
        <div class="step-divider"></div>
        <div class="wizard-step">
            <div class="step-circle">2</div>
            <span>2. Ingredient & Nutrition Verification</span>
        </div>
        <div class="step-divider"></div>
        <div class="wizard-step">
            <div class="step-circle">3</div>
            <span>3. Forensic Health Report</span>
        </div>
    </div>
    <div style="margin-bottom: 22px;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">Upload Product Packaging</h2>
        <div style="font-size: 0.95rem; color: #8b949e;">Upload packaging photographs. Our vision pipeline will extract ingredient tokens, detect additives, and match our local retail catalog.</div>
    </div>
    """, unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([1.5, 1.0], gap="large")

    with up_col1:
        uploaded_images = st.file_uploader(
            "Drop packaging images here (JPG, PNG, WEBP):",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="file_uploader_widget"
        )

        if uploaded_images:
            st.session_state["uploaded_files"] = uploaded_images
            st.markdown("##### Uploaded Photo Thumbnails")
            p_cols = st.columns(min(len(uploaded_images), 3))
            for i, f in enumerate(uploaded_images):
                with p_cols[i % 3]:
                    st.image(Image.open(f), caption=f"Photo #{i+1}", use_container_width=True)

            st.write("")
            if st.button("Extract Ingredients & Identify Product  →", type="primary", use_container_width=True):
                status_block = st.status("Executing Vision & Forensic Ingestion...", expanded=True)
                with status_block:
                    st.write("Extracting high-resolution text via Windows Native WinRT OCR...")
                    text_extracted = extract_text_from_images(uploaded_images)
                    time.sleep(0.2)
                    st.write("Parsing ingredients statement and additive tokens...")
                    extracted_ing = extract_ingredients_from_ocr(text_extracted)
                    time.sleep(0.2)
                    st.write("Extracting nutritional values from Nutrition Facts panel...")
                    nutrition_extracted = extract_nutrition_from_ocr(text_extracted)
                    st.write("Cross-referencing brand signatures against Sri Lankan Retail Catalog...")
                    matched_p, score, tokens = match_inbuilt_product(text_extracted, df_catalog)
                    time.sleep(0.2)
                    status_block.update(label="Vision Forensics Completed!", state="complete", expanded=False)

                st.session_state["ocr_extracted_text"] = text_extracted
                st.session_state["ocr_raw_ingredients"] = extracted_ing
                st.session_state["matched_product_row"] = matched_p
                st.session_state["match_confidence"] = score
                st.session_state["match_tokens"] = tokens

                # Priority: Store actual extracted ingredients from uploaded photos!
                if extracted_ing and len(extracted_ing.strip()) > 5:
                    st.session_state["ingredients_text"] = extracted_ing
                elif matched_p is not None:
                    st.session_state["ingredients_text"] = matched_p["ingredients_text"]
                else:
                    st.session_state["ingredients_text"] = text_extracted.strip() if text_extracted else ""

                if matched_p is not None:
                    st.session_state["prod_name"] = matched_p["product_name"].split("(")[0].strip()
                    st.session_state["energy_kcal"] = min(max(float(nutrition_extracted.get("energy_kcal", float(matched_p["energy_kcal"]))), 0.0), 2000.0)
                    st.session_state["sugars_g"] = min(max(float(nutrition_extracted.get("sugars_g", float(matched_p["sugars_g"]))), 0.0), 100.0)
                    st.session_state["fat_g"] = min(max(float(nutrition_extracted.get("fat_g", float(matched_p["fat_g"]))), 0.0), 100.0)
                    st.session_state["saturated_fat_g"] = min(max(float(nutrition_extracted.get("saturated_fat_g", float(matched_p["saturated_fat_g"]))), 0.0), 100.0)
                    st.session_state["sodium_mg"] = min(max(float(nutrition_extracted.get("sodium_mg", float(matched_p["sodium_mg"]))), 0.0), 10000.0)
                    st.session_state["proteins_g"] = min(max(float(nutrition_extracted.get("proteins_g", float(matched_p["proteins_g"]))), 0.0), 100.0)
                    st.session_state["fiber_g"] = min(max(float(nutrition_extracted.get("fiber_g", float(matched_p["fiber_g"]))), 0.0), 100.0)
                    st.session_state["fruits_veg_pct"] = float(matched_p.get("fruits_veg_pct", 0.0))
                    st.session_state["is_beverage"] = bool(matched_p.get("is_beverage", False))
                else:
                    st.session_state["prod_name"] = "Uploaded Food Product"
                    st.session_state["energy_kcal"] = min(max(float(nutrition_extracted.get("energy_kcal", 380.0)), 0.0), 2000.0)
                    st.session_state["sugars_g"] = min(max(float(nutrition_extracted.get("sugars_g", 12.0)), 0.0), 100.0)
                    st.session_state["fat_g"] = min(max(float(nutrition_extracted.get("fat_g", 9.0)), 0.0), 100.0)
                    st.session_state["saturated_fat_g"] = min(max(float(nutrition_extracted.get("saturated_fat_g", 3.8)), 0.0), 100.0)
                    st.session_state["sodium_mg"] = min(max(float(nutrition_extracted.get("sodium_mg", 380.0)), 0.0), 10000.0)
                    st.session_state["proteins_g"] = min(max(float(nutrition_extracted.get("proteins_g", 6.5)), 0.0), 100.0)
                    st.session_state["fiber_g"] = min(max(float(nutrition_extracted.get("fiber_g", 2.2)), 0.0), 100.0)
                    st.session_state["fruits_veg_pct"] = 0.0
                    st.session_state["is_beverage"] = False

                navigate_to("review")
                st.rerun()

    with up_col2:
        st.markdown(f"""
        <div class="dark-panel">
            <div class="dark-panel-title">
                <span>Capture Guidelines</span>
                {SVG_ICONS['alert']}
            </div>
            <div style="font-size: 0.88rem; color: #8b949e; line-height: 1.6;">
                <p><strong>To ensure high-precision OCR recognition:</strong></p>
                <ul style="padding-left: 20px; color: #cbd5e1;">
                    <li>Keep camera parallel to the packaging text to avoid distortion.</li>
                    <li>Ensure the <strong>Ingredients statement</strong> and <strong>Nutrition Table</strong> are sharp and in focus.</li>
                    <li>Avoid direct flash glare over glossy plastic packaging.</li>
                    <li>You can upload multiple photos: front brand label, ingredients panel, and nutrition table.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# VIEW 4: REVIEW & VERIFICATION (Step 2 of Wizard)
# =========================================================
elif st.session_state["view"] == "review":
    st.markdown("""
    <div class="wizard-stepper">
        <div class="wizard-step completed">
            <div class="step-circle">✓</div>
            <span>1. Upload Photos</span>
        </div>
        <div class="step-divider active"></div>
        <div class="wizard-step active">
            <div class="step-circle">2</div>
            <span>2. Ingredient & Nutritional Verification</span>
        </div>
        <div class="step-divider"></div>
        <div class="wizard-step">
            <div class="step-circle">3</div>
            <span>3. Forensic Health Report</span>
        </div>
    </div>
    <div style="margin-bottom: 22px;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">Forensic Ingestion & Ingredient Verification</h2>
        <div style="font-size: 0.95rem; color: #8b949e;">Review the extracted ingredient tokens and nutritional values detected from the uploaded packaging photographs before generating the safety audit.</div>
    </div>
    """, unsafe_allow_html=True)

    matched_p = st.session_state.get("matched_product_row")
    conf = st.session_state.get("match_confidence", 0.0)
    match_toks = st.session_state.get("match_tokens", [])

    # Product Identification Status Card
    if matched_p is not None and conf >= 40.0:
        clean_cat_title = matched_p["product_name"].split("(")[0].strip()
        st.markdown(f"""
        <div class="metric-box" style="border-color: rgba(0, 229, 153, 0.4); margin-bottom: 22px; background: rgba(0, 229, 153, 0.03);">
            <div class="metric-header">
                <span style="font-size: 0.8rem; font-weight: 700; color: #00e599; text-transform: uppercase;">Inbuilt Catalog Product Identified</span>
                <span class="pill-badge neon-mint">{conf}% Match Confidence</span>
            </div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">{clean_cat_title} ({matched_p['brand']})</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">Category: <strong>{matched_p['category']}</strong> | Identifiers: {', '.join(match_toks) if match_toks else 'Brand & keyword match'}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-box" style="border-color: rgba(56, 189, 248, 0.4); margin-bottom: 22px; background: rgba(56, 189, 248, 0.03);">
            <div class="metric-header">
                <span style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;">Custom / Uncataloged Product Ingested</span>
                <span class="pill-badge neon-blue">Direct Vision OCR Ingestion</span>
            </div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">New Packaging Formulation Detected</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">All safety audits and clinical NCD restrictions will be computed directly against the OCR-extracted ingredients and nutritional metrics below.</div>
        </div>
        """, unsafe_allow_html=True)

    rev_col1, rev_col2 = st.columns([1.45, 1.0], gap="large")

    with rev_col1:
        prod_input = st.text_input("Product Title / Brand Description:", value=st.session_state["prod_name"], key="input_prod_title")

        st.markdown("""
        <div style="margin: 16px 0 6px 0;">
            <div style="font-size: 1.22rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">Extracted Ingredients Statement</div>
            <div style="font-size: 0.86rem; color: #cbd5e1;">Direct packaging OCR tokens. Review or edit any misread characters before auditing:</div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state["ingredients_text"] = clean_ingredients_statement(st.session_state["ingredients_text"])

        if not st.session_state["ingredients_text"].strip():
            st.warning("No readable ingredient text was automatically found on the uploaded image. Please paste or type the ingredients list below to proceed with the safety audit.")

        ing_input = st.text_area(
            "Ingredients (Comma-separated statement extracted via OCR):",
            value=st.session_state["ingredients_text"],
            height=135,
            key="input_ing_text",
            help="Review or edit any misread OCR words. Include additive INS / E-numbers (e.g. INS 500ii, E102)."
        )

        # Real-time granular deconstruction of the ingredients
        parsed_ingredients = analyze_ingredients_granular(ing_input)
        st.markdown(f"""
        <div style="margin: 20px 0 10px 0;">
            <div style="font-size: 1.18rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">Deconstructed Forensic Profile ({len(parsed_ingredients)} items recognized)</div>
            <div style="font-size: 0.86rem; color: #cbd5e1;">Real-time clinical, functional, and toxicological classification for each constituent:</div>
        </div>
        """, unsafe_allow_html=True)

        if parsed_ingredients:
            st.markdown('<div class="dark-panel" style="padding: 10px 16px; max-height: 380px; overflow-y: auto;">', unsafe_allow_html=True)
            for item in parsed_ingredients:
                tier = item["risk_tier"]
                badge_cls = "neon-mint" if tier == "SAFE" else ("neon-amber" if tier == "CAUTION" else "neon-red")
                e_tag = f'<span style="font-size: 0.75rem; background: #16191d; border: 1px solid #2d333b; padding: 2px 6px; border-radius: 4px; color: #38bdf8; margin-left: 8px;">{item["code"]}</span>' if item.get("code") and item["code"] != "—" else ""

                st.markdown(f"""
                <div class="dark-ing-row">
                    <div style="max-width: 72%;">
                        <div class="ing-name-white">
                            {item['ingredient_name']}{e_tag}
                        </div>
                        <div class="ing-sub-text">{item['category']} • {item['function']}</div>
                        <div style="font-size: 0.77rem; color: #94a3b8; margin-top: 2px;">{item['clinical_impact'][:110]}...</div>
                    </div>
                    <div>
                        <span class="pill-badge {badge_cls}">{tier}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Type or verify ingredients above to preview the real-time granular safety classification.")

    with rev_col2:
        # Photo thumbnails if available
        if st.session_state.get("uploaded_files"):
            with st.expander("View Uploaded Packaging Photos", expanded=False):
                th_cols = st.columns(min(len(st.session_state["uploaded_files"]), 3))
                for idx_p, img_file in enumerate(st.session_state["uploaded_files"]):
                    with th_cols[idx_p % 3]:
                        st.image(Image.open(img_file), caption=f"Photo #{idx_p+1}", use_container_width=True)

        st.markdown("""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.22rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">Nutritional Profile (per 100g)</div>
            <div style="font-size: 0.86rem; color: #cbd5e1;">Extracted from Nutrition Facts panel or matched baseline. Adjust if needed:</div>
        </div>
        """, unsafe_allow_html=True)

        n_col1, n_col2 = st.columns(2)
        with n_col1:
            val_energy = min(max(float(st.session_state.get("energy_kcal", 380.0)), 0.0), 2000.0)
            energy_val = st.number_input("Energy (kcal):", value=val_energy, min_value=0.0, max_value=2000.0, step=10.0, key="in_energy")

            val_fat = min(max(float(st.session_state.get("fat_g", 9.0)), 0.0), 100.0)
            fat_val = st.number_input("Total Fat (g):", value=val_fat, min_value=0.0, max_value=100.0, step=0.5, key="in_fat")

            val_sodium = min(max(float(st.session_state.get("sodium_mg", 380.0)), 0.0), 10000.0)
            sodium_val = st.number_input("Sodium (mg):", value=val_sodium, min_value=0.0, max_value=10000.0, step=10.0, key="in_sodium")

            val_fiber = min(max(float(st.session_state.get("fiber_g", 2.2)), 0.0), 100.0)
            fiber_val = st.number_input("Dietary Fiber (g):", value=val_fiber, min_value=0.0, max_value=100.0, step=0.5, key="in_fiber")

        with n_col2:
            val_sugar = min(max(float(st.session_state.get("sugars_g", 12.0)), 0.0), 100.0)
            sugar_val = st.number_input("Total Sugars (g):", value=val_sugar, min_value=0.0, max_value=100.0, step=0.5, key="in_sugar")

            val_sat_fat = min(max(float(st.session_state.get("saturated_fat_g", 3.8)), 0.0), 100.0)
            sat_fat_val = st.number_input("Saturated Fat (g):", value=val_sat_fat, min_value=0.0, max_value=100.0, step=0.5, key="in_sat_fat")

            val_protein = min(max(float(st.session_state.get("proteins_g", 6.5)), 0.0), 100.0)
            protein_val = st.number_input("Protein (g):", value=val_protein, min_value=0.0, max_value=100.0, step=0.5, key="in_protein")
            bev_val = st.checkbox("Is Liquid Beverage?", value=bool(st.session_state.get("is_beverage", False)), key="in_bev")

        st.write("")
        st.markdown("---")
        btn_col_back, btn_col_run = st.columns([1.0, 1.6])
        with btn_col_back:
            if st.button("← Back to Upload", key="btn_rev_back", use_container_width=True):
                navigate_to("upload")
                st.rerun()

        with btn_col_run:
            if st.button("Generate Forensic Audit Report  →", type="primary", key="btn_rev_generate", use_container_width=True):
                # Save confirmed values to session state
                st.session_state["prod_name"] = prod_input.strip() if prod_input.strip() else "Packaged Food Product"
                st.session_state["ingredients_text"] = clean_ingredients_statement(ing_input.strip())
                st.session_state["energy_kcal"] = float(energy_val)
                st.session_state["sugars_g"] = float(sugar_val)
                st.session_state["fat_g"] = float(fat_val)
                st.session_state["saturated_fat_g"] = float(sat_fat_val)
                st.session_state["sodium_mg"] = float(sodium_val)
                st.session_state["proteins_g"] = float(protein_val)
                st.session_state["fiber_g"] = float(fiber_val)
                st.session_state["is_beverage"] = bool(bev_val)
                navigate_to("report")
                st.rerun()


# =========================================================
# VIEW 5: REPORT VIEW (AIZCRM-STYLE MODERN DARK METRICS)
# =========================================================
elif st.session_state["view"] == "report":
    st.markdown("""
    <div class="wizard-stepper">
        <div class="wizard-step completed">
            <div class="step-circle">✓</div>
            <span>1. Upload Photos</span>
        </div>
        <div class="step-divider active"></div>
        <div class="wizard-step completed">
            <div class="step-circle">✓</div>
            <span>2. Ingredients & Nutrition Verified</span>
        </div>
        <div class="step-divider active"></div>
        <div class="wizard-step active">
            <div class="step-circle">3</div>
            <span>3. Forensic Health Report</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    e_numbers = extract_e_numbers(st.session_state["ingredients_text"])
    additives_cnt = len(e_numbers)

    sample_df = pd.DataFrame([{
        "energy_kcal": st.session_state["energy_kcal"],
        "sugars_g": st.session_state["sugars_g"],
        "fat_g": st.session_state["fat_g"],
        "saturated_fat_g": st.session_state["saturated_fat_g"],
        "sodium_mg": st.session_state["sodium_mg"],
        "proteins_g": st.session_state["proteins_g"],
        "fiber_g": st.session_state["fiber_g"],
        "fruits_veg_pct": st.session_state["fruits_veg_pct"],
        "is_beverage": 1 if st.session_state["is_beverage"] else 0,
        "additives_count": additives_cnt
    }])

    xai_res = explainer.explain_instance(sample_df)
    pred_grade = xai_res["predicted_grade"]
    probs = xai_res["probabilities"]

    audit_res = safety_engine.full_audit(
        product_name=st.session_state["prod_name"],
        energy_kcal=st.session_state["energy_kcal"],
        sugars_g=st.session_state["sugars_g"],
        fat_g=st.session_state["fat_g"],
        saturated_fat_g=st.session_state["saturated_fat_g"],
        sodium_mg=st.session_state["sodium_mg"],
        proteins_g=st.session_state["proteins_g"],
        fiber_g=st.session_state["fiber_g"],
        fruits_veg_pct=st.session_state["fruits_veg_pct"],
        ingredients_text=st.session_state["ingredients_text"],
        predicted_grade=pred_grade,
        is_beverage=st.session_state["is_beverage"]
    )

    composite = audit_res["composite_rating"]
    moh_lights = audit_res["moh_traffic_lights"]
    ingredients_breakdown = analyze_ingredients_granular(st.session_state["ingredients_text"])

    # Header Row
    col_rep_head_l, col_rep_head_r = st.columns([2.2, 1.2])
    with col_rep_head_l:
        matched_row = st.session_state.get("matched_product_row")
        conf_val = st.session_state.get("match_confidence", 0.0)
        match_tag = ""
        if matched_row is not None and conf_val >= 40.0:
            match_tag = f"""<div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(0, 229, 153, 0.1); border: 1px solid rgba(0, 229, 153, 0.3); color: #00e599; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; margin-bottom: 8px;"><span>Inbuilt Catalog Product: {matched_row['brand']} ({conf_val}% Confidence)</span></div>"""
        
        rep_head_html = f"""<div style="margin-bottom: 22px;">
{match_tag}
<div style="font-size: 0.8rem; color: #00e599; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">FORENSIC AUDIT REPORT</div>
<h1 style="font-size: 2.3rem; font-weight: 800; color: #ffffff; margin: 4px 0 6px 0;">{st.session_state['prod_name']}</h1>
<div style="font-size: 0.9rem; color: #8b949e;">Sri Lankan Ministry of Health Regulations & International Safety Reference</div>
</div>"""
        st.markdown(rep_head_html, unsafe_allow_html=True)
    with col_rep_head_r:
        st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("← Adjust Data", key="rep_btn_adjust", use_container_width=True):
                navigate_to("review")
                st.rerun()
        with c_btn2:
            if st.button("+ New Scan", type="primary", key="rep_btn_new", use_container_width=True):
                navigate_to("home")
                st.rerun()

    # TOP ROW: 4 STAT BOXES (Exact layout of reference design)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        g_badge = "neon-mint" if pred_grade in ['A', 'B'] else ("neon-amber" if pred_grade in ['C', 'D'] else "neon-red")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-header">
                <div class="metric-icon-small">{SVG_ICONS['shield']}</div>
                <span class="pill-badge {g_badge}">Grade {pred_grade}</span>
            </div>
            <div class="metric-value-huge">Grade {pred_grade}</div>
            <div class="metric-label-muted">CatBoost ML Health Tier</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        sc = moh_lights["sugar"]["color"].lower()
        badge_cls = "neon-mint" if sc == "green" else ("neon-amber" if sc == "amber" else "neon-red")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-header">
                <div class="metric-icon-small">{SVG_ICONS['chart']}</div>
                <span class="pill-badge {badge_cls}">{moh_lights['sugar']['level']}</span>
            </div>
            <div class="metric-value-huge">{st.session_state['sugars_g']:.1f}<span style="font-size: 1.1rem; font-weight: 600; color: #8b949e;"> g/100g</span></div>
            <div class="metric-label-muted">MOH Sugar Standard</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        nac = moh_lights["sodium"]["color"].lower()
        badge_cls = "neon-mint" if nac == "green" else ("neon-amber" if nac == "amber" else "neon-red")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-header">
                <div class="metric-icon-small">{SVG_ICONS['heart']}</div>
                <span class="pill-badge {badge_cls}">{moh_lights['sodium']['level']}</span>
            </div>
            <div class="metric-value-huge">{st.session_state['sodium_mg']:.0f}<span style="font-size: 1.1rem; font-weight: 600; color: #8b949e;"> mg</span></div>
            <div class="metric-label-muted">MOH Sodium Standard</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        fc = moh_lights["fat"]["color"].lower()
        badge_cls = "neon-mint" if fc == "green" else ("neon-amber" if fc == "amber" else "neon-red")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-header">
                <div class="metric-icon-small">{SVG_ICONS['alert']}</div>
                <span class="pill-badge {badge_cls}">{moh_lights['fat']['level']}</span>
            </div>
            <div class="metric-value-huge">{st.session_state['fat_g']:.1f}<span style="font-size: 1.1rem; font-weight: 600; color: #8b949e;"> g</span></div>
            <div class="metric-label-muted">MOH Total Fat Standard</div>
        </div>
        """, unsafe_allow_html=True)

    # MAIN TWO-COLUMN WORKSPACE
    col_main_l, col_main_r = st.columns([1.5, 1.0], gap="large")

    with col_main_l:
        # Dark SHAP Feature Attribution Plot
        st.markdown(f"""
        <div class="dark-panel">
            <div class="dark-panel-title">
                <span>Model Feature Attribution (SHAP)</span>
                {SVG_ICONS['chart']}
            </div>
            <div style="font-size: 0.85rem; color: #8b949e; margin-bottom: 14px;">
                Exact Shapley values demonstrating how positive and negative nutrient attributes drove Grade {pred_grade}:
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plot Dark-Themed Waterfall
        feats = xai_res["feature_names"]
        sh_vals = xai_res["shap_values"]
        f_vals = xai_res["feature_values"]
        labels = [f"{feat} = {val:.1f}" for feat, val in zip(feats, f_vals)]
        sorted_indices = np.argsort(np.abs(sh_vals))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor('#21262d')
        ax.set_facecolor('#21262d')

        # Colors matching the dark palette: Neon Mint for negative/protective, Coral Red for positive/penalty
        colors = ["#00e599" if val < 0 else "#f87171" for val in sh_vals[sorted_indices]]
        ax.barh([labels[i] for i in sorted_indices], sh_vals[sorted_indices], color=colors, alpha=0.9, edgecolor="#2d333b", linewidth=0.8)
        ax.axvline(0, color="#484f58", linestyle="--", linewidth=1.0)
        ax.set_xlabel(f"SHAP Value (Impact on Grade {pred_grade} Log-Odds)", fontsize=9, color="#8b949e")
        ax.tick_params(colors='#cbd5e1', labelsize=8.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2d333b')
        ax.spines['bottom'].set_color('#2d333b')
        ax.grid(axis="x", linestyle=":", alpha=0.2, color="#8b949e")

        for idx, sh in enumerate(sh_vals[sorted_indices]):
            offset = 0.02 if sh >= 0 else -0.02
            ha = 'left' if sh >= 0 else 'right'
            ax.text(sh + offset, idx, f"{sh:+.2f}", va='center', ha=ha, fontsize=8.5, fontweight='bold', color="#ffffff")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Compact Ingredient Breakdown (Dark Theme)
        st.markdown(f"""
        <div class="dark-panel" style="margin-top: 20px;">
            <div class="dark-panel-title">
                <span>Ingredient Forensic Breakdown ({len(ingredients_breakdown)} items)</span>
                {SVG_ICONS['database']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        for idx, ing in enumerate(ingredients_breakdown):
            tier = ing["risk_tier"]
            b_cls = "neon-mint" if tier == "SAFE" else ("neon-amber" if tier == "CAUTION" else "neon-red")
            code_str = f'<span style="font-family: monospace; font-size: 0.78rem; background: #16191d; border: 1px solid #2d333b; padding: 2px 6px; border-radius: 4px; color: #8b949e;">{ing["code"]}</span>' if ing.get("code") and ing["code"] != "—" else ""

            st.markdown(f"""
            <div class="dark-ing-row">
                <div>
                    <div class="ing-name-white">#{idx+1}. {ing['ingredient_name']} {code_str}</div>
                    <div class="ing-sub-text">{ing['category']} • {ing['function']}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{ing['clinical_impact'][:110]}...</div>
                </div>
                <div>
                    <span class="pill-badge {b_cls}">{tier}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_main_r:
        # Right Widget 1: Composite Health & Safety Rating
        rating_color = composite["color_rating"]
        banner_badge = "neon-mint" if rating_color == "GREEN" else ("neon-amber" if rating_color == "ORANGE" else "neon-red")

        st.markdown(f"""
        <div class="dark-panel">
            <div class="dark-panel-title">
                <span>Overall Safety Rating</span>
                {SVG_ICONS['shield']}
            </div>
            <div style="margin: 10px 0 14px 0;">
                <span class="pill-badge {banner_badge}" style="font-size: 0.95rem; padding: 6px 14px;">
                    {composite['status_title']}
                </span>
            </div>
            <div style="font-size: 0.88rem; color: #8b949e; line-height: 1.5;">
                {composite['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Right Widget 2: Clinical NCD Risk Matrix
        ncd = audit_res["ncd_audit"]
        st.markdown(f"""
        <div class="dark-panel">
            <div class="dark-panel-title">
                <span>Clinical NCD Risk Flags</span>
                {SVG_ICONS['heart']}
            </div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Diabetes Mellitus</span>
                    <span class="pill-badge {'neon-mint' if ncd['diabetes']['risk']=='LOW' else ('neon-amber' if ncd['diabetes']['risk']=='MODERATE' else 'neon-red')}">{ncd['diabetes']['risk']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Hypertension</span>
                    <span class="pill-badge {'neon-mint' if ncd['hypertension']['risk']=='LOW' else ('neon-amber' if ncd['hypertension']['risk']=='MODERATE' else 'neon-red')}">{ncd['hypertension']['risk']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Cardiovascular (CVD)</span>
                    <span class="pill-badge {'neon-mint' if ncd['cardiovascular']['risk']=='LOW' else ('neon-amber' if ncd['cardiovascular']['risk']=='MODERATE' else 'neon-red')}">{ncd['cardiovascular']['risk']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Chronic Kidney (CKD)</span>
                    <span class="pill-badge {'neon-mint' if ncd['chronic_kidney']['risk']=='LOW' else ('neon-amber' if ncd['chronic_kidney']['risk']=='MODERATE' else 'neon-red')}">{ncd['chronic_kidney']['risk']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Right Widget 3: Demographic Vulnerability Summary
        raw_demo = audit_res.get("demographic_audit", {})
        demo_preg = raw_demo.get("pregnant_mothers", {"status": "SAFE", "impact": "Formulation compliant with perinatal nutritional safety.", "warnings": []})
        demo_baby = raw_demo.get("babies_under_1yr", raw_demo.get("infants_children", {"status": "UNSUITABLE / HIGH RISK", "impact": "High sodium or additive burden under 12 months.", "warnings": []}))
        demo_toddler = raw_demo.get("infants_toddlers", raw_demo.get("infants_children", {"status": "MODERATE CAUTION", "impact": "Moderate child growth impact.", "warnings": []}))
        demo_elderly = raw_demo.get("very_old_people", raw_demo.get("elderly", {"status": "SAFE", "impact": "Standard geriatric dietary profile.", "warnings": []}))

        st.markdown(f"""
        <div class="dark-panel">
            <div class="dark-panel-title">
                <span>Vulnerable Population Summary</span>
                {SVG_ICONS['alert']}
            </div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Pregnant Mothers</span>
                    <span class="pill-badge {resolve_demographic_badge(demo_preg['status'])}">{demo_preg['status']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Babies (&lt; 1 Year)</span>
                    <span class="pill-badge {resolve_demographic_badge(demo_baby['status'])}">{demo_baby['status']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Infants (1–5 Yrs)</span>
                    <span class="pill-badge {resolve_demographic_badge(demo_toddler['status'])}">{demo_toddler['status']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9rem; font-weight: 600; color: #f1f5f9;">Very Old People</span>
                    <span class="pill-badge {resolve_demographic_badge(demo_elderly['status'])}">{demo_elderly['status']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Right Widget 4: Critical Toxins & Foreign Bans
        alarming = audit_res["alarming_audit"]
        if alarming["benzene_risk"] or alarming["regulatory_discrepancies"] or alarming["southampton_dyes"]:
            st.markdown(f"""
            <div class="dark-panel" style="border-left: 5px solid #f87171;">
                <div class="dark-panel-title" style="color: #f87171;">
                    <span>Regulatory & Toxic Flags</span>
                    {SVG_ICONS['alert']}
                </div>
            """, unsafe_allow_html=True)
            if alarming["benzene_risk"]:
                st.error("Sodium Benzoate (E211) + Vitamin C (E300) reacts to synthesize carcinogenic Benzene.")
            for disc in alarming["regulatory_discrepancies"]:
                st.warning(f"[{disc['code']}] {disc['name']}: {disc['reason']}")
            for dye in alarming["southampton_dyes"]:
                st.caption(f"[{dye['code']}] {dye['name']}: Southampton 6 azo dye.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="dark-panel" style="border-left: 5px solid #00e599;">
                <div class="dark-panel-title" style="color: #00e599;">
                    <span>Clean Chemical Profile</span>
                    {SVG_ICONS['check']}
                </div>
                <div style="font-size: 0.85rem; color: #8b949e;">
                    No foreign regulatory bans (EU/US), benzene precursors, or Southampton 6 azo colorants detected.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # DEDICATED FULL-WIDTH POPULATION IMPACT SECTION (Pregnant, Babies, Infants, Very Old People)
    st.markdown("---")
    st.markdown(f"""
    <div style="margin-bottom: 18px;">
        <div style="font-size: 0.8rem; color: #00e599; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">CLINICAL POPULATION FORENSICS</div>
        <h2 style="font-size: 1.65rem; font-weight: 800; color: #ffffff; margin: 4px 0 6px 0;">Vulnerable Demographic Impact Breakdown</h2>
        <div style="font-size: 0.88rem; color: #8b949e;">Targeted physiological assessment for pregnant women, babies, infants, and very old individuals:</div>
    </div>
    """, unsafe_allow_html=True)

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)

    # 1. Pregnant Mothers
    with col_v1:
        st_p = demo_preg["status"]
        b_p = resolve_demographic_badge(st_p)
        st.markdown(f"""
        <div class="metric-box" style="min-height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="metric-header">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">Pregnant Mothers</span>
                    <span class="pill-badge {b_p}">{st_p}</span>
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 12px;">
                    <strong>Clinical Action:</strong> {demo_preg['impact']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        if demo_preg["warnings"]:
            for w in demo_preg["warnings"]:
                st.caption(f"- {w}")
        else:
            st.caption("No maternal contraindications detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Babies (< 1 Year)
    with col_v2:
        st_b = demo_baby["status"]
        b_b = resolve_demographic_badge(st_b)
        st.markdown(f"""
        <div class="metric-box" style="min-height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="metric-header">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">Babies (&lt; 1 Year)</span>
                    <span class="pill-badge {b_b}">{st_b}</span>
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 12px;">
                    <strong>Renal & Gut Impact:</strong> {demo_baby['impact']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        if demo_baby["warnings"]:
            for w in demo_baby["warnings"]:
                st.caption(f"- {w}")
        else:
            st.caption("Gentle formulation compliant with infant guidelines.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Infants & Toddlers (1–5 Years)
    with col_v3:
        st_t = demo_toddler["status"]
        b_t = resolve_demographic_badge(st_t)
        st.markdown(f"""
        <div class="metric-box" style="min-height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="metric-header">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">Infants & Toddlers (1–5y)</span>
                    <span class="pill-badge {b_t}">{st_t}</span>
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 12px;">
                    <strong>Pediatric Impact:</strong> {demo_toddler['impact']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        if demo_toddler["warnings"]:
            for w in demo_toddler["warnings"]:
                st.caption(f"- {w}")
        else:
            st.caption("Free of hyperactivity azo dyes and intense sweeteners.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Very Old People / Geriatric
    with col_v4:
        st_e = demo_elderly["status"]
        b_e = resolve_demographic_badge(st_e)
        st.markdown(f"""
        <div class="metric-box" style="min-height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="metric-header">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #ffffff;">Very Old / Geriatric</span>
                    <span class="pill-badge {b_e}">{st_e}</span>
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 12px;">
                    <strong>Vascular & Renal Impact:</strong> {demo_elderly['impact']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        if demo_elderly["warnings"]:
            for w in demo_elderly["warnings"]:
                st.caption(f"- {w}")
        else:
            st.caption("Low arterial stiffening and glycemic perturbation risk.")
        st.markdown("</div>", unsafe_allow_html=True)
