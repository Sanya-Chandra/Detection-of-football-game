"""
Sports Analytics CV — UI Components
Reusable Streamlit UI widgets for the sports analytics dashboard.
"""

import streamlit as st
import numpy as np
import cv2
from typing import Dict



def apply_custom_css() -> None:
    """Inject premium custom CSS for the sports analytics theme."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=Rajdhani:wght@400;500;600;700;900&family=Orbitron:wght@400;700;900&display=swap');

    /* ── Root Variables ───────────────────────────────── */
    :root {
        --primary:      #00FF87;
        --primary-dim:  rgba(0,255,135,0.15);
        --secondary:    #00D4FF;
        --amber:        #FFB300;
        --amber-dim:    rgba(255,179,0,0.15);
        --blue:         #0096FF;
        --blue-dim:     rgba(0,150,255,0.15);
        --red:          #FF4560;
        --red-dim:      rgba(255,69,96,0.15);
        --accent:       #FF6B35;
        --purple:       #a855f7;
        --bg-dark:      #060b14;
        --bg-mid:       #0d1520;
        --bg-card:      #0d1826;
        --bg-card2:     #101e30;
        --glass:        rgba(13,24,38,0.9);
        --text:         #e2e8f0;
        --text-muted:   #7a92b2;
        --border:       rgba(0,255,135,0.18);
        --border-soft:  rgba(255,255,255,0.06);
        --glow:         0 0 40px rgba(0,255,135,0.10);
    }

    /* ── Global ───────────────────────────────────────── */
    html, body, .stApp {
        background: radial-gradient(ellipse at 20% 0%, #0a1f1a 0%, #060b14 50%, #06091a 100%) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Remove default streamlit padding ─────────────── */
    .block-container {
        padding-top: 1rem !important;
        max-width: 1400px !important;
    }

    /* ── Headers ──────────────────────────────────────── */
    h1, h2, h3, h4 {
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 0.04em;
    }
    h1 { font-size: 2.6rem !important; font-weight: 900 !important;
         background: linear-gradient(135deg, #00FF87, #0096FF);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2 { font-size: 1.7rem !important; font-weight: 700 !important; color: var(--primary) !important; }
    h3 { font-size: 1.2rem !important; font-weight: 600 !important; color: #c8d8f0 !important; }

    /* ── Sidebar ──────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07101e 0%, #0d1a2e 60%, #0a1120 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebarNav"] { display: none !important; }

    /* ── Sidebar markdown & labels ────────────────────── */
    [data-testid="stSidebar"] .stMarkdown p { color: var(--text-muted) !important; font-size: 0.85rem; }
    [data-testid="stSidebar"] h3 { color: var(--primary) !important; font-size: 1rem !important; letter-spacing: 0.08em; }

    /* ── Tabs ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid var(--border-soft) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 8px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,255,135,0.2), rgba(0,150,255,0.15)) !important;
        color: var(--primary) !important;
        box-shadow: 0 0 12px rgba(0,255,135,0.2) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #00FF87 0%, #0096FF 100%) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1rem !important;
        letter-spacing: 0.04em !important;
        transition: all 0.25s ease !important;
        padding: 0.55rem 1.8rem !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 28px rgba(0,255,135,0.45) !important;
        filter: brightness(1.1) !important;
    }

    /* ── File Uploader ────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--bg-card) !important;
        border: 2px dashed rgba(0,255,135,0.25) !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        transition: border-color 0.3s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary) !important;
        box-shadow: var(--glow) !important;
    }

    /* ── Sliders ──────────────────────────────────────── */
    [data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }
    [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    }

    /* ── Progress bar ─────────────────────────────────── */
    .stProgress > div > div { background: linear-gradient(90deg, var(--primary), var(--secondary)) !important; }

    /* ── Metric Cards ─────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        box-shadow: var(--glow) !important;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
    [data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 800 !important; font-size: 1.8rem !important; }

    /* ── Alert/Info boxes ─────────────────────────────── */
    .stAlert {
        background: var(--bg-card2) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Dataframes ───────────────────────────────────── */
    .stDataFrame {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Selectbox ────────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div {
        background: var(--bg-card2) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }

    /* ── Checkbox ─────────────────────────────────────── */
    .stCheckbox > label > div:first-child {
        border-color: var(--primary) !important;
    }

    /* ── Scrollbar ────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: rgba(0,255,135,0.3); border-radius: 3px; }

    /* ── HERO BANNER ──────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #071523 0%, #0d2137 45%, #071220 100%);
        border: 1px solid rgba(0,255,135,0.25);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(0,255,135,0.08) 0%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(0,150,255,0.07) 0%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Orbitron', 'Rajdhani', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00FF87 0%, #00d4ff 50%, #0096FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.3rem 0;
        line-height: 1.1;
        letter-spacing: 0.03em;
    }
    .hero-subtitle {
        color: #8aa8c8;
        font-size: 1.05rem;
        margin: 0 0 1rem 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,255,135,0.1);
        color: #00FF87;
        border: 1px solid rgba(0,255,135,0.3);
        border-radius: 30px;
        padding: 0.25rem 0.9rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 0 10px rgba(0,255,135,0.1);
        transition: all 0.2s ease;
    }
    .hero-live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: var(--primary);
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px var(--primary);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }

    /* ── STAT CARD ────────────────────────────────────── */
    .stat-card {
        background: linear-gradient(145deg, #0d1826, #0a1420);
        border: 1px solid rgba(0,255,135,0.18);
        border-radius: 16px;
        padding: 1.3rem 1rem;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 36px rgba(0,255,135,0.18);
        border-color: rgba(0,255,135,0.38);
    }
    .stat-card-value {
        font-family: 'Orbitron', 'Rajdhani', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        line-height: 1;
        text-shadow: 0 0 22px rgba(0,255,135,0.45);
    }
    .stat-card-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        margin-top: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 600;
    }
    /* Speed card — amber */
    .stat-card.speed { border-color: rgba(255,179,0,0.25); }
    .stat-card.speed::before { background: linear-gradient(90deg, transparent, var(--amber), transparent); }
    .stat-card.speed .stat-card-value { color: var(--amber); text-shadow: 0 0 22px rgba(255,179,0,0.4); }
    .stat-card.speed:hover { box-shadow: 0 14px 36px rgba(255,179,0,0.18); border-color: rgba(255,179,0,0.4); }
    /* Possession card — blue */
    .stat-card.poss { border-color: rgba(0,150,255,0.25); }
    .stat-card.poss::before { background: linear-gradient(90deg, transparent, var(--blue), transparent); }
    .stat-card.poss .stat-card-value { color: var(--blue); text-shadow: 0 0 22px rgba(0,150,255,0.4); }
    .stat-card.poss:hover { box-shadow: 0 14px 36px rgba(0,150,255,0.18); border-color: rgba(0,150,255,0.4); }
    /* Ball card — red */
    .stat-card.ball { border-color: rgba(255,69,96,0.25); }
    .stat-card.ball::before { background: linear-gradient(90deg, transparent, var(--red), transparent); }
    .stat-card.ball .stat-card-value { color: var(--red); text-shadow: 0 0 22px rgba(255,69,96,0.4); }
    .stat-card.ball:hover { box-shadow: 0 14px 36px rgba(255,69,96,0.18); border-color: rgba(255,69,96,0.4); }

    /* ── WELCOME MODAL ────────────────────────────────── */
    .welcome-overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(6,11,20,0.92);
        backdrop-filter: blur(12px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.4s ease;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }

    .welcome-card {
        background: linear-gradient(145deg, #0d1a2e, #111c2d);
        border: 1px solid rgba(0,255,135,0.3);
        border-radius: 24px;
        padding: 3rem 3.5rem;
        max-width: 580px;
        width: 90%;
        text-align: center;
        position: relative;
        box-shadow: 0 30px 100px rgba(0,0,0,0.7), 0 0 60px rgba(0,255,135,0.08);
        animation: slideUp 0.5s ease;
    }
    .welcome-card::before {
        content: '';
        position: absolute;
        top: 0; left: 10%; right: 10%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
        border-radius: 2px;
    }
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        filter: drop-shadow(0 0 20px rgba(0,255,135,0.5));
    }
    .welcome-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00FF87, #0096FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .welcome-subtitle {
        color: #7a92b2;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    .welcome-features {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-bottom: 2rem;
        text-align: left;
    }
    .welcome-feature-item {
        background: rgba(0,255,135,0.06);
        border: 1px solid rgba(0,255,135,0.15);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        font-size: 0.82rem;
        color: #c0d4e8;
    }
    .welcome-feature-item span { color: var(--primary); margin-right: 6px; }
    .welcome-btn {
        background: linear-gradient(135deg, #00FF87, #0096FF);
        color: #000 !important;
        border: none;
        border-radius: 12px;
        padding: 0.85rem 3rem;
        font-size: 1rem;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 0.06em;
        cursor: pointer;
        transition: all 0.25s ease;
        width: 100%;
        text-transform: uppercase;
        box-shadow: 0 4px 30px rgba(0,255,135,0.35);
    }
    .welcome-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 40px rgba(0,255,135,0.5);
        filter: brightness(1.08);
    }

    /* ── Sidebar logo card ────────────────────────────── */
    .sidebar-logo {
        background: linear-gradient(145deg, rgba(0,255,135,0.08), rgba(0,150,255,0.06));
        border: 1px solid rgba(0,255,135,0.2);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar-logo-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00FF87, #0096FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.04em;
    }
    .sidebar-logo-sub {
        color: #4a6a8a;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .sidebar-device-badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .sidebar-device-badge.gpu {
        background: rgba(0,255,135,0.12);
        color: #00FF87;
        border: 1px solid rgba(0,255,135,0.3);
    }
    .sidebar-device-badge.cpu {
        background: rgba(150,150,150,0.1);
        color: #94a3b8;
        border: 1px solid rgba(150,150,150,0.2);
    }
    .sidebar-section-header {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #3a5a7a !important;
        margin: 1rem 0 0.4rem 0;
        padding-left: 0.2rem;
    }
    .sidebar-stat {
        background: rgba(13,25,45,0.7);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.82rem;
    }
    .sidebar-stat .label { color: #4a6a8a; flex: 1; }
    .sidebar-stat .value { color: #c8d8f0; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)



def hero_banner() -> None:
    """Render the premium top hero banner focused on the 6 key showcase metrics."""
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">⚽ Sports Analytics CV</p>
        <p class="hero-subtitle">
            <span class="hero-live-dot"></span>
            AI-powered player tracking &amp; real-time sports intelligence
        </p>
        <span class="hero-badge">🎯 Player Tracking</span>
        <span class="hero-badge">⚡ Speed &amp; Distance</span>
        <span class="hero-badge">🔥 Heatmaps</span>
        <span class="hero-badge">⚽ Ball Trail</span>
        <span class="hero-badge">🤝 Possession</span>
        <span class="hero-badge">⚡ GPU Ready</span>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(config: dict) -> None:
    """Render the rich premium sidebar."""
    with st.sidebar:
        # Logo card
        try:
            import torch
            is_gpu = torch.cuda.is_available()
            device_label = "🟢 GPU · CUDA" if is_gpu else "⚪ CPU Mode"
            badge_class = "gpu" if is_gpu else "cpu"
        except Exception:
            device_label = "⚪ CPU Mode"
            badge_class = "cpu"

        st.markdown(f"""
        <div class="sidebar-logo">
            <div style="font-size:2.2rem; margin-bottom:0.4rem; filter:drop-shadow(0 0 10px rgba(0,255,135,0.5));">⚽</div>
            <div class="sidebar-logo-title">Sports Analytics CV</div>
            <div class="sidebar-logo-sub">v1.0 · AI Vision Platform</div>
            <span class="sidebar-device-badge {badge_class}">{device_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # System Stats section
        st.markdown('<div class="sidebar-section-header">⚡ System</div>', unsafe_allow_html=True)
        try:
            import torch
            cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—"
            st.markdown(f"""
            <div class="sidebar-stat">
                <span class="label">GPU</span>
                <span class="value">{gpu_name[:18]}</span>
            </div>
            <div class="sidebar-stat">
                <span class="label">CUDA</span>
                <span class="value">{cuda_ver}</span>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

        model_name = config.get('detection', {}).get('model', 'yolov8n.pt')
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="label">Model</span>
            <span class="value">{model_name}</span>
        </div>
        """, unsafe_allow_html=True)

        # Quick links
        st.markdown('<div class="sidebar-section-header">🔗 Quick Links</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:0.4rem;">
            <a href="storage/" target="_blank" style="
                background: rgba(0,150,255,0.08);
                border: 1px solid rgba(0,150,255,0.2);
                border-radius:8px; padding:0.45rem 0.8rem;
                color:#60a0d0; text-decoration:none; font-size:0.82rem;
                display:flex; align-items:center; gap:6px;">
                📖 Documentation
            </a>
            <a href="storage/" target="_blank" style="
                background: rgba(0,255,135,0.06);
                border: 1px solid rgba(0,255,135,0.15);
                border-radius:8px; padding:0.45rem 0.8rem;
                color:#00cc70; text-decoration:none; font-size:0.82rem;
                display:flex; align-items:center; gap:6px;">
                📂 Output Files
            </a>
        </div>
        """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.05);
                    text-align:center;">
            <div style="color:#2a4a6a; font-size:0.65rem; letter-spacing:0.05em; text-transform:uppercase;">
                Built with YOLOv8 · MediaPipe · Streamlit
            </div>
            <div style="color:#1a3a5a; font-size:0.62rem; margin-top:3px;">© 2025 Sports Analytics CV</div>
        </div>
        """, unsafe_allow_html=True)


def stat_card(value: str, label: str) -> str:
    """Return HTML for a single premium stat card."""
    return f"""
    <div class="stat-card">
        <div class="stat-card-value">{value}</div>
        <div class="stat-card-label">{label}</div>
    </div>
    """


def render_stat_row(stats: Dict[str, str], variants: Dict[str, str] = None) -> None:
    """Render a row of stat cards. variants maps label -> css class (speed/poss/ball)"""
    variants = variants or {}
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        css_class = variants.get(label, "")
        col.markdown(
            f'<div class="stat-card {css_class}">'
            f'<div class="stat-card-value">{value}</div>'
            f'<div class="stat-card-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV frame to RGB for Streamlit display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def display_frame(frame: np.ndarray, caption: str = "", use_container_width: bool = True) -> None:
    """Display a BGR frame in Streamlit (converts to RGB automatically)."""
    rgb = bgr_to_rgb(frame)
    st.image(rgb, caption=caption, use_container_width=use_container_width)


def loading_spinner(message: str = "Analyzing..."):
    """Context manager — Streamlit spinner wrapper."""
    return st.spinner(message)


def progress_bar_context(total: int, label: str = "Processing frames..."):
    """Return a configured Streamlit progress bar."""
    bar = st.progress(0, text=label)
    return bar
