"""
AI Project Intelligence & Risk Advisor — Shared Modern UI Design System
Soft, colorful, airy, rounded aesthetic with Inter typography and custom components.
"""

import streamlit as st


def inject_css():
    """Inject modern, customized CSS into the Streamlit app."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Reset Streamlit Defaults & Set Background ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #1F2340 !important;
    }

    .stApp {
        background: linear-gradient(135deg, #F6F4FF 0%, #EAF4FF 100%) !important;
        background-attachment: fixed !important;
    }

    /* Hide default Streamlit header, footer, decoration & reduce padding */
    #MainMenu, footer, header, div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1350px !important;
    }

    /* ── Top Navigation Bar & App Brand ── */
    .brand-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0 1.2rem 0;
        margin-bottom: 0.5rem;
    }
    .brand-logo-text {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #7C5CFC 0%, #5B8DEF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .brand-badge {
        background: #EDEAFE;
        color: #7C5CFC;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid #D8D2FD;
    }

    /* ── Modern Top Pill Navigation ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF !important;
        border-radius: 30px !important;
        padding: 6px 8px !important;
        gap: 6px !important;
        border: 1px solid #ECE9FB !important;
        box-shadow: 0 4px 20px rgba(124, 92, 252, 0.06) !important;
        display: flex !important;
        justify-content: flex-start !important;
        margin-bottom: 1.5rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 24px !important;
        color: #7A7F9A !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 8px 20px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #7C5CFC !important;
        background: #F6F4FF !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7C5CFC 0%, #5B8DEF 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(124, 92, 252, 0.3) !important;
    }

    /* ── Modern Cards & Containers ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border-radius: 16px !important;
        border: 1px solid #ECE9FB !important;
        box-shadow: 0 4px 20px rgba(124, 92, 252, 0.06) !important;
        padding: 1.2rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 28px rgba(124, 92, 252, 0.1) !important;
    }

    .ui-card {
        background: #FFFFFF;
        border-radius: 16px;
        border: 1px solid #ECE9FB;
        box-shadow: 0 4px 20px rgba(124, 92, 252, 0.06);
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .ui-card:hover {
        box-shadow: 0 8px 28px rgba(124, 92, 252, 0.1);
        transform: translateY(-2px);
    }

    /* ── KPI Metric Cards ── */
    .kpi-wrapper {
        background: #FFFFFF;
        border-radius: 16px;
        border: 1px solid #ECE9FB;
        box-shadow: 0 4px 20px rgba(124, 92, 252, 0.06);
        padding: 1.1rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: all 0.2s ease;
    }
    .kpi-wrapper:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(124, 92, 252, 0.12);
    }
    .kpi-icon-circle {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
    }
    .kpi-icon-violet { background: #EDEAFE; color: #7C5CFC; }
    .kpi-icon-coral  { background: #FDE8E9; color: #EF6B73; }
    .kpi-icon-amber  { background: #FEF3DD; color: #F5B547; }
    .kpi-icon-mint   { background: #E6F8F1; color: #34C38F; }

    .kpi-info { flex: 1; }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #7A7F9A;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .kpi-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #1F2340;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #7A7F9A;
        margin-top: 3px;
    }

    /* ── Typography & Headers ── */
    .page-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #1F2340;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 0.92rem;
        color: #7A7F9A;
        font-weight: 400;
        margin-bottom: 1.4rem;
    }

    /* ── Semantic Badges ── */
    .badge-coral  { background: #FDE8E9; color: #EF6B73; border: 1px solid #FCD0D3; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }
    .badge-amber  { background: #FEF3DD; color: #D98200; border: 1px solid #FDE3B6; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }
    .badge-mint   { background: #E6F8F1; color: #1E8E62; border: 1px solid #BFF1DE; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }
    .badge-violet { background: #EDEAFE; color: #6952D4; border: 1px solid #D8D2FD; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }
    .badge-grey   { background: #F1F3F7; color: #7A7F9A; border: 1px solid #E1E4EC; padding: 3px 10px; border-radius: 12px; font-size: 0.76rem; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid #ECE9FB !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C5CFC 0%, #5B8DEF 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(124, 92, 252, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(124, 92, 252, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #1F2340 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #F6F4FF !important;
        border-color: #7C5CFC !important;
        color: #7C5CFC !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 2px dashed #D8D2FD !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #7C5CFC !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        border: 1px solid #ECE9FB !important;
        overflow: hidden !important;
        box-shadow: 0 2px 10px rgba(124, 92, 252, 0.04) !important;
    }

    /* ── Modern Fallback Table (When PyArrow is blocked by OS Application Control) ── */
    .modern-table-container {
        width: 100%;
        overflow-x: auto;
        background: #FFFFFF;
        border-radius: 14px;
        border: 1px solid #ECE9FB;
        box-shadow: 0 4px 16px rgba(124, 92, 252, 0.05);
        margin: 0.8rem 0;
    }
    .modern-data-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: #1F2340;
    }
    .modern-data-table thead th {
        background: #F6F4FF;
        color: #5B5875;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 2px solid #ECE9FB;
    }
    .modern-data-table tbody td {
        padding: 12px 16px;
        border-bottom: 1px solid #F0EDFA;
        vertical-align: middle;
        line-height: 1.45;
    }
    .modern-data-table tbody tr:last-child td {
        border-bottom: none;
    }
    .modern-data-table tbody tr:hover td {
        background-color: #FAF9FF;
    }

    /* ── Narrative Highlight Cards (for Goals & Scope summaries) ── */
    .narrative-card {
        background: #FFFFFF;
        border: 1px solid #ECE9FB;
        border-left: 4px solid #7C5CFC;
        border-radius: 12px;
        padding: 1.15rem 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 10px rgba(124, 92, 252, 0.04);
    }
    .narrative-card-header {
        font-weight: 700;
        color: #7C5CFC;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .narrative-card p {
        font-size: 0.93rem;
        line-height: 1.65;
        color: #2D3154;
        margin-bottom: 0.65rem;
    }
    .narrative-card p:last-child {
        margin-bottom: 0;
    }


    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #1F2340 !important;
        border: 1px solid #ECE9FB !important;
    }
    .streamlit-expanderContent {
        background: #FFFFFF !important;
        border-radius: 0 0 12px 12px !important;
        border: 1px solid #ECE9FB !important;
        border-top: none !important;
    }

    /* ── Rounded Progress Bars ── */
    .stProgress > div > div > div > div {
        border-radius: 8px !important;
        background: linear-gradient(90deg, #7C5CFC, #5B8DEF) !important;
    }
    .stProgress > div > div {
        border-radius: 8px !important;
        background-color: #EDEAFE !important;
        height: 10px !important;
    }

    /* ── Empty State Card ── */
    .empty-state {
        text-align: center;
        padding: 3.5rem 2rem;
        background: #FFFFFF;
        border-radius: 20px;
        border: 1px solid #ECE9FB;
        box-shadow: 0 4px 24px rgba(124, 92, 252, 0.07);
        max-width: 600px;
        margin: 2rem auto;
    }
    .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        display: inline-block;
    }

    /* ── AI Chat Floating Widget ── */
    .chat-floating-btn {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 58px;
        height: 58px;
        border-radius: 29px;
        background: linear-gradient(135deg, #7C5CFC 0%, #5B8DEF 100%);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        box-shadow: 0 8px 24px rgba(124, 92, 252, 0.35);
        cursor: pointer;
        z-index: 9999;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .chat-floating-btn:hover {
        transform: scale(1.06);
        box-shadow: 0 12px 30px rgba(124, 92, 252, 0.45);
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #EDEAFE !important;
        border: 1px solid #D8D2FD !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #FFFFFF !important;
        border: 1px solid #ECE9FB !important;
        box-shadow: 0 2px 8px rgba(124, 92, 252, 0.04) !important;
    }

    /* Source chip in chat */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #F6F4FF;
        color: #7C5CFC;
        padding: 3px 9px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #D8D2FD;
        margin-right: 4px;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
