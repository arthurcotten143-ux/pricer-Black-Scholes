"""
Black-Scholes Pricer — Streamlit
Compatible GitHub Codespaces / navigateur
Lancer avec : streamlit run streamlit_bs_pricer.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BS Pricer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .block-container { padding-top: 1rem; }
    [data-testid="stSidebar"] { background-color: #ffffff; }
    h1, h2, h3 { color: #1b5e20; font-family: monospace; font-weight: bold; }
    p, span, div { color: #000000; }
    .metric-label { color: #1b5e20 !important; font-size: 0.75rem !important; font-weight: bold !important; }
    .stMetric { background-color: #ffffff; border-radius: 8px; padding: 8px; border: 1px solid #c8e6c9; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="metric-container"] label {
        color: #1b5e20 !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: bold !important;
    }
    .stMarkdown { color: #000000; }
</style>
""", unsafe_allow_html=True)

# ─── STYLE MATPLOTLIB ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f5f5f5",
    "axes.edgecolor":   "#1b5e20",
    "axes.labelcolor":  "#1b5e20",
    "text.color":       "#000000",
    "xtick.color":      "#000000",
    "ytick.color":      "#000000",
    "grid.color":       "#c8e6c9",
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
})

BG     = "#ffffff"
PANEL  = "#f5f5f5"
BORDER = "#1b5e20"
ACCENT = "#2e7d32"
GREEN  = "#2e7d32"
RED    = "#c62828"
YELLOW = "#f57c00"
PURPLE = "#6a1b9a"
CYAN   = "#00838f"
ORANGE = "#ef6c00"
GRAY   = "#424242"
TEXT   = "#000000"
TITLE  = "#1b5e20"

# ─── BLACK-SCHOLES ────────────────────────────────────────────────────────────

def bs(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10:
        return max(S-K, 0) if opt=="call" else max(K-S, 0)
    if sigma <= 1e-10:
        return max(S*np.exp(-q*T)-K*np.exp(-r*T), 0) if opt=="call" \
               else max(K*np.exp(-r*T)-S*np.exp(-q*T), 0)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def greeks(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return {k: 0.0 for k in ["delta","gamma","vega","theta","rho"]}
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.pdf(d1)
    if opt == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*norm.cdf(d2)
                 + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)
                 - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = np.exp(-q*T)*nd1 / (S*sigma*np.sqrt(T))
    vega  = S*np.exp(-q*T)*nd1*np.sq
