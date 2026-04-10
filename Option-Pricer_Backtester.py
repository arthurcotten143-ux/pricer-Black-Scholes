"""
Options Pricer — Streamlit
Run with: streamlit run Option-Pricer_Clean.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Options Pricer", page_icon="○", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "app"

# Minimal styling
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { font-size: 1.4rem; font-weight: 600; }
    h2 { font-size: 1.1rem; font-weight: 500; }
    h3 { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# Chart style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#333",
    "text.color": "#333",
    "xtick.color": "#333",
    "ytick.color": "#333",
    "grid.color": "#ddd",
    "grid.linewidth": 0.5,
    "font.size": 8,
})

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

def bs(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10:
        return max(S-K, 0) if opt == "call" else max(K-S, 0)
    if sigma <= 1e-10:
        return max(S*np.exp(-q*T)-K*np.exp(-r*T), 0) if opt == "call" else max(K*np.exp(-r*T)-S*np.exp(-q*T), 0)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def greeks(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return {k: 0.0 for k in ["delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.pdf(d1)
    if opt == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = np.exp(-q*T)*nd1 / (S*sigma*np.sqrt(T))
    vega = S*np.exp(-q*T)*nd1*np.sqrt(T) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

def prob_itm(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return 1.0 if (opt == "call" and S > K) or (opt == "put" and S < K) else 0.0
    d2 = (np.log(S/K) + (r-q-0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d2) if opt == "call" else norm.cdf(-d2)

def skewed_iv(base_iv, S, K, T, skew_slope, skew_convexity, term_slope):
    moneyness = 1.0 - K / S
    term_adj = T - 0.25
    return max(base_iv + skew_slope * moneyness + skew_convexity * (moneyness ** 2) + term_slope * term_adj, 0.01)

@st.cache_data(ttl=300)
def monte_carlo_pricer(S, K, T, r, sigma, q, opt, n_sims, n_steps, seed):
    np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.standard_normal((n_sims, n_steps))
    S_T = S * np.exp(np.sum((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
    payoffs = np.maximum(S_T-K, 0) if opt == "call" else np.maximum(K-S_T, 0)
    price = np.exp(-r*T) * np.mean(payoffs)
    se = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_sims)
    return {"price": price, "std_error": se, "paths": S_T[:500]}

@st.cache_data(ttl=300)
def backtest_strategy(strategy, S0, K, T, r, sigma, q, n_days, n_sims):
    np.random.seed(42)
    dt = T / n_days
    Z = np.random.standard_normal((n_sims, n_days))
    S_f = S0 * np.exp(np.sum((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
    rows = []
    for Se in S_f:
        if strategy == "long_call":
            pnl = max(Se-K, 0) - bs(S0, K, T, r, sigma, q, "call")
        elif strategy == "long_put":
            pnl = max(K-Se, 0) - bs(S0, K, T, r, sigma, q, "put")
        elif strategy == "covered_call":
            pnl = (Se-S0) + bs(S0, K, T, r, sigma, q, "call") - max(Se-K, 0)
        elif strategy == "protective_put":
            pnl = (Se-S0) - bs(S0, K, T, r, sigma, q, "put") + max(K-Se, 0)
        elif strategy == "straddle":
            pnl = max(Se-K, 0) + max(K-Se, 0) - bs(S0, K, T, r, sigma, q, "call") - bs(S0, K, T, r, sigma, q, "put")
        else:
            Kc, Kp = K*1.05, K*0.95
            pnl = max(Se-Kc, 0) + max(Kp-Se, 0) - bs(S0, Kc, T, r, sigma, q, "call") - bs(S0, Kp, T, r, sigma, q, "put")
        rows.append({"final_spot": Se, "pnl": pnl, "return_pct": (pnl/S0)*100})
    return pd.DataFrame(rows)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Options Pricer")
    st.divider()
    
    if st.session_state.page == "app":
        if st.button("Formula Reference"):
            st.session_state.page = "docs"
            st.rerun()
    else:
        if st.button("Back"):
            st.session_state.page = "app"
            st.rerun()
    
    st.divider()

    if st.session_state.page == "app":
        mode = st.selectbox("Mode", ["Pricing", "Implied Volatility", "Backtesting"])
        pricing_method = "Black-Scholes"
        if mode == "Pricing":
            st.divider()
            pricing_method = st.selectbox("Model", ["Black-Scholes", "Monte Carlo"])

        st.divider()
        st.markdown("**Parameters**")
        S = st.number_input("Spot S", value=100.0, step=1.0)
        K = st.number_input("Strike K", value=100.0, step=1.0)
        T_day = st.number_input("Maturity (days)", value=30, step=1, min_value=1)
        r = st.number_input("Rate r (%)", value=5.0, step=0.1) / 100
        sigma = st.number_input("Vol σ (%)", value=20.0, step=0.5) / 100
        q = st.number_input("Div yield q (%)", value=0.0, step=0.1) / 100
        opt = st.radio("Type", ["call", "put"], horizontal=True)

        prem, n_sims, n_steps, seed = 0.0, 100000, 252, 42
        skew_slope, skew_convexity, term_slope = 0.0, 0.0, 0.0
        strategy, backtest_days, n_simulations = "long_call", 30, 1000

        if mode == "Pricing":
            prem = st.number_input("Premium paid ($)", value=0.0, step=0.01)
            if pricing_method == "Monte Carlo":
                st.divider()
                n_sims = st.selectbox("Simulations", [10000, 50000, 100000], index=2)
                seed = st.number_input("Seed", value=42, step=1)

        elif mode == "Implied Volatility":
            st.divider()
            skew_slope = st.slider("Skew slope", -0.50, 0.50, 0.10, 0.01)
            skew_convexity = st.slider("Skew convexity", 0.00, 0.50, 0.05, 0.01)
            term_slope = st.slider("Term slope", -0.20, 0.20, 0.02, 0.01)

        elif mode == "Backtesting":
            st.divider()
            strategy = st.selectbox("Strategy", ["long_call", "long_put", "covered_call", "protective_put", "straddle", "strangle"])
            backtest_days = st.slider("Horizon (days)", 1, min(365, T_day), min(T_day, 30))
            n_simulations = st.selectbox("Simulations", [100, 500, 1000], index=2)

        st.divider()
        run = st.button("Run", use_container_width=True)
    else:
        S, K, T_day, r, sigma, q, opt = 100.0, 100.0, 30, 0.05, 0.20, 0.0, "call"
        mode, pricing_method, prem, n_sims, n_steps, seed = "Pricing", "Black-Scholes", 0, 0, 0, 0
        strategy, backtest_days, n_simulations = "long_call", 30, 1000
        skew_slope, skew_convexity, term_slope = 0.0, 0.0, 0.0
        run = False

# ─── DOCS PAGE ────────────────────────────────────────────────────────────────

if st.session_state.page == "docs":
    st.markdown("# Formula Reference")
    st.caption("Arthur Cotten")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Black-Scholes", "Monte Carlo", "Greeks"])

    with tab1:
        st.markdown("**Call:** C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)")
        st.markdown("**Put:** P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)")
        st.markdown("d1 = [ln(S/K) + (r-q+σ²/2)·T] / (σ·√T)")
        st.markdown("d2 = d1 - σ·√T")

    with tab2:
        st.markdown("**GBM:** S(t+dt) = S(t)·exp[(r-q-σ²/2)·dt + σ·√dt·Z]")
        st.markdown("**Price:** C ≈ e^(-rT)·mean(payoffs)")
        st.markdown("**SE:** σ(payoffs)/√n")

    with tab3:
        st.markdown("| Greek | Call | Put |")
        st.markdown("|---|---|---|")
        st.markdown("| Delta | e^(-qT)·N(d1) | -e^(-qT)·N(-d1) |")
        st.markdown("| Gamma | e^(-qT)·N'(d1)/(S·σ·√T) | same |")
        st.markdown("| Vega | S·e^(-qT)·N'(d1)·√T/100 | same |")
        st.markdown("| Theta | complex | complex |")
        st.markdown("| Rho | K·T·e^(-rT)·N(d2)/100 | -K·T·e^(-rT)·N(-d2)/100 |")

# ─── APP PAGE ─────────────────────────────────────────────────────────────────

elif st.session_state.page == "app":

    st.markdown(f"# {'Pricing' if mode == 'Pricing' else 'IV Surface' if mode == 'Implied Volatility' else 'Backtest'}")
    st.caption("Arthur Cotten")
    st.divider()

    T = T_day / 365

    # ── PRICING ───────────────────────────────────────────────────────────────
    if mode == "Pricing":
        if pricing_method == "Black-Scholes":
            price = bs(S, K, T, r, sigma, q, opt)
            g = greeks(S, K, T, r, sigma, q, opt)
            std_error, mc_paths = None, None
        else:
            mc = monte_carlo_pricer(S, K, T, r, sigma, q, opt, n_sims, n_steps, int(seed))
            price, std_error, mc_paths = mc["price"], mc["std_error"], mc["paths"]
            g = greeks(S, K, T, r, sigma, q, opt)

        prob = prob_itm(S, K, T, r, sigma, q, opt)
        cost = prem if prem > 0 else price
        be = (K + cost) if opt == "call" else (K - cost)
        intr = max(S-K, 0) if opt == "call" else max(K-S, 0)
        tv = price - intr

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${price:.4f}")
        c2.metric("Break-even", f"${be:.2f}")
        c3.metric("Prob ITM", f"{prob*100:.1f}%")
        c4.metric("Time Value", f"${tv:.4f}")

        st.divider()
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Δ", f"{g['delta']:+.4f}")
        g2.metric("Γ", f"{g['gamma']:.5f}")
        g3.metric("ν", f"{g['vega']:.4f}")
        g4.metric("Θ", f"{g['theta']:+.4f}")
        g5.metric("ρ", f"{g['rho']:+.4f}")

        st.divider()
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            Sr = np.linspace(S*0.7, S*1.3, 200)
            pnl = (np.maximum(Sr-K, 0) - cost if opt == "call" else np.maximum(K-Sr, 0) - cost)
            ax.fill_between(Sr, pnl, 0, where=pnl >= 0, alpha=0.3, color="#2e7d32")
            ax.fill_between(Sr, pnl, 0, where=pnl < 0, alpha=0.3, color="#c62828")
            ax.plot(Sr, pnl, color="#333", lw=1)
            ax.axhline(0, color="#999", lw=0.5)
            ax.axvline(K, color="#1976d2", lw=0.8, ls="--", label=f"K={K}")
            ax.axvline(be, color="#2e7d32", lw=0.8, ls="--", label=f"BE={be:.1f}")
            ax.set_xlabel("Spot")
            ax.set_ylabel("P&L")
            ax.legend(fontsize=7, frameon=False)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            if mc_paths is not None:
                fig2, ax2 = plt.subplots(figsize=(3, 2.5))
                ax2.hist(mc_paths, bins=30, color="#1976d2", alpha=0.7, edgecolor="white", lw=0.3)
                ax2.axvline(K, color="#c62828", lw=0.8, ls="--")
                ax2.set_xlabel("S(T)")
                ax2.set_ylabel("Freq")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                plt.close()

    # ── IMPLIED VOLATILITY ────────────────────────────────────────────────────
    elif mode == "Implied Volatility":
        iv_skew = skewed_iv(sigma, S, K, T, skew_slope, skew_convexity, term_slope)
        theo_price = bs(S, K, T, r, sigma, q, opt)
        market_price = bs(S, K, T, r, iv_skew, q, opt)

        c1, c2, c3 = st.columns(3)
        c1.metric("Flat σ", f"{sigma*100:.1f}%")
        c2.metric("Skew IV", f"{iv_skew*100:.1f}%")
        c3.metric("Price gap", f"${market_price - theo_price:+.4f}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            stks = np.linspace(S*0.7, S*1.3, 30)
            ivs = [skewed_iv(sigma, S, k, T, skew_slope, skew_convexity, term_slope)*100 for k in stks]
            ax.plot([k/S for k in stks], ivs, color="#1976d2", lw=1, marker="o", ms=2)
            ax.axhline(sigma*100, color="#c62828", lw=0.8, ls="--", label=f"Flat {sigma*100:.0f}%")
            ax.set_xlabel("K/S")
            ax.set_ylabel("IV (%)")
            ax.legend(fontsize=7, frameon=False)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            mats = np.linspace(max(T, 7/365), min(T*3, 2.0), 15)
            ivs = [skewed_iv(sigma, S, K, m, skew_slope, skew_convexity, term_slope)*100 for m in mats]
            ax.plot([m*365 for m in mats], ivs, color="#1976d2", lw=1, marker="o", ms=2)
            ax.axhline(sigma*100, color="#c62828", lw=0.8, ls="--")
            ax.set_xlabel("Maturity (days)")
            ax.set_ylabel("IV (%)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # ── BACKTESTING ───────────────────────────────────────────────────────────
    elif mode == "Backtesting":
        df = backtest_strategy(strategy, S, K, T, r, sigma, q, backtest_days, n_simulations)

        mp = df['pnl'].mean()
        wr = (df['pnl'] > 0).sum() / len(df) * 100
        mg, ml = df['pnl'].max(), df['pnl'].min()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg P&L", f"${mp:.2f}")
        c2.metric("Win Rate", f"{wr:.0f}%")
        c3.metric("Max Gain", f"${mg:.2f}")
        c4.metric("Max Loss", f"${ml:.2f}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            pa = df['pnl'].values
            ax.hist(pa[pa >= 0], bins=25, color="#2e7d32", alpha=0.7, edgecolor="white", lw=0.3)
            ax.hist(pa[pa < 0], bins=25, color="#c62828", alpha=0.7, edgecolor="white", lw=0.3)
            ax.axvline(0, color="#333", lw=0.5)
            ax.axvline(mp, color="#1976d2", lw=0.8, ls="--")
            ax.set_xlabel("P&L")
            ax.set_ylabel("Freq")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            sp, pn = df['final_spot'].values, df['pnl'].values
            ax.scatter(sp[pn >= 0], pn[pn >= 0], alpha=0.4, s=6, color="#2e7d32")
            ax.scatter(sp[pn < 0], pn[pn < 0], alpha=0.4, s=6, color="#c62828")
            ax.axhline(0, color="#333", lw=0.5)
            ax.axvline(K, color="#1976d2", lw=0.8, ls="--")
            ax.set_xlabel("Final Spot")
            ax.set_ylabel("P&L")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        st.divider()
        pcts = [5, 25, 50, 75, 95]
        pct_data = {f"{p}%": [f"${df['pnl'].quantile(p/100):.2f}"] for p in pcts}
        st.dataframe(pd.DataFrame(pct_data), hide_index=True)
