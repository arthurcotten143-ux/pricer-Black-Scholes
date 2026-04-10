"""
Options Pricer
streamlit run pricer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

st.set_page_config(page_title="Pricer", layout="wide")

st.markdown("""
<style>
.block-container{padding:1rem 1rem;}
div[data-testid="stVerticalBlock"]{gap:0.2rem;}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white","axes.edgecolor":"#666","font.size":7})

def bs(S,K,T,r,sig,q=0,opt="call"):
    if T<=0:return max(S-K,0) if opt=="call" else max(K-S,0)
    d1=(np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2=d1-sig*np.sqrt(T)
    if opt=="call":return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)

def greeks(S,K,T,r,sig,q=0,opt="call"):
    if T<=0 or sig<=0:return {"delta":0,"gamma":0,"vega":0,"theta":0,"rho":0}
    d1=(np.log(S/K)+(r-q+0.5*sig**2)*T)/(sig*np.sqrt(T))
    d2=d1-sig*np.sqrt(T)
    nd1=norm.pdf(d1)
    if opt=="call":
        delta=np.exp(-q*T)*norm.cdf(d1)
        theta=(-(S*np.exp(-q*T)*nd1*sig)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
        rho=K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta=-np.exp(-q*T)*norm.cdf(-d1)
        theta=(-(S*np.exp(-q*T)*nd1*sig)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
        rho=-K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    gamma=np.exp(-q*T)*nd1/(S*sig*np.sqrt(T))
    vega=S*np.exp(-q*T)*nd1*np.sqrt(T)/100
    return {"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}

def mc(S,K,T,r,sig,q,opt,n):
    np.random.seed(42)
    Z=np.random.standard_normal(n)
    ST=S*np.exp((r-q-0.5*sig**2)*T+sig*np.sqrt(T)*Z)
    pay=np.maximum(ST-K,0) if opt=="call" else np.maximum(K-ST,0)
    return np.exp(-r*T)*np.mean(pay),np.exp(-r*T)*np.std(pay)/np.sqrt(n),ST[:300]

def backtest(strat,S,K,T,r,sig,q,days,n):
    np.random.seed(42)
    dt=T/days
    Z=np.random.standard_normal((n,days))
    Sf=S*np.exp(np.sum((r-q-0.5*sig**2)*dt+sig*np.sqrt(dt)*Z,axis=1))
    pnls=[]
    for Se in Sf:
        if strat=="call":pnl=max(Se-K,0)-bs(S,K,T,r,sig,q,"call")
        elif strat=="put":pnl=max(K-Se,0)-bs(S,K,T,r,sig,q,"put")
        elif strat=="straddle":pnl=max(Se-K,0)+max(K-Se,0)-bs(S,K,T,r,sig,q,"call")-bs(S,K,T,r,sig,q,"put")
        else:pnl=0
        pnls.append(pnl)
    return np.array(pnls),Sf

# SIDEBAR
with st.sidebar:
    st.write("**PRICER**")
    mode=st.selectbox("",["Pricing","Backtest"],label_visibility="collapsed")
    method="BS"
    if mode=="Pricing":method=st.radio("",["BS","MC"],horizontal=True,label_visibility="collapsed")
    st.write("---")
    S=st.number_input("S",value=100.0)
    K=st.number_input("K",value=100.0)
    T_d=st.number_input("T(d)",value=30,min_value=1)
    r=st.number_input("r%",value=5.0)/100
    sig=st.number_input("σ%",value=20.0)/100
    q=st.number_input("q%",value=0.0)/100
    opt=st.radio("",["call","put"],horizontal=True,label_visibility="collapsed")
    if mode=="Backtest":
        strat=st.selectbox("",["call","put","straddle"],label_visibility="collapsed")
        nsim=st.selectbox("n",[500,1000,2000],index=1)

T=T_d/365

# MAIN
if mode=="Pricing":
    if method=="BS":
        p=bs(S,K,T,r,sig,q,opt)
        g=greeks(S,K,T,r,sig,q,opt)
        se=None
        paths=None
    else:
        p,se,paths=mc(S,K,T,r,sig,q,opt,50000)
        g=greeks(S,K,T,r,sig,q,opt)
    
    be=K+p if opt=="call" else K-p
    intr=max(S-K,0) if opt=="call" else max(K-S,0)
    d2=(np.log(S/K)+(r-q-0.5*sig**2)*T)/(sig*np.sqrt(T))
    prob_itm=norm.cdf(d2) if opt=="call" else norm.cdf(-d2)
    
    col1,col2=st.columns([1,2])
    
    with col1:
        st.write(f"**{opt.upper()}**")
        st.write(f"Price = {p:.4f}")
        st.write(f"Break-even = {be:.2f}")
        st.write(f"P(ITM) = {prob_itm*100:.1f}%")
        st.write(f"Time value = {p-intr:.4f}")
        if se:st.write(f"SE = ±{se:.4f}")
        st.write("")
        st.write(f"Delta = {g['delta']:.4f}")
        st.write(f"Gamma = {g['gamma']:.5f}")
        st.write(f"Vega = {g['vega']:.4f}")
        st.write(f"Theta = {g['theta']:.4f}")
        st.write(f"Rho = {g['rho']:.4f}")
    
    with col2:
        fig,ax=plt.subplots(figsize=(5,2.2))
        Sr=np.linspace(S*0.7,S*1.3,150)
        pnl=np.maximum(Sr-K,0)-p if opt=="call" else np.maximum(K-Sr,0)-p
        ax.fill_between(Sr,pnl,0,where=pnl>=0,alpha=0.3,color="green")
        ax.fill_between(Sr,pnl,0,where=pnl<0,alpha=0.3,color="red")
        ax.plot(Sr,pnl,color="black",lw=0.8)
        ax.axhline(0,color="gray",lw=0.4)
        ax.axvline(K,color="blue",lw=0.5,ls="--")
        ax.axvline(be,color="green",lw=0.5,ls="--")
        ax.set_xlim(S*0.7,S*1.3)
        ax.set_xlabel("Spot")
        ax.set_ylabel("P&L")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        if paths is not None:
            fig,ax=plt.subplots(figsize=(5,1.8))
            ax.hist(paths,bins=25,color="steelblue",alpha=0.7,edgecolor="white",lw=0.2)
            ax.axvline(K,color="red",lw=0.5,ls="--")
            ax.set_xlabel("S(T)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

else:
    pnls,Sf=backtest(strat,S,K,T,r,sig,q,T_d,nsim)
    
    col1,col2=st.columns([1,2])
    
    with col1:
        st.write(f"**{strat.upper()}**")
        st.write(f"Avg P&L = {pnls.mean():.2f}")
        st.write(f"Win rate = {(pnls>0).mean()*100:.0f}%")
        st.write(f"Max = {pnls.max():.2f}")
        st.write(f"Min = {pnls.min():.2f}")
        st.write("")
        pcts=np.percentile(pnls,[5,25,50,75,95])
        st.write(f"P5 = {pcts[0]:.2f}")
        st.write(f"P25 = {pcts[1]:.2f}")
        st.write(f"P50 = {pcts[2]:.2f}")
        st.write(f"P75 = {pcts[3]:.2f}")
        st.write(f"P95 = {pcts[4]:.2f}")
    
    with col2:
        fig,ax=plt.subplots(figsize=(5,2))
        ax.hist(pnls[pnls>=0],bins=20,color="green",alpha=0.6,edgecolor="white",lw=0.2)
        ax.hist(pnls[pnls<0],bins=20,color="red",alpha=0.6,edgecolor="white",lw=0.2)
        ax.axvline(0,color="black",lw=0.4)
        ax.set_xlabel("P&L")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        fig,ax=plt.subplots(figsize=(5,1.8))
        ax.scatter(Sf[pnls>=0],pnls[pnls>=0],s=3,alpha=0.4,color="green")
        ax.scatter(Sf[pnls<0],pnls[pnls<0],s=3,alpha=0.4,color="red")
        ax.axhline(0,color="black",lw=0.4)
        ax.axvline(K,color="blue",lw=0.5,ls="--")
        ax.set_xlabel("S(T)")
        ax.set_ylabel("P&L")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
