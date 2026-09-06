import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI — Smart Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model ────────────────────────────────────────────────────────────────
model         = pickle.load(open("best_model.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# ── Global CSS / Animations ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ─── Base Reset ─── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #0a1628 70%, #060d1a 100%) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(16, 185, 129, 0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, 0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: bgShift 12s ease-in-out infinite alternate;
}

@keyframes bgShift {
    0%   { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(2%, 2%) rotate(3deg); }
}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

/* ─── Animated Hero Banner ─── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(99,102,241,0.18) 0%,
        rgba(16,185,129,0.12) 50%,
        rgba(6,182,212,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    animation: heroSlideIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
}

@keyframes heroSlideIn {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -2px; left: -2px; right: -2px; bottom: -2px;
    border-radius: 24px;
    background: linear-gradient(135deg, #6366f1, #10b981, #06b6d4);
    z-index: -1;
    opacity: 0.3;
    animation: borderGlow 4s ease-in-out infinite alternate;
}

@keyframes borderGlow {
    0%   { opacity: 0.2; }
    100% { opacity: 0.5; }
}

.hero-banner::after {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    top: -100px; right: -50px;
    border-radius: 50%;
    animation: orb1 8s ease-in-out infinite;
}

@keyframes orb1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50%       { transform: translate(-20px, 20px) scale(1.1); }
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc 0%, #34d399 50%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
    animation: textShimmer 3s ease-in-out infinite alternate;
}

@keyframes textShimmer {
    0%   { filter: brightness(1); }
    100% { filter: brightness(1.2); }
}

.hero-subtitle {
    color: rgba(200, 210, 255, 0.75);
    font-size: 1.1rem;
    font-weight: 400;
    margin: 0;
    animation: fadeInUp 1s 0.3s both;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
    animation: pulse-badge 2.5s ease-in-out infinite;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

@keyframes pulse-badge {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(16,185,129,0); }
}

/* ─── Floating Particles ─── */
.particles {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.particle {
    position: absolute;
    border-radius: 50%;
    animation: floatParticle linear infinite;
    opacity: 0;
}

@keyframes floatParticle {
    0%   { opacity: 0;    transform: translateY(100vh) scale(0); }
    10%  { opacity: 0.6; }
    90%  { opacity: 0.3; }
    100% { opacity: 0;    transform: translateY(-10vh) scale(1); }
}

/* ─── Glass Cards ─── */
.glass-card {
    background: rgba(15, 20, 40, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: cardFadeIn 0.6s ease both;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent);
}

@keyframes cardFadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-card:hover {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
    transform: translateY(-2px);
}

/* ─── Section headers ─── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.5), transparent);
}

/* ─── Number inputs ─── */
[data-testid="stNumberInput"] > div {
    background: rgba(15, 20, 45, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stNumberInput"] > div:focus-within {
    border-color: rgba(99, 102, 241, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

[data-testid="stNumberInput"] input {
    color: #e0e8ff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInput"] label {
    color: rgba(180, 195, 255, 0.85) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ─── Predict Button ─── */
[data-testid="stButton"] > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button::after {
    content: '' !important;
    position: absolute !important;
    top: 50%; left: 50% !important;
    width: 0; height: 0 !important;
    background: rgba(255,255,255,0.2) !important;
    border-radius: 50% !important;
    transform: translate(-50%, -50%) !important;
    transition: width 0.5s, height 0.5s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 50%, #4f46e5 100%) !important;
}

.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ─── Expander ─── */
[data-testid="stExpander"] {
    background: rgba(15, 20, 40, 0.5) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

[data-testid="stExpander"] summary {
    color: #a5b4fc !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ─── Progress bars ─── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #10b981) !important;
    border-radius: 4px !important;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stProgress"] > div {
    background: rgba(99,102,241,0.1) !important;
    border-radius: 4px !important;
}

/* ─── Result Cards ─── */
.result-card {
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    animation: resultReveal 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both;
    position: relative;
    overflow: hidden;
}

@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.85) translateY(20px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

.result-safe {
    background: linear-gradient(135deg,
        rgba(16, 185, 129, 0.15) 0%,
        rgba(6, 182, 212, 0.10) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 40px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(255,255,255,0.1);
}

.result-risk {
    background: linear-gradient(135deg,
        rgba(239, 68, 68, 0.15) 0%,
        rgba(251, 146, 60, 0.10) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.2), inset 0 1px 0 rgba(255,255,255,0.1);
}

.result-icon {
    font-size: 4rem;
    display: block;
    margin-bottom: 1rem;
    animation: iconBounce 0.6s 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}

@keyframes iconBounce {
    from { transform: scale(0) rotate(-20deg); }
    to   { transform: scale(1) rotate(0deg); }
}

.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.result-title-safe  { color: #34d399; }
.result-title-risk  { color: #f87171; }

.result-subtitle {
    color: rgba(200, 210, 240, 0.7);
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* ─── Confidence Meter ─── */
.conf-meter {
    background: rgba(0,0,0,0.3);
    border-radius: 50px;
    height: 14px;
    overflow: hidden;
    margin: 1rem 0 0.3rem 0;
    position: relative;
}

.conf-fill-safe {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
    animation: fillBar 1.2s 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
    box-shadow: 0 0 12px rgba(16,185,129,0.6);
}

.conf-fill-risk {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #ef4444, #f87171, #fca5a5);
    animation: fillBar 1.2s 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
    box-shadow: 0 0 12px rgba(239,68,68,0.6);
}

@keyframes fillBar {
    from { width: 0%; opacity: 0.5; }
    to   { opacity: 1; }
}

/* ─── Stat pills ─── */
.stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 1.5rem;
}

.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 50px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    color: rgba(200,215,255,0.8);
    font-weight: 500;
    animation: pillFade 0.5s ease both;
}

@keyframes pillFade {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─── Quick-fill buttons ─── */
.stButton > button[data-baseweb="button"] {
    font-size: 0.88rem !important;
}

/* ─── Caption ─── */
[data-testid="stCaptionContainer"] {
    color: rgba(150, 165, 210, 0.5) !important;
    font-size: 0.78rem !important;
    text-align: center !important;
    margin-top: 2rem !important;
}

/* ─── Metrics ─── */
[data-testid="metric-container"] {
    background: rgba(15, 20, 45, 0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem;
    animation: cardFadeIn 0.5s ease both;
}

[data-testid="metric-container"] label {
    color: rgba(160, 175, 255, 0.7) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #a5b4fc !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

/* ─── Divider ─── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(99,102,241,0.4);
    border-radius: 3px;
}

/* ─── fadeInUp utility ─── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to   { opacity: 1; transform: translateY(0); }
}

.fade-in-1 { animation: fadeInUp 0.5s 0.1s both; }
.fade-in-2 { animation: fadeInUp 0.5s 0.2s both; }
.fade-in-3 { animation: fadeInUp 0.5s 0.3s both; }
.fade-in-4 { animation: fadeInUp 0.5s 0.4s both; }

/* ─── Shimmer loader ─── */
.shimmer {
    background: linear-gradient(90deg,
        rgba(99,102,241,0.1) 25%,
        rgba(99,102,241,0.25) 50%,
        rgba(99,102,241,0.1) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 8px;
    height: 8px;
}

@keyframes shimmer {
    0%   { background-position: 200% center; }
    100% { background-position: -200% center; }
}

/* ─── Pulse dot ─── */
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #34d399;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulseDot 2s ease-in-out infinite;
}

@keyframes pulseDot {
    0%, 100% { transform: scale(1);   opacity: 1; box-shadow: 0 0 0 0 rgba(52,211,153,0.6); }
    50%       { transform: scale(1.2); opacity: 0.8; box-shadow: 0 0 0 6px rgba(52,211,153,0); }
}
</style>

<!-- Floating particles -->
<div class="particles">
  <div class="particle" style="width:3px;height:3px;background:#6366f1;left:10%;animation-duration:18s;animation-delay:0s;"></div>
  <div class="particle" style="width:2px;height:2px;background:#10b981;left:25%;animation-duration:22s;animation-delay:3s;"></div>
  <div class="particle" style="width:4px;height:4px;background:#06b6d4;left:40%;animation-duration:16s;animation-delay:6s;"></div>
  <div class="particle" style="width:2px;height:2px;background:#a5b4fc;left:60%;animation-duration:20s;animation-delay:1s;"></div>
  <div class="particle" style="width:3px;height:3px;background:#34d399;left:75%;animation-duration:25s;animation-delay:4s;"></div>
  <div class="particle" style="width:2px;height:2px;background:#6366f1;left:88%;animation-duration:19s;animation-delay:8s;"></div>
  <div class="particle" style="width:5px;height:5px;background:rgba(99,102,241,0.4);left:50%;animation-duration:28s;animation-delay:2s;"></div>
</div>
""", unsafe_allow_html=True)

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">
        <span class="pulse-dot"></span>AI-Powered · Clinical Grade
    </div>
    <div class="hero-title">🩺 DiabetesAI</div>
    <p class="hero-subtitle">
        Advanced ensemble machine learning model trained on the Pima Indians Diabetes Dataset.<br>
        Get instant, confidence-rated predictions — enter patient details below.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model Stats Row ───────────────────────────────────────────────────────────
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("🎯 Accuracy",  "74.7%",  "Validated")
mc2.metric("📈 ROC-AUC",   "0.83",   "Excellent")
mc3.metric("🧬 Features",  "14",     "8 + 6 engineered")
mc4.metric("🏥 Patients",  "768",    "Training set")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Reference Guide ───────────────────────────────────────────────────────────
with st.expander("📋 Reference Guide — Normal & At-Risk Ranges"):
    st.markdown("""
| Parameter | What it means | ✅ Normal | ⚠️ At-Risk |
|---|---|---|---|
| **Pregnancies** | Number of times pregnant | 0 – 3 | 4 or more |
| **Glucose Level** | Blood sugar (mg/dL) | 70 – 99 (fasting) | 126+ (diabetic range) |
| **Blood Pressure** | Diastolic BP (mm Hg) | 60 – 80 | 90+ |
| **Skin Thickness** | Triceps skinfold (mm) | 10 – 25 | 35+ |
| **Insulin** | 2-hour serum insulin (uU/ml) | 16 – 166 | 200+ |
| **BMI** | Body Mass Index (kg/m²) | 18.5 – 24.9 | 30+ (obese) |
| **Diabetes Pedigree** | Family history score | 0.078 – 0.5 | 0.8+ |
| **Age** | Patient age (years) | Any | 45+ (higher risk) |

---
**🟢 Healthy Profile** → Pregnancies: 1 · Glucose: 85 · BP: 70 · Skin: 20 · Insulin: 80 · BMI: 22.5 · DPF: 0.200 · Age: 28

**🔴 At-Risk Profile** → Pregnancies: 6 · Glucose: 160 · BP: 90 · Skin: 35 · Insulin: 250 · BMI: 35.0 · DPF: 0.850 · Age: 52
""")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Quick-fill Buttons ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ Quick Fill</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)

if col_a.button("🟢 Load Healthy Profile", use_container_width=True):
    st.session_state.pregnancies = 1;  st.session_state.glucose = 85
    st.session_state.bp = 70;          st.session_state.skin = 20
    st.session_state.insulin = 80;     st.session_state.bmi = 22.5
    st.session_state.dpf = 0.200;      st.session_state.age = 28
    st.toast("✅ Healthy profile loaded!", icon="🟢")

if col_b.button("🔴 Load At-Risk Profile", use_container_width=True):
    st.session_state.pregnancies = 6;  st.session_state.glucose = 160
    st.session_state.bp = 90;          st.session_state.skin = 35
    st.session_state.insulin = 250;    st.session_state.bmi = 35.0
    st.session_state.dpf = 0.850;      st.session_state.age = 52
    st.toast("⚠️ At-risk profile loaded!", icon="🔴")

if col_c.button("🔄 Reset All Fields", use_container_width=True):
    st.session_state.pregnancies = 0;  st.session_state.glucose = 0
    st.session_state.bp = 0;           st.session_state.skin = 0
    st.session_state.insulin = 0;      st.session_state.bmi = 0.0
    st.session_state.dpf = 0.000;      st.session_state.age = 1
    st.toast("🔄 Fields reset!", icon="🔄")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Patient Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="glass-card fade-in-1">', unsafe_allow_html=True)
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, max_value=20, step=1,
        key="pregnancies", help="Number of times pregnant. Normal: 0-3 | High risk: 4+")
    glucose = st.number_input("🩸 Glucose Level (mg/dL)", min_value=0, max_value=300,
        key="glucose", help="Fasting blood glucose. Normal: 70-99 | Pre-diabetic: 100-125 | Diabetic: 126+")
    bp = st.number_input("💓 Blood Pressure (mm Hg)", min_value=0, max_value=200,
        key="bp", help="Diastolic blood pressure. Normal: 60-80 | High: 90+")
    skin_thickness = st.number_input("📏 Skin Thickness (mm)", min_value=0, max_value=100,
        key="skin", help="Triceps skinfold thickness. Normal: 10-25 mm | High: 35+")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card fade-in-2">', unsafe_allow_html=True)
    insulin = st.number_input("💉 Insulin (uU/ml)", min_value=0, max_value=1000,
        key="insulin", help="2-hour serum insulin. Normal: 16-166 | High: 200+")
    bmi = st.number_input("⚖️ BMI (kg/m²)", min_value=0.0, max_value=100.0, format="%.1f",
        key="bmi", help="Body Mass Index. Healthy: 18.5-24.9 | Overweight: 25-29.9 | Obese: 30+")
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=5.0, format="%.3f",
        key="dpf", help="Family history score. Low: <0.5 | High: 0.8+")
    age = st.number_input("🎂 Age (years)", min_value=1, max_value=120,
        key="age", help="Patient age. Risk increases after 45.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Live Range Visualiser ─────────────────────────────────────────────────────
with st.expander("📊 Live Health Range Visualiser"):
    def range_bar(label, value, low, high, unit=""):
        pct = min(int((value / (high * 1.5 + 0.001)) * 100), 100)
        if value == 0:
            status = "⚫ Not entered"
            color = "#6b7280"
        elif value < low:
            status = "🔵 Low"
            color = "#60a5fa"
        elif low <= value <= high:
            status = "🟢 Normal"
            color = "#34d399"
        else:
            status = "🔴 High"
            color = "#f87171"
        st.markdown(f"**{label}:** `{value}{unit}` — {status}")
        st.progress(pct)

    r1, r2 = st.columns(2)
    with r1:
        range_bar("🤰 Pregnancies",    pregnancies,     0,    3)
        range_bar("🩸 Glucose",        glucose,         70,   99,   " mg/dL")
        range_bar("💓 Blood Pressure", bp,              60,   80,   " mm Hg")
        range_bar("📏 Skin Thickness", skin_thickness,  10,   25,   " mm")
    with r2:
        range_bar("💉 Insulin",        insulin,         16,   166,  " uU/ml")
        range_bar("⚖️ BMI",            bmi,             18.5, 24.9, " kg/m²")
        range_bar("🧬 Pedigree Score", dpf,             0.0,  0.5)
        range_bar("🎂 Age",            age,             1,    44,   " yrs")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔬 Run Prediction Analysis", use_container_width=True)

if predict_clicked:
    # Animated loading
    with st.spinner(""):
        progress_placeholder = st.empty()
        progress_placeholder.markdown("""
<div style="text-align:center; padding: 1.5rem;">
    <div style="font-size:1.1rem; color:#a5b4fc; font-family:'Space Grotesk',sans-serif; margin-bottom:1rem;">
        ⚙️ Analyzing patient data...
    </div>
    <div class="shimmer" style="max-width:400px; margin:0 auto;"></div>
</div>
""", unsafe_allow_html=True)
        time.sleep(1.2)
    progress_placeholder.empty()

    # ── Feature engineering ───────────────────────────────────────────────────
    glucose_eff  = glucose         if glucose         > 0 else 117.0
    bp_eff       = bp              if bp              > 0 else 72.0
    skin_eff     = skin_thickness  if skin_thickness  > 0 else 23.0
    insulin_eff  = insulin         if insulin         > 0 else 30.5
    bmi_eff      = bmi             if bmi             > 0 else 32.0

    glucose_bmi     = glucose_eff * bmi_eff
    glucose_insulin = glucose_eff / (insulin_eff + 1)
    high_glucose    = int(glucose_eff >= 126)
    obese           = int(bmi_eff >= 30)
    senior          = int(age >= 45)
    bmi_age         = bmi_eff * age

    row = pd.DataFrame([[
        pregnancies, glucose_eff, bp_eff, skin_eff, insulin_eff, bmi_eff, dpf, age,
        glucose_bmi, glucose_insulin, high_glucose, obese, senior, bmi_age
    ]], columns=feature_names)

    prediction = model.predict(row)
    proba      = model.predict_proba(row)[0]
    diabetes_prob = proba[1] * 100
    no_diab_prob  = proba[0] * 100

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Prediction Result</div>', unsafe_allow_html=True)

    if prediction[0] == 0:
        conf_pct = no_diab_prob
        fill_class = "conf-fill-safe"
        st.markdown(f"""
<div class="result-card result-safe">
    <span class="result-icon">✅</span>
    <div class="result-title result-title-safe">No Diabetes Detected</div>
    <div class="result-subtitle">Based on the provided patient data, diabetes is unlikely.</div>
    <div class="conf-meter">
        <div class="{fill_class}" style="width:{conf_pct:.1f}%"></div>
    </div>
    <div style="color:#34d399; font-size:1.4rem; font-weight:700; font-family:'Space Grotesk',sans-serif; margin-top:0.5rem;">
        {conf_pct:.1f}% Confidence
    </div>
    <div class="stat-row">
        <div class="stat-pill">🩺 No Diabetes: {no_diab_prob:.1f}%</div>
        <div class="stat-pill">⚠️ Diabetes: {diabetes_prob:.1f}%</div>
        <div class="stat-pill">🤰 Pregnancies: {pregnancies}</div>
        <div class="stat-pill">🩸 Glucose: {glucose} mg/dL</div>
        <div class="stat-pill">⚖️ BMI: {bmi:.1f}</div>
        <div class="stat-pill">🎂 Age: {age} yrs</div>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        conf_pct = diabetes_prob
        fill_class = "conf-fill-risk"
        st.markdown(f"""
<div class="result-card result-risk">
    <span class="result-icon">⚠️</span>
    <div class="result-title result-title-risk">Diabetes Risk Detected</div>
    <div class="result-subtitle">The model indicates a significant likelihood of diabetes — please consult a medical professional.</div>
    <div class="conf-meter">
        <div class="{fill_class}" style="width:{conf_pct:.1f}%"></div>
    </div>
    <div style="color:#f87171; font-size:1.4rem; font-weight:700; font-family:'Space Grotesk',sans-serif; margin-top:0.5rem;">
        {conf_pct:.1f}% Confidence
    </div>
    <div class="stat-row">
        <div class="stat-pill">⚠️ Diabetes: {diabetes_prob:.1f}%</div>
        <div class="stat-pill">🩺 No Diabetes: {no_diab_prob:.1f}%</div>
        <div class="stat-pill">🤰 Pregnancies: {pregnancies}</div>
        <div class="stat-pill">🩸 Glucose: {glucose} mg/dL</div>
        <div class="stat-pill">⚖️ BMI: {bmi:.1f}</div>
        <div class="stat-pill">🎂 Age: {age} yrs</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── About this prediction ─────────────────────────────────────────────────
    with st.expander("🧠 About this Prediction"):
        st.markdown(f"""
**Model Architecture:** Soft-Voting Ensemble (Logistic Regression + Random Forest + Gradient Boosting)

| Metric | Value |
|---|---|
| Test Accuracy | 74.7% |
| ROC-AUC | 0.83 |
| Total Features | 14 (8 raw + 6 engineered) |
| Training Samples | 768 patients |

**Derived Features Auto-Computed:**
- `glucose × BMI` = `{glucose_bmi:.1f}`
- `glucose / (insulin+1)` = `{glucose_insulin:.3f}`
- `high_glucose (≥126)` = `{high_glucose}`
- `obese (BMI ≥30)` = `{obese}`
- `senior (age ≥45)` = `{senior}`
- `BMI × age` = `{bmi_age:.1f}`
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🔒 DiabetesAI is for educational & demo purposes only. Not a substitute for professional medical advice.")
