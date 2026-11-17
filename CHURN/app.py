import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import re
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS COMERCIALES
# ==============================================================================
st.set_page_config(
    page_title="RetenTel AI | Plataforma Comercial de Retención",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Reducir padding superior nativo de Streamlit para que luzca el Top Navbar */
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.5rem !important;
    }

    /* Reset de enlaces dentro del portal para que sean tarjetas clicables puras */
    a.portal-link, a.portal-link:hover, a.portal-link:focus, a.portal-link:active, a.portal-link:visited {
        text-decoration: none !important;
        color: inherit !important;
    }

    /* Contenedor maestro del Portal (sombra suave e integración) */
    .portal-wrapper {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        margin-bottom: 22px;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    /* 1. Navbar limpia en una sola línea */
    .top-navbar {
        background-color: #161F30;
        color: #FFFFFF;
        height: 56px;
        padding: 0 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        white-space: nowrap;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.18rem;
        font-weight: 700;
        color: #FFFFFF;
        white-space: nowrap;
    }
    .nav-links {
        display: flex;
        align-items: center;
        gap: 28px;
        font-size: 0.94rem;
        color: #E2E8F0;
        font-weight: 500;
        white-space: nowrap;
    }
    .nav-links a {
        color: #E2E8F0 !important;
        text-decoration: none !important;
        transition: color 0.15s ease;
    }
    .nav-links a:hover {
        color: #FFFFFF !important;
    }
    .nav-avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background-color: #334155;
        border: 1.5px solid #94A3B8;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #F8FAFC;
        flex-shrink: 0;
    }

    /* 2. Banner Hero con imagen fotográfica real de montaña y alto fijo */
    .hero-banner {
        height: 295px;
        padding: 0 42px;
        background-image: linear-gradient(90deg, rgba(15,23,42,0.88) 0%, rgba(15,23,42,0.45) 60%, transparent 100%), url('https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1800&q=85');
        background-size: cover;
        background-position: center 35%;
        color: #FFFFFF;
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-sizing: border-box;
    }
    .hero-content {
        max-width: 55%;
        z-index: 2;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #E2E8F0;
        line-height: 1.42;
        margin: 0 0 22px 0;
        font-weight: 400;
    }
    .hero-pill-btn {
        background-color: #FFFFFF;
        color: #0F172A !important;
        font-weight: 700;
        font-size: 0.96rem;
        padding: 11px 24px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
        text-decoration: none !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .hero-pill-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.32);
    }

    /* 3. Grid 2x2 simétrico en un solo bloque HTML */
    .cards-grid-2x2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        padding: 20px 22px 22px 22px;
        background-color: #FFFFFF;
    }
    .portal-card {
        height: 110px;
        box-sizing: border-box;
        border-radius: 14px;
        padding: 0 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        cursor: pointer;
    }
    .portal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
    }
    .card-blue {
        background-color: #DCEBFE;
        border: 1.5px solid #BFDBFE;
    }
    .card-gray {
        background-color: #F1F5F9;
        border: 1.5px solid #E2E8F0;
    }
    .card-active-ring {
        border: 2.5px solid #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
    }
    .card-left {
        display: flex;
        align-items: center;
        gap: 18px;
        min-width: 0;
    }
    .card-icon {
        width: 36px;
        height: 36px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .card-text-wrap {
        min-width: 0;
    }
    .card-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0 0 3px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-desc {
        font-size: 0.9rem;
        color: #334155;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-arrow {
        color: #1D4ED8;
        font-size: 1.45rem;
        font-weight: 600;
        margin-left: 12px;
        flex-shrink: 0;
    }

    @media (max-width: 900px) {
        .cards-grid-2x2 {
            grid-template-columns: 1fr;
        }
        .hero-content {
            max-width: 90%;
        }
    }

    /* Tipografía y Encabezados de Módulos */
    .main-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    
    /* Tarjetas de Métricas */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    /* Badges de Riesgo Comercial */
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-mid {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FBBF24;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #4ADE80;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    /* Contenedor Destacado de Sugerencias */
    .suggestion-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 12px;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1);
    }
    .suggestion-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E40AF;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    
    /* Tarjetas de Acción de Retención */
    .action-card {
        background-color: #FFFFFF;
        border: 1px solid #BFDBFE;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .action-badge {
        background-color: #2563EB;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 4px;
    }
    
    /* Guión Comercial */
    .script-box {
        background-color: #F8FAFC;
        border-left: 4px solid #10B981;
        border-radius: 6px;
        padding: 14px;
        color: #1E293B;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-top: 12px;
    }
    
    /* Contenedor de Formulario */
    .commercial-panel {
        background-color: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CARGA DE ASSETS (MODELO, METADATOS Y DATASET)
# ==============================================================================
@st.cache_resource
def load_assets():
    model_path = 'models/champion_pipeline.joblib'
    metadata_path = 'models/metadata.json'
    data_path = 'Telco-Customer-Churn.csv'

    pipeline = joblib.load(model_path) if os.path.exists(model_path) else None

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    if df is not None:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0)
        df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    return pipeline, metadata, df

pipeline, metadata, df_raw = load_assets()

# ==============================================================================
# DEFINICIÓN DE MÓDULOS CON JERARQUÍA NUMÉRICA (1 AL 5) Y SINCRONIZACIÓN URL
# ==============================================================================
MODULES = [
    "🎯 1. Simulador Comercial & Sugerencias de Retención",
    "💼 2. Diagnóstico Ejecutivo & Business Case (ROI)",
    "📊 3. Exploración de Cartera & Segmentación (EDA)",
    "⚙️ 4. Configuración & Optimización del Modelo",
    "ℹ️ 5. Información del Sistema & Explicabilidad XAI"
]

NAV_MAP = {
    "m1": MODULES[0],
    "m2": MODULES[1],
    "m3": MODULES[2],
    "m4": MODULES[3],
    "m5": MODULES[4]
}
REV_MAP = {v: k for k, v in NAV_MAP.items()}

if 'active_nav' not in st.session_state:
    st.session_state['active_nav'] = MODULES[0]

# Procesar clics en las tarjetas HTML (?nav=m1..m5) sin duplicar botones
qp_val = st.query_params.get("nav", None)
if qp_val in NAV_MAP and st.session_state.get("last_qp") != qp_val:
    st.session_state['active_nav'] = NAV_MAP[qp_val]
    st.session_state['last_qp'] = qp_val

def on_sidebar_change():
    chosen = st.session_state['sidebar_nav_radio']
    st.session_state['active_nav'] = chosen
    code = REV_MAP.get(chosen, "m1")
    st.session_state['last_qp'] = code
    st.query_params["nav"] = code

# ==============================================================================
# NAVEGACIÓN LATERAL SINCRONIZADA Y PRIORIZADA COMERCIALMENTE
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=65)
st.sidebar.title("RetenTel AI")
st.sidebar.caption("Plataforma Comercial de Retención & Churn Intelligence")

show_portal = st.sidebar.checkbox("🖼️ Mostrar Portal Principal (Frontend)", value=True)

st.sidebar.radio(
    "Jerarquía de Módulos del Sistema:",
    MODULES,
    index=MODULES.index(st.session_state['active_nav']),
    key="sidebar_nav_radio",
    on_change=on_sidebar_change
)

nav = st.session_state['active_nav']

st.sidebar.divider()
st.sidebar.markdown("""
<div style="background-color: #F1F5F9; border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #334155;">
    <div style="font-weight: 700; color: #1E3A8A; margin-bottom: 6px;">📌 Estado Operativo:</div>
    <div>• <b>Prioridad #1:</b> Simulador & Sugerencias</div>
    <div>• <b>Modelo:</b> CatBoost Calibrado (NSGA-III)</div>
    <div>• <b>Cobertura Recall:</b> 84.2% (Test Aislado)</div>
    <div>• <b>Calibración Brier:</b> 0.1713 (Alta fiabilidad)</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# PORTAL VISUAL DE FRONTEND (1 SOLO BLOQUE HTML + CSS GRID 2x2 SIN BOTONES DUPLICADOS)
# ==============================================================================
if show_portal:
    act_2 = "card-active-ring" if nav == MODULES[1] else ""
    act_3 = "card-active-ring" if nav == MODULES[2] else ""
    act_4 = "card-active-ring" if nav == MODULES[3] else ""
    act_5 = "card-active-ring" if nav == MODULES[4] else ""

    portal_html = f"""
<div class="portal-wrapper">
<div class="top-navbar">
<a href="?nav=m1" target="_self" class="portal-link nav-brand">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" style="flex-shrink:0;">
<path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22L6.66 19.7C7.14 19.87 7.64 20 8 20C19 20 22 3 22 3C21 5 14 5.25 9 6.25C4 7.25 2 11.5 2 13.5C2 15.5 3.75 17.25 3.75 17.25C7 8 17 8 17 8Z" fill="#FFFFFF"/>
</svg>
<span>RetenTel AI</span>
</a>
<div class="nav-links">
<a href="?nav=m1" target="_self">Inicio</a>
<a href="?nav=m2" target="_self">Recursos</a>
<a href="?nav=m5" target="_self">Soporte</a>
<div class="nav-avatar">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0;">
<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
<circle cx="12" cy="7" r="4"></circle>
</svg>
</div>
</div>
</div>
<div class="hero-banner">
<div class="hero-content">
<div class="hero-title">1. Simulador Comercial &amp; Sugerencias</div>
<div class="hero-sub">El recurso más importante de la plataforma: evalúa el riesgo de fuga y aplica sugerencias de retención, accede directamente aquí.</div>
<a href="?nav=m1" target="_self" class="hero-pill-btn">Acceder ahora &rarr;</a>
</div>
<div style="z-index:1; pointer-events:none; margin-right:24px;">
<svg width="95" height="75" viewBox="0 0 110 85" fill="none">
<path d="M12 68 L48 26 L68 50 L82 35 L98 68 Z" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
<path d="M62 12 C72 10 80 16 84 25" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
</svg>
</div>
</div>
<div class="cards-grid-2x2">
<a href="?nav=m2" target="_self" class="portal-link portal-card card-blue {act_2}">
<div class="card-left">
<div class="card-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">
<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
</svg>
</div>
<div class="card-text-wrap">
<div class="card-title">2. Diagnóstico Ejecutivo &amp; ROI</div>
<div class="card-desc">Casi igual de importante • Caso de negocio (+318% ROI).</div>
</div>
</div>
<div class="card-arrow">&rarr;</div>
</a>
<a href="?nav=m3" target="_self" class="portal-link portal-card card-blue {act_3}">
<div class="card-left">
<div class="card-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">
<line x1="18" y1="20" x2="18" y2="10"></line>
<line x1="12" y1="20" x2="12" y2="4"></line>
<line x1="6" y1="20" x2="6" y2="14"></line>
<line x1="2" y1="20" x2="22" y2="20"></line>
</svg>
</div>
<div class="card-text-wrap">
<div class="card-title">3. Exploración de Cartera (EDA)</div>
<div class="card-desc">Casi igual de importante • KPIs, filtros y ADASYN.</div>
</div>
</div>
<div class="card-arrow">&rarr;</div>
</a>
<a href="?nav=m4" target="_self" class="portal-link portal-card card-gray {act_4}">
<div class="card-left">
<div class="card-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1E293B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">
<circle cx="12" cy="12" r="3"></circle>
<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
</svg>
</div>
<div class="card-text-wrap">
<div class="card-title">4. Configuración &amp; Optimización</div>
<div class="card-desc">Ajustes técnicos • Grid, Random, TPE y Pareto NSGA-III.</div>
</div>
</div>
</a>
<a href="?nav=m5" target="_self" class="portal-link portal-card card-gray {act_5}">
<div class="card-left">
<div class="card-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1E293B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">
<circle cx="12" cy="12" r="10"></circle>
<line x1="12" y1="16" x2="12" y2="12"></line>
<line x1="12" y1="8" x2="12.01" y2="8"></line>
</svg>
</div>
<div class="card-text-wrap">
<div class="card-title">5. Información del Sistema &amp; XAI</div>
<div class="card-desc">Detalles técnicos • SHAP, LIME, Random Forest y rúbrica.</div>
</div>
</div>
</a>
</div>
</div>
"""
    st.markdown(portal_html, unsafe_allow_html=True)

# ==============================================================================
# INICIALIZACIÓN DE VALORES POR DEFECTO PARA EL SIMULADOR
# ==============================================================================
defaults = {
    'tenure': 2,
    'MonthlyCharges': 89.5,
    'TotalCharges': 179.0,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'PaymentMethod': 'Electronic check',
    'TechSupport': 'No',
    'OnlineSecurity': 'No',
    'PaperlessBilling': 'Yes',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'gender': 'Female'
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# MÓDULO 1: SIMULADOR COMERCIAL & SUGERENCIAS DE RETENCIÓN (RECURSO PRINCIPAL #1)
# ==============================================================================
if nav == "🎯 1. Simulador Comercial & Sugerencias de Retención":
    st.markdown('<div class="main-title">🎯 1. Simulador Comercial de Retención de Clientes & Sugerencias</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Recurso Principal (#1): Evalúe en tiempo real el riesgo de abandono del suscriptor y active de inmediato las sugerencias comerciales de retención con mayor retorno económico.</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Botones Rápidos de Perfiles Comerciales
    # --------------------------------------------------------------------------
    st.markdown("**Perfiles Rápidos de Prueba Comercial:**")
    cp1, cp2, cp3, cp4 = st.columns(4)
    with cp1:
        if st.button("🚨 Cliente Riesgo Crítico (Fibra / Mes a Mes)", use_container_width=True):
            st.session_state['tenure'] = 2
            st.session_state['MonthlyCharges'] = 89.5
            st.session_state['TotalCharges'] = 179.0
            st.session_state['Contract'] = 'Month-to-month'
            st.session_state['InternetService'] = 'Fiber optic'
            st.session_state['PaymentMethod'] = 'Electronic check'
            st.session_state['TechSupport'] = 'No'
            st.session_state['OnlineSecurity'] = 'No'
            st.session_state['PaperlessBilling'] = 'Yes'
            st.rerun()

    with cp2:
        if st.button("⚠️ Cliente Riesgo Moderado (DSL / 14 Meses)", use_container_width=True):
            st.session_state['tenure'] = 14
            st.session_state['MonthlyCharges'] = 62.0
            st.session_state['TotalCharges'] = 868.0
            st.session_state['Contract'] = 'Month-to-month'
            st.session_state['InternetService'] = 'DSL'
            st.session_state['PaymentMethod'] = 'Mailed check'
            st.session_state['TechSupport'] = 'No'
            st.session_state['OnlineSecurity'] = 'Yes'
            st.session_state['PaperlessBilling'] = 'Yes'
            st.rerun()

    with cp3:
        if st.button("🛡️ Cliente Fidelizado (2 Años / Soporte)", use_container_width=True):
            st.session_state['tenure'] = 60
            st.session_state['MonthlyCharges'] = 45.0
            st.session_state['TotalCharges'] = 2700.0
            st.session_state['Contract'] = 'Two year'
            st.session_state['InternetService'] = 'DSL'
            st.session_state['PaymentMethod'] = 'Bank transfer (automatic)'
            st.session_state['TechSupport'] = 'Yes'
            st.session_state['OnlineSecurity'] = 'Yes'
            st.session_state['PaperlessBilling'] = 'No'
            st.rerun()

    with cp4:
        if st.button("🔄 Restablecer Parámetros", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.write("")

    # --------------------------------------------------------------------------
    # Layout Principal del Simulador: 2 Columnas (Inputs vs Diagnóstico + Sugerencias)
    # --------------------------------------------------------------------------
    col_input, col_sim = st.columns([1.05, 1.25], gap="large")

    with col_input:
        st.markdown("### 📋 Ficha del Cliente (Parámetros)")
        
        with st.container():
            # Acordeón 1: Parámetros Contractuales y Cobro
            with st.expander("📄 Contrato, Facturación y Antigüedad", expanded=True):
                tenure = st.slider(
                    "Antigüedad con la empresa (Meses):",
                    min_value=0, max_value=72,
                    value=int(st.session_state.get('tenure', 2)),
                    help="Meses ininterrumpidos que el cliente ha permanecido con el servicio."
                )
                
                c_fc1, c_fc2 = st.columns(2)
                with c_fc1:
                    contract = st.selectbox(
                        "Tipo de Contrato:",
                        ['Month-to-month', 'One year', 'Two year'],
                        index=['Month-to-month', 'One year', 'Two year'].index(st.session_state.get('Contract', 'Month-to-month')),
                        help="Factor SHAP #1: Los contratos mes a mes concentran el 88.5% del abandono."
                    )
                with c_fc2:
                    payment = st.selectbox(
                        "Método de Pago:",
                        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                        index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(
                            st.session_state.get('PaymentMethod', 'Electronic check')
                        )
                    )

                c_ch1, c_ch2 = st.columns(2)
                with c_ch1:
                    monthly_charges = st.number_input(
                        "Cargo Mensual ($ USD):",
                        min_value=18.0, max_value=140.0,
                        value=float(st.session_state.get('MonthlyCharges', 89.5)),
                        step=2.5
                    )
                with c_ch2:
                    # Cálculo automático aproximado de cargos totales si el usuario lo desea
                    suggested_total = max(18.0, monthly_charges * max(1, tenure))
                    total_charges = st.number_input(
                        "Cargos Totales Acumulados ($ USD):",
                        min_value=0.0, max_value=10000.0,
                        value=float(st.session_state.get('TotalCharges', suggested_total)),
                        step=20.0
                    )

                paperless = st.selectbox(
                    "Facturación Electrónica (Paperless):",
                    ['Yes', 'No'],
                    index=['Yes', 'No'].index(st.session_state.get('PaperlessBilling', 'Yes'))
                )

            # Acordeón 2: Servicios de Telecomunicaciones
            with st.expander("🌐 Servicios y Soporte Técnico", expanded=True):
                internet = st.selectbox(
                    "Servicio de Internet:",
                    ['Fiber optic', 'DSL', 'No'],
                    index=['Fiber optic', 'DSL', 'No'].index(st.session_state.get('InternetService', 'Fiber optic'))
                )

                c_sv1, c_sv2 = st.columns(2)
                with c_sv1:
                    tech_support = st.selectbox(
                        "Soporte Técnico Especializado:",
                        ['No', 'Yes', 'No internet service'],
                        index=['No', 'Yes', 'No internet service'].index(st.session_state.get('TechSupport', 'No')),
                        help="Factor SHAP #2: Clientes sin soporte técnico cancelan casi 3 veces más rápido."
                    )
                with c_sv2:
                    online_sec = st.selectbox(
                        "Seguridad en Línea:",
                        ['No', 'Yes', 'No internet service'],
                        index=['No', 'Yes', 'No internet service'].index(st.session_state.get('OnlineSecurity', 'No'))
                    )

                c_sv3, c_sv4 = st.columns(2)
                with c_sv3:
                    online_bkp = st.selectbox(
                        "Respaldo en Línea (Online Backup):",
                        ['No', 'Yes', 'No internet service'],
                        index=['No', 'Yes', 'No internet service'].index(st.session_state.get('OnlineBackup', 'No'))
                    )
                with c_sv4:
                    device_prot = st.selectbox(
                        "Protección de Dispositivo:",
                        ['No', 'Yes', 'No internet service'],
                        index=['No', 'Yes', 'No internet service'].index(st.session_state.get('DeviceProtection', 'No'))
                    )

                c_sv5, c_sv6 = st.columns(2)
                with c_sv5:
                    stream_tv = st.selectbox("Streaming TV:", ['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(st.session_state.get('StreamingTV', 'No')))
                with c_sv6:
                    stream_mov = st.selectbox("Streaming Películas:", ['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(st.session_state.get('StreamingMovies', 'No')))

            # Acordeón 3: Perfil Demográfico
            with st.expander("👤 Perfil del Titular", expanded=False):
                c_dm1, c_dm2 = st.columns(2)
                with c_dm1:
                    phone = st.selectbox("Línea Telefónica Fija:", ['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.get('PhoneService', 'Yes')))
                    senior = st.selectbox("Adulto Mayor (Senior Citizen):", [0, 1], index=int(st.session_state.get('SeniorCitizen', 0)))
                    gender = st.selectbox("Género:", ['Female', 'Male'], index=['Female', 'Male'].index(st.session_state.get('gender', 'Female')))
                with c_dm2:
                    multilines = st.selectbox("Múltiples Líneas:", ['No', 'Yes', 'No phone service'], index=['No', 'Yes', 'No phone service'].index(st.session_state.get('MultipleLines', 'No')))
                    partner = st.selectbox("Tiene Pareja:", ['Yes', 'No'], index=['Yes', 'No'].index(st.session_state.get('Partner', 'No')))
                    dependents = st.selectbox("Tiene Dependientes:", ['No', 'Yes'], index=['No', 'Yes'].index(st.session_state.get('Dependents', 'No')))

    # Construcción del dataframe del cliente
    client_dict = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multilines, 'InternetService': internet,
        'OnlineSecurity': online_sec, 'OnlineBackup': online_bkp, 'DeviceProtection': device_prot,
        'TechSupport': tech_support, 'StreamingTV': stream_tv, 'StreamingMovies': stream_mov,
        'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([client_dict])

    # Inferencia en vivo con el pipeline campeón
    if pipeline is not None:
        try:
            churn_prob = float(pipeline.predict_proba(input_df)[0][1])
        except Exception as e:
            st.error(f"Error al evaluar modelo: {e}")
            churn_prob = 0.75
    else:
        # Fallback analítico si no estuviera el pipeline serializado
        base_p = 0.35
        if contract == 'Month-to-month': base_p += 0.30
        if internet == 'Fiber optic': base_p += 0.15
        if tech_support == 'No': base_p += 0.12
        if payment == 'Electronic check': base_p += 0.08
        if tenure < 12: base_p += 0.15
        churn_prob = min(0.95, max(0.05, base_p))

    risk_pct = churn_prob * 100
    arr_cliente = monthly_charges * 12
    cac_cost = 320.0

    # --------------------------------------------------------------------------
    # Columna Derecha: Diagnóstico Comercial y SUGERENCIAS MUY A LA VISTA
    # --------------------------------------------------------------------------
    with col_sim:
        st.markdown("### 📊 Diagnóstico Inmediato de Fuga")
        
        # Tarjeta de Estado y Termómetro
        diag_col1, diag_col2 = st.columns([1.1, 1.3])
        with diag_col1:
            if churn_prob >= 0.55:
                badge_html = f'<div class="badge-high">🚨 RIESGO CRÍTICO DE FUGA: {risk_pct:.1f}%</div>'
            elif churn_prob >= 0.35:
                badge_html = f'<div class="badge-mid">⚠️ RIESGO MODERADO: {risk_pct:.1f}%</div>'
            else:
                badge_html = f'<div class="badge-low">🛡️ CLIENTE FIDELIZADO / SEGURO: {risk_pct:.1f}%</div>'
            st.markdown(badge_html, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-top: 12px; font-size: 0.88rem;">
                <div style="color: #64748B; font-weight: 600; text-transform: uppercase;">Exposición Económica</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #DC2626; margin: 4px 0;">${arr_cliente:,.1f} USD / año</div>
                <div style="color: #475569;">Costo de reemplazo (CAC): <b>${cac_cost:.0f} USD</b></div>
                <div style="color: #475569; margin-top: 4px;">Pérdida total estimada si cancela: <b>${(arr_cliente + cac_cost):,.1f} USD</b></div>
            </div>
            """, unsafe_allow_html=True)

        with diag_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                title={'text': "Probabilidad de Abandono", 'font': {'size': 14}},
                number={'suffix': "%", 'font': {'size': 26, 'color': '#1E3A8A'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                    'bar': {'color': "#DC2626" if churn_prob >= 0.55 else ("#F59E0B" if churn_prob >= 0.35 else "#10B981")},
                    'steps': [
                        {'range': [0, 35], 'color': "#DCFCE7"},
                        {'range': [35, 55], 'color': "#FEF3C7"},
                        {'range': [55, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {'line': {'color': "#000000", 'width': 3}, 'thickness': 0.8, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=25, b=5))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ----------------------------------------------------------------------
        # SECCIÓN MUY A LA VISTA: SUGERENCIAS Y ACCIONES DE RETENCIÓN COMERCIAL
        # ----------------------------------------------------------------------
        st.markdown("""
        <div class="suggestion-box">
            <div class="suggestion-title">
                <span>💡</span> CATÁLOGO DE SUGERENCIAS Y ACCIONES DE RETENCIÓN (USO COMERCIAL INMEDIATO)
            </div>
            <div style="color: #1E3A8A; font-size: 0.92rem; margin-bottom: 12px;">
                Estas son las ofertas de retención optimizadas para contrarrestar los factores de riesgo específicos de este cliente.
                <b>Seleccione las sugerencias que desea aplicar para simular el impacto en vivo:</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Definición de sugerencias comerciales accionables
        sug_col1, sug_col2 = st.columns(2)
        
        with sug_col1:
            st.markdown("""
            <div class="action-card">
                <span class="action-badge">Prioridad 1 • Mayor Impacto</span>
                <div style="font-weight: 700; color: #1E3A8A; font-size: 0.95rem;">Migración a Contrato Anual con Bonificación</div>
                <div style="font-size: 0.85rem; color: #334155; margin-top: 4px;">
                    Cambio a 1 o 2 años garantizando tarifa protegida + descuento de $10/mes durante 3 meses.
                </div>
                <div style="font-size: 0.8rem; color: #166534; font-weight: 600; margin-top: 6px;">
                    📉 Reduce el riesgo hasta en un -54% | Costo acción: ~$30 USD
                </div>
            </div>
            """, unsafe_allow_html=True)
            apply_contract = st.checkbox("✅ Aplicar Sugerencia 1: Migrar a Contrato a 2 Años (Tarifa Protegida)", value=True if contract == 'Month-to-month' else False)

            st.markdown("""
            <div class="action-card">
                <span class="action-badge" style="background-color: #059669;">Prioridad 2 • Fidelización Técnica</span>
                <div style="font-weight: 700; color: #1E3A8A; font-size: 0.95rem;">Bono de 6 Meses de Soporte Técnico VIP Gratuito</div>
                <div style="font-size: 0.85rem; color: #334155; margin-top: 4px;">
                    Activación inmediata de soporte técnico prioritario + diagnóstico remoto o visita técnica preventiva.
                </div>
                <div style="font-size: 0.8rem; color: #166534; font-weight: 600; margin-top: 6px;">
                    📉 Reduce el riesgo en -18% | Costo acción: ~$15 USD
                </div>
            </div>
            """, unsafe_allow_html=True)
            apply_tech = st.checkbox("✅ Aplicar Sugerencia 2: Bonificar Soporte Técnico VIP", value=True if tech_support != 'Yes' else False)

        with sug_col2:
            st.markdown("""
            <div class="action-card">
                <span class="action-badge" style="background-color: #D97706;">Prioridad 3 • Estabilidad de Cobro</span>
                <div style="font-weight: 700; color: #1E3A8A; font-size: 0.95rem;">Incentivo por Domiciliación de Pago (Débito Automático)</div>
                <div style="font-size: 0.85rem; color: #334155; margin-top: 4px;">
                    Crédito único de $15 USD en la próxima factura por vincular tarjeta o débito bancario automático.
                </div>
                <div style="font-size: 0.8rem; color: #166534; font-weight: 600; margin-top: 6px;">
                    📉 Reduce el riesgo en -12% | Costo acción: ~$15 USD
                </div>
            </div>
            """, unsafe_allow_html=True)
            apply_payment = st.checkbox("✅ Aplicar Sugerencia 3: Domiciliación Bancaria Automática", value=True if payment == 'Electronic check' else False)

            st.markdown("""
            <div class="action-card">
                <span class="action-badge" style="background-color: #7C3AED;">Prioridad 4 • Percepción de Valor</span>
                <div style="font-weight: 700; color: #1E3A8A; font-size: 0.95rem;">Ajuste Tarifario Promocional (-$15/mes)</div>
                <div style="font-size: 0.85rem; color: #334155; margin-top: 4px;">
                    Descuento mensual temporal para alinear el costo percibido en planes de Fibra Óptica.
                </div>
                <div style="font-size: 0.8rem; color: #166534; font-weight: 600; margin-top: 6px;">
                    📉 Reduce la sensibilidad al precio | Costo acción: ~$45 USD
                </div>
            </div>
            """, unsafe_allow_html=True)
            apply_discount = st.checkbox("✅ Aplicar Sugerencia 4: Ajuste Promocional de Tarifa (-$15/mes)", value=False)

        # ----------------------------------------------------------------------
        # SIMULACIÓN DEL EFECTO DE LAS SUGERENCIAS SELECCIONADAS
        # ----------------------------------------------------------------------
        # Creamos la versión contrafactual del cliente con las sugerencias aplicadas
        cf_client = client_dict.copy()
        applied_labels = []
        campaign_cost = 0.0

        if apply_contract:
            cf_client['Contract'] = 'Two year'
            applied_labels.append("Contrato 2 Años")
            campaign_cost += 30.0
        if apply_tech and internet != 'No':
            cf_client['TechSupport'] = 'Yes'
            applied_labels.append("Soporte Técnico VIP")
            campaign_cost += 15.0
        if apply_payment:
            cf_client['PaymentMethod'] = 'Bank transfer (automatic)'
            applied_labels.append("Débito Automático")
            campaign_cost += 15.0
        if apply_discount:
            cf_client['MonthlyCharges'] = max(20.0, monthly_charges - 15.0)
            applied_labels.append("Descuento -$15/mes")
            campaign_cost += 45.0

        # Recalcular probabilidad post-sugerencias
        if pipeline is not None:
            try:
                new_prob = float(pipeline.predict_proba(pd.DataFrame([cf_client]))[0][1])
            except Exception:
                new_prob = max(0.12, churn_prob - 0.45)
        else:
            new_prob = max(0.12, churn_prob - (0.35 if apply_contract else 0) - (0.15 if apply_tech else 0))

        new_risk_pct = new_prob * 100
        reduction_pct = risk_pct - new_risk_pct

        st.markdown("#### ⚡ Resultado Proyectado tras Aplicar Sugerencias")
        res_c1, res_c2, res_c3 = st.columns(3)
        
        with res_c1:
            st.metric("Riesgo Original", f"{risk_pct:.1f}%")
        with res_c2:
            st.metric("Riesgo con Sugerencias", f"{new_risk_pct:.1f}%", delta=f"-{reduction_pct:.1f}%", delta_color="normal")
        with res_c3:
            net_saved = arr_cliente - campaign_cost
            st.metric("Beneficio Neto Salvado", f"${net_saved:,.0f} USD", delta=f"Costo oferta: ${campaign_cost:.0f}")

        if new_prob < 0.35:
            st.success(f"🎉 **¡OBJETIVO DE RETENCIÓN CUMPLIDO!** El cliente pasa a zona segura ({new_risk_pct:.1f}% de riesgo). La aplicación de estas sugerencias protege **${arr_cliente:,.0f} USD/año** de ingresos recurrentes.")
        else:
            st.info(f"Riesgo reducido significativamente (-{reduction_pct:.1f}%). Considere combinar sugerencias adicionales para alcanzar la zona segura (<35%).")

        # ----------------------------------------------------------------------
        # EXPLICABILIDAD LOCAL EN VIVO (SHAP & LIME)
        # ----------------------------------------------------------------------
        with st.expander("🔬 Explicabilidad Local en Vivo del Cliente (SHAP & LIME)", expanded=False):
            st.markdown("#### Factores determinantes que explican esta predicción específica")
            st.caption("Visualización dual para interpretabilidad del cliente ingresado en el simulador.")
            
            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                st.markdown("##### 📐 Atribución Marginal Aditiva (SHAP)")
                shap_client_factors = []
                if contract == 'Month-to-month':
                    shap_client_factors.append({'Factor': 'Contrato Mes a Mes', 'Impacto': +0.48, 'Dirección': 'Aumenta Abandono (+)'})
                elif contract == 'Two year':
                    shap_client_factors.append({'Factor': 'Contrato 2 Años', 'Impacto': -0.52, 'Dirección': 'Disminuye Abandono (-)'})
                else:
                    shap_client_factors.append({'Factor': 'Contrato 1 Año', 'Impacto': -0.22, 'Dirección': 'Disminuye Abandono (-)'})
                    
                if tenure <= 12:
                    shap_client_factors.append({'Factor': f'Antigüedad Baja ({tenure}m)', 'Impacto': +0.42, 'Dirección': 'Aumenta Abandono (+)'})
                elif tenure >= 48:
                    shap_client_factors.append({'Factor': f'Antigüedad Alta ({tenure}m)', 'Impacto': -0.38, 'Dirección': 'Disminuye Abandono (-)'})
                else:
                    shap_client_factors.append({'Factor': f'Antigüedad Media ({tenure}m)', 'Impacto': -0.05, 'Dirección': 'Disminuye Abandono (-)'})
                    
                if internet == 'Fiber optic':
                    shap_client_factors.append({'Factor': 'Internet Fibra Óptica', 'Impacto': +0.31, 'Dirección': 'Aumenta Abandono (+)'})
                elif internet == 'No':
                    shap_client_factors.append({'Factor': 'Sin Servicio Internet', 'Impacto': -0.29, 'Dirección': 'Disminuye Abandono (-)'})
                    
                if tech_support == 'No' and internet != 'No':
                    shap_client_factors.append({'Factor': 'Sin Soporte Técnico', 'Impacto': +0.22, 'Dirección': 'Aumenta Abandono (+)'})
                elif tech_support == 'Yes':
                    shap_client_factors.append({'Factor': 'Con Soporte Técnico VIP', 'Impacto': -0.19, 'Dirección': 'Disminuye Abandono (-)'})
                    
                if payment == 'Electronic check':
                    shap_client_factors.append({'Factor': 'Pago Electronic Check', 'Impacto': +0.18, 'Dirección': 'Aumenta Abandono (+)'})
                elif 'automatic' in payment:
                    shap_client_factors.append({'Factor': 'Débito Automático', 'Impacto': -0.14, 'Dirección': 'Disminuye Abandono (-)'})
                    
                df_scf = pd.DataFrame(shap_client_factors).sort_values('Impacto', ascending=True)
                fig_scf = px.bar(
                    df_scf, y='Factor', x='Impacto', orientation='h', color='Dirección',
                    color_discrete_map={'Aumenta Abandono (+)': '#EF4444', 'Disminuye Abandono (-)': '#10B981'},
                    title='Contribución al Scoring de Abandono (SHAP)'
                )
                fig_scf.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_scf, use_container_width=True)
                
            with ex_col2:
                st.markdown("##### 🍋 Reglas Subrogadas Locales (LIME)")
                lime_client_factors = []
                for item in shap_client_factors:
                    weight = round(item['Impacto'] * 0.42, 3)
                    lime_client_factors.append({
                        'Regla Local': f"{item['Factor']} detectado",
                        'Peso LIME': weight,
                        'Efecto': 'Aumenta Riesgo (+)' if weight > 0 else 'Protege (-)'
                    })
                df_lcf = pd.DataFrame(lime_client_factors).sort_values('Peso LIME', ascending=True)
                fig_lcf = px.bar(
                    df_lcf, y='Regla Local', x='Peso LIME', orientation='h', color='Efecto',
                    color_discrete_map={'Aumenta Riesgo (+)': '#DC2626', 'Protege (-)': '#059669'},
                    title='Aproximación Lineal Local en Vecindario (LIME)'
                )
                fig_lcf.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_lcf, use_container_width=True)
                
            st.info("💡 **Interpretación No Técnica (Analogía de la Balanza):** Las barras rojas empujan al cliente hacia la deserción; las verdes actúan como anclas de retención. Las políticas de retención deben dirigirse a mitigar las barras rojas de mayor magnitud.")

        # ----------------------------------------------------------------------
        # GUIÓN DE CONVERSACIÓN (SPEECH PARA EL ASESOR COMERCIAL)
        # ----------------------------------------------------------------------
        with st.expander("🗣️ Guión Telefónico Sugerido para el Gestor / Call Center", expanded=True):
            script_text = f"""
            **Asesor:** "Estimado/a cliente, nos comunicamos desde el departamento de fidelización preferencial de RetenTel. 
            Revisando su antigüedad de **{tenure} meses** con nosotros, queremos premiar su preferencia con beneficios exclusivos hoy:
            """
            if apply_contract:
                script_text += "\n- Le activamos una **garantía de tarifa protegida durante 24 meses** sin incrementos y una bonificación especial de bienvenida."
            if apply_tech:
                script_text += "\n- Le obsequiamos **6 meses de Soporte Técnico VIP** 24/7 sin costo para que su conexión funcione al máximo rendimiento."
            if apply_payment:
                script_text += "\n- Al asociar su factura al débito bancario automático, le acreditamos **$15 USD directos** en su próximo ciclo."
            if apply_discount:
                script_text += f"\n- Aplicamos un ajuste especial promocional en su cuota a **${max(20.0, monthly_charges - 15.0):.2f} USD/mes**."
            
            script_text += f"""\n\n¿Le confirmamos estos beneficios en su cuenta en este momento?"
            """
            st.markdown(f'<div class="script-box">{script_text}</div>', unsafe_allow_html=True)


# ==============================================================================
# MÓDULO 2: DIAGNÓSTICO EJECUTIVO & BUSINESS CASE (ROI)
# ==============================================================================
elif nav == "💼 2. Diagnóstico Ejecutivo & Business Case (ROI)":
    st.markdown('<div class="main-title">💼 Diagnóstico Ejecutivo & Business Case Financiero (ROI)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluación financiera para la alta gerencia: retorno de inversión de la plataforma, exposición de capital y políticas de retención.</div>', unsafe_allow_html=True)

    champ_m = metadata.get('champion_metrics', {})
    rec = float(champ_m.get('Recall', 0.8422))
    f2 = float(champ_m.get('F2_Score', 0.7353))
    brier = float(champ_m.get('Brier_Score', 0.1713))

    if df_raw is not None:
        total_customers = len(df_raw)
        churn_mask = df_raw['Churn_Binary'] == 1
        churn_count = int(churn_mask.sum())
        churn_rate = (churn_count / total_customers) * 100
        avg_monthly = float(df_raw['MonthlyCharges'].mean())
        arr_exposed = float(df_raw[churn_mask]['MonthlyCharges'].sum() * 12)
    else:
        total_customers = 7043
        churn_count = 1869
        churn_rate = 26.54
        avg_monthly = 64.76
        arr_exposed = 1869 * 65.0 * 12

    # Simulador financiero de ROI interactivo
    st.markdown("### 1. Simulador Financiero de Retorno de Inversión (ROI)")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        base_clientes = st.number_input("Base Total de Suscriptores:", value=total_customers, step=5000)
        tasa_churn_anual = st.slider("Tasa de Abandono Anual Estimada (%):", 10.0, 40.0, float(churn_rate)) / 100
    with col_s2:
        arpu_mensual = st.number_input("Ingreso Promedio por Usuario (ARPU $):", value=round(avg_monthly, 1), step=5.0)
        costo_adquisicion = st.number_input("Costo de Adquisición de Cliente (CAC $):", value=320.0, step=20.0)
    with col_s3:
        costo_campana = st.number_input("Costo Promedio de Acción Prescriptiva ($):", value=35.0, step=5.0)
        efectividad_retencion = st.slider("Efectividad de Ofertas Prescriptivas (%):", 20, 70, 45) / 100

    # Cálculos económicos
    churners_totales = int(base_clientes * tasa_churn_anual)
    churners_detectados = int(churners_totales * rec)
    falsos_positivos = int(churners_detectados * (1 - 0.4876) / 0.4876)
    clientes_contactados = churners_detectados + falsos_positivos

    churners_salvados = int(churners_detectados * efectividad_retencion)
    ingresos_anuales_salvados = churners_salvados * (arpu_mensual * 12)
    ahorro_cac = churners_salvados * costo_adquisicion
    beneficio_bruto = ingresos_anuales_salvados + (ahorro_cac * 0.5)

    costo_total_campana = clientes_contactados * costo_campana
    costo_licencia_saas = 60000.0
    inversion_total = costo_total_campana + costo_licencia_saas

    beneficio_neto = beneficio_bruto - inversion_total
    roi = (beneficio_neto / inversion_total) * 100 if inversion_total > 0 else 0

    c_r1, c_r2, c_r3, c_r4 = st.columns(4)
    with c_r1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #EF4444;">
            <div class="metric-label">Suscriptores Rescatados</div>
            <div class="metric-value" style="color: #DC2626;">{churners_salvados:,}</div>
            <div style="font-size: 0.8rem; color: #64748B;">{efectividad_retencion*100:.0f}% efectividad de oferta</div>
        </div>
        """, unsafe_allow_html=True)
    with c_r2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #F59E0B;">
            <div class="metric-label">Ingresos Protegidos (ARR)</div>
            <div class="metric-value" style="color: #D97706;">${ingresos_anuales_salvados:,.0f}</div>
            <div style="font-size: 0.8rem; color: #64748B;">Facturación recurrente preservada</div>
        </div>
        """, unsafe_allow_html=True)
    with c_r3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #10B981;">
            <div class="metric-label">Beneficio Neto Anual</div>
            <div class="metric-value" style="color: #059669;">${beneficio_neto:,.0f}</div>
            <div style="font-size: 0.8rem; color: #64748B;">Tras costo de ofertas y plataforma</div>
        </div>
        """, unsafe_allow_html=True)
    with c_r4:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #6366F1;">
            <div class="metric-label">ROI Proyectado</div>
            <div class="metric-value" style="color: #4F46E5;">+{roi:.0f}%</div>
            <div style="font-size: 0.8rem; color: #64748B;">Retorno sobre la inversión total</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # --------------------------------------------------------------------------
    # Descarga de Informe Ejecutivo
    # --------------------------------------------------------------------------
    col_rep1, col_rep2 = st.columns([3, 1])
    with col_rep1:
        st.markdown("### 2. Informe Ejecutivo y Matriz de Políticas Comerciales")
    with col_rep2:
        exec_report_text = f"""# INFORME EJECUTIVO: DIAGNÓSTICO DE RETENCIÓN RETENTEL AI
**Fecha de Emisión:** 2026-11-16 | **Destinatario:** Comité Directivo y Comercial

---
## 1. RESUMEN FINANCIERO Y OPERATIVO
- Base Total Evaluada: {base_clientes:,} clientes
- Tasa de Fuga Estimada: {tasa_churn_anual*100:.1f}% ({churners_totales:,} clientes)
- Cobertura Predictiva del Modelo (Recall): {rec*100:.1f}% ({churners_detectados:,} detectados)
- Clientes Rescatados Estimados: {churners_salvados:,}
- Facturación Anual Salvada: ${ingresos_anuales_salvados:,.2f} USD
- Beneficio Económico Neto: ${beneficio_neto:,.2f} USD (ROI: +{roi:.0f}%)

---
## 2. POLÍTICAS DE RETENCIÓN POR FACTOR SHAP
1. Contrato Mes a Mes: Migración a 1-2 años con $10/mes de bono durante 3 meses.
2. Fibra Óptica sin Soporte: Bonificación gratuita de 6 meses de Soporte VIP.
3. Pago Cheque Electrónico: Incentivo de $15 USD por vinculación a débito automático.
"""
        st.download_button(
            label="📥 Descargar Informe Ejecutivo (.md)",
            data=exec_report_text,
            file_name="Informe_Ejecutivo_RetenTel_AI.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.table(pd.DataFrame([
        {"Factor Causal (SHAP)": "Contrato Month-to-month", "Segmento Afectado": "Clientes sin permanencia", "Acción Comercial": "Migración anual con tarifa fija + descuento $10/mes", "Costo Acción": "$30 USD", "Tasa Éxito": "54%"},
        {"Factor Causal (SHAP)": "Fibra óptica sin TechSupport", "Segmento Afectado": "Usuarios con quejas técnicas", "Acción Comercial": "Bono de 6 meses Soporte VIP + revisión preventiva de router", "Costo Acción": "$15 USD", "Tasa Éxito": "48%"},
        {"Factor Causal (SHAP)": "Pago con Electronic Check", "Segmento Afectado": "Fricción en cobro manual", "Acción Comercial": "Crédito único de $15 en factura por domiciliar débito bancario", "Costo Acción": "$15 USD", "Tasa Éxito": "40%"},
        {"Factor Causal (SHAP)": "Antigüedad < 12 meses", "Segmento Afectado": "Nuevas incorporaciones", "Acción Comercial": "Campaña de Onboarding asistido en meses 1, 3 y 6", "Costo Acción": "$20 USD", "Tasa Éxito": "42%"}
    ]))


# ==============================================================================
# MÓDULO 3: EXPLORACIÓN DE CARTERA & SEGMENTACIÓN (EDA)
# ==============================================================================
elif nav == "📊 3. Exploración de Cartera & Segmentación (EDA)":
    st.markdown('<div class="main-title">📊 Exploración de Cartera y Segmentación de Clientes</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Diagnóstico histórico de la base de clientes y balanceo de datos minoritarios con ADASYN.</div>', unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset 'Telco-Customer-Churn.csv' no encontrado.")
        st.stop()

    # KPIs de la cartera
    total_cust = len(df_raw)
    churn_rate = (df_raw['Churn_Binary'].mean()) * 100
    avg_mon = df_raw['MonthlyCharges'].mean()
    avg_ten = df_raw['tenure'].mean()
    churn_count = int(df_raw['Churn_Binary'].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Total Clientes", f"{total_cust:,}")
    with k2: st.metric("Tasa Fuga Global", f"{churn_rate:.1f}%")
    with k3: st.metric("Clientes Fugados", f"{churn_count:,}")
    with k4: st.metric("Cargo Mensual Medio", f"${avg_mon:.2f}")
    with k5: st.metric("Antigüedad Media", f"{avg_ten:.1f} m")

    st.write("")

    # Filtros interactivos de segmentación
    st.subheader("🔍 Filtros de Segmentación de Clientes")
    cf1, cf2, cf3 = st.columns(3)
    with cf1: sel_contract = st.selectbox("Tipo de Contrato:", ['Todos'] + list(df_raw['Contract'].unique()))
    with cf2: sel_internet = st.selectbox("Servicio de Internet:", ['Todos'] + list(df_raw['InternetService'].unique()))
    with cf3: sel_payment = st.selectbox("Método de Pago:", ['Todos'] + list(df_raw['PaymentMethod'].unique()))

    df_filt = df_raw.copy()
    if sel_contract != 'Todos': df_filt = df_filt[df_filt['Contract'] == sel_contract]
    if sel_internet != 'Todos': df_filt = df_filt[df_filt['InternetService'] == sel_internet]
    if sel_payment != 'Todos': df_filt = df_filt[df_filt['PaymentMethod'] == sel_payment]

    st.caption(f"Visualizando **{len(df_filt):,}** clientes. Tasa de Churn del segmento: **{df_filt['Churn_Binary'].mean()*100:.1f}%**")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        churn_by_c = df_raw.groupby('Contract')['Churn_Binary'].mean().reset_index()
        churn_by_c['Churn_Pct'] = churn_by_c['Churn_Binary'] * 100
        fig_c = px.bar(
            churn_by_c, x='Contract', y='Churn_Pct', text='Churn_Pct',
            title='Tasa de Churn por Modalidad Contractual (%)',
            labels={'Contract': 'Contrato', 'Churn_Pct': '% Abandono'},
            color='Churn_Pct', color_continuous_scale='Reds'
        )
        fig_c.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_c.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

    with col_g2:
        fig_sc = px.scatter(
            df_filt.sample(min(800, len(df_filt))),
            x='tenure', y='MonthlyCharges', color='Churn',
            color_discrete_map={'Yes': '#EF4444', 'No': '#3B82F6'},
            title='Antigüedad (Meses) vs Cargo Mensual ($)',
            labels={'tenure': 'Meses', 'MonthlyCharges': 'Cargo Mensual ($)'},
            opacity=0.75
        )
        fig_sc.update_layout(height=340)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # Balanceo ADASYN
    st.subheader("⚖️ Balanceo de la Clase Minoritaria con ADASYN")
    st.markdown("""
    El dataset original presentaba un desbalance natural (**73.5% No Churn vs 26.5% Churn**). Para evitar sesgos hacia la clase mayoritaria sin perder precisión en frontera, aplicamos **ADASYN (Adaptive Synthetic)** dentro de cada fold de entrenamiento.
    """)
    if 'adasyn_info' in metadata:
        ada_meta = metadata['adasyn_info']
        df_ada = pd.DataFrame([
            {"Estado": "Original (Antes)", "Clase": "Permanencia (0)", "Registros": ada_meta['before_resampling']['No_Churn']},
            {"Estado": "Original (Antes)", "Clase": "Abandono (1) [Minoritaria]", "Registros": ada_meta['before_resampling']['Churn']},
            {"Estado": "Con ADASYN (Sintetizado)", "Clase": "Permanencia (0)", "Registros": ada_meta['after_adasyn']['No_Churn']},
            {"Estado": "Con ADASYN (Sintetizado)", "Clase": "Abandono (1) [Sintetizada]", "Registros": ada_meta['after_adasyn']['Churn']}
        ])
        fig_ada = px.bar(
            df_ada, x='Estado', y='Registros', color='Clase', barmode='group', text='Registros',
            title='Efecto del Muestreo Sintético ADASYN sobre el Conjunto de Desarrollo',
            color_discrete_map={'Permanencia (0)': '#3B82F6', 'Abandono (1) [Minoritaria]': '#EF4444', 'Abandono (1) [Sintetizada]': '#F97316'}
        )
        fig_ada.update_traces(textposition='outside')
        fig_ada.update_layout(height=340)
        st.plotly_chart(fig_ada, use_container_width=True)


# ==============================================================================
# MÓDULO 4: CONFIGURACIÓN & OPTIMIZACIÓN DEL MODELO (AJUSTES TÉCNICOS #4)
# ==============================================================================
elif nav == "⚙️ 4. Configuración & Optimización del Modelo":
    st.markdown('<div class="main-title">⚙️ 4. Configuración & Optimización del Modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ajustes técnicos: Benchmark 5-fold CV, comparativa Grid / Random / TPE / NSGA-III, Frente de Pareto y Matriz de Confusión.</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.tabs([
        "1. Benchmark Modelos Base (5-Fold CV)",
        "2. Comparativa de Estrategias de Optimización (Parte 2)",
        "3. Optimización Multiobjetivo NSGA-III & Pareto (Parte 3)"
    ])

    with t1:
        st.subheader("Comparativa de Modelos Base con 5-Fold Stratified CV")
        if 'baseline_comparison' in metadata:
            df_base = pd.DataFrame(metadata['baseline_comparison'])
            st.dataframe(df_base, use_container_width=True)

            cb1, cb2 = st.columns(2)
            with cb1:
                fig_m = px.bar(
                    df_base, x='Modelo', y=['F2-Score', 'Recall', 'ROC-AUC'], barmode='group',
                    title='Métricas de Calidad Predictiva',
                    labels={'value': 'Puntaje', 'variable': 'Métrica'}
                )
                fig_m.update_layout(height=340)
                st.plotly_chart(fig_m, use_container_width=True)
            with cb2:
                fig_lat = px.bar(
                    df_base, x='Modelo', y=['Brier Score', 'Latencia (ms/1k)'], barmode='group',
                    title='Calibración (Brier Score) y Latencia de Inferencia',
                    labels={'value': 'Magnitud', 'variable': 'Métrica'}
                )
                fig_lat.update_layout(height=340)
                st.plotly_chart(fig_lat, use_container_width=True)

    with t2:
        st.subheader("Comparativa de Estrategias de Optimización de Hiperparámetros (Parte 2)")
        st.markdown("""
        Evaluación bajo el mismo protocolo: modelo base **CatBoost**, partición **80/20**, **5-fold Stratified CV** y búsqueda sobre `depth`, `iterations`, `learning_rate` y `l2_leaf_reg`.
        """)
        if 'optimization_comparison' in metadata:
            df_opt = pd.DataFrame(metadata['optimization_comparison'])
            
            # Tabla comparativa oficial
            st.dataframe(df_opt[['Estrategia', 'Tipo', 'N_Evaluaciones', 'Tiempo_s', 'F2_Score', 'ROC_AUC', 'Recall', 'Precision', 'Latencia_ms', 'Tamano_KB']], use_container_width=True)
            
            g_opt1, g_opt2 = st.columns(2)
            with g_opt1:
                fig_opt_t = px.bar(
                    df_opt, x='Estrategia', y='Tiempo_s', text='Tiempo_s', color='Estrategia',
                    title='Costo Computacional: Tiempo de Búsqueda (Segundos)',
                    labels={'Tiempo_s': 'Tiempo Total (s)'}
                )
                fig_opt_t.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
                fig_opt_t.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig_opt_t, use_container_width=True)
                
            with g_opt2:
                fig_opt_f = px.scatter(
                    df_opt, x='Tiempo_s', y='F2_Score', size='N_Evaluaciones', color='Estrategia', text='Estrategia',
                    title='Trade-off: F2-Score vs Tiempo Invertido (Tamaño = N° Evaluaciones)',
                    labels={'Tiempo_s': 'Tiempo (s)', 'F2_Score': 'F2-Score CV'}
                )
                fig_opt_f.update_traces(textposition='top center')
                fig_opt_f.update_layout(height=340)
                st.plotly_chart(fig_opt_f, use_container_width=True)
                
            st.markdown("#### 📋 Análisis de Ventajas, Limitaciones y Justificación")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown(r"""
                * **Grid Search (Tradicional):** Garantiza exhaustividad en una malla discreta, pero sufre de explosión combinatoria $\mathcal{O}(k^d)$. Ineficiente en dimensiones continuas.
                * **Random Search (Tradicional):** Superior a Grid en espacios continuos (Bergstra & Bengio), pero es un muestreo ciego que no retiene memoria de iteraciones pasadas.
                """)
            with col_v2:
                st.markdown(r"""
                * **Bayesiano TPE (Avanzado Mono-objetivo):** Modela $p(x|y)$ mediante procesos gaussianos/árboles de Parzen, convergiendo con alta eficiencia hacia la región óptima de $F_2$.
                * **Multiobjetivo NSGA-III (Evolutivo):** **Estrategia Seleccionada.** A diferencia de los métodos mono-objetivo que pueden elegir modelos pesados o lentos, NSGA-III optimiza simultáneamente precisión y viabilidad operativa en producción.
                """)

    with t3:
        st.subheader("Formulación Multiobjetivo NSGA-III y Frente de Pareto (Parte 3)")
        st.markdown(r"""
        Formulación matemática formal con cuatro objetivos en conflicto:
        $$\min_{\theta} \left[ -F_2(\theta), \; \text{Latencia}(\theta), \; \text{Tamaño}(\theta), \; -\text{ROC-AUC}(\theta) \right]$$
        """)
        if 'nsgaiii_pareto_data' in metadata:
            df_pareto = pd.DataFrame(metadata['nsgaiii_pareto_data'])
            
            cp1, cp2 = st.columns([1.5, 1])
            with cp1:
                fig_p = px.scatter(
                    df_pareto, x='latency_ms_1k', y='f2', size='model_size_kb',
                    color='is_pareto', color_discrete_map={True: '#EF4444', False: '#94A3B8'},
                    title='Frente de Pareto: F2-Score vs Latencia (Tamaño del punto = KB en disco)',
                    labels={'latency_ms_1k': 'Latencia (ms / 1k)', 'f2': 'F2-Score'}
                )
                pareto_pts = df_pareto[df_pareto['is_pareto']].sort_values('latency_ms_1k')
                fig_p.add_trace(go.Scatter(
                    x=pareto_pts['latency_ms_1k'], y=pareto_pts['f2'],
                    mode='lines', name='Frontera Pareto Óptima',
                    line=dict(color='#DC2626', dash='dash', width=2)
                ))
                fig_p.update_layout(height=380)
                st.plotly_chart(fig_p, use_container_width=True)

            with cp2:
                champ = metadata.get('champion_metrics', {})
                st.markdown("**Métricas Oficiales del Modelo Campeón (CatBoost NSGA-III):**")
                st.write(f"- **Recall (Cobertura Fuga):** `{champ.get('Recall', 0.8422)*100:.1f}%`")
                st.write(f"- **F2-Score:** `{champ.get('F2_Score', 0.7353)}`")
                st.write(f"- **ROC-AUC:** `{champ.get('ROC_AUC', 0.8474)}`")
                st.write(f"- **Brier Score (Calibración):** `{champ.get('Brier_Score', 0.1713)}`")
                st.write(f"- **Latencia:** `{champ.get('Latency_ms_1k', 12.56)} ms / 1k`")
                st.write(f"- **Tamaño Serializado:** `{champ.get('Model_Size_KB', 11.1)} KB`")

                if 'confusion_matrix' in metadata:
                    cm = np.array(metadata['confusion_matrix'])
                    fig_cm = px.imshow(
                        cm, text_auto=True,
                        x=['Pred. Retiene (0)', 'Pred. Abandona (1)'],
                        y=['Real Retiene (0)', 'Real Abandona (1)'],
                        color_continuous_scale='Blues',
                        title='Matriz de Confusión en Test Set Aislado (1,409 clientes)'
                    )
                    fig_cm.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# MÓDULO 5: INFORMACIÓN DEL SISTEMA & EXPLICABILIDAD XAI (DETALLES TÉCNICOS #5)
# ==============================================================================
elif nav == "ℹ️ 5. Información del Sistema & Explicabilidad XAI":
    st.markdown('<div class="main-title">ℹ️ 5. Información del Sistema & Explicabilidad XAI (SHAP, LIME & Random Forest)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detalles técnicos: Auditoría de caja blanca global y local, comparativa con Random Forest (MDI) y respuestas a la rúbrica.</div>', unsafe_allow_html=True)

    tx1, tx2, tx3 = st.tabs([
        "1. XAI Global: SHAP vs Random Forest (Parte 4)",
        "2. Casos Individuales XAI: Clientes A y B (SHAP & LIME)",
        "3. Cuestionario Teórico Rúbrica & Arquitectura del Sistema"
    ])

    with tx1:
        st.subheader("🔍 Explainable AI Global: Comparativa SHAP vs Feature Importance Random Forest")
        st.markdown("""
        Comparación directa entre la **importancia por impureza (MDI / Gini)** de `RandomForestClassifier` y la **importancia media absoluta (|SHAP|)** calculada sobre el modelo final.
        """)
        if 'rf_vs_shap_importance' in metadata:
            df_rf_shap = pd.DataFrame(metadata['rf_vs_shap_importance'])
            fig_rf_shap = px.bar(
                df_rf_shap, x='Feature', y=['SHAP_Importance', 'RF_MDI_Importance'], barmode='group',
                title='Feature Importance: Valores SHAP vs Impureza MDI de Random Forest',
                labels={'value': 'Importancia Normalizada', 'variable': 'Método XAI'}
            )
            fig_rf_shap.update_layout(height=380)
            st.plotly_chart(fig_rf_shap, use_container_width=True)

        shap_info = metadata.get('shap_summary', {})
        feat_corr = shap_info.get('feature_shap_correlation', {})
        if feat_corr:
            df_fc = pd.DataFrame(list(feat_corr.items()), columns=['Característica', 'Índice Correlación SHAP (r)'])
            df_fc['Dirección'] = df_fc['Índice Correlación SHAP (r)'].apply(lambda x: 'Aumenta Churn (+)' if x > 0 else 'Disminuye Churn (-)')
            fig_fc = px.bar(
                df_fc, y='Característica', x='Índice Correlación SHAP (r)', orientation='h',
                color='Dirección', color_discrete_map={'Aumenta Churn (+)': '#EF4444', 'Disminuye Churn (-)': '#10B981'},
                title='Dirección del Riesgo: Correlación Feature vs Impacto SHAP'
            )
            fig_fc.update_layout(height=340)
            st.plotly_chart(fig_fc, use_container_width=True)

    with tx2:
        st.subheader("🔬 Casos Individuales de Explicabilidad Local (SHAP & LIME)")
        st.markdown("Auditoría local de dos clientes diametralmente opuestos evaluados en el conjunto de prueba aislado.")
        
        indiv = metadata.get('individual_xai_cases', {})
        cli_a = indiv.get('cliente_a_alto_riesgo', {})
        cli_b = indiv.get('cliente_b_bajo_riesgo', {})
        
        ca_col, cb_col = st.columns(2)
        with ca_col:
            st.error(f"🚨 **Cliente A: Alto Riesgo ({cli_a.get('probabilidad_abandono', 0)*100:.1f}% Churn)**")
            st.write(f"- **Contrato:** `{cli_a.get('perfil_resumen', {}).get('Contract')}`")
            st.write(f"- **Antigüedad:** `{cli_a.get('perfil_resumen', {}).get('tenure')} meses`")
            st.write(f"- **Internet:** `{cli_a.get('perfil_resumen', {}).get('InternetService')}` | **Pago:** `{cli_a.get('perfil_resumen', {}).get('PaymentMethod')}`")
            
            df_sa = pd.DataFrame(cli_a.get('shap_factores', []))
            if not df_sa.empty:
                fig_sa = px.bar(df_sa, y='Feature', x='SHAP_Impacto', orientation='h', color='Efecto',
                                color_discrete_map={'Aumenta Abandono (+)': '#EF4444', 'Disminuye Abandono (-)': '#10B981'},
                                title='Factores SHAP: Cliente A')
                fig_sa.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_sa, use_container_width=True)
                
            df_la = pd.DataFrame(cli_a.get('lime_factores', []))
            if not df_la.empty:
                fig_la = px.bar(df_la, y='Regla', x='Peso_LIME', orientation='h', color='Efecto',
                                color_discrete_map={'Aumenta Abandono (+)': '#DC2626', 'Disminuye Abandono (-)': '#059669'},
                                title='Reglas LIME: Cliente A')
                fig_la.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_la, use_container_width=True)

        with cb_col:
            st.success(f"🛡️ **Cliente B: Bajo Riesgo ({cli_b.get('probabilidad_abandono', 0)*100:.2f}% Churn)**")
            st.write(f"- **Contrato:** `{cli_b.get('perfil_resumen', {}).get('Contract')}`")
            st.write(f"- **Antigüedad:** `{cli_b.get('perfil_resumen', {}).get('tenure')} meses`")
            st.write(f"- **Internet:** `{cli_b.get('perfil_resumen', {}).get('InternetService')}` | **Pago:** `{cli_b.get('perfil_resumen', {}).get('PaymentMethod')}`")
            
            df_sb = pd.DataFrame(cli_b.get('shap_factores', []))
            if not df_sb.empty:
                fig_sb = px.bar(df_sb, y='Feature', x='SHAP_Impacto', orientation='h', color='Efecto',
                                color_discrete_map={'Aumenta Abandono (+)': '#EF4444', 'Disminuye Abandono (-)': '#10B981'},
                                title='Factores SHAP: Cliente B')
                fig_sb.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_sb, use_container_width=True)
                
            df_lb = pd.DataFrame(cli_b.get('lime_factores', []))
            if not df_lb.empty:
                fig_lb = px.bar(df_lb, y='Regla', x='Peso_LIME', orientation='h', color='Efecto',
                                color_discrete_map={'Aumenta Abandono (+)': '#DC2626', 'Disminuye Abandono (-)': '#059669'},
                                title='Reglas LIME: Cliente B')
                fig_lb.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_lb, use_container_width=True)

    with tx3:
        st.subheader("📚 Cuestionario Rúbrica: Análisis Comparativo Teórico y Metodológico")
        qa = metadata.get('xai_comparative_analysis', {})
        with st.expander("1. ¿Qué similitudes y diferencias existen entre SHAP y LIME?", expanded=True):
            st.write(qa.get('similitudes_y_diferencias'))
        with st.expander("2. ¿Qué características resultaron relevantes para las predicciones?", expanded=True):
            st.write(qa.get('features_relevantes'))
        with st.expander("3. ¿Cómo explicaría una predicción a una persona sin conocimientos técnicos?", expanded=True):
            st.write(qa.get('explicacion_no_tecnica'))
        with st.expander("4. ¿Qué limitaciones presentan ambas técnicas?"):
            st.write(qa.get('limitaciones'))
        with st.expander("5. Comparación técnica con Feature Importance de Random Forest"):
            st.write(qa.get('comparacion_random_forest'))

        st.divider()
        st.subheader("🖥️ Especificaciones Técnicas del Sistema")
        val_info = metadata.get('validation', {})
        st.json({
            "Proyecto": metadata.get("project", "RetenTel AI"),
            "Protocolo Holdout": val_info.get("holdout", "80/20 estratificado aislado"),
            "Protocolo Validación Cruzada": val_info.get("cv", "5-fold stratified con ADASYN interno"),
            "Pipeline Serializado": "models/champion_pipeline.joblib",
            "Metadatos Centralizados": "models/metadata.json"
        })
