"""Inicio: la promesa sobre la montaña, el producto con datos reales, dos destinos y la verificación del modelo."""
import pandas as pd
import streamlit as st

import components as c
import core
import plans

X, y, p = core.holdout()
alert = c.alert_threshold()
m = core.metrics_at(y, p, alert)
cost = c.costs()


@st.cache_data(show_spinner=False)
def sample_scored(threshold: float) -> pd.DataFrame:
    return core.score_portfolio(pd.read_csv(core.SAMPLE_PATH), threshold)


@st.cache_data(show_spinner=False)
def sample_features() -> pd.DataFrame:
    return core.prepare_portfolio(pd.read_csv(core.SAMPLE_PATH)).reset_index(drop=True)[core.model_features()]


scored = sample_scored(alert)
top = scored.head(5)[["Cliente", "Probabilidad", "Palanca principal"]].rename(columns={"Palanca principal": "palanca"})
n_alto = int((scored["Nivel"] == "Alto").sum())
BUDGET = 1500
campaign = plans.campaign_at(
    plans.campaign_curve(c.cached_portfolio_plans(sample_features(), cost["support"], cost["security"], cost["contact"]), "valor"),
    BUDGET,
)

with st.container(key="hero"):
    copy_col, product_col = st.columns([1, 1.05], gap="large", vertical_alignment="center")
    with copy_col:
        st.html(
            '<div class="hero"><h1>Sepa qué clientes se van y qué hacer con cada uno.</h1>'
            "<p>RetenTel AI estima el riesgo de abandono de cada cliente y calcula el plan de retención que más valor deja.</p></div>"
        )
        with st.container(key="hero_cta", horizontal=True):
            with st.container(key="cta_main"):
                st.page_link(c.P_ANALIZAR, label="Analizar un cliente")
            with st.container(key="cta_alt"):
                st.page_link(c.P_CARTERA, label="Revisar la cartera")
    with product_col:
        st.html(c.preview_card(top, alert, len(scored)))

st.html('<div class="sec-title">De la lista a la decisión</div>')
left, right = st.columns(2, gap="medium")
with left:
    c.tile(
        "cartera", "Cartera de clientes",
        "Cargue un CSV, ordene a sus clientes por riesgo y descargue la lista de contacto.",
        f'<div class="tile-fig">{n_alto} de {len(scored)}<small>en riesgo alto</small></div>',
        c.P_CARTERA, "Abrir la cartera  →",
    )
with right:
    c.tile(
        "campana", "Campaña",
        "Con un presupuesto dado, elige a quién contactar y con qué plan para retener el mayor ingreso.",
        f'<div class="tile-fig">{c.usd(campaign["valor"])}<small>con {c.usd(BUDGET)} de presupuesto</small></div>',
        c.P_CAMPANA, "Planificar una campaña  →",
    )

with st.container(key="band"):
    st.html(
        '<div class="band-grid"><div><h2>Un modelo que se puede verificar</h2>'
        f"<p>Recalculado con {c.num(len(y))} clientes que no usó para entrenar.</p></div>"
        f'<div class="band-num"><b>{m["recall"]:.0%}</b><span>de quienes se van, detectados</span></div>'
        f'<div class="band-num"><b>{m["precision"]:.0%}</b><span>de las alertas son reales</span></div>'
        f'<div class="band-num"><b>{c.num(m["auc"], 2)}</b><span>AUC</span></div></div>'
    )
    st.page_link(c.P_CALIDAD, label="Ver la calidad del modelo  →")

with st.container(key="sistema_link"):
    st.page_link(c.P_SISTEMA, label="Información del sistema  →")
