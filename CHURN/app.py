"""RetenTel AI: portal de retención de clientes. Punto de entrada.

Ejecutar desde esta carpeta:  streamlit run app.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import streamlit as st

import components as c
import core

HERE = Path(__file__).resolve().parent
log = logging.getLogger("retentel")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="RetenTel AI · Retención de clientes",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "RetenTel AI: riesgo de abandono por cliente, con explicación y acciones sugeridas."},
)

css = (HERE / "static" / "style.css").read_text(encoding="utf-8")
hero = quote((HERE / "static" / "hero.svg").read_text(encoding="utf-8"))
st.markdown(
    f'<style>:root{{--hero-img:url("data:image/svg+xml;charset=utf-8,{hero}")}}\n{css}</style>',
    unsafe_allow_html=True,
)
st.session_state.setdefault("alert", core.ALERT_DEFAULT)

# Comprobación de arranque: sin estos archivos ninguna página puede funcionar.
missing_files = core.missing_assets()
if missing_files:
    log.error("Faltan archivos del portal: %s", missing_files)
    c.fatal_panel(
        "Faltan archivos del portal",
        "No se encontraron: " + ", ".join(missing_files) + ". Vuelva a copiar la carpeta assets/ desde el notebook "
        "(champion_pipeline.joblib y metadata.json) y recargue la página.",
    )
    st.stop()

c.shell()

pages = [
    st.Page("views/inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("views/analizar.py", title="Analizar cliente", icon=":material/person_search:"),
    st.Page("views/cartera.py", title="Cartera", icon=":material/table_rows:"),
    st.Page("views/campana.py", title="Campaña", icon=":material/campaign:"),
    st.Page("views/calidad.py", title="Calidad del modelo", icon=":material/fact_check:"),
    st.Page("views/sistema.py", title="Sistema", icon=":material/info:"),
]

try:
    st.navigation(pages, position="top").run()
except Exception:  # noqa: BLE001 - los saltos de página y st.stop no son Exception, así que pasan de largo
    log.exception("Error no controlado en una página")
    c.fatal_panel(
        "Algo salió mal en esta página",
        "El problema quedó registrado. Recargue la página; si se repite con el mismo archivo o los mismos datos, "
        "avise al equipo técnico con la hora del error.",
    )
c.footer()
