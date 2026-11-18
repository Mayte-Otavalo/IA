"""Calidad del modelo: métricas en el conjunto de prueba, matriz de confusión y umbral de alerta."""
import altair as alt
import pandas as pd
import streamlit as st

import components as c
import core


@st.cache_data(show_spinner=False)
def curve_data():
    _, y, p = core.holdout()
    return core.threshold_curve(y, p)


@st.cache_data(show_spinner="Calculando factores globales…")
def drivers():
    X, _, _ = core.holdout()
    return core.global_drivers(X.iloc[:200])


def _sync_alert() -> None:
    st.session_state["alert"] = st.session_state["alert_slider"] / 100  # el control va en %, el portal usa fracciones


c.page_header("Calidad del modelo", "Desempeño sobre los 1.409 clientes de prueba.", chip=False)

if "alert_slider" not in st.session_state:  # el valor inicial va en el estado, no en el argumento: si no, el control se reinicia al moverlo
    st.session_state["alert_slider"] = int(round(c.alert_threshold() * 100))
with st.container(key="panel_umbral"):
    c.panel_head("Umbral de alerta", "Se aplica en todo el portal")
    st.slider(
        "Umbral de alerta", 5, 75, format="%d%%",
        key="alert_slider", on_change=_sync_alert, label_visibility="collapsed",
    )
alert = c.alert_threshold()

X, y, p = core.holdout()
m = core.metrics_at(y, p, alert)

st.html('<div style="height:14px"></div>' + c.strip([
    ("Detecta", f"{m['recall']:.0%}", "De quienes se van", "bajo"),
    ("Acierta", f"{m['precision']:.0%}", "De las alertas son reales", "medio"),
    ("Marca", f"{m['flagged_share']:.0%}", "De los clientes"),
    ("AUC", c.num(m["auc"], 2), "0,5 es azar, 1 es perfecto"),
    ("Brier", c.num(m["brier"], 3), "Menor es mejor"),
]) + '<div style="height:14px"></div>')

left, right = st.columns([1, 1.25], gap="medium")
with left:
    with st.container(key="panel_matriz"):
        c.panel_head("Resultado por tipo de cliente", f"{c.num(len(y))} clientes")
        st.html(
            '<table class="cm"><tr><th></th><th>No marcado</th><th>Marcado en riesgo</th></tr>'
            '<tr><th>Se quedó</th>'
            f'<td class="cell ok"><small>Correcto</small>{c.num(m["tn"])}</td>'
            f'<td class="cell bad"><small>Falsa alerta</small>{c.num(m["fp"])}</td></tr>'
            '<tr><th>Se fue</th>'
            f'<td class="cell bad"><small>No detectado</small>{c.num(m["fn"])}</td>'
            f'<td class="cell ok"><small>Detectado</small>{c.num(m["tp"])}</td></tr>'
            "</table>"
        )
with right:
    with st.container(key="panel_compromiso"):
        c.panel_head("Compromiso según el umbral")
        curve = curve_data().melt("umbral", var_name="Medida", value_name="Valor")
        lines = alt.Chart(curve).mark_line(strokeWidth=2.5).encode(
            x=alt.X("umbral:Q", axis=alt.Axis(format="%", title=None, tickCount=8)),
            y=alt.Y("Valor:Q", axis=alt.Axis(format="%", title=None, tickCount=5)),
            color=alt.Color("Medida:N", scale=alt.Scale(
                domain=["Detectados", "Acierto al marcar", "Clientes marcados"],
                range=["#108C3D", "#0A72EF", "#8898AA"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["umbral:Q", "Medida:N", alt.Tooltip("Valor:Q", format=".1%")],
        )
        rule = alt.Chart(pd.DataFrame({"umbral": [alert]})).mark_rule(color="#061B31", strokeDash=[4, 4], strokeWidth=1.5).encode(x="umbral:Q")
        c.show_chart((lines + rule).properties(height=250))

st.html('<div style="height:14px"></div>')
with st.container(key="panel_factores"):
    c.panel_head("Qué más mueve la estimación", "Impacto medio en puntos, sobre 200 clientes")
    drv = drivers().head(8)
    st.html(c.bar_list([(r.label, r.impact * 100) for r in drv.itertuples()],
                       fmt=lambda v: f"{v:.1f}".replace(".", ",")))
