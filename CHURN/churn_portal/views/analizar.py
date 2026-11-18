"""Analizar cliente: el riesgo, su evolución, la evidencia del historial y el plan de retención que más valor deja."""
import altair as alt
import pandas as pd
import streamlit as st

import components as c
import core
import plans
from core import options, value_label

c.page_header("Analizar un cliente", "Riesgo, evidencia y el plan de retención que más valor deja.")
alert = c.alert_threshold()

# Perfiles de ejemplo. El primero es el "Cliente A de alto riesgo" del reporte del equipo.
BASE = {
    "tenure": 1, "MonthlyCharges": 95.1, "TotalCharges": 95.1,
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check", "PaperlessBilling": "Yes",
    "InternetService": "Fiber optic", "PhoneService": "Yes", "MultipleLines": "No",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
}
PRESETS = {
    "Nuevo cliente de fibra (alto riesgo)": BASE,
    "Cliente típico": {
        **BASE, "tenure": 29, "MonthlyCharges": 70.4, "TotalCharges": 2050.0, "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check", "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    },
    "Cliente fiel con contrato de 2 años (bajo riesgo)": {
        **BASE, "tenure": 60, "MonthlyCharges": 64.8, "TotalCharges": 3900.0, "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)", "PaperlessBilling": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No", "Partner": "Yes", "Dependents": "Yes",
    },
}
KEYS = {col: f"f_{col}" for col in BASE}
CUSTOM = "Personalizado"


def load_preset() -> None:
    chosen = st.session_state["preset"]
    if chosen in PRESETS:  # «Personalizado» conserva los datos que ya hay en el formulario
        for col, value in PRESETS[chosen].items():
            st.session_state[KEYS[col]] = value


if "f_tenure" not in st.session_state:
    for col, value in BASE.items():
        st.session_state[KEYS[col]] = value


def pick(column: str, text: str):
    opts = options(column)
    return st.selectbox(text, opts, format_func=value_label, key=KEYS[column])


with st.container(key="row_top"):
    form_col, main_col = st.columns([0.85, 2.15], gap="medium")

with form_col:
    with st.container(key="panel_form"):
        c.panel_head("Perfil del cliente")
        st.selectbox("Perfil de partida", [*PRESETS, CUSTOM], key="preset", on_change=load_preset)
        tab_a, tab_s, tab_p = st.tabs(["Cuenta", "Servicios", "Perfil"])
        with tab_a:
            tenure = st.slider("Antigüedad (meses)", 0, 72, key=KEYS["tenure"])
            contract = pick("Contract", "Tipo de contrato")
            payment = pick("PaymentMethod", "Método de pago")
            paperless = pick("PaperlessBilling", "Factura electrónica")
            m1, m2 = st.columns(2)
            monthly = m1.number_input("Cargo mensual (USD)", 0.0, 200.0, step=1.0, key=KEYS["MonthlyCharges"])
            total = m2.number_input("Cargo total (USD)", 0.0, 12000.0, step=10.0, key=KEYS["TotalCharges"])
        with tab_s:
            internet = pick("InternetService", "Servicio de internet")
            phone = pick("PhoneService", "Servicio telefónico")
            multiple = pick("MultipleLines", "Líneas múltiples")
            s1, s2 = st.columns(2)
            with s1:
                security = pick("OnlineSecurity", "Seguridad en línea")
                protection = pick("DeviceProtection", "Protección de equipos")
                tv = pick("StreamingTV", "Streaming de TV")
            with s2:
                backup = pick("OnlineBackup", "Respaldo en línea")
                tech = pick("TechSupport", "Soporte técnico")
                movies = pick("StreamingMovies", "Streaming de películas")
        with tab_p:
            gender = pick("gender", "Género")
            senior = st.selectbox("Adulto mayor", [0, 1], format_func=lambda v: value_label(v, "SeniorCitizen"),
                                  key=KEYS["SeniorCitizen"])
            partner = pick("Partner", "Tiene pareja")
            dependents = pick("Dependents", "Tiene dependientes")

row = {
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
    "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
    "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": total,
}
X_one = pd.DataFrame([row])[core.model_features()]
res = core.explain_customer(X_one)
p = res["p"]
level = core.band(p, alert)
cost = c.costs()
pl = plans.retention_plans(
    X_one, service_cost={"TechSupport": cost["support"], "OnlineSecurity": cost["security"]}, contact_cost=cost["contact"])
picks = plans.pick_plans(pl, alert)
names = list(picks)
if st.session_state.get("plan_choice") not in names:
    st.session_state["plan_choice"] = names[0]
plan = picks[st.session_state["plan_choice"]]
curve = plans.lifecycle(X_one, plan)
similar = plans.similar_customers(X_one.iloc[0])
HEADLINE = {"Alto": "Conviene actuar ya", "Medio": "Vale la pena vigilar", "Bajo": "Sin acción urgente"}


def when(column: str) -> str:
    month = plans.months_to_threshold(curve, column, int(tenure), alert)
    return "No en 72 meses" if month is None else ("Ya está bajo el umbral" if month == int(tenure) else f"Mes {month}")


with main_col:
    # --- riesgo y ciclo de vida ---
    with st.container(key="panel_riesgo"):
        c.panel_head("Riesgo y ciclo de vida", f"Umbral de alerta {alert:.0%}")
        left, right = st.columns([1, 1.9], gap="medium")
        with left:
            st.html(c.verdict(p, alert, level, HEADLINE[level]))
            rate = f"{similar['rate']:.0%}" if similar["n"] else "—"
            st.html(c.kv_rows([
                ("Ingreso anual en riesgo", c.usd(monthly * 12 * p)),
                ("Clientes parecidos", c.num(similar["n"])),
                ("Se fueron de verdad", rate),
            ]) + f'<div class="fine">{similar["criterio"].capitalize()}, en el historial de {c.num(len(core.load_reference()))} clientes.</div>')
        with right:
            long = curve.melt("meses", var_name="serie", value_name="riesgo")
            long["serie"] = long["serie"].map({"actual": "Sin cambios", "con_plan": "Con el plan"})
            domain, colors = ["Sin cambios", "Con el plan"], ["#DF1B41", "#0A72EF"]
            color = alt.Color("serie:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient="top", title=None))
            lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
                x=alt.X("meses:Q", axis=alt.Axis(title="Meses de antigüedad", tickCount=7)),
                y=alt.Y("riesgo:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None, tickCount=5)),
                color=color,
                tooltip=[alt.Tooltip("meses:Q", title="Mes"), alt.Tooltip("serie:N", title=""), alt.Tooltip("riesgo:Q", format=".0%", title="Riesgo")],
            )
            threshold = alt.Chart(pd.DataFrame({"y": [alert]})).mark_rule(strokeDash=[4, 4], color="#8898AA").encode(y="y:Q")
            now = alt.Chart(pd.DataFrame({"x": [int(tenure)]})).mark_rule(color="#061B31", strokeWidth=1).encode(x="x:Q")
            dots = alt.Chart(pd.DataFrame({"meses": [int(tenure)] * 2, "riesgo": [p, float(plan["p_plan"])], "serie": domain})).mark_point(
                filled=True, size=110, stroke="#fff", strokeWidth=2).encode(x="meses:Q", y="riesgo:Q", color=color)
            c.show_chart((lines + threshold + now + dots).properties(height=260))
            st.html(c.kv_rows([
                ("Baja del umbral sin cambios", when("actual")),
                ("Baja del umbral con el plan", when("con_plan")),
            ]))

# --- plan de retención ---
with st.container(key="panel_plan"):
    c.panel_head("Plan de retención", f"{len(pl)} planes evaluados · efecto estimado por el modelo, no medido")
    choice = st.segmented_control("Plan", names, key="plan_choice", label_visibility="collapsed") or names[0]
    plan = picks[choice]
    st.html(c.strip([
        ("Riesgo con el plan", f"{plan['p_plan']:.0%}", f"Antes: {p:.0%}", "bajo" if plan["p_plan"] < alert else "medio"),
        ("Inversión esperada", c.usd(plan["costo"]), f"Incluye {c.usd(cost['contact'])} de contacto"),
        ("Ingreso protegido", c.usd(plan["protegido"]), "Próximos 12 meses"),
        ("Valor neto", c.usd(plan["valor"]), "Ingreso protegido menos inversión", "bajo" if plan["valor"] > 0 else "medio"),
    ]))
    st.html('<div style="height:6px"></div>')
    steps_col, chart_col = st.columns([1, 2.1], gap="large")
    with steps_col:
        st.html('<div class="sub" style="margin-top:4px">Qué ofrecer</div>' + c.act_rows(plans.plan_steps(X_one.iloc[0], plan)))
    with chart_col:
        front = []
        top_value = -1e18
        for r in pl.sort_values("p_plan", ascending=False).itertuples():
            if r.valor > top_value + 1e-9:
                front.append({"p_plan": r.p_plan, "valor": r.valor})
                top_value = r.valor
        front = pd.DataFrame(front).sort_values("p_plan")
        scatter = alt.Chart(pl[["p_plan", "valor", "etiqueta"]]).mark_circle(size=42, color="#B8C7DD", opacity=0.9).encode(
            x=alt.X("p_plan:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="%", title="Riesgo con el plan", tickCount=6)),
            y=alt.Y("valor:Q", axis=alt.Axis(format="$,.0f", title="Valor neto", tickCount=5)),
            tooltip=[alt.Tooltip("etiqueta:N", title="Plan"), alt.Tooltip("p_plan:Q", format=".0%", title="Riesgo"),
                     alt.Tooltip("valor:Q", format="$,.0f", title="Valor neto")])
        frontier = alt.Chart(front).mark_line(color="#0A72EF", strokeWidth=2, interpolate="step-after").encode(x="p_plan:Q", y="valor:Q")
        chosen = alt.Chart(pd.DataFrame({"p_plan": [float(plan["p_plan"])], "valor": [float(plan["valor"])]})).mark_point(
            filled=True, size=190, color="#0A72EF", stroke="#fff", strokeWidth=2).encode(x="p_plan:Q", y="valor:Q")
        nothing = alt.Chart(pd.DataFrame({"p_plan": [p], "valor": [0.0]})).mark_point(
            size=110, color="#64748D", strokeWidth=2, shape="diamond").encode(x="p_plan:Q", y="valor:Q")
        c.show_chart((scatter + frontier + nothing + chosen).properties(height=250))
    with st.expander("Supuestos de costo"):
        a, b, d = st.columns(3)
        a.number_input("Soporte técnico (USD/mes)", 0.0, 50.0, plans.SERVICE_COST["TechSupport"], 0.5, key="cost_support")
        b.number_input("Seguridad en línea (USD/mes)", 0.0, 50.0, plans.SERVICE_COST["OnlineSecurity"], 0.5, key="cost_security")
        d.number_input("Contacto (USD por cliente)", 0.0, 100.0, plans.CONTACT_COST, 0.5, key="cost_contact")

# --- qué pesa más ---
with st.container(key="panel_factores"):
    c.panel_head("Qué pesa más", "Frente a un cliente típico, en puntos")
    f = res["factors"].copy()
    f = f.reindex(f["contrib"].abs().sort_values(ascending=False).index).head(6)
    st.html(c.diverging_list([(f"{lab}: {val}", float(cb) * 100) for lab, val, cb in zip(f["label"], f["value"], f["contrib"])]))
