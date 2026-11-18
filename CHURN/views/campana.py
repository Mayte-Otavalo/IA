"""Campaña: dado un presupuesto, a quién contactar y con qué plan para retener el mayor ingreso."""
import dataclasses
import html
import io
import time
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import components as c
import core
import mailer
import plans


@st.cache_data(ttl=120, show_spinner=False)
def credential_status(host: str, port: int, username: str, password: str) -> tuple[bool, str]:
    """Comprueba las credenciales iniciando sesión, sin enviar nada. Se guarda 2 minutos para no consultar en cada interacción."""
    return mailer.check_credentials(mailer.Config(host=host, port=port, username=username, password=password))


mail_cfg = c.email_config()
if mail_cfg.configured:
    creds_ok, creds_detail = credential_status(mail_cfg.host, mail_cfg.port, mail_cfg.username, mail_cfg.password)
else:
    creds_ok, creds_detail = False, ""
real = mail_cfg.configured and creds_ok

c.page_header("Campaña", "A quién contactar y con qué plan para sacar el máximo valor de un presupuesto.", chip=False,
              tag="Agente de campaña conectado" if real else "Agente de campaña sin conexión", tag_ok=real)

SAMPLE = "Muestra de 200 clientes"


@st.cache_data(show_spinner=False)
def sample_features() -> pd.DataFrame:
    return core.prepare_portfolio(pd.read_csv(core.SAMPLE_PATH)).reset_index(drop=True)[core.model_features()]


sources = [SAMPLE]
if "cartera_X" in st.session_state:
    sources.append(f"Cartera cargada ({c.num(len(st.session_state['cartera_X']))})")
source = SAMPLE
if len(sources) > 1:  # el selector solo aparece cuando hay una cartera cargada que elegir
    source = st.segmented_control("Base", sources, default=SAMPLE, key="camp_base", label_visibility="collapsed") or SAMPLE

@st.cache_data(show_spinner=False)
def sample_meta() -> pd.DataFrame:
    return pd.read_csv(core.SAMPLE_PATH).fillna("")[["customerID", "nombre", "correo"]]


DEMO_ID = "CLI-0200"  # el cliente de la muestra que representa a quien presenta la demostración
if source == SAMPLE:
    X = sample_features()
    meta = sample_meta().copy()
else:
    X = st.session_state["cartera_X"]
    meta = st.session_state.get("cartera_meta")
    if meta is None or len(meta) != len(X):
        meta = pd.DataFrame({"customerID": [f"CLI-{i + 1:04d}" for i in range(len(X))], "nombre": "", "correo": ""})
meta = meta.reset_index(drop=True)
if "correo" not in meta:
    meta["correo"] = ""
# Los correos de la muestra son sintéticos (example.com, un dominio reservado que no recibe nada).
# El cliente de demostración usa el nombre y la dirección de los secretos locales.
is_demo = ((meta["customerID"] == DEMO_ID) & (source == SAMPLE)).tolist()
for i, demo in enumerate(is_demo):
    if demo:
        if mail_cfg.demo_name:  # sin secretos, el cliente conserva el nombre y la dirección sintética de la muestra
            meta.loc[i, "nombre"] = mail_cfg.demo_name
        if mail_cfg.demo_email:
            meta.loc[i, "correo"] = mail_cfg.demo_email
names = [str(nombre) if demo else str(cid) for cid, nombre, demo in zip(meta["customerID"], meta["nombre"], is_demo)]
emails = [mailer.normalize(e) for e in meta["correo"]]
# Sin credenciales el portal simula el envío al cliente de demostración: no sale ningún correo.
demo_addresses = tuple(e for e, demo in zip(emails, is_demo) if demo and e)
send_cfg = mail_cfg if mail_cfg.configured else dataclasses.replace(mail_cfg, allowed=demo_addresses)

cost = c.costs()
pp = c.cached_portfolio_plans(X, cost["support"], cost["security"], cost["contact"])
curve_value = plans.campaign_curve(pp, "valor")
curve_risk = plans.campaign_curve(pp, "riesgo")
max_cost = float(curve_value["costo"].iloc[-1])
if max_cost <= 0:
    c.empty_state("Ningún plan deja valor con estos costos", "Baje los costos de la oferta o el contacto para ver una campaña.")
    st.stop()

step = 50 if max_cost > 600 else 10
max_budget = int(np.ceil(max_cost / step) * step)
default_budget = max(step, int(round(max_cost * 0.35 / step) * step))
if st.session_state.get("camp_budget", default_budget) > max_budget:
    st.session_state["camp_budget"] = max_budget

with st.container(key="panel_presupuesto"):
    c.panel_head("Presupuesto", f"Gasto esperado en ofertas y contactos · máximo útil {c.usd(max_cost)}")
    budget = st.slider("Presupuesto (USD)", 0, max_budget, default_budget, step, key="camp_budget", label_visibility="collapsed")

best = plans.campaign_at(curve_value, budget)
naive = plans.campaign_at(curve_risk, budget)
ratio = (best["valor"] / naive["valor"]) if naive["valor"] > 0 else None

st.html('<div style="height:14px"></div>' + c.strip([
    ("Clientes contactados", f"{c.num(best['clientes'])} de {c.num(len(X))}", "Con su plan"),
    ("Inversión esperada", c.usd(best["costo"]), "Ofertas y contactos"),
    ("Valor neto esperado", c.usd(best["valor"]), "Ingreso retenido menos inversión", "bajo"),
    ("Frente a ordenar por riesgo", f"{ratio:.1f}×".replace(".", ",") if ratio else "—", f"{c.usd(naive['valor'])} con el mismo presupuesto"),
]) + '<div style="height:14px"></div>')

chart_col, list_col = st.columns([1.25, 1], gap="medium")
with chart_col:
    with st.container(key="panel_curva"):
        c.panel_head("Valor según el presupuesto", "Estimado por el modelo, no medido")
        data = pd.concat([
            curve_value.assign(serie="Por valor esperado"),
            curve_risk.assign(serie="Por riesgo (mismo plan)"),
        ])
        color = alt.Color("serie:N", scale=alt.Scale(domain=["Por valor esperado", "Por riesgo (mismo plan)"], range=["#0A72EF", "#8898AA"]),
                          legend=alt.Legend(orient="top", title=None))
        lines = alt.Chart(data).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
            x=alt.X("costo:Q", axis=alt.Axis(format="$,.0f", title="Presupuesto", tickCount=6), scale=alt.Scale(domain=[0, max_cost])),
            y=alt.Y("valor:Q", axis=alt.Axis(format="$,.0f", title=None, tickCount=5)),
            color=color,
            tooltip=[alt.Tooltip("serie:N", title=""), alt.Tooltip("clientes:Q", title="Clientes"),
                     alt.Tooltip("costo:Q", format="$,.0f", title="Presupuesto"), alt.Tooltip("valor:Q", format="$,.0f", title="Valor")])
        rule = alt.Chart(pd.DataFrame({"x": [budget]})).mark_rule(color="#061B31", strokeDash=[4, 4]).encode(x="x:Q")
        dots = alt.Chart(pd.DataFrame({"costo": [best["costo"], naive["costo"]], "valor": [best["valor"], naive["valor"]],
                                       "serie": ["Por valor esperado", "Por riesgo (mismo plan)"]})).mark_point(
            filled=True, size=110, stroke="#fff", strokeWidth=2).encode(x="costo:Q", y="valor:Q", color=color)
        c.show_chart((lines + rule + dots).properties(height=300))
with list_col:
    with st.container(key="panel_planes"):
        c.panel_head("Planes más repetidos", f"Entre los {c.num(best['clientes'])} contactados")
        chosen = pp[pp["fila"].isin(best["filas"])]
        counts = chosen["plan"].value_counts().head(5)
        st.html(c.bar_list([(plan_name, float(n)) for plan_name, n in counts.items()], fmt=lambda v: f"{v:.0f}"))
        st.html(c.kv_rows([
            ("Riesgo medio antes", f"{chosen['p0'].mean():.0%}" if len(chosen) else "—"),
            ("Riesgo medio con el plan", f"{chosen['p_plan'].mean():.0%}" if len(chosen) else "—"),
        ]))

with st.container(key="panel_lista"):
    c.panel_head("Lista de campaña", f"{c.num(best['clientes'])} clientes, en el orden en que conviene contactarlos")
    order = {fila: k for k, fila in enumerate(best["filas"])}
    sel = pp[pp["fila"].isin(best["filas"])].assign(_k=lambda d: d["fila"].map(order)).sort_values("_k")
    table = pd.DataFrame({
        "Cliente": [names[int(f)] for f in sel["fila"]],
        "Correo": [emails[int(f)] or "—" for f in sel["fila"]],
        "Riesgo": sel["p0"].to_numpy() * 100,
        "Con plan": sel["p_plan"].to_numpy() * 100,
        "Plan": sel["plan"].to_numpy(),
        "Inversión": sel["costo"].to_numpy(),
        "Valor": sel["valor"].to_numpy(),
    })
    st.dataframe(
        table, hide_index=True, height=360,
        column_config={
            "Cliente": st.column_config.TextColumn("Cliente", width="small"),
            "Correo": st.column_config.TextColumn("Correo", width="medium"),
            "Riesgo": st.column_config.NumberColumn("Riesgo", format="%.0f%%", width="small"),
            "Con plan": st.column_config.NumberColumn("Con plan", format="%.0f%%", width="small"),
            "Plan": st.column_config.TextColumn("Plan", width="large"),
            "Inversión": st.column_config.NumberColumn("Inversión", format="$%.0f", width="small"),
            "Valor": st.column_config.NumberColumn("Valor", format="$%.0f", width="small"),
        },
    )
    out = io.StringIO()
    table.to_csv(out, index=False)
    st.download_button("Descargar lista de campaña", data=out.getvalue().encode("utf-8-sig"),
                       file_name="campana.csv", mime="text/csv", icon=":material/download:")
    with st.expander("Supuestos de costo"):
        a, b, d = st.columns(3)
        a.number_input("Soporte técnico (USD/mes)", 0.0, 50.0, plans.SERVICE_COST["TechSupport"], 0.5, key="cost_support")
        b.number_input("Seguridad en línea (USD/mes)", 0.0, 50.0, plans.SERVICE_COST["OnlineSecurity"], 0.5, key="cost_security")
        d.number_input("Contacto (USD por cliente)", 0.0, 100.0, plans.CONTACT_COST, 0.5, key="cost_contact")


# ---------- agente de correo ----------

STEPS = ["Eligiendo a los clientes de la campaña", "Redactando el plan de cada uno", "Enviando por correo"]


@st.dialog("Lanzar campaña", width="medium")
def launch_dialog(messages: list, cfg: mailer.Config, summary: dict) -> None:
    results = st.session_state.get("mail_results")
    if st.session_state.get("mail_done") and results:  # el envío terminó: se muestra el resultado
        st.html(c.send_result(results, len(messages), st.session_state.get("mail_when", "")))
        if st.button("Cerrar", type="primary", width="stretch"):
            st.session_state["mail_done"] = False
            st.rerun()
        return
    view = st.empty()
    with view.container():
        st.html(c.fleet_card(len(messages)))
        st.html(c.kv_rows([
            ("Clientes", c.num(len(messages))),
            ("Inversión esperada", c.usd(summary["costo"])),
            ("Valor neto esperado", c.usd(summary["valor"])),
        ]))
        go, cancel = st.columns(2)
        launch = go.button("Enviar ahora", type="primary", width="stretch", icon=":material/send:")
        if cancel.button("Cancelar", width="stretch"):
            st.rerun()
    if launch:
        view.empty()
        slot = st.empty()
        for done in range(2):
            slot.html(c.steps_card(done, STEPS))
            time.sleep(0.7)
        slot.html(c.steps_card(2, STEPS))
        st.session_state["mail_results"] = mailer.send_campaign(messages, cfg)
        slot.html(c.steps_card(3, STEPS))
        time.sleep(0.6)
        st.session_state["mail_done"], st.session_state["mail_when"] = True, datetime.now().strftime("%H:%M")
        st.rerun(scope="fragment")


rows_by_fila = pp.set_index("fila")
messages = []
for fila in best["filas"]:
    plan_row, customer = rows_by_fila.loc[fila], X.iloc[fila]
    subject, text, body = mailer.compose(str(meta["nombre"].iloc[fila]), int(customer["tenure"]), plans.offer_lines(customer, plan_row),
                                         reply_to=mail_cfg.username)
    messages.append(mailer.Message(names[fila], emails[fila], subject, text, body))
authorized = [m for m in messages if m.email in send_cfg.allowed]

st.html('<div style="height:14px"></div>')
with st.container(key="panel_envio"):
    chosen_plans = pp[pp["fila"].isin(best["filas"])]
    st.html(c.agent_panel(len(messages), best["valor"], float(chosen_plans["p0"].mean()) if len(chosen_plans) else 0.0,
                          float(chosen_plans["p_plan"].mean()) if len(chosen_plans) else 0.0, real))
    blocked = (mail_cfg.configured and not creds_ok) or not authorized
    with st.container(key="send_launch"):
        if st.button("Lanzar campaña por correo", icon=":material/send:", disabled=blocked, width="stretch"):
            st.session_state["mail_done"] = False
            launch_dialog(messages, send_cfg, best)
    if mail_cfg.configured and not creds_ok:
        st.html(f'<div class="fine fine-error">{html.escape(creds_detail)}</div>')
    elif not send_cfg.allowed:
        st.html('<div class="fine">No hay direcciones autorizadas: configure [email] en .streamlit/secrets.toml.</div>')
    elif not authorized:
        st.html('<div class="fine">Ningún cliente de esta campaña tiene una dirección autorizada.</div>')
    elif st.session_state.get("mail_when"):
        st.html(f'<div class="fine">Último envío a las {html.escape(st.session_state["mail_when"])}.</div>')
