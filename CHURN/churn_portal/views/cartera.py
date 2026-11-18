"""Cartera de clientes: puntúa una lista, la ordena por riesgo y muestra el detalle del cliente elegido."""
import io

import pandas as pd
import streamlit as st

import components as c
import core

c.page_header("Cartera de clientes", "Clientes ordenados por riesgo, con la palanca que más lo reduce.")
alert = c.alert_threshold()


@st.cache_data(show_spinner="Calculando el riesgo de cada cliente…")
def score(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return core.score_portfolio(df, threshold)


source = st.segmented_control(
    "Datos", ["Muestra de 200 clientes", "Mi archivo CSV"], default="Muestra de 200 clientes",
    key="cartera_source", label_visibility="collapsed",
) or "Muestra de 200 clientes"

if source == "Mi archivo CSV":
    up_col, tpl_col = st.columns([3, 1], gap="large", vertical_alignment="bottom")
    with up_col:
        upload = st.file_uploader(
            "Archivo CSV con una fila por cliente", type=["csv"],
            help=f"Hasta {core.MAX_FILE_MB} MB y {core.MAX_ROWS:,} clientes. Acepta coma, punto y coma o tabulador.".replace(",", "."),
        )
    with tpl_col:
        st.download_button("Descargar plantilla", data=core.portfolio_template().encode("utf-8-sig"),
                           file_name="plantilla_cartera.csv", mime="text/csv", icon=":material/description:",
                           width="stretch")
    st.html('<div class="privacy">Los datos se procesan en memoria durante su sesión y no se guardan.</div>')
    with st.expander("Columnas necesarias"):
        st.write(", ".join(core.label(col) if col in core.LABELS else col for col in core.model_features()))
        st.caption("Si incluye «customerID», se usa para nombrar a cada cliente. Las columnas adicionales se ignoran.")
    if upload is None:
        c.empty_state("Todavía no hay una cartera cargada",
                      "Suba un CSV o descargue la plantilla para ver el formato. También puede usar la muestra de 200 clientes.")
        st.stop()
    try:
        parsed = core.read_portfolio_csv(upload.getvalue())
    except ValueError as exc:
        c.fatal_panel("No se pudo leer el archivo", str(exc))
        st.stop()
    raw, errors, notes = core.validate_portfolio(parsed)
    for message in errors:
        st.error(message, icon=":material/error:")
    if raw is None:
        st.stop()
    for message in notes:
        st.warning(message, icon=":material/warning:")
else:
    raw = pd.read_csv(core.SAMPLE_PATH)

result = score(raw, alert)
prepared = core.prepare_portfolio(raw).reset_index(drop=True)
st.session_state["cartera_X"] = prepared[core.model_features()].copy()  # lo usa Campaña
lower = {str(col).lower(): col for col in prepared.columns}
mail_col = next((lower[k] for k in ("correo", "email", "e-mail", "mail") if k in lower), None)
name_col = next((lower[k] for k in ("nombre", "name") if k in lower), None)
st.session_state["cartera_meta"] = pd.DataFrame({
    "customerID": prepared["customerID"].astype(str) if "customerID" in prepared.columns
    else pd.Series([f"CLI-{i + 1:04d}" for i in range(len(prepared))]),
    "nombre": prepared[name_col].fillna("").astype(str) if name_col else "",
    "correo": prepared[mail_col].fillna("").astype(str) if mail_col else "",
})

counts = result["Nivel"].value_counts()
n_alto, n_medio, n_bajo = (int(counts.get(k, 0)) for k in ["Alto", "Medio", "Bajo"])
at_risk_month = float((result["_p"] * result["_monthly"]).sum())

st.html(c.strip([
    ("Clientes", c.num(len(result)), "En la lista"),
    ("Riesgo alto", c.num(n_alto), f"{n_alto / len(result):.0%} de la lista", "alto"),
    ("Ingreso mensual en riesgo", c.usd(at_risk_month), "Probabilidad × cargo", "medio"),
    ("Probabilidad media", f"{result['Probabilidad'].mean():.0%}", "Promedio de la lista"),
]))
st.html(f'<div style="height:14px"></div>{c.stacked_bar([("Alto", n_alto, "#DF1B41"), ("Medio", n_medio, "#E39B00"), ("Bajo", n_bajo, "#15BE53")])}'
        '<div style="height:6px"></div>')

table_col, detail_col = st.columns([1.9, 1], gap="medium")

with table_col:
    with st.container(key="panel_tabla"):
        nivel = st.segmented_control("Nivel", ["Todos", "Alto", "Medio", "Bajo"], default="Todos",
                                     key="cartera_nivel", label_visibility="collapsed") or "Todos"
        view = result if nivel == "Todos" else result[result["Nivel"] == nivel]
        view = view.reset_index(drop=True)
        shown = pd.DataFrame({
            "Cliente": view["Cliente"],
            "Riesgo": view["Probabilidad"] * 100,
            "Palanca principal": view["Palanca principal"],
            "Baja (pts)": view["Reducción"] * 100,
        })
        if shown.empty:
            c.empty_state("Sin clientes en este nivel", "Cambie el filtro para ver el resto de la lista.")
            st.stop()
        event = st.dataframe(
            shown, hide_index=True, height=440, key=f"cartera_tabla_{nivel}_{source}",
            on_select="rerun", selection_mode="single-row",
            column_config={
                "Cliente": st.column_config.TextColumn("Cliente", width="small"),
                "Riesgo": st.column_config.ProgressColumn("Riesgo", format="%.0f%%", min_value=0, max_value=100, width="small"),
                "Palanca principal": st.column_config.TextColumn("Palanca principal", width="large"),
                "Baja (pts)": st.column_config.NumberColumn("Baja (pts)", format="−%.0f", width="small"),
            },
        )
        export = result[["Cliente", "Probabilidad", "Nivel", "Palanca principal"]].copy()
        export["Probabilidad"] = (export["Probabilidad"] * 100).round(1).astype(str) + " %"
        buffer = io.StringIO()
        export.to_csv(buffer, index=False)
        st.download_button("Descargar lista de contacto", data=buffer.getvalue().encode("utf-8-sig"),
                           file_name="riesgo_clientes.csv", mime="text/csv", icon=":material/download:")

picked = event.selection.rows[0] if event.selection.rows else 0
sel = view.iloc[min(picked, len(view) - 1)]
X_one = prepared.iloc[[int(sel["_row"])]][core.model_features()].reset_index(drop=True)
res = core.explain_customer(X_one)
level = core.band(float(sel["Probabilidad"]), alert)
levers = core.what_if(X_one)
cust = X_one.iloc[0]
f = res["factors"].copy()
f = f.reindex(f["contrib"].abs().sort_values(ascending=False).index).head(5)

with detail_col:
    with st.container(key="panel_detalle"):
        months = int(cust["tenure"])
        c.panel_head(str(sel["Cliente"]), f"{months} {'mes' if months == 1 else 'meses'} · {c.usd(float(cust['MonthlyCharges']))} al mes")
        st.html(c.verdict(float(sel["Probabilidad"]), alert, level, {"Alto": "Conviene actuar ya",
                "Medio": "Vale la pena vigilar", "Bajo": "Sin acción urgente"}[level]))
        st.html('<div class="sub">Qué pesa más</div>' + c.diverging_list(
            [(f"{lab}: {val}", float(cb) * 100) for lab, val, cb in zip(f["label"], f["value"], f["contrib"])]))
        if not levers.empty:
            st.html('<div class="sub">Cómo reducirlo</div>'
                    + c.lever_rows(levers, core.combined_what_if(X_one, levers), float(sel["Probabilidad"])))
        if st.button("Abrir en Analizar", icon=":material/open_in_new:", key="open_in_analizar"):
            for col in core.model_features():
                value = cust[col]
                st.session_state[f"f_{col}"] = value.item() if hasattr(value, "item") else value
            st.session_state["preset"] = "Personalizado"
            st.switch_page(c.P_ANALIZAR)
