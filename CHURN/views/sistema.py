"""Información del sistema: modelo cargado, trazabilidad frente al reporte y límites de uso."""
import streamlit as st

import components as c
import core

card = core.model_card()
meta = core.load_metadata()
X, y, p = core.holdout()
m = core.metrics_at(y, p, c.alert_threshold())

c.page_header("Sistema", "Modelo cargado, datos y trazabilidad frente al reporte.", chip=False)

CLASS_LABELS = {"RandomForestClassifier": "Bosque aleatorio", "CatBoostClassifier": "CatBoost"}
clf_label = CLASS_LABELS.get(card["classifier"], card["classifier"])
unit = "árboles" if card["classifier"] == "RandomForestClassifier" else "iteraciones"
size_kb = card["size_kb"]
size_txt = f"{c.num(size_kb, 1)} KB" if size_kb < 1024 else f"{c.num(size_kb / 1024, 1)} MB"

st.html(c.strip([
    ("Modelo", clf_label, f"{card['trees']} {unit}, profundidad {card['depth']}"),
    ("Variables", str(card["features"]), "Datos del cliente"),
    ("Historial de referencia", c.num(len(core.load_reference())), "Clientes con su resultado"),
    ("Tamaño", size_txt, "champion_pipeline.joblib"),
]) + '<div style="height:14px"></div>')


def status(ok: bool | None) -> str:
    if ok is None:
        return '<span class="badge" style="background:#EEF1F6;color:#667085">No comparable</span>'
    return '<span class="badge badge-bajo">Coincide</span>' if ok else '<span class="badge badge-medio">No coincide</span>'


rows = [
    ("Tipo de modelo", f"{clf_label} ({card['classifier']})", card["reported_champion"], clf_label == card["reported_champion"]),
    ("AUC (prueba)", c.num(m["auc"], 4), c.num(card["reported_auc"], 4) if card["reported_auc"] is not None else "—",
     None if card["reported_auc"] is None else abs(m["auc"] - card["reported_auc"]) < 0.002),
    ("Brier (prueba)", c.num(m["brier"], 4), c.num(card["reported_brier"], 4) if card["reported_brier"] is not None else "—",
     None if card["reported_brier"] is None else abs(m["brier"] - card["reported_brier"]) < 0.002),
    ("Recall", f"{c.num(m['recall'], 4)} (umbral {c.alert_threshold():.0%})",
     f"{c.num(card['reported_recall'], 4)} (umbral no reportado)" if card["reported_recall"] is not None else "—", None),
]
body = "".join(f"<tr><td>{n}</td><td>{a}</td><td>{b}</td><td>{status(ok)}</td></tr>" for n, a, b, ok in rows)

with st.container(key="panel_traza"):
    c.panel_head("Trazabilidad frente al reporte", "Cada cifra se recalcula con el modelo cargado")
    st.html('<table class="tbl"><tr><th>Medida</th><th>Modelo cargado</th><th>Reporte</th><th>Estado</th></tr>' + body + "</table>")

st.html('<div style="height:14px"></div>')
left, right = st.columns(2, gap="medium")
with left:
    with st.container(key="panel_datos"):
        c.panel_head("Datos y método")
        st.html(c.kv_rows([
            ("Historial", "IBM Telco Customer Churn"),
            ("Partición de prueba", f"20 %, estratificada, semilla 42 ({c.num(len(y))} clientes)"),
            ("Umbral de alerta", f"{c.alert_threshold():.0%}"),
        ]))
with right:
    with st.container(key="panel_limites"):
        c.panel_head("Límites de uso")
        st.html(c.kv_rows([
            ("Alcance", "Estima asociaciones, no causas"),
            ("Validez", "Entrenado con un historial de referencia"),
            ("Probabilidades", "No calibradas"),
            ("Planes y campañas", "Efecto estimado, no medido: valídelo con un piloto"),
            ("Modelo", "El portal no lo reentrena"),
        ]))
