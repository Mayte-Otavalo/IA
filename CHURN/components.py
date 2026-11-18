"""Piezas de interfaz reutilizables (HTML y SVG sencillos). Sin lógica de negocio."""
from __future__ import annotations

import html
from urllib.parse import quote

import altair as alt
import streamlit as st

import core
import mailer
import plans

P_INICIO = "views/inicio.py"
P_ANALIZAR = "views/analizar.py"
P_CARTERA = "views/cartera.py"
P_CAMPANA = "views/campana.py"
P_CALIDAD = "views/calidad.py"
P_SISTEMA = "views/sistema.py"

LEVEL_COLOR = {"Alto": "#DF1B41", "Medio": "#B25E09", "Bajo": "#108C3D"}
UP, DOWN = "#DF1B41", "#108C3D"  # sube el riesgo / baja el riesgo
FONT = "'RetenSans', 'Source Sans 3', system-ui, -apple-system, 'Segoe UI', sans-serif"

_ICON_PATHS = {
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
}


def svg_data(svg: str, cls: str = "", width: int | None = None) -> str:
    """SVG embebido como imagen. st.html elimina <svg> en línea, pero conserva <img> con datos."""
    w = f' width="{width}"' if width else ""
    return f'<img class="{cls}" alt=""{w} src="data:image/svg+xml;charset=utf-8,{quote(svg)}">'


def icon(name: str, size: int = 22, color: str = "#0B1B33") -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICON_PATHS[name]}</svg>"
    )
    return svg_data(svg, cls="ico", width=size)


def logo(size: int = 34) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 40 40">'
        '<rect width="40" height="40" rx="9" fill="#0D253D"/>'
        '<path d="M8 27l8-9 6 6 10-12" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="32" cy="12" r="3.2" fill="#3B9BFF"/></svg>'
    )
    return svg_data(svg, width=size)


def num(x: float, dec: int = 0) -> str:
    """Número con formato hispano: 1.409 y 0,84."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def usd(x: float) -> str:
    return f"USD {num(x)}"


def costs() -> dict:
    """Supuestos de costo compartidos entre pantallas (se editan en Analizar y en Campaña)."""
    ss = st.session_state
    return {
        "support": float(ss.get("cost_support", plans.SERVICE_COST["TechSupport"])),
        "security": float(ss.get("cost_security", plans.SERVICE_COST["OnlineSecurity"])),
        "contact": float(ss.get("cost_contact", plans.CONTACT_COST)),
    }


@st.cache_data(show_spinner="Buscando el mejor plan para cada cliente…")
def cached_portfolio_plans(X, support: float, security: float, contact: float):
    """Mejor plan de cada cliente de X con los costos dados. Se guarda en caché porque es lo más pesado de la campaña."""
    return plans.portfolio_plans(
        X, service_cost={"TechSupport": support, "OnlineSecurity": security}, contact_cost=contact)


def email_config() -> mailer.Config:
    """Configuración del agente de correo desde los secretos locales. Sin secretos: nadie autorizado."""
    try:
        section = dict(st.secrets.get("email", {}))
    except Exception:  # noqa: BLE001 - sin archivo de secretos
        section = {}
    return mailer.load_config(section)


def delivery_animation(kind: str) -> str:
    """Un avión de papel cruza de RetenTel al teléfono y se dibuja una marca de verificación. kind: ok, demo o error."""
    color = {"ok": "#108C3D", "demo": "#0A72EF", "error": "#DF1B41"}[kind]
    mark = (
        '<path d="M309 48l8 8 14-16" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" '
        'stroke-dasharray="40" stroke-dashoffset="40"><animate attributeName="stroke-dashoffset" from="40" to="0" begin="1.5s" dur="0.4s" fill="freeze"/></path>'
        if kind != "error" else
        '<path d="M310 40l16 16M326 40l-16 16" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round"/>'
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="96" viewBox="0 0 360 96">'
        '<rect x="14" y="30" width="36" height="36" rx="9" fill="#0D253D"/>'
        '<path d="M22 54l7-8 5 5 9-10" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M64 48H292" stroke="#C9D5E3" stroke-width="2" stroke-dasharray="3 7" stroke-linecap="round"/>'
        f'<g><path d="M-12 -9L14 0L-12 9L-6 0Z" fill="{color}"/>'
        '<animateMotion path="M72 48H282" dur="1.3s" begin="0.15s" fill="freeze" calcMode="spline" keyTimes="0;1" keySplines="0.45 0 0.2 1"/>'
        '<animate attributeName="opacity" from="1" to="0" begin="1.35s" dur="0.15s" fill="freeze"/></g>'
        f'<circle cx="318" cy="48" r="23" fill="{color}" opacity="0">'
        '<animate attributeName="opacity" from="0" to="1" begin="1.2s" dur="0.2s" fill="freeze"/>'
        '<animate attributeName="r" values="16;26;23" begin="1.2s" dur="0.4s" fill="freeze"/></circle>'
        f"{mark}</svg>"
    )
    return svg_data(svg, cls="delivery", width=360)


ENVELOPE = ('<svg viewBox="0 0 32 24" width="30" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
            '<rect x="2" y="2" width="28" height="20" rx="3"/><path d="M3 5l13 9 13-9"/></svg>')


def agent_panel(chosen: int, value: float, risk_before: float, risk_after: float, live: bool) -> str:
    """Cabecera del agente: de la campaña elegida a la reducción de riesgo que se espera de ella."""
    wire = '<div class="wire"><i></i><i></i><i></i></div>'
    status = "Agente de campaña en vivo" if live else "Agente de campaña · sin conexión"
    drop = max(0.0, risk_before - risk_after) * 100
    return (
        f'<div class="agent"><div class="agent-live{" real" if live else ""}"><i></i>{status}</div>'
        '<h3>Un agente que le escribe a cada cliente su propio plan</h3>'
        '<div class="pipe">'
        f'<div class="node"><b>{num(chosen)}</b><span>clientes elegidos</span><small>Valor neto esperado {usd(value)}</small></div>{wire}'
        f'<div class="node"><b>{num(chosen)}</b><span>mensajes redactados</span><small>Cada uno con el plan exacto del cliente</small></div>{wire}'
        f'<div class="node on"><b>−{drop:.0f} pts</b><span>riesgo medio esperado</span><small>De {risk_before:.0%} a {risk_after:.0%} con el plan</small></div>'
        "</div></div>"
    )


def fleet_card(count: int) -> str:
    """De RetenTel a los clientes de la campaña: un sobre viaja hacia un grupo de avatares."""
    colors = ["#0A72EF", "#3B8DF0", "#5BA7FF"]
    stack = "".join(f'<span class="r-avatar s" style="background:{c}">{l}</span>' for c, l in zip(colors, "ABC"))
    return (
        '<div class="route">'
        '<div class="r-node"><span class="r-logo">R</span><b>RetenTel</b><small>Agente de retención</small></div>'
        f'<div class="r-wire"><span>{ENVELOPE}</span></div>'
        f'<div class="r-node"><span class="r-stack">{stack}</span><b>{num(count)} clientes</b><small>Cada uno con su plan</small></div>'
        "</div>"
    )


def steps_card(done: int, labels: list[str]) -> str:
    """Pasos del agente mientras trabaja: los hechos llevan marca, el actual gira."""
    items = []
    for i, label in enumerate(labels):
        kind = "done" if i < done else "now" if i == done else ""
        items.append(f'<li class="{kind}"><i></i>{html.escape(label)}</li>')
    return f'<ul class="steps">{"".join(items)}</ul>'


def send_result(results, total: int, when: str) -> str:
    """Resultado de un envío para la ventana de lanzamiento: la animación y el resumen de la campaña, sin destacar a ningún cliente."""
    done = [r for r in results if r.estado != "Omitido"]
    if not done:
        return ('<div class="done"><div class="done-text"><b>No se envió ningún mensaje</b>'
                '<span>Ningún cliente de la campaña tiene una dirección autorizada.</span></div></div>')
    first = done[0]
    kind = {"Enviado": "ok", "Simulado": "demo", "Error": "error"}[first.estado]
    title = {"Enviado": "Campaña enviada", "Simulado": "Simulación completada", "Error": "No se pudo enviar"}[first.estado]
    rows = kv_rows([("Clientes en la campaña", num(total)), ("Hora", when), ("Estado", first.detalle)])
    note = '<div class="fine">Envío limitado a la dirección autorizada.</div>' if len(done) < len(results) else ""
    return (f'<div class="done"><div class="done-art">{delivery_animation(kind)}</div>'
            f'<div class="done-text"><b>{title}</b>{rows}{note}</div></div>')


def alert_threshold() -> float:
    """Umbral de alerta compartido entre páginas. Se cambia en Calidad del modelo."""
    return float(st.session_state.get("alert", core.ALERT_DEFAULT))


# ---------- estructura general ----------

def shell() -> None:
    """Marca y estado del modelo en la barra superior. Se llama una vez desde app.py."""
    auc = core.load_metadata().get("champion_metrics", {}).get("ROC_AUC")
    auc_txt = f" · AUC {num(auc, 3)}" if auc else ""
    name = {"CatBoostClassifier": "CatBoost", "RandomForestClassifier": "Bosque aleatorio"}.get(
        core.model_card()["classifier"], core.model_card()["classifier"])
    st.html(
        f'<div class="brand">{logo(30)}<div>RetenTel AI</div></div>'
        f'<div class="model-pill"><i></i>{html.escape(name)}{auc_txt}</div>'
    )


def footer() -> None:
    st.html(f'<div class="foot">RetenTel AI · {html.escape(core.model_card()["classifier"].replace("Classifier", ""))} · '
            "IBM Telco Customer Churn, 7.043 clientes</div>")


def page_header(title: str, lead: str = "", crumb: str = "", chip: bool = True, tag: str = "", tag_ok: bool = True) -> None:
    """Título de la pantalla, una línea que dice qué se ve y, a la derecha, el umbral de alerta vigente o una etiqueta de estado."""
    if tag:
        chip_html = f'<div class="chip tag{"" if tag_ok else " off"}"><i></i><b>{html.escape(tag)}</b></div>'
    else:
        chip_html = f'<div class="chip"><i></i>Umbral de alerta <b>{alert_threshold():.0%}</b></div>' if chip else ""
    lead_html = f"<p>{html.escape(lead)}</p>" if lead else ""
    st.html(f'<div class="page-head"><div><h1>{html.escape(title)}</h1>{lead_html}</div>{chip_html}</div>')


def panel_head(title: str, meta: str = "") -> None:
    """Cabecera de un panel: título a la izquierda y un dato breve a la derecha."""
    meta_html = f"<span>{meta}</span>" if meta else ""
    st.html(f'<div class="ph"><b>{html.escape(title)}</b>{meta_html}</div>')


def section(title: str, hint: str = "") -> None:
    panel_head(title, html.escape(hint) if hint else "")


def fatal_panel(title: str, text: str) -> None:
    """Error que impide usar la página, con una acción clara. Nunca muestra trazas al usuario."""
    st.html('<div class="fatal" role="alert"><h2>' + html.escape(title) + "</h2><p>" + html.escape(text) + "</p></div>")


def empty_state(title: str, text: str) -> None:
    """Estado vacío: dice qué falta y qué hacer, en lugar de dejar un hueco."""
    st.html('<div class="empty"><b>' + html.escape(title) + "</b><span>" + html.escape(text) + "</span></div>")


def notice(text: str, kind: str = "info") -> None:
    st.html(f'<div class="notice notice-{kind}">{text}</div>')


# ---------- portada ----------

def preview_card(rows, alert: float, total: int) -> str:
    """Vista del producto con datos reales: los clientes de mayor riesgo de la muestra y su palanca principal."""
    body = "".join(
        f'<div class="pv-row"><span class="pv-id">{html.escape(str(r.Cliente))}</span>'
        f'<span class="pv-lever">{html.escape(str(r.palanca))}</span>'
        f'<span class="pill pill-alto">{r.Probabilidad:.0%}</span></div>'
        for r in rows.itertuples()
    )
    return (
        '<div class="preview"><div class="pv-head"><b>Clientes con mayor riesgo</b>'
        f'<span>Muestra de {num(total)} · umbral {alert:.0%}</span></div>{body}'
        '<div class="pv-foot">Cada fila muestra la acción que más reduce el riesgo de ese cliente.</div></div>'
    )


def tile(key: str, title: str, text: str, figure: str, page: str, cta: str) -> None:
    """Destino principal: toda la tarjeta es clicable y muestra una cifra real."""
    with st.container(key=f"tile_{key}", border=True):
        st.html(f'<div class="tile-body"><h3>{html.escape(title)}</h3><p>{html.escape(text)}</p>{figure}</div>')
        st.page_link(page, label=cta)


# ---------- datos: métricas, barras y filas ----------

def strip(items: list[tuple]) -> str:
    """Franja de métricas en un solo panel. items = [(etiqueta, cifra, nota, tono)]; el tono pone un punto de estado."""
    cells = ""
    for it in items:
        label, value, note = it[0], it[1], (it[2] if len(it) > 2 else "")
        tone = it[3] if len(it) > 3 else ""
        dot = f'<i class="dot dot-{tone}"></i>' if tone in ("alto", "medio", "bajo") else ""
        note_html = f'<div class="s-note">{html.escape(note)}</div>' if note else ""
        cells += (f'<div class="s-cell"><div class="s-label">{dot}{html.escape(label)}</div>'
                  f'<div class="s-value">{html.escape(value)}</div>{note_html}</div>')
    return f'<div class="strip" style="--n:{len(items)}">{cells}</div>'


def pill(level: str) -> str:
    key = {"Alto": "alto", "Medio": "medio", "Bajo": "bajo"}[level]
    return f'<span class="pill pill-{key}"><i></i>{level}</span>'


def risk_badge(level: str) -> str:
    return pill(level)


def bar_list(rows: list[tuple[str, float]], fmt=lambda v: f"{v:.1f}", color: str = "#0A72EF") -> str:
    """Barras horizontales con la cifra a la derecha. rows = [(etiqueta, valor)]."""
    top = max((abs(v) for _, v in rows), default=1) or 1
    body = "".join(
        f'<div class="b-row"><span class="b-label">{html.escape(lab)}</span>'
        f'<span class="b-track"><span class="b-fill" style="width:{abs(v) / top * 100:.1f}%;background:{color}"></span></span>'
        f'<span class="b-val">{fmt(v)}</span></div>'
        for lab, v in rows
    )
    return f'<div class="bars">{body}</div>'


def diverging_list(rows: list[tuple[str, float]], fmt=lambda v: f"{v:+.1f}".replace(".", ",")) -> str:
    """Barras que parten de un eje central: a la derecha sube el riesgo, a la izquierda baja."""
    top = max((abs(v) for _, v in rows), default=1) or 1
    body = ""
    for lab, v in rows:
        w = abs(v) / top * 50
        pos = f"left:50%;width:{w:.1f}%;background:{UP}" if v > 0 else f"right:50%;width:{w:.1f}%;background:{DOWN}"
        body += (f'<div class="b-row"><span class="b-label">{html.escape(lab)}</span>'
                 f'<span class="b-track dv"><span class="b-fill" style="{pos}"></span></span>'
                 f'<span class="b-val" style="color:{UP if v > 0 else DOWN}">{fmt(v)}</span></div>')
    return f'<div class="bars">{body}</div>'


def kv_rows(rows: list[tuple[str, str]], strong_last: bool = False) -> str:
    """Filas de dato: concepto a la izquierda y cifra a la derecha."""
    out = ""
    for i, (k, v) in enumerate(rows):
        cls = " kv-total" if strong_last and i == len(rows) - 1 else ""
        out += f'<div class="kv{cls}"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
    return f'<div class="kvs">{out}</div>'


def lever_rows(levers, combined: float | None = None, p: float | None = None) -> str:
    """Palancas con el riesgo antes y después; la última fila, todas aplicadas juntas."""
    out = "".join(
        f'<div class="lever"><div><b>{html.escape(r.palanca)}</b>'
        f'<span>{r.p_actual:.0%} → {r.p_nueva:.0%}</span></div><em>−{r.reduccion * 100:.0f} pts</em></div>'
        for r in levers.itertuples()
    )
    if combined is not None and p is not None:
        out += (f'<div class="lever lever-total"><div><b>Todas juntas</b><span>{p:.0%} → {combined:.0%}</span></div>'
                f'<em>−{(p - combined) * 100:.0f} pts</em></div>')
    return f'<div class="levers">{out}</div>'


def act_rows(items: list[str]) -> str:
    """Lista numerada de acciones."""
    return '<div class="acts">' + "".join(
        f'<div class="act"><i>{n}</i><span>{html.escape(t)}</span></div>' for n, t in enumerate(items, start=1)
    ) + "</div>"


def verdict(p: float, alert: float, level: str, headline: str, sentence: str = "") -> str:
    """Veredicto de un cliente: la cifra es lo primero que se lee, con el umbral de alerta en la barra."""
    sent = f"<p>{sentence}</p>" if sentence else ""
    return (
        f'<div class="verdict" style="--lvl:{LEVEL_COLOR[level]}">'
        f'<div class="verdict-top"><div class="verdict-num">{round(p * 100)}<sup>%</sup></div>'
        f'<div>{pill(level)}<h2>{html.escape(headline)}</h2></div></div>{sent}{gauge(p, alert, level)}</div>'
    )


def gauge(p: float, alert: float, level: str = "Alto") -> str:
    pct = max(0.0, min(1.0, p)) * 100
    mark = max(0.0, min(1.0, alert)) * 100
    return (
        f'<div class="gauge" style="--lvl:{LEVEL_COLOR[level]}">'
        f'<div class="gauge-fill" style="width:{pct:.1f}%"></div>'
        f'<div class="gauge-mark" style="left:{mark:.1f}%" title="Umbral de alerta"></div></div>'
        f'<div class="gauge-scale"><span>0%</span><span>Umbral {alert:.0%}</span><span>100%</span></div>'
    )


def stacked_bar(parts: list[tuple[str, int, str]]) -> str:
    """Barra apilada con leyenda. parts = [(nombre, cantidad, color)]."""
    total = max(1, sum(n for _, n, _ in parts))
    segs = "".join(
        f'<div title="{html.escape(name)}: {n}" style="width:{n / total * 100:.2f}%;background:{color}"></div>'
        for name, n, color in parts if n
    )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(name)} <b>{num(n)}</b> · {n / total:.0%}</span>'
        for name, n, color in parts
    )
    return f'<div class="stack">{segs}</div><div class="legend">{legend}</div>'


# ---------- gráficos de líneas ----------

def style_chart(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (
        chart.configure_view(stroke=None)
        .configure(background="transparent", font=FONT)
        .configure_axis(
            labelColor="#64748D", titleColor="#64748D", labelFontSize=12, titleFontSize=12, titleFontWeight=500,
            labelFont=FONT, titleFont=FONT, gridColor="#EEF3F9", domain=False, ticks=False, labelPadding=8, titlePadding=12,
        )
        .configure_legend(labelColor="#273951", labelFont=FONT, labelFontSize=12, symbolType="stroke")
    )


def show_chart(chart: alt.TopLevelMixin) -> None:
    st.altair_chart(style_chart(chart), theme=None)
