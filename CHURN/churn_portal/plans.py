"""Planes de retención: qué ofrecer a cada cliente y a quién contactar con un presupuesto dado.

Para un cliente se prueban todas las combinaciones razonables de contrato, servicios, método de pago y descuento.
Cada combinación se puntúa con el modelo cargado y se valora como el cambio esperado del ingreso de los próximos
`horizon` meses, descontando lo que cuesta la oferta:

    ingreso esperado = (1 − riesgo) × (cargo mensual tras el descuento − costo de servicios) × meses
    valor del plan   = ingreso esperado con el plan − ingreso esperado sin hacer nada

Son estimaciones del modelo, no efectos causales medidos, y los costos de los servicios son supuestos editables.
No depende de Streamlit.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import core

HORIZON = 12
DISCOUNTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
SERVICE_COST = {"TechSupport": 4.0, "OnlineSecurity": 3.0}  # USD por mes y cliente: supuestos
CONTACT_COST = 5.0  # USD por cliente contactado (llamada o mensaje del equipo): supuesto
MAX_CUSTOMERS = 2000  # en carteras más grandes se evalúan los clientes de mayor riesgo

CONTRACT_RANK = {"Month-to-month": 0, "One year": 1, "Two year": 2}
CONTRACT_TEXT = {"One year": "Contrato anual", "Two year": "Contrato de 2 años"}
PAY_AUTO = "Bank transfer (automatic)"
PAY_MANUAL = ("Electronic check", "Mailed check")
PLAN_COLUMNS = ["Contract", "TechSupport", "OnlineSecurity", "PaymentMethod", "disc"]


def _candidates(row: pd.Series) -> pd.DataFrame:
    """Combinaciones posibles para un cliente: solo cambios que mejoran lo que ya tiene."""
    contracts = [row["Contract"]] + [c for c, r in CONTRACT_RANK.items() if r > CONTRACT_RANK[row["Contract"]]]
    support = [row["TechSupport"]] + (["Yes"] if row["TechSupport"] == "No" else [])
    security = [row["OnlineSecurity"]] + (["Yes"] if row["OnlineSecurity"] == "No" else [])
    payment = [row["PaymentMethod"]] + ([PAY_AUTO] if row["PaymentMethod"] in PAY_MANUAL else [])
    return pd.DataFrame(list(itertools.product(contracts, support, security, payment, DISCOUNTS)), columns=PLAN_COLUMNS)


def apply_plan(X: pd.DataFrame, plan: pd.Series | dict) -> pd.DataFrame:
    """Copia de X con el plan aplicado. El cargo total se mantiene proporcional al cargo mensual."""
    out = X.copy()
    base_monthly = out["MonthlyCharges"].to_numpy(dtype=float)
    for col in ("Contract", "TechSupport", "OnlineSecurity", "PaymentMethod"):
        out[col] = plan[col]
    factor = 1.0 - float(plan["disc"])
    out["MonthlyCharges"] = base_monthly * factor
    out["TotalCharges"] = out["TotalCharges"].to_numpy(dtype=float) * factor
    return out


def _label(row: pd.Series, plan: pd.Series) -> str:
    parts = []
    if plan["Contract"] != row["Contract"]:
        parts.append(CONTRACT_TEXT[plan["Contract"]])
    if plan["TechSupport"] != row["TechSupport"]:
        parts.append("soporte técnico incluido")
    if plan["OnlineSecurity"] != row["OnlineSecurity"]:
        parts.append("seguridad en línea incluida")
    if plan["PaymentMethod"] != row["PaymentMethod"]:
        parts.append("pago con débito automático")
    if plan["disc"] > 0:
        parts.append(f"{plan['disc']:.0%} de descuento durante {HORIZON} meses".replace("%", " %"))
    return (parts[0][0].upper() + parts[0][1:] + ("; " + "; ".join(parts[1:]) if len(parts) > 1 else "")) if parts else "Sin cambios"


def _score_many(X: pd.DataFrame, horizon: int, service_cost: dict | None, contact_cost: float) -> pd.DataFrame:
    """Planes de uno o varios clientes con sus cifras, en una sola llamada al modelo. Columna «fila»: cliente de X."""
    sc = {**SERVICE_COST, **(service_cost or {})}
    X = X[core.model_features()].reset_index(drop=True)
    frames = []
    for i in range(len(X)):
        cand = _candidates(X.iloc[i])
        cand["fila"] = i
        frames.append(cand)
    plans = pd.concat(frames, ignore_index=True)
    base = X.iloc[plans["fila"].to_numpy()].reset_index(drop=True)
    disc = plans["disc"].to_numpy(dtype=float)

    X_plans = base.copy()
    for col in ("Contract", "TechSupport", "OnlineSecurity", "PaymentMethod"):
        X_plans[col] = plans[col].to_numpy()
    X_plans["MonthlyCharges"] = base["MonthlyCharges"].to_numpy(dtype=float) * (1 - disc)
    X_plans["TotalCharges"] = base["TotalCharges"].to_numpy(dtype=float) * (1 - disc)

    p0 = core.predict_proba(X)[plans["fila"].to_numpy()]
    p1 = core.predict_proba(X_plans)
    m0 = base["MonthlyCharges"].to_numpy(dtype=float)
    service = ((plans["TechSupport"].to_numpy() != base["TechSupport"].to_numpy()) * sc["TechSupport"]
               + (plans["OnlineSecurity"].to_numpy() != base["OnlineSecurity"].to_numpy()) * sc["OnlineSecurity"])
    changed = (
        (plans["Contract"].to_numpy() != base["Contract"].to_numpy())
        | (plans["TechSupport"].to_numpy() != base["TechSupport"].to_numpy())
        | (plans["OnlineSecurity"].to_numpy() != base["OnlineSecurity"].to_numpy())
        | (plans["PaymentMethod"].to_numpy() != base["PaymentMethod"].to_numpy())
        | (disc > 0)
    )
    contact = np.where(changed, contact_cost, 0.0)

    revenue = (1 - p1) * (m0 * (1 - disc) - service) * horizon
    base_revenue = (1 - p0) * m0 * horizon
    plans["p0"] = p0
    plans["p_plan"] = p1
    plans["costo"] = (1 - p1) * (disc * m0 + service) * horizon + contact   # gasto esperado: oferta + contacto
    plans["protegido"] = (p0 - p1) * m0 * horizon                           # ingreso que deja de perderse
    plans["valor"] = revenue - base_revenue - contact                       # cambio neto esperado del ingreso
    return plans


def retention_plans(X_one: pd.DataFrame, horizon: int = HORIZON, service_cost: dict | None = None,
                    contact_cost: float = CONTACT_COST) -> pd.DataFrame:
    """Todos los planes posibles para un cliente, valorados y ordenados por valor."""
    plans = _score_many(X_one, horizon, service_cost, contact_cost).drop(columns="fila")
    row = X_one.iloc[0]
    plans["etiqueta"] = [_label(row, plans.iloc[i]) for i in range(len(plans))]
    return plans.sort_values("valor", ascending=False).reset_index(drop=True)


def plan_steps(row: pd.Series, plan: pd.Series) -> list[str]:
    """Pasos concretos del plan, en frases cortas."""
    steps = []
    if plan["Contract"] != row["Contract"]:
        steps.append(f"Ofrecer {CONTRACT_TEXT[plan['Contract']].lower()}")
    if plan["TechSupport"] != row["TechSupport"]:
        steps.append("Incluir soporte técnico")
    if plan["OnlineSecurity"] != row["OnlineSecurity"]:
        steps.append("Incluir seguridad en línea")
    if plan["PaymentMethod"] != row["PaymentMethod"]:
        steps.append("Pasar el pago a débito automático")
    if plan["disc"] > 0:
        steps.append(f"Descuento de {plan['disc'] * 100:.0f} % durante {HORIZON} meses")
    return steps or ["Mantener el contacto habitual"]


def pick_plans(plans: pd.DataFrame, alert: float) -> dict[str, pd.Series]:
    """Tres planes con intención distinta. Nunca recomienda un plan que pierda dinero."""
    useful = plans[plans["valor"] > 1e-9]
    picks: dict[str, pd.Series] = {}
    if useful.empty:
        picks["Sin cambios"] = plans.iloc[(plans["valor"].abs()).argmin()]
        return picks
    picks["Recomendado"] = useful.sort_values(["valor", "costo"], ascending=[False, True]).iloc[0]
    below = useful[useful["p_plan"] < alert]
    if not below.empty:
        picks["Mínimo para bajar del umbral"] = below.sort_values("costo").iloc[0]
    picks["Máxima retención"] = useful.sort_values("p_plan").iloc[0]
    seen, unique = set(), {}
    for name, plan in picks.items():  # si dos intenciones dan el mismo plan, se muestra una sola
        key = plan["etiqueta"]
        if key not in seen:
            seen.add(key)
            unique[name] = plan
    return unique


def lifecycle(X_one: pd.DataFrame, plan: pd.Series | None = None) -> pd.DataFrame:
    """Riesgo del cliente según su antigüedad (0 a 72 meses), con y sin plan, y el resto de datos fijos."""
    row = X_one.iloc[0]
    t = np.arange(0, 73)
    ratio = float(row["TotalCharges"]) / (float(row["tenure"]) * float(row["MonthlyCharges"])) if row["tenure"] > 0 and row["MonthlyCharges"] > 0 else 1.0
    X_t = pd.concat([X_one] * len(t), ignore_index=True)
    X_t["tenure"] = t
    X_t["TotalCharges"] = t * float(row["MonthlyCharges"]) * ratio
    out = pd.DataFrame({"meses": t, "actual": core.predict_proba(X_t)})
    if plan is not None:
        out["con_plan"] = core.predict_proba(apply_plan(X_t, plan))
    return out


def months_to_threshold(curve: pd.DataFrame, column: str, tenure: int, alert: float) -> int | None:
    """Primer mes, desde la antigüedad actual, en que el riesgo baja del umbral. None si nunca ocurre."""
    ahead = curve[curve["meses"] >= tenure]
    below = ahead[ahead[column] < alert]
    return None if below.empty else int(below["meses"].iloc[0])


def similar_customers(row: pd.Series) -> dict:
    """Historial real: cuántos clientes parecidos hay y qué porcentaje se fue (no es una predicción)."""
    ref = core.load_reference()
    tenure, monthly = float(row["tenure"]), float(row["MonthlyCharges"])
    for span_t, span_m, same_internet in ((6, 0.20, True), (12, 0.30, True), (18, 0.40, False)):
        mask = (
            (ref["Contract"] == row["Contract"])
            & ((ref["tenure"] - tenure).abs() <= span_t)
            & ((ref["MonthlyCharges"] - monthly).abs() <= monthly * span_m)
        )
        if same_internet:
            mask &= ref["InternetService"] == row["InternetService"]
        if int(mask.sum()) >= 30:
            break
    sel = ref[mask]
    n = len(sel)
    return {"n": n, "rate": float(sel["churn"].mean()) if n else float("nan"),
            "criterio": f"mismo contrato, antigüedad ±{span_t} meses y cargo ±{span_m:.0%}".replace("%", " %")}


def portfolio_plans(X: pd.DataFrame, horizon: int = HORIZON, service_cost: dict | None = None,
                    contact_cost: float = CONTACT_COST, max_customers: int = MAX_CUSTOMERS) -> pd.DataFrame:
    """El mejor plan de cada cliente de una cartera. En carteras muy grandes evalúa los de mayor riesgo."""
    X = X[core.model_features()].reset_index(drop=True)
    order = np.arange(len(X))
    if len(X) > max_customers:
        order = np.argsort(-core.predict_proba(X))[:max_customers]
    scored = _score_many(X.iloc[order].reset_index(drop=True), horizon, service_cost, contact_cost)
    best = scored.loc[scored.groupby("fila")["valor"].idxmax()].reset_index(drop=True)
    Xo = X.iloc[order].reset_index(drop=True)
    rows = [{"fila": int(order[int(r["fila"])]), "p0": r["p0"], "p_plan": r["p_plan"], "costo": r["costo"],
             "protegido": r["protegido"], "valor": r["valor"], "plan": _label(Xo.iloc[int(r["fila"])], r),
             **{col: r[col] for col in PLAN_COLUMNS}}
            for _, r in best.iterrows()]
    return pd.DataFrame(rows)


def campaign_curve(plans: pd.DataFrame, order: str) -> pd.DataFrame:
    """Valor y gasto acumulados al contactar clientes en un orden. order: «valor» (por retorno) o «riesgo»."""
    d = plans[plans["valor"] > 1e-9].copy()
    if order == "valor":
        d = d.assign(_k=d["valor"] / d["costo"].clip(lower=1e-9)).sort_values("_k", ascending=False)
    else:
        d = d.sort_values("p0", ascending=False)
    curve = pd.DataFrame({"clientes": np.arange(1, len(d) + 1), "costo": d["costo"].cumsum().to_numpy(),
                          "valor": d["valor"].cumsum().to_numpy(), "fila": d["fila"].to_numpy()})
    return pd.concat([pd.DataFrame({"clientes": [0], "costo": [0.0], "valor": [0.0], "fila": [-1]}), curve], ignore_index=True)


def campaign_at(curve: pd.DataFrame, budget: float) -> dict:
    """Resultado de la campaña con un presupuesto de gasto esperado."""
    ok = curve[curve["costo"] <= budget + 1e-9]
    last = ok.iloc[-1]
    return {"clientes": int(last["clientes"]), "costo": float(last["costo"]), "valor": float(last["valor"]),
            "filas": [int(f) for f in ok["fila"].tolist() if f >= 0]}


def offer_lines(row, plan) -> list[str]:
    """Lo que se le ofrece al cliente, en lenguaje de cliente y a partir del plan exacto de la campaña."""
    lines = []
    if plan["Contract"] != row["Contract"]:
        lines.append("Pasar a un " + CONTRACT_TEXT[plan["Contract"]].lower())
    if plan["TechSupport"] != row["TechSupport"]:
        lines.append("Soporte técnico incluido, sin costo adicional")
    if plan["OnlineSecurity"] != row["OnlineSecurity"]:
        lines.append("Seguridad en línea incluida, sin costo adicional")
    if plan["PaymentMethod"] != row["PaymentMethod"]:
        lines.append("Pago con débito automático, para que no tenga que acordarse cada mes")
    if float(plan["disc"]) > 0:
        lines.append(f"{float(plan['disc']) * 100:.0f} % de descuento en su factura durante {HORIZON} meses")
    return lines
