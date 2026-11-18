"""Lógica del portal: carga del modelo, puntuación y explicaciones. No depende de Streamlit.

Todo lo que se muestra en la interfaz sale de aquí. El portal nunca entrena ni reajusta el modelo:
solo lo carga y lo evalúa sobre datos ya existentes.
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ASSETS = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS / "champion_pipeline.joblib"
METADATA_PATH = ASSETS / "metadata.json"
TELCO_PATH = ASSETS / "Telco-Customer-Churn.csv"
SAMPLE_PATH = ASSETS / "sample_test_data.csv"

SEED = 42
TEST_SIZE = 0.2
ALERT_DEFAULT = 0.40  # umbral de alerta por defecto (ver Calidad del modelo)

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

LABELS = {
    "gender": "Género", "SeniorCitizen": "Adulto mayor", "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes", "tenure": "Antigüedad (meses)",
    "PhoneService": "Servicio telefónico", "MultipleLines": "Líneas múltiples",
    "InternetService": "Servicio de internet", "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea", "DeviceProtection": "Protección de equipos",
    "TechSupport": "Soporte técnico", "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de películas", "Contract": "Tipo de contrato",
    "PaperlessBilling": "Factura electrónica", "PaymentMethod": "Método de pago",
    "MonthlyCharges": "Cargo mensual", "TotalCharges": "Cargo total acumulado",
}

VALUE_LABELS = {
    "Yes": "Sí", "No": "No", "Female": "Mujer", "Male": "Hombre",
    "No internet service": "Sin servicio de internet", "No phone service": "Sin servicio telefónico",
    "DSL": "DSL", "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mes a mes", "One year": "Anual (1 año)", "Two year": "Bienal (2 años)",
    "Electronic check": "Cheque electrónico", "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia automática",
    "Credit card (automatic)": "Tarjeta de crédito automática",
}


def label(col: str) -> str:
    return LABELS.get(col, col)


def value_label(value, column: str | None = None) -> str:
    """Texto legible para un valor. Solo SeniorCitizen usa 0/1 como Sí/No; el resto de números se muestra tal cual."""
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if column == "SeniorCitizen":
        return "Sí" if value == 1 else "No"
    if isinstance(value, str):
        return VALUE_LABELS.get(value, value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def missing_assets() -> list[str]:
    """Archivos que el portal necesita para arrancar y que no están en assets/."""
    return [p.name for p in (MODEL_PATH, METADATA_PATH, TELCO_PATH, SAMPLE_PATH) if not p.exists()]


# ---------- carga ----------

@lru_cache(maxsize=1)
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # aviso de versión de scikit-learn al cargar el pickle
        return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def model_features() -> list[str]:
    return list(load_model().named_steps["preprocessor"].feature_names_in_)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_reference() -> pd.DataFrame:
    """Historial de referencia (7.043 clientes) con la etiqueta de abandono como 0/1."""
    df = pd.read_csv(TELCO_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)  # igual que el notebook
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def options(column: str) -> list:
    """Opciones de un campo categórico, tomadas del historial de referencia."""
    return sorted(load_reference()[column].dropna().unique().tolist(), key=str)


@lru_cache(maxsize=1)
def holdout():
    """Partición de prueba del equipo (80/20 estratificada, semilla 42): 1.409 clientes.

    Devuelve X, y (0/1) y la probabilidad de abandono del modelo cargado.
    """
    df = load_reference()
    features = model_features()
    X = df[features]
    y = df["churn"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_te = X_te.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True).to_numpy()
    return X_te, y_te, predict_proba(X_te)


# ---------- puntuación ----------

def predict_proba(X: pd.DataFrame) -> np.ndarray:
    return load_model().predict_proba(X[model_features()])[:, 1]


def band(p: float, alert: float) -> str:
    """Alto: en o sobre el umbral de alerta. Medio: la mitad del umbral hasta el umbral. Bajo: el resto."""
    if p >= alert:
        return "Alto"
    if p >= alert / 2:
        return "Medio"
    return "Bajo"


def baseline_row() -> pd.Series:
    """Cliente típico: mediana en numéricos y valor más frecuente en categóricos."""
    ref = load_reference()
    values = {}
    for col in model_features():
        if col in NUMERIC:
            values[col] = float(ref[col].median())
        else:
            values[col] = ref[col].mode().iloc[0]
    return pd.Series(values)


def contributions(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Cuánto cambia la probabilidad al reemplazar cada dato por el del cliente típico.

    Positivo: ese dato eleva el riesgo frente a un cliente típico. Negativo: lo reduce.
    Es una comparación de sensibilidad del modelo, no una relación causal.
    """
    features = model_features()
    base = baseline_row()
    p0 = predict_proba(X)
    cols = {}
    for f in features:
        Xa = X.copy()
        Xa[f] = base[f]
        cols[f] = p0 - predict_proba(Xa)
    return pd.DataFrame(cols, index=X.index), p0


# Acciones sugeridas: se activan solo si el dato del cliente es de riesgo y aporta al riesgo.
ACTION_RULES = {
    "Contract": (lambda v: v == "Month-to-month", "Ofrecer pasar a un contrato anual con un beneficio de precio."),
    "tenure": (lambda v: v <= 12, "Activar un seguimiento de bienvenida durante el primer año."),
    "TechSupport": (lambda v: v == "No", "Ofrecer soporte técnico incluido en su plan."),
    "OnlineSecurity": (lambda v: v == "No", "Ofrecer el paquete de seguridad en línea."),
    "PaymentMethod": (lambda v: v == "Electronic check", "Invitar a pagar con débito automático."),
    "InternetService": (lambda v: v == "Fiber optic", "Revisar que el precio de la fibra óptica sea competitivo."),
    "MonthlyCharges": (lambda v: v >= 70, "Revisar si el plan actual se ajusta al uso del cliente."),
    "OnlineBackup": (lambda v: v == "No", "Ofrecer respaldo en línea como beneficio."),
    "DeviceProtection": (lambda v: v == "No", "Ofrecer protección de equipos."),
}
MIN_CONTRIBUTION = 0.005  # menos de medio punto porcentual no se considera un factor


def actions_for_row(values: pd.Series, contrib: pd.Series, max_items: int = 3) -> list[str]:
    out: list[str] = []
    for feature in contrib.sort_values(ascending=False).index:
        if contrib[feature] < MIN_CONTRIBUTION:
            break
        rule = ACTION_RULES.get(feature)
        if rule is None:
            continue
        condition, text = rule
        if condition(values[feature]) and text not in out:
            out.append(text)
        if len(out) == max_items:
            break
    return out or ["Mantener el contacto habitual; no hay un factor de riesgo dominante."]


def explain_customer(X_one: pd.DataFrame) -> dict:
    """Probabilidad, factores y acciones para un cliente."""
    contrib_df, p = contributions(X_one)
    values = X_one.iloc[0]
    contrib = contrib_df.iloc[0]
    factors = (
        pd.DataFrame({
            "feature": contrib.index,
            "label": [label(f) for f in contrib.index],
            "value": [value_label(values[f], f) for f in contrib.index],
            "contrib": contrib.values,
        })
        .sort_values("contrib", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "p": float(p[0]),
        "factors": factors,
        "actions": actions_for_row(values, contrib),
    }


def global_drivers(X: pd.DataFrame) -> pd.DataFrame:
    """Impacto medio absoluto de cada dato sobre la probabilidad, en un conjunto de clientes."""
    contrib_df, _ = contributions(X)
    impact = contrib_df.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({
        "feature": impact.index,
        "label": [label(f) for f in impact.index],
        "impact": impact.values,
    })


# ---------- cartera ----------

def missing_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in model_features() if c not in df.columns]


def prepare_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos correctos para el modelo. Ignora columnas extra como la etiqueta o predicciones previas."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)  # igual que el notebook
    out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    return out


MAX_ROWS = 20_000
MAX_FILE_MB = 25


def read_portfolio_csv(raw: bytes) -> pd.DataFrame:
    """Lee un CSV subido por el usuario. Acepta coma, punto y coma o tabulador, y UTF-8 o Latin-1.

    Lanza ValueError con un mensaje que el usuario puede entender.
    """
    import io

    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"El archivo pesa más de {MAX_FILE_MB} MB. Divídalo en partes más pequeñas.")
    if not raw.strip():
        raise ValueError("El archivo está vacío.")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise ValueError("No se pudo leer el archivo como CSV. Revise que tenga una fila de títulos y una fila por cliente.") from exc
    raise ValueError("No se pudo leer la codificación del archivo. Guárdelo como CSV UTF-8.") from last_error


def validate_portfolio(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Valida una cartera antes de puntuarla.

    Devuelve (cartera limpia, errores, avisos). Con errores no hay cartera: el usuario debe corregir el archivo.
    Los avisos describen filas que se descartaron o datos que conviene revisar; el resto se puntúa.
    """
    errors: list[str] = []
    notes: list[str] = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        return None, ["El archivo no tiene filas de clientes."], notes
    missing = missing_columns(df)
    if missing:
        shown = ", ".join(missing[:5]) + (f" y {len(missing) - 5} más" if len(missing) > 5 else "")
        errors.append(f"Faltan {len(missing)} columnas: {shown}. Descargue la plantilla para ver el formato.")
    if len(df) > MAX_ROWS:
        errors.append(f"El archivo tiene {len(df):,} filas y el máximo es {MAX_ROWS:,}.".replace(",", "."))
    if errors:
        return None, errors, notes

    out = prepare_portfolio(df).reset_index(drop=True)
    keep = pd.Series(True, index=out.index)

    bad_num = out["tenure"].isna() | out["MonthlyCharges"].isna()
    if bad_num.any():
        notes.append(f"{int(bad_num.sum())} filas sin antigüedad o cargo mensual válidos (se descartaron).")
        keep &= ~bad_num
    out_of_range = (out["tenure"] < 0) | (out["tenure"] > 120) | (out["MonthlyCharges"] < 0) | (out["MonthlyCharges"] > 1000)
    out_of_range &= keep
    if out_of_range.any():
        notes.append(f"{int(out_of_range.sum())} filas con antigüedad o cargo fuera de rango (se descartaron).")
        keep &= ~out_of_range

    unknown_cols: dict[str, list[str]] = {}
    for col in CATEGORICAL:
        if col == "SeniorCitizen":
            continue
        known = set(options(col))
        bad = ~out[col].isin(known) & keep
        if bad.any():
            unknown_cols[col] = sorted({str(v) for v in out.loc[bad, col].unique()})[:3]
            keep &= ~bad
    for col, values in unknown_cols.items():
        notes.append(f"Valores no reconocidos en «{label(col)}»: {', '.join(values)}. Esas filas se descartaron.")

    bad_senior = ~out["SeniorCitizen"].isin([0, 1]) & keep
    if bad_senior.any():
        notes.append(f"{int(bad_senior.sum())} filas con «Adulto mayor» distinto de 0 o 1 (se descartaron).")
        keep &= ~bad_senior

    clean = out[keep].reset_index(drop=True)
    if clean.empty:
        return None, ["Ninguna fila es válida. Revise los valores con la plantilla."] + notes, []
    if len(clean) < len(out):
        notes.append(f"Se puntuaron {len(clean):,} de {len(out):,} filas.".replace(",", "."))
    return clean, [], notes


def portfolio_template() -> str:
    """CSV de ejemplo con las columnas exactas que necesita el modelo (tres clientes de la muestra)."""
    sample = pd.read_csv(SAMPLE_PATH).head(3)[model_features()]
    sample.insert(0, "customerID", [f"CLI-{i:04d}" for i in range(1, len(sample) + 1)])
    return sample.to_csv(index=False)


def best_levers(X: pd.DataFrame) -> pd.DataFrame:
    """Para cada cliente, la palanca que más baja su riesgo (calculada por lotes). Columnas: palanca, reduccion."""
    X = X[model_features()].reset_index(drop=True)
    p0 = predict_proba(X)
    best = np.zeros(len(X))
    text = np.array([""] * len(X), dtype=object)
    for feature, new_value, applies, label_text in LEVERS:
        mask = X[feature].map(applies).to_numpy(dtype=bool)
        if not mask.any():
            continue
        X_new = X[mask].copy()
        X_new[feature] = new_value
        drop = p0[mask] - predict_proba(X_new)
        idx = np.flatnonzero(mask)
        better = drop > best[idx]
        best[idx[better]] = drop[better]
        text[idx[better]] = label_text
    return pd.DataFrame({"palanca": text, "reduccion": best})


def score_portfolio(df: pd.DataFrame, alert: float) -> pd.DataFrame:
    """Puntúa una cartera: probabilidad, nivel y la palanca que más baja el riesgo de cada cliente."""
    X = prepare_portfolio(df).reset_index(drop=True)
    p = predict_proba(X)
    levers = best_levers(X)
    has_lever = levers["reduccion"].to_numpy() >= MIN_LEVER_DROP
    if "customerID" in X.columns:
        names = X["customerID"].astype(str)
    else:
        names = pd.Series([f"CLI-{i + 1:04d}" for i in range(len(X))])
    out = pd.DataFrame({"Cliente": names})
    out["Probabilidad"] = p
    out["Nivel"] = [band(v, alert) for v in p]
    out["Palanca principal"] = np.where(has_lever, levers["palanca"], "Mantener el contacto habitual")
    out["Reducción"] = np.where(has_lever, levers["reduccion"], 0.0)
    out["_p"] = p
    out["_monthly"] = X["MonthlyCharges"].to_numpy()
    out["_row"] = np.arange(len(X))
    return out.sort_values("Probabilidad", ascending=False).reset_index(drop=True)


# ---------- métricas ----------

def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "f2": float(fbeta_score(y, yhat, beta=2, zero_division=0)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "flagged_share": float(yhat.mean()),
    }


CURVE_MAX = 0.75  # por encima quedan muy pocos clientes marcados y la precisión deja de ser estable


def threshold_curve(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.05, CURVE_MAX + 0.001, 0.01), 2):
        m = metrics_at(y, p, t)
        rows.append({"umbral": t, "Detectados": m["recall"],
                     "Acierto al marcar": m["precision"],
                     "Clientes marcados": m["flagged_share"]})
    return pd.DataFrame(rows)


# ---------- impacto (hipotético) ----------

def simulate_impact(p: np.ndarray, monthly: np.ndarray, alert: float,
                    success: float, cost_per_contact: float, months: int) -> dict:
    """Escenario hipotético. Supone que la oferta retiene a una fracción fija de los clientes marcados."""
    contacted = p >= alert
    n = int(contacted.sum())
    expected_churn = float(p[contacted].sum())
    saved = expected_churn * success
    revenue = float((p[contacted] * monthly[contacted]).sum() * success * months)
    cost = n * cost_per_contact
    net = revenue - cost
    return {
        "contacted": n,
        "expected_churn": expected_churn,
        "saved": saved,
        "revenue": revenue,
        "cost": cost,
        "net": net,
        "roi": (net / cost) if cost > 0 else float("nan"),
    }


def break_even_success(p: np.ndarray, monthly: np.ndarray, alert: float,
                       cost_per_contact: float, months: int) -> float:
    """Tasa de éxito de la oferta con la que la campaña no gana ni pierde. NaN si no hay a quién contactar."""
    contacted = p >= alert
    n = int(contacted.sum())
    at_stake = float((p[contacted] * monthly[contacted]).sum() * months)
    if n == 0 or at_stake <= 0:
        return float("nan")
    return n * cost_per_contact / at_stake


# Palancas de retención: cambio que se simula, condición para que aplique y texto para el usuario.
LEVERS = [
    ("Contract", "One year", lambda v: v == "Month-to-month", "Pasar a contrato anual"),
    ("Contract", "Two year", lambda v: v != "Two year", "Pasar a contrato de 2 años"),
    ("TechSupport", "Yes", lambda v: v == "No", "Incluir soporte técnico"),
    ("OnlineSecurity", "Yes", lambda v: v == "No", "Incluir seguridad en línea"),
    ("OnlineBackup", "Yes", lambda v: v == "No", "Incluir respaldo en línea"),
    ("DeviceProtection", "Yes", lambda v: v == "No", "Incluir protección de equipos"),
    ("PaymentMethod", "Bank transfer (automatic)", lambda v: v in ("Electronic check", "Mailed check"),
     "Pasar a débito automático"),
]
MIN_LEVER_DROP = 0.01  # una palanca que baja menos de 1 punto porcentual no se muestra


def what_if(X_one: pd.DataFrame, max_items: int = 4) -> pd.DataFrame:
    """Qué pasa con el riesgo si se cambia un dato del cliente (sensibilidad del modelo, no causalidad).

    Devuelve una fila por palanca con la probabilidad actual y la nueva, ordenadas por reducción.
    """
    row = X_one.iloc[0]
    p0 = float(predict_proba(X_one)[0])
    rows = []
    for feature, new_value, applies, text in LEVERS:
        if not applies(row[feature]):
            continue
        X_new = X_one.copy()
        X_new[feature] = new_value
        p_new = float(predict_proba(X_new)[0])
        if p0 - p_new >= MIN_LEVER_DROP:
            rows.append({"palanca": text, "feature": feature, "nuevo": new_value, "p_actual": p0, "p_nueva": p_new,
                         "reduccion": p0 - p_new})
    out = pd.DataFrame(rows, columns=["palanca", "feature", "nuevo", "p_actual", "p_nueva", "reduccion"])
    if out.empty:
        return out
    # Para un mismo dato (p. ej. contrato) se conserva la palanca que más baja el riesgo.
    out = out.sort_values("reduccion", ascending=False).drop_duplicates("feature")
    return out.head(max_items).reset_index(drop=True)


def combined_what_if(X_one: pd.DataFrame, levers: pd.DataFrame) -> float:
    """Probabilidad si se aplican a la vez todas las palancas de la tabla."""
    X_new = X_one.copy()
    for feature, new_value in zip(levers["feature"], levers["nuevo"]):
        X_new[feature] = new_value
    return float(predict_proba(X_new)[0])


def impact_curve(p: np.ndarray, monthly: np.ndarray, alert: float,
                 cost_per_contact: float, months: int) -> pd.DataFrame:
    rows = []
    for s in np.arange(0.05, 0.401, 0.05):
        r = simulate_impact(p, monthly, alert, float(s), cost_per_contact, months)
        rows.append({"Tasa de éxito": round(float(s), 2), "Ganancia neta (hipotética)": r["net"]})
    return pd.DataFrame(rows)


# ---------- información del sistema ----------

REPORTED_CHAMPION = "CatBoost"  # así lo nombra el reporte y el notebook del equipo


def model_card() -> dict:
    """Datos del modelo cargado y de lo que dice el reporte. Sirve para detectar discrepancias."""
    clf = load_model().named_steps["classifier"]
    params = clf.get_params()
    champ = load_metadata().get("champion_metrics", {})
    return {
        "classifier": type(clf).__name__,
        "trees": params.get("n_estimators", params.get("iterations")),  # RandomForest: n_estimators; CatBoost: iterations
        "depth": params.get("max_depth", params.get("depth")),
        "features": len(model_features()),
        "size_kb": MODEL_PATH.stat().st_size / 1024,
        "reported_champion": REPORTED_CHAMPION,
        "reported_auc": champ.get("ROC_AUC"),
        "reported_brier": champ.get("Brier_Score"),
        "reported_recall": champ.get("Recall"),
    }
