"""Pruebas sin navegador: lógica de negocio y que cada página se renderiza sin excepciones.

Ejecutar desde churn_portal/:  python -m pytest -q
"""
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

# Inicio se prueba a través de app.py (sus enlaces solo existen dentro de st.navigation).
PAGES = [
    "views/analizar.py",
    "views/cartera.py",
    "views/campana.py",
    "views/calidad.py",
    "views/sistema.py",
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Las pruebas nunca abren una conexión de correo, aunque haya credenciales reales en los secretos locales."""
    import mailer

    st.cache_data.clear()  # cada prueba parte de una caché vacía, para que el estado de las credenciales no se arrastre
    monkeypatch.setattr(mailer, "check_credentials", lambda cfg: (True, "servidor de prueba"))


# ---------- lógica ----------

def test_model_loads_with_expected_features():
    assert len(core.model_features()) == 19
    assert core.model_card()["classifier"] == "CatBoostClassifier"


def test_holdout_has_1409_rows_and_matches_reported_metrics():
    X, y, p = core.holdout()
    assert len(X) == 1409 and len(y) == 1409
    m = core.metrics_at(y, p, 0.5)
    ref = core.load_metadata()["champion_metrics"]
    assert abs(m["auc"] - ref["ROC_AUC"]) < 0.001
    assert abs(m["brier"] - ref["Brier_Score"]) < 0.001
    assert abs(m["recall"] - ref["Recall"]) < 0.001


def test_sample_predictions_match_file():
    sample = pd.read_csv(core.SAMPLE_PATH)
    X = core.prepare_portfolio(sample)
    p = core.predict_proba(X)
    assert np.allclose(p, sample["Predicted_Probability"].to_numpy(), atol=1e-9)


@pytest.mark.parametrize("p, expected", [(0.45, "Alto"), (0.40, "Alto"), (0.25, "Medio"), (0.19, "Bajo")])
def test_band_thresholds(p, expected):
    assert core.band(p, 0.40) == expected


def test_simulate_impact_arithmetic():
    p = np.array([0.8, 0.6, 0.1])
    monthly = np.array([100.0, 50.0, 70.0])
    r = core.simulate_impact(p, monthly, alert=0.5, success=0.2, cost_per_contact=5.0, months=12)
    assert r["contacted"] == 2
    assert math.isclose(r["expected_churn"], 1.4)
    assert math.isclose(r["saved"], 0.28)
    assert math.isclose(r["revenue"], (0.8 * 100 + 0.6 * 50) * 0.2 * 12)
    assert math.isclose(r["cost"], 10.0)
    assert math.isclose(r["net"], r["revenue"] - 10.0)


def test_simulate_impact_without_contacts_has_nan_roi():
    r = core.simulate_impact(np.array([0.1]), np.array([50.0]), alert=0.5, success=0.2, cost_per_contact=5.0, months=12)
    assert r["contacted"] == 0 and math.isnan(r["roi"])


def test_break_even_success_gives_zero_net():
    p = np.array([0.8, 0.6, 0.1])
    monthly = np.array([100.0, 50.0, 70.0])
    s = core.break_even_success(p, monthly, 0.5, 5.0, 12)
    assert 0 < s < 1
    assert math.isclose(core.simulate_impact(p, monthly, 0.5, s, 5.0, 12)["net"], 0.0, abs_tol=1e-9)


def test_break_even_success_is_nan_without_contacts():
    assert math.isnan(core.break_even_success(np.array([0.1]), np.array([50.0]), 0.5, 5.0, 12))


def test_what_if_levers_lower_risk_and_are_sorted():
    sample = pd.read_csv(core.SAMPLE_PATH)
    X = core.prepare_portfolio(sample)[core.model_features()]
    p = core.predict_proba(X)
    X_one = X.iloc[[int(np.argmax(p))]].reset_index(drop=True)  # el cliente de mayor riesgo de la muestra
    levers = core.what_if(X_one)
    assert (levers["reduccion"] > 0).all()
    assert levers["reduccion"].is_monotonic_decreasing
    assert levers["feature"].is_unique
    if not levers.empty:
        assert core.combined_what_if(X_one, levers) <= float(p.max()) + 1e-9


def _sample_rows(n=6):
    return pd.read_csv(core.SAMPLE_PATH).head(n)[core.model_features()]


def test_template_is_a_valid_portfolio():
    df = core.read_portfolio_csv(core.portfolio_template().encode("utf-8"))
    clean, errors, notes = core.validate_portfolio(df)
    assert errors == [] and notes == []
    assert len(clean) == 3 and "customerID" in clean.columns


def test_read_csv_accepts_semicolons_and_latin1():
    df = _sample_rows()
    raw = df.to_csv(index=False, sep=";").replace("Fiber optic", "Fibra óptica").encode("latin-1")
    parsed = core.read_portfolio_csv(raw)
    assert list(parsed.columns) == list(df.columns) and len(parsed) == len(df)


@pytest.mark.parametrize("raw", [b"", b"   \n"])
def test_read_csv_rejects_empty_files(raw):
    with pytest.raises(ValueError, match="vacío"):
        core.read_portfolio_csv(raw)


def test_read_csv_rejects_oversize_files():
    with pytest.raises(ValueError, match="MB"):
        core.read_portfolio_csv(b"a" * (core.MAX_FILE_MB * 1024 * 1024 + 1))


def test_validate_reports_missing_columns_without_scoring():
    clean, errors, _ = core.validate_portfolio(pd.DataFrame({"tenure": [1, 2]}))
    assert clean is None and "Faltan" in errors[0]


def test_validate_drops_bad_rows_and_says_why():
    df = _sample_rows().reset_index(drop=True)
    df.loc[1, "Contract"] = "Mensual"       # categoría desconocida
    df.loc[2, "tenure"] = -4                # fuera de rango
    df.loc[3, "MonthlyCharges"] = None      # vacío
    clean, errors, notes = core.validate_portfolio(df)
    assert errors == [] and len(clean) == 3
    text = " ".join(notes)
    assert "Mensual" in text and "fuera de rango" in text and "sin antigüedad" in text and "3 de 6" in text


def test_validate_returns_an_error_when_no_row_is_valid():
    df = _sample_rows().reset_index(drop=True)
    df["Contract"] = "Mensual"
    clean, errors, _ = core.validate_portfolio(df)
    assert clean is None and errors


def test_validated_portfolio_can_be_scored():
    clean, _, _ = core.validate_portfolio(_sample_rows())
    out = core.score_portfolio(clean, 0.4)
    assert len(out) == len(clean) and out["Probabilidad"].between(0, 1).all()


def test_missing_assets_is_empty_when_the_package_is_complete():
    assert core.missing_assets() == []


def test_actions_are_suggested_only_for_risk_factors():
    values = pd.Series({"Contract": "Two year", "tenure": 40, "TechSupport": "Yes"})
    contrib = pd.Series({"Contract": 0.2, "tenure": -0.1, "TechSupport": 0.0})
    out = core.actions_for_row(values, contrib)
    assert out == ["Mantener el contacto habitual; no hay un factor de riesgo dominante."]


def test_actions_follow_risk_factors():
    values = pd.Series({"Contract": "Month-to-month", "tenure": 3, "TechSupport": "No"})
    contrib = pd.Series({"Contract": 0.2, "tenure": 0.1, "TechSupport": 0.02})
    out = core.actions_for_row(values, contrib)
    assert out[0].startswith("Ofrecer pasar a un contrato anual")
    assert len(out) <= 3


def test_portfolio_scoring_returns_one_row_per_customer():
    sample = pd.read_csv(core.SAMPLE_PATH).drop(columns=["Actual_Churn", "Predicted_Probability"])
    out = core.score_portfolio(sample, 0.40)
    assert len(out) == len(sample)
    assert set(out["Nivel"]) <= {"Alto", "Medio", "Bajo"}
    assert out["Probabilidad"].is_monotonic_decreasing


def test_missing_columns_detected():
    assert core.missing_columns(pd.DataFrame({"tenure": [1]})) != []


def test_value_labels_only_map_senior_flag_to_yes_no():
    assert core.value_label(1) == "1"
    assert core.value_label(1, "SeniorCitizen") == "Sí"
    assert core.value_label(0, "SeniorCitizen") == "No"
    assert core.value_label("Fiber optic") == "Fibra óptica"


def test_threshold_curve_stops_at_75_percent():
    X, y, p = core.holdout()
    curve = core.threshold_curve(y, p)
    assert curve["umbral"].max() <= 0.75 + 1e-9
    assert {"Detectados", "Acierto al marcar", "Clientes marcados"} <= set(curve.columns)


# ---------- páginas ----------

@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_cartera_scores_an_uploaded_file_and_reports_dropped_rows():
    df = pd.read_csv(core.SAMPLE_PATH).head(8)[core.model_features()]
    df.loc[1, "Contract"] = "Mensual"
    df.loc[2, "tenure"] = -3
    raw = df.to_csv(index=False, sep=";").encode("latin-1")
    at = AppTest.from_file(str(ROOT / "views/cartera.py"), default_timeout=180)
    at.session_state["cartera_source"] = "Mi archivo CSV"
    at.run()
    at.file_uploader[0].upload("cartera.csv", raw, "text/csv").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert any("Mensual" in w.value for w in at.warning)
    assert any("6 de 8" in w.value for w in at.warning)
    assert len(at.dataframe[0].value) == 6


def test_cartera_rejects_a_file_with_missing_columns():
    at = AppTest.from_file(str(ROOT / "views/cartera.py"), default_timeout=180)
    at.session_state["cartera_source"] = "Mi archivo CSV"
    at.run()
    at.file_uploader[0].upload("x.csv", b"tenure,MonthlyCharges\n1,50\n", "text/csv").run()
    assert not at.exception
    assert any("Faltan" in e.value for e in at.error)


def test_app_entrypoint_renders_home_without_exceptions():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_every_source_file_compiles():
    """AppTest no avisa de un error de sintaxis en app.py; esta prueba sí."""
    import ast

    files = [ROOT / "app.py", ROOT / "core.py", ROOT / "components.py", ROOT / "plans.py", *(ROOT / "views").glob("*.py")]
    assert len(files) >= 9
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_every_sample_address_is_synthetic_and_the_demo_customer_has_a_name():
    raw = pd.read_csv(core.SAMPLE_PATH).fillna("")
    assert raw["correo"].str.match(r"^[a-z]+\.\d{4}@example\.com$").all() and raw["correo"].is_unique  # dominio reservado: no recibe nada
    demo = raw[raw["customerID"] == "CLI-0200"]
    assert len(demo) == 1 and demo["nombre"].iloc[0] == "Jaher"


def test_the_demo_customer_is_in_the_default_campaign():
    import plans

    raw = pd.read_csv(core.SAMPLE_PATH).fillna("")
    X = core.prepare_portfolio(raw).reset_index(drop=True)[core.model_features()]
    curve = plans.campaign_curve(plans.portfolio_plans(X), "valor")
    budget = max(50, int(round(curve["costo"].iloc[-1] * 0.35 / 50) * 50))
    demo = int(raw.index[raw["customerID"] == "CLI-0200"][0])
    assert demo in plans.campaign_at(curve, budget)["filas"]


def _campana_button(monkeypatch, cfg, creds=(True, "servidor de prueba")):
    import components
    import mailer

    monkeypatch.setattr(components, "email_config", lambda: cfg)
    monkeypatch.setattr(mailer, "check_credentials", lambda c: creds)
    at = AppTest.from_file(str(ROOT / "views/campana.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return next(b for b in at.button if "correo" in b.label)


def test_campana_without_secrets_runs_the_simulation(monkeypatch):
    import mailer

    assert not _campana_button(monkeypatch, mailer.Config()).disabled


def test_campana_button_follows_the_allowlist_when_connected(monkeypatch):
    import mailer

    connected = dict(username="agente@example.org", password="x" * 16)
    assert _campana_button(monkeypatch, mailer.Config(**connected, demo_email="yo@example.org", allowed=("yo@example.org",))).disabled is False
    assert _campana_button(monkeypatch, mailer.Config(**connected, demo_email="yo@example.org", allowed=("otro@example.org",))).disabled


def test_campana_blocks_the_button_when_credentials_are_invalid(monkeypatch):
    import mailer

    cfg = mailer.Config(username="agente@example.org", password="x" * 16, demo_email="yo@example.org", allowed=("yo@example.org",))
    assert _campana_button(monkeypatch, cfg, creds=(False, "Las credenciales no son válidas")).disabled


def test_moving_the_alert_slider_stores_a_fraction_and_keeps_the_control():
    at = AppTest.from_file(str(ROOT / "views/calidad.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    slider = next(s for s in at.slider if s.key == "alert_slider")
    assert slider.value == round(core.ALERT_DEFAULT * 100)
    slider.set_value(55).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["alert"] == pytest.approx(0.55)  # el portal usa fracciones, no porcentajes
    assert next(s for s in at.slider if s.key == "alert_slider").value == 55  # el control conserva lo que se eligió


def test_campana_header_shows_the_agent_status_tag(monkeypatch):
    import components
    import mailer

    for cfg, creds, expected in [(mailer.Config(username="a@example.org", password="x" * 16), (True, "ok"), "Agente de campaña conectado"),
                                 (mailer.Config(), (True, "ok"), "Agente de campaña sin conexión")]:
        monkeypatch.setattr(components, "email_config", lambda cfg=cfg: cfg)
        monkeypatch.setattr(mailer, "check_credentials", lambda c, creds=creds: creds)
        st.cache_data.clear()
        at = AppTest.from_file(str(ROOT / "views/campana.py"), default_timeout=180)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert expected in " ".join(str(e.value) for e in at.get("html"))
