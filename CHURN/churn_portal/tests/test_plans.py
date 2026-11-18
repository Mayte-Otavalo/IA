"""Pruebas del motor de planes de retención: invariantes que las cifras deben cumplir siempre."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core  # noqa: E402
import plans  # noqa: E402


@pytest.fixture(scope="module")
def sample_X():
    return core.prepare_portfolio(pd.read_csv(core.SAMPLE_PATH)).reset_index(drop=True)[core.model_features()]


@pytest.fixture(scope="module")
def risky(sample_X):
    p = core.predict_proba(sample_X)
    return sample_X.iloc[[int(p.argmax())]].reset_index(drop=True)


@pytest.fixture(scope="module")
def portfolio(sample_X):
    return plans.portfolio_plans(sample_X)


def test_doing_nothing_is_a_plan_worth_exactly_zero(risky):
    pl = plans.retention_plans(risky)
    nothing = pl[pl["etiqueta"] == "Sin cambios"].iloc[0]
    assert nothing["valor"] == pytest.approx(0.0, abs=1e-9)
    assert nothing["costo"] == pytest.approx(0.0, abs=1e-9)
    assert nothing["p_plan"] == pytest.approx(nothing["p0"])


def test_plans_never_downgrade_what_the_customer_already_has(risky):
    pl = plans.retention_plans(risky)
    row = risky.iloc[0]
    rank = plans.CONTRACT_RANK
    assert (pl["Contract"].map(rank) >= rank[row["Contract"]]).all()
    if row["TechSupport"] == "Yes":
        assert (pl["TechSupport"] == "Yes").all()


def test_value_is_protected_revenue_minus_investment(risky):
    pl = plans.retention_plans(risky)
    row = risky.iloc[0]
    m0, h = float(row["MonthlyCharges"]), plans.HORIZON
    for r in pl.head(10).itertuples():
        # valor = ingreso esperado con el plan − ingreso esperado sin plan; se reconstruye con la fórmula del módulo
        disc = r.disc
        service = (r.TechSupport != row["TechSupport"]) * plans.SERVICE_COST["TechSupport"] \
            + (r.OnlineSecurity != row["OnlineSecurity"]) * plans.SERVICE_COST["OnlineSecurity"]
        changed = (r.Contract != row["Contract"]) or (r.TechSupport != row["TechSupport"]) \
            or (r.OnlineSecurity != row["OnlineSecurity"]) or (r.PaymentMethod != row["PaymentMethod"]) or disc > 0
        contact = plans.CONTACT_COST if changed else 0.0
        expected = (1 - r.p_plan) * (m0 * (1 - disc) - service) * h - (1 - r.p0) * m0 * h - contact
        assert r.valor == pytest.approx(expected, rel=1e-9, abs=1e-6)


def test_picked_plans_always_leave_value(risky):
    picks = plans.pick_plans(plans.retention_plans(risky), 0.4)
    assert picks and all(pl["valor"] > 0 for name, pl in picks.items() if name != "Sin cambios")
    assert picks["Recomendado"]["valor"] == max(pl["valor"] for pl in picks.values())


def test_a_customer_with_nothing_to_gain_gets_no_recommendation(risky):
    pl = plans.retention_plans(risky, contact_cost=1e9)
    assert list(plans.pick_plans(pl, 0.4)) == ["Sin cambios"]


def test_the_plan_lowers_the_lifecycle_curve(risky):
    best = plans.pick_plans(plans.retention_plans(risky), 0.4)["Recomendado"]
    curve = plans.lifecycle(risky, best)
    assert len(curve) == 73 and (curve["con_plan"] < curve["actual"]).all()
    t = int(risky["tenure"].iloc[0])
    assert curve["actual"].iloc[t] == pytest.approx(float(core.predict_proba(risky)[0]), abs=0.02)


def test_months_to_threshold_is_none_when_risk_never_drops():
    curve = pd.DataFrame({"meses": np.arange(73), "actual": np.full(73, 0.9)})
    assert plans.months_to_threshold(curve, "actual", 0, 0.4) is None
    curve["actual"] = np.linspace(0.9, 0.1, 73)
    assert plans.months_to_threshold(curve, "actual", 0, 0.4) > 0


def test_similar_customers_come_from_real_history(risky):
    s = plans.similar_customers(risky.iloc[0])
    assert s["n"] >= 1 and 0.0 <= s["rate"] <= 1.0 and "meses" in s["criterio"]


def test_portfolio_has_one_best_plan_per_customer(portfolio, sample_X):
    assert len(portfolio) == len(sample_X)
    assert (portfolio["valor"] >= -1e-9).all()  # el mejor plan nunca es peor que no hacer nada


def test_campaign_curves_grow_and_end_at_the_same_total(portfolio):
    by_value, by_risk = plans.campaign_curve(portfolio, "valor"), plans.campaign_curve(portfolio, "riesgo")
    for curve in (by_value, by_risk):
        assert curve["costo"].is_monotonic_increasing and curve["valor"].is_monotonic_increasing
    assert by_value["costo"].iloc[-1] == pytest.approx(by_risk["costo"].iloc[-1])
    assert by_value["valor"].iloc[-1] == pytest.approx(by_risk["valor"].iloc[-1])


@pytest.mark.parametrize("budget", [300, 800, 1500, 2500])
def test_ordering_by_expected_value_never_loses_to_ordering_by_risk(portfolio, budget):
    best = plans.campaign_at(plans.campaign_curve(portfolio, "valor"), budget)
    naive = plans.campaign_at(plans.campaign_curve(portfolio, "riesgo"), budget)
    assert best["costo"] <= budget + 1e-6 and naive["costo"] <= budget + 1e-6
    assert best["valor"] >= naive["valor"] - 1e-6


def test_a_zero_budget_contacts_nobody(portfolio):
    assert plans.campaign_at(plans.campaign_curve(portfolio, "valor"), 0)["clientes"] == 0


def test_very_large_portfolios_are_capped_to_the_riskiest(sample_X):
    big = pd.concat([sample_X] * 12, ignore_index=True)  # 2.400 clientes
    out = plans.portfolio_plans(big, max_customers=300)
    assert len(out) == 300
    p = core.predict_proba(big)
    assert out["p0"].min() >= np.sort(p)[-300] - 1e-9
