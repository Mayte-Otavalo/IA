"""Pruebas del agente de correo: ninguna toca la red; el servidor SMTP se sustituye por uno falso."""
import pathlib
import smtplib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mailer  # noqa: E402

OK = "demo.autorizado@example.org"
OTHER = "otra.persona@example.net"


def config(**kw):
    base = dict(username="agente@example.org", password="abcdabcdabcdabcd", allowed=(OK,))
    base.update(kw)
    return mailer.Config(**base)


def message(address, cliente="A"):
    return mailer.Message(cliente, mailer.normalize(address), "Asunto", "texto", "<p>html</p>")


class FakeSender:
    def __init__(self, fail_for=()):
        self.calls, self.fail_for = [], set(fail_for)

    def __call__(self, cfg, to, subject, text, html_body):
        self.calls.append(to)
        if to in self.fail_for:
            raise RuntimeError("rechazado")
        return "<id@example.org>"


# ---------- direcciones ----------

def test_normalize_accepts_addresses_and_rejects_the_rest():
    assert mailer.normalize("  Ana.Perez@Example.ORG ") == "ana.perez@example.org"
    for bad in (None, "", "sin-arroba", "a@b", "a b@c.com", "a@b.com, c@d.com", "a@b.com\nBcc: x@y.com"):
        assert mailer.normalize(bad) is None


# ---------- configuración ----------

def test_config_without_secrets_has_nobody_authorized():
    cfg = mailer.load_config(None)
    assert cfg.allowed == () and not cfg.configured and cfg.demo_email is None and cfg.demo_name == ""


def test_config_normalizes_allowlist_and_strips_app_password_spaces():
    cfg = mailer.load_config({"username": "u@gmail.com", "app_password": "abcd efgh ijkl mnop",
                              "allowed": [" Uno@Example.org ", "roto", "uno@example.org"], "demo_email": "Uno@Example.org", "demo_name": " Ana "})
    assert cfg.allowed == ("uno@example.org",) and cfg.demo_email == "uno@example.org" and cfg.demo_name == "Ana"
    assert cfg.password == "abcdefghijklmnop" and cfg.configured
    assert (cfg.host, cfg.port) == ("smtp.gmail.com", 465)


def test_config_ignores_a_bad_port():
    assert mailer.load_config({"port": "no"}).port == 465


# ---------- el envío ----------

def test_only_the_authorized_address_is_sent_to():
    fake = FakeSender()
    out = mailer.send_campaign([message(OK), message(OTHER, "B")], config(), fake)
    assert [r.estado for r in out] == [mailer.SENT, mailer.SKIPPED]
    assert fake.calls == [OK]  # la otra dirección nunca llega al emisor


def test_without_credentials_nothing_is_sent_and_the_result_is_simulated():
    fake = FakeSender()
    out = mailer.send_campaign([message(OK)], config(username="", password=""), fake)
    assert out[0].estado == mailer.SIMULATED and fake.calls == []


def test_empty_allowlist_sends_nothing():
    fake = FakeSender()
    out = mailer.send_campaign([message(OK)], config(allowed=()), fake)
    assert out[0].estado == mailer.SKIPPED and fake.calls == []


def test_one_message_per_address_per_run():
    fake = FakeSender()
    out = mailer.send_campaign([message(OK, "A"), message(OK, "B")], config(), fake)
    assert [r.estado for r in out] == [mailer.SENT, mailer.SKIPPED] and fake.calls == [OK]


def test_customers_without_an_address_are_skipped():
    out = mailer.send_campaign([mailer.Message("A", None, "s", "t", "h")], config(), FakeSender())
    assert out[0].estado == mailer.SKIPPED and out[0].destino == "—"


def test_the_per_run_cap_applies_even_if_the_allowlist_is_larger():
    addresses = [f"d{i}@example.org" for i in range(mailer.MAX_PER_RUN + 2)]
    fake = FakeSender()
    out = mailer.send_campaign([message(a, str(i)) for i, a in enumerate(addresses)], config(allowed=tuple(addresses)), fake)
    assert len(fake.calls) == mailer.MAX_PER_RUN
    assert [r.estado for r in out].count(mailer.SKIPPED) == 2


def test_a_failure_is_reported_and_does_not_stop_the_run():
    other = "segundo@example.org"
    fake = FakeSender(fail_for={OK})
    out = mailer.send_campaign([message(OK), message(other, "B")], config(allowed=(OK, other)), fake)
    assert [r.estado for r in out] == [mailer.FAILED, mailer.SENT]


def test_results_show_the_address_the_message_went_to():
    out = mailer.send_campaign([message(OK)], config(), FakeSender())
    assert out[0].destino == OK


# ---------- el mensaje ----------

def test_compose_has_subject_text_and_html_with_the_offer_and_an_exit():
    subject, text, body = mailer.compose("Ana", 3, ["Pasar a un contrato de 2 años", "10 % de descuento"], reply_to="agente@example.org")
    assert "Ana" in subject and "3 meses" in text and "✓ Pasar a un contrato de 2 años" in text
    assert "BAJA" in text and "10 % de descuento" in body
    assert "mailto:agente@example.org?subject=" in body  # los botones responden al agente


def test_compose_without_name_and_one_month():
    subject, text, _ = mailer.compose("", 1, ["x"])
    assert text.startswith("Hola,") and "1 mes " in text and not subject.startswith(",")


def test_compose_escapes_html():
    _, _, body = mailer.compose("<b>Ana</b>", 3, ["<script>alert(1)</script>"])
    assert "<script>" not in body and "&lt;script&gt;" in body


# ---------- SMTP (servidor falso) ----------

class FakeSMTP:
    instances = []
    login_error = None

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.sent, self.closed = host, port, [], False
        FakeSMTP.instances.append(self)

    def login(self, user, password):
        if FakeSMTP.login_error:
            raise FakeSMTP.login_error
        self.user = user

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        self.closed = True

    def close(self):
        self.closed = True


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances, FakeSMTP.login_error = [], None
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def test_smtp_send_builds_a_multipart_message(smtp):
    mailer.smtp_send(config(), OK, "Asunto con ñ", "texto plano", "<p>html</p>")
    (server,) = smtp.instances
    (msg,) = server.sent
    assert msg["To"] == OK and "agente@example.org" in msg["From"] and msg["Subject"] == "Asunto con ñ"
    assert msg["Reply-To"] == "agente@example.org" and "BAJA" in msg["List-Unsubscribe"]
    assert msg.get_body(("plain",)).get_content().strip() == "texto plano"
    assert "<p>html</p>" in msg.get_body(("html",)).get_content()
    assert server.closed and (server.host, server.port) == ("smtp.gmail.com", 465)


def test_a_wrong_password_gives_an_understandable_message(smtp):
    smtp.login_error = smtplib.SMTPAuthenticationError(535, b"nope")
    ok, detail = mailer.check_credentials(config())
    assert not ok and "contraseña de aplicación" in detail and smtp.instances[0].closed


def test_check_credentials_logs_in_without_sending(smtp):
    ok, detail = mailer.check_credentials(config())
    assert ok and "smtp.gmail.com" in detail and smtp.instances[0].sent == []


def test_check_credentials_without_them_does_not_connect(smtp):
    ok, _ = mailer.check_credentials(config(username="", password=""))
    assert not ok and smtp.instances == []


def test_a_connection_failure_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise OSError("sin red")

    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", boom)
    ok, detail = mailer.check_credentials(config())
    assert not ok and "No se pudo conectar" in detail
    out = mailer.send_campaign([message(OK)], config())
    assert out[0].estado == mailer.FAILED


def test_a_rejected_send_is_a_failed_result_and_closes_the_connection(smtp):
    class Rejecting(FakeSMTP):
        def send_message(self, msg):
            raise smtplib.SMTPRecipientsRefused({OK: (550, b"no")})

    mailer.smtplib.SMTP_SSL = Rejecting
    out = mailer.send_campaign([message(OK)], config())
    assert out[0].estado == mailer.FAILED and "rechazó" in out[0].detalle and Rejecting.instances[-1].closed


# ---------- secretos ----------

def test_no_personal_address_or_password_is_committed_to_the_sources():
    """La dirección real y la contraseña de aplicación viven solo en .streamlit/secrets.toml (ignorado por git)."""
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if not secrets.exists():
        pytest.skip("no hay secrets.toml en este equipo")
    import tomllib

    section = tomllib.loads(secrets.read_text(encoding="utf-8")).get("email", {})
    cfg = mailer.load_config(section)
    needles = {cfg.username.lower(), cfg.password.lower(), cfg.demo_email or "", *cfg.allowed} - {""}
    if not needles:
        pytest.skip("la sección [email] de los secretos está vacía")
    leaks = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".csv", ".md", ".json", ".toml", ".example", ".txt", ".html", ".css"} \
                and path != secrets and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts:
            text = path.read_text(encoding="utf-8", errors="ignore").lower().replace(" ", "")
            leaks += [str(path.relative_to(ROOT)) for n in needles if n in text]
    assert leaks == [], f"datos privados aparecen en: {leaks}"
