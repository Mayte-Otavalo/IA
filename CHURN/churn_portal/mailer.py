"""Agente de correo para campañas: redacta el mensaje de cada plan y lo envía solo a direcciones autorizadas.

Reglas de seguridad, aplicadas aquí y no en la interfaz:
- Solo se envía a direcciones que están en la lista autorizada de `.streamlit/secrets.toml`. Cualquier otra se omite.
- Como máximo un mensaje por dirección y por envío, y como máximo `MAX_PER_RUN` mensajes reales por envío.
- Sin credenciales no se envía nada: el resultado es «simulado».
Las direcciones y la contraseña viven solo en los secretos locales, nunca en el código ni en los datos.
Envía por SMTP con la biblioteca estándar (Gmail por defecto), sin servicios de terceros. No depende de Streamlit.
"""
from __future__ import annotations

import html
import re
import smtplib
import ssl
from urllib.parse import quote
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

MAX_PER_RUN = 5
DEFAULT_HOST, DEFAULT_PORT = "smtp.gmail.com", 465
BRAND = "RetenTel"

SENT, SIMULATED, SKIPPED, FAILED = "Enviado", "Simulado", "Omitido", "Error"

_ADDRESS = re.compile(r"^[^@\s<>,;]+@[^@\s<>,;]+\.[^@\s<>,;]+$")


def normalize(address) -> str | None:
    """Dirección de correo en minúsculas, o None si no tiene forma de dirección."""
    text = str(address or "").strip().lower()
    return text if _ADDRESS.match(text) else None


@dataclass(frozen=True)
class Config:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    username: str = ""
    password: str = ""
    allowed: tuple[str, ...] = ()
    demo_email: str | None = None
    demo_name: str = ""

    @property
    def configured(self) -> bool:
        """Hay credenciales para enviar de verdad."""
        return bool(self.username and self.password)


def load_config(section: Mapping | None) -> Config:
    """Configuración desde la sección [email] de los secretos. Sin sección, no hay direcciones autorizadas."""
    section = dict(section or {})
    allowed = tuple(sorted({n for n in (normalize(a) for a in section.get("allowed", [])) if n}))
    try:
        port = int(section.get("port") or DEFAULT_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    return Config(
        host=str(section.get("host") or DEFAULT_HOST),
        port=port,
        username=str(section.get("username", "") or "").strip(),
        # Google muestra la contraseña de aplicación en grupos de cuatro letras separados por espacios.
        password=re.sub(r"\s", "", str(section.get("app_password", "") or "")),
        allowed=allowed,
        demo_email=normalize(section.get("demo_email")),
        demo_name=str(section.get("demo_name", "") or "").strip(),
    )


# ---------- el mensaje ----------

INK, SLATE, BLUE, LINE, PAPER = "#061B31", "#64748D", "#0A72EF", "#E5EDF5", "#F6F9FC"


def compose(nombre: str, tenure: int, offer: list[str], reply_to: str = "") -> tuple[str, str, str]:
    """Correo del plan: asunto, texto y HTML. Se presenta, dice por qué escribe, lista la propuesta y da una salida clara."""
    name = (nombre or "").strip()
    greeting = f"Hola {name}" if name else "Hola"
    months = f"{int(tenure)} {'mes' if int(tenure) == 1 else 'meses'}"
    subject = f"{name}, una propuesta para mantener su servicio" if name else "Una propuesta para mantener su servicio"
    intro = f"Llevamos {months} juntos y queremos cuidar su servicio. Revisamos su cuenta y preparamos esto para usted:"
    yes_link = f"mailto:{reply_to}?subject={quote('SÍ, quiero mi plan')}" if reply_to else ""
    no_link = f"mailto:{reply_to}?subject={quote('BAJA')}" if reply_to else ""
    outro = "Si le interesa, responda SÍ a este correo y la activamos. Si prefiere no recibir más mensajes, responda BAJA."
    text = (f"{greeting},\n\n{intro}\n\n" + "\n".join(f"  ✓ {line}" for line in offer)
            + f"\n\n{outro}\n\nEl equipo de {BRAND}\n")
    rows = "".join(
        f'<tr><td style="padding:9px 0;border-top:1px solid {LINE};font-size:15px;color:{INK};line-height:1.45">'
        f'<span style="color:{BLUE};font-weight:700;padding-right:10px">✓</span>{html.escape(line)}</td></tr>' for line in offer)
    buttons = (
        f'<a href="{html.escape(yes_link)}" style="display:inline-block;background:{BLUE};color:#fff;text-decoration:none;'
        f'font-weight:600;font-size:15px;padding:12px 22px;border-radius:6px">Sí, quiero este plan</a>'
        f'<a href="{html.escape(no_link)}" style="display:inline-block;color:{SLATE};text-decoration:none;font-size:14px;padding:12px 16px">No, gracias</a>'
        if reply_to else "")
    body = (
        f'<div style="background:{PAPER};padding:28px 12px;font-family:\'Segoe UI\',Helvetica,Arial,sans-serif">'
        f'<table role="presentation" width="100%" style="max-width:560px;margin:0 auto;background:#fff;border-radius:8px;border:1px solid {LINE}" cellpadding="0" cellspacing="0">'
        f'<tr><td style="background:{INK};padding:18px 28px;border-radius:8px 8px 0 0;color:#fff;font-size:17px;font-weight:600;letter-spacing:.01em">{BRAND}</td></tr>'
        f'<tr><td style="padding:28px 28px 8px">'
        f'<p style="margin:0 0 6px;font-size:26px;font-weight:300;color:{INK};line-height:1.25">{html.escape(greeting)}</p>'
        f'<p style="margin:0 0 20px;font-size:15px;color:{SLATE};line-height:1.55">{html.escape(intro)}</p>'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 24px">{rows}</table>'
        f'<div style="margin:0 0 8px">{buttons}</div></td></tr>'
        f'<tr><td style="padding:18px 28px 26px;font-size:12.5px;color:{SLATE};line-height:1.5;border-top:1px solid {LINE}">'
        f'Recibe este correo porque es cliente de {BRAND}. Para no recibir más mensajes, responda BAJA.</td></tr></table></div>'
    )
    return subject, text, body


# ---------- el envío ----------

@dataclass(frozen=True)
class Message:
    cliente: str
    email: str | None  # ya normalizado, o None si el cliente no tiene correo válido
    subject: str
    text: str
    html: str


@dataclass(frozen=True)
class Result:
    cliente: str
    destino: str  # la dirección, tal como se envió
    estado: str
    detalle: str = ""


AUTH_HINT = ("El servidor rechazó el usuario o la contraseña. Con Gmail se necesita una contraseña de aplicación "
             "(16 letras, requiere la verificación en dos pasos), no la contraseña de la cuenta.")


def _connect(config: Config) -> smtplib.SMTP_SSL:
    """Abre la conexión segura y entra con las credenciales. Lanza RuntimeError con un mensaje entendible."""
    try:
        server = smtplib.SMTP_SSL(config.host, config.port, timeout=15, context=ssl.create_default_context())
    except (smtplib.SMTPException, OSError) as exc:
        raise RuntimeError(f"No se pudo conectar con {config.host}.") from exc
    try:
        server.login(config.username, config.password)
    except smtplib.SMTPAuthenticationError as exc:
        server.close()
        raise RuntimeError(AUTH_HINT) from exc
    except (smtplib.SMTPException, OSError) as exc:
        server.close()
        raise RuntimeError(f"No se pudo iniciar sesión en {config.host}.") from exc
    return server


def check_credentials(config: Config) -> tuple[bool, str]:
    """Inicia sesión en el servidor sin enviar nada. Devuelve (válidas, descripción)."""
    if not config.configured:
        return False, "Faltan username o app_password."
    try:
        server = _connect(config)
    except RuntimeError as exc:
        return False, str(exc)
    server.quit()
    return True, f"{config.host} · {config.username}"


def smtp_send(config: Config, to: str, subject: str, text: str, html_body: str) -> str:
    """Envía un correo y devuelve su identificador."""
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, formataddr((BRAND, config.username)), to
    msg["Message-ID"] = make_msgid(domain=config.username.split("@")[-1] or None)
    msg["Reply-To"] = config.username
    msg["List-Unsubscribe"] = f"<mailto:{config.username}?subject=BAJA>"
    msg.set_content(text)
    msg.add_alternative(html_body, subtype="html")
    server = _connect(config)
    try:
        server.send_message(msg)
    except smtplib.SMTPException as exc:
        raise RuntimeError(f"El servidor rechazó el envío: {exc}") from exc
    finally:
        server.quit()
    return msg["Message-ID"]


def send_campaign(messages: list[Message], config: Config,
                  post: Callable[[Config, str, str, str, str], str] = smtp_send) -> list[Result]:
    """Envía cada mensaje solo si su dirección está autorizada. Devuelve un resultado por cliente, sin lanzar errores."""
    results: list[Result] = []
    sent_to: set[str] = set()
    real = 0
    for m in messages:
        shown = m.email or "—"
        if m.email is None:
            results.append(Result(m.cliente, shown, SKIPPED, "Sin correo válido"))
        elif m.email not in config.allowed:
            results.append(Result(m.cliente, shown, SKIPPED, "Fuera de la dirección autorizada (demostración)"))
        elif m.email in sent_to:
            results.append(Result(m.cliente, shown, SKIPPED, "Ya recibió un mensaje en este envío"))
        elif real >= MAX_PER_RUN:
            results.append(Result(m.cliente, shown, SKIPPED, f"Límite de {MAX_PER_RUN} mensajes por envío"))
        elif not config.configured:
            sent_to.add(m.email)
            results.append(Result(m.cliente, shown, SIMULATED, "Sin credenciales de correo: no se envió nada"))
        else:
            sent_to.add(m.email)
            real += 1
            try:
                post(config, m.email, m.subject, m.text, m.html)
                results.append(Result(m.cliente, shown, SENT, "Aceptado por el servidor de correo"))
            except RuntimeError as exc:
                results.append(Result(m.cliente, shown, FAILED, str(exc)))
    return results
