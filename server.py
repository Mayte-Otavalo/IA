"""
Servidor Federado HTTP/REST (Computadora 1) con:
  - Agregacion FedAvg ponderada por tamano de muestra local (n_k)
  - Deteccion de desconexion de clientes mediante Heartbeat + Timeout por ronda
  - Continuacion automatica con los clientes disponibles (sin bloqueo indefinido)
  - Sincronizacion automatica del modelo global cuando un cliente reconecta
  - Dashboard Web en vivo en http://<IP_SERVIDOR>:5000 para demostracion en clase
"""

import os
import sys
import json
import time
import socket
import argparse
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from federated_model import FederatedMLP, fed_avg
from prepare_data import FEATURE_COLS


def get_local_ip() -> str:
    """Obtiene la direccion IP de la computadora en la red local (LAN/Wi-Fi)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class FederatedCoordinator:
    """Coordinador central del servidor federado con tolerancia a fallos y timeouts."""

    def __init__(
        self,
        total_rounds: int = 15,
        expected_clients: int = 3,
        min_clients: int = 1,
        round_timeout: float = 12.0,
        heartbeat_timeout: float = 5.0,
        local_epochs: int = 8,
        lr: float = 0.025,
        batch_size: int = 64,
        auto_start: bool = True,
        round_pause: float = 1.0,
        scenario_name: str = "iid",
        output_json: str | None = None,
    ):
        self.lock = threading.RLock()
        self.total_rounds = total_rounds
        self.expected_clients = expected_clients
        self.min_clients = min_clients
        self.round_timeout = round_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.auto_start = auto_start
        self.round_pause = round_pause
        self.scenario_name = scenario_name
        self.output_json = output_json

        # Cargar conjunto de prueba global (exclusivo del servidor para evaluacion imparcial)
        test_df = pd.read_csv("data/test_global.csv")
        self.X_test = test_df[FEATURE_COLS].values
        self.y_test = test_df["target"].values.astype(int)

        # Inicializar modelo global MLP con semilla fija para comparacion justa IID vs non-IID
        self.global_model = FederatedMLP(seed=42)
        self.initial_eval = self.global_model.evaluate(self.X_test, self.y_test)

        # Estado del sistema
        self.state = "WAITING_FOR_CLIENTS"  # WAITING_FOR_CLIENTS | TRAINING_ROUND | FINISHED
        self.current_round = 0
        self.round_start_time = 0.0
        self.clients = {}  # client_id -> dict de estado
        self.round_submissions = {}  # client_id -> update dict de la ronda actual
        self.history = []  # historial de metricas por ronda
        self.event_logs = []  # bitacora de eventos con timestamp

        self._log_event(
            f"Servidor Federado iniciado ({self.global_model.count_parameters()} parametros MLP). "
            f"Evaluacion inicial Ronda 0 -> Acc={self.initial_eval['accuracy']:.4f}, F1={self.initial_eval['f1']:.4f}"
        )

    def _log_event(self, message: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"time": ts, "level": level, "message": message}
        self.event_logs.append(entry)
        if len(self.event_logs) > 120:
            self.event_logs = self.event_logs[-120:]
        prefix = {"INFO": "[INFO]", "WARN": "[ALERTA DESCONEXION]", "SUCCESS": "[RECONEXION / AGREGACION]"}.get(level, "[INFO]")
        print(f"{ts} {prefix} {message}", flush=True)

    def register_client(self, client_id: int, ip_addr: str, num_samples: int, class_dist: dict) -> dict:
        with self.lock:
            now = time.time()
            was_known = client_id in self.clients
            prev_status = self.clients[client_id]["status"] if was_known else None

            self.clients[client_id] = {
                "client_id": client_id,
                "ip": ip_addr,
                "num_samples": num_samples,
                "class_dist": class_dist,
                "last_seen": now,
                "status": "ONLINE",
                "last_round_submitted": self.clients[client_id]["last_round_submitted"] if was_known else 0,
                "last_metrics": self.clients[client_id].get("last_metrics", {}) if was_known else {},
            }

            if was_known and prev_status == "DISCONNECTED":
                self._log_event(
                    f"Cliente {client_id} ({ip_addr}) se ha RECONECTADO en Ronda {self.current_round}. "
                    f"Sincronizando modelo global actual...",
                    level="SUCCESS",
                )
            else:
                self._log_event(
                    f"Cliente {client_id} registrado desde {ip_addr} con n={num_samples} muestras locales "
                    f"(Clases: {class_dist})."
                )

            # Iniciar automaticamente cuando los clientes esperados se han registrado
            if self.state == "WAITING_FOR_CLIENTS" and self.auto_start and len(self.clients) >= self.expected_clients:
                self._start_next_round_locked()

            return self._build_client_Directive_locked(client_id)

    def heartbeat(self, client_id: int, ip_addr: str) -> dict:
        with self.lock:
            now = time.time()
            if client_id not in self.clients:
                self.clients[client_id] = {
                    "client_id": client_id,
                    "ip": ip_addr,
                    "num_samples": 0,
                    "class_dist": {},
                    "last_seen": now,
                    "status": "ONLINE",
                    "last_round_submitted": 0,
                    "last_metrics": {},
                }
            else:
                if self.clients[client_id]["status"] == "DISCONNECTED":
                    self._log_event(
                        f"¡Cliente {client_id} ({ip_addr}) ha vuelto a conectarse en la Ronda {self.current_round}! "
                        f"Enviando pesos globales sincronizados de la Ronda {self.current_round}.",
                        level="SUCCESS",
                    )
                self.clients[client_id]["last_seen"] = now
                self.clients[client_id]["ip"] = ip_addr
                self.clients[client_id]["status"] = "ONLINE"

            return self._build_client_Directive_locked(client_id)

    def _build_client_Directive_locked(self, client_id: int) -> dict:
        already_submitted = client_id in self.round_submissions
        need_train = (self.state == "TRAINING_ROUND") and (not already_submitted)
        return {
            "state": self.state,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "need_train": need_train,
            "local_epochs": self.local_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "global_weights": self.global_model.get_weights() if need_train else None,
        }

    def submit_update(self, payload: dict) -> dict:
        with self.lock:
            cid = int(payload["client_id"])
            r = int(payload["round"])
            now = time.time()

            if cid in self.clients:
                if self.clients[cid]["status"] == "DISCONNECTED":
                    self._log_event(
                        f"Cliente {cid} reconectado y entregando actualizacion en Ronda {r}.",
                        level="SUCCESS",
                    )
                self.clients[cid]["last_seen"] = now
                self.clients[cid]["status"] = "ONLINE"

            if self.state != "TRAINING_ROUND" or r != self.current_round:
                self._log_event(
                    f"Actualizacion tardia descartada de Cliente {cid} (Ronda cliente={r}, Ronda servidor={self.current_round}). "
                    f"El cliente sera sincronizado con la ronda actual.",
                    level="WARN",
                )
                return {"accepted": False, "reason": "stale_round", "current_round": self.current_round}

            self.round_submissions[cid] = {
                "client_id": cid,
                "num_samples": int(payload["num_samples"]),
                "weights": payload["weights"],
                "metrics": payload["metrics"],
                "timestamp": now,
            }
            if cid in self.clients:
                self.clients[cid]["last_round_submitted"] = r
                self.clients[cid]["last_metrics"] = payload["metrics"]

            m = payload["metrics"]
            self._log_event(
                f"Ronda {r}/{self.total_rounds} | Recibidos pesos locales de Cliente {cid} "
                f"(n={payload['num_samples']}, Acc local={m['accuracy']:.4f}, F1 local={m['f1']:.4f}, Loss={m['loss']:.4f})."
            )

            # Si todos los clientes esperados ya enviaron su actualizacion, agregar de inmediato
            if len(self.round_submissions) >= self.expected_clients:
                self._finalize_round_locked(reason="ALL_CLIENTS_RESPONDED")

            return {"accepted": True, "current_round": self.current_round}

    def _start_next_round_locked(self):
        if self.current_round >= self.total_rounds:
            self.state = "FINISHED"
            self._log_event(f"Entrenamiento Federado completado ({self.total_rounds} rondas).", level="SUCCESS")
            self._save_results_locked()
            return

        self.current_round += 1
        self.state = "TRAINING_ROUND"
        self.round_start_time = time.time()
        self.round_submissions = {}
        self._log_event(
            f"--- Iniciando Ronda {self.current_round}/{self.total_rounds} "
            f"(Timeout de ronda: {self.round_timeout:.1f}s) ---"
        )

    def _finalize_round_locked(self, reason: str):
        r = self.current_round
        updates = list(self.round_submissions.values())
        if not updates:
            self._log_event(
                f"Ronda {r}: Ningun cliente respondio antes del timeout ({self.round_timeout:.1f}s). Reintentando ronda...",
                level="WARN",
            )
            self.round_start_time = time.time()
            return

        # Identificar clientes desconectados o ausentes en esta ronda
        active_ids = sorted([u["client_id"] for u in updates])
        all_known_ids = sorted(list(self.clients.keys()))
        missing_ids = [cid for cid in all_known_ids if cid not in active_ids]

        if missing_ids:
            for m_id in missing_ids:
                self.clients[m_id]["status"] = "DISCONNECTED"
            self._log_event(
                f"Ronda {r} finalizada por TIMEOUT ({reason}): Clientes sin respuesta -> {missing_ids}. "
                f"Continuando FedAvg con clientes disponibles: {active_ids}.",
                level="WARN",
            )

        # Ejecutar agregacion FedAvg con los clientes disponibles
        new_weights, agg_info = fed_avg(updates)
        self.global_model.set_weights(new_weights)
        global_metrics = self.global_model.evaluate(self.X_test, self.y_test)

        round_record = {
            "round": r,
            "reason": reason,
            "active_clients": active_ids,
            "missing_clients": missing_ids,
            "num_active_clients": len(active_ids),
            "weighting_factors": agg_info["weighting_factors"],
            "client_drift_l2": agg_info["client_drift_l2"],
            "mean_client_drift_l2": agg_info["mean_client_drift_l2"],
            "global_metrics": global_metrics,
            "client_metrics": {f"client_{u['client_id']}": u["metrics"] for u in updates},
        }
        self.history.append(round_record)

        self._log_event(
            f"Ronda {r} Agregada (FedAvg con {active_ids}, pesos={agg_info['weighting_factors']}) -> "
            f"GLOBAL Test: Acc={global_metrics['accuracy']:.4f} | Precision={global_metrics['precision']:.4f} | "
            f"Recall={global_metrics['recall']:.4f} | F1={global_metrics['f1']:.4f} | "
            f"Loss={global_metrics['loss']:.4f} | Drift L2={agg_info['mean_client_drift_l2']:.4f}",
            level="SUCCESS",
        )

        self._save_results_locked()

        # Pausa breve opcional para visualizacion en vivo y pasar a siguiente ronda
        if self.round_pause > 0:
            time.sleep(self.round_pause)
        self._start_next_round_locked()

    def _save_results_locked(self):
        os.makedirs("results", exist_ok=True)
        target_path = self.output_json or f"results/history_{self.scenario_name}.json"
        data = {
            "scenario": self.scenario_name,
            "total_rounds": self.total_rounds,
            "round_timeout": self.round_timeout,
            "local_epochs": self.local_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "initial_eval": self.initial_eval,
            "history": self.history,
            "events": self.event_logs,
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def monitor_loop(self):
        """Hilo supervisor que verifica heartbeats y ejecuta el timeout de cada ronda sin bloquear el servidor."""
        while True:
            time.sleep(0.4)
            with self.lock:
                if self.state == "FINISHED":
                    break

                now = time.time()
                # 1. Verificar clientes que dejaron de enviar heartbeat
                for cid, cinfo in self.clients.items():
                    if cinfo["status"] == "ONLINE" and (now - cinfo["last_seen"]) > self.heartbeat_timeout:
                        cinfo["status"] = "DISCONNECTED"
                        self._log_event(
                            f"Cliente {cid} ({cinfo['ip']}) perdio conexion (sin señal por >{self.heartbeat_timeout:.1f}s).",
                            level="WARN",
                        )

                # 2. Verificar timeout de la ronda activa
                if self.state == "TRAINING_ROUND":
                    elapsed = now - self.round_start_time
                    if elapsed >= self.round_timeout:
                        if len(self.round_submissions) >= self.min_clients:
                            self._finalize_round_locked(reason="ROUND_TIMEOUT_PARTIAL_CLIENTS")
                        else:
                            self._log_event(
                                f"Timeout de ronda {self.current_round} alcanzado ({elapsed:.1f}s), "
                                f"esperando al menos {self.min_clients} cliente(s) disponible(s)...",
                                level="WARN",
                            )
                            self.round_start_time = now

    def get_dashboard_state(self) -> dict:
        with self.lock:
            now = time.time()
            elapsed = (now - self.round_start_time) if self.state == "TRAINING_ROUND" else 0.0
            remaining = max(0.0, self.round_timeout - elapsed) if self.state == "TRAINING_ROUND" else 0.0
            return {
                "scenario": self.scenario_name,
                "state": self.state,
                "current_round": self.current_round,
                "total_rounds": self.total_rounds,
                "round_timeout": self.round_timeout,
                "round_elapsed": round(elapsed, 1),
                "round_remaining": round(remaining, 1),
                "clients": list(self.clients.values()),
                "submitted_in_round": list(self.round_submissions.keys()),
                "initial_eval": self.initial_eval,
                "history": self.history,
                "events": self.event_logs[-25:],
            }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>Servidor de Aprendizaje Federado - Air Quality</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 18px; background: #0f172a; color: #f8fafc; }
    h1 { margin: 0 0 6px 0; font-size: 22px; color: #38bdf8; }
    .subtitle { color: #94a3b8; font-size: 13px; margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }
    .clients-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 10px; }
    .client-box { padding: 12px; border-radius: 8px; background: #0f172a; border: 2px solid #475569; }
    .client-box.ONLINE { border-color: #22c55e; }
    .client-box.DISCONNECTED { border-color: #ef4444; background: #2a1215; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }
    .badge.ONLINE { background: #16a34a; color: white; }
    .badge.DISCONNECTED { background: #dc2626; color: white; animation: pulse 1s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }
    th, td { border-bottom: 1px solid #334155; padding: 6px 8px; text-align: center; }
    th { background: #0f172a; color: #38bdf8; }
    .logs { font-family: Consolas, monospace; font-size: 12px; max-height: 220px; overflow-y: auto; background: #090d16; padding: 10px; border-radius: 6px; }
    .log-WARN { color: #f87171; font-weight: bold; }
    .log-SUCCESS { color: #4ade80; }
    .log-INFO { color: #cbd5e1; }
    .btn { background: #0284c7; color: white; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Servidor Federado (FedAvg) — Air Quality Dataset</h1>
  <div class="subtitle" id="status-bar">Cargando estado del servidor...</div>

  <div class="card" style="margin-bottom: 16px;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <strong>Estado de las Computadoras Clientes (Red Local)</strong>
      <button class="btn" onclick="startTraining()">Iniciar / Forzar Ronda</button>
    </div>
    <div class="clients-grid" id="clients-container"></div>
  </div>

  <div class="grid">
    <div class="card">
      <strong>Evolución por Ronda (Modelo Global en Test Server)</strong>
      <table>
        <thead>
          <tr>
            <th>Ronda</th>
            <th>Clientes Activos</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Loss</th>
            <th>Drift L2</th>
          </tr>
        </thead>
        <tbody id="history-body"></tbody>
      </table>
    </div>
    <div class="card">
      <strong>Bitácora de Red, Agregación y Desconexiones en Tiempo Real</strong>
      <div class="logs" id="logs-container"></div>
    </div>
  </div>

  <script>
    async function refresh() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        document.getElementById('status-bar').innerHTML =
          `Escenario: <b>${data.scenario.toUpperCase()}</b> | Estado: <b>${data.state}</b> | ` +
          `Ronda: <b>${data.current_round}/${data.total_rounds}</b> | ` +
          `Timeout restante en ronda: <b>${data.round_remaining}s</b> (de ${data.round_timeout}s)`;

        const cDiv = document.getElementById('clients-container');
        cDiv.innerHTML = '';
        [1, 2, 3].forEach(cid => {
          const c = data.clients.find(x => x.client_id === cid);
          if (!c) {
            cDiv.innerHTML += `<div class="client-box"><b>Cliente ${cid}</b><br><span style="color:#94a3b8">No registrado aún</span></div>`;
          } else {
            const submitted = data.submitted_in_round.includes(cid) ? '✅ Pesos enviados' : '⏳ Entrenando...';
            const m = c.last_metrics || {};
            cDiv.innerHTML += `
              <div class="client-box ${c.status}">
                <div style="display:flex; justify-content:space-between;">
                  <b>Computadora Cliente ${c.client_id}</b>
                  <span class="badge ${c.status}">${c.status === 'ONLINE' ? 'CONECTADO' : 'DESCONECTADO'}</span>
                </div>
                <div style="font-size:12px; margin-top:6px; color:#cbd5e1;">
                  IP: ${c.ip} | Muestras: <b>${c.num_samples}</b><br>
                  Última ronda enviada: <b>${c.last_round_submitted}</b> (${c.status === 'ONLINE' ? submitted : '⚠️ Sin conexión'})<br>
                  Último Acc Local: <b>${m.accuracy !== undefined ? m.accuracy : '-'}</b> | F1 Local: <b>${m.f1 !== undefined ? m.f1 : '-'}</b>
                </div>
              </div>`;
          }
        });

        const tbody = document.getElementById('history-body');
        tbody.innerHTML = '';
        data.history.slice().reverse().forEach(h => {
          const gm = h.global_metrics;
          const miss = h.missing_clients.length > 0 ? ` (⚠️ Sin C${h.missing_clients.join(',C')})` : '';
          tbody.innerHTML += `
            <tr>
              <td><b>#${h.round}</b></td>
              <td>[${h.active_clients.join(', ')}]${miss}</td>
              <td><b>${(gm.accuracy*100).toFixed(2)}%</b></td>
              <td>${(gm.precision*100).toFixed(2)}%</td>
              <td>${(gm.recall*100).toFixed(2)}%</td>
              <td><b>${(gm.f1*100).toFixed(2)}%</b></td>
              <td>${gm.loss.toFixed(4)}</td>
              <td>${h.mean_client_drift_l2.toFixed(4)}</td>
            </tr>`;
        });

        const logsDiv = document.getElementById('logs-container');
        logsDiv.innerHTML = data.events.slice().reverse().map(
          e => `<div class="log-${e.level}">[${e.time}] ${e.message}</div>`
        ).join('');
      } catch (e) {}
    }
    async function startTraining() {
      await fetch('/api/start', {method: 'POST'});
      refresh();
    }
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


def make_handler(coordinator: FederatedCoordinator):
    class FederatedRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Silenciar logs HTTP estandar para mantener limpia la consola del servidor
            return

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                body = DASHBOARD_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._send_json(coordinator.get_dashboard_state())
            else:
                self._send_json({"error": "Not found"}, status=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(raw) if raw else {}
            client_ip = self.client_address[0]

            if self.path == "/register":
                resp = coordinator.register_client(
                    client_id=int(payload["client_id"]),
                    ip_addr=client_ip,
                    num_samples=int(payload["num_samples"]),
                    class_dist=payload.get("class_dist", {}),
                )
                self._send_json(resp)
            elif self.path == "/heartbeat":
                resp = coordinator.heartbeat(
                    client_id=int(payload["client_id"]),
                    ip_addr=client_ip,
                )
                self._send_json(resp)
            elif self.path == "/submit_update":
                resp = coordinator.submit_update(payload)
                self._send_json(resp)
            elif self.path == "/api/start":
                with coordinator.lock:
                    if coordinator.state == "WAITING_FOR_CLIENTS":
                        coordinator._start_next_round_locked()
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "Not found"}, status=404)

    return FederatedRequestHandler


def main():
    parser = argparse.ArgumentParser(description="Servidor de Aprendizaje Federado (FedAvg)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host de escucha (0.0.0.0 para red local)")
    parser.add_argument("--port", type=int, default=5000, help="Puerto TCP del servidor")
    parser.add_argument("--rounds", type=int, default=15, help="Numero total de rondas federadas")
    parser.add_argument("--expected-clients", type=int, default=3, help="Numero esperado de clientes")
    parser.add_argument("--min-clients", type=int, default=1, help="Minimo de clientes para agregar tras timeout")
    parser.add_argument("--round-timeout", type=float, default=12.0, help="Segundos de timeout maximo por ronda")
    parser.add_argument("--heartbeat-timeout", type=float, default=5.0, help="Segundos para detectar desconexion")
    parser.add_argument("--local-epochs", type=int, default=8, help="Epocas de entrenamiento local por ronda")
    parser.add_argument("--lr", type=float, default=0.025, help="Learning rate local")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size local")
    parser.add_argument("--round-pause", type=float, default=0.5, help="Pausa en segundos entre rondas")
    parser.add_argument("--scenario", type=str, default="iid", help="Etiqueta del escenario (iid, non_iid, disconnect)")
    parser.add_argument("--output", type=str, default=None, help="Ruta JSON para guardar resultados")
    parser.add_argument("--exit-when-done", action="store_true", help="Cerrar el servidor al terminar las rondas")
    args = parser.parse_args()

    coordinator = FederatedCoordinator(
        total_rounds=args.rounds,
        expected_clients=args.expected_clients,
        min_clients=args.min_clients,
        round_timeout=args.round_timeout,
        heartbeat_timeout=args.heartbeat_timeout,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        auto_start=True,
        round_pause=args.round_pause,
        scenario_name=args.scenario,
        output_json=args.output,
    )

    local_ip = get_local_ip()
    print("=" * 78)
    print(f" SERVIDOR DE APRENDIZAJE FEDERADO INICIADO — ESCENARIO: {args.scenario.upper()}")
    print(f" Escuchando en red local: http://{local_ip}:{args.port}")
    print(f" Dashboard en vivo:       http://localhost:{args.port}")
    print(f" Rondas: {args.rounds} | Clientes esperados: {args.expected_clients} | Timeout/ronda: {args.round_timeout}s")
    print("=" * 78, flush=True)

    monitor_thread = threading.Thread(target=coordinator.monitor_loop, daemon=True)
    monitor_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(coordinator))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        while True:
            time.sleep(0.5)
            if args.exit_when_done and coordinator.state == "FINISHED":
                time.sleep(1.5)
                break
    except KeyboardInterrupt:
        print("\nCerrando servidor federado...")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
