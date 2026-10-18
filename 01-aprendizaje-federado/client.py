"""
Cliente de Aprendizaje Federado (Computadoras 2, 3 y 4).

Responsabilidades:
  1. Cargar exclusivamente su particion local de datos (data/{iid|non_iid}/client_{id}.csv).
     Los datos locales NUNCA se envian al servidor.
  2. Registrarse ante el Servidor Federado por HTTP en la red local.
  3. Recibir los pesos del modelo global MLP (W1, b1, W2, b2, W3, b3) en cada ronda.
  4. Entrenar localmente el modelo MLP sobre sus datos privados y evaluar metricas locales.
  5. Enviar unicamente los pesos entrenados y el tamano de muestra (n_k) al servidor.
  6. Tolerar desconexiones de red (ej. cuando el profesor desactiva la tarjeta de red):
     detecta la caida de red sin cerrarse, reintenta automaticamente y, al volver la
     conexion, sincroniza el modelo global de la ronda actual del servidor.
"""

import time
import argparse
from datetime import datetime

import pandas as pd
import requests

from federated_model import FederatedMLP
from prepare_data import FEATURE_COLS


def log_client(client_id: int, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} [Cliente {client_id}] {msg}", flush=True)


def load_local_dataset(client_id: int, mode: str):
    path = f"data/{mode}/client_{client_id}.csv"
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)
    counts = {int(k): int(v) for k, v in df["target"].value_counts().sort_index().items()}
    return X, y, counts, path


def main():
    parser = argparse.ArgumentParser(description="Cliente de Aprendizaje Federado (Air Quality)")
    parser.add_argument("--client-id", type=int, required=True, choices=[1, 2, 3], help="ID del cliente (1, 2 o 3)")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:5000", help="URL del Servidor Federado (ej. http://192.168.1.10:5000)")
    parser.add_argument("--mode", type=str, default="iid", choices=["iid", "non_iid"], help="Distribucion de datos locales (iid o non_iid)")
    parser.add_argument("--poll-interval", type=float, default=0.8, help="Segundos entre heartbeats al servidor")
    parser.add_argument(
        "--disconnect-rounds",
        type=str,
        default="",
        help="Lista opcional de rondas separadas por coma en las que este cliente simula desconexion de red (ej. '5,6,7')",
    )
    args = parser.parse_args()

    cid = args.client_id
    server_url = args.server.rstrip("/")
    disconnect_rounds = {
        int(x.strip()) for x in args.disconnect_rounds.split(",") if x.strip().isdigit()
    }

    X_local, y_local, class_dist, csv_path = load_local_dataset(cid, args.mode)
    num_samples = len(X_local)
    local_model = FederatedMLP(seed=42)

    print("=" * 72)
    print(f" CLIENTE FEDERADO {cid} — MODO: {args.mode.upper()}")
    print(f" Servidor objetivo: {server_url}")
    print(f" Datos privados:    {csv_path} (n={num_samples} muestras, Clases={class_dist})")
    if disconnect_rounds:
        print(f" Simulacion de desconexion programada en rondas: {sorted(disconnect_rounds)}")
    print("=" * 72, flush=True)

    # 1. Registro inicial con reintentos ante fallos de red
    registered = False
    while not registered:
        try:
            resp = requests.post(
                f"{server_url}/register",
                json={
                    "client_id": cid,
                    "num_samples": num_samples,
                    "class_dist": class_dist,
                },
                timeout=3.0,
            )
            resp.raise_for_status()
            registered = True
            log_client(cid, "Registrado exitosamente en el Servidor Federado.")
        except requests.RequestException as exc:
            log_client(cid, f"Esperando conexion con el servidor en {server_url}... ({exc.__class__.__name__})")
            time.sleep(1.5)

    last_trained_round = 0
    was_offline = False
    simulated_drop_done = set()

    # 2. Bucle principal de participacion federada y recuperacion de conexion
    try:
        while True:
            expected_next_round = last_trained_round + 1
            if expected_next_round in disconnect_rounds and expected_next_round not in simulated_drop_done:
                if not was_offline:
                    log_client(
                        cid,
                        f"[PRUEBA DE DESCONEXION] Tarjeta de red desactivada antes de Ronda {expected_next_round}. "
                        f"Cero paquetes enviados al servidor durante las rondas {sorted(disconnect_rounds)}...",
                    )
                    was_offline = True
                # Calcular cuantas rondas consecutivas se desconecta y esperar sin enviar paquetes HTTP
                consecutive = 0
                r_check = expected_next_round
                while r_check in disconnect_rounds:
                    simulated_drop_done.add(r_check)
                    consecutive += 1
                    r_check += 1
                # Dormir lo suficiente para que el servidor dispare timeout en cada una de esas rondas
                time.sleep(consecutive * 1.75 + 0.25)
                last_trained_round = r_check - 1
                continue

            try:
                resp = requests.post(
                    f"{server_url}/heartbeat",
                    json={"client_id": cid},
                    timeout=3.0,
                )
                resp.raise_for_status()
                directive = resp.json()
            except requests.RequestException:
                if not was_offline:
                    log_client(
                        cid,
                        "[ALERTA RED] CONEXION PERDIDA con el servidor (Tarjeta de red desconectada o timeout). "
                        "Modo espera activo: reintentando reconexion automaticamente...",
                    )
                    was_offline = True
                time.sleep(1.5)
                continue

            current_round = int(directive.get("current_round", 0))
            state = directive.get("state", "WAITING_FOR_CLIENTS")

            if state == "FINISHED":
                log_client(cid, f"El servidor indico que las {directive.get('total_rounds')} rondas finalizaron. Cerrando cliente.")
                break

            if was_offline:
                log_client(
                    cid,
                    f"[RECONEXION Y SINCRONIZACION] Conexion restablecida con el servidor en la Ronda {current_round}. "
                    f"Sincronizando modelo global mas reciente...",
                )
                was_offline = False

            if state == "TRAINING_ROUND" and directive.get("need_train") and current_round > last_trained_round:
                global_weights = directive.get("global_weights")
                if global_weights is not None:
                    # Sincronizar el modelo local con los pesos globales de la ronda actual
                    local_model.set_weights(global_weights)
                    pre_eval = local_model.evaluate(X_local, y_local)

                    local_epochs = int(directive.get("local_epochs", 8))
                    lr = float(directive.get("lr", 0.025))
                    batch_size = int(directive.get("batch_size", 64))

                    # Entrenar localmente sobre los datos privados del cliente
                    post_metrics = local_model.train_local(
                        X_local,
                        y_local,
                        epochs=local_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        seed=42 + current_round * 10 + cid,
                    )

                    log_client(
                        cid,
                        f"Ronda {current_round}/{directive['total_rounds']} entrenada localmente "
                        f"(Acc pre={pre_eval['accuracy']:.4f} -> post={post_metrics['accuracy']:.4f}, "
                        f"F1={post_metrics['f1']:.4f}, Loss={post_metrics['loss']:.4f}). Enviando pesos...",
                    )

                    # Enviar solo los pesos entrenados al servidor (nunca los datos)
                    sub_resp = requests.post(
                        f"{server_url}/submit_update",
                        json={
                            "client_id": cid,
                            "round": current_round,
                            "num_samples": num_samples,
                            "weights": local_model.get_weights(),
                            "metrics": post_metrics,
                        },
                        timeout=4.0,
                    )
                    sub_data = sub_resp.json()
                    if sub_data.get("accepted"):
                        last_trained_round = current_round
                    else:
                        log_client(
                            cid,
                            f"Servidor reporto ronda desfasada ({sub_data.get('reason')}). Re-sincronizando en siguiente pulso...",
                        )

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        log_client(cid, "Cliente detenido manualmente por el usuario.")


if __name__ == "__main__":
    main()
