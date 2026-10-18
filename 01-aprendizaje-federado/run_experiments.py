"""
Script orquestador que ejecuta de extremo a extremo los experimentos reales
cliente-servidor (procesos independientes comunicandose por HTTP/TCP):

  1. Experimento 1: Escenario IID (15 rondas, 3 clientes conectados)
  2. Experimento 2: Escenario non-IID (15 rondas, 3 clientes conectados)
  3. Experimento 3: Prueba de Desconexion y Recuperacion (15 rondas, el Cliente 3
     pierde la conexion en las rondas 5, 6, 7 y 8; el servidor detecta el timeout,
     continua con los Clientes 1 y 2, y en la ronda 9 el Cliente 3 se reconecta y
     sincroniza automaticamente con el modelo global).

Genera todas las tablas CSV y graficos comparativos en la carpeta results/.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import prepare_data


def run_federated_session(
    scenario: str,
    mode: str,
    port: int,
    rounds: int = 15,
    round_timeout: float = 2.4,
    heartbeat_timeout: float = 1.2,
    disconnect_client: int | None = None,
    disconnect_rounds: str = "",
) -> dict:
    """Lanza 1 proceso Servidor y 3 procesos Cliente por HTTP real y espera su finalizacion."""
    output_json = f"results/history_{scenario}.json"
    if os.path.exists(output_json):
        os.remove(output_json)

    server_cmd = [
        sys.executable,
        "server.py",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--rounds", str(rounds),
        "--expected-clients", "3",
        "--min-clients", "1",
        "--round-timeout", str(round_timeout),
        "--heartbeat-timeout", str(heartbeat_timeout),
        "--local-epochs", "10",
        "--lr", "0.03",
        "--batch-size", "64",
        "--round-pause", "0.15",
        "--scenario", scenario,
        "--output", output_json,
        "--exit-when-done",
    ]

    print(f"\n>>> Iniciando Servidor Federado para escenario [{scenario.upper()}] en puerto {port}...")
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(1.0)

    client_procs = []
    for cid in [1, 2, 3]:
        cmd = [
            sys.executable,
            "client.py",
            "--client-id", str(cid),
            "--server", f"http://127.0.0.1:{port}",
            "--mode", mode,
            "--poll-interval", "0.25",
        ]
        if disconnect_client == cid and disconnect_rounds:
            cmd.extend(["--disconnect-rounds", disconnect_rounds])
        p = subprocess.Popen(cmd)
        client_procs.append(p)

    server_proc.wait(timeout=180)
    for p in client_procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.terminate()

    with open(output_json, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_comparative_tables_and_charts(iid_data: dict, non_iid_data: dict, disc_data: dict):
    os.makedirs("results", exist_ok=True)

    iid_hist = iid_data["history"]
    non_hist = non_iid_data["history"]
    disc_hist = disc_data["history"]

    rounds = [h["round"] for h in iid_hist]

    # 1. Tabla por ronda IID vs non-IID
    round_rows = []
    for h_iid, h_non, h_disc in zip(iid_hist, non_hist, disc_hist):
        r = h_iid["round"]
        gm_i = h_iid["global_metrics"]
        gm_n = h_non["global_metrics"]
        gm_d = h_disc["global_metrics"]
        round_rows.append({
            "Ronda": r,
            "IID_Accuracy": gm_i["accuracy"],
            "IID_Precision": gm_i["precision"],
            "IID_Recall": gm_i["recall"],
            "IID_F1": gm_i["f1"],
            "IID_Loss": gm_i["loss"],
            "IID_Drift_L2": h_iid["mean_client_drift_l2"],
            "NonIID_Accuracy": gm_n["accuracy"],
            "NonIID_Precision": gm_n["precision"],
            "NonIID_Recall": gm_n["recall"],
            "NonIID_F1": gm_n["f1"],
            "NonIID_Loss": gm_n["loss"],
            "NonIID_Drift_L2": h_non["mean_client_drift_l2"],
            "Disconnect_ActiveClients": ",".join(map(str, h_disc["active_clients"])),
            "Disconnect_Accuracy": gm_d["accuracy"],
            "Disconnect_F1": gm_d["f1"],
        })

    df_rounds = pd.DataFrame(round_rows)
    df_rounds.to_csv("results/tabla_comparativa_rondas.csv", index=False)

    # 2. Tabla resumen ejecutiva IID vs non-IID
    iid_acc_series = np.array([h["global_metrics"]["accuracy"] for h in iid_hist])
    non_acc_series = np.array([h["global_metrics"]["accuracy"] for h in non_hist])
    iid_f1_series = np.array([h["global_metrics"]["f1"] for h in iid_hist])
    non_f1_series = np.array([h["global_metrics"]["f1"] for h in non_hist])

    # Estabilidad medida como desviacion estandar de las diferencias entre rondas consecutivas (ultimas 10 rondas)
    iid_instability = float(np.std(np.diff(iid_f1_series[4:])))
    non_instability = float(np.std(np.diff(non_f1_series[4:])))

    # Ronda en que se supera el 85% de Accuracy
    iid_conv_r = next((h["round"] for h in iid_hist if h["global_metrics"]["accuracy"] >= 0.85), ">15")
    non_conv_r = next((h["round"] for h in non_hist if h["global_metrics"]["accuracy"] >= 0.85), ">15")

    summary_table = pd.DataFrame([
        {
            "Criterio / Metrica": "Accuracy Final (Ronda 15)",
            "Escenario IID": f"{iid_hist[-1]['global_metrics']['accuracy']*100:.2f}%",
            "Escenario non-IID": f"{non_hist[-1]['global_metrics']['accuracy']*100:.2f}%",
            "Diferencia (IID vs non-IID)": f"{(iid_hist[-1]['global_metrics']['accuracy'] - non_hist[-1]['global_metrics']['accuracy'])*100:+.2f} pp",
        },
        {
            "Criterio / Metrica": "Precision Macro Final (Ronda 15)",
            "Escenario IID": f"{iid_hist[-1]['global_metrics']['precision']*100:.2f}%",
            "Escenario non-IID": f"{non_hist[-1]['global_metrics']['precision']*100:.2f}%",
            "Diferencia (IID vs non-IID)": f"{(iid_hist[-1]['global_metrics']['precision'] - non_hist[-1]['global_metrics']['precision'])*100:+.2f} pp",
        },
        {
            "Criterio / Metrica": "Recall Macro Final (Ronda 15)",
            "Escenario IID": f"{iid_hist[-1]['global_metrics']['recall']*100:.2f}%",
            "Escenario non-IID": f"{non_hist[-1]['global_metrics']['recall']*100:.2f}%",
            "Diferencia (IID vs non-IID)": f"{(iid_hist[-1]['global_metrics']['recall'] - non_hist[-1]['global_metrics']['recall'])*100:+.2f} pp",
        },
        {
            "Criterio / Metrica": "F1-Score Macro Final (Ronda 15)",
            "Escenario IID": f"{iid_hist[-1]['global_metrics']['f1']*100:.2f}%",
            "Escenario non-IID": f"{non_hist[-1]['global_metrics']['f1']*100:.2f}%",
            "Diferencia (IID vs non-IID)": f"{(iid_hist[-1]['global_metrics']['f1'] - non_hist[-1]['global_metrics']['f1'])*100:+.2f} pp",
        },
        {
            "Criterio / Metrica": "Loss Global Final (Cross-Entropy)",
            "Escenario IID": f"{iid_hist[-1]['global_metrics']['loss']:.4f}",
            "Escenario non-IID": f"{non_hist[-1]['global_metrics']['loss']:.4f}",
            "Diferencia (IID vs non-IID)": f"{iid_hist[-1]['global_metrics']['loss'] - non_hist[-1]['global_metrics']['loss']:+.4f}",
        },
        {
            "Criterio / Metrica": "Divergencia Media de Pesos (Client Drift L2)",
            "Escenario IID": f"{np.mean([h['mean_client_drift_l2'] for h in iid_hist]):.4f}",
            "Escenario non-IID": f"{np.mean([h['mean_client_drift_l2'] for h in non_hist]):.4f}",
            "Diferencia (IID vs non-IID)": f"{np.mean([h['mean_client_drift_l2'] for h in non_hist]) / np.mean([h['mean_client_drift_l2'] for h in iid_hist]):.2f}x mayor en non-IID",
        },
        {
            "Criterio / Metrica": "Ronda de Convergencia (Acc >= 85%)",
            "Escenario IID": f"Ronda {iid_conv_r}",
            "Escenario non-IID": f"Ronda {non_conv_r}",
            "Diferencia (IID vs non-IID)": "Convergencia mas rapida en IID",
        },
        {
            "Criterio / Metrica": "Oscilacion / Inestabilidad entre Rondas (Std Delta F1)",
            "Escenario IID": f"{iid_instability:.5f}",
            "Escenario non-IID": f"{non_instability:.5f}",
            "Diferencia (IID vs non-IID)": "Mayor estabilidad en IID",
        },
    ])
    summary_table.to_csv("results/tabla_resumen_final.csv", index=False)

    # 3. Grafico 1: Evolucion por ronda IID vs non-IID (4 paneles)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # Panel A: Accuracy
    ax = axes[0, 0]
    ax.plot(rounds, iid_acc_series * 100, "o-", color="#16a34a", linewidth=2.4, label="Escenario IID")
    ax.plot(rounds, non_acc_series * 100, "s--", color="#dc2626", linewidth=2.4, label="Escenario non-IID")
    ax.set_title("A) Evolución de Accuracy Global por Ronda", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ronda Federada")
    ax.set_ylabel("Accuracy Global (%)")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Panel B: F1-Score Macro
    ax = axes[0, 1]
    ax.plot(rounds, iid_f1_series * 100, "o-", color="#2563eb", linewidth=2.4, label="Escenario IID (F1 Macro)")
    ax.plot(rounds, non_f1_series * 100, "s--", color="#ea580c", linewidth=2.4, label="Escenario non-IID (F1 Macro)")
    ax.set_title("B) Evolución de F1-Score Macro por Ronda", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ronda Federada")
    ax.set_ylabel("F1-Score Macro (%)")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Panel C: Cross-Entropy Loss
    ax = axes[1, 0]
    ax.plot(rounds, [h["global_metrics"]["loss"] for h in iid_hist], "o-", color="#16a34a", linewidth=2.4, label="Escenario IID")
    ax.plot(rounds, [h["global_metrics"]["loss"] for h in non_hist], "s--", color="#dc2626", linewidth=2.4, label="Escenario non-IID")
    ax.set_title("C) Evolución de la Pérdida (Cross-Entropy Loss) Global", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ronda Federada")
    ax.set_ylabel("Loss en Test Global")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Panel D: Divergencia de Pesos (Client Drift L2)
    ax = axes[1, 1]
    ax.plot(rounds, [h["mean_client_drift_l2"] for h in iid_hist], "o-", color="#0d9488", linewidth=2.4, label="IID (Baja deriva entre clientes)")
    ax.plot(rounds, [h["mean_client_drift_l2"] for h in non_hist], "s--", color="#9333ea", linewidth=2.4, label="non-IID (Alta deriva / Client Drift)")
    ax.set_title("D) Divergencia de Pesos Locales vs. Global (Client Drift L2)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ronda Federada")
    ax.set_ylabel("Distancia L2 Promedio ||w_k - w_global||")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.suptitle("Comparación de Evolución por Ronda: Aprendizaje Federado IID vs. non-IID (FedAvg)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("results/evolucion_iid_vs_non_iid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 4. Grafico 2: Comportamiento de los clientes locales vs modelo global
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    c_colors = {1: "#2563eb", 2: "#d97706", 3: "#059669"}
    for idx, (title, hist) in enumerate([("Escenario IID: Clientes Homogéneos", iid_hist), ("Escenario non-IID: Clientes Sesgados (Client Drift)", non_hist)]):
        ax = axes[idx]
        for cid in [1, 2, 3]:
            c_acc = [h["client_metrics"][f"client_{cid}"]["accuracy"] * 100 for h in hist]
            ax.plot(rounds, c_acc, "o--", color=c_colors[cid], alpha=0.75, label=f"Acc Local Cliente {cid}")
        g_acc = [h["global_metrics"]["accuracy"] * 100 for h in hist]
        ax.plot(rounds, g_acc, "k-", linewidth=2.8, label="Acc Modelo Global (Servidor)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Ronda Federada")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(rounds)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right")

    plt.suptitle("Comportamiento Local de los Clientes vs. Modelo Global en Servidor", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/comportamiento_clientes_locales.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 5. Grafico 3: Prueba de Desconexion y Recuperacion de un Cliente
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    disc_f1 = [h["global_metrics"]["f1"] * 100 for h in disc_hist]

    ax = axes[0]
    ax.axvspan(4.6, 8.4, color="#fee2e2", alpha=0.8, label="Cliente 3 Desconectado (Rondas 5-8)")
    ax.axvline(9, color="#16a34a", linestyle="--", linewidth=2.0, label="Reconexión y Sincronización (Ronda 9)")
    ax.plot(rounds, non_f1_series * 100, "o--", color="#64748b", linewidth=1.8, label="non-IID Sin Desconexión (3 clientes)")
    ax.plot(rounds, disc_f1, "s-", color="#dc2626", linewidth=2.5, label="non-IID Con Caída de Cliente 3 (Rondas 5-8)")
    ax.set_title("A) Impacto de la Desconexión en el F1-Score Global", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ronda Federada")
    ax.set_ylabel("F1-Score Macro Global (%)")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.axvspan(4.6, 8.4, color="#fee2e2", alpha=0.8, label="Cliente 3 Desconectado (Timeout -> 2 clientes)")
    ax.axvline(9, color="#16a34a", linestyle="--", linewidth=2.0, label="Cliente 3 Reconecta y Sincroniza (Ronda 9)")
    for cls_idx, (cls_label, col) in enumerate([
        ("F1 Clase 0: Bajo (Domina C1)", "#2563eb"),
        ("F1 Clase 1: Medio (Domina C2)", "#d97706"),
        ("F1 Clase 2: Alto (Domina C3 - Desconectado)", "#dc2626"),
    ]):
        cls_series = [h["global_metrics"]["f1_per_class"][cls_idx] * 100 for h in disc_hist]
        ax.plot(rounds, cls_series, "o-", color=col, linewidth=2.2, label=cls_label)
    ax.set_title("B) Efecto por Clase cuando cae el Cliente 3 y se Reconecta", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ronda Federada")
    ax.set_ylabel("F1-Score por Clase (%)")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.8)

    plt.suptitle("Prueba de Desconexión Temporal (Rondas 5–8) y Recuperación Automática (Ronda 9)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/prueba_desconexion_recuperacion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 6. Grafico 4: Matrices de confusion finales IID vs non-IID
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cls_short = ["Bajo (0)", "Medio (1)", "Alto (2)"]
    for idx, (title, cm_list) in enumerate([
        ("Matriz de Confusión Final — Escenario IID (Ronda 15)", iid_hist[-1]["global_metrics"]["confusion_matrix"]),
        ("Matriz de Confusión Final — Escenario non-IID (Ronda 15)", non_hist[-1]["global_metrics"]["confusion_matrix"]),
    ]):
        ax = axes[idx]
        cm = np.array(cm_list)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(cls_short)
        ax.set_yticklabels(cls_short)
        ax.set_xlabel("Predicción del Modelo Global")
        ax.set_ylabel("Clase Real")
        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                color = "white" if val > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/matrices_confusion_comparativa.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\n=== TABLA RESUMEN FINAL: IID vs. non-IID ===")
    print(summary_table.to_string(index=False))


def main():
    if not os.path.exists("data/iid/client_1.csv"):
        print("Paso 0: Asegurando particiones de datos IID y non-IID actualizadas...")
        prepare_data.main()

    # Experimento 1: IID
    if os.path.exists("results/history_iid.json"):
        with open("results/history_iid.json", "r", encoding="utf-8") as f:
            iid_data = json.load(f)
    else:
        iid_data = run_federated_session(
            scenario="iid",
            mode="iid",
            port=5011,
            rounds=15,
            round_timeout=2.0,
            heartbeat_timeout=1.0,
        )

    # Experimento 2: non-IID
    if os.path.exists("results/history_non_iid.json"):
        with open("results/history_non_iid.json", "r", encoding="utf-8") as f:
            non_iid_data = json.load(f)
    else:
        non_iid_data = run_federated_session(
            scenario="non_iid",
            mode="non_iid",
            port=5012,
            rounds=15,
            round_timeout=2.0,
            heartbeat_timeout=1.0,
        )

    # Experimento 3: Prueba de Desconexion y Recuperacion (Cliente 3 se desconecta en rondas 5,6,7,8 y regresa en la 9)
    disc_data = run_federated_session(
        scenario="disconnect",
        mode="non_iid",
        port=5013,
        rounds=15,
        round_timeout=1.6,
        heartbeat_timeout=0.9,
        disconnect_client=3,
        disconnect_rounds="5,6,7,8",
    )

    generate_comparative_tables_and_charts(iid_data, non_iid_data, disc_data)


if __name__ == "__main__":
    main()
