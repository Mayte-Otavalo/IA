"""
Orquestador Automatizado de Experimentos (Escenarios A, B, C, D y E)
File Systems & Storage — Distributed Systems Evaluation (PC / Container <-> Raspberry Pi).

Implementa el Protocolo Experimental (Diapositivas 4, 5, 6, 7, 8 y 10):
- Escenario A: Tamaño de archivo (small, medium, large), CSV vs. Parquet y patrones de acceso (full, columns, range, write).
- Escenario B: Comparación Local Baseline (PC/Contenedor) vs. Remoto (Raspberry Pi).
- Escenario C: Comparación de transporte TCP Sockets vs. gRPC.
- Escenario D: Clientes concurrentes (1, 2, 4, 8 peticiones simultáneas) midiendo latencia y uso de CPU/RAM.
- Escenario E: Fallo del servicio en la Raspberry Pi (No retry vs. Timeout + retry), midiendo detección y recuperación.
- Calcula mediana (p50), percentil 95 (p95), desviación estándar y exporta evidencias CSV y gráficas.
"""

import os
import sys
import time
import json
import platform
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from prepare_datasets import prepare_all_datasets, STORAGE_DIR
from server.file_server import DistributedFileService, TCPFileServer, GRPCFileServer, HAS_GRPC
from client.file_client import DistributedStorageClient

RESULTS_DIR = os.path.join(BASE_DIR, "results")


def record_environment_metadata(output_dir: str, pi_host: str, runs_per_setup: int):
    """Guarda versiones de Python, SO y parámetros experimentales (Diapositiva 8: Reproducibility)."""
    os.makedirs(output_dir, exist_ok=True)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "grpc_available": HAS_GRPC,
        "target_pi_host": pi_host,
        "runs_per_setup": runs_per_setup
    }
    with open(os.path.join(output_dir, "stack_environment.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def run_all_scenarios(
    pi_host: str = "127.0.0.1",
    tcp_port: int = 5000,
    grpc_port: int = 50051,
    runs: int = 10,
    local_demo: bool = False
) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prepare_all_datasets(STORAGE_DIR)
    record_environment_metadata(RESULTS_DIR, pi_host, runs)

    tcp_server = None
    grpc_server = None
    if local_demo:
        print("\n[Modo Demo Local] Iniciando File/I/O Service en segundo plano (127.0.0.1)...")
        service = DistributedFileService(storage_dir=STORAGE_DIR)
        tcp_server = TCPFileServer(service, host="127.0.0.1", port=tcp_port)
        grpc_server = GRPCFileServer(service, host="127.0.0.1", port=grpc_port)
        tcp_server.start_background()
        grpc_server.start_background()
        time.sleep(0.8)

    client = DistributedStorageClient(
        pi_host=pi_host,
        tcp_port=tcp_port,
        grpc_port=grpc_port,
        timeout_sec=4.0,
        local_storage_dir=STORAGE_DIR
    )

    all_records: List[Dict[str, Any]] = []

    try:
        # =========================================================================
        # ESCENARIO A: Tamaño de archivo, CSV vs. Parquet y Patrones de Acceso
        # =========================================================================
        print(f"\n>>> Ejecutando ESCENARIO A: CSV vs. Parquet, Tamaños y Patrones de Acceso ({runs} runs c/u)...")
        for size_label in ["small", "medium", "large"]:
            for fmt in ["csv", "parquet"]:
                for op in ["read_full", "read_columns", "read_range", "write_file"]:
                    req = {
                        "operation": op,
                        "size_label": size_label,
                        "file_format": fmt,
                        "columns": ["CO(GT)", "T", "RH"],
                        "start_row": 100,
                        "end_row": 600
                    }
                    for r in range(1, runs + 1):
                        res = client.execute_remote(req, transport="tcp", retry_enabled=True)
                        res["scenario"] = "A_Storage_And_Access"
                        res["concurrency"] = 1
                        res["retry_policy"] = "timeout_retry"
                        res["run_id"] = r
                        all_records.append(res)

        # =========================================================================
        # ESCENARIO B: Local Baseline (PC/Contenedor) vs. Remoto (Raspberry Pi)
        # =========================================================================
        print(f">>> Ejecutando ESCENARIO B: Local Baseline vs. Remoto Raspberry Pi ({runs} runs c/u)...")
        for size_label in ["small", "medium", "large"]:
            for fmt in ["csv", "parquet"]:
                req = {"operation": "read_full", "size_label": size_label, "file_format": fmt}
                for r in range(1, runs + 1):
                    # 1. Local Baseline
                    res_local = client.execute_local_baseline(req)
                    res_local["scenario"] = "B_Local_vs_Remote"
                    res_local["concurrency"] = 1
                    res_local["retry_policy"] = "none"
                    res_local["run_id"] = r
                    all_records.append(res_local)

                    # 2. Remote Pi Access
                    res_rem = client.execute_remote(req, transport="tcp", retry_enabled=True)
                    res_rem["scenario"] = "B_Local_vs_Remote"
                    res_rem["concurrency"] = 1
                    res_rem["retry_policy"] = "timeout_retry"
                    res_rem["run_id"] = r
                    all_records.append(res_rem)

        # =========================================================================
        # ESCENARIO C: Transporte TCP Sockets vs. gRPC
        # =========================================================================
        print(f">>> Ejecutando ESCENARIO C: TCP Sockets vs. gRPC ({runs} runs c/u)...")
        transports = ["tcp", "grpc"] if HAS_GRPC else ["tcp"]
        if not HAS_GRPC:
            print("    [Aviso] grpcio no detectado; ejecutando TCP (instale 'pip install grpcio' para incluir gRPC).")
        for size_label in ["small", "medium", "large"]:
            for fmt in ["csv", "parquet"]:
                req = {"operation": "read_full", "size_label": size_label, "file_format": fmt}
                for tr in transports:
                    for r in range(1, runs + 1):
                        res = client.execute_remote(req, transport=tr, retry_enabled=True)
                        res["scenario"] = "C_TCP_vs_gRPC"
                        res["concurrency"] = 1
                        res["retry_policy"] = "timeout_retry"
                        res["run_id"] = r
                        all_records.append(res)

        # =========================================================================
        # ESCENARIO D: Clientes Concurrentes (1, 2, 4, 8 hilos simultáneos)
        # =========================================================================
        print(f">>> Ejecutando ESCENARIO D: Clientes Concurrentes (1, 2, 4, 8 clientes)...")
        for num_clients in [1, 2, 4, 8]:
            for fmt in ["csv", "parquet"]:
                req = {"operation": "read_full", "size_label": "medium", "file_format": fmt}

                def _worker_task(run_idx: int):
                    c_worker = DistributedStorageClient(
                        pi_host=pi_host, tcp_port=tcp_port, grpc_port=grpc_port,
                        timeout_sec=6.0, local_storage_dir=STORAGE_DIR
                    )
                    out = c_worker.execute_remote(req, transport="tcp", retry_enabled=True)
                    out["scenario"] = "D_Concurrency"
                    out["concurrency"] = num_clients
                    out["retry_policy"] = "timeout_retry"
                    out["run_id"] = run_idx
                    return out

                with ThreadPoolExecutor(max_workers=num_clients) as pool:
                    futures_list = [pool.submit(_worker_task, i + 1) for i in range(num_clients * 2)]
                    for fut in as_completed(futures_list):
                        all_records.append(fut.result())

        # =========================================================================
        # ESCENARIO E: Fallo del Servicio en la Raspberry Pi (No Retry vs. Retry)
        # =========================================================================
        print(f">>> Ejecutando ESCENARIO E: Fallo del Servicio (No retry vs. Timeout + retry)...")
        req_read = {"operation": "read_full", "size_label": "small", "file_format": "parquet"}

        # E1: Sin reintento (No retry) durante caída de 1.5s
        client.execute_remote({"operation": "simulate_failure", "duration_sec": 1.5}, transport="tcp")
        res_no_retry = client.execute_remote(req_read, transport="tcp", retry_enabled=False)
        res_no_retry["scenario"] = "E_Fault_Tolerance"
        res_no_retry["concurrency"] = 1
        res_no_retry["retry_policy"] = "no_retry"
        res_no_retry["run_id"] = 1
        all_records.append(res_no_retry)

        # Restaurar servicio antes de la prueba con reintentos
        client.execute_remote({"operation": "recover_service"}, transport="tcp")

        # E2: Con Timeout + Retry durante caída de 1.8s (se recupera automáticamente al 3er intento)
        client.execute_remote({"operation": "simulate_failure", "duration_sec": 1.8}, transport="tcp")
        res_with_retry = client.execute_remote(
            req_read, transport="tcp", retry_enabled=True, max_retries=6, retry_delay_sec=0.7
        )
        res_with_retry["scenario"] = "E_Fault_Tolerance"
        res_with_retry["concurrency"] = 1
        res_with_retry["retry_policy"] = "timeout_retry"
        res_with_retry["run_id"] = 1
        all_records.append(res_with_retry)

    finally:
        if tcp_server:
            tcp_server.stop()
        if grpc_server:
            grpc_server.stop()

    df_runs = pd.DataFrame(all_records)
    per_run_csv = os.path.join(RESULTS_DIR, "per_run_metrics.csv")
    df_runs.to_csv(per_run_csv, index=False)

    summary_df = build_summary_table(df_runs)
    summary_csv = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)

    print_executive_report(summary_df, df_runs, per_run_csv, summary_csv)
    generate_plots(summary_df, RESULTS_DIR)
    return summary_df


def build_summary_table(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Calcula mediana (p50), p95, desviación estándar y promedios de recursos por configuración."""
    group_cols = ["scenario", "mode", "transport", "operation", "size_label", "file_format", "concurrency", "retry_policy"]
    rows = []

    for keys, grp in df_runs.groupby(group_cols, dropna=False):
        wall = grp["total_wall_time_ms"].values
        disk = grp["disk_io_time_ms"].values
        ser = grp["serialization_time_ms"].values
        net = grp["network_overhead_ms"].values
        tput = grp["throughput_mb_s"].values

        row = dict(zip(group_cols, keys))
        row.update({
            "runs": len(grp),
            "success_rate": round((grp["status"] == "success").mean() * 100.0, 1),
            "p50_latency_ms": round(float(np.percentile(wall, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(wall, 95)), 3),
            "std_latency_ms": round(float(np.std(wall)), 3),
            "median_disk_io_ms": round(float(np.median(disk)), 3),
            "median_serialization_ms": round(float(np.median(ser)), 3),
            "median_network_ms": round(float(np.median(net)), 3),
            "median_throughput_mb_s": round(float(np.median(tput)), 3),
            "file_size_kb": round(float(grp["file_size_bytes"].max()) / 1024.0, 2),
            "bytes_tx_kb": round(float(grp["bytes_transmitted"].median()) / 1024.0, 2),
            "avg_pi_cpu_pct": round(float(grp["server_cpu_percent"].mean()), 2),
            "peak_pi_ram_mb": round(float(grp["server_ram_mb"].max()), 2),
            "total_failed_attempts": int(grp["failed_attempts"].sum()),
            "recovery_time_ms": round(float(grp["recovery_time_ms"].max()), 3),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def print_executive_report(summary_df: pd.DataFrame, df_runs: pd.DataFrame, per_run_csv: str, summary_csv: str):
    print("\n" + "=" * 95)
    print("  RESUMEN CUANTITATIVO DE RESULTADOS (p50, p95, Disco I/O vs Red, CPU/RAM y Fallos)")
    print("=" * 95)
    cols_to_show = [
        "scenario", "mode", "transport", "operation", "size_label",
        "file_format", "concurrency", "p50_latency_ms", "p95_latency_ms",
        "median_disk_io_ms", "median_network_ms", "file_size_kb"
    ]
    print(summary_df[cols_to_show].to_string(index=False))
    print("-" * 95)
    print(f"  [Evidencia 1] Logs por corrida guardados en : {per_run_csv}")
    print(f"  [Evidencia 2] Tabla resumen (p50/p95) en    : {summary_csv}")
    print("=" * 95 + "\n")


def generate_plots(summary_df: pd.DataFrame, output_dir: str):
    """Genera gráficas comparativas para la presentación si matplotlib está disponible."""
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. CSV vs Parquet (Escenario A - read_full)
        sub_a = summary_df[
            (summary_df["scenario"] == "A_Storage_And_Access") &
            (summary_df["operation"] == "read_full")
        ]
        if not sub_a.empty:
            piv_a = sub_a.pivot(index="size_label", columns="file_format", values="p50_latency_ms")
            piv_a.plot(kind="bar", ax=axes[0], title="Escenario A: Latencia p50 (CSV vs Parquet)")
            axes[0].set_ylabel("Latencia p50 (ms)")

        # 2. Local vs Remote (Escenario B)
        sub_b = summary_df[summary_df["scenario"] == "B_Local_vs_Remote"]
        if not sub_b.empty:
            piv_b = sub_b.pivot_table(index=["size_label", "file_format"], columns="mode", values="p50_latency_ms")
            piv_b.plot(kind="bar", ax=axes[1], title="Escenario B: Local Baseline vs Remoto Pi")
            axes[1].set_ylabel("Latencia p50 (ms)")

        # 3. Concurrencia (Escenario D)
        sub_d = summary_df[summary_df["scenario"] == "D_Concurrency"]
        if not sub_d.empty:
            piv_d = sub_d.pivot(index="concurrency", columns="file_format", values="p95_latency_ms")
            piv_d.plot(kind="line", marker="o", ax=axes[2], title="Escenario D: Concurrencia vs Latencia p95")
            axes[2].set_ylabel("Latencia p95 (ms)")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "evaluation_plots.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  [Evidencia 3] Gráficas generadas en         : {plot_path}\n")
    except Exception as e:
        print(f"  [Nota] No se generaron gráficas opcionales ({e}). Los CSV están listos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutor de Experimentos A-E (File Systems & Storage)")
    parser.add_argument("--pi-host", default="127.0.0.1", help="IP de la Raspberry Pi (default: 127.0.0.1)")
    parser.add_argument("--tcp-port", type=int, default=5000, help="Puerto TCP en la Raspberry Pi")
    parser.add_argument("--grpc-port", type=int, default=50051, help="Puerto gRPC en la Raspberry Pi")
    parser.add_argument("--runs", type=int, default=10, help="Número de repeticiones por prueba (default: 10)")
    parser.add_argument("--local-demo", action="store_true", help="Levantar el servidor localmente para probar en 1 sola PC")
    args = parser.parse_args()

    run_all_scenarios(
        pi_host=args.pi_host,
        tcp_port=args.tcp_port,
        grpc_port=args.grpc_port,
        runs=args.runs,
        local_demo=args.local_demo
    )
