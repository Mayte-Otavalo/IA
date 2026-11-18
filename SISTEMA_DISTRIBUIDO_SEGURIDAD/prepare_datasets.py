"""
Preparación de Datasets Reproducibles (CSV vs. Parquet) — File Systems & Storage Evaluation.
Cumple con Diapositivas 4 (Escenario A), 5 (Storage), 7 (Prepare data: Fixed-seed datasets; verify logical equivalence)
y 8 (Reproducibility: Fixed data / seeds).
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage_data")

FIXED_SEED = 42
DATASET_SIZES = {
    "small": 1000,
    "medium": 5000,
    "large": 20000
}

NUMERIC_COLS = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
    'T', 'RH', 'AH'
]


def load_base_dataframe() -> pd.DataFrame:
    """Reutiliza los datasets existentes de Calidad del Aire del proyecto."""
    candidate_paths = [
        os.path.join(BASE_DIR, "server", "global_test.csv"),
        os.path.join(BASE_DIR, "server", "global_val.csv"),
        os.path.join(BASE_DIR, "client", "local_data", "client_1", "iid_train.csv"),
        os.path.join(BASE_DIR, "client", "local_data", "client_2", "iid_train.csv"),
    ]
    dfs = []
    for p in candidate_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError("No se encontraron los archivos CSV base en server/ o client/local_data/")
    return pd.concat(dfs, ignore_index=True)


def generate_fixed_seed_dataset(base_df: pd.DataFrame, num_rows: int, seed: int = FIXED_SEED) -> pd.DataFrame:
    """Genera un dataset determinístico con semilla fija del tamaño exacto solicitado."""
    rng = np.random.default_rng(seed + num_rows)
    indices = rng.integers(0, len(base_df), size=num_rows)
    sampled = base_df.iloc[indices].reset_index(drop=True).copy()
    sampled["record_id"] = np.arange(num_rows, dtype=np.int64)
    # Ordenar columnas con record_id primero
    cols = ["record_id"] + [c for c in sampled.columns if c != "record_id"]
    return sampled[cols]


def verify_logical_equivalence(csv_path: str, parquet_path: str) -> bool:
    """Verifica que el archivo CSV y el archivo Parquet contengan exactamente los mismos datos lógicos."""
    df_csv = pd.read_csv(csv_path)
    df_pq = pd.read_parquet(parquet_path)

    if df_csv.shape != df_pq.shape:
        return False
    if list(df_csv.columns) != list(df_pq.columns):
        return False

    for col in NUMERIC_COLS:
        if col in df_csv.columns:
            if not np.allclose(df_csv[col].values, df_pq[col].values, rtol=1e-5, atol=1e-6):
                return False
    return True


def prepare_all_datasets(output_dir: str = STORAGE_DIR) -> dict:
    """Genera los datasets CSV y Parquet en tamaños fijos y registra métricas de almacenamiento."""
    os.makedirs(output_dir, exist_ok=True)
    base_df = load_base_dataframe()
    summary = {
        "seed": FIXED_SEED,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {}
    }

    print("=" * 78)
    print("  PREPARACIÓN DE DATASETS FIJOS (CSV vs. PARQUET) — SEMILLA FIJA = 42")
    print("=" * 78)
    print(f"  {'Etiqueta':<10} | {'Filas':<8} | {'CSV (KB)':<12} | {'Parquet (KB)':<14} | {'Ratio Comp.':<12} | {'Equivalencia'}")
    print("-" * 78)

    for label, rows in DATASET_SIZES.items():
        df = generate_fixed_seed_dataset(base_df, rows, seed=FIXED_SEED)
        csv_path = os.path.join(output_dir, f"dataset_{label}.csv")
        parquet_path = os.path.join(output_dir, f"dataset_{label}.parquet")

        # Medir escritura CSV
        t0 = time.perf_counter()
        df.to_csv(csv_path, index=False)
        csv_write_ms = (time.perf_counter() - t0) * 1000.0

        # Medir escritura Parquet
        t0 = time.perf_counter()
        df.to_parquet(parquet_path, index=False, compression="snappy")
        pq_write_ms = (time.perf_counter() - t0) * 1000.0

        csv_bytes = os.path.getsize(csv_path)
        pq_bytes = os.path.getsize(parquet_path)
        ratio = csv_bytes / pq_bytes if pq_bytes > 0 else 1.0
        is_equiv = verify_logical_equivalence(csv_path, parquet_path)

        summary["datasets"][label] = {
            "rows": rows,
            "columns": len(df.columns),
            "csv_path": csv_path,
            "parquet_path": parquet_path,
            "csv_size_bytes": csv_bytes,
            "parquet_size_bytes": pq_bytes,
            "compression_ratio_csv_over_parquet": round(ratio, 3),
            "initial_csv_write_ms": round(csv_write_ms, 3),
            "initial_parquet_write_ms": round(pq_write_ms, 3),
            "logical_equivalence_verified": is_equiv
        }

        print(
            f"  {label:<10} | {rows:<8} | {csv_bytes / 1024.0:<12.2f} | "
            f"{pq_bytes / 1024.0:<14.2f} | {ratio:<11.2f}x | {'OK' if is_equiv else 'ERROR'}"
        )

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print(f"  Datasets guardados en: {output_dir}")
    print(f"  Resumen de verificación: {summary_path}\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de datasets CSV y Parquet con semilla fija")
    parser.add_argument("--output-dir", default=STORAGE_DIR, help="Directorio de salida para los datasets")
    args = parser.parse_args()
    prepare_all_datasets(args.output_dir)
