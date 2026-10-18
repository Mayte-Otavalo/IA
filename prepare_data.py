"""
Preprocesamiento y particionamiento del dataset Air Quality (UCI / Kaggle)
para Aprendizaje Federado en escenarios IID y non-IID.

Dataset: https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set/data
Problema: Clasificacion multiclase del Nivel de Contaminacion por CO (3 clases):
  - Clase 0: Bajo  (CO(GT) <= 1.4 mg/m^3)
  - Clase 1: Medio (1.4 < CO(GT) <= 2.5 mg/m^3)
  - Clase 2: Alto  (CO(GT) > 2.5 mg/m^3)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_COLS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "Hour",
]

CLASS_NAMES = {
    0: "Bajo (<=1.4 mg/m³)",
    1: "Medio (1.4-2.5 mg/m³)",
    2: "Alto (>2.5 mg/m³)",
}


def load_and_clean_air_quality(csv_path: str = "data/AirQuality.csv") -> pd.DataFrame:
    """Carga AirQuality.csv, limpia valores faltantes (-200) y construye la variable objetivo."""
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # Extraer la hora (formato '18.00.00' -> 18)
    df["Hour"] = df["Time"].astype(str).str.slice(0, 2).astype(float)

    # NMHC(GT) tiene >90% de valores faltantes (-200), se descarta
    if "NMHC(GT)" in df.columns:
        df = df.drop(columns=["NMHC(GT)"])

    # Reemplazar -200 por NaN en las columnas relevantes
    cols_to_check = ["CO(GT)"] + FEATURE_COLS
    for col in cols_to_check:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[cols_to_check] = df[cols_to_check].replace(-200.0, np.nan)
    df = df.dropna(subset=cols_to_check).reset_index(drop=True)

    # Definir las 3 clases de calidad del aire basadas en terciles fisicos de CO(GT)
    # <= 1.4 mg/m3: Bajo (0), 1.4 - 2.5 mg/m3: Medio (1), > 2.5 mg/m3: Alto (2)
    conditions = [
        df["CO(GT)"] <= 1.4,
        (df["CO(GT)"] > 1.4) & (df["CO(GT)"] <= 2.5),
        df["CO(GT)"] > 2.5,
    ]
    df["target"] = np.select(conditions, [0, 1, 2]).astype(int)
    return df


def create_iid_splits(df_train: pd.DataFrame, num_clients: int = 3) -> list[pd.DataFrame]:
    """
    Divide los datos de entrenamiento de forma IID (Independiente e Identicamente Distribuida)
    usando muestreo estratificado equitativo entre los 3 clientes.
    """
    df_shuffled = df_train.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    clients_data = [[] for _ in range(num_clients)]

    for cls in sorted(df_shuffled["target"].unique()):
        cls_df = df_shuffled[df_shuffled["target"] == cls].reset_index(drop=True)
        n = len(cls_df)
        for i in range(num_clients):
            start_idx = (i * n) // num_clients
            end_idx = ((i + 1) * n) // num_clients
            clients_data[i].append(cls_df.iloc[start_idx:end_idx])

    return [
        pd.concat(client_chunks, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED + i).reset_index(drop=True)
        for i, client_chunks in enumerate(clients_data)
    ]


def create_non_iid_splits(df_train: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Divide los datos de entrenamiento de forma claramente non-IID (sesgo extremo de clases + sesgo ambiental):
      - Cliente 1 (Estacion Residencial / Baja contaminacion): ~78% Clase 0, ~16% Clase 1, ~6% Clase 2
      - Cliente 2 (Estacion Urbana / Contaminacion moderada):  ~12% Clase 0, ~76% Clase 1, ~12% Clase 2
      - Cliente 3 (Estacion Industrial / Alta contaminacion):  ~6% Clase 0, ~16% Clase 1, ~78% Clase 2
    """
    # Proporciones de cada clase que van a (Cliente 1, Cliente 2, Cliente 3)
    # Filas = Clases (0, 1, 2), Columnas = Clientes (1, 2, 3)
    allocation_matrix = np.array([
        [0.90, 0.07, 0.03],  # Clase 0 (Bajo) se concentra 90% en el Cliente 1
        [0.06, 0.88, 0.06],  # Clase 1 (Medio) se concentra 88% en el Cliente 2
        [0.03, 0.07, 0.90],  # Clase 2 (Alto) se concentra 90% en el Cliente 3
    ])

    clients_chunks = [[], [], []]
    for cls in [0, 1, 2]:
        # Ordenar ademas por Temperatura/Hora para acentuar el sesgo de dominio local
        cls_df = df_train[df_train["target"] == cls].sample(frac=1.0, random_state=RANDOM_SEED + cls * 10).reset_index(drop=True)
        n = len(cls_df)
        p1, p2, _ = allocation_matrix[cls]
        idx1 = int(round(n * p1))
        idx2 = idx1 + int(round(n * p2))

        clients_chunks[0].append(cls_df.iloc[:idx1])
        clients_chunks[1].append(cls_df.iloc[idx1:idx2])
        clients_chunks[2].append(cls_df.iloc[idx2:])

    return [
        pd.concat(chunks).sample(frac=1.0, random_state=RANDOM_SEED + 100 + i).reset_index(drop=True)
        for i, chunks in enumerate(clients_chunks)
    ]


def plot_distributions(iid_clients: list[pd.DataFrame], non_iid_clients: list[pd.DataFrame], out_dir: str = "results"):
    """Genera graficos y tabla comparativa demostrando las distribuciones IID y non-IID."""
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []
    for mode_name, client_list in [("IID", iid_clients), ("non-IID", non_iid_clients)]:
        for idx, cdf in enumerate(client_list, start=1):
            counts = cdf["target"].value_counts().to_dict()
            total = len(cdf)
            c0 = counts.get(0, 0)
            c1 = counts.get(1, 0)
            c2 = counts.get(2, 0)
            summary_rows.append({
                "Escenario": mode_name,
                "Cliente": f"Cliente {idx}",
                "Total Muestras": total,
                "Clase 0 (Bajo)": c0,
                "% Bajo": round(100 * c0 / total, 2),
                "Clase 1 (Medio)": c1,
                "% Medio": round(100 * c1 / total, 2),
                "Clase 2 (Alto)": c2,
                "% Alto": round(100 * c2 / total, 2),
                "Media CO(GT)": round(cdf["CO(GT)"].mean(), 2),
                "Media Temp (T)": round(cdf["T"].mean(), 2),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "tabla_distribucion_clientes.csv"), index=False)

    # Grafico comparativo de barras apiladas y agrupadas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Clase 0: Bajo (<=1.4)", "Clase 1: Medio (1.4-2.5)", "Clase 2: Alto (>2.5)"]

    for ax_idx, (mode_name, client_list) in enumerate([("Escenario 1: IID (Balanceado)", iid_clients), ("Escenario 2: non-IID (Sesgado)", non_iid_clients)]):
        ax = axes[ax_idx]
        x = np.arange(len(client_list))
        width = 0.25

        for cls in [0, 1, 2]:
            pcts = [
                100.0 * (cdf["target"] == cls).sum() / len(cdf)
                for cdf in client_list
            ]
            counts = [
                (cdf["target"] == cls).sum()
                for cdf in client_list
            ]
            bars = ax.bar(x + (cls - 1) * width, pcts, width, label=labels[cls], color=colors[cls], edgecolor="black", alpha=0.9)
            for bar, pct, cnt in zip(bars, pcts, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{pct:.1f}%\n({cnt})",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax.set_title(mode_name, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Cliente {i+1}\n(n={len(client_list[i])})" for i in range(len(client_list))], fontsize=10.5)
        ax.set_ylabel("Porcentaje de Muestras (%)", fontsize=11)
        ax.set_ylim(0, 108)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", frameon=True)

    plt.suptitle("Demostración de Distribución de Clases por Cliente: IID vs. non-IID (Air Quality Dataset)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribucion_iid_vs_non_iid.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_df


def main():
    print("1. Cargando y limpiando AirQuality.csv...")
    df = load_and_clean_air_quality("data/AirQuality.csv")
    print(f"   Total de registros validos: {len(df)}")
    print("   Distribucion global de clases:")
    for cls, cnt in df["target"].value_counts().sort_index().items():
        print(f"     Clase {cls} ({CLASS_NAMES[cls]}): {cnt} ({100*cnt/len(df):.2f}%)")

    # Dividir en Train (80%) y Test Global en el Servidor (20%)
    df_train, df_test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_SEED
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Ajustar escalador de caracteristicas sobre el conjunto de entrenamiento y estandarizar
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_test_scaled[FEATURE_COLS] = scaler.transform(df_test[FEATURE_COLS])

    # Guardar parametros del escalador y conjunto de prueba global
    os.makedirs("data/iid", exist_ok=True)
    os.makedirs("data/non_iid", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    keep_cols = FEATURE_COLS + ["CO(GT)", "target"]
    df_test_scaled[keep_cols].to_csv("data/test_global.csv", index=False)

    with open("data/scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": FEATURE_COLS,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "class_names": CLASS_NAMES,
            },
            f,
            indent=2,
        )

    # Generar particiones IID y non-IID
    iid_clients = create_iid_splits(df_train_scaled[keep_cols], num_clients=3)
    non_iid_clients = create_non_iid_splits(df_train_scaled[keep_cols])

    for i, cdf in enumerate(iid_clients, start=1):
        cdf.to_csv(f"data/iid/client_{i}.csv", index=False)
    for i, cdf in enumerate(non_iid_clients, start=1):
        cdf.to_csv(f"data/non_iid/client_{i}.csv", index=False)

    summary_df = plot_distributions(iid_clients, non_iid_clients, out_dir="results")
    print("\n2. Tabla de distribucion generada:")
    print(summary_df.to_string(index=False))
    print("\nArchivos guardados en data/iid/, data/non_iid/, data/test_global.csv y results/distribucion_iid_vs_non_iid.png")


if __name__ == "__main__":
    main()
