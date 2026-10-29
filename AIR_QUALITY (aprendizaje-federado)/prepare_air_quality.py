"""
Script completo de EDA y Tratamiento de Datos: Air Quality Dataset (UCI / Kaggle)
Genera el dataset limpio y la partición federada (IID y non-IID) para 4 clientes.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import KNNImputer

def main():
    csv_path = "air_quality_data/AirQualityUCI.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} no encontrado.")
        return

    print("1. Cargando datos...")
    df = pd.read_csv(csv_path, sep=';', decimal=',', na_values=[-200, -200.0])
    
    # 2. Limpieza básica inicial
    df = df.dropna(how='all', subset=['Date'])
    df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
    
    # Formateo de fecha y hora
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False),
        format='%d/%m/%Y %H:%M:%S'
    )
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    print(f"Dimensiones iniciales: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Rango temporal: de {df['DateTime'].min()} a {df['DateTime'].max()}")

    # 3. Tratamiento de Valores Nulos
    # NMHC(GT) tiene >90% de nulos -> Se descarta para el modelado principal
    print("\n2. Tratamiento de valores faltantes (-200)...")
    df = df.drop(columns=['NMHC(GT)'])
    
    # Sensores y variables ambientales (3.9% nulos) -> Interpolación temporal lineal
    sensor_cols = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
        'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]
    for col in sensor_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    # Target prioritario: C6H6(GT) tiene 3.9% nulos -> interpolación
    df['C6H6(GT)'] = df['C6H6(GT)'].interpolate(method='linear', limit_direction='both')
    
    # Gases de referencia con ~17-18% nulos (CO, NOx, NO2) -> KNN Imputer multivariado
    ref_cols = ['CO(GT)', 'NOx(GT)', 'NO2(GT)']
    numeric_features = sensor_cols + ['C6H6(GT)'] + ref_cols
    
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    
    print("Nulos restantes tras imputación:", df[numeric_features].isnull().sum().sum())

    # 4. Feature Engineering
    print("\n3. Ingeniería de características...")
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Codificación cíclica (24h y 12 meses)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0)
    
    # Ratios de sensores e interacciones
    df['S1_S2_Ratio'] = df['PT08.S1(CO)'] / (df['PT08.S2(NMHC)'] + 1e-5)
    df['S4_S5_Ratio'] = df['PT08.S4(NO2)'] / (df['PT08.S5(O3)'] + 1e-5)
    
    # Guardar versión limpia CSV
    os.makedirs("air_quality_processed", exist_ok=True)
    df.to_csv("air_quality_processed/air_quality_cleaned.csv", index=False)
    print("Dataset limpio guardado en: air_quality_processed/air_quality_cleaned.csv")

    # 5. Preparación para Aprendizaje Federado (Federated Learning)
    # Seleccionamos variables predictoras (X) y objetivo (y: C6H6 o CO)
    feature_cols = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'T', 'RH', 'AH', 'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos',
        'IsWeekend', 'S1_S2_Ratio', 'S4_S5_Ratio'
    ]
    target_col = 'C6H6(GT)'  # Concentración de Benceno como target de regresión
    
    scaler = RobustScaler()
    X = scaler.fit_transform(df[feature_cols]).astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    indices = np.arange(len(df), dtype=np.uint32)

    # Crear shards federados para 4 clientes (c1, c2, c3, c4)
    # Partición IID (mezcla estratificada aleatoria)
    np.random.seed(42)
    iid_perm = np.random.permutation(len(df))
    shards_iid = np.array_split(iid_perm, 4)
    
    # Partición Non-IID (ordenado por concentración de contaminación o estacionalidad)
    # Ordenado por el target y (o por meses estacionales)
    noniid_order = np.argsort(y)  # c1 = baja polución, c4 = alta polución
    shards_noniid = np.array_split(noniid_order, 4)

    for mode, shards in [("iid", shards_iid), ("noniid", shards_noniid)]:
        out_dir = f"air_quality_processed/{mode}"
        os.makedirs(out_dir, exist_ok=True)
        for i, client_id in enumerate(["c1", "c2", "c3", "c4"]):
            idx_shard = shards[i]
            shard_path = os.path.join(out_dir, f"{client_id}.npz")
            np.savez_compressed(
                shard_path,
                X=X[idx_shard],
                y=y[idx_shard],
                idx=indices[idx_shard]
            )
            print(f"[{mode.upper()}] Shard {client_id}: {len(idx_shard)} muestras guardado en {shard_path}")

    # Shard público para destilación federada (500 muestras)
    public_idx = np.random.choice(len(df), size=500, replace=False)
    np.savez_compressed(
        "air_quality_processed/public.npz",
        X=X[public_idx]
    )
    print(f"Conjunto público (public.npz): 500 muestras guardado.")
    print("\nProcesamiento y preparación completados con éxito.")

if __name__ == "__main__":
    main()
