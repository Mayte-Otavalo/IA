"""
Pipeline de Tratamiento y Ventaneo Temporal para LSTM/GRU en Aprendizaje Federado
Target: Monóxido de Carbono CO(GT)
Modelos: LSTM / GRU
Dataset: Air Quality (UCI / Kaggle)
"""

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

def create_sequences(X_data: np.ndarray, y_data: np.ndarray, seq_len: int = 24):
    """
    Transforma matrices 2D en secuencias temporales 3D para LSTM/GRU.
    Entrada:
      X_data: (T, num_features)
      y_data: (T,)
      seq_len: longitud de ventana histórica (ej. 24 horas)
    Salida:
      X_seq: (N, seq_len, num_features)
      y_seq: (N,) correspondiente al valor objetivo en t
    """
    X_seq, y_seq = [], []
    for i in range(len(X_data) - seq_len):
        X_seq.append(X_data[i : i + seq_len])
        y_seq.append(y_data[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def main():
    # Buscar archivo original
    candidate_paths = [
        "air_quality_data/AirQualityUCI.csv",
        r"C:\Users\mayte.otavalo\Desktop\MAYTE\MACHINE\IA\AIR_QUALITY (aprendizaje-federado)\air_quality_data\AirQualityUCI.csv"
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if not csv_path:
        print("Error: No se encontró AirQualityUCI.csv en las rutas conocidas.")
        return

    print(f"1. Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', decimal=',', na_values=[-200, -200.0])
    
    # 2. Limpieza preliminar
    df = df.dropna(how='all', subset=['Date'])
    df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], errors='ignore')
    
    df['DateTime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False),
        format='%d/%m/%Y %H:%M:%S'
    )
    df = df.sort_values('DateTime').reset_index(drop=True)
    print(f"Dimensiones iniciales: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 3. Tratamiento de Faltantes (-200)
    print("\n2. Tratamiento especializado para Monóxido de Carbono CO(GT) y Sensores...")
    # Descartar NMHC(GT) debido al 90.2% de nulos
    df = df.drop(columns=['NMHC(GT)'])
    
    # Sensores y meteorología: interpolación temporal
    sensor_cols = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
        'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]
    for col in sensor_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
    ref_cols = ['C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
    for col in ref_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

    # Imputación guiada de CO(GT) usando KNN multivariado con el sensor PT08.S1(CO) y Benzene
    all_num_cols = ['CO(GT)'] + sensor_cols + ref_cols
    imputer = KNNImputer(n_neighbors=5)
    df[all_num_cols] = imputer.fit_transform(df[all_num_cols])
    
    print("Nulos restantes tras imputación:", df[all_num_cols].isnull().sum().sum())

    # 4. Feature Engineering
    print("\n3. Ingeniería de características para modelado temporal...")
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.day_of_week
    df['Month'] = df['DateTime'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(np.float32)
    
    # Codificación cíclica continua
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0).astype(np.float32)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12.0).astype(np.float32)
    
    # Ratios fisicoquímicos
    df['S1_S2_Ratio'] = (df['PT08.S1(CO)'] / (df['PT08.S2(NMHC)'] + 1e-5)).astype(np.float32)
    df['S4_S5_Ratio'] = (df['PT08.S4(NO2)'] / (df['PT08.S5(O3)'] + 1e-5)).astype(np.float32)

    # 5. Definición de Features y Target CO(GT)
    feature_cols = [
        'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH',
        'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 'IsWeekend',
        'S1_S2_Ratio', 'S4_S5_Ratio'
    ]
    target_col = 'CO(GT)'  # MONÓXIDO DE CARBONO COMO TARGET PRINCIPAL
    
    print(f"Target seleccionado: {target_col}")
    print(f"Número de características predictoras (F): {len(feature_cols)}")

    # Escalado Robusto para mitigar picos de contaminación vehicular
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[feature_cols]).astype(np.float32)
    y_scaled = df[target_col].values.astype(np.float32)

    # 6. Construcción de Secuencias (Sliding Window: seq_len = 24 horas)
    SEQ_LEN = 24
    print(f"\n4. Construyendo ventanas deslizantes de {SEQ_LEN} horas para LSTM/GRU...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len=SEQ_LEN)
    indices = np.arange(len(X_seq), dtype=np.uint32)
    print(f"Dimensiones de entrada para LSTM/GRU (N, Seq_Len, Features): {X_seq.shape}")
    print(f"Dimensiones de objetivo (N,): {y_seq.shape}")

    # 7. Particionamiento Federado para 4 Nodos (c1, c2, c3, c4)
    # Partición IID: Barajado aleatorio uniforme
    np.random.seed(42)
    iid_perm = np.random.permutation(len(X_seq))
    shards_iid = np.array_split(iid_perm, 4)
    
    # Partición Non-IID: Segmentación por niveles de exposición a Monóxido de Carbono
    # c1 = baja concentración de CO, c2 y c3 = media, c4 = alta concentración (picos de tráfico)
    co_sorted_idx = np.argsort(y_seq)
    shards_noniid = np.array_split(co_sorted_idx, 4)

    # Carpetas destino
    target_dirs = ["air_quality_processed"]
    desktop_dest = r"C:\Users\mayte.otavalo\Desktop\MAYTE\MACHINE\IA\AIR_QUALITY (aprendizaje-federado)\air_quality_processed"
    if os.path.exists(os.path.dirname(desktop_dest)):
        target_dirs.append(desktop_dest)

    for base_out in target_dirs:
        os.makedirs(base_out, exist_ok=True)
        # Guardar dataset procesado completo
        df.to_csv(os.path.join(base_out, "air_quality_co_cleaned.csv"), index=False)
        
        # Guardar Shards IID y Non-IID
        for mode, shards in [("iid", shards_iid), ("noniid", shards_noniid)]:
            mode_dir = os.path.join(base_out, mode)
            os.makedirs(mode_dir, exist_ok=True)
            for i, client_id in enumerate(["c1", "c2", "c3", "c4"]):
                idx_shard = shards[i]
                shard_path = os.path.join(mode_dir, f"{client_id}.npz")
                np.savez_compressed(
                    shard_path,
                    X=X_seq[idx_shard],
                    y=y_seq[idx_shard],
                    idx=indices[idx_shard]
                )
                print(f"Guardado {mode.upper()} Shard {client_id}: {len(idx_shard)} secuencias en {shard_path}")
        
        # Shard público para destilación federada (500 secuencias no etiquetadas)
        public_idx = np.random.choice(len(X_seq), size=500, replace=False)
        np.savez_compressed(
            os.path.join(base_out, "public.npz"),
            X=X_seq[public_idx]
        )
        print(f"Guardado conjunto público en {os.path.join(base_out, 'public.npz')}")

    print("\nProcesamiento para LSTM/GRU finalizado exitosamente.")

if __name__ == "__main__":
    main()
