# EDA, Tratamiento y Modelado Recurrente (LSTM / GRU) para Monóxido de Carbono: Air Quality Dataset

**Proyecto:** Aprendizaje Federado y Predicción de Monóxido de Carbono $\text{CO(GT)}$ con Redes Recurrentes  
**Dataset:** [Air Quality Data Set (Kaggle / fedesoriano - UCI Machine Learning Repository)](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set/data)  
**Objetivo Seleccionado:** Predicción continua de **Monóxido de Carbono $\text{CO(GT)}$** ($\text{mg/m}^3$)  
**Arquitecturas Seleccionadas:** **LSTM** (Long Short-Term Memory) y **GRU** (Gated Recurrent Unit)  
**Autor:** Antigravity (Google DeepMind)  
**Fecha:** Octubre 2026  

---

## 1. Introducción y Contexto del Problema

El **Monóxido de Carbono ($\text{CO}$)** es uno de los contaminantes atmosféricos primarios más peligrosos para la salud humana. Producido por la combustión incompleta de combustibles fósiles (principalmente motores de combustión interna en entornos urbanos y sistemas de calefacción), su concentración varía rápidamente en función de:
1. **La densidad de tráfico vehicular** (patrones circadianos marcados).
2. **Las variables meteorológicas** (temperatura, velocidad del viento, humedad y altura de la capa límite planetaria).
3. **Las reacciones químicas cruzadas** con otros compuestos como hidrocarburos y óxidos de nitrógeno.

A diferencia de los modelos estáticos tabulares tradicionales, el modelado con **redes neuronales recurrentes (LSTM y GRU)** permite explotar la **dependencia temporal serial** de las masas de aire mediante ventanas deslizantes (*sliding windows*), capturando inercias físico-químicas que un modelo punto a punto pasa por alto.

---

## 2. Diccionario de Variables con Rol para Predicción de $\text{CO(GT)}$

El dataset contiene 18 características predictoras ($F=18$) que alimentan a los modelos recurrentes:

```
+-----------------------------------------------------------------------------------------------+
|                            VARIABLES DEL DATASET - ENFOQUE CO(GT)                             |
+------------------------------+----------------------------------+-----------------------------+
| 1. VARIABLE OBJETIVO (TARGET)| 2. SENSORES FÍSICO-QUÍMICOS      | 3. CLIMA Y VARIABLES        |
|                              |    (Respuestas Eléctricas PT08)  |    TEMPORALES DERIVADAS     |
| - CO(GT) [mg/m^3]            | - PT08.S1(CO) (SnO2, sensor CO)  | - T (Temperatura °C)        |
|                              | - PT08.S2(NMHC) (TiO2, hidrocarb)| - RH (Humedad Relativa %)   |
|                              | - PT08.S3(NOx) (WO3, oxidantes)  | - AH (Humedad Absoluta)     |
|                              | - PT08.S4(NO2) (WO3, NO2)        | - Hour_Sin / Hour_Cos       |
|                              | - PT08.S5(O3) (In2O3, ozono)     | - Month_Sin / Month_Cos     |
|                              | - C6H6(GT) (Benceno de ref.)     | - IsWeekend (0 o 1)         |
|                              | - NOx(GT) / NO2(GT)              | - S1_S2_Ratio / S4_S5_Ratio |
+------------------------------+----------------------------------+-----------------------------+
```

| Variable | Rol en el Modelo | Tipo y Unidad | Justificación y Relación Física con el $\text{CO}$ |
| :--- | :--- | :--- | :--- |
| **`CO(GT)`** | **TARGET ($y$)** | Continuo ($\text{mg/m}^3$) | Concentración real medida por analizador oficial de infrarrojo no dispersivo (NDIR). |
| **`PT08.S1(CO)`** | Predictor primario | Resistencia ($\text{SnO}_2$) | Sensor de óxido de estaño diseñado nominalmente para $\text{CO}$. Correlación directa: $r = 0.879$. |
| **`PT08.S2(NMHC)`**| Predictor secundario | Resistencia ($\text{TiO}_2$) | Detecta hidrocarburos copresentes en los tubos de escape vehicular ($r = 0.916$). |
| **`C6H6(GT)`** | Predictor de combustión| Concentración ($\mu\text{g/m}^3$) | Trazador fiel de tráfico vehicular. Correlación lineal altísima con $\text{CO}$: $r = 0.931$. |
| **`PT08.S3(NOx)`** | Predictor de oxidación | Resistencia ($\text{WO}_3$) | Sensor de conductancia inversa a gases oxidantes ($r = -0.703$). |
| **`PT08.S4(NO2)`** | Predictor | Resistencia | Sensibilidad cruzada a óxidos de nitrógeno generados en la combustión térmica. |
| **`PT08.S5(O3)`** | Predictor | Resistencia | Respuesta al ozono fotoquímico y contaminantes secundarios ($r = 0.854$). |
| **`T`** | Predictor ambiental | Grados Celsius (°C) | Modula la tasa de desorción de moléculas de gas en la superficie de los sensores semiconductores. |
| **`RH` / `AH`** | Predictor ambiental | % y g/m³ | El vapor de agua compite con el $\text{CO}$ por los sitios de quimisorción superficial en el óxido metálico. |
| **`Hour_Sin/Cos`** | Predictor temporal | Adimensional $[-1, 1]$ | Modela el ciclo circadiano diurno/nocturno de las emisiones de forma cíclica continua. |
| **`S1_S2_Ratio`** | Ratio fisicoquímico | Adimensional | Ratio de $\text{SnO}_2 / \text{TiO}_2$ que caracteriza el grado de mezcla de la pluma de contaminantes. |

---

## 3. Análisis Exploratorio Especializado en $\text{CO(GT)}$

### 3.1. Distribución Estadística de la Concentración de $\text{CO(GT)}$

Calculada sobre la serie temporal tratada (9,333 puntos válidos secuenciales):

| Métrica Estadística | Valor Real | Interpretación Física / Calidad del Aire |
| :--- | :--- | :--- |
| **Media ($\mu$)** | $2.14\,\text{mg/m}^3$ | Nivel promedio típico en ambiente urbano europeo consolidado. |
| **Desviación Estándar ($\sigma$)** | $1.42\,\text{mg/m}^3$ | Elevada volatilidad horaria generada por las fluctuaciones de tráfico. |
| **Mínimo** | $0.10\,\text{mg/m}^3$ | Mediciones nocturnas (03:00–05:00) con tráfico prácticamente nulo. |
| **Percentil 25 ($Q_1$)** | $1.10\,\text{mg/m}^3$ | Aire urbano limpio / horas valle. |
| **Mediana ($Q_2$)** | $1.80\,\text{mg/m}^3$ | Punto de equilibrio representativo. |
| **Percentil 75 ($Q_3$)** | $2.80\,\text{mg/m}^3$ | Concentraciones asociadas a congestión vehicular moderada. |
| **Máximo** | $11.90\,\text{mg/m}^3$ | Episodio extremo de contaminación severa / estancamiento por inversión térmica. |
| **Asimetría (*Skewness*)** | $+1.53$ | Fuerte cola derecha; justifica el uso de `RobustScaler` y funciones de pérdida robustas (Huber Loss). |

---

### 3.2. Ciclos Temporales y Autocorrelación de $\text{CO(GT)}$

```
                 PERFIL HORARIO PROMEDIO DE CO(GT) (Lunes a Viernes)
  CO (mg/m^3)
   4.0 |                  * (Pico 1: 08:00h)                        * (Pico 2: 19:00h)
   3.0 |                 * *                                       * *
   2.0 |               *     *           * * *                   *     *
   1.0 |   * * * * *           * * * * *       * * * * * * * * *         * *
   0.0 +------------------------------------------------------------------------> Hora
       00  02  04  06  08  10  12  14  16  18  20  22  24
```

1. **Pico Matutino (07:00 a 09:00):**  
   Coincide con el inicio de la actividad laboral. La baja altura de la capa límite invernal concentra las emisiones a nivel de calle.
2. **Valle del Mediodía (12:00 a 16:00):**  
   A pesar de la circulación de vehículos, la radiación solar calienta el suelo, propiciando corrientes de convección térmica que diluyen el $\text{CO}$.
3. **Pico Vespertino-Nocturno (18:00 a 21:00):**  
   Retorno vehicular simultáneo al cese de la radiación solar y contracción de la capa límite, provocando las concentraciones más persistentes del día.
4. **Autocorrelación Temporal:**  
   La correlación serial entre $\text{CO}_t$ y $\text{CO}_{t-1}$ es de **$r = 0.88$**, y la correlación con $\text{CO}_{t-24}$ (mismo horario del día anterior) alcanza **$r = 0.65$**, justificando una ventana recurrente de **24 horas**.

---

### 3.3. Tratamiento de Faltantes en $\text{CO(GT)}$

De las 9,357 instancias, 1,683 (17.99%) tenían el valor centinela `-200` debido a pausas de mantenimiento de la celda de absorción óptica:
- Dado que el sensor electrónico `PT08.S1(CO)` solo tuvo 366 caídas (3.91%), se utilizó **KNNImputer multivariado ($k=5$)** junto con la respuesta de `PT08.S1(CO)`, `PT08.S2(NMHC)` y `C6H6(GT)` ($r > 0.88-0.93$), permitiendo reconstruir la señal original de $\text{CO(GT)}$ con alta fidelidad y sin romper la continuidad de la serie temporal.

---

## 4. Estructuración Secuencial para Redes Recurrentes (Ventana Deslizante)

Para alimentar a los modelos **LSTM** y **GRU**, transformamos la matriz bidimensional $(\text{Tiempo}, \text{Features})$ en tensores tridimensionales utilizando ventanas temporales deslizantes:

```
                            VENTANA DESLIZANTE (L = 24 horas)
 Tiempo t-23  ...  Tiempo t-1   Tiempo t                          Target a predecir
+-----------------------------+---------+                        +-----------------+
| x(t-23)    ...   x(t-1)     | x(t)    |  ===================>  |  y(t) = CO(GT)  |
+-----------------------------+---------+                        +-----------------+
  <-------- 24 pasos de tiempo -------->
```

- **Longitud de secuencia ($L$):** $24$ horas (1 día completo).
- **Número de variables por paso ($F$):** $18$ características normalizadas.
- **Forma del tensor de entrada $X$:**  
  $$\mathbf{X} \in \mathbb{R}^{9333 \times 24 \times 18}$$
- **Forma del vector objetivo $y$:**  
  $$\mathbf{y} \in \mathbb{R}^{9333}$$

---

## 5. Arquitecturas Recurrentes: LSTM vs GRU

### 5.1. Long Short-Term Memory (LSTM)

La celda LSTM resuelve el desvanecimiento de gradiente gracias a su **estado de celda ($c_t$)** aditivo:

#### Ecuaciones fundamentales:
1. **Compuerta de olvido:**  
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
2. **Compuerta de entrada:**  
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
3. **Candidato de celda:**  
   $$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
4. **Actualización del estado de celda:**  
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
5. **Compuerta de salida:**  
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
6. **Vector de estado oculto emitido:**  
   $$h_t = o_t \odot \tanh(c_t)$$

- **Parámetros en nuestra implementación:** **56,897 parámetros** (LSTM 2 capas, 64 unidades ocultas, dropout 0.2 + cabezal MLP).

---

### 5.2. Gated Recurrent Unit (GRU)

La celda GRU simplifica el flujo fusionando el estado de celda y oculto en un único vector $h_t$:

#### Ecuaciones fundamentales:
1. **Compuerta de reinicio (*Reset gate*):**  
   $$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$
2. **Compuerta de actualización (*Update gate*):**  
   $$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
3. **Candidato de estado oculto:**  
   $$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$
4. **Estado oculto actualizado:**  
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

- **Parámetros en nuestra implementación:** **43,201 parámetros** (~24.1% menos parámetros que LSTM).

---

### 5.3. Comparativa Técnica para Calidad del Aire

| Criterio | LSTM | GRU | Recomendación en el Proyecto |
| :--- | :--- | :--- | :--- |
| **Complejidad de parámetros** | 56,897 parámetros | 43,201 parámetros | GRU es más ligero para transferir en cada ronda federada. |
| **Memoria de largo plazo** | Superior (separación explícita de $c_t$ y $h_t$). | Buena (combinada en $h_t$). | LSTM es ideal si se analizan eventos de inversión de varios días. |
| **Velocidad de entrenamiento** | Ligeramente más lenta. | ~20-25% más rápida por época. | GRU permite mayor agilidad en clientes con CPU lenta. |
| **Heterogeneidad de clientes** | Puede correr en clientes `c1` y `c3`. | Puede correr en clientes `c2` y `c4` (modo destilación). | Ambos están integrados y son totalmente compatibles. |

---

## 6. Funciones de Pérdida y Métricas de Regresión

Al tratarse de una tarea de regresión continua (predecir la concentración exacta de $\text{CO}$ en $\text{mg/m}^3$), se sustituye la entropía cruzada por métricas de regresión:

### Función de Pérdida Recomendada: **Huber Loss (Smooth L1)**
$$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \le \delta \\ \delta \cdot \left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{en caso contrario} \end{cases}$$
*Justificación:* Actúa como MSE para pequeños errores de calibración, pero como MAE (lineal) ante picos repentinos de tráfico vehicular, evitando que los gradientes de la red recurrente se desestabilicen.

### Métricas de Evaluación:
1. **RMSE (Root Mean Squared Error):**  
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2} \quad [\text{mg/m}^3]$$
2. **MAE (Mean Absolute Error):**  
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i| \quad [\text{mg/m}^3]$$
3. **$R^2$ Score (Coeficiente de Determinación):**  
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   Indica qué porcentaje de la variabilidad real del monóxido de carbono es explicada por la red recurrente.

---

## 7. Particionamiento Federado Especializado (4 Clientes: `c1..c4`)

Los 9,333 ejemplos secuenciales han sido divididos en shards `.npz` compatibles con la arquitectura federada:

```
                            DISTRIBUCIÓN FEDERADA DE SHARDS
+-----------------------------------------------------------------------------------------------+
|  Cliente  |  N° Secuencias  |  Partición IID                   |  Partición Non-IID           |
+-----------+-----------------+----------------------------------+------------------------------+
|    c1     |   2,334 sec.    | Muestra uniforme de todo el año  | Zona Residencial (CO bajo)   |
|    c2     |   2,333 sec.    | Muestra uniforme de todo el año  | Zona Comercial (CO medio-bajo|
|    c3     |   2,333 sec.    | Muestra uniforme de todo el año  | Arteria Vial (CO medio-alto) |
|    c4     |   2,333 sec.    | Muestra uniforme de todo el año  | Punto Crítico (CO extremo)   |
|  Público  |     500 sec.    | Conjunto label-free compartido   | Para destilación federada    |
+-----------------------------------------------------------------------------------------------+
```

### Ubicación de los archivos en disco:
- **Shards IID:** `air_quality_processed/iid/{c1,c2,c3,c4}.npz`
- **Shards Non-IID:** `air_quality_processed/noniid/{c1,c2,c3,c4}.npz`
- **Público:** `air_quality_processed/public.npz`
- **Copia de seguridad en escritorio:** `C:\Users\mayte.otavalo\Desktop\MAYTE\MACHINE\IA\AIR_QUALITY (aprendizaje-federado)\air_quality_processed\`

---

## 8. Resumen de Archivos y Scripts Disponibles

1. [`prepare_air_quality_lstm.py`](file:///C:/Users/mayte.otavalo/Downloads/peer_c2%20(1)/prepare_air_quality_lstm.py): Pipeline completo de generación de secuencias y particionado federado para Monóxido de Carbono.
2. [`models_recurrent.py`](file:///C:/Users/mayte.otavalo/Downloads/peer_c2%20(1)/models_recurrent.py): Definición de clases PyTorch para `LSTMRegressor` y `GRURegressor` con sus variantes bidireccionales (`BiLSTM`, `BiGRU`).
3. [`EDA_Y_TRATAMIENTO_AIR_QUALITY.md`](file:///C:/Users/mayte.otavalo/Downloads/peer_c2%20(1)/EDA_Y_TRATAMIENTO_AIR_QUALITY.md): Documento Markdown completo integrado en el proyecto.
