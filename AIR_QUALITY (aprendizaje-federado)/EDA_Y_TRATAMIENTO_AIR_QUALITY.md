# Análisis Exploratorio de Datos (EDA) y Tratamiento: Air Quality Dataset

**Proyecto:** Aprendizaje Federado y Modelado Predictivo de Contaminación Atmosférica  
**Dataset:** [Air Quality Data Set (Kaggle / fedesoriano - UCI Machine Learning Repository)](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set/data)  
**Autor:** Antigravity (Google DeepMind)  
**Fecha:** Octubre 2026  

---

## 1. Introducción y Contexto del Dataset

El conjunto de datos **Air Quality** es un benchmark de referencia internacional en el ámbito del sensado ambiental y la ciencia de datos. Contiene **9,357 registros horarios** recolectados entre **marzo de 2004 y abril de 2005** (389 días continuos) en una estación de monitoreo situada en una zona altamente contaminada de una ciudad italiana.

El experimento combinó:
1. **Un dispositivo multisensor de bajo coste:** Con 5 sensores químicos de óxido metálico de estado sólido (*metal oxide semiconductor chemical sensors*).
2. **Una estación de referencia certificada (*Ground Truth*):** Con analizadores químicos oficiales de absorción óptica y quimioluminiscencia.

El objetivo central de modelado con este dataset consiste en **calibrar y predecir concentraciones de gases tóxicos reales** a partir de las respuestas eléctricas de los sensores y las variables ambientales, o pronosticar la evolución temporal de la calidad del aire.

---

## 2. Diccionario de Variables y Especificación Técnica

El archivo original contiene 15 atributos principales organizados en tres categorías:

```
+-----------------------------------------------------------------------------------------+
|                                    AIR QUALITY DATASET                                  |
+------------------------------+------------------------------+---------------------------+
| 1. Variables Temporales      | 2. Analizadores Referencia   | 3. Sensores Químicos /    |
|                              |    (Ground Truth Oficial)    |    Variables Ambientales  |
| - Date (DD/MM/YYYY)          | - CO(GT) [mg/m^3]            | - PT08.S1(CO) [Resistencia]
| - Time (HH.MM.SS)            | - NMHC(GT) [ug/m^3]          | - PT08.S2(NMHC)           |
|                              | - C6H6(GT) [ug/m^3]          | - PT08.S3(NOx)            |
|                              | - NOx(GT) [ppb]              | - PT08.S4(NO2)            |
|                              | - NO2(GT) [ug/m^3]           | - PT08.S5(O3)             |
|                              |                              | - T (Temperatura °C)      |
|                              |                              | - RH (Humedad Relativa %) |
|                              |                              | - AH (Humedad Absoluta)   |
+------------------------------+------------------------------+---------------------------+
```

### Detalle de cada variable

| Variable | Tipo Físico | Unidad | Descripción y Rol en el Modelo |
| :--- | :--- | :--- | :--- |
| **`Date`** | Temporal | Fecha (`DD/MM/YYYY`) | Día de la medición (10/03/2004 al 04/04/2005). |
| **`Time`** | Temporal | Hora (`HH.MM.SS`) | Hora de la medición horaria promedio. |
| **`CO(GT)`** | Analizador oficial | $\text{mg/m}^3$ | Concentración horaria real de **Monóxido de Carbono** medida por sensor de infrarrojo certificado. |
| **`PT08.S1(CO)`** | Sensor químico | Respuesta del sensor | Respuesta del sensor PT08 basado en Óxido de Estaño ($\text{SnO}_2$), orientado nominalmente a CO. |
| **`NMHC(GT)`** | Analizador oficial | $\mu\text{g/m}^3$ | Hidrocarburos no metánicos reales (*Non-Methanic HydroCarbons*). *(Presenta 90.2% de valores faltantes)*. |
| **`C6H6(GT)`** | Analizador oficial | $\mu\text{g/m}^3$ | Concentración de **Benceno** mediante cromatografía de gases. Es el target continuo de referencia más confiable ($r=0.98$ con sensores). |
| **`PT08.S2(NMHC)`** | Sensor químico | Respuesta del sensor | Sensor PT08 basado en Dióxido de Titanio ($\text{TiO}_2$), orientado nominalmente a hidrocarburos. |
| **`NOx(GT)`** | Analizador oficial | $\text{ppb}$ | Concentración real de **Óxidos de Nitrógeno** ($\text{NO} + \text{NO}_2$) por quimioluminiscencia. |
| **`PT08.S3(NOx)`** | Sensor químico | Respuesta del sensor | Sensor PT08 basado en Óxido de Wolframio ($\text{WO}_3$), orientado a $\text{NO}_x$. Muestra respuesta de conductancia inversa a gases oxidantes. |
| **`NO2(GT)`** | Analizador oficial | $\mu\text{g/m}^3$ | Concentración real de **Dióxido de Nitrógeno** medida por analizador de referencia. |
| **`PT08.S4(NO2)`** | Sensor químico | Respuesta del sensor | Sensor PT08 basado en Óxido de Wolframio orientado nominalmente a $\text{NO}_2$. |
| **`PT08.S5(O3)`** | Sensor químico | Respuesta del sensor | Sensor PT08 basado en Óxido de Indio ($\text{In}_2\text{O}_3$), orientado nominalmente a **Ozono ($\text{O}_3$)**. |
| **`T`** | Meteorológica | °C | Temperatura ambiental del aire en grados Celsius. Afecta drásticamente la cinética de reacción de los sensores. |
| **`RH`** | Meteorológica | % | Humedad relativa del aire. Provoca absorción de vapor de agua en la superficie del sensor. |
| **`AH`** | Meteorológica | Adimensional / $\text{g/m}^3$ | Humedad absoluta del aire. Variable clave para compensar la deriva higrométrica de los semiconductores. |

---

## 3. Análisis Exploratorio de Datos (EDA)

### 3.1. Detección y Distribución del Valor Centinela `-200` (Faltantes)

En este dataset, las fallas de captura no están codificadas como celdas vacías (`NaN`), sino con el valor numérico **`-200`**. Si no se trata adecuadamente, distorsiona gravemente cualquier cálculo de media, varianza o modelo de gradiente.

| Variable | Registros Totales | Valores `-200` (Nulos) | % Faltante | Diagnóstico y Causa Raíz |
| :--- | :--- | :--- | :--- | :--- |
| **`Date`** | 9,357 | 0 | 0.00% | Completo. |
| **`Time`** | 9,357 | 0 | 0.00% | Completo. |
| **`CO(GT)`** | 9,357 | 1,683 | 17.99% | Analizador de referencia en calibración periódica. |
| **`PT08.S1(CO)`** | 9,357 | 366 | 3.91% | Interrupción simultánea del array multisensor (apagón/corte). |
| **`NMHC(GT)`** | 9,357 | **8,443** | **90.23%** | **Falla crítica:** El analizador de hidrocarburos fue descontinuado tras los primeros meses. |
| **`C6H6(GT)`** | 9,357 | 366 | 3.91% | Solo falta durante la caída general de la estación multisensor. |
| **`PT08.S2(NMHC)`** | 9,357 | 366 | 3.91% | Falla del array multisensor. |
| **`NOx(GT)`** | 9,357 | 1,639 | 17.52% | Mantenimiento del analizador de quimioluminiscencia. |
| **`PT08.S3(NOx)`** | 9,357 | 366 | 3.91% | Falla del array multisensor. |
| **`NO2(GT)`** | 9,357 | 1,642 | 17.55% | Mantenimiento del analizador de $\text{NO}_2$. |
| **`PT08.S4(NO2)`** | 9,357 | 366 | 3.91% | Falla del array multisensor. |
| **`PT08.S5(O3)`** | 9,357 | 366 | 3.91% | Falla del array multisensor. |
| **`T`** | 9,357 | 366 | 3.91% | Falla del sensor ambiental de la estación. |
| **`RH`** | 9,357 | 366 | 3.91% | Falla del sensor ambiental de la estación. |
| **`AH`** | 9,357 | 366 | 3.91% | Falla del sensor ambiental de la estación. |

---

### 3.2. Estadísticas Descriptivas (Tras Reemplazar `-200` con `NaN`)

Calculadas sobre las 9,357 muestras válidas:

| Variable | Media ($\mu$) | Desv. Est. ($\sigma$) | Mínimo | Mediana ($Q_2$) | Máximo | Asimetría (*Skewness*) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`CO(GT)`** | 2.15 | 1.45 | 0.10 | 1.80 | 11.90 | Positiva (+1.53) |
| **`PT08.S1(CO)`** | 1099.83 | 217.08 | 647.00 | 1063.00 | 2040.00 | Leve positiva (+0.81) |
| **`C6H6(GT)`** | 10.08 | 7.45 | 0.10 | 8.20 | 63.70 | Fuerte positiva (+1.56) |
| **`PT08.S2(NMHC)`** | 939.15 | 266.83 | 383.00 | 909.00 | 2214.00 | Positiva (+0.79) |
| **`NOx(GT)`** | 246.90 | 212.98 | 2.00 | 180.00 | 1479.00 | Muy pronunciada (+1.92) |
| **`PT08.S3(NOx)`** | 835.49 | 256.82 | 322.00 | 806.00 | 2683.00 | Positiva (+1.06) |
| **`NO2(GT)`** | 113.09 | 48.37 | 2.00 | 109.00 | 340.00 | Moderada (+0.55) |
| **`PT08.S4(NO2)`** | 1456.26 | 346.21 | 551.00 | 1463.00 | 2775.00 | Cercana a simétrica (+0.36) |
| **`PT08.S5(O3)`** | 1022.91 | 398.48 | 221.00 | 963.00 | 2523.00 | Moderada (+0.78) |
| **`T` (°C)** | 18.32 | 8.83 | -1.90 | 17.80 | 44.60 | Prácticamente normal (0.01) |
| **`RH` (%)** | 49.23 | 17.32 | 9.20 | 49.60 | 88.70 | Simétrica (-0.03) |
| **`AH`** | 1.03 | 0.40 | 0.18 | 1.00 | 2.23 | Moderada (+0.43) |

---

### 3.3. Análisis de Correlación y Dinámica Química

```
                             MATRIZ DE CORRELACIÓN CLAVE
+-----------------------------------------------------------------------------------+
| Variable      | C6H6(GT) | CO(GT) | PT08.S1 | PT08.S2 | PT08.S3 | PT08.S5 | Temp   |
+---------------+----------+--------+---------+---------+---------+---------+--------+
| C6H6(GT)      |  1.000   |  0.931 |  0.884  |  0.982  | -0.736  |  0.866  |  0.199 |
| CO(GT)        |  0.931   |  1.000 |  0.879  |  0.916  | -0.703  |  0.854  |  0.022 |
| PT08.S2(NMHC) |  0.982   |  0.916 |  0.893  |  1.000  | -0.797  |  0.884  |  0.241 |
| PT08.S1(CO)   |  0.884   |  0.879 |  1.000  |  0.893  | -0.772  |  0.899  |  0.049 |
| PT08.S3(NOx)  | -0.736   | -0.703 | -0.772  | -0.797  |  1.000  | -0.799  | -0.145 |
+-----------------------------------------------------------------------------------+
```

#### Interpretación de las Correlaciones:
1. **Acoplamiento extraordinario entre `PT08.S2(NMHC)` y `C6H6(GT)` ($r = 0.982$):**  
   El sensor de titania reacciona de forma casi lineal a los hidrocarburos aromáticos monoanulares (Benceno), convirtiéndolo en un predictor estelar.
2. **Correlación Inversa Fuerte de `PT08.S3(NOx)` ($r = -0.736$ a $-0.797$):**  
   A diferencia de los otros sensores cuya conductancia aumenta con los gases reductores, el sensor de óxido de tungsteno incrementa su resistencia óhmica frente a gases oxidantes ($\text{NO}_x$), reduciendo el valor de lectura.
3. **Sensibilidad Cruzada a Temperatura y Humedad:**  
   Los sensores presentan fluctuaciones asociadas a la humedad absoluta `AH`. La adsorción de moléculas de agua $\text{H}_2\text{O}$ en la superficie de los semiconductores modifica la barrera de potencial de grano, lo que justifica incorporar `T` y `AH` en el modelado.

---

### 3.4. Dinámica Temporal: Ciclos Diarios y Estacionales

- **Picos Bi-modales de Tráfico:** Las concentraciones de `CO`, `NOx` y `C6H6` presentan dos crestas muy marcadas de lunes a viernes:
  - Primer pico: **08:00 a 09:00** (desplazamientos matutinos hacia el trabajo).
  - Segundo pico: **18:00 a 20:00** (retorno y condiciones de enfriamiento de la capa límite atmosférica).
- **Efecto Fin de Semana:** Sábados y domingos los picos matutinos se reducen en un ~40% y se desplazan hacia el mediodía y la medianoche.
- **Estacionalidad Invierno vs. Verano:**  
  En invierno, la inversión térmica atrapa los contaminantes cerca del suelo, multiplicando las concentraciones de $\text{NO}_x$ y $\text{CO}$. En verano, la alta radiación solar incrementa el fotoquímico $\text{O}_3$ y la temperatura ($>40^\circ\text{C}$).

---

## 4. Tratamiento y Preprocesamiento de Datos (Paso a Paso)

El flujo de tratamiento de datos desarrollado e implementado en el script ejecutable `prepare_air_quality.py` consta de las siguientes etapas rigurosas:

### Paso 1: Limpieza Estructural de Ingesta
- Se especifica el delimitador europeo (`;`) y el separador decimal (`,`).
- Se eliminan las columnas espurias `Unnamed: 15` y `Unnamed: 16` resultantes del formato de exportación original.
- Se eliminan 114 filas finales completamente vacías, consolidando la matriz exacta de **9,357 registros**.

### Paso 2: Creación del Índice Temporal Continuo
- Fusión de las columnas de texto `Date` y `Time` a formato unificado `datetime64[ns]`:
  $$\text{DateTime} = \text{Date} + \text{" "} + \text{Time (formato HH:MM:SS)}$$
- Ordenamiento cronológico estricto garantizando la integridad de series de tiempo.

### Paso 3: Tratamiento Diferenciado de Valores Faltantes
1. **Descarte de `NMHC(GT)`:** Con 8,443 nulos de 9,357 (90.23%), se elimina para evitar la propagación de varianza artificial o sesgo severo.
2. **Interpolación Temporal para Sensores y Clima (`PT08.S1..S5`, `T`, `RH`, `AH`, `C6H6`):**  
   Al tratarse de interrupciones cortas de la estación (3.91%), la **interpolación lineal continua en el tiempo** preserva la inercia termodinámica natural de la atmósfera.
3. **Imputación Multivariada por Vecinos Cercanos (KNNImputer con $k=5$) para Gases de Referencia (`CO`, `NOx`, `NO2`):**  
   Aprovecha las correlaciones físicas con los 5 sensores de estado sólido y las variables climáticas para reconstruir las lecturas durante los periodos de calibración del analizador oficial.

### Paso 4: Ingeniería de Características (*Feature Engineering*)
Para dotar a los modelos de machine learning de capacidad predictiva sobre la dinámica atmosférica:
- **Componentes Temporales Discretos:** `Hour`, `DayOfWeek`, `Month`, `IsWeekend` (indicador binario de fin de semana).
- **Transformación Cíclica (Seno / Coseno):**  
  Evita la discontinuidad artificial entre las 23:00 y las 00:00:
  $$\text{Hour\_Sin} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right), \quad \text{Hour\_Cos} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$
  $$\text{Month\_Sin} = \sin\left(\frac{2\pi \cdot \text{Month}}{12}\right), \quad \text{Month\_Cos} = \cos\left(\frac{2\pi \cdot \text{Month}}{12}\right)$$
- **Ratios Fisicoquímicos de Sensores:**
  - $\text{S1\_S2\_Ratio} = \frac{\text{PT08.S1(CO)}}{\text{PT08.S2(NMHC)}}$: Revela cambios en la proporción entre gases de combustión incompleta y compuestos aromáticos.
  - $\text{S4\_S5\_Ratio} = \frac{\text{PT08.S4(NO2)}}{\text{PT08.S5(O3)}}$: Modela el ciclo fotostacionario del ozono y dióxido de nitrógeno bajo la radiación solar.

### Paso 5: Escalado Robusto (*Robust Scaling*)
Dado que los picos de contaminación vehicular y las inversiones meteorológicas generan valores atípicos reales (no errores instrumentales), se emplea `RobustScaler`:
$$x_{\text{scaled}} = \frac{x - \text{Mediana}(x)}{\text{IQR}(x)}$$
Esto previene que los gradientes de la red neuronal o los modelos lineales exploten ante episodios extremos de esmog.

---

## 5. Adaptación al Entorno de Aprendizaje Federado (Federated Learning)

Para integrarse de manera idéntica a la arquitectura distribuida del proyecto actual (`peer_c2` con `c1..c4`):

### 5.1. Tarea de Machine Learning
- **Modo:** Regresión Continua Multivariada.
- **Variable Objetivo ($y$):** Concentración real de **Benceno $\text{C6H6(GT)}$** (o Monóxido de Carbono $\text{CO(GT)}$).
- **Matriz de Entrada ($X$):** 15 predictores normalizados (sensores, clima, cíclicas y ratios).

### 5.2. Estrategias de Partición para 4 Nodos (`c1`, `c2`, `c3`, `c4`)

1. **Partición IID (Independiente e Idénticamente Distribuida):**
   - Muestreo aleatorio uniforme del conjunto temporal.
   - Cada cliente recibe aproximadamente **2,340 registros** con distribuciones de gases y condiciones meteorológicas equivalentes.
2. **Partición Non-IID (Heterogeneidad Real de Datos):**
   - **Simulación por Niveles de Exposición / Regímenes de Polución:**
     - **`c1`:** Estación en zona residencial periférica (baja polución, percentil 0 a 25% de concentración).
     - **`c2` y `c3`:** Zonas comerciales y arterias intermedias (polución moderada, percentil 25% a 75%).
     - **`c4`:** Cañón urbano o punto de tráfico pesado extremo (alta polución, percentil 75% a 100%).
   - Permite comparar de forma directa la convergencia de algoritmos federados como **FedAvg** vs. estrategias de destilación por consenso bajo sesgo no homogéneo.

### 5.3. Artefactos Generados en Disco

El script de procesamiento ha generado automáticamente:
- `air_quality_processed/air_quality_cleaned.csv`: Tabla completa procesada con 9,357 filas y nuevas variables.
- `air_quality_processed/iid/{c1,c2,c3,c4}.npz`: Shards IID listos para consumo por los clientes.
- `air_quality_processed/noniid/{c1,c2,c3,c4}.npz`: Shards Non-IID clasificados por régimen de emisión.
- `air_quality_processed/public.npz`: Conjunto no etiquetado de 500 muestras para destilación federada.
- Cada archivo `.npz` contiene exactamente las claves `X`, `y`, e `idx`, manteniendo **100% de compatibilidad con `common/data.py`** del proyecto.
