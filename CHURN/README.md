# Guía de Uso: Tarea 7 - Telco Churn ML, XAI y Business Pitch

Este repositorio contiene la solución completa para la **Tarea 7: Optimización de Machine Learning, Explainable AI y Business Pitch** (Asignatura: Aprendizaje Automático).

---

## Contenido del Proyecto
- **`app.py`**: Aplicación web interactiva en **Streamlit** con 4 módulos:
  1. *A. Análisis Exploratorio (EDA)* con gráficos interactivos y KPIs.
  2. *B. Desempeño y Pareto* con comparativa de optimizaciones (Grid, Random, TPE, NSGA-II).
  3. *C. Predicción & Explicabilidad (XAI)* con formularios y desgloses de SHAP y LIME.
  4. *D. Aplicación de Negocio & Pitch* con simulador financiero de ROI interactivo.
- **`train_and_evaluate.py`**: Script de entrenamiento, optimización, cálculo de métricas y serialización de modelos.
- **`business_pitch.md`**: Documento ejecutivo de la propuesta de negocio (*RetenTel AI*), justificación económica y guion de negociación comercial para el roleplay presencial.
- **`models/`**: Modelos serializados (`champion_pipeline.joblib`, `preprocessor.joblib`, `metadata.json`).
- **`Telco-Customer-Churn.csv`**: Dataset de IBM / Kaggle con 7,043 clientes.

---

## Opción A: Ejecución en Google Colab (Recomendada)
1. Abre [Google Colab](https://colab.research.google.com/).
2. Haz clic en **Subir (Upload)** y selecciona el archivo [`Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb`](file:///C:/Users/mayte.otavalo/Desktop/MAYTE/MACHINE/TAREA%20CHURN/Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb).
3. En el menú superior de Colab, selecciona **Entorno de ejecución $\rightarrow$ Ejecutar todas**.
4. En la celda final de la **Parte 5**:
   - Se imprimirá tu dirección IP pública (ejemplo: `34.125.80.20`).
   - Se generará un enlace público de `localtunnel` (ejemplo: `https://famous-fox-42.loca.lt`).
   - Haz clic en el enlace, pega tu IP en el campo de contraseña ("Tunnel Password") y haz clic en "Submit".
   - ¡Tu aplicación Streamlit se abrirá en vivo e interactiva directamente en tu navegador!

---

## Opción B: Ejecución Local en tu Computadora
1. **Entrenar y verificar el modelo (opcional si ya existen los archivos en `models/`):**
   ```bash
   python train_and_evaluate.py
   ```
2. **Lanzar la aplicación interactiva de Streamlit:**
   ```bash
   streamlit run app.py
   ```
3. Se abrirá automáticamente en tu navegador en: `http://localhost:8501`.

---

##  Resumen de Resultados Técnicos

| Estrategia de Optimización | Evaluaciones | Tiempo (s) | Best CV F1 | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grid Search** | 54 | 30.1s | 0.6367 | 78.3% | 0.6288 | 0.8424 |
| **Random Search** | 25 | 13.9s | 0.6367 | 78.3% | 0.6288 | 0.8424 |
| **Bayesian (Optuna TPE)** | 25 | 53.0s | **0.6388** | **78.6%** | **0.6289** | **0.8412** |
| **NSGA-II (Multiobjetivo)** | 35 | 98.3s | 0.6348 | 73.5% | 0.6194 | 0.8373 |

### Variables más determinantes identificadas por SHAP:
1. `tenure` (Antigüedad del cliente: menor a 12 meses dispara el riesgo de fuga).
2. `InternetService_Fiber optic` (Clientes con fibra sin soporte técnico presentan alta deserción).
3. `Contract_Two year` / `Contract_One year` (Los contratos anuales actúan como el mayor escudo de retención).
4. `PaymentMethod_Electronic check` (Métodos de pago manuales tienen una correlación marcada con el churn).
5. `MonthlyCharges` / `TotalCharges` (Cargos mensuales elevados sin servicios de valor añadido).
