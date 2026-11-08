# Guía de Uso: Tarea 7 - RetenTel AI (Telco Churn ML, NSGA-III, XAI & Business Pitch)

---

##  Contenido del Proyecto

- **`app.py`**: Aplicación web interactiva en **Streamlit** con 5 módulos:
  1. *A. Análisis Exploratorio (EDA) & ADASYN*: Distribución de clases, impacto del remuestreo adaptativo y KPIs.
  2. *B. Desempeño y Pareto NSGA-III*: Comparativa de modelos (**Regresión Logística** vs **CatBoost**) bajo 4 objetivos de optimización, visualización del Frente de Pareto y calibración probabilística con **Brier Score**.
  3. *C. Predicción & Explicabilidad (XAI)*: Formulario interactivo, gauge de riesgo, desgloses **SHAP** y **LIME**, **Índice de Correlación** de top features e **Intervención Contrafactual (What-If)** prescriptiva.
  4. *D. Aplicación de Negocio & Pitch*: Simulador financiero de ROI interactivo (+440% ROI) y matriz prescriptiva.
  5. *E. Asistente IA RetenTel (Chatbot RAG Documental Guardrailed)*: Chatbot conversacional con guardrail estricto que **solo responde preguntas basadas en la documentación oficial del proyecto**, bloqueando consultas fuera de tema.
- **`train_and_evaluate.py`**: Script de entrenamiento, balanceo con ADASYN, optimización multiobjetivo con NSGA-III, cálculo de métricas y serialización de modelos.
- **`business_pitch.md`**: Propuesta de negocio ejecutiva (*RetenTel AI*), justificación financiera y guion de negociación comercial para el roleplay presencial.
- **`models/`**: Artefactos serializados (`champion_pipeline.joblib`, `preprocessor.joblib`, `metadata.json`).
- **`Telco-Customer-Churn.csv`**: Dataset de IBM / Kaggle con 7,043 clientes.

---

##  Opción A: Ejecución en Google Colab (Recomendada)
1. Abre [Google Colab](https://colab.research.google.com/).
2. Haz clic en **Subir (Upload)** y selecciona el archivo [`Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb`](file:///C:/Users/mayte.otavalo/Desktop/MAYTE/MACHINE/IA/CHURN/Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb).
3. En el menú superior de Colab, selecciona **Entorno de ejecución $\rightarrow$ Ejecutar todas**.
4. En la celda final de la **Parte 5**:
   - Se imprimirá tu dirección IP pública (ejemplo: `34.125.80.20`).
   - Se generará un enlace público de `localtunnel` (ejemplo: `https://famous-fox-42.loca.lt`).
   - Haz clic en el enlace, pega tu IP en el campo de contraseña ("Tunnel Password") y haz clic en "Submit".
   - ¡Tu aplicación Streamlit se abrirá en vivo e interactiva directamente en tu navegador!

---

## Opción B: Ejecución Local en tu Computadora
1. **Entrenar y verificar el modelo:**
   ```bash
   python train_and_evaluate.py
   ```
2. **Lanzar la aplicación interactiva de Streamlit:**
   ```bash
   streamlit run app.py
   ```
3. Se abrirá automáticamente en tu navegador en: `http://localhost:8501`.

---

## Resumen de Resultados Técnicos

### 1. Comparativa de Modelos Base (con Balanceo ADASYN)
| Modelo | Recall | F1-Score | F2-Score ($\beta=2$) | ROC-AUC | Brier Score | Latencia (ms / 1k) | Tamaño (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | 83.16% | 0.6385 | 0.7266 | 0.8396 | 0.1767 | ~0.5 ms | 0.8 KB |
| **CatBoost (Campeón NSGA-III)** | 71.39% | 0.6305 | **0.6780** | **0.8442** | **0.1467** | ~5.9 ms | 12.7 KB |

> [!NOTE]
> **Calibración con Brier Score:** El modelo CatBoost optimizado alcanza un Brier Score de **0.1467** (frente a 0.25 de azar), certificando que las probabilidades de fuga son consistentes y fiables para la toma de decisiones comerciales.

### 2. Optimización Multiobjetivo (NSGA-III)
Se optimizaron 4 objetivos simultáneos:
1. **Maximizar F2-Score** (priorización 2x de Recall para no dejar escapar clientes en fuga).
2. **Minimizar Latencia de Inferencia** (scoring en streaming < 6 ms por 1k).
3. **Minimizar Tamaño del Modelo** (< 13 KB para microservicios edge).
4. **Maximizar ROC-AUC** (discriminación global > 0.84).

### 3. Explicabilidad XAI e Índice de Correlación SHAP
- `Contract_Two year`: $r = -0.9929$ (Fuerte factor protector contra el churn).
- `tenure`: $r = -0.7750$ (Mayor antigüedad reduce drásticamente el riesgo).
- `InternetService_Fiber optic`: $r = +0.9972$ (Correlación directa con aumento de fuga sin soporte técnico).
- `PaymentMethod_Electronic check`: $r = +0.9936$ (Fricción en pagos manuales).
- `TechSupport_Yes`: $r = -0.9837$ (Blindaje de retención).

### 4. Motor Contrafactual (What-If)
Para clientes de alto riesgo (ej. 76.5% probabilidad de churn), el algoritmo contrafactual identifica que modificando el contrato a anual, bonificando soporte técnico y habilitando débito automático se reduce la probabilidad a **17.8%** (reducción neta de **58.8%**).

### 5. Chatbot RAG Guardrailed
Incluye un módulo conversacional con guardrail de seguridad estricto que **solo responde a preguntas soportadas por la documentación del proyecto** (métrica ROI, NSGA-III, ADASYN, Brier score, XAI), y devuelve un mensaje de advertencia claro ante consultas fuera de tema.
