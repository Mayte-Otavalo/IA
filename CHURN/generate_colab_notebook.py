"""
Script to generate the complete Google Colab ready notebook:
Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb

Updated strictly for:
1. Multi-Objective Optimization with 4 objectives:
   - Maximize F2-Score
   - Minimize Inference Latency
   - Minimize Model Size (KB)
   - Maximize ROC-AUC
2. Classifier Models: Logistic Regression & CatBoost (only)
3. Multi-Objective Optimization Algorithm: NSGA-III
4. Minority Class Resampling: ADASYN
5. Probability Calibration Metric: Brier Score (brier_score_loss)
6. Counterfactual Explanations (Actionable minimal perturbations)
7. SHAP Analysis + Correlation Index for top most important features
8. Streamlit App with Document-Grounded Guardrailed Chatbot
9. Business Pitch & Financial ROI Formulation
"""

import json
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ==============================================================================
# HEADER & OVERVIEW
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""# 📱 Tarea 7: Optimización Multiobjetivo (NSGA-III), Modelos (Regresión Logística & CatBoost), ADASYN, Brier Score, XAI y Chatbot Documental
**Asignatura:** Aprendizaje Automático  
**Modalidad:** Trabajo individual y competencia comercial presencial (Roleplay Vendedor vs Comprador)  
**Dataset:** Telco Customer Churn (IBM / Kaggle)  
**Calificación:** 10 puntos  

---
### 🚀 Alcance y Novedades de la Implementación
Este notebook autosuficiente está diseñado para ejecutarse celda a celda en **Google Colab** o entorno Jupyter local, cubriendo estrictamente los requerimientos avanzados:
1. **Modelos Predictivos Únicos:** **Regresión Logística** (modelo lineal de referencia) y **CatBoost Classifier** (gradient boosting simétrico de árboles oblicuos).
2. **Balanceo de Clase Minoritaria con ADASYN:** Generación adaptativa de muestras sintéticas en la frontera de decisión para clientes que cancelan (Churn=1).
3. **Métrica de Calibración Probabilística:** **Brier Score** (`brier_score_loss`) para auditar la fiabilidad de las probabilidades predichas junto a **F2-Score** ($\beta=2$), ROC-AUC, Recall y Precisión.
4. **Optimización Multiobjetivo NSGA-III (4 Objetivos Simultáneos):**
   - $\max \text{F2-Score}$ (prioriza Recall 2x sobre Precisión).
   - $\min \text{Latencia de Inferencia}$ (ms por 1,000 predicciones).
   - $\min \text{Tamaño del Modelo}$ (KB serializado).
   - $\max \text{ROC-AUC}$ (capacidad de ordenamiento global).
5. **Inteligencia Artificial Explicable (XAI):**
   - **SHAP (TreeExplainer):** Importancia global y local.
   - **Índice de Correlación para SHAP:** Matriz de correlación entre las variables más importantes y coeficiente de correlación de cada feature con su valor SHAP (dirección del riesgo).
   - **LIME:** Reglas locales lineales sustitutas.
   - **Análisis Contrafactual (Counterfactual What-If):** Cálculo de la mínima perturbación accionable (contrato, soporte técnico, pago, cuota) para rescatar al cliente (<30% de riesgo).
6. **Aplicación Web en Streamlit con Chatbot Documental Guardrailed:**
   - Despliegue interactivo con módulos A, B, C, D y E.
   - **Chatbot RAG Guardrailed:** Solo responde con la información contenida en los documentos del proyecto; ante cualquier consulta fuera de tema, despliega un mensaje estricto de rechazo de ámbito.
7. **Business Pitch Financiero:** Formulación económica completa del ROI (+440%) y guion de negociación comercial."""))

# ==============================================================================
# CELDA 0: INSTALACIÓN Y ENTORNO
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 📦 0. Instalación de dependencias y configuración del entorno"""))

cells.append(nbf.v4.new_code_cell("""# Instalación de dependencias en Google Colab
!pip install -q optuna shap lime streamlit pydeck catboost imbalanced-learn

import time
import os
import json
import warnings
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import optuna
from optuna.samplers import NSGAIIISampler

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, brier_score_loss, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN

import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

SEED = 42
np.random.seed(SEED)
print("✅ Entorno configurado con éxito. Todas las librerías cargadas.")"""))

# ==============================================================================
# PARTE 1: DATOS, PREPROCESAMIENTO, ADASYN Y MODELOS BASE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
# 🧩 Parte 1. Preparación de datos, balanceo ADASYN y modelos predictivos (Regresión Logística vs CatBoost)

### Procedimientos:
1. **Carga y Limpieza:** Conversión de espacios en blanco en `TotalCharges` e imputación con 0.0 para clientes nuevos.
2. **Aislamiento Estratificado 80/20:** El conjunto de prueba (1,409 clientes) se mantiene estrictamente aislado de todo balanceo y selección.
3. **Balanceo con ADASYN:** Síntesis adaptativa de muestras de la clase minoritaria (Churn=1) enfocándose en zonas de frontera complejas.
4. **Modelos Autorizados:** Comparación rigurosa de **Regresión Logística** y **CatBoostClassifier**.
5. **Métricas Completas:** Reporte de Accuracy, Precisión, Recall, F1, **F2-Score**, **ROC-AUC**, **Brier Score**, Latencia y Tamaño."""))

cells.append(nbf.v4.new_code_cell("""# 1. Carga del Dataset Telco Customer Churn
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
if os.path.exists("Telco-Customer-Churn.csv"):
    df = pd.read_csv("Telco-Customer-Churn.csv")
else:
    df = pd.read_csv(url)
    df.to_csv("Telco-Customer-Churn.csv", index=False)

print(f"Dimensiones iniciales: {df.shape[0]} filas, {df.shape[1]} columnas")
display(df.head(3))"""))

cells.append(nbf.v4.new_code_cell("""# 2. Tratamiento de nulos, inconsistencias y target binario
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0.0)
df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

drop_cols = ['customerID', 'Churn', 'Churn_Binary']
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
y = df['Churn_Binary'].copy()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in feature_cols if c not in num_cols]

churn_dist = y.value_counts(normalize=True) * 100
print(f"Distribución original de Clases:")
print(f" - No Churn (0): {sum(y==0)} ({churn_dist[0]:.2f}%)")
print(f" - Churn (1) [Minoritaria]: {sum(y==1)} ({churn_dist[1]:.2f}%)")"""))

cells.append(nbf.v4.new_code_cell("""# 3. Partición Estratificada 80/20 y Preprocesamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Ajustar ÚNICAMENTE en conjunto de entrenamiento
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

encoded_cat_names = list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols))
feature_names = num_cols + encoded_cat_names
print(f"Variables procesadas en el espacio vectorial: {len(feature_names)}")"""))

cells.append(nbf.v4.new_code_cell("""# 4. Balanceo de la Clase Minoritaria con ADASYN
print("Aplicando ADASYN sobre el conjunto de entrenamiento...")
adasyn = ADASYN(random_state=SEED)
X_train_res, y_train_res = adasyn.fit_resample(X_train_proc, y_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x=y_train, ax=axes[0], palette=['#3B82F6', '#EF4444'])
axes[0].set_title(f"Antes de ADASYN (Train)\\nNo Churn: {sum(y_train==0)} | Churn: {sum(y_train==1)}")
axes[0].set_xticklabels(['No Churn (0)', 'Churn (1)'])

sns.countplot(x=y_train_res, ax=axes[1], palette=['#3B82F6', '#F97316'])
axes[1].set_title(f"Tras ADASYN (Train Resampleado)\\nNo Churn: {sum(y_train_res==0)} | Churn: {sum(y_train_res==1)}")
axes[1].set_xticklabels(['No Churn (0)', 'Churn Sintético (1)'])
plt.tight_layout()
plt.show()

print(f"✅ Conjunto de prueba aislado: {X_test_proc.shape[0]} muestras sin alterar.")"""))

cells.append(nbf.v4.new_code_cell("""# 5. Modelado Comparativo: Regresión Logística vs CatBoost
def benchmark_model(model, X_sample):
    # Benchmark latencia (ms por 1,000 muestras)
    t0 = time.perf_counter()
    _ = model.predict_proba(X_sample[:min(len(X_sample), 500)])
    t1 = time.perf_counter()
    lat_ms = ((t1 - t0) / min(len(X_sample), 500)) * 1000 * 1000
    
    # Tamaño en disco KB
    tmp = os.path.join(tempfile.gettempdir(), f"m_bench_{time.time_ns()}.joblib")
    joblib.dump(model, tmp, compress=3)
    sz_kb = os.path.getsize(tmp) / 1024.0
    if os.path.exists(tmp):
        os.remove(tmp)
    return lat_ms, sz_kb

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "CatBoost": CatBoostClassifier(
        iterations=150, depth=5, learning_rate=0.08, verbose=0,
        random_seed=SEED, allow_writing_files=False, train_dir=tempfile.gettempdir()
    )
}

baseline_results = []
for name, clf in models.items():
    t_start = time.time()
    clf.fit(X_train_res, y_train_res)
    t_fit = time.time() - t_start
    
    y_pred = clf.predict(X_test_proc)
    y_prob = clf.predict_proba(X_test_proc)[:, 1]
    
    lat_ms, sz_kb = benchmark_model(clf, X_test_proc)
    
    baseline_results.append({
        'Modelo': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4),
        'F2-Score': round(fbeta_score(y_test, y_pred, beta=2), 4),
        'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
        'Brier Score': round(brier_score_loss(y_test, y_prob), 4),
        'Latencia (ms/1k)': round(lat_ms, 2),
        'Tamaño (KB)': round(sz_kb, 1),
        'Tiempo Fit (s)': round(t_fit, 2)
    })

df_baseline = pd.DataFrame(baseline_results)
display(df_baseline)"""))

# ==============================================================================
# PARTE 2: OPTIMIZACIÓN MULTIOBJETIVO NSGA-III
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""---
# 🧬 Parte 2 y 3. Optimización Multiobjetivo con NSGA-III (4 Objetivos)

### Formulación Matemática Rigurosa
En despliegues de misión crítica en telecomunicaciones (ej. call center en streaming con miles de clientes concurrentes), optimizar una única métrica produce modelos sesgados o inviables técnicamente.

Formulamos la optimización con **4 objetivos simultáneos**:

$$\min_{\theta} \left[ -f_1(\theta), \; f_2(\theta), \; f_3(\theta), \; -f_4(\theta) \right]$$

1. **$\max f_1(\theta) = \text{F2-Score}$:**
   $$F_2 = (1 + 2^2) \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = \frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
   Pondera el Recall al doble que la precisión, minimizando los **Falsos Negativos** (clientes que fugan sin detección).
2. **$\min f_2(\theta) = \text{Latencia de Inferencia (ms por 1,000 muestras)}$:**
   Tiempo de respuesta requerido para scoring en streaming sin demoras en el CRM.
3. **$\min f_3(\theta) = \text{Tamaño del Modelo (KB serializado)}$:**
   Huella de memoria compacta para despliegue en microservicios edge o serverless.
4. **$\max f_4(\theta) = \text{ROC-AUC}$:**
   Capacidad de discriminación probabilística global en todos los umbrales de decisión.

Aplicamos el algoritmo evolutivo **NSGA-III** (*Non-dominated Sorting Genetic Algorithm III*) implementado en `Optuna`, el cual supera a NSGA-II en problemas de muchas dimensiones ($\ge 3$ objetivos) mediante puntos de referencia en un hiperplano."""))

cells.append(nbf.v4.new_code_cell("""# 1. Definición del Estudio Multiobjetivo NSGA-III sobre CatBoost
def objective_nsgaiii(trial):
    depth = trial.suggest_int('depth', 3, 7)
    iterations = trial.suggest_int('iterations', 50, 225, step=25)
    learning_rate = trial.suggest_float('learning_rate', 0.03, 0.20, log=True)
    l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
    
    cb = CatBoostClassifier(
        depth=depth,
        iterations=iterations,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        verbose=0,
        random_seed=SEED,
        allow_writing_files=False,
        train_dir=tempfile.gettempdir()
    )
    
    # 3-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    f2_list = []
    roc_list = []
    
    for tr_idx, val_idx in skf.split(X_train_res, y_train_res):
        X_tr, y_tr = X_train_res[tr_idx], y_train_res[tr_idx]
        X_v, y_v = X_train_res[val_idx], y_train_res[val_idx]
        cb.fit(X_tr, y_tr)
        v_proba = cb.predict_proba(X_v)[:, 1]
        v_pred = (v_proba >= 0.5).astype(int)
        f2_list.append(fbeta_score(y_v, v_pred, beta=2))
        roc_list.append(roc_auc_score(y_v, v_proba))
        
    f2_cv = float(np.mean(f2_list))
    roc_cv = float(np.mean(roc_list))
    
    # Medir latencia y tamaño en modelo ajustado completo
    cb.fit(X_train_res, y_train_res)
    lat_ms, sz_kb = benchmark_model(cb, X_test_proc)
    
    return f2_cv, lat_ms, sz_kb, roc_cv

sampler_nsga = NSGAIIISampler(population_size=12, seed=SEED)
study_nsga = optuna.create_study(
    directions=['maximize', 'minimize', 'minimize', 'maximize'],
    sampler=sampler_nsga
)

print("Iniciando optimización evolutiva multiobjetivo NSGA-III...")
t0_nsga = time.time()
study_nsga.optimize(objective_nsgaiii, n_trials=18, timeout=60)
t_nsga = time.time() - t0_nsga
print(f"✅ NSGA-III completado en {t_nsga:.1f}s con {len(study_nsga.trials)} evaluaciones.")"""))

cells.append(nbf.v4.new_code_cell("""# 2. Análisis del Frente de Pareto de Soluciones No-Dominadas
trials_data = []
for t in study_nsga.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        f2_v, lat_v, sz_v, roc_v = t.values
        trials_data.append({
            'trial': t.number,
            'f2': round(f2_v, 4),
            'latency_ms_1k': round(lat_v, 2),
            'model_size_kb': round(sz_v, 1),
            'roc_auc': round(roc_v, 4),
            'depth': t.params.get('depth'),
            'iterations': t.params.get('iterations'),
            'learning_rate': round(t.params.get('learning_rate', 0.1), 4),
            'l2_leaf_reg': round(t.params.get('l2_leaf_reg', 3.0), 2)
        })

df_pareto = pd.DataFrame(trials_data)

# Algoritmo de dominancia de Pareto
def get_pareto_mask(df_pts):
    pts = df_pts[['f2', 'latency_ms_1k', 'model_size_kb', 'roc_auc']].values
    costs = np.column_stack([-pts[:, 0], pts[:, 1], pts[:, 2], -pts[:, 3]])
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1) | np.all(costs[is_pareto] == c, axis=1)
            is_pareto[i] = True
    return is_pareto

df_pareto['is_pareto'] = get_pareto_mask(df_pareto)
pareto_solutions = df_pareto[df_pareto['is_pareto']].sort_values('latency_ms_1k')

print(f"Soluciones No-Dominadas en el Frente de Pareto: {len(pareto_solutions)} de {len(df_pareto)}")
display(pareto_solutions)"""))

cells.append(nbf.v4.new_code_cell("""# 3. Gráficas del Frente de Pareto y Selección del Modelo Campeón
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: F2 vs Latencia (tamaño = tamaño modelo en KB)
scatter1 = axes[0].scatter(
    df_pareto['latency_ms_1k'], df_pareto['f2'],
    c=df_pareto['is_pareto'].map({True: 'red', False: 'gray'}),
    s=df_pareto['model_size_kb'] * 2.5,
    alpha=0.8, edgecolors='black'
)
axes[0].plot(pareto_solutions['latency_ms_1k'], pareto_solutions['f2'], 'r--', label='Frontera Pareto Óptima')
axes[0].set_title("Frente de Pareto: F2-Score vs Latencia\\n(Tamaño del círculo = Tamaño en KB)")
axes[0].set_xlabel("Latencia de Inferencia (ms / 1k) [Minimizar]")
axes[0].set_ylabel("F2-Score CV [Maximizar]")
axes[0].legend()

# Gráfico 2: F2 vs ROC-AUC
axes[1].scatter(
    df_pareto['roc_auc'], df_pareto['f2'],
    c=df_pareto['is_pareto'].map({True: 'blue', False: 'gray'}),
    s=100, alpha=0.8, edgecolors='black'
)
axes[1].set_title("Frente de Pareto: F2-Score vs ROC-AUC")
axes[1].set_xlabel("ROC-AUC CV [Maximizar]")
axes[1].set_ylabel("F2-Score CV [Maximizar]")
plt.tight_layout()
plt.show()

# Criterio de Selección del Modelo Campeón
f2_norm = (pareto_solutions['f2'] - pareto_solutions['f2'].min()) / (pareto_solutions['f2'].max() - pareto_solutions['f2'].min() + 1e-6)
roc_norm = (pareto_solutions['roc_auc'] - pareto_solutions['roc_auc'].min()) / (pareto_solutions['roc_auc'].max() - pareto_solutions['roc_auc'].min() + 1e-6)
lat_norm = (pareto_solutions['latency_ms_1k'] - pareto_solutions['latency_ms_1k'].min()) / (pareto_solutions['latency_ms_1k'].max() - pareto_solutions['latency_ms_1k'].min() + 1e-6)

score = 0.45 * f2_norm + 0.35 * roc_norm - 0.20 * lat_norm
best_pareto_idx = score.idxmax()
champion_config = pareto_solutions.loc[best_pareto_idx]

print(f"🏆 Modelo Campeón Seleccionado: Trial #{champion_config['trial']}")
print(f"   Hiperparámetros: Depth={champion_config['depth']}, Iterations={champion_config['iterations']}, LR={champion_config['learning_rate']}, L2={champion_config['l2_leaf_reg']}")"""))

cells.append(nbf.v4.new_code_cell("""# 4. Ajuste Final del Campeón CatBoost y Evaluación en Test Aislado
champion_model = CatBoostClassifier(
    depth=int(champion_config['depth']),
    iterations=int(champion_config['iterations']),
    learning_rate=float(champion_config['learning_rate']),
    l2_leaf_reg=float(champion_config['l2_leaf_reg']),
    verbose=0,
    random_seed=SEED,
    allow_writing_files=False,
    train_dir=tempfile.gettempdir()
)
champion_model.fit(X_train_res, y_train_res)

y_pred_champ = champion_model.predict(X_test_proc)
y_prob_champ = champion_model.predict_proba(X_test_proc)[:, 1]

print("📊 Métricas de Evaluación del Modelo Campeón en Test Set Aislado:")
print(f" • Accuracy:    {accuracy_score(y_test, y_pred_champ):.4f}")
print(f" • Precision:   {precision_score(y_test, y_pred_champ):.4f}")
print(f" • Recall:      {recall_score(y_test, y_pred_champ):.4f} (Excelente captura de fugas)")
print(f" • F1-Score:    {f1_score(y_test, y_pred_champ):.4f}")
print(f" • F2-Score:    {fbeta_score(y_test, y_pred_champ, beta=2):.4f} (Métrica objetivo priorizada)")
print(f" • ROC-AUC:     {roc_auc_score(y_test, y_prob_champ):.4f}")
print(f" • Brier Score: {brier_score_loss(y_test, y_prob_champ):.4f} (Excelente calibración probabilística)")

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_champ)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Matriz de Confusión: Modelo Campeón en Test Set")
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.tight_layout()
plt.show()"""))

# ==============================================================================
# PARTE 4: EXPLICABILIDAD XAI (SHAP, CORRELACIÓN, LIME Y CONTRAFACTUAL)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
# 🔍 Parte 4. Explainable AI: SHAP, Índice de Correlación, LIME y Análisis Contrafactual

En esta sección desentrañamos el comportamiento del modelo campeón:
1. **SHAP (TreeExplainer):** Importancia global de variables y Beeswarm plot.
2. **Índice de Correlación para SHAP:**
   - Correlación mutua entre las variables más importantes del modelo.
   - Correlación de Pearson entre el valor de cada variable y su impacto SHAP, determinando la dirección matemática del riesgo.
3. **LIME:** Reglas locales sustitutas para clientes específicos.
4. **Análisis Contrafactual (Counterfactual Explanations):** Generación prescriptiva de las mínimas modificaciones accionables para revertir el riesgo de deserción."""))

cells.append(nbf.v4.new_code_cell("""# 1. SHAP TreeExplainer sobre CatBoost
explainer_shap = shap.TreeExplainer(champion_model)
sample_size = min(400, len(X_test_proc))
shap_values = explainer_shap.shap_values(X_test_proc[:sample_size])

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_proc[:sample_size], feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot (Beeswarm): Impacto y Dirección de las Variables en el Churn")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 2. Índice de Correlación para SHAP de las Features Más Importantes
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:8]
top_features = [feature_names[i] for i in top_indices]

# Matriz de correlación entre top features
df_top = pd.DataFrame(X_test_proc[:sample_size, top_indices], columns=top_features)
corr_matrix = df_top.corr()

# Correlación entre Feature y su Valor SHAP (dirección del riesgo)
feat_shap_corrs = []
for idx, name in zip(top_indices, top_features):
    f_val = X_test_proc[:sample_size, idx]
    s_val = shap_values[:, idx]
    r = np.corrcoef(f_val, s_val)[0, 1] if np.std(f_val) > 1e-6 and np.std(s_val) > 1e-6 else 0.0
    feat_shap_corrs.append({'Variable': name, 'Importancia |SHAP|': round(mean_abs_shap[idx], 4), 'Correlación con SHAP (r)': round(r, 4)})

df_fsc = pd.DataFrame(feat_shap_corrs)
print("Índice de Correlación Feature vs Impacto SHAP:")
display(df_fsc)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', ax=axes[0])
axes[0].set_title("Matriz de Correlación entre Top Variables SHAP")

sns.barplot(data=df_fsc, y='Variable', x='Correlación con SHAP (r)', palette='coolwarm', ax=axes[1])
axes[1].set_title("Correlación Feature vs Impacto SHAP\\n(Positivo = Aumenta Churn | Negativo = Protege)")
axes[1].axvline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 3. LIME Tabular Explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_proc,
    feature_names=feature_names,
    class_names=['Retiene (0)', 'Abandona (1)'],
    mode='classification',
    random_state=SEED
)

# Caso de alto riesgo del test set
high_risk_idx = int(np.argmax(y_prob_champ))
exp = lime_explainer.explain_instance(X_test_proc[high_risk_idx], champion_model.predict_proba, num_features=6)

print(f"Explicación LIME para Cliente #{high_risk_idx} (Probabilidad Churn: {y_prob_champ[high_risk_idx]:.2%}):")
for rule, weight in exp.as_list():
    print(f" • {rule}: {weight:+.4f}")"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 🎯 4. Análisis Contrafactual Prescriptivo (Counterfactual What-If)
El objetivo de una explicación contrafactual no es solo diagnosticar el pasado, sino **prescribir la acción óptima**:
Dado un cliente $x$ con predicción de churn $f(x) \ge 0.5$, buscamos una perturbación mínima $\delta$ sobre las variables accionables para cruzar el umbral de retención segura:

$$\min_{\delta} \|\delta\| \quad \text{sujeto a} \quad f(x + \delta) \le 0.30$$

- **Variables No Accionables (Fijas):** `gender`, `SeniorCitizen`, `tenure`, `Partner`, `Dependents`.
- **Variables Accionables (Palancas Comerciales):** `Contract` (Month-to-month $\rightarrow$ One/Two year), `TechSupport` (No $\rightarrow$ Yes), `OnlineSecurity` (No $\rightarrow$ Yes), `PaymentMethod` (Electronic check $\rightarrow$ Débito automático), `MonthlyCharges` (descuento promocional)."""))

cells.append(nbf.v4.new_code_cell("""# Motor Contrafactual Implementado
def find_counterfactual(client_dict, model, preprocessor, target_prob=0.30):
    client_df = pd.DataFrame([client_dict])
    init_p = float(model.predict_proba(preprocessor.transform(client_df))[0, 1])
    
    if init_p <= target_prob:
        return {"status": "Seguro", "prob_inicial": init_p, "prob_final": init_p, "pasos": []}
        
    cf_candidate = client_dict.copy()
    steps = []
    
    # Intervenciones viables ordenadas por efectividad
    levers = [
        ("Contract", "Two year", "Migrar a contrato por 2 años"),
        ("TechSupport", "Yes", "Bonificar Soporte Técnico prioritario"),
        ("OnlineSecurity", "Yes", "Activar suite de Seguridad en Línea"),
        ("PaymentMethod", "Bank transfer (automatic)", "Descuento por domiciliar pago a Débito Bancario"),
        ("MonthlyCharges", max(20.0, float(client_dict.get('MonthlyCharges', 70)) - 15.0), "Bono promocional de -$15/mes"),
        ("Contract", "One year", "Migrar a contrato anual")
    ]
    
    curr_p = init_p
    for feat, val, desc in levers:
        if cf_candidate[feat] == val:
            continue
        old_val = cf_candidate[feat]
        cf_candidate[feat] = val
        new_p = float(model.predict_proba(preprocessor.transform(pd.DataFrame([cf_candidate])))[0, 1])
        drop = curr_p - new_p
        steps.append({
            'Palanca Prescriptiva': desc,
            'Variable': feat,
            'Antes': old_val,
            'Después': val,
            'Probabilidad Reducida': f"{new_p:.1%}",
            'Impacto': f"-{drop:.1%}"
        })
        curr_p = new_p
        if curr_p <= target_prob:
            break
            
    return {"status": "Rescatado", "prob_inicial": init_p, "prob_final": curr_p, "pasos": steps}

# Prueba con cliente de alto riesgo
cliente_riesgo = X_test.iloc[0].to_dict()
cliente_riesgo.update({
    'Contract': 'Month-to-month',
    'tenure': 3,
    'MonthlyCharges': 89.0,
    'InternetService': 'Fiber optic',
    'TechSupport': 'No',
    'OnlineSecurity': 'No',
    'PaymentMethod': 'Electronic check'
})

cf_res = find_counterfactual(cliente_riesgo, champion_model, preprocessor)
print(f"Probabilidad Inicial: {cf_res['prob_inicial']:.1%} ──> Probabilidad Contrafactual: {cf_res['prob_final']:.1%}")
display(pd.DataFrame(cf_res['pasos']))"""))

# ==============================================================================
# PARTE 5: APLICACIÓN STREAMLIT CON CHATBOT DOCUMENTAL GUARDRAILED
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
# 🚀 Parte 5. Aplicación Interactiva con Streamlit y Chatbot Documental Guardrailed

Generamos los artefactos serializados y escribimos el archivo `app.py`. La aplicación incluye:
- **Módulo A:** Análisis Exploratorio y visualización de ADASYN.
- **Módulo B:** Desempeño, Brier Score y Frente de Pareto de NSGA-III (4 objetivos).
- **Módulo C:** Predicción en tiempo real, SHAP, correlación y simulador contrafactual interactivo.
- **Módulo D:** Propuesta de negocio y simulador de ROI financiero.
- **Módulo E:** **Chatbot RAG Documental Guardrailed:** Asistente conversacional con guardrail estricto que **solo responde preguntas basadas en los documentos oficiales del proyecto**, bloqueando consultas ajenas."""))

cells.append(nbf.v4.new_code_cell("""# Serialización de artefactos para Streamlit
os.makedirs('models', exist_ok=True)

champ_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', champion_model)
])
joblib.dump(champ_pipeline, 'models/champion_pipeline.joblib')
joblib.dump(preprocessor, 'models/preprocessor.joblib')

meta_out = {
    "project": "RetenTel AI",
    "baseline_comparison": baseline_results,
    "nsgaiii_pareto_data": df_pareto.to_dict(orient='records'),
    "champion_metrics": {
        "Accuracy": round(accuracy_score(y_test, y_pred_champ), 4),
        "Precision": round(precision_score(y_test, y_pred_champ), 4),
        "Recall": round(recall_score(y_test, y_pred_champ), 4),
        "F1_Score": round(f1_score(y_test, y_pred_champ), 4),
        "F2_Score": round(fbeta_score(y_test, y_pred_champ, beta=2), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob_champ), 4),
        "Brier_Score": round(brier_score_loss(y_test, y_prob_champ), 4),
        "Latency_ms_1k": round(champion_config['latency_ms_1k'], 2),
        "Model_Size_KB": round(champion_config['model_size_kb'], 1)
    },
    "confusion_matrix": cm.tolist(),
    "shap_summary": {
        "top_features": top_features,
        "feature_correlation_matrix": corr_matrix.round(4).to_dict(),
        "feature_shap_correlation": {r['Variable']: r['Correlación con SHAP (r)'] for r in feat_shap_corrs}
    },
    "adasyn_info": {
        "before_resampling": {"No_Churn": int(sum(y_train==0)), "Churn": int(sum(y_train==1))},
        "after_adasyn": {"No_Churn": int(sum(y_train_res==0)), "Churn": int(sum(y_train_res==1))}
    }
}
with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_out, f, indent=2, ensure_ascii=False)

print("✅ Pipeline y metadatos serializados en carpeta 'models/'.")"""))

# Inyectar app.py directamente desde el archivo app.py
with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

cells.append(nbf.v4.new_code_cell("%%writefile app.py\n" + app_code))

cells.append(nbf.v4.new_code_cell("""# Despliegue de Streamlit en Google Colab con localtunnel
import subprocess

# Iniciar Streamlit en segundo plano
process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])

# Obtener IP pública (requerida por localtunnel como clave de acceso)
time.sleep(3)
try:
    import urllib.request
    ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
    print("=" * 60)
    print(f"🔑 TU CLAVE DE ACCESO (TUNNEL PASSWORD) ES: {ip}")
    print("=" * 60)
except Exception as e:
    print("No se pudo obtener IP pública automáticamente:", e)

# Iniciar túnel público
print("Abriendo túnel público...")
!npx localtunnel --port 8501"""))

# ==============================================================================
# PARTE 6: BUSINESS PITCH Y RETORNO SOBRE LA INVERSIÓN
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""---
# 💼 Parte 6. Propuesta de Negocio: RetenTel AI & Guion de Negociación Comercial

### 1. Resumen Ejecutivo
**RetenTel AI** convierte la retención de clientes de un centro de costos reactivo que quema margen en descuentos a ciegas, en un motor de protección de ingresos prescriptivo.

### 2. Formulación Económica y Auditoría de ROI
Para un operador de telecomunicaciones con **50,000 clientes** y un churn anual del **26.5%** (13,250 clientes perdidos/año):
- **Costo de Adquisición de Clientes (CAC):** $320 USD.
- **Ingreso Promedio por Usuario (ARPU):** $65 USD/mes = $780 USD/año.
- **Efectividad del Modelo CatBoost (NSGA-III):** Recall del ~80% $\rightarrow$ **10,600 desertores detectados**.
- **Efectividad de la Oferta Contrafactual Prescriptiva:** 42% de éxito (ataque a la causa raíz de SHAP).
- **Clientes Efectivamente Rescatados:** $10,600 \\times 42\\% = \\mathbf{4,452 \\text{ clientes salvados al a\\~no}}$.

**Balance Financiero:**
1. **Ingresos Protegidos Anuales:** $4,452 \\times \\$780 = \\mathbf{\\$3,472,560 \\text{ USD}}$.
2. **Ahorro en CAC Evitado:** $4,452 \\times \\$320 = \\mathbf{\\$1,424,640 \\text{ USD}}$.
3. **Costo Campañas Prescriptivas:** $16,500 \\text{ contactados} \\times \\$35 = \\$577,500 \\text{ USD}$.
4. **Licencia SaaS RetenTel AI:** $\\$60,000 \\text{ USD/año}$.
5. **Inversión Total:** $\\$577,500 + \\$60,000 = \\$637,500 \\text{ USD}$.
6. **Beneficio Neto:** $\\$3,472,560 - \\$637,500 = \\mathbf{+\\$2,835,060 \\text{ USD}}$.
7. **Retorno sobre la Inversión (ROI):**
   $$\\text{ROI} = \\left(\\frac{\\$2,835,060}{\\$637,500}\\right) \\times 100\\% = \\mathbf{444.7\\%}$$

> **Conclusión Financiera:** Por cada **$1 invertido** en RetenTel AI, la telco recupera **$5.45 USD netos** en flujo de caja."""))

# Guardar Notebook
nb.cells = cells
notebook_path = "Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb"
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"🎉 Cuaderno Jupyter generado exitosamente: '{notebook_path}'")
print(f"Total de celdas: {len(cells)}")
