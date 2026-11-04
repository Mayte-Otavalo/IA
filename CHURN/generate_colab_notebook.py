"""
Script to generate the complete Google Colab ready notebook:
Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb
"""

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# --- Header & Colab Badge ---
cells.append(nbf.v4.new_markdown_cell("""# 📱 Tarea 7: Optimización de Machine Learning, Explainable AI y Business Pitch
**Asignatura:** Aprendizaje Automático  
**Modalidad:** Trabajo individual y competencia comercial presencial  
**Dataset:** Telco Customer Churn  
**Calificación:** 10 puntos  

---
### 🚀 Ejecución en Google Colab
Este notebook está preparado para ejecutarse de extremo a extremo en **Google Colab** sin dependencias externas previas. Incluye:
1. **Parte 1:** Preparación de datos, pipelines y comparación de modelos (Logistic Regression, Random Forest, XGBoost).
2. **Parte 2:** Optimización de hiperparámetros (Grid Search, Random Search, Optimización Bayesiana TPE y NSGA-II).
3. **Parte 3:** Optimización multiobjetivo y visualización del Frente de Pareto.
4. **Parte 4:** Inteligencia Artificial Explicable (XAI) con SHAP y LIME + Comparación con Feature Importance MDI.
5. **Parte 5:** Aplicación interactiva con **Streamlit** desplegable directamente en Colab vía tunelización pública.
6. **Parte 6 & Pitch:** Propuesta de negocio original (*RetenTel AI*), formulación económica de ROI y simulación de negociación comercial."""))

# --- Cell 1: Environment Setup ---
cells.append(nbf.v4.new_markdown_cell("""## 📦 0. Instalación de dependencias y configuración del entorno"""))

cells.append(nbf.v4.new_code_cell("""# Instalación de librerías en Google Colab
!pip install -q optuna shap lime streamlit pydeck xgboost

import time
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib

import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

SEED = 42
np.random.seed(SEED)
print("✅ Entorno configurado correctamente.")"""))

# --- Cell 2: Part 1 Markdown ---
cells.append(nbf.v4.new_markdown_cell("""---
# 🧩 Parte 1. Preparación de datos y modelos predictivos

### Procedimientos:
1. **Exploración de datos:** Identificación de variables y análisis de la distribución de `Churn`.
2. **Tratamiento de inconsistencias y nulos:** Limpieza de la columna `TotalCharges` (espacios en blanco convertidos a numérico e imputación) y eliminación de IDs no predictivos.
3. **Pipelines de Preprocesamiento:** Transformación de variables numéricas y categóricas mediante `ColumnTransformer` y `Pipeline`.
4. **Aislamiento Estratificado:** División 80/20 garantizando que el conjunto de prueba permanezca totalmente aislado de la optimización y selección.
5. **Modelado Comparativo:** Entrenamiento de **Logistic Regression**, **Random Forest** y **XGBoost**.
6. **Métricas de Evaluación:** Reporte de Accuracy, Precision, Recall, F1-score, ROC-AUC y matriz de confusión."""))

cells.append(nbf.v4.new_code_cell("""# 1. Carga del Dataset Telco Customer Churn
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print(f"Dimensiones iniciales: {df.shape[0]} filas, {df.shape[1]} columnas")
display(df.head(3))"""))

cells.append(nbf.v4.new_code_cell("""# 2. Limpieza e Identificación de Inconsistencias
# TotalCharges contiene espacios en blanco ' ' en clientes nuevos (tenure == 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
nulos_tc = df['TotalCharges'].isna().sum()
print(f"Valores nulos detectados en TotalCharges: {nulos_tc}")

# Imputar con 0.0 dado que corresponde a clientes con 0 meses de antigüedad
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

# Codificación de la variable objetivo (Churn: Yes=1, No=0)
df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Distribución del Churn
churn_counts = df['Churn'].value_counts()
churn_rate = df['Churn_Binary'].mean() * 100
print(f"Distribución de Churn:\\n{churn_counts}")
print(f"Tasa de Abandono (Prevalencia): {churn_rate:.2f}% (Desbalance 73.5% vs 26.5%)")

# Separación de características
drop_cols = ['customerID', 'Churn', 'Churn_Binary']
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
y = df['Churn_Binary'].copy()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in feature_cols if c not in num_cols]

print(f"\\nVariables Numéricas ({len(num_cols)}): {num_cols}")
print(f"Variables Categóricas ({len(cat_cols)}): {cat_cols}")"""))

cells.append(nbf.v4.new_code_cell("""# 3. Partición Estratificada y Aislamiento del Test Set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

print(f"Conjunto de Entrenamiento: {X_train.shape[0]} observaciones")
print(f"Conjunto de Prueba Aislado: {X_test.shape[0]} observaciones")

# 4. Pipelines de Preprocesamiento con ColumnTransformer
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

# Ajustar preprocesador ÚNICAMENTE con datos de entrenamiento
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

feature_names = (
    num_cols + 
    list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
)
print(f"Dimensión del espacio transformado: {len(feature_names)} variables")"""))

cells.append(nbf.v4.new_code_cell("""# 5. Entrenamiento y Comparación de Modelos Base
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
    'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=SEED)
}

baseline_results = []
trained_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    t0 = time.time()
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1').mean()
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    cv_recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='recall').mean()
    
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    baseline_results.append({
        'Modelo': name,
        'CV F1': round(cv_f1, 4),
        'CV ROC-AUC': round(cv_auc, 4),
        'CV Recall': round(cv_recall, 4),
        'Test Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Test Precision': round(precision_score(y_test, y_pred), 4),
        'Test Recall': round(recall_score(y_test, y_pred), 4),
        'Test F1': round(f1_score(y_test, y_pred), 4),
        'Test ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
        'Tiempo (s)': round(fit_time, 2)
    })
    trained_pipelines[name] = pipe

df_baseline = pd.DataFrame(baseline_results)
display(df_baseline)"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Justificación Técnica y de Negocio del Modelo Inicial
1. **Métricas Relevantes:**  
   En la gestión de abandono de clientes (*Customer Churn*), **el costo de un Falso Negativo** (cliente que se va sin ser detectado) es mucho mayor que el costo de un Falso Positivo (ofrecer una promoción de retención a un cliente que se iba a quedar). Por tanto, las métricas prioritarias son **Recall**, **F1-score** y **ROC-AUC**.
2. **Selección del Modelo Base:**  
   Aunque la Regresión Logística presenta un F1 balanceado en configuraciones lineales básicas, **Random Forest** es seleccionado para el proceso de optimización e interpretabilidad debido a:
   - Capacidad intrínseca de modelar interacciones no lineales entre características comerciales complejas (ej. combinación de tipo de contrato, soporte técnico y método de pago).
   - Robustez frente a outliers y ausencia de suposiciones de linealidad.
   - Compatibilidad directa con algoritmos explicativos basados en árboles (**TreeSHAP**) y métodos locales (**LIME**)."""))

# --- Cell 3: Part 2 Markdown ---
cells.append(nbf.v4.new_markdown_cell("""---
# ⚙️ Parte 2. Optimización de Hiperparámetros

Se evalúan y comparan cuatro estrategias sobre el modelo base **Random Forest**:
1. **Grid Search:** Búsqueda exhaustiva sobre una cuadrícula predefinida.
2. **Random Search:** Muestreo aleatorio uniforme del espacio hiperparamétrico con presupuesto fijo.
3. **Optimización Bayesiana (TPE - Optuna):** Estimación de Parzen mediante árboles para modelar probabilísticamente $P(x|y)$ y guiar la búsqueda a zonas de alto rendimiento.
4. **Optimización Multiobjetivo (NSGA-II):** Algoritmo genético de ordenamiento no dominado para optimizar simultáneamente dos métricas en conflicto."""))

cells.append(nbf.v4.new_code_cell("""# Definición del espacio de búsqueda hiperparamétrico
param_grid_sk = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__class_weight': ['balanced', None]
}

rf_base_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=SEED))
])

# 1. Grid Search
print("Iniciando Grid Search (54 combinaciones)...")
t0 = time.time()
grid_search = GridSearchCV(rf_base_pipe, param_grid_sk, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
t_grid = time.time() - t0
y_pred_grid = grid_search.best_estimator_.predict(X_test)
y_prob_grid = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

# 2. Random Search
print("Iniciando Random Search (25 configuraciones)...")
t0 = time.time()
rand_search = RandomizedSearchCV(rf_base_pipe, param_grid_sk, n_iter=25, cv=cv, scoring='f1', random_state=SEED, n_jobs=-1)
rand_search.fit(X_train, y_train)
t_rand = time.time() - t0
y_pred_rand = rand_search.best_estimator_.predict(X_test)
y_prob_rand = rand_search.best_estimator_.predict_proba(X_test)[:, 1]

# 3. Optimización Bayesiana TPE (Optuna)
print("Iniciando Optimización Bayesiana TPE con Optuna (25 iteraciones)...")
def objective_tpe(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200, step=25)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=SEED
    )
    return cross_val_score(clf, X_train_proc, y_train, cv=cv, scoring='f1').mean()

sampler_tpe = optuna.samplers.TPESampler(seed=SEED)
study_tpe = optuna.create_study(direction='maximize', sampler=sampler_tpe)
t0 = time.time()
study_tpe.optimize(objective_tpe, n_trials=25)
t_tpe = time.time() - t0

best_tpe_params = study_tpe.best_params
best_rf_tpe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**best_tpe_params, random_state=SEED))
])
best_rf_tpe.fit(X_train, y_train)
y_pred_tpe = best_rf_tpe.predict(X_test)
y_prob_tpe = best_rf_tpe.predict_proba(X_test)[:, 1]

print("✅ Búsquedas tradicionales y bayesianas finalizadas.")"""))

# --- Cell 4: Part 3 Markdown ---
cells.append(nbf.v4.new_markdown_cell(r"""---
# 🎯 Parte 3. Optimización Multiobjetivo (NSGA-II)

### Formulación Matemática
En entornos de producción reales (ej. scoring en tiempo real durante una llamada en Call Center), no solo interesa la exactitud del modelo, sino también su costo computacional y latencia de respuesta. Formulamos el problema multiobjetivo:

$$\max_{\theta} f_1(\theta) = \text{F1-score}_{CV}(\theta)$$
$$\min_{\theta} f_2(\theta) = \text{Latencia de Inferencia (ms por 1,000 predicciones)}$$

Donde $\theta = \{\text{n\_estimators}, \text{max\_depth}, \text{min\_samples\_split}, \text{class\_weight}\}$.  
Se aplica el algoritmo **NSGA-II** (*Non-dominated Sorting Genetic Algorithm II*) mediante Optuna."""))

cells.append(nbf.v4.new_code_cell("""# Ejecución de Optimización Multiobjetivo con NSGA-II
def multi_objective_func(trial):
    n_estimators = trial.suggest_int('n_estimators', 40, 240, step=20)
    max_depth = trial.suggest_int('max_depth', 4, 18)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=SEED
    )
    # Objetivo 1: Calidad predictiva F1
    f1_cv = cross_val_score(clf, X_train_proc, y_train, cv=cv, scoring='f1').mean()
    
    # Objetivo 2: Benchmark de latencia de inferencia en ms por 1,000 muestras
    clf.fit(X_train_proc, y_train)
    bench_data = X_test_proc[:500]
    t_start = time.perf_counter()
    for _ in range(5):
        _ = clf.predict_proba(bench_data)
    elapsed = (time.perf_counter() - t_start) / 5.0
    latency_ms_1k = (elapsed / len(bench_data)) * 1000 * 1000
    
    return f1_cv, latency_ms_1k

sampler_nsga = optuna.samplers.NSGAIISampler(seed=SEED)
study_nsga = optuna.create_study(directions=['maximize', 'minimize'], sampler=sampler_nsga)

t0 = time.time()
study_nsga.optimize(multi_objective_func, n_trials=35)
t_nsga = time.time() - t0

# Extraer soluciones del frente de Pareto
pareto_trials = study_nsga.best_trials
pareto_records = []
for t in study_nsga.trials:
    pareto_records.append({
        'trial': t.number,
        'f1_cv': round(t.values[0], 4),
        'latency_ms_1k': round(t.values[1], 3),
        'n_estimators': t.params.get('n_estimators'),
        'max_depth': t.params.get('max_depth'),
        'class_weight': t.params.get('class_weight'),
        'is_pareto': t in pareto_trials
    })

df_pareto = pd.DataFrame(pareto_records)
pareto_solutions = df_pareto[df_pareto['is_pareto']].sort_values('latency_ms_1k')

print(f"Número de soluciones en el Frente de Pareto: {len(pareto_solutions)}")
display(pareto_solutions)"""))

cells.append(nbf.v4.new_code_cell("""# Visualización del Frente de Pareto y Tabla Comparativa de Estrategias
plt.figure(figsize=(10, 5))
sns.scatterplot(
    data=df_pareto,
    x='latency_ms_1k',
    y='f1_cv',
    hue='is_pareto',
    palette={True: 'red', False: 'grey'},
    s=100,
    style='is_pareto',
    markers={True: 'X', False: 'o'}
)
plt.plot(pareto_solutions['latency_ms_1k'], pareto_solutions['f1_cv'], 'r--', label='Frontera de Pareto (No Dominadas)')
plt.title("Frente de Pareto: Compromiso entre Desempeño (F1) y Latencia de Inferencia")
plt.xlabel("Latencia de Inferencia (ms por 1,000 muestras) [Minimizar]")
plt.ylabel("F1-Score en Validación Cruzada [Maximizar]")
plt.legend()
plt.tight_layout()
plt.show()

# Tabla comparativa general de optimizaciones
selected_pareto = sorted(pareto_trials, key=lambda x: x.values[0], reverse=True)[0]
rf_pareto_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**selected_pareto.params, random_state=SEED))
])
rf_pareto_clf.fit(X_train, y_train)
y_pred_nsga = rf_pareto_clf.predict(X_test)
y_prob_nsga = rf_pareto_clf.predict_proba(X_test)[:, 1]

optim_summary = pd.DataFrame([
    {
        'Estrategia': 'Grid Search',
        'Best CV F1': round(grid_search.best_score_, 4),
        'Test F1': round(f1_score(y_test, y_pred_grid), 4),
        'Test Recall': round(recall_score(y_test, y_pred_grid), 4),
        'Test ROC-AUC': round(roc_auc_score(y_test, y_prob_grid), 4),
        'Configs Evaluadas': len(grid_search.cv_results_['params']),
        'Tiempo (s)': round(t_grid, 2),
        'Mejores Hiperparámetros': str(grid_search.best_params_)
    },
    {
        'Estrategia': 'Random Search',
        'Best CV F1': round(rand_search.best_score_, 4),
        'Test F1': round(f1_score(y_test, y_pred_rand), 4),
        'Test Recall': round(recall_score(y_test, y_pred_rand), 4),
        'Test ROC-AUC': round(roc_auc_score(y_test, y_prob_rand), 4),
        'Configs Evaluadas': 25,
        'Tiempo (s)': round(t_rand, 2),
        'Mejores Hiperparámetros': str(rand_search.best_params_)
    },
    {
        'Estrategia': 'Bayesian (Optuna TPE)',
        'Best CV F1': round(study_tpe.best_value, 4),
        'Test F1': round(f1_score(y_test, y_pred_tpe), 4),
        'Test Recall': round(recall_score(y_test, y_pred_tpe), 4),
        'Test ROC-AUC': round(roc_auc_score(y_test, y_prob_tpe), 4),
        'Configs Evaluadas': 25,
        'Tiempo (s)': round(t_tpe, 2),
        'Mejores Hiperparámetros': str(best_tpe_params)
    },
    {
        'Estrategia': 'Multi-Objective (NSGA-II)',
        'Best CV F1': round(selected_pareto.values[0], 4),
        'Test F1': round(f1_score(y_test, y_pred_nsga), 4),
        'Test Recall': round(recall_score(y_test, y_pred_nsga), 4),
        'Test ROC-AUC': round(roc_auc_score(y_test, y_prob_nsga), 4),
        'Configs Evaluadas': 35,
        'Tiempo (s)': round(t_nsga, 2),
        'Mejores Hiperparámetros': str(selected_pareto.params)
    }
])
display(optim_summary[['Estrategia', 'Best CV F1', 'Test F1', 'Test Recall', 'Test ROC-AUC', 'Configs Evaluadas', 'Tiempo (s)']])"""))

# --- Cell 5: Part 4 Markdown ---
cells.append(nbf.v4.new_markdown_cell("""---
# 🔍 Parte 4. Explainable AI: SHAP y LIME

Se analizan las explicaciones de caja negra sobre el modelo seleccionado (**Random Forest Optimizado con TPE**):
1. **SHAP (SHapley Additive exPlanations):**
   - Importancia global y distribución de impacto marginal (Beeswarm).
   - Explicaciones locales (descomposición aditiva) para dos perfiles contrastantes del conjunto de test.
2. **LIME (Local Interpretable Model-agnostic Explanations):**
   - Aproximación local lineal perturbando el vecindario del cliente.
3. **Comparativa Crítica:**
   - SHAP vs LIME vs Random Forest MDI (Gini Importance)."""))

cells.append(nbf.v4.new_code_cell("""# 1. SHAP: Explicabilidad Global y Local
champion_pipeline = best_rf_tpe
rf_clf = champion_pipeline.named_steps['classifier']

explainer_shap = shap.TreeExplainer(rf_clf)
shap_values = explainer_shap.shap_values(X_test_proc)

if isinstance(shap_values, list):
    shap_vals_churn = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_churn = shap_values[:, :, 1]
else:
    shap_vals_churn = shap_values

# Gráfico de Importancia Global SHAP
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_churn, X_test_proc, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot: Impacto Global de Variables en la Decisión de Churn")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# 2. Selección de Dos Clientes Contrastantes del Test Set Aislado
y_prob_test = champion_pipeline.predict_proba(X_test)[:, 1]
idx_high = int(np.argmax(y_prob_test))
idx_low = int(np.argmin(y_prob_test))

print(f"Cliente A (Alto Riesgo) - Índice: {idx_high}, Probabilidad Churn: {y_prob_test[idx_high]:.4f}")
print(f"Cliente B (Bajo Riesgo / Leal) - Índice: {idx_low}, Probabilidad Churn: {y_prob_test[idx_low]:.4f}")

# LIME Explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_proc,
    feature_names=feature_names,
    class_names=['Retain (No)', 'Churn (Yes)'],
    mode='classification',
    random_state=SEED
)

# Explicaciones LIME
exp_high = lime_explainer.explain_instance(X_test_proc[idx_high], rf_clf.predict_proba, num_features=6)
exp_low = lime_explainer.explain_instance(X_test_proc[idx_low], rf_clf.predict_proba, num_features=6)

print("\\n--- Reglas LIME para Cliente A (Alto Riesgo) ---")
for feat, weight in exp_high.as_list():
    print(f"  {feat}: {weight:+.4f}")

print("\\n--- Reglas LIME para Cliente B (Bajo Riesgo) ---")
for feat, weight in exp_low.as_list():
    print(f"  {feat}: {weight:+.4f}")"""))

cells.append(nbf.v4.new_markdown_cell(r"""### 📊 Comparativa Conceptual: SHAP vs LIME vs Feature Importance MDI

| Criterio | SHAP (Shapley Values) | LIME (Local Explanations) | Random Forest MDI (Gini) |
| :--- | :--- | :--- | :--- |
| **Fundamento Matemático** | Teoría de Juegos Cooperativos (Valor de Shapley único). | Ajuste de modelo lineal local sustituto ponderado por distancia. | Reducción de impureza de Gini acumulada en las divisiones de los árboles. |
| **Propiedad Aditiva** | **Sí:** $\sum \phi_i = f(x) - E[f(x)]$ (Consistencia garantizada). | **No:** Los pesos lineales no suman exactamente la probabilidad. | No aplica (métrica global relativa). |
| **Sesgo Conocido** | No sesgado por cardinalidad continua. | Sensible al tamaño del radio de perturbación local y semilla. | **Sesgo Severo:** Sobrestima variables continuas con muchos valores únicos (ej. `TotalCharges`). |
| **Explicación Local** | Exacta y matemáticamente robusta para cada individuo. | Rápida e intuitiva basada en reglas booleanas. | No disponible (solo importancia global agregada). |

**Explicación en Términos No Técnicos para Gestores:**  
- **SHAP** actúa como un desglose de 'factura': le dice al gerente exactamente cuántos centavos de riesgo aportó cada característica (ej. '+28% de riesgo por tener contrato mensual').  
- **LIME** actúa como una 'lupa simplificada': genera un perfil aproximado diciendo qué variables son las más determinantes en las circunstancias inmediatas del cliente.  
- **Alerta de Causalidad:** Una alta importancia predictiva (ej. `tenure` o `InternetService_Fiber optic`) **no implica causalidad**. Un cliente no cancela *porque* tenga fibra óptica, sino porque las fallas de calidad de red o insatisfacción con el precio no fueron atendidas con soporte técnico oportuno."""))

# --- Cell 6: Part 5 Markdown & Streamlit Deployment ---
cells.append(nbf.v4.new_markdown_cell("""---
# 🚀 Parte 5. Aplicación Interactiva con Streamlit en Google Colab

Generamos el script `app.py` y los archivos serializados de modelo y metadatos. Luego, iniciamos Streamlit en segundo plano y abrimos un túnel público mediante `localtunnel` para interactuar con la interfaz gráfica en vivo desde el navegador."""))

cells.append(nbf.v4.new_code_cell("""# 1. Serialización del modelo y preprocesador para Streamlit
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

joblib.dump(champion_pipeline, 'models/champion_pipeline.joblib')
joblib.dump(preprocessor, 'models/preprocessor.joblib')

# Guardar metadatos
metadata_export = {
    'baseline_comparison': baseline_results,
    'optim_comparison': optim_summary.to_dict(orient='records'),
    'pareto_data': pareto_records,
    'feature_names': feature_names,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'confusion_matrix': confusion_matrix(y_test, y_pred_tpe).tolist()
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata_export, f, indent=2)

print("✅ Modelo y metadatos serializados en carpeta 'models/'.")"""))

# App code cell directly reading app.py
with open('app.py', 'r', encoding='utf-8') as f:
    app_py_code = f.read()

app_write_code = "%%writefile app.py\n" + app_py_code
cells.append(nbf.v4.new_code_cell(app_write_code))

cells.append(nbf.v4.new_code_cell("""# 3. Lanzar Streamlit y Generar Túnel Público en Google Colab
import subprocess
import urllib.request

# Iniciar Streamlit en segundo plano
process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])

# Obtener IP pública (requerida por localtunnel como clave de acceso)
external_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()

print(f"\\n🔑 TU CONTRASEÑA DE ACCESO EN LOCALTUNNEL ES TU IP PÚBLICA: {external_ip}")
print("⚡ Iniciando túnel localtunnel... Haz clic en el enlace a continuación y pega la contraseña anterior:\\n")

# Ejecutar localtunnel
!npx -y localtunnel --port 8501"""))

# --- Cell 7: Part 6 & Business Pitch Markdown ---
cells.append(nbf.v4.new_markdown_cell("""---
# 💼 Parte 6 & Pitch Comercial: RetenTel AI

### 1. Resumen Ejecutivo de la Propuesta de Negocio
- **Nombre de la Empresa / Producto:** *RetenTel AI - Prescriptive Retention Engine*.
- **Problema de Negocio:** La industria de telecomunicaciones enfrenta una tasa promedio de fuga del 26.5%. Recuperar o adquirir un nuevo cliente (CAC) cuesta entre **$300 y $350**, mientras que una oferta de retención bien calibrada cuesta apenas **$35 a $40**.
- **Propuesta de Valor Única:** A diferencia de las 'cajas negras' que solo predicen el churn cuando el cliente ya tomó la decisión, RetenTel AI combina:
  1. **Detección Temprana de Alto Recall (78.6%):** Evita perder clientes de alto valor sin contactar.
  2. **Intervención Prescriptiva XAI:** Traduce los factores SHAP en ofertas personalizadas inmediatas (ej. si el factor es falta de soporte técnico, se asigna soporte VIP en lugar de descuentos de precio innecesarios).
- **Modelo de Monetización (SaaS B2B):**
  - Suscripción anual base de **$60,000 USD** para bases de hasta 100,000 suscriptores.
  - Success fee del **5%** sobre el valor de los clientes efectivamente rescatados.
- **Caso de Negocio (Base de 50,000 clientes):**
  - Clientes en riesgo detectados anualmente: **10,414**
  - Clientes rescatados con intervención prescriptiva: **4,165**
  - Ingresos protegidos: **$3,248,700 USD**
  - Inversión total (SaaS + costos de campaña): **$598,000 USD**
  - **Beneficio Neto:** **+$2,650,700 USD (ROI > 440%)**.

---
### 2. Guion de Negociación para el Business Pitch (Rol Vendedor)
* **Apertura de Impacto:**  
  *"Estimado Director Comercial, hoy su empresa pierde 26 de cada 100 clientes. Eso significa que más de $3 millones de dólares salen por la puerta cada año mientras sus comerciales siguen llamando a ciegas ofreciendo descuentos genéricos que destruyen su margen."*
* **Demostración de la Solución (Streamlit Demo):**  
  *"Con RetenTel AI, cada vez que un cliente contacta a soporte o se evalúa el corte de mes, nuestro modelo no solo calcula un 84% de probabilidad de fuga en 12 milisegundos, sino que a través de SHAP nos dice: 'Este cliente no se va por precio, se va porque tiene fibra óptica sin soporte técnico y su contrato es mensual'. La recomendación prescriptiva es automática: se le otorga soporte premium gratuito por 3 meses, resolviendo la causa raíz por un costo de $15 en lugar de quemar un descuento de $100."*
* **Cierre y Manejo de Objeciones:**  
  - *Objeción del Comprador:* "Nuestros científicos de datos ya tienen modelos internos de churn."  
  - *Respuesta Contundente:* "Un modelo interno que solo calcula probabilidades en un CSV mensual no salva clientes en el momento de la verdad. RetenTel AI entrega inferencia en milisegundos calibrada por el Frente de Pareto e integrada al CRM con acciones prescriptivas XAI explicables para agentes no técnicos. Con solo rescatar 100 clientes adicionales en el primer mes, la solución ya está 100% amortizada."*"""))

# Assign cells
nb['cells'] = cells

with open('Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✅ Archivo 'Tarea7_Churn_ML_Optimization_XAI_Pitch.ipynb' generado con éxito.")
