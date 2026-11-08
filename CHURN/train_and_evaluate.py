"""
Training and Multi-Objective Optimization Script for Telco Customer Churn
Updated for:
- Models: Logistic Regression & CatBoostClassifier (only)
- Imbalance handling: ADASYN oversampling for minority class
- Multi-Objective Optimization: NSGA-III with 4 objectives:
    1. Maximize F2-Score
    2. Minimize Inference Latency (ms per 1k inferences)
    3. Minimize Model Size (KB serialized)
    4. Maximize ROC-AUC
- Metric: Brier Score (brier_score_loss) for probability calibration
- Explainability: SHAP + Feature Correlation Index for top features
- Counterfactual Explanations: Minimal actionable interventions
"""

import time
import os
import json
import joblib
import tempfile
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import NSGAIIISampler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, brier_score_loss, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN
import shap

SEED = 42
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("=" * 70)
print("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO Y OPTIMIZACIÓN MULTIOBJETIVO")
print("=" * 70)

# 1. CARGA Y LIMPIEZA DE DATOS
print("\n[1/6] 📥 Cargando y limpiando dataset Telco Churn...")
csv_path = "Telco-Customer-Churn.csv"
if not os.path.exists(csv_path):
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.to_csv(csv_path, index=False)
else:
    df = pd.read_csv(csv_path)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0.0)
df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

drop_cols = ['customerID', 'Churn', 'Churn_Binary']
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
y = df['Churn_Binary'].copy()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in feature_cols if c not in num_cols]

print(f"Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Distribución original Churn: No={sum(y==0)} ({(1-y.mean())*100:.1f}%), Sí={sum(y==1)} ({y.mean()*100:.1f}%)")

# Split estratificado 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"Conjunto Entrenamiento: {X_train.shape[0]} registros | Conjunto Prueba (Aislado): {X_test.shape[0]} registros")

# 2. PREPROCESADOR
print("\n[2/6] ⚙️ Construyendo pipeline de transformación...")
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

preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

encoded_cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
feature_names = list(num_cols) + list(encoded_cat_names)
print(f"Total características procesadas: {len(feature_names)}")

# 3. BALANCEO CON ADASYN
print("\n[3/6] ⚖️ Aplicando ADASYN para sintetizar muestras de la clase minoritaria (Churn=1)...")
adasyn = ADASYN(random_state=SEED)
X_train_res, y_train_res = adasyn.fit_resample(X_train_proc, y_train)

adasyn_info = {
    "before_resampling": {"No_Churn": int(sum(y_train == 0)), "Churn": int(sum(y_train == 1))},
    "after_adasyn": {"No_Churn": int(sum(y_train_res == 0)), "Churn": int(sum(y_train_res == 1))}
}
print(f"Entrenamiento antes de ADASYN: {adasyn_info['before_resampling']}")
print(f"Entrenamiento tras ADASYN:    {adasyn_info['after_adasyn']}")

# Función utilitaria para medir latencia y tamaño
def measure_latency_and_size(model, X_sample, n_runs=1000):
    # Latencia por 1k muestras
    t0 = time.perf_counter()
    _ = model.predict_proba(X_sample[:min(len(X_sample), 500)])
    t1 = time.perf_counter()
    latency_ms_1k = ((t1 - t0) / min(len(X_sample), 500)) * 1000 * 1000
    
    # Tamaño en KB
    tmp_path = os.path.join(tempfile.gettempdir(), f"model_size_test_{time.time_ns()}.joblib")
    joblib.dump(model, tmp_path, compress=3)
    size_kb = os.path.getsize(tmp_path) / 1024.0
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return latency_ms_1k, size_kb

# 4. MODELOS BASE (REGRESIÓN LOGÍSTICA & CATBOOST)
print("\n[4/6] 🥊 Evaluando modelos base: Regresión Logística vs CatBoost...")
baseline_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "CatBoost": CatBoostClassifier(
        iterations=150, depth=5, learning_rate=0.08, verbose=0,
        random_seed=SEED, allow_writing_files=False, train_dir=tempfile.gettempdir()
    )
}

baseline_comparison = []
for name, model in baseline_models.items():
    t_start = time.time()
    model.fit(X_train_res, y_train_res)
    fit_time = time.time() - t_start
    
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    roc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    lat_ms, sz_kb = measure_latency_and_size(model, X_test_proc)
    
    res = {
        "Model": name,
        "Test_Accuracy": round(acc, 4),
        "Test_Precision": round(prec, 4),
        "Test_Recall": round(rec, 4),
        "Test_F1": round(f1, 4),
        "Test_F2": round(f2, 4),
        "Test_ROC_AUC": round(roc, 4),
        "Test_Brier_Score": round(brier, 4),
        "Latency_ms_1k": round(lat_ms, 2),
        "Model_Size_KB": round(sz_kb, 1),
        "Fit_Time_s": round(fit_time, 2)
    }
    baseline_comparison.append(res)
    print(f"-> {name}: F2={f2:.4f} | Recall={rec:.4f} | ROC-AUC={roc:.4f} | Brier={brier:.4f} | Latencia={lat_ms:.2f}ms | Tamaño={sz_kb:.1f}KB")

# 5. OPTIMIZACIÓN MULTIOBJETIVO CON NSGA-III
print("\n[5/6] 🧬 Ejecutando Optimización Multiobjetivo NSGA-III sobre CatBoost...")
print("      Objetivos: 1) Max F2-Score | 2) Min Latencia | 3) Min Tamaño (KB) | 4) Max ROC-AUC")

def objective_nsgaiii(trial):
    depth = trial.suggest_int('depth', 3, 7)
    iterations = trial.suggest_int('iterations', 50, 250, step=25)
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
    
    # 3-Fold Stratified CV para eficiencia
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
        
    f2_mean = float(np.mean(f2_list))
    roc_mean = float(np.mean(roc_list))
    
    # Medir latencia y tamaño en modelo final ajustado
    cb.fit(X_train_res, y_train_res)
    lat_ms, sz_kb = measure_latency_and_size(cb, X_test_proc)
    
    return f2_mean, lat_ms, sz_kb, roc_mean

sampler_nsga = NSGAIIISampler(population_size=12, seed=SEED)
study_nsga = optuna.create_study(
    directions=['maximize', 'minimize', 'minimize', 'maximize'],
    sampler=sampler_nsga
)

t0_opt = time.time()
study_nsga.optimize(objective_nsgaiii, n_trials=18, timeout=50)
opt_duration = time.time() - t0_opt
print(f"Optimización NSGA-III finalizada en {opt_duration:.1f}s ({len(study_nsga.trials)} evaluaciones realizadas).")

# Determinar soluciones Pareto no-dominadas
trials_data = []
for t in study_nsga.trials:
    if t.state == optuna.trial.TrialState.COMPLETE:
        f2_v, lat_v, sz_v, roc_v = t.values
        trials_data.append({
            "trial_id": t.number,
            "f2": round(f2_v, 4),
            "latency_ms_1k": round(lat_v, 2),
            "model_size_kb": round(sz_v, 1),
            "roc_auc": round(roc_v, 4),
            "depth": t.params.get('depth'),
            "iterations": t.params.get('iterations'),
            "learning_rate": round(t.params.get('learning_rate', 0.1), 4),
            "l2_leaf_reg": round(t.params.get('l2_leaf_reg', 3.0), 2)
        })

df_pareto_trials = pd.DataFrame(trials_data)

# Algoritmo de dominancia de Pareto para 4 objetivos:
# Maximizar f2, Minimizar latency, Minimizar size, Maximizar roc
def check_pareto_dominance(df_pts):
    pts = df_pts[['f2', 'latency_ms_1k', 'model_size_kb', 'roc_auc']].values
    # Invertir para minimizar todo: [-f2, +latency, +size, -roc]
    costs = np.column_stack([-pts[:, 0], pts[:, 1], pts[:, 2], -pts[:, 3]])
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1) | np.all(costs[is_pareto] == c, axis=1)
            is_pareto[i] = True
    return is_pareto

df_pareto_trials['is_pareto'] = check_pareto_dominance(df_pareto_trials)
pareto_solutions = df_pareto_trials[df_pareto_trials['is_pareto']]
print(f"Soluciones no-dominadas en Frente de Pareto: {len(pareto_solutions)} de {len(df_pareto_trials)}")

# Seleccionar Modelo Campeón del Frente de Pareto
# Criterio: Balance óptimo (Score compuesto = F2 * 0.4 + ROC * 0.4 - norm(lat)*0.1 - norm(size)*0.1)
f2_norm = (pareto_solutions['f2'] - pareto_solutions['f2'].min()) / (pareto_solutions['f2'].max() - pareto_solutions['f2'].min() + 1e-6)
roc_norm = (pareto_solutions['roc_auc'] - pareto_solutions['roc_auc'].min()) / (pareto_solutions['roc_auc'].max() - pareto_solutions['roc_auc'].min() + 1e-6)
lat_norm = (pareto_solutions['latency_ms_1k'] - pareto_solutions['latency_ms_1k'].min()) / (pareto_solutions['latency_ms_1k'].max() - pareto_solutions['latency_ms_1k'].min() + 1e-6)
sz_norm = (pareto_solutions['model_size_kb'] - pareto_solutions['model_size_kb'].min()) / (pareto_solutions['model_size_kb'].max() - pareto_solutions['model_size_kb'].min() + 1e-6)

composite_score = 0.4 * f2_norm + 0.4 * roc_norm - 0.1 * lat_norm - 0.1 * sz_norm
best_trial_idx = composite_score.idxmax()
champion_trial = pareto_solutions.loc[best_trial_idx]

print(f"🏆 Trial Campeón seleccionado: #{champion_trial['trial_id']} (F2={champion_trial['f2']}, ROC={champion_trial['roc_auc']}, Latencia={champion_trial['latency_ms_1k']}ms, Tamaño={champion_trial['model_size_kb']}KB)")

# Re-entrenar Champion CatBoost con los hiperparámetros ganadores
champion_params = {
    'depth': int(champion_trial['depth']),
    'iterations': int(champion_trial['iterations']),
    'learning_rate': float(champion_trial['learning_rate']),
    'l2_leaf_reg': float(champion_trial['l2_leaf_reg']),
    'verbose': 0,
    'random_seed': SEED,
    'allow_writing_files': False,
    'train_dir': tempfile.gettempdir()
}
champion_model = CatBoostClassifier(**champion_params)
champion_model.fit(X_train_res, y_train_res)

# Evaluación del campeón en conjunto de prueba aislado
y_test_pred = champion_model.predict(X_test_proc)
y_test_proba = champion_model.predict_proba(X_test_proc)[:, 1]

champ_metrics = {
    "Accuracy": round(accuracy_score(y_test, y_test_pred), 4),
    "Precision": round(precision_score(y_test, y_test_pred), 4),
    "Recall": round(recall_score(y_test, y_test_pred), 4),
    "F1_Score": round(f1_score(y_test, y_test_pred), 4),
    "F2_Score": round(fbeta_score(y_test, y_test_pred, beta=2), 4),
    "ROC_AUC": round(roc_auc_score(y_test, y_test_proba), 4),
    "Brier_Score": round(brier_score_loss(y_test, y_test_proba), 4),
    "Latency_ms_1k": round(champion_trial['latency_ms_1k'], 2),
    "Model_Size_KB": round(champion_trial['model_size_kb'], 1),
    "Hyperparameters": champion_params
}
cm = confusion_matrix(y_test, y_test_pred).tolist()
print("\n📊 Métricas del Modelo Campeón en Test Set Aislado:")
for k, v in champ_metrics.items():
    if k != 'Hyperparameters':
        print(f"   • {k}: {v}")

# 6. SHAP, ÍNDICE DE CORRELACIÓN Y CONTRAFACTUALES
print("\n[6/6] 🔍 Computando SHAP, Índices de Correlación de Top Features y Contrafactuales...")

# SHAP TreeExplainer
explainer = shap.TreeExplainer(champion_model)
sample_size = min(400, len(X_test_proc))
shap_vals = explainer.shap_values(X_test_proc[:sample_size])

mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
top_feat_indices = np.argsort(mean_abs_shap)[::-1][:8]
top_features = [feature_names[i] for i in top_feat_indices]
top_shap_importance = [round(float(mean_abs_shap[i]), 4) for i in top_feat_indices]

# Matriz de Correlación entre las top features
df_top_feats = pd.DataFrame(X_test_proc[:sample_size, top_feat_indices], columns=top_features)
corr_matrix_top = df_top_feats.corr().round(4).to_dict()

# Índice de Correlación entre Feature y su Valor SHAP (dirección del impacto)
feat_shap_corr = {}
for idx, f_name in zip(top_feat_indices, top_features):
    f_vals = X_test_proc[:sample_size, idx]
    s_vals = shap_vals[:, idx]
    # Pearson corr
    if np.std(f_vals) > 1e-6 and np.std(s_vals) > 1e-6:
        r = np.corrcoef(f_vals, s_vals)[0, 1]
    else:
        r = 0.0
    feat_shap_corr[f_name] = round(float(r), 4)

print(f"Top 8 Features por SHAP:")
for f_name, imp, r in zip(top_features, top_shap_importance, feat_shap_corr.values()):
    print(f"   • {f_name}: Importancia Media |SHAP|={imp:.4f} | Correlación con SHAP={r:+.4f}")

# MOTOR CONTRAFACTUAL (Counterfactual Search Engine)
# Encuentra la mínima modificación accionable para revertir una predicción de Churn
def generate_counterfactual(client_series, model, preprocessor, target_prob=0.30):
    client_df = pd.DataFrame([client_series])
    init_proba = float(model.predict_proba(preprocessor.transform(client_df))[0, 1])
    
    if init_proba <= target_prob:
        return {
            "status": "already_retained",
            "initial_prob": round(init_proba, 4),
            "cf_prob": round(init_proba, 4),
            "changes": []
        }
        
    cf_candidate = client_series.copy()
    changes_applied = []
    
    # Acciones prescriptivas ordenadas por impacto y viabilidad comercial
    candidate_actions = [
        ("Contract", "Two year", "Cambio a contrato por 2 años (fidelización institucional)"),
        ("TechSupport", "Yes", "Activación gratuita de soporte técnico prioritario"),
        ("OnlineSecurity", "Yes", "Activación de protección y seguridad en línea"),
        ("PaymentMethod", "Bank transfer (automatic)", "Migración a débito automático"),
        ("MonthlyCharges", max(20.0, float(client_series.get('MonthlyCharges', 70.0)) - 15.0), "Bono/Descuento de $15/mes"),
        ("Contract", "One year", "Cambio a contrato anual con tarifa protegida")
    ]
    
    current_proba = init_proba
    for feat, new_val, desc in candidate_actions:
        prev_val = cf_candidate[feat]
        if prev_val == new_val:
            continue
        cf_candidate[feat] = new_val
        test_df = pd.DataFrame([cf_candidate])
        new_p = float(model.predict_proba(preprocessor.transform(test_df))[0, 1])
        
        changes_applied.append({
            "feature": feat,
            "original_value": prev_val,
            "counterfactual_value": new_val,
            "intervention": desc,
            "prob_reduction": round(float(current_proba - new_p), 4),
            "new_prob": round(float(new_p), 4)
        })
        current_proba = new_p
        if current_proba <= target_prob:
            break
            
    return {
        "status": "success",
        "initial_prob": round(init_proba, 4),
        "cf_prob": round(current_proba, 4),
        "total_drop": round(float(init_proba - current_proba), 4),
        "changes": changes_applied
    }

# Generar ejemplo contrafactual con caso de alto riesgo
high_risk_case = X_test.iloc[0].to_dict()
# Forzar perfil riesgoso típico
high_risk_case.update({
    'Contract': 'Month-to-month',
    'tenure': 3,
    'MonthlyCharges': 89.0,
    'InternetService': 'Fiber optic',
    'TechSupport': 'No',
    'OnlineSecurity': 'No',
    'PaymentMethod': 'Electronic check'
})
sample_cf = generate_counterfactual(high_risk_case, champion_model, preprocessor)
print(f"\nEjemplo Contrafactual:")
print(f"Probabilidad Inicial: {sample_cf['initial_prob']*100:.1f}% -> Contrafactual: {sample_cf['cf_prob']*100:.1f}% (Reducción: {sample_cf['total_drop']*100:.1f}%)")
print(f"Intervenciones requeridas: {len(sample_cf['changes'])}")

# SERIALIZACIÓN COMPLETA
print("\n💾 Serializando artefactos en directorio 'models/'...")

# Pipeline unificado estándar de scikit-learn
champion_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', champion_model)
])
joblib.dump(champion_pipeline, 'models/champion_pipeline.joblib')
joblib.dump(preprocessor, 'models/preprocessor.joblib')

metadata = {
    "project": "RetenTel AI - Churn Multi-Objective Intelligence",
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "models_evaluated": ["Logistic Regression", "CatBoost"],
    "resampling_strategy": "ADASYN (Adaptive Synthetic Sampling)",
    "optimization_algorithm": "NSGA-III (Non-dominated Sorting Genetic Algorithm III)",
    "optimization_objectives": [
        "Maximize F2-Score (Recall Prioritized)",
        "Minimize Inference Latency (ms / 1k)",
        "Minimize Model Size (KB)",
        "Maximize ROC-AUC"
    ],
    "calibration_metric": "Brier Score (Mean Squared Probability Error)",
    "adasyn_info": adasyn_info,
    "baseline_comparison": baseline_comparison,
    "nsgaiii_pareto_data": df_pareto_trials.to_dict(orient='records'),
    "champion_metrics": champ_metrics,
    "confusion_matrix": cm,
    "shap_summary": {
        "top_features": top_features,
        "importance_mean_abs": top_shap_importance,
        "feature_correlation_matrix": corr_matrix_top,
        "feature_shap_correlation": feat_shap_corr
    },
    "sample_counterfactual": sample_cf
}

with open('models/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ Pipeline y metadatos serializados exitosamente en 'models/'.")
print("=" * 70)
