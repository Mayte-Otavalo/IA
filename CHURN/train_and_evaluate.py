"""
Training and Optimization Script for Telco Customer Churn
Covers:
- Part 1: Data prep, pipelines, model comparisons
- Part 2: Hyperparameter optimization (Grid, Random, Optuna TPE, NSGA-II)
- Part 3: Multi-objective optimization (F1-score vs Inference Latency & Recall vs Precision)
- Part 4: SHAP & LIME precomputations
- Model and metadata serialization for Streamlit
"""

import time
import os
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap
import lime
import lime.lime_tabular

# Set random seed
SEED = 42
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("--> 1. Loading and cleaning data...")
url = "Telco-Customer-Churn.csv"
if not os.path.exists(url):
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.to_csv('Telco-Customer-Churn.csv', index=False)
else:
    df = pd.read_csv(url)

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
# Customers with tenure == 0 have TotalCharges NaN; fill with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

# Target encoding
df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Feature definitions
drop_cols = ['customerID', 'Churn', 'Churn_Binary']
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
y = df['Churn_Binary'].copy()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in feature_cols if c not in num_cols]

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(feature_cols)} ({len(num_cols)} num, {len(cat_cols)} cat)")
print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

# Stratified Split (80/20) - Test set isolated
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Preprocessor
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

# Fit preprocessor on train only
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

feature_names = (
    num_cols + 
    list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
)

print(f"Processed feature dimensions: {len(feature_names)}")

# -------------------------------------------------------------
# Part 1: Model Comparison (Baseline)
# -------------------------------------------------------------
print("\n--> 2. Comparing Baseline Models...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Random Forest': RandomForestClassifier(random_state=SEED, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=SEED, eval_metric='logloss', n_estimators=100)
}

baseline_results = []
trained_models = {}

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    start = time.time()
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1').mean()
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    cv_recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='recall').mean()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    res = {
        'Model': name,
        'CV_F1': round(cv_f1, 4),
        'CV_ROC_AUC': round(cv_auc, 4),
        'CV_Recall': round(cv_recall, 4),
        'Test_Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Test_Precision': round(precision_score(y_test, y_pred), 4),
        'Test_Recall': round(recall_score(y_test, y_pred), 4),
        'Test_F1': round(f1_score(y_test, y_pred), 4),
        'Test_ROC_AUC': round(roc_auc_score(y_test, y_prob), 4),
        'Train_Time_s': round(fit_time, 2)
    }
    baseline_results.append(res)
    trained_models[name] = pipe

df_baseline = pd.DataFrame(baseline_results)
print(df_baseline.to_string(index=False))

# -------------------------------------------------------------
# Part 2: Hyperparameter Optimization Strategies
# -------------------------------------------------------------
print("\n--> 3. Running Hyperparameter Optimization on Random Forest...")

# Parameter space
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__class_weight': ['balanced', None]
}

rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=SEED))
])

# 1. Grid Search
print("Running Grid Search (54 candidates)...")
start = time.time()
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring='f1', n_jobs=-1, return_train_score=False)
grid.fit(X_train, y_train)
time_grid = time.time() - start
best_rf_grid = grid.best_estimator_
y_pred_grid = best_rf_grid.predict(X_test)
y_prob_grid = best_rf_grid.predict_proba(X_test)[:, 1]

grid_res = {
    'Strategy': 'Grid Search',
    'Best_CV_F1': round(grid.best_score_, 4),
    'Test_F1': round(f1_score(y_test, y_pred_grid), 4),
    'Test_ROC_AUC': round(roc_auc_score(y_test, y_prob_grid), 4),
    'Test_Recall': round(recall_score(y_test, y_pred_grid), 4),
    'Evaluated_Configs': len(grid.cv_results_['params']),
    'Time_Seconds': round(time_grid, 2),
    'Best_Params': str(grid.best_params_)
}

# 2. Random Search
print("Running Random Search (25 candidates)...")
start = time.time()
random_search = RandomizedSearchCV(
    rf_pipe, param_distributions=param_grid, n_iter=25, cv=cv, scoring='f1',
    random_state=SEED, n_jobs=-1
)
random_search.fit(X_train, y_train)
time_random = time.time() - start
best_rf_random = random_search.best_estimator_
y_pred_rnd = best_rf_random.predict(X_test)
y_prob_rnd = best_rf_random.predict_proba(X_test)[:, 1]

rnd_res = {
    'Strategy': 'Random Search',
    'Best_CV_F1': round(random_search.best_score_, 4),
    'Test_F1': round(f1_score(y_test, y_pred_rnd), 4),
    'Test_ROC_AUC': round(roc_auc_score(y_test, y_prob_rnd), 4),
    'Test_Recall': round(recall_score(y_test, y_pred_rnd), 4),
    'Evaluated_Configs': 25,
    'Time_Seconds': round(time_random, 2),
    'Best_Params': str(random_search.best_params_)
}

# 3. Bayesian Optimization (Optuna TPE)
print("Running Bayesian Optimization TPE (25 trials)...")
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
    # Using preprocessed X_train_proc for speed
    scores = cross_val_score(clf, X_train_proc, y_train, cv=cv, scoring='f1')
    return scores.mean()

sampler_tpe = optuna.samplers.TPESampler(seed=SEED)
study_tpe = optuna.create_study(direction='maximize', sampler=sampler_tpe)
start = time.time()
study_tpe.optimize(objective_tpe, n_trials=25)
time_tpe = time.time() - start

best_tpe_params = study_tpe.best_params
best_rf_tpe_clf = RandomForestClassifier(**best_tpe_params, random_state=SEED)
best_rf_tpe = Pipeline([('preprocessor', preprocessor), ('classifier', best_rf_tpe_clf)])
best_rf_tpe.fit(X_train, y_train)

y_pred_tpe = best_rf_tpe.predict(X_test)
y_prob_tpe = best_rf_tpe.predict_proba(X_test)[:, 1]

tpe_res = {
    'Strategy': 'Bayesian (Optuna TPE)',
    'Best_CV_F1': round(study_tpe.best_value, 4),
    'Test_F1': round(f1_score(y_test, y_pred_tpe), 4),
    'Test_ROC_AUC': round(roc_auc_score(y_test, y_prob_tpe), 4),
    'Test_Recall': round(recall_score(y_test, y_pred_tpe), 4),
    'Evaluated_Configs': 25,
    'Time_Seconds': round(time_tpe, 2),
    'Best_Params': str(best_tpe_params)
}

# -------------------------------------------------------------
# Part 3: Multi-Objective Optimization (NSGA-II)
# -------------------------------------------------------------
print("\n--> 4. Running Multi-Objective Optimization (NSGA-II: Max F1 vs Min Latency)...")
# Objective 1: Maximize CV F1-Score
# Objective 2: Minimize Mean Inference Time per 1,000 samples (ms)

def multi_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 40, 250, step=20)
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
    
    # Objective 1: F1 CV
    f1_cv = cross_val_score(clf, X_train_proc, y_train, cv=cv, scoring='f1').mean()
    
    # Objective 2: Inference latency on 1000 samples (benchmarked)
    clf.fit(X_train_proc, y_train)
    bench_data = X_test_proc[:500]
    t0 = time.perf_counter()
    for _ in range(5):
        _ = clf.predict_proba(bench_data)
    t_elapsed = (time.perf_counter() - t0) / 5.0
    latency_ms_per_1k = (t_elapsed / len(bench_data)) * 1000 * 1000  # ms per 1000 samples
    
    return f1_cv, latency_ms_per_1k

sampler_nsga = optuna.samplers.NSGAIISampler(seed=SEED)
study_nsga = optuna.create_study(directions=['maximize', 'minimize'], sampler=sampler_nsga)
start = time.time()
study_nsga.optimize(multi_objective, n_trials=35)
time_nsga = time.time() - start

# Extract Pareto front trials
pareto_trials = study_nsga.best_trials
pareto_data = []
for t in study_nsga.trials:
    is_pareto = t in pareto_trials
    pareto_data.append({
        'trial_number': t.number,
        'f1_cv': round(t.values[0], 4),
        'latency_ms_1k': round(t.values[1], 3),
        'n_estimators': t.params.get('n_estimators'),
        'max_depth': t.params.get('max_depth'),
        'min_samples_split': t.params.get('min_samples_split'),
        'class_weight': t.params.get('class_weight'),
        'is_pareto': is_pareto
    })

df_pareto = pd.DataFrame(pareto_data)

# Pick best balanced Pareto solution (highest F1 with low latency)
sorted_pareto = sorted([t for t in pareto_trials], key=lambda x: x.values[0], reverse=True)
selected_pareto_trial = sorted_pareto[0]
best_nsga_params = selected_pareto_trial.params

best_rf_nsga_clf = RandomForestClassifier(**best_nsga_params, random_state=SEED)
best_rf_nsga = Pipeline([('preprocessor', preprocessor), ('classifier', best_rf_nsga_clf)])
best_rf_nsga.fit(X_train, y_train)

y_pred_nsga = best_rf_nsga.predict(X_test)
y_prob_nsga = best_rf_nsga.predict_proba(X_test)[:, 1]

nsga_res = {
    'Strategy': 'Multi-Objective (NSGA-II)',
    'Best_CV_F1': round(selected_pareto_trial.values[0], 4),
    'Test_F1': round(f1_score(y_test, y_pred_nsga), 4),
    'Test_ROC_AUC': round(roc_auc_score(y_test, y_prob_nsga), 4),
    'Test_Recall': round(recall_score(y_test, y_pred_nsga), 4),
    'Evaluated_Configs': 35,
    'Time_Seconds': round(time_nsga, 2),
    'Best_Params': str(best_nsga_params)
}

optim_comparison = pd.DataFrame([grid_res, rnd_res, tpe_res, nsga_res])
print("\n--- Optimization Comparison Table ---")
print(optim_comparison.to_string(index=False))

# Select Champion Model (Optuna TPE / NSGA-II balanced model)
champion_pipeline = best_rf_tpe
joblib.dump(champion_pipeline, 'models/champion_pipeline.joblib')
joblib.dump(preprocessor, 'models/preprocessor.joblib')
print("\nChampion model saved to models/champion_pipeline.joblib")

# -------------------------------------------------------------
# Part 4: Explainability (SHAP & LIME)
# -------------------------------------------------------------
print("\n--> 5. Computing SHAP and LIME Explanations...")
# Extract classifier and transformed feature matrix
fitted_clf = champion_pipeline.named_steps['classifier']

# SHAP TreeExplainer
explainer_shap = shap.TreeExplainer(fitted_clf)
shap_values = explainer_shap.shap_values(X_test_proc)

# Handle shap values structure (for binary classification, index 1 corresponds to Churn=1)
if isinstance(shap_values, list):
    shap_vals_churn = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_churn = shap_values[:, :, 1]
else:
    shap_vals_churn = shap_values

# Global feature importance via SHAP
mean_abs_shap = np.mean(np.abs(shap_vals_churn), axis=0)
df_shap_imp = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': mean_abs_shap,
    'RF_MDI_Importance': fitted_clf.feature_importances_
}).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

print("\nTop 10 Features by SHAP Importance:")
print(df_shap_imp.head(10).to_string(index=False))

# LIME Explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_proc,
    feature_names=feature_names,
    class_names=['Retain (No)', 'Churn (Yes)'],
    mode='classification',
    random_state=SEED
)

# Identify two contrasting customers in test set:
# Customer A: High probability of Churn
# Customer B: Low probability of Churn
y_probs_test = champion_pipeline.predict_proba(X_test)[:, 1]
idx_high_risk = int(np.argmax(y_probs_test))
idx_low_risk = int(np.argmin(y_probs_test))

sample_high = X_test.iloc[idx_high_risk].to_dict()
sample_low = X_test.iloc[idx_low_risk].to_dict()

# Compute LIME explanations
exp_high = lime_explainer.explain_instance(
    data_row=X_test_proc[idx_high_risk],
    predict_fn=fitted_clf.predict_proba,
    num_features=8
)
exp_low = lime_explainer.explain_instance(
    data_row=X_test_proc[idx_low_risk],
    predict_fn=fitted_clf.predict_proba,
    num_features=8
)

lime_high_list = exp_high.as_list()
lime_low_list = exp_low.as_list()

# Local SHAP for the two customers
shap_high = dict(zip(feature_names, [round(float(v), 4) for v in shap_vals_churn[idx_high_risk]]))
shap_low = dict(zip(feature_names, [round(float(v), 4) for v in shap_vals_churn[idx_low_risk]]))

# Sort top features
shap_high_top = sorted(shap_high.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
shap_low_top = sorted(shap_low.items(), key=lambda x: abs(x[1]), reverse=True)[:8]

# Save all metadata for Streamlit and Colab
metadata = {
    'baseline_comparison': baseline_results,
    'optim_comparison': optim_comparison.to_dict(orient='records'),
    'pareto_data': pareto_data,
    'feature_names': feature_names,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'feature_importance': df_shap_imp.head(20).to_dict(orient='records'),
    'idx_high_risk': idx_high_risk,
    'idx_low_risk': idx_low_risk,
    'sample_high_risk': {
        'features': sample_high,
        'prob_churn': float(y_probs_test[idx_high_risk]),
        'lime_rules': lime_high_list,
        'shap_top': shap_high_top
    },
    'sample_low_risk': {
        'features': sample_low,
        'prob_churn': float(y_probs_test[idx_low_risk]),
        'lime_rules': lime_low_list,
        'shap_top': shap_low_top
    },
    'confusion_matrix': confusion_matrix(y_test, champion_pipeline.predict(X_test)).tolist(),
    'test_size': len(y_test),
    'train_size': len(y_train),
    'churn_prevalence': float(y.mean())
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save sample dataset for Streamlit demo
sample_demo_df = X_test.copy()
sample_demo_df['Actual_Churn'] = y_test.values
sample_demo_df['Predicted_Probability'] = y_probs_test
sample_demo_df.head(200).to_csv('data/sample_test_data.csv', index=False)

print("\nFinished successfully! All models and metadata exported.")
