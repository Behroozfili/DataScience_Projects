#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Hinzugefügt
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error
)

import matplotlib.pyplot as plt
import os

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# -------------------------------------
# 0. Grafikverzeichnis erstellen
# -------------------------------------
graphics_dir = "treegraficsubung"
os.makedirs(graphics_dir, exist_ok=True)

# -------------------------------------
# 1. Datensatz laden
# -------------------------------------
url = "https://raw.githubusercontent.com/Statsomat/Datasets/master/data-ml/cal_housing/CaliforniaHousing/cal_housing.data"
california = pd.read_csv(url)

column_names = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households',
    'median_income', 'median_house_value'
]
california.columns = column_names

print("Dataset Overview:")
print(california.describe())

# -------------------------------------
# 2. Daten vorbereiten
# -------------------------------------
X = california.drop('median_house_value', axis=1)
y = california['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
feature_names = list(X_train.columns)

# -------------------------------------
# 3. Modelle trainieren und Fehler für Plot sammeln
# -------------------------------------
MAX_TREES = 100

# -- Entscheidungsbaum (als Referenz für ein einzelnes Modell) --
dt_regr = DecisionTreeRegressor(max_depth=3, criterion="squared_error", random_state=42)
dt_regr.fit(X_train, y_train)
dt_test_mse = mean_squared_error(y_test, dt_regr.predict(X_test))
dt_test_errors = [dt_test_mse] * MAX_TREES # Konstanter Fehler für den Plot

# -- Random Forest --
print("\nTrainiere Random Forest und sammle Fehler...")
rf_regr = RandomForestRegressor(n_estimators=MAX_TREES, max_features=2, random_state=42, oob_score=False)
rf_regr.fit(X_train, y_train)

rf_test_errors = []
all_tree_preds = np.array([tree.predict(X_test) for tree in rf_regr.estimators_])
cumulative_preds = np.cumsum(all_tree_preds, axis=0)
for i in range(MAX_TREES):
    current_mean_pred = cumulative_preds[i] / (i + 1)
    rf_test_errors.append(mean_squared_error(y_test, current_mean_pred))


# -- Gradient Boosting --
print("\nTrainiere Gradient Boosting und sammle Fehler...")
gb_regr = GradientBoostingRegressor(n_estimators=MAX_TREES, max_depth=4, learning_rate=0.1, random_state=42)
gb_regr.fit(X_train, y_train)

gb_test_errors = []
for y_pred_stage in gb_regr.staged_predict(X_test):
    gb_test_errors.append(mean_squared_error(y_test, y_pred_stage))

# -------------------------------------
# 4. Plot Fehler vs. Anzahl der Bäume (ähnlich Hastie et al. Fig 15.3)
# -------------------------------------
print("\nErzeuge Plot: Fehler vs. Anzahl der Bäume...")
plt.figure(figsize=(12, 7))
plt.plot(range(1, MAX_TREES + 1), dt_test_errors, label='Decision Tree (max_depth=3) - Test MSE', linestyle='--', color='gray')
plt.plot(range(1, MAX_TREES + 1), rf_test_errors, label=f'Random Forest (m=2) - Test MSE', color='blue')
plt.plot(range(1, MAX_TREES + 1), gb_test_errors, label=f'Gradient Boosting (depth=4) - Test MSE', color='green')

plt.xlabel("Anzahl der Bäume (Number of Trees)")
plt.ylabel("Test MSE (Mittlerer quadratischer Fehler)")
plt.title("Modellvergleich: Test MSE vs. Anzahl der Bäume")
plt.legend()
plt.grid(True)
plt.ylim(0, max(max(rf_test_errors), max(gb_test_errors), dt_test_mse) * 1.1)
plt.savefig(os.path.join(graphics_dir, "error_vs_trees_comparison.png"))
plt.show()
plt.close()

# -------------------------------------
# 5. Bewertungsmetriken für finale Modelle
# -------------------------------------
def calculate_metrics(y_true, y_pred, model_name=""):
    residuals = y_true - y_pred
    n = len(y_true)
    metrics = {
        f'MSE_{model_name}': mean_squared_error(y_true, y_pred),
        f'RMSE_{model_name}': np.sqrt(mean_squared_error(y_true, y_pred)),
        f'MAE_{model_name}': mean_absolute_error(y_true, y_pred),
        f'R²_{model_name}': r2_score(y_true, y_pred),
        f'Standard Error_{model_name}': np.std(residuals) / np.sqrt(n),
        f'Sample Size_{model_name}': n
    }
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


all_metrics_train = {}
all_metrics_test = {}

# Decision Tree Metriken
all_metrics_train.update(calculate_metrics(y_train, dt_regr.predict(X_train), "DT_Train"))
all_metrics_test.update(calculate_metrics(y_test, dt_regr.predict(X_test), "DT_Test"))

# Random Forest Metriken (mit MAX_TREES Bäumen)
all_metrics_train.update(calculate_metrics(y_train, rf_regr.predict(X_train), "RF_Train"))
all_metrics_test.update(calculate_metrics(y_test, rf_regr.predict(X_test), "RF_Test"))

# Gradient Boosting Metriken (mit MAX_TREES Bäumen)
all_metrics_train.update(calculate_metrics(y_train, gb_regr.predict(X_train), "GB_Train"))
all_metrics_test.update(calculate_metrics(y_test, gb_regr.predict(X_test), "GB_Test"))

print("\nUmfassende Modellbewertung (Finale Modelle mit 100 Bäumen):")
print("=" * 70)

print("\n--- Trainingsmetriken ---")
train_df = pd.DataFrame(all_metrics_train, index=[0]).T
train_df.columns = ["Wert"]
print(train_df)

print("\n--- Testmetriken ---")
test_df = pd.DataFrame(all_metrics_test, index=[0]).T
test_df.columns = ["Wert"]
print(test_df)

comparison_data = {
    'Decision Tree': {
        'Train MSE': all_metrics_train['MSE_DT_Train'],
        'Test MSE': all_metrics_test['MSE_DT_Test'],
        'Train R²': all_metrics_train['R²_DT_Train'],
        'Test R²': all_metrics_test['R²_DT_Test'],
    },
    'Random Forest': {
        'Train MSE': all_metrics_train['MSE_RF_Train'],
        'Test MSE': all_metrics_test['MSE_RF_Test'],
        'Train R²': all_metrics_train['R²_RF_Train'],
        'Test R²': all_metrics_test['R²_RF_Test'],
    },
    'Gradient Boosting': {
        'Train MSE': all_metrics_train['MSE_GB_Train'],
        'Test MSE': all_metrics_test['MSE_GB_Test'],
        'Train R²': all_metrics_train['R²_GB_Train'],
        'Test R²': all_metrics_test['R²_GB_Test'],
    }
}
comparison_df = pd.DataFrame(comparison_data).T
print("\n\n--- Vergleichstabelle (MSE und R²) ---")
print(comparison_df)