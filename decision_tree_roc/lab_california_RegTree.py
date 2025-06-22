#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
)
import dtreeviz
import matplotlib.pyplot as plt # Für das Speichern und Anzeigen von Plots
import os # Für Datei- und Ordneroperationen

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# -------------------------------------
# 0. Grafikverzeichnis erstellen
# -------------------------------------
graphics_dir = "treegrafics"
os.makedirs(graphics_dir, exist_ok=True)

# -------------------------------------
# 1. Datensatz laden
# -------------------------------------
url = "https://raw.githubusercontent.com/Statsomat/Datasets/master/data-ml/cal_housing/CaliforniaHousing/cal_housing.data"
california = pd.read_csv(url)

# Korrekte Spaltennamen definieren
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
X = california.drop('median_house_value', axis=1) # Features (Merkmale)
y = california['median_house_value'] # Zielvariable

# Aufteilung in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
feature_names = list(X_train.columns) # Liste der Merkmalsnamen

# -------------------------------------
# 3. Entscheidungsbaum-Modell trainieren
# -------------------------------------
# DecisionTreeRegressor mit maximaler Tiefe 3 und quadratischem Fehler als Kriterium
regr_1 = DecisionTreeRegressor(max_depth=3, criterion="squared_error")
regr_1.fit(X_train, y_train) # Modell mit Trainingsdaten trainieren

# -------------------------------------
# 4. Visualisierung und Speichern
# -------------------------------------
# dtreeviz Modell-Objekt erstellen
viz_rmodel = dtreeviz.model(
    regr_1, X_train, y_train,
    target_name='median_house_value', # Name der Zielvariable
    feature_names=feature_names,      # Namen der Merkmale
    class_names=None                  # Für Regression auf None setzen
)

# Visualisierungen:

# Visualisierung des Merkmalsraums für 'median_income'
print("\nErzeuge und speichere rtree_feature_space für median_income...")
viz_rmodel.rtree_feature_space(features=['median_income']) # Erstellt den Plot auf der aktuellen Figur
fig1 = plt.gcf() # Holt die aktuelle Figur
fig1.savefig(os.path.join(graphics_dir, "feature_space_median_income.png")) # Als PNG speichern
plt.show() # Plot anzeigen
plt.close(fig1) # Figur schließen

# 3D-Visualisierung des Merkmalsraums für 'median_income' und 'longitude'
print("\nErzeuge und speichere rtree_feature_space3D...")
viz_rmodel.rtree_feature_space3D(
    features=['median_income', 'longitude'],
    fontsize=10, elev=30, azim=20,
    show={'splits', 'title'}, # Zeige Splits und Titel
    colors={'tessellation_alpha': 0.5} # Transparenz der Tessellation
)
fig2 = plt.gcf() # Holt die aktuelle Figur
fig2.savefig(os.path.join(graphics_dir, "feature_space_3d.png"))
plt.show()
plt.close(fig2)

# Visualisierung des Merkmalsraums für 'median_income' und 'longitude' (2D)
print("\nErzeuge und speichere rtree_feature_space für median_income und longitude...")
viz_rmodel.rtree_feature_space(features=['median_income', 'longitude'])
fig3 = plt.gcf() # Holt die aktuelle Figur
fig3.savefig(os.path.join(graphics_dir, "feature_space_median_income_longitude.png"))
plt.show()
plt.close(fig3)

# Baumansicht mit dtreeviz generieren und speichern
print("\nErzeuge und speichere Baumansicht (SVG)...")
v = viz_rmodel.view(orientation="LR") # LR = Left to Right Orientierung
v.save(os.path.join(graphics_dir, "tree_view.svg")) # Als SVG speichern

# Baum mit matplotlib plotten und speichern
print("\nErzeuge und speichere sklearn plot_tree...")
fig_sklearn, ax_sklearn = plt.subplots(figsize=(20, 10)) # Neue Figur und Achse erstellen, Größe anpassen
plot_tree(regr_1, filled=True, feature_names=feature_names, rounded=True, fontsize=7, ax=ax_sklearn)
fig_sklearn.savefig(os.path.join(graphics_dir, "sklearn_plot_tree.png"))
plt.show()
plt.close(fig_sklearn)

# -------------------------------------
# 5. Bewertungsmetriken
# -------------------------------------
# Funktion zur Berechnung verschiedener Metriken
def calculate_metrics(y_true, y_pred):
    residuals = y_true - y_pred # Residuen (Fehler)
    n = len(y_true) # Anzahl der Datenpunkte
    return {
        'MSE': mean_squared_error(y_true, y_pred),             # Mittlerer quadratischer Fehler
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),   # Wurzel des mittleren quadratischen Fehlers
        'MAE': mean_absolute_error(y_true, y_pred),            # Mittlerer absoluter Fehler
        'R²': r2_score(y_true, y_pred),                        # Bestimmtheitsmaß
        'Standard Error': np.std(residuals) / np.sqrt(n),      # Standardfehler der Vorhersage
        'Sample Size': n                                       # Stichprobengröße
    }

# Vorhersagen für Trainings- und Testdaten
y_train_pred = regr_1.predict(X_train)
y_test_pred = regr_1.predict(X_test)

# Metriken für Trainings- und Testdaten berechnen
train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Ergebnisse in einer Tabelle zusammenfassen
results_table = pd.DataFrame({
    'Training': train_metrics,
    'Test': test_metrics
}).round(4) # Auf 4 Nachkommastellen runden

# Umfassende Modellbewertung ausgeben
print("\nComprehensive Model Evaluation:")
print("=" * 45)
print(results_table)