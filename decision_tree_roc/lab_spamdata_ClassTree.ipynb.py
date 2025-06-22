#!/usr/bin/env python
# coding: utf-8

# -------------------------------------
# 1. Import Libraries
# -------------------------------------
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import dtreeviz
import os # Hinzugefügt für Ordneroperationen, falls benötigt

# Optional: matplotlib font manager und andere Warnungen unterdrücken
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.axes._base').setLevel(logging.ERROR) # Für spezifische Achsen-Warnungen

# -------------------------------------
# 2. Dataset Loading and Initial Processing
# -------------------------------------
def load_dataset(filename, github_raw_url="https://raw.githubusercontent.com/Statsomat/Datasets/master/data-ml/"):
    """
    Load a dataset from the course GitHub repository.
    """
    full_url = f"{github_raw_url}{filename}"
    return pd.read_csv(full_url, header=None)

# Datensatz laden
df_raw = load_dataset('spamdata.csv')
print("Raw dataset shape:", df_raw.shape)
print("\nRaw dataset head:")
print(df_raw.head())

# Funktion zur Verarbeitung jeder Zeile
def process_row(row_str):
    values = row_str.iloc[0].split(' ')
    return pd.Series(values)

# Funktion auf jede Zeile im DataFrame anwenden
df_processed = df_raw.apply(process_row, axis=1)
print("\nProcessed dataset head:")
print(df_processed.head())

# -------------------------------------
# 3. Define Column Names and Apply to DataFrame
# -------------------------------------
# Spaltennamen basierend auf Attributinformationen erstellen
word_freq_cols = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
    'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp',
    'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
    'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
    'word_freq_edu', 'word_freq_table', 'word_freq_conference'
]
char_freq_cols = [
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#'
]
capital_run_length_cols = [
    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'
]
target_col = ['spam']
column_names = word_freq_cols + char_freq_cols + capital_run_length_cols + target_col

df = df_processed.copy() # Kopie erstellen, um das Original df_processed nicht zu verändern
df.columns = column_names

print("\nDataFrame head with column names:")
print(df.head())
print("\nDataFrame shape:", df.shape)

# -------------------------------------
# 4. Data Type Conversion and Basic EDA
# -------------------------------------
# Alle Spalten in geeignete numerische Typen konvertieren
for column in df.columns:
    df[column] = pd.to_numeric(df[column])

print("\nDataFrame info after type conversion:")
df.info()

# EDA der Zielvariable
print("\nTarget variable 'spam' value counts:")
print(df['spam'].value_counts())
print(f"Percentage of spam emails: {df['spam'].mean() * 100:.2f}%")

# -------------------------------------
# 5. Prepare Data for Machine Learning
# -------------------------------------
X = df.drop('spam', axis=1)  # Merkmale
y = df['spam']               # Zielvariable (Ground Truth)
class_names = ['non-spam', 'spam']
feature_names = list(X.columns) # Merkmalsnamen für Visualisierungen verwenden

# Aufteilung des Datensatzes in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------------
# 6. Train Decision Tree Classifier
# -------------------------------------
dtc_spam = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=10,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=123,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    class_weight=None,
    ccp_alpha=0.00
)
dtc_spam.fit(X_train, y_train)

# -------------------------------------
# 7. Visualization with dtreeviz
# -------------------------------------
# Sicherstellen, dass das Grafikverzeichnis existiert
graphics_dir = "tree_graphics_spam" # Eindeutiger Name für dieses Skript
os.makedirs(graphics_dir, exist_ok=True)

viz_model = dtreeviz.model(
    dtc_spam,
    X_train=X_train, y_train=y_train,
    feature_names=feature_names, # Korrigiert: feature_names statt features
    target_name='spam',          # Zielvariablenname korrigiert
    class_names=class_names
)

print("\nGenerating and saving dtreeviz tree view (SVG)...")
dtreeviz_view = viz_model.view(scale=0.8)
dtreeviz_view.save(os.path.join(graphics_dir, "dtreeviz_spam_tree.svg"))
# dtreeviz_view # Im Skript wird dies nicht automatisch angezeigt, nur im Notebook

# -------------------------------------
# 8. Model Evaluation
# -------------------------------------
# Vorhersagen für den Testdatensatz
y_pred = dtc_spam.predict(X_test)
print("\nSample predictions (first 10):", y_pred[:10])

# Genauigkeit auf dem Testdatensatz berechnen
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Zusätzliche Metriken berechnen
print("\nClassification Report (Test Dataset):")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nConfusion Matrix (Test Dataset):")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------
# 9. ROC Curve
# -------------------------------------
# Für die ROC-Kurve benötigen wir Wahrscheinlichkeits-Scores
y_scores = dtc_spam.predict_proba(X_test)[:, 1] # Wahrscheinlichkeit der positiven Klasse (Spam)
print("\nSample probability scores (first 10):", y_scores[:10])

# ROC-Kurve berechnen
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# ROC-Kurve plotten
print("\nGenerating and saving ROC curve...")
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') # Diagonale Linie
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve for Decision Tree Spam Classifier')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)

# Genauigkeit auf dem Plot anzeigen
plt.text(0.7, 0.3, f'Accuracy: {accuracy:.4f}', fontsize=12)

# Einige Schwellenwertpunkte auf der Kurve markieren
indices_to_mark = np.linspace(0, len(thresholds) - 1, 5, dtype=int)
for i in indices_to_mark:
    threshold_val = thresholds[i]
    if threshold_val <= 1.0: # Nur gültige Schwellenwerte plotten
        plt.plot(fpr[i], tpr[i], 'ro')
        plt.annotate(f"{threshold_val:.2f}", (fpr[i], tpr[i]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')

# Optimalen Schwellenwert markieren (nächster Punkt zur oberen linken Ecke)
optimal_idx = np.argmin(np.sqrt((1 - tpr)**2 + fpr**2)) # Distanz zum Punkt (0,1) minimieren
optimal_threshold = thresholds[optimal_idx]
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'go', markersize=10, label=f"Optimal Threshold: {optimal_threshold:.2f}")
plt.annotate(f"Optimal: {optimal_threshold:.2f}",
             (fpr[optimal_idx], tpr[optimal_idx]),
             textcoords="offset points",
             xytext=(0,20),
             ha='center',
             fontweight='bold')
plt.legend(loc="lower right") # Legende erneut aufrufen, um den optimalen Punkt einzuschließen

# Plot speichern und anzeigen
plt.tight_layout()
plt.savefig(os.path.join(graphics_dir, 'decision_tree_roc_curve.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close() # Figur schließen

print("\nScript finished.")