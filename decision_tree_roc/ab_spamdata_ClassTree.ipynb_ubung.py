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
from sklearn.ensemble import RandomForestClassifier # Hinzugefügt für Random Forest
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score # Für weitere Plots
)
import matplotlib.pyplot as plt
import dtreeviz # Optional für detaillierte Baumvisualisierung
import os

# matplotlib font manager und andere Warnungen unterdrücken
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.axes._base').setLevel(logging.ERROR)

# -------------------------------------
# 2. Dataset Loading and Initial Processing
# -------------------------------------
def load_dataset(filename, github_raw_url="https://raw.githubusercontent.com/Statsomat/Datasets/master/data-ml/"):
    full_url = f"{github_raw_url}{filename}"
    return pd.read_csv(full_url, header=None)

df_raw = load_dataset('spamdata.csv')
print("Raw dataset shape:", df_raw.shape)

def process_row(row_str):
    values = row_str.iloc[0].split(' ')
    return pd.Series(values)

df_processed = df_raw.apply(process_row, axis=1)

# -------------------------------------
# 3. Define Column Names and Apply to DataFrame
# -------------------------------------
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

df = df_processed.copy()
df.columns = column_names

# -------------------------------------
# 4. Data Type Conversion and Basic EDA
# -------------------------------------
for column in df.columns:
    df[column] = pd.to_numeric(df[column])

print("\nTarget variable 'spam' value counts:")
print(df['spam'].value_counts())
print(f"Percentage of spam emails: {df['spam'].mean() * 100:.2f}%")

# -------------------------------------
# 5. Prepare Data for Machine Learning
# -------------------------------------
X = df.drop('spam', axis=1)
y = df['spam']
class_names = ['non-spam', 'spam']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------------
# 6. Grafikverzeichnis erstellen
# -------------------------------------
graphics_dir = "tree_graphics_spam"
os.makedirs(graphics_dir, exist_ok=True)

# -------------------------------------
# 7. Train and Evaluate Decision Tree Classifier (Original Model)
# -------------------------------------
print("\n--- Training Decision Tree Classifier ---")
dtc_spam = DecisionTreeClassifier(
    criterion='gini', max_depth=8, min_samples_split=2, min_samples_leaf=10,
    random_state=123, ccp_alpha=0.00
)
dtc_spam.fit(X_train, y_train)

y_pred_dt = dtc_spam.predict(X_test)
y_scores_dt = dtc_spam.predict_proba(X_test)[:, 1]

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"\nDecision Tree - Accuracy: {accuracy_dt:.4f}")
print("\nDecision Tree - Classification Report (Test Dataset):")
print(classification_report(y_test, y_pred_dt, target_names=class_names))
print("\nDecision Tree - Confusion Matrix (Test Dataset):")
print(confusion_matrix(y_test, y_pred_dt))

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_scores_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# -------------------------------------
# 8. Train and Evaluate Random Forest Classifier
# -------------------------------------
print("\n--- Training Random Forest Classifier ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_leaf=10)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
y_scores_rf = rf_clf.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest - Accuracy: {accuracy_rf:.4f}")
print("\nRandom Forest - Classification Report (Test Dataset):")
print(classification_report(y_test, y_pred_rf, target_names=class_names))
print("\nRandom Forest - Confusion Matrix (Test Dataset):")
print(confusion_matrix(y_test, y_pred_rf))

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_scores_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# -------------------------------------
# 9. Comparative Visualizations
# -------------------------------------

# --- Vergleichende ROC-Kurven ---
print("\nGenerating and saving comparative ROC curves...")
plt.figure(figsize=(10, 8))
plt.plot(fpr_dt, tpr_dt, color='blue', lw=2, label=f'Decision Tree ROC (AUC = {roc_auc_dt:.3f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest ROC (AUC = {roc_auc_rf:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Vergleich der ROC-Kurven: Decision Tree vs. Random Forest')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(graphics_dir, 'comparative_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# --- Vergleichende Precision-Recall-Kurven ---
print("\nGenerating and saving comparative Precision-Recall curves...")
precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_scores_dt)
ap_dt = average_precision_score(y_test, y_scores_dt)

precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_scores_rf)
ap_rf = average_precision_score(y_test, y_scores_rf)

plt.figure(figsize=(10, 8))
plt.plot(recall_dt, precision_dt, color='blue', lw=2, label=f'Decision Tree PR (AP = {ap_dt:.3f})')
plt.plot(recall_rf, precision_rf, color='green', lw=2, label=f'Random Forest PR (AP = {ap_rf:.3f})')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Vergleich der Precision-Recall-Kurven: Decision Tree vs. Random Forest')
plt.legend(loc="lower left")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(graphics_dir, 'comparative_pr_curves.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# --- Feature Importances Plot (für beide Modelle) ---
print("\nGenerating and saving feature importances plot...")

importances_dt = dtc_spam.feature_importances_
indices_dt = np.argsort(importances_dt)[::-1]
top_n_features = 10 # Anzahl der wichtigsten Merkmale anzeigen

importances_rf = rf_clf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]

fig, axes = plt.subplots(2, 1, figsize=(12, 12)) # Zwei Subplots

# Decision Tree Feature Importances
axes[0].set_title("Decision Tree - Top Feature Importances")
axes[0].bar(range(top_n_features), importances_dt[indices_dt][:top_n_features], align="center")
axes[0].set_xticks(range(top_n_features))
axes[0].set_xticklabels(np.array(feature_names)[indices_dt][:top_n_features], rotation=45, ha="right")
axes[0].set_ylabel("Importance")

# Random Forest Feature Importances
axes[1].set_title("Random Forest - Top Feature Importances")
axes[1].bar(range(top_n_features), importances_rf[indices_rf][:top_n_features], align="center", color="green")
axes[1].set_xticks(range(top_n_features))
axes[1].set_xticklabels(np.array(feature_names)[indices_rf][:top_n_features], rotation=45, ha="right")
axes[1].set_ylabel("Importance")

plt.tight_layout()
plt.savefig(os.path.join(graphics_dir, 'comparative_feature_importances.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("\nScript finished.")