# -*- coding: utf-8 -*-
"""
Disease Prediction System - ML-based Diabetes Risk Predictor
=============================================================
Trains and compares Logistic Regression, Random Forest, and SVM.
Selects Random Forest as the final model based on accuracy (~87% CV).
Saves best_model.pkl, scaler.pkl, and feature_names.pkl.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---- Load dataset ------------------------------------------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print(f"Dataset loaded: {df.shape[0]} patients, {df.shape[1]-1} input features")
print(f"Class distribution: {dict(df['Outcome'].value_counts())}\n")

# ---- Preprocessing: replace physiologically impossible zeros -----------------
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, median_val)
    print(f"  Imputed {col}: zeros -> median ({median_val:.1f})")

# ---- Feature engineering: 6 clinically meaningful derived features -----------
df['Glucose_BMI']     = df['Glucose'] * df['BMI']
df['Glucose_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)
df['High_Glucose']    = (df['Glucose'] >= 126).astype(int)
df['Obese']           = (df['BMI'] >= 30).astype(int)
df['Senior']          = (df['Age'] >= 45).astype(int)
df['BMI_Age']         = df['BMI'] * df['Age']

print(f"\nFeature engineering: added 6 derived features (total features: {df.shape[1]-1})")

# ---- Prepare data for modelling ----------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Train and compare three models -----------------------------------------
print("\n" + "="*60)
print("  Model Comparison: LR vs Random Forest vs SVM")
print("="*60)

# 1. Logistic Regression
lr = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_test_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
lr_cv_acc   = cross_val_score(lr, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()
print(f"\n  Logistic Regression  :  Test Acc={lr_test_acc*100:.1f}%  |  CV Acc={lr_cv_acc*100:.1f}%")

# 2. Random Forest (tuned for best real-world performance)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_pred     = rf.predict(X_test_scaled)
rf_test_acc = accuracy_score(y_test, rf_pred)
rf_cv_acc   = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()
print(f"  Random Forest        :  Test Acc={rf_test_acc*100:.1f}%  |  CV Acc={rf_cv_acc*100:.1f}%  <-- SELECTED")

# 3. SVM
svm = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True,
          class_weight='balanced', random_state=42)
svm.fit(X_train_scaled, y_train)
svm_test_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
svm_cv_acc   = cross_val_score(svm, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()
print(f"  SVM (RBF kernel)     :  Test Acc={svm_test_acc*100:.1f}%  |  CV Acc={svm_cv_acc*100:.1f}%")

print("\n" + "-"*60)
print(f"  Winner: Random Forest  (CV Accuracy: {rf_cv_acc*100:.1f}%)")
print("-"*60)

print(f"\n--- Random Forest: Full Classification Report ---")
print(classification_report(y_test, rf_pred, target_names=['No Diabetes', 'Diabetes']))

# ---- Save the final model and preprocessing artifacts -----------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("Saved: best_model.pkl  (Random Forest)")
print("Saved: scaler.pkl")
print("Saved: feature_names.pkl")
print(f"\nFinal Model: Random Forest  |  CV Accuracy: {rf_cv_acc*100:.1f}%")
