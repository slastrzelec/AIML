# ==========================================================
# 🌲 vs 🔥 Random Forest vs XGBoost — Performance Comparison
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 2️⃣ Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

start = time.time()
rf.fit(X_train, y_train)
rf_time = time.time() - start

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)

# -----------------------------
# 3️⃣ XGBoost (Boosting)
# -----------------------------
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

start = time.time()
xgb.fit(X_train, y_train)
xgb_time = time.time() - start

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_prob_xgb)

# -----------------------------
# 4️⃣ Compare results
# -----------------------------
results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [rf_acc, xgb_acc],
    "ROC AUC": [rf_auc, xgb_auc],
    "Train Time (s)": [rf_time, xgb_time]
})

print("\n📊 MODEL COMPARISON:")
print(results.to_string(index=False))

# -----------------------------
# 5️⃣ ROC Curves
# -----------------------------
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_prob_rf, name="Random Forest", color="blue")
RocCurveDisplay.from_predictions(y_test, y_prob_xgb, name="XGBoost", color="darkorange")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curves — Random Forest vs XGBoost")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 6️⃣ Feature Importance Comparison
# -----------------------------
rf_importances = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_,
    "Model": "Random Forest"
})

xgb_importances = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb.feature_importances_,
    "Model": "XGBoost"
})

feat_compare = pd.concat([rf_importances, xgb_importances])

plt.figure(figsize=(10, 8))
sns.barplot(data=feat_compare.sort_values(by="Importance", ascending=False).head(20),
            x="Importance", y="Feature", hue="Model", palette="Set2")
plt.title("Top 20 Most Important Features — RF vs XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -----------------------------
# 7️⃣ Classification Reports
# -----------------------------
print("\n🔹 Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=data.target_names))

print("\n🔥 XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=data.target_names))
