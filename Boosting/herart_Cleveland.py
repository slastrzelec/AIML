# ===============================
# Cleveland Heart Disease - Boosting
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
path = r"C:\Users\slast\OneDrive\Pulpit\AIML\Boosting\Heart_disease_cleveland_new.csv"
df = pd.read_csv(path)

# -----------------------------
# 2️⃣ Prepare features and target
# -----------------------------
X = df.drop('target', axis=1)
y = df['target']

# One-hot encode categorical features
categorical_cols = ['cp', 'restecg', 'slope', 'thal', 'ca']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optional: scale features (not strictly necessary for XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3️⃣ Train XGBoost classifier
# -----------------------------
xgb = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train_scaled, y_train)

# -----------------------------
# 4️⃣ Predictions & evaluation
# -----------------------------
y_pred = xgb.predict(X_test_scaled)
y_prob = xgb.predict_proba(X_test_scaled)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ XGBoost Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease','Disease']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease','Disease'], yticklabels=['No Disease','Disease'])
plt.title("Confusion Matrix — XGBoost")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob, name='XGBoost', color='darkorange')
plt.plot([0,1],[0,1], linestyle='--', color='navy')
plt.title("ROC Curve — XGBoost")
plt.show()

# -----------------------------
# 5️⃣ Feature importance
# -----------------------------
importances = xgb.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='rocket')
plt.title("Feature Importance — XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
