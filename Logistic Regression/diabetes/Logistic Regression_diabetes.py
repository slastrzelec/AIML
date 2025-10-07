import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score # Added cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# 1️⃣ Load dataset
data = load_diabetes(as_frame=True)
X = data.data
y = data.target

# 2️⃣ Convert to classes (1 = high disease progression, e.g., above mean)
y_class = (y > y.mean()).astype(int)

# 3️⃣ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# 4️⃣ Pipeline: Scaling + Logistic Regression
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs'))

# 4b 🆕 Cross-Validation
cv_scores = cross_val_score(model, X, y_class, cv=5, scoring='accuracy')

print("📊 Cross-Validation Results (5-Fold):")
print(f"Mean Accuracy (CV): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
print("---")

# Fit the model on the training data for prediction and reporting
model.fit(X_train, y_train)

# 5️⃣ Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6️⃣ Confusion matrix
ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    display_labels=['Healthy', 'Diseased'],
    cmap='Blues',
    values_format='d'
)
plt.title('Confusion Matrix')
plt.show()

# 7️⃣ ROC curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('ROC Curve for Diabetes Classification Model')
plt.show()

# 8️⃣ Classification report
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# 9️⃣ Feature importance (coefficients)
coefficients = model.named_steps['logisticregression'].coef_[0]
intercept = model.named_steps['logisticregression'].intercept_[0] # 🆕 Getting the intercept
features = X.columns

# 🆕 Displaying the Intercept
print(f"\n🧠 Intercept (Bias term): {intercept:.3f}")
print("   - This is the log-odds of being 'Diseased' (class 1) when all features are at their mean level (after scaling).")

importance_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

# Feature descriptions
feature_info = {
    "age": "Patient age (normalized values)",
    "sex": "Gender (normalized value)",
    "bmi": "Body Mass Index",
    "bp": "Average blood pressure",
    "s1": "Total cholesterol",
    "s2": "LDL cholesterol (bad)",
    "s3": "HDL cholesterol (good)",
    "s4": "TCH/HDL ratio",
    "s5": "Log of serum triglycerides",
    "s6": "Blood glucose level"
}

feature_info_df = pd.DataFrame(list(feature_info.items()), columns=["Feature", "Description"])
importance_with_desc = importance_df.merge(feature_info_df, on="Feature", how="left")
importance_with_desc["Coefficient"] = importance_with_desc["Coefficient"].round(3)

print("\n📘 Feature Impact and Description:\n")
print(importance_with_desc)

# 🔟 Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(importance_with_desc['Feature'], importance_with_desc['Coefficient'], color='orange')
plt.xlabel('Coefficient Value (Impact on Prediction)')
plt.ylabel('Features')
plt.title('Feature Importance in Diabetes Classification Model')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()