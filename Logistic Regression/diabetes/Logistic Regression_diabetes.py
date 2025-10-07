import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# 1️⃣ Wczytanie danych
data = load_diabetes(as_frame=True)
X = data.data
y = data.target

# 2️⃣ Zamiana na klasy (1 = duża progresja choroby, np. górne 50%)
y_class = (y > y.mean()).astype(int)

# 3️⃣ Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# 4️⃣ Pipeline: Skalowanie + regresja logistyczna
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs'))
model.fit(X_train, y_train)

# 5️⃣ Przewidywanie
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6️⃣ Macierz pomyłek
ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    display_labels=['Zdrowy', 'Chory'],
    cmap='Blues',
    values_format='d'
)
plt.title('Macierz pomyłek')
plt.show()

# 7️⃣ Krzywa ROC
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('Krzywa ROC dla modelu klasyfikacji cukrzycy')
plt.show()

# 8️⃣ Raport klasyfikacji
print("📋 Raport klasyfikacji:")
print(classification_report(y_test, y_pred))

# 9️⃣ Wpływ cech (ważność cech)
coefficients = model.named_steps['logisticregression'].coef_[0]
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

# Opisy cech
feature_info = {
    "age": "Wiek pacjenta (wartości znormalizowane)",
    "sex": "Płeć (wartość znormalizowana)",
    "bmi": "Body Mass Index",
    "bp": "Średnie ciśnienie krwi",
    "s1": "Całkowity cholesterol (Total Cholesterol)",
    "s2": "Lipoproteiny LDL (zły cholesterol)",
    "s3": "Lipoproteiny HDL (dobry cholesterol)",
    "s4": "Stosunek TCH/HDL",
    "s5": "Logarytm stężenia trójglicerydów (LTG)",
    "s6": "Poziom glukozy (GLU)"
}

feature_info_df = pd.DataFrame(list(feature_info.items()), columns=["Feature", "Opis"])
importance_with_desc = importance_df.merge(feature_info_df, on="Feature", how="left")
importance_with_desc["Coefficient"] = importance_with_desc["Coefficient"].round(3)

print("\n📘 Wpływ cech i ich znaczenie:\n")
print(importance_with_desc)

#  🔟 Wykres ważności cech
plt.figure(figsize=(10, 6))
plt.barh(importance_with_desc['Feature'], importance_with_desc['Coefficient'], color='orange')
plt.xlabel('Wartość współczynnika (wpływ na predykcję)')
plt.ylabel('Cechy')
plt.title('Ważność cech w modelu klasyfikacji cukrzycy')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
