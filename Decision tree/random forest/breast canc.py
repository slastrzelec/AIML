from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------------------
# Load data and create DataFrame
# ------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

# ------------------------------
# Split data
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Train Random Forest model
# ------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
print(f"✅ Random Forest model accuracy: {accuracy:.4f}")

# ------------------------------
# Feature importance
# ------------------------------
feat_importances = pd.DataFrame({
    "Feature": data.feature_names,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n🔹 Top 10 Features:")
print(feat_importances.head(10))

top_features = feat_importances.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color='skyblue')
plt.title("Top 15 Features (Random Forest)")
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.tight_layout()
plt.show()

# ------------------------------
# Correlation heatmap
# ------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# ------------------------------
# Confusion matrix and classification report
# ------------------------------
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(classification_report(y_test, y_pred, target_names=data.target_names))

# ------------------------------
# Surrogate decision tree
# ------------------------------
surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
surrogate.fit(X_train, rf.predict(X_train))

plt.figure(figsize=(20, 10))
plot_tree(
    surrogate,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    node_ids=True
)
plt.title("Surrogate Decision Tree (node IDs shown)")
plt.show()

# ------------------------------
# Decision path for a single sample
# ------------------------------
sample_idx = 0
sample = X_test[sample_idx].reshape(1, -1)
node_indicator = surrogate.decision_path(sample)
leaf_id = surrogate.apply(sample)

print(f"\nSample index: {sample_idx}")
print(f"Predicted class by surrogate tree: {surrogate.predict(sample)[0]}")
print(f"Leaf node reached: {leaf_id[0]}")
print("Nodes visited in the tree:", node_indicator.indices)

print("\nDecision path explanation:")
for node_id in node_indicator.indices:
    feature = surrogate.tree_.feature[node_id]
    threshold = surrogate.tree_.threshold[node_id]
    if feature != _tree.TREE_UNDEFINED:
        feature_name = data.feature_names[feature]
        print(f"Node {node_id}: if {feature_name} <= {threshold:.2f}")

# ----------------------------------------
# ROC Curve and AUC (Corrected Code)
# ----------------------------------------
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

# 1. Calculate the probability of the positive class (1 - benign)
y_prob = rf.predict_proba(X_test)[:, 1]

# 2. Calculate the ROC curve points and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\n✨ ROC AUC Score: {roc_auc:.4f}")

# 3. Plot the ROC Curve
plt.figure(figsize=(8, 8))

# Use 'name' instead of 'estimator_name' (avoids one FutureWarning)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, 
                              name='Random Forest') 

# Plot the curve using the correct argument: curve_kwargs
# This resolves the AttributeError you encountered.
roc_display.plot(curve_kwargs={'color':'darkorange', 'lw':2}) 

# Plot the diagonal line (random guess baseline)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title(f'Receiver Operating Characteristic (ROC) Curve\n(AUC = {roc_auc:.4f})')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ----------------------------------------
# Visualization of Key Feature Distributions (Box Plots)
# ----------------------------------------
print("\n📊 Visualizing distribution of the Top 3 features by class.")

# Get the names of the top 3 most important features
top_3_features = feat_importances['Feature'].head(3).tolist()

plt.figure(figsize=(15, 5))
plt.suptitle('Distribution of Top 3 Features by Class (0=Malignant, 1=Benign)', fontsize=16)

# Map the target back to the DataFrame for plotting
df_plot = pd.DataFrame(X, columns=data.feature_names)
df_plot['Target'] = y

for i, feature in enumerate(top_3_features):
    plt.subplot(1, 3, i + 1)
    # Use seaborn for a clean box plot
    sns.boxplot(x='Target', y=feature, data=df_plot, palette='Set2')
    plt.title(feature, fontsize=12)
    plt.xlabel('Diagnosis (0: Malignant, 1: Benign)')
    plt.ylabel(feature)
    plt.grid(axis='y', linestyle='--')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
plt.show()

print("✅ Box plots showing class separation for key features have been generated.")