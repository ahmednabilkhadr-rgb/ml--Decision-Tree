# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# Load dataset

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.iloc[:, 0].copy()

print("Dataset loaded successfully")

# Explore dataset

print("\nFirst 5 rows:")
print(X.head())

print("\nDataset information:")
X.info()

print("\nDataset description:")
print(X.describe())

print("\nDataset shape:")
print(X.shape)

print("\nDataset variables:")
print(heart_disease.variables)

# Check missing values

X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

print("\nMissing values:")
print(X.isnull().sum())

print("\nTotal missing values:")
print(X.isnull().sum().sum())

# Remove rows with missing target

valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# Check duplicates

print("\nDuplicate rows:")
print(X.duplicated().sum())

# Convert target into binary classification
# 0 = No Heart Disease
# 1 = Heart Disease

y = (y > 0).astype(int)

print("\nTarget distribution:")
print(y.value_counts())

# Explore target distribution

plt.figure(figsize=(7, 5))

sns.countplot(x=y)

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.savefig("target_distribution.png")
plt.show()

# Explore feature distributions

X.hist(figsize=(15, 12), bins=20)

plt.suptitle("Feature Distributions")

plt.tight_layout()
plt.savefig("feature_distributions.png")
plt.show()

# Correlation analysis

correlation_data = X.copy()
correlation_data["target"] = y

plt.figure(figsize=(13, 9))

sns.heatmap(
    correlation_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Missing values visualization

plt.figure(figsize=(12, 5))

sns.heatmap(
    X.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Values")

plt.tight_layout()
plt.savefig("missing_values.png")
plt.show()

# Select features

print("\nFeatures:")
print(X.columns.tolist())

print("\nNumber of features:")
print(X.shape[1])

# Identify categorical and numerical features

categorical_features = [
    "cp",
    "restecg",
    "slope",
    "thal"
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# Preprocessing

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Build decision tree model

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

# Train model and find best parameters

parameters = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [2, 3, 4, 5, 6, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    model,
    parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross validation score:")
print(grid_search.best_score_)

# Make predictions

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]

# Evaluate model

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

auc = roc_auc_score(y_test, y_prob)

print("\nModel evaluation")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")
print("ROC AUC:", round(auc, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ROC curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Decision Tree AUC = {auc:.3f}"
)

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# Precision recall curve

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_test, y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(pr_recall, pr_precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")

plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.show()

# Feature importance

feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = best_model.named_steps[
    "classifier"
].feature_importances_

importance_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_data = importance_data.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature importance:")
print(importance_data)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_data,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Visualize decision tree

plt.figure(figsize=(22, 12))

plot_tree(
    best_model.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Visualization")

plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300)
plt.show()

# Compare training and testing accuracy

train_pred = best_model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print("\nTraining accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing accuracy:", round(test_accuracy * 100, 2), "%")

plt.figure(figsize=(7, 5))

plt.bar(
    ["Training", "Testing"],
    [train_accuracy * 100, test_accuracy * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("Training vs Testing Accuracy")

plt.tight_layout()
plt.savefig("train_test_accuracy.png")
plt.show()

# Save model

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/decision_tree_model.pkl"
)

model_info = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": auc,
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy,
    "features": X.columns.tolist()
}

joblib.dump(
    model_info,
    "models/model_info.pkl"
)

print("\nModel saved successfully")

print("\nAll machine learning steps completed!")