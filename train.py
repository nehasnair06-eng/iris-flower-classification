import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# 1. LOAD DATASET
# =========================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset loaded successfully!")
print("Features:", feature_names)
print("Classes:", target_names)
print("Dataset shape:", X.shape)


# =========================================================
# 2. CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(X, columns=feature_names)

df["target"] = y
df["species"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nClass distribution:")
print(df["species"].value_counts())


# =========================================================
# 3. CHECK MISSING VALUES
# =========================================================

print("\nMissing values:")
print(df.isnull().sum())


# =========================================================
# 4. VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="species"
)

plt.title("Sepal Length vs Sepal Width")
plt.savefig("iris_plot.png")
plt.show()


# =========================================================
# 5. FEATURES AND TARGET
# =========================================================

X = df[feature_names]
y = df["target"]


# =========================================================
# 6. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 7. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# 8. LOGISTIC REGRESSION
# =========================================================

logistic_model = LogisticRegression()

logistic_model.fit(X_train_scaled, y_train)

logistic_pred = logistic_model.predict(X_test_scaled)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

print("\nLogistic Regression Accuracy:",
      logistic_accuracy)


# =========================================================
# 9. KNN
# =========================================================

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

print("KNN Accuracy:", knn_accuracy)


# =========================================================
# 10. RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("Random Forest Accuracy:", rf_accuracy)


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        knn_accuracy,
        rf_accuracy
    ]
})

print("\n================ MODEL COMPARISON ================")
print(results)


# =========================================================
# 12. SELECT BEST MODEL
# =========================================================

if rf_accuracy >= max(logistic_accuracy, knn_accuracy):

    best_model = rf_model
    best_model_name = "Random Forest"
    best_scaler = None

elif logistic_accuracy >= knn_accuracy:

    best_model = logistic_model
    best_model_name = "Logistic Regression"
    best_scaler = scaler

else:

    best_model = knn_model
    best_model_name = "KNN"
    best_scaler = scaler


print("\nBest Model:", best_model_name)


# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

if best_scaler is not None:
    best_predictions = best_model.predict(X_test_scaled)
else:
    best_predictions = best_model.predict(X_test)


cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()


# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=target_names
    )
)


# =========================================================
# 15. SAVE MODEL
# =========================================================

joblib.dump(
    best_model,
    "model/iris_model.pkl"
)

joblib.dump(
    best_scaler,
    "model/scaler.pkl"
)

joblib.dump(
    target_names,
    "model/target_names.pkl"
)

print("\nModel saved successfully!")