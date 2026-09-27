import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("data/Titanic-Dataset.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = df[features]
y = df["Survived"]

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed!")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree.fit(X_train_processed, y_train)

y_pred_tree = decision_tree.predict(X_test_processed)

accuracy_tree = accuracy_score(y_test, y_pred_tree)

print("Decision Tree Accuracy:", accuracy_tree)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

random_forest.fit(X_train_processed, y_train)

y_pred_rf = random_forest.predict(X_test_processed)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", accuracy_rf)

from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting = GradientBoostingClassifier(
    random_state=42
)

gradient_boosting.fit(X_train_processed, y_train)

y_pred_gb = gradient_boosting.predict(X_test_processed)

accuracy_gb = accuracy_score(y_test, y_pred_gb)

print("Gradient Boosting Accuracy:", accuracy_gb)

import tensorflow as tf

X_train_dl = X_train_processed
X_test_dl = X_test_processed

print("DL data prepared:", X_train_dl.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

ann = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_dl.shape[1],)),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
ann.fit(
    X_train_dl,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=1
)
ann_loss, ann_accuracy = ann.evaluate(
    X_test_dl,
    y_test,
    verbose=0
)

print("ANN Accuracy:", ann_accuracy)

X_train_cnn = X_train_dl.reshape(
    X_train_dl.shape[0],
    X_train_dl.shape[1],
    1
)

X_test_cnn = X_test_dl.reshape(
    X_test_dl.shape[0],
    X_test_dl.shape[1],
    1
)

print("CNN data prepared:", X_train_cnn.shape)

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

cnn = Sequential([
    Conv1D(32, kernel_size=3, activation="relu",
           input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation="relu"),
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.30),
    Dense(1, activation="sigmoid")
])

cnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
cnn.fit(
    X_train_cnn,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=1
)

cnn_loss, cnn_accuracy = cnn.evaluate(
    X_test_cnn,
    y_test,
    verbose=0
)

print("CNN Accuracy:", cnn_accuracy)

from xgboost import XGBClassifier

xgboost_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgboost_model.fit(X_train_processed, y_train)

y_pred_xgb = xgboost_model.predict(X_test_processed)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print("XGBoost Accuracy:", accuracy_xgb)

print("\n===== MODEL COMPARISON =====")

results = {
    "Logistic Regression": accuracy,
    "Decision Tree": accuracy_tree,
    "Random Forest": accuracy_rf,
    "Gradient Boosting": accuracy_gb,
    "XGBoost": accuracy_xgb,
    "ANN": ann_accuracy,
    "CNN": cnn_accuracy
}

for model_name, score in results.items():
    print(model_name, ":", score)

# Find the best ML model based on accuracy
ml_results = {
    "Logistic Regression": accuracy,
    "Decision Tree": accuracy_tree,
    "Random Forest": accuracy_rf,
    "Gradient Boosting": accuracy_gb,
    "XGBoost": accuracy_xgb
}

best_model_name = max(ml_results, key=ml_results.get)

print("\nBest ML Model:", best_model_name)
print("Best ML Accuracy:", ml_results[best_model_name])

import joblib

# Select the actual best ML model
best_models = {
    "Logistic Regression": model,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Gradient Boosting": gradient_boosting,
    "XGBoost": xgboost_model
}

best_model = best_models[best_model_name]

# Create one complete pipeline
best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

# Train the complete pipeline on the original training data
best_pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(best_pipeline, "titanic_best_model.pkl")

print("\nBest model saved as titanic_best_model.pkl")

# ==========================================
# FINAL MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("\n===== FINAL MODEL EVALUATION =====")

# ML predictions
ml_predictions = {
    "Logistic Regression": (
        model.predict(X_test_processed),
        model.predict_proba(X_test_processed)[:, 1]
    ),
    "Decision Tree": (
        decision_tree.predict(X_test_processed),
        decision_tree.predict_proba(X_test_processed)[:, 1]
    ),
    "Random Forest": (
        random_forest.predict(X_test_processed),
        random_forest.predict_proba(X_test_processed)[:, 1]
    ),
    "Gradient Boosting": (
        gradient_boosting.predict(X_test_processed),
        gradient_boosting.predict_proba(X_test_processed)[:, 1]
    ),
    "XGBoost": (
        xgboost_model.predict(X_test_processed),
        xgboost_model.predict_proba(X_test_processed)[:, 1]
    )
}

# ANN predictions
ann_prob = ann.predict(X_train_dl if False else X_test_dl, verbose=0).ravel()
ann_pred = (ann_prob >= 0.5).astype(int)

ml_predictions["ANN"] = (ann_pred, ann_prob)

# CNN predictions
cnn_prob = cnn.predict(X_test_cnn, verbose=0).ravel()
cnn_pred = (cnn_prob >= 0.5).astype(int)

ml_predictions["CNN"] = (cnn_pred, cnn_prob)

# Calculate metrics
for model_name, (predictions, probabilities) in ml_predictions.items():

    accuracy_score_value = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n", model_name)
    print("Accuracy :", round(accuracy_score_value, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

# Confusion matrix for the best ML model
best_predictions = ml_predictions[best_model_name][0]

cm = confusion_matrix(y_test, best_predictions)

print("\n===== CONFUSION MATRIX =====")
print(cm)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    best_predictions,
    target_names=["Did Not Survive", "Survived"]
))

# ==========================================
# CONFUSION MATRIX GRAPH
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Did Not Survive", "Survived"],
    yticklabels=["Did Not Survive", "Survived"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")

plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()

# ==========================================
# MODEL COMPARISON GRAPH
# ==========================================

model_names = list(results.keys())
accuracy_scores = list(results.values())

plt.figure(figsize=(10, 5))

plt.bar(model_names, accuracy_scores)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.xticks(rotation=30)
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig("model_comparison.png")
plt.show()

# ==========================================
# SAVE MODEL RESULTS
# ==========================================

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

results_df.to_csv("model_results.csv", index=False)

print("\nModel results saved to model_results.csv")