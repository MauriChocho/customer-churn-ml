# Entrenamiento del modelo baseline para prediccion de churn
# Se usa el DummyClassifier como referencia para comparar con los demas modelos

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocessing import ID_COL, TARGET, build_preprocessor

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_PATH = Path("data/raw/customer_churn_historical.csv")


df = pd.read_csv(DATA_PATH)


# Saco el target y el ID de las variables predictoras
X = df.drop(columns=[TARGET, ID_COL])


# Paso Yes/No a 1/0
y = df[TARGET].map({
    "No": 0,
    "Yes": 1,
})


# Separamos los datos para entrenar y probar
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

preprocessor = build_preprocessor()


model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DummyClassifier(
        strategy="most_frequent",
        random_state=42,
    )),
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# Probabilidades de la clase 1 para calcular ROC-AUC
y_proba = model_pipeline.predict_proba(X_test)[:, 1]


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)

print("\nEntrenamiento finalizado correctamente.")

print("\nMetricas del modelo Baseline (DummyClassifier):")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print("\nMatriz de confusion:")
print(cm)