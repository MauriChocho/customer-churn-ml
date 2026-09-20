# Entrenamiento del modelo Decision Tree

from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocessing import ID_COL, TARGET, build_preprocessor


import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Ruta del dataset
DATA_PATH = Path("data/raw/customer_churn_historical.csv")


# Cargar dataset
df = pd.read_csv(DATA_PATH)


# Separar variables predictoras
X = df.drop(columns=[TARGET, ID_COL])


# Preparar variable objetivo
y = df[TARGET].map({
    "No": 0,
    "Yes": 1,
})


# Separar datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Construir preprocessing comun
preprocessor = build_preprocessor()


# Construir pipeline completo
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        class_weight="balanced"
    )),
])

# Entrenar pipeline completo
model_pipeline.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model_pipeline.predict(X_test)

# Calcular matriz de confusion
cm = confusion_matrix(y_test, y_pred)

# Obtener probabilidades de churn
y_proba = model_pipeline.predict_proba(X_test)[:, 1]


# Calcular metricas sobre test
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)

# Calcular F1 sobre train, para comparar y detectar overfitting
y_train_pred = model_pipeline.predict(X_train)
f1_train = f1_score(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)

# Validacion del split
print("Forma de X_train:", X_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de y_test:", y_test.shape)

print("\nDistribucion de y_train:")
print(y_train.value_counts(normalize=True).round(4))

print("\nDistribucion de y_test:")
print(y_test.value_counts(normalize=True).round(4))

print("\nEntrenamiento finalizado correctamente.")

print("\nMetricas del modelo Decision Tree (test):")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print("\nMetricas sobre train (para detectar overfitting):")
print(f"F1-score (train): {f1_train:.4f}")
print(f"Accuracy (train): {accuracy_train:.4f}")

print("\nMatriz de confusion:")
print(cm)

# Mostrar matriz de confusion en grafico
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
disp.plot()

plt.title("Matriz de confusion - Decision Tree")
plt.show()