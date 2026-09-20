# Entrenamiento del modelo Logistic Regression
# objetivo: entrenar y evaluar Logistic Regression usando el pipeline comun del equipo

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

from src.features.preprocessing import TARGET, ID_COL, build_preprocessor

# Cargar datos historicos
df = pd.read_csv("data/raw/customer_churn_historical.csv")

# Separar features (X) y target (y)
X = df.drop(columns=[TARGET, ID_COL])
y = df[TARGET].map({"Yes": 1, "No": 0})

# Split acordado por el equipo
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Filas de entrenamiento:", X_train.shape[0])
print("Filas de test:", X_test.shape[0])
print("Proporcion de Churn en train:", y_train.mean().round(4))
print("Proporcion de Churn en test:", y_test.mean().round(4))

# Armar el pipeline completo: preprocesamiento + modelo
preprocessor = build_preprocessor()

modelo_logreg = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        C=0.1
    )),
])

print(modelo_logreg)

# Entrenar el pipeline completo
modelo_logreg.fit(X_train, y_train)

print("Entrenamiento finalizado.")

# Predicciones sobre el set de test
y_pred = modelo_logreg.predict(X_test)
y_proba = modelo_logreg.predict_proba(X_test)[:, 1]

# Metricas acordadas por el equipo
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("--- Metricas Logistic Regression ---")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

# Matriz de confusion y accuracy (como referencia adicional)
matriz = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy (referencia, no es la metrica principal):", round(accuracy, 4))
print("Matriz de confusion:")
print(matriz)

# Guardar el pipeline completo entrenado
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(modelo_logreg, MODEL_PATH)

print("Pipeline guardado en:", MODEL_PATH)
