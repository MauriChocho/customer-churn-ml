# Entrenamiento del modelo SVM (Support Vector Classifier) para prediccion de churn
# Objetivo: preparar los datos y entrenar el modelo usando el preprocessing comun
# Ejecutar desde la raiz del proyecto con: python -m src.training.train_svm

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
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

# Importar funciones compartidas para registrar el entrenamiento en MLflow
from src.utils.mlflow_utils import configurar_mlflow, registrar_experimento


# Configurar MLflow para registrar el entrenamiento en DagsHub
configurar_mlflow()


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
# Mismas condiciones que el resto del equipo, para que la comparacion sea justa
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Preprocessing comun (definido en src/features/preprocessing.py)
preprocessor = build_preprocessor()


# Construir pipeline completo
# class_weight="balanced" se eligio porque mejoro el F1-score frente a la
# configuracion por defecto (ver alternativas probadas en pre_train_svm.py)
# Se usa CalibratedClassifierCV en vez de SVC(probability=True) porque ese
# parametro quedo deprecado en scikit-learn 1.9
svm_base = SVC(
    kernel="rbf",
    C=1.0,
    class_weight="balanced",
    random_state=42,
)

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", CalibratedClassifierCV(svm_base, ensemble=False)),
])

# Entrenar pipeline completo
model_pipeline.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calcular matriz de confusion
cm = confusion_matrix(y_test, y_pred)


# Calcular metricas
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)

# Obtener los parámetros del SVM y de la calibración
# Se toman de svm_base porque model_pipeline.named_steps["model"] es el
# CalibratedClassifierCV, y su get_params() mezcla los parámetros del SVC
# con un objeto entero, cosa que en MLflow queda poco legible
calibrador = model_pipeline.named_steps["model"]

parametros = svm_base.get_params()
parametros["calibracion_method"] = calibrador.method
parametros["calibracion_ensemble"] = calibrador.ensemble
parametros["calibracion_cv"] = calibrador.cv

# Preparar las métricas con las claves acordadas
metricas = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "accuracy": accuracy,
}

# Preparar las etiquetas para identificar el modelo
etiquetas = {
    "modelo": "SVM",
}

# Registrar el entrenamiento del SVM en MLflow
registrar_experimento(
    nombre_run="svm",
    parametros=parametros,
    metricas=metricas,
    etiquetas=etiquetas,
    pipeline=model_pipeline,
)

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

print("\nMetricas del modelo SVM:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print("\nMatriz de confusion:")
print(cm)

