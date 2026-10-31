# Inferencia: carga del modelo y prediccion
# Adaptado del script de prueba de serializacion

from pathlib import Path

import joblib
import pandas as pd

from src.features.preprocessing import (
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
)


# Ruta del modelo
# Se calcula desde la ubicacion de este archivo (src/inference/predict.py),
# asi funciona sin importar desde que carpeta se ejecute
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
MODEL_PATH = RAIZ_PROYECTO / "models" / "churn_pipeline.joblib"

# Columnas que el pipeline espera como entrada (se reutilizan las del preprocessing)
COLUMNAS_ENTRADA = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Etiquetas de la prediccion
ETIQUETA_CHURN = "CHURN"
ETIQUETA_NO_CHURN = "NO_CHURN"

# Umbrales de risk_level
# Acordados por el equipo (propuesta B), segun el analisis de umbrales
# Justificacion en src/evaluation/justificacion_umbrales.md
UMBRAL_MEDIUM = 0.20
UMBRAL_HIGH = 0.50


def cargar_modelo(ruta=MODEL_PATH):
    # Carga el pipeline completo (preprocessing + modelo) desde el archivo serializado
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {ruta}. "
            "Descargalo con: dvc pull models/churn_pipeline.joblib.dvc"
        )

    return joblib.load(ruta)


def calcular_risk_level(probability):
    # Convierte la probabilidad de churn en un nivel de riesgo
    if probability >= UMBRAL_HIGH:
        return "HIGH"
    if probability >= UMBRAL_MEDIUM:
        return "MEDIUM"
    return "LOW"


def predecir(pipeline, datos_cliente, model_version):
    # Recibe un diccionario con los datos de un cliente y devuelve la prediccion

    # Validar que esten todas las columnas que necesita el pipeline
    faltantes = [col for col in COLUMNAS_ENTRADA if col not in datos_cliente]
    if faltantes:
        raise ValueError(f"Faltan columnas en los datos del cliente: {faltantes}")

    # Armar un DataFrame de una fila con las columnas esperadas
    # (si llegan columnas de mas, como customerID, se ignoran)
    cliente = pd.DataFrame([datos_cliente], columns=COLUMNAS_ENTRADA)

    # El pipeline ya incluye el preprocessing, no hay que transformar nada antes
    clase = int(pipeline.predict(cliente)[0])
    probabilidad = float(pipeline.predict_proba(cliente)[0, 1])

    return {
        "prediction": ETIQUETA_CHURN if clase == 1 else ETIQUETA_NO_CHURN,
        "probability": probabilidad,
        "risk_level": calcular_risk_level(probabilidad),
        "model_version": str(model_version),
    }


# Prueba manual
# Solo corre si se ejecuta este archivo directamente, no al importarlo
if __name__ == "__main__":
    pipeline = cargar_modelo()

    df = pd.read_csv(RAIZ_PROYECTO / "data" / "raw" / "customer_churn_historical.csv")
    X = df.drop(columns=["Churn", "customerID"])

    # Toma una sola fila para simular una nueva prediccion
    cliente_prueba = X.iloc[[0]]

    prediccion = pipeline.predict(cliente_prueba)
    print("Prediccion:", prediccion)

    # Prueba de la funcion predecir con el mismo cliente
    resultado = predecir(pipeline, X.iloc[0].to_dict(), model_version="sin-definir")
    print("Resultado:", resultado)