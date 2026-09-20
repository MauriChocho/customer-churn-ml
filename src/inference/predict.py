# Imports
# Librerias necesarias para cargar el pipeline serializado

import pandas as pd
import joblib
from pathlib import Path


# Ruta del modelo
# Define la ubicacion del pipeline entrenado

MODEL_PATH = Path("models") / "churn_pipeline.joblib"


# Cargar pipeline
# Carga el pipeline entrenado desde el archivo serializado

pipeline = joblib.load(MODEL_PATH)


# Cargar datos para prueba
# Carga el dataset historico y prepara las variables de entrada

df = pd.read_csv("data/raw/customer_churn_historical.csv")

X = df.drop(columns=["Churn", "customerID"])


# Seleccionar un cliente de prueba
# Toma una sola fila para simular una nueva prediccion

cliente_prueba = X.iloc[[0]]


# Realizar prediccion
# Usa el pipeline cargado para predecir churn del cliente seleccionado

prediccion = pipeline.predict(cliente_prueba)

print("Prediccion:", prediccion)