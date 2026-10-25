# Inferencia: carga del modelo y prediccion
# Adaptado del script de prueba de serializacion

from pathlib import Path

import joblib


# Ruta del modelo
# Se calcula desde la ubicacion de este archivo (src/inference/predict.py),
# asi funciona sin importar desde que carpeta se ejecute
RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
MODEL_PATH = RAIZ_PROYECTO / "models" / "churn_pipeline.joblib"


def cargar_modelo(ruta=MODEL_PATH):
    # Carga el pipeline completo (preprocessing + modelo) desde el archivo serializado
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {ruta}. "
            "Descargalo con: dvc pull models/churn_pipeline.joblib.dvc"
        )

    return joblib.load(ruta)


# Prueba manual
# Solo corre si se ejecuta este archivo directamente, no al importarlo
if __name__ == "__main__":
    import pandas as pd

    pipeline = cargar_modelo()

    df = pd.read_csv(RAIZ_PROYECTO / "data" / "raw" / "customer_churn_historical.csv")
    X = df.drop(columns=["Churn", "customerID"])

    # Toma una sola fila para simular una nueva prediccion
    cliente_prueba = X.iloc[[0]]

    prediccion = pipeline.predict(cliente_prueba)
    print("Prediccion:", prediccion)