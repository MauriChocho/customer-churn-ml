# Importar librerías
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.features.preprocessing import ID_COL, TARGET, build_preprocessor

DATA_PATH = Path("data/raw/customer_churn_historical.csv")


# Cargar una muestra de las variables predictoras
def cargar_muestra(cantidad=200):
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=[TARGET, ID_COL]).head(cantidad)


# Convertir la salida a matriz densa si el ColumnTransformer la devuelve dispersa
def a_matriz(salida):
    return salida.toarray() if sparse.issparse(salida) else np.asarray(salida)


# Probar que el preprocesamiento no pierde registros y genera una salida válida
def test_preprocesamiento_conserva_registros_y_salida_valida():
    X = cargar_muestra()

    salida = a_matriz(build_preprocessor().fit_transform(X))

    assert salida.ndim == 2
    assert salida.shape[0] == len(X)
    assert np.isfinite(salida.astype(float)).all()


# Probar que datos nuevos generan la misma cantidad de columnas que en el entrenamiento
def test_preprocesamiento_mantiene_columnas_en_datos_nuevos():
    X = cargar_muestra()
    X_entrenamiento, X_nuevos = X.iloc[:150], X.iloc[150:]

    preprocesador = build_preprocessor()
    columnas_entrenamiento = a_matriz(preprocesador.fit_transform(X_entrenamiento)).shape[1]
    salida_nuevos = a_matriz(preprocesador.transform(X_nuevos))

    assert salida_nuevos.shape == (len(X_nuevos), columnas_entrenamiento)