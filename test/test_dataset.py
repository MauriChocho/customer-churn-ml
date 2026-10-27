# Pruebas de estructura del dataset historico
# Objetivo: comprobar que el CSV tenga las columnas y tipos que necesitan el preprocessing y los modelos para entrenar y predecir.
# Requiere haber ejecutado "dvc pull" para tener el dataset local.

from pathlib import Path

import pandas as pd
import pytest

from src.features.preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    TARGET,
)

DATA_PATH = Path("data/raw/customer_churn_historical.csv")


# Cargar el dataset una sola vez para todas las pruebas del archivo
@pytest.fixture(scope="module")
def df():
    assert DATA_PATH.exists(), (
        f"No se encontro {DATA_PATH}. Ejecutar 'dvc pull' antes de correr las pruebas."
    )
    return pd.read_csv(DATA_PATH)


# Probar que existan todas las columnas que usa el proyecto
def test_columnas_obligatorias(df):
    columnas_esperadas = (
        [ID_COL, TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    )

    faltantes = [col for col in columnas_esperadas if col not in df.columns]

    assert faltantes == [], f"Faltan columnas en el dataset: {faltantes}"


# Probar que las variables numericas sean numericas
def test_tipos_numericos(df):
    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} deberia ser numerica"


# Probar que la variable objetivo solo tenga los valores esperados
def test_valores_target(df):
    assert set(df[TARGET].unique()) == {"Yes", "No"}


# Probar que SeniorCitizen este codificada como 0/1
def test_senior_citizen_binaria(df):
    assert set(df["SeniorCitizen"].unique()) <= {0, 1}


# Probar que no haya clientes duplicados
def test_customer_id_unico(df):
    assert df[ID_COL].is_unique