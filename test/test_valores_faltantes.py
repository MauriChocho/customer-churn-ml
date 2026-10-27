# Pruebas de manejo de valores faltantes en el preprocessing
# Objetivo: comprobar que el preprocesador complete los nulos y no pierda registros.
# Se usan datos chicos construidos aca, sin depender del CSV real.

import numpy as np
import pandas as pd

from src.features.preprocessing import build_preprocessor


# Construir un pequeño DataFrame con el mismo esquema que el dataset real
def crear_clientes():
    cliente_base = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }

    clientes = pd.DataFrame([cliente_base] * 4)

    # Variar algunos valores para que la mediana tenga sentido
    clientes["tenure"] = [1, 12, 24, 48]
    clientes["MonthlyCharges"] = [20.0, 50.0, 70.0, 100.0]
    clientes["TotalCharges"] = [20.0, 600.0, 1680.0, 4800.0]

    return clientes


# Caso real del EDA: TotalCharges vacio en clientes nuevos
def test_total_charges_faltante():
    clientes = crear_clientes()
    clientes.loc[0, "TotalCharges"] = np.nan

    preprocessor = build_preprocessor()
    resultado = preprocessor.fit_transform(clientes)

    if hasattr(resultado, "toarray"):
        resultado = resultado.toarray()

    # No quedan nulos luego del preprocessing
    assert not np.isnan(resultado).any()

    # No se pierden registros
    assert resultado.shape[0] == len(clientes)


# Nulos en varias columnas numericas a la vez
def test_varios_numericos_faltantes():
    clientes = crear_clientes()
    clientes.loc[1, "tenure"] = np.nan
    clientes.loc[2, "MonthlyCharges"] = np.nan
    clientes.loc[3, "TotalCharges"] = np.nan

    preprocessor = build_preprocessor()
    resultado = preprocessor.fit_transform(clientes)

    if hasattr(resultado, "toarray"):
        resultado = resultado.toarray()

    assert not np.isnan(resultado).any()
    assert resultado.shape[0] == len(clientes)


# Un cliente nuevo con nulo, usando un preprocesador ya entrenado
# (simula lo que pasa en inferencia: se usa la mediana aprendida en train)
def test_faltante_en_datos_nuevos():
    clientes = crear_clientes()

    preprocessor = build_preprocessor()
    preprocessor.fit(clientes)

    cliente_nuevo = clientes.iloc[[0]].copy()
    cliente_nuevo["TotalCharges"] = np.nan

    resultado = preprocessor.transform(cliente_nuevo)

    if hasattr(resultado, "toarray"):
        resultado = resultado.toarray()

    assert not np.isnan(resultado).any()
    assert resultado.shape[0] == 1