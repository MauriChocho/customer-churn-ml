# Pruebas de la capa de inferencia (src/inference/predict.py)
# Los clientes de ejemplo se arman dentro de la prueba, sin depender del CSV
# Requiere tener bajado el modelo: dvc pull models/churn_pipeline.joblib.dvc

import pytest

from src.inference.predict import (
    ETIQUETA_CHURN,
    ETIQUETA_NO_CHURN,
    UMBRAL_HIGH,
    UMBRAL_MEDIUM,
    calcular_risk_level,
    cargar_modelo,
    predecir,
)

# Cliente de ejemplo con las 19 columnas que espera el pipeline
CLIENTE_EJEMPLO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
}


@pytest.fixture(scope="module")
def pipeline():
    # Carga el modelo una sola vez para todas las pruebas de este archivo
    return cargar_modelo()


def test_prediccion_valida(pipeline):
    resultado = predecir(pipeline, CLIENTE_EJEMPLO, model_version="test")

    assert resultado["prediction"] in (ETIQUETA_CHURN, ETIQUETA_NO_CHURN)


def test_estructura_y_tipos_de_la_respuesta(pipeline):
    resultado = predecir(pipeline, CLIENTE_EJEMPLO, model_version=3)

    # Los cuatro campos que exige la consigna, ni mas ni menos
    assert set(resultado.keys()) == {
        "prediction",
        "probability",
        "risk_level",
        "model_version",
    }

    assert isinstance(resultado["prediction"], str)
    assert isinstance(resultado["probability"], float)
    assert 0.0 <= resultado["probability"] <= 1.0
    assert resultado["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    # La version se devuelve siempre como texto
    assert resultado["model_version"] == "3"


def test_columna_extra_se_ignora(pipeline):
    # Si llega customerID de mas, no debe romper la prediccion
    cliente = dict(CLIENTE_EJEMPLO, customerID="0000-TEST")

    resultado = predecir(pipeline, cliente, model_version="test")

    assert resultado["prediction"] in (ETIQUETA_CHURN, ETIQUETA_NO_CHURN)


@pytest.mark.parametrize("columna", ["tenure", "Contract", "SeniorCitizen"])
def test_faltan_columnas(pipeline, columna):
    # Si falta una columna, debe avisar cual es con un error claro
    cliente = dict(CLIENTE_EJEMPLO)
    del cliente[columna]

    with pytest.raises(ValueError, match=columna):
        predecir(pipeline, cliente, model_version="test")


def test_risk_level_respeta_los_umbrales():
    # Se usan las constantes para verificar los niveles de riesgo
    assert UMBRAL_MEDIUM < UMBRAL_HIGH
    assert calcular_risk_level(0.0) == "LOW"
    assert calcular_risk_level(UMBRAL_MEDIUM) == "MEDIUM"
    assert calcular_risk_level(UMBRAL_HIGH) == "HIGH"
    assert calcular_risk_level(1.0) == "HIGH"
def test_umbrales_acordados_por_el_equipo():
    # Umbrales acordados por el equipo
    # Si se cambian a proposito, hay que actualizar esta prueba
    assert UMBRAL_MEDIUM == 0.20
    assert UMBRAL_HIGH == 0.50