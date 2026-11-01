# Pruebas del contrato de salida de la API (PredictionResponse)
# No necesitan el modelo real ni el dataset

from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas import PredictionResponse
from src.inference.predict import (
    ETIQUETA_CHURN,
    ETIQUETA_NO_CHURN,
    UMBRAL_HIGH,
    UMBRAL_MEDIUM,
    calcular_risk_level,
    predecir,
)

# Los cuatro campos de la consigna más request_id, que lo agrega el endpoint
CAMPOS_RESPUESTA = {"prediction", "probability", "risk_level", "model_version", "request_id"}

RESPUESTA_VALIDA = {
    "prediction": "CHURN",
    "probability": 0.81,
    "risk_level": "HIGH",
    "model_version": "1",
    "request_id": "test-123",
}

# Cliente con las 19 columnas que espera el pipeline
CLIENTE = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 32,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 101.35,
    "TotalCharges": 2865.88,
}


# Pipeline falso: devuelve la probabilidad indicada, sin cargar el modelo real
class PipelineFalso:
    def __init__(self, probabilidad):
        self.probabilidad = probabilidad

    def predict(self, X):
        return np.array([1 if self.probabilidad >= 0.5 else 0])

    def predict_proba(self, X):
        return np.array([[1 - self.probabilidad, self.probabilidad]])


def test_respuesta_valida_se_acepta():
    respuesta = PredictionResponse(**RESPUESTA_VALIDA)

    assert respuesta.model_dump() == RESPUESTA_VALIDA


def test_la_respuesta_tiene_los_campos_acordados():
    assert set(PredictionResponse.model_fields) == CAMPOS_RESPUESTA


@pytest.mark.parametrize("campo", sorted(CAMPOS_RESPUESTA))
def test_falta_un_campo(campo):
    respuesta = dict(RESPUESTA_VALIDA)
    del respuesta[campo]

    with pytest.raises(ValidationError):
        PredictionResponse(**respuesta)


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("prediction", "Yes"),
        ("prediction", 1),
        ("probability", 1.5),
        ("probability", -0.1),
        ("risk_level", "VERY_HIGH"),
        ("risk_level", "high"),
        ("model_version", 1),
        ("request_id", None),
    ],
)
def test_valor_invalido_se_rechaza(campo, valor):
    respuesta = dict(RESPUESTA_VALIDA, **{campo: valor})

    with pytest.raises(ValidationError):
        PredictionResponse(**respuesta)


def test_etiquetas_coinciden_con_la_inferencia():
    # Si alguien cambia las etiquetas en predict.py, esta prueba avisa
    etiquetas_contrato = set(get_args(PredictionResponse.model_fields["prediction"].annotation))

    assert etiquetas_contrato == {ETIQUETA_CHURN, ETIQUETA_NO_CHURN}


def test_niveles_de_riesgo_coinciden_con_la_inferencia():
    niveles_contrato = set(get_args(PredictionResponse.model_fields["risk_level"].annotation))
    niveles_inferencia = {
        calcular_risk_level(0.0),
        calcular_risk_level(UMBRAL_MEDIUM),
        calcular_risk_level(UMBRAL_HIGH),
    }

    assert niveles_contrato == niveles_inferencia


@pytest.mark.parametrize("probabilidad", [0.05, 0.30, 0.80])
def test_la_salida_de_predecir_cumple_el_contrato(probabilidad):
    # Una probabilidad por cada nivel de riesgo (LOW, MEDIUM, HIGH)
    resultado = predecir(PipelineFalso(probabilidad), CLIENTE, model_version="1")

    # predecir() no genera request_id: lo agrega el endpoint, acá se simula
    resultado["request_id"] = "test-123"

    respuesta = PredictionResponse(**resultado)

    assert respuesta.probability == pytest.approx(probabilidad)