# Pruebas para verificar los datos que recibe la API
# Se usan datos de ejemplo, sin necesidad de cargar el dataset ni el modelo

import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest
from src.inference.predict import COLUMNAS_ENTRADA

# Cliente valido: es el ejemplo predict_request_valid.json que entrego el profesor
CLIENTE_VALIDO = {
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


def campos_con_error(error):
    return {detalle["loc"][0] for detalle in error.errors()}


def test_cliente_valido_se_acepta():
    cliente = PredictionRequest(**CLIENTE_VALIDO)

    assert cliente.tenure == 32
    assert cliente.Contract == "Month-to-month"
    assert cliente.MonthlyCharges == 101.35


def test_ejemplo_invalido_del_profesor():
    # Es el ejemplo predict_request_invalid.json, con tres campos incorrectos
    cliente = dict(
        CLIENTE_VALIDO,
        tenure=-12,
        Contract="Weekly",
        MonthlyCharges="not-a-number",
    )

    with pytest.raises(ValidationError) as error:
        PredictionRequest(**cliente)

    assert campos_con_error(error.value) == {"tenure", "Contract", "MonthlyCharges"}


@pytest.mark.parametrize(
    "campo", ["gender", "SeniorCitizen", "tenure", "Contract", "MonthlyCharges"]
)
def test_falta_un_campo_obligatorio(campo):
    cliente = dict(CLIENTE_VALIDO)
    del cliente[campo]

    with pytest.raises(ValidationError) as error:
        PredictionRequest(**cliente)

    assert campos_con_error(error.value) == {campo}


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("gender", "male"),
        ("InternetService", "Cable"),
        ("Contract", "Weekly"),
        ("PaymentMethod", "Cash"),
        ("StreamingTV", "Maybe"),
    ],
)
def test_categoria_que_no_existe(campo, valor):
    cliente = dict(CLIENTE_VALIDO, **{campo: valor})

    with pytest.raises(ValidationError) as error:
        PredictionRequest(**cliente)

    assert campos_con_error(error.value) == {campo}


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("tenure", -1),
        ("MonthlyCharges", 0),
        ("MonthlyCharges", -5.0),
        ("TotalCharges", -1.0),
        ("SeniorCitizen", 2),
    ],
)
def test_valor_fuera_de_rango(campo, valor):
    cliente = dict(CLIENTE_VALIDO, **{campo: valor})

    with pytest.raises(ValidationError) as error:
        PredictionRequest(**cliente)

    assert campos_con_error(error.value) == {campo}


def test_tenure_no_tiene_maximo():
    # Un cliente con más de 72 meses de antigüedad también es válido
    cliente = dict(CLIENTE_VALIDO, tenure=80)

    assert PredictionRequest(**cliente).tenure == 80


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("tenure", "abc"),
        ("tenure", 32.5),
        ("MonthlyCharges", "abc"),
    ],
)
def test_tipo_de_dato_incorrecto(campo, valor):
    cliente = dict(CLIENTE_VALIDO, **{campo: valor})

    with pytest.raises(ValidationError) as error:
        PredictionRequest(**cliente)

    assert campos_con_error(error.value) == {campo}


def test_total_charges_es_opcional():
    # Se acepta tanto si no se envia el campo como si llega en null
    sin_campo = dict(CLIENTE_VALIDO)
    del sin_campo["TotalCharges"]
    con_null = dict(CLIENTE_VALIDO, TotalCharges=None)

    assert PredictionRequest(**sin_campo).TotalCharges is None
    assert PredictionRequest(**con_null).TotalCharges is None


def test_campo_de_mas_se_ignora():
    # Si llega customerID, no genera error y no queda en los datos del modelo
    cliente = dict(CLIENTE_VALIDO, customerID="0000-TEST")

    datos = PredictionRequest(**cliente).model_dump()

    assert "customerID" not in datos


def test_el_contrato_tiene_las_columnas_del_pipeline():
    # Comprueba que los campos de la API coincidan con los del pipeline
    datos = PredictionRequest(**CLIENTE_VALIDO).model_dump()

    assert set(datos) == set(COLUMNAS_ENTRADA)
