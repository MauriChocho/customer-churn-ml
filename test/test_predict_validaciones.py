# Pruebas de entradas invalidas en POST /predict
# Cada caso tiene que devolver HTTP 422 e indicar el campo con error

from fastapi.testclient import TestClient

from app.main import app

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


def enviar(cliente):
    with TestClient(app) as api:
        return api.post("/predict", json=cliente)


def campos_con_error(respuesta):
    # En la API el error viene como ["body", "campo"]
    return {detalle["loc"][1] for detalle in respuesta.json()["detail"]}


def test_predict_falta_un_campo():
    cliente = dict(CLIENTE_VALIDO)
    del cliente["Contract"]

    respuesta = enviar(cliente)

    assert respuesta.status_code == 422
    assert campos_con_error(respuesta) == {"Contract"}


def test_predict_categoria_que_no_existe():
    cliente = dict(CLIENTE_VALIDO, Contract="Weekly")

    respuesta = enviar(cliente)

    assert respuesta.status_code == 422
    assert campos_con_error(respuesta) == {"Contract"}


def test_predict_tipo_de_dato_incorrecto():
    cliente = dict(CLIENTE_VALIDO, tenure="abc")

    respuesta = enviar(cliente)

    assert respuesta.status_code == 422
    assert campos_con_error(respuesta) == {"tenure"}


def test_predict_pedido_vacio():
    # Tienen que aparecer todos los campos obligatorios: todos menos TotalCharges
    respuesta = enviar({})

    assert respuesta.status_code == 422
    assert campos_con_error(respuesta) == set(CLIENTE_VALIDO) - {"TotalCharges"}