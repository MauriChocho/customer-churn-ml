# Pruebas del contrato HTTP de la API y de la documentacion Swagger/OpenAPI
# Complementan a test_predict.py: aca no se repite el caso valido ni los codigos 503/500/422,
# sino que se controla la consistencia entre endpoints y lo que la API documenta.
# No necesitan el modelo real: se reemplaza por un pipeline falso al iniciar la API.

import json
from uuid import UUID

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as api

# Cliente valido de ejemplo (19 columnas que espera el pipeline)
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

CAMPOS_RESPUESTA = {"prediction", "probability", "risk_level", "model_version", "request_id"}


# Pipeline falso: devuelve siempre la misma probabilidad, sin cargar el modelo real
class PipelineFalso:
    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


# Inicia la API con el pipeline falso en lugar del modelo descargado con DVC
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "cargar_modelo", lambda: PipelineFalso())

    with TestClient(api.app) as c:
        yield c


# Esquema OpenAPI que genera FastAPI (es lo que muestra Swagger en /docs)
@pytest.fixture
def openapi(client):
    return client.get("/openapi.json").json()


# ---------- Consistencia entre endpoints ----------

def test_model_version_igual_en_health_predict_y_model_meta(client):
    # La version que informa /health tiene que ser la misma que devuelve /predict
    # y la que figura en models/model_meta.json (trazabilidad del modelo)
    version_health = client.get("/health").json()["model_version"]
    version_predict = client.post("/predict", json=CLIENTE_VALIDO).json()["model_version"]
    version_meta = json.loads(api.META_PATH.read_text())["model_version"]

    assert version_health == version_predict == version_meta


def test_request_id_es_un_uuid_distinto_en_cada_pedido(client):
    primera = client.post("/predict", json=CLIENTE_VALIDO).json()["request_id"]
    segunda = client.post("/predict", json=CLIENTE_VALIDO).json()["request_id"]

    # Tiene formato UUID y no se repite entre pedidos
    assert UUID(primera).version == 4
    assert primera != segunda


def test_la_api_no_devuelve_campos_fuera_del_contrato(client, monkeypatch):
    # Si la inferencia devolviera un dato de mas, response_model lo filtra
    def predecir_con_campo_extra(*args, **kwargs):
        return {
            "prediction": "CHURN",
            "probability": 0.8,
            "risk_level": "HIGH",
            "model_version": "1",
            "dato_interno": "no deberia salir",
        }

    monkeypatch.setattr(api, "predecir", predecir_con_campo_extra)

    respuesta = client.post("/predict", json=CLIENTE_VALIDO)

    assert respuesta.status_code == 200
    assert set(respuesta.json()) == CAMPOS_RESPUESTA


def test_la_api_no_entrega_una_respuesta_fuera_del_contrato(monkeypatch):
    # Si la inferencia devolviera una probabilidad invalida, la API no la entrega como valida
    def predecir_con_probabilidad_invalida(*args, **kwargs):
        return {
            "prediction": "CHURN",
            "probability": 1.5,
            "risk_level": "HIGH",
            "model_version": "1",
        }

    monkeypatch.setattr(api, "cargar_modelo", lambda: PipelineFalso())
    monkeypatch.setattr(api, "predecir", predecir_con_probabilidad_invalida)

    # raise_server_exceptions=False: queremos ver la respuesta HTTP, no la excepcion de Python
    with TestClient(api.app, raise_server_exceptions=False) as c:
        respuesta = c.post("/predict", json=CLIENTE_VALIDO)

    assert respuesta.status_code == 500


# ---------- Documentacion Swagger / OpenAPI ----------

def test_swagger_esta_disponible(client):
    respuesta = client.get("/docs")

    assert respuesta.status_code == 200
    assert "swagger" in respuesta.text.lower()


def test_openapi_documenta_los_dos_endpoints(openapi):
    assert "get" in openapi["paths"]["/health"]
    assert "post" in openapi["paths"]["/predict"]


def test_openapi_predict_usa_los_contratos_de_entrada_y_salida(openapi):
    predict = openapi["paths"]["/predict"]["post"]

    entrada = predict["requestBody"]["content"]["application/json"]["schema"]
    salida = predict["responses"]["200"]["content"]["application/json"]["schema"]

    assert entrada["$ref"] == "#/components/schemas/PredictionRequest"
    assert salida["$ref"] == "#/components/schemas/PredictionResponse"

    # FastAPI documenta tambien la respuesta de error de validacion
    assert "422" in predict["responses"]


def test_openapi_contrato_de_salida_completo_y_con_ejemplo(openapi):
    esquema = openapi["components"]["schemas"]["PredictionResponse"]

    # Los cinco campos aparecen como obligatorios
    assert set(esquema["required"]) == CAMPOS_RESPUESTA

    # El ejemplo definido en schemas.py se muestra en Swagger
    assert esquema["examples"]


def test_openapi_contrato_de_entrada_marca_los_obligatorios(openapi):
    esquema = openapi["components"]["schemas"]["PredictionRequest"]

    # Todos los campos son obligatorios salvo TotalCharges, que es opcional
    assert set(esquema["required"]) == set(CLIENTE_VALIDO) - {"TotalCharges"}
