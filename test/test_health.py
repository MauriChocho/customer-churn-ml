# Pruebas del endpoint GET /health, la API informa si esta lista y que version de modelo cargo.

import pytest
from fastapi.testclient import TestClient

import app.main as api


# Cliente de prueba: arranque de la API (carga del modelo)
@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


# La API levanta, carga el modelo y responde 200
def test_health_ok(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200

    datos = respuesta.json()
    assert datos["status"] == "ok"
    assert datos["model_loaded"] is True
    assert isinstance(datos["model_version"], str)


# Si el modelo no se puede cargar, la API sigue viva pero /health responde 503
def test_health_sin_modelo(monkeypatch):
    def simular_error():
        raise FileNotFoundError("modelo no encontrado")

    monkeypatch.setattr(api, "cargar_modelo", simular_error)

    with TestClient(api.app) as c:
        respuesta = c.get("/health")

    assert respuesta.status_code == 503
    assert respuesta.json()["model_loaded"] is False