# Pruebas del endpoint POST /predict

from fastapi.testclient import TestClient

from app.main import app


# Datos del cliente valido de ejemplo
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


# Verifica que la API acepte un cliente valido y devuelva una prediccion HTTP 200
def test_predict_cliente_valido():
    # Iniciamos la API para que se cargue el modelo
    with TestClient(app) as cliente:
        respuesta = cliente.post("/predict", json=CLIENTE_VALIDO)

    # Verificamos que la solicitud se proceso correctamente
    assert respuesta.status_code == 200

    # Verificamos los datos que devuelve la prediccion
    resultado = respuesta.json()
    assert resultado["prediction"] in ("CHURN", "NO_CHURN")
    assert 0 <= resultado["probability"] <= 1
    assert resultado["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(resultado["model_version"], str)
    
    # Verificamos que la respuesta tenga los cinco campos del contrato
    assert set(resultado.keys()) == {
        "prediction",
        "probability",
        "risk_level",
        "model_version",
        "request_id",
    }

    # Verificamos que se genere un identificador para la solicitud
    assert isinstance(resultado["request_id"], str)
    assert resultado["request_id"]


# Verifica que la API devuelva HTTP 503 cuando el modelo no esta disponible
def test_predict_modelo_no_disponible():
    # Iniciamos la API y simulamos que el modelo no esta disponible
    with TestClient(app) as cliente:
        app.state.pipeline = None

        respuesta = cliente.post("/predict", json=CLIENTE_VALIDO)

    # Sin modelo cargado, el endpoint debe devolver HTTP 503
    assert respuesta.status_code == 503
    assert respuesta.json()["detail"] == "El modelo no esta disponible"

# Verifica que la API devuelva HTTP 500 si ocurre un error inesperado al predecir
def test_predict_error_interno(monkeypatch):
    # Simulamos un fallo en la funcion de inferencia
    def predecir_con_error(*args, **kwargs):
        raise RuntimeError("Fallo simulado del modelo")

    monkeypatch.setattr("app.main.predecir", predecir_con_error)

    # Enviamos un cliente valido para llegar hasta la inferencia
    with TestClient(app) as cliente:
        respuesta = cliente.post("/predict", json=CLIENTE_VALIDO)

    # La API debe responder con un error controlado, sin exponer detalles internos
    assert respuesta.status_code == 500
    assert respuesta.json()["detail"] == "Error interno al realizar la prediccion"
    
# Verifica que la API rechace datos invalidos con HTTP 422
def test_predict_datos_invalidos():
    # Copiamos un cliente valido para modificar solamente un campo
    datos_invalidos = CLIENTE_VALIDO.copy()

    # tenure representa los meses como cliente y no puede ser negativo
    datos_invalidos["tenure"] = -1

    # Enviamos los datos incorrectos al endpoint
    with TestClient(app) as cliente:
        respuesta = cliente.post("/predict", json=datos_invalidos)

    # FastAPI debe rechazar la solicitud antes de llamar al modelo
    assert respuesta.status_code == 422