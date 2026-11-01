# Prueba del endpoint POST /predict con un cliente valido

from fastapi.testclient import TestClient

from app.main import app

# Datos del cliente valido 
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