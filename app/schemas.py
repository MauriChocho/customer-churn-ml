# Datos que recibe la API para hacer una predicción de churn
# Los nombres tienen que coincidir con los del preprocessing

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Opciones que tiene cada campo en el dataset
SiNo = Literal["No", "Yes"]
SiNoSinInternet = Literal["No", "No internet service", "Yes"]


# Si llega customerID u otro campo que no usamos, se ignora
class PredictionRequest(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: SiNo
    Dependents: SiNo

    # No se fija un máximo: un cliente con más de 72 meses también es válido
    tenure: int = Field(ge=0)

    PhoneService: SiNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: SiNoSinInternet
    OnlineBackup: SiNoSinInternet
    DeviceProtection: SiNoSinInternet
    TechSupport: SiNoSinInternet
    StreamingTV: SiNoSinInternet
    StreamingMovies: SiNoSinInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SiNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(gt=0)

    # Si viene vacío, después se completa con la mediana
    TotalCharges: float | None = Field(default=None, ge=0)

# Datos que devuelve la API después de una predicción
# Los valores salen de predecir() en src/inference/predict.py
class PredictionResponse(BaseModel):
    prediction: Literal["CHURN", "NO_CHURN"]
    probability: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    request_id: str

    # Ejemplo que se muestra en Swagger (/docs)
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prediction": "CHURN",
                    "probability": 0.81,
                    "risk_level": "HIGH",
                    "model_version": "1",
                    "request_id": "3f1c9a2e-8b7d-4e5f-9a1b-2c3d4e5f6a7b",
                }
            ]
        }
    )