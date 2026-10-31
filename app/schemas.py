
# Datos que recibe la API para hacer una predicción de churn
# Los nombres tienen que coincidir con los del preprocessing

from typing import Literal

from pydantic import BaseModel, Field

# Opciones que tiene cada campo en el dataset
SiNo = Literal["No", "Yes"]
SiNoSinInternet = Literal["No", "No internet service", "Yes"]


# Si llega customerID u otro campo que no usamos, se ignora
class PredictionRequest(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: SiNo
    Dependents: SiNo

    # En el dataset tenure llega hasta 72 meses
    tenure: int = Field(ge=0, le=72)

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
