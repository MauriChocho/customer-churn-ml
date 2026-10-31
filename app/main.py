#Crea la aplicacion FastAPI, cargar el modelo una sola vez al arrancar y exponer GET /health para saber si el servicio esta listo.

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.inference.predict import cargar_modelo

# Logs basicos: se ven en la terminal y en "docker compose logs"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("churn-api")

# Version del modelo registrado en el Model Registry (churn_logistic_regression v1)
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")


# Se ejecuta una vez al iniciar la API (antes de recibir pedidos)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.model_version = MODEL_VERSION

    try:
        app.state.pipeline = cargar_modelo()
        logger.info("Modelo cargado correctamente. Version: %s", MODEL_VERSION)
    except Exception:
        # La API arranca igual, pero /health avisa que no esta lista
        logger.exception("No se pudo cargar el modelo")

    yield

    logger.info("API detenida")


app = FastAPI(
    title="Customer Churn API",
    description="Predice la probabilidad de abandono de un cliente y su nivel de riesgo.",
    version="0.2.0",
    lifespan=lifespan,
)


# Estado del servicio y version del modelo cargado
@app.get("/health", tags=["Estado"])
def health():
    if app.state.pipeline is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "model_loaded": False,
                "model_version": None,
            },
        )

    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": app.state.model_version,
    }