# Crea la aplicacion FastAPI, carga el modelo una sola vez al arrancar
# y expone GET /health para saber si el servicio esta listo.

import json                      
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path        

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import PredictionRequest
from src.inference.predict import cargar_modelo, predecir

# Logs basicos: se ven en la terminal y en "docker compose logs"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("churn-api")

# archivo generado por scripts/exportar_modelo.py
META_PATH = Path(__file__).resolve().parents[1] / "models" / "model_meta.json"


#  la version del modelo exportado; si no existe, usa la variable de entorno
def leer_version():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())["model_version"]
    return os.getenv("MODEL_VERSION", "desconocida")


# Se ejecuta una vez al iniciar la API (antes de recibir pedidos)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.model_version = leer_version()          

    try:
        app.state.pipeline = cargar_modelo()
        logger.info("Modelo cargado correctamente. Version: %s", app.state.model_version)  
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
    
# Recibe los datos validados del cliente y devuelve la prediccion
@app.post("/predict", tags=["Prediccion"])
def predict(request: PredictionRequest):
    # Si el modelo no esta disponible, no podemos realizar la prediccion
    if app.state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no esta disponible",
        )

    # Pydantic entrega un diccionario con las variables validadas
    datos_cliente = request.model_dump()

    # Reutilizamos la inferencia y el modelo cargado al iniciar FastAPI
    # Si ocurre un error inesperado al predecir, devolvemos HTTP 500
    try:
        resultado = predecir(
            app.state.pipeline,
            datos_cliente,
            app.state.model_version,
        )
    except Exception:
        logger.exception("Error al realizar la prediccion")
        raise HTTPException(
            status_code=500,
            detail="Error interno al realizar la prediccion",
        ) from None

    return resultado