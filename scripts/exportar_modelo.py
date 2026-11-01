# Exporta el modelo registrado en el Model Registry a models/
# Uso: python -m scripts.exportar_modelo
import json
from pathlib import Path

import dagshub
import joblib
import mlflow
from mlflow import MlflowClient

NOMBRE = "churn_logistic_regression"
VERSION = "1"
CARPETA = Path("models")

dagshub.init(repo_owner="MauriChocho", repo_name="customer-churn-ml", mlflow=True)

# Bajar el pipeline registrado (preprocessing + modelo)
pipeline = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{VERSION}")
joblib.dump(pipeline, CARPETA / "churn_pipeline.joblib")

# Guarda de dónde salió, para la trazabilidad
info = MlflowClient().get_model_version(NOMBRE, VERSION)
meta = {"model_name": NOMBRE, "model_version": VERSION, "run_id": info.run_id}
(CARPETA / "model_meta.json").write_text(json.dumps(meta, indent=2))

print("Modelo exportado:", meta)