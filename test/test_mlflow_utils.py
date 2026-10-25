# Importar librerías
from contextlib import nullcontext

import dagshub
import mlflow

from src.utils.mlflow_utils import configurar_mlflow, registrar_experimento


# Probar la configuración de MLflow
def test_configurar_mlflow(monkeypatch):
    llamadas = {}

    # Reemplazar la conexión real con DagsHub
    def simular_dagshub_init(**kwargs):
        llamadas["dagshub"] = kwargs

    # Reemplazar la selección real del experimento
    def simular_set_experiment(nombre):
        llamadas["experimento"] = nombre

    monkeypatch.setattr(dagshub, "init", simular_dagshub_init)
    monkeypatch.setattr(mlflow, "set_experiment", simular_set_experiment)

    # Ejecutar la función que queremos probar
    configurar_mlflow()

    # Verificar que utilizó la configuración esperada
    assert llamadas["dagshub"] == {
        "repo_owner": "MauriChocho",
        "repo_name": "customer-churn-ml",
        "mlflow": True
    }

    assert llamadas["experimento"] == "churn-classification"
    

# Probar el registro de un experimento en MLflow
def test_registrar_experimento(monkeypatch):
    llamadas = {}

    # Simular el inicio de una ejecución de MLflow
    def simular_start_run(run_name):
        llamadas["nombre_run"] = run_name
        return nullcontext()

    monkeypatch.setattr(mlflow, "start_run", simular_start_run)
    
    # Simular el registro de parámetros
    def simular_log_params(parametros):
        llamadas["parametros"] = parametros

    monkeypatch.setattr(mlflow, "log_params", simular_log_params)
    
    # Simular el registro de métricas
    def simular_log_metrics(metricas):
        llamadas["metricas"] = metricas

    monkeypatch.setattr(mlflow, "log_metrics", simular_log_metrics)
    
    # Simular el registro de etiquetas
    def simular_set_tags(etiquetas):
        llamadas["etiquetas"] = etiquetas

    monkeypatch.setattr(mlflow, "set_tags", simular_set_tags)
    
    # Simular el registro del pipeline entrenado
    def simular_log_model(pipeline, name):
        llamadas["pipeline"] = pipeline
        llamadas["nombre_modelo"] = name

    monkeypatch.setattr(mlflow.sklearn, "log_model", simular_log_model)
    
    # Preparar datos de ejemplo para probar la función
    nombre_run = "random_forest"

    parametros = {
        "n_estimators": 100,
        "random_state": 42
    }

    metricas = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.79,
        "f1": 0.80,
        "roc_auc": 0.88
    }

    etiquetas = {
        "modelo": "RandomForest"
    }

    pipeline = object()
    
    # Ejecutar la función con los datos de prueba
    registrar_experimento(
        nombre_run,
        parametros,
        metricas,
        etiquetas,
        pipeline
    )
    
    # Verificar que se registraron los datos correctamente
    assert llamadas["nombre_run"] == nombre_run
    assert llamadas["parametros"] == parametros
    assert llamadas["metricas"] == metricas
    assert llamadas["etiquetas"] == etiquetas
    assert llamadas["pipeline"] is pipeline
    assert llamadas["nombre_modelo"] == "model"