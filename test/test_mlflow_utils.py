# Importar librerías
import dagshub
import mlflow

from src.utils.mlflow_utils import configurar_mlflow


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