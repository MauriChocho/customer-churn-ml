# Importar librerías
import dagshub
import mlflow


# Configurar la conexión con MLflow y DagsHub
def configurar_mlflow():
    # Conectar MLflow con el repositorio del equipo en DagsHub
    dagshub.init(
        repo_owner="MauriChocho",
        repo_name="customer-churn-ml",
        mlflow=True
    )

    # Seleccionar el experimento compartido por todos los modelos
    mlflow.set_experiment("churn-classification")