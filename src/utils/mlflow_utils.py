# Importar librerías
import dagshub
import mlflow
import mlflow.sklearn


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


# Registrar los resultados de un entrenamiento en MLflow
def registrar_experimento(nombre_run, parametros, metricas, etiquetas, pipeline):
    # Iniciar una ejecución para registrar el entrenamiento
    with mlflow.start_run(run_name=nombre_run):

        # Registrar los parámetros utilizados para entrenar el modelo
        mlflow.log_params(parametros)

        # Registrar las métricas obtenidas al evaluar el modelo
        mlflow.log_metrics(metricas)

        # Registrar las etiquetas asociadas al modelo
        mlflow.set_tags(etiquetas)

        # Registrar el pipeline completo de machine learning
        mlflow.sklearn.log_model(pipeline, name="model")