# Importar librerías
import runpy

import pytest
from sklearn.pipeline import Pipeline

import src.utils.mlflow_utils as mlflow_utils

# Claves de métricas acordadas por el equipo
METRICAS_ACORDADAS = {"precision", "recall", "f1", "roc_auc", "accuracy"}

# Scripts integrados con MLflow y el nombre de run que deben usar
# Agregar una línea cuando cada integrante termine su integración
MODELOS = [
    ("src.training.train_svm", "svm"),
    ("src.training.train_random_forest", "random_forest"),
    # ("src.training.train_baseline", "baseline"),
    # ("src.training.train_regresion_logistica", "regresion_logistica"),
    # ("src.training.train_arbol_decision", "arbol_decision"),
]


# Probar que cada script entrega correctamente su información a MLflow
@pytest.mark.parametrize("modulo, nombre_run_esperado", MODELOS)
def test_script_registra_en_mlflow(monkeypatch, modulo, nombre_run_esperado):
    # Evitar que los scripts abran ventanas de gráficos durante la prueba
    monkeypatch.setenv("MPLBACKEND", "Agg")

    llamadas = {}

    # Simular la conexión con DagsHub
    def simular_configurar_mlflow():
        llamadas["configurado"] = True

    # Simular el registro y guardar lo que recibió
    def simular_registrar_experimento(nombre_run, parametros, metricas, etiquetas, pipeline):
        llamadas["nombre_run"] = nombre_run
        llamadas["parametros"] = parametros
        llamadas["metricas"] = metricas
        llamadas["etiquetas"] = etiquetas
        llamadas["pipeline"] = pipeline

    monkeypatch.setattr(mlflow_utils, "configurar_mlflow", simular_configurar_mlflow)
    monkeypatch.setattr(mlflow_utils, "registrar_experimento", simular_registrar_experimento)

    # Ejecutar el script de entrenamiento completo
    runpy.run_module(modulo, run_name="__main__")

    # Verificar que configuró MLflow
    assert llamadas.get("configurado") is True

    # Verificar el nombre del run
    assert llamadas["nombre_run"] == nombre_run_esperado

    # Verificar que envió exactamente las cinco métricas acordadas, con valores válidos
    assert set(llamadas["metricas"]) == METRICAS_ACORDADAS
    for valor in llamadas["metricas"].values():
        assert 0 <= valor <= 1

    # Verificar que envió parámetros y una etiqueta que identifica el modelo
    assert len(llamadas["parametros"]) > 0
    assert "modelo" in llamadas["etiquetas"]

    # Verificar que envió el pipeline completo (preprocesamiento + modelo)
    assert isinstance(llamadas["pipeline"], Pipeline)
    assert hasattr(llamadas["pipeline"], "predict_proba")