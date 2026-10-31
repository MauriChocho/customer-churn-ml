# Pruebas de la carga del modelo (cargar_modelo en src/inference/predict.py)
# La ultima prueba requiere tener bajado el modelo: dvc pull models/churn_pipeline.joblib.dvc

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.inference.predict import cargar_modelo


def test_cargar_modelo_devuelve_el_pipeline_guardado(tmp_path):
    # Armar un pipeline chico y guardarlo en una carpeta temporal
    ruta = tmp_path / "modelo_de_prueba.joblib"
    pipeline_original = Pipeline([
        ("preprocessing", StandardScaler()),
        ("model", DummyClassifier()),
    ])
    joblib.dump(pipeline_original, ruta)

    # Cargarlo con la funcion del equipo
    pipeline_cargado = cargar_modelo(ruta)

    # Verificar que es un pipeline con los mismos pasos
    assert isinstance(pipeline_cargado, Pipeline)
    assert list(pipeline_cargado.named_steps) == ["preprocessing", "model"]


def test_cargar_modelo_sin_archivo_avisa_como_descargarlo(tmp_path):
    ruta = tmp_path / "no_existe.joblib"

    with pytest.raises(FileNotFoundError, match="dvc pull"):
        cargar_modelo(ruta)


def test_el_modelo_real_es_un_pipeline_completo():
    # Usa el archivo real de models/, que se baja con dvc pull
    pipeline = cargar_modelo()

    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "predict_proba")