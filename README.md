# Customer Churn ML

Proyecto integrador de la materia **Laboratorio de Minería de Datos** (ISTEA) — predicción de churn de clientes con un enfoque de MLOps end-to-end.

## Problema de negocio

Una empresa de telecomunicaciones necesita identificar clientes con riesgo de abandonar el
servicio (`Churn`). El sistema debe estimar la probabilidad de abandono de un cliente y
devolver un nivel de riesgo (`LOW` / `MEDIUM` / `HIGH`) consumible por otros sistemas
(por ejemplo, un CRM o una campaña de retención).
   El objetivo de esta primera etapa es desarrollar y comparar modelos capaces de predecir si un cliente presenta riesgo de churn.
A partir de esta prediccion, la empresa podria identificar clientes con mayor riesgo de abandono y orientar futuras acciones de retencion.


## Dataset

El proyecto utiliza el dataset historico:

`data/raw/customer_churn_historical.csv`

La variable objetivo es `Churn`, que indica si un cliente abandono o no el servicio.

El dataset no se versiona directamente con Git. Se encuentra gestionado con DVC y almacenado en el remote configurado en DagsHub.


## Estructura del repositorio

```
customer-churn-ml/
├── app/              # servicio de API (FastAPI) - entregas posteriores
├── data/
│   └── raw/          # dataset historico, versionado con DVC
├── models/           # pipelines/modelos serializados
├── monitoring/       # monitoreo del modelo en produccion - entregas posteriores
├── notebooks/        # notebooks de EDA y experimentacion
├── scripts/          # scripts auxiliares
├── src/
│   ├── data/          # carga y separacion de datos
│   ├── evaluation/    # calculo de metricas
│   ├── features/      # preprocessing / feature engineering
│   ├── inference/     # carga del pipeline entrenado y ejecucion de predicciones
│   └── training/      # entrenamiento de modelos
├── test/              # tests
├── requirements.txt
└── README.md
```

## Cómo levantar el proyecto

1. Clonar el repositorio
```bash
   git clone https://github.com/MauriChocho/customer-churn-ml.git
   cd customer-churn-ml
```

2. Crear y activar un entorno virtual
```bash
   python -m venv venv
   source .\venv\Scripts\Activate.ps1    # Windows
   source venv/bin/activate   # Linux/Mac
```

3. Instalar dependencias
```bash
   pip install -r requirements.txt
```

4. Configuración de acceso a los datos (DVC + DagsHub)

Los datos del proyecto están versionados con DVC y alojados en DagsHub, no en este repositorio.

Para poder descargarlos, cada persona debe configurar su propio acceso:

1. Crear una cuenta en DagsHub (si no la tenés).
2. Generar un token personal desde Settings → Tokens.
3. Configurar la autenticación local (reemplazando por tus propios datos):
 -Apartado "Data" -> Setup Credentials

   - dvc remote modify origin --local auth basic
   - dvc remote modify origin --local user <tu_usuario_dagshub>
   - dvc remote modify origin --local password <tu_token>

4. Recuperar los archivos versionados con DVC:

   dvc pull

Este comando recupera los archivos administrados por DVC, incluyendo el dataset historico y el pipeline serializado del modelo

## Ejecucion de los modulos

Los comandos deben ejecutarse desde la carpeta raiz del proyecto.

Esto es importante para que Python pueda reconocer correctamente los modulos internos dentro de `src/`.

### Seguimiento de experimentos con MLflow y DagsHub

El proyecto utiliza MLflow integrado con DagsHub para registrar y comparar los resultados de los modelos de clasificacion de churn.

Para evitar repetir codigo en cada script de entrenamiento, se crearon dos funciones compartidas en `src/utils/mlflow_utils.py`:

- `configurar_mlflow()`: configura la conexion con DagsHub y selecciona el experimento compartido `churn-classification`.
- `registrar_experimento()`: registra el nombre de la ejecucion, los parametros, las metricas, las etiquetas y el pipeline entrenado.

Los cinco modelos utilizan el mismo experimento para facilitar la comparacion de sus resultados.

#### Configurar MLflow

Para utilizar las funciones compartidas, cada script de entrenamiento debe importarlas desde `src/utils/mlflow_utils.py`:

```python
from src.utils.mlflow_utils import configurar_mlflow, registrar_experimento
```

Antes de comenzar el entrenamiento, se debe configurar la conexión con DagsHub:

```python
configurar_mlflow()
```

Esta función utiliza el repositorio `MauriChocho/customer-churn-ml` y selecciona el experimento `churn-classification`.

La primera vez que se ejecute, DagsHub puede solicitar autorización desde el navegador. Cada integrante debe utilizar su propia cuenta y tener los permisos correspondientes.

No se deben incluir contraseñas ni tokens de acceso en el código.

#### Registrar un entrenamiento

Después de entrenar el modelo y calcular las métricas, se deben preparar los parámetros, las métricas y las etiquetas que se enviarán a MLflow.

Por ejemplo:

```python
parametros = model_pipeline.named_steps["model"].get_params()

metricas = {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "accuracy": accuracy
}

etiquetas = {
    "modelo": "RandomForest"
}
```

Luego se utiliza la función compartida para registrar el entrenamiento:

```python
registrar_experimento(
    nombre_run="random_forest",
    parametros=parametros,
    metricas=metricas,
    etiquetas=etiquetas,
    pipeline=model_pipeline
)
```

Cada integrante debe adaptar `nombre_run` y la etiqueta `modelo` a su algoritmo, conservando los mismos nombres de métricas para facilitar la comparación.

La función registra el pipeline completo, incluyendo el preprocesamiento y el modelo entrenado, utilizando el formato `cloudpickle`. Se utiliza cloudpickle porque, durante las pruebas con Random Forest, el formato skops presentó un error al serializar algunos componentes del modelo. Con cloudpickle se pudo registrar correctamente el pipeline completo en DagsHub.

#### Pruebas automáticas de MLflow

Las funciones compartidas tienen pruebas automáticas en `test/test_mlflow_utils.py`.

Para ejecutarlas, desde la carpeta raíz del proyecto:

```bash
python -m pytest test/test_mlflow_utils.py -v
```

Se realizan dos pruebas:

- `test_configurar_mlflow`: verifica la configuración del repositorio de DagsHub y del experimento compartido.
- `test_registrar_experimento`: verifica el registro de parámetros, métricas, etiquetas y pipeline, incluyendo el formato `cloudpickle`.

Las pruebas utilizan `monkeypatch` para simular las llamadas a MLflow, por lo que no necesitan conectarse realmente a DagsHub.

Resultado verificado:

```text
2 passed
```

### Análisis Exploratorio de Datos (EDA)
El diagnóstico inicial de los datos se encuentra modularizado en el script `src/data/01_eda.py`. 
El análisis completo se puede leer en `src/data/reporte_eda.md`.

Para ejecutar la inspección por consola:

```bash
python -m src.data.01_eda
```

### Validar el preprocessing

Para ejecutar y validar el preprocessing comun:

```bash
python -m src.features.preprocessing
```

Este modulo aplica las transformaciones definidas para las variables numericas, categoricas y binarias.

### Entrenar Baseline

Para entrenar y evaluar el modelo baseline:

python -m src.training.train_baseline

El script:
- carga el dataset
- separa las variables predictoras y la variable objetivo
- realiza la division train/test
- aplica el preprocessing comun
- entrena un DummyClassifier como referencia minima
- calcula las metricas de evaluacion
- muestra la matriz de confusion


 Este modelo se utiliza como referencia minima para comparar el rendimiento de los demas modelos

 
### Entrenar Random Forest

Para entrenar y evaluar el modelo Random Forest:

```bash
python -m src.training.train_random_forest
```

El script:

- carga el dataset
- separa las variables predictoras y la variable objetivo
- realiza la division train/test
- aplica el preprocessing comun
- entrena el modelo
- calcula las metricas de evaluacion
- muestra la matriz de confusion
- registra los parametros, las metricas, las etiquetas y el pipeline entrenado en MLflow/DagsHub


### Exploración de configuraciones de SVM

Para probar distintas configuraciones de SVM utilizando el mismo split y preprocessing:

```bash
python -m src.training.pre_train_svm
```
Este script:

- carga el dataset
- realiza la division train/test
- aplica el preprocessing comun
- prueba distintas configuraciones de SVM
- muestra la matriz de confusion de cada configuracion

### Entrenar modelo SVM
Luego ejecutamos el entrenamiento:

```bash
python -m src.training.train_svm
```

El script:

- carga el dataset
- separa las variables predictoras y la variable objetivo
- realiza la division train/test
- aplica el preprocessing comun
- entrena la configuracion seleccionada de SVM
- calcula Precision, Recall, F1-score, ROC-AUC y Accuracy
- muestra la matriz de confusion

### Decision Tree Classifier

Para entrenar y evaluar el modelo Decision Tree:

```bash
python -m src.training.train_decision_tree
```

El script:

- carga el dataset
- separa variables predictoras y objetivo
- division train/test
- aplica processing comun
- entrena un DecisionTreeClassifier con max_depth=10 y class_weight="balanced"
- calcula Precision, F1-Score, Accuracy, Recall,ROC-AUC sobre test
- calcula F1-Score y Accuracy sobre train para evaluar posible overfitting
- muestra matriz de confusion
- genera grafico de la matriz


### Entrenar y evaluar Logistic Regression

Para entrenar y evaluar el modelo Logistic Regression:

```bash
python -m src.training.train_logistic_regression
```

El script:
- carga el dataset
- separa las variables predictoras y la variable objetivo
- realiza la division train/test
- aplica el preprocessing comun
- entrena Logistic Regression con la configuracion seleccionada
- calcula Precision, Recall, F1-score, ROC-AUC y Accuracy
- muestra la matriz de confusion
- guarda el pipeline completo entrenado en models/churn_pipeline.joblib

Este modelo fue seleccionado como modelo candidato del proyecto.

## Resultados de los modelos

Los resultados y la comparacion entre los distintos modelos se documentan en:

```text
src/evaluation/comparacion_modelos.md
```

La comparacion utiliza como metrica principal el F1-score y considera tambien Precision, Recall, ROC-AUC y Accuracy.

## Modelo candidato seleccionado

Luego de comparar Baseline, Decision Tree, Random Forest, Logistic Regression y SVM, se selecciono **Logistic Regression** como modelo candidato.

La metrica principal utilizada para la seleccion fue el **F1-score**.

La configuracion seleccionada obtuvo:

- F1-score: 0.5779
- Recall: 0.7231
- ROC-AUC: 0.8118

El detalle completo de la comparacion y la justificacion de la seleccion se encuentra en:

`src/evaluation/comparacion_modelos.md`


## Probar el pipeline serializado

El modelo candidato se guarda como un pipeline completo en:

`models/churn_pipeline.joblib`

Para comprobar que el pipeline puede cargarse y utilizarse para realizar predicciones:

```bash
python -m src.inference.predict


El script carga el pipeline serializado y realiza una prediccion de prueba.

## Reproduccion del flujo

Una vez clonado el repositorio y configurado el entorno:

1. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

2. Configurar las credenciales de DagsHub.

3. Recuperar los archivos administrados por DVC:

```bash
python -m dvc pull
```

4. Entrenar el modelo candidato:

```bash
python -m src.training.train_logistic_regression
```

5. Probar el pipeline serializado:

```bash
python -m src.inference.predict
```

De esta manera se puede reproducir el entrenamiento y verificar el funcionamiento del modelo desde la linea de comandos.

## Stack utilizado (Entrega 1)

Python, pandas, NumPy, scikit-learn, joblib, Git/GitHub y DVC + DagsHub.


## Integrantes

- Mauricio Chocholacek
- Paola Pierolivo
- Veronica Gonzalez
- Denise Valdivieso
- Daiana Marilyn Ruiz díaz


## Estado actual

🚧 Proyecto en desarrollo — Entrega 1 en curso.