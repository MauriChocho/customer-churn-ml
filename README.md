# Customer Churn ML

Proyecto integrador de la materia **Laboratorio de Minería de Datos** (ISTEA) — predicción de churn de clientes con un enfoque de MLOps end-to-end.

## Problema de negocio

Una empresa de telecomunicaciones necesita identificar clientes con riesgo de abandonar el
servicio (`Churn`). El sistema debe estimar la probabilidad de abandono de un cliente y
devolver un nivel de riesgo (`LOW` / `MEDIUM` / `HIGH`) consumible por otros sistemas
(por ejemplo, un CRM o una campaña de retención).
   El objetivo es desarrollar y comparar modelos capaces de predecir si un cliente presenta riesgo de churn.
A partir de esta prediccion, la empresa podria identificar clientes con mayor riesgo de abandono y orientar futuras acciones de retencion.

## Cómo funciona el sistema

```
Cliente (19 variables en JSON)
   │
   ▼
POST /predict  ──►  Pydantic valida la entrada  ──►  si es inválida: error 422
   │
   ▼
Pipeline del modelo (versión 1 del Model Registry)
   ├─ Preprocessing: imputa nulos, codifica categorías y escala
   └─ Logistic Regression: calcula la probabilidad de abandono
   │
   ▼
Umbrales de riesgo: LOW < 0.20 · MEDIUM: 0.20 ≤ probabilidad < 0.50 · HIGH ≥ 0.50
   │
   ▼
Respuesta: prediction, probability, risk_level, model_version, request_id
```

- El modelo se carga **una sola vez** al iniciar la API, desde `models/churn_pipeline.joblib`.
- `prediction` usa un corte de 0.50 (`CHURN` / `NO_CHURN`). `risk_level` gradúa el riesgo
  para orientar acciones de retención: un cliente puede ser `NO_CHURN` y aun así `MEDIUM`.
- `GET /health` informa si el modelo está cargado y qué versión se está usando.


## Dataset

El proyecto utiliza el dataset historico:

`data/raw/customer_churn_historical.csv`

La variable objetivo es `Churn`, que indica si un cliente abandono o no el servicio.

El dataset no se versiona directamente con Git. Se encuentra gestionado con DVC y almacenado en el remote configurado en DagsHub.


## Estructura del repositorio

```
customer-churn-ml/
├── app/
│   ├── main.py        # API FastAPI: carga del modelo, /health y /predict
│   └── schemas.py     # contratos Pydantic de entrada y salida
├── data/
│   └── raw/           # dataset historico, versionado con DVC
├── models/
│   ├── churn_pipeline.joblib.dvc  # modelo servido (version 1 del Model Registry)
│   └── model_meta.json            # nombre, version y run de MLflow del modelo
├── monitoring/        # reservado para monitoreo en la entrega final
├── notebooks/         # reservado para notebooks de análisis y experimentación
├── scripts/
│   └── exportar_modelo.py  # exporta el modelo registrado a models/
├── src/
│   ├── data/          # EDA y reporte
│   ├── evaluation/    # comparacion de modelos y analisis de umbrales
│   ├── features/      # preprocessing
│   ├── inference/     # carga del pipeline y predicciones
│   ├── training/      # entrenamiento de modelos con MLflow
│   └── utils/         # funciones compartidas de MLflow
├── test/              # pruebas automaticas (pytest)
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
   .\venv\Scripts\Activate.ps1    # Windows
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

#### Prueba de integración de los modelos con MLflow

`test/test_mlflow_utils.py` prueba las funciones compartidas con datos de ejemplo. `test/test_integracion_modelos.py` prueba que cada script de entrenamiento les pase bien la información.

La prueba ejecuta cada script de entrenamiento completo, pero reemplaza `configurar_mlflow()` y `registrar_experimento()` por versiones simuladas que guardan lo que reciben. Por eso no se conecta a DagsHub ni crea runs nuevos.

También simula `joblib.dump`. El script de regresión logística guarda el pipeline en `models/churn_pipeline.joblib`, que es el mismo archivo del modelo exportado del Model Registry que usa la API. Sin esa simulación, correr la prueba reemplazaba el modelo servido por uno reentrenado localmente, con otra huella MD5. La prueba compara la huella del modelo antes y después de cada script para asegurarse de que no cambie.

Para cada modelo verifica que:

- el modelo que usa la API no se modifique
- se llame a `configurar_mlflow()`
- el `nombre_run` sea el esperado
- se envíen exactamente las cinco métricas acordadas (`precision`, `recall`, `f1`, `roc_auc`, `accuracy`), con valores entre 0 y 1
- se envíen parámetros y la etiqueta `modelo`
- se registre el pipeline completo (preprocessing + modelo), con `predict_proba`

Es una sola prueba parametrizada. Los modelos que cubre están en la lista `MODELOS`:

| Script | Run esperado |
|---|---|
| `src.training.train_svm` | `svm` |
| `src.training.train_random_forest` | `random_forest` |
| `src.training.train_baseline` | `baseline` |
| `src.training.train_logistic_regression` | `logistic_regression` |
| `src.training.train_decision_tree` | `decision_tree` |

Si se agrega un modelo nuevo, alcanza con sumar una línea a `MODELOS`.

Requisito: haber ejecutado `dvc pull`, porque los scripts entrenan con el dataset real. Durante la prueba se fija `MPLBACKEND=Agg` para que los gráficos de los scripts no abran ventanas.

```bash
python -m pytest test/test_integracion_modelos.py -v
```

Resultado verificado:

```text
5 passed
```

Pueden aparecer tres avisos esperables: dos `FigureCanvasAgg is non-interactive` (por los `plt.show()` de Random Forest y Decision Tree) y un `UndefinedMetricWarning` del baseline, que nunca predice churn y por eso su Precision queda indefinida.

#### Pruebas de datos y preprocesamiento

Verifican que el dataset tenga la estructura esperada y que el preprocesamiento
maneje correctamente los valores faltantes.

Requisito: haber ejecutado `dvc pull`, ya que `test_dataset.py` y `test_preprocessing.py` utilizan el dataset real.

Para ejecutarlas, desde la carpeta raíz del proyecto:

```bash
python -m pytest test/test_dataset.py test/test_valores_faltantes.py test/test_preprocessing.py -v
```

`test_dataset.py` utiliza el dataset local `data/raw/customer_churn_historical.csv`.

> Si el archivo no está, la prueba lo indica; se recupera con `dvc checkout` o `dvc pull`.

**`test/test_dataset.py`** — estructura del dataset histórico:

- `test_columnas_obligatorias`: existen todas las columnas que usa el preprocessing, más `customerID` y `Churn`.
- `test_tipos_numericos`: `tenure`, `MonthlyCharges`, `TotalCharges` y `SeniorCitizen` son numéricas.
- `test_valores_target`: `Churn` solo contiene `Yes` y `No`.
- `test_senior_citizen_binaria`: `SeniorCitizen` está codificada como 0/1.
- `test_customer_id_unico`: no hay clientes duplicados.

**`test/test_valores_faltantes.py`** — manejo de nulos en el preprocessing:

Utiliza un pequeño conjunto de clientes construido dentro de la prueba, por lo que no depende del CSV.

- `test_total_charges_faltante`: completa un `TotalCharges` vacío (caso detectado en el EDA: 26 nulos).
- `test_varios_numericos_faltantes`: completa nulos en varias columnas numéricas a la vez.
- `test_faltante_en_datos_nuevos`: un preprocesador ya entrenado completa nulos en un cliente nuevo, como ocurrirá en inferencia.

En todos los casos se verifica que no queden nulos y que no se pierdan registros.

**`test/test_preprocessing.py`** — transformaciones del preprocessing:

Utiliza una muestra de 200 clientes del dataset real, por lo que requiere `dvc pull`.

- `test_preprocesamiento_conserva_registros_y_salida_valida`: el preprocesador transforma la muestra sin perder registros y la salida es una matriz numérica sin valores nulos ni infinitos.
- `test_preprocesamiento_mantiene_columnas_en_datos_nuevos`: un preprocesador entrenado con una parte de la muestra genera la misma cantidad de columnas al transformar clientes que no vio. Es lo que pasa en inferencia: si un cliente nuevo generara otra cantidad de columnas, el modelo no podría predecir.

Resultado verificado:

```text
10 passed
```

Para ejecutar todas las pruebas del proyecto:

```bash
python -m pytest test/ -v
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
- registra los parametros, las metricas, las etiquetas y el pipeline entrenado en MLflow/DagsHub


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
- entrena la configuracion seleccionada de SVM (`kernel="rbf"`, `C=1.0`, `class_weight="balanced"`)
- calcula Precision, Recall, F1-score, ROC-AUC y Accuracy
- muestra la matriz de confusion
- registra los parametros, las metricas, las etiquetas y el pipeline entrenado en MLflow/DagsHub (run `svm`)

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
- registra los parametros, las metricas, las etiquetas y el pipeline entrenado en MLflow/DagsHub


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
- registra los parametros, las metricas, las etiquetas y el pipeline entrenado en MLflow/DagsHub


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

### Modelo registrado en el Model Registry

El modelo candidato esta registrado en el Model Registry de MLflow, en el DagsHub del equipo:

- Nombre: `churn_logistic_regression`
- Version: 1
- Origen: run `logistic_regression` del experimento `churn-classification` (Run ID `a1033ab77d204947a0cd51e8eaafc3ae`, commit `ee6fd3b`)

Para cargar esta version desde Python:

```python
import dagshub
import mlflow

dagshub.init(repo_owner="MauriChocho", repo_name="customer-churn-ml", mlflow=True)

modelo = mlflow.sklearn.load_model("models:/churn_logistic_regression/1")
```

El resultado es el pipeline completo, con los pasos `preprocessing` y `model`.

Para conectarse a DagsHub hay que contar con acceso al repositorio. La primera vez puede solicitar autorizacion desde el navegador. No se deben incluir contraseñas ni tokens en el codigo.

### Exportación del modelo registrado

La API y los contenedores no se conectan a DagsHub. Utilizan una copia local de la
versión registrada, exportada una sola vez desde el Model Registry y versionada con DVC.
De esta manera, la aplicación usa una versión identificable del modelo sin necesitar
tokens ni conexión a internet.

Para exportar el modelo (requiere acceso a DagsHub), desde la carpeta raíz del proyecto:

```bash
python -m scripts.exportar_modelo
```

El script:

- descarga `churn_logistic_regression` versión 1 desde el Model Registry
- la guarda en `models/churn_pipeline.joblib` (el archivo que utiliza `cargar_modelo()`)
- genera `models/model_meta.json` con el origen del modelo:

```json
{
  "model_name": "churn_logistic_regression",
  "model_version": "1",
  "run_id": "a1033ab77d204947a0cd51e8eaafc3ae"
}
```

Luego se versiona el modelo con DVC y se sube al remote de DagsHub:

```bash
python -m dvc add models/churn_pipeline.joblib
python -m dvc push models/churn_pipeline.joblib.dvc
```

Para comprobar que el modelo quedó subido:

```bash
python -m dvc status -c models/churn_pipeline.joblib.dvc
```

**Importante:** después de exportar, ejecutar siempre `dvc push` y subir a Git en el
**mismo commit** `models/churn_pipeline.joblib.dvc` y `models/model_meta.json`.
Así la versión indicada en el JSON corresponde siempre al modelo que descarga DVC.

Para obtener el modelo exportado en otra computadora:

```bash
git pull
python -m dvc pull models/churn_pipeline.joblib.dvc
```

### Trazabilidad del modelo servido

| Elemento | Valor |
|---|---|
| Modelo registrado | `churn_logistic_regression`, versión 1 |
| Run de MLflow | `a1033ab77d204947a0cd51e8eaafc3ae` (experimento `churn-classification`) |
| Commit de entrenamiento | `ee6fd3bb68` |
| MD5 del dataset | `026e0519053e9c810c8da233c7ad5449` |
| MD5 del modelo servido | `2657e6f63a25ceda66e407c2755abf79` |

Recorrido: la API informa `model_version` → `models/model_meta.json` indica el Run →
en DagsHub, el Run muestra parámetros, métricas y el commit → ese commit apunta, mediante
DVC, a la versión del dataset utilizada.

Límite: la relación con los datos se reconstruye a partir del commit y su referencia DVC;
no demuestra retrospectivamente que el entrenamiento se ejecutara sin cambios locales.

## Análisis de umbrales de riesgo

Para definir los niveles de riesgo LOW, MEDIUM y HIGH, se compararon tres propuestas utilizando las probabilidades de Logistic Regression obtenidas mediante validación cruzada sobre los datos de entrenamiento.

| Propuesta | MEDIUM desde | HIGH desde | Abandonos en LOW | Abandonos en HIGH |
|---|---:|---:|---:|---:|
| A | 0.30 | 0.50 | 126 | 1135 |
| B | 0.20 | 0.50 | 68 | 1135 |
| C | 0.30 | 0.60 | 126 | 979 |

Se seleccionó la **propuesta B (0.20 / 0.50)**, priorizando reducir los abandonos reales clasificados como LOW, sin disminuir los abandonos identificados en HIGH.

Los niveles definidos son:
- **LOW:** score menor a 0.20.
- **MEDIUM:** score desde 0.20 hasta menos de 0.50.
- **HIGH:** score igual o mayor a 0.50.

La metodología, los resultados completos y las limitaciones del análisis están documentados en [justificacion_umbrales.md](src/evaluation/justificacion_umbrales.md).

Para reproducir el análisis:

```bash
python -m src.evaluation.analisis_umbrales
```

## Inferencia: cargar el modelo y predecir

El módulo `src/inference/predict.py` carga el pipeline entrenado (preprocessing + modelo) y devuelve la predicción de un cliente. No repite ninguna transformación: el preprocessing ya está dentro del pipeline.

Antes de usarlo hay que descargar el modelo (para configurar el acceso a DagsHub, ver "Cómo levantar el proyecto"):

```bash
dvc pull models/churn_pipeline.joblib.dvc
```

### Uso

```python
from src.inference.predict import cargar_modelo, predecir

pipeline = cargar_modelo()

cliente = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
}

resultado = predecir(pipeline, cliente, model_version="1")
print(resultado)
```

El cliente es un diccionario con las 19 columnas que espera el pipeline (sin `customerID`).

### Respuesta

| Campo | Tipo | Descripción |
|---|---|---|
| `prediction` | texto | `CHURN` o `NO_CHURN`|
| `probability` | número entre 0 y 1 | probabilidad estimada de abandono |
| `risk_level` | texto | `LOW`, `MEDIUM` o `HIGH` |
| `model_version` | texto | versión del modelo que se le pasa a la función |

El nivel de riesgo se calcula con `UMBRAL_MEDIUM` y `UMBRAL_HIGH`, definidos al inicio de `predict.py` y acordados por el equipo (propuesta B):

| Nivel | Probabilidad de abandono |
|---|---|
| `LOW` | menor a 0,20 |
| `MEDIUM` | desde 0,20 y menor a 0,50 |
| `HIGH` | desde 0,50 |

El análisis y la justificación de estos valores están en la sección "Análisis de umbrales de riesgo" de este README y en [justificacion_umbrales.md](src/evaluation/justificacion_umbrales.md). Si se cambian los umbrales, hay que actualizar también la prueba `test_umbrales_acordados_por_el_equipo`.

### Errores

- Si falta una columna, `predecir` lanza un `ValueError` que indica cuáles faltan.
- Si llegan columnas de más (por ejemplo `customerID`), se ignoran.
- Si no está el modelo, `cargar_modelo` lanza un `FileNotFoundError` que indica el comando `dvc pull`.

### Pruebas

```bash
python -m pytest test/test_inferencia.py -v
python -m src.inference.predict
```

El segundo comando es una prueba manual con el primer cliente del dataset (requiere tener descargado `data/raw/customer_churn_historical.csv` con DVC).

## Contrato de entrada de la API

La API recibe los datos de un cliente para predecir si puede abandonar el servicio (churn). Antes de hacer la predicción, esos datos se validan usando Pydantic.

El modelo de entrada está definido en `app/schemas.py`, dentro de la clase `PredictionRequest`. Los campos y sus nombres coinciden con los que utiliza el pipeline de preprocessing (`src/features/preprocessing.py`).

### Campos

| Campo | Tipo | Valores permitidos |
|---|---|---|
| `gender` | texto | `Female`, `Male` |
| `SeniorCitizen` | entero | `0`, `1` |
| `Partner` | texto | `No`, `Yes` |
| `Dependents` | texto | `No`, `Yes` |
| `tenure` | entero | mayor o igual a 0 |
| `PhoneService` | texto | `No`, `Yes` |
| `MultipleLines` | texto | `No`, `No phone service`, `Yes` |
| `InternetService` | texto | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | texto | `No`, `No internet service`, `Yes` |
| `OnlineBackup` | texto | `No`, `No internet service`, `Yes` |
| `DeviceProtection` | texto | `No`, `No internet service`, `Yes` |
| `TechSupport` | texto | `No`, `No internet service`, `Yes` |
| `StreamingTV` | texto | `No`, `No internet service`, `Yes` |
| `StreamingMovies` | texto | `No`, `No internet service`, `Yes` |
| `Contract` | texto | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | texto | `No`, `Yes` |
| `PaymentMethod` | texto | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |
| `MonthlyCharges` | decimal | mayor que 0 |
| `TotalCharges` | decimal, opcional | mayor o igual a 0 |

### Validaciones

- `TotalCharges` puede no enviarse o recibirse como `null`. En esos casos, el pipeline completa el valor faltante utilizando la mediana.
- Si se envían campos que no están definidos en el contrato, como `customerID`, se ignoran.
- `tenure` no puede ser negativo y no tiene un máximo: un cliente con más de 72 meses de antigüedad también es válido.
- Pydantic permite convertir textos que representan números en los campos `tenure`, `MonthlyCharges` y `TotalCharges`. Por ejemplo, `"32"` se convierte en `32`. Si no puede realizar la conversión, devuelve un error.
- `SeniorCitizen` debe recibirse como un número entero (`0` o `1`), no como texto.
- Cuando hay más de un dato incorrecto, se informan todos los errores encontrados en la misma respuesta.

### Ejemplo de uso

```python
from app.schemas import PredictionRequest

cliente = PredictionRequest(**datos)

# Se obtienen los datos validados para enviarlos a la predicción
datos_validados = cliente.model_dump()
```

### Integración con la API

Este contrato se utiliza en el endpoint `POST /predict`. Cuando los datos recibidos no cumplen con las validaciones, FastAPI devuelve un error HTTP 422 sin llegar al modelo. En la respuesta, `detail` incluye un error por cada dato incorrecto, y `loc` indica el campo que falló (por ejemplo, `["body", "Contract"]`). Las pruebas por HTTP están en `test/test_predict_validaciones.py` y se ejecutan con `python -m pytest test/test_predict_validaciones.py -v`.

### Pruebas

Para ejecutar las pruebas del contrato de entrada:

```bash
python -m pytest test/test_contrato_entrada.py -v
```

Las pruebas se pueden ejecutar sin tener descargados el dataset ni el modelo. Se verifica que se pueda ingresar un cliente válido y que se rechacen los datos incompletos, las categorías que no pertenecen al dataset, los valores fuera de rango y los tipos incorrectos. También se comprueba que los campos definidos en el contrato coincidan con las columnas que utiliza el pipeline.

## Contrato de salida de la API

La respuesta de `POST /predict` está definida en `app/schemas.py`, en la clase `PredictionResponse`. El endpoint la usa como `response_model`, así que FastAPI valida lo que devuelve la API y lo muestra en Swagger.

### Campos

| Campo | Tipo | Valores | Origen |
|---|---|---|---|
| `prediction` | texto | `CHURN`, `NO_CHURN` | etiquetas definidas en `src/inference/predict.py` |
| `probability` | decimal | entre 0 y 1 | probabilidad de abandono (`predict_proba`) |
| `risk_level` | texto | `LOW`, `MEDIUM`, `HIGH` | `calcular_risk_level()`, con los umbrales acordados |
| `model_version` | texto | por ejemplo `"1"` | versión de `models/model_meta.json` |
| `request_id` | texto | UUID | lo genera el endpoint en cada pedido |

Los cuatro primeros campos son los que pide la consigna. `request_id` se agregó para identificar cada predicción; no lo calcula `predecir()`, lo agrega el endpoint.

Ejemplo de respuesta:

```json
{
  "prediction": "CHURN",
  "probability": 0.81,
  "risk_level": "HIGH",
  "model_version": "1",
  "request_id": "3f1c9a2e-8b7d-4e5f-9a1b-2c3d4e5f6a7b"
}
```

Este mismo ejemplo aparece en Swagger (`/docs`).

### Pruebas

```bash
python -m pytest test/test_contrato_salida.py -v
```

No necesitan el modelo ni el dataset: para probar `predecir()` se usa un pipeline falso que devuelve la probabilidad indicada.

- `test_respuesta_valida_se_acepta`: una respuesta correcta pasa la validación.
- `test_la_respuesta_tiene_los_campos_acordados`: el contrato tiene exactamente los cinco campos.
- `test_falta_un_campo`: si falta cualquiera de los campos, se rechaza.
- `test_valor_invalido_se_rechaza`: se rechazan valores fuera del contrato (probabilidad fuera de 0-1, nivel de riesgo inexistente, etiqueta distinta, versión numérica, `request_id` vacío).
- `test_etiquetas_coinciden_con_la_inferencia` y `test_niveles_de_riesgo_coinciden_con_la_inferencia`: los valores permitidos en el contrato son los mismos que usa `predict.py`. Si alguien cambia una etiqueta o un nivel en un solo lado, la prueba avisa.
- `test_la_salida_de_predecir_cumple_el_contrato`: lo que devuelve `predecir()`, más el `request_id` que agrega el endpoint, cumple el contrato para los tres niveles de riesgo.

Resultado verificado:

```text
20 passed
```

## API (FastAPI)

La API expone el modelo como un servicio HTTP. El modelo se carga una sola vez al
iniciar la aplicación, y su versión se toma de `models/model_meta.json`.

### Levantar la API

Desde la carpeta raíz del proyecto, con el entorno virtual activado:

```bash
python -m uvicorn app.main:app --reload
```

- `http://127.0.0.1:8000/health`: estado del servicio y versión del modelo cargado.
- `http://127.0.0.1:8000/docs`: documentación Swagger, para probar los endpoints.

### GET /health

Respuesta cuando el modelo está cargado (código 200):

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1"
}
```

Si el modelo no se pudo cargar, la API arranca igual y responde 503 con
`"model_loaded": false`. Esto permite que Docker detecte que el servicio no está listo.

### Pruebas

```bash
python -m pytest test/test_health.py -v
```

- `test_health_ok`: la API carga el modelo y `/health` responde 200 con la versión.
- `test_health_sin_modelo`: si el modelo no carga, `/health` responde 503.

Resultado verificado:

```text
2 passed
```
### POST /predict

Este endpoint recibe los datos de un cliente y devuelve una predicción de abandono utilizando el modelo entrenado.

Está implementado en `app/main.py` y utiliza los contratos `PredictionRequest` y `PredictionResponse` definidos en `app/schemas.py`.

Pydantic valida los datos antes de realizar la predicción. El contrato tiene 19 campos de entrada, de los cuales 18 son obligatorios y `TotalCharges` es opcional. Los tipos y valores permitidos se detallan en [Contrato de entrada de la API](#contrato-de-entrada-de-la-api).

El endpoint reutiliza el pipeline cargado al iniciar FastAPI y llama a la función `predecir()` de `src/inference/predict.py`. No vuelve a cargar ni entrenar el modelo para cada solicitud.

#### Ejemplo de solicitud

Con la API iniciada, abrí [Swagger](http://127.0.0.1:8000/docs). Desplegá `POST /predict` y hacé clic en **Try it out**. Pegá el siguiente JSON y presioná **Execute**. Si la solicitud es válida y el modelo está cargado, la API responde con HTTP `200`.

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 846.0
}
```

#### Respuesta de la API

Cuando la solicitud es válida y la predicción se realiza correctamente, devuelve HTTP 200 con cinco campos:

| Campo | Descripción |
|---|---|
| `prediction` | Predicción del modelo: `CHURN` o `NO_CHURN`. |
| `probability` | Probabilidad estimada de abandono, entre 0 y 1. |
| `risk_level` | Nivel de riesgo: `LOW`, `MEDIUM` o `HIGH`. |
| `model_version` | Versión del modelo utilizado para predecir. |
| `request_id` | Identificador único generado para cada solicitud mediante `uuid4()`. |

El campo `request_id` se agrega en el endpoint y no forma parte de la respuesta original de la función `predecir()`.

#### Códigos HTTP

| Código | Situación |
|---|---|
| `200` | La predicción se realizó correctamente. |
| `422` | Los datos enviados no cumplen las validaciones de `PredictionRequest`. |
| `500` | Ocurrió un error inesperado durante la inferencia. |
| `503` | El modelo no está disponible para realizar predicciones. |

#### Pruebas HTTP

Las pruebas del endpoint están implementadas en `test/test_predict.py`.

Para ejecutarlas desde la carpeta raíz del proyecto:

```bash
python -m pytest test/test_predict.py -v
```

Se verifican cuatro situaciones:

- Cliente válido: el endpoint responde HTTP 200 y devuelve los cinco campos esperados.
- Datos inválidos: un valor negativo de `tenure` produce HTTP 422.
- Error inesperado de inferencia: el endpoint responde HTTP 500.
- Modelo no disponible: el endpoint responde HTTP 503.

Las pruebas utilizan `monkeypatch` para simular las condiciones necesarias sin modificar la implementación real de inferencia.

### Pruebas del contrato HTTP y Swagger

```bash
python -m pytest test/test_contrato_http.py -v
```

Complementan a `test/test_predict.py`: no repiten el caso válido ni los códigos 503, 500 y 422, sino que controlan que los endpoints sean consistentes entre sí y que la documentación generada por FastAPI refleje los contratos.

No necesitan el modelo real: al iniciar la API se reemplaza la carga del modelo por un pipeline falso.

Consistencia entre endpoints:

- `test_model_version_igual_en_health_predict_y_model_meta`: la versión que informa `/health` es la misma que devuelve `/predict` y la que figura en `models/model_meta.json`.
- `test_request_id_es_un_uuid_distinto_en_cada_pedido`: cada predicción tiene un `request_id` con formato UUID que no se repite.
- `test_la_api_no_devuelve_campos_fuera_del_contrato`: si la inferencia devolviera un dato de más, la API no lo expone.
- `test_la_api_no_entrega_una_respuesta_fuera_del_contrato`: si la inferencia devolviera una probabilidad inválida, la API responde con error en lugar de entregarla.

Documentación Swagger / OpenAPI:

- `test_swagger_esta_disponible`: `/docs` responde.
- `test_openapi_documenta_los_dos_endpoints`: aparecen `GET /health` y `POST /predict`.
- `test_openapi_predict_usa_los_contratos_de_entrada_y_salida`: `/predict` documenta `PredictionRequest` como entrada, `PredictionResponse` como salida y el error 422.
- `test_openapi_contrato_de_salida_completo_y_con_ejemplo`: los cinco campos de salida figuran como obligatorios y se muestra el ejemplo.
- `test_openapi_contrato_de_entrada_marca_los_obligatorios`: todos los campos de entrada son obligatorios salvo `TotalCharges`.

Resultado verificado:

```text
9 passed
```

## Docker y Docker Compose

La API se empaqueta en una imagen Docker que incluye Python, las dependencias, el código
(`app/` y `src/`) y el modelo exportado (`models/`). El contenedor no se conecta a DagsHub
ni necesita credenciales: usa la versión 1 del modelo, copiada al construir la imagen.

### Archivos

| Archivo | Función |
|---|---|
| `Dockerfile` | Construye la imagen: `python:3.13-slim`, dependencias de `requirements-docker.txt`, código y modelo |
| `requirements-docker.txt` | Dependencias mínimas que necesita la API |
| `.dockerignore` | Excluye de la imagen datos, entornos virtuales, tests y archivos con credenciales |
| `compose.yaml` | Levanta el servicio `api` en el puerto 8000, con healthcheck sobre `/health` |

### Requisitos

- Docker Desktop instalado y abierto, con el estado **Engine running**
  (en Windows requiere WSL 2 y virtualización habilitada).
- El puerto 8000 libre (detener `uvicorn` si está en ejecución).

### 1. Obtener el modelo antes de construir la imagen

El modelo no se versiona en Git. Debe descargarse con DVC **antes** del build, porque la
imagen lo copia desde `models/`:

```bash
python -m dvc pull models/churn_pipeline.joblib.dvc
```

Para verificar que es la versión registrada (Windows PowerShell):

```powershell
Get-FileHash models\churn_pipeline.joblib -Algorithm MD5
```

Debe dar `2657E6F63A25CEDA66E407C2755ABF79`.

> Si se construye la imagen sin el modelo, la API arranca igual, pero `/health`
> responde 503 y el contenedor queda en estado `unhealthy`.

### 2. Construir y levantar

Desde la carpeta raíz del proyecto:

```bash
docker compose up --build
```

La primera vez tarda algunos minutos. Al terminar, debe aparecer:

```text
Modelo cargado correctamente. Version: 1
Uvicorn running on http://0.0.0.0:8000
```

### 3. Verificar

- `http://localhost:8000/health` → `{"status":"ok","model_loaded":true,"model_version":"1"}`
- `http://localhost:8000/docs` → `POST /predict` → *Try it out* → *Execute*: respuesta 200
  con `prediction`, `probability`, `risk_level`, `model_version` y `request_id`.

En otra terminal, también desde la raíz del proyecto:

```bash
docker compose ps      # el servicio debe figurar como (healthy) tras unos 30 segundos
docker compose logs    # carga del modelo y un registro por cada solicitud
```

### 4. Detener

```bash
docker compose down
```

### Verificación realizada

Recorrido probado desde un clon nuevo, en Windows con Docker Desktop y WSL 2:
build correcto, `/health` 200 con versión 1, `/predict` 200, entrada inválida 422
y contenedor `(healthy)`.



## Pruebas automáticas (pytest)

Para ejecutar todas las pruebas, desde la carpeta raíz del proyecto:

```bash
python -m pytest test/ -v
```

Algunas pruebas usan el dataset o el modelo real, así que antes hay que ejecutar `python -m dvc pull` (ver "Cómo levantar el proyecto"). Sin esos archivos, esas pruebas fallan indicando qué falta.

### Archivos de prueba

| Archivo | Qué prueba | Necesita `dvc pull` |
|---|---|---|
| `test_dataset.py` | estructura y tipos del dataset histórico | sí (dataset) |
| `test_valores_faltantes.py` | imputación de nulos en el preprocessing | no |
| `test_preprocessing.py` | transformación sin pérdida de registros y columnas consistentes | sí (dataset) |
| `test_mlflow_utils.py` | funciones compartidas de MLflow | no |
| `test_integracion_modelos.py` | que cada script de entrenamiento registre bien en MLflow | sí (dataset) |
| `test_carga_modelo.py` | carga del pipeline serializado | solo la prueba del modelo real |
| `test_inferencia.py` | predicción, estructura de la salida, umbrales de riesgo | sí, salvo las de umbrales |
| `test_contrato_entrada.py` | validaciones de `PredictionRequest` | no |
| `test_contrato_salida.py` | validaciones de `PredictionResponse` y coherencia con la inferencia | no |
| `test_health.py` | `GET /health` con y sin modelo | solo `test_health_ok` |
| `test_predict.py` | `POST /predict`: caso válido, 503, 500 y 422 | sí, salvo 503 y 422 |
| `test_predict_validaciones.py` | errores 422 por HTTP con el campo que falló | no |
| `test_contrato_http.py` | consistencia entre endpoints y documentación Swagger/OpenAPI | no |

### Cobertura de las pruebas obligatorias de la consigna

| Prueba obligatoria | Dónde se cubre |
|---|---|
| Carga del modelo o pipeline | `test_carga_modelo.py` |
| Ejecución de una predicción válida | `test_inferencia.py`, `test_predict.py` |
| Estructura y tipos de la respuesta | `test_inferencia.py`, `test_contrato_salida.py`, `test_predict.py`, `test_contrato_http.py` |
| Endpoint `/health` | `test_health.py` |
| Endpoint `/predict` | `test_predict.py`, `test_contrato_http.py` |
| Entradas inválidas y errores | `test_contrato_entrada.py`, `test_predict_validaciones.py`, `test_predict.py` (422, 503, 500), `test_inferencia.py` (columnas faltantes) |

### Límites de la suite

- Las pruebas de MLflow simulan las llamadas, así que no verifican la conexión real con DagsHub. Esa conexión se comprueba ejecutando los scripts de entrenamiento y revisando los runs en el experimento `churn-classification`.
- `test_integracion_modelos.py` entrena los cinco modelos, por lo que es la parte más lenta de la suite.
- Docker y Docker Compose no se prueban con pytest: se verifican con `docker build`, `docker compose up` y una solicitud real a `/health` y `/predict`.

### Resultado de la suite completa

En el commit `0b894ff`, `python -m pytest test/ -v` terminó con 91 pruebas aprobadas, 4 avisos y ningún fallo, en 35,85 segundos.

La huella MD5 de `models/churn_pipeline.joblib` coincidió antes y después de ejecutar la suite completa.

## Reproduccion del flujo

Una vez clonado el repositorio y configurado el entorno, desde la carpeta raíz del proyecto:

1. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

2. Configurar las credenciales de DagsHub (ver "Configuración de acceso a los datos").

3. Recuperar los archivos administrados por DVC (dataset histórico y modelo exportado del Model Registry):

```bash
python -m dvc pull
```

4. Ejecutar las pruebas:

```bash
python -m pytest test/ -v
```

5. Levantar la API con el modelo registrado (versión 1):

```bash
python -m uvicorn app.main:app
```

Luego verificar en el navegador:

- `http://127.0.0.1:8000/health`: debe responder `"status": "ok"` y `"model_version": "1"`.
- `http://127.0.0.1:8000/docs`: permite enviar una solicitud de prueba a `POST /predict`.

### Reentrenamiento (opcional)

El modelo que utiliza la API es la versión 1 exportada del Model Registry
(ver "Exportación del modelo registrado"). No es necesario reentrenar para usar el proyecto.

Para reproducir el entrenamiento del modelo candidato:

```bash
python -m src.training.train_logistic_regression
```

Este script **sobrescribe** `models/churn_pipeline.joblib`. Desde ese momento, el archivo
ya no corresponde a la versión indicada en `models/model_meta.json`.

Para volver al modelo registrado:

```bash
python -m dvc checkout models/churn_pipeline.joblib.dvc
```

Para que la API utilice un modelo nuevo, este debe registrarse en MLflow y luego repetirse la exportación del modelo registrado.

## Stack utilizado

- **Datos y modelado:** Python, pandas, NumPy, scikit-learn, joblib.
- **Análisis y visualización:** Jupyter, matplotlib, seaborn.
- **Versionado:** Git/GitHub, DVC + DagsHub (dataset y modelo).
- **Experimentos y registro de modelos:** MLflow (tracking y Model Registry en DagsHub).
- **API:** FastAPI, Uvicorn, Pydantic.
- **Testing:** pytest, httpx (cliente de prueba de FastAPI).
- **Contenedores:** Docker y Docker Compose.

## Integrantes

- Mauricio Chocholacek
- Paola Pierolivo
- Veronica Gonzalez
- Denise Valdivieso
- Daiana Marilyn Ruiz díaz


## Estado actual

Segunda entrega implementada: API de inferencia con FastAPI y Pydantic, pruebas automatizadas con pytest y contenerizacion con Docker y Docker Compose.

La ejecucion de Docker Compose fue verificada desde un clon nuevo, con carga del modelo version 1 y respuestas HTTP 200 para solicitudes validas y HTTP 422 para entradas invalidas.

Las etapas de CI/CD, despliegue y monitoreo corresponden a la entrega final.