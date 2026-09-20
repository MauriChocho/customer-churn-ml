# Customer Churn ML

Proyecto integrador de la materia **Laboratorio de Minería de Datos** (ISTEA) — predicción de churn de clientes con un enfoque de MLOps end-to-end.

## Problema de negocio

Una empresa de telecomunicaciones necesita identificar clientes con riesgo de abandonar el
servicio (`Churn`). El sistema debe estimar la probabilidad de abandono de un cliente y
devolver un nivel de riesgo (`LOW` / `MEDIUM` / `HIGH`) consumible por otros sistemas
(por ejemplo, un CRM o una campaña de retención).


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
│   ├── inference/     # logica de prediccion - entregas posteriores
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
   source venv/Scripts/activate    # Windows
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

4. Descargar los datos:

   dvc pull


## Stack utilizado (Entrega 1)

Python, pandas, scikit-learn, Git/GitHub, DVC + DagsHub.
MLflow, FastAPI y Docker se incorporan en entregas posteriores.

## Estado actual

🚧 Proyecto en desarrollo — Entrega 1 en curso.

## Ejecucion de los modulos

Los comandos deben ejecutarse desde la carpeta raiz del proyecto.

Esto es importante para que Python pueda reconocer correctamente los modulos internos dentro de `src/`.

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


### Exploración de hiperparámetros de SVM
Imprime en consola: dimensiones y tipos de datos, duplicados, validación de `customerID`, valores nulos, distribución de `Churn`, estadísticas descriptivas y relación de las variables con el target.

```bash
python -m src.training.pre_train_svm
```



### 3. Entrenamiento del modelo SVM final
Entrena y compara 4 configuraciones de SVM (kernel, C, class_weight) sobre el mismo split y preprocessing, e imprime Precision, Recall, F1-score, ROC-AUC, Accuracy y matriz de confusión de cada una. Es el script que generó la evidencia documentada en `src/evaluation/comparacion_modelos.md`.

```bash
python -m src.training.train_svm
```

Entrena el pipeline completo (preprocessing + SVM) con la configuración elegida (`kernel="rbf", C=1.0, class_weight="balanced"`, calibrada con `CalibratedClassifierCV`) e imprime sus métricas finales.
### Resultados de los modelos

Los resultados y la comparacion entre los distintos modelos se documentan en:

```text
src/evaluation/comparacion_modelos.md
```

La comparacion utiliza como metrica principal el F1-score y considera tambien Precision, Recall, ROC-AUC y Accuracy.