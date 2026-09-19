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