# Imagen de la API Customer Churn (FastAPI + scikit-learn)
FROM python:3.13-slim

# Logs sin buffer (se ven al instante en "docker compose logs")
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1) Dependencias primero: Docker reutiliza esta capa si no cambian
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 2) Codigo de la aplicacion
COPY app/ ./app/
COPY src/ ./src/

# 3) Modelo exportado (churn_pipeline.joblib) y su metadata (model_meta.json)
#    El .joblib debe existir en models/ antes de hacer "docker build" (ver README).
COPY models/ ./models/

EXPOSE 8000

# La API escucha en 0.0.0.0 para ser accesible desde fuera del contenedor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
