# Analisis de umbrales de riesgo para Logistic Regression
# Objetivo: justificar los niveles LOW, MEDIUM y HIGH
# utilizando probabilidades obtenidas con validacion cruzada.

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    TARGET,
    ID_COL,
    build_preprocessor,
)

# Cargar el mismo dataset utilizado en el entrenamiento original
df = pd.read_csv("data/raw/customer_churn_historical.csv")

# Separar las variables predictoras (X) y la variable objetivo (y)
X = df.drop(columns=[TARGET, ID_COL])
y = df[TARGET].map({"Yes": 1, "No": 0})

# Reproducir la division acordada por el equipo en la primera entrega
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Comprobar la cantidad de registros en cada conjunto
print("Filas de entrenamiento:", X_train.shape[0])
print("Filas de prueba:", X_test.shape[0])

# Configurar validacion cruzada estratificada de 5 particiones
# Mantiene aproximadamente la proporcion de churn en cada grupo
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Construir el mismo pipeline utilizado en la primera entrega
modelo_logreg = Pipeline([
    ("preprocessing", build_preprocessor()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        C=0.1,
    )),
])

# Obtener probabilidades mediante validacion cruzada
# Cada cliente recibe una prediccion de un modelo que no lo uso para entrenar
probabilidades = cross_val_predict(
    modelo_logreg,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
)[:, 1]

# Mostrar las primeras probabilidades obtenidas
print("Primeras 10 probabilidades de churn:")
print(probabilidades[:10])

print("Cantidad de probabilidades:", len(probabilidades))

# Verificar que todas las probabilidades sean validas
print("Probabilidad minima:", probabilidades.min())
print("Probabilidad maxima:", probabilidades.max())
print("Valores faltantes:", pd.isna(probabilidades).sum())

# Comparar distintas propuestas de umbrales de riesgo
# Los valores son alternativas evaluadas antes de la decision del equipo
propuestas_umbrales = {
    "A": (0.30, 0.50),
    "B": (0.20, 0.50),
    "C": (0.30, 0.60),
}

# Guardar los resultados para construir una tabla comparativa
resumen = []

for nombre, (umbral_medium, umbral_high) in propuestas_umbrales.items():

    # Clasificar a los clientes segun los umbrales de cada propuesta
    niveles = pd.cut(
        probabilidades,
        bins=[-float("inf"), umbral_medium, umbral_high, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH"],
        right=False,
    )

    # Comparar el riesgo asignado con el abandono real
    resultados = pd.DataFrame({
        "nivel_riesgo": niveles,
        "churn_real": y_train.to_numpy(),
    })

    tabla = pd.crosstab(
        resultados["nivel_riesgo"],
        resultados["churn_real"],
    ).reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)

    print(f"\n--- Propuesta {nombre}: {umbral_medium} / {umbral_high} ---")
    print(tabla)

    # Obtener los clientes de riesgo bajo y alto
    clientes_low = resultados[resultados["nivel_riesgo"] == "LOW"]
    clientes_high = resultados[resultados["nivel_riesgo"] == "HIGH"]

    # Guardar los indicadores principales de cada propuesta
    resumen.append({
        "propuesta": nombre,
        "umbral_medium": umbral_medium,
        "umbral_high": umbral_high,
        "clientes_low": len(clientes_low),
        "abandonos_low": clientes_low["churn_real"].sum(),
        "clientes_high": len(clientes_high),
        "abandonos_high": clientes_high["churn_real"].sum(),
        "falsas_alarmas_high": (
            len(clientes_high) - clientes_high["churn_real"].sum()
        ),
    })

# Mostrar el resumen de todas las propuestas
tabla_resumen = pd.DataFrame(resumen)

print("\nResumen comparativo de propuestas:")
print(tabla_resumen.to_string(index=False))