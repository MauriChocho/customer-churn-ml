# Entrenamiento del modelo Logistic Regression
# objetivo: entrenar y evaluar Logistic Regression usando el pipeline comun del equipo

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.preprocessing import TARGET, ID_COL, build_preprocessor

# Cargar datos historicos
df = pd.read_csv("data/raw/customer_churn_historical.csv")

# Separar features (X) y target (y)
X = df.drop(columns=[TARGET, ID_COL])
y = df[TARGET].map({"Yes": 1, "No": 0})

# Split acordado por el equipo
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Filas de entrenamiento:", X_train.shape[0])
print("Filas de test:", X_test.shape[0])
print("Proporcion de Churn en train:", y_train.mean().round(4))
print("Proporcion de Churn en test:", y_test.mean().round(4))
