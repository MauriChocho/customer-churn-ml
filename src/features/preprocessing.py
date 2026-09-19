# Preprocesamiento de datos para el modelo de churn
# objetivo: Definir cómo se transforman las variables

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Definicion de columnas

# Variable objetivo
TARGET = "Churn"

# Identificador del cliente
# No se utiliza como predictor
ID_COL = "customerID"

# Variables numericas
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Variable binaria ya codificada como 0/1
BINARY_FEATURES = [
    "SeniorCitizen",
]

# Variables categoricas
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]



# Pipeline para variables numericas

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])


# Pipeline para variables categoricas

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])


# Preprocesador comun

def build_preprocessor():
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("binary", "passthrough", BINARY_FEATURES),
    ])

    return preprocessor


# Prueba local del preprocesador

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    df = pd.read_csv("data/raw/customer_churn_historical.csv")

    X = df.drop(columns=[TARGET, ID_COL])

    preprocessor = build_preprocessor()

    X_preprocessed = preprocessor.fit_transform(X)

    # Convertir a array solo para verificar valores nulos
    if hasattr(X_preprocessed, "toarray"):
        X_check = X_preprocessed.toarray()
    else:
        X_check = X_preprocessed

    print("Forma original de X:", X.shape)
    print("Forma luego del preprocessing:", X_preprocessed.shape)
    print("Cantidad de nulos luego del preprocessing:", np.isnan(X_check).sum())