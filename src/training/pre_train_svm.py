# Script de exploracion de hiperparametros para SVM
# Objetivo: probar distintas configuraciones sobre el mismo split/preprocessing
# y dejar registrada la evidencia de cada corrida (ver evaluacion.md)
# Ejecutar desde la raiz del proyecto con: python -m src.training.pre_train_svm

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.preprocessing import ID_COL, TARGET, build_preprocessor


DATA_PATH = Path("data/raw/customer_churn_historical.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET, ID_COL])
y = df[TARGET].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Cada tupla es (nombre, parametros de SVC). Se cambia una sola cosa por vez
# respecto a la configuracion por defecto, para poder ver que efecto tiene
# cada cambio por separado.
# Se usa CalibratedClassifierCV en vez de SVC(probability=True) porque ese
# parametro quedo deprecado en scikit-learn 1.9
configs = [
    ("default (rbf, C=1)", dict(kernel="rbf", C=1.0, random_state=42)),
    ("linear, C=1", dict(kernel="linear", C=1.0, random_state=42)),
    ("rbf, class_weight=balanced", dict(kernel="rbf", C=1.0, class_weight="balanced", random_state=42)),
    ("rbf, C=5", dict(kernel="rbf", C=5.0, random_state=42)),
]

for name, params in configs:
    preprocessor = build_preprocessor()
    svm_base = SVC(**params)
    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("model", CalibratedClassifierCV(svm_base, ensemble=False)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    p = precision_score(y_test, y_pred)
    r = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name}: P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc:.4f} Acc={acc:.4f}")
    print(cm)
    print()