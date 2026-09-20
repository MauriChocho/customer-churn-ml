# Comparacion de modelos

Este documento resume los resultados obtenidos por los distintos modelos probados para predecir churn.

Para que la comparacion sea justa, todos los modelos deben utilizar:

- el mismo dataset
- el mismo preprocessing
- la misma division train/test
- `test_size=0.20`
- `random_state=42`
- `stratify=y`

La metrica principal acordada para comparar los modelos es **F1-score**, ya que el problema presenta desbalance entre clientes que abandonan y clientes que no abandonan.

Tambien se analizan Precision, Recall, ROC-AUC y Accuracy como metricas complementarias.

## Resultados generales

| Modelo | Precision | Recall | F1-score | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.7360 |
| Logistic Regression | 0.4812 | 0.7231 | 0.5779 | 0.8118 | 0.7211 |
| Decision Tree | 0.4527 | 0.6048 | 0.5178 | 0.7089 | 0.7026 |
| Random Forest | 0.6564 | 0.4005 | 0.4975 | 0.7930 | 0.7864 |
| SVM           | 0.6293 | 0.4973 | 0.5556 | 0.8017 | 0.7899 |

## Detalle por modelo

### Baseline (DummyClassifier)

Se utilizo un DummyClassifier con `strategy="most_frequent"` como punto de referencia minimo. Al tratarse de un baseline, no se probaron variantes de configuracion.

Configuracion utilizada:

```python
DummyClassifier(
    strategy="most_frequent",
    random_state=42
)
```

Matriz de confusion:

```text
[[1037    0]
 [ 372    0]]
```

Interpretacion:

- 1037 clientes sin churn fueron clasificados correctamente, porque el modelo siempre predice la clase mayoritaria.
- 372 clientes con churn no fueron detectados en ningun caso.
- El modelo no distingue entre clases: Precision, Recall y F1-score son 0, y el ROC-AUC (0.5000) equivale a una prediccion al azar.

Este resultado confirma que Accuracy sola no es una metrica adecuada para este problema: un modelo que no aprendio nada obtiene un Accuracy relativamente alto (0.7360) unicamente por el desbalance de clases.


### Decision Tree

Se probo el modelo de arbol de decision con parametros standard y otras variables para mejorar resultados.

|      Configuracion     | Precision | Recall | F1-score | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
|        Standard        | 0.4928 | 0.4624 | 0.4771 | 0.6458 | 0.7324 |
|       max_depth=10      | 0.5302 | 0.4489 | 0.4862 | 0.7041 | 0.7495 |
| max_depth=10 + class_weight="balanced" | 0.4527 | 0.6048 | 0.5178 | 0.7089 | 0.7026 |

** Aclaración: cada train sumo el parametro anterior.

**Matriz de confusion**
765 → Verdaderos No-Churn: clientes que no se iban, y el modelo acerto que no se iban.
225 → Verdaderos Churn: clientes que si se iban, y el modelo acerto que se iban.
272 → Falsos Positivos: el modelo dijo "este se va" pero en realidad no se iba.
147 → Falsos Negativos: el modelo dijo "este se queda" pero en realidad si se iba.

**Observaciones:**
- El modelo standard (sin restricciones) tiene overfitting severo: F1-score de 1.0000 en train vs. 0.4771 en test.
- Limitar la profundidad (max_depth=10) mejoro Precision y ROC-AUC, aunque bajo levemente el Recall.
- Sumar parametro class_weight="balanced" mejoro significativamente el Recall (0.4489 → 0.6048), dando el mejor F1-score general.

**Mejor Configuración final:** max_depth=10 + class_weight="balanced" (mejor F1-score y mejor ROC-AUC)


### Random Forest

Se probaron distintas configuraciones de Random Forest manteniendo constante el preprocessing y la division de datos.

| Configuracion | Precision | Recall | F1-score | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Configuracion inicial | 0.6564 | 0.4005 | 0.4975 | 0.7930 | 0.7864 |
| `class_weight="balanced"` | 0.6283 | 0.3817 | 0.4749 | 0.7896 | 0.7771 |
| `n_estimators=300, max_depth=8` | 0.6667 | 0.3548 | 0.4632 | 0.8022 | 0.7828 |
| `max_depth=12` | 0.6466 | 0.4032 | 0.4967 | 0.7972 | 0.7842 |

Configuracion seleccionada:

```python
RandomForestClassifier(
    random_state=42
)
```

Se mantuvo esta configuracion porque obtuvo el mayor F1-score entre las variantes probadas.

Matriz de confusion:

```text
[[959  78]
 [223 149]]
```

Interpretacion:

- 959 clientes sin churn fueron clasificados correctamente.
- 78 clientes sin churn fueron clasificados incorrectamente como churn.
- 223 clientes con churn no fueron detectados.
- 149 clientes con churn fueron detectados correctamente.

El Recall relativamente bajo indica que el modelo todavia deja sin detectar una parte importante de los clientes que realmente abandonan.

### Logistic Regression

Se probaron distintas configuraciones de Logistic Regression manteniendo constante el preprocessing y la division de datos.

| Configuracion | Precision | Recall | F1-score | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Configuracion inicial | 0.6614 | 0.4516 | 0.5367 | 0.8119 | 0.7942 |
| `class_weight="balanced"` | 0.4803 | 0.7204 | 0.5763 | 0.8117 | 0.7204 |
| `class_weight="balanced", C=0.1` | 0.4812 | 0.7231 | 0.5779 | 0.8118 | 0.7211 |

Configuracion seleccionada:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced",
    C=0.1
)
```

Se selecciono la configuracion `class_weight="balanced", C=0.1` porque obtuvo el mayor F1-score entre las variantes probadas (0.5779), que es la metrica principal acordada para comparar los modelos. Ademas, mejoro significativamente el Recall, pasando de 0.4516 a 0.7231, lo que permite detectar una mayor cantidad de clientes que realmente abandonan.

Como contrapartida, la Precision bajo de 0.6614 a 0.4812, por lo que aumentan los falsos positivos. Esto significa que el modelo identifica como posibles churn a mas clientes que finalmente no abandonan. Sin embargo, para este proyecto se prioriza el equilibrio entre Precision y Recall medido mediante F1-score.

Matriz de confusion:

```text
[[747  290]
 [103 269]]
```

Interpretacion:

- 747 clientes sin churn fueron clasificados correctamente.
- 290 clientes sin churn fueron clasificados incorrectamente como churn.
- 103 clientes con churn no fueron detectados.
- 269 clientes con churn fueron detectados correctamente.

Comparado con Random Forest (F1 0.4975), Logistic Regression obtuvo un mejor F1-score (0.5779), un Recall considerablemente mayor (0.7231 vs. 0.4005) y un ROC-AUC superior (0.8118 vs. 0.7930). Sin embargo, Random Forest obtuvo mejores resultados en Precision y Accuracy. La configuracion seleccionada de Logistic Regression logra detectar una mayor proporcion de clientes con churn, aunque a costa de generar mas falsos positivos.


### Pruebas con el modelo SVM (Support Vector Machine)

Para probar una alternativa distinta a los árboles y a la regresión logística, entrenamos un modelo SVM usando exactamente la misma partición y preprocesamiento que acordamos para todo el proyecto (`test_size=0.20`, `stratify=y`, `random_state=42`).
Al ejecutar el entrenamiento (`train_svm.py`), obtuvimos los siguientes resultados sobre el conjunto de prueba:

* **Accuracy:** 78.99%
* **Precision:** 62.93%
* **Recall:** 49.73%
* **F1-score:** 55.56%
* **ROC-AUC:** 80.17%

#### ¿Qué nos muestra la matriz de confusión?
Matriz de confusion:
```text
[[928 109]
 [187 185]]
```

* **928 clientes que se quedaban** El modelo predijo correctamente que permanecían en la empresa (Verdaderos Negativos).
* **109 clientes estables** El modelo los clasificó por error como si fueran a darse de baja (Falsos Positivos).
* **185 clientes en riesgo real** El modelo los identificó a tiempo para poder aplicar acciones de retención (Verdaderos Positivos).
* **187 clientes en riesgo real** El modelo no los detectó y los consideró estables (Falsos Negativos).

#### Conclusión del experimento

Lo positivo de probar SVM es que **mejoró la detección de clientes en fuga respecto al Random Forest inicial**: el F1-score subió de 0.4975 a 0.5556 y el Recall mejoró de 40.05% a 49.73%. 

Sin embargo, el punto débil sigue siendo que se le escapan 187 clientes en riesgo (casi el 50% del total de casos de Churn). Para el negocio esto es relevante porque un falso negativo significa perder al cliente sin haber actuado preventivamente.


## Seleccion del modelo candidato

La seleccion final se realizara cuando todos los modelos hayan sido entrenados y evaluados utilizando las mismas condiciones.

El criterio principal sera el F1-score, complementado con Recall, Precision y ROC-AUC.