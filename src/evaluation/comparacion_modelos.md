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
| Baseline | — | — | — | — | — |
| Logistic Regression | 0.6614 | 0.4516 | 0.5367 | 0.8119 | 0.7942 |
| Decision Tree | — | — | — | — | — |
| Random Forest | 0.6564 | 0.4005 | 0.4975 | 0.7930 | 0.7864 |

## Detalle por modelo

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

Configuracion utilizada:

​```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
​```

Metricas obtenidas:

| Metrica | Valor |
|---|---|
| Precision | 0.6614 |
| Recall | 0.4516 |
| F1-score | 0.5367 |
| ROC-AUC | 0.8119 |
| Accuracy | 0.7942 |

Matriz de confusion:

​```text
[[951  86]
 [204 168]]
​```

Interpretacion:

- 951 clientes sin churn fueron clasificados correctamente.
- 86 clientes sin churn fueron clasificados incorrectamente como churn.
- 204 clientes con churn no fueron detectados.
- 168 clientes con churn fueron detectados correctamente.

Comparado con Random Forest (F1 0.4975), Logistic Regression obtuvo mejor resultado en las cuatro metricas principales (F1 0.5367, ROC-AUC 0.8119). El Recall sigue siendo el punto debil de ambos modelos: mas de la mitad de los clientes que se van no son detectados a tiempo.

## Seleccion del modelo candidato

La seleccion final se realizara cuando todos los modelos hayan sido entrenados y evaluados utilizando las mismas condiciones.

El criterio principal sera el F1-score, complementado con Recall, Precision y ROC-AUC.