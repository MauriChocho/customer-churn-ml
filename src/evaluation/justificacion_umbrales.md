
# Justificacion de umbrales de riesgo de churn

## Objetivo

El objetivo de este analisis es definir y justificar los umbrales utilizados para clasificar a los clientes en tres niveles de riesgo: LOW, MEDIUM y HIGH.

Para ello, se analizaron las probabilidades estimadas por el modelo Logistic Regression, seleccionado por el equipo en la primera entrega.

## Metodologia

Se utilizo el dataset `customer_churn_historical.csv`, manteniendo la division acordada por el equipo:

- 80% de los datos para entrenamiento (5634 clientes).
- 20% de los datos para prueba (1409 clientes).
- `random_state=42` y `stratify=y`.

Para comparar los umbrales se aplico validacion cruzada estratificada de 5 particiones sobre el conjunto de entrenamiento.

En cada particion se entreno el pipeline completo, incluyendo el preprocesamiento y Logistic Regression con los parametros seleccionados originalmente.

De esta manera, se obtuvieron probabilidades de validacion para los 5634 clientes de entrenamiento sin utilizar el conjunto de prueba para seleccionar los umbrales.

El analisis puede reproducirse ejecutando:

```powershell
python -m src.evaluation.analisis_umbrales
```

## Propuestas evaluadas

Se compararon tres alternativas:

| Propuesta | LOW | MEDIUM | HIGH |
|---|---|---|---|
| A | Score menor a 0.30 | Desde 0.30 hasta menos de 0.50 | Desde 0.50 |
| B | Score menor a 0.20 | Desde 0.20 hasta menos de 0.50 | Desde 0.50 |
| C | Score menor a 0.30 | Desde 0.30 hasta menos de 0.60 | Desde 0.60 |

## Resultados obtenidos

| Indicador | A | B | C |
|---|---:|---:|---:|
| Clientes en LOW | 2284 | 1586 | 2284 |
| Abandonos reales en LOW | 126 | 68 | 126 |
| Clientes en MEDIUM | 1097 | 1795 | 1611 |
| Abandonos reales en MEDIUM | 224 | 282 | 380 |
| Clientes en HIGH | 2253 | 2253 | 1739 |
| Abandonos reales en HIGH | 1135 | 1135 | 979 |
| Clientes que no abandonaron en HIGH | 1118 | 1118 | 760 |

Los resultados muestran que la propuesta B reduce la cantidad de abandonos reales clasificados como LOW, mientras mantiene la cantidad de abandonos identificados en HIGH.

La propuesta C reduce las falsas alarmas de HIGH, pero tambien deja mas abandonos reales fuera del nivel de riesgo alto.

## Umbrales seleccionados

El equipo selecciono la **propuesta B**, con los siguientes valores:

- **LOW:** score menor a 0.20.
- **MEDIUM:** score desde 0.20 hasta menos de 0.50.
- **HIGH:** score igual o mayor a 0.50.

## Justificacion de la decision

El objetivo del proyecto es anticipar el abandono de clientes para facilitar acciones preventivas.

Por este motivo, se priorizo reducir la cantidad de clientes que efectivamente abandonan y quedan clasificados como riesgo bajo.

Con la propuesta B, los abandonos reales en LOW disminuyen de 126 a 68 respecto de la propuesta A. Esto permite identificar como riesgo medio a 58 clientes adicionales que realmente abandonaron.

Como contrapartida, MEDIUM aumenta de 1097 a 1795 clientes. Esto implica que un mayor numero de clientes podria requerir seguimiento preventivo.

Se mantuvo el limite HIGH en 0.50 porque permite conservar 1135 abandonos reales dentro del grupo de riesgo alto.

Subir ese limite a 0.60 reduciria las falsas alarmas, pero tambien trasladaria 156 abandonos reales desde HIGH hacia MEDIUM.

Dado que no contamos con informacion sobre costos de retencion ni capacidad operativa para atender alertas, no se considero adecuado priorizar esa reduccion de falsas alarmas por encima de la deteccion de posibles abandonos.

## Limitaciones del analisis

- Los resultados se obtuvieron mediante validacion cruzada sobre el conjunto de entrenamiento.
- Los umbrales fueron seleccionados comparando tres alternativas, no mediante una busqueda exhaustiva.
- El modelo utiliza `class_weight="balanced"`, por lo que sus scores no deben interpretarse automaticamente como probabilidades perfectamente calibradas.
- Los niveles LOW, MEDIUM y HIGH son una clasificacion adicional. Su definicion no modifica las predicciones binarias originales ni las metricas previamente registradas.
- No se dispone de informacion sobre costos de retencion o capacidad operativa de la empresa para evaluar el impacto economico de cada alternativa.

## Conclusion

Se seleccionaron los umbrales 0.20 y 0.50 porque permiten reducir los abandonos reales clasificados como riesgo bajo y mantener la deteccion de posibles abandonos en el grupo HIGH.

La decision responde al objetivo preventivo del proyecto y se encuentra respaldada por la comparacion de resultados obtenidos mediante validacion cruzada.

Los valores seleccionados se utilizaran en la funcion `calcular_risk_level()` de la capa de inferencia.
