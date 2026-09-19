# Análisis de Datos (EDA)
--
-- Dataset analizado: data/raw/customer_churn_historical.csv
--


## ¿Qué encontramos en los datos?

Estuve revisando la tabla histórica de clientes y preparé este resumen con los puntos clave a tener en cuenta antes de armar el modelo:

1. Estructura general  
   El archivo cuenta con 7.043 registros y 21 columnas. La gran mayoría son datos de texto (servicios contratados, tipo de contrato, datos del cliente) y 3 de las columnas son numéricas (meses de antigüedad, cobros mensuales y cobros totales).

2. Calidad de datos y faltantes
   - No hay registros duplicados en la tabla.
   - Encontramos 26 valores faltantes únicamente en la columna TotalCharges. Para no tocar el archivo original a mano, lo resolveremos en el preprocesamiento completando esos nulos con la mediana.
   - La columna customerID se usa para identidicar a cada cliente de forma única. Como no aporta un valor predictivo no la cosnideramos una variable de analisis.

3. Variable de estudio Churn (Fuga de clientes) Variable categorida con dos valores posibles:Yes o No 
   - El 73.6% de los clientes permanece en la empresa (No).
   - El 26.4% se dio de baja (Yes).
   - Al haber menos clientes que se van que los que se quedan, tenemos que asegurarnos de que la separación de datos entre entrenamiento y prueba conserve esta misma proporción (usando la opción de estratificado stratify=y).
   --- Validación de customerID ---
   Clientes únicos: 7043
   Total de registros: 7043
   customerID nulos: 0
   Validación OK: todos los customerID son únicos.
   
## Estadísticas descriptivas

 Variable         Media    Mediana  Std      Min  Max 

 tenure           35.17    35       18.90    0    72 
 MonthlyCharges   68.17    73.91    24.98    18   114.41 
 TotalCharges     2312.08  2013.20  1573.97  0    7761.34 

*tenure* (antigüedad del cliente en meses) tiene bastante dispersión, va
desde clientes nuevos hasta clientes de 6 años.
*TotalCharges* tiene una diferencia grande entre media y mediana, lo que sugiere que la distribución está sesgada hacia la derecha (hay clientes con montos acumulados mucho más
altos que el resto).

## Promedio de variables numéricas según Churn 
       tenure  MonthlyCharges  TotalCharges
Churn                                      
No      37.28           63.94       2324.70
Yes     29.28           79.98       2276.81

Los clientes que se dan de baja tienen en promedio menos antigüedad (29 vs37 meses) y pagan una tarifa mensual más alta (79.98 vs 63.94). 
Tiene sentido: clientes más nuevos y con planes más caros parecen ser más propensos a irse.

Con el tipo de contrato se ve más claro:

Proporción de Churn según tipo de contrato
Churn               No      Yes
Contract                    
Month-to-month      61.27   38.73
One year            85.03   14.97
Two year            91.03   8.97

Los clientes con contrato mes a mes tienen una tasa de abandono mucho más alta que los que tienen contrato a uno o dos años. 
Esta variable parece ser una buena opción para predecir el churn.
---

**Resumen de columnas para el Preprocesamiento (para Pao)

Para facilitar el armado del preprocesador y los pipelines, dejo agrupadas las columnas según su tipo:

* Evaludar la posibilidad de eliminar la columna customerID porque no es de relevancia para la predicción
* Variable predictora (Target): Churn (mapear Yes como 1 y No como 0)
* Variables numéricas (3): tenure, MonthlyCharges, TotalCharges
* Variables categóricas (16): gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod