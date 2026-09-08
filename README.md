# Predicción de generación solar (AC_POWER) con árbol de decisión

## Objetivo

Predecir la potencia de salida (`AC_POWER`) de inversores solares a partir de
variables meteorológicas, usando el dataset [Solar Power Generation
Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
de Kaggle. El foco del proyecto es **modelado predictivo**, no solo
exploración de datos: se entrena, valida y compara un modelo de regresión
capaz de generalizar a datos no vistos.

## Datos

Dos plantas solares en India, cada una con dos ficheros:

| Fichero | Nivel | Columnas clave |
|---|---|---|
| `Plant_X_Generation_Data.csv` | Por inversor (`SOURCE_KEY`) | `DATE_TIME`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD` |
| `Plant_X_Weather_Sensor_Data.csv` | Por planta (un único sensor) | `DATE_TIME`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION` |

Periodo: 34 días, intervalos de 15 minutos. 22 inversores por planta.

## Metodología

1. **Limpieza de fechas**: los dos ficheros de la Planta 1 usaban formatos de
   fecha distintos (`DD-MM-YYYY` vs `YYYY-MM-DD`); se parsearon
   explícitamente con `format=` para evitar ambigüedad.
2. **Fusión**: generación + clima por `DATE_TIME` + `PLANT_ID` (no por
   `SOURCE_KEY`, que en el fichero de clima es un único sensor por planta).
3. **Detección de anomalías**: filas con irradiación alta y `AC_POWER = 0`
   señalan fallos puntuales de inversor. En la Planta 1 son marginales
   (0.07% de las filas, concentradas en 2 inversores); en la Planta 2 son
   mucho más frecuentes (5.2%) y afectan a los 22 inversores simultáneamente
   en varios instantes — evidencia de eventos de corte a nivel de planta
   completa (curtailment), no de fallos de equipo aislados.
4. **Feature engineering**:
   - `SOURCE_KEY` codificado con `LabelEncoder` (válido para árboles, que no
     asumen orden entre categorías al dividir por umbrales sucesivos).
   - `n_inversores_cero_dia`: número de inversores con `AC_POWER = 0` en el
     mismo instante, calculado solo sobre horas con irradiación (excluyendo
     la noche, donde el valor sería 22 de forma trivial y no informativa).
5. **Modelo**: `DecisionTreeRegressor` de scikit-learn.
6. **Validación**: split 80/20, con `GridSearchCV` (5-fold) sobre
   `max_depth` y `min_samples_leaf` para controlar el overfitting.
7. **Métrica de overfitting**: brecha relativa `(RMSE_test - RMSE_train) /
   RMSE_train`, más informativa que comparar solo R² (que comprime
   diferencias de error cuando ya está cerca de 1).

## Resultados

| Planta | Modelo final | RMSE test | R² test | Brecha train-test |
|---|---|---|---|---|
| 1 | `max_depth=None, min_samples_leaf=10` | 37.28 | 0.9910 | 12.5% |
| 2 | `max_depth=15, min_samples_leaf=20` | 184.88 | 0.7378 | 9.9% |

### Hallazgos clave

- **`IRRADIATION` domina la predicción en ambas plantas** (>85% de
  importancia), coherente con la física del problema: sin irradiación no hay
  generación fotovoltaica posible.
- **La Planta 1 es mucho más predecible que la Planta 2.** En una franja
  fija de irradiación (0.75-0.80), la potencia media varía solo un ~12%
  entre los 22 inversores de la Planta 1, pero un ~150% entre los de la
  Planta 2 (367 a 910). Esto indica heterogeneidad estructural real entre
  inversores en la Planta 2 — no ruido, sino una diferencia de
  comportamiento persistente y severa.
- **Eliminar los instantes de corte total de planta no mejoró el modelo de
  la Planta 2** (R² test pasó de 0.7378 a 0.7278): la causa principal del
  peor ajuste no son esos eventos puntuales, sino la heterogeneidad
  constante entre inversores.
- **Overfitting de árboles profundos**: un `max_depth` alto sin restricción
  de `min_samples_leaf` puede crear hojas con una única fila de soporte,
  haciendo que una anomalía puntual (un fallo de inversor) contamine la
  predicción de instancias similares en test. `min_samples_leaf` corrige
  esto de forma más directa que solo limitar la profundidad.

## Herramientas

Python, pandas, scikit-learn (`DecisionTreeRegressor`, `GridSearchCV`,
`LabelEncoder`), matplotlib.

## Posibles extensiones futuras

- Investigar si el bajo rendimiento de los inversores más flojos de la
  Planta 2 es constante en el tiempo o varía por fecha (degradación
  progresiva vs fallo desde el inicio).
- Comparar contra otros modelos (Random Forest, Gradient Boosting) para ver
  si la heterogeneidad de la Planta 2 se puede compensar mejor con ensambles
  de árboles.
- Incorporar `DAILY_YIELD`/`TOTAL_YIELD` tras resolver las inconsistencias
  de magnitud detectadas entre inversores.
