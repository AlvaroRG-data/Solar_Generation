# %%
import pandas as pd
# %%
gen = pd.read_csv("Plant_1_Generation_Data.csv")
weather = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
# %%
print(gen.shape, weather.shape)
# %%
print(gen.head())
print(weather.head())
# %%
print(gen.dtypes)
print(weather.dtypes)
# Comentarios sobre el data interesantes, las fechas tienen distinto formato
#luego la source_key son distintos, esto es por la cantidad de medidores de cada set. Es por ello que sería
#ridiculo plantear una division por source_key
# %%
#Vamos a juntar las fechas en el mismo formato
gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format="%d-%m-%Y %H:%M")
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
print(gen.dtypes)
print(weather.dtypes)
#revisamos que esta bien
# %%
#Juntamos para poder trabajar con solo un dataset
plant1 = gen.merge(
    weather[["DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]],
    on=["DATE_TIME", "PLANT_ID"],
    how="left"
)
print(plant1.shape)
print(plant1.head())
print(plant1.isna().sum())
#%%
print(gen["SOURCE_KEY"].nunique())
print(gen["DATE_TIME"].nunique())
print(gen.shape[0])
# %%
import matplotlib.pyplot as plt
# %%
plt.figure(figsize=(7, 5))
plt.scatter(plant1["IRRADIATION"], plant1["AC_POWER"], alpha=0.3, s=5)
plt.xlabel("IRRADIATION")
plt.ylabel("AC_POWER")
plt.title("AC_POWER vs IRRADIATION - Planta 1")
plt.show()

#Cosas interesantes: 
# -Es posiblemente una recta, calculable por minimos cuadrados
# -Hay puntos con AC 0, seguramente que esten desconectados, vamos a revisarlo. (Si esto fuese
# algo mayor que un ejercicio, hubiese estado bien investigar mayor información sobre estos procesos)
# %%
anomalias = plant1[(plant1["IRRADIATION"] > 0.5) & (plant1["AC_POWER"] == 0)]
print(anomalias.shape)
print(anomalias["SOURCE_KEY"].value_counts())
print(anomalias["DATE_TIME"].dt.hour.value_counts())
#Los horarios son tarde, lo cual no tiene sentido que sea cuando esta arrancando el sistema
#Por otro lado esta concentrado en 8 unidades, y principalmente en 2. Aun asi son eventos aislados.
#Como mucho en algo real se podría ver si es algo esporárico o si interesa revisar esos sistemas.
#Aun asi, son muy pocos casos
# %%
plant1_clean = plant1.drop(anomalias.index)
print(plant1_clean.shape)
# %%
franja = plant1_clean[(plant1_clean["IRRADIATION"] > 0.75) & (plant1_clean["IRRADIATION"] < 0.80)]
print(franja.groupby("SOURCE_KEY")["AC_POWER"].mean().sort_values())
#Esto es para ver como estan los inversores especificos, como se puede observar cada uno tiene un rendimiento unico,
#asi que hemos dado con un resultado interesante
# %%
#Vamos a continuar por arboles, pues nos interesa hacerlo segun como de eficiente sea
from sklearn.model_selection import train_test_split
#%%
X = plant1_clean[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "SOURCE_KEY"]]
y = plant1_clean["AC_POWER"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
# %%
#Como los label son dificiles de trabajar al no ser numericos, tenemos dos opciones, crear una columna para cada uno,
# o crear un label para cada uno, iremos con el label
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
#%%
le = LabelEncoder()

X_train = X_train.copy()
X_test = X_test.copy()

X_train["SOURCE_KEY"] = le.fit_transform(X_train["SOURCE_KEY"])
X_test["SOURCE_KEY"] = le.transform(X_test["SOURCE_KEY"])

print(X_train["SOURCE_KEY"].unique())

# %%
#Con esto podemos entrenar el arbol, para poder hacer una prediccion
tree = DecisionTreeRegressor(max_depth=6, random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
# %%
from sklearn.metrics import root_mean_squared_error, r2_score

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#El error cuadratico medio depende de cada ejercicio, el R^2 es de los apuntes de MDLR, es el porcentaje de varianza que se "explica"
#Para este caso, RMSE bajo, lo cual es bueno, y R^2 alto, lo cual tambien es bueno
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
# %%
#Ahora vamos a ver un par de cosas que son interesantes (que no vimos en teoría de MLDR) vamos primero a ver el R^2 del conj entrenador
#si nos sale mucho más alto que el conjunto de prueba significa que tenemos overfitting, ie. tenemos más ramas de las necesarias
y_train_pred = tree.predict(X_train)
rmse_train = root_mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print(f"Train — RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")
#Todo perfecto pues

# %%
#Ahora vamos a ver la importancia de cada una de los valores, esto tambien lo hicimos en MLDR, pero con otros metodos
importancias = pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importancias)
#Sorpresa, la source_key no da mucha importancia, pero esto no descarta que no tenga importancia, pues el arbol todavía
#puede ser bastante pequeño como para que tenga relevancia, en 6 pasos puede que solo haya dividido en grupos de radiación
# pues (obviamente) tiene mucho peso

#La conclusión que sacamos de todo es que podemos añadirle mayor profundidad, pues no hay overfittin y algo que hemos observado
#no tiene reflejo en un arbol tan pequeño
# %%
#repetimos el proceso, esta vez del tiron realente
tree_deep = DecisionTreeRegressor(max_depth=12  , random_state=42)
tree_deep.fit(X_train, y_train)

y_pred_deep = tree_deep.predict(X_test)
y_train_pred_deep = tree_deep.predict(X_train)

print("Train:", root_mean_squared_error(y_train, y_train_pred_deep), r2_score(y_train, y_train_pred_deep))
print("Test:", root_mean_squared_error(y_test, y_pred_deep), r2_score(y_test, y_pred_deep))

importancias_deep = pd.Series(tree_deep.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importancias_deep)

#Vemos que hay overfitting, sobre todo se nota en el error cuadratico, pues al tener unas cifras cercanas al 1 R^2 aumenta poco.
#De hecho, si aumentamos la profundidad se ve más, que aumenta el fallo en el test y decrece mucho el del train, ie. los datos son tan precisos que predicen peor

# %%
#la proxima traigo todo la vd
from sklearn.model_selection import GridSearchCV
#%%
#Vamos a continuar con validación cruzada para no estar con esto media hora probando simplemente
param_grid = {"max_depth": [4, 6, 8, 10, 12, 15, None]}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5
)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(-grid.best_score_)
#Obtenemos que 10 tiene buena pinta
# %%
#Vamos a comparar ahora 
best_tree = grid.best_estimator_

y_pred_best = best_tree.predict(X_test)
y_train_pred_best = best_tree.predict(X_train)

print("Train:", root_mean_squared_error(y_train, y_train_pred_best), r2_score(y_train, y_train_pred_best))
print("Test:", root_mean_squared_error(y_test, y_pred_best), r2_score(y_test, y_pred_best))

# %%
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_best, alpha=0.3, s=5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Predicción perfecta")
plt.xlabel("AC_POWER real")
plt.ylabel("AC_POWER predicho")
plt.title("Predicho vs Real - Árbol de decisión (max_depth=10)")
plt.legend()
plt.show()

#Destacan en el grafico un conjunto de puntos con produccion pero predicción nula
#lo cual es al menos raro, así que vamos a investigarlo un poco
#%%
resultados = X_test.copy()
resultados["AC_POWER_real"] = y_test
resultados["AC_POWER_predicho"] = y_pred_best

casos = resultados[(resultados["AC_POWER_predicho"] < 50) & (resultados["AC_POWER_real"] > 200)]

print(casos.shape)
print(casos[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "AC_POWER_real", "AC_POWER_predicho"]])
#Fuah, extraño en verdad, pues tienen IRRADIATION sobre 0.45, lo cual no es una cantidad nula
#debemos investigar esto en mas detalle
# %%
print(casos.index)
print(X_test.loc[casos.index, "SOURCE_KEY"])
#BINGO, fijate que son solo de dos paneles especificos, por tanto, veamos y revisemos los que fallaban
print(le.inverse_transform([0, 20]))
print(anomalias["SOURCE_KEY"].value_counts())
#Exacto, se trata de 1 de ellos, pero el otro es un poco más extraño.
# %%
#Procedemos a investigar
X_train_check = X_train.copy()
X_train_check["AC_POWER"] = y_train

zbi_code = le.transform(["zBIq5rxdHJRwDNY"])[0]

subset = X_train_check[
    (X_train_check["SOURCE_KEY"] == zbi_code) &
    (X_train_check["IRRADIATION"] > 0.40) &
    (X_train_check["IRRADIATION"] < 0.50)
]

print(subset.shape)
print(subset["AC_POWER"].describe())
#vuelven a ser cosas rarillas, el conjunto de datos es amplio y con buenos datos, si bien hay un minimo de 0, es raro
# %%
#Vamos a ver la hoja en especifico
hojas_train = best_tree.apply(X_train)
hojas_test_casos = best_tree.apply(X_test.loc[casos.index])

print(hojas_test_casos)

# Filas de train que caen exactamente en esa misma hoja
hoja_objetivo = hojas_test_casos[0]
mismo_grupo = X_train[hojas_train == hoja_objetivo].copy()
mismo_grupo["AC_POWER"] = y_train[hojas_train == hoja_objetivo]

print(mismo_grupo.shape)
print(mismo_grupo)
#Vemos un solo caso, que se trata de una fila anomala, asi que lo que podemos hacer es ver un poco de manejo sobre las hojas,
# es decir, plantear un minimo de lineas de arbol
# %%
#Probemos ahora con esto en cuenta
param_grid2 = {"max_depth": [6, 8, 10, 12, 15, None], "min_samples_leaf": [1, 5, 20, 50]}

grid2 = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid2,
    scoring="neg_root_mean_squared_error",
    cv=5
)
grid2.fit(X_train, y_train)

print(grid2.best_params_)
print(-grid2.best_score_)
#Sale mejor, pues es un limite mejor que un numero de pasos
# %%
#Probamos
best_tree2 = grid2.best_estimator_

y_pred_best2 = best_tree2.predict(X_test)
y_train_pred_best2 = best_tree2.predict(X_train)

print("Train:", root_mean_squared_error(y_train, y_train_pred_best2), r2_score(y_train, y_train_pred_best2))
print("Test:", root_mean_squared_error(y_test, y_pred_best2), r2_score(y_test, y_pred_best2))
#Ugh? hay mejora, pero una disparidad tan grande entre train y test merece la pena investigar
# %%
#Repetimos todo el proceso pero esta vez del tiron
param_grid3 = {"max_depth": [None, 15, 20], "min_samples_leaf": [5, 10, 20, 30, 50]}

grid3 = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid3,
    scoring="neg_root_mean_squared_error",
    cv=5
)
grid3.fit(X_train, y_train)

print(grid3.best_params_)
print(-grid3.best_score_)

best_tree3 = grid3.best_estimator_
y_pred_best3 = best_tree3.predict(X_test)
y_train_pred_best3 = best_tree3.predict(X_train)

rmse_train3 = root_mean_squared_error(y_train, y_train_pred_best3)
rmse_test3 = root_mean_squared_error(y_test, y_pred_best3)

print("Train:", rmse_train3, r2_score(y_train, y_train_pred_best3))
print("Test:", rmse_test3, r2_score(y_test, y_pred_best3))
print("Brecha relativa:", (rmse_test3 - rmse_train3) / rmse_train3 * 100, "%")
#Bingo, vemos que 10 es mejor minimo, ademas con menor disparidad
# %%
#vamos a ver las importancias, pues puede ser interesante
importancias3 = pd.Series(best_tree3.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importancias3)
#Nada a destacar en verdad, como mucho que ahora influye source_key un poco mas
# %%
#Con esto tenemos un estudio bastante completo de la primera planta, vamos ahora a ver la segunda planta
#que debería de ser similar, asi que seguramente avance más rápido por aquí. 
gen2 = pd.read_csv("Plant_2_Generation_Data.csv")
weather2 = pd.read_csv("Plant_2_Weather_Sensor_Data.csv")

print(gen2.shape, weather2.shape)
print(gen2.head())
print(weather2.head())
print(gen2.dtypes)
print(weather2.dtypes)
#Destacable, los dias presentan mismo formato, pero vamos a volverlo un formato de fecha
# %%
gen2["DATE_TIME"] = pd.to_datetime(gen2["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
weather2["DATE_TIME"] = pd.to_datetime(weather2["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

print(gen2["SOURCE_KEY"].nunique())
print(gen2["DATE_TIME"].nunique())

# %%
#juntamos tambien
plant2 = gen2.merge(
    weather2[["DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]],
    on=["DATE_TIME", "PLANT_ID"],
    how="left"
)

print(plant2.shape)
print(plant2.isna().sum())
#A destacar, mejor cobertura ahora, no hay discontinuidades

# %%
anomalias2 = plant2[(plant2["IRRADIATION"] > 0.5) & (plant2["AC_POWER"] == 0)]
print(anomalias2.shape)
print(anomalias2["SOURCE_KEY"].value_counts())
print(anomalias2["DATE_TIME"].dt.hour.value_counts())
#PFF, muchas mas anomalias, se ve que es un problema general, no estarán limpias o algo. Pero se ve
#que es algo de las horas, seguramente a ciertas horas les de más sombra
# %%
#veamos si es un caso expecifico o algo general, ie. si se trata de una desconexión de un día, de sombra
#que da todos los dias o similar
conteo_por_momento = anomalias2.groupby("DATE_TIME")["SOURCE_KEY"].nunique().sort_values(ascending=False)
print(conteo_por_momento.head(15))
#Se ve como algo general, tenemos pues que decidir que hacer. Decido construir una variable que mire si hay fallo general

# %%
ceros_por_instante = plant2.groupby("DATE_TIME")["AC_POWER"].apply(lambda x: (x == 0).sum())
ceros_por_instante.name = "n_inversores_cero"

plant2 = plant2.merge(ceros_por_instante, on="DATE_TIME", how="left")

print(plant2[["DATE_TIME", "SOURCE_KEY", "AC_POWER", "n_inversores_cero"]].head(10))
print(plant2["n_inversores_cero"].describe())

#Problema, la noche no está diferenciada, por ello vamos a poner una cota sobre la irradiacion
# %%
plant2_dia = plant2[plant2["IRRADIATION"] > 0.1]

ceros_dia = plant2_dia.groupby("DATE_TIME")["AC_POWER"].apply(lambda x: (x == 0).sum())
ceros_dia.name = "n_inversores_cero_dia"

plant2 = plant2.drop(columns=["n_inversores_cero"])  # descartamos la versión anterior, mal definida
plant2 = plant2.merge(ceros_dia, on="DATE_TIME", how="left")

plant2["n_inversores_cero_dia"] = plant2["n_inversores_cero_dia"].fillna(0)

print(plant2["n_inversores_cero_dia"].describe())
# %%
#Empezamos a ver algo más del árbol. El prcedimiento es el mismo, con Grid search
X2 = plant2[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "SOURCE_KEY", "n_inversores_cero_dia"]]
y2 = plant2["AC_POWER"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

le2 = LabelEncoder()
X2_train = X2_train.copy()
X2_test = X2_test.copy()
X2_train["SOURCE_KEY"] = le2.fit_transform(X2_train["SOURCE_KEY"])
X2_test["SOURCE_KEY"] = le2.transform(X2_test["SOURCE_KEY"])

param_grid_p2 = {"max_depth": [None, 15, 20], "min_samples_leaf": [5, 10, 20, 30, 50]}

grid_p2 = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid_p2,
    scoring="neg_root_mean_squared_error",
    cv=5
)
grid_p2.fit(X2_train, y2_train)

print(grid_p2.best_params_)
print(-grid_p2.best_score_)

best_tree_p2 = grid_p2.best_estimator_
y2_pred = best_tree_p2.predict(X2_test)
y2_train_pred = best_tree_p2.predict(X2_train)

rmse_train_p2 = root_mean_squared_error(y2_train, y2_train_pred)
rmse_test_p2 = root_mean_squared_error(y2_test, y2_pred)

print("Train:", rmse_train_p2, r2_score(y2_train, y2_train_pred))
print("Test:", rmse_test_p2, r2_score(y2_test, y2_pred))
print("Brecha relativa:", (rmse_test_p2 - rmse_train_p2) / rmse_train_p2 * 100, "%")

importancias_p2 = pd.Series(best_tree_p2.feature_importances_, index=X2_train.columns).sort_values(ascending=False)
print(importancias_p2)
# %%
plt.figure(figsize=(7, 5))
plt.scatter(plant2["IRRADIATION"], plant2["AC_POWER"], alpha=0.3, s=5)
plt.xlabel("IRRADIATION")
plt.ylabel("AC_POWER")
plt.title("AC_POWER vs IRRADIATION - Planta 2")
plt.show()
#muchos puntos de baja produccion con irradiation alta
# %%
#Miremos si se trata de algo de algunas en especifico
franja2 = plant2[(plant2["IRRADIATION"] > 0.75) & (plant2["IRRADIATION"] < 0.80)]
print(franja2.groupby("SOURCE_KEY")["AC_POWER"].mean().sort_values())
#BINGO BANGO
# %%
#antes de dar nuestra investigación por concluida, vamos a plantear una ultima cuestion ¿que pasa si eliminamos los cortes?
#a lo mejor asi se explica mejor el tema
instantes_corte = ceros_dia[ceros_dia == 22].index
print(len(instantes_corte), "instantes de corte total")

plant2_sin_corte = plant2[~plant2["DATE_TIME"].isin(instantes_corte)]
print(plant2.shape, "->", plant2_sin_corte.shape)

# %%
X2b = plant2_sin_corte[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "SOURCE_KEY", "n_inversores_cero_dia"]]
y2b = plant2_sin_corte["AC_POWER"]

X2b_train, X2b_test, y2b_train, y2b_test = train_test_split(X2b, y2b, test_size=0.2, random_state=42)

le2b = LabelEncoder()
X2b_train = X2b_train.copy()
X2b_test = X2b_test.copy()
X2b_train["SOURCE_KEY"] = le2b.fit_transform(X2b_train["SOURCE_KEY"])
X2b_test["SOURCE_KEY"] = le2b.transform(X2b_test["SOURCE_KEY"])

grid_p2b = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid_p2,
    scoring="neg_root_mean_squared_error",
    cv=5
)
grid_p2b.fit(X2b_train, y2b_train)

best_tree_p2b = grid_p2b.best_estimator_
y2b_pred = best_tree_p2b.predict(X2b_test)
y2b_train_pred = best_tree_p2b.predict(X2b_train)

rmse_train_p2b = root_mean_squared_error(y2b_train, y2b_train_pred)
rmse_test_p2b = root_mean_squared_error(y2b_test, y2b_pred)

print(grid_p2b.best_params_)
print("Train:", rmse_train_p2b, r2_score(y2b_train, y2b_train_pred))
print("Test:", rmse_test_p2b, r2_score(y2b_test, y2b_pred))
print("Brecha relativa:", (rmse_test_p2b - rmse_train_p2b) / rmse_train_p2b * 100, "%")
#No hay mejora significativa, de hecho empeora ligeramente, por tanto rechazamos la hipotesis
# %%
