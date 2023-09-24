import pandas as pd
import numpy as np
from fastapi import FastAPI,  HTTPException
import ast
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Cambiamos a tipo date las fechas
df_limpio["release_date"] = pd.to_datetime(df_limpio["release_date"])

# Cambiamos el tipo de dato de sentiment a categórico
df_limpio["sentiment"] = df_limpio["sentiment"].astype("category")

app = FastAPI(title='Proyecto de Machine Learning Operations de juegos de Steam')

# Variables globales
df_limpio = None
rmse_train = None
df_entrenado = None
tree_regressor = None
label_encoder = None

# Cargar datos iniciales
@app.on_event("startup")
async def load_data_and_model():
    global df_limpio, df_entrenado, tree_regressor, label_encoder,rmse_train

    # Cargar los datos del CSV usado para las funciones, resultado del ETL
    df_limpio = pd.read_csv('nuevos_datos.csv')

    # Cargar los datos del CSV usado para el entrenamiento del modelo, resultado del EDA
    df_entrenado = pd.read_csv('df_modelo.csv')

    # Cargar el LabelEncoder usado para los genres desde el archivo .pkl
    label_encoder = joblib.load('label_encoder.pkl')

    # Feature engineering
    features = ["early_access","genres_encoded","metascore","año"]
    X = df_entrenado[features]
    y = df_entrenado["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    tree_regressor = DecisionTreeRegressor(max_depth=30, min_samples_leaf=1, min_samples_split=2)

    tree_regressor.fit(X_train, y_train)

    y_train_pred = tree_regressor.predict(X_train)
    y_test_pred = tree_regressor.predict(X_test)

    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

@app.get('/genero/{anio}')
async def genero(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')
   # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['año'] == int(anio)]

    if df_anio.empty:
        raise HTTPException(status_code=404, detail='No se encontraron juegos para el año proporcionado.')

    # Contar la frecuencia de cada género
    generos_contados = df_anio['genres'].explode().value_counts()

    # Tomar los 5 géneros más vendidos
    top_generos = generos_contados.head(5)

    # Obtener la lista de géneros junto con su posición en la lista
    top_generos_con_posicion = [(posicion + 1, genero) for posicion, genero in enumerate(top_generos.index)]

    return top_generos_con_posicion

@app.get('/juegos/{anio}')
async def juegos(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')

    # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['año'] == anio_entero]

    if df_anio.empty:
        raise HTTPException(status_code=404, detail='No se encontraron juegos para el año proporcionado.')

    # Obtener los juegos lanzados en ese año
    juegos_lanzados = df_anio['app_name'].tolist()

    return juegos_lanzados

@app.get('/specs/{anio}')
async def specs(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')

    # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['año'] == anio_entero]

    if df_anio.empty:
        raise HTTPException(status_code=404, detail='No se encontraron juegos para el año proporcionado.')

    # Contar las especificaciones más comunes
    specs_contados = df_anio['specs'].explode().value_counts()

    # Obtener las 5 especificaciones más comunes junto con su posición
    top_specs = specs_contados.head(5)
    top_specs_con_posicion = [(posicion + 1, espec) for posicion, espec in enumerate(top_specs.index)]

    return top_specs_con_posicion

@app.get('/earlyaccess/{anio}')
async def earlyaccess(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')

    # Filtrar los registros del año ingresado y que estén en early access
    df_anio_early_access = df_limpio[(df_limpio['release_date'].dt.year == anio_entero) & (df_limpio['early_access'] == True)]

    if df_anio_early_access.empty:
        raise HTTPException(status_code=404, detail='No se encontraron juegos en early access para el año proporcionado.')

    # Obtener la cantidad de juegos en early access
    cantidad_early_access = df_anio_early_access.shape[0]

    return cantidad_early_access

@app.get('/sentiment/{anio}')
async def sentiment(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')

    # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['release_date'].dt.year == anio_entero]

    if df_anio.empty:
        raise HTTPException(status_code=404, detail='No se encontraron registros para el año proporcionado.')

    # Obtener el conteo de las categorías de sentimiento
    conteo_categorias = df_anio['sentiment'].value_counts()

    return conteo_categorias.to_dict()

@app.get('/metascore/{anio}')
async def metascore(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')

    # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['release_date'].dt.year == anio_entero]

    if df_anio.empty:
        raise HTTPException(status_code=404, detail='No se encontraron registros para el año proporcionado.')

    # Ordenar los juegos por su puntaje (metascore) en orden descendente
    df_ordenado = df_anio.sort_values(by='metascore', ascending=False)

    # Tomar los 5 juegos con mayor puntaje
    top_5_juegos = df_ordenado.head(5)

    return top_5_juegos[['app_name', 'metascore']].to_dict(orient='records')

@app.get("/prediction/")
async def prediction(genre: str, early_access: bool, metascore: int, year: int):
    # Verificar que el género ingresado esté presente en el LabelEncoder
    if genre not in label_encoder.classes_:
        genres_list = ", ".join(label_encoder.classes_)
        print(f"Error: El género '{genre}' no está presente en el dataset.")
        print(f"Los géneros disponibles son: {genres_list}")
        return None, None
    
    # Obtener el valor codificado del género usando el LabelEncoder
    genre_encoded = label_encoder.transform([genre])[0]
    
    # Verificar que el metascore ingresado esté presente en el dataset
    if metascore not in df_entrenado["metascore"].unique():
        metascores_list = ", ".join(map(str, df_entrenado["metascore"].unique()))
        print(f"Error: El metascore '{metascore}' no está presente en el dataset.")
        print(f"Los metascores disponibles son: {metascores_list}")
        return None, None
    
    # Verificar que el año ingresado esté presente en el dataset
    if year not in df_entrenado["año"].unique():
        min_year = df_entrenado["año"].min()
        max_year = df_entrenado["año"].max()
        print(f"Error: El año '{year}' no está presente en el dataset.")
        print(f"El rango de años disponibles es de {min_year} a {max_year}.")
        return None, None
    
    # Crear un DataFrame con las características ingresadas
    data = pd.DataFrame({
        "early_access": [early_access],
        "genres_encoded": [genre_encoded],
        "metascore": [metascore],
        "año": [year]
    })
    
    # Realizar la predicción del precio utilizando el modelo entrenado
    price_pred = tree_regressor.predict(data)[0]
    
    return {"predicción_de_precio": price_pred, "rmse": rmse_train}