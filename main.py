import pandas as pd
import numpy as np
from fastapi import FastAPI,  HTTPException
import ast
from datetime import datetime

df_limpio = pd.read_json('nuevos_datos.json')

df_limpio['release_date'] = pd.to_datetime(df_limpio["release_date"], errors='coerce')
df_limpio['metascore'] = pd.to_numeric(df_limpio['metascore'], errors='coerce')

app = FastAPI(title='Proyecto de Machine Learning Operations de juegos de Steam')

data = pd.read_json('datos_exportados.json')

data['release_date'] = pd.to_datetime(data['release_date'], format='mixed', errors='coerce')

@app.get('/genero/{anio}')
def genero(anio):
   # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['release_date'].dt.year == int(anio)]

    # Contar la frecuencia de cada género
    generos_contados = df_anio['genres'].explode().value_counts()

    # Tomar los 5 géneros más vendidos
    top_generos = generos_contados.head(5)

    # Obtener la lista de géneros junto con su posición en la lista
    top_generos_con_posicion = [(posicion + 1, genero) for posicion, genero in enumerate(top_generos.index)]

    return top_generos_con_posicion

@app.get('/juegos/{anio}')
def juegos(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')

    # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['release_date'].dt.year == anio_entero]

    if df_anio.empty:
        raise HTTPException(status_code=404, detail='No se encontraron juegos para el año proporcionado.')

    # Obtener los juegos lanzados en ese año
    juegos_lanzados = df_anio['app_name'].tolist()

    return juegos_lanzados

@app.get('/specs/{anio}')
def specs(anio):
    try:
        # Validar que el año sea un valor numérico válido
        anio_entero = int(anio)
    except ValueError:
        raise HTTPException(status_code=400, detail='El año proporcionado no es válido.')

    # Filtrar los registros del año ingresado
    df_anio = df_limpio[df_limpio['release_date'].dt.year == anio_entero]

    if df_anio.empty:
        raise HTTPException(status_code=404, detail='No se encontraron juegos para el año proporcionado.')

    # Contar las especificaciones más comunes
    specs_contados = df_anio['specs'].explode().value_counts()

    # Obtener las 5 especificaciones más comunes junto con su posición
    top_specs = specs_contados.head(5)
    top_specs_con_posicion = [(posicion + 1, espec) for posicion, espec in enumerate(top_specs.index)]

    return top_specs_con_posicion

@app.get('/earlyaccess/{anio}')
def earlyaccess(anio):
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
def sentiment(anio):
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
def metascore(anio):
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