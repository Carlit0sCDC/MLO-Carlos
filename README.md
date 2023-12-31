# Proyecto de MLOps: Sistema de Recomendación de Videojuegos en Steam

Este proyecto aborda la creación de un sistema de recomendación de videojuegos para usuarios en la plataforma Steam. Comenzaremos desde cero, enfrentando desafíos de ingeniería de datos y culminando con un modelo de aprendizaje automático que será accesible a través de una API desarrollada con FastAPI.

## Contexto del Proyecto

### Descripción del Rol
Soy un Data Scientist en Steam y mi tarea es crear un sistema de recomendación de videojuegos para nuestros usuarios. Los datos disponibles son poco maduros y carecen de estructura adecuada, lo que hace que mi trabajo sea un desafío.

### Objetivo
Crear un Minimum Viable Product (MVP) para el sistema de recomendación de videojuegos.

### Pasos y Metodología

### 1. Ingeniería de Datos
- **Transformaciones y Feature Engineering**: Realizamos procesos de limpieza y transformación de datos, así como ingeniería de características para preparar los datos para el modelo de ML.

### 2. Desarrollo de la API
- Utilizamos el framework FastAPI para crear una API que exponga los datos de la empresa.

#### Consultas Disponibles en la API
- `def genero(anio)`: Consulta para obtener información sobre videojuegos por género en un año específico.
- `def juegos(anio)`: Consulta para obtener una lista de juegos disponibles en un año específico.
- `def specs(anio)`: Consulta para obtener especificaciones técnicas de videojuegos en un año específico.
- `def earlyaccess(anio)`: Consulta para obtener información sobre videojuegos en acceso temprano en un año específico.
- `def sentiment(anio)`: Consulta para obtener análisis de sentimiento de reseñas de videojuegos en un año específico.
- `def metascore(anio)`: Consulta para obtener puntuaciones de Metacritic de videojuegos en un año específico.

### 3. Deployment
- Implementamos las funciones de consulta en una plataforma como Render para que sean accesibles en línea.

### 4. Análisis Exploratorio de Datos (EDA)
- Realizamos un análisis en profundidad de los datos para comprender las relaciones entre las variables y detectar posibles outliers o patrones interesantes.

### 5. Modelo de Aprendizaje Automático
- Desarrollamos un modelo de aprendizaje automático para construir el sistema de recomendación.
- Función `def prediccion(año, metascore)`: Predice recomendaciones de videojuegos en función del año y la puntuación de Metacritic.

## Pasos para Ejecutar el Proyecto en Visual Studio Code

1. Clona este repositorio en tu entorno local.
2. Abre Visual Studio Code.
3. Instala las dependencias necesarias utilizando `pip install -r requirements.txt`.
4. Ejecuta el servidor de la API utilizando `uvicorn main:app --host 0.0.0.0 --port 8000`.
5. Accede a las consultas disponibles a través de `http://localhost:8000`.

## Conclusiones

Este proyecto demuestra cómo abordar un proyecto completo de MLOps, desde la ingeniería de datos hasta la implementación de un modelo de aprendizaje automático y la exposición de datos a través de una API. La automatización de procesos y la implementación de buenas prácticas son esenciales para lograr un sistema efectivo y eficiente.

¡Espero que esta guía sea útil para comprender y ejecutar el proyecto! Si tienes alguna pregunta o comentario, no dudes en contactarme.

## Deploy en Render del proyecto:
https://mlops-carlos-diaz-colodrero.onrender.com/docs

## Video del proyecto:
https://youtu.be/ilYEx2NHkUc

## Medios de contacto e informacion del autor del proyecto:
### - GitHub: https://github.com/Carlit0sCDC
### - Linkedin: www.linkedin.com/in/carlos-diaz-colodrero-12a4261a1
### - Correo/Hotmail: carlosdc2019@hotmail.com

### Disclaimer:
La finalidad de este proyecto es meramente educacional y no propone ni induce la presentacion de un proyecto real relacionado o planteado con las autoridades oficiales de Steam. Todos los analisis hechos fueron a modo de evaluacion del nivel tecnico del autor de dicho proyecto.
