# Vision Computacional para Humanidades Digitales

Proyecto de **exploración visual-semántica de colecciones de arte**
utilizando modelos de visión por computador y búsqueda vectorial.

El sistema permite analizar grandes conjuntos de imágenes de obras de
arte mediante **embeddings generados por CLIP** y **búsqueda de
similitud con FAISS**, facilitando nuevas formas de investigación en
**historia del arte y humanidades digitales**.

El proyecto combina:

-   análisis visual automatizado
-   búsqueda semántica por lenguaje natural
-   exploración interactiva del espacio visual de las imágenes

Esto permite identificar relaciones entre obras que no son fácilmente
detectables mediante metadatos tradicionales.

------------------------------------------------------------------------

# Estructura del repositorio

    Vision_computacionalHD/
    │
    ├── analisis_de_imagen.ipynb
    │   Notebook de análisis exploratorio.
    │   Permite generar embeddings visuales con CLIP,
    │   explorar similitudes y realizar pruebas experimentales.
    │
    ├── app.py
    │   Aplicación Streamlit para búsqueda visual-semántica.
    │   Permite consultar un índice FAISS y visualizar resultados.
    │
    └── README.md

------------------------------------------------------------------------

# Tecnologías utilizadas

El proyecto utiliza herramientas de **IA abierta y reproducible**:

-   Python
-   OpenAI CLIP
-   FAISS (Facebook AI Similarity Search)
-   Streamlit
-   PyTorch
-   Transformers
-   MySQL
-   Pandas
-   NumPy
-   Plotly
-   Pillow

------------------------------------------------------------------------

# Funcionamiento general

El sistema sigue el siguiente flujo:

### 1. Ingesta de obras

Las obras se almacenan en una base de datos MySQL con campos como:

-   título
-   artista
-   técnica
-   fecha
-   colección
-   URL de la imagen

------------------------------------------------------------------------

### 2. Generación de embeddings

Cada imagen es procesada con el modelo **CLIP**, generando un vector
numérico (embedding) que representa su contenido visual.

Estos vectores capturan información sobre:

-   composición
-   objetos
-   estilos
-   contextos visuales

------------------------------------------------------------------------

### 3. Indexación con FAISS

Los embeddings se almacenan en un **índice vectorial FAISS**, que
permite realizar búsquedas extremadamente rápidas incluso en colecciones
de millones de imágenes.

------------------------------------------------------------------------

### 4. Exploración interactiva

La aplicación **Streamlit (`app.py`)** permite:

-   búsqueda **texto → imágenes**
-   búsqueda **imagen → imágenes**
-   exploración por **metadatos**
-   visualización del **espacio visual (PCA 2D)**

------------------------------------------------------------------------

# Aplicaciones en historia del arte

Este enfoque permite analizar:

-   relaciones iconográficas
-   recurrencias formales
-   patrones compositivos
-   proximidades estilísticas
-   asociaciones semánticas mediante lenguaje natural

El sistema permite explorar colecciones desde una perspectiva **visual y
semántica simultáneamente**.

------------------------------------------------------------------------

# Ejemplo de consulta

Un investigador puede buscar:

    three men sitting in a park, modernist painting, flat colors

El sistema recuperará imágenes visualmente relacionadas, incluso si los
metadatos no contienen esa descripción.

------------------------------------------------------------------------

# Instalación

Crear entorno virtual:

    python -m venv .venv

Activar entorno:

Linux / Mac:

    source .venv/bin/activate

Windows:

    .venv\Scripts\activate

Instalar dependencias:

    pip install streamlit torch transformers faiss-cpu mysql-connector-python plotly pandas numpy pillow

------------------------------------------------------------------------

# Ejecutar la aplicación

    streamlit run app.py

La aplicación abrirá una interfaz interactiva para explorar el índice
visual. 
- Nota: Es necesaria conexion a Base de datos, es posible adaptar el proyecto a un csv, como data_img.csv

------------------------------------------------------------------------

# Ejemplo de datasets utilizados

El proyecto puede integrarse con colecciones de museos y archivos como:

-   Art Institute of Chicago
-   Museo Thyssen-Bornemisza
-   Colección BADAC (Beatriz González)
-   catálogos de museos abiertos

------------------------------------------------------------------------

# Objetivo del proyecto

Explorar cómo las técnicas de **visión computacional y búsqueda
vectorial** pueden ampliar las metodologías de investigación en historia
del arte, permitiendo nuevas formas de análisis visual a gran escala.

------------------------------------------------------------------------

# Autor

**Iván Barajas Hurtado**

Proyecto en desarrollo dentro del campo de **Humanidades Digitales y
análisis visual computacional**.
