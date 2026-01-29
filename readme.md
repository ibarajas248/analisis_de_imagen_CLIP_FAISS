
# Proyecto: Análisis de Imagen y Búsqueda Visual
## CLIP + FAISS aplicado a Humanidades Digitales

### Autor
Iván Barajas Hurtado

---

## 1. Descripción general

Este proyecto implementa un **sistema de análisis y exploración visual** basado en *embeddings semánticos* obtenidos con **CLIP (OpenAI)** y búsqueda eficiente mediante **FAISS**.

El objetivo principal es permitir:
- Búsqueda de imágenes **por texto** (lenguaje natural)
- Búsqueda de imágenes **por similitud visual**
- Exploración de archivos culturales sin depender exclusivamente de metadatos tradicionales

El proyecto se enmarca en prácticas de **Humanidades Digitales**, curaduría algorítmica y análisis visual asistido por IA.

---

## 2. Arquitectura del proyecto

```
humanidades_digitales/
│
├── analisis_de_imagen/
│   ├── analisis_de_imagen.ipynb
│   ├── analisis_de_imagen_documentado.ipynb
│   ├── metadata.csv / parquet
│   ├── index.faiss
│
├── streamlit_app/
│   └── app.py
│
└── README.md (este documento)
```

---

## 3. Componentes principales

### 3.1 Notebook `analisis_de_imagen.ipynb`

Funciones principales:
- Carga y limpieza de metadata visual
- Generación de embeddings CLIP (imagen y texto)
- Normalización de vectores
- Construcción y persistencia de índice FAISS
- Búsqueda semántica y visual
- Visualización de resultados

Versión documentada:
- `analisis_de_imagen_documentado.ipynb`
- Incluye Markdown explicativo antes de cada paso
- No altera el código original

---

### 3.2 Modelo de embeddings (CLIP)

- Modelo: `openai/clip-vit-base-patch32` (configurable)
- Espacio vectorial compartido texto-imagen
- Dimensión: 512
- Normalización L2 para similitud por producto interno (Inner Product)

Esto permite comparar:
- texto ↔ imagen
- imagen ↔ imagen

---

### 3.3 Índice FAISS

- Tipo recomendado: `IndexFlatIP`
- Métrica: similitud coseno (vía normalización)
- Persistencia en disco (`.faiss`)
- IDs alineados con columna `id` de la metadata

Ventajas:
- Escalabilidad
- Velocidad de consulta
- Separación entre embeddings y metadata

---

### 3.4 Aplicación Streamlit

Archivo: `app.py`

Funcionalidades:
- Interfaz web interactiva
- Búsqueda por texto
- Búsqueda por imagen (upload o ruta local)
- Visualización tabular y visual
- Configuración dinámica (modelo, top-k, rutas)

Ejecución:
```bash
streamlit run app.py
```

---

## 4. Dependencias

```
streamlit
pandas
numpy
faiss-cpu
torch
transformers
pillow
```

---

## 5. Enfoque conceptual

Este proyecto propone una lectura alternativa del archivo visual:

- El archivo no se ordena por autor o cronología
- Se navega por **resonancia visual y semántica**
- Se habilita una curaduría algorítmica no jerárquica

Esto lo posiciona como:
- Herramienta de investigación
- Dispositivo curatorial
- Base para instalaciones interactivas
- Prototipo de archivo vivo

---

## 6. Posibles extensiones

- Integración con base de datos SQL
- API REST para consultas externas
- FAISS GPU
- Interfaces museográficas
- Anotación colaborativa
- Exportación de recorridos visuales

---

## 7. Licencia y uso

Proyecto de investigación y creación.
Uso académico, experimental y artístico.

---

> “No se busca clasificar imágenes, sino pensar con ellas.”
