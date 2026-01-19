# BoeFácil 🔍

> **Sistema de búsqueda semántica del Boletín Oficial del Estado (BOE)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

BoeFácil es un sistema avanzado de búsqueda semántica para el Boletín Oficial del Estado español. Combina técnicas de procesamiento de lenguaje natural (NLP), embeddings de alta dimensionalidad y búsqueda vectorial para facilitar el acceso y la exploración de documentos oficiales publicados en el BOE.

## 🎯 Características Principales

- **🔍 Búsqueda Semántica Avanzada**: Utiliza el modelo BGE-M3 para comprender el significado de las consultas
- **⚡ Alto Rendimiento**: Indexación con FAISS para búsquedas ultrarrápidas en millones de documentos
- **🎨 Interfaz Web Intuitiva**: Aplicación Streamlit para búsquedas interactivas
- **📊 Filtros Avanzados**: Filtrado por fecha, ministerio, sección, rango, departamento y más
- **📈 Sistema de Evaluación**: Framework completo para comparar métodos de búsqueda (BGE-M3 vs TF-IDF)
- **🔄 Pipeline Automatizado**: Proceso completo desde la descarga hasta la indexación
- **📝 Conversión Inteligente**: HTML → Markdown con extracción de tablas
- **🧩 Chunking Optimizado**: División inteligente de documentos para mejor precisión

## 📋 Tabla de Contenidos

- [Instalación](#-instalación)
- [Inicio Rápido](#-inicio-rápido)
- [Arquitectura](#-arquitectura)
- [Pipeline de Procesamiento](#-pipeline-de-procesamiento)
- [Uso](#-uso)
- [Evaluación y Benchmarks](#-evaluación-y-benchmarks)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 8GB+ RAM recomendado para procesar grandes volúmenes
- (Opcional) GPU para generación más rápida de embeddings

### Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/jbarrerobuch/BoeFacil.git
cd BoeFacil
```

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

### Dependencias Principales

- **pandas**: Manipulación y análisis de datos
- **scikit-learn**: Machine Learning y métricas
- **sentence-transformers**: Generación de embeddings (BGE-M3)
- **faiss-cpu**: Búsqueda vectorial eficiente
- **streamlit**: Interfaz web interactiva
- **beautifulsoup4**: Parsing HTML
- **markdownify**: Conversión HTML a Markdown

## ⚡ Inicio Rápido

### 1. Construir el Índice de Búsqueda

```python
from src.lib.index_builder import build_index_from_parquets

# Construir índice desde archivos parquet
db = build_index_from_parquets(
    parquet_files=["samples/parquet/boe_data_20231229.parquet"],
    output_index_path="indices/boe_index.faiss",
    output_metadata_path="indices/metadata.json",
    index_type="IVF"
)

print(f"Índice construido con {db.get_stats()['total_vectors']} vectores")
```

### 2. Realizar Búsquedas

```python
from src.lib.boe_search_api import BOESearchAPI

# Inicializar API de búsqueda
api = BOESearchAPI(
    index_path="indices/boe_index.faiss",
    metadata_path="indices/metadata.json"
)

# Búsqueda simple
results = api.search("impuesto sobre sociedades", limit=5)

for result in results:
    print(f"Título: {result['item_titulo']}")
    print(f"Fecha: {result['fecha_publicacion']}")
    print(f"Similitud: {result['similarity_score']:.3f}\n")
```

### 3. Lanzar la Interfaz Web

```bash
cd src/ui
streamlit run streamlit_app.py
```

Accede a la aplicación en: `http://localhost:8501`

## 🏗️ Arquitectura

BoeFácil está diseñado con una arquitectura modular que separa responsabilidades:

```
┌─────────────────────────────────────────────────────────┐
│                    INTERFAZ USUARIO                      │
│              (Streamlit UI / Python API)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  CAPA DE BÚSQUEDA                        │
│         BOESearchAPI + AdvancedFilter                    │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│               BASE DE DATOS VECTORIAL                    │
│         FAISS Index + Metadata (JSON/Parquet)            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│              PROCESAMIENTO DE DATOS                      │
│    Descarga → Parsing → Chunking → Embeddings           │
└─────────────────────────────────────────────────────────┘
```

### Componentes Principales

- **`src/lib/boe.py`**: Cliente para la API oficial del BOE
- **`src/lib/vector_db.py`**: Gestión de índices FAISS y metadatos
- **`src/lib/boe_search_api.py`**: API de búsqueda con filtros
- **`src/lib/chunk_utils.py`**: Estrategias de chunking inteligente
- **`src/lib/index_builder.py`**: Construcción y actualización de índices
- **`src/lib/advanced_filter.py`**: Sistema de filtros avanzados
- **`src/ui/`**: Interfaz web Streamlit

## 🔄 Pipeline de Procesamiento

El pipeline completo transforma datos crudos del BOE en un índice de búsqueda optimizado:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   1. BOE    │────▶│  2. HTML→   │────▶│ 3. Chunking │────▶│4. Embeddings│
│  Descarga   │     │  Markdown   │     │             │     │   (BGE-M3)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                     │
                                                                     ▼
                                                          ┌─────────────────┐
                                                          │  5. Indexación  │
                                                          │  FAISS + Meta   │
                                                          └─────────────────┘
```

### Ejecutar Pipeline Completo

```bash
python src/pipeline_completo.py \
    --fecha-inicio 2023-12-01 \
    --fecha-fin 2023-12-31 \
    --output-dir samples
```

Pasos del pipeline:

1. **Descarga**: Obtiene sumarios y documentos completos del BOE
2. **Aplanado**: Estructura los datos en formato tabular (Parquet)
3. **Conversión**: HTML → Markdown + extracción de tablas
4. **Chunking**: División inteligente en fragmentos de 512 tokens
5. **Embeddings**: Generación con modelo BGE-M3
6. **Indexación**: Construcción del índice FAISS

## 📚 Uso

### API de Python

#### Búsqueda Básica

```python
from src.lib.boe_search_api import BOESearchAPI

api = BOESearchAPI("indices/boe_index.faiss", "indices/metadata.json")

# Búsqueda simple
results = api.search("normativa educación universitaria", limit=10)
```

#### Búsqueda con Filtros

```python
# Filtrar por fecha
results = api.search(
    query="subvenciones agricultura",
    limit=10,
    filters={
        'fecha_publicacion': '20231229',  # Formato YYYYMMDD
    }
)

# Filtrar por múltiples criterios
results = api.search(
    query="política monetaria",
    limit=10,
    filters={
        'fecha_desde': '20230101',
        'fecha_hasta': '20231231',
        'ministerio': 'Ministerio de Economía',
        'seccion': 'I'
    }
)
```

#### Búsqueda Avanzada

```python
from src.lib.advanced_filter import AdvancedFilter

# Crear filtros complejos
af = AdvancedFilter()
af.add_date_range('20230101', '20231231')
af.add_ministry_filter('Ministerio de Hacienda')
af.add_section_filter(['I', 'II'])

# Aplicar filtros
results = api.search("impuestos", limit=10)
filtered_results = af.apply(results)
```

### Interfaz de Línea de Comandos

```bash
# Demo interactivo
python scripts/search_demo.py

# Búsqueda directa
python scripts/search_demo.py --query "ley orgánica educación" --limit 5

# Ejemplo de filtros
python scripts/filter_demo.py
```

### Actualización Diaria del Índice

```python
from src.lib.index_builder import add_daily_parquet_to_index

# Agregar nuevos datos al índice existente
db = add_daily_parquet_to_index(
    parquet_path="samples/parquet/boe_data_20240101.parquet",
    index_path="indices/boe_index.faiss",
    metadata_path="indices/metadata.json"
)
```

## 📊 Evaluación y Benchmarks

BoeFácil incluye un framework completo para evaluar y comparar diferentes métodos de búsqueda.

### Resultados del Experimento Comparativo

Se realizó una evaluación rigurosa comparando **BGE-M3** (embeddings semánticos) vs **TF-IDF** (baseline clásico):

| Métrica | BGE-M3 | TF-IDF | Ganador |
|---------|--------|--------|---------|
| **Precision@5** | 0.130 ± 0.239 | **0.230 ± 0.348** | TF-IDF |
| **MAP** | 0.060 ± 0.129 | **0.142 ± 0.276** | TF-IDF |
| **Tiempo promedio** | **0.521s** | 4.991s | BGE-M3 |
| **Eficiencia*** | **0.2497** | 0.0461 | BGE-M3 |

*Eficiencia = Precisión / Tiempo (mayor es mejor)

**Hallazgos clave**:
- ⚡ **BGE-M3 es 9.6x más rápido** que TF-IDF
- 📊 **TF-IDF tiene mejor precisión** en búsquedas de entidades específicas
- 🎯 **BGE-M3 sobresale** en consultas conceptuales legales
- 💡 **Recomendación**: Sistema híbrido que combine ambos métodos

### Ejecutar Benchmarks

```bash
# Evaluación completa
python scripts/evaluation_benchmark.py

# Análisis de resultados
python scripts/analyze_results.py

# Análisis detallado por categorías
python scripts/detailed_analysis.py
```

Los resultados se guardan en `evaluation_results/`:
- `summary_metrics.json`: Métricas agregadas
- `detailed_results.csv`: Resultados por query
- `analisis_academico.md`: Análisis académico completo
- `visualizations/`: Gráficos comparativos

## 📁 Estructura del Proyecto

```
BoeFacil/
├── src/
│   ├── lib/                      # Biblioteca principal
│   │   ├── boe.py               # Cliente API BOE
│   │   ├── vector_db.py         # Gestión FAISS
│   │   ├── boe_search_api.py    # API de búsqueda
│   │   ├── chunk_utils.py       # Chunking inteligente
│   │   ├── index_builder.py     # Construcción índices
│   │   ├── advanced_filter.py   # Filtros avanzados
│   │   ├── utils_markdown.py    # HTML→Markdown
│   │   └── ...
│   ├── ui/                       # Interfaz Streamlit
│   │   ├── streamlit_app.py     # App principal
│   │   ├── components/          # Componentes UI
│   │   ├── pages/               # Páginas múltiples
│   │   └── utils/               # Utilidades UI
│   ├── pipeline_completo.py     # Pipeline end-to-end
│   └── ...
├── scripts/                      # Scripts auxiliares
│   ├── example_usage.py         # Ejemplos de uso
│   ├── search_demo.py           # Demo búsqueda
│   ├── evaluation_benchmark.py  # Evaluación
│   └── ...
├── test/                         # Tests unitarios
├── config/                       # Configuraciones
│   └── vector_db_config.json
├── evaluation_results/           # Resultados evaluación
├── requirements.txt             # Dependencias Python
├── LICENSE                      # Licencia MIT
└── README.md                    # Este archivo
```

## 🧪 Tests

Ejecutar tests unitarios:

```bash
# Todos los tests
python -m pytest test/

# Test específico
python -m pytest test/test_chunk_utils.py -v

# Con cobertura
python -m pytest test/ --cov=src --cov-report=html
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Si quieres mejorar BoeFácil:

1. **Fork** el repositorio
2. Crea una **rama** para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. Abre un **Pull Request**

### Guías de Contribución

- Sigue el estilo de código existente (PEP 8 para Python)
- Añade tests para nuevas funcionalidades
- Actualiza la documentación según sea necesario
- Asegúrate de que todos los tests pasen antes de enviar el PR

## 📖 Documentación Adicional

- **Pipeline Completo**: Ver `src/pipeline_completo.py` (documentación inline)
- **Análisis Académico**: `evaluation_results/analisis_academico.md`
- **Ejemplos de Uso**: `scripts/example_usage.py`
- **Demo de Filtros**: `scripts/filter_demo.py`

## 🎓 Uso Académico

Este proyecto fue desarrollado como parte de un Trabajo de Fin de Máster sobre búsqueda semántica en documentos legales. Incluye:

- Framework de evaluación riguroso
- Comparación de métodos (BGE-M3 vs TF-IDF)
- Análisis estadístico completo
- Visualizaciones y métricas académicas

Si utilizas este proyecto en investigación académica, por favor considera citarlo.

## 🔒 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License - Copyright (c) 2025 Pepe B
```

## 👤 Autor

**jbarrero**

- GitHub: [@jbarrerobuch](https://github.com/jbarrerobuch)

## 🙏 Agradecimientos

- **BOE.es**: Por proporcionar acceso público a los datos oficiales
- **Sentence Transformers**: Por los modelos de embeddings BGE-M3
- **FAISS**: Por la biblioteca de búsqueda vectorial eficiente
- **Streamlit**: Por facilitar la creación de interfaces web interactivas

## 📞 Soporte

Si encuentras algún problema o tienes preguntas:

- 🐛 Reporta bugs en [GitHub Issues](https://github.com/jbarrerobuch/BoeFacil/issues)
- 💬 Únete a las [Discusiones](https://github.com/jbarrerobuch/BoeFacil/discussions)
- 📧 Contacta al autor a través de GitHub

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**