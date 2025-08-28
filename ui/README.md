# BoeFacil - Interfaz Streamlit 🔍

Interfaz web intuitiva para el buscador semántico del BOE (Boletín Oficial del Estado).

## 🚀 Inicio Rápido

### Activar entorno y lanzar aplicación:

1. Navega al directorio del proyecto
2. Activa entorno
3. Lanza la aplicación: streamlit run ui\streamlit_app.py

### O usar el launcher automático:
```bash
python run_ui.py
```

## 📋 Funcionalidades Actuales (Fase 1)

### ✅ Implementadas:
- **🔍 Búsqueda semántica básica**: Consultas en lenguaje natural
- **📊 Métricas del sistema**: Total de documentos, modelo usado, estado
- **📄 Visualización de resultados**: Cards expandibles con metadatos completos
- **🎯 Scoring visual**: Puntuación de similitud con colores
- **📝 Preview de contenido**: Texto truncado con opción de expandir
- **⚙️ Controles básicos**: Número de resultados ajustable
- **🎨 Diseño profesional**: Tema BOE con CSS personalizado

### 🔧 Próximamente (Siguientes Fases):
- **📅 Filtros por fecha**: Selector de rangos temporales
- **🏛️ Filtros por ministerio**: Autocompletado de departamentos
- **📂 Filtros por sección**: Selección múltiple de secciones BOE
- **📊 Dashboard estadísticas**: Visualizaciones interactivas
- **🔗 Documentos similares**: Búsqueda de contenido relacionado
- **📤 Export de resultados**: Descarga en CSV/JSON
- **📚 Historial**: Búsquedas recientes y favoritos

## 🏗️ Arquitectura

```
ui/
├── streamlit_app.py          # 🏠 Aplicación principal
├── components/               # 🧩 Componentes reutilizables (próximamente)
│   ├── search_interface.py   # 🔍 Interfaz de búsqueda
│   ├── filters.py           # 🎛️ Panel de filtros
│   ├── results_display.py   # 📄 Visualización de resultados
│   └── stats_dashboard.py   # 📊 Dashboard estadísticas
├── utils/
│   └── ui_helpers.py        # 🛠️ Utilidades auxiliares
└── config/
    └── streamlit_config.toml # ⚙️ Configuración Streamlit
```

## 🎨 Características de Diseño

### Tema BOE Institucional:
- **🎨 Colores**: Azul institucional (#1f4e79) como color primario
- **📱 Responsive**: Adaptado para desktop y móvil
- **🃏 Cards**: Resultados en tarjetas con sombras
- **📊 Métricas**: Indicadores visuales destacados
- **🎯 UX**: Interfaz intuitiva y profesional

### Componentes Visuales:
- **Header gradient**: Título con degradado azul
- **Botones hover**: Efectos de elevación
- **Cards con sombra**: Resultados estructurados
- **Métricas destacadas**: Stats del sistema
- **Sidebar organizado**: Información y próximas funciones

## 🔧 Configuración

### Variables de Entorno:
- **Índice FAISS**: `indices/boe_index.faiss`
- **Metadatos**: `indices/metadata.json`
- **Modelo**: `pablosi/bge-m3-trained-2`

### Streamlit Config:
- **Puerto**: 8501 (por defecto)
- **Tema**: Personalizado BOE
- **Cache**: Habilitado para performance
- **CORS**: Deshabilitado para desarrollo

## 📊 Performance

### Optimizaciones Actuales:
- **@st.cache_resource**: API BOE cacheada para evitar reinicios
- **Lazy loading**: Resultados se cargan bajo demanda
- **Text truncation**: Preview eficiente de contenido largo

### Métricas Esperadas:
- **Tiempo carga inicial**: ~3-5 segundos
- **Tiempo búsqueda**: ~1-2 segundos
- **Memoria**: ~200-500MB (según tamaño índice)

## 🐛 Troubleshooting

### Errores Comunes:

#### ❌ "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit plotly
```

#### ❌ "No se encontró el índice FAISS"
- Verificar que existe `indices/boe_index.faiss`
- Ejecutar `scripts/build_vector_index.py` si es necesario

#### ❌ "Error al inicializar la API BOE"
- Verificar paths en `streamlit_app.py`
- Comprobar permisos de archivos
- Revisar logs en terminal

### Logs:
Los logs se muestran en la terminal donde ejecutas Streamlit.

## 🚀 Desarrollo

### Añadir nuevos componentes:
1. Crear archivo en `ui/components/`
2. Importar en `streamlit_app.py`
3. Integrar en la función `main()`

### Testing local:
```bash
# Verificar sintaxis
python -m py_compile ui/streamlit_app.py

# Lanzar en modo desarrollo
streamlit run ui/streamlit_app.py --logger.level debug
```

---

**🎯 Status: Fase 1 Completada ✅**

La interfaz básica está funcionando correctamente. Las siguientes fases añadirán filtros avanzados, dashboard de estadísticas y funcionalidades premium.
