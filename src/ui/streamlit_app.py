import streamlit as st
import sys
from pathlib import Path
import logging
from typing import Optional

# Configurar el path para importar la API BOE
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Desde src/ui/ subir a raíz del proyecto

# Agregar src al path para imports absolutos
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Importar componentes de UI
try:
    from ui.components.filters import BOEFilters
except ImportError as e:
    st.error(f"Error al importar componentes UI: {e}")
    st.stop()

try:
    from lib.boe_search_api import BOESearchAPI
except ImportError as e:
    st.error(f"Error al importar BOESearchAPI: {e}")
    st.error(f"PROJECT_ROOT: {PROJECT_ROOT}")
    st.stop()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="BoeFacil - Buscador BOE",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/jbarrerobuch/BoeFacil',
        'Report a bug': 'https://github.com/jbarrerobuch/BoeFacil/issues',
        'About': """
        # BoeFacil 🔍
        
        Buscador semántico del BOE (Boletín Oficial del Estado).
        
        **Funcionalidades:**
        - Búsqueda semántica avanzada
        - Filtros por fecha, ministerio y sección
        - Visualización de resultados detallada
        - Estadísticas del índice BOE
        
        Desarrollado con ❤️ usando Streamlit y FAISS por JBarrero 😎.
        """
    }
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    /* Tema principal BOE */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Título principal */
    .title-container {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f4e79, #2d5aa0);
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Cards de resultados */
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Métricas */
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79, #2d5aa0);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2d5aa0, #1f4e79);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_api():
    """
    Inicializa la API BOE con cache para evitar recargas.
    
    Returns:
        BOESearchAPI: Instancia inicializada de la API
    """
    try:
        # Rutas de los archivos de índice
        index_path = PROJECT_ROOT / "indices" / "boe_index.faiss"
        metadata_path = PROJECT_ROOT / "indices" / "metadata.json"
        
        # Verificar que los archivos existen
        if not index_path.exists():
            st.error(f"❌ No se encontró el índice FAISS en: {index_path}")
            st.stop()
            
        if not metadata_path.exists():
            st.error(f"❌ No se encontró el archivo de metadatos en: {metadata_path}")
            st.stop()
        
        # Inicializar API
        api = BOESearchAPI(
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            model_name="pablosi/bge-m3-trained-2"
        )
        
        logger.info("✅ API BOE inicializada correctamente")
        return api
        
    except Exception as e:
        st.error(f"❌ Error al inicializar la API BOE: {e}")
        st.stop()

def main():
    """Función principal de la aplicación Streamlit."""
    
    # Inicializar sistema de filtros
    filters = BOEFilters()
    
    # Título principal con estilo
    st.markdown("""
    <div class="title-container">
        <h1>🔍 BoeFacil</h1>
        <p style="margin: 0; font-size: 1.2em; opacity: 0.9;">
            Buscador Semántico del BOE
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar API
    with st.spinner("🚀 Inicializando motor de búsqueda BOE..."):
        api = initialize_api()
    
    # Obtener estadísticas básicas
    try:
        stats = api.get_stats()
        # Obtener el número de documentos desde las estadísticas del índice
        index_stats = stats.get('index_stats', {})
        total_docs = index_stats.get('total_vectors', stats.get('total_searchable_documents', 'N/A'))
        unique_boes = stats.get('unique_boes', 'N/A')
        date_range = stats.get('date_range', 'N/A')
        
        # Mostrar métricas básicas
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📄 Chunks Indexados", 
                value=f"{total_docs:,}" if isinstance(total_docs, int) else total_docs
            )
        
        with col2:
            st.metric(
                label="📰 BOEs Indexados", 
                value=f"{unique_boes:,}" if isinstance(unique_boes, int) else unique_boes
            )
        
        with col3:
            st.metric(
                label="📅 Rango de Años", 
                value=date_range
            )
        
        with col4:
            st.metric(
                label="🤖 Modelo Embeddings", 
                value="BGE-M3"
            )
            
        with col5:
            st.metric(
                label="✅ Estado", 
                value="Listo"
            )
            
    except Exception as e:
        st.warning(f"⚠️ No se pudieron cargar las estadísticas: {e}")
    
    # Manejar sugerencias de búsqueda
    if 'search_suggestion' in st.session_state:
        suggestion = st.session_state['search_suggestion']
        del st.session_state['search_suggestion']
        st.info(f"💡 Búsqueda sugerida: **{suggestion}**")
        # Auto-rellenar el campo de búsqueda (esto se verá en la siguiente recarga)
        
    # Interfaz de búsqueda básica
    st.markdown("### 🔍 Búsqueda Semántica")
    
    # Usar sugerencia si existe
    default_query = ""
    if 'search_suggestion' in st.session_state:
        default_query = st.session_state.get('search_suggestion', "")
    
    # Barra de búsqueda principal
    query = st.text_input(
        "Consulta de búsqueda:",
        value=default_query,
        placeholder="Ej: Real decreto sobre impuestos, ministerio hacienda presupuesto...",
        help="Introduce cualquier consulta en lenguaje natural. El sistema entiende conceptos, fechas, ministerios y más.",
        label_visibility="collapsed"
    )
    
    # Controles básicos
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_button = st.button("🔍 Buscar", type="primary", use_container_width=True)
    
    with col2:
        num_results = st.selectbox(
            "Resultados:",
            options=[5, 10, 20, 30, 50],
            index=0,  # Por defecto 5
            help="Número máximo de resultados a mostrar"
        )
    
    # Realizar búsqueda si hay consulta
    if search_button and query.strip():
        # Obtener parámetros de filtros
        filter_params = filters.get_filter_parameters()
        num_active_filters = len(filter_params)
        
        # Mensaje de búsqueda dinámico
        if num_active_filters > 0:
            search_message = f"🎯 Buscando '{query}' con {num_active_filters} filtro(s)..."
        else:
            search_message = f"🔍 Buscando '{query}' en toda la base de datos..."
        
        with st.spinner(search_message):
            try:
                # Mostrar información de filtros aplicados
                if num_active_filters > 0:
                    filter_info = []
                    if 'start_date' in filter_params or 'end_date' in filter_params:
                        start = filter_params.get('start_date', '...')
                        end = filter_params.get('end_date', '...')
                        filter_info.append(f"📅 {start} → {end}")
                    if 'ministry' in filter_params:
                        filter_info.append(f"🏛️ {filter_params['ministry']}")
                    if 'section' in filter_params:
                        filter_info.append(f"📂 Sección {filter_params['section']}")
                    if 'min_tokens' in filter_params or 'max_tokens' in filter_params:
                        min_tok = filter_params.get('min_tokens', 10)
                        max_tok = filter_params.get('max_tokens', 10000)
                        filter_info.append(f"📊 {min_tok}-{max_tok} tokens")
                    
                    st.info(f"🎯 **Filtros aplicados**: {' • '.join(filter_info)}")
                
                # Decidir qué método de API usar según filtros activos
                if filters.has_active_filters():
                    # Usar búsqueda avanzada con filtros
                    results = api.advanced_search(
                        query=query.strip(),
                        limit=num_results,
                        **filter_params
                    )
                else:
                    # Usar búsqueda simple
                    results = api.search(query.strip(), limit=num_results)
                
                if results:
                    # Mensaje de éxito con más información
                    if num_active_filters > 0:
                        st.success(f"✅ **{len(results)} resultados encontrados** con filtros aplicados")
                    else:
                        st.success(f"✅ **{len(results)} resultados encontrados** en búsqueda general")
                    
                    # Mostrar estadísticas de relevancia
                    if results:
                        scores = [r.get('similarity_score', 0) for r in results]
                        avg_score = sum(scores) / len(scores)
                        max_score = max(scores)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎯 Relevancia Promedio", f"{avg_score:.3f}")
                        with col2:
                            st.metric("⭐ Mejor Coincidencia", f"{max_score:.3f}")
                        with col3:
                            st.metric("📄 Documentos", len(results))
                    
                    # Mostrar resultados
                    st.markdown("### 📋 Resultados")
                    
                    for i, result in enumerate(results):
                        with st.expander(
                            f"📄 Resultado {i+1} - Similitud: {result.get('similarity_score', 0):.3f}",
                            expanded=(i < 3)  # Expandir los primeros 3 resultados
                        ):
                            # Información básica
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**📅 Fecha:** {result.get('fecha_publicacion', 'N/A')}")
                                st.write(f"**🏛️ Ministerio:** {result.get('departamento_nombre', 'N/A')}")
                                st.write(f"**📂 Sección:** {result.get('seccion_nombre', 'N/A')}")
                            
                            with col2:
                                st.write(f"**🆔 ID:** {result.get('chunk_id', 'N/A')}")
                                st.write(f"**📊 Tokens:** {result.get('tokens_aproximados', 'N/A')}")
                                st.write(f"**🎯 Score:** {result.get('similarity_score', 0):.4f}")
                            
                            # Título
                            titulo = result.get('item_titulo', 'Sin título')
                            st.markdown(f"**📋 Título:** {titulo}")
                            
                            # Texto del documento
                            texto = result.get('texto', '')
                            if texto:
                                # Mostrar preview del texto
                                if len(texto) > 500:
                                    st.markdown(f"**📝 Contenido:** {texto[:500]}...")
                                    
                                    # Botón para mostrar texto completo
                                    if st.button(f"Ver texto completo", key=f"full_text_{i}"):
                                        st.text_area(
                                            "Texto completo:",
                                            value=texto,
                                            height=300,
                                            key=f"full_text_area_{i}"
                                        )
                                else:
                                    st.markdown(f"**📝 Contenido:** {texto}")
                
                else:
                    if num_active_filters > 0:
                        st.warning("🔍 **No se encontraron resultados** con los filtros aplicados.")
                        st.markdown("""
                        **💡 Sugerencias para encontrar resultados:**
                        - Intenta **ampliar el rango de fechas** si tienes filtros temporales
                        - **Quita algunos filtros** organizacionales o de contenido
                        - Usa **términos más generales** en la búsqueda
                        - Verifica la **ortografía** de los términos de búsqueda
                        """)
                        
                        # Botón rápido para limpiar filtros
                        if st.button("🗑️ Quitar todos los filtros y buscar de nuevo"):
                            filters.clear_all_filters()
                            st.rerun()
                    else:
                        st.warning("🔍 **No se encontraron resultados** para tu búsqueda.")
                        st.markdown("""
                        **💡 Sugerencias para mejorar tu búsqueda:**
                        - Intenta con **términos más generales** o **sinónimos**
                        - Revisa la **ortografía** de las palabras clave
                        - Usa **menos palabras** en la consulta
                        - Prueba **conceptos relacionados** al tema
                        
                        **Ejemplos de búsquedas exitosas:**
                        - "presupuestos generales estado"
                        - "real decreto impuestos"
                        - "ministerio hacienda"
                        """)
                        
                        # Sugerencias de búsquedas populares
                        st.markdown("**🔥 Búsquedas populares:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💰 Presupuestos", key="sugg1"):
                                st.session_state['search_suggestion'] = "presupuestos generales"
                            if st.button("📜 Decretos", key="sugg2"):
                                st.session_state['search_suggestion'] = "real decreto"
                        with col2:
                            if st.button("🏛️ Hacienda", key="sugg3"):
                                st.session_state['search_suggestion'] = "ministerio hacienda"
                            if st.button("⚖️ Justicia", key="sugg4"):
                                st.session_state['search_suggestion'] = "administración justicia"
                    
            except Exception as e:
                st.error(f"❌ Error durante la búsqueda: {e}")
                logger.error(f"Error en búsqueda: {e}", exc_info=True)
    
    elif search_button and not query.strip():
        st.warning("⚠️ Por favor, introduce una consulta para buscar.")
    
    # Panel de filtros y información en el sidebar
    with st.sidebar:
        # Navegación principal
        st.markdown("## 🧭 Navegación")
        
        # Botón para gestión de datos
        if st.button("🔧 Gestión de Datos", use_container_width=True, type="secondary"):
            st.switch_page("pages/1_Data_Management.py")
        
        st.markdown("---")
        
        st.markdown("## 🎛️ Filtros de Búsqueda")
        
        # Renderizar filtros temporales
        filters.render_temporal_filters()
        
        st.markdown("---")
        
        # Renderizar filtros organizacionales
        filters.render_organizational_filters(api)
        
        st.markdown("---")
        
        # Renderizar filtros de contenido
        filters.render_content_filters()
        
        st.markdown("---")
        
        # Resumen de filtros activos
        filters.render_filter_summary()
        
        st.markdown("---")
        
        # Información adicional
        st.markdown("### ℹ️ Información")
        
        st.markdown("""
        **¿Cómo usar BoeFacil?**
        
        1. 🔍 **Búsqueda libre:** Escribe cualquier consulta en lenguaje natural
        2. 🎛️ **Aplica filtros:** Usa los filtros del sidebar para refinar
        3. 📊 **Ajusta resultados:** Selecciona cuántos resultados ver
        4. 📄 **Explora:** Haz clic en los resultados para ver detalles
        
        **Ejemplos de búsquedas:**
        - "Real decreto sobre impuestos"
        - "ministerio hacienda presupuesto"
        - "BOE diciembre 2023"
        - "disposiciones generales"
        """)
        
        st.markdown("---")
        
        st.markdown("""
        **🆕 Funcionalidades:**
        - ✅ Filtros por fecha con presets
        - ✅ Filtros por ministerio
        - ✅ Filtros por sección BOE
        - ✅ Filtros por longitud de documento
        - 🔧 Dashboard de estadísticas (próximamente)
        - 🔧 Búsqueda de documentos similares (próximamente)
        """)

if __name__ == "__main__":
    main()
