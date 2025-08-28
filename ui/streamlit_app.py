#!/usr/bin/env python3
"""
BoeFacil - Interfaz Streamlit
============================

Aplicación web para búsqueda semántica en el BOE (Boletín Oficial del Estado).
Utiliza la API BOESearchAPI para proporcionar una interfaz intuitiva y completa.

Autor: BoeFacil Team
Fecha: Agosto 2025
"""

import streamlit as st
import sys
from pathlib import Path
import logging
from typing import Optional

# Configurar el path para importar la API BOE
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

try:
    from lib.boe_search_api import BOESearchAPI
except ImportError as e:
    st.error(f"Error al importar BOESearchAPI: {e}")
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
        total_docs = stats.get('total_documents', 'N/A')
        
        # Mostrar métricas básicas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📄 Documentos Indexados", 
                value=f"{total_docs:,}" if isinstance(total_docs, int) else total_docs
            )
        
        with col2:
            st.metric(
                label="🤖 Modelo Embeddings", 
                value="BGE-M3"
            )
            
        with col3:
            st.metric(
                label="✅ Estado", 
                value="Listo"
            )
            
    except Exception as e:
        st.warning(f"⚠️ No se pudieron cargar las estadísticas: {e}")
    
    # Interfaz de búsqueda básica
    st.markdown("### 🔍 Búsqueda Semántica")
    
    # Barra de búsqueda principal
    query = st.text_input(
        "",
        placeholder="Ej: Nombramiento de Santos Cerdán...",
        help="Introduce cualquier consulta en lenguaje natural. El sistema entiende conceptos, fechas, ministerios y más."
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
        with st.spinner(f"🔍 Buscando '{query}'..."):
            try:
                results = api.search(query.strip(), limit=num_results)
                
                if results:
                    st.success(f"✅ Se encontraron {len(results)} resultados")
                    
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
                    st.warning("🔍 No se encontraron resultados para tu búsqueda. Intenta con términos diferentes.")
                    
            except Exception as e:
                st.error(f"❌ Error durante la búsqueda: {e}")
                logger.error(f"Error en búsqueda: {e}", exc_info=True)
    
    elif search_button and not query.strip():
        st.warning("⚠️ Por favor, introduce una consulta para buscar.")
    
    # Información adicional en el sidebar
    with st.sidebar:
        st.markdown("### ℹ️ Información")
        
        st.markdown("""
        **¿Cómo usar el buscador?**
        
        1. 🔍 **Búsqueda libre:** Escribe cualquier consulta en lenguaje natural
        2. 📊 **Ajusta resultados:** Selecciona cuántos resultados ver
        3. 📄 **Explora:** Haz clic en los resultados para ver detalles
        
        **Ejemplos de búsquedas:**
        - "Real decreto sobre impuestos"
        - "ministerio hacienda presupuesto"
        - "BOE diciembre 2023"
        - "disposiciones generales"
        """)
        
        st.markdown("---")
        
        st.markdown("""
        **🔧 Próximamente:**
        - Filtros avanzados por fecha
        - Filtros por ministerio
        - Dashboard de estadísticas
        - Búsqueda de documentos similares
        """)

if __name__ == "__main__":
    main()
