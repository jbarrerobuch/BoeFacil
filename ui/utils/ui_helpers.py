"""
Funciones auxiliares para la interfaz Streamlit de BoeFacil.

Este módulo contiene utilidades comunes para formateo, validación
y manipulación de datos en la interfaz de usuario.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import re
from datetime import datetime, date

def format_similarity_score(score: float) -> str:
    """
    Formatea el score de similitud para mostrar en la UI.
    
    Args:
        score: Score de similitud (0.0 - 1.0)
    
    Returns:
        String formateado con color y emoji
    """
    if score >= 0.9:
        return f"🟢 {score:.3f} (Excelente)"
    elif score >= 0.8:
        return f"🟡 {score:.3f} (Muy bueno)"
    elif score >= 0.7:
        return f"🟠 {score:.3f} (Bueno)"
    elif score >= 0.6:
        return f"🔴 {score:.3f} (Regular)"
    else:
        return f"⚫ {score:.3f} (Bajo)"

def truncate_text(text: str, max_length: int = 300, suffix: str = "...") -> str:
    """
    Trunca texto para preview en la UI.
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        suffix: Sufijo a agregar si se trunca
    
    Returns:
        Texto truncado
    """
    if not text or len(text) <= max_length:
        return text
    
    # Buscar último espacio antes del límite para no cortar palabras
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # Si hay espacio cerca del final
        truncated = truncated[:last_space]
    
    return truncated + suffix

def highlight_query_terms(text: str, query: str) -> str:
    """
    Resalta términos de búsqueda en el texto usando markdown.
    
    Args:
        text: Texto donde resaltar
        query: Consulta de búsqueda
    
    Returns:
        Texto con términos resaltados
    """
    if not query or not text:
        return text
    
    # Extraer palabras clave de la query (ignorar palabras comunes)
    stop_words = {'el', 'la', 'de', 'en', 'y', 'a', 'que', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'un', 'una', 'unos', 'unas', 'mi', 'tu', 'su', 'sus', 'este', 'esta', 'estos', 'estas'}
    words = [w.strip().lower() for w in re.findall(r'\w+', query) if len(w.strip()) > 2 and w.lower() not in stop_words]
    
    highlighted_text = text
    for word in words:
        # Usar expresión regular para encontrar coincidencias (ignorando caso)
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**{word}**", highlighted_text)
    
    return highlighted_text

def format_date_spanish(date_str: str) -> str:
    """
    Formatea fecha en español.
    
    Args:
        date_str: Fecha en formato YYYY-MM-DD
    
    Returns:
        Fecha formateada en español
    """
    if not date_str:
        return "Fecha no disponible"
    
    try:
        # Parsear fecha
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Nombres de meses en español
        months = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
        ]
        
        return f"{dt.day} de {months[dt.month - 1]} de {dt.year}"
        
    except (ValueError, IndexError):
        return date_str

def create_result_card(result: Dict[str, Any], index: int, query: str = "") -> None:
    """
    Crea una tarjeta visual para mostrar un resultado de búsqueda.
    
    Args:
        result: Diccionario con datos del resultado
        index: Índice del resultado
        query: Consulta original para highlighting
    """
    # Obtener datos del resultado
    chunk_id = result.get('chunk_id', 'N/A')
    similarity = result.get('similarity_score', 0)
    fecha = result.get('fecha_publicacion', 'N/A')
    departamento = result.get('departamento_nombre', 'N/A')
    seccion = result.get('seccion_nombre', 'N/A')
    titulo = result.get('item_titulo', 'Sin título')
    texto = result.get('texto', '')
    tokens = result.get('tokens_aproximados', 'N/A')
    
    # Container principal
    with st.container():
        # Header del resultado
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 📄 Resultado {index + 1}")
            st.markdown(f"**{titulo[:100]}{'...' if len(titulo) > 100 else ''}**")
        
        with col2:
            st.metric("Similitud", f"{similarity:.3f}")
        
        # Información principal
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**📅 Fecha:** {format_date_spanish(fecha)}")
            st.markdown(f"**🆔 ID:** `{chunk_id}`")
        
        with col2:
            st.markdown(f"**🏛️ Ministerio:** {departamento}")
            st.markdown(f"**📊 Tokens:** {tokens}")
        
        with col3:
            st.markdown(f"**📂 Sección:** {seccion}")
            st.markdown(f"**🎯 Score:** {format_similarity_score(similarity)}")
        
        # Contenido del documento
        if texto:
            st.markdown("**📝 Contenido:**")
            
            # Aplicar highlighting si hay query
            display_text = highlight_query_terms(texto, query) if query else texto
            
            # Mostrar preview o texto completo
            if len(texto) > 500:
                preview_text = truncate_text(display_text, 500)
                st.markdown(preview_text)
                
                # Botón para expandir
                if st.button(f"Ver texto completo", key=f"expand_{chunk_id}"):
                    st.text_area(
                        "Texto completo:",
                        value=texto,
                        height=300,
                        key=f"full_{chunk_id}"
                    )
            else:
                st.markdown(display_text)
        
        # Separador visual
        st.markdown("---")

def validate_date_range(start_date: date, end_date: date) -> bool:
    """
    Valida que el rango de fechas sea correcto.
    
    Args:
        start_date: Fecha de inicio
        end_date: Fecha de fin
    
    Returns:
        True si el rango es válido
    """
    if start_date > end_date:
        st.error("❌ La fecha de inicio debe ser anterior a la fecha de fin")
        return False
    
    # Verificar que no sea un rango demasiado amplio (más de 5 años)
    if (end_date - start_date).days > 365 * 5:
        st.warning("⚠️ Rango de fechas muy amplio. Los resultados pueden ser lentos.")
    
    return True

def show_loading_message(operation: str) -> None:
    """
    Muestra mensaje de carga personalizado.
    
    Args:
        operation: Descripción de la operación
    """
    st.info(f"🔄 {operation}...")

def show_no_results_message(query: str = "") -> None:
    """
    Muestra mensaje cuando no hay resultados.
    
    Args:
        query: Consulta que no dio resultados
    """
    st.warning("🔍 No se encontraron resultados.")
    
    if query:
        st.markdown(f"""
        **Sugerencias para mejorar tu búsqueda:**
        - Intenta con términos más generales
        - Revisa la ortografía
        - Prueba sinónimos o términos relacionados
        - Reduce el número de palabras en la consulta
        
        **Consulta actual:** `{query}`
        """)

def export_results_to_csv(results: List[Dict[str, Any]]) -> str:
    """
    Convierte resultados a formato CSV para descarga.
    
    Args:
        results: Lista de resultados de búsqueda
    
    Returns:
        String CSV
    """
    import csv
    from io import StringIO
    
    output = StringIO()
    
    if not results:
        return ""
    
    # Headers
    fieldnames = [
        'chunk_id', 'similarity_score', 'fecha_publicacion',
        'departamento_nombre', 'seccion_nombre', 'item_titulo',
        'tokens_aproximados', 'texto'
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for result in results:
        # Limpiar datos para CSV
        row = {}
        for field in fieldnames:
            value = result.get(field, '')
            # Limpiar saltos de línea y comillas para CSV
            if isinstance(value, str):
                value = value.replace('\n', ' ').replace('\r', ' ')
            row[field] = value
        writer.writerow(row)
    
    return output.getvalue()

# Cache para datos que no cambian frecuentemente
@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_cached_ministries(api):
    """Obtiene lista de ministerios con cache."""
    return api.get_available_ministries()

@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_cached_sections(api):
    """Obtiene lista de secciones con cache."""
    return api.get_available_sections()

@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_cached_stats(api):
    """Obtiene estadísticas con cache."""
    return api.get_stats()
