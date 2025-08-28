"""
Componente de filtros avanzados para BoeFacil.

Este módulo implementa todos los filtros disponibles para búsquedas BOE:
- Filtros temporales (fechas, rangos, presets)
- Filtros organizacionales (ministerios, secciones)
- Filtros de contenido (tokens, tipos de documento)
"""

import streamlit as st
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class BOEFilters:
    """
    Clase para manejar todos los filtros de búsqueda BOE.
    
    Maneja el estado de los filtros en st.session_state y proporciona
    métodos para renderizar la UI y obtener los valores actuales.
    """
    
    def __init__(self):
        """Inicializa el sistema de filtros."""
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa variables de estado de sesión para filtros."""
        
        # Filtros temporales
        if 'filter_start_date' not in st.session_state:
            st.session_state.filter_start_date = None
        if 'filter_end_date' not in st.session_state:
            st.session_state.filter_end_date = None
        if 'filter_date_preset' not in st.session_state:
            st.session_state.filter_date_preset = "Sin filtro"
        
        # Filtros organizacionales
        if 'filter_ministry' not in st.session_state:
            st.session_state.filter_ministry = None
        if 'filter_sections' not in st.session_state:
            st.session_state.filter_sections = []
        
        # Filtros de contenido
        if 'filter_min_tokens' not in st.session_state:
            st.session_state.filter_min_tokens = 50
        if 'filter_max_tokens' not in st.session_state:
            st.session_state.filter_max_tokens = 5000
        if 'filter_doc_type' not in st.session_state:
            st.session_state.filter_doc_type = "Todos"
    
    def render_temporal_filters(self) -> bool:
        """
        Renderiza los filtros temporales en la UI.
        
        Returns:
            bool: True si algún filtro temporal está activo
        """
        st.markdown("### 🗓️ Filtros Temporales")
        
        # Presets de fecha
        preset_options = [
            "Sin filtro",
            "Último mes", 
            "Últimos 3 meses",
            "Último año",
            "2024",
            "2023",
            "Personalizado"
        ]
        
        preset = st.selectbox(
            "Período:",
            options=preset_options,
            index=preset_options.index(st.session_state.filter_date_preset),
            help="Selecciona un período predefinido o personaliza las fechas",
            key="date_preset_selector"
        )
        
        # Actualizar estado del preset
        if preset != st.session_state.filter_date_preset:
            st.session_state.filter_date_preset = preset
            self._apply_date_preset(preset)
            st.rerun()
        
        # Mostrar selectores de fecha si es personalizado o hay fechas activas
        show_date_inputs = (preset == "Personalizado" or 
                           st.session_state.filter_start_date is not None or 
                           st.session_state.filter_end_date is not None)
        
        if show_date_inputs:
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Fecha inicio:",
                    value=st.session_state.filter_start_date,
                    min_value=date(2020, 1, 1),
                    max_value=date.today(),
                    help="Fecha de inicio del rango de búsqueda",
                    key="start_date_input"
                )
                
                if start_date != st.session_state.filter_start_date:
                    st.session_state.filter_start_date = start_date
                    if preset != "Personalizado":
                        st.session_state.filter_date_preset = "Personalizado"
            
            with col2:
                end_date = st.date_input(
                    "Fecha fin:",
                    value=st.session_state.filter_end_date,
                    min_value=date(2020, 1, 1),
                    max_value=date.today(),
                    help="Fecha de fin del rango de búsqueda",
                    key="end_date_input"
                )
                
                if end_date != st.session_state.filter_end_date:
                    st.session_state.filter_end_date = end_date
                    if preset != "Personalizado":
                        st.session_state.filter_date_preset = "Personalizado"
        
        # Validación de rango de fechas
        if (st.session_state.filter_start_date and 
            st.session_state.filter_end_date and 
            st.session_state.filter_start_date > st.session_state.filter_end_date):
            st.error("❌ La fecha de inicio debe ser anterior a la fecha de fin")
            return False
        
        # Mostrar información del filtro activo
        if st.session_state.filter_start_date or st.session_state.filter_end_date:
            self._show_active_date_filter()
            return True
        
        return False
    
    def _apply_date_preset(self, preset: str):
        """
        Aplica un preset de fecha predefinido.
        
        Args:
            preset: Nombre del preset seleccionado
        """
        today = date.today()
        
        if preset == "Sin filtro":
            st.session_state.filter_start_date = None
            st.session_state.filter_end_date = None
        
        elif preset == "Último mes":
            st.session_state.filter_end_date = today
            st.session_state.filter_start_date = today - timedelta(days=30)
        
        elif preset == "Últimos 3 meses":
            st.session_state.filter_end_date = today
            st.session_state.filter_start_date = today - timedelta(days=90)
        
        elif preset == "Último año":
            st.session_state.filter_end_date = today
            st.session_state.filter_start_date = today - timedelta(days=365)
        
        elif preset == "2024":
            st.session_state.filter_start_date = date(2024, 1, 1)
            st.session_state.filter_end_date = date(2024, 12, 31)
        
        elif preset == "2023":
            st.session_state.filter_start_date = date(2023, 1, 1)
            st.session_state.filter_end_date = date(2023, 12, 31)
        
        elif preset == "2022":
            st.session_state.filter_start_date = date(2022, 1, 1)
            st.session_state.filter_end_date = date(2022, 12, 31)
        
        # "Personalizado" no modifica las fechas
    
    def _show_active_date_filter(self):
        """Muestra información visual del filtro de fecha activo."""
        start_str = st.session_state.filter_start_date.strftime("%d/%m/%Y") if st.session_state.filter_start_date else "..."
        end_str = st.session_state.filter_end_date.strftime("%d/%m/%Y") if st.session_state.filter_end_date else "..."
        
        st.info(f"📅 Filtro activo: {start_str} → {end_str}")
        
        # Botón para limpiar filtro temporal
        if st.button("🗑️ Limpiar filtro temporal", key="clear_date_filter"):
            st.session_state.filter_start_date = None
            st.session_state.filter_end_date = None
            st.session_state.filter_date_preset = "Sin filtro"
            st.rerun()
    
    def render_organizational_filters(self, api) -> bool:
        """
        Renderiza los filtros organizacionales en la UI.
        
        Args:
            api: Instancia de BOESearchAPI para obtener datos
        
        Returns:
            bool: True si algún filtro organizacional está activo
        """
        st.markdown("### 🏛️ Filtros Organizacionales")
        
        has_active_filters = False
        
        # Filtro de ministerio
        try:
            with st.spinner("Cargando ministerios..."):
                ministries = self._get_cached_ministries(api)
            
            ministry_options = ["Sin filtro"] + ministries
            current_ministry = st.session_state.filter_ministry or "Sin filtro"
            
            # Asegurar que el ministerio actual esté en las opciones
            if current_ministry not in ministry_options:
                ministry_options.append(current_ministry)
            
            selected_ministry = st.selectbox(
                "Ministerio/Departamento:",
                options=ministry_options,
                index=ministry_options.index(current_ministry),
                help="Filtrar por ministerio o departamento específico",
                key="ministry_selector"
            )
            
            if selected_ministry != current_ministry:
                st.session_state.filter_ministry = selected_ministry if selected_ministry != "Sin filtro" else None
            
            if st.session_state.filter_ministry:
                st.success(f"🏛️ Ministerio: {st.session_state.filter_ministry}")
                has_active_filters = True
        
        except Exception as e:
            st.warning(f"⚠️ No se pudieron cargar los ministerios: {e}")
            logger.warning(f"Error cargando ministerios: {e}")
        
        # Filtro de secciones
        try:
            sections = self._get_cached_sections(api)
            
            if sections:
                section_options = [f"{s['codigo']} - {s['nombre']}" for s in sections]
                
                selected_sections = st.multiselect(
                    "Secciones BOE:",
                    options=section_options,
                    default=[s for s in section_options if self._is_section_selected(s)],
                    help="Selecciona una o más secciones del BOE",
                    key="sections_multiselect"
                )
                
                # Actualizar estado
                section_codes = [s.split(" - ")[0] for s in selected_sections]
                if section_codes != st.session_state.filter_sections:
                    st.session_state.filter_sections = section_codes
                
                if st.session_state.filter_sections:
                    st.success(f"📂 Secciones: {', '.join(st.session_state.filter_sections)}")
                    has_active_filters = True
        
        except Exception as e:
            st.warning(f"⚠️ No se pudieron cargar las secciones: {e}")
            logger.warning(f"Error cargando secciones: {e}")
        
        return has_active_filters
    
    def _is_section_selected(self, section_option: str) -> bool:
        """Verifica si una sección está seleccionada."""
        section_code = section_option.split(" - ")[0]
        return section_code in st.session_state.filter_sections
    
    def _get_cached_ministries(self, api) -> List[str]:
        """Obtiene lista de ministerios con cache."""
        try:
            return api.get_available_ministries(limit=100)
        except Exception as e:
            logger.error(f"Error obteniendo ministerios: {e}")
            return []
    
    def _get_cached_sections(self, api) -> List[Dict[str, str]]:
        """Obtiene lista de secciones con cache."""
        try:
            return api.get_available_sections()
        except Exception as e:
            logger.error(f"Error obteniendo secciones: {e}")
            return []
    
    def render_content_filters(self) -> bool:
        """
        Renderiza los filtros de contenido en la UI.
        
        Returns:
            bool: True si algún filtro de contenido está activo
        """
        st.markdown("### 📄 Filtros de Contenido")
        
        # Slider para tokens
        col1, col2 = st.columns([3, 1])
        
        with col1:
            token_range = st.slider(
                "Rango de tokens:",
                min_value=10,
                max_value=10000,
                value=(st.session_state.filter_min_tokens, st.session_state.filter_max_tokens),
                step=50,
                help="Filtra documentos por longitud (número de tokens)",
                key="token_range_slider"
            )
            
            # Actualizar estado
            if (token_range[0] != st.session_state.filter_min_tokens or 
                token_range[1] != st.session_state.filter_max_tokens):
                st.session_state.filter_min_tokens = token_range[0]
                st.session_state.filter_max_tokens = token_range[1]
        
        with col2:
            # Presets de tipo de documento
            doc_types = {
                "Todos": (10, 10000),
                "Cortos": (10, 500),
                "Medios": (500, 2000),
                "Largos": (2000, 10000)
            }
            
            doc_type = st.selectbox(
                "Tipo:",
                options=list(doc_types.keys()),
                index=list(doc_types.keys()).index(st.session_state.filter_doc_type),
                help="Presets de longitud de documento",
                key="doc_type_selector"
            )
            
            # Aplicar preset si cambió
            if doc_type != st.session_state.filter_doc_type:
                st.session_state.filter_doc_type = doc_type
                if doc_type != "Todos":
                    min_tok, max_tok = doc_types[doc_type]
                    st.session_state.filter_min_tokens = min_tok
                    st.session_state.filter_max_tokens = max_tok
                    st.rerun()
        
        # Mostrar filtro activo
        has_filter = (st.session_state.filter_min_tokens > 10 or 
                     st.session_state.filter_max_tokens < 10000)
        
        if has_filter:
            st.info(f"📊 Tokens: {st.session_state.filter_min_tokens:,} - {st.session_state.filter_max_tokens:,}")
        
        return has_filter
    
    def render_filter_summary(self) -> int:
        """
        Renderiza un resumen de todos los filtros activos.
        
        Returns:
            int: Número de filtros activos
        """
        active_filters = []
        
        # Contar filtros activos
        if st.session_state.filter_start_date or st.session_state.filter_end_date:
            active_filters.append("📅 Temporal")
        
        if st.session_state.filter_ministry:
            active_filters.append("🏛️ Ministerio")
        
        if st.session_state.filter_sections:
            active_filters.append("📂 Secciones")
        
        if (st.session_state.filter_min_tokens > 10 or 
            st.session_state.filter_max_tokens < 10000):
            active_filters.append("📊 Tokens")
        
        # Mostrar resumen
        if active_filters:
            st.markdown("#### 🎯 Filtros Activos")
            st.write(" • ".join(active_filters))
            
            # Botón para limpiar todos los filtros
            if st.button("🗑️ Limpiar todos los filtros", key="clear_all_filters"):
                self.clear_all_filters()
                st.rerun()
        else:
            st.info("ℹ️ No hay filtros activos - se mostrarán todos los resultados")
        
        return len(active_filters)
    
    def clear_all_filters(self):
        """Limpia todos los filtros activos."""
        st.session_state.filter_start_date = None
        st.session_state.filter_end_date = None
        st.session_state.filter_date_preset = "Sin filtro"
        st.session_state.filter_ministry = None
        st.session_state.filter_sections = []
        st.session_state.filter_min_tokens = 50
        st.session_state.filter_max_tokens = 5000
        st.session_state.filter_doc_type = "Todos"
    
    def get_filter_parameters(self) -> Dict[str, Any]:
        """
        Obtiene los parámetros de filtro actuales para la API.
        
        Returns:
            Dict con parámetros para api.advanced_search()
        """
        params = {}
        
        # Filtros temporales
        if st.session_state.filter_start_date:
            params['start_date'] = st.session_state.filter_start_date.strftime("%Y-%m-%d")
        
        if st.session_state.filter_end_date:
            params['end_date'] = st.session_state.filter_end_date.strftime("%Y-%m-%d")
        
        # Filtros organizacionales
        if st.session_state.filter_ministry:
            params['ministry'] = st.session_state.filter_ministry
        
        if st.session_state.filter_sections:
            # Para múltiples secciones, usar la primera (API limitation)
            # En futuro se puede extender la API para soportar múltiples
            params['section'] = st.session_state.filter_sections[0]
        
        # Filtros de contenido
        if st.session_state.filter_min_tokens > 10:
            params['min_tokens'] = st.session_state.filter_min_tokens
        
        if st.session_state.filter_max_tokens < 10000:
            params['max_tokens'] = st.session_state.filter_max_tokens
        
        return params
    
    def has_active_filters(self) -> bool:
        """
        Verifica si hay algún filtro activo.
        
        Returns:
            bool: True si hay filtros activos
        """
        return bool(
            st.session_state.filter_start_date or
            st.session_state.filter_end_date or
            st.session_state.filter_ministry or
            st.session_state.filter_sections or
            st.session_state.filter_min_tokens > 10 or
            st.session_state.filter_max_tokens < 10000
        )
