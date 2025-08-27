"""
API simplificada para búsquedas BOE de alto nivel.

Este módulo proporciona una interfaz simplificada y fácil de usar para realizar
búsquedas en la base de datos vectorial BOE sin necesidad de conocer los detalles
técnicos del motor de búsqueda subyacente.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .search_engine import BOESearchEngine

# Configuración del logger
logger = logging.getLogger(__name__)

class BOESearchAPI:
    """
    API simplificada para búsquedas BOE.
    
    Proporciona métodos intuitivos para realizar búsquedas comunes sin necesidad
    de conocer los detalles técnicos del motor de búsqueda FAISS subyacente.
    """
    
    def __init__(
        self, 
        index_path: str, 
        metadata_path: str, 
        model_name: str = "pablosi/bge-m3-trained-2"
    ):
        """
        Inicializa la API de búsqueda BOE.
        
        Args:
            index_path: Ruta al archivo de índice FAISS
            metadata_path: Ruta al archivo de metadatos JSON
            model_name: Nombre del modelo SentenceTransformer
        """
        self.engine = BOESearchEngine(index_path, metadata_path, model_name)
        logger.info("BOESearchAPI inicializada correctamente")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Búsqueda inteligente que funciona con cualquier tipo de consulta.
        
        Esta función detecta automáticamente el tipo de búsqueda más apropiado
        basándose en el contenido de la consulta.
        
        Args:
            query: Consulta en lenguaje natural
            limit: Número máximo de resultados
        
        Returns:
            Lista de documentos relevantes
        
        Examples:
            # Búsqueda semántica simple
            results = api.search("Real decreto sobre impuestos")
            
            # Búsqueda con departamento implícito
            results = api.search("ministerio hacienda presupuesto")
            
            # Búsqueda con fecha implícita
            results = api.search("BOE diciembre 2023")
        """
        logger.info(f"Búsqueda inteligente: '{query}' (limit={limit})")
        
        if not query or not query.strip():
            logger.warning("Consulta vacía")
            return []
        
        # Por ahora, usar búsqueda semántica directa
        # En futuras versiones se puede implementar detección automática de tipos e integración con LLM.
        return self.engine.search_by_text(query.strip(), k=limit)
    
    def find_by_date(
        self, 
        date: str, 
        query: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos de una fecha específica.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            query: Consulta opcional para filtrar por contenido
            limit: Número máximo de resultados
        
        Returns:
            Lista de documentos de la fecha especificada
        
        Examples:
            # Todos los documentos del 29 de diciembre 2023
            results = api.find_by_date("2023-12-29")
            
            # Documentos de esa fecha sobre "hacienda"
            results = api.find_by_date("2023-12-29", "hacienda")
        """
        logger.info(f"Búsqueda por fecha: {date} (query={bool(query)})")
        
        return self.engine.search_by_date_range(
            start_date=date,
            end_date=date,
            query_text=query,
            k=limit
        )
    
    def find_by_date_range(
        self, 
        start_date: str, 
        end_date: str, 
        query: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos en un rango de fechas.
        
        Args:
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            query: Consulta opcional para filtrar por contenido
            limit: Número máximo de resultados
        
        Returns:
            Lista de documentos en el rango especificado
        
        Examples:
            # Documentos de enero 2023
            results = api.find_by_date_range("2023-01-01", "2023-01-31")
            
            # Documentos de enero 2023 sobre "impuestos"
            results = api.find_by_date_range("2023-01-01", "2023-01-31", "impuestos")
        """
        logger.info(f"Búsqueda por rango: {start_date} - {end_date}")
        
        return self.engine.search_by_date_range(
            start_date=start_date,
            end_date=end_date,
            query_text=query,
            k=limit
        )
    
    def find_by_ministry(
        self, 
        ministry: str, 
        query: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos de un ministerio específico.
        
        Args:
            ministry: Nombre del ministerio (puede ser parcial)
            query: Consulta opcional para filtrar por contenido
            limit: Número máximo de resultados
        
        Returns:
            Lista de documentos del ministerio especificado
        
        Examples:
            # Todos los documentos de Hacienda
            results = api.find_by_ministry("HACIENDA")
            
            # Documentos de Hacienda sobre "presupuesto"
            results = api.find_by_ministry("HACIENDA", "presupuesto")
            
            # También funciona con nombres parciales
            results = api.find_by_ministry("hacienda", "iva")
        """
        logger.info(f"Búsqueda por ministerio: '{ministry}' (query={bool(query)})")
        
        return self.engine.search_by_department(
            department=ministry,
            query_text=query,
            k=limit,
            exact_match=False  # Permite coincidencias parciales
        )
    
    def find_by_section(
        self, 
        section: str, 
        query: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos de una sección específica del BOE.
        
        Args:
            section: Código ("I", "II") o nombre de la sección
            query: Consulta opcional para filtrar por contenido
            limit: Número máximo de resultados
        
        Returns:
            Lista de documentos de la sección especificada
        
        Examples:
            # Documentos de la sección I
            results = api.find_by_section("I")
            
            # Documentos de Disposiciones Generales
            results = api.find_by_section("DISPOSICIONES GENERALES")
            
            # Con filtro adicional
            results = api.find_by_section("I", "real decreto")
        """
        logger.info(f"Búsqueda por sección: '{section}' (query={bool(query)})")
        
        # Determinar si es código o nombre
        filters = {}
        if len(section) <= 3 and section.upper() in ['I', 'II', 'III', 'IV', 'V']:
            # Probablemente es un código
            filters['seccion_codigo'] = section.upper()
        else:
            # Probablemente es nombre o parte del nombre
            filters['seccion_nombre_contains'] = section
        
        return self.engine.advanced_search(
            query_text=query,
            filters=filters,
            k=limit
        )
    
    def find_similar(self, document_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Encontrar documentos similares a uno específico.
        
        Args:
            document_id: ID del chunk de referencia (chunk_id)
            limit: Número máximo de resultados similares
        
        Returns:
            Lista de documentos similares
        
        Examples:
            # Encontrar documentos similares
            results = api.find_similar("BOE-A-2023-12345_001")
        """
        logger.info(f"Búsqueda de similares para: {document_id}")
        
        return self.engine.search_similar_documents(
            chunk_id=document_id,
            k=limit,
            exclude_same_document=True
        )
    
    def advanced_search(
        self,
        query: Optional[str] = None,
        ministry: Optional[str] = None,
        section: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda avanzada con múltiples filtros de forma simple.
        
        Args:
            query: Texto de búsqueda semántica
            ministry: Ministerio o departamento (nombre parcial)
            section: Sección del BOE (código o nombre)
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            min_tokens: Número mínimo de tokens
            max_tokens: Número máximo de tokens
            limit: Número máximo de resultados
        
        Returns:
            Lista de documentos que cumplen todos los criterios
        
        Examples:
            # Búsqueda compleja
            results = api.advanced_search(
                query="impuestos sociedades",
                ministry="HACIENDA",
                section="I",
                start_date="2023-01-01",
                end_date="2023-12-31",
                min_tokens=500
            )
        """
        logger.info("Búsqueda avanzada simplificada")
        
        # Construir filtros
        filters = {}
        date_range = {}
        
        # Filtro de ministerio
        if ministry:
            filters['departamento_nombre_contains'] = ministry
        
        # Filtro de sección
        if section:
            if len(section) <= 3 and section.upper() in ['I', 'II', 'III', 'IV', 'V']:
                filters['seccion_codigo'] = section.upper()
            else:
                filters['seccion_nombre_contains'] = section
        
        # Filtros numéricos
        if min_tokens:
            filters['tokens_min'] = min_tokens
        if max_tokens:
            filters['tokens_max'] = max_tokens
        
        # Rango de fechas
        if start_date:
            date_range['start_date'] = start_date
        if end_date:
            date_range['end_date'] = end_date
        
        return self.engine.advanced_search(
            query_text=query,
            filters=filters if filters else None,
            date_range=date_range if date_range else None,
            k=limit
        )
    
    def get_document_details(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene detalles completos de un documento específico.
        
        Args:
            chunk_id: ID del chunk
        
        Returns:
            Diccionario con todos los metadatos del documento
        
        Examples:
            details = api.get_document_details("BOE-A-2023-12345_001")
        """
        return self.engine.db.get_document_by_id(chunk_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos.
        
        Returns:
            Diccionario con estadísticas del índice
        """
        return self.engine.get_search_stats()
    
    def get_available_ministries(self, limit: int = 50) -> List[str]:
        """
        Obtiene lista de ministerios disponibles en la base de datos.
        
        Args:
            limit: Máximo número de ministerios a devolver
        
        Returns:
            Lista de nombres de ministerios únicos
        """
        # Obtener muestra de documentos para extraer ministerios
        sample_results = self.engine.search_by_text("documento", k=min(1000, self.engine.db.index.ntotal))
        
        ministries = set()
        for result in sample_results:
            dept_name = result.get('departamento_nombre', '').strip()
            if dept_name:
                ministries.add(dept_name)
        
        return sorted(list(ministries))[:limit]
    
    def get_available_sections(self) -> List[Dict[str, str]]:
        """
        Obtiene lista de secciones disponibles en la base de datos.
        
        Returns:
            Lista de diccionarios con código y nombre de sección
        """
        # Obtener muestra para extraer secciones
        sample_results = self.engine.search_by_text("documento", k=min(1000, self.engine.db.index.ntotal))
        
        sections = {}
        for result in sample_results:
            code = result.get('seccion_codigo', '').strip()
            name = result.get('seccion_nombre', '').strip()
            if code and name:
                sections[code] = name
        
        return [{'codigo': code, 'nombre': name} for code, name in sorted(sections.items())]
