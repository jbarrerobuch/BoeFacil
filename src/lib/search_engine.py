"""
Motor de búsqueda semántica para documentos BOE usando FAISS.

Este módulo proporciona un motor de búsqueda completo que permite realizar
consultas semánticas, filtradas y avanzadas sobre la base de datos vectorial BOE.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

from .vector_db import VectorDatabase
from .advanced_filter import AdvancedFilter

# Configuración del logger
logger = logging.getLogger(__name__)

class BOESearchEngine:
    """
    Motor de búsqueda semántica para documentos BOE.
    
    Proporciona funcionalidades completas para:
    - Búsqueda por texto natural (semántica)
    - Búsqueda de documentos similares
    - Búsqueda filtrada por metadatos
    - Búsqueda por rango de fechas
    - Búsqueda avanzada combinando múltiples criterios
    """
    
    def __init__(
        self, 
        index_path: str, 
        metadata_path: str, 
        model_name: str = "pablosi/bge-m3-trained-2"
    ):
        """
        Inicializa el motor de búsqueda.
        
        Args:
            index_path: Ruta al archivo de índice FAISS
            metadata_path: Ruta al archivo de metadatos JSON
            model_name: Nombre del modelo SentenceTransformer para embeddings
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        
        # Inicializar base de datos vectorial
        logger.info(f"Cargando índice desde: {index_path}")
        self.db = VectorDatabase()
        self.db.load_index(index_path, metadata_path)

        stats = self.db.get_stats()
        logger.info(f"Índice cargado: {stats['total_vectors']} vectores, dimensión {stats['dimension']}")
        
        # Inicializar modelo de embeddings
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Modelo de embeddings cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo {model_name}: {e}")
            raise
        
        # Inicializar sistema de filtros avanzados
        self.filter_engine = AdvancedFilter()
        
        logger.info("Motor de búsqueda BOE inicializado correctamente")
    
    def search_by_text(
        self, 
        query_text: str, 
        k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda semántica por texto natural.
        
        Args:
            query_text: Texto de la consulta en lenguaje natural
            k: Número máximo de resultados a devolver
            filters: Filtros opcionales de metadatos
        
        Returns:
            Lista de documentos ordenados por relevancia semántica
        
        Example:
            results = engine.search_by_text("Real decreto sobre impuestos", k=5)
        """
        logger.info(f"Búsqueda por texto: '{query_text}' (k={k})")
        
        if not query_text or not query_text.strip():
            logger.warning("Consulta vacía")
            return []
        
        start_time = time.time()
        
        # Generar embedding de la consulta
        try:
            query_embedding = self.embedding_model.encode([query_text.strip()])[0]
            query_embedding = query_embedding.astype(np.float32)
            logger.debug(f"Embedding generado: shape {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Error generando embedding para '{query_text}': {e}")
            return []
        
        # Realizar búsqueda vectorial
        try:
            # Buscar más resultados si hay filtros (para poder filtrar después)
            search_k = k * 3 if filters else k
            search_k = min(search_k, self.db.index.ntotal)
            
            results = self.db.search(query_embedding, k=search_k)
            logger.debug(f"Búsqueda vectorial completada: {len(results)} resultados iniciales")
            
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            return []
        
        # Aplicar filtros si se especifican
        if filters:
            try:
                results = self.filter_engine.apply_metadata_filters(results, filters)
                logger.debug(f"Filtros aplicados: {len(results)} resultados después del filtrado")
            except Exception as e:
                logger.error(f"Error aplicando filtros: {e}")
                # Continuar sin filtros en caso de error
        
        # Limitar a k resultados
        results = results[:k]
        
        duration = time.time() - start_time
        logger.info(f"Búsqueda completada en {duration:.3f}s: {len(results)} resultados")
        
        return results
    
    def search_similar_documents(
        self, 
        chunk_id: str, 
        k: int = 10,
        exclude_same_document: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Encuentra documentos similares a un chunk específico.
        
        Args:
            chunk_id: ID del chunk de referencia
            k: Número de resultados similares
            exclude_same_document: Si excluir chunks del mismo documento padre
        
        Returns:
            Lista de documentos similares ordenados por similitud
        
        Example:
            results = engine.search_similar_documents("BOE-A-2023-12345_001", k=5)
        """
        logger.info(f"Búsqueda de similares para chunk: {chunk_id}")
        
        # Encontrar el documento de referencia
        reference_doc = self.db.get_document_by_id(chunk_id)
        if not reference_doc:
            logger.warning(f"Chunk no encontrado: {chunk_id}")
            return []
        
        # Obtener el embedding del documento de referencia
        # Buscar el índice del chunk en los metadatos
        chunk_index = None
        for i, metadata in enumerate(self.db.metadata):
            if metadata.get('chunk_id') == chunk_id:
                chunk_index = i
                break
        
        if chunk_index is None:
            logger.error(f"No se encontró el índice para chunk {chunk_id}")
            return []
        
        # Extraer el vector del índice FAISS
        try:
            # Reconstruir el vector desde el índice
            vector = self.db.index.reconstruct(chunk_index)
            vector = vector.astype(np.float32)
        except Exception as e:
            logger.error(f"Error extrayendo vector para {chunk_id}: {e}")
            return []
        
        # Realizar búsqueda de similitud
        try:
            # Buscar más resultados para poder filtrar
            search_k = k + 10 if exclude_same_document else k
            search_k = min(search_k, self.db.index.ntotal)
            
            results = self.db.search(vector, k=search_k)
            
            # Excluir el documento original y posiblemente del mismo documento padre
            filtered_results = []
            reference_item_id = reference_doc.get('item_id')
            
            for result in results:
                # Excluir el chunk original
                if result.get('chunk_id') == chunk_id:
                    continue
                
                # Excluir chunks del mismo documento si se solicita
                if exclude_same_document and result.get('item_id') == reference_item_id:
                    continue
                
                filtered_results.append(result)
                
                if len(filtered_results) >= k:
                    break
            
            logger.info(f"Encontrados {len(filtered_results)} documentos similares")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de similares: {e}")
            return []
    
    def search_by_date_range(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        query_text: Optional[str] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda filtrada por rango de fechas.
        
        Args:
            start_date: Fecha de inicio (formato YYYY-MM-DD)
            end_date: Fecha de fin (formato YYYY-MM-DD)
            query_text: Texto opcional para búsqueda semántica adicional
            k: Número de resultados
        
        Returns:
            Lista de documentos en el rango de fechas especificado
        
        Example:
            # Documentos de enero 2023
            results = engine.search_by_date_range("2023-01-01", "2023-01-31")
            
            # Con búsqueda semántica
            results = engine.search_by_date_range(
                "2023-01-01", "2023-02-15", 
                query_text="ministerio hacienda"
            )
        """
        logger.info(f"Búsqueda por fechas: {start_date} - {end_date}")
        
        # Validar que al menos una fecha esté especificada
        if not start_date and not end_date:
            logger.warning("Debe especificar al menos start_date o end_date")
            return []
        
        # Construir filtro de fecha
        date_range = {}
        if start_date:
            date_range['start_date'] = start_date
        if end_date:
            date_range['end_date'] = end_date
        
        # Si hay consulta de texto, hacer búsqueda semántica primero
        if query_text and query_text.strip():
            # Búsqueda semántica amplia para luego filtrar por fecha
            results = self.search_by_text(query_text, k=k*5)  # Buscar más para filtrar
        else:
            # Solo filtro por fecha - obtener todos los documentos relevantes
            # Esto requiere una búsqueda que devuelva muchos resultados
            dummy_query = "documento"  # Consulta genérica
            results = self.search_by_text(dummy_query, k=min(10000, self.db.index.ntotal))
        
        # Aplicar filtro de fecha
        try:
            filtered_results = self.filter_engine.apply_date_range_filter(results, date_range)
            logger.info(f"Filtro de fecha aplicado: {len(filtered_results)} resultados")
        except Exception as e:
            logger.error(f"Error aplicando filtro de fecha: {e}")
            return []
        
        # Si no había búsqueda semántica, ordenar por fecha (más reciente primero)
        if not query_text or not query_text.strip():
            filtered_results = sorted(
                filtered_results, 
                key=lambda x: x.get('fecha_publicacion', ''), 
                reverse=True
            )
        
        return filtered_results[:k]
    
    def search_by_department(
        self, 
        department: str, 
        query_text: Optional[str] = None,
        k: int = 10,
        exact_match: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda filtrada por departamento.
        
        Args:
            department: Nombre o código del departamento
            query_text: Texto opcional para búsqueda semántica adicional
            k: Número de resultados
            exact_match: Si True, busca coincidencia exacta; si False, contiene
        
        Returns:
            Lista de documentos del departamento especificado
        
        Example:
            # Todos los documentos de Hacienda
            results = engine.search_by_department("HACIENDA")
            
            # Con búsqueda semántica
            results = engine.search_by_department(
                "HACIENDA", 
                query_text="presupuesto"
            )
        """
        logger.info(f"Búsqueda por departamento: '{department}' (exact={exact_match})")
        
        if not department or not department.strip():
            logger.warning("Departamento no especificado")
            return []
        
        # Construir filtros de departamento
        filters = {}
        department = department.strip()
        
        if exact_match:
            # Intentar tanto código como nombre exacto
            filters = {
                'departamento_codigo': department,
                'departamento_nombre': department
            }
        else:
            # Búsqueda que contiene el texto
            filters = {
                'departamento_nombre_contains': department
            }
        
        # Realizar búsqueda
        if query_text and query_text.strip():
            # Búsqueda semántica con filtros
            return self.search_by_text(query_text, k=k, filters=filters)
        else:
            # Solo filtros - búsqueda genérica
            return self.search_by_text("documento", k=k*5, filters=filters)[:k]
    
    def advanced_search(
        self,
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        date_range: Optional[Dict[str, str]] = None,
        ranking_strategy: str = "semantic",
        k: int = 10,
        require_all_filters: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda avanzada combinando múltiples criterios de forma flexible.
        
        Args:
            query_text: Texto opcional para búsqueda semántica
            filters: Filtros opcionales de metadatos
            date_range: Rango de fechas opcional {'start_date': '2023-01-01', 'end_date': '2023-02-15'}
            ranking_strategy: Estrategia de ordenamiento ('semantic', 'temporal', 'hybrid')
            k: Número de resultados
            require_all_filters: Si True usa AND entre filtros, si False usa OR
        
        Returns:
            Lista de documentos que cumplen los criterios especificados
        
        Examples:
            # Solo búsqueda semántica
            results = engine.advanced_search(query_text="impuestos sociedades")
            
            # Solo filtros de metadatos
            results = engine.advanced_search(
                filters={'departamento_nombre_contains': 'HACIENDA'}
            )
            
            # Solo rango de fechas
            results = engine.advanced_search(
                date_range={'start_date': '2023-01-01', 'end_date': '2023-02-15'}
            )
            
            # Combinación completa
            results = engine.advanced_search(
                query_text="impuesto sociedades",
                filters={
                    'seccion_codigo': ['I', 'II'],
                    'departamento_nombre_contains': 'HACIENDA',
                    'tokens_min': 500
                },
                date_range={'start_date': '2023-01-01', 'end_date': '2023-02-15'}
            )
        """
        logger.info("Iniciando búsqueda avanzada")
        logger.debug(f"Parámetros: query_text={bool(query_text)}, filters={bool(filters)}, date_range={bool(date_range)}")
        
        start_time = time.time()
        results = []
        
        # Validar que al menos un criterio esté especificado
        if not any([query_text, filters, date_range]):
            logger.warning("Debe especificar al menos un criterio de búsqueda")
            return []
        
        try:
            # Paso 1: Búsqueda semántica inicial (si se especifica)
            if query_text and query_text.strip():
                logger.debug("Realizando búsqueda semántica")
                # Buscar más resultados para poder aplicar filtros después
                semantic_k = k * 5 if (filters or date_range) else k
                semantic_k = min(semantic_k, self.db.index.ntotal)
                
                results = self.search_by_text(query_text, k=semantic_k)
                logger.debug(f"Búsqueda semántica: {len(results)} resultados")
                
            else:
                # No hay búsqueda semántica - obtener conjunto amplio para filtrar
                logger.debug("No hay búsqueda semántica, obteniendo conjunto amplio")
                # Usar búsqueda genérica para obtener muchos documentos
                dummy_query = "documento"
                results = self.search_by_text(dummy_query, k=min(10000, self.db.index.ntotal))
                logger.debug(f"Conjunto inicial: {len(results)} resultados")
            
            # Paso 2: Aplicar filtros de metadatos (si se especifican)
            if filters:
                logger.debug(f"Aplicando filtros de metadatos: {list(filters.keys())}")
                try:
                    results = self.filter_engine.apply_metadata_filters(
                        results, filters, require_all=require_all_filters
                    )
                    logger.debug(f"Después de filtros de metadatos: {len(results)} resultados")
                except Exception as e:
                    logger.error(f"Error aplicando filtros de metadatos: {e}")
                    # Continuar sin filtros en caso de error
            
            # Paso 3: Aplicar filtro de fecha (si se especifica)
            if date_range:
                logger.debug(f"Aplicando filtro de fecha: {date_range}")
                try:
                    results = self.filter_engine.apply_date_range_filter(results, date_range)
                    logger.debug(f"Después de filtro de fecha: {len(results)} resultados")
                except Exception as e:
                    logger.error(f"Error aplicando filtro de fecha: {e}")
                    # Continuar sin filtro de fecha en caso de error
            
            # Paso 4: Aplicar estrategia de ranking
            if ranking_strategy == "temporal" and not query_text:
                # Ordenar por fecha (más reciente primero) cuando no hay búsqueda semántica
                results = sorted(
                    results, 
                    key=lambda x: x.get('fecha_publicacion', ''), 
                    reverse=True
                )
                logger.debug("Aplicado ranking temporal")
            elif ranking_strategy == "hybrid" and query_text:
                # En futuras versiones se implementarán rankings más sofisticados
                # Por ahora mantener el orden semántico
                logger.debug("Aplicado ranking híbrido (semántico por defecto)")
            # Para "semantic": mantener el orden original de la búsqueda vectorial
            
            # Paso 5: Limitar resultados
            final_results = results[:k]
            
            duration = time.time() - start_time
            logger.info(f"Búsqueda avanzada completada en {duration:.3f}s: {len(final_results)} resultados finales")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda avanzada: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del motor de búsqueda.
        
        Returns:
            Diccionario con estadísticas del índice y modelo
        """
        db_stats = self.db.get_stats()
        
        return {
            'index_stats': db_stats,
            'model_name': self.model_name,
            'embedding_dimension': getattr(self.embedding_model, 'get_sentence_embedding_dimension', lambda: 'unknown')(),
            'total_searchable_documents': db_stats.get('total_vectors', 0)
        }
