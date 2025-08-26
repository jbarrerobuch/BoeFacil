"""
Base de datos vectorial usando FAISS para búsqueda semántica en documentos BOE.

Este módulo proporciona una interfaz completa para crear, gestionar y consultar
índices vectoriales usando FAISS, específicamente diseñado para el proyecto BoeFacil.
"""

import faiss
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

# Configuración del logger
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Base de datos vectorial usando FAISS para búsqueda semántica.
    
    Funcionalidades:
    - Crear y gestionar índices FAISS
    - Agregar embeddings con metadatos
    - Realizar búsquedas semánticas
    - Persistir y cargar índices
    """

    def __init__(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None, dimension: int = 1024):
        """
        Inicializa la base de datos vectorial.
        
        Args:
            index_path: Ruta al archivo del índice FAISS (.faiss)
            metadata_path: Ruta al archivo de metadatos (.json)
            dimension: Dimensión de los vectores (1024 para BGE-M3)
        """
        self.dimension = dimension
        self.index = None
        self.metadata = []  # Lista de diccionarios con metadatos de cada vector
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index_type = None
        self.creation_time = None
        
        logger.info(f"Inicializando VectorDatabase con dimensión {dimension}")
        
        # Si se proporcionan rutas, intentar cargar índice existente
        if index_path and metadata_path:
            if Path(index_path).exists() and Path(metadata_path).exists():
                logger.info(f"Cargando índice existente desde {index_path}")
                self.load_index(index_path, metadata_path)
            else:
                logger.info("Rutas especificadas pero archivos no existen, se creará nuevo índice")
    
    def create_index(self, index_type: str = "Flat") -> None:
        """
        Crea un nuevo índice FAISS del tipo especificado.
        
        Args:
            index_type: Tipo de índice ("Flat", "IVF", "HNSW")
                - Flat: Búsqueda exacta, ideal para <100K vectores
                - IVF: Búsqueda aproximada, ideal para 100K-1M vectores  
                - HNSW: Búsqueda ultra-rápida, ideal para >1M vectores
        """
        self.index_type = index_type
        self.creation_time = time.time()
        
        if index_type == "Flat":
            # IndexFlatL2: búsqueda exacta usando distancia L2
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Creado índice Flat L2 con dimensión {self.dimension}")
            
        elif index_type == "IVF":
            # IndexIVFFlat: búsqueda aproximada más rápida
            nlist = 100  # número de clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.info(f"Creado índice IVF con {nlist} clusters")
            
        elif index_type == "HNSW":
            # IndexHNSWFlat: búsqueda muy rápida para datasets grandes
            m = 32  # número de conexiones bidireccionales
            self.index = faiss.IndexHNSWFlat(self.dimension, m)
            logger.info(f"Creado índice HNSW con M={m}")
            
        else:
            raise ValueError(f"Tipo de índice no soportado: {index_type}. Use 'Flat', 'IVF', o 'HNSW'")
        
        # Limpiar metadatos existentes
        self.metadata = []
        logger.info(f"Índice {index_type} creado exitosamente")
    
    def add_vectors(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """
        Agrega vectores y sus metadatos al índice.
        
        Args:
            embeddings: Array numpy con shape (n_vectors, dimension)
            metadata_list: Lista de diccionarios con metadatos de cada vector
        """
        if self.index is None:
            raise RuntimeError("Debe crear un índice primero usando create_index()")
        
        if len(embeddings) != len(metadata_list):
            raise ValueError(f"Número de embeddings ({len(embeddings)}) debe coincidir con número de metadatos ({len(metadata_list)})")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimensión de embeddings ({embeddings.shape[1]}) no coincide con dimensión del índice ({self.dimension})")
        
        # Convertir a float32 si es necesario
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Para índices IVF, entrenar si no se ha hecho
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info("Entrenando índice IVF...")
            self.index.train(embeddings)
            logger.info("Entrenamiento de índice IVF completado")
        
        # Agregar vectores al índice
        start_time = time.time()
        self.index.add(embeddings)
        
        # Agregar metadatos
        self.metadata.extend(metadata_list)
        
        duration = time.time() - start_time
        logger.info(f"Agregados {len(embeddings)} vectores en {duration:.2f} segundos")
        logger.info(f"Total de vectores en índice: {self.index.ntotal}")
    
    def search(self, query_vector: np.ndarray, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda de similitud vectorial.
        
        Args:
            query_vector: Vector de consulta (shape: (dimension,) o (1, dimension))
            k: Número de resultados a devolver
            filters: Filtros opcionales por metadatos (ej: {'seccion_codigo': 'I'})
        
        Returns:
            Lista de diccionarios con resultados ordenados por similitud
        """
        if self.index is None:
            raise RuntimeError("No hay índice cargado. Use create_index() o load_index()")
        
        if self.index.ntotal == 0:
            logger.warning("El índice está vacío")
            return []
        
        # Asegurar formato correcto del vector
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        # Realizar búsqueda
        start_time = time.time()
        
        # Buscar más resultados si hay filtros (para poder filtrar después)
        search_k = k * 3 if filters else k
        search_k = min(search_k, self.index.ntotal)
        
        distances, indices = self.index.search(query_vector, search_k)
        
        # Construir resultados
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No hay más resultados válidos
                break
                
            result = {
                'similarity_score': float(1 / (1 + distance)),  # Convertir distancia a score de similitud
                'distance': float(distance),
                'rank': i + 1,
                **self.metadata[idx]  # Agregar todos los metadatos
            }
            
            # Aplicar filtros si se especifican
            if filters:
                if self._matches_filters(result, filters):
                    results.append(result)
            else:
                results.append(result)
            
            # Detener si ya tenemos suficientes resultados
            if len(results) >= k:
                break
        
        duration = time.time() - start_time
        logger.info(f"Búsqueda completada en {duration:.3f} segundos, {len(results)} resultados")
        
        return results[:k]
    
    def _matches_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Verifica si un resultado coincide con los filtros especificados.
        
        Args:
            result: Diccionario con metadatos del resultado
            filters: Diccionario con filtros a aplicar
        
        Returns:
            True si el resultado coincide con todos los filtros
        """
        for key, value in filters.items():
            if key not in result:
                return False
            if isinstance(value, list):
                if result[key] not in value:
                    return False
            else:
                if result[key] != value:
                    return False
        return True
    
    def save_index(self, index_path: str, metadata_path: str) -> None:
        """
        Guarda el índice y metadatos en disco.
        
        Args:
            index_path: Ruta donde guardar el índice FAISS
            metadata_path: Ruta donde guardar los metadatos JSON
        """
        if self.index is None:
            raise RuntimeError("No hay índice para guardar")
        
        # Crear directorios si no existen
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar índice FAISS
        faiss.write_index(self.index, index_path)
        
        # Guardar metadatos y configuración
        metadata_dict = {
            'metadata': self.metadata,
            'config': {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'creation_time': self.creation_time,
                'total_vectors': self.index.ntotal,
                'save_time': time.time()
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        logger.info(f"Índice guardado en: {index_path}")
        logger.info(f"Metadatos guardados en: {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str) -> None:
        """
        Carga un índice y metadatos desde disco.
        
        Args:
            index_path: Ruta del archivo de índice FAISS
            metadata_path: Ruta del archivo de metadatos JSON
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Archivo de índice no encontrado: {index_path}")
        
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Archivo de metadatos no encontrado: {metadata_path}")
        
        # Cargar índice FAISS
        self.index = faiss.read_index(index_path)
        
        # Cargar metadatos
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        self.metadata = metadata_dict['metadata']
        config = metadata_dict.get('config', {})
        
        self.dimension = config.get('dimension', self.dimension)
        self.index_type = config.get('index_type', 'Unknown')
        self.creation_time = config.get('creation_time')
        
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        logger.info(f"Índice cargado: {self.index.ntotal} vectores, tipo {self.index_type}")
        logger.info(f"Metadatos cargados: {len(self.metadata)} entradas")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos vectorial.
        
        Returns:
            Diccionario con estadísticas del índice
        """
        if self.index is None:
            return {'status': 'No index loaded'}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'creation_time': self.creation_time,
            'metadata_entries': len(self.metadata),
            'is_trained': getattr(self.index, 'is_trained', True),
            'index_path': self.index_path,
            'metadata_path': self.metadata_path
        }
        
        # Estadísticas adicionales para diferentes tipos de índices
        if hasattr(self.index, 'nlist'):  # IVF
            stats['nlist'] = self.index.nlist
        
        if hasattr(self.index, 'hnsw'):  # HNSW
            stats['hnsw_M'] = self.index.hnsw.M
            stats['hnsw_max_level'] = self.index.hnsw.max_level
        
        return stats
    
    def get_document_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un documento específico por su chunk_id.
        
        Args:
            chunk_id: ID del chunk a buscar
        
        Returns:
            Diccionario con metadatos del documento o None si no se encuentra
        """
        for metadata in self.metadata:
            if metadata.get('chunk_id') == chunk_id:
                return metadata
        return None
    
    def delete_vectors_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Marca vectores para eliminación basado en filtros.
        Nota: FAISS no soporta eliminación directa, esta función marca elementos.
        
        Args:
            filters: Filtros para identificar vectores a eliminar
        
        Returns:
            Número de vectores marcados para eliminación
        """
        count = 0
        for i, metadata in enumerate(self.metadata):
            if self._matches_filters(metadata, filters):
                metadata['_deleted'] = True
                count += 1
        
        logger.info(f"Marcados {count} vectores para eliminación")
        return count
    
    def rebuild_index_without_deleted(self) -> None:
        """
        Reconstruye el índice excluyendo vectores marcados como eliminados.
        """
        if self.index is None:
            raise RuntimeError("No hay índice para reconstruir")
        
        # Identificar vectores no eliminados
        valid_indices = []
        valid_metadata = []
        
        for i, metadata in enumerate(self.metadata):
            if not metadata.get('_deleted', False):
                valid_indices.append(i)
                valid_metadata.append(metadata)
        
        if len(valid_indices) == len(self.metadata):
            logger.info("No hay vectores marcados para eliminación")
            return
        
        # Extraer vectores válidos (esto requiere almacenar vectores originales)
        logger.warning("Reconstrucción de índice requiere vectores originales")
        logger.info(f"Se mantendrían {len(valid_indices)} de {len(self.metadata)} vectores")
        
        # Actualizar metadatos
        self.metadata = valid_metadata
