"""
Constructor de índices FAISS desde archivos parquet con embeddings.

Este módulo proporciona funcionalidades para cargar archivos parquet que contienen
embeddings y metadatos, y construir índices vectoriales FAISS optimizados para
búsqueda semántica en documentos BOE.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import time

from .vector_db import VectorDatabase

# Configuración del logger
logger = logging.getLogger(__name__)

def load_parquet_with_embeddings(parquet_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Carga un archivo parquet y extrae embeddings y metadatos.
    
    Args:
        parquet_path: Ruta al archivo parquet con embeddings
        
    Returns:
        Tuple con:
        - embeddings: Array numpy con shape (n_chunks, dimension)
        - metadata_list: Lista de diccionarios con metadatos de cada chunk
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si la estructura del archivo no es válida
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Archivo parquet no encontrado: {parquet_path}")
    
    logger.info(f"Cargando archivo parquet: {parquet_path}")
    
    try:
        # Cargar DataFrame
        df = pd.read_parquet(parquet_path)
        logger.info(f"Cargadas {len(df)} filas del archivo {parquet_path.name}")
        
        # Verificar columnas requeridas
        required_columns = ['embeddings']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes en {parquet_path}: {missing_columns}")
        
        # Extraer embeddings
        embeddings_list = df['embeddings'].tolist()
        
        # Convertir embeddings a numpy array
        try:
            embeddings = np.array(embeddings_list, dtype=np.float32)
            logger.info(f"Embeddings shape: {embeddings.shape}")
            
            # Validar dimensión
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings deben ser 2D, encontrado: {embeddings.ndim}D")
                
        except Exception as e:
            raise ValueError(f"Error convirtiendo embeddings a numpy array: {e}")
        
        # Construir metadatos preservando jerarquía sumario_id → item_id → chunk_id
        metadata_list = []
        for idx, row in df.iterrows():
            # Extraer metadatos básicos con valores por defecto seguros
            metadata = {
                # Identificadores únicos (jerarquía completa)
                'sumario_id': _safe_get(row, 'sumario_id'),
                'item_id': _safe_get(row, 'item_id'),
                'chunk_id': _safe_get(row, 'chunk_id'),
                
                # Información del documento
                'texto': _safe_get(row, 'texto', ''),
                'item_titulo': _safe_get(row, 'item_titulo', ''),
                
                # Categorización
                'fecha_publicacion': _safe_get(row, 'fecha_publicacion'),
                'seccion_codigo': _safe_get(row, 'seccion_codigo'),
                'seccion_nombre': _safe_get(row, 'seccion_nombre', ''),
                'departamento_codigo': _safe_get(row, 'departamento_codigo'),
                'departamento_nombre': _safe_get(row, 'departamento_nombre', ''),
                'epigrafe_nombre': _safe_get(row, 'epigrafe_nombre', ''),
                
                # Metadatos técnicos
                'tokens_aproximados': _safe_get(row, 'tokens_aproximados', 0),
                'chunk_numero': _safe_get(row, 'chunk_numero', 0),
                'total_chunks_fila': _safe_get(row, 'total_chunks_fila', 1),
                
                # URLs de referencia
                'sumario_url_pdf': _safe_get(row, 'sumario_url_pdf', ''),
                'item_url_pdf': _safe_get(row, 'item_url_pdf', ''),
                
                # Metadatos de origen
                'source_file': parquet_path.name,
                'load_timestamp': time.time()
            }
            
            metadata_list.append(metadata)
        
        logger.info(f"Extraídos {len(metadata_list)} metadatos")
        
        # Validación final
        if len(embeddings) != len(metadata_list):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_list)} metadatos")
        
        return embeddings, metadata_list
        
    except Exception as e:
        logger.error(f"Error cargando {parquet_path}: {e}")
        raise

def _safe_get(row: pd.Series, column: str, default: Any = None) -> Any:
    """
    Extrae valor de una fila de DataFrame de forma segura.
    
    Args:
        row: Fila del DataFrame
        column: Nombre de la columna
        default: Valor por defecto si no existe o es nulo
        
    Returns:
        Valor de la columna o valor por defecto
    """
    if column not in row.index:
        return default
    
    value = row[column]
    
    # Manejar valores nulos/NaN
    if pd.isna(value):
        return default
        
    return value

def validate_embeddings(embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> bool:
    """
    Valida que los embeddings y metadatos sean consistentes.
    
    Args:
        embeddings: Array numpy con embeddings
        metadata: Lista de metadatos
        
    Returns:
        True si la validación pasa
        
    Raises:
        ValueError: Si la validación falla
    """
    logger.info("Validando embeddings y metadatos...")
    
    # Validar forma de embeddings
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings deben ser 2D, encontrado: {embeddings.ndim}D")
    
    n_vectors, dimension = embeddings.shape
    
    # Validar tipo de datos
    if embeddings.dtype != np.float32:
        logger.warning(f"Embeddings no son float32: {embeddings.dtype}, convirtiendo...")
        embeddings = embeddings.astype(np.float32)
    
    # Validar correspondencia con metadatos
    if len(metadata) != n_vectors:
        raise ValueError(f"Número de metadatos ({len(metadata)}) no coincide con embeddings ({n_vectors})")
    
    # Validar que no hay vectores nulos
    null_vectors = np.isnan(embeddings).all(axis=1).sum()
    if null_vectors > 0:
        logger.warning(f"Encontrados {null_vectors} vectores completamente nulos")
    
    # Validar chunk_ids únicos
    chunk_ids = [meta.get('chunk_id') for meta in metadata]
    unique_chunk_ids = set(chunk_ids)
    if len(unique_chunk_ids) != len(chunk_ids):
        duplicates = len(chunk_ids) - len(unique_chunk_ids)
        logger.warning(f"Encontrados {duplicates} chunk_ids duplicados")
    
    # Validar jerarquía de IDs
    missing_ids = []
    for i, meta in enumerate(metadata):
        if not meta.get('sumario_id'):
            missing_ids.append(f"sumario_id en índice {i}")
        if not meta.get('item_id'):
            missing_ids.append(f"item_id en índice {i}")
        if not meta.get('chunk_id'):
            missing_ids.append(f"chunk_id en índice {i}")
    
    if missing_ids:
        logger.warning(f"IDs faltantes: {missing_ids[:10]}...")  # Mostrar solo primeros 10
    
    logger.info(f"Validación completada: {n_vectors} vectores, dimensión {dimension}")
    return True

def build_index_from_parquets(
    parquet_files: List[str], 
    output_index_path: str,
    output_metadata_path: str,
    index_type: str = "IVF",
    dimension: int = 1024
) -> VectorDatabase:
    """
    Construye un índice FAISS desde múltiples archivos parquet.
    
    Args:
        parquet_files: Lista de rutas a archivos parquet
        output_index_path: Ruta donde guardar el índice FAISS
        output_metadata_path: Ruta donde guardar los metadatos
        index_type: Tipo de índice FAISS (default: "IVF")
        dimension: Dimensión esperada de embeddings (default: 1024)
        
    Returns:
        VectorDatabase con índice construido y cargado
    """
    logger.info(f"Construyendo índice {index_type} desde {len(parquet_files)} archivos")
    
    # Validar archivos de entrada
    for file_path in parquet_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    # Crear directorio de salida
    Path(output_index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_metadata_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Crear base de datos vectorial
    db = VectorDatabase(dimension=dimension)
    db.create_index(index_type)
    
    # Procesar archivos uno por uno
    total_vectors = 0
    total_files = len(parquet_files)
    
    logger.info("Procesando archivos parquet...")
    
    for i, parquet_file in enumerate(tqdm(parquet_files, desc="Procesando archivos")):
        try:
            logger.info(f"Procesando archivo {i+1}/{total_files}: {Path(parquet_file).name}")
            
            # Cargar embeddings y metadatos
            embeddings, metadata = load_parquet_with_embeddings(parquet_file)
            
            # Validar datos
            validate_embeddings(embeddings, metadata)
            
            # Agregar al índice
            db.add_vectors(embeddings, metadata)
            
            total_vectors += len(embeddings)
            logger.info(f"Agregados {len(embeddings)} vectores. Total acumulado: {total_vectors}")
            
        except Exception as e:
            logger.error(f"Error procesando {parquet_file}: {e}")
            raise
    
    # Guardar índice
    logger.info(f"Guardando índice con {total_vectors} vectores...")
    db.save_index(output_index_path, output_metadata_path)
    
    # Generar reporte
    construction_report = {
        "total_vectors": total_vectors,
        "total_files_processed": total_files,
        "index_type": index_type,
        "dimension": dimension,
        "files_processed": [Path(f).name for f in parquet_files],
        "output_files": {
            "index": output_index_path,
            "metadata": output_metadata_path
        },
        "construction_timestamp": time.time()
    }
    
    # Guardar reporte
    report_path = Path(output_metadata_path).parent / "construction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(construction_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Índice construido exitosamente: {total_vectors} vectores")
    logger.info(f"Reporte guardado en: {report_path}")
    
    return db

def add_daily_parquet_to_index(
    new_parquet_path: str, 
    index_path: str, 
    metadata_path: str
) -> VectorDatabase:
    """
    Agrega un nuevo archivo parquet diario al índice existente.
    
    Args:
        new_parquet_path: Ruta al nuevo archivo parquet
        index_path: Ruta del índice FAISS existente
        metadata_path: Ruta de metadatos existente
        
    Returns:
        VectorDatabase actualizada
    """
    logger.info(f"Agregando parquet diario: {new_parquet_path}")
    
    # Cargar índice existente
    db = VectorDatabase()
    db.load_index(index_path, metadata_path)
    
    stats_before = db.get_stats()
    logger.info(f"Índice actual: {stats_before['total_vectors']} vectores")
    
    # Procesar nuevo archivo
    embeddings, metadata = load_parquet_with_embeddings(new_parquet_path)
    validate_embeddings(embeddings, metadata)
    
    # Verificar duplicados por chunk_id
    existing_chunk_ids = set(meta.get('chunk_id') for meta in db.metadata)
    new_chunk_ids = set(meta.get('chunk_id') for meta in metadata)
    
    duplicates = existing_chunk_ids.intersection(new_chunk_ids)
    if duplicates:
        logger.warning(f"Encontrados {len(duplicates)} chunk_ids duplicados, se omitirán")
        
        # Filtrar duplicados
        filtered_embeddings = []
        filtered_metadata = []
        
        for i, meta in enumerate(metadata):
            if meta.get('chunk_id') not in existing_chunk_ids:
                filtered_embeddings.append(embeddings[i])
                filtered_metadata.append(meta)
        
        if filtered_embeddings:
            embeddings = np.array(filtered_embeddings, dtype=np.float32)
            metadata = filtered_metadata
        else:
            logger.warning("No hay vectores nuevos para agregar después de filtrar duplicados")
            return db
    
    # Agregar nuevos vectores
    db.add_vectors(embeddings, metadata)
    
    # Guardar índice actualizado
    db.save_index(index_path, metadata_path)
    
    stats_after = db.get_stats()
    new_vectors = stats_after['total_vectors'] - stats_before['total_vectors']
    
    logger.info(f"Agregados {new_vectors} vectores nuevos")
    logger.info(f"Total vectores en índice: {stats_after['total_vectors']}")
    
    return db

def get_parquet_files_from_directory(directory: str, pattern: str = "*.parquet") -> List[str]:
    """
    Obtiene lista de archivos parquet de un directorio.
    
    Args:
        directory: Directorio a buscar
        pattern: Patrón de archivos (default: "*.parquet")
        
    Returns:
        Lista de rutas a archivos parquet
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    
    parquet_files = list(directory.glob(pattern))
    parquet_files = [str(f) for f in parquet_files]
    
    logger.info(f"Encontrados {len(parquet_files)} archivos parquet en {directory}")
    
    return sorted(parquet_files)  # Ordenar para procesamiento consistente
