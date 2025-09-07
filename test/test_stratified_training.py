"""
Script de prueba para validar el entrenamiento estratificado de índices IVF.

Este script prueba la nueva funcionalidad de muestreo estratificado
implementada en index_builder.py para entrenar índices IVF.
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np

# Agregar src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from lib.index_builder import (
    build_index_from_parquets, 
    get_parquet_files_from_directory,
    create_stratified_sample,
    load_parquet_with_embeddings
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stratified_sampling():
    """
    Prueba la función de muestreo estratificado con datos reales.
    """
    logger.info("=== Prueba de Muestreo Estratificado ===")
    
    # Buscar archivos parquet en el directorio samples
    samples_dir = Path(__file__).parent.parent / "samples"
    parquet_files = get_parquet_files_from_directory(str(samples_dir))
    
    if not parquet_files:
        logger.error("No se encontraron archivos parquet en samples/")
        return False
    
    # Usar solo el primer archivo para la prueba
    test_file = parquet_files[0]
    logger.info(f"Usando archivo de prueba: {Path(test_file).name}")
    
    try:
        # Cargar datos
        embeddings, metadata = load_parquet_with_embeddings(test_file)
        logger.info(f"Cargados {len(embeddings)} vectores para prueba")
        
        # Crear muestra estratificada
        sample_embeddings, sample_metadata, remaining_embeddings, remaining_metadata = create_stratified_sample(
            embeddings, 
            metadata, 
            sample_ratio=0.3,
            stratify_keys=['seccion_codigo', 'departamento_codigo']
        )
        
        # Verificar resultados
        total_original = len(embeddings)
        total_sample = len(sample_embeddings)
        total_remaining = len(remaining_embeddings)
        
        logger.info(f"Resultados del muestreo:")
        logger.info(f"  Original: {total_original} vectores")
        logger.info(f"  Muestra: {total_sample} vectores ({total_sample/total_original*100:.1f}%)")
        logger.info(f"  Restantes: {total_remaining} vectores ({total_remaining/total_original*100:.1f}%)")
        
        # Verificar que no hay superposición
        sample_chunk_ids = set(meta['chunk_id'] for meta in sample_metadata)
        remaining_chunk_ids = set(meta['chunk_id'] for meta in remaining_metadata)
        overlap = sample_chunk_ids.intersection(remaining_chunk_ids)
        
        if overlap:
            logger.error(f"Error: {len(overlap)} chunk_ids duplicados entre muestra y restantes")
            return False
        else:
            logger.info("✓ No hay superposición entre muestra y datos restantes")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en prueba de muestreo estratificado: {e}")
        return False

def test_ivf_training():
    """
    Prueba la construcción de índice IVF con entrenamiento estratificado.
    """
    logger.info("=== Prueba de Construcción de Índice IVF ===")
    
    # Buscar archivos parquet
    samples_dir = Path(__file__).parent.parent / "samples"
    parquet_files = get_parquet_files_from_directory(str(samples_dir))
    
    if not parquet_files:
        logger.error("No se encontraron archivos parquet en samples/")
        return False
    
    # Usar solo los primeros archivos para la prueba (limitar para que sea rápido)
    test_files = parquet_files[:2] if len(parquet_files) > 1 else parquet_files
    logger.info(f"Usando {len(test_files)} archivos para prueba")
    
    # Rutas de salida para prueba
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    index_path = output_dir / "test_ivf_index.faiss"
    metadata_path = output_dir / "test_metadata.json"
    
    try:
        # Construir índice IVF con entrenamiento estratificado
        db = build_index_from_parquets(
            parquet_files=test_files,
            output_index_path=str(index_path),
            output_metadata_path=str(metadata_path),
            index_type="IVF",
            dimension=1024,
            training_sample_ratio=0.3
        )
        
        # Verificar estadísticas
        stats = db.get_stats()
        logger.info(f"Índice IVF construido exitosamente:")
        logger.info(f"  Vectores totales: {stats['total_vectors']}")
        logger.info(f"  Tipo: {stats['index_type']}")
        logger.info(f"  Entrenado: {stats['is_trained']}")
        logger.info(f"  Clusters (nlist): {stats.get('nlist', 'N/A')}")
        
        # Verificar que el índice está entrenado
        if not stats['is_trained']:
            logger.error("Error: El índice IVF no está entrenado")
            return False
        
        logger.info("✓ Índice IVF entrenado correctamente")
        
        # Hacer una búsqueda de prueba
        if stats['total_vectors'] > 0:
            # Usar el primer vector como consulta de prueba
            test_vector = np.random.random((1024,)).astype(np.float32)
            results = db.search(test_vector, k=5)
            logger.info(f"✓ Búsqueda de prueba completada: {len(results)} resultados")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en construcción de índice IVF: {e}")
        return False
    
    finally:
        # Limpiar archivos de prueba
        try:
            if index_path.exists():
                index_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            (output_dir / "construction_report.json").unlink(missing_ok=True)
            output_dir.rmdir()
        except:
            pass

def main():
    """
    Ejecuta todas las pruebas.
    """
    logger.info("Iniciando pruebas de entrenamiento estratificado...")
    
    success = True
    
    # Prueba 1: Muestreo estratificado
    if not test_stratified_sampling():
        success = False
    
    # Prueba 2: Construcción de índice IVF
    if not test_ivf_training():
        success = False
    
    if success:
        logger.info("🎉 Todas las pruebas pasaron exitosamente")
        return 0
    else:
        logger.error("❌ Algunas pruebas fallaron")
        return 1

if __name__ == "__main__":
    exit(main())
