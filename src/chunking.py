#!/usr/bin/env python3
"""
Script para procesar archivos parquet y generar chunks de markdown
Carga todos los archivos parquet de samples/parquet_plano, los combina en un DataFrame
y aplica la función chunking_markdown_df para generar chunks que se guardan en samples/chunks
"""

import os
import pandas as pd
import glob
import logging
from pathlib import Path

# Importar la función de chunking desde el módulo lib
from lib import chunk_utils

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Función principal que carga los parquet files y aplica chunking
    """
    # Definir rutas
    base_path = Path(__file__).parent.parent  # Directorio raíz del proyecto
    parquet_dir = base_path / "samples" / "parquet_plano"
    chunks_dir = base_path / "samples" / "chunks"
    
    logger.info(f"Directorio base: {base_path}")
    logger.info(f"Directorio parquet: {parquet_dir}")
    logger.info(f"Directorio chunks: {chunks_dir}")
    
    # Verificar que existe el directorio de parquet
    if not parquet_dir.exists():
        logger.error(f"El directorio {parquet_dir} no existe")
        return
    
    # Buscar todos los archivos parquet
    parquet_pattern = str(parquet_dir / "*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    
    if not parquet_files:
        logger.error(f"No se encontraron archivos parquet en {parquet_dir}")
        return
    
    logger.info(f"Encontrados {len(parquet_files)} archivos parquet")
    
    # Cargar y combinar todos los archivos parquet
    dataframes = []
    for file_path in parquet_files:
        try:
            logger.info(f"Cargando {os.path.basename(file_path)}...")
            df = pd.read_parquet(file_path)
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Error cargando {file_path}: {e}")
            continue
    
    if not dataframes:
        logger.error("No se pudo cargar ningún archivo parquet")
        return
    
    # Combinar todos los DataFrames
    logger.info("Combinando DataFrames...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"DataFrame combinado: {combined_df.shape[0]} filas, {combined_df.shape[1]} columnas")
    logger.info(f"Columnas disponibles: {list(combined_df.columns)}")
    
    # Verificar que existe la columna markdown
    if 'markdown' not in combined_df.columns:
        logger.error("No se encontró la columna 'markdown' en el DataFrame")
        logger.info(f"Columnas disponibles: {list(combined_df.columns)}")
        return
    
    # Aplicar chunking con los parámetros especificados
    logger.info("Aplicando chunking_markdown_df...")
    chunks = chunk_utils.chunking_markdown_df(
        df=combined_df,
        columna_markdown='markdown',
        max_tokens=1000,
        texto_id=None,  # Se extraerá automáticamente de 'item_id'
        store_path=str(chunks_dir)
    )
    
    logger.info(f"Proceso completado. Generados {len(chunks)} chunks totales")
    
    # Mostrar estadísticas
    if chunks:
        chunks_guardados = sum(1 for chunk in chunks if 'archivo_guardado' in chunk)
        logger.info(f"Chunks guardados en archivos: {chunks_guardados}")
        
        # Estadísticas por tokens
        tokens_stats = [chunk['tokens_aproximados'] for chunk in chunks]
        logger.info(f"Tokens por chunk - Promedio: {sum(tokens_stats)/len(tokens_stats):.1f}, "
                   f"Mín: {min(tokens_stats)}, Máx: {max(tokens_stats)}")
    
    # Crear directorio para guardar los archivos parquet de chunks
    parquet_chunks_dir = base_path / "samples" / "parquet_chunks"
    os.makedirs(parquet_chunks_dir, exist_ok=True)
    logger.info(f"Directorio parquet_chunks creado/verificado: {parquet_chunks_dir}")

    # Guardar los chunks en un archivo parquet
    logger.info("Guardando chunks en archivo parquet...")
    chunks_df = pd.DataFrame(chunks)  # Convertir la lista de chunks a un DataFrame
    parquet_file_path = parquet_chunks_dir / "chunks.parquet"
    try:
        chunks_df.to_parquet(parquet_file_path, index=False)
        logger.info(f"Chunks guardados en: {parquet_file_path}")
    except Exception as e:
        logger.error(f"Error guardando los chunks en archivo parquet: {e}")

if __name__ == "__main__":
    main()