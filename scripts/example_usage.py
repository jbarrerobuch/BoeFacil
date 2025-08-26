#!/usr/bin/env python3
"""
Script de ejemplo para usar la base de datos vectorial BoeFacil.

Este script muestra cómo:
1. Construir un índice desde archivos parquet
2. Realizar búsquedas semánticas 
3. Agregar nuevos datos diariamente

Uso:
    python scripts/example_usage.py
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lib.vector_db import VectorDatabase
from lib.index_builder import build_index_from_parquets, get_parquet_files_from_directory
import numpy as np

def example_build_index():
    """Ejemplo: Construir índice desde archivos parquet existentes."""
    print("=== Ejemplo 1: Construir índice desde parquets ===")
    
    # Obtener archivos parquet
    parquet_files = get_parquet_files_from_directory("samples")
    print(f"Archivos encontrados: {[Path(f).name for f in parquet_files]}")
    
    if not parquet_files:
        print("No se encontraron archivos parquet en samples/")
        return None
    
    # Construir índice
    db = build_index_from_parquets(
        parquet_files=parquet_files,
        output_index_path="indices/boe_index.faiss",
        output_metadata_path="indices/metadata.json",
        index_type="IVF"
    )
    
    stats = db.get_stats()
    print(f"Índice construido: {stats['total_vectors']} vectores")
    
    return db

def example_load_and_search():
    """Ejemplo: Cargar índice existente y realizar búsquedas."""
    print("\n=== Ejemplo 2: Cargar índice y buscar ===")
    
    # Verificar que existe el índice
    index_path = "indices/boe_index.faiss"
    metadata_path = "indices/metadata.json"
    
    if not Path(index_path).exists():
        print(f"Índice no encontrado en {index_path}")
        print("Ejecuta primero example_build_index()")
        return
    
    # Cargar índice
    db = VectorDatabase()
    db.load_index(index_path, metadata_path)
    
    stats = db.get_stats()
    print(f"Índice cargado: {stats['total_vectors']} vectores")
    
    # Ejemplo de búsqueda (con vector aleatorio como demo)
    print("\n--- Ejemplo de búsqueda ---")
    query_vector = np.random.random(1024).astype(np.float32)
    
    # Búsqueda simple
    results = db.search(query_vector, k=5)
    
    print(f"Resultados encontrados: {len(results)}")
    for i, result in enumerate(results):
        print(f"{i+1}. Similarity: {result['similarity_score']:.3f}")
        print(f"   Chunk ID: {result.get('chunk_id', 'N/A')}")
        print(f"   Título: {result.get('item_titulo', 'N/A')[:100]}...")
        print(f"   Fecha: {result.get('fecha_publicacion', 'N/A')}")
        print()

def example_filtered_search():
    """Ejemplo: Búsqueda con filtros por metadatos."""
    print("\n=== Ejemplo 3: Búsqueda con filtros ===")
    
    # Cargar índice
    db = VectorDatabase()
    db.load_index("indices/boe_index.faiss", "indices/metadata.json")
    
    # Vector de consulta de ejemplo
    query_vector = np.random.random(1024).astype(np.float32)
    
    # Búsqueda filtrada por fecha
    filters = {
        'fecha_publicacion': '2023-12-29'  # Solo documentos de esta fecha
    }
    
    results = db.search(query_vector, k=5, filters=filters)
    
    print(f"Resultados filtrados por fecha: {len(results)}")
    for result in results:
        print(f"- {result.get('chunk_id')}: {result.get('fecha_publicacion')}")

def example_daily_update():
    """Ejemplo: Agregar nuevo archivo parquet diario."""
    print("\n=== Ejemplo 4: Actualización diaria ===")
    
    # Este sería el flujo para agregar un nuevo BOE diario
    # (requiere que tengas un archivo nuevo para probar)
    
    new_parquet = "samples/boe_data_20250827.parquet"  # Archivo hipotético
    
    if Path(new_parquet).exists():
        from lib.index_builder import add_daily_parquet_to_index
        
        db = add_daily_parquet_to_index(
            new_parquet,
            "indices/boe_index.faiss", 
            "indices/metadata.json"
        )
        
        stats = db.get_stats()
        print(f"Índice actualizado: {stats['total_vectors']} vectores totales")
    else:
        print(f"Archivo {new_parquet} no existe (ejemplo teórico)")
        print("Para probar:")
        print("1. Genera un nuevo archivo parquet con embeddings")
        print("2. Ejecuta add_daily_parquet_to_index() con ese archivo")

def main():
    """Ejecuta todos los ejemplos."""
    print("=== Ejemplos de Uso - Base de Datos Vectorial BoeFacil ===\n")
    
    try:
        # Ejemplo 1: Construir índice
        db = example_build_index()
        
        if db is not None:
            # Ejemplo 2: Cargar y buscar
            example_load_and_search()
            
            # Ejemplo 3: Búsqueda con filtros
            example_filtered_search()
            
            # Ejemplo 4: Actualización diaria
            example_daily_update()
        
        print("\n=== Ejemplos completados ===")
        print("Tu base de datos vectorial está lista para usar!")
        
    except Exception as e:
        print(f"Error en los ejemplos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
