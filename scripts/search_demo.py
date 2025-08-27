#!/usr/bin/env python3
"""
Demo interactivo del sistema de búsqueda BOE.

Este script proporciona una interfaz de línea de comandos para probar todas las
funcionalidades del motor de búsqueda BOE de forma interactiva.

Uso:
    # Demo interactivo
    python scripts/search_demo.py
    
    # Búsqueda directa desde línea de comandos
    python scripts/search_demo.py --query "impuestos sociedades" --limit 5
"""

import argparse
import sys
from pathlib import Path
import json

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lib.boe_search_api import BOESearchAPI
from lib.search_engine import BOESearchEngine

def format_result(result: dict, index: int) -> str:
    """Formatea un resultado de búsqueda para mostrar."""
    lines = []
    lines.append(f"\n=== Resultado {index + 1} ===")
    lines.append(f"Chunk ID: {result.get('chunk_id', 'N/A')}")
    lines.append(f"Similarity: {result.get('similarity_score', 0):.3f}")
    lines.append(f"Fecha: {result.get('fecha_publicacion', 'N/A')}")
    lines.append(f"Sección: {result.get('seccion_nombre', 'N/A')}")
    lines.append(f"Departamento: {result.get('departamento_nombre', 'N/A')}")
    lines.append(f"Título: {result.get('item_titulo', 'N/A')[:100]}...")
    lines.append(f"Tokens: {result.get('tokens_aproximados', 'N/A')}")
    
    # Mostrar extracto del texto
    texto = result.get('texto', '')
    if texto:
        extracto = texto[:300] + "..." if len(texto) > 300 else texto
        lines.append(f"Texto: {extracto}")
    
    return "\n".join(lines)

def demo_basic_search(api: BOESearchAPI):
    """Demuestra búsqueda básica."""
    print("\n" + "="*50)
    print("DEMO: Búsqueda Básica")
    print("="*50)
    
    queries = [
        "Real decreto impuestos sociedades",
        "ministerio hacienda presupuesto",
        "disposiciones generales tributarias"
    ]
    
    for query in queries:
        print(f"\n🔍 Búsqueda: '{query}'")
        results = api.search(query, limit=3)
        
        if results:
            for i, result in enumerate(results):
                print(format_result(result, i))
        else:
            print("No se encontraron resultados.")
        
        input("\nPresiona Enter para continuar...")

def demo_date_search(api: BOESearchAPI):
    """Demuestra búsqueda por fechas."""
    print("\n" + "="*50)
    print("DEMO: Búsqueda por Fechas")
    print("="*50)
    
    print("\n🗓️ Documentos del 29 de diciembre 2023:")
    results = api.find_by_date("2023-12-29", limit=3)
    
    if results:
        for i, result in enumerate(results):
            print(format_result(result, i))
    else:
        print("No se encontraron documentos en esa fecha.")
    
    input("\nPresiona Enter para continuar...")
    
    print("\n🗓️ Documentos de diciembre 2023 sobre 'hacienda':")
    results = api.find_by_date_range("2023-12-01", "2023-12-31", "hacienda", limit=3)
    
    if results:
        for i, result in enumerate(results):
            print(format_result(result, i))
    else:
        print("No se encontraron documentos en ese rango.")
    
    input("\nPresiona Enter para continuar...")

def demo_ministry_search(api: BOESearchAPI):
    """Demuestra búsqueda por ministerio."""
    print("\n" + "="*50)
    print("DEMO: Búsqueda por Ministerio")
    print("="*50)
    
    print("\n🏛️ Documentos del Ministerio de Hacienda:")
    results = api.find_by_ministry("HACIENDA", limit=3)
    
    if results:
        for i, result in enumerate(results):
            print(format_result(result, i))
    else:
        print("No se encontraron documentos de ese ministerio.")
    
    input("\nPresiona Enter para continuar...")

def demo_advanced_search(api: BOESearchAPI):
    """Demuestra búsqueda avanzada."""
    print("\n" + "="*50)
    print("DEMO: Búsqueda Avanzada")
    print("="*50)
    
    print("\n🔬 Búsqueda avanzada: Hacienda + Sección I + Documentos largos")
    results = api.advanced_search(
        query="impuestos",
        ministry="HACIENDA",
        section="I",
        min_tokens=500,
        limit=3
    )
    
    if results:
        for i, result in enumerate(results):
            print(format_result(result, i))
    else:
        print("No se encontraron documentos con esos criterios.")
    
    input("\nPresiona Enter para continuar...")

def demo_similar_search(api: BOESearchAPI):
    """Demuestra búsqueda de similares."""
    print("\n" + "="*50)
    print("DEMO: Búsqueda de Documentos Similares")
    print("="*50)
    
    # Primero encontrar un documento
    print("\n🔍 Buscando un documento de referencia...")
    results = api.search("real decreto", limit=1)
    
    if results:
        reference = results[0]
        chunk_id = reference.get('chunk_id')
        print(f"\n📄 Documento de referencia: {chunk_id}")
        print(f"Título: {reference.get('item_titulo', 'N/A')[:100]}...")
        
        print(f"\n🔗 Documentos similares a {chunk_id}:")
        similar = api.find_similar(chunk_id, limit=3)
        
        if similar:
            for i, result in enumerate(similar):
                print(format_result(result, i))
        else:
            print("No se encontraron documentos similares.")
    else:
        print("No se encontró documento de referencia.")
    
    input("\nPresiona Enter para continuar...")

def interactive_mode(api: BOESearchAPI):
    """Modo interactivo para búsquedas."""
    print("\n" + "="*50)
    print("MODO INTERACTIVO")
    print("="*50)
    print("Escribe 'help' para ver comandos disponibles")
    print("Escribe 'quit' para salir")
    
    while True:
        try:
            query = input("\n🔍 Búsqueda: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'salir']:
                print("¡Hasta luego!")
                break
            
            if query.lower() == 'help':
                print_help()
                continue
            
            if query.lower() == 'stats':
                stats = api.get_stats()
                print(f"\n📊 Estadísticas:")
                print(f"Total documentos: {stats['index_stats'].get('total_vectors', 'N/A')}")
                print(f"Modelo: {stats.get('model_name', 'N/A')}")
                print(f"Dimensión: {stats.get('embedding_dimension', 'N/A')}")
                continue
            
            if query.lower().startswith('ministerios'):
                print("\n🏛️ Ministerios disponibles:")
                ministries = api.get_available_ministries(limit=20)
                for ministry in ministries:
                    print(f"  - {ministry}")
                continue
            
            if query.lower().startswith('secciones'):
                print("\n📑 Secciones disponibles:")
                sections = api.get_available_sections()
                for section in sections:
                    print(f"  - {section['codigo']}: {section['nombre']}")
                continue
            
            # Búsqueda normal
            results = api.search(query, limit=5)
            
            if results:
                print(f"\n📄 Encontrados {len(results)} resultados:")
                for i, result in enumerate(results):
                    print(format_result(result, i))
            else:
                print("❌ No se encontraron resultados.")
                
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_help():
    """Muestra ayuda del modo interactivo."""
    print("""
📖 Comandos disponibles:

Búsquedas:
  <texto>              - Búsqueda semántica
  
Información:
  stats                - Estadísticas del índice
  ministerios          - Lista ministerios disponibles
  secciones            - Lista secciones disponibles
  help                 - Mostrar esta ayuda
  quit/exit/salir      - Salir

Ejemplos de búsquedas:
  real decreto impuestos
  ministerio hacienda
  disposiciones generales 2023
    """)

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Demo del sistema de búsqueda BOE")
    
    parser.add_argument(
        "--index-path",
        default="indices/boe_index.faiss",
        help="Ruta al índice FAISS"
    )
    
    parser.add_argument(
        "--metadata-path", 
        default="indices/metadata.json",
        help="Ruta a metadatos"
    )
    
    parser.add_argument(
        "--query",
        help="Realizar búsqueda directa desde línea de comandos"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Número máximo de resultados"
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo" if "--query" not in sys.argv else "direct",
        help="Modo de ejecución"
    )
    
    args = parser.parse_args()
    
    # Verificar que existen los archivos de índice
    if not Path(args.index_path).exists():
        print(f"❌ Error: Índice no encontrado en {args.index_path}")
        print("💡 Ejecuta primero: python scripts/build_vector_index.py --input-dir samples --output-dir indices")
        return 1
    
    if not Path(args.metadata_path).exists():
        print(f"❌ Error: Metadatos no encontrados en {args.metadata_path}")
        return 1
    
    try:
        print("🚀 Inicializando motor de búsqueda BOE...")
        api = BOESearchAPI(args.index_path, args.metadata_path)
        print("✅ Motor de búsqueda listo!")
        
        stats = api.get_stats()
        print(f"📊 {stats['index_stats'].get('total_vectors', 'N/A')} documentos indexados")
        
    except Exception as e:
        print(f"❌ Error inicializando motor de búsqueda: {e}")
        return 1
    
    # Búsqueda directa desde línea de comandos
    if args.query:
        print(f"\n🔍 Búsqueda: '{args.query}'")
        results = api.search(args.query, limit=args.limit)
        
        if results:
            print(f"\n📄 Encontrados {len(results)} resultados:")
            for i, result in enumerate(results):
                print(format_result(result, i))
        else:
            print("❌ No se encontraron resultados.")
        
        return 0
    
    # Modo demo completo
    if args.mode == "demo":
        print("\n🎯 Iniciando demos del sistema de búsqueda...")
        
        demo_basic_search(api)
        demo_date_search(api)
        demo_ministry_search(api)
        demo_advanced_search(api)
        demo_similar_search(api)
        
        print("\n✅ Demos completados!")
        
        # Preguntar si quiere modo interactivo
        response = input("\n¿Quieres probar el modo interactivo? [y/N]: ")
        if response.lower() == 'y':
            interactive_mode(api)
    
    # Modo interactivo directo
    elif args.mode == "interactive":
        interactive_mode(api)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
