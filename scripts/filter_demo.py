#!/usr/bin/env python3
"""
Ejemplos de uso del sistema de filtros avanzados con formato de fecha BOE correcto.

Este script demuestra cómo usar el sistema de filtros con fechas en formato YYYYMMDD
que es el formato real usado en los documentos BOE.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lib.advanced_filter import AdvancedFilter

def demo_date_formats():
    """Demuestra el uso correcto de formatos de fecha."""
    print("=== Demostración de Formatos de Fecha BOE ===\n")
    
    filter_system = AdvancedFilter()
    
    # Mostrar ayuda de formatos
    print("📅 Formatos de fecha soportados:")
    help_info = filter_system.get_date_format_help()
    print(f"- Documentos BOE: {help_info['documento_format']}")
    for fmt in help_info['input_formats']:
        print(f"- Entrada: {fmt}")
    print()
    
    # Ejemplos de conversión
    print("🔄 Ejemplos de conversión:")
    test_dates = [
        "2023-01-01",    # Formato estándar
        "2023-12-29",    # Formato estándar
        "20230101",      # Formato BOE directo
        "20231229"       # Formato BOE directo
    ]
    
    for date in test_dates:
        converted = AdvancedFilter.convert_date_to_boe_format(date)
        print(f"  {date} → {converted}")
    print()

def demo_date_range_examples():
    """Ejemplos de rangos de fecha para usar en búsquedas."""
    print("📋 Ejemplos de rangos de fecha para búsquedas:\n")
    
    examples = [
        {
            "desc": "Buscar en enero 2023",
            "range": {'start_date': '2023-01-01', 'end_date': '2023-01-31'}
        },
        {
            "desc": "Buscar desde diciembre 2023",
            "range": {'start_date': '2023-12-01'}
        },
        {
            "desc": "Buscar hasta febrero 2023",
            "range": {'end_date': '2023-02-28'}
        },
        {
            "desc": "Buscar día específico (formato BOE)",
            "range": {'start_date': '20231229', 'end_date': '20231229'}
        },
        {
            "desc": "Buscar entre dos fechas (formatos mixtos)",
            "range": {'start_date': '2023-01-01', 'end_date': '20230215'}
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['desc']}")
        print(f"   date_range = {example['range']}")
        print()

def demo_combined_filters():
    """Ejemplos de filtros combinados incluyendo fechas."""
    print("🔍 Ejemplos de filtros avanzados combinados:\n")
    
    examples = [
        {
            "desc": "Ministerio de Hacienda en Q4 2023",
            "filters": {
                'departamento_nombre_contains': 'HACIENDA'
            },
            "date_range": {
                'start_date': '2023-10-01',
                'end_date': '2023-12-31'
            }
        },
        {
            "desc": "Disposiciones Generales documentos largos",
            "filters": {
                'seccion_nombre_contains': 'DISPOSICIONES',
                'tokens_min': 1000
            },
            "date_range": {
                'start_date': '2023-01-01'
            }
        },
        {
            "desc": "BOE específico del 29 de diciembre",
            "filters": {
                'chunk_numero_max': 3  # Solo primeros 3 chunks
            },
            "date_range": {
                'start_date': '20231229',
                'end_date': '20231229'
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['desc']}")
        if 'filters' in example:
            print(f"   filters = {example['filters']}")
        if 'date_range' in example:
            print(f"   date_range = {example['date_range']}")
        print()

def demo_filter_validation():
    """Demuestra la validación de filtros."""
    print("✅ Demostración de validación de filtros:\n")
    
    filter_system = AdvancedFilter()
    
    # Filtros válidos
    valid_filters = {
        'departamento_nombre_contains': 'HACIENDA',
        'tokens_min': 500,
        'seccion_codigo': ['I', 'II']
    }
    
    # Filtros inválidos
    invalid_filters = {
        'invalid_filter': 'test',
        'tokens_min': 'not_a_number',
        'departamento_nombre_contains': 'HACIENDA'  # Este es válido
    }
    
    print("🟢 Validando filtros válidos:")
    result = filter_system.validate_filters(valid_filters)
    print(f"   Válidos: {result['valid_filters']}")
    print(f"   Inválidos: {result['invalid_filters']}")
    print()
    
    print("🔴 Validando filtros con errores:")
    result = filter_system.validate_filters(invalid_filters)
    print(f"   Válidos: {result['valid_filters']}")
    print(f"   Inválidos: {result['invalid_filters']}")
    print()

def main():
    """Ejecuta todas las demostraciones."""
    print("🎯 Sistema de Filtros Avanzados BOE - Ejemplos de Uso\n")
    print("=" * 60)
    
    try:
        demo_date_formats()
        print("=" * 60)
        
        demo_date_range_examples()
        print("=" * 60)
        
        demo_combined_filters()
        print("=" * 60)
        
        demo_filter_validation()
        print("=" * 60)
        
        print("✅ Todas las demostraciones completadas exitosamente!")
        print("\n💡 Consejos:")
        print("- Usa formato YYYY-MM-DD para fechas de entrada (más legible)")
        print("- El sistema convierte automáticamente al formato BOE YYYYMMDD")
        print("- Combina filtros de fecha con filtros de metadatos para búsquedas precisas")
        print("- Usa la validación de filtros para detectar errores antes de buscar")
        
    except Exception as e:
        print(f"❌ Error en las demostraciones: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
