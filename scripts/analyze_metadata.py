#!/usr/bin/env python3
"""
Script para analizar la estructura del metadata.json de FAISS
"""

import json
from pathlib import Path

def analyze_metadata():
    metadata_path = Path('indices/metadata.json')
    
    if not metadata_path.exists():
        print("❌ No se encuentra el archivo metadata.json")
        return
    
    print("📊 Analizando estructura del metadata.json...")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Tipo de datos: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Número de entradas: {len(data)}")
        
        # Obtener una muestra de claves
        keys = list(data.keys())
        print(f"Ejemplos de claves: {keys[:3]}")
        
        # Analizar el primer elemento
        if keys:
            first_key = keys[0]
            first_value = data[first_key]
            print(f"\nEstructura del primer elemento:")
            print(f"Clave: {first_key}")
            print(f"Tipo de valor: {type(first_value)}")
            
            if isinstance(first_value, dict):
                print(f"Campos disponibles: {list(first_value.keys())}")
                
                # Mostrar algunos campos importantes
                for field in ['texto', 'item_titulo', 'seccion_nombre', 'departamento_nombre']:
                    if field in first_value:
                        content = str(first_value[field])
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"{field}: {preview}")
            else:
                print(f"Contenido: {str(first_value)[:200]}...")
    
    elif isinstance(data, list):
        print(f"Es una lista con {len(data)} elementos")
        if data:
            print(f"Tipo del primer elemento: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"Campos del primer elemento: {list(data[0].keys())}")

if __name__ == "__main__":
    analyze_metadata()
