#!/usr/bin/env python3
"""
Script de entrada para SageMaker Training Job.
Este script importa y ejecuta el script tokenizer principal.
"""
import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run.py")

def main():
    """Punto de entrada principal"""
    logger.info("=== INICIANDO SCRIPT TOKENIZER PARA SAGEMAKER ===")
    
    # Mostrar información del entorno para diagnóstico
    logger.info(f"Directorio actual: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Contenido del directorio actual: {os.listdir('.')}")
    
    # Añadir el directorio actual al path de Python
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Intentar importar directamente el archivo tokenizer.py
        logger.info("Intentando importar tokenizer directamente...")
        import tokenizer
        tokenizer_main = tokenizer.main
    except ImportError as e:
        logger.warning(f"Importación directa falló: {e}")
        logger.info("Intentando importación desde paquete tokenizer...")
        
        try:
            from tokenizer import tokenizer as tokenizer_module
            tokenizer_main = tokenizer_module.main
        except ImportError as e:
            logger.error(f"Importación desde paquete falló: {e}")
            logger.info("Buscando archivo tokenizer.py en el sistema...")
            
            # Buscar el archivo en el sistema de archivos
            py_files = list(Path('.').glob('**/*.py'))
            logger.info(f"Archivos Python encontrados: {py_files}")
            
            raise ImportError(f"No se pudo encontrar tokenizer.py en ninguna ubicación: {e}")
    
    # Ejecutar la función principal
    logger.info("Ejecutando función principal de tokenizer")
    tokenizer_main()
    
    logger.info("=== EJECUCIÓN COMPLETADA ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"ERROR EN EJECUCIÓN: {e}")
        sys.exit(1)
