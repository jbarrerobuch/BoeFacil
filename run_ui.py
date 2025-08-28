#!/usr/bin/env python3
"""
Launcher para la aplicación BoeFacil Streamlit.

Este script facilita el lanzamiento de la aplicación web sin necesidad
de recordar comandos complejos.

Uso:
    python run_ui.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Lanza la aplicación Streamlit."""
    
    # Ruta del archivo principal de Streamlit
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Error: No se encontró el archivo {app_path}")
        sys.exit(1)
    
    print("🚀 Iniciando BoeFacil...")
    print(f"📁 Archivo: {app_path}")
    print("🌐 La aplicación se abrirá en tu navegador automáticamente")
    print("⏹️  Presiona Ctrl+C para detener la aplicación")
    print("-" * 50)
    
    try:
        # Lanzar Streamlit
        cmd = [
            "C:\\Users\\soyel\\miniconda3\\envs\\boefacil\\Scripts\\streamlit.exe",
            "run", 
            str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego! BoeFacil se ha cerrado correctamente.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
