#!/bin/bash
# Script para verificar la configuración

echo "Verificando archivos y directorios..."

# Verificar estructura modular
if [ -d "modules" ]; then
  echo "✅ Directorio 'modules' encontrado"
  
  # Verificar archivos en el directorio 'modules'
  MODULE_FILES=("__init__.py" "checkpoint.py" "performance.py" "utils.py" "s3_utils.py")
  for file in "${MODULE_FILES[@]}"; do
    if [ -f "modules/$file" ]; then
      echo "  ✅ Módulo 'modules/$file' existe"
    else
      echo "  ❌ Módulo 'modules/$file' NO encontrado"
    fi
  done
else
  echo "❌ Directorio 'modules' NO encontrado"
fi

# Verificar archivos principales
MAIN_FILES=("tokenizer.py" "__init__.py" "README.md" "package_and_upload.sh" "job_maker.sh")
for file in "${MAIN_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "✅ Archivo '$file' encontrado"
  else
    echo "❌ Archivo '$file' NO encontrado"
  fi
done

# Verificar archivos obsoletos
OBSOLETE_FILES=("s3_utils.py" "sm_utils.py")
for file in "${OBSOLETE_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "⚠️  Archivo '$file' obsoleto encontrado - considere eliminarlo"
  else
    echo "✅ Archivo obsoleto '$file' correctamente eliminado"
  fi
done

echo ""
echo "Estructura actual:"
find . -type f -name "*.py" -o -name "*.sh" | sort

echo ""
echo "Para continuar el proceso:"
echo "1. Suba esta carpeta completa a AWS CloudShell:"
echo "   - En CloudShell: Haga clic en 'Acciones' -> 'Cargar archivo'"
echo "   - Para subir la carpeta completa: Comprima la carpeta sagemaker_tokenizer como ZIP"
echo "   - Suba el archivo ZIP y descomprímalo en CloudShell con: unzip nombre-archivo.zip"
echo "   - Alternativamente, suba cada archivo individualmente seleccionando múltiples archivos"
echo ""
echo "2. Ejecute los siguientes comandos en CloudShell:"
echo "   chmod +x *.sh                           # Dar permisos de ejecución"
echo "   ./verify_structure.sh                  # Verificar estructura"
echo "   ./package_and_upload.sh                # Empaquetar y subir código"
echo "   ./job_maker.sh                         # Crear y lanzar el Training Job"
