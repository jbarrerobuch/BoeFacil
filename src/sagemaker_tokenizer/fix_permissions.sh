#!/bin/bash
# Script para corregir los permisos de los archivos en CloudShell

echo "=== Corrigiendo permisos de archivos y directorios ==="
echo "Este script otorgará permisos de lectura, escritura y ejecución a todos los archivos"

# Obtener el directorio actual
CURRENT_DIR=$(pwd)
echo "Directorio actual: $CURRENT_DIR"

# Confirmar con el usuario
echo -n "¿Desea corregir los permisos de todos los archivos en este directorio? (s/n): "
read CONFIRM

if [ "$CONFIRM" != "s" ] && [ "$CONFIRM" != "S" ]; then
    echo "Operación cancelada por el usuario."
    exit 0
fi

echo "Corrigiendo permisos..."

# Corregir permisos de directorios
find . -type d -exec chmod 755 {} \;
echo "✅ Permisos de directorios corregidos (755 - rwxr-xr-x)"

# Corregir permisos de archivos
find . -type f -exec chmod 644 {} \;
echo "✅ Permisos básicos de archivos corregidos (644 - rw-r--r--)"

# Otorgar permisos de ejecución a scripts .sh
find . -name "*.sh" -exec chmod 755 {} \;
echo "✅ Permisos de ejecución añadidos a scripts .sh (755 - rwxr-xr-x)"

# Listar archivos sh en este directorio
echo ""
echo "Scripts .sh encontrados:"
find . -maxdepth 1 -name "*.sh" -type f | sort

echo ""
echo "Para ejecutar los scripts, use:"
echo "./nombre_del_script.sh"
echo ""
echo "=== Corrección de permisos completada ==="
