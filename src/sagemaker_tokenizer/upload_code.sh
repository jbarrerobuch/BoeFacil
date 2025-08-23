#!/bin/bash
# Script para preparar y subir el código fuente a S3

# Variables de configuración
S3_CODE_LOCATION="s3://boe-facil/job_embeddings/code/"

# Crear directorio temporal para los archivos
echo "Creando directorio temporal para el código..."
TMP_DIR=$(mktemp -d)

echo "Copiando archivos al directorio temporal..."
# Copiar el script principal
cp tokenizer.py "${TMP_DIR}/"

# Crear directorio para módulos
mkdir -p "${TMP_DIR}/modules"

# Copiar los módulos
cp modules/__init__.py "${TMP_DIR}/modules/"
cp modules/checkpoint.py "${TMP_DIR}/modules/"
cp modules/performance.py "${TMP_DIR}/modules/"
cp modules/utils.py "${TMP_DIR}/modules/"
cp modules/s3_utils.py "${TMP_DIR}/modules/"

echo "Subiendo archivos a S3..."
# Subir el script principal
aws s3 cp "${TMP_DIR}/tokenizer.py" "${S3_CODE_LOCATION}tokenizer.py"

# Subir los módulos
aws s3 cp "${TMP_DIR}/modules" "${S3_CODE_LOCATION}modules" --recursive

echo "Limpiando directorio temporal..."
rm -rf "${TMP_DIR}"

echo "Código subido exitosamente a ${S3_CODE_LOCATION}"
echo "Ahora puedes ejecutar job_maker.sh para crear el trabajo de entrenamiento"
