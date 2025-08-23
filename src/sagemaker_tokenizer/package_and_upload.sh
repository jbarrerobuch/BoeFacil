#!/bin/bash
# Script para preparar y subir el código fuente a S3 como un paquete Python

# Variables de configuración
S3_CODE_LOCATION="s3://boe-facil/job_embeddings/code/"

# Crear directorio temporal para los archivos
echo "Creando directorio temporal para el código..."
TMP_DIR=$(mktemp -d)
PKG_DIR="${TMP_DIR}/sagemaker_package"

echo "Preparando estructura de paquete Python..."
mkdir -p "${PKG_DIR}"

# Crear archivo setup.py para el paquete
cat > "${PKG_DIR}/setup.py" << EOF
from setuptools import setup, find_packages

setup(
    name="tokenizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=3.3.0",
        "numpy<2.0",
        "psutil"
    ],
)
EOF

# Crear directorio para el paquete tokenizer
mkdir -p "${PKG_DIR}/tokenizer"
mkdir -p "${PKG_DIR}/tokenizer/modules"

# Copiar archivos
echo "Copiando archivos al paquete..."
cp __init__.py "${PKG_DIR}/tokenizer/"
cp tokenizer.py "${PKG_DIR}/tokenizer/"
cp modules/__init__.py "${PKG_DIR}/tokenizer/modules/"
cp modules/checkpoint.py "${PKG_DIR}/tokenizer/modules/"
cp modules/performance.py "${PKG_DIR}/tokenizer/modules/"
cp modules/utils.py "${PKG_DIR}/tokenizer/modules/"
cp modules/s3_utils.py "${PKG_DIR}/tokenizer/modules/"

# Crear archivo __main__.py para permitir ejecución como módulo
cat > "${PKG_DIR}/tokenizer/__main__.py" << EOF
"""
Punto de entrada para el paquete tokenizer
"""
from .tokenizer import main

if __name__ == "__main__":
    main()
EOF

# Crear archivo README.md
cat > "${PKG_DIR}/README.md" << EOF
# SageMaker Tokenizer

Paquete para generar embeddings en SageMaker Training Jobs.
EOF

# Crear archivo de entrada para simplificar la ejecución
cat > "${PKG_DIR}/run.py" << EOF
"""
Script de entrada para SageMaker Training Job.
"""
from tokenizer.tokenizer import main

if __name__ == "__main__":
    main()
EOF

# Empaquetar el código
echo "Empaquetando código..."
cd "${PKG_DIR}"
pip install --target "${PKG_DIR}/package" .

# Comprimir el paquete y los archivos de entrada
echo "Comprimiendo archivos..."
cd "${PKG_DIR}"
tar -czf "${TMP_DIR}/code.tar.gz" .

# Subir a S3
echo "Subiendo paquete a S3..."
aws s3 cp "${TMP_DIR}/code.tar.gz" "${S3_CODE_LOCATION}code.tar.gz"

echo "Limpiando directorio temporal..."
rm -rf "${TMP_DIR}"

echo "Código subido exitosamente a ${S3_CODE_LOCATION}code.tar.gz"
echo "Ahora puedes ejecutar job_maker.sh para crear el trabajo de entrenamiento"
