#!/bin/bash
# Script para crear un SageMaker Training Job con instancias spot

# IMPORTANTE: Antes de ejecutar este script, debes empaquetar y subir el código a S3 usando:
#   bash package_and_upload.sh
#
# Este script empaquetará y subirá tokenizer.py y sus módulos como un paquete Python
# a la ubicación S3 especificada en S3_CODE_LOCATION

# Variables de configuración - AJUSTA ESTAS VARIABLES A TUS NECESIDADES
JOB_NAME="embedding-training-SPOT-mlm7ixlarge-001"  # Nombre único para el job
ROLE_ARN="arn:aws:iam::192644798139:role/service-role/SageMaker-all"  # REEMPLAZAR con tu ARN de rol
REGION="eu-west-3"  # Región de París para coincidir con la imagen Docker
IMAGE_URI="763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04"

# Rutas S3 - REEMPLAZAR con tus rutas
S3_CODE_LOCATION="s3://boe-facil/job_embeddings/code"
S3_INPUT_LOCATION="s3://boe-facil/test_job/input/"
S3_OUTPUT_LOCATION="s3://boe-facil/test_job/output/"
S3_CHECKPOINT_LOCATION="s3://boe-facil/test_job/checkpoints/"

# Configuración de instancia
INSTANCE_TYPE="ml.m7i.xlarge"  # Instancia recomendada
MAX_RUNTIME_SECONDS=600  # Tiempo máximo de ejecución en segundos (10 minutos)
MAX_WAIT_SECONDS=1800    # Tiempo máximo de espera para instancias spot (30 minutos)

# Crear un directorio temporal para guardar la configuración
TMP_DIR=$(mktemp -d -t sagemaker-XXXXXXXXXX)
CONFIG_FILE="${TMP_DIR}/training-job-config.json"

echo "Usando directorio temporal: ${TMP_DIR}"

# Crear archivo de configuración JSON en el directorio temporal
cat > "${CONFIG_FILE}" << EOF
{
  "TrainingJobName": "${JOB_NAME}",
  "RoleArn": "${ROLE_ARN}",
  "AlgorithmSpecification": {
    "TrainingImage": "${IMAGE_URI}",
    "TrainingInputMode": "File",
    "ContainerEntrypoint": [
      "python3",
      "/opt/ml/input/data/code/run.py"
    ],
    "ContainerArguments": [
      "--model",
      "pablosi/bge-m3-trained-2",
      "--batch-size",
      "8"
    ]
  },
  "HyperParameters": {},
  "InputDataConfig": [
    {
      "ChannelName": "dataset",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "${S3_INPUT_LOCATION}",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "application/x-parquet"
    },
    {
      "ChannelName": "code",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "${S3_CODE_LOCATION}",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "application/x-tar"
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "${S3_OUTPUT_LOCATION}"
  },
  "CheckpointConfig": {
    "S3Uri": "${S3_CHECKPOINT_LOCATION}",
    "LocalPath": "/opt/ml/checkpoints"
  },
  "ResourceConfig": {
    "InstanceType": "${INSTANCE_TYPE}",
    "InstanceCount": 1,
    "VolumeSizeInGB": 30
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": ${MAX_RUNTIME_SECONDS},
    "MaxWaitTimeInSeconds": ${MAX_WAIT_SECONDS}
  },
  "EnableManagedSpotTraining": true,
  "EnableNetworkIsolation": false,
  "EnableInterContainerTrafficEncryption": false
}
EOF

# Verificar que el archivo se creó correctamente
if [ -f "${CONFIG_FILE}" ]; then
  echo "✅ Archivo de configuración creado correctamente en: ${CONFIG_FILE}"
  echo "Configuración del Training Job:"
  cat "${CONFIG_FILE}"
else
  echo "❌ ERROR: No se pudo crear el archivo de configuración"
  exit 1
fi

# Crear el training job
echo "Creando SageMaker Training Job..."
aws sagemaker create-training-job --cli-input-json "file://${CONFIG_FILE}" --region ${REGION}

# Verificar si el comando fue exitoso
if [ $? -eq 0 ]; then
  echo "✅ Training job '${JOB_NAME}' creado exitosamente con instancias spot"
else
  echo "❌ ERROR: No se pudo crear el Training Job"
  echo "Verifique que tiene permisos para crear SageMaker Training Jobs y que las rutas S3 son correctas"
  exit 1
fi

echo "Para monitorizar el estado del job, ejecuta:"
echo "aws sagemaker describe-training-job --training-job-name ${JOB_NAME} --region ${REGION} --query 'TrainingJobStatus'"

# Limpiar archivos temporales
echo "Limpiando archivos temporales..."
rm -rf "${TMP_DIR}"

echo "✅ Proceso completado"