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

# Crear un archivo JSON temporal con la configuración del job
cat > training-job-config.json << EOF
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

# Mostrar la configuración final
echo "Configuración del Training Job:"
cat training-job-config.json

# Crear el training job
aws sagemaker create-training-job --cli-input-json file://training-job-config.json --region ${REGION}

echo "Training job '${JOB_NAME}' creado con instancias spot"
echo "Para monitorizar el estado del job, ejecuta:"
echo "aws sagemaker describe-training-job --training-job-name ${JOB_NAME} --region ${REGION} --query 'TrainingJobStatus'"

# Opcional: limpiar el archivo de configuración
rm training-job-config.json