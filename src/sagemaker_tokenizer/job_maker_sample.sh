#!/bin/bash
# Script para crear un SageMaker Training Job con instancias spot

# Variables de configuración - AJUSTA ESTAS VARIABLES A TUS NECESIDADES
JOB_NAME="embedding-training-SPOT-mlg4dnxlarge-$(date +%Y%m%d-%H%M%S)"  # Nombre único para el job
ROLE_ARN="arn:aws:iam::*****"  # REEMPLAZAR con tu ARN de rol
REGION="aws-region"  # Región de París para coincidir con la imagen Docker
IMAGE_URI="763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04"

# Rutas S3 - REEMPLAZAR con tus rutas
S3_CODE_LOCATION="s3://nombre-bucket/code/"
S3_INPUT_LOCATION="s3://nombre-bucket/input/"
S3_OUTPUT_LOCATION="s3://nombre-bucket/output/"
S3_CHECKPOINT_LOCATION="s3://nombre-bucket/checkpoints/"

# Configuración de instancia
INSTANCE_TYPE="ml.g4dn.xlarge"  # Instancia recomendada
MAX_RUNTIME_SECONDS=3600  # Tiempo máximo de ejecución en segundos (1 hora)
MAX_WAIT_SECONDS=10800    # Tiempo máximo de espera para instancias spot (3 horas)

# Verificar que el archivo code.tar.gz existe en S3
echo "Verificando que el archivo code.tar.gz existe en S3..."
aws s3 ls "${S3_CODE_LOCATION}code.tar.gz" --region ${REGION}
if [ $? -ne 0 ]; then
    echo "❌ ERROR: El archivo code.tar.gz no se encuentra en ${S3_CODE_LOCATION}/code.tar.gz"
    echo "Por favor, sube el archivo code.tar.gz a S3 antes de continuar."
    exit 1
fi
echo "✅ Archivo code.tar.gz encontrado en S3"

# Crear un directorio temporal para guardar la configuración
TMP_DIR=$(mktemp -d -t sagemaker-XXXXXXXXXX)
CONFIG_FILE="${TMP_DIR}/training-job-config.json"

echo "Usando directorio temporal: ${TMP_DIR}"
echo "Creando job con nombre: ${JOB_NAME}"

# Crear archivo de configuración JSON en el directorio temporal
cat > "${CONFIG_FILE}" << EOF
{
  "TrainingJobName": "${JOB_NAME}",
  "RoleArn": "${ROLE_ARN}",
  "AlgorithmSpecification": {
    "TrainingImage": "${IMAGE_URI}",
    "TrainingInputMode": "File",
    "ContainerEntrypoint": [
      "sh",
      "-c",
      "cd /opt/ml/input/data/code && tar -xzf code.tar.gz 2>/dev/null || true && ls -la && python3 run.py"
    ],
    "ContainerArguments": [
      "--model",
      "pablosi/bge-m3-trained-2",
      "--batch-size",
      "0"
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
      "ContentType": "application/gzip"
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
  echo ""
  echo "Para monitorizar el estado del job, ejecuta:"
  echo "aws sagemaker describe-training-job --training-job-name ${JOB_NAME} --region ${REGION} --query 'TrainingJobStatus'"
  echo ""
  echo "Para ver los logs del job:"
  echo "aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/TrainingJobs --region ${REGION}"
  echo "aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs/${JOB_NAME} --log-stream-name ${JOB_NAME}/algo-1-* --region ${REGION}"
else
  echo "❌ ERROR: No se pudo crear el Training Job"
  echo "Verifique que tiene permisos para crear SageMaker Training Jobs y que las rutas S3 son correctas"
  exit 1
fi

# Limpiar archivos temporales
echo "Limpiando archivos temporales..."
rm -rf "${TMP_DIR}"

echo "✅ Proceso completado"
