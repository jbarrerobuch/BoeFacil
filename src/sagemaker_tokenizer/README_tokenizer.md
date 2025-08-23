# SageMaker Embedding Processing Script

Este script genera embeddings de textos usando el modelo BGE-M3 en SageMaker Processing Jobs.

## Uso Local:

```bash
python src/tokenizer.py --input-path ./samples/parquet --output-path ./output --model pablosi/bge-m3-trained-2 --batch-size 32
```

## Uso en SageMaker Processing:

### Imagen OFICIAL HuggingFace para eu-west-3:
```
763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04
```
**✓ Ventajas de esta imagen:**
- PyTorch 2.6.0 (último estable)
- Transformers 4.49.0 (compatible con BGE-M3)
- Python 3.12 (máximo rendimiento)
- CUDA 12.4 (última versión)
- Imagen oficial de AWS para HuggingFace

### Configuración recomendada:

#### **Opción 1 - Más eficiente (si está disponible):**
- **`ml.g4dn.xlarge`** (4 vCPU, 16 GB RAM, 1 NVIDIA T4) - **~$0.526/hora** (On-Demand)
- **`ml.g4dn.xlarge` SPOT** - **~$0.158/hora** (70% descuento)
- **Batch size:** 32 (auto-detectado)
- **Nota:** Recomendada con spot instances por su buena relación precio/rendimiento

#### **Opción 2 - CPU con mejor precio (sin GPU):**
- **`ml.m5.xlarge`** (4 vCPU, 16 GB RAM) - **~$0.192/hora**
- **Batch size:** 8 (CPU processing)
- **Nota:** Mucho más lento pero más barato para pruebas

#### **Opción 3 - GPU alternativa (si disponible):**
- **`ml.p3.2xlarge`** (8 vCPU, 61 GB RAM, 1 NVIDIA V100) - **~$3.06/hora**
- **Batch size:** 64 (auto-detectado)
- **Nota:** Muy caro pero máximo rendimiento

#### **Recomendación para producción:** 
- Usa `ml.g4dn.xlarge` como **instancia spot** para máximo ahorro
- El script tiene checkpointing para garantizar resiliencia en caso de terminación de la instancia spot

### Argumentos del container (uno por línea):
```
python3
/opt/ml/processing/input/code/tokenizer.py
--input-path
/opt/ml/processing/input
--output-path
/opt/ml/processing/output
--model
pablosi/bge-m3-trained-2
--batch-size
0
```

**Argumentos adicionales opcionales:**
```
# Desactivar checkpointing (procesar todos los archivos)
--no-checkpoint
```

**Notas:** 
- `--batch-size 0` activa la detección automática del batch size óptimo según la GPU.
- El script incluye soporte para instancias spot con checkpointing automático.

### Mapeo de entradas/salidas:
- **Code Input:** S3 con tokenizer.py → `/opt/ml/processing/input/code/`
- **Data Input:** S3 bucket con archivos parquet → `/opt/ml/processing/input`
- **Output:** S3 bucket para embeddings → `/opt/ml/processing/output`

## Notas técnicas:
- Instala automáticamente PyTorch GPU (CUDA 12.4) y sentence-transformers 3.3.0
- Genera embeddings de 1024 dimensiones usando BGE-M3
- Incluye limpieza de memoria GPU y logging detallado
- Procesa archivos parquet con columna `texto`

## Soporte para Spot Instances:

El script incluye funcionalidades de checkpointing para garantizar la resiliencia cuando se ejecuta en instancias spot:

- **Checkpoints automáticos**: Guarda el progreso cada 5 archivos procesados
- **Manejo de señales**: Captura las señales SIGTERM/SIGINT emitidas cuando una instancia spot está programada para terminar
- **Reanudación inteligente**: Al reiniciar, salta archivos ya procesados en ejecuciones anteriores
- **Información detallada**: El archivo de resultados JSON incluye estadísticas de checkpoint

### Argumentos de checkpoint:

```bash
# Desactivar checkpointing (procesar todos los archivos, incluso los ya procesados)
--no-checkpoint
```

### Funcionamiento:

1. Al iniciar, verifica si existe un checkpoint previo y carga la lista de archivos ya procesados
2. Durante el procesamiento, salta automáticamente los archivos ya procesados
3. Al recibir una señal de terminación, guarda inmediatamente el checkpoint antes de salir
4. Cada 5 archivos procesados, actualiza el checkpoint en disco
5. Al finalizar, incluye información detallada sobre el estado del checkpoint

Esta funcionalidad es especialmente útil para:
- Trabajos de SageMaker en instancias spot (más económicas pero pueden terminar inesperadamente)
- Procesamiento de grandes volúmenes de datos que requieren múltiples ejecuciones
- Recuperación automática después de errores o terminaciones
