#!/usr/bin/env python3
"""
Pipeline Completo de Procesamiento BOE - BoeFacil
================================================

Script unificado que ejecuta todo el pipeline de procesamiento desde la descarga
de datos BOE hasta la generación de embeddings e indexación FAISS.

Pipeline de procesamiento:
1. Descarga sumarios BOE (JSON)
2. Aplanado de datos y descarga de cuerpos completos (Parquet)
3. Conversión HTML → Markdown y extracción de tablas
4. Generación de chunks de texto optimizados
5. Generación de embeddings localmente (CPU/GPU)
6. Actualización del índice FAISS

Estructura de directorios:
samples/
├── json/           # Paso 1: Sumarios BOE
├── parquet/        # Paso 2-3: Datos aplanados con markdown
├── chunks/         # Paso 4: Chunks en parquet (cada fila = un chunk)
└── embeddings/     # Paso 5: Parquet con embeddings → listo para FAISS

Autor: jbarrero
Fecha: Septiembre 2025
"""

import os
import sys
import json
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Configurar path para imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Importar módulos del proyecto
from lib import boe, utils_boe, utils, chunk_utils, index_builder
from lib.utils_markdown import extraer_tablas_markdown, eliminar_tablas_markdown, convertir_tablas_markdown_a_texto

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def embed_texts(texts: List[str], model: Any, batch_size: int) -> List[List[float]]:
    """
    Devuelve una lista de embeddings (lista de floats) para los textos de entrada.

    Utiliza SentenceTransformer.encode_document, funcion especifica para generar embeddings de documentos largos.
    Internamente con el batch_size configurado.
    Incluye lógica de reintentos para errores de memoria CUDA.
    """
    if model is None:
        raise RuntimeError("sentence_transformers not available; cannot produce embeddings")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # encode returns a numpy array; convert to python lists for parquet-friendly storage
            embeddings = model.encode_document(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.tolist()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and attempt < max_retries - 1:
                logger.warning("CUDA OOM on attempt %d, reducing batch size and retrying...", attempt + 1)
                # Limpiar memoria CUDA
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                except:
                    pass
                # Reduce batch size for retry
                batch_size = max(1, batch_size // 2)
                logger.info("Reduced batch size to %d for retry", batch_size)
            else:
                raise e

class PipelineBOE:
    """
    Clase principal para ejecutar el pipeline completo de procesamiento BOE
    """
    
    def __init__(self, base_dir: str = "samples", fecha_procesamiento: str = None):
        """
        Inicializar pipeline con directorios base organizados por fecha
        
        Args:
            base_dir: Directorio base para almacenar todos los datos
            fecha_procesamiento: Fecha en formato YYYYMMDD para crear subdirectorio específico
        """
        self.base_dir = Path(base_dir)
        self.fecha_procesamiento = fecha_procesamiento
        
        # Crear subdirectorio por fecha si se especifica
        if fecha_procesamiento:
            self.fecha_dir = self.base_dir / fecha_procesamiento
        else:
            self.fecha_dir = self.base_dir
        
        # Directorios organizados por fecha
        self.json_dir = self.fecha_dir / "json"
        self.parquet_dir = self.fecha_dir / "parquet"
        self.chunks_dir = self.fecha_dir / "chunks"
        self.embeddings_dir = self.fecha_dir / "embeddings"
        self.indices_dir = Path("indices")
        
        # Configuración de embeddings
        self.embedding_model_name = "pablosi/bge-m3-trained-2"  # Modelo preentrenado para BOE
        self.embedding_dimension = 1024
        self.max_tokens_per_chunk = 1000
        self.batch_size = None  # Auto-detectar según GPU disponible
        
        # Crear directorios necesarios
        self._create_directories()
        
        # Estado del pipeline
        self.stats = {
            "inicio": None,
            "fin": None,
            "fecha_procesada": fecha_procesamiento,
            "directorio_fecha": str(self.fecha_dir) if fecha_procesamiento else None,
            "archivos_procesados": 0,
            "chunks_generados": 0,
            "embeddings_creados": 0,
            "indices_actualizados": 0
        }
    
    def _create_directories(self):
        """Crear estructura de directorios necesaria organizados por fecha"""
        for directory in [self.json_dir, self.parquet_dir, self.chunks_dir, 
                         self.embeddings_dir, self.indices_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorios creados en: {self.fecha_dir}")
    
    def paso_1_descargar_boe(self, fecha: str) -> bool:
        """
        Paso 1: Descarga sumario BOE para la fecha especificada
        
        Args:
            fecha: Fecha en formato YYYY-MM-DD (un solo día)
            
        Returns:
            True si la descarga fue exitosa
        """
        logger.info("=== PASO 1: Descarga de sumario BOE ===")
        
        try:
            # Convertir fecha YYYY-MM-DD a YYYYMMDD
            fecha_dt = datetime.strptime(fecha, '%Y-%m-%d')
            fecha_str = fecha_dt.strftime('%Y%m%d')
            
            logger.info(f"Descargando BOE para fecha: {fecha}")
            
            # Descargar BOE específico para la fecha
            boe.extraer(fecha=fecha_str, output_dir=str(self.fecha_dir))
            
            logger.info(f"Descarga completada para: {fecha}")
            self.stats["archivos_procesados"] = 1
            return True
            
        except Exception as e:
            logger.error(f"Error en paso 1: {e}")
            return False
    
    def paso_2_aplanar_y_descargar_cuerpos(self) -> bool:
        """
        Paso 2: Procesa archivos JSON, aplana datos y descarga cuerpos completos
        
        Returns:
            True si el procesamiento fue exitoso
        """
        logger.info("=== PASO 2: Aplanado de datos y descarga de cuerpos ===")
        
        try:
            # Buscar archivos JSON
            json_files = list(self.json_dir.glob("*.json"))
            
            if not json_files:
                logger.error(f"No se encontraron archivos JSON en {self.json_dir}")
                return False
            
            logger.info(f"Procesando {len(json_files)} archivos JSON")
            
            with tqdm(json_files, desc="Procesando JSON") as pbar:
                for json_file in pbar:
                    pbar.set_description(f"Procesando {json_file.name}")
                    
                    try:
                        # Cargar JSON
                        with open(json_file, 'r', encoding='utf-8') as f:
                            datos_json = json.load(f)
                        
                        # Aplanar datos usando utils_boe
                        logger.debug(f"Aplanando datos de {json_file.name}")
                        flattened_data = utils_boe.flatten_boe(datos_json)
                        
                        # Crear DataFrame
                        df = pd.DataFrame(flattened_data)
                        
                        if len(df) == 0:
                            logger.warning(f"No hay datos válidos en {json_file.name}")
                            continue
                        
                        # Generar nombre de archivo parquet
                        fecha = json_file.name.replace('boe_', '').replace('.json', '')
                        parquet_file = self.parquet_dir / f"boe_data_{fecha}.parquet"
                        
                        # Guardar como parquet
                        df.to_parquet(parquet_file, index=False)
                        logger.debug(f"Guardado: {parquet_file}")
                        
                    except Exception as e:
                        logger.error(f"Error procesando {json_file.name}: {e}")
                        continue
                    
                    # Pausa breve
                    time.sleep(0.1)
            
            logger.info("Aplanado completado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en paso 2: {e}")
            return False
    
    def paso_3_convertir_html_markdown(self) -> bool:
        """
        Paso 3: Convierte HTML a Markdown y extrae tablas por separado
        
        Returns:
            True si la conversión fue exitosa
        """
        logger.info("=== PASO 3: Conversión HTML → Markdown ===")
        
        try:
            # Buscar archivos parquet
            parquet_files = list(self.parquet_dir.glob("*.parquet"))
            
            if not parquet_files:
                logger.error(f"No se encontraron archivos parquet en {self.parquet_dir}")
                return False
            
            logger.info(f"Convirtiendo {len(parquet_files)} archivos parquet")
            
            with tqdm(parquet_files, desc="Convirtiendo HTML") as pbar:
                for parquet_file in pbar:
                    pbar.set_description(f"Procesando {parquet_file.name}")
                    
                    try:
                        # Cargar parquet
                        df = pd.read_parquet(parquet_file)
                        
                        if 'html' not in df.columns:
                            logger.warning(f"No hay columna 'html' en {parquet_file.name}")
                            continue
                        
                        logger.debug(f"Convirtiendo HTML a markdown en {parquet_file.name}")
                        
                        # Convertir HTML a markdown usando utils
                        df['markdown'] = df['html'].apply(
                            lambda html: utils.extraer_texto_de_html(html) if pd.notna(html) else ""
                        )
                        
                        # Extraer tablas del markdown
                        df['tablas'] = df['markdown'].apply(
                            lambda md: utils.dataframes_a_json(extraer_tablas_markdown(md)) if pd.notna(md) else ""
                        )
                        
                        # Convertir tablas del markdown a texto
                        df['markdown'] = df['markdown'].apply(
                            lambda md: convertir_tablas_markdown_a_texto(md) if pd.notna(md) else ""
                        )
                        
                        # Guardar archivo actualizado
                        df.to_parquet(parquet_file, index=False)
                        logger.debug(f"Actualizado: {parquet_file}")
                        
                    except Exception as e:
                        logger.error(f"Error procesando {parquet_file.name}: {e}")
                        continue
            
            logger.info("Conversión HTML → Markdown completada")
            return True
            
        except Exception as e:
            logger.error(f"Error en paso 3: {e}")
            return False
    
    def paso_4_generar_chunks(self) -> bool:
        """
        Paso 4: Genera chunks de texto optimizados para embeddings
        Procesa cada archivo parquet individualmente
        
        Returns:
            True si la generación de chunks fue exitosa
        """
        logger.info("=== PASO 4: Generación de chunks ===")
        
        try:
            # Buscar archivos parquet con markdown
            parquet_files = list(self.parquet_dir.glob("*.parquet"))
            
            if not parquet_files:
                logger.error(f"No se encontraron archivos parquet en {self.parquet_dir}")
                return False
            
            logger.info(f"Procesando {len(parquet_files)} archivos parquet individualmente")
            
            total_chunks_generados = 0
            
            with tqdm(parquet_files, desc="Generando chunks") as pbar:
                for parquet_file in pbar:
                    pbar.set_description(f"Procesando {parquet_file.name}")
                    
                    try:
                        # Cargar archivo parquet individual
                        df = pd.read_parquet(parquet_file)
                        
                        if 'markdown' not in df.columns:
                            logger.warning(f"No hay columna 'markdown' en {parquet_file.name}")
                            continue
                        
                        if len(df) == 0:
                            logger.warning(f"Archivo vacío: {parquet_file.name}")
                            continue
                        
                        logger.debug(f"Procesando: {parquet_file.name} ({len(df)} filas)")
                        
                        # Generar nombre único para los chunks de este archivo
                        fecha = parquet_file.stem.replace('boe_data_', '')
                        chunks_output_path = self.chunks_dir / f"chunks_{fecha}.parquet"
                        
                        # Generar chunks para este archivo específico
                        chunks = chunk_utils.chunking_markdown_df(
                            df=df,
                            columna_markdown='markdown',
                            max_tokens=self.max_tokens_per_chunk,
                            texto_id=None,  # Se extraerá automáticamente
                            store_path=str(self.chunks_dir)  # Solo para los archivos JSON individuales si es necesario
                        )
                        
                        # Convertir chunks a DataFrame si no está ya en ese formato
                        if chunks and isinstance(chunks[0], dict):
                            chunks_df = pd.DataFrame(chunks)
                        else:
                            chunks_df = chunks if isinstance(chunks, pd.DataFrame) else pd.DataFrame()
                        
                        if len(chunks_df) > 0:
                            # Guardar chunks como parquet para este archivo específico
                            chunks_df.to_parquet(chunks_output_path, index=False)
                            total_chunks_generados += len(chunks_df)
                            logger.debug(f"Guardado: {chunks_output_path} ({len(chunks_df)} chunks)")
                        else:
                            logger.warning(f"No se generaron chunks para {parquet_file.name}")
                        
                    except Exception as e:
                        logger.error(f"Error procesando {parquet_file.name}: {e}")
                        continue
            
            self.stats["chunks_generados"] = total_chunks_generados
            logger.info(f"Total chunks generados: {total_chunks_generados}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en paso 4: {e}")
            return False
    
    def paso_5_generar_embeddings_local(self) -> bool:
        """
        Paso 5: Genera embeddings localmente usando sentence_transformers
        Procesa cada archivo de chunks individualmente y genera embeddings
        
        Returns:
            True si la generación de embeddings fue exitosa
        """
        logger.info("=== PASO 5: Generación de embeddings locales ===")
        
        try:
            # Importar sentence_transformers
            try:
                from sentence_transformers import SentenceTransformer
                import torch
            except ImportError:
                logger.error("sentence_transformers no está instalado. Instalar con: pip install sentence-transformers")
                return False
            
            # Detectar dispositivo (GPU/CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Usando dispositivo: {device}")
            
            # Cargar modelo de embeddings
            logger.info(f"Cargando modelo: {self.embedding_model_name}")
            model = SentenceTransformer(self.embedding_model_name, device=device)
            
            # Auto-detectar batch size según dispositivo
            if self.batch_size is None:
                self.batch_size = 32 if device == "cuda" else 8
            logger.info(f"Batch size: {self.batch_size}")
            
            # Buscar archivos de chunks (parquet)
            chunk_files = list(self.chunks_dir.glob("*.parquet"))
            
            if not chunk_files:
                logger.error(f"No se encontraron archivos parquet de chunks en {self.chunks_dir}")
                return False
            
            logger.info(f"Procesando {len(chunk_files)} archivos de chunks individualmente")
            
            total_embeddings_creados = 0
            
            with tqdm(chunk_files, desc="Generando embeddings") as pbar:
                for chunk_file in pbar:
                    pbar.set_description(f"Procesando {chunk_file.name}")
                    
                    try:
                        # Cargar DataFrame de chunks para este archivo
                        chunks_df = pd.read_parquet(chunk_file)
                        
                        if len(chunks_df) == 0:
                            logger.warning(f"Archivo de chunks vacío: {chunk_file.name}")
                            continue
                        
                        # Verificar que existe la columna 'texto'
                        if 'texto' not in chunks_df.columns:
                            logger.error(f"No se encontró la columna 'texto' en {chunk_file.name}")
                            logger.info(f"Columnas disponibles: {list(chunks_df.columns)}")
                            continue
                        
                        # Extraer textos para embeddings
                        textos = chunks_df['texto'].fillna('').astype(str).tolist()
                        
                        if not textos:
                            logger.warning(f"No hay textos válidos en {chunk_file.name}")
                            continue
                        
                        logger.debug(f"Generando embeddings para {len(textos)} chunks de {chunk_file.name}")
                        
                        # Generar embeddings usando embed_texts (maneja batches internamente)
                        try:
                            embeddings = embed_texts(
                                texts=textos,
                                model=model,
                                batch_size=self.batch_size
                            )
                            
                        except Exception as e:
                            logger.error(f"Error generando embeddings para {chunk_file.name}: {e}")
                            # Rellenar con embeddings vacíos para mantener índices
                            embeddings = [[0.0] * self.embedding_dimension] * len(textos)
                        
                        # Crear DataFrame con embeddings para este archivo
                        embeddings_df = chunks_df.copy()
                        embeddings_df['embeddings'] = embeddings
                        
                        # Generar nombre del archivo de embeddings
                        fecha = chunk_file.stem.replace('chunks_', '')
                        embeddings_file = self.embeddings_dir / f"embeddings_{fecha}.parquet"
                        
                        # Guardar embeddings en archivo parquet
                        embeddings_df.to_parquet(embeddings_file, index=False)
                        
                        total_embeddings_creados += len(embeddings)
                        logger.debug(f"Embeddings guardados: {embeddings_file} ({len(embeddings)} embeddings)")
                        
                    except Exception as e:
                        logger.error(f"Error procesando {chunk_file.name}: {e}")
                        continue
            
            self.stats["embeddings_creados"] = total_embeddings_creados
            logger.info(f"Total embeddings generados: {total_embeddings_creados}")
            
            # Limpiar memoria
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error en paso 5: {e}")
            return False

    def paso_6_actualizar_indice(self) -> bool:
        """
        Paso 6: Actualiza índice FAISS existente con los nuevos embeddings
        
        Returns:
            True si la actualización fue exitosa
        """
        logger.info("=== PASO 6: Actualización del índice FAISS ===")
        
        try:
            # Buscar archivos de embeddings
            embedding_files = list(self.embeddings_dir.glob("*.parquet"))
            
            if not embedding_files:
                logger.error(f"No se encontraron archivos de embeddings en {self.embeddings_dir}")
                return False
            
            # Configurar rutas de índice
            index_path = self.indices_dir / "boe_index.faiss"
            metadata_path = self.indices_dir / "metadata.json"
            
            logger.info(f"Procesando {len(embedding_files)} archivos de embeddings")
            
            if not index_path.exists():
                logger.error(f"No existe índice FAISS en {index_path}. Debe crearse primero usando la construcción manual.")
                return False
            
            # Actualizar índice existente (archivo por archivo)
            logger.info("Actualizando índice FAISS existente")
            
            for embedding_file in embedding_files:
                try:
                    result = index_builder.add_daily_parquet_to_index(
                        str(embedding_file),
                        str(index_path),
                        str(metadata_path)
                    )
                    logger.debug(f"Agregado al índice: {embedding_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error agregando {embedding_file.name} al índice: {e}")
                    continue
            
            logger.info("Índice FAISS actualizado exitosamente")
            
            # Obtener estadísticas del índice
            if hasattr(result, 'get_stats'):
                stats = result.get_stats()
                logger.info(f"Estadísticas del índice:")
                logger.info(f"  - Total vectores: {stats.get('total_vectors', 'N/A'):,}")
                logger.info(f"  - Tipo de índice: {stats.get('index_type', 'N/A')}")
                logger.info(f"  - Dimensión: {stats.get('dimension', 'N/A')}")
            
            self.stats["indices_actualizados"] = len(embedding_files)
            return True
            
        except Exception as e:
            logger.error(f"Error en paso 6: {e}")
            return False
    
    def ejecutar_pipeline_completo(
        self, 
        fecha: str
    ) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de procesamiento BOE para una fecha específica
        
        Args:
            fecha: Fecha en formato YYYY-MM-DD (un solo día)
            
        Returns:
            Diccionario con estadísticas del procesamiento
        """
        logger.info("🚀 INICIANDO PIPELINE COMPLETO BOE")
        logger.info(f"Fecha: {fecha}")
        logger.info(f"Directorio base: {self.base_dir}")
        logger.info(f"Directorio fecha: {self.fecha_dir}")
        
        self.stats["inicio"] = datetime.now().isoformat()
        inicio_tiempo = time.time()
        
        try:
            # Paso 1: Descarga BOE
            if not self.paso_1_descargar_boe(fecha):
                raise Exception("Error en Paso 1: Descarga BOE")
            
            # Paso 2: Aplanar y descargar cuerpos
            if not self.paso_2_aplanar_y_descargar_cuerpos():
                raise Exception("Error en Paso 2: Aplanado de datos")
            
            # Paso 3: Conversión HTML → Markdown
            if not self.paso_3_convertir_html_markdown():
                raise Exception("Error en Paso 3: Conversión HTML → Markdown")
            
            # Paso 4: Generación de chunks
            if not self.paso_4_generar_chunks():
                raise Exception("Error en Paso 4: Generación de chunks")
            
            # Paso 5: Generación de embeddings
            if not self.paso_5_generar_embeddings_local():
                raise Exception("Error en Paso 5: Generación de embeddings")
            
            # Paso 6: Actualización de índice
            if not self.paso_6_actualizar_indice():
                raise Exception("Error en Paso 6: Actualización de índice")
            
            # Finalizar estadísticas
            tiempo_total = time.time() - inicio_tiempo
            self.stats["fin"] = datetime.now().isoformat()
            self.stats["tiempo_total_segundos"] = tiempo_total
            
            logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info(f"Tiempo total: {tiempo_total:.2f} segundos")
            logger.info(f"Fecha procesada: {fecha}")
            logger.info(f"Chunks generados: {self.stats['chunks_generados']}")
            logger.info(f"Embeddings creados: {self.stats['embeddings_creados']}")
            
            return {"success": True, "stats": self.stats}
            
        except Exception as e:
            tiempo_total = time.time() - inicio_tiempo
            self.stats["fin"] = datetime.now().isoformat()
            self.stats["tiempo_total_segundos"] = tiempo_total
            self.stats["error"] = str(e)
            
            logger.error(f"❌ ERROR EN PIPELINE: {e}")
            return {"success": False, "error": str(e), "stats": self.stats}


def main():
    """Función principal para ejecutar desde línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline completo de procesamiento BOE para un día específico")
    parser.add_argument("--fecha", required=True, help="Fecha en formato YYYY-MM-DD")
    parser.add_argument("--base-dir", default="samples", help="Directorio base (default: samples)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Logging detallado")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ejecutar pipeline
    fecha_dt = datetime.strptime(args.fecha, '%Y-%m-%d')
    fecha_dir = fecha_dt.strftime('%Y%m%d')
    
    pipeline = PipelineBOE(
        base_dir=args.base_dir,
        fecha_procesamiento=fecha_dir
    )
    resultado = pipeline.ejecutar_pipeline_completo(
        fecha=args.fecha
    )
    
    # Mostrar resultado
    if resultado["success"]:
        print("✅ Pipeline completado exitosamente")
        return 0
    else:
        print(f"❌ Error en pipeline: {resultado['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
