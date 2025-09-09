"""
Gestión de Datos - Módulo de Administración de Índices BOE
============================================================

Este módulo permite la gestión completa de los índices vectoriales del BOE,
incluyendo construcción desde cero, actualización incremental y procesamiento ETL.

Autor: jbarrero
Fecha: Septiembre 2025
"""

import streamlit as st
import os
import sys
import datetime
from datetime import timedelta
from pathlib import Path

# Obtener directorio raíz del proyecto (subir desde src/ui/pages/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Agregar src al path para imports absolutos
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Importar módulos con manejo de errores
try:
    from lib.index_builder import build_index_from_parquets, get_parquet_files_from_directory
    from pipeline_completo import PipelineBOE
    imports_ok = True
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    build_index_from_parquets = None
    get_parquet_files_from_directory = None
    PipelineBOE = None
    imports_ok = False

# Configuración de la página
st.set_page_config(
    page_title="Gestión de Datos - BoeFacil",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Función principal de la página de gestión de datos"""
    
    # Título principal con estilo
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 2px solid #ff6b35;">
        <h1 style="color: #2c3e50; margin: 0;">🔧 Gestión de Datos</h1>
        <p style="color: #7f8c8d; margin: 0.5rem 0 0 0;">
            Administración completa de índices vectoriales BOE
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Espacio
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar para navegación
    with st.sidebar:
        st.markdown("### 🏠 Navegación")
        
        # Botón de regreso
        if st.button("🏠 Página Principal", use_container_width=True, type="secondary"):
            st.switch_page("streamlit_app.py")
        
        st.markdown("---")
        
        st.markdown("### 📋 Procesos disponibles")
        
        # Selector de proceso
        proceso_seleccionado = st.selectbox(
            "Selecciona el proceso:",
            [
                "Construcción de Índice",
                "Actualización Incremental por día",
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Información de estado del sistema
        st.markdown("### ℹ️ Estado del Sistema")
        
        # Verificar estado del índice actual
        indices_dir = PROJECT_ROOT / "indices"
        if indices_dir.exists() and (indices_dir / "boe_index.faiss").exists():
            st.success("✅ Índice existente detectado")
            metadata_file = indices_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    total_chunks = len(metadata) if isinstance(metadata, list) else metadata.get('total_vectors', 0)
                    st.info(f"📊 Chunks actuales: {total_chunks:,}")
                    
                    # Información adicional si existe
                    report_file = indices_dir / "construction_report.json"
                    if report_file.exists():
                        with open(report_file, 'r', encoding='utf-8') as f:
                            report = json.load(f)
                        if 'construction_date' in report:
                            st.info(f"📅 Última construcción: {report['construction_date']}")
                
                except Exception as e:
                    st.warning(f"⚠️ Error leyendo metadata: {str(e)}")
        else:
            st.warning("⚠️ No se encontró índice existente")
        
        # Directorio de datos
        samples_dir = PROJECT_ROOT / "samples"
        if samples_dir.exists():
            parquet_files = list(samples_dir.rglob("*.parquet"))  # Búsqueda recursiva
            st.info(f"📁 Archivos parquet: {len(parquet_files)} (total)")
        else:
            st.error("❌ Directorio de origen de datos no encontrado")

    # Contenido principal basado en el proceso seleccionado
    if "Construcción de Índice" in proceso_seleccionado:
        nueva_construccion()
    elif "Actualización Incremental por día" in proceso_seleccionado:
        actualizacion_dia()

def nueva_construccion():
    """Interfaz para la Construcción de Índice desde cero"""

    st.markdown("## 🏗️ Construcción de Índice desde Cero")

    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h4 style="color: #2c3e50; margin-top: 0;">📋 Descripción del Proceso</h4>
        <p style="margin-bottom: 0;">
            Este proceso construye un índice vectorial completamente nuevo desde los archivos parquet disponibles
            en la carpeta especificada. Se eliminará cualquier índice existente y se creará uno nuevo.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de configuración
    st.markdown("### ⚙️ Configuración de Construcción")
    
    # Selector de carpeta origen
    st.markdown("#### 📁 Carpeta de Datos Origen")
    
    # Carpeta por defecto
    carpeta_por_defecto = str(PROJECT_ROOT / "samples")
    
    carpeta_origen = st.text_input(
        "Ruta de la carpeta con archivos parquet:",
        value=carpeta_por_defecto,
        help="Especifica la ruta completa de la carpeta que contiene los archivos parquet a procesar"
    )
    
    # Verificar carpeta y mostrar archivos disponibles
    carpeta_path = Path(carpeta_origen)
    
    if carpeta_path.exists() and carpeta_path.is_dir():
        parquet_files = list(carpeta_path.rglob("*.parquet"))  # Búsqueda recursiva
        parquet_files.sort()
        
        if parquet_files:
            st.success(f"✅ Carpeta válida encontrada: `{carpeta_origen}`")
            st.info(f"📊 **Archivos parquet detectados:** {len(parquet_files)} (incluyendo subcarpetas)")
            
            # Mostrar lista de archivos en un expander
            with st.expander(f"📄 Ver archivos ({len(parquet_files)} archivos)", expanded=False):
                for i, archivo in enumerate(parquet_files, 1):
                    file_size = archivo.stat().st_size / (1024 * 1024)  # MB
                    # Mostrar ruta relativa desde la carpeta origen
                    ruta_relativa = archivo.relative_to(carpeta_path)
                    st.write(f"{i}. `{ruta_relativa}` ({file_size:.1f} MB)")
            
            archivos_validos = True
            
        else:
            st.warning(f"⚠️ No se encontraron archivos parquet en: `{carpeta_origen}`")
            archivos_validos = False
    else:
        st.error(f"❌ La carpeta no existe o no es válida: `{carpeta_origen}`")
        archivos_validos = False
    
    # Sección de advertencias y confirmaciones
    st.markdown("### ⚠️ Advertencias Importantes")
    
    st.warning("""
    **ATENCIÓN:** Esta operación:
    - Eliminará el índice vectorial existente si existe
    - Procesará todos los archivos parquet de la carpeta especificada
    - Puede tardar varios minutos dependiendo del volumen de datos
    - Requiere espacio en disco suficiente para el nuevo índice
    """)
    
    # Confirmación del usuario
    confirmacion = st.checkbox(
        "He leído las advertencias y confirmo que deseo proceder con la construcción del índice",
        value=False
    )
    
    # Botón de construcción
    st.markdown("### 🚀 Iniciar Construcción")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        if st.button(
            "🏗️ Construir Índice",
            disabled=not confirmacion or not archivos_validos,
            use_container_width=True,
            type="primary"
        ):
            ejecutar_construccion_indice(carpeta_origen)

def actualizacion_dia():
    """Interfaz para el procesamiento ETL Completo de un día específico del BOE"""

    st.markdown("## 🔄 Procesamiento ETL Completo de un día específico del BOE")

    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h4 style="color: #2c3e50; margin-top: 0;">📋 Descripción del Proceso</h4>
        <p style="margin-bottom: 0;">
            Este proceso ejecuta el pipeline completo para un día específico del BOE desde la descarga
            hasta la generación de embeddings e indexación FAISS. Los datos se organizan por fecha 
            en subdirectorios (YYYYMMDD).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline de pasos
    st.markdown("### 🔄 Pipeline de Procesamiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Pasos del Pipeline:**
        1. 📥 Descarga sumario BOE (JSON)
        2. 🔄 Aplanado y descarga de cuerpos
        3. 📝 Conversión HTML → Markdown
        4. ✂️ Generación de chunks
        5. 🧠 Generación de embeddings
        6. 📊 Actualización índice FAISS
        """)
    
    with col2:
        st.markdown("""
        **Estructura por fecha:**
        - `samples/YYYYMMDD/json/` → Sumarios BOE
        - `samples/YYYYMMDD/parquet/` → Datos aplanados
        - `samples/YYYYMMDD/chunks/` → Chunks de texto
        - `samples/YYYYMMDD/embeddings/` → Embeddings
        - `indices/` → Índice FAISS compartido
        """)
    
    # Configuración de fecha (un solo día)
    st.markdown("### 📅 Selección de Fecha")
    
    fecha_boe = st.date_input(
        "Fecha del BOE a procesar:",
        value=datetime.date.today() - timedelta(days=1),  # Ayer por defecto
        help="Selecciona un día específico para procesar. Los datos se guardarán en samples/YYYYMMDD/"
    )
    
    # Información de la fecha seleccionada
    fecha_str = fecha_boe.strftime('%Y-%m-%d')
    fecha_dir = fecha_boe.strftime('%Y%m%d')
    
    st.info(f"""
    **📊 Fecha seleccionada:** {fecha_str}
    
    **📁 Directorio de destino:** `samples/{fecha_dir}/`
    
    **📅 Día de la semana:** {fecha_boe.strftime('%A')} ({fecha_boe.strftime('%d de %B de %Y')})
    """)
    
    # Configuración avanzada
    with st.expander("⚙️ Configuración Avanzada", expanded=False):
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            directorio_base = st.text_input(
                "Directorio base:",
                value="samples",
                help="Directorio donde se guardarán todos los datos procesados (se creará subdirectorio por fecha)"
            )
            
            modelo_embeddings = st.selectbox(
                "Modelo de embeddings:",
                ["pablosi/bge-m3-trained-2", "BAAI/bge-m3"],
                index=0,
                help="Modelo para generar embeddings de texto"
            )
        
        with col_config2:
            max_tokens_chunk = st.number_input(
                "Máximo tokens por chunk:",
                min_value=100,
                max_value=2000,
                value=1000,
                help="Número máximo de tokens por chunk de texto"
            )
            
            batch_size = st.number_input(
                "Batch size embeddings:",
                min_value=1,
                max_value=128,
                value=32,
                help="Tamaño del lote para generar embeddings (auto-detectado si se deja por defecto)"
            )
    
    # Advertencias
    st.markdown("### ⚠️ Advertencias Importantes")
    
    st.warning(f"""
    **ATENCIÓN:** Este proceso:
    - Descargará datos del BOE para la fecha {fecha_str} (requiere conexión a internet)
    - Puede tardar entre 15-30 minutos para un día promedio
    - Creará el directorio `samples/{fecha_dir}/` con toda la estructura de datos
    - Utilizará GPU si está disponible para generar embeddings
    - Actualizará el índice vectorial existente (debe existir un índice base)
    - Los archivos de un día pueden ocupar entre 50-200 MB dependiendo del contenido
    """)
    
    # Confirmación
    confirmacion_pipeline = st.checkbox(
        f"He leído las advertencias y confirmo que deseo procesar el BOE del {fecha_str}",
        value=False,
        help="Confirma que entiendes el proceso y que se creará la estructura de datos por fecha"
    )
    
    # Botón de ejecución
    st.markdown("### 🚀 Ejecutar Pipeline")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        if st.button(
            f"🔄 Procesar BOE del {fecha_str}",
            disabled=not confirmacion_pipeline,
            use_container_width=True,
            type="primary"
        ):
            ejecutar_pipeline_etl_completo(
                fecha=fecha_str,
                directorio_base=directorio_base,
                modelo_embeddings=modelo_embeddings,
                max_tokens_chunk=max_tokens_chunk,
                batch_size=batch_size
            )

def ejecutar_construccion_indice(carpeta_origen):
    """Ejecuta la construcción del índice vectorial"""
    
    try:
        # Crear contenedor de progreso
        progress_container = st.empty()
        
        with progress_container.container():
            st.markdown("### 🔄 Construcción en Progreso...")
            
            # Barra de progreso principal
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Área de logs detallados
            with st.expander("📋 Logs Detallados", expanded=False):
                log_area = st.empty()
            
            # Obtener lista de archivos parquet (búsqueda recursiva)
            carpeta_path = Path(carpeta_origen)
            parquet_files = list(carpeta_path.rglob("*.parquet"))
            
            if not parquet_files:
                st.error("❌ No se encontraron archivos parquet en la carpeta especificada")
                return
            
            # Configurar rutas de salida
            output_dir = PROJECT_ROOT / "indices"
            index_path = output_dir / "boe_index.faiss"
            metadata_path = output_dir / "metadata.json"
            report_path = output_dir / "construction_report.json"
            
            # Crear directorio de salida si no existe
            output_dir.mkdir(exist_ok=True)
            
            # Actualizar progreso y estado
            status_text.text("Inicializando construcción del índice...")
            progress_bar.progress(10)
            log_area.text("📂 Archivos encontrados: " + ", ".join([f.name for f in parquet_files[:5]]) + 
                         (f" ... y {len(parquet_files)-5} más" if len(parquet_files) > 5 else ""))
            
            status_text.text(f"Procesando {len(parquet_files)} archivos parquet...")
            progress_bar.progress(25)
            log_area.text(f"📊 Iniciando construcción con {len(parquet_files)} archivos\n🗂️ Destino: {index_path}")
            
            # Construir índice usando la función importada
            if build_index_from_parquets and imports_ok:
                try:
                    status_text.text("Construyendo índice FAISS...")
                    progress_bar.progress(50)
                    
                    # Convertir paths a strings para la función
                    parquet_file_paths = [str(f) for f in parquet_files]
                    
                    log_area.text(f"📊 Llamando a build_index_from_parquets...\n📁 Archivos: {len(parquet_file_paths)}\n📝 Destino índice: {index_path}\n📄 Destino metadata: {metadata_path}")
                    
                    # Llamar a la función real de construcción
                    result = build_index_from_parquets(
                        parquet_files=parquet_file_paths,
                        output_index_path=str(index_path),
                        output_metadata_path=str(metadata_path),
                        index_type="IVF",
                        dimension=1024
                    )
                    
                    progress_bar.progress(75)
                    status_text.text("Optimizando y guardando índice...")
                    
                    # Crear reporte de construcción
                    import datetime
                    
                    # Manejar el resultado que puede ser un objeto VectorDatabase
                    result_info = "Construcción completada"
                    if result:
                        if hasattr(result, 'index') and hasattr(result, 'metadata'):
                            # Es un objeto VectorDatabase, extraer información relevante
                            try:
                                total_vectors = result.index.ntotal if hasattr(result.index, 'ntotal') else 'Desconocido'
                                result_info = f"Índice FAISS creado con {total_vectors} vectores"
                            except:
                                result_info = "Objeto VectorDatabase creado exitosamente"
                        else:
                            # Convertir a string si es serializable
                            try:
                                result_info = str(result)
                            except:
                                result_info = "Construcción completada"
                    
                    report_data = {
                        "construction_date": datetime.datetime.now().isoformat(),
                        "source_directory": str(carpeta_path),
                        "files_processed": len(parquet_file_paths),
                        "index_path": str(index_path),
                        "metadata_path": str(metadata_path),
                        "result": result_info
                    }
                    
                    with open(report_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(report_data, f, indent=2, ensure_ascii=False)
                    
                    progress_bar.progress(90)
                    log_area.text(f"✅ Índice construido exitosamente\n📊 Resultado: {result_info}\n📄 Reporte guardado en: {report_path}")
                    
                except Exception as construction_error:
                    st.error(f"❌ Error durante la construcción del índice: {str(construction_error)}")
                    log_area.text(f"❌ Error detallado: {str(construction_error)}\n❌ Tipo de error: {type(construction_error).__name__}")
                    
                    # Mostrar información adicional del error en el debug
                    import traceback
                    error_traceback = traceback.format_exc()
                    
                    with st.expander("🔍 Stack Trace Completo", expanded=False):
                        st.code(error_traceback)
                    
                    return
            else:
                st.error("❌ No se pudieron importar las funciones de construcción de índice")
                return
            
            status_text.text("¡Construcción completada!")
            progress_bar.progress(100)
            
            # Mensaje de éxito
            st.success(f"""
            ✅ **Índice construido exitosamente**
            
            - Carpeta origen: `{carpeta_origen}`
            - Archivos procesados: {len(parquet_files)}
            - Índice guardado en: `indices/`
            """)
            
            # Aviso de recarga automática
            st.info("""
            🔄 **Reinicia la aplicación para usar el nuevo índice...**

            La aplicación necesita reiniciarse para cargar el nuevo índice vectorial.
            """)
            
            # Esperar un momento para que el usuario lea el mensaje
            import time
            time.sleep(3)
            
            # Limpiar cache y recargar aplicación
            st.cache_resource.clear()
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error durante la construcción: {str(e)}")
        
        # Área de debugging
        with st.expander("🔍 Información de Debug", expanded=False):
            st.code(str(e))

def ejecutar_pipeline_etl_completo(
    fecha: str,
    directorio_base: str,
    modelo_embeddings: str,
    max_tokens_chunk: int,
    batch_size: int
):
    """Ejecuta el pipeline ETL completo de procesamiento BOE para una fecha específica"""
    
    try:
        # Verificar que el pipeline esté disponible
        if not PipelineBOE or not imports_ok:
            st.error("❌ No se pudo importar el módulo del pipeline completo")
            return
        
        # Convertir fecha a formato YYYYMMDD para directorio
        fecha_dt = datetime.datetime.strptime(fecha, '%Y-%m-%d')
        fecha_dir = fecha_dt.strftime('%Y%m%d')
        
        # Crear contenedor de progreso
        progress_container = st.empty()
        
        with progress_container.container():
            st.markdown("### 🔄 Pipeline ETL en Progreso...")
            
            # Información del proceso
            st.info(f"""
            **Configuración del pipeline:**
            - 📅 Fecha: {fecha}
            - 📁 Directorio base: `{directorio_base}`
            - � Directorio específico: `{directorio_base}/{fecha_dir}/`
            - 🧠 Modelo embeddings: `{modelo_embeddings}`
            - ✂️ Tokens por chunk: {max_tokens_chunk}
            - 🔄 Batch size: {batch_size}
            - 📊 Modo: Actualizar índice existente
            """)
            
            # Barra de progreso principal (6 pasos)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Área de logs en tiempo real
            log_container = st.container()
            with log_container:
                st.markdown("#### 📋 Logs del Pipeline")
                log_area = st.empty()
            
            # Inicializar pipeline
            status_text.text("Inicializando pipeline...")
            progress_bar.progress(0)
            
            try:
                # Crear instancia del pipeline con configuración personalizada
                pipeline = PipelineBOE(
                    base_dir=directorio_base,
                    fecha_procesamiento=fecha_dir
                )
                
                # Aplicar configuraciones personalizadas
                pipeline.embedding_model_name = modelo_embeddings
                pipeline.max_tokens_per_chunk = max_tokens_chunk
                if batch_size > 0:
                    pipeline.batch_size = batch_size
                
                log_area.text(f"✅ Pipeline inicializado\n📁 Directorio base: {directorio_base}\n📂 Directorio fecha: {fecha_dir}\n🧠 Modelo: {modelo_embeddings}")
                
                # Paso 1: Descarga BOE
                status_text.text("Paso 1/6: Descargando sumario BOE...")
                progress_bar.progress(1/6)
                
                resultado_paso1 = pipeline.paso_1_descargar_boe(fecha)
                if not resultado_paso1:
                    raise Exception("Error en la descarga de sumario BOE")
                
                log_area.text(f"✅ Paso 1 completado: Sumario BOE descargado para {fecha}")
                
                # Paso 2: Aplanado de datos
                status_text.text("Paso 2/6: Aplanando datos y descargando cuerpos...")
                progress_bar.progress(2/6)
                
                resultado_paso2 = pipeline.paso_2_aplanar_y_descargar_cuerpos()
                if not resultado_paso2:
                    raise Exception("Error en el aplanado de datos")
                
                log_area.text(f"✅ Paso 2 completado: Datos aplanados y cuerpos descargados")
                
                # Paso 3: Conversión HTML → Markdown
                status_text.text("Paso 3/6: Convirtiendo HTML a Markdown...")
                progress_bar.progress(3/6)
                
                resultado_paso3 = pipeline.paso_3_convertir_html_markdown()
                if not resultado_paso3:
                    raise Exception("Error en la conversión HTML → Markdown")
                
                log_area.text(f"✅ Paso 3 completado: HTML convertido a Markdown")
                
                # Paso 4: Generación de chunks
                status_text.text("Paso 4/6: Generando chunks de texto...")
                progress_bar.progress(4/6)
                
                resultado_paso4 = pipeline.paso_4_generar_chunks()
                if not resultado_paso4:
                    raise Exception("Error en la generación de chunks")
                
                log_area.text(f"✅ Paso 4 completado: {pipeline.stats['chunks_generados']} chunks generados")
                
                # Paso 5: Generación de embeddings
                status_text.text("Paso 5/6: Generando embeddings (puede tardar varios minutos)...")
                progress_bar.progress(5/6)
                
                resultado_paso5 = pipeline.paso_5_generar_embeddings_local()
                if not resultado_paso5:
                    raise Exception("Error en la generación de embeddings")
                
                log_area.text(f"✅ Paso 5 completado: {pipeline.stats['embeddings_creados']} embeddings generados")
                
                # Paso 6: Actualización de índice
                status_text.text("Paso 6/6: Actualizando índice FAISS...")
                progress_bar.progress(6/6)
                
                resultado_paso6 = pipeline.paso_6_actualizar_indice()
                if not resultado_paso6:
                    raise Exception("Error en la actualización del índice")
                
                log_area.text(f"✅ Paso 6 completado: Índice FAISS actualizado")
                
                # Completar progreso
                status_text.text("¡Pipeline ETL completado exitosamente!")
                progress_bar.progress(1.0)
                
                # Estadísticas finales
                stats = pipeline.stats
                
                st.success(f"""
                ✅ **Pipeline ETL completado exitosamente**
                
                **Resumen del procesamiento:**
                - 📅 Fecha procesada: {fecha}
                - � Directorio: {directorio_base}/{fecha_dir}/
                - ✂️ Chunks generados: {stats.get('chunks_generados', 0):,}
                - 🧠 Embeddings creados: {stats.get('embeddings_creados', 0):,}
                - 📊 Archivos indexados: {stats.get('indices_actualizados', 0)}
                - ⏱️ Tiempo total: {stats.get('tiempo_total_segundos', 0):.1f} segundos
                
                **Archivos generados:**
                - `{directorio_base}/{fecha_dir}/json/` → Sumario BOE
                - `{directorio_base}/{fecha_dir}/parquet/` → Datos procesados
                - `{directorio_base}/{fecha_dir}/chunks/` → Chunks de texto
                - `{directorio_base}/{fecha_dir}/embeddings/` → Embeddings
                - `indices/` → Índice FAISS actualizado
                """)
                
                # Mostrar estadísticas detalladas
                with st.expander("📊 Estadísticas Detalladas", expanded=False):
                    st.json(stats)
                
                # Aviso de recarga automática
                st.info("""
                🔄 **Recargando aplicación para usar el nuevo índice...**
                
                La aplicación se reiniciará automáticamente para cargar el índice actualizado.
                """)
                
                # Esperar un momento y recargar
                import time
                time.sleep(3)
                
                # Limpiar cache y recargar aplicación
                st.cache_resource.clear()
                st.rerun()
                
            except Exception as pipeline_error:
                st.error(f"❌ Error durante el pipeline: {str(pipeline_error)}")
                log_area.text(f"❌ Error detallado: {str(pipeline_error)}\n❌ Tipo: {type(pipeline_error).__name__}")
                
                # Mostrar información de debug
                with st.expander("🔍 Información de Debug", expanded=False):
                    import traceback
                    st.code(traceback.format_exc())
                
                return
        
    except Exception as e:
        st.error(f"❌ Error general en el pipeline: {str(e)}")
        
        # Área de debugging
        with st.expander("🔍 Información de Debug", expanded=False):
            st.code(str(e))

if __name__ == "__main__":
    main()
