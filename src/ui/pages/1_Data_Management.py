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
from pathlib import Path

# Obtener directorio raíz del proyecto (subir desde src/ui/pages/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Agregar src al path para imports absolutos
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Importar módulos con manejo de errores
try:
    from lib.index_builder import build_index_from_parquets, get_parquet_files_from_directory
    imports_ok = True
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    build_index_from_parquets = None
    get_parquet_files_from_directory = None
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
        
        st.markdown("### 📋 Fases Disponibles")
        
        # Selector de fase
        fase_seleccionada = st.selectbox(
            "Selecciona la fase:",
            [
                "Fase 1: Construcción de Índice",
                "Fase 2: Actualización Incremental",
                "Fase 3: Procesamiento ETL Completo"
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
            st.error("❌ Directorio de samples no encontrado")
    
    # Contenido principal basado en la fase seleccionada
    if "Fase 1" in fase_seleccionada:
        mostrar_fase_1()
    elif "Fase 2" in fase_seleccionada:
        mostrar_fase_2()
    elif "Fase 3" in fase_seleccionada:
        mostrar_fase_3()

def mostrar_fase_1():
    """Interfaz para la Fase 1: Construcción de Índice desde cero"""
    
    st.markdown("## 🏗️ Fase 1: Construcción de Índice desde Cero")
    
    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
        <h4 style="color: #2c3e50; margin-top: 0;">📋 Descripción de la Fase</h4>
        <p style="margin-bottom: 0;">
            Esta fase construye un índice vectorial completamente nuevo desde los archivos parquet disponibles
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

def mostrar_fase_2():
    """Interfaz para la Fase 2: Actualización Incremental"""
    
    st.markdown("## 🔄 Fase 2: Actualización Incremental")
    
    st.info("""
    **🚧 Próximamente**
    
    Esta fase permitirá actualizar el índice existente con nuevos documentos
    sin necesidad de reconstruir todo desde cero.
    """)

def mostrar_fase_3():
    """Interfaz para la Fase 3: Procesamiento ETL Completo"""
    
    st.markdown("## 🔄 Fase 3: Procesamiento ETL Completo")
    
    st.info("""
    **🚧 Próximamente**
    
    Esta fase incluirá el pipeline completo:
    - Descarga de nuevos BOEs
    - Conversión HTML a Markdown
    - Procesamiento y chunking
    - Actualización del índice vectorial
    """)

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
            
            # LÓGICA REAL: Construir índice usando la función importada
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
            🔄 **Recargando aplicación para usar el nuevo índice...**
            
            La aplicación se reiniciará automáticamente en unos segundos para cargar el nuevo índice vectorial.
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

if __name__ == "__main__":
    main()
