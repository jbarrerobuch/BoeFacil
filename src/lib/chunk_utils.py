import pandas as pd
import re
import logging
import os

# Configuración del logger
logger = logging.getLogger(__name__)

def contar_tokens(texto):
    """
    Aproxima el número de tokens en un texto.
    Utiliza una heurística simple: divide por espacios y signos de puntuación.
    
    Args:
        texto (str): El texto a contar
        
    Returns:
        int: Número aproximado de tokens
    """
    if not texto or not isinstance(texto, str):
        return 0
    
    # Eliminar espacios extra y dividir por espacios y signos de puntuación
    tokens = re.findall(r'\w+|[.,!?;]', texto.strip())
    return len(tokens)

def chunking(texto, max_tokens=1000):
    """
    Divide un texto en chunks basándose en el número máximo de tokens.
    Intenta mantener párrafos completos cuando sea posible.
    
    Args:
        texto (str): El texto a dividir
        max_tokens (int): Número máximo de tokens por chunk. Por defecto 1000.
        
    Returns:
        list: Lista de chunks de texto
    """
    if not texto or not isinstance(texto, str):
        return []
    
    # Si el texto completo cabe en un chunk, devolverlo entero
    if contar_tokens(texto) <= max_tokens:
        return [texto.strip()]
    
    chunks = []
    
    # Dividir por párrafos (doble salto de línea)
    parrafos = texto.split('\n\n')
    
    chunk_actual = ""
    tokens_actuales = 0
    
    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            continue
            
        tokens_parrafo = contar_tokens(parrafo)
        
        # Si el párrafo solo ya excede el límite, dividirlo por oraciones
        if tokens_parrafo > max_tokens:
            # Guardar el chunk actual si tiene contenido
            if chunk_actual.strip():
                chunks.append(chunk_actual.strip())
                chunk_actual = ""
                tokens_actuales = 0
            
            # Dividir el párrafo largo por oraciones
            oraciones = re.split(r'(?<=[.!?])\s+', parrafo)
            
            for oracion in oraciones:
                tokens_oracion = contar_tokens(oracion)
                
                # Si incluso una oración es muy larga, dividirla por palabras
                if tokens_oracion > max_tokens:
                    if chunk_actual.strip():
                        chunks.append(chunk_actual.strip())
                        chunk_actual = ""
                        tokens_actuales = 0
                    
                    palabras = oracion.split()
                    chunk_palabras = ""
                    
                    for palabra in palabras:
                        if contar_tokens(chunk_palabras + " " + palabra) <= max_tokens:
                            if chunk_palabras:
                                chunk_palabras += " " + palabra
                            else:
                                chunk_palabras = palabra
                        else:
                            if chunk_palabras:
                                chunks.append(chunk_palabras)
                            chunk_palabras = palabra
                    
                    if chunk_palabras:
                        chunk_actual = chunk_palabras
                        tokens_actuales = contar_tokens(chunk_palabras)
                
                elif tokens_actuales + tokens_oracion <= max_tokens:
                    if chunk_actual:
                        chunk_actual += " " + oracion
                    else:
                        chunk_actual = oracion
                    tokens_actuales += tokens_oracion
                else:
                    if chunk_actual.strip():
                        chunks.append(chunk_actual.strip())
                    chunk_actual = oracion
                    tokens_actuales = tokens_oracion
        
        # Si el párrafo cabe en el chunk actual
        elif tokens_actuales + tokens_parrafo <= max_tokens:
            if chunk_actual:
                chunk_actual += "\n\n" + parrafo
            else:
                chunk_actual = parrafo
            tokens_actuales += tokens_parrafo
        
        # Si el párrafo no cabe, guardar el chunk actual e iniciar uno nuevo
        else:
            if chunk_actual.strip():
                chunks.append(chunk_actual.strip())
            chunk_actual = parrafo
            tokens_actuales = tokens_parrafo
    
    # Agregar el último chunk si tiene contenido
    if chunk_actual.strip():
        chunks.append(chunk_actual.strip())
    
    return chunks

def chunking_markdown_df(df:pd.DataFrame, columna_markdown:str = 'markdown', max_tokens:int = 1000, texto_id:str = None, store_path:str = None) -> list:
    """
    Procesa un DataFrame que contiene una columna con texto en markdown,
    dividiéndolo en chunks según el número máximo de tokens especificado.
    Opcionalmente guarda los chunks en archivos markdown.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna con markdown
        columna_markdown (str): Nombre de la columna que contiene el markdown. Por defecto 'markdown'.
        max_tokens (int): Número máximo de tokens por chunk. Por defecto 1000.
        texto_id (str): ID del texto para nombrar los archivos. Si es None y existe la columna 'item_id', 
                       se extraerá automáticamente del primer valor no nulo de esa columna.
        store_path (str): Ruta donde guardar los archivos markdown. Si es None, no se guardan archivos.
        
    Returns:
        list: Lista de chunks de texto markdown
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("El parámetro df debe ser un DataFrame de pandas")
        return []
    
    if columna_markdown not in df.columns:
        logger.error(f"La columna '{columna_markdown}' no existe en el DataFrame")
        return []
    
    if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
        logger.error("max_tokens debe ser un número positivo")
        return []
    
    # Validar parámetros para guardado de archivos
    if store_path is not None:
        if texto_id is None:
            # Verificar que existe la columna item_id para extraer el ID de cada fila
            if 'item_id' not in df.columns:
                logger.error("texto_id es requerido cuando se especifica store_path. " + 
                           "Proporciona texto_id explícitamente o asegúrate de que el DataFrame tenga una columna 'item_id'")
                return []
            else:
                logger.info("Se usará la columna 'item_id' para extraer el ID de cada fila")
        
        # Crear directorio si no existe
        try:
            os.makedirs(store_path, exist_ok=True)
            logger.info(f"Directorio creado/verificado: {store_path}")
        except Exception as e:
            logger.error(f"Error creando directorio {store_path}: {e}")
            return []
    
    todos_los_chunks = []
    
    for idx, row in df.iterrows():
        texto_markdown = row[columna_markdown]
        
        if pd.isna(texto_markdown) or not isinstance(texto_markdown, str):
            logger.warning(f"Fila {idx}: contenido vacío o no es string en la columna '{columna_markdown}'")
            continue
        
        # Determinar el texto_id para esta fila específica
        current_texto_id = texto_id
        if current_texto_id is None and store_path is not None:
            # Extraer item_id de esta fila específica
            if pd.isna(row['item_id']):
                logger.warning(f"Fila {idx}: item_id es nulo, se omitirá el guardado de archivos para esta fila")
                current_texto_id = f"fila_{idx}"  # Fallback usando el índice de la fila
            else:
                current_texto_id = str(row['item_id'])
        
        chunks = chunking(texto_markdown, max_tokens)
        
        # Añadir metadatos y guardar archivos si es necesario
        for i, chunk in enumerate(chunks):
            chunk_info = {
                'texto': chunk,
                'fila_original': idx,
                'chunk_numero': i,  # Número dentro de la fila (empezar desde 0)
                'total_chunks_fila': len(chunks),
                'tokens_aproximados': contar_tokens(chunk),
                'item_id': current_texto_id if store_path is not None else (str(row['item_id']) if 'item_id' in df.columns and not pd.isna(row['item_id']) else None)
            }
            
            # Guardar archivo markdown si se especificó store_path
            if store_path is not None:
                filename = f'{current_texto_id}_{i}.md'
                filepath = os.path.join(store_path, filename)
                
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(chunk)
                    
                    # Añadir información del archivo guardado a los metadatos
                    chunk_info['archivo_guardado'] = filepath
                    chunk_info['nombre_archivo'] = filename
                    
                    logger.debug(f"Chunk guardado en: {filepath}")
                    
                except Exception as e:
                    logger.error(f"Error guardando chunk en {filepath}: {e}")
                    chunk_info['error_guardado'] = str(e)
            
            todos_los_chunks.append(chunk_info)
        
        logger.debug(f"Fila {idx}: dividida en {len(chunks)} chunks")
    
    logger.info(f"Procesadas {len(df)} filas, generados {len(todos_los_chunks)} chunks totales")
    
    if store_path is not None:
        chunks_guardados = sum(1 for chunk in todos_los_chunks if 'archivo_guardado' in chunk)
        logger.info(f"Guardados {chunks_guardados} chunks en archivos markdown en {store_path}")

    return todos_los_chunks