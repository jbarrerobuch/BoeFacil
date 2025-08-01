import re
import pandas as pd
from io import StringIO
import logging

def limpiar_markdown(texto):
    texto_sin_markdown = re.sub(r'[#*_\[\]`]', '', texto)
    texto_sin_markdown = re.sub(r'\n', ' ', texto_sin_markdown)  # Reemplazar saltos de línea por espacios
    texto_sin_markdown = re.sub(r'\\', '', texto_sin_markdown)  # Eliminar caracteres de escape
    return texto_sin_markdown

def extraer_tablas_markdown(texto: str, logger: logging.Logger = None) -> list:
    """
    Extrae tablas de un texto en formato Markdown, soportando cabeceras agrupadas.

    Parámetros:
        texto (str): El texto en formato Markdown del que se extraerán las tablas.

    Devuelve:
        list: Una lista de DataFrames, cada uno representando una tabla extraída del texto.
    """
    tablas = []
    regex_tabla = r'(\|.+\|\n\|[-| ]+\|\n(?:\|.*\|\n?)+)'
    matches = re.findall(regex_tabla, texto)

    for match in matches:
        tabla_str = str(match).strip().replace('\r', '')
        lineas = [l for l in tabla_str.split('\n') if l.strip()]
        
        # Detectar la línea separadora
        sep_idx = None
        for idx, linea in enumerate(lineas):
            if re.match(r'^\|[-| ]+\|$', linea):
                sep_idx = idx
                break
                
        if sep_idx is not None and sep_idx >= 1:
            # Obtener los encabezados
            header_line = lineas[sep_idx-1]
            header_cells = [c.strip() for c in header_line.split('|')[1:-1]]
            
            # Verificar si hay encabezados vacíos (tabla sin encabezado real)
            headers_vacios = all(h == '' for h in header_cells)
            
            if headers_vacios:
                # Tabla sin encabezado válido
                data_lines = lineas[sep_idx+1:]
                data = []
                for line in data_lines:
                    if line.strip():
                        row = [c.strip() for c in line.split('|')[1:-1]]
                        # Filtrar filas que solo contengan separadores
                        if not all(re.match(r'^-+$', cell.strip()) for cell in row if cell.strip()):
                            data.append(row)
                
                if data:
                    # Crear DataFrame sin encabezados
                    df = pd.DataFrame(data)
                    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                    tablas.append(df)
                    
            elif sep_idx+1 < len(lineas):
                # Verificar si es multiindex comparando con la siguiente línea
                next_line = lineas[sep_idx+1]
                next_cells = [c.strip() for c in next_line.split('|')[1:-1]]
                
                # Si la siguiente línea parece ser encabezados secundarios (no datos típicos)
                is_multiindex = False
                if len(next_cells) == len(header_cells):
                    # Verificar si algún encabezado principal está vacío (indicando agrupación)
                    if any(h == '' for h in header_cells):
                        is_multiindex = True
                    # O si los valores en next_cells parecen ser encabezados (no datos numéricos)
                    # y si hay repeticiones en header_cells
                    elif (len(set(header_cells)) < len(header_cells) and
                          not any(cell.isdigit() or re.match(r'^\d+[\.,]\d+$', cell) for cell in next_cells if cell)):
                        is_multiindex = True
                
                if is_multiindex:
                    # Rellenar celdas vacías en header_cells con el valor anterior
                    for i in range(len(header_cells)):
                        if header_cells[i] == '':
                            header_cells[i] = header_cells[i-1] if i > 0 else ''
                    
                    columns = list(zip(header_cells, next_cells))
                    data_lines = lineas[sep_idx+2:]
                    data = []
                    for line in data_lines:
                        if line.strip():
                            row = [c.strip() for c in line.split('|')[1:-1]]
                            data.append(row)
                    
                    try:
                        df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
                        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                        tablas.append(df)
                    except Exception as e:
                        if logger: logger.error(f"Error al procesar la tabla multiindex: {tabla_str}\n{e}")
                        pass
                else:
                    # Header simple
                    data_lines = lineas[sep_idx+1:]
                    tabla_limpia = '\n'.join([lineas[sep_idx-1]] + data_lines)
                    try:
                        df = pd.read_csv(StringIO(tabla_limpia), sep='|', engine='python')
                        df = df.dropna(axis=1, how='all')
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        df.columns = [col.strip() for col in df.columns]
                        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                        tablas.append(df)
                    except Exception as e:
                        if logger: logger.error(f"Error al procesar la tabla simple: {tabla_str}\n{e}")
                        pass
        else:
            # Fallback: intentar leer como tabla markdown simple
            try:
                df = pd.read_csv(StringIO(tabla_str), sep='|', engine='python')
                df = df.dropna(axis=1, how='all')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df.columns = [col.strip() for col in df.columns]
                df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                tablas.append(df)
            except Exception as e:
                if logger: logger.error(f"Error al procesar la tabla simple: {tabla_str}\n{e}")
                pass
    return tablas

def tablas_a_texto(dataframes: list) -> str:
    """
    Convierte una lista de DataFrames a texto plano siguiendo el formato:
    [encabezado principal]/[encabezado secundario] : [valor]
    
    Parámetros:
        dataframes (list): Lista de DataFrames extraídos de tablas markdown
        
    Devuelve:
        str: Texto convertido de las tablas, separado por saltos de línea dobles
    """
    textos_tablas = []
    
    for df in dataframes:
        if df.empty:
            continue
            
        filas_texto = []
        
        # Verificar si es multiindex
        if isinstance(df.columns, pd.MultiIndex):
            # Tabla con encabezado multiindex
            for _, row in df.iterrows():
                pares_fila = []
                for (header_main, header_sub), valor in zip(df.columns, row):
                    if pd.notna(valor) and str(valor).strip():
                        pares_fila.append(f"{header_main}/{header_sub} : {valor}")
                if pares_fila:
                    filas_texto.append(", ".join(pares_fila))
        else:
            # Verificar si es una tabla sin encabezados válidos
            # Características: columnas vacías, números, o patrones como .1, .2
            headers_sin_significado = all(
                col == '' or 
                str(col).strip() == '' or
                (isinstance(col, str) and re.match(r'^\.?\d*$', col.strip())) or
                isinstance(col, (int, float))
                for col in df.columns
            )
            
            if headers_sin_significado:
                # Tabla sin encabezado: concatenar valores por filas separados por comas
                for _, row in df.iterrows():
                    valores_fila = []
                    for valor in row:
                        if (pd.notna(valor) and str(valor).strip() and 
                            not re.match(r'^-+$', str(valor).strip())):
                            valores_fila.append(str(valor).strip())
                    if valores_fila:
                        filas_texto.append(", ".join(valores_fila))
            else:
                # Tabla con encabezado simple
                for _, row in df.iterrows():
                    pares_fila = []
                    for col, valor in zip(df.columns, row):
                        if (pd.notna(valor) and str(valor).strip() and 
                            not re.match(r'^-+$', str(valor).strip())):
                            pares_fila.append(f"{col} : {valor}")
                    if pares_fila:
                        filas_texto.append(", ".join(pares_fila))
        
        if filas_texto:
            textos_tablas.append("\n\n".join(filas_texto))
    
    return "\n\n".join(textos_tablas)

def convertir_tablas_markdown_a_texto(texto: str, logger: logging.Logger = None) -> str:
    """
    Convierte todas las tablas markdown de un texto a formato de texto plano,
    reemplazando las tablas originales por su representación en texto.
    
    Parámetros:
        texto (str): El texto en formato Markdown con tablas
        logger (logging.Logger): Logger opcional para errores
        
    Devuelve:
        str: Texto con las tablas convertidas a formato de texto plano
    """
    # Extraer las tablas
    tablas = extraer_tablas_markdown(texto, logger)
    
    if not tablas:
        return texto
    
    # Convertir tablas a texto
    texto_tablas = tablas_a_texto(tablas)
    
    # Reemplazar las tablas markdown originales por el texto convertido
    regex_tabla = r'(\|.+\|\n\|[-| ]+\|\n(?:\|.*\|\n?)*)'
    
    # Contar las tablas para reemplazar una por una
    matches = list(re.finditer(regex_tabla, texto))
    texto_convertido = texto
    
    # Reemplazar desde el final hacia el inicio para mantener los índices
    for i, match in enumerate(reversed(matches)):
        # Obtener el texto de la tabla correspondiente (en orden inverso)
        idx_tabla = len(matches) - 1 - i
        if idx_tabla < len(tablas):
            tabla_texto = tablas_a_texto([tablas[idx_tabla]])
            texto_convertido = texto_convertido[:match.start()] + tabla_texto + texto_convertido[match.end():]
    
    return texto_convertido

def eliminar_tablas_markdown(texto: str) -> str:
    """
    Elimina todas las tablas en formato Markdown de un texto y devuelve el texto sin las tablas.
    Una tabla Markdown se detecta por el patrón de líneas que empiezan y terminan con '|',
    seguidas de una línea separadora y al menos una fila de datos.
    """
    regex_tabla = r'(\|.+\|\n\|[-| ]+\|\n(?:\|.*\|\n?)*)'
    texto_sin_tablas = re.sub(regex_tabla, '', texto)
    return texto_sin_tablas


