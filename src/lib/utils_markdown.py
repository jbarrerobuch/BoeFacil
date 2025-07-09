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
        if sep_idx is not None and sep_idx >= 1 and sep_idx+1 < len(lineas):
            n_cols = len([c for c in lineas[sep_idx].split('|') if c.strip()])
            header_main = [c.strip() for c in lineas[sep_idx-1].split('|')[1:-1]]
            header_sub = [c.strip() for c in lineas[sep_idx+1].split('|')[1:-1]]
            # Si hay columnas sin nombre en header_main, es multiindex
            if any(h == '' for h in header_main):
                # Rellenar celdas vacías en header_main con el valor anterior (agrupación)
                for i in range(len(header_main)):
                    if header_main[i] == '':
                        header_main[i] = header_main[i-1] if i > 0 else ''
                columns = list(zip(header_main, header_sub))
                data_lines = lineas[sep_idx+2:]
                data = [ [c.strip() for c in l.split('|')[1:-1]] for l in data_lines if l.strip() ]
                try:
                    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    tablas.append(df)
                except Exception as e:
                    if logger: logger.error(f"Error al procesar la tabla: {tabla_str}\n{e}")
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
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
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
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                tablas.append(df)
            except Exception as e:
                if logger: logger.error(f"Error al procesar la tabla simple: {tabla_str}\n{e}")
                pass
    return tablas

def eliminar_tablas_markdown(texto: str) -> str:
    """
    Elimina todas las tablas en formato Markdown de un texto y devuelve el texto sin las tablas.
    Una tabla Markdown se detecta por el patrón de líneas que empiezan y terminan con '|',
    seguidas de una línea separadora y al menos una fila de datos.
    """
    regex_tabla = r'(\|.+\|\n\|[-| ]+\|\n(?:\|.*\|\n?)*)'
    texto_sin_tablas = re.sub(regex_tabla, '', texto)
    return texto_sin_tablas


