import boto3
import os
import json
import pandas as pd
import json
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import logging

# Configuración del logger
logger = logging.getLogger(__name__)

def guardar_en_s3(body, bucket, key, content_type = None):
    """
    Guardar el contenido en un bucket de S3 con la clave especificada.
    
    Args:
        body (str): contenido a guardar en S3.
        bucket (str): nombre del bucket de S3.
        key (str): key donde se guardará el contenido en S3.
        content_type (str, optional): tipo de contenido del objeto. Por defecto es None.
    """
    
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type)

def guardar_en_json(json_data: str, output_path: str):
    """
    Guardar el contenido en un archivo JSON en la ruta especificada.
    
    Args:
        json_data (str): The content to save.
        output_path (str): The path where the JSON file will be saved.
    """
    # Asegura que el directorio existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Guarda el JSON correctamente
    with open(output_path, 'w', encoding='utf-8') as f:
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except Exception:
                data = json_data
        else:
            data = json_data
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Datos JSON guardados en {output_path}")

def dataframes_a_json(lista_dfs):
    """
    Convierte una lista de DataFrames en un solo objeto JSON (lista de diccionarios).
    Si hay nombres de columnas no únicos, los renombra para que sean únicos.
    Soporta DataFrames con columnas MultiIndex convirtiendo las claves a string.
    Args:
        lista_dfs (list): Lista de pandas.DataFrame
    Returns:
        str: Cadena JSON representando la lista de DataFrames como lista de listas de diccionarios.
    """
    def make_unique_columns(cols):
        seen = {}
        new_cols = []
        for col in cols:
            col_str = str(col)
            if col_str in seen:
                seen[col_str] += 1
                new_cols.append(f"{col_str}_{seen[col_str]}")
            else:
                seen[col_str] = 0
                new_cols.append(col_str)
        return new_cols

    def dict_keys_to_str(d):
        if isinstance(d, dict):
            return {str(k): dict_keys_to_str(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [dict_keys_to_str(i) for i in d]
        else:
            return d

    lista_dicts = []
    for df in lista_dfs:
        # Si las columnas no son únicas, renombrarlas
        if not df.columns.is_unique:
            if isinstance(df.columns, pd.MultiIndex):
                # Para MultiIndex, convertir a strings y hacer únicos
                cols = ['|'.join(map(str, col)) for col in df.columns]
            else:
                cols = [str(col) for col in df.columns]
            df = df.copy()
            df.columns = make_unique_columns(cols)
        lista_dicts.append(dict_keys_to_str(df.to_dict(orient='records')))
    return json.dumps(lista_dicts, ensure_ascii=False, indent=4)

def extraer_texto_de_html(html, div_id='textoxslt'):
    """
    Extrae el texto de un HTML, buscando el div con id definido. 
    Ignorando links y blockquotes.
    Si no se encuentra, devuelve todo el HTML convertido a Markdown.
    Args:
        html (str): El contenido HTML del que se extraerá el texto.
        div_id (str): El id del div del que se extraerá el texto. Por defecto es 'textoxslt'.
    
    Returns:
        str: El texto extraído del HTML, convertido a Markdown.
    """

    soup = BeautifulSoup(html, 'html.parser')
    main = soup.find('div', id=div_id)
    if main:
        logger.debug("Texto extraído del HTML")
        return md(str(main), strip=["a", "blockquote"])
    else:
        logger.warning(f"No se encontró el div con id '{div_id}', extrayendo todo el HTML")
        return md(str(html), strip=["a", "blockquote"])
