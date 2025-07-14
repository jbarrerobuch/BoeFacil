import requests
import time
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import logging

from . import utils_markdown
from . import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def flatten_boe(data:dict) -> list:
    """
    Aplanado de la estructura de datos anidada del BOE.
    
    Args:
        data (dict): The BOE API response data
        
    Returns:
        list: A list of dictionaries containing flattened data
    """
    all_items = []
    
    # Check if we have a sumario
    if 'data' not in data or 'sumario' not in data['data']:
        return all_items
    
    sumario = data['data']['sumario']
    metadatos = sumario.get('metadatos', {})
    
    # Process each diario
    for diario in sumario.get('diario', []):
        diario_num = diario.get('numero', '')
        sumario_diario = diario.get('sumario_diario', {})
        
        # Process each section
        for seccion in diario.get('seccion', []):
            # Handle the case where seccion is a dict, not a list
            if not isinstance(seccion, dict):
                continue
                
            seccion_codigo = seccion.get('codigo', '')
            seccion_nombre = seccion.get('nombre', '')
            
            # Check if seccion has texto field with items
            if 'texto' in seccion and isinstance(seccion['texto'], dict) and 'departamento' in seccion['texto']:
                seccion_texto_dept = seccion['texto']['departamento']
                if not isinstance(seccion_texto_dept, list):
                    seccion_texto_dept = [seccion_texto_dept]
                    
                for departamento in seccion_texto_dept:
                    if not isinstance(departamento, dict):
                        continue
                        
                    depto_codigo = departamento.get('codigo', '')
                    depto_nombre = departamento.get('nombre', '')
                    
                    # Process epigrafes in texto->departamento if they exist
                    if 'epigrafe' in departamento:
                        epigrafes = departamento['epigrafe']
                        if not isinstance(epigrafes, list):
                            epigrafes = [epigrafes]
                            
                        for epigrafe in epigrafes:
                            if not isinstance(epigrafe, dict):
                                continue
                                
                            epigrafe_nombre = epigrafe.get('nombre', '')
                            
                            # Process items
                            if 'item' in epigrafe:
                                items = epigrafe['item']
                                if not isinstance(items, list):
                                    items = [items]
                                
                                for item in items:
                                    if isinstance(item, dict):
                                        # Process this item
                                        process_item(item, all_items, metadatos, diario_num, sumario_diario,
                                                  seccion_codigo, seccion_nombre, depto_codigo, depto_nombre, epigrafe_nombre)
            
            # Process each department
            if 'departamento' in seccion:
                departamentos = seccion['departamento']
                if not isinstance(departamentos, list):
                    departamentos = [departamentos] # Convert single department to list

                for departamento in departamentos:
                    if not isinstance(departamento, dict):
                        continue
                        
                    depto_codigo = departamento.get('codigo', '')
                    depto_nombre = departamento.get('nombre', '')
                    
                    # Handle the special case where 'texto' contains 'item' directly
                    if 'texto' in departamento and isinstance(departamento['texto'], dict) and 'item' in departamento['texto']:
                        items = departamento['texto']['item']
                        if not isinstance(items, list):
                            items = [items]  # Convert single item to list
                        
                        for item in items:
                            if isinstance(item, dict):
                                # Process this item
                                process_item(item, all_items, metadatos, diario_num, sumario_diario,
                                          seccion_codigo, seccion_nombre, depto_codigo, depto_nombre, 'Texto')
                    
                    # Process epigrafes if they exist
                    if 'epigrafe' in departamento:
                        epigrafes = departamento['epigrafe']
                        if not isinstance(epigrafes, list):
                            epigrafes = [epigrafes]
                            
                        for epigrafe in epigrafes:
                            if not isinstance(epigrafe, dict):
                                continue
                                
                            epigrafe_nombre = epigrafe.get('nombre', '')
                            
                            # Process items in the epigrafe
                            if 'item' in epigrafe:
                                items = epigrafe['item']
                                if not isinstance(items, list):
                                    items = [items]  # Convert single item to list
                                
                                for item in items:
                                    if isinstance(item, dict):
                                        # Process this item
                                        process_item(item, all_items, metadatos, diario_num, sumario_diario,
                                                  seccion_codigo, seccion_nombre, depto_codigo, depto_nombre, epigrafe_nombre)
                    
                    # Process items directly in the department if they exist
                    if 'item' in departamento:
                        items = departamento['item']
                        if not isinstance(items, list):
                            items = [items]  # Convert single item to list
                        
                        for item in items:
                            if isinstance(item, dict):
                                # Process this item
                                process_item(item, all_items, metadatos, diario_num, sumario_diario,
                                          seccion_codigo, seccion_nombre, depto_codigo, depto_nombre, '')
    
    return all_items

def process_item(item, all_items, metadatos, diario_num, sumario_diario, seccion_codigo, seccion_nombre, depto_codigo, depto_nombre, epigrafe_nombre):
    """
    Función de ayuda para procesar un elemento individual y agregarlo a la lista all_items.
    """
    # extraccion de la URL del PDF y su tamaño
    url_pdf_text = ''
    pdf_kbytes = 0  # Default size if not provided
    if 'url_pdf' in item:
        url_pdf = item['url_pdf']
        if isinstance(url_pdf, dict):
            url_pdf_text = url_pdf.get('texto', '')
            pdf_kbytes = url_pdf.get('szKBytes', 0)
        elif isinstance(url_pdf, str):
            url_pdf_text = url_pdf
            pdf_kbytes = 0  # Default to 0 if not provided

    # extracción de la URL del sumario diario PDF
    sumario_url_pdf = ''
    if sumario_diario and 'url_pdf' in sumario_diario:
        if isinstance(sumario_diario['url_pdf'], dict):
            sumario_url_pdf = sumario_diario['url_pdf'].get('texto', '')
        elif isinstance(sumario_diario['url_pdf'], str):
            sumario_url_pdf = sumario_diario['url_pdf']

    # Extracción de la URL HTML del item
    item_url_html = item.get('url_html', '')
    item_html = ''
    item_markdown = ''
    response = requests.Response()  # Initialize response variable

    if item_url_html:
        try:
            time.sleep(0.1)
            max_retries = 5
            retry_delay = 0.25  # seconds
            for retry in range(max_retries):
                try:
                    response = requests.get(item_url_html, timeout=30)
                    if response.status_code == 200:
                        break
                    elif response.status_code == 429:
                        sleep_time = retry_delay * (2 ** retry)
                        logger.warning(f"Limite de requests alcanzado. Esperando {sleep_time} segundos antes de reintentar...")
                        time.sleep(sleep_time)
                    else:
                        logger.warning(f"Request fallido con status code {response.status_code}, reintento en {retry_delay} segundos...")
                        time.sleep(retry_delay)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error: {str(e)}, reintento en {retry_delay} segundos...")
                    time.sleep(retry_delay)
            # Si la respuesta es exitosa, procesar el HTML
        except Exception as e:
            logger.error(f"Error obteniendo HTML {item_url_html}: {str(e)}")
        else:
            if response.status_code == 200:
                item_html = response.text
                # Convertir HTML a Markdown
                item_markdown = utils.extraer_texto_de_html(item_html)
                # Extraer tablas del Markdown
                tablas_markdown = utils_markdown.extraer_tablas_markdown(item_markdown)
                # Eliminar tablas del Markdown
                if len(tablas_markdown) > 0:
                    tablas_json = utils.dataframes_a_json(tablas_markdown)
                    texto_markdown = utils_markdown.eliminar_tablas_markdown(item_markdown)
                else:
                    tablas_json = None
                    texto_markdown = item_markdown
            else:
                logger.error(f"fetch HTML fallido tras {max_retries} intentos: {item_url_html}")
                item_markdown = ''
    
    # URL al documento XML
    item_url_xml = item.get('url_xml', '')

    item_data = {
        'fecha_publicacion': metadatos.get('fecha_publicacion', ''),
        'publicacion': metadatos.get('publicacion', ''),
        'diario_numero': diario_num,
        'sumario_id': sumario_diario.get('identificador', ''),
        'sumario_url_pdf': sumario_url_pdf,
        'seccion_codigo': seccion_codigo,
        'seccion_nombre': seccion_nombre,
        'departamento_codigo': depto_codigo,
        'departamento_nombre': depto_nombre,
        'epigrafe_nombre': epigrafe_nombre,
        'item_id': item.get('identificador', ''),
        'item_titulo': item.get('titulo', ''),
        'item_url_pdf': url_pdf_text,
        'item_url_html': item_url_html,
        'item_url_xml': item_url_xml,
        'html': item_html,
        'markdown': texto_markdown,
        'tablas': tablas_json,
        'szKBytes': pdf_kbytes,
    }
    all_items.append(item_data)

def extraer_texto_de_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Intenta extraer el contenido principal (ajusta según la estructura de tus datos)
    main = soup.find('div', id='textoxslt')
    if main:
        logger.debug("Texto extraído del HTML")
        return md(str(main))
    else:
        logger.warning("No se encontró el div con id 'textoxslt', extrayendo todo el HTML")
        return md(str(html))

def metricas_de_textos(dataframe, columna='markdown'):
    """
    Calcula métricas de texto en una columna de un DataFrame: número de párrafos, caracteres y palabras,
    excluyendo los caracteres y símbolos de formato Markdown.

    Parámetros:
        dataframe (pd.DataFrame): El DataFrame que contiene la columna de texto.
        columna (str): El nombre de la columna que contiene el texto Markdown. Por defecto es 'markdown'.

    Devuelve:
        tuple: Tres pd.Series con el conteo de caracteres, palabras y párrafos para cada fila.
    """

    num_parrafos = dataframe[columna].apply(lambda texto: texto.count('\n\n') + 1)
    texto_limpio = dataframe[columna].apply(utils_markdown.limpiar_markdown)
    num_caracteres = texto_limpio.apply(len)
    num_palabras = texto_limpio.apply(lambda texto: len(texto.split()))
    
    return num_caracteres, num_palabras, num_parrafos