import os
import json
from datetime import datetime
import requests
import pandas as pd
import io
import logging

from . import utils_boe
from . import utils


def extraer(fecha:str = "", s3_bucket:str = "", output_dir:str = "output_csv", log_level=logging.INFO):
    """
    Obtener datos del BOE para una fecha especifica y lo guarda en el entorno especificado (local o S3).
    
    Args:
        fecha (str, optional): fecha del BOE a descargar. string format "YYYYMMDD". Si no se define por defecto es hoy.
        s3_bucket (str, optional): S3 bucket para guardar los datos. Por defecto es "" (local).
        csv_output_dir (str, optional): Directorio donde guardar el CSV. Por defecto es "output_csv".
        data (json, optional): The data to process. If provided, the function will not make an API request.
            Defaults to None. If data is provided, date is required.
    Returns:
        pandas.DataFrame: A DataFrame containing the flattened BOE data
    """

    # Validación del valor de fecha

    if len(fecha) == 0 :
        # Si no hay fecha, se usa la fecha actual por defecto
        fecha = datetime.today().strftime('%Y%m%d')
    else:
        try:
            datetime.strptime(fecha, '%Y%m%d')
        except ValueError:
            raise ValueError("La fecha debe estar en el formato 'YYYYMMDD'.")
    
    # Detalles de la API del BOE
    url = f"https://www.boe.es/datosabiertos/api/boe/sumario/{fecha}"
    headers = {"Accept": "application/json"}
    
    # Relizar la solicitud a la API del BOE
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error: código {response.status_code}")
        
        datos_json = response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error en la request de la API: {e}")
        return pd.DataFrame()
    
    else:

        # Guardar JSON en subcarpeta 'json' dentro de csv_output_dir
        json_dir = os.path.join(output_dir, 'json')
        output_file = f"boe_{fecha}.json"
        output_path = os.path.join(json_dir, output_file)

        if not s3_bucket:
            utils.guardar_en_json(
                json_data=datos_json,
                output_path=output_path
            )
        else:
            utils.guardar_en_s3(
                body=json.dumps(datos_json),
                bucket = s3_bucket,
                key = output_path,
                content_type='application/json'
            )
            print(f"Datos JSON guardados en S3 {s3_bucket} with key {output_path}")

    # Process the nested structure
    print("Aplanando datos del BOE...")
    flattened_data = utils_boe.flatten_boe(datos_json)
    
    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)

    # Guardar Parquet en subcarpeta 'parquet' dentro de output_dir
    parquet_output_file = f"boe_data_{fecha}.parquet"
    parquet_subdir = os.path.join(output_dir, 'parquet')
    parquet_output_path = os.path.join(parquet_subdir, parquet_output_file)

    if s3_bucket:
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        utils.guardar_en_s3(
            body=parquet_buffer.getvalue(),
            bucket = s3_bucket,
            key = parquet_output_path,
            content_type='application/octet-stream'
        )
        print(f"Datos guardados en S3 {s3_bucket} with key {parquet_output_path}")
    else:
        os.makedirs(parquet_subdir, exist_ok=True)
        df.to_parquet(parquet_output_path, index=False)
        print(f"Datos guardado en local {parquet_output_path}")
    
    return df
