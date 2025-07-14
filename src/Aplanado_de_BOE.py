import os
import io
import json
import pandas as pd
import logging
import time
import datetime as dt
from lib import utils_boe, utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = "samples/parquet"

# Leer todos los archivos JSON de la carpeta samples/json
dataframes = []
samples_json_dir = os.path.join('samples', 'json')
#file_list = os.listdir(samples_json_dir)
file_list = [f'boe_data_20241231.parquet']

for filename in file_list:
    print(f"Procesando archivo: {filename}")
    if filename.endswith('.json'):
        filepath = os.path.join(samples_json_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            datos_json = json.load(f)
        # Process the nested structure
        logger.info(f"Aplanando datos del BOE para {filename}...")
        flattened_data = utils_boe.flatten_boe(datos_json)
        df = pd.DataFrame(flattened_data)
        # Guardar Parquet en subcarpeta 'parquet' dentro de output_dir
        fecha = filename.replace('boe_', '').replace('.json', '')
        parquet_output_file = f"boe_data_{fecha}.parquet"
        parquet_subdir = os.path.join(output_dir, 'parquet')
        parquet_output_path = os.path.join(parquet_subdir, parquet_output_file)

        os.makedirs(parquet_subdir, exist_ok=True)
        df.to_parquet(parquet_output_path, index=False)
        logger.info(f"Datos guardado en local {parquet_output_path}")
    
    time.sleep(0.5)  # Pausa para evitar sobrecargar el sistema
    
