import os
import pandas as pd
from glob import glob
from tqdm import tqdm

from lib.utils_markdown import extraer_tablas_markdown, eliminar_tablas_markdown, convertir_tablas_markdown_a_texto
from lib.utils import extraer_texto_de_html  # Ajusta el import según tu estructura

DESTINATION_DIR = 'samples/parquet_plano'
PARQUET_DIR = 'samples/parquet'

def procesar_archivos_parquet(source_dir, destination_dir):
    archivos = glob(os.path.join(source_dir, '*.parquet'))
    for archivo in tqdm(archivos, desc="Procesando archivos parquet"):
        df = pd.read_parquet(archivo)
        # Extraer markdown del HTML
        df['markdown'] = df['html'].apply(lambda txt: convertir_tablas_markdown_a_texto(extraer_texto_de_html(txt)))

        # Guardar el resultado
        df.to_parquet(os.path.join(destination_dir, os.path.basename(archivo)), index=False)

if __name__ == "__main__":
    procesar_archivos_parquet(PARQUET_DIR, DESTINATION_DIR)