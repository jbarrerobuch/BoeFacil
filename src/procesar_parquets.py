import os
import pandas as pd
from glob import glob
from tqdm import tqdm

from lib.utils_markdown import extraer_tablas_markdown, eliminar_tablas_markdown
from lib.utils_html import extraer_texto_de_html  # Ajusta el import según tu estructura

PARQUET_DIR = 'samples/parquet'

def procesar_archivos_parquet(parquet_dir):
    archivos = glob(os.path.join(parquet_dir, '*.parquet'))
    for archivo in tqdm(archivos, desc="Procesando archivos parquet"):
        df = pd.read_parquet(archivo)
        # Extraer markdown del HTML
        df['markdown'] = df['html'].apply(extraer_texto_de_html)
        # Extraer tablas del markdown
        df['tablas'] = df['markdown'].apply(lambda md: extraer_tablas_markdown(md))
        # Eliminar tablas del markdown
        df['markdown'] = df['markdown'].apply(eliminar_tablas_markdown)
        # Guardar el resultado
        df.to_parquet(archivo.replace('.parquet', '_procesado.parquet'), index=False)

if __name__ == "__main__":
    procesar_archivos_parquet(PARQUET_DIR)