import os
import pandas as pd
from glob import glob
from tqdm import tqdm

from lib.utils_markdown import extraer_tablas_markdown, eliminar_tablas_markdown
from lib.utils import extraer_texto_de_html, dataframes_a_json

def procesar_archivo_parquet(filepath:str):
        
        df = pd.read_parquet(filepath)

        # Extraer markdown del HTML
        df['markdown'] = df['html'].apply(extraer_texto_de_html)
        # Extraer tablas del markdown
        df['tablas'] = df['markdown'].apply(lambda md: dataframes_a_json(extraer_tablas_markdown(md)))
        # Eliminar tablas del markdown
        df['markdown'] = df['markdown'].apply(eliminar_tablas_markdown)
        
        return df
        

if __name__ == "__main__":
    PARQUET_DIR = 'samples/parquet'
    archivos = glob(os.path.join(PARQUET_DIR, '*.parquet'))
    for archivo in tqdm(archivos, desc="Procesando archivos parquet"):
        df = procesar_archivo_parquet(archivo)
        df.to_parquet(archivo, index=False)