from lib import boe
from datetime import datetime as dt, timedelta
import pandas as pd

# Configuración de la fecha de extracción y el rango de fechas
extract_date = "2024-01-01"
END_DATE = "2024-01-01"

end_dt = dt.strptime(END_DATE, '%Y-%m-%d')
extract_dt = dt.strptime(extract_date, '%Y-%m-%d')

while extract_dt <= end_dt:
    print(f"Extrayendo datos del BOE para la fecha: {extract_dt.strftime('%Y-%m-%d')}")
    boe.extraer(fecha=extract_dt.strftime('%Y%m%d'), output_dir="samples")
    extract_dt = (extract_dt + timedelta(days=1))

print("Extracción completada.")
