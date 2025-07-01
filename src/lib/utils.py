import boto3
import os
import json

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