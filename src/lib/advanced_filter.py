"""
Sistema de filtros avanzados para búsquedas en base de datos vectorial BOE.

Este módulo proporciona funcionalidades para aplicar filtros complejos y flexibles
sobre metadatos de documentos BOE, incluyendo filtros por fecha, departamentos,
secciones, y criterios numéricos.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# Configuración del logger
logger = logging.getLogger(__name__)

class AdvancedFilter:
    """
    Sistema de filtros avanzados para metadatos de documentos BOE.
    
    Proporciona filtros flexibles para:
    - Rangos de fechas
    - Departamentos y secciones (código o nombre, exacto o contiene)
    - Criterios numéricos (tokens, chunk numbers)
    - Combinaciones complejas con lógica AND/OR
    """
    
    def __init__(self):
        """Inicializa el sistema de filtros."""
        self.supported_filters = {
            # Filtros de secciones
            'seccion_codigo',
            'seccion_nombre', 
            'seccion_nombre_contains',
            
            # Filtros de departamentos
            'departamento_codigo',
            'departamento_nombre',
            'departamento_nombre_contains',
            
            # Filtros de epígrafes
            'epigrafe_nombre',
            'epigrafe_nombre_contains',
            
            # Filtros numéricos
            'tokens_min',
            'tokens_max',
            'chunk_numero_min',
            'chunk_numero_max',
            
            # Filtros de identificadores
            'sumario_id',
            'item_id'
        }
        
        logger.debug(f"AdvancedFilter inicializado con {len(self.supported_filters)} tipos de filtros")
    
    def apply_metadata_filters(
        self, 
        results: List[Dict[str, Any]], 
        filters: Dict[str, Any],
        require_all: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Aplica filtros de metadatos de forma flexible.
        
        Args:
            results: Lista de resultados a filtrar
            filters: Diccionario con filtros a aplicar
            require_all: Si True usa AND (todos los filtros), si False usa OR (cualquier filtro)
        
        Returns:
            Lista de resultados filtrados
        """
        if not filters:
            return results
        
        logger.debug(f"Aplicando filtros de metadatos: {list(filters.keys())} (require_all={require_all})")
        
        # Validar filtros soportados
        unsupported = set(filters.keys()) - self.supported_filters
        if unsupported:
            logger.warning(f"Filtros no soportados ignorados: {unsupported}")
        
        filtered_results = []
        
        for result in results:
            if require_all:
                # Lógica AND: debe cumplir TODOS los filtros
                if self._matches_all_filters(result, filters):
                    filtered_results.append(result)
            else:
                # Lógica OR: debe cumplir AL MENOS UN filtro
                if self._matches_any_filter(result, filters):
                    filtered_results.append(result)
        
        logger.debug(f"Filtros aplicados: {len(filtered_results)} de {len(results)} resultados")
        return filtered_results
    
    def apply_date_range_filter(
        self, 
        results: List[Dict[str, Any]], 
        date_range: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Filtra resultados por rango de fechas flexible.
        
        Los documentos BOE usan formato de fecha YYYYMMDD (ej: 20231229).
        Las fechas de entrada pueden ser en formato YYYY-MM-DD o YYYYMMDD.
        
        Args:
            results: Lista de resultados a filtrar
            date_range: Diccionario con 'start_date' y/o 'end_date'
        
        Returns:
            Lista de resultados filtrados por fecha
        
        Examples:
            # Solo fecha inicio (formato YYYY-MM-DD)
            date_range = {'start_date': '2023-01-01'}
            
            # Solo fecha fin (formato YYYYMMDD)
            date_range = {'end_date': '20231231'}
            
            # Rango completo (formatos mixtos)
            date_range = {'start_date': '2023-01-01', 'end_date': '20230215'}
        """
        if not date_range:
            return results
        
        start_date = date_range.get('start_date')
        end_date = date_range.get('end_date')
        input_date_format = date_range.get('input_format', '%Y-%m-%d')  # Formato de entrada del usuario
        doc_date_format = '%Y%m%d'  # Formato de fecha en documentos BOE (YYYYMMDD)
        
        logger.debug(f"Aplicando filtro de fecha: {start_date} - {end_date}")
        
        # Validar que al menos una fecha esté especificada
        if not start_date and not end_date:
            logger.warning("Rango de fechas vacío")
            return results
        
        # Convertir fechas string a datetime
        start_dt = None
        end_dt = None
        
        try:
            if start_date:
                # Si la fecha de entrada ya está en formato YYYYMMDD, usarla directamente
                if len(start_date) == 8 and start_date.isdigit():
                    start_dt = datetime.strptime(start_date, doc_date_format)
                else:
                    # Convertir desde formato de entrada (ej: 2023-01-01) a datetime
                    start_dt = datetime.strptime(start_date, input_date_format)
                    
            if end_date:
                # Si la fecha de entrada ya está en formato YYYYMMDD, usarla directamente
                if len(end_date) == 8 and end_date.isdigit():
                    end_dt = datetime.strptime(end_date, doc_date_format)
                else:
                    # Convertir desde formato de entrada (ej: 2023-12-31) a datetime
                    end_dt = datetime.strptime(end_date, input_date_format)
                    
        except ValueError as e:
            logger.error(f"Error en formato de fecha: {e}")
            logger.error(f"Formato esperado de entrada: {input_date_format} (ej: 2023-01-01)")
            logger.error(f"O formato directo de documento: {doc_date_format} (ej: 20230101)")
            return results
        
        filtered_results = []
        skipped_count = 0
        
        for result in results:
            doc_date_str = result.get('fecha_publicacion')
            
            if not doc_date_str:
                skipped_count += 1
                continue
            
            try:
                # Los documentos BOE usan formato YYYYMMDD
                doc_date = datetime.strptime(doc_date_str, doc_date_format)
                
                # Verificar si está en el rango
                in_range = True
                
                if start_dt and doc_date < start_dt:
                    in_range = False
                if end_dt and doc_date > end_dt:
                    in_range = False
                
                if in_range:
                    filtered_results.append(result)
                    
            except ValueError:
                # Formato de fecha inválido en el documento
                logger.debug(f"Formato de fecha inválido en documento: {doc_date_str}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            logger.debug(f"Omitidos {skipped_count} documentos por fecha inválida/faltante")
        
        logger.debug(f"Filtro de fecha aplicado: {len(filtered_results)} de {len(results)} resultados")
        return filtered_results
    
    def _matches_all_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Verifica si un resultado cumple TODOS los filtros (lógica AND).
        
        Args:
            result: Resultado a verificar
            filters: Filtros a aplicar
        
        Returns:
            True si cumple todos los filtros
        """
        for filter_key, filter_value in filters.items():
            if not self._matches_single_filter(result, filter_key, filter_value):
                return False
        return True
    
    def _matches_any_filter(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Verifica si un resultado cumple AL MENOS UN filtro (lógica OR).
        
        Args:
            result: Resultado a verificar
            filters: Filtros a aplicar
        
        Returns:
            True si cumple al menos un filtro
        """
        for filter_key, filter_value in filters.items():
            if self._matches_single_filter(result, filter_key, filter_value):
                return True
        return False
    
    def _matches_single_filter(self, result: Dict[str, Any], filter_key: str, filter_value: Any) -> bool:
        """
        Verifica si un resultado cumple un filtro específico.
        
        Args:
            result: Resultado a verificar
            filter_key: Clave del filtro
            filter_value: Valor del filtro
        
        Returns:
            True si cumple el filtro
        """
        # Filtros de secciones
        if filter_key == 'seccion_codigo':
            return self._matches_list_filter(result.get('seccion_codigo'), filter_value)
        
        elif filter_key == 'seccion_nombre':
            return self._matches_exact_filter(result.get('seccion_nombre'), filter_value)
        
        elif filter_key == 'seccion_nombre_contains':
            return self._matches_contains_filter(result.get('seccion_nombre'), filter_value)
        
        # Filtros de departamentos
        elif filter_key == 'departamento_codigo':
            return self._matches_list_filter(result.get('departamento_codigo'), filter_value)
        
        elif filter_key == 'departamento_nombre':
            return self._matches_exact_filter(result.get('departamento_nombre'), filter_value)
        
        elif filter_key == 'departamento_nombre_contains':
            return self._matches_contains_filter(result.get('departamento_nombre'), filter_value)
        
        # Filtros de epígrafes
        elif filter_key == 'epigrafe_nombre':
            return self._matches_exact_filter(result.get('epigrafe_nombre'), filter_value)
        
        elif filter_key == 'epigrafe_nombre_contains':
            return self._matches_contains_filter(result.get('epigrafe_nombre'), filter_value)
        
        # Filtros numéricos
        elif filter_key == 'tokens_min':
            return self._matches_numeric_min(result.get('tokens_aproximados'), filter_value)
        
        elif filter_key == 'tokens_max':
            return self._matches_numeric_max(result.get('tokens_aproximados'), filter_value)
        
        elif filter_key == 'chunk_numero_min':
            return self._matches_numeric_min(result.get('chunk_numero'), filter_value)
        
        elif filter_key == 'chunk_numero_max':
            return self._matches_numeric_max(result.get('chunk_numero'), filter_value)
        
        # Filtros de identificadores
        elif filter_key == 'sumario_id':
            return self._matches_exact_filter(result.get('sumario_id'), filter_value)
        
        elif filter_key == 'item_id':
            return self._matches_exact_filter(result.get('item_id'), filter_value)
        
        else:
            logger.warning(f"Filtro no reconocido: {filter_key}")
            return True  # No filtrar si el filtro no es reconocido
    
    def _matches_exact_filter(self, doc_value: Any, filter_value: Any) -> bool:
        """Coincidencia exacta."""
        if doc_value is None:
            return False
        return str(doc_value) == str(filter_value)
    
    def _matches_list_filter(self, doc_value: Any, filter_value: Any) -> bool:
        """Coincidencia con lista de valores."""
        if doc_value is None:
            return False
        
        # Si filter_value es una lista, verificar si doc_value está en ella
        if isinstance(filter_value, list):
            return str(doc_value) in [str(v) for v in filter_value]
        else:
            # Si no es lista, tratar como coincidencia exacta
            return str(doc_value) == str(filter_value)
    
    def _matches_contains_filter(self, doc_value: Any, filter_value: Any) -> bool:
        """Verificar si doc_value contiene filter_value (case insensitive)."""
        if doc_value is None or filter_value is None:
            return False
        
        doc_str = str(doc_value).upper()
        filter_str = str(filter_value).upper()
        
        return filter_str in doc_str
    
    def _matches_numeric_min(self, doc_value: Any, filter_value: Any) -> bool:
        """Verificar si doc_value >= filter_value."""
        try:
            doc_num = float(doc_value) if doc_value is not None else 0
            filter_num = float(filter_value)
            return doc_num >= filter_num
        except (ValueError, TypeError):
            return False
    
    def _matches_numeric_max(self, doc_value: Any, filter_value: Any) -> bool:
        """Verificar si doc_value <= filter_value."""
        try:
            doc_num = float(doc_value) if doc_value is not None else 0
            filter_num = float(filter_value)
            return doc_num <= filter_num
        except (ValueError, TypeError):
            return False
    
    def validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y normaliza filtros antes de aplicar.
        
        Args:
            filters: Filtros a validar
        
        Returns:
            Diccionario con filtros validados y mensajes de error
        """
        result = {
            'valid_filters': {},
            'invalid_filters': {},
            'warnings': []
        }
        
        for key, value in filters.items():
            if key not in self.supported_filters:
                result['invalid_filters'][key] = f"Filtro no soportado: {key}"
                continue
            
            # Validaciones específicas por tipo de filtro
            try:
                if key.endswith('_min') or key.endswith('_max'):
                    # Filtros numéricos
                    float(value)  # Verificar que sea numérico
                elif key in ['seccion_codigo', 'departamento_codigo']:
                    # Códigos pueden ser string o lista
                    if isinstance(value, list):
                        value = [str(v) for v in value]
                    else:
                        value = str(value)
                else:
                    # Filtros de texto
                    value = str(value)
                
                result['valid_filters'][key] = value
                
            except (ValueError, TypeError) as e:
                result['invalid_filters'][key] = f"Valor inválido para {key}: {e}"
        
        logger.debug(f"Validación de filtros: {len(result['valid_filters'])} válidos, {len(result['invalid_filters'])} inválidos")
        
        return result
    
    def get_supported_filters(self) -> Dict[str, str]:
        """
        Devuelve documentación de filtros soportados.
        
        Returns:
            Diccionario con descripción de cada filtro
        """
        return {
            'seccion_codigo': 'Código de sección (string o lista): ["I", "II"]',
            'seccion_nombre': 'Nombre exacto de sección: "DISPOSICIONES GENERALES"',
            'seccion_nombre_contains': 'Sección que contiene texto: "DISPOSICIONES"',
            
            'departamento_codigo': 'Código de departamento (string o lista): ["04", "05"]',
            'departamento_nombre': 'Nombre exacto de departamento: "MINISTERIO DE HACIENDA"',
            'departamento_nombre_contains': 'Departamento que contiene: "HACIENDA"',
            
            'epigrafe_nombre': 'Nombre exacto de epígrafe: "OTROS"',
            'epigrafe_nombre_contains': 'Epígrafe que contiene: "IMPUESTO"',
            
            'tokens_min': 'Número mínimo de tokens: 500',
            'tokens_max': 'Número máximo de tokens: 1500',
            'chunk_numero_min': 'Número mínimo de chunk: 1',
            'chunk_numero_max': 'Número máximo de chunk: 3',
            
            'sumario_id': 'ID exacto de sumario: "sumario-20231229"',
            'item_id': 'ID exacto de item: "BOE-A-2023-12345"'
        }
    
    def get_date_format_help(self) -> Dict[str, str]:
        """
        Devuelve ayuda sobre formatos de fecha soportados.
        
        Returns:
            Diccionario con ejemplos de formatos de fecha
        """
        return {
            'documento_format': 'YYYYMMDD (formato interno BOE): 20231229',
            'input_formats': [
                'YYYY-MM-DD (recomendado): 2023-12-29',
                'YYYYMMDD (directo): 20231229'
            ],
            'examples': {
                'single_date': "date_range = {'start_date': '2023-01-01'}",
                'date_range': "date_range = {'start_date': '2023-01-01', 'end_date': '2023-12-31'}",
                'mixed_formats': "date_range = {'start_date': '2023-01-01', 'end_date': '20231231'}"
            }
        }
    
    @staticmethod
    def convert_date_to_boe_format(date_str: str) -> str:
        """
        Convierte una fecha en formato YYYY-MM-DD a formato BOE YYYYMMDD.
        
        Args:
            date_str: Fecha en formato YYYY-MM-DD
            
        Returns:
            Fecha en formato YYYYMMDD
            
        Example:
            convert_date_to_boe_format("2023-12-29") -> "20231229"
        """
        try:
            if len(date_str) == 8 and date_str.isdigit():
                # Ya está en formato YYYYMMDD
                return date_str
            elif len(date_str) == 10 and date_str.count('-') == 2:
                # Formato YYYY-MM-DD, convertir a YYYYMMDD
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                return dt.strftime('%Y%m%d')
            else:
                logger.warning(f"Formato de fecha no reconocido: {date_str}")
                return date_str
        except ValueError as e:
            logger.error(f"Error convirtiendo fecha {date_str}: {e}")
            return date_str
