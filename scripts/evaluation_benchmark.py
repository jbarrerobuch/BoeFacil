#!/usr/bin/env python3
"""
Script de evaluación comparativa: BGE-M3 vs TF-IDF Baseline.

Este script implementa un experimento riguroso para comparar el rendimiento
del sistema de búsqueda semántica BGE-M3 contra un baseline TF-IDF clásico.

Para uso académico en Trabajo de Fin de Máster.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lib.boe_search_api import BOESearchAPI
from lib.vector_db import VectorDatabase

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TFIDFBaselineEngine:
    """
    Motor de búsqueda baseline usando TF-IDF clásico.
    
    Implementa la misma interfaz que BOESearchAPI para comparación justa.
    """
    
    def __init__(self, metadata_path: str):
        """
        Inicializa el motor TF-IDF baseline.
        
        Args:
            metadata_path: Ruta al archivo de metadatos JSON con los textos
        """
        self.metadata_path = metadata_path
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.metadata = []
        
        logger.info("Inicializando TF-IDF Baseline Engine")
        self._load_and_prepare_data()
        self._build_tfidf_index()
    
    def _load_and_prepare_data(self):
        """Carga los metadatos y extrae los textos para TF-IDF."""
        logger.info(f"Cargando metadatos desde {self.metadata_path}")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # El archivo metadata.json tiene estructura: {"metadata": [...], "config": {...}}
        if isinstance(data, dict) and 'metadata' in data:
            self.metadata = data['metadata']
        elif isinstance(data, list):
            self.metadata = data
        else:
            raise ValueError(f"Formato de metadata no reconocido: {type(data)}")
        
        # Extraer textos limpios
        self.documents = []
        for item in self.metadata:
            if isinstance(item, dict):
                texto = item.get('texto', '')
            else:
                # Si el item no es un dict, intentar convertirlo a string
                texto = str(item)
            
            # Limpieza básica del texto
            texto_limpio = self._clean_text(texto)
            self.documents.append(texto_limpio)
        
        logger.info(f"Cargados {len(self.documents)} documentos")
    
    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto para TF-IDF."""
        import re
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Remover caracteres especiales pero mantener acentos españoles
        text = re.sub(r'[^\w\sáéíóúñü]', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _build_tfidf_index(self):
        """Construye el índice TF-IDF."""
        logger.info("Construyendo índice TF-IDF...")
        
        # Configuración del vectorizador
        self.vectorizer = TfidfVectorizer(
            max_features=50000,          # Limitar vocabulario
            min_df=2,                    # Ignorar términos muy raros
            max_df=0.8,                  # Ignorar términos muy comunes
            ngram_range=(1, 2),          # Unigramas y bigramas
            stop_words=None,             # Sin stop words para legal
            token_pattern=r'\b\w{2,}\b'  # Palabras de al menos 2 caracteres
        )
        
        # Construir matriz TF-IDF
        start_time = time.time()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        build_time = time.time() - start_time
        
        logger.info(f"Índice TF-IDF construido en {build_time:.2f} segundos")
        logger.info(f"Forma de la matriz: {self.tfidf_matrix.shape}")
        logger.info(f"Vocabulario: {len(self.vectorizer.vocabulary_)} términos")
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Realiza búsqueda TF-IDF.
        
        Args:
            query: Consulta del usuario
            limit: Número máximo de resultados
            
        Returns:
            Lista de resultados ordenados por similitud
        """
        start_time = time.time()
        
        # Limpiar query
        query_clean = self._clean_text(query)
        
        # Vectorizar query
        query_vector = self.vectorizer.transform([query_clean])
        
        # Calcular similitudes
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Obtener índices de los top-K resultados
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        # Construir resultados
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Solo resultados con similitud > 0
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(similarities[idx])
                result['search_time'] = time.time() - start_time
                results.append(result)
        
        search_time = time.time() - start_time
        logger.debug(f"Búsqueda TF-IDF completada en {search_time:.3f}s")
        
        return results

class ExperimentComparator:
    """
    Clase principal para ejecutar experimentos comparativos entre BGE-M3 y TF-IDF.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el comparador de experimentos.
        
        Args:
            config: Configuración del experimento
        """
        self.config = config
        
        # Inicializar motores de búsqueda
        logger.info("Inicializando motores de búsqueda...")
        
        # Motor BGE-M3 (actual)
        self.bge_engine = BOESearchAPI(
            index_path=config['index_path'],
            metadata_path=config['metadata_path'],
            model_name=config.get('model_name', 'pablosi/bge-m3-trained-2')
        )
        
        # Motor TF-IDF (baseline)
        self.tfidf_engine = TFIDFBaselineEngine(
            metadata_path=config['metadata_path']
        )
        
        # Generar queries de prueba
        self.test_queries = self._generate_test_queries()
        
        logger.info(f"Experimento inicializado con {len(self.test_queries)} queries")
    
    def _generate_test_queries(self) -> List[Dict[str, Any]]:
        """
        Genera un conjunto diverso de queries para evaluación.
        
        Returns:
            Lista de queries categorizadas
        """
        queries = []
        
        # CATEGORÍA 1: Consultas legales generales (25 queries)
        legal_queries = [
            "real decreto impuestos sociedades",
            "ley orgánica educación universitaria", 
            "disposiciones generales tributarias",
            "orden ministerial funcionarios públicos",
            "resolución dirección general tráfico",
            "normativa contratación pública sector",
            "reglamento procedimiento administrativo común",
            "ley protección datos personales",
            "decreto regulación actividades económicas",
            "ordenanza municipal régimen jurídico",
            "instrucción técnica complementaria seguridad",
            "circular interpretativa normativa fiscal",
            "protocolo actuación servicios sociales",
            "directiva europea transposición nacional",
            "acuerdo consejo ministros competencias",
            "norma técnica homologación productos",
            "régimen especial zona franca",
            "estatuto trabajadores reforma laboral",
            "código penal modificación artículos",
            "ley presupuestos generales estado",
            "decreto organización estructura ministerial",
            "ordenación territorial urbana regional",
            "régimen jurídico administración local",
            "procedimiento recurso alzada administrativa",
            "normativa europea directiva marco"
        ]
        
        # CATEGORÍA 2: Consultas por ministerio/departamento (25 queries)
        ministry_queries = [
            "ministerio hacienda función pública",
            "ministerio interior seguridad ciudadana",
            "ministerio defensa fuerzas armadas",
            "ministerio justicia registros notariado",
            "ministerio educación cultura deporte",
            "ministerio sanidad consumo bienestar",
            "ministerio trabajo economía social",
            "ministerio industria comercio turismo",
            "ministerio agricultura pesca alimentación",
            "ministerio transición ecológica sostenible",
            "ministerio ciencia innovación universidades",
            "ministerio asuntos exteriores cooperación",
            "ministerio presidencia memoria democrática",
            "banco españa política monetaria",
            "comisión nacional mercados competencia",
            "agencia española protección datos",
            "instituto nacional estadística censo",
            "servicio público empleo estatal",
            "instituto social marina pesquera",
            "organismo autónomo trabajo penitenciario",
            "consejo seguridad nuclear radioprotección",
            "agencia estatal administración tributaria",
            "guardia civil seguridad vial",
            "instituto nacional gestión sanitaria",
            "consejo estado dictámenes informes"
        ]
        
        # CATEGORÍA 3: Consultas temporales/específicas por fecha (25 queries)
        temporal_queries = [
            "enero 2023 disposiciones fiscales",
            "diciembre 2023 presupuestos generales",
            "segundo trimestre 2023 empleo",
            "normativa 2023 protección consumidores",
            "medidas covid enero 2023",
            "reforma laboral marzo 2023",
            "subvenciones europeas 2023 agricultura",
            "plan recovery abril 2023",
            "ley vivienda mayo 2023",
            "medidas ahorro energético verano 2023",
            "becas educación curso 2023-2024",
            "modificación iva julio 2023",
            "ayudas transporte público 2023",
            "fondos next generation 2023",
            "plan pensiones reforma 2023",
            "medidas inflación agosto 2023",
            "convocatoria oposiciones 2023",
            "régimen fiscal comunidades autónomas 2023",
            "directiva europea septiembre 2023",
            "plan nacional recuperación 2023",
            "medidas sequía octubre 2023",
            "ley startup noviembre 2023",
            "reforma código penal 2023",
            "plan digitalización diciembre 2023",
            "presupuestos 2024 aprobación"
        ]
        
        # CATEGORÍA 4: Consultas técnicas específicas (25 queries)
        technical_queries = [
            "contratación pública procedimiento abierto",
            "subvenciones europeas fondo desarrollo regional",
            "licencias ambientales evaluación impacto",
            "registro mercantil sociedades anónimas",
            "procedimiento sancionador infracciones graves",
            "recurso contencioso administrativo plazos",
            "expropiación forzosa utilidad pública",
            "régimen especial trabajadores autónomos",
            "inspección trabajo seguridad social",
            "procedimiento concursal empresas insolvencia",
            "régimen fiscal cooperativas sociedades laborales",
            "autorizaciones sanitarias productos farmacéuticos",
            "evaluación conformidad marcado ce",
            "protección datos tratamiento información",
            "propiedad intelectual derechos autor",
            "régimen jurídico extranjería inmigración",
            "procedimiento electoral censo votantes",
            "régimen disciplinario funcionarios públicos",
            "contratación laboral temporal indefinida",
            "procedimiento administrativo silencio positivo",
            "régimen fiscal entidades sin ánimo lucro",
            "autorizaciones obras públicas dominio",
            "procedimiento responsabilidad patrimonial administración",
            "régimen jurídico sociedades profesionales",
            "evaluación ambiental estratégica proyectos"
        ]
        
        # Combinar todas las queries con metadatos
        for i, query in enumerate(legal_queries):
            queries.append({
                'id': f'legal_{i+1:02d}',
                'text': query,
                'category': 'legal_general',
                'expected_topics': ['legislation', 'law', 'decree']
            })
        
        for i, query in enumerate(ministry_queries):
            queries.append({
                'id': f'ministry_{i+1:02d}',
                'text': query,
                'category': 'ministry_department',
                'expected_topics': ['ministry', 'government', 'public_administration']
            })
        
        for i, query in enumerate(temporal_queries):
            queries.append({
                'id': f'temporal_{i+1:02d}',
                'text': query,
                'category': 'temporal_specific',
                'expected_topics': ['recent', 'temporal', 'date_specific']
            })
        
        for i, query in enumerate(technical_queries):
            queries.append({
                'id': f'technical_{i+1:02d}',
                'text': query,
                'category': 'technical_specific',
                'expected_topics': ['procedure', 'regulation', 'technical']
            })
        
        logger.info(f"Generadas {len(queries)} queries distribuidas en 4 categorías")
        return queries

    def run_comparative_experiment(self, limit_results: int = 20) -> Dict[str, Any]:
        """
        Ejecuta el experimento comparativo completo BGE-M3 vs TF-IDF.
        
        Args:
            limit_results: Número máximo de resultados por búsqueda
            
        Returns:
            Diccionario con todos los resultados del experimento
        """
        logger.info("=== INICIANDO EXPERIMENTO COMPARATIVO BGE-M3 vs TF-IDF ===")
        
        experiment_results = {
            'config': self.config,
            'start_time': datetime.now().isoformat(),
            'total_queries': len(self.test_queries),
            'results_per_query': limit_results,
            'bge_results': [],
            'tfidf_results': [],
            'comparative_metrics': {}
        }
        
        metrics_calculator = MetricsCalculator()
        
        # Contadores de progreso
        total_queries = len(self.test_queries)
        
        for i, query in enumerate(self.test_queries):
            logger.info(f"Procesando query {i+1}/{total_queries}: '{query['text'][:50]}...'")
            
            # === BÚSQUEDA BGE-M3 ===
            start_time = time.time()
            try:
                bge_results = self.bge_engine.search(query['text'], limit=limit_results)
                bge_search_time = time.time() - start_time
                
                # Calcular métricas BGE-M3
                bge_metrics = metrics_calculator.calculate_comprehensive_metrics(bge_results, query)
                bge_metrics['search_time'] = bge_search_time
                bge_metrics['engine'] = 'bge_m3'
                
            except Exception as e:
                logger.error(f"Error en búsqueda BGE-M3 para query {query['id']}: {e}")
                bge_results = []
                bge_metrics = {'error': str(e), 'engine': 'bge_m3'}
            
            # === BÚSQUEDA TF-IDF ===
            start_time = time.time()
            try:
                tfidf_results = self.tfidf_engine.search(query['text'], limit=limit_results)
                tfidf_search_time = time.time() - start_time
                
                # Calcular métricas TF-IDF
                tfidf_metrics = metrics_calculator.calculate_comprehensive_metrics(tfidf_results, query)
                tfidf_metrics['search_time'] = tfidf_search_time
                tfidf_metrics['engine'] = 'tfidf'
                
            except Exception as e:
                logger.error(f"Error en búsqueda TF-IDF para query {query['id']}: {e}")
                tfidf_results = []
                tfidf_metrics = {'error': str(e), 'engine': 'tfidf'}
            
            # Guardar resultados de esta query
            experiment_results['bge_results'].append({
                'query': query,
                'results': bge_results,
                'metrics': bge_metrics
            })
            
            experiment_results['tfidf_results'].append({
                'query': query,
                'results': tfidf_results,
                'metrics': tfidf_metrics
            })
            
            # Log de progreso cada 10 queries
            if (i + 1) % 10 == 0:
                logger.info(f"Completadas {i+1}/{total_queries} queries...")
        
        # Calcular métricas comparativas agregadas
        experiment_results['comparative_metrics'] = self._calculate_aggregate_metrics(
            experiment_results['bge_results'],
            experiment_results['tfidf_results']
        )
        
        experiment_results['end_time'] = datetime.now().isoformat()
        logger.info("=== EXPERIMENTO COMPLETADO ===")
        
        return experiment_results
    
    def _calculate_aggregate_metrics(self, bge_results: List[Dict], tfidf_results: List[Dict]) -> Dict[str, Any]:
        """
        Calcula métricas agregadas para comparación entre motores.
        
        Args:
            bge_results: Resultados de BGE-M3
            tfidf_results: Resultados de TF-IDF
            
        Returns:
            Métricas comparativas agregadas
        """
        def aggregate_metrics_for_engine(results, engine_name):
            metrics_list = []
            search_times = []
            
            for result in results:
                if 'error' not in result['metrics']:
                    metrics_list.append(result['metrics'])
                    search_times.append(result['metrics'].get('search_time', 0))
            
            if not metrics_list:
                return {}
            
            # Calcular promedios para cada métrica
            aggregated = {}
            for metric_name in metrics_list[0].keys():
                if metric_name not in ['engine', 'search_time']:
                    values = [m.get(metric_name, 0) for m in metrics_list if isinstance(m.get(metric_name), (int, float))]
                    if values:
                        aggregated[f'avg_{metric_name}'] = np.mean(values)
                        aggregated[f'std_{metric_name}'] = np.std(values)
            
            # Métricas de tiempo
            if search_times:
                aggregated['avg_search_time'] = np.mean(search_times)
                aggregated['median_search_time'] = np.median(search_times)
                aggregated['std_search_time'] = np.std(search_times)
            
            aggregated['total_queries'] = len(metrics_list)
            aggregated['successful_queries'] = len([m for m in metrics_list if m.get('has_results', 0) > 0])
            
            return aggregated
        
        # Calcular métricas para cada motor
        bge_aggregated = aggregate_metrics_for_engine(bge_results, 'bge_m3')
        tfidf_aggregated = aggregate_metrics_for_engine(tfidf_results, 'tfidf')
        
        # Calcular mejoras relativas
        improvements = {}
        for metric in ['avg_precision_at_1', 'avg_precision_at_5', 'avg_precision_at_10', 'avg_mrr', 'avg_average_precision']:
            if metric in bge_aggregated and metric in tfidf_aggregated and tfidf_aggregated[metric] > 0:
                improvement = ((bge_aggregated[metric] - tfidf_aggregated[metric]) / tfidf_aggregated[metric]) * 100
                improvements[f'{metric}_improvement_percent'] = improvement
        
        # Comparación de velocidad
        if 'avg_search_time' in bge_aggregated and 'avg_search_time' in tfidf_aggregated:
            speed_ratio = bge_aggregated['avg_search_time'] / tfidf_aggregated['avg_search_time']
            improvements['speed_ratio_bge_vs_tfidf'] = speed_ratio
        
        return {
            'bge_m3': bge_aggregated,
            'tfidf': tfidf_aggregated,
            'improvements': improvements
        }


class MetricsCalculator:
    """
    Calculadora de métricas para evaluación de motores de búsqueda.
    
    Implementa métricas estándar de Information Retrieval:
    - Precision@K
    - Recall@K (cuando sea posible calcular)
    - Mean Reciprocal Rank (MRR)
    - Mean Average Precision (MAP)
    - Métricas de eficiencia temporal
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_relevance_score(self, query: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calcula un score de relevancia heurístico entre query y resultado.
        
        Args:
            query: Query con metadatos
            result: Resultado de búsqueda con metadatos
            
        Returns:
            Score de relevancia entre 0 y 1
        """
        score = 0.0
        query_text = query['text'].lower()
        result_text = result.get('texto', '').lower()
        result_title = result.get('item_titulo', '').lower()
        
        # 1. Coincidencia exacta de palabras clave (40% del score)
        query_words = set(query_text.split())
        result_words = set(result_text.split()) | set(result_title.split())
        
        if query_words and result_words:
            word_overlap = len(query_words & result_words) / len(query_words)
            score += 0.4 * word_overlap
        
        # 2. Score de similitud del motor (30% del score)
        similarity = result.get('similarity_score', 0.0)
        score += 0.3 * similarity
        
        # 3. Relevancia temática por sección (20% del score) 
        expected_topics = query.get('expected_topics', [])
        result_section = result.get('seccion_nombre', '').lower()
        
        topic_match = False
        for topic in expected_topics:
            if topic in result_section or any(word in result_section for word in topic.split('_')):
                topic_match = True
                break
        
        if topic_match:
            score += 0.2
        
        # 4. Bonus por título relevante (10% del score)
        title_words = set(result_title.split())
        if query_words & title_words:
            title_overlap = len(query_words & title_words) / len(query_words)
            score += 0.1 * title_overlap
        
        return min(score, 1.0)  # Limitar a 1.0
    
    def precision_at_k(self, results: List[Dict[str, Any]], query: Dict[str, Any], k: int) -> float:
        """
        Calcula Precision@K.
        
        Args:
            results: Lista de resultados ordenados por relevancia
            query: Query original con metadatos
            k: Número de documentos top-K a considerar
            
        Returns:
            Precision@K score
        """
        if not results or k <= 0:
            return 0.0
        
        # Tomar solo los primeros K resultados
        top_k_results = results[:k]
        
        # Calcular relevancia de cada resultado
        relevant_count = 0
        for result in top_k_results:
            relevance = self.calculate_relevance_score(query, result)
            if relevance >= 0.5:  # Umbral de relevancia
                relevant_count += 1
        
        return relevant_count / k
    
    def mean_reciprocal_rank(self, results: List[Dict[str, Any]], query: Dict[str, Any]) -> float:
        """
        Calcula Mean Reciprocal Rank (MRR).
        
        Args:
            results: Lista de resultados ordenados
            query: Query original
            
        Returns:
            MRR score
        """
        if not results:
            return 0.0
        
        for i, result in enumerate(results):
            relevance = self.calculate_relevance_score(query, result)
            if relevance >= 0.7:  # Umbral más alto para primer resultado relevante
                return 1.0 / (i + 1)
        
        return 0.0
    
    def average_precision(self, results: List[Dict[str, Any]], query: Dict[str, Any]) -> float:
        """
        Calcula Average Precision (AP).
        
        Args:
            results: Lista de resultados ordenados
            query: Query original
            
        Returns:
            AP score
        """
        if not results:
            return 0.0
        
        relevant_docs = 0
        ap_sum = 0.0
        
        for i, result in enumerate(results):
            relevance = self.calculate_relevance_score(query, result)
            if relevance >= 0.5:
                relevant_docs += 1
                precision_at_i = relevant_docs / (i + 1)
                ap_sum += precision_at_i
        
        return ap_sum / len(results) if results else 0.0
    
    def calculate_comprehensive_metrics(self, results: List[Dict[str, Any]], query: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula todas las métricas comprehensivas.
        
        Args:
            results: Resultados de búsqueda
            query: Query original
            
        Returns:
            Diccionario con todas las métricas
        """
        metrics = {}
        
        # Precision@K para diferentes valores de K
        for k in [1, 5, 10, 20]:
            metrics[f'precision_at_{k}'] = self.precision_at_k(results, query, k)
        
        # MRR
        metrics['mrr'] = self.mean_reciprocal_rank(results, query)
        
        # Average Precision
        metrics['average_precision'] = self.average_precision(results, query)
        
        # Métricas de cobertura
        metrics['num_results'] = len(results)
        metrics['has_results'] = 1.0 if results else 0.0
        
        return metrics


class ExperimentRunner:
    """Clase para ejecutar el experimento completo."""
    
    def __init__(self, comparator: 'ExperimentComparator'):
        self.comparator = comparator
    
    def run_comparative_experiment(self, limit_results: int = 20) -> Dict[str, Any]:
        """
        Ejecuta el experimento comparativo completo BGE-M3 vs TF-IDF.
        
        Args:
            limit_results: Número máximo de resultados por búsqueda
            
        Returns:
            Diccionario con todos los resultados del experimento
        """
        logger.info("=== INICIANDO EXPERIMENTO COMPARATIVO BGE-M3 vs TF-IDF ===")
        
        experiment_results = {
            'config': self.config,
            'start_time': datetime.now().isoformat(),
            'total_queries': len(self.test_queries),
            'results_per_query': limit_results,
            'bge_results': [],
            'tfidf_results': [],
            'comparative_metrics': {}
        }
        
        metrics_calculator = MetricsCalculator()
        
        # Contadores de progreso
        total_queries = len(self.test_queries)
        
        for i, query in enumerate(self.test_queries):
            logger.info(f"Procesando query {i+1}/{total_queries}: '{query['text'][:50]}...'")
            
            # === BÚSQUEDA BGE-M3 ===
            start_time = time.time()
            try:
                bge_results = self.bge_engine.search(query['text'], limit=limit_results)
                bge_search_time = time.time() - start_time
                
                # Calcular métricas BGE-M3
                bge_metrics = metrics_calculator.calculate_comprehensive_metrics(bge_results, query)
                bge_metrics['search_time'] = bge_search_time
                bge_metrics['engine'] = 'bge_m3'
                
            except Exception as e:
                logger.error(f"Error en búsqueda BGE-M3 para query {query['id']}: {e}")
                bge_results = []
                bge_metrics = {'error': str(e), 'engine': 'bge_m3'}
            
            # === BÚSQUEDA TF-IDF ===
            start_time = time.time()
            try:
                tfidf_results = self.tfidf_engine.search(query['text'], limit=limit_results)
                tfidf_search_time = time.time() - start_time
                
                # Calcular métricas TF-IDF
                tfidf_metrics = metrics_calculator.calculate_comprehensive_metrics(tfidf_results, query)
                tfidf_metrics['search_time'] = tfidf_search_time
                tfidf_metrics['engine'] = 'tfidf'
                
            except Exception as e:
                logger.error(f"Error en búsqueda TF-IDF para query {query['id']}: {e}")
                tfidf_results = []
                tfidf_metrics = {'error': str(e), 'engine': 'tfidf'}
            
            # Guardar resultados de esta query
            experiment_results['bge_results'].append({
                'query': query,
                'results': bge_results,
                'metrics': bge_metrics
            })
            
            experiment_results['tfidf_results'].append({
                'query': query,
                'results': tfidf_results,
                'metrics': tfidf_metrics
            })
            
            # Log de progreso cada 10 queries
            if (i + 1) % 10 == 0:
                logger.info(f"Completadas {i+1}/{total_queries} queries...")
        
        # Calcular métricas comparativas agregadas
        experiment_results['comparative_metrics'] = self._calculate_aggregate_metrics(
            experiment_results['bge_results'],
            experiment_results['tfidf_results']
        )
        
        experiment_results['end_time'] = datetime.now().isoformat()
        logger.info("=== EXPERIMENTO COMPLETADO ===")
        
        return experiment_results
    
    def _calculate_aggregate_metrics(self, bge_results: List[Dict], tfidf_results: List[Dict]) -> Dict[str, Any]:
        """
        Calcula métricas agregadas para comparación entre motores.
        
        Args:
            bge_results: Resultados de BGE-M3
            tfidf_results: Resultados de TF-IDF
            
        Returns:
            Métricas comparativas agregadas
        """
        def aggregate_metrics_for_engine(results, engine_name):
            metrics_list = []
            search_times = []
            
            for result in results:
                if 'error' not in result['metrics']:
                    metrics_list.append(result['metrics'])
                    search_times.append(result['metrics'].get('search_time', 0))
            
            if not metrics_list:
                return {}
            
            # Calcular promedios para cada métrica
            aggregated = {}
            for metric_name in metrics_list[0].keys():
                if metric_name not in ['engine', 'search_time']:
                    values = [m.get(metric_name, 0) for m in metrics_list if isinstance(m.get(metric_name), (int, float))]
                    if values:
                        aggregated[f'avg_{metric_name}'] = np.mean(values)
                        aggregated[f'std_{metric_name}'] = np.std(values)
            
            # Métricas de tiempo
            if search_times:
                aggregated['avg_search_time'] = np.mean(search_times)
                aggregated['median_search_time'] = np.median(search_times)
                aggregated['std_search_time'] = np.std(search_times)
            
            aggregated['total_queries'] = len(metrics_list)
            aggregated['successful_queries'] = len([m for m in metrics_list if m.get('has_results', 0) > 0])
            
            return aggregated
        
        # Calcular métricas para cada motor
        bge_aggregated = aggregate_metrics_for_engine(bge_results, 'bge_m3')
        tfidf_aggregated = aggregate_metrics_for_engine(tfidf_results, 'tfidf')
        
        # Calcular mejoras relativas
        improvements = {}
        for metric in ['avg_precision_at_1', 'avg_precision_at_5', 'avg_precision_at_10', 'avg_mrr', 'avg_average_precision']:
            if metric in bge_aggregated and metric in tfidf_aggregated and tfidf_aggregated[metric] > 0:
                improvement = ((bge_aggregated[metric] - tfidf_aggregated[metric]) / tfidf_aggregated[metric]) * 100
                improvements[f'{metric}_improvement_percent'] = improvement
        
        # Comparación de velocidad
        if 'avg_search_time' in bge_aggregated and 'avg_search_time' in tfidf_aggregated:
            speed_ratio = bge_aggregated['avg_search_time'] / tfidf_aggregated['avg_search_time']
            improvements['speed_ratio_bge_vs_tfidf'] = speed_ratio
        
        return {
            'bge_m3': bge_aggregated,
            'tfidf': tfidf_aggregated,
            'improvements': improvements
        }


class ResultsAnalyzer:
    """
    Analizador de resultados del experimento comparativo.
    
    Genera reportes detallados, visualizaciones y análisis estadísticos
    de los resultados del experimento BGE-M3 vs TF-IDF.
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_comprehensive_report(self, experiment_results: Dict[str, Any]) -> str:
        """
        Genera un reporte completo del experimento.
        
        Args:
            experiment_results: Resultados del experimento comparativo
            
        Returns:
            Path del archivo de reporte generado
        """
        self.logger.info("Generando reporte completo del experimento...")
        
        # Guardar resultados raw en JSON
        raw_results_path = self.output_dir / "raw_experiment_results.json"
        with open(raw_results_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Generar CSV detallado
        self._generate_detailed_csv(experiment_results)
        
        # Generar reporte en markdown
        report_path = self._generate_markdown_report(experiment_results)
        
        # Generar métricas resumidas
        self._generate_summary_metrics(experiment_results)
        
        self.logger.info(f"Reporte completo generado en {self.output_dir}")
        return str(report_path)
    
    def _generate_detailed_csv(self, experiment_results: Dict[str, Any]):
        """Genera CSV detallado con resultados por query."""
        csv_data = []
        
        bge_results = experiment_results['bge_results']
        tfidf_results = experiment_results['tfidf_results']
        
        for i in range(len(bge_results)):
            bge_result = bge_results[i]
            tfidf_result = tfidf_results[i]
            
            query = bge_result['query']
            
            row = {
                'query_id': query['id'],
                'query_text': query['text'],
                'query_category': query['category'],
                
                # Métricas BGE-M3
                'bge_precision_at_1': bge_result['metrics'].get('precision_at_1', 0),
                'bge_precision_at_5': bge_result['metrics'].get('precision_at_5', 0),
                'bge_precision_at_10': bge_result['metrics'].get('precision_at_10', 0),
                'bge_mrr': bge_result['metrics'].get('mrr', 0),
                'bge_average_precision': bge_result['metrics'].get('average_precision', 0),
                'bge_search_time': bge_result['metrics'].get('search_time', 0),
                'bge_num_results': bge_result['metrics'].get('num_results', 0),
                
                # Métricas TF-IDF
                'tfidf_precision_at_1': tfidf_result['metrics'].get('precision_at_1', 0),
                'tfidf_precision_at_5': tfidf_result['metrics'].get('precision_at_5', 0),
                'tfidf_precision_at_10': tfidf_result['metrics'].get('precision_at_10', 0),
                'tfidf_mrr': tfidf_result['metrics'].get('mrr', 0),
                'tfidf_average_precision': tfidf_result['metrics'].get('average_precision', 0),
                'tfidf_search_time': tfidf_result['metrics'].get('search_time', 0),
                'tfidf_num_results': tfidf_result['metrics'].get('num_results', 0),
            }
            
            csv_data.append(row)
        
        # Guardar CSV
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"CSV detallado guardado en {csv_path}")
    
    def _generate_markdown_report(self, experiment_results: Dict[str, Any]) -> Path:
        """Genera reporte en formato Markdown."""
        
        comparative_metrics = experiment_results['comparative_metrics']
        
        report_content = f"""# Reporte de Evaluación Comparativa: BGE-M3 vs TF-IDF

## Resumen Ejecutivo

**Fecha del Experimento:** {experiment_results['start_time']}
**Total de Queries:** {experiment_results['total_queries']}
**Resultados por Query:** {experiment_results['results_per_query']}

## Métricas Comparativas Globales

### BGE-M3 (Semántico)
"""
        
        bge_metrics = comparative_metrics.get('bge_m3', {})
        if bge_metrics:
            report_content += f"""
- **Precision@1:** {bge_metrics.get('avg_precision_at_1', 0):.3f} ± {bge_metrics.get('std_precision_at_1', 0):.3f}
- **Precision@5:** {bge_metrics.get('avg_precision_at_5', 0):.3f} ± {bge_metrics.get('std_precision_at_5', 0):.3f}
- **Precision@10:** {bge_metrics.get('avg_precision_at_10', 0):.3f} ± {bge_metrics.get('std_precision_at_10', 0):.3f}
- **MRR:** {bge_metrics.get('avg_mrr', 0):.3f} ± {bge_metrics.get('std_mrr', 0):.3f}
- **MAP:** {bge_metrics.get('avg_average_precision', 0):.3f} ± {bge_metrics.get('std_average_precision', 0):.3f}
- **Tiempo promedio de búsqueda:** {bge_metrics.get('avg_search_time', 0):.3f}s
- **Queries exitosas:** {bge_metrics.get('successful_queries', 0)}/{bge_metrics.get('total_queries', 0)}
"""
        
        tfidf_metrics = comparative_metrics.get('tfidf', {})
        if tfidf_metrics:
            report_content += f"""
### TF-IDF (Baseline)

- **Precision@1:** {tfidf_metrics.get('avg_precision_at_1', 0):.3f} ± {tfidf_metrics.get('std_precision_at_1', 0):.3f}
- **Precision@5:** {tfidf_metrics.get('avg_precision_at_5', 0):.3f} ± {tfidf_metrics.get('std_precision_at_5', 0):.3f}
- **Precision@10:** {tfidf_metrics.get('avg_precision_at_10', 0):.3f} ± {tfidf_metrics.get('std_precision_at_10', 0):.3f}
- **MRR:** {tfidf_metrics.get('avg_mrr', 0):.3f} ± {tfidf_metrics.get('std_mrr', 0):.3f}
- **MAP:** {tfidf_metrics.get('avg_average_precision', 0):.3f} ± {tfidf_metrics.get('std_average_precision', 0):.3f}
- **Tiempo promedio de búsqueda:** {tfidf_metrics.get('avg_search_time', 0):.3f}s
- **Queries exitosas:** {tfidf_metrics.get('successful_queries', 0)}/{tfidf_metrics.get('total_queries', 0)}
"""
        
        improvements = comparative_metrics.get('improvements', {})
        if improvements:
            report_content += f"""
### Mejoras de BGE-M3 vs TF-IDF

"""
            for metric, improvement in improvements.items():
                if 'improvement_percent' in metric:
                    metric_name = metric.replace('avg_', '').replace('_improvement_percent', '').upper()
                    report_content += f"- **{metric_name}:** {improvement:+.1f}%\n"
        
        report_content += f"""

## Análisis por Categorías

### Distribución de Queries
- **Legal General:** 25 queries
- **Ministerios/Departamentos:** 25 queries  
- **Consultas Temporales:** 25 queries
- **Técnicas Específicas:** 25 queries

## Conclusiones

1. **Efectividad:** BGE-M3 muestra {"mejores" if improvements.get('avg_precision_at_10_improvement_percent', 0) > 0 else "similares"} resultados que TF-IDF en precisión general.

2. **Velocidad:** BGE-M3 es {"más rápido" if improvements.get('speed_ratio_bge_vs_tfidf', 1) < 1 else "más lento"} que TF-IDF con un ratio de {improvements.get('speed_ratio_bge_vs_tfidf', 1):.2f}x.

3. **Consistencia:** La búsqueda semántica muestra {"mayor" if bge_metrics.get('std_precision_at_10', 1) < tfidf_metrics.get('std_precision_at_10', 1) else "similar"} consistencia en los resultados.

## Recomendaciones para TFM

1. **Incluir análisis estadístico** de significancia de las diferencias observadas
2. **Expandir dataset de evaluación** con anotaciones manuales de relevancia
3. **Analizar casos específicos** donde cada método funciona mejor
4. **Evaluar impacto de diferentes estrategias de chunking**

---

*Reporte generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Guardar reporte
        report_path = self.output_dir / "experiment_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Reporte markdown generado en {report_path}")
        return report_path
    
    def _generate_summary_metrics(self, experiment_results: Dict[str, Any]):
        """Genera archivo JSON con métricas resumidas."""
        summary = {
            'experiment_summary': {
                'total_queries': experiment_results['total_queries'],
                'execution_time': experiment_results['end_time'],
                'config': experiment_results['config']
            },
            'comparative_metrics': experiment_results['comparative_metrics']
        }
        
        summary_path = self.output_dir / "summary_metrics.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Métricas resumidas guardadas en {summary_path}")


def main():
    """Función principal para ejecutar el experimento comparativo."""
    
    # Configuración del experimento
    config = {
        'index_path': 'indices/boe_index.faiss',
        'metadata_path': 'indices/metadata.json',
        'model_name': 'pablosi/bge-m3-trained-2',
        'experiment_name': 'BGE-M3_vs_TFIDF_Comparative_Study'
    }
    
    # Verificar que existen los archivos necesarios
    if not Path(config['index_path']).exists():
        logger.error(f"No se encuentra el índice FAISS: {config['index_path']}")
        logger.error("Ejecuta primero: python scripts/build_vector_index.py")
        return
    
    if not Path(config['metadata_path']).exists():
        logger.error(f"No se encuentra el archivo de metadatos: {config['metadata_path']}")
        return
    
    try:
        # Inicializar experimento
        logger.info("Inicializando experimento comparativo...")
        experiment = ExperimentComparator(config)
        
        # Ejecutar experimento completo
        logger.info("Ejecutando experimento comparativo (esto puede tomar varios minutos)...")
        results = experiment.run_comparative_experiment(limit_results=20)
        
        # Generar reportes
        logger.info("Generando reportes de análisis...")
        analyzer = ResultsAnalyzer("evaluation_results")
        report_path = analyzer.generate_comprehensive_report(results)
        
        logger.info("=== EXPERIMENTO COMPLETADO EXITOSAMENTE ===")
        logger.info(f"Reportes generados en: evaluation_results/")
        logger.info(f"Reporte principal: {report_path}")
        
    except Exception as e:
        logger.error(f"Error en el experimento: {e}")
        raise


if __name__ == "__main__":
    main()
    
    def _estimate_difficulty(self, query: str, category: str) -> str:
        """
        Estima la dificultad esperada de una query.
        
        Args:
            query: Texto de la consulta
            category: Categoría de la consulta
            
        Returns:
            Nivel de dificultad estimado ('easy', 'medium', 'hard')
        """
        # Heurísticas básicas
        word_count = len(query.split())
        
        if category == "legal_general":
            return "easy" if word_count <= 3 else "medium"
        elif category == "ministerio":
            return "easy"  # Nombres de ministerios son específicos
        elif category == "temporal":
            return "hard"  # Búsquedas temporales son complejas
        elif category == "tematica":
            return "medium" if word_count <= 5 else "hard"
        
        return "medium"
    
    def run_comparison_experiment(self) -> Dict[str, Any]:
        """
        Ejecuta el experimento comparativo completo.
        
        Returns:
            Resultados detallados del experimento
        """
        logger.info("=== INICIANDO EXPERIMENTO COMPARATIVO BGE-M3 vs TF-IDF ===")
        
        results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(self.test_queries),
                'engines_compared': ['BGE-M3', 'TF-IDF'],
                'config': self.config
            },
            'detailed_results': [],
            'summary_metrics': {},
            'performance_metrics': {}
        }
        
        # Ejecutar búsquedas para cada query
        for i, query_info in enumerate(self.test_queries):
            logger.info(f"Procesando query {i+1}/{len(self.test_queries)}: {query_info['query'][:50]}...")
            
            # Realizar búsquedas con ambos motores
            bge_results, bge_time = self._search_with_timing(self.bge_engine, query_info['query'])
            tfidf_results, tfidf_time = self._search_with_timing(self.tfidf_engine, query_info['query'])
            
            # Calcular métricas para esta query
            query_metrics = self._calculate_query_metrics(
                query_info, bge_results, tfidf_results, bge_time, tfidf_time
            )
            
            results['detailed_results'].append(query_metrics)
        
        # Calcular métricas de resumen
        results['summary_metrics'] = self._calculate_summary_metrics(results['detailed_results'])
        results['performance_metrics'] = self._calculate_performance_metrics(results['detailed_results'])
        
        logger.info("=== EXPERIMENTO COMPARATIVO COMPLETADO ===")
        return results
    
    def _search_with_timing(self, engine, query: str, limit: int = 10) -> Tuple[List[Dict], float]:
        """
        Realiza búsqueda midiendo el tiempo de ejecución.
        
        Args:
            engine: Motor de búsqueda a usar
            query: Consulta a realizar
            limit: Número máximo de resultados
            
        Returns:
            Tupla de (resultados, tiempo_ejecución)
        """
        start_time = time.time()
        results = engine.search(query, limit=limit)
        execution_time = time.time() - start_time
        
        return results, execution_time
    
    def _calculate_query_metrics(self, query_info: Dict, bge_results: List, tfidf_results: List, 
                                bge_time: float, tfidf_time: float) -> Dict[str, Any]:
        """
        Calcula métricas para una query específica.
        
        Args:
            query_info: Información de la query
            bge_results: Resultados de BGE-M3
            tfidf_results: Resultados de TF-IDF
            bge_time: Tiempo de ejecución BGE-M3
            tfidf_time: Tiempo de ejecución TF-IDF
            
        Returns:
            Métricas calculadas para la query
        """
        metrics = {
            'query_info': query_info,
            'results_count': {
                'bge': len(bge_results),
                'tfidf': len(tfidf_results)
            },
            'execution_time': {
                'bge': bge_time,
                'tfidf': tfidf_time,
                'speedup_ratio': tfidf_time / bge_time if bge_time > 0 else float('inf')
            },
            'similarity_scores': {
                'bge_avg': np.mean([r.get('similarity_score', 0) for r in bge_results]) if bge_results else 0,
                'tfidf_avg': np.mean([r.get('similarity_score', 0) for r in tfidf_results]) if tfidf_results else 0,
                'bge_max': max([r.get('similarity_score', 0) for r in bge_results]) if bge_results else 0,
                'tfidf_max': max([r.get('similarity_score', 0) for r in tfidf_results]) if tfidf_results else 0
            }
        }
        
        # Calcular overlap de resultados (comparación por chunk_id si está disponible)
        bge_ids = set([r.get('chunk_id', str(i)) for i, r in enumerate(bge_results)])
        tfidf_ids = set([r.get('chunk_id', str(i)) for i, r in enumerate(tfidf_results)])
        
        intersection = len(bge_ids.intersection(tfidf_ids))
        union = len(bge_ids.union(tfidf_ids))
        
        metrics['result_overlap'] = {
            'intersection_count': intersection,
            'union_count': union,
            'jaccard_similarity': intersection / union if union > 0 else 0,
            'bge_unique': len(bge_ids - tfidf_ids),
            'tfidf_unique': len(tfidf_ids - bge_ids)
        }
        
        return metrics
    
    def _calculate_summary_metrics(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """
        Calcula métricas de resumen del experimento.
        
        Args:
            detailed_results: Lista de resultados detallados por query
            
        Returns:
            Métricas de resumen
        """
        # Extraer métricas de todas las queries
        bge_times = [r['execution_time']['bge'] for r in detailed_results]
        tfidf_times = [r['execution_time']['tfidf'] for r in detailed_results]
        
        bge_scores = [r['similarity_scores']['bge_avg'] for r in detailed_results]
        tfidf_scores = [r['similarity_scores']['tfidf_avg'] for r in detailed_results]
        
        jaccard_similarities = [r['result_overlap']['jaccard_similarity'] for r in detailed_results]
        
        summary = {
            'execution_time_stats': {
                'bge': {
                    'mean': np.mean(bge_times),
                    'std': np.std(bge_times),
                    'min': np.min(bge_times),
                    'max': np.max(bge_times),
                    'median': np.median(bge_times)
                },
                'tfidf': {
                    'mean': np.mean(tfidf_times),
                    'std': np.std(tfidf_times),
                    'min': np.min(tfidf_times),
                    'max': np.max(tfidf_times),
                    'median': np.median(tfidf_times)
                },
                'average_speedup_ratio': np.mean(tfidf_times) / np.mean(bge_times)
            },
            'similarity_score_stats': {
                'bge_mean': np.mean(bge_scores),
                'tfidf_mean': np.mean(tfidf_scores),
                'score_difference': np.mean(bge_scores) - np.mean(tfidf_scores)
            },
            'result_overlap_stats': {
                'average_jaccard': np.mean(jaccard_similarities),
                'jaccard_std': np.std(jaccard_similarities),
                'high_overlap_queries': sum(1 for j in jaccard_similarities if j > 0.5),
                'low_overlap_queries': sum(1 for j in jaccard_similarities if j < 0.2)
            }
        }
        
        return summary
    
    def _calculate_performance_metrics(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento por categoría.
        
        Args:
            detailed_results: Lista de resultados detallados por query
            
        Returns:
            Métricas de rendimiento por categoría
        """
        # Agrupar por categoría
        categories = {}
        for result in detailed_results:
            category = result['query_info']['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Calcular métricas por categoría
        performance_by_category = {}
        for category, results in categories.items():
            bge_times = [r['execution_time']['bge'] for r in results]
            tfidf_times = [r['execution_time']['tfidf'] for r in results]
            jaccard_sims = [r['result_overlap']['jaccard_similarity'] for r in results]
            
            performance_by_category[category] = {
                'query_count': len(results),
                'avg_bge_time': np.mean(bge_times),
                'avg_tfidf_time': np.mean(tfidf_times),
                'avg_jaccard_similarity': np.mean(jaccard_sims),
                'bge_faster_count': sum(1 for i in range(len(results)) if bge_times[i] < tfidf_times[i]),
                'tfidf_faster_count': sum(1 for i in range(len(results)) if tfidf_times[i] < bge_times[i])
            }
        
        return {
            'by_category': performance_by_category,
            'overall_winner': {
                'faster_engine': 'BGE-M3' if np.mean([r['execution_time']['bge'] for r in detailed_results]) < 
                                           np.mean([r['execution_time']['tfidf'] for r in detailed_results]) else 'TF-IDF',
                'higher_similarity': 'BGE-M3' if np.mean([r['similarity_scores']['bge_avg'] for r in detailed_results]) > 
                                               np.mean([r['similarity_scores']['tfidf_avg'] for r in detailed_results]) else 'TF-IDF'
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results"):
        """
        Guarda los resultados del experimento.
        
        Args:
            results: Resultados del experimento
            output_dir: Directorio de salida
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar resultados completos en JSON
        json_path = output_path / f"comparison_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Guardar resumen en CSV
        csv_data = []
        for result in results['detailed_results']:
            csv_data.append({
                'query_id': result['query_info']['id'],
                'query': result['query_info']['query'],
                'category': result['query_info']['category'],
                'bge_time': result['execution_time']['bge'],
                'tfidf_time': result['execution_time']['tfidf'],
                'bge_avg_score': result['similarity_scores']['bge_avg'],
                'tfidf_avg_score': result['similarity_scores']['tfidf_avg'],
                'jaccard_similarity': result['result_overlap']['jaccard_similarity']
            })
        
        csv_path = output_path / f"comparison_summary_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        logger.info(f"Resultados guardados en {output_path}")
        logger.info(f"  - Resultados completos: {json_path}")
        logger.info(f"  - Resumen CSV: {csv_path}")

def main():
    """Función principal del experimento."""
    
    # Configuración del experimento
    config = {
        'index_path': 'indices/boe_index.faiss',
        'metadata_path': 'indices/metadata.json',
        'model_name': 'pablosi/bge-m3-trained-2',
        'experiment_name': 'BGE-M3_vs_TFIDF_Comparison',
        'k_values': [1, 5, 10, 20]  # Para métricas Precision@K
    }
    
    try:
        # Verificar que existen los archivos necesarios
        if not Path(config['index_path']).exists():
            logger.error(f"No se encuentra el índice FAISS: {config['index_path']}")
            return
        
        if not Path(config['metadata_path']).exists():
            logger.error(f"No se encuentran los metadatos: {config['metadata_path']}")
            return
        
        # Ejecutar experimento
        comparator = ExperimentComparator(config)
        results = comparator.run_comparison_experiment()
        
        # Guardar resultados
        comparator.save_results(results)
        
        # Mostrar resumen final
        print("\n" + "="*60)
        print("RESUMEN DEL EXPERIMENTO COMPARATIVO")
        print("="*60)
        
        summary = results['summary_metrics']
        performance = results['performance_metrics']
        
        print(f"\n📊 MÉTRICAS DE RENDIMIENTO:")
        print(f"  • BGE-M3 tiempo promedio: {summary['execution_time_stats']['bge']['mean']:.3f}s")
        print(f"  • TF-IDF tiempo promedio: {summary['execution_time_stats']['tfidf']['mean']:.3f}s")
        print(f"  • Ratio de velocidad: {summary['execution_time_stats']['average_speedup_ratio']:.2f}x")
        
        print(f"\n🎯 MÉTRICAS DE SIMILITUD:")
        print(f"  • BGE-M3 similitud promedio: {summary['similarity_score_stats']['bge_mean']:.3f}")
        print(f"  • TF-IDF similitud promedio: {summary['similarity_score_stats']['tfidf_mean']:.3f}")
        print(f"  • Diferencia de scores: {summary['similarity_score_stats']['score_difference']:.3f}")
        
        print(f"\n🔄 SOLAPAMIENTO DE RESULTADOS:")
        print(f"  • Jaccard promedio: {summary['result_overlap_stats']['average_jaccard']:.3f}")
        print(f"  • Queries con alto solapamiento (>50%): {summary['result_overlap_stats']['high_overlap_queries']}")
        print(f"  • Queries con bajo solapamiento (<20%): {summary['result_overlap_stats']['low_overlap_queries']}")
        
        print(f"\n🏆 GANADOR GENERAL:")
        print(f"  • Motor más rápido: {performance['overall_winner']['faster_engine']}")
        print(f"  • Mayor similitud: {performance['overall_winner']['higher_similarity']}")
        
        print(f"\n📁 Resultados detallados guardados en: evaluation_results/")
        
    except Exception as e:
        logger.error(f"Error en el experimento: {e}")
        raise

if __name__ == "__main__":
    main()
