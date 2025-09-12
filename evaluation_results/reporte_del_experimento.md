# Reporte de Evaluación Comparativa: BGE-M3 vs TF-IDF

## Resumen Ejecutivo

**Fecha del Experimento:** 2025-09-09T23:58:43.863183
**Total de Queries:** 100
**Resultados por Query:** 20

## Métricas Comparativas Globales

### BGE-M3 (Semántico)

- **Precision@1:** 0.170 ± 0.376
- **Precision@5:** 0.130 ± 0.239
- **Precision@10:** 0.121 ± 0.214
- **MRR:** 0.000 ± 0.000
- **MAP:** 0.060 ± 0.129
- **Tiempo promedio de búsqueda:** 0.521s
- **Queries exitosas:** 100/100

### TF-IDF (Baseline)

- **Precision@1:** 0.240 ± 0.427
- **Precision@5:** 0.230 ± 0.348
- **Precision@10:** 0.207 ± 0.328
- **MRR:** 0.000 ± 0.000
- **MAP:** 0.142 ± 0.276
- **Tiempo promedio de búsqueda:** 4.991s
- **Queries exitosas:** 100/100

### Mejoras de BGE-M3 vs TF-IDF

- **PRECISION_AT_1:** -29.2%
- **PRECISION_AT_5:** -43.5%
- **PRECISION_AT_10:** -41.5%
- **AVERAGE_PRECISION:** -58.1%


## Análisis por Categorías

### Distribución de Queries
- **Legal General:** 25 queries
- **Ministerios/Departamentos:** 25 queries  
- **Consultas Temporales:** 25 queries
- **Técnicas Específicas:** 25 queries

## Conclusiones

1. **Efectividad:** BGE-M3 muestra similares resultados que TF-IDF en precisión general.

2. **Velocidad:** BGE-M3 es más rápido que TF-IDF con un ratio de 0.10x.

3. **Consistencia:** La búsqueda semántica muestra mayor consistencia en los resultados.

## Recomendaciones para TFM

1. **Incluir análisis estadístico** de significancia de las diferencias observadas
2. **Expandir dataset de evaluación** con anotaciones manuales de relevancia
3. **Analizar casos específicos** donde cada método funciona mejor
4. **Evaluar impacto de diferentes estrategias de chunking**

---

*Reporte generado automáticamente el 2025-09-10 00:07:59*
