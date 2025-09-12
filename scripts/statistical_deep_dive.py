#!/usr/bin/env python3
"""
Análisis estadístico profundo de los resultados BGE-M3 vs TF-IDF
para explicación académica detallada.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def explain_wilcoxon_test():
    """Explica paso a paso el test de Wilcoxon aplicado."""
    print("🧮 EXPLICACIÓN DETALLADA DEL TEST DE WILCOXON")
    print("=" * 60)
    
    # Cargar datos
    df = pd.read_csv('evaluation_results/detailed_results.csv')
    
    bge_p5 = df['bge_precision_at_5'].values
    tfidf_p5 = df['tfidf_precision_at_5'].values
    
    print(f"📊 Datos cargados: {len(bge_p5)} pares de observaciones")
    print(f"   BGE-M3 P@5: μ={bge_p5.mean():.3f}, σ={bge_p5.std():.3f}")
    print(f"   TF-IDF P@5: μ={tfidf_p5.mean():.3f}, σ={tfidf_p5.std():.3f}")
    
    # Calcular diferencias
    differences = bge_p5 - tfidf_p5
    print(f"\n🔍 Análisis de diferencias (BGE-M3 - TF-IDF):")
    print(f"   Media de diferencias: {differences.mean():.3f}")
    print(f"   Mediana de diferencias: {np.median(differences):.3f}")
    print(f"   Diferencias positivas (BGE > TF-IDF): {(differences > 0).sum()}")
    print(f"   Diferencias negativas (TF-IDF > BGE): {(differences < 0).sum()}")
    print(f"   Empates: {(differences == 0).sum()}")
    
    # Test de normalidad
    shapiro_stat, shapiro_p = stats.shapiro(differences[differences != 0])
    print(f"\n📈 Test de normalidad de diferencias (Shapiro-Wilk):")
    print(f"   Estadístico: {shapiro_stat:.4f}")
    print(f"   P-valor: {shapiro_p:.6f}")
    print(f"   ¿Distribución normal?: {'NO' if shapiro_p < 0.05 else 'SÍ'}")
    print(f"   → {'Justifica uso de Wilcoxon (no paramétrico)' if shapiro_p < 0.05 else 'Podría usarse t-test paramétrico'}")
    
    # Test de Wilcoxon paso a paso
    print(f"\n🎯 TEST DE WILCOXON PASO A PASO:")
    
    # Filtrar empates (el test los ignora)
    non_zero_diff = differences[differences != 0]
    print(f"   1. Eliminamos {(differences == 0).sum()} empates")
    print(f"   2. Quedan {len(non_zero_diff)} diferencias no nulas")
    
    # Calcular rangos de valores absolutos
    abs_diff = np.abs(non_zero_diff)
    ranks = stats.rankdata(abs_diff)
    print(f"   3. Calculamos rangos de |diferencias|")
    
    # Separar rangos positivos y negativos
    positive_ranks = ranks[non_zero_diff > 0]
    negative_ranks = ranks[non_zero_diff < 0]
    
    w_plus = positive_ranks.sum()
    w_minus = negative_ranks.sum()
    
    print(f"   4. Suma de rangos positivos (W+): {w_plus}")
    print(f"   5. Suma de rangos negativos (W-): {w_minus}")
    print(f"   6. Estadístico W = min(W+, W-): {min(w_plus, w_minus)}")
    
    # Ejecutar test oficial
    statistic, p_value = stats.wilcoxon(bge_p5, tfidf_p5, alternative='two-sided')
    print(f"   7. P-valor (dos colas): {p_value:.6f}")
    
    # Interpretación
    alpha = 0.05
    print(f"\n💡 INTERPRETACIÓN ACADÉMICA:")
    print(f"   H0: No hay diferencia entre BGE-M3 y TF-IDF")
    print(f"   H1: Sí hay diferencia entre BGE-M3 y TF-IDF")
    print(f"   Nivel de significancia α = {alpha}")
    print(f"   P-valor = {p_value:.6f}")
    
    if p_value < alpha:
        print(f"   ✅ p < α → RECHAZAMOS H0")
        print(f"   🎯 Conclusión: Diferencia ESTADÍSTICAMENTE SIGNIFICATIVA")
        print(f"   📊 Confianza: {(1-p_value)*100:.2f}%")
    else:
        print(f"   ❌ p ≥ α → NO RECHAZAMOS H0")
        print(f"   🤷 Conclusión: No hay evidencia suficiente de diferencia")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'differences': differences,
        'w_plus': w_plus,
        'w_minus': w_minus
    }

def analyze_correlation():
    """Analiza la correlación entre métodos."""
    print("\n\n🔗 ANÁLISIS DE CORRELACIÓN DETALLADO")
    print("=" * 60)
    
    df = pd.read_csv('evaluation_results/detailed_results.csv')
    
    bge_p5 = df['bge_precision_at_5']
    tfidf_p5 = df['tfidf_precision_at_5']
    
    # Correlaciones
    pearson_r, pearson_p = stats.pearsonr(bge_p5, tfidf_p5)
    spearman_r, spearman_p = stats.spearmanr(bge_p5, tfidf_p5)
    
    print(f"📊 CORRELACIÓN PEARSON (lineal):")
    print(f"   Coeficiente r: {pearson_r:.3f}")
    print(f"   P-valor: {pearson_p:.6f}")
    print(f"   Interpretación: {interpret_correlation(pearson_r)}")
    
    print(f"\n📊 CORRELACIÓN SPEARMAN (monotónica):")
    print(f"   Coeficiente ρ: {spearman_r:.3f}")  
    print(f"   P-valor: {spearman_p:.6f}")
    print(f"   Interpretación: {interpret_correlation(spearman_r)}")
    
    # Análisis de concordancia
    both_zero = ((bge_p5 == 0) & (tfidf_p5 == 0)).sum()
    both_nonzero = ((bge_p5 > 0) & (tfidf_p5 > 0)).sum()
    bge_only = ((bge_p5 > 0) & (tfidf_p5 == 0)).sum()
    tfidf_only = ((bge_p5 == 0) & (tfidf_p5 > 0)).sum()
    
    print(f"\n🎯 ANÁLISIS DE CONCORDANCIA:")
    print(f"   Ambos fallan (0,0): {both_zero} queries ({both_zero/len(df)*100:.1f}%)")
    print(f"   Ambos funcionan (+,+): {both_nonzero} queries ({both_nonzero/len(df)*100:.1f}%)")
    print(f"   Solo BGE-M3 funciona (+,0): {bge_only} queries ({bge_only/len(df)*100:.1f}%)")
    print(f"   Solo TF-IDF funciona (0,+): {tfidf_only} queries ({tfidf_only/len(df)*100:.1f}%)")
    
    # Coeficiente de concordancia de Kendall
    tau, tau_p = stats.kendalltau(bge_p5, tfidf_p5)
    print(f"\n📊 CONCORDANCIA DE KENDALL:")
    print(f"   Tau (τ): {tau:.3f}")
    print(f"   P-valor: {tau_p:.6f}")
    print(f"   Interpretación: {interpret_concordance(tau)}")

def interpret_correlation(r):
    """Interpreta el coeficiente de correlación."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "Muy débil o inexistente"
    elif abs_r < 0.3:
        return "Débil"
    elif abs_r < 0.5:
        return "Moderada" 
    elif abs_r < 0.7:
        return "Fuerte"
    else:
        return "Muy fuerte"

def interpret_concordance(tau):
    """Interpreta el coeficiente de concordancia de Kendall."""
    abs_tau = abs(tau)
    if abs_tau < 0.1:
        return "Concordancia muy baja"
    elif abs_tau < 0.3:
        return "Concordancia baja"
    elif abs_tau < 0.5:
        return "Concordancia moderada"
    elif abs_tau < 0.7:
        return "Concordancia alta"
    else:
        return "Concordancia muy alta"

def analyze_consistency():
    """Analiza la consistencia de cada método."""
    print("\n\n📊 ANÁLISIS DE CONSISTENCIA Y VARIABILIDAD")
    print("=" * 60)
    
    df = pd.read_csv('evaluation_results/detailed_results.csv')
    
    bge_p5 = df['bge_precision_at_5']
    tfidf_p5 = df['tfidf_precision_at_5']
    
    # Estadísticas descriptivas
    print(f"📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   BGE-M3:")
    print(f"      Media: {bge_p5.mean():.3f}")
    print(f"      Mediana: {bge_p5.median():.3f}")
    print(f"      Desv. Estándar: {bge_p5.std():.3f}")
    print(f"      Coef. Variación: {bge_p5.std()/bge_p5.mean():.3f}")
    print(f"      IQR: {bge_p5.quantile(0.75) - bge_p5.quantile(0.25):.3f}")
    
    print(f"\n   TF-IDF:")
    print(f"      Media: {tfidf_p5.mean():.3f}")
    print(f"      Mediana: {tfidf_p5.median():.3f}")
    print(f"      Desv. Estándar: {tfidf_p5.std():.3f}")
    print(f"      Coef. Variación: {tfidf_p5.std()/tfidf_p5.mean():.3f}")
    print(f"      IQR: {tfidf_p5.quantile(0.75) - tfidf_p5.quantile(0.25):.3f}")
    
    # Test de varianzas iguales
    levene_stat, levene_p = stats.levene(bge_p5, tfidf_p5)
    print(f"\n🔬 TEST DE IGUALDAD DE VARIANZAS (Levene):")
    print(f"   Estadístico: {levene_stat:.4f}")
    print(f"   P-valor: {levene_p:.6f}")
    print(f"   ¿Varianzas iguales?: {'NO' if levene_p < 0.05 else 'SÍ'}")
    
    # Interpretación de consistencia
    print(f"\n💡 INTERPRETACIÓN DE CONSISTENCIA:")
    if bge_p5.std() < tfidf_p5.std():
        print(f"   ✅ BGE-M3 es MÁS CONSISTENTE (menor variabilidad)")
        print(f"   📊 Ventaja: {((tfidf_p5.std() - bge_p5.std())/tfidf_p5.std())*100:.1f}% menos variable")
        print(f"   🎯 Para aplicaciones de producción: BGE-M3 más predecible")
    else:
        print(f"   ✅ TF-IDF es MÁS CONSISTENTE (menor variabilidad)")
        print(f"   📊 Ventaja: {((bge_p5.std() - tfidf_p5.std())/bge_p5.std())*100:.1f}% menos variable")
        print(f"   🎯 Para aplicaciones de producción: TF-IDF más predecible")

def create_statistical_visualizations():
    """Crea visualizaciones para explicar los hallazgos estadísticos."""
    print("\n\n📊 GENERANDO VISUALIZACIONES ESTADÍSTICAS")
    print("=" * 60)
    
    df = pd.read_csv('evaluation_results/detailed_results.csv')
    
    bge_p5 = df['bge_precision_at_5']
    tfidf_p5 = df['tfidf_precision_at_5']
    differences = bge_p5 - tfidf_p5
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribución de diferencias
    axes[0,0].hist(differences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(0, color='red', linestyle='--', label='No diferencia')
    axes[0,0].axvline(differences.mean(), color='orange', linestyle='-', label=f'Media={differences.mean():.3f}')
    axes[0,0].set_title('Distribución de Diferencias (BGE-M3 - TF-IDF)')
    axes[0,0].set_xlabel('Diferencia en Precision@5')
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].legend()
    
    # 2. Scatter plot con línea de correlación
    axes[0,1].scatter(tfidf_p5, bge_p5, alpha=0.6, color='green')
    
    # Línea de regresión
    z = np.polyfit(tfidf_p5, bge_p5, 1)
    p = np.poly1d(z)
    axes[0,1].plot(tfidf_p5.sort_values(), p(tfidf_p5.sort_values()), "r--", alpha=0.8)
    
    # Línea de igualdad
    max_val = max(tfidf_p5.max(), bge_p5.max())
    axes[0,1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Igualdad perfecta')
    
    axes[0,1].set_title(f'Correlación: r={stats.pearsonr(tfidf_p5, bge_p5)[0]:.3f}')
    axes[0,1].set_xlabel('TF-IDF Precision@5')
    axes[0,1].set_ylabel('BGE-M3 Precision@5')
    axes[0,1].legend()
    
    # 3. Box plots comparativos
    data_for_box = [bge_p5, tfidf_p5]
    axes[1,0].boxplot(data_for_box, labels=['BGE-M3', 'TF-IDF'])
    axes[1,0].set_title('Distribución de Precision@5 por Método')
    axes[1,0].set_ylabel('Precision@5')
    
    # 4. Q-Q plot para normalidad
    from scipy.stats import probplot
    probplot(differences[differences != 0], dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot: Diferencias vs Normal')
    
    plt.tight_layout()
    
    # Guardar
    viz_path = Path('evaluation_results/visualizations')
    viz_path.mkdir(exist_ok=True)
    plt.savefig(viz_path / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Visualizaciones guardadas en {viz_path / 'statistical_analysis.png'}")
    
    plt.close()

def main():
    """Función principal del análisis estadístico."""
    print("🔬 ANÁLISIS ESTADÍSTICO PROFUNDO PARA TFM")
    print("=" * 70)
    print("📚 Explicación detallada de hallazgos estadísticos")
    print("=" * 70)
    
    try:
        # Ejecutar análisis
        wilcoxon_results = explain_wilcoxon_test()
        analyze_correlation()
        analyze_consistency()
        create_statistical_visualizations()
        
        print(f"\n✅ ANÁLISIS ESTADÍSTICO COMPLETADO")
        print(f"📊 Visualizaciones en: evaluation_results/visualizations/")
        print(f"\n🎓 PARA TU TFM:")
        print(f"   • Usa la explicación de Wilcoxon en la sección de Metodología")
        print(f"   • Incluye interpretación de correlación en Resultados")
        print(f"   • Menciona consistencia en Discusión")
        print(f"   • Las visualizaciones apoyan tu argumentación estadística")
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
