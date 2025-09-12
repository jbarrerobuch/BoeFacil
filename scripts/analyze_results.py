#!/usr/bin/env python3
"""
Script de análisis profundo de los resultados del experimento comparativo BGE-M3 vs TF-IDF.

Genera análisis estadísticos detallados, visualizaciones y insights académicos
para el Trabajo de Fin de Máster.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para español
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-v0_8')

class ExperimentAnalyzer:
    """
    Analizador avanzado de resultados experimentales para TFM.
    """
    
    def __init__(self, data_path: str = "evaluation_results"):
        self.data_path = Path(data_path)
        self.df = None
        self.summary_stats = {}
        
    def load_data(self):
        """Carga los datos experimentales."""
        print("📊 Cargando datos experimentales...")
        
        # Cargar CSV detallado
        csv_path = self.data_path / "detailed_results.csv"
        if csv_path.exists():
            self.df = pd.read_csv(csv_path)
            print(f"✅ Datos cargados: {len(self.df)} queries analizadas")
        else:
            raise FileNotFoundError(f"No se encuentra {csv_path}")
        
        # Cargar métricas resumidas
        summary_path = self.data_path / "summary_metrics.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                self.summary_stats = json.load(f)
    
    def analyze_by_category(self):
        """Análisis detallado por categorías de queries."""
        print("\n🔍 ANÁLISIS POR CATEGORÍAS")
        print("=" * 50)
        
        categories = self.df['query_category'].unique()
        category_results = {}
        
        for category in categories:
            cat_data = self.df[self.df['query_category'] == category]
            
            bge_metrics = {
                'precision_at_1': cat_data['bge_precision_at_1'].mean(),
                'precision_at_5': cat_data['bge_precision_at_5'].mean(),
                'precision_at_10': cat_data['bge_precision_at_10'].mean(),
                'avg_search_time': cat_data['bge_search_time'].mean(),
                'count': len(cat_data)
            }
            
            tfidf_metrics = {
                'precision_at_1': cat_data['tfidf_precision_at_1'].mean(),
                'precision_at_5': cat_data['tfidf_precision_at_5'].mean(),
                'precision_at_10': cat_data['tfidf_precision_at_10'].mean(),
                'avg_search_time': cat_data['tfidf_search_time'].mean(),
                'count': len(cat_data)
            }
            
            category_results[category] = {
                'bge': bge_metrics,
                'tfidf': tfidf_metrics
            }
            
            print(f"\n📋 {category.upper().replace('_', ' ')} ({len(cat_data)} queries)")
            print(f"   BGE-M3    - P@1: {bge_metrics['precision_at_1']:.3f}, P@5: {bge_metrics['precision_at_5']:.3f}, Tiempo: {bge_metrics['avg_search_time']:.3f}s")
            print(f"   TF-IDF    - P@1: {tfidf_metrics['precision_at_1']:.3f}, P@5: {tfidf_metrics['precision_at_5']:.3f}, Tiempo: {tfidf_metrics['avg_search_time']:.3f}s")
            
            # Calcular mejoras
            p5_improvement = ((bge_metrics['precision_at_5'] - tfidf_metrics['precision_at_5']) / max(tfidf_metrics['precision_at_5'], 0.001)) * 100
            speed_improvement = ((tfidf_metrics['avg_search_time'] - bge_metrics['avg_search_time']) / tfidf_metrics['avg_search_time']) * 100
            
            print(f"   MEJORAS   - Precisión P@5: {p5_improvement:+.1f}%, Velocidad: {speed_improvement:+.1f}%")
        
        return category_results
    
    def statistical_analysis(self):
        """Análisis estadístico de significancia."""
        print("\n📈 ANÁLISIS ESTADÍSTICO")
        print("=" * 50)
        
        # Test de normalidad
        bge_p5 = self.df['bge_precision_at_5']
        tfidf_p5 = self.df['tfidf_precision_at_5']
        
        # Test de Wilcoxon (no paramétrico)
        try:
            statistic, p_value = stats.wilcoxon(bge_p5, tfidf_p5, alternative='two-sided')
            print(f"🧮 Test de Wilcoxon (Precision@5):")
            print(f"   Estadístico: {statistic:.3f}")
            print(f"   P-valor: {p_value:.6f}")
            print(f"   Significativo (α=0.05): {'SÍ' if p_value < 0.05 else 'NO'}")
        except Exception as e:
            print(f"   Error en test estadístico: {e}")
        
        # Estadísticas descriptivas
        print(f"\n📊 Estadísticas Descriptivas (Precision@5):")
        print(f"   BGE-M3: μ={bge_p5.mean():.3f}, σ={bge_p5.std():.3f}, mediana={bge_p5.median():.3f}")
        print(f"   TF-IDF: μ={tfidf_p5.mean():.3f}, σ={tfidf_p5.std():.3f}, mediana={tfidf_p5.median():.3f}")
        
        # Correlación entre métricas
        correlation = self.df['bge_precision_at_5'].corr(self.df['tfidf_precision_at_5'])
        print(f"   Correlación BGE-M3 vs TF-IDF: {correlation:.3f}")
    
    def identify_best_worst_cases(self):
        """Identifica los mejores y peores casos para cada método."""
        print("\n🎯 ANÁLISIS DE CASOS ESPECÍFICOS")
        print("=" * 50)
        
        # Calcular diferencia de rendimiento
        self.df['bge_advantage'] = self.df['bge_precision_at_5'] - self.df['tfidf_precision_at_5']
        
        # Mejores casos para BGE-M3
        best_bge = self.df.nlargest(5, 'bge_advantage')[['query_text', 'query_category', 'bge_precision_at_5', 'tfidf_precision_at_5', 'bge_advantage']]
        print("🚀 TOP 5 CASOS DONDE BGE-M3 SUPERA A TF-IDF:")
        for idx, row in best_bge.iterrows():
            print(f"   '{row['query_text'][:50]}...' ({row['query_category']})")
            print(f"      BGE: {row['bge_precision_at_5']:.3f} vs TF-IDF: {row['tfidf_precision_at_5']:.3f} (Δ={row['bge_advantage']:+.3f})")
        
        # Mejores casos para TF-IDF
        best_tfidf = self.df.nsmallest(5, 'bge_advantage')[['query_text', 'query_category', 'bge_precision_at_5', 'tfidf_precision_at_5', 'bge_advantage']]
        print("\n📚 TOP 5 CASOS DONDE TF-IDF SUPERA A BGE-M3:")
        for idx, row in best_tfidf.iterrows():
            print(f"   '{row['query_text'][:50]}...' ({row['query_category']})")
            print(f"      TF-IDF: {row['tfidf_precision_at_5']:.3f} vs BGE: {row['bge_precision_at_5']:.3f} (Δ={row['bge_advantage']:+.3f})")
    
    def analyze_speed_vs_accuracy(self):
        """Análisis del trade-off velocidad vs precisión."""
        print("\n⚡ ANÁLISIS VELOCIDAD vs PRECISIÓN")
        print("=" * 50)
        
        bge_speed = self.df['bge_search_time'].mean()
        tfidf_speed = self.df['tfidf_search_time'].mean()
        speed_ratio = tfidf_speed / bge_speed
        
        bge_precision = self.df['bge_precision_at_5'].mean()
        tfidf_precision = self.df['tfidf_precision_at_5'].mean()
        precision_ratio = tfidf_precision / bge_precision if bge_precision > 0 else float('inf')
        
        print(f"🏃 Velocidad promedio:")
        print(f"   BGE-M3: {bge_speed:.3f}s")
        print(f"   TF-IDF: {tfidf_speed:.3f}s")
        print(f"   Ratio: TF-IDF es {speed_ratio:.1f}x más lento")
        
        print(f"\n🎯 Precisión promedio (P@5):")
        print(f"   BGE-M3: {bge_precision:.3f}")
        print(f"   TF-IDF: {tfidf_precision:.3f}")
        print(f"   Ratio: TF-IDF es {precision_ratio:.1f}x más preciso")
        
        # Eficiencia = Precisión / Tiempo
        bge_efficiency = bge_precision / bge_speed
        tfidf_efficiency = tfidf_precision / tfidf_speed
        
        print(f"\n⚖️ Eficiencia (Precisión/Tiempo):")
        print(f"   BGE-M3: {bge_efficiency:.4f}")
        print(f"   TF-IDF: {tfidf_efficiency:.4f}")
        print(f"   Ventaja BGE-M3: {bge_efficiency/tfidf_efficiency:.1f}x más eficiente")
    
    def generate_visualizations(self):
        """Genera visualizaciones para el TFM."""
        print("\n📊 GENERANDO VISUALIZACIONES")
        print("=" * 50)
        
        # Configurar estilo
        plt.style.use('default')
        fig_path = self.data_path / "visualizations"
        fig_path.mkdir(exist_ok=True)
        
        # 1. Comparación por categorías
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision@5 por categoría
        cat_data = self.df.groupby('query_category')[['bge_precision_at_5', 'tfidf_precision_at_5']].mean()
        cat_data.plot(kind='bar', ax=axes[0,0], title='Precision@5 por Categoría')
        axes[0,0].set_ylabel('Precision@5')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(['BGE-M3', 'TF-IDF'])
        
        # Tiempo de búsqueda por categoría
        time_data = self.df.groupby('query_category')[['bge_search_time', 'tfidf_search_time']].mean()
        time_data.plot(kind='bar', ax=axes[0,1], title='Tiempo de Búsqueda por Categoría')
        axes[0,1].set_ylabel('Tiempo (segundos)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(['BGE-M3', 'TF-IDF'])
        
        # Distribución de Precision@5
        axes[1,0].hist(self.df['bge_precision_at_5'], alpha=0.7, bins=20, label='BGE-M3', color='blue')
        axes[1,0].hist(self.df['tfidf_precision_at_5'], alpha=0.7, bins=20, label='TF-IDF', color='orange')
        axes[1,0].set_title('Distribución de Precision@5')
        axes[1,0].set_xlabel('Precision@5')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].legend()
        
        # Scatter plot: Precisión vs Tiempo
        axes[1,1].scatter(self.df['bge_search_time'], self.df['bge_precision_at_5'], 
                         alpha=0.6, label='BGE-M3', color='blue')
        axes[1,1].scatter(self.df['tfidf_search_time'], self.df['tfidf_precision_at_5'], 
                         alpha=0.6, label='TF-IDF', color='orange')
        axes[1,1].set_title('Trade-off Velocidad vs Precisión')
        axes[1,1].set_xlabel('Tiempo de Búsqueda (s)')
        axes[1,1].set_ylabel('Precision@5')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(fig_path / "comparison_analysis.png", dpi=300, bbox_inches='tight')
        print(f"✅ Gráficos guardados en {fig_path / 'comparison_analysis.png'}")
        
        # 2. Gráfico de barras comparativo resumido
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = ['Precision@1', 'Precision@5', 'Precision@10']
        bge_values = [self.df['bge_precision_at_1'].mean(), 
                      self.df['bge_precision_at_5'].mean(),
                      self.df['bge_precision_at_10'].mean()]
        tfidf_values = [self.df['tfidf_precision_at_1'].mean(),
                        self.df['tfidf_precision_at_5'].mean(), 
                        self.df['tfidf_precision_at_10'].mean()]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, bge_values, width, label='BGE-M3', color='skyblue')
        ax.bar(x + width/2, tfidf_values, width, label='TF-IDF', color='lightcoral')
        
        ax.set_xlabel('Métricas')
        ax.set_ylabel('Valores')
        ax.set_title('Comparación de Métricas de Precisión')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Añadir valores en las barras
        for i, (bge_val, tfidf_val) in enumerate(zip(bge_values, tfidf_values)):
            ax.text(i - width/2, bge_val + 0.01, f'{bge_val:.3f}', ha='center', fontsize=9)
            ax.text(i + width/2, tfidf_val + 0.01, f'{tfidf_val:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(fig_path / "precision_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico de precisión guardado en {fig_path / 'precision_comparison.png'}")
        
        plt.close('all')
    
    def generate_academic_insights(self):
        """Genera insights académicos para el TFM."""
        print("\n🎓 INSIGHTS ACADÉMICOS PARA TFM")
        print("=" * 50)
        
        insights = []
        
        # Insight 1: Dominancia de TF-IDF
        tfidf_wins = (self.df['tfidf_precision_at_5'] > self.df['bge_precision_at_5']).sum()
        total_queries = len(self.df)
        win_rate = (tfidf_wins / total_queries) * 100
        
        insights.append(f"📈 TF-IDF superó a BGE-M3 en {tfidf_wins}/{total_queries} queries ({win_rate:.1f}%)")
        
        # Insight 2: Ventaja en velocidad
        speed_advantage = self.df['tfidf_search_time'].mean() / self.df['bge_search_time'].mean()
        insights.append(f"⚡ BGE-M3 es {speed_advantage:.1f}x más rápido que TF-IDF")
        
        # Insight 3: Categoría con mejor rendimiento BGE-M3
        cat_performance = self.df.groupby('query_category')['bge_advantage'].mean()
        best_cat = cat_performance.idxmax()
        best_advantage = cat_performance.max()
        insights.append(f"🎯 BGE-M3 funciona mejor en categoria '{best_cat}' (ventaja: {best_advantage:+.3f})")
        
        # Insight 4: Variabilidad
        bge_std = self.df['bge_precision_at_5'].std()
        tfidf_std = self.df['tfidf_precision_at_5'].std()
        consistency = "BGE-M3" if bge_std < tfidf_std else "TF-IDF"
        insights.append(f"📊 {consistency} muestra mayor consistencia (menor desviación estándar)")
        
        # Insight 5: Queries sin resultados
        bge_no_results = (self.df['bge_precision_at_5'] == 0).sum()
        tfidf_no_results = (self.df['tfidf_precision_at_5'] == 0).sum()
        insights.append(f"🔍 Queries sin resultados relevantes: BGE-M3={bge_no_results}, TF-IDF={tfidf_no_results}")
        
        for insight in insights:
            print(f"   {insight}")
        
        return insights
    
    def generate_tfm_recommendations(self):
        """Genera recomendaciones específicas para el TFM."""
        print("\n💡 RECOMENDACIONES PARA TFM")
        print("=" * 50)
        
        recommendations = [
            "🔬 METODOLÓGICAS:",
            "   • Implementar evaluación con anotadores humanos (ground truth real)",
            "   • Probar diferentes estrategias de chunking (semántico vs fijo)",
            "   • Evaluar modelos multilinguë (español + catalán + euskera + gallego)",
            "   • Comparar con otros embeddings (SentenceBERT, RoBERTa, etc.)",
            "",
            "⚙️ TÉCNICAS:",
            "   • Desarrollar sistema híbrido que combine TF-IDF + BGE-M3",
            "   • Implementar re-ranking con múltiples señales",
            "   • Optimizar chunking para preservar contexto legal",
            "   • Ajustar fine-tuning del modelo BGE-M3 en dominio legal",
            "",
            "📊 EXPERIMENTALES:",
            "   • Ampliar dataset a otros años (2020-2024)",
            "   • Incluir otras fuentes legales (DOGC, BOJA, etc.)",
            "   • Evaluar robustez con queries adversariales",
            "   • Medir impacto de usuario real en aplicación web",
            "",
            "🎯 APLICACIÓN:",
            "   • Sistema adaptativo: TF-IDF para queries exactas, BGE-M3 para conceptuales",
            "   • Interfaz con sugerencias de reformulación de queries",
            "   • Cache inteligente aprovechando velocidad BGE-M3",
            "   • API híbrida con selección automática de método"
        ]
        
        for rec in recommendations:
            print(rec)

def main():
    """Función principal del análisis."""
    print("🔬 ANÁLISIS PROFUNDO DE RESULTADOS EXPERIMENTALES")
    print("=" * 60)
    print("📚 Para Trabajo de Fin de Máster en Data Science & NLP")
    print("=" * 60)
    
    try:
        # Inicializar analizador
        analyzer = ExperimentAnalyzer()
        analyzer.load_data()
        
        # Ejecutar análisis completo
        analyzer.analyze_by_category()
        analyzer.statistical_analysis()
        analyzer.identify_best_worst_cases()
        analyzer.analyze_speed_vs_accuracy()
        analyzer.generate_visualizations()
        analyzer.generate_academic_insights()
        analyzer.generate_tfm_recommendations()
        
        print(f"\n✅ ANÁLISIS COMPLETADO")
        print(f"📁 Resultados disponibles en: evaluation_results/")
        print(f"📊 Visualizaciones en: evaluation_results/visualizations/")
        
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
