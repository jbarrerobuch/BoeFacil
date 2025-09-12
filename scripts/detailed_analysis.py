import pandas as pd

# Cargar datos
df = pd.read_csv('evaluation_results/detailed_results.csv')

# Calcular ventaja de TF-IDF
df['tfidf_advantage'] = df['tfidf_precision_at_5'] - df['bge_precision_at_5']

# Top casos donde TF-IDF supera a BGE-M3
top_tfidf = df.nlargest(10, 'tfidf_advantage')[['query_text', 'query_category', 'bge_precision_at_5', 'tfidf_precision_at_5', 'tfidf_advantage']]

print('=== TOP 10 CASOS DONDE TF-IDF SUPERA SIGNIFICATIVAMENTE A BGE-M3 ===\n')
for idx, row in top_tfidf.iterrows():
    print(f"Query: '{row['query_text']}'")
    print(f"Categoria: {row['query_category']}")
    print(f"BGE-M3 P@5: {row['bge_precision_at_5']:.3f}")
    print(f"TF-IDF P@5: {row['tfidf_precision_at_5']:.3f}")
    print(f"Ventaja TF-IDF: +{row['tfidf_advantage']:.3f}")
    print("---\n")

# Análisis por categoría
print('\n=== ANÁLISIS DETALLADO POR CATEGORÍA ===\n')
for category in df['query_category'].unique():
    cat_data = df[df['query_category'] == category]
    bge_wins = (cat_data['bge_precision_at_5'] > cat_data['tfidf_precision_at_5']).sum()
    tfidf_wins = (cat_data['tfidf_precision_at_5'] > cat_data['bge_precision_at_5']).sum()
    ties = (cat_data['tfidf_precision_at_5'] == cat_data['bge_precision_at_5']).sum()
    
    print(f"{category.upper().replace('_', ' ')}:")
    print(f"  BGE-M3 gana: {bge_wins}/{len(cat_data)} queries ({bge_wins/len(cat_data)*100:.1f}%)")
    print(f"  TF-IDF gana: {tfidf_wins}/{len(cat_data)} queries ({tfidf_wins/len(cat_data)*100:.1f}%)")
    print(f"  Empates: {ties}/{len(cat_data)} queries ({ties/len(cat_data)*100:.1f}%)")
    print(f"  Promedio BGE-M3 P@5: {cat_data['bge_precision_at_5'].mean():.3f}")
    print(f"  Promedio TF-IDF P@5: {cat_data['tfidf_precision_at_5'].mean():.3f}")
    print()
