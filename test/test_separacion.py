import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.lib.utils_markdown import convertir_tablas_markdown_a_texto

# Test específico para ver las separaciones de filas
texto_tabla_simple = """
| Nombre | Edad | Ciudad |
|--------|------|--------|
| Juan   | 25   | Madrid |
| Ana    | 30   | Barcelona |
| Pedro  | 35   | Sevilla |
"""

print("=== TABLA SIMPLE CON 3 FILAS ===")
resultado = convertir_tablas_markdown_a_texto(texto_tabla_simple)
print(repr(resultado))  # usar repr para ver los caracteres de escape
print("\n--- FORMATO LEGIBLE ---")
print(resultado)
