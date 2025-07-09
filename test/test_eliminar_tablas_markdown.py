import sys
import unittest
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lib import utils_markdown

class TestEliminarTablasMarkdown(unittest.TestCase):
    def test_eliminar_tabla_simple(self):
        texto = """
Texto antes de la tabla.
| Col1 | Col2 |
|------|------|
|  1   |  2   |
|  3   |  4   |
Texto después de la tabla.
"""
        esperado = """
Texto antes de la tabla.
Texto después de la tabla.
"""
        resultado = utils_markdown.eliminar_tablas_markdown(texto)
        self.assertEqual(resultado.strip(), esperado.strip())

    def test_eliminar_varias_tablas(self):
        texto = """
Intro
| A | B |
|---|---|
| 1 | 2 |
Medio
| X | Y |
|---|---|
| 3 | 4 |
Final
"""
        esperado = """
Intro
Medio
Final
"""
        resultado = utils_markdown.eliminar_tablas_markdown(texto)
        self.assertEqual(resultado.strip(), esperado.strip())

    def test_sin_tablas(self):
        texto = "Solo texto plano sin tablas."
        resultado = utils_markdown.eliminar_tablas_markdown(texto)
        self.assertEqual(resultado, texto)

if __name__ == "__main__":
    unittest.main()
