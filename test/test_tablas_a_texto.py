import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from src.lib.utils_markdown import convertir_tablas_markdown_a_texto, tablas_a_texto, extraer_tablas_markdown

class TestTablasATexto(unittest.TestCase):
    
    def test_tabla_simple(self):
        """Test para tabla con encabezado simple"""
        texto_markdown = """
Aquí hay una tabla simple:

| Nombre | Edad | Ciudad |
|--------|------|--------|
| Juan   | 25   | Madrid |
| Ana    | 30   | Barcelona |

Fin del texto.
"""
        resultado = convertir_tablas_markdown_a_texto(texto_markdown)
        
        self.assertIn("Nombre : Juan", resultado)
        self.assertIn("Edad : 25", resultado)
        self.assertIn("Ciudad : Madrid", resultado)
        self.assertIn("Nombre : Ana", resultado)
        self.assertIn("Edad : 30", resultado)
        self.assertIn("Ciudad : Barcelona", resultado)
        
        # Verificar que no queden las tablas markdown originales
        self.assertNotIn("|", resultado)
    
    def test_tabla_multiindex(self):
        """Test para tabla con encabezado multiindex"""
        texto_markdown = """
Tabla con encabezados agrupados:

| Datos Personales |  | Trabajo |
|------------------|------------------|---------|
| Nombre           | Edad             | Empresa |
| Juan             | 25               | TechCorp |
| Ana              | 30               | DataInc |

Fin del texto.
"""
        resultado = convertir_tablas_markdown_a_texto(texto_markdown)
        
        self.assertIn("Datos Personales/Nombre : Juan", resultado)
        self.assertIn("Datos Personales/Edad : 25", resultado)
        self.assertIn("Trabajo/Empresa : TechCorp", resultado)
        self.assertIn("Datos Personales/Nombre : Ana", resultado)
        
        # Verificar que no queden las tablas markdown originales
        self.assertNotIn("|", resultado)
    
    def test_tabla_sin_encabezado(self):
        """Test para tabla sin encabezado válido"""
        texto_markdown = """
Tabla sin encabezados:

|   |   |   |
|---|---|---|
| Juan | 25 | Madrid |
| Ana  | 30 | Barcelona |

Fin del texto.
"""
        resultado = convertir_tablas_markdown_a_texto(texto_markdown)
        
        self.assertIn("Juan : 25 : Madrid", resultado)
        self.assertIn("Ana : 30 : Barcelona", resultado)
        
        # Verificar que no queden las tablas markdown originales
        self.assertNotIn("|", resultado)
    
    def test_multiples_tablas(self):
        """Test para múltiples tablas en el mismo texto"""
        texto_markdown = """
Primera tabla:

| Nombre | Edad |
|--------|------|
| Juan   | 25   |

Segunda tabla:

| Datos | Datos |
|-------|-------|
| Valor | Cantidad |
| A     | 10    |

Fin del texto.
"""
        resultado = convertir_tablas_markdown_a_texto(texto_markdown)
        
        self.assertIn("Nombre : Juan", resultado)
        self.assertIn("Edad : 25", resultado)
        self.assertIn("Datos/Valor : A", resultado)
        self.assertIn("Datos/Cantidad : 10", resultado)
        
        # Verificar que haya separación entre tablas
        partes = resultado.split("\n\n")
        self.assertGreater(len(partes), 1)
    
    def test_texto_sin_tablas(self):
        """Test para texto que no contiene tablas"""
        texto_simple = "Este es un texto simple sin tablas markdown."
        resultado = convertir_tablas_markdown_a_texto(texto_simple)
        
        self.assertEqual(texto_simple, resultado)
    
    def test_tablas_vacias(self):
        """Test para tablas vacías o con datos faltantes"""
        texto_markdown = """
Tabla con datos faltantes:

| Nombre | Edad | Ciudad |
|--------|------|--------|
| Juan   |      | Madrid |
|        | 30   |        |

Fin del texto.
"""
        resultado = convertir_tablas_markdown_a_texto(texto_markdown)
        
        # Solo deben aparecer los valores no vacíos
        self.assertIn("Nombre : Juan", resultado)
        self.assertIn("Ciudad : Madrid", resultado)
        self.assertIn("Edad : 30", resultado)
        
        # No deben aparecer valores vacíos
        self.assertNotIn(" : \n", resultado)
        self.assertNotIn(" :  ", resultado)

if __name__ == '__main__':
    unittest.main()
