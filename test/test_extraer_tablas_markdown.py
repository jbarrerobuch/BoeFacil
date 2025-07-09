import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lib import utils_markdown

class TestExtraerTablasMarkdown(unittest.TestCase):
    def test_tabla_simple(self):
        markdown = """
                | Nombre | Edad |
                |--------|------|
                | Ana    | 23   |
                | Luis   | 31   |
                """
        tablas = utils_markdown.extraer_tablas_markdown(markdown)
        self.assertEqual(len(tablas), 1)
        df = tablas[0]
        # El DataFrame puede tener columnas vacías por el separador, así que quitamos las vacías
        df = df.dropna(axis=1, how='all')
        self.assertIn('Nombre', df.columns)
        self.assertIn('Edad', df.columns)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.iloc[0]['Nombre'], 'Ana')
        self.assertEqual(df.iloc[1]['Edad'], 31)

    def test_sin_tabla(self):
        markdown = "Esto es un texto sin tablas."
        tablas = utils_markdown.extraer_tablas_markdown(markdown)
        self.assertEqual(tablas, [])

    def test_varias_tablas(self):
        markdown = """
                | A | B |
                |---|---|
                | 1 | 2 |

                Texto entre tablas

                | X | Y |
                |---|---|
                | 9 | 8 |
                """
        tablas = utils_markdown.extraer_tablas_markdown(markdown)
        self.assertGreaterEqual(len(tablas), 2)
    
    def test_tabla_grande(self):
        markdown = """
| Puesto adjudicado | | | | | Puesto de procedencia | | | Datos personales adjudicatario/a | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N.º  Orden | Puesto | Localidad | Nivel | C. E. | Ministerio, Centro Directivo, Provincia | Nivel | C. E. | Apellidos y nombre | NRP | Grupo | Cuerpo o Escala | Situación |
|  | **Secretaría de Estado de Transportes, Movilidad y Agenda Urbana** |  |  |  |  |  |  |  |  |  |  |  |
|  | *Secretaría General de Agenda Urbana y Vivienda* |  |  |  |  |  |  |  |  |  |  |  |
|  | Unidad Temporal para Ejecución del Plan de Recuperación, Transformación y Resiliencia |  |  |  |  |  |  |  |  |  |  |  |
| 1 | Coordinador/Coordinadora (5576267). | Madrid. | 29 | 22.243,06 | Ministerio de Transportes, Movilidad y Agenda Urbana. Secretaría General de Agenda Urbana y Vivienda Madrid. (Adscripción Provisional). | 29 | 22.243,06 | Lorite Becerra, Rita María. | \*\*\*6617\*02 A5900 | A1 | E. Técnicos Facultativos Superior OO.AA. M. Ambiente. | Activo. |
|  | *Subsecretaría de Transportes, Movilidad y* *Agenda Urbana* |  |  |  |  |  |  |  |  |  |  |  |
|  | Dirección General de Organización e Inspección  Unidad de Apoyo |  |  |  |  |  |  |  |  |  |  |  |
| 2 | Coordinador/Coordinador (5639713). | Madrid. | 29 | 22.243,06 | Ministerio de Trabajo y Economía Social. Secretaría General Técnica. Madrid. | 28 | 15.486,66 | Ramos Arteaga, Ángel. | \*\*\*0030\*13 A0304 | A1 | C. Facultativo de Archiveros Bibliotecarios y Arqueólogos. | Activo. |
"""
        tablas = utils_markdown.extraer_tablas_markdown(markdown)
        self.assertGreaterEqual(len(tablas), 1)
        df = tablas[0]
        self.assertEqual(df.shape, (7, 13))
    
    def test_sin_tablas(self):
        markdown = """Texto sin tablas ni formato especial."""
        tablas = utils_markdown.extraer_tablas_markdown(markdown)
        self.assertEqual(tablas, [])

if __name__ == "__main__":
    unittest.main()