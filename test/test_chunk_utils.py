import unittest
import pandas as pd
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lib import chunk_utils

class TestChunkUtils(unittest.TestCase):
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear directorio temporal para tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Datos de prueba
        self.texto_corto = "Este es un texto corto que no necesita división."
        self.texto_largo = """Este es el primer párrafo. Contiene varias oraciones para probar la funcionalidad.
        
Este es el segundo párrafo. También tiene múltiples oraciones para aumentar el contenido.

Este es el tercer párrafo. Añade más contenido para superar el límite de tokens y forzar la división del texto en chunks más pequeños.

Este es el cuarto párrafo. Continúa expandiendo el texto para asegurar que se generen múltiples chunks durante las pruebas de la funcionalidad."""
        
        # DataFrame de prueba
        self.df_test = pd.DataFrame({
            'markdown': [
                self.texto_corto,
                self.texto_largo,
                "Otro texto de prueba con contenido moderado.",
                None,  # Valor nulo para probar manejo de errores
                123    # Valor no string para probar validación
            ],
            'contenido': [
                "Contenido alternativo corto",
                "Contenido alternativo largo que también puede ser dividido en chunks si es necesario para las pruebas.",
                "Más contenido de prueba",
                "Contenido válido",
                "Último contenido"
            ]
        })
    
    def tearDown(self):
        """Limpieza después de cada test"""
        # Eliminar directorio temporal
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_contar_tokens_texto_valido(self):
        """Test función contar_tokens con texto válido"""
        # Texto simple
        resultado = chunk_utils.contar_tokens("Hola mundo")
        self.assertEqual(resultado, 2)
        
        # Texto con puntuación
        resultado = chunk_utils.contar_tokens("Hola, mundo!")
        self.assertEqual(resultado, 4)  # ['Hola', ',', 'mundo', '!']
        
        # Texto vacío
        resultado = chunk_utils.contar_tokens("")
        self.assertEqual(resultado, 0)
        
        # Texto con espacios extra
        resultado = chunk_utils.contar_tokens("  Hola   mundo  ")
        self.assertEqual(resultado, 2)
    
    def test_contar_tokens_casos_especiales(self):
        """Test función contar_tokens con casos especiales"""
        # None
        resultado = chunk_utils.contar_tokens(None)
        self.assertEqual(resultado, 0)
        
        # No string
        resultado = chunk_utils.contar_tokens(123)
        self.assertEqual(resultado, 0)
        
        # String vacío
        resultado = chunk_utils.contar_tokens("")
        self.assertEqual(resultado, 0)
    
    def test_chunking_texto_corto(self):
        """Test función chunking con texto que no necesita división"""
        resultado = chunk_utils.chunking(self.texto_corto, max_tokens=100)
        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0], self.texto_corto.strip())
    
    def test_chunking_texto_largo(self):
        """Test función chunking con texto que necesita división"""
        resultado = chunk_utils.chunking(self.texto_largo, max_tokens=50)
        self.assertGreater(len(resultado), 1)
        
        # Verificar que cada chunk no excede el límite (aproximadamente)
        for chunk in resultado:
            tokens = chunk_utils.contar_tokens(chunk)
            self.assertLessEqual(tokens, 60)  # Margen de tolerancia
    
    def test_chunking_casos_especiales(self):
        """Test función chunking con casos especiales"""
        # Texto None
        resultado = chunk_utils.chunking(None)
        self.assertEqual(resultado, [])
        
        # Texto vacío
        resultado = chunk_utils.chunking("")
        self.assertEqual(resultado, [])
        
        # No string
        resultado = chunk_utils.chunking(123)
        self.assertEqual(resultado, [])
    
    def test_chunking_markdown_df_basico(self):
        """Test función chunking_markdown_df básica sin guardar archivos"""
        resultado = chunk_utils.chunking_markdown_df(self.df_test, max_tokens=50)
        
        # Verificar que se generaron chunks
        self.assertGreater(len(resultado), 0)
        
        # Verificar estructura de chunks
        for chunk in resultado:
            self.assertIn('texto', chunk)
            self.assertIn('item_id', chunk)
            self.assertIn('chunk_numero', chunk)
            self.assertIn('total_chunks_fila', chunk)
            self.assertIn('tokens_aproximados', chunk)
            
            # Verificar que chunk_numero empieza en 0
            self.assertGreaterEqual(chunk['chunk_numero'], 0)
    
    def test_chunking_markdown_df_con_guardado(self):
        """Test función chunking_markdown_df guardando archivos"""
        texto_id = "test_documento"
        
        resultado = chunk_utils.chunking_markdown_df(
            self.df_test, 
            max_tokens=30,
            texto_id=texto_id,
            store_path=self.temp_dir
        )
        
        # Verificar que se generaron chunks
        self.assertGreater(len(resultado), 0)
        
        # Verificar que se guardaron archivos
        archivos_guardados = [chunk for chunk in resultado if 'archivo_guardado' in chunk]
        self.assertGreater(len(archivos_guardados), 0)
        
        # Verificar que los archivos existen
        for chunk in archivos_guardados:
            self.assertTrue(os.path.exists(chunk['archivo_guardado']))
            
            # Verificar formato del nombre
            nombre_archivo = chunk['nombre_archivo']
            self.assertTrue(nombre_archivo.startswith(f"{texto_id}_"))
            self.assertTrue(nombre_archivo.endswith(".md"))
            
            # Verificar contenido del archivo
            with open(chunk['archivo_guardado'], 'r', encoding='utf-8') as f:
                contenido = f.read()
                self.assertEqual(contenido, chunk['texto'])
    
    def test_chunking_markdown_df_columna_inexistente(self):
        """Test con columna que no existe"""
        resultado = chunk_utils.chunking_markdown_df(
            self.df_test, 
            columna_markdown='columna_inexistente'
        )
        self.assertEqual(resultado, [])
    
    def test_chunking_markdown_df_no_dataframe(self):
        """Test con parámetro que no es DataFrame"""
        resultado = chunk_utils.chunking_markdown_df("no es dataframe")
        self.assertEqual(resultado, [])
    
    def test_chunking_markdown_df_max_tokens_invalido(self):
        """Test con max_tokens inválido"""
        resultado = chunk_utils.chunking_markdown_df(
            self.df_test, 
            max_tokens=0
        )
        self.assertEqual(resultado, [])
    
    def test_chunking_markdown_df_store_path_sin_texto_id(self):
        """Test con store_path pero sin texto_id"""
        resultado = chunk_utils.chunking_markdown_df(
            self.df_test,
            store_path=self.temp_dir
        )
        self.assertEqual(resultado, [])
    
    def test_chunking_markdown_df_columna_alternativa(self):
        """Test usando columna alternativa"""
        resultado = chunk_utils.chunking_markdown_df(
            self.df_test,
            columna_markdown='contenido',
            max_tokens=20
        )
        
        self.assertGreater(len(resultado), 0)
        
        # Verificar que se procesaron las filas válidas
        item_ids_procesados = set(chunk['item_id'] for chunk in resultado)
        self.assertIn('fila_0', item_ids_procesados)  # Primera fila válida
        self.assertIn('fila_1', item_ids_procesados)  # Segunda fila válida
    
    def test_numeracion_chunks_desde_cero(self):
        """Test que verifica que los chunks empiecen en 0"""
        resultado = chunk_utils.chunking_markdown_df(
            pd.DataFrame({'markdown': [self.texto_largo]}),
            max_tokens=30
        )
        
        # Debe haber múltiples chunks
        self.assertGreater(len(resultado), 1)
        
        # El primer chunk debe ser 0
        primer_chunk = min(resultado, key=lambda x: x['chunk_numero'])
        self.assertEqual(primer_chunk['chunk_numero'], 0)
        
        # Verificar secuencia consecutiva
        numeros_chunks = sorted([chunk['chunk_numero'] for chunk in resultado if chunk['item_id'] == 'fila_0'])
        self.assertEqual(numeros_chunks, list(range(len(numeros_chunks))))
    
    def test_manejo_valores_nulos_y_no_string(self):
        """Test manejo de valores None y no-string en DataFrame"""
        df_problematico = pd.DataFrame({
            'markdown': [
                "Texto válido",
                None,
                123,
                "",
                "Otro texto válido"
            ]
        })
        
        resultado = chunk_utils.chunking_markdown_df(df_problematico, max_tokens=50)
        
        # Solo deben procesarse las filas válidas (0 y 4)
        item_ids_procesados = set(chunk['item_id'] for chunk in resultado)
        self.assertEqual(item_ids_procesados, {'fila_0', 'fila_4'})
    
    def test_directorio_creacion_automatica(self):
        """Test que el directorio se crea automáticamente"""
        directorio_nuevo = os.path.join(self.temp_dir, 'subdir', 'chunks')
        
        resultado = chunk_utils.chunking_markdown_df(
            pd.DataFrame({'markdown': [self.texto_corto]}),
            texto_id="test",
            store_path=directorio_nuevo
        )
        
        # Verificar que se creó el directorio
        self.assertTrue(os.path.exists(directorio_nuevo))
        
        # Verificar que se guardó el archivo
        self.assertGreater(len(resultado), 0)
        self.assertIn('archivo_guardado', resultado[0])

if __name__ == '__main__':
    unittest.main()
