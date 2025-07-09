import unittest
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lib import utils

class TestDataframesAJson(unittest.TestCase):
    def test_lista_dataframes_simple(self):
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'x': [5], 'y': [6]})
        resultado = utils.dataframes_a_json([df1, df2])
        esperado = json.dumps([
            [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}],
            [{'x': 5, 'y': 6}]
        ], ensure_ascii=False, indent=4)
        self.assertEqual(json.loads(resultado), json.loads(esperado))

    def test_dataframe_multiindex(self):
        arrays = [
            ['A', 'A', 'B', 'B'],
            ['uno', 'dos', 'uno', 'dos']
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['primero', 'segundo'])
        df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=index)
        resultado = utils.dataframes_a_json([df])
        # El resultado debe ser una lista de listas de diccionarios, con claves como tuplas stringificadas
        esperado = [
            [
                {"('A', 'uno')": 1, "('A', 'dos')": 2, "('B', 'uno')": 3, "('B', 'dos')": 4},
                {"('A', 'uno')": 5, "('A', 'dos')": 6, "('B', 'uno')": 7, "('B', 'dos')": 8}
            ]
        ]
        self.assertEqual(json.loads(resultado), esperado)

if __name__ == "__main__":
    unittest.main()
