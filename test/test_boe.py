import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.lib import boe

class TestBoeExtraer(unittest.TestCase):
    @patch('src.boe.requests.get')
    @patch('src.boe.flatten_boe')
    @patch('src.boe.utils.guardar_en_json')
    def test_extraer_local(self, mock_guardar_en_json, mock_flatten_boe, mock_requests_get):
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'mock': 'data'}
        mock_requests_get.return_value = mock_response
        # Mock flatten_boe
        mock_flatten_boe.return_value = [{'a': 1, 'b': 2}]
        # Call function
        df = boe.extraer(fecha='20240627')
        # Assertions
        mock_requests_get.assert_called_once()
        mock_guardar_en_json.assert_called_once()
        mock_flatten_boe.assert_called_once_with({'mock': 'data'})
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('a', df.columns)
        self.assertIn('b', df.columns)

    @patch('src.boe.requests.get')
    @patch('src.boe.flatten_boe')
    @patch('src.boe.utils.guardar_en_s3')
    def test_extraer_s3(self, mock_guardar_en_s3, mock_flatten_boe, mock_requests_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'mock': 'data'}
        mock_requests_get.return_value = mock_response
        mock_flatten_boe.return_value = [{'a': 1, 'b': 2}]
        df = boe.extraer(fecha='20240627', s3_bucket='my-bucket')
        mock_guardar_en_s3.assert_called()
        mock_flatten_boe.assert_called_once_with({'mock': 'data'})
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
