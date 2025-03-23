import pytest
import pandas as pd
from src.data_processing import load_data, handle_missing

def test_data_loading():
    config = {'data_paths': {
        'raw_medical': 'test_data/medical.csv',
        'raw_lifestyle': 'test_data/lifestyle.json',
        'external_pollution': 'test_data/pollution.geojson'
    }}
    medical, lifestyle, pollution = load_data(config)
    assert isinstance(medical, pd.DataFrame)
    assert isinstance(lifestyle, pd.DataFrame)
    assert not pollution.empty

def test_missing_handling():
    test_df = pd.DataFrame({'a': [1, None, 3], 'b': ['x', None, 'z']})
    cleaned = handle_missing(test_df)
    assert not cleaned.isnull().any().any()
