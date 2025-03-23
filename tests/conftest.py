import pytest
import pandas as pd

@pytest.fixture
def sample_medical_data():
    return pd.DataFrame({
        'patient_id': ['P001', 'P002'],
        'age': [45, 32],
        'bmi': [28.1, 24.5],
        'disease_risk': [1, 0]
    })

@pytest.fixture
def sample_config():
    return {
        'data_paths': {
            'raw_medical': 'data/raw/sample_medical.csv',
            'external_pollution': 'data/external/test_pollution.geojson'
        },
        'model_params': {'test_size': 0.2}
    }
