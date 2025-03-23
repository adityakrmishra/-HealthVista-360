import pytest
from src.models import train_model
from sklearn.exceptions import NotFittedError
import joblib

def test_model_training(tmp_path):
    model_path = tmp_path / "test_model.pkl"
    test_config = {
        'model_params': {'test_size': 0.3, 'random_state': 42},
        'data_paths': {'raw_medical': 'data/raw/sample_medical.csv'}
    }
    
    # Mock training process
    model = train_model(test_config)
    joblib.dump(model, model_path)
    
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, 'predict'), "Model should have predict method"

def test_model_predictions():
    # Test prediction shape
    model = joblib.load("models/baseline_model.pkl")
    test_data = pd.DataFrame([[45, 28.1, 140, 110]], 
                           columns=['age', 'bmi', 'blood_pressure', 'glucose'])
    preds = model.predict(test_data)
    assert preds.shape == (1,), "Predictions should match input shape"
