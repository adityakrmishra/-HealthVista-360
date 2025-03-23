from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from .data_processing import load_config

def train_model():
    config = load_config()
    data = pd.read_csv("data/processed/clean_dataset.csv")
    
    X = data.drop('disease_risk', axis=1)
    y = data['disease_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model_params']['test_size'],
        random_state=config['model_params']['random_state'])
    
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config['model_params']['logistic_regression']['C'],
            max_iter=config['model_params']['logistic_regression']['max_iter']
        )
    )
    
    model.fit(X_train, y_train)
    print(f"Test accuracy: {model.score(X_test, y_test):.2f}")
    
    save_model(model, "models/baseline_model.pkl")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

if __name__ == "__main__":
    train_model()
