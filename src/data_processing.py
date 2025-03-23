import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import yaml
import os

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def load_data(config):
    medical = pd.read_csv(config['data_paths']['raw_medical'])
    lifestyle = pd.read_json(config['data_paths']['raw_lifestyle'])
    pollution = gpd.read_file(config['data_paths']['external_pollution'])
    return medical, lifestyle, pollution

def handle_missing(data, num_strategy='median', cat_strategy='mode'):
    numeric_cols = data.select_dtypes(include='number').columns
    cat_cols = data.select_dtypes(include='object').columns
    
    num_imputer = SimpleImputer(strategy=num_strategy)
    cat_imputer = SimpleImputer(strategy=cat_strategy)
    
    data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
    return data

def encode_categorical(data):
    encoder = OneHotEncoder(handle_unknown='ignore')
    cat_cols = data.select_dtypes(include='object').columns
    encoded = encoder.fit_transform(data[cat_cols])
    return pd.concat([data.drop(cat_cols, axis=1), 
                    pd.DataFrame(encoded.toarray())], axis=1)

def process_data():
    config = load_config()
    medical, lifestyle, pollution = load_data(config)
    
    # Merge datasets
    merged = pd.merge(medical, lifestyle, on='patient_id')
    merged = pd.merge(merged, pollution, on='zip_code')
    
    # Handle missing values
    cleaned = handle_missing(merged, 
                           config['preprocessing']['numeric_impute'],
                           config['preprocessing']['categorical_impute'])
    
    # Encode categorical variables
    final_data = encode_categorical(cleaned)
    
    # Save processed data
    final_data.to_csv("data/processed/clean_dataset.csv", index=False)

if __name__ == "__main__":
    process_data()
