from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from .feature_engineering import GeoSpatialFeatures, LifestyleTransformer

def build_pipeline(config):
    geo_features = GeoSpatialFeatures(config['data_paths']['external_pollution'])
    lifestyle_transformer = LifestyleTransformer()
    
    preprocessor = ColumnTransformer([
        ('geo', geo_features, ['zip_code']),
        ('lifestyle', lifestyle_transformer, ['exercise_frequency', 'diet_quality'])
    ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
