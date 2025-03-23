from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class GeoSpatialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, pollution_data_path):
        self.pollution = gpd.read_file(pollution_data_path)
    
    def transform(self, X, y=None):
        X_geo = X.merge(self.pollution, on='zip_code')
        X_geo['pollution_exposure'] = np.log1p(X_geo['pm2_5'])
        return X_geo.drop(columns='geometry')

class LifestyleTransformer(TransformerMixin):
    def fit(self, X, y=None):
        self.activity_levels = X['exercise_frequency'].value_counts(normalize=True)
        return self
    
    def transform(self, X):
        X['activity_score'] = X['exercise_frequency'].map(self.activity_levels)
        return X
