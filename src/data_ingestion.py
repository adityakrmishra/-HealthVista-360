"""
Health Data Ingestion Pipeline
- Pulls data from APIs
- Handles incremental loading
- Manages data versioning
"""
import requests
import hashlib
from datetime import datetime
import os
from .utils import create_directory

class HealthDataIngestor:
    def __init__(self, config):
        self.config = config
        self.raw_data_path = config['data_paths']['raw']
        create_directory(self.raw_data_path)
        
    def _download_external_data(self, url, params):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise DataIngestionError(f"API request failed: {str(e)}")
    
    def _generate_data_hash(self, data):
        return hashlib.sha256(str(data).encode()).hexdigest()
    
    def ingest_patient_records(self):
        # Example EHR API integration
        api_data = self._download_external_data(
            self.config['ehr_api']['endpoint'],
            {'token': self.config['ehr_api']['token']}
        )
        
        filename = f"patient_records_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(self.raw_data_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(api_data, f)
        
        return self._generate_data_hash(api_data)

class DataIngestionError(Exception):
    pass

if __name__ == "__main__":
    from .utils import load_config
    config = load_config()
    ingestor = HealthDataIngestor(config)
    ingestor.ingest_patient_records()
