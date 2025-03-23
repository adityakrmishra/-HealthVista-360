"""
Custom Exception Classes for HealthVista-360

Defines a hierarchy of exceptions for different error scenarios in:
- Data processing
- Model operations
- API interactions
- External service communication
"""

class HealthVistaError(Exception):
    """Base exception class for all project-specific errors"""
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        
    def __str__(self):
        return f"[{self.timestamp}] {self.__class__.__name__}: {super().__str__()}"

class DataError(HealthVistaError):
    """Base class for data-related errors"""
    pass

class DataValidationError(DataError):
    """Raised when data validation checks fail"""
    def __init__(self, message: str, invalid_records: list = None):
        super().__init__(message, {'invalid_records': invalid_records or []})

class DataIngestionError(DataError):
    """Raised during data collection/loading failures"""
    pass

class ModelError(HealthVistaError):
    """Base class for model-related errors"""
    pass

class ModelTrainingError(ModelError):
    """Raised when model training fails"""
    def __init__(self, message: str, training_metrics: dict = None):
        super().__init__(message, {'training_metrics': training_metrics or {}})

class ModelPredictionError(ModelError):
    """Raised when model inference fails"""
    def __init__(self, message: str, input_data: dict = None):
        super().__init__(message, {'input_data': input_data or {}})

class APIServiceError(HealthVistaError):
    """Base class for API-related errors"""
    pass

class AuthorizationError(APIServiceError):
    """Raised for authentication/authorization failures"""
    pass

class RateLimitExceededError(APIServiceError):
    """Raised when API rate limits are exceeded"""
    def __init__(self, message: str, reset_time: float = None):
        super().__init__(message, {'rate_limit_reset': reset_time})

class ExternalServiceError(HealthVistaError):
    """Base class for third-party service errors"""
    pass

class DatabaseConnectionError(ExternalServiceError):
    """Raised for database connectivity issues"""
    pass

class GeospatialServiceError(ExternalServiceError):
    """Raised for failures in geospatial data services"""
    pass
