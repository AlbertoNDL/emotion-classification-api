from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings and configuration.
    This class manages all configuration parameters for the ML training pipeline,
    including API settings and MLflow tracking configuration.
    Attributes:
        API_PREFIX (str): The API version prefix for all endpoints. Defaults to "/v1".
        API_KEY_NAME (str): The HTTP header name for API authentication. Defaults to "X-API-Key".
        API_KEY (str): The secret API key for authentication. Defaults to "super-secret-key".
        MLFLOW_TRACKING_URI (str): The URI for MLflow experiment tracking. 
            Defaults to "file:./mlruns" for local file storage.
    Configuration:
        Environment variables can be loaded from a .env file to override default values.
    """

    API_PREFIX: str = "/v1"
    API_KEY_NAME: str = "X-API-Key"
    API_KEY: str = "super-secret-key"

    ONNX_MODEL_PATH: str = "models/onnx/model.onnx"
    HF_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"

    class Config:
        env_file = ".env"


settings = Settings()
