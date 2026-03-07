from pydantic_settings import BaseSettings
from pathlib import Path


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

    MODEL_DIR: str = "models/emotions-v1.1"
    ONNX_MODEL_RELATIVE_PATH: str = "onnx/model.onnx"
    ONNX_MODEL_PATH: str | None = None
    HF_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    MODEL_PROVIDER: str = "cpu"

    @property
    def resolved_onnx_model_path(self) -> str:
        if self.ONNX_MODEL_PATH:
            return self.ONNX_MODEL_PATH
        return str(Path(self.MODEL_DIR) / self.ONNX_MODEL_RELATIVE_PATH)

    class Config:
        env_file = ".env"


settings = Settings()
