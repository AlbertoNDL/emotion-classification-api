import logging
from fastapi import Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from app.config import settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API Key for authentication (example: super-secret-key)",
)

def get_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Validate and retrieve the API key from the request header.
    This function checks if the provided API key matches the configured API key
    in the application settings. If the key is valid, it returns the API key.
    If the key is invalid or missing, it raises an HTTP 401 Unauthorized exception.
    Args:
        api_key (str): The API key extracted from the request header via dependency injection.
                       Defaults to the value from the api_key_header dependency.
    Returns:
        str: The valid API key.
    Raises:
        HTTPException: If the provided API key does not match the configured API key.
                       Status code: 401 (Unauthorized)
                       Detail: "Invalid or missing API key"
    """

    # log api_key for debugging purposes
    logger.debug(f"API Key received: {api_key}")
    
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return api_key