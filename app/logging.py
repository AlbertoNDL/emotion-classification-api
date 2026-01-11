import logging
import sys


def setup_logging() -> None:
    """
    Configure the logging system for the application.
    Sets up basic logging with INFO level and a standardized format that includes
    timestamp, log level, logger name, and message. Logs are directed to stdout.
    Also suppresses verbose logging from the uvicorn.access logger by setting its
    level to WARNING to reduce noise in the logs.
    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
