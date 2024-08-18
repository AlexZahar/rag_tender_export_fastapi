import os
import logging
from llama_index.core import set_global_handler

logger = logging.getLogger(__name__)

def setup_telemetry():
    PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
    if not PHOENIX_API_KEY:
        logger.error("PHOENIX_API_KEY is not set. Please set this environment variable.")
        return False

    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
    set_global_handler("arize_phoenix", endpoint="https://llamatrace.com/v1/traces")
    logger.info("Telemetry setup completed.")
    return True