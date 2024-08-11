import logging

logger = logging.getLogger(__name__)

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from rag_tender_export_fastapi.config.settings import load_config

config=load_config()

def setup_telemetry():
    try:
        endpoint = config["telemetry_url"]
        tracer_provider = trace_sdk.TracerProvider()
        span_processor = SimpleSpanProcessor(OTLPSpanExporter(endpoint))
        tracer_provider.add_span_processor(span_processor)
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Telemetry setup completed successfully.")
    except Exception as e:
        logger.error(f"Failed to set up telemetry: {e}")
