import logging
import os
from fastapi import FastAPI
from rag_tender_export_fastapi.services.rag_service import RAG
from rag_tender_export_fastapi.config.settings import load_config
from rag_tender_export_fastapi.services.telemetry_service import setup_telemetry
from rag_tender_export_fastapi.models.models import Query, Response, SourceNode
from llama_index.core import QueryBundle
from telemetry import setup_telemetry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = load_config()

if config.get("enable_tracing", False):
    telemetry_setup_success = setup_telemetry()
    if not telemetry_setup_success:
        logger.warning("Telemetry setup failed. Continuing without telemetry.")

rag = RAG(config_file=config)
index = rag.milvus_index()

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Research RAG"}

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    # Use the parser if enabled
    if query.use_parser:
        parsed_query = rag.parse_query(query.query, num_queries=1)[0]
        final_query = rag.generate_final_query(parsed_query)
    else:
        final_query = query.query

    # Create the query engine
    query_engine = rag.query_engine(
        index,
        rerank=query.rerank,
        alpha=query.alpha,
        similarity_top_k=query.similarity_top_k,
        response_mode=query.response_mode,
        hyde_transform=query.hyde_transform
    )

    # Execute the query
    query_bundle = QueryBundle(query_str=final_query)
    # cleaned_query = clean_text(query_bundle)
    # print("query CLEAN TEXT:", cleaned_query)
    response = query_engine.query(query_bundle)

    # Create a list of SourceNode objects
    source_nodes = [
        SourceNode(text=node.node.text, score=node.score)
        for node in response.source_nodes
    ]

    response_object = Response(
        search_result=str(response).strip(),
        source_nodes=source_nodes
    )
    print("response_object", response_object)
    return response_object
