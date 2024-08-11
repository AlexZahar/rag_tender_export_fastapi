from fastapi import FastAPI
from services.rag_service import RAG
from config.settings import load_config
from services.telemetry_service import setup_telemetry
from models.models import Query, Response, SourceNode

setup_telemetry()
config = load_config()

rag = RAG(config_file=config)
index = rag.milvus_index()

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Research RAG"}

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    # Format the query
    parsed_query = rag.parse_query(query, num_queries=1)[0]  # Assuming we want the first generated query

    # Generate the final query
    final_query = rag.generate_final_query(parsed_query)

    # Create the query engine
    query_engine = rag.query_engine(
        index,
        rerank=True,
        similarity_top_k=query.similarity_top_k,
        response_mode="tree_summarize",
        hydeTransform=True
    )

    # Execute the query
    response = query_engine.query(final_query)

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