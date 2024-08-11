from fastapi import FastAPI
from services.rag_service import RAG
from config.settings import load_config
from services.telemetry_service import setup_telemetry
from models.models import Query, Response, SourceNode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from services.query_parser_service import generate_queries


setup_telemetry()
config=load_config()

rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)
rag = RAG(config_file=config)
index = rag.milvus_index()
hyde = HyDEQueryTransform(include_original=True)

app = FastAPI()
embed_model = HuggingFaceEmbedding(model_name=config["embedding_model"], trust_remote_code=True) 

@app.get("/")
def root():
    return {"message": "Research RAG"}

a = "Basierend auf den folgenden Eigenschaften:"
b = "geben Sie bitte die Knauf System ID an, die diesen Eigenschaften entspricht."

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    queries = generate_queries(query.query, llm)
    query_engine = index.as_query_engine(
        vector_store_query_mode="hybrid", 
        similarity_top_k=query.similarity_top_k,
        alpha=0.5,
        output=Response, 
        node_postprocessors=[rerank],
        response_mode="tree_summarize", 
        verbose=True,
        embedding=embed_model
    )
    query_engine = TransformQueryEngine(query_engine, query_transform=hyde)
    response = query_engine.query(a + query.query + b)

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