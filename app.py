from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel, Field
import yaml
from rag import RAG
from typing import List
config_file = "config.yml"

with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)

class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)


class SourceNode(BaseModel):
    text: str
    score: float

class Response(BaseModel):
    search_result: str
    source_nodes: List[SourceNode]
rag = RAG(config_file=config)
index = rag.milvus_index()


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Research RAG"}

a = "Basierend auf den folgenden Eigenschaften:"
b = "geben Sie bitte die Knauf System ID an, die diesen Eigenschaften entspricht."

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    query_engine = index.as_query_engine(
        vector_store_query_mode="hybrid", 
        # similarity_top_k=query.similarity_top_k, #Weird behavior
        alpha=0.5,
        output=Response, 
        response_mode="tree_summarize", 
        verbose=True
    )
    response = query_engine.query(a + query.query + b)
    print("response", response)
    print("response.source_nodes", response.source_nodes)

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