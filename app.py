from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel, Field
import yaml
from rag import RAG
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from typing import List
config_file = "config.yml"
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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

### Local LLM
# llm = Ollama(model=config["llm_name"], url=config["llm_url"])
llm = OpenAI(model=config["llm_name"])

Settings.llm = OpenAI(model=config["llm_name"])
Settings.embed_model = HuggingFaceEmbedding(model_name=config["embedding_model"], trust_remote_code=True) 
# Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
# Settings.num_output = 512
Settings.context_window = 3900

rag = RAG(config_file=config, llm=llm)
index = rag.milvus_index()


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Research RAG"}

a = "Basierend auf den folgenden Eigenschaften:"
b = "geben Sie bitte die Knauf System ID an, die diesen Eigenschaften entspricht."

@app.post("/api/search", response_model=Response, status_code=200)
def search(query: Query):
    query_engine = index.as_query_engine(vector_store_query_mode="hybrid", output=Response, response_mode="tree_summarize", verbose=True)
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