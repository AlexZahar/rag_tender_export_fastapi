from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel, Field
import yaml
from rag import RAG
from llama_index.llms.openai import OpenAI

config_file = "config.yml"

with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)


class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)


class Response(BaseModel):
    search_result: str 
    source: str



# llm = Ollama(model=config["llm_name"], url=config["llm_url"])
llm = OpenAI(model=config["llm_name"], url=config["llm_url"])
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
    print("response.source_nodes[0]", response.source_nodes[0])
    response_object = Response(
        search_result=str(response).strip(), source=response.source_nodes[0].text
    )
    print("response_object", response_object)
    # response_object = Response(
    #     search_result=str(response).strip(), source=[response.metadata[k]["name"] for k in response.metadata.keys()][0]
    # )
    return response_object