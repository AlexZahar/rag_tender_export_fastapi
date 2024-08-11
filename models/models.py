from pydantic import BaseModel
from typing import List, Optional

class Query(BaseModel):
    query: str
    similarity_top_k: int
    rerank: bool
    hyde_transform: bool
    alpha: float
    response_mode: str
    use_parser: bool

class SourceNode(BaseModel):
    text: str
    score: float

class Response(BaseModel):
    search_result: str
    source_nodes: List[SourceNode]