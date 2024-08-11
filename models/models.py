from typing import List, Optional
from pydantic import BaseModel, Field

class Query(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=10)

class SourceNode(BaseModel):
    text: str
    score: float

class Response(BaseModel):
    search_result: str
    source_nodes: List[SourceNode]
