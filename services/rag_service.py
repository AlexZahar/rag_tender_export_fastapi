from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)

from services.query_parser_service import generate_queries
class RAG:
    def __init__(self, config_file):
        self.config = config_file
        Settings.llm = OpenAI(model=self.config["llm_name"])
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"], trust_remote_code=True) 
        Settings.node_parser = SentenceSplitter(chunk_size=912, chunk_overlap=40)
        Settings.num_output = 512
        Settings.context_window = 3900

    def milvus_index(self):
        milvus_vector_store = MilvusVectorStore(
            uri=self.config["milvus"]["uri"],
            dim=self.config["milvus"]["dim"],
            overwrite=False,
            enable_sparse=self.config["milvus"]["enable_sparse"],
            hybrid_ranker=self.config["milvus"]["hybrid_ranker"],
            hybrid_ranker_params=self.config["milvus"]["hybrid_ranker_params"],
            verbose=self.config["milvus"]["verbose"]
        )
  

        index = VectorStoreIndex.from_vector_store(
            vector_store=milvus_vector_store
        )
        return index
    
    def query_engine(self, index, rerank=True, alpha=0.5, similarity_top_k=5, response_mode="tree_summarize", hydeTransform=False):
        node_postprocessors = []
        hyde = HyDEQueryTransform(include_original=True)

        if rerank:
            rerank_processor = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
            node_postprocessors.append(rerank_processor)
        
        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid", 
            similarity_top_k=similarity_top_k,
            alpha=alpha,
            node_postprocessors=node_postprocessors,
            response_mode=response_mode, 
            verbose=True
        )
        if hydeTransform:
            query_engine = TransformQueryEngine(query_engine, query_transform=hyde)

        return query_engine
    
    def generate_final_query(self, query):
        a = "Basierend auf den folgenden Eigenschaften:"
        b = "</br> Geben Sie bitte die Knauf System ID an, die diesen Eigenschaften entspricht."
        return a + query + b
    
    def get_llm(self):
        llm = OpenAI(model=self.config["llm_name"])
        return llm
    
    def get_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"], trust_remote_code=True)
        return embed_model

    def parse_query(self, query, num_queries=1):
        llm = self.get_llm()
        queries = generate_queries(query.query, llm, num_queries)
        return queries