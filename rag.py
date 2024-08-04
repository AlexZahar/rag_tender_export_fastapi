from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

class RAG:
    def __init__(self, config_file, llm):
        self.config = config_file
        self.llm = llm  # ollama llm
    
    def load_embedder(self):
        embed_model = HuggingFaceEmbedding(model_name=self.config['embedding_model'], trust_remote_code=True)
        return embed_model

    def milvus_index(self):
        milvus_vector_store = MilvusVectorStore(
            uri="./milvus_innenwande_tender_fastapi.db",
            dim=768,
            overwrite=False,
            enable_sparse=True,
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 50},
            verbose=True,
        )
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.load_embedder()
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=milvus_vector_store, service_context=service_context
        )
        return index