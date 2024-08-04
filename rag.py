from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

class RAG:
    def __init__(self, config_file, llm):
        self.config = config_file
        self.llm = llm  # ollama llm
    
    def load_embedding_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.config['embedding_model'], trust_remote_code=True)
        return embed_model

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
            vector_store=milvus_vector_store, embed_model=self.load_embedding_model(), llm=self.llm
        )
        return index