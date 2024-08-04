from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI


class RAG:
    def __init__(self, config_file):
        self.config = config_file
        Settings.llm = OpenAI(model=self.config["llm_name"])
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"], trust_remote_code=True) 
        # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        # Settings.num_output = 512
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