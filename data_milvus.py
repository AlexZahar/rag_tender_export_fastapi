import pandas as pd
from llama_index.core import (
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
    Document
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
import argparse
import yaml


class Data:
    def __init__(self, config):
        self.config = config

    def ingest(self, embedder, llm):
        print("Indexing data...")
        
        # Load data from CSV
        df = pd.read_csv(self.config["data_path"])
        
        # Create documents from DataFrame
        documents = [
            Document(
                text=f"Knauf System ID: {row['name']}, Properties: {row['long_tender_text']}",
                metadata={"name": row['name']}
            ) for i, row in df.iterrows()
        ]

        
        milvus_vector_store = MilvusVectorStore(
            uri=self.config["milvus"]["uri"],
            dim=self.config["milvus"]["dim"],
            overwrite=True,
            enable_sparse=self.config["milvus"]["enable_sparse"],
            hybrid_ranker=self.config["milvus"]["hybrid_ranker"],
            hybrid_ranker_params=self.config["milvus"]["hybrid_ranker_params"],
            verbose=self.config["milvus"]["verbose"]
        )
        storage_context = StorageContext.from_defaults(vector_store=milvus_vector_store)
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embedder
        )

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True, service_context=service_context
        )
        print(
            f"Data indexed successfully to Milvus."
        )
        return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ingest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ingest data to Milvus vector Database.",
    )

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)

    if args.ingest:
        print("Loading Embedder...")
        llm = OpenAI(config["llm_name"])
        embed_model = HuggingFaceEmbedding(model_name=config["embedding_model"], trust_remote_code=True) 
        data.ingest(embedder=embed_model, llm=llm)