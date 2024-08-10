import pandas as pd
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    Document
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import argparse
import yaml
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.llms import Ollama

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

        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"])
        qdrant_vector_store = QdrantVectorStore(
            client=client, collection_name=self.config["collection_name"]
        )
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
        service_context = ServiceContext.from_defaults(
            llm=None, embed_model=embedder, chunk_size=self.config["chunk_size"]
        )

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, service_context=service_context
        )
        print(
            f"Data indexed successfully to Qdrant. Collection: {self.config['collection_name']}"
        )
        return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--ingest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ingest data to Qdrant vector Database.",
    )

    args = parser.parse_args()
    config_file = "config.yml"
    with open(config_file, "r") as conf:
        config = yaml.safe_load(conf)
    data = Data(config)

    if args.ingest:
        print("Loading Embedder...")
        embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=config["embedding_model"])
        )
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        data.ingest(embedder=embed_model, llm=llm)