from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    get_response_synthesizer
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI

from src.services.query_parser_service import generate_queries


class RAG:
    def __init__(self, config_file):
        self.config = config_file
        Settings.llm = OpenAI(model=self.config["llm_name"])
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"], trust_remote_code=True)
        Settings.num_output = 512
        Settings.context_window = 3900

    def milvus_index(self):
        milvus_vector_store = MilvusVectorStore(
            uri=self.config["milvus"]["uri"],
            collection_name=self.config["milvus"]["collection_name"],
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

    @staticmethod
    def create_hyde_transform():
        custom_prompt = """
        Process the construction specification texts to extract and retain key details with high precision. 
        Ensure normalization of units and terms, but specifically focus on retaining detailed information about
        construction standards, product codes, material specifics, and performance characteristics. 
        Identify and tag entities like product codes, dimensions, fire resistance class, construction standards (e.g., DIN EN 14195), 
        and details of the construction elements such as type of beplankung, verspachtelung, and unterkonstruktion.
        Extract and maintain all identifiers and numeric values associated with standards, measurements, and quality grades. 
        Output should provide a comprehensive summary that includes these specifics in a clean, concise format, structured in German without new lines.

        {context_str}

        Output:"""
        new_hyde_tmpl = PromptTemplate(custom_prompt)
        return HyDEQueryTransform(include_original=True, hyde_prompt=new_hyde_tmpl)

    def query_engine(self,
                     index,
                     rerank=True,
                     alpha=0.5,
                     similarity_top_k=5,
                     response_mode="tree_summarize",
                     hyde_transform=False):
        node_postprocessors = []
        response_synthesizer = self.create_response_synthesizer()

        if rerank:
            rerank_processor = SentenceTransformerRerank(model=self.config["reranker_model"], top_n=3)
            node_postprocessors.append(rerank_processor)

        query_engine = index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=similarity_top_k,
            alpha=alpha,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
            response_mode=response_mode,
            verbose=True
        )

        if hyde_transform:
            hyde = self.create_hyde_transform()
            query_engine = TransformQueryEngine(query_engine, query_transform=hyde)

        return query_engine

    @staticmethod
    def generate_final_query(query: str) -> str:
        """Generate a final query for RAG model."""
        prefix = "Basierend auf den folgenden Eigenschaften:"
        suffix = "Geben Sie bitte die Knauf System ID an, die diesen Eigenschaften entspricht."
        return prefix + query + suffix


    def get_llm(self):
        llm = OpenAI(model=self.config["llm_name"])
        return llm

    def get_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"], trust_remote_code=True)
        return embed_model

    def parse_query(self, query, num_queries=1):
        llm = self.get_llm()
        queries = generate_queries(query, llm, num_queries)
        return queries

    def create_response_synthesizer(self):
        qa_prompt_tmpl_str = """\
            Context information is below.
            ---------------------
            {context_str}
            ---------------------
            Anhand der gegebenen Kontextinformationen und ohne Vorwissen, analysieren Sie die Anfrage, die einen Ausschreibungstext eines Wettbewerbers enthält. Identifizieren Sie die Knauf System-ID, die den in der Anfrage beschriebenen Eigenschaften und Spezifikationen am nächsten kommt. Es muss keine "100%ige" Übereinstimmung sein, eine Genauigkeit von über "70%" ist ausreichend. Berücksichtigen Sie dabei Faktoren wie Wandtyp, Dicke, Feuerwiderstand, Plattentyp und andere relevante Details. Geben Sie nur die passende Knauf System-ID an.
            Falls kein System im Kontext der Anfrage zu mindestens 70% übereinstimmt, antworten Sie mit "Keine passende Übereinstimmung gefunden."
            Erfinden Sie keine Informationen und geben Sie keine zusätzlichen Details an.
    
            Query: {query_str}
            Answer: \
            """
        qa_prompt = PromptTemplate(qa_prompt_tmpl_str)
        response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_prompt,
        )
        return response_synthesizer
