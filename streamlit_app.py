
import streamlit as st
import requests

# Assuming you have these models defined
from models.models import Query, Response, SourceNode

st.set_page_config(page_title="Research RAG", layout="wide")

st.title("Research RAG")

# Sidebar configuration
st.sidebar.title("Configuration")
show_config = st.sidebar.checkbox("Show Configuration Options", value=False)

if show_config:
    st.sidebar.subheader("Query Engine Configuration")

    rerank = st.sidebar.radio(
        "Rerank",
        options=[True, False],
        index=0,
        help="Enable or disable reranking of search results."
    )

    hyde_transform = st.sidebar.radio(
        "HyDE Transform",
        options=[True, False],
        index=0,
        help="Enable or disable Hypothetical Document Embeddings (HyDE) transformation."
    )

    similarity_top_k = st.sidebar.number_input(
        "Similarity Top K",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of top similar documents to retrieve."
    )

    alpha = st.sidebar.slider(
        "Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Weight between keyword search (0.0) and vector search (1.0) in hybrid mode. Default is 0.5."
    )

    response_mode_options = [
        "refine",
        "compact",
        "tree_summarize",
        "simple_summarize",
        "no_text",
        "accumulate",
        "compact_accumulate"
    ]

    response_mode_descriptions = {
        "refine": "Processes each text chunk sequentially, refining the answer iteratively.",
        "compact": "Similar to refine, but concatenates chunks to fit more into the context window.",
        "tree_summarize": "Recursively summarizes chunks until a final answer is obtained.",
        "simple_summarize": "Truncates all chunks to fit into a single LLM prompt for quick summarization.",
        "no_text": "Only retrieves nodes without sending them to the LLM.",
        "accumulate": "Applies the query to each chunk separately and accumulates the responses.",
        "compact_accumulate": "Similar to accumulate, but compacts each LLM prompt like the compact mode."
    }

    response_mode = st.sidebar.selectbox(
        "Response Mode",
        options=response_mode_options,
        index=2,  # Default to "tree_summarize"
        format_func=lambda x: x.replace("_", " ").title(),
        help="Select the mode for processing and combining retrieved text chunks."
    )

    # Display the description of the selected response mode
    st.sidebar.info(response_mode_descriptions[response_mode])

else:
    # Default values when configuration is hidden
    rerank = True
    hyde_transform = False
    similarity_top_k = 5
    alpha = 0.5
    response_mode = "tree_summarize"

# Main area for query input and results
query = st.text_area("Enter your query", height=100)

if st.button("Search"):
    if query:
        # Prepare the request payload
        payload = Query(
            query=query,
            similarity_top_k=similarity_top_k,
            rerank=rerank,
            hyde_transform=hyde_transform,
            alpha=alpha,
            response_mode=response_mode
        )

        # Make the API request
        response = requests.post("http://localhost:8000/api/search", json=payload.dict())

        if response.status_code == 200:
            result = Response(**response.json())
            
            st.subheader("Search Result")
            st.write(result.search_result)

            st.subheader("Source Nodes")
            for node in result.source_nodes:
                with st.expander(f"Score: {node.score:.4f}"):
                    st.write(node.text)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a query.")