import streamlit as st
import requests
from typing import List, Optional
from rag_tender_export_fastapi.models.models import Query, Response, SourceNode

st.set_page_config(page_title="Research RAG", layout="wide")

st.title("Research RAG")

# Default configuration
DEFAULT_CONFIG = {
    "rerank": True,
    "hyde_transform": True,
    "use_parser": False,
    "similarity_top_k": 5,
    "alpha": 0.5,
    "response_mode": "tree_summarize"
}

# Sidebar configuration
st.sidebar.title("Configuration")

# Add a reset button next to the checkbox
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    show_config = st.checkbox("Show Configuration Options", value=False, key="show_config_checkbox")
with col2:
    if st.button("Reset", key="reset_button"):
        st.session_state.update(DEFAULT_CONFIG)
        st.rerun()

if show_config:
    st.sidebar.subheader("Query Engine Configuration")

    rerank = st.sidebar.checkbox(
        "Rerank",
        value=st.session_state.get("rerank", DEFAULT_CONFIG["rerank"]),
        key="rerank",
        help="Enable or disable reranking of search results."
    )

    hyde_transform = st.sidebar.checkbox(
        "HyDE Transform",
        value=st.session_state.get("hyde_transform", DEFAULT_CONFIG["hyde_transform"]),
        key="hyde_transform",
        help="Enable or disable Hypothetical Document Embeddings (HyDE) transformation."
    )

    use_parser = st.sidebar.checkbox(
        "Use Query Parser",
        value=st.session_state.get("use_parser", DEFAULT_CONFIG["use_parser"]),
        key="use_parser",
        help="Enable or disable query parsing."
    )

    # Put sliders in a single row
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        similarity_top_k = st.slider(
            "Similarity Top K",
            min_value=1,
            max_value=10,
            value=st.session_state.get("similarity_top_k", DEFAULT_CONFIG["similarity_top_k"]),
            step=1,
            key="similarity_top_k",
            help="Number of top similar documents to retrieve."
        )

    with col2:
        alpha = st.slider(
            "Alpha",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("alpha", DEFAULT_CONFIG["alpha"]),
            step=0.01,
            key="alpha",
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
        index=response_mode_options.index(st.session_state.get("response_mode", DEFAULT_CONFIG["response_mode"])),
        format_func=lambda x: x.replace("_", " ").title(),
        key="response_mode",
        help="Select the mode for processing and combining retrieved text chunks."
    )

    # Display the description of the selected response mode
    st.sidebar.info(response_mode_descriptions[response_mode])

else:
    # Use session state or default values when configuration is hidden
    rerank = st.session_state.get("rerank", DEFAULT_CONFIG["rerank"])
    hyde_transform = st.session_state.get("hyde_transform", DEFAULT_CONFIG["hyde_transform"])
    use_parser = st.session_state.get("use_parser", DEFAULT_CONFIG["use_parser"])
    similarity_top_k = st.session_state.get("similarity_top_k", DEFAULT_CONFIG["similarity_top_k"])
    alpha = st.session_state.get("alpha", DEFAULT_CONFIG["alpha"])
    response_mode = st.session_state.get("response_mode", DEFAULT_CONFIG["response_mode"])

# Main area for query input and results
query = st.text_area("Enter your query", height=100)

# Create a placeholder for the search button
search_button_placeholder = st.empty()

# Create a placeholder for the results
results_placeholder = st.container()

if search_button_placeholder.button("Search", key="search_button"):
    if query:
        # Disable the search button
        search_button_placeholder.empty()
        disabled_button = st.button("Searching...", disabled=True)

        # Show loading spinner
        with st.spinner("Searching..."):
            # Prepare the request payload
            payload = Query(
                query=query,
                similarity_top_k=similarity_top_k,
                rerank=rerank,
                hyde_transform=hyde_transform,
                use_parser=use_parser,
                alpha=alpha,
                response_mode=response_mode
            )

            # Make the API request
            response = requests.post("http://localhost:8000/api/search", json=payload.dict())

            if response.status_code == 200:
                result = Response(**response.json())
                
                with results_placeholder:
                    st.subheader("Search Result")
                    st.write(result.search_result)

                    st.subheader("Source Nodes")
                    for node in result.source_nodes:
                        with st.expander(f"Score: {node.score:.4f}"):
                            st.write(node.text)
            else:
                with results_placeholder:
                    st.error(f"Error: {response.status_code} - {response.text}")

        # Re-enable the search button
        search_button_placeholder.button("Search", key="search_button_after")
    else:
        st.warning("Please enter a query.")