import streamlit as st
import requests
import json
from typing import Dict, Any

FASTAPI_ENDPOINT = "http://127.0.0.1:8000/api/search"

def parse_response(response: str) -> Dict[str, Any]:
    data = json.loads(response)
    return {
        "search_result": data["search_result"],
        "source_nodes": data["source_nodes"]
    }

def display_source_node(node: Dict[str, Any]):
    st.markdown(f"**Score:** {node['score']:.4f}")
    st.markdown(f"**Content:**\n{node['text']}")
    st.markdown("---")

st.set_page_config(page_title="Find Your Knauf System", layout="wide")

# Custom CSS to style the search bar, slider, and layout
st.markdown("""
<style>
    .stTextInput > div > div > input {
        max-width: 400px;
        min-height: 40px;
        max-height: 100px;
        overflow-y: auto;
    }
    .stSlider > div {
        width: 250px;
    }
    .main .block-container {
        margin-top: -50px;
    }
    .title-and-slider {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .title-and-slider h1 {
        margin-bottom: 0;
    }
    .slider-container {
        width: 250px;
    }
</style>
""", unsafe_allow_html=True)

# Title and slider in the same row
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="title-and-slider"><h1>Find Your Knauf System</h1></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="slider-container">', unsafe_allow_html=True)
    similarity_top_k = st.slider("Similar results", min_value=1, max_value=10, value=5, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

query = st.text_area("Enter competitor tender text", height=60, max_chars=None, key="query")

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            response = requests.post(FASTAPI_ENDPOINT, json={"query": query, "similarity_top_k": similarity_top_k})
        
        if response.status_code == 200:
            parsed_response = parse_response(response.text)
            
            if not parsed_response["search_result"] and not parsed_response["source_nodes"]:
                st.warning("Sorry, no system could be identified.")
            else:
                st.success("Search completed successfully!")
                
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    st.subheader("Search Result")
                    st.info(parsed_response["search_result"] or "No specific system identified.")
                
                with result_col2:
                    st.subheader(f"Source Nodes")
                    if parsed_response["source_nodes"]:
                        for idx, node in enumerate(parsed_response["source_nodes"], 1):
                            st.markdown(f"### Source {idx}")
                            display_source_node(node)
                    else:
                        st.info("No relevant source nodes found.")
        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.text)
    else:
        st.warning("Please enter a query.")

st.sidebar.header("About")
st.sidebar.info("This app uses a RAG (Retrieval-Augmented Generation) system to find relevant Knauf systems based on competitor tender text.")