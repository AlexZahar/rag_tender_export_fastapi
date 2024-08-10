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

st.set_page_config(page_title="Find Your Code", layout="wide")

# Custom CSS to style the search bar
st.markdown("""
<style>
    .stTextInput > div > div > input {
        max-width: 400px;
        min-height: 40px;
        max-height: 100px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

st.title("Find Your Knauf System")

query = st.text_area("Enter competitor tender text", height=180, max_chars=None, key="query")

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            response = requests.post(FASTAPI_ENDPOINT, json={"query": query, "similarity_top_k": 2})
        
        if response.status_code == 200:
            parsed_response = parse_response(response.text)
            
            st.success("Search completed successfully!")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Search Result")
                st.info(parsed_response["search_result"])
            
            with col2:
                st.subheader("Source Nodes")
                for idx, node in enumerate(parsed_response["source_nodes"], 1):
                    st.markdown(f"### Source {idx}")
                    display_source_node(node)
        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.text)
    else:
        st.warning("Please enter a query.")

st.sidebar.header("About")
st.sidebar.info("This app uses a RAG (Retrieval-Augmented Generation) system to find relevant code snippets based on your query.")