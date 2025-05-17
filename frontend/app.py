import streamlit as st
import requests
import pandas as pd
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint (configurable)
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def check_api_health():
    """Check if the API is available and return health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unavailable", "error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "unavailable", "error": str(e)}

def search_documents(query, top_k=5):
    """Send search query to API and return results"""
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(f"{API_URL}/search", json=payload)
        
        if response.status_code == 200:
            return response.json()["results"]
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return []

# Sidebar for configuration
with st.sidebar:
    st.title("Semantic Search")
    st.markdown("Search documents based on meaning, not just keywords.")
    
    # API connection status
    st.subheader("API Connection")
    health_status = check_api_health()
    
    if health_status["status"] == "healthy":
        st.success("✅ API is connected")
        st.info(f"Documents indexed: {health_status.get('documents_indexed', 'Unknown')}")
    else:
        st.error("❌ API is unavailable")
        st.warning(f"Error: {health_status.get('error', 'Unknown error')}")
        st.info("Make sure the API is running and accessible.")
        st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000")
    
    # Search settings
    st.subheader("Search Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    # About section
    st.subheader("About")
    st.markdown("""
    This application uses semantic search to find documents based on meaning rather than exact keyword matching.
    
    Powered by:
    - Sentence Transformers
    - FAISS (Facebook AI Similarity Search)
    - FastAPI
    - Streamlit
    """)

# Main content
st.title("🔍 Semantic Document Search")

# Search input
query = st.text_input("Enter your search query", placeholder="What are you looking for?")

# Search button
col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("Search", type="primary", use_container_width=True)

# Execute search when button is clicked or Enter is pressed
if search_button or (query and len(query.strip()) > 0):
    if not query or len(query.strip()) == 0:
        st.warning("Please enter a search query")
    else:
        with st.spinner("Searching documents..."):
            results = search_documents(query, top_k)
        
        if results:
            st.subheader(f"Found {len(results)} results")
            
            # Display results in a nice format
            for i, result in enumerate(results):
                score_percentage = round(float(result["score"]) * 100, 2)
                
                with st.container():
                    st.markdown(f"### Result {i+1} - Relevance: {score_percentage}%")
                    st.markdown(f"**Document ID:** {result['document_id']}")
                    
                    # Display the text in an expandable container if it's long
                    text = result["text"]
                    if len(text) > 300:
                        st.markdown(f"{text[:300]}...")
                        with st.expander("Show full text"):
                            st.markdown(text)
                    else:
                        st.markdown(text)
                    
                    st.divider()
        else:
            st.info("No results found. Try a different query.")

# Initial state - show instructions
if not query:
    st.info("Enter a search query above to find relevant documents.")
    
    # Example queries
    st.subheader("Example Queries")
    example_queries = [
        "Machine learning techniques for natural language processing",
        "Climate change impact on agriculture",
        "Renewable energy solutions for sustainable development",
        "Data privacy regulations in technology"
    ]
    
    for example in example_queries:
        if st.button(example, key=f"example_{example}"):
            # This will be handled in the next run of the script
            st.session_state.query = example
            st.rerun()