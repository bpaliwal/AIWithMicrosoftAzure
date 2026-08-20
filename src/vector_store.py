"""
Vector Store Configuration
RUBRIC: Vector Store & RAG Setup (12 marks total)
- Azure AI Search vector store initialized correctly (4 marks)
- Azure OpenAI embeddings configured properly (4 marks)
- LangChain AzureSearch integration is correct (4 marks)

TASK: Initialize Azure AI Search vector store with embeddings
"""
from src.config import Config
from langchain_community.vectorstores import AzureSearch

def get_vector_store(embedding_function):
    """
    Returns Azure AI Search vector store (no ChromaDB option)
    
    HINT: This function should:
    1. Get Azure Search credentials from Config
    2. Validate that endpoint and key are present
    3. Initialize AzureSearch with correct parameters
    4. Return the vector store instance
    
    Args:
        embedding_function: The LangChain embedding function to use
    """
    
    # HINT: Get configuration values from Config class
    endpoint = Config.AZURE_SEARCH_ENDPOINT
    key = Config.AZURE_SEARCH_KEY
    index_name = Config.AZURE_SEARCH_INDEX_NAME

    print(f"DEBUG: Configured Azure Search endpoint: {endpoint}")
    print(f"DEBUG: Configured Azure Search index: {index_name}")

    # HINT: Validate that required credentials are present
    if not endpoint or not key or not index_name:
        raise ValueError("AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, and AZURE_SEARCH_INDEX_NAME must be set.")

    if not endpoint or not key or not index_name:
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, "
            "and AZURE_SEARCH_INDEX_NAME must be set."
        )

    print("DEBUG: Initializing AzureSearch vector store...")

    vector_store = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=key,
        index_name=index_name,
        embedding_function=embedding_function,
    )

    print("DEBUG: AzureSearch vector store initialized successfully.")

    return vector_store
