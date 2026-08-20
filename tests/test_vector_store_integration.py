"""
Tests for LangChain Azure AI Search vector store integration.

RUBRIC:
Vector Store & RAG Setup
- Azure AI Search vector store initialized correctly (4 marks)
- Azure OpenAI embeddings configured properly (4 marks)
- LangChain AzureSearch integration is correct (4 marks)
"""

from src.config import Config
from src.vector_store import get_vector_store
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch


def test_azure_search_configuration():
    """Verify Azure AI Search configuration is available."""

    assert Config.AZURE_SEARCH_ENDPOINT
    assert Config.AZURE_SEARCH_KEY
    assert Config.AZURE_SEARCH_INDEX_NAME


def test_azure_openai_embedding_configuration():
    """Verify Azure OpenAI embeddings can be initialized."""

    assert Config.AZURE_OPENAI_API_KEY
    assert Config.AZURE_OPENAI_ENDPOINT
    assert Config.AZURE_OPENAI_API_VERSION
    assert Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    embeddings = AzureOpenAIEmbeddings(
        api_key=Config.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=Config.AZURE_OPENAI_API_VERSION,
    )

    assert embeddings is not None


def test_langchain_azure_search_initialization():
    """
    Verify LangChain AzureSearch is initialized using the
    configured Azure AI Search service and embedding function.
    """

    embeddings = AzureOpenAIEmbeddings(
        api_key=Config.AZURE_OPENAI_API_KEY,
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=Config.AZURE_OPENAI_API_VERSION,
    )

    vector_store = get_vector_store(embeddings)

    assert vector_store is not None
    assert isinstance(vector_store, AzureSearch)
    