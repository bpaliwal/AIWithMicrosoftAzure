"""
Tests for Azure AI Search vector store.

RUBRIC:
Vector Store & RAG Setup
- Azure AI Search vector store initialized correctly (4 marks)
- Azure OpenAI embeddings configured properly (4 marks)
- LangChain AzureSearch integration is correct (4 marks)
"""

from unittest.mock import MagicMock, patch

from src.config import Config
from src.vector_store import get_vector_store


def test_azure_search_configuration():
    """Azure AI Search configuration is available."""

    assert Config.AZURE_SEARCH_ENDPOINT
    assert Config.AZURE_SEARCH_KEY
    assert Config.AZURE_SEARCH_INDEX_NAME


def test_azure_search_vector_store_initialization():
    """AzureSearch is initialized with the configured endpoint, key, index and embeddings."""

    embedding_function = MagicMock(name="AzureOpenAIEmbeddings")

    with patch("src.vector_store.AzureSearch") as MockAzureSearch:

        mock_vector_store = MagicMock(name="AzureSearchVectorStore")
        MockAzureSearch.return_value = mock_vector_store

        vector_store = get_vector_store(embedding_function)

        assert MockAzureSearch.called
        assert vector_store is mock_vector_store

        kwargs = MockAzureSearch.call_args.kwargs

        assert kwargs["azure_search_endpoint"] == Config.AZURE_SEARCH_ENDPOINT
        assert kwargs["azure_search_key"] == Config.AZURE_SEARCH_KEY
        assert kwargs["index_name"] == Config.AZURE_SEARCH_INDEX_NAME
        assert kwargs["embedding_function"] is embedding_function


def test_azure_search_requires_configuration():
    """AzureSearch initialization fails when required configuration is missing."""

    embedding_function = MagicMock(name="AzureOpenAIEmbeddings")

    with patch("src.vector_store.AzureSearch"):

        original_endpoint = Config.AZURE_SEARCH_ENDPOINT

        try:
            Config.AZURE_SEARCH_ENDPOINT = None

            try:
                get_vector_store(embedding_function)
                assert False, "Expected ValueError for missing endpoint"
            except ValueError as exc:
                assert "AZURE_SEARCH_ENDPOINT" in str(exc)

        finally:
            Config.AZURE_SEARCH_ENDPOINT = original_endpoint
