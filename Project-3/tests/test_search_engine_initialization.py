"""
Tests for TravelSearchEngine initialization.

RUBRIC:
Search Engine Implementation
- TravelSearchEngine initialized correctly (4 marks)
"""

from unittest.mock import MagicMock, patch

from src.search_engine import TravelSearchEngine


def test_travel_search_engine_initialization():
    """
    Verify that TravelSearchEngine initializes all required
    RAG components:
      - Governance gate
      - Azure Chat OpenAI
      - Azure OpenAI embeddings
      - Azure AI Search vector store
    """

    with patch("src.search_engine.mlflow.set_tracking_uri"), \
         patch("src.search_engine.mlflow.set_experiment"), \
         patch("src.search_engine.GovernanceGate") as mock_governance, \
         patch("src.search_engine.AzureChatOpenAI") as mock_chat, \
         patch("src.search_engine.AzureOpenAIEmbeddings") as mock_embeddings, \
         patch("src.search_engine.get_vector_store") as mock_vector_store:

        mock_vector_store.return_value = MagicMock()

        engine = TravelSearchEngine()

        assert engine is not None
        assert engine.governance_gate is not None
        assert engine.llm is not None
        assert engine.embeddings is not None
        assert engine.vector_store is not None

        mock_governance.assert_called_once()
        mock_chat.assert_called_once()
        mock_embeddings.assert_called_once()
        mock_vector_store.assert_called_once_with(engine.embeddings)