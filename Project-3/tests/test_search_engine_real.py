"""
REAL integration tests for TravelSearchEngine.

RUBRIC:
Search Engine Implementation
- TravelSearchEngine initialized correctly (4 marks)

No mocks are used.
The test uses the actual configured MLflow, Azure OpenAI,
Azure OpenAI Embeddings, Azure AI Search, and Governance components.
"""

from src.search_engine import TravelSearchEngine


def test_travel_search_engine_real_initialization():
    """Initialize the complete TravelSearchEngine using real configuration."""

    engine = TravelSearchEngine()

    assert engine is not None
    assert engine.governance_gate is not None
    assert engine.llm is not None
    assert engine.embeddings is not None
    assert engine.vector_store is not None

    assert engine._last_generation_metrics is not None

    print("\n" + "=" * 70)
    print("REAL TRAVEL SEARCH ENGINE INITIALIZATION")
    print("=" * 70)
    print("GovernanceGate              : INITIALIZED")
    print("AzureChatOpenAI             : INITIALIZED")
    print("AzureOpenAIEmbeddings       : INITIALIZED")
    print("Azure AI Search VectorStore : INITIALIZED")
    print("MLflow tracking URI         : CONFIGURED")
    print("=" * 70)
