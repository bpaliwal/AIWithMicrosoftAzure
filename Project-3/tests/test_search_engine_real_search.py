import pytest

from src.search_engine import TravelSearchEngine


@pytest.mark.integration
def test_real_search_by_text():
    """
    Real integration test for TravelSearchEngine.search_by_text().

    This test:
    - Uses the actual Azure OpenAI embeddings configuration
    - Uses the actual Azure AI Search vector store
    - Performs a real similarity search
    - Verifies that documents are retrieved
    - Captures source/category/content evidence
    """

    query = "What is the baggage allowance for Air India?"

    print("=" * 70)
    print("REAL AZURE AI SEARCH - search_by_text() EVIDENCE")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    engine = TravelSearchEngine()

    results, returned_query = engine.search_by_text(
        query,
        k=5,
    )

    print(f"Returned query: {returned_query}")
    print(f"Documents retrieved: {len(results)}")
    print()

    assert returned_query == query
    assert len(results) > 0, (
        "Real Azure AI Search returned no documents. "
        "Verify that the Azure Search index contains indexed documents."
    )

    print("RETRIEVED DOCUMENTS")
    print("-" * 70)

    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "N/A")
        category = doc.metadata.get("category", "N/A")
        content = doc.page_content.replace("\n", " ").strip()

        print(f"\nDocument {i}")
        print(f"Source   : {source}")
        print(f"Category : {category}")
        print(f"Preview  : {content[:500]}")

    print()
    print("=" * 70)
    print("RESULT: REAL SIMILARITY SEARCH SUCCESSFUL")
    print("=" * 70)