"""
Azure Production Configuration Smoke Test

Purpose:
    Verify that the application can run locally using the
    production Azure configuration before Docker deployment.

Checks:
    1. Azure AI Search is selected instead of ChromaDB.
    2. Azure AI Search configuration is available.
    3. TravelSearchEngine initializes successfully.
    4. Azure AI Search retrieves relevant documents.
    5. Azure OpenAI generates a response.

This test does NOT start Streamlit.
"""

from src.config import Config
from src.search_engine import TravelSearchEngine


def test_azure_production_smoke():

    print("=" * 70)
    print("AZURE PRODUCTION CONFIGURATION SMOKE TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    print("VECTOR_STORE_TYPE:", Config.VECTOR_STORE_TYPE)
    print("AZURE_SEARCH_ENDPOINT:", Config.AZURE_SEARCH_ENDPOINT)
    print("AZURE_SEARCH_INDEX_NAME:", Config.AZURE_SEARCH_INDEX_NAME)
    print(
        "AZURE_STORAGE_CONTAINER_NAME:",
        Config.AZURE_STORAGE_CONTAINER_NAME,
    )

    assert (
        Config.VECTOR_STORE_TYPE == "azure_search"
    ), (
        "Production smoke test requires "
        "VECTOR_STORE_TYPE=azure_search"
    )

    assert Config.AZURE_SEARCH_ENDPOINT
    assert Config.AZURE_SEARCH_INDEX_NAME

    # ---------------------------------------------------------------
    # Initialize search engine
    # ---------------------------------------------------------------

    print("\nInitializing TravelSearchEngine...")

    engine = TravelSearchEngine()

    print("\nTravelSearchEngine initialized successfully.")

    # ---------------------------------------------------------------
    # Test query
    # ---------------------------------------------------------------

    query = (
        "What is the specific reimbursement percentage for "
        "an Air India passenger who is involuntarily downgraded "
        "on a domestic flight?"
    )

    print("\nQuery:")
    print(query)

    # ---------------------------------------------------------------
    # Azure AI Search retrieval
    # ---------------------------------------------------------------

    print("\nRunning Azure AI Search retrieval...")

    results, processed_query = engine.search_by_text(
        query,
        k=5,
    )

    print("\nRETRIEVAL RESULT")
    print("-" * 70)

    print("Processed query:", processed_query)
    print("Documents retrieved:", len(results))

    assert len(results) > 0, (
        "Azure AI Search returned no documents."
    )

    for i, doc in enumerate(results, 1):

        print(
            f"\n{i}. Source: "
            f"{doc.metadata.get('source', 'Unknown')}"
        )

        print(
            "   Category: "
            f"{doc.metadata.get('category', 'N/A')}"
        )

        content = (
            doc.page_content
            .replace("\n", " ")
        )

        print(
            "   Content: "
            f"{content[:250]}"
        )

    # ---------------------------------------------------------------
    # Azure OpenAI response generation
    # ---------------------------------------------------------------

    print("\nGenerating answer using Azure OpenAI...")

    answer = engine.synthesize_response(
        results,
        processed_query,
    )

    print("\nGENERATED ANSWER")
    print("-" * 70)
    print(answer)

    assert answer
    assert len(answer.strip()) > 0

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "RESULT: AZURE SEARCH + AZURE OPENAI "
        "SMOKE TEST SUCCESSFUL"
    )
    print("=" * 70)
