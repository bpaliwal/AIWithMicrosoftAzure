import os

from dotenv import load_dotenv

from src.search_engine import TravelSearchEngine


load_dotenv()


def test_blob_azure_rag_smoke():

    print("=" * 70)
    print("AZURE PRODUCTION RAG SMOKE TEST")
    print("=" * 70)

    print(f"BLOB_STORAGE: {os.getenv('BLOB_STORAGE')}")
    print(
        f"AZURE_STORAGE_CONTAINER_NAME: "
        f"{os.getenv('AZURE_STORAGE_CONTAINER_NAME')}"
    )
    print(
        f"VECTOR_STORE_TYPE_PROD: "
        f"{os.getenv('VECTOR_STORE_TYPE_PROD')}"
    )
    print(
        f"AZURE_SEARCH_INDEX_NAME: "
        f"{os.getenv('AZURE_SEARCH_INDEX_NAME')}"
    )

    assert os.getenv("BLOB_STORAGE", "").lower() == "true"
    assert os.getenv("VECTOR_STORE_TYPE_PROD") == "azure_search"

    print("\nInitializing TravelSearchEngine...")
    engine = TravelSearchEngine()

    print("\nTravelSearchEngine initialized successfully.")

    query = (
        "What is the specific reimbursement percentage for an "
        "Air India passenger who is involuntarily downgraded "
        "on a domestic flight?"
    )

    print("\nQuery:")
    print(query)

    print("\nRunning Azure AI Search retrieval...")

    results, processed_query = engine.search_by_text(
        query,
        k=5,
    )

    print("\nRETRIEVAL RESULT")
    print("-" * 70)
    print(f"Documents retrieved: {len(results)}")

    assert len(results) > 0

    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Source: {doc.metadata.get('source')}")
        print(f"   Category: {doc.metadata.get('category')}")
        print(
            f"   Content: "
            f"{doc.page_content[:300].replace(chr(10), ' ')}"
        )

    print("\nGenerating answer using Azure OpenAI...")

    answer = engine.synthesize_response(
        docs=results,
        user_query=query,
    )
    print("\nGENERATED ANSWER")
    print("-" * 70)
    print(answer)

    assert answer
    assert "75%" in answer

    print("\n" + "=" * 70)
    print("RESULT: AZURE BLOB + AZURE SEARCH + AZURE OPENAI RAG TEST SUCCESSFUL")
    print("=" * 70)