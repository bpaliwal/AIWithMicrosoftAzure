from src.search_engine import TravelSearchEngine


def test_missing_implementation_document():

    print("=" * 70)
    print("CHECK MISSING DOT CONSUMER CREDIT PROTECTION DOCUMENT")
    print("=" * 70)

    engine = TravelSearchEngine()

    queries = [
        "Implementation of the Consumer Credit Protection Act",
        "Consumer Credit Protection Act air carriers foreign air carriers",
        "credit protection act airline",
        "air carriers foreign air carriers consumer credit",
    ]

    found = []

    for query in queries:

        print(f"\nQuery: {query}")

        results, _ = engine.search_by_text(query, k=10)

        for doc in results:

            source = doc.metadata.get("source")
            print(f"  Source: {source}")

            if source and "Implementation of the Consumer Credit Protection Act" in source:
                found.append(doc)

    print("\n" + "=" * 70)

    if found:
        print(
            "RESULT: DOCUMENT FOUND IN AZURE AI SEARCH"
        )
    else:
        print(
            "RESULT: DOCUMENT NOT FOUND IN AZURE AI SEARCH"
        )

    print("=" * 70)

    assert found, (
        "The Implementation of the Consumer Credit Protection Act "
        "document was not found in Azure AI Search."
    )
