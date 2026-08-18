"""
Verify that the Azure AI Search production index contains
the expected travel source documents.
"""

from collections import Counter

from src.search_engine import TravelSearchEngine


EXPECTED_SOURCES = {
    "air-india-coc.pdf",
    "air-india-general-booking-policies-oct2025.pdf",
    "ai-schedule-change-policy-updated.pdf",
    "U.S. Department of Transportation - Aircraft Dissinection.pdf",
    "U.S. Department of Transportation - Implementation of the Consumer Credit Protection Act With Respect to Air Carriers and Foreign Air Carriers.pdf",
    "U.S. Department of Transportation - Air Consumer Privacy.pdf",
    "U.S. Department of Transportation - Aviation Industry Bankruptcy and Service Cessation.pdf",
    "U.S. Department of Transportation - Refunds and Other Consumer Protections.pdf",
    "U.S. Department of Transportation - Refunds for Airline Fare and Ancillary Service Fees.pdf",
}


def test_azure_search_index_sources():

    print("=" * 70)
    print("AZURE AI SEARCH INDEX SOURCE VERIFICATION")
    print("=" * 70)

    engine = TravelSearchEngine()

    found_sources = set()

    # Use representative queries covering the knowledge base.
    queries = [
        "Air India baggage booking cancellation policy",
        "Air India name change churn void policy",
        "Air India schedule change refund",
        "DOT airline refund credit card",
        "DOT baggage fee refund",
        "DOT privacy COPPA airline",
        "airline bankruptcy credit card",
        "aircraft disinsection countries",
        "Implementation of the Consumer Credit Protection Act air carriers foreign air carriers",
    ]
    
    for query in queries:

        results, _ = engine.search_by_text(
            query,
            k=10,
        )

        for doc in results:
            source = doc.metadata.get("source")

            if source:
                found_sources.add(source)

    print("\nSources discovered through Azure AI Search:")
    print("-" * 70)

    for source in sorted(found_sources):
        print("✓", source)

    print("\nExpected sources:", len(EXPECTED_SOURCES))
    print("Discovered sources:", len(found_sources))

    missing = EXPECTED_SOURCES - found_sources

    if missing:

        print("\nMISSING SOURCES")
        print("-" * 70)

        for source in sorted(missing):
            print("✗", source)

    else:

        print("\nAll expected source documents were discovered.")

    print("\n" + "=" * 70)

    assert not missing, (
        "Expected source documents were not found "
        "through Azure AI Search."
    )

    print(
        "RESULT: AZURE AI SEARCH SOURCE VERIFICATION SUCCESSFUL"
    )

    print("=" * 70)
