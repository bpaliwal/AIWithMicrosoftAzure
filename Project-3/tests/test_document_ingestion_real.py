import pytest

from src.data_loader import TravelDataLoader


def test_real_document_ingestion_loads_all_documents():
    """
    REAL document ingestion test.

    Demonstrates that the TravelDataLoader discovers and loads
    the documents from the configured data directory.
    """

    print("=" * 70)
    print("REAL DOCUMENT INGESTION - LOAD ALL DOCUMENTS")
    print("=" * 70)

    loader = TravelDataLoader()

    documents = loader.load_all_travel_documents()

    print()
    print("INGESTION RESULT")
    print("-" * 70)
    print(f"Documents loaded : {len(documents)}")

    assert documents, "No documents were loaded."

    # Every loaded document should contain text.
    assert all(
        hasattr(doc, "page_content") and doc.page_content.strip()
        for doc in documents
    )

    # Every document should have metadata.
    assert all(
        hasattr(doc, "metadata")
        for doc in documents
    )

    # The current knowledge base contains 9 PDFs and loads
    # their pages as individual LangChain Documents.
    #
    # Current expected count is 152 based on the configured
    # data directory.
    assert len(documents) == 152

    print()
    print("SAMPLE DOCUMENT METADATA")
    print("-" * 70)

    for i, doc in enumerate(documents[:5], start=1):
        print(f"Document {i}")
        print(f"Source   : {doc.metadata.get('source', 'Unknown')}")
        print(f"Category : {doc.metadata.get('category', 'Unknown')}")
        print(f"Characters: {len(doc.page_content)}")
        print()

    print("=" * 70)
    print("RESULT: REAL DOCUMENT INGESTION SUCCESSFUL")
    print("=" * 70)
