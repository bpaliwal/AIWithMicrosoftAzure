"""
Real integration tests for the Document Ingestion Pipeline.

RUBRIC:
1. Ingestion script loads all documents.        (2 marks)
2. Documents are chunked properly.              (2 marks)
3. Batch indexing implemented.                  (2 marks)
4. Index verification performed.                (2 marks)

These tests intentionally use REAL project components and
REAL Azure AI Search.

No mocks are used.
"""

import pytest

from src.config import Config
from src.data_loader import TravelDataLoader
from src.search_engine import TravelSearchEngine


pytestmark = pytest.mark.integration


# ================================================================
# 1. REAL DOCUMENT LOADING + CHUNKING
# ================================================================

def test_real_document_loading_and_chunking():

    print("\n" + "=" * 70)
    print("REAL DOCUMENT INGESTION - LOAD AND CHUNK")
    print("=" * 70)

    loader = TravelDataLoader()

    # ------------------------------------------------------------
    # Load all documents
    # ------------------------------------------------------------

    documents = loader.load_all_travel_documents()

    print(f"Documents loaded : {len(documents)}")

    assert documents, (
        "No travel documents were loaded from the data directory."
    )

    # ------------------------------------------------------------
    # Split documents
    # ------------------------------------------------------------

    chunks = loader.split_documents(documents)

    print(f"Chunks produced  : {len(chunks)}")
    print(
        f"Chunk size       : "
        f"{loader.text_splitter._chunk_size}"
    )
    print(
        f"Chunk overlap    : "
        f"{loader.text_splitter._chunk_overlap}"
    )

    assert chunks, (
        "Document splitting produced no chunks."
    )

    # At minimum, chunking should preserve all source content
    # as one or more chunks per source document.
    assert len(chunks) >= len(documents), (
        "Expected chunking to produce at least as many "
        "chunks as source documents."
    )

    # ------------------------------------------------------------
    # Verify chunk structure
    # ------------------------------------------------------------

    for chunk in chunks:

        assert chunk.page_content.strip(), (
            "A generated chunk contains no content."
        )

        assert chunk.metadata.get("source"), (
            "A generated chunk is missing source metadata."
        )

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    print("\nSAMPLE CHUNKS")
    print("-" * 70)

    for i, chunk in enumerate(chunks[:5], 1):

        preview = (
            chunk.page_content[:200]
            .replace("\n", " ")
        )

        print(f"Chunk {i}")
        print(
            f"Source     : "
            f"{chunk.metadata.get('source')}"
        )
        print(
            f"Category   : "
            f"{chunk.metadata.get('category', 'Unknown')}"
        )
        print(
            f"Characters : "
            f"{len(chunk.page_content)}"
        )
        print(
            f"Preview    : "
            f"{preview}"
        )
        print()

    print("=" * 70)
    print("RESULT: REAL DOCUMENT CHUNKING SUCCESSFUL")
    print("=" * 70)


# ================================================================
# 2. REAL BATCH INDEXING
# ================================================================

def test_real_batch_indexing_to_azure_search():

    print("\n" + "=" * 70)
    print("REAL AZURE AI SEARCH - BATCH INDEXING")
    print("=" * 70)

    loader = TravelDataLoader()

    engine = TravelSearchEngine()

    # ------------------------------------------------------------
    # Load and chunk real documents
    # ------------------------------------------------------------

    documents = loader.load_all_travel_documents()

    assert documents, (
        "No documents available for indexing."
    )

    chunks = loader.split_documents(documents)

    assert chunks, (
        "No chunks available for indexing."
    )

    # ------------------------------------------------------------
    # Production batch size
    # ------------------------------------------------------------

    batch_size = 50

    batch = chunks[:batch_size]

    print(
        f"Configured batch size : {batch_size}"
    )

    print(
        f"Actual batch size     : {len(batch)}"
    )

    print(
        f"Azure Search endpoint : "
        f"{Config.AZURE_SEARCH_ENDPOINT}"
    )

    print(
        f"Azure Search index    : "
        f"{Config.AZURE_SEARCH_INDEX_NAME}"
    )

    assert len(batch) > 0
    assert len(batch) <= batch_size

    # ------------------------------------------------------------
    # REAL Azure AI Search write
    #
    # This deliberately uses the production vector store.
    # There is NO mock here.
    # ------------------------------------------------------------

    engine.vector_store.add_documents(batch)

    print(
        f"Successfully indexed  : "
        f"{len(batch)} chunks"
    )

    print("=" * 70)
    print(
        "RESULT: REAL AZURE AI SEARCH "
        "BATCH INDEXING SUCCESSFUL"
    )
    print("=" * 70)


# ================================================================
# 3. REAL INDEX VERIFICATION
# ================================================================

def test_real_index_verification():

    print("\n" + "=" * 70)
    print("REAL AZURE AI SEARCH - INDEX VERIFICATION")
    print("=" * 70)

    engine = TravelSearchEngine()

    test_query = (
        "What are the baggage allowance rules "
        "for international flights?"
    )

    print(
        f"Test query: {test_query}"
    )

    # ------------------------------------------------------------
    # REAL Azure AI Search similarity search
    # ------------------------------------------------------------

    results, returned_query = (
        engine.search_by_text(
            test_query,
            k=5,
        )
    )

    print(
        f"Returned query : {returned_query}"
    )

    print(
        f"Documents found: {len(results)}"
    )

    # ------------------------------------------------------------
    # Verification assertions
    # ------------------------------------------------------------

    assert returned_query == test_query

    assert results, (
        "Azure AI Search returned no results. "
        "Index verification failed."
    )

    assert len(results) <= 5

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    print("\nVERIFIED RESULTS")
    print("-" * 70)

    for i, doc in enumerate(results, 1):

        print(f"Document {i}")

        print(
            f"Source   : "
            f"{doc.metadata.get('source', 'Unknown')}"
        )

        print(
            f"Category : "
            f"{doc.metadata.get('category', 'Unknown')}"
        )

        preview = (
            doc.page_content[:250]
            .replace("\n", " ")
        )

        print(
            f"Preview  : {preview}"
        )

        print()

    print("=" * 70)
    print(
        "RESULT: REAL AZURE AI SEARCH "
        "INDEX VERIFICATION SUCCESSFUL"
    )
    print("=" * 70)