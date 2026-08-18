"""
Tests for TravelDataLoader.

RUBRIC:
Data Loading & Preparation
- TravelDataLoader implemented correctly (3 marks)
- PDFs loaded directly from data/ folder (3 marks)
- Document categorization based on filename (2 marks)
- Text chunking configured properly (2 marks)
"""

from pathlib import Path

from langchain_core.documents import Document

from src.data_loader import TravelDataLoader


def test_travel_data_loader_initializes():
    """TravelDataLoader initializes with the required text splitter."""

    loader = TravelDataLoader()

    assert loader is not None
    assert loader.text_splitter is not None


def test_pdfs_loaded_from_data_directory():
    """PDF documents are loaded directly from the data/ directory."""

    loader = TravelDataLoader()

    documents = loader.load_pdfs_from_data_directory()

    assert isinstance(documents, list)
    assert len(documents) > 0

    for document in documents:
        assert isinstance(document, Document)
        assert document.metadata["file_type"] == "pdf"
        assert document.metadata["source"].endswith(".pdf")


def test_document_categorization():
    """Documents are categorized based on filename."""

    loader = TravelDataLoader()

    assert (
        loader._categorize_document("air-india-policy.pdf")
        == "air_india_policies"
    )

    assert (
        loader._categorize_document("ai-schedule.pdf")
        == "air_india_policies"
    )

    assert (
        loader._categorize_document(
            "U.S. Department of Transportation Regulations.pdf"
        )
        == "us_dot_regulations"
    )

    assert (
        loader._categorize_document("booking-policy.pdf")
        == "booking_policies"
    )

    assert (
        loader._categorize_document("refund-policy.pdf")
        == "refund_policies"
    )

    assert (
        loader._categorize_document("privacy-policy.pdf")
        == "privacy_policies"
    )

    assert (
        loader._categorize_document("general-travel.pdf")
        == "general"
    )


def test_text_chunking_configuration():
    """Text splitter uses the required chunk size and overlap."""

    loader = TravelDataLoader()

    assert loader.text_splitter._chunk_size == 1000
    assert loader.text_splitter._chunk_overlap == 200


def test_documents_are_split_into_chunks():
    """Loaded documents can be split into chunks."""

    loader = TravelDataLoader()

    document = Document(
        page_content="Travel information. " * 300,
        metadata={
            "source": "test.pdf",
            "file_type": "pdf",
            "category": "general",
        },
    )

    chunks = loader.split_documents([document])

    assert len(chunks) > 1

    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= 1000