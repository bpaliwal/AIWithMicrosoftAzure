"""
Data Loader for Travel Documents
RUBRIC: Data Loading & Preparation (10 marks total)
- TravelDataLoader implemented correctly (3 marks)
- PDFs loaded from local data/ folder OR Azure Blob Storage (3 marks)
- Document categorization based on filename (2 marks)
- Text chunking configured properly (2 marks)

Supports:
- BLOB_STORAGE=False -> load PDFs from local data/ directory
- BLOB_STORAGE=True  -> load PDFs from Azure Blob Storage

TASK: Load travel documents, categorize them, and chunk them.
"""

import io
import os
import pandas as pd
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from azure.storage.blob import BlobServiceClient


# Load environment variables
load_dotenv()


# Local data directory
DATA_DIR = Path("data")


class TravelDataLoader:
    """Loads travel documents from local storage or Azure Blob Storage."""

    def __init__(self):
        # Storage mode
        self.blob_storage = os.getenv("BLOB_STORAGE", "False").lower() == "true"

        # Azure Blob configuration
        self.storage_connection_string = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        self.storage_container_name = os.getenv(
            "AZURE_STORAGE_CONTAINER_NAME",
            "travel-documents"
        )

        # Initialize Azure Blob client only when Blob Storage is enabled
        self.blob_service_client = None
        self.container_client = None

        if self.blob_storage:
            if not self.storage_connection_string:
                raise ValueError(
                    "BLOB_STORAGE=True but "
                    "AZURE_STORAGE_CONNECTION_STRING is not configured."
                )

            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.storage_connection_string
            )

            self.container_client = (
                self.blob_service_client.get_container_client(
                    self.storage_container_name
                )
            )

            print(
                f"☁️  Azure Blob Storage enabled: "
                f"container='{self.storage_container_name}'"
            )
        else:
            print(f"📁 Local storage enabled: directory='{DATA_DIR}'")

        # HINT: Initialize text splitter with chunk_size=1000,
        # chunk_overlap=200
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    # =====================================================================
    # LOCAL PDF LOADING
    # =====================================================================

    def load_pdfs_from_data_directory(self) -> List[Document]:
        """
        Load all PDFs directly from local data/ directory.

        Used when:
            BLOB_STORAGE=False
        """

        documents = []

        if not DATA_DIR.exists():
            print(f"Warning: Directory {DATA_DIR} does not exist")
            return documents

        pdf_files = list(DATA_DIR.glob("*.pdf"))

        print(f"Found {len(pdf_files)} PDF files in data/")

        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()

                for doc in docs:
                    doc.metadata.update({
                        "source": pdf_file.name,
                        "file_type": "pdf",
                        "category": self._categorize_document(
                            pdf_file.name
                        )
                    })

                documents.extend(docs)

                print(
                    f"  ✓ Loaded: {pdf_file.name} "
                    f"({len(docs)} pages)"
                )

            except Exception as e:
                print(
                    f"  ✗ Error loading "
                    f"{pdf_file.name}: {e}"
                )

        return documents

    # =====================================================================
    # AZURE BLOB PDF LOADING
    # =====================================================================

    def load_pdfs_from_blob_storage(self) -> List[Document]:
        """
        Load all PDFs from Azure Blob Storage.

        Used when:
            BLOB_STORAGE=True

        PDFs are downloaded into memory and converted into
        LangChain Documents. No local files are created.
        """

        documents = []

        print(
            f"☁️  Loading PDFs from Azure Blob container "
            f"'{self.storage_container_name}'..."
        )

        try:
            blobs = self.container_client.list_blobs()

            pdf_blobs = [
                blob for blob in blobs
                if blob.name.lower().endswith(".pdf")
            ]

            print(
                f"Found {len(pdf_blobs)} PDF files "
                f"in Azure Blob Storage"
            )

            for blob in pdf_blobs:
                try:
                    print(f"  Downloading: {blob.name}")

                    blob_client = self.container_client.get_blob_client(
                        blob.name
                    )

                    pdf_bytes = blob_client.download_blob().readall()

                    # Read PDF directly from memory
                    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))

                    blob_filename = Path(blob.name).name

                    page_count = 0

                    for page_number, page in enumerate(
                        pdf_reader.pages
                    ):
                        text = page.extract_text() or ""

                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": blob_filename,
                                    "file_type": "pdf",
                                    "category": self._categorize_document(
                                        blob_filename
                                    ),
                                    "page": page_number
                                }
                            )
                        )

                        page_count += 1

                    print(
                        f"  ✓ Loaded: {blob_filename} "
                        f"({page_count} pages)"
                    )

                except Exception as e:
                    print(
                        f"  ✗ Error loading "
                        f"{blob.name}: {e}"
                    )

        except Exception as e:
            print(
                f"  ✗ Error accessing Azure Blob Storage: {e}"
            )
            raise

        return documents

    # =====================================================================
    # DOCUMENT CATEGORIZATION
    # =====================================================================

    def _categorize_document(self, filename: str) -> str:
        """Categorize document based on filename."""

        filename_lower = filename.lower()

        if (
            "air-india" in filename_lower
            or "ai-schedule" in filename_lower
        ):
            return "air_india_policies"

        elif (
            "u.s. department" in filename_lower
            or "transportation" in filename_lower
        ):
            return "us_dot_regulations"

        elif "refund" in filename_lower:
            return "refund_policies"

        elif "privacy" in filename_lower:
            return "privacy_policies"

        elif (
            "booking" in filename_lower
            or "policy" in filename_lower
        ):
            return "booking_policies"

        else:
            return "general"

    # =====================================================================
    # CSV LOADING
    # =====================================================================
    def load_csvs_from_blob_storage(self) -> List[Document]:
        """
        Load all CSV files from Azure Blob Storage.

        Used when:
            BLOB_STORAGE=True
        """

        documents = []

        print(
            f"☁️  Looking for CSV files in Azure Blob container "
            f"'{self.storage_container_name}'..."
        )

        try:
            blobs = self.container_client.list_blobs()

            csv_blobs = [
                blob for blob in blobs
                if blob.name.lower().endswith(".csv")
            ]

            print(
                f"Found {len(csv_blobs)} CSV files "
                f"in Azure Blob Storage"
            )

            for blob in csv_blobs:
                try:
                    print(f"  Downloading: {blob.name}")

                    blob_client = self.container_client.get_blob_client(
                        blob.name
                    )

                    csv_bytes = blob_client.download_blob().readall()

                    # Read CSV directly from memory
                    df = pd.read_csv(io.BytesIO(csv_bytes))

                    blob_filename = Path(blob.name).name

                    for idx, row in df.iterrows():

                        content = " | ".join(
                            f"{col}: {val}"
                            for col, val in row.items()
                        )

                        documents.append(
                            Document(
                                page_content=content,
                                metadata={
                                    "source": blob_filename,
                                    "file_type": "csv",
                                    "row_index": idx,
                                    "category": self._categorize_document(
                                        blob_filename
                                    )
                                }
                            )
                        )

                    print(
                        f"  ✓ Loaded: {blob_filename} "
                        f"({len(df)} rows)"
                    )

                except Exception as e:
                    print(
                        f"  ✗ Error loading "
                        f"{blob.name}: {e}"
                    )

        except Exception as e:
            print(
                f"  ✗ Error accessing Azure Blob Storage: {e}"
            )
            raise

        return documents

    def load_csvs_from_data_directory(self) -> List[Document]:
        """
        Load all CSVs from local data/ directory.

        CSV loading remains local because the current knowledge
        base uses PDFs as the production document source.
        """

        documents = []

        if not DATA_DIR.exists():
            print(f"Warning: Directory {DATA_DIR} does not exist")
            return documents

        csv_files = list(DATA_DIR.glob("*.csv"))

        print(f"Found {len(csv_files)} CSV files in data/")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(str(csv_file))

                for idx, row in df.iterrows():

                    content = " | ".join(
                        f"{col}: {val}"
                        for col, val in row.items()
                    )

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": csv_file.name,
                                "file_type": "csv",
                                "row_index": idx,
                                "category": self._categorize_document(
                                    csv_file.name
                                )
                            }
                        )
                    )

                print(
                    f"  ✓ Loaded: {csv_file.name} "
                    f"({len(df)} rows)"
                )

            except Exception as e:
                print(
                    f"  ✗ Error loading "
                    f"{csv_file.name}: {e}"
                )

        return documents

    # =====================================================================
    # LOAD ALL DOCUMENTS
    # =====================================================================

    def load_all_travel_documents(self) -> List[Document]:
        """
        Load all PDFs and CSVs.

        Storage source is controlled by BLOB_STORAGE:

            False -> local data/
            True  -> Azure Blob Storage
        """

        all_documents = []

        print("\n📂 Loading travel knowledge base...")
        if self.blob_storage:
            print("   Source: Azure Blob Storage")
        else:
            print("   Source: Local data/")

        print("=" * 60)

        if self.blob_storage:
            # Production: Azure Blob Storage
            pdf_docs = self.load_pdfs_from_blob_storage()
            csv_docs = self.load_csvs_from_blob_storage()

        else:
            # Local development: data/
            pdf_docs = self.load_pdfs_from_data_directory()
            csv_docs = self.load_csvs_from_data_directory()

        all_documents.extend(pdf_docs)
        all_documents.extend(csv_docs)

        print("=" * 60)
        print(
            f"✅ Total documents loaded: "
            f"{len(all_documents)}"
        )

        return all_documents

    # =====================================================================
    # CHUNK DOCUMENTS
    # =====================================================================

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split documents into chunks.
        """

        print(
            f"\n✂️  Splitting "
            f"{len(documents)} documents into chunks..."
        )

        chunks = self.text_splitter.split_documents(documents)

        print(
            f"✅ Created {len(chunks)} chunks"
        )

        print(
            f"   Average chunk size: "
            f"{sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars"
        )

        return chunks

# =====================================================================
# STANDALONE TEST
# =====================================================================

if __name__ == "__main__":

    loader = TravelDataLoader()

    docs = loader.load_all_travel_documents()

    chunks = loader.split_documents(docs)

    print("\n📊 Summary:")
    print(f"   Storage mode: "
          f"{'Azure Blob Storage' if loader.blob_storage else 'Local data/'}")
    print(f"   Total documents: {len(docs)}")
    print(f"   Total chunks: {len(chunks)}")
